"""Calibration responses never enter discovery selection, fitting or tuning."""
import importlib
import json

import numpy as np
import polars as pl
import pytest
from scipy.stats import norm

import mboostrs as mb
from mboostrs.cvrisk import cv_weights

module = importlib.import_module('mboostrs.discover')


def draw(seed=631, n=300):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    return pl.DataFrame({'rowid': np.arange(n), 'x': x,
                         'y': .8*x+(.3+.3*np.abs(x))*rng.normal(size=n)})


def test_internal_holdout_excludes_rows_at_every_stage_and_resists_response_mutation(monkeypatch):
    data = draw(n=600)
    held = np.flatnonzero(cv_weights(600, type='kfold', B=2, seed=7)[:, 0] == 0)
    expected = set(range(600))-set(held.tolist())
    seen = set()
    for name in ('_infer_family', '_draft', '_mean_effects', '_interactions', '_spread', '_tune_and_fit', '_refit'):
        original = getattr(module._Analysis, name)
        def checked(self, *args, _name=name, _original=original, **kwargs):
            assert set(self.frame['rowid'].to_list()) == expected
            seen.add(_name)
            return _original(self, *args, **kwargs)
        monkeypatch.setattr(module._Analysis, name, checked)
    query = pl.DataFrame({'x': [-1., 0., 1.], 'rowid': [-1, -2, -3]})
    options = dict(newdata=query, seed=7, candidates={'mu': ['x'], 'sigma': ['x']})
    first = mb.discover(data, 'y', **options)
    changed = data.with_columns(pl.when(pl.col('rowid').is_in(held.tolist()))
        .then(1000.+100*pl.col('rowid')).otherwise(pl.col('y')).alias('y'))
    second = mb.discover(changed, 'y', **options)
    assert seen == {'_infer_family', '_draft', '_mean_effects', '_interactions', '_spread', '_tune_and_fit', '_refit'}
    assert (first.family, first.formula, first.spread_formula, first.mstop) == (
        second.family, second.formula, second.spread_formula, second.mstop)
    np.testing.assert_array_equal(first.predictions['mean'], second.predictions['mean'])
    if 'sigma' in first.predictions:
        np.testing.assert_array_equal(first.predictions['sigma'], second.predictions['sigma'])
    # Check the fitted objects after calibration, not only the mean column
    # that discovery computed before calibrating its band.
    np.testing.assert_equal(first.model.coef(), second.model.coef())
    np.testing.assert_equal(first.model.mstop, second.model.mstop)
    np.testing.assert_equal(first.model.predict(query, type='response', extrapolation='clip'),
                            second.model.predict(query, type='response', extrapolation='clip'))
    assert (first.refit is None) == (second.refit is None)
    if first.refit is not None:
        np.testing.assert_equal(first.refit.predict(query, type='response'),
                                second.refit.predict(query, type='response'))
    assert first.interval.quantile != second.interval.quantile
    parts = first.provenance['partition']
    assert set(parts['selection_rows']) == expected
    assert set(parts['calibration_rows']) == set(held)
    assert first.interval.n_calibration == 300
    assert first.log[0]['stage'] == 'calibration'


def test_internal_and_caller_supplied_partition_produce_the_same_model_and_band():
    data = draw(n=600)
    weights = cv_weights(600, type='kfold', B=2, seed=3)[:, 0]
    selection, held = np.flatnonzero(weights > 0), np.flatnonzero(weights == 0)
    query = data.head(9).drop('y')
    options = dict(newdata=query, seed=3, interactions=False)
    internal = mb.discover(data, 'y', **options)
    explicit = mb.discover(data[selection], 'y', calibration=data[held], **options)
    assert internal.formula == explicit.formula and internal.mstop == explicit.mstop
    np.testing.assert_array_equal(internal.predictions.to_numpy(), explicit.predictions.to_numpy())
    assert explicit.provenance['partition']['calibration_input'] == 'calibration'
    assert len(explicit.provenance['partition']['selection_rows']) == 300
    assert 'interval' in json.loads(json.dumps(explicit.to_dict()))


def test_analysis_without_prediction_bands_keeps_every_input_row():
    data = draw(n=80)
    result = mb.discover(data, 'y', seed=1, spread=False, interactions=False)
    assert result.provenance['partition']['selection_rows'] == list(range(80))
    assert result.provenance['partition']['calibration_rows'] == []
    assert result.interval is None
    assert len(result.to_dict(include_arrays=True)['fitted']) == 80


def test_inadequate_or_reused_calibration_is_rejected_before_selection(monkeypatch):
    monkeypatch.setattr(module, '_Analysis', lambda *a, **kw: pytest.fail('selection must not start'))
    data = draw(n=50)
    with pytest.raises(ValueError, match='selection rows'):
        mb.discover(data, 'y', newdata=data.head(1))
    with pytest.raises(ValueError, match='at least .* calibration'):
        mb.discover(data, 'y', calibration=draw(seed=9, n=5))
    with pytest.raises(ValueError, match='same object'):
        mb.discover(data, 'y', calibration=data)


@pytest.fixture(scope='module')
def lss():
    data = draw(n=500)
    return mb.gamboostLSS({'mu': 'y ~ bols(x)', 'sigma': 'y ~ bbs(x)'}, data,
                         family=mb.GaussianLSS(), control=mb.boost_control(120))


def test_external_lss_uses_fitted_quantiles_and_no_refit(lss, monkeypatch):
    refit = importlib.import_module('mboostrs._refit')
    monkeypatch.setattr(refit, 'refit_spec', lambda *a, **kw: pytest.fail('external calibration must not refit'))
    calibration = draw(seed=632, n=99).with_columns(pl.col('y')*2)
    query = draw(seed=633, n=10).drop('y')
    params = lss.params(calibration, type='response')
    low = params['mu']+norm.ppf(.05)*params['sigma']
    high = params['mu']+norm.ppf(.95)*params['sigma']
    q = np.sort(np.maximum(low-calibration['y'].to_numpy(), calibration['y'].to_numpy()-high))[89]
    interval = mb.conformal_interval(lss, query, calibration=calibration, level=.9)
    target = lss.params(query, type='response')
    np.testing.assert_allclose(interval.lower, target['mu']+norm.ppf(.05)*target['sigma']-q)
    np.testing.assert_allclose(interval.upper, target['mu']+norm.ppf(.95)*target['sigma']+q)
    assert interval.quantile == pytest.approx(q)
    assert interval.n_training == 500 and interval.n_calibration == 99
    assert interval.guaranteed_coverage == .9
    with pytest.raises(ValueError, match='beyond'):
        mb.conformal_interval(lss, query, calibration=calibration, mstop=121)
    with pytest.raises(ValueError, match="mstop='tune'"):
        mb.conformal_interval(lss, query, calibration=calibration, mstop='tune')
    with pytest.raises(NotImplementedError, match='calibration='):
        mb.conformal_interval(lss, query, calibration=calibration, method='cqr')


def test_lss_external_calibration_reuses_learned_predictor_recipes():
    rng = np.random.default_rng(634)
    h = np.exp(rng.normal(size=400))
    x = rng.normal(size=400)
    y = .5*np.log(h)+.2*x+rng.normal(size=400)
    x[::11] = np.nan
    data = pl.DataFrame({'h': h, 'x': x, 'y': y})
    fit = mb.gamboostLSS({'mu': 'y ~ bbs(log(h)) + bols(x)', 'sigma': 'y ~ bols(log(h))'},
        data.head(250), family=mb.GaussianLSS(), control=mb.boost_control(80), na_action='impute')
    query = data.tail(20).drop('y')
    before = fit.predict_parameters(query)
    interval = mb.conformal_interval(fit, query, calibration=data.slice(250, 130), extrapolation='clip')
    assert np.all(np.isfinite(interval.lower))
    after = fit.predict_parameters(query)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])


def test_missing_calibration_response_is_a_clear_schema_error(monkeypatch):
    monkeypatch.setattr(module, '_Analysis', lambda *a, **kw: pytest.fail('selection must not start'))
    with pytest.raises(ValueError, match="calibration rows have no response column 'y'"):
        mb.discover(draw(), 'y', calibration=draw(seed=8, n=30).drop('y'))


def test_invalid_calibration_values_do_not_produce_a_coverage_claim(lss):
    held = draw(seed=8, n=30).with_columns(pl.lit(float('nan')).alias('y'))
    with pytest.raises(ValueError, match='finite observations'):
        mb.conformal_interval(lss, calibration=held)


def test_external_lss_response_transform_returns_original_scale_bounds():
    data = draw(n=300).with_columns((pl.col('y')/4).exp().alias('y'))
    fit = mb.gamboostLSS({'mu': 'log(y) ~ bols(x)', 'sigma': 'log(y) ~ bols(x)'}, data.head(200),
        family=mb.GaussianLSS(), control=mb.boost_control(40))
    query, held = data.tail(10).drop('y'), data.slice(200, 90)
    interval = mb.conformal_interval(fit, query, calibration=held, extrapolation='allow')
    assert interval.fitted_mean is None
    assert np.all(np.isfinite(interval.lower))
    expected = fit.prediction_interval(level=.9, newdata=query)
    # Calibration widens the original-scale family quantiles, not log(y).
    np.testing.assert_allclose(interval.lower, np.asarray(expected['lower'])-interval.quantile)
    np.testing.assert_allclose(interval.upper, np.asarray(expected['upper'])+interval.quantile)


def test_budget_skip_does_not_attach_an_uncomputed_coverage_claim():
    data = draw(n=120)
    result = mb.discover(data, 'y', newdata=data.head(3).drop('y'), seed=4, budget_seconds=0)
    assert len(result.provenance['partition']['selection_rows']) == 60
    assert len(result.provenance['partition']['calibration_rows']) == 60
    assert result.interval is None
    assert result.predictions['mean'].is_finite().all()
    assert result.predictions['lower'].is_nan().all()
    assert not any("guarantee '" in row['reason'] for row in result.log)


def test_missing_calibration_predictor_logs_refusal_without_changing_the_mean():
    data = draw(n=150)
    query = data.head(3).drop('y')
    result = mb.discover(data, 'y', calibration=draw(seed=11, n=40).drop('x'),
        newdata=query, mandatory_terms=['x'], seed=0, interactions=False, spread=False)
    assert result.interval is None
    assert result.predictions['mean'].is_finite().all()
    assert result.predictions['lower'].is_nan().all()
    assert any(row['stage'] == 'predict' and 'error' in row['reason'] for row in result.log)


def test_noncyclic_lss_held_out_calibration_uses_the_total_update_prefix():
    data = draw(n=260)
    model = mb.gamboostLSS({'mu': 'y ~ bols(x)', 'sigma': 'y ~ bols(x)'}, data.head(180),
        family=mb.GaussianLSS(), mstop=80, method='noncyclic')
    assert model.stopping_budget().total == 80
    query, held = data.tail(10).drop('y'), data.slice(180, 70)
    result = mb.conformal_interval(model, query, calibration=held, mstop=30)
    params = model.params(held, type='response', mstop=30)
    low = params['mu']+norm.ppf(.05)*params['sigma']
    high = params['mu']+norm.ppf(.95)*params['sigma']
    scores = np.maximum(low-held['y'].to_numpy(), held['y'].to_numpy()-high)
    expected = np.sort(scores)[63]
    assert result.quantile == pytest.approx(expected)
    with pytest.raises(ValueError, match='beyond'):
        mb.conformal_interval(model, query, calibration=held, mstop=81)
