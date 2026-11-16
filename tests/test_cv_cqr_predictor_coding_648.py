"""Raw-pipeline prediction and independent CV+/CQR references (#648)."""
import warnings

import numpy as np
import pytest

import mboostrs as mb
from benches.audit_conformal_predictor_coding import fit, inputs, reference


@pytest.mark.parametrize('frontend', ['glm', 'gam'])
@pytest.mark.parametrize('method', ['cv+', 'cqr'])
@pytest.mark.parametrize('recipe', ['scale', 'log', 'missing'])
def test_matches_independent_public_pipeline_reference(frontend, method, recipe):
    frame, query = inputs('gaussian', recipe)
    expected = reference(frame, query, frontend, 'gaussian', recipe, method, 30)
    result = mb.conformal_interval(fit(frame, frontend, 'gaussian', recipe), query,
        method=method, mstop=30, seed=4, n_folds=4)
    for name in ('lower', 'upper', 'predicted'):
        np.testing.assert_allclose(getattr(result, name), expected[name], atol=1e-10, rtol=0)
    assert result.n_training == expected['n_training']
    assert result.n_calibration == expected['n_calibration']
    assert result.guaranteed_coverage == pytest.approx(expected['guaranteed_coverage'])


@pytest.mark.parametrize('frontend', ['glm', 'gam'])
@pytest.mark.parametrize('method', ['cv+', 'cqr'])
def test_affine_predictor_formulas_have_identical_intervals(frontend, method):
    frame, query = inputs('poisson', 'bare')
    a = mb.conformal_interval(fit(frame, frontend, 'poisson', 'bare'), query,
        method=method, mstop=30, seed=4, n_folds=4)
    b = mb.conformal_interval(fit(frame, frontend, 'poisson', 'scale'), query,
        method=method, mstop=30, seed=4, n_folds=4)
    np.testing.assert_allclose(a.lower, b.lower, atol=1e-10, rtol=0)
    np.testing.assert_allclose(a.upper, b.upper, atol=1e-10, rtol=0)


@pytest.mark.parametrize('method', ['cv+', 'cqr'])
def test_training_rows_use_raw_pipeline_and_original_order(method):
    frame, _ = inputs('gaussian', 'missing')
    expected = reference(frame, None, 'gam', 'gaussian', 'missing', method, 30)
    result = mb.conformal_interval(fit(frame, 'gam', 'gaussian', 'missing'),
        method=method, mstop=30, seed=4, n_folds=4)
    np.testing.assert_allclose(result.lower, expected['lower'], atol=1e-10, rtol=0)
    np.testing.assert_allclose(result.upper, expected['upper'], atol=1e-10, rtol=0)


def test_unequal_folds_chunking_and_extended_prefix_match_reference(monkeypatch):
    from mboostrs import conformal

    frame, query = inputs('gaussian', 'missing', n=241)
    expected = reference(frame, query, 'gam', 'gaussian', 'missing', 'cv+', 80)
    monkeypatch.setattr(conformal, '_cv_plus_chunk', lambda *_: 1)
    result = mb.conformal_interval(fit(frame, 'gam', 'gaussian', 'missing'), query,
        method='cv+', mstop=80, seed=4, n_folds=4)
    np.testing.assert_allclose(result.lower, expected['lower'], atol=1e-10, rtol=0)
    np.testing.assert_allclose(result.upper, expected['upper'], atol=1e-10, rtol=0)
    np.testing.assert_allclose(result.predicted, expected['predicted'], atol=1e-10, rtol=0)
    assert result.n_set_aside == 1
    assert result.n_calibration == 240 and result.n_training == 180
    assert result.guaranteed_coverage == pytest.approx(expected['guaranteed_coverage'])


@pytest.mark.parametrize('method', ['cv+', 'cqr'])
def test_tuning_relearns_inner_fold_preprocessing_and_matches_reference(method):
    from benches.audit_conformal_predictor_coding import observe_inner_cv

    frame, query = inputs('gaussian', 'scale')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        expected = reference(frame, query, 'glm', 'gaussian', 'scale', method, 'tune')
        trace = []
        with observe_inner_cv(trace):
            result = mb.conformal_interval(fit(frame, 'glm', 'gaussian', 'scale'), query,
                method=method, mstop='tune', seed=4, n_folds=4)
    np.testing.assert_allclose(result.lower, expected['lower'], atol=1e-10, rtol=0)
    np.testing.assert_allclose(result.upper, expected['upper'], atol=1e-10, rtol=0)
    assert trace and all(stage['mode'] == 'refit' for stage in trace)
    assert [stage['optimum'] for stage in trace] == [stage['optimum'] for stage in expected['inner_cv']]


@pytest.mark.parametrize('method', ['cv+', 'cqr'])
def test_actual_outer_and_inner_fit_rows_exclude_their_held_rows(monkeypatch, method):
    from mboostrs import conformal
    from mboostrs._refit import RefitSpec

    frame, query = inputs('gaussian', 'scale')
    query['rowid'] = np.array([-3, -2, -1])
    model = fit(frame, 'gam', 'gaussian', 'scale')
    built, predicted, inner_parent, inner_checked = [], [], [], []
    build, predict = RefitSpec.build, conformal._predict_raw
    cv = type(model).cvrisk

    def record_build(spec, **kwargs):
        ids = set(np.asarray(spec.data['rowid'], dtype=int))
        if inner_parent:
            assert ids.issubset(inner_parent[-1])
            assert len(inner_parent[-1]-ids) == len(inner_parent[-1])//5
            inner_checked.append(ids)
        built.append(ids)
        return build(spec, **kwargs)

    def record_inner_cv(fitted, *args, **kwargs):
        inner_parent.append(set(np.asarray(fitted._data['rowid'], dtype=int)))
        try:
            return cv(fitted, *args, **kwargs)
        finally:
            inner_parent.pop()

    def record_prediction(fitted, rows, *args):
        training = set(np.asarray(fitted._data['rowid'], dtype=int))
        held = set(np.asarray(rows['rowid'], dtype=int))
        if min(held) >= 0:
            assert not training.intersection(held)
            predicted.append((training, held))
        return predict(fitted, rows, *args)

    monkeypatch.setattr(RefitSpec, 'build', record_build)
    monkeypatch.setattr(conformal, '_predict_raw', record_prediction)
    monkeypatch.setattr(type(model), 'cvrisk', record_inner_cv)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mb.conformal_interval(model, query, method=method, mstop='tune', seed=4, n_folds=4)
    assert built and predicted and inner_checked and not inner_parent
    if method == 'cqr':
        calibration = set(np.random.default_rng(4).permutation(240)[:120])
        assert all(not rows.intersection(calibration) for rows in built)
        assert {len(rows) for rows in built} == {120, 96}
    else:
        held_folds = [set(rows) for rows in np.random.default_rng(4).permutation(240).reshape(4, 60)]
        for training, held in predicted:
            assert held in held_folds
            assert training == set(range(240))-held
        assert {len(rows) for rows in built} == {180, 144}


@pytest.mark.parametrize('method', ['cv+', 'cqr'])
def test_fixed_prefix_refits_keep_case_weights_aligned(monkeypatch, method):
    from mboostrs._refit import RefitSpec

    frame, query = inputs('gaussian', 'scale')
    weights = np.linspace(.5, 2., 240)
    model = mb.gamboost('y ~ bols(scale(x))', frame, weights=weights,
                        control=mb.boost_control(60))
    captured, build = [], RefitSpec.build

    def record(spec, **kwargs):
        rowids = np.asarray(spec.data['rowid'], dtype=int)
        actual = kwargs.get('weights', spec.arguments.get('weights'))
        np.testing.assert_array_equal(actual, weights[rowids])
        captured.append(rowids)
        return build(spec, **kwargs)

    monkeypatch.setattr(RefitSpec, 'build', record)
    with pytest.warns(UserWarning, match='case weights'):
        mb.conformal_interval(model, query, method=method, mstop=30, seed=4, n_folds=4)
    assert len(captured) == (4 if method == 'cv+' else 2)


@pytest.mark.parametrize('method', ['cv+', 'cqr'])
def test_misaligned_raw_rows_refuse_before_refitting(monkeypatch, method):
    from dataclasses import replace
    from mboostrs import _refit

    frame, query = inputs('gaussian', 'scale')
    model = fit(frame, 'gam', 'gaussian', 'scale')
    original = _refit.refit_spec

    def misaligned(fitted):
        spec = original(fitted)
        return replace(spec, raw_data={k: v[:-1] for k, v in frame.items()})

    monkeypatch.setattr(_refit, 'refit_spec', misaligned)
    with pytest.raises(NotImplementedError, match='raw inputs aligned'):
        mb.conformal_interval(model, query, method=method, mstop=30, seed=4, n_folds=4)


@pytest.mark.parametrize('method', ['cv+', 'cqr'])
def test_missing_raw_inputs_refuse_by_name(monkeypatch, method):
    from dataclasses import replace
    from mboostrs import _refit

    frame, query = inputs('gaussian', 'scale')
    model = fit(frame, 'gam', 'gaussian', 'scale')
    original = _refit.refit_spec
    monkeypatch.setattr(_refit, 'refit_spec', lambda fitted: replace(original(fitted), raw_data=None))
    with pytest.raises(NotImplementedError, match='raw'):
        mb.conformal_interval(model, query, method=method, mstop=30, seed=4, n_folds=4)


@pytest.mark.parametrize('method', ['cv+', 'cqr'])
def test_missing_query_predictor_refuses_without_an_interval(method):
    frame, _ = inputs('gaussian', 'scale')
    with pytest.raises((KeyError, ValueError), match='x'):
        mb.conformal_interval(fit(frame, 'gam', 'gaussian', 'scale'), {'other': np.zeros(3)},
            method=method, mstop=30, seed=4, n_folds=4)


@pytest.mark.parametrize('frontend', ['glm', 'gam'])
@pytest.mark.parametrize('budget', [30, 'tune'])
def test_cqr_calibration_changes_leave_quantile_fits_and_stops_identical(monkeypatch, frontend, budget):
    from mboostrs import conformal

    frame, query = inputs('gaussian', 'scale')
    calibration = np.random.default_rng(4).permutation(240)[:120]
    changed = {name: values.copy() for name, values in frame.items()}
    changed['x'][calibration] += 10
    changed['y'][calibration] += 4
    captured, original = [], conformal._quantile_refit

    def record(*args):
        model = original(*args)
        captured.append((model.mstop, model.coef(), model.predict(query, type='response')))
        return model

    monkeypatch.setattr(conformal, '_quantile_refit', record)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = mb.conformal_interval(fit(frame, frontend, 'gaussian', 'scale'), query,
            method='cqr', mstop=budget, seed=4)
        b = mb.conformal_interval(fit(changed, frontend, 'gaussian', 'scale'), query,
            method='cqr', mstop=budget, seed=4)
    assert len(captured) == 4
    for before, after in zip(captured[:2], captured[2:], strict=True):
        assert before[0] == after[0]
        assert before[1].keys() == after[1].keys()
        for name in before[1]:
            np.testing.assert_array_equal(before[1][name], after[1][name])
        np.testing.assert_array_equal(before[2], after[2])
    assert a.quantile != b.quantile
    np.testing.assert_allclose(a.predicted, b.predicted, atol=1e-12, rtol=0)


@pytest.mark.parametrize('method', ['cv+', 'cqr'])
def test_each_fit_uses_its_own_learned_log_floor(method):
    frame, _ = inputs('gaussian', 'log')
    query = {'x': np.array([0., 1., 2.])}
    expected = reference(frame, query, 'gam', 'gaussian', 'log', method, 30)
    result = mb.conformal_interval(fit(frame, 'gam', 'gaussian', 'log'), query,
        method=method, mstop=30, seed=4, n_folds=4)
    np.testing.assert_allclose(result.lower, expected['lower'], atol=1e-10, rtol=0)
    np.testing.assert_allclose(result.upper, expected['upper'], atol=1e-10, rtol=0)


def test_cqr_replays_categorical_encoding_from_raw_labels():
    frame, _ = inputs('gaussian', 'bare')
    frame['group'] = np.where(np.arange(240) % 2, 'B', 'A')
    query = {'group': np.array(['A', 'B'])}
    perm = np.random.default_rng(4).permutation(240)
    training = {k: v[perm[120:]] for k, v in frame.items()}
    calibration = {k: v[perm[:120]] for k, v in frame.items()}
    low = mb.glmboost('y ~ group', training, family=mb.Quantile(tau=.05),
                      control=mb.boost_control(30))
    high = mb.glmboost('y ~ group', training, family=mb.Quantile(tau=.95),
                       control=mb.boost_control(30))
    a, b = low.predict(calibration, type='response'), high.predict(calibration, type='response')
    correction = np.sort(np.maximum(np.minimum(a, b)-calibration['y'],
                                    calibration['y']-np.maximum(a, b)))[108]
    a, b = low.predict(query, type='response'), high.predict(query, type='response')
    model = mb.glmboost('y ~ group', frame, control=mb.boost_control(60))
    result = mb.conformal_interval(model, query, method='cqr', mstop=30, seed=4)
    np.testing.assert_allclose(result.lower, np.minimum(a, b)-correction, atol=1e-10, rtol=0)
    np.testing.assert_allclose(result.upper, np.maximum(a, b)+correction, atol=1e-10, rtol=0)
