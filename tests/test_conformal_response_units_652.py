"""Original response units across standard conformal refits (#652)."""
import numpy as np
import pytest

import mboostrs as mb
from benches.audit_conformal_response_units import (
    TRANSFORMS, fit, inputs, original_prediction, reference, subset,
)


@pytest.mark.parametrize('frontend', ['glm', 'gam'])
@pytest.mark.parametrize('transform', TRANSFORMS)
@pytest.mark.parametrize('method', ['split', 'cv+', 'cqr'])
def test_original_response_reference(frontend, transform, method):
    data, query = inputs()
    expected = reference(data, query, frontend, transform, method, 30)
    interval = mb.conformal_interval(fit(data, frontend, transform), query,
                                     method=method, mstop=30, seed=4, n_folds=4)
    for field in ('lower', 'upper', 'predicted'):
        np.testing.assert_allclose(getattr(interval, field), expected[field], atol=1e-10, rtol=0)
    if transform != 'bare':
        assert interval.fitted_mean is None
    if interval.quantile is not None:
        assert interval.quantile == pytest.approx(expected['quantile'], abs=1e-10)


@pytest.mark.parametrize('frontend', ['glm', 'gam'])
@pytest.mark.parametrize('transform', TRANSFORMS)
def test_external_calibration_and_training_targets(frontend, transform):
    data, query = inputs()
    p = np.random.default_rng(4).permutation(240)
    train, held = subset(data, p[120:]), subset(data, p[:120])
    model = fit(train, frontend, transform)
    for target in (query, None):
        expected = reference(data, target, frontend, transform, 'split', 30, 'external')
        interval = mb.conformal_interval(model, target, calibration=held, mstop=30)
        for field in ('lower', 'upper', 'predicted'):
            np.testing.assert_allclose(getattr(interval, field), expected[field], atol=1e-10, rtol=0)


@pytest.mark.parametrize('transform', ['center', 'scale'])
@pytest.mark.parametrize('method', ['split', 'cv+', 'cqr'])
def test_tuned_intervals_run_on_a_fold_fixed_transform(transform, method):
    # Tuning on the fold-fixed transform: the transform is learned on the rows
    # the refit uses, once, and every tuning fold reuses it (#663).
    data, query = inputs()
    interval = mb.conformal_interval(fit(data, 'gam', transform), query,
                                     method=method, mstop='tune', seed=4, n_folds=4)
    assert np.all(np.isfinite(interval.lower)) and np.all(np.isfinite(interval.upper))
    assert np.all(interval.lower <= interval.upper)


@pytest.mark.parametrize('transform', ['center', 'scale', 'log'])
def test_response_and_missing_predictor_recipes_use_proper_training_rows(monkeypatch, transform):
    from mboostrs._refit import RefitSpec

    data, query = inputs()
    data['x'][::13] = np.nan
    query['x'][1] = np.nan
    p = np.random.default_rng(4).permutation(240)
    proper = subset(data, p[120:])
    changed = {key: value.copy() for key, value in data.items()}
    changed['y'][p[:120]] *= 10
    changed['x'][p[:120]] += 100
    captured, build = [], RefitSpec.build

    def record(spec, **kwargs):
        model = build(spec, **kwargs)
        captured.append(model)
        return model

    monkeypatch.setattr(RefitSpec, 'build', record)
    intervals = []
    for frame in (data, changed):
        model = fit(frame, 'gam', transform, predictor='scale(x)', na_action='impute')
        intervals.append(mb.conformal_interval(model, query, method='cqr', seed=4, mstop=30))
    assert len(captured) == 4
    for first, second in zip(captured[:2], captured[2:], strict=True):
        assert first._response_transform == second._response_transform
        assert first._transform_recipe == second._transform_recipe
        direct = fit(proper, 'gam', transform, steps=30, family=first.family,
                     predictor='scale(x)', na_action='impute')
        assert first._response_transform == direct._response_transform
        np.testing.assert_array_equal(original_prediction(first, proper, query, transform),
                                      original_prediction(second, proper, query, transform))
    assert intervals[0].quantile != intervals[1].quantile


def test_cv_plus_remainder_chunking_and_response_scale(monkeypatch):
    from mboostrs import conformal

    data, query = inputs(241)
    expected = reference(data, query, 'gam', 'scale', 'cv+', 30)
    monkeypatch.setattr(conformal, '_cv_plus_chunk', lambda *_: 1)
    result = mb.conformal_interval(fit(data, 'gam', 'scale'), query,
                                   method='cv+', seed=4, n_folds=4, mstop=30)
    np.testing.assert_allclose(result.lower, expected['lower'], atol=1e-10, rtol=0)
    np.testing.assert_allclose(result.upper, expected['upper'], atol=1e-10, rtol=0)
    assert result.n_set_aside == 1


def test_omitted_rows_remain_aligned():
    data, query = inputs()
    data['x'][0] = np.nan
    retained = {key: value[1:] for key, value in data.items()}
    model = fit(data, 'gam', 'scale', na_action='omit')
    expected = reference(retained, query, 'gam', 'scale', 'cqr', 30)
    result = mb.conformal_interval(model, query, method='cqr', mstop=30, seed=4)
    np.testing.assert_allclose(result.lower, expected['lower'], atol=1e-10, rtol=0)


@pytest.mark.parametrize('invalid', [np.nan, np.inf])
def test_invalid_raw_calibration_response_refuses(invalid):
    data, query = inputs()
    held = subset(data, np.arange(120))
    held['y'][0] = invalid
    with pytest.raises(ValueError, match='finite observations'):
        mb.conformal_interval(fit(subset(data, np.arange(120, 240)), 'glm', 'scale'),
                              query, calibration=held, mstop=30)


def test_missing_raw_response_refuses_instead_of_using_encoded_column():
    data, query = inputs()
    model = fit(data, 'gam', 'scale')
    held = {'x': data['x'][:120], 'scale__y': np.ones(120)}
    with pytest.raises(ValueError, match="no response column 'y'"):
        mb.conformal_interval(model, query, calibration=held, mstop=30)


def test_inverse_domain_refusal_is_explicit():
    data, _ = inputs()
    model = fit(data, 'glm', 'exp')
    # A Gaussian fit to exp(y) can extrapolate to a negative value, outside log's domain.
    with pytest.raises(ValueError, match="inverse transform's domain"):
        mb.conformal_interval(model, {'x': np.array([-1e6])}, method='split', mstop=30, seed=4)


def test_tuned_log_response_functions_and_stops_exclude_calibration_rows(monkeypatch):
    from mboostrs import conformal

    data, query = inputs()
    held = np.random.default_rng(4).permutation(240)[:120]
    changed = {key: value.copy() for key, value in data.items()}
    changed['y'][held] *= 10
    captured, quantile_refit = [], conformal._quantile_refit

    def record(*args, **kwargs):
        model = quantile_refit(*args, **kwargs)
        captured.append(model)
        return model

    monkeypatch.setattr(conformal, '_quantile_refit', record)
    results = [mb.conformal_interval(fit(frame, 'gam', 'log'), query,
                                    method='cqr', mstop='tune', seed=4)
               for frame in (data, changed)]
    for first, second in zip(captured[:2], captured[2:], strict=True):
        assert first.mstop == second.mstop
        assert first._response_transform == second._response_transform
        np.testing.assert_array_equal(first.predict(query), second.predict(query))
    assert results[0].quantile != results[1].quantile


def test_cv_plus_extended_path_point_is_in_original_units():
    data, query = inputs()
    result = mb.conformal_interval(fit(data, 'glm', 'scale'), query,
                                   method='cv+', mstop=80, seed=4, n_folds=4)
    full = fit(data, 'glm', 'scale', steps=80)
    expected = original_prediction(full, data, query, 'scale')
    np.testing.assert_allclose(result.predicted, expected, atol=1e-10, rtol=0)
