"""Proper-training predictor coding for internal split calibration (#646)."""
import numpy as np
import pytest

import mboostrs as mb


def frame():
    rng = np.random.default_rng(636)
    x = rng.uniform(-1, 1, 240)
    return {'x': x, 'y': rng.poisson(np.exp(.3+.6*x)).astype(float)}


def fitted(data, kind, term, na_action='error'):
    if kind == 'gam':
        return mb.gamboost(f'y ~ bols({term})', data, family=mb.Poisson(),
                           control=mb.boost_control(60), na_action=na_action)
    return mb.glmboost(f'y ~ {term}', data, family=mb.Poisson(),
                       control=mb.boost_control(60), na_action=na_action)


@pytest.mark.parametrize('kind', ['glm', 'gam'])
@pytest.mark.parametrize('term', ['scale(x)', 'center(x)', 'log(x)', 'x'])
def test_internal_and_equivalent_external_partition_agree(kind, term):
    data = frame()
    if term == 'log(x)':
        data['x'] = np.exp(data['x'])
    if term == 'x':
        data['x'][::13] = np.nan
    action = 'impute' if term == 'x' else 'error'
    perm = np.random.default_rng(4).permutation(240)
    train, cal = ({k: v[idx] for k, v in data.items()} for idx in (perm[120:], perm[:120]))
    query = {'x': np.array([.2, .5, .8])}
    full, proper = fitted(data, kind, term, action), fitted(train, kind, term, action)
    # Poisson fits resolve score="auto" to the count-rank score, whose tie
    # randomization draws one uniform per calibration and query row from a
    # stream keyed by the seed (#636, #650). The internal split and the
    # equivalent held-out calibration see the same rows, so they agree
    # exactly when both draw from the same seed -- the agreement #646 pins
    # for the predictor coding holds under the randomization only then.
    internal = mb.conformal_interval(full, query, seed=4, mstop=30)
    external = mb.conformal_interval(proper, query, calibration=cal, seed=4, mstop=30)
    np.testing.assert_allclose(internal.lower, external.lower, atol=1e-11, rtol=0)
    np.testing.assert_allclose(internal.upper, external.upper, atol=1e-11, rtol=0)
    np.testing.assert_allclose(internal.predicted, external.predicted, atol=1e-11, rtol=0)
    assert internal.quantile == pytest.approx(external.quantile, abs=1e-11)
    assert internal.score == external.score
    at_training = mb.conformal_interval(full, seed=4, mstop=30)
    direct = mb.conformal_interval(proper, data, calibration=cal, seed=4, mstop=30)
    np.testing.assert_allclose(at_training.lower, direct.lower, atol=1e-11, rtol=0)
    np.testing.assert_allclose(at_training.upper, direct.upper, atol=1e-11, rtol=0)


@pytest.mark.parametrize('kind', ['glm', 'gam'])
def test_calibration_predictors_cannot_change_proper_training_point(kind):
    data = frame()
    cal_idx = np.random.default_rng(4).permutation(240)[:120]
    changed = {k: v.copy() for k, v in data.items()}
    changed['x'][cal_idx] += 10
    query = {'x': np.array([-.5, 0., .5])}
    before = mb.conformal_interval(fitted(data, kind, 'scale(x)'), query, seed=4, mstop=30)
    after = mb.conformal_interval(fitted(changed, kind, 'scale(x)'), query, seed=4, mstop=30)
    np.testing.assert_array_equal(before.predicted, after.predicted)
    assert before.quantile != after.quantile
