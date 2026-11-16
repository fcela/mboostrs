"""Count-score numerical references and independent calibration (#636)."""
import numpy as np
import pytest
from scipy.stats import poisson

import mboostrs as mb
from mboostrs import _rust

SCORES = ("poisson_normalized", "poisson_quantile")


@pytest.mark.parametrize("score", SCORES)
@pytest.mark.parametrize("level", [.5, .8, .9, .99])
def test_core_matches_independent_poisson_reference(score, level):
    mu = np.array([0., 1e-12, .01, .3, 1., 4., 25., 100., 1e5])
    y = np.array([0., 1., 0., 2., 1., 10., 20., 130., 100100.])
    if score == "poisson_normalized":
        scale = np.sqrt(np.maximum(mu, 1e-12))
        expected = np.abs(y-mu)/scale
        lo, hi = mu-2*scale, mu+2*scale
    else:
        lo, hi = poisson.ppf((1-level)/2, mu), poisson.ppf((1+level)/2, mu)
        expected = np.maximum(lo-y, y-hi)
        lo, hi = lo-2, hi+2
    np.testing.assert_allclose(_rust.conformal_poisson_scores(y, mu, level, score), expected,
                               rtol=1e-12, atol=1e-12)
    got_lo, got_hi = _rust.conformal_poisson_band(mu, level, 2., score)
    np.testing.assert_allclose(got_lo, lo, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got_hi, hi, rtol=1e-12, atol=1e-12)


def data():
    rng = np.random.default_rng(636)
    x = rng.uniform(-1, 1, 240)
    return {"x": x, "y": rng.poisson(np.exp(.3+.6*x)).astype(float)}


def fit(rows, kind="gam", term="x"):
    return (mb.gamboost(f"y ~ bols({term})", rows, family=mb.Poisson(), control=mb.boost_control(60))
            if kind == "gam" else
            mb.glmboost(f"y ~ {term}", rows, family=mb.Poisson(), control=mb.boost_control(60)))


@pytest.mark.parametrize("score", SCORES)
@pytest.mark.parametrize("kind", ["glm", "gam"])
@pytest.mark.parametrize("term", ["x", "scale(x)"])
def test_internal_external_and_fixed_prefix_agree(score, kind, term):
    frame = data()
    perm = np.random.default_rng(4).permutation(240)
    train, cal = ({k: v[idx] for k, v in frame.items()} for idx in (perm[120:], perm[:120]))
    query = {"x": np.array([-.5, 0., .5])}
    full, proper = fit(frame, kind, term), fit(train, kind, term)
    internal = mb.conformal_interval(full, query, seed=4, mstop=30, score=score)
    external = mb.conformal_interval(proper, query, calibration=cal, mstop=30, score=score)
    np.testing.assert_allclose(internal.lower, external.lower, atol=1e-12)
    np.testing.assert_allclose(internal.upper, external.upper, atol=1e-12)
    assert internal.score == external.score == score
    assert internal.n_training == internal.n_calibration == 120
    # Calibration cannot alter the fitted mean even when its correction changes.
    coefficients = proper.coef()
    prediction = proper.predict(query, type="response").copy()
    changed = {**cal, "y": cal["y"]+20}
    changed_interval = mb.conformal_interval(proper, query, calibration=changed, mstop=30, score=score)
    assert changed_interval.quantile != external.quantile
    np.testing.assert_array_equal(proper.predict(query, type="response"), prediction)
    for key, value in coefficients.items():
        np.testing.assert_array_equal(proper.coef()[key], value)


@pytest.mark.parametrize("score", SCORES)
@pytest.mark.parametrize("bad", [-1., .5, np.inf, np.nan])
def test_invalid_count_responses_refuse(score, bad):
    frame = data()
    calibration = {**frame, "y": frame["y"].copy()}
    calibration["y"][0] = bad
    with pytest.raises(ValueError, match="finite|nonnegative integer"):
        mb.conformal_interval(fit(frame), {"x": np.array([0.])}, calibration=calibration, score=score)


@pytest.mark.parametrize("score", SCORES)
def test_unsupported_routes_and_invalid_means_refuse(score):
    model = fit(data())
    for method in ("cv+", "cqr", "distributional"):
        with pytest.raises(ValueError, match="only for standard-model"):
            mb.conformal_interval(model, method=method, score=score)
    with pytest.raises(ValueError, match="score must"):
        mb.conformal_interval(model, score="unknown")
    for mu in (-1., np.inf, np.nan):
        with pytest.raises(ValueError, match="nonnegative predicted means"):
            _rust.conformal_poisson_scores(np.array([0.]), np.array([mu]), .9, score)


def test_extreme_means_are_finite_or_explicitly_refused():
    lo, hi = _rust.conformal_poisson_band(np.array([1e300]), .9, 2., "poisson_normalized")
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
    with pytest.raises(RuntimeError, match="exact-count quantile range"):
        _rust.conformal_poisson_band(np.array([1e300]), .9, 2., "poisson_quantile")
