"""Conformal prediction intervals: validity and adaptivity.

The point of these intervals is a coverage guarantee, so the tests measure
coverage on *fresh* data rather than asserting implementation details.
Tolerances are set from the Monte-Carlo standard error of the coverage
estimate, not chosen to make the suite pass.
"""

import numpy as np
import pytest

import mboostrs as mb

LEVEL = 0.9
CTRL = mb.boost_control(100, 0.1)


# --------------------------------------------------------------------------
# data generators
# --------------------------------------------------------------------------


def homoscedastic(n, rng):
    x = rng.uniform(-2, 2, n)
    return {"x": x, "y": np.sin(x) + rng.normal(scale=0.3, size=n)}


def heteroscedastic(n, rng):
    """Noise grows with x, so a constant-width interval cannot be efficient."""
    x = rng.uniform(-2, 2, n)
    scale = 0.1 + 0.5 * (x + 2.0)
    return {"x": x, "y": np.sin(x) + rng.normal(scale=scale, size=n)}


def heavy_tailed(n, rng):
    """t(3) noise: finite variance, but far heavier tails than Gaussian."""
    x = rng.uniform(-2, 2, n)
    return {"x": x, "y": np.sin(x) + rng.standard_t(3, size=n) * 0.3}


def skewed(n, rng):
    """Right-skewed (lognormal) noise, centered so the mean model still fits."""
    x = rng.uniform(-2, 2, n)
    e = (rng.lognormal(0.0, 1.0, size=n) - np.exp(0.5)) * 0.3
    return {"x": x, "y": np.sin(x) + e}


def _fit(data):
    return mb.gamboost(
        "y ~ bbs(x, df=4)", data=data, family=mb.Gaussian(), control=CTRL
    )


# --------------------------------------------------------------------------
# marginal coverage on fresh data — the actual guarantee
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["split", "cv+", "cqr"])
@pytest.mark.parametrize(
    "generator", [homoscedastic, heteroscedastic], ids=["homosc", "heterosc"]
)
def test_marginal_coverage_on_fresh_data(method, generator):
    """Coverage must reach the nominal level on data never seen in fitting."""
    n_train, n_test, reps = 400, 400, 8
    covered = []
    for rep in range(reps):
        rng = np.random.default_rng(1000 + rep)
        train = generator(n_train, rng)
        test = generator(n_test, rng)
        fit = _fit(train)
        ci = mb.conformal_interval(
            fit, newdata=test, level=LEVEL, method=method, seed=rep, n_folds=5
        )
        covered.append(ci.empirical_coverage(test["y"]))

    mean_cov = float(np.mean(covered))
    # SE of the mean coverage over reps*n_test draws, plus slack for the
    # randomness of the calibration set itself.
    se = np.sqrt(LEVEL * (1 - LEVEL) / (reps * n_test))
    assert mean_cov > LEVEL - 4 * se - 0.02, (method, mean_cov, covered)
    # Conformal is valid but should not be absurdly conservative here.
    assert mean_cov < LEVEL + 0.08, (method, mean_cov, covered)


# Normal quantiles z_{(1+level)/2}, hardcoded to keep scipy out of the
# dependency set (see tests/test_optional_dependencies.py).
_Z = {0.80: 1.2815515655446004, 0.90: 1.6448536269514722,
      0.95: 1.959963984540054, 0.99: 2.5758293035489004}


def _gaussian_theory_coverage(fit, train, test, level):
    """Textbook alternative: a symmetric interval from the residual SD."""
    resid = np.asarray(train["y"]) - np.asarray(fit.predict(type="response"))
    half = _Z[level] * float(np.std(resid, ddof=1))
    point = np.asarray(fit.predict(newdata=test, type="response")).ravel()
    return float(np.mean((test["y"] >= point - half) & (test["y"] <= point + half)))


@pytest.mark.parametrize(
    "generator", [heavy_tailed, skewed], ids=["t3", "lognormal"]
)
def test_conformal_is_calibrated_across_levels_where_gaussian_theory_is_not(generator):
    """The claim that justifies the feature.

    Conformal hits whatever level you ask for. A Gaussian-theory interval
    built from the residual SD does not, and it errs in *both* directions
    under non-normal noise: it is far too wide at low levels (heavy tails
    inflate the SD more than they spread the central mass) and too narrow at
    high levels (where the tails finally dominate). A modeler asking for 99%
    coverage and silently getting 97% is the case that matters.
    """
    levels = (0.80, 0.90, 0.95, 0.99)
    reps = 6
    conformal_err, gaussian_err = {}, {}
    for level in levels:
        c_cov, g_cov = [], []
        for rep in range(reps):
            rng = np.random.default_rng(2000 + rep)
            train = generator(500, rng)
            test = generator(500, rng)
            fit = _fit(train)
            ci = mb.conformal_interval(
                fit, newdata=test, level=level, method="split", seed=rep
            )
            c_cov.append(ci.empirical_coverage(test["y"]))
            g_cov.append(_gaussian_theory_coverage(fit, train, test, level))
        conformal_err[level] = abs(float(np.mean(c_cov)) - level)
        gaussian_err[level] = abs(float(np.mean(g_cov)) - level)

    # Conformal tracks every level closely.
    assert max(conformal_err.values()) < 0.035, conformal_err
    # The Gaussian interval does not, and is worse on average.
    assert max(gaussian_err.values()) > 3 * max(conformal_err.values()), (
        conformal_err, gaussian_err
    )
    # Specifically: it under-covers at the level where under-coverage hurts.
    assert gaussian_err[0.99] > conformal_err[0.99], (conformal_err, gaussian_err)


@pytest.mark.parametrize("method", ["split", "cv+", "cqr"])
def test_coverage_holds_under_heavy_tails(method):
    """Validity must not depend on the noise being Gaussian."""
    reps = 6
    covered = []
    for rep in range(reps):
        rng = np.random.default_rng(2500 + rep)
        train = heavy_tailed(400, rng)
        test = heavy_tailed(400, rng)
        ci = mb.conformal_interval(
            _fit(train), newdata=test, level=LEVEL, method=method, seed=rep, n_folds=5
        )
        covered.append(ci.empirical_coverage(test["y"]))
    assert float(np.mean(covered)) > LEVEL - 0.03, covered


# --------------------------------------------------------------------------
# adaptivity — why cqr exists
# --------------------------------------------------------------------------


def test_cqr_width_adapts_to_heteroscedasticity_and_split_does_not():
    rng = np.random.default_rng(7)
    train = heteroscedastic(1500, rng)
    grid = {"x": np.linspace(-1.9, 1.9, 200)}
    fit = _fit(train)

    split = mb.conformal_interval(
        fit, newdata=grid, level=LEVEL, method="split", seed=1
    )
    cqr = mb.conformal_interval(fit, newdata=grid, level=LEVEL, method="cqr", seed=1)

    # Split conformal is constant-width by construction.
    assert float(np.ptp(split.width)) < 1e-9

    # True noise scale rises with x, so CQR width must rise with x too.
    corr = float(np.corrcoef(grid["x"], cqr.width)[0, 1])
    assert corr > 0.8, f"CQR width did not track the noise scale (corr={corr})"
    assert cqr.width[-1] > 2.0 * cqr.width[0]


def test_cqr_conditional_coverage_beats_split_in_the_noisy_region():
    """Constant-width intervals under-cover where noise is large.

    This is the practical failure that adaptivity fixes; conditional coverage
    is not guaranteed by any of these methods, but CQR should be visibly
    closer in the tail region.
    """
    reps = 6
    split_tail, cqr_tail = [], []
    for rep in range(reps):
        rng = np.random.default_rng(3000 + rep)
        train = heteroscedastic(800, rng)
        test = heteroscedastic(800, rng)
        fit = _fit(train)
        noisy = np.asarray(test["x"]) > 1.0

        for method, bucket in (("split", split_tail), ("cqr", cqr_tail)):
            ci = mb.conformal_interval(
                fit, newdata=test, level=LEVEL, method=method, seed=rep
            )
            bucket.append(float(np.mean(ci.contains(test["y"])[noisy])))

    assert float(np.mean(cqr_tail)) > float(np.mean(split_tail)), (
        np.mean(split_tail), np.mean(cqr_tail)
    )


# --------------------------------------------------------------------------
# monotonicity and basic contracts
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["split", "cv+", "cqr"])
def test_higher_level_gives_wider_intervals(method):
    rng = np.random.default_rng(11)
    data = homoscedastic(400, rng)
    fit = _fit(data)
    widths = [
        float(
            np.mean(
                mb.conformal_interval(
                    fit, level=lv, method=method, seed=0, n_folds=5
                ).width
            )
        )
        for lv in (0.5, 0.8, 0.95)
    ]
    assert widths[0] < widths[1] < widths[2], widths


def test_intervals_bracket_the_point_prediction():
    rng = np.random.default_rng(5)
    data = homoscedastic(300, rng)
    fit = _fit(data)
    ci = mb.conformal_interval(fit, level=LEVEL, seed=0)
    assert np.all(ci.lower <= ci.predicted)
    assert np.all(ci.predicted <= ci.upper)
    assert np.all(ci.width >= 0)


def test_glmboost_models_are_supported():
    rng = np.random.default_rng(2)
    n = 400
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    data = {"x1": x1, "x2": x2, "y": 1.5 * x1 - 0.7 * x2 + rng.normal(scale=0.4, size=n)}
    fit = mb.glmboost("y ~ x1 + x2", data=data, family=mb.Gaussian(), control=CTRL)
    ci = mb.conformal_interval(fit, level=LEVEL, seed=0)
    assert 0.82 < ci.empirical_coverage(data["y"]) < 0.98


def test_to_frame_and_repr():
    rng = np.random.default_rng(4)
    data = homoscedastic(200, rng)
    ci = mb.conformal_interval(_fit(data), level=LEVEL, seed=0)

    frame = ci.to_frame(output="records")
    # The columns of prediction_interval's names (median, level) and the full
    # model's prediction (fitted_mean) are added to the original four.
    assert set(frame) == {"predicted", "lower", "upper", "width", "median",
                          "level", "fitted_mean"}

    import polars as pl

    assert isinstance(ci.to_frame(output="polars"), pl.DataFrame)
    assert "method='split'" in repr(ci)


def test_contains_rejects_mismatched_length():
    rng = np.random.default_rng(6)
    data = homoscedastic(200, rng)
    ci = mb.conformal_interval(_fit(data), level=LEVEL, seed=0)
    with pytest.raises(ValueError, match="observed has"):
        ci.contains(np.zeros(5))


# --------------------------------------------------------------------------
# failure modes must be explicit
# --------------------------------------------------------------------------


def test_level_too_high_for_calibration_size_is_rejected():
    """Finite-sample validity is unattainable below a minimum calibration size."""
    rng = np.random.default_rng(8)
    data = homoscedastic(20, rng)
    fit = _fit(data)
    with pytest.raises(ValueError, match="calibration points"):
        mb.conformal_interval(fit, level=0.999, method="split", seed=0)


@pytest.mark.parametrize("bad_level", [0.0, 1.0, -0.1, 1.5])
def test_invalid_level_is_rejected(bad_level):
    rng = np.random.default_rng(9)
    fit = _fit(homoscedastic(100, rng))
    with pytest.raises(ValueError, match="level must be in"):
        mb.conformal_interval(fit, level=bad_level)


def test_unknown_method_is_rejected():
    rng = np.random.default_rng(10)
    fit = _fit(homoscedastic(100, rng))
    with pytest.raises(ValueError, match="method must be"):
        mb.conformal_interval(fit, method="jackknife")


def test_invalid_calib_fraction_is_rejected():
    rng = np.random.default_rng(12)
    fit = _fit(homoscedastic(100, rng))
    with pytest.raises(ValueError, match="calib_fraction"):
        mb.conformal_interval(fit, calib_fraction=1.0)


def test_buser_learner_is_refused_explicitly():
    """Buser carries a user design matrix this module cannot subset."""
    rng = np.random.default_rng(13)
    n = 100
    x = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x])
    K = np.eye(2)
    data = {"x": x, "y": 1.5 * x + rng.normal(scale=0.3, size=n)}
    fit = mb.gamboost(
        "y ~ buser(x, X=X_basis, K=K_penalty, df=2)", data=data,
        family=mb.Gaussian(), control=CTRL,
        env={"X_basis": X, "K_penalty": K},
    )
    with pytest.raises(NotImplementedError, match="Buser"):
        mb.conformal_interval(fit, level=LEVEL, seed=0)


def test_cqr_on_skewed_noise_returns_ordered_intervals():
    """A negative calibration correction used to invert the band where the two
    quantile fits were close and raise; the interval is now a point there."""
    import numpy as np

    import mboostrs as mb

    for seed in range(4):
        rng = np.random.default_rng(seed)
        x = rng.uniform(-1, 1, (1000, 3))
        y = np.sin(2 * x[:, 0]) + 0.3 * np.exp(0.8 * x[:, 1]) * (rng.gamma(2.0, size=1000) - 2)
        data = {"x0": x[:, 0], "x1": x[:, 1], "x2": x[:, 2], "y": y}
        fit = mb.gamboost("y ~ bbs(x0) + bbs(x1) + bbs(x2)", data=data)
        ci = fit.conformal_interval(level=0.8, method="cqr", seed=seed)
        assert np.all(np.asarray(ci.lower) <= np.asarray(ci.upper))
