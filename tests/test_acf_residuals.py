"""Serial correlation left in the residuals, as `gamlss::acfResid` reports it.

Boosting assumes observations are independent given the predictors. When they
are ordered in time, structure the model failed to absorb shows up as
correlation between neighbouring residuals; checking the powers as well
separates which moment is still dependent.
"""

import numpy as np
import pytest
from mboostrs.diagnostics import _durbin_levinson, acf_residuals

import mboostrs as mb


def _ar1_series(n=300, phi=0.7, seed=9):
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=n)
    series = np.zeros(n)
    for i in range(1, n):
        series[i] = phi * series[i - 1] + noise[i]
    return series


def test_acf_and_pacf_match_r():
    """Reference values from R's own `acf` and `pacf` on the same series."""
    series = _ar1_series()
    n = len(series)
    centered = series - series.mean()
    denominator = float(np.dot(centered, centered))
    acf = np.array([
        float(np.dot(centered[: n - lag], centered[lag:])) / denominator
        for lag in range(7)
    ])
    pacf = _durbin_levinson(acf, 6)

    np.testing.assert_allclose(
        acf[1:],
        [0.6903225885, 0.5253570375, 0.3828005676, 0.2949544570,
         0.2486081823, 0.2000190407],
        rtol=0, atol=1e-9,
    )
    # An AR(1) has one non-zero partial autocorrelation, at lag 1.
    np.testing.assert_allclose(
        pacf[1:],
        [0.6903225885, 0.0932492517, -0.0200789138, 0.0248554769,
         0.0489415553, -0.0037330070],
        rtol=0, atol=1e-9,
    )


def _fit(y, x):
    return mb.glmboost("y ~ x", data={"x": x, "y": y}, family=mb.Gaussian(),
                       control=mb.boost_control(100, 0.1))


def test_independent_residuals_stay_inside_the_band():
    rng = np.random.default_rng(1)
    n = 400
    x = rng.normal(size=n)
    frame = acf_residuals(_fit(1.5 * x + rng.normal(0, 0.5, n), x),
                          max_lag=5, output="records")
    outside = [row for row in frame if abs(row["acf"]) > row["bound"]]
    # A 95% band on 20 (power, lag) cells: a couple of crossings is expected.
    assert len(outside) <= 3


def test_serially_correlated_residuals_are_flagged_at_lag_one():
    """The case the diagnostic exists for: the model fits the mean correctly,
    but the errors are autocorrelated, which no residual-versus-fitted plot
    would show."""
    rng = np.random.default_rng(4)
    n = 400
    x = rng.normal(size=n)
    frame = acf_residuals(_fit(1.5 * x + _ar1_series(n, 0.7, seed=4), x),
                          max_lag=6, output="records")
    level_lag1 = next(r for r in frame if r["power"] == 1 and r["lag"] == 1)
    assert level_lag1["acf"] > 4 * level_lag1["bound"]


def test_variance_dependence_shows_in_the_squared_residuals():
    """Level correlation and variance correlation are different failures, and
    the powers are what tells them apart."""
    rng = np.random.default_rng(6)
    n = 500
    x = rng.normal(size=n)
    # Independent in level, but the scale follows a slow cycle.
    scale = 0.4 + 0.35 * (1 + np.sin(np.linspace(0, 6 * np.pi, n)))
    frame = acf_residuals(_fit(1.5 * x + rng.normal(0, scale), x),
                          max_lag=8, output="records")
    squared = [r for r in frame if r["power"] == 2]
    assert max(abs(r["acf"]) for r in squared) > squared[0]["bound"]


def test_default_max_lag_follows_r():
    rng = np.random.default_rng(2)
    n = 200
    x = rng.normal(size=n)
    frame = acf_residuals(_fit(x + rng.normal(0, 0.5, n), x), output="records")
    # R's acf default is 10 * log10(n).
    assert max(r["lag"] for r in frame) == int(np.floor(10 * np.log10(n)))


def test_powers_are_configurable():
    rng = np.random.default_rng(2)
    n = 200
    x = rng.normal(size=n)
    frame = acf_residuals(_fit(x + rng.normal(0, 0.5, n), x),
                          max_lag=4, powers=(1, 2), output="records")
    assert {r["power"] for r in frame} == {1, 2}
    assert len(frame) == 8


def test_too_few_observations_is_rejected():
    rng = np.random.default_rng(2)
    x = rng.normal(size=2)
    with pytest.raises(ValueError, match="at least three observations"):
        acf_residuals(_fit(x, x), max_lag=1)
