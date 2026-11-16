"""Tests for the standalone diagnostics module (standard + LSS models)."""

import numpy as np
import pytest
from mboostrs.diagnostics import (
    calibration_frame,
    coef_all,
    coverage,
    deviance_increment,
    ipc_weights,
    outlier_frame,
    predict_all,
    q_stats,
    quantile_residuals,
    residual_frame,
    worm_data,
)
from mboostrs.gamlss import PoissonLSS

import mboostrs as mb
from conftest import r_call


@pytest.fixture
def gaussian_glmboost():
    rng = np.random.default_rng(42)
    n = 50
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 + 0.5 * x1 - 0.3 * x2 + rng.normal(scale=0.3, size=n)
    data = {"y": y, "x1": x1, "x2": x2}
    mod = mb.glmboost("y ~ x1 + x2", data=data, family=mb.Gaussian(),
                      control=mb.boost_control(50, 0.1))
    return mod, data


@pytest.fixture
def poisson_glmboostlss():
    rng = np.random.default_rng(99)
    n = 40
    x1 = rng.normal(size=n)
    mu = np.exp(0.3 + 0.4 * x1)
    y = rng.poisson(mu)
    data = {"y": y.astype(float), "x1": x1}
    mod = mb.glmboostLSS("y ~ x1", data=data, families=PoissonLSS(),
                         control=mb.boost_control(20, 0.1))
    return mod, data


def test_predict_all_standard(gaussian_glmboost):
    mod, data = gaussian_glmboost
    df = predict_all(mod)
    assert len(df) == 50
    assert "fitted" in df.columns
    assert "y" in df.columns


def test_predict_all_accepts_predictor_only_newdata(gaussian_glmboost):
    mod, data = gaussian_glmboost
    df = predict_all(mod, newdata={"x1": data["x1"][:7], "x2": data["x2"][:7]})
    assert len(df) == 7
    assert "fitted" in df.columns
    assert "y" not in df.columns


def test_predict_all_lss(poisson_glmboostlss):
    mod, data = poisson_glmboostlss
    df = predict_all(mod)
    assert len(df) == 40
    assert "mu" in df.columns


def test_coef_all_standard(gaussian_glmboost):
    mod, data = gaussian_glmboost
    df = coef_all(mod)
    assert "learner" in df.columns
    assert "coefficient" in df.columns


def test_coef_all_vector_valued_gam_coefficients():
    x = np.linspace(0.0, 1.0, 40)
    y = np.sin(4.0 * x)
    mod = mb.gamboost(
        "y ~ bbs(x)", data={"y": y, "x": x},
        control=mb.boost_control(5, 0.1),
    )
    df = coef_all(mod)
    assert len(df) > 0
    assert "component" in df.columns
    assert np.isfinite(df["coefficient"].to_numpy()).all()


def test_deviance_increment_standard(gaussian_glmboost):
    mod, data = gaussian_glmboost
    dev = deviance_increment(mod)
    assert len(dev) == 50
    assert np.all(dev >= 0)


def test_deviance_increment_rejects_unsupported_loss():
    x = np.linspace(-1.0, 1.0, 30)
    mod = mb.glmboost(
        "y ~ x", data={"y": x**2, "x": x}, family=mb.Laplace(),
        control=mb.boost_control(5, 0.1),
    )
    with pytest.raises(NotImplementedError, match="Laplace"):
        deviance_increment(mod)


def test_lss_pearson_residuals_and_nonlocation_guard():
    rng = np.random.default_rng(123)
    x = rng.normal(size=50)
    y = 0.5 * x + rng.normal(scale=0.4, size=50)
    mod = mb.glmboostLSS(
        "y ~ x", data={"y": y, "x": x}, families=mb.GaussianLSS(),
        control=mb.boost_control(10, 0.1),
    )
    residuals = mod.residuals(parameter="mu", type="pearson")
    assert residuals.shape == y.shape
    assert np.isfinite(residuals).all()
    with pytest.raises(NotImplementedError, match="location parameter"):
        mod.residuals(parameter="sigma", type="response")


def test_quantile_residuals_standard_gaussian(gaussian_glmboost):
    mod, data = gaussian_glmboost
    resid = quantile_residuals(mod)
    assert len(resid) == 50
    assert np.isfinite(resid).all()


def test_quantile_residuals_standard_poisson_raises():
    rng = np.random.default_rng(7)
    n = 30
    x1 = rng.normal(size=n)
    y = rng.poisson(np.exp(0.2 * x1)).astype(float)
    mod = mb.glmboost("y ~ x1", data={"y": y, "x1": x1}, family=mb.Poisson(),
                      control=mb.boost_control(20, 0.1))
    with pytest.raises(NotImplementedError):
        quantile_residuals(mod)


def test_residual_frame_standard(gaussian_glmboost):
    mod, data = gaussian_glmboost
    for kind in ("response", "pearson"):
        df = residual_frame(mod, kind=kind)
        assert len(df) == 50
        assert "residual" in df.columns
        assert "y" in df.columns


def test_q_stats_standard(gaussian_glmboost):
    mod, data = gaussian_glmboost
    stats = q_stats(mod, xvar="x1", n_intervals=5)
    assert len(stats) <= 5
    assert "mean" in stats.columns
    assert "variance" in stats.columns


def test_q_stats_lss(poisson_glmboostlss):
    mod, data = poisson_glmboostlss
    stats = q_stats(mod, xvar="x1", n_intervals=4, randomized=False)
    assert len(stats) <= 4
    assert "skewness" in stats.columns


def test_worm_data(gaussian_glmboost):
    mod, data = gaussian_glmboost
    wd = worm_data(mod)
    assert len(wd) == 50
    assert "deviation" in wd.columns
    assert "expected" in wd.columns


def test_lss_worm_plot_method(poisson_glmboostlss):
    import altair as alt

    mod, data = poisson_glmboostlss
    chart = mod.worm_plot(xvar="x1", n_intervals=4)
    assert isinstance(chart, alt.VConcatChart)
    spec = chart.to_dict()
    assert spec

    single = mod.worm_plot(n_intervals=1)
    assert single.to_dict()

    from mboostrs.plot import plot_worm

    chart_seeded = mod.worm_plot(xvar="x1", n_intervals=4, random_state=0)
    direct = plot_worm(mod, xvar="x1", n_intervals=4, random_state=0)
    assert direct.to_dict() == chart_seeded.to_dict()


def test_calibration_frame(gaussian_glmboost):
    mod, data = gaussian_glmboost
    cf = calibration_frame(mod, xvar="x1", bins=5)
    assert len(cf) <= 5
    assert "observed_mean" in cf.columns
    assert "fitted_mean" in cf.columns


def test_outlier_frame(gaussian_glmboost):
    mod, data = gaussian_glmboost
    of = outlier_frame(mod)
    assert len(of) == 50
    assert "residual" in of.columns


def test_coverage():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    lower = np.array([0.5, 1.5, 2.5, 3.5])
    upper = np.array([1.5, 2.5, 3.5, 4.5])
    rate, width = coverage(y, lower, upper)
    assert rate == 1.0
    assert width == 1.0


def test_residual_frame_lss(poisson_glmboostlss):
    mod, data = poisson_glmboostlss
    df = residual_frame(mod, kind="quantile", randomized=False)
    assert len(df) == 40
    assert "residual" in df.columns
    assert "mu" in df.columns


def test_ipc_weights_censored_get_zero():
    time = np.array([1.0, 2.0, 3.0, 4.0])
    status = np.array([1.0, 0.0, 1.0, 0.0])
    w = ipc_weights(time, status)
    assert w[1] == 0.0
    assert w[3] == 0.0
    assert w[0] > 0.0
    assert w[2] > 0.0


def test_ipc_weights_all_events_equal_one():
    # With no censoring the censoring-distribution KM never drops, so every
    # weight is exactly status / 1 = status.
    rng = np.random.default_rng(7)
    time = rng.exponential(size=30)
    status = np.ones_like(time)
    w = ipc_weights(time, status)
    np.testing.assert_allclose(w, 1.0)


def test_ipc_weights_respects_maxweight():
    rng = np.random.default_rng(3)
    n = 200
    time = rng.exponential(scale=0.5, size=n)
    status = rng.binomial(1, 0.3, size=n).astype(float)
    w = ipc_weights(time, status, maxweight=2.0)
    assert np.all(w <= 2.0)
    assert np.any(w == 2.0)


def test_ipc_weights_rejects_bad_input():
    with pytest.raises(ValueError):
        ipc_weights(np.array([1.0, -1.0]), np.array([1.0, 0.0]))
    with pytest.raises(ValueError):
        ipc_weights(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        ipc_weights(np.array([1.0]), np.array([1.0, 0.0]))


@pytest.mark.r_parity
def test_ipc_weights_matches_r_mboost():
    import rpy2.robjects as ro

    reference = ro.r(
        """
        function() {
            if (!requireNamespace("mboost", quietly = TRUE)) stop("no mboost")
            if (!requireNamespace("survival", quietly = TRUE)) stop("no survival")
            suppressMessages(library(mboost))
            suppressMessages(library(survival))
            set.seed(11)
            n <- 60
            time <- round(rexp(n, rate = 0.3), 2)
            status <- rbinom(n, 1, 0.65)
            # force exact ties between an event and a censoring at the same time
            time[5] <- time[3]; status[5] <- 1 - status[3]
            time[10] <- time[7]; status[10] <- 1 - status[7]
            w <- IPCweights(Surv(time, status), maxweight = 5)
            list(time = time, status = as.double(status), w = as.double(w))
        }
        """
    )
    result = r_call(reference, packages=("mboost", "survival"))

    time = np.asarray(result.rx2("time"), dtype=np.float64)
    status = np.asarray(result.rx2("status"), dtype=np.float64)
    w_r = np.asarray(result.rx2("w"), dtype=np.float64)

    w_py = ipc_weights(time, status, maxweight=5.0)
    np.testing.assert_allclose(w_py, w_r, rtol=1e-10, atol=1e-10)
