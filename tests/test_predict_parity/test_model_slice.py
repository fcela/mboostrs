"""R-parity tests for model slicing (__getitem__)."""

import mboostrs as mboost
import numpy as np
import pytest
from mboostrs.families import Gaussian

rtol_glm = 1e-10
rtol_gam = 1e-6
atol = 1e-9


def _run_r_glmboost_slice(mstop, nu, n=200, seed=42):
    import rpy2.robjects as ro

    r_code = """
    function(mstop, nu, n, seed) {
        set.seed(seed)
        x1 <- rnorm(n)
        x2 <- rnorm(n)
        y <- 1.5*x1 + 0.8*x2 + rnorm(n, sd=0.3)
        df <- data.frame(x1 = x1, x2 = x2, y = y)
        fit <- mboost::glmboost(
            y ~ x1 + x2, data = df,
            family = mboost::Gaussian(), center = TRUE,
            control = mboost::boost_control(mstop = mstop, nu = nu)
        )
        k <- as.integer(mstop / 2)
        fit_k <- fit[k]
        fitted_k <- as.numeric(stats::predict(fit_k, type = "link"))
        risk_k <- as.numeric(mboost::risk(fit_k))
        list(
            x1 = as.numeric(x1), x2 = as.numeric(x2), y = as.numeric(y),
            fitted_k = fitted_k, risk_k = risk_k, k = k
        )
    }
    """
    r_func = ro.r(r_code)
    result = r_func(mstop, nu, n, seed)
    return {
        "x1": np.asarray(result[0], dtype=np.float64),
        "x2": np.asarray(result[1], dtype=np.float64),
        "y": np.asarray(result[2], dtype=np.float64),
        "fitted_k": np.asarray(result[3], dtype=np.float64),
        "risk_k": np.asarray(result[4], dtype=np.float64),
        "k": int(result[5][0]),
    }


@pytest.mark.r_parity
def test_glmboost_slice_fitted():
    r = _run_r_glmboost_slice(50, 0.1)
    data = {"y": r["y"], "x1": r["x1"], "x2": r["x2"]}
    model = mboost.glmboost(
        "y ~ x1 + x2", data=data, family=Gaussian(),
        control=mboost.boost_control(50, 0.1),
    )
    k = r["k"]
    model_k = model[:k]
    np.testing.assert_allclose(model_k.fitted_, r["fitted_k"], rtol=rtol_glm,
                               err_msg=f"model[:{k}].fitted_ does not match R")


@pytest.mark.r_parity
def test_glmboost_slice_risk():
    r = _run_r_glmboost_slice(50, 0.1)
    data = {"y": r["y"], "x1": r["x1"], "x2": r["x2"]}
    model = mboost.glmboost(
        "y ~ x1 + x2", data=data, family=Gaussian(),
        control=mboost.boost_control(50, 0.1),
    )
    k = r["k"]
    model_k = model[:k]
    np.testing.assert_allclose(model_k.risk_, r["risk_k"], rtol=rtol_glm,
                               err_msg=f"model[:{k}].risk_ does not match R")


@pytest.mark.r_parity
def test_glmboost_slice_mstop():
    r = _run_r_glmboost_slice(50, 0.1)
    data = {"y": r["y"], "x1": r["x1"], "x2": r["x2"]}
    model = mboost.glmboost(
        "y ~ x1 + x2", data=data, family=Gaussian(),
        control=mboost.boost_control(50, 0.1),
    )
    k = r["k"]
    model_k = model[:k]
    assert model_k.mstop == k, f"model[:{k}].mstop should be {k}, got {model_k.mstop}"


def _run_r_gamboost_slice(mstop, nu, n=100, seed=42):
    import rpy2.robjects as ro

    r_code = """
    function(mstop, nu, n, seed) {
        library(mboost)
        set.seed(seed)
        x1 <- runif(n, 0, 1)
        x2 <- rnorm(n)
        y <- sin(2 * pi * x1) + 0.5 * x2 + rnorm(n, sd = 0.3)
        df <- data.frame(x1 = x1, x2 = x2, y = y)
        fit <- gamboost(
            y ~ bbs(x1, df = 4, knots = 20) + bols(x2),
            data = df, family = Gaussian(),
            control = boost_control(mstop = mstop, nu = nu)
        )
        k <- as.integer(mstop / 2)
        fit_k <- fit[k]
        fitted_k <- as.numeric(stats::predict(fit_k, type = "link"))
        risk_k <- as.numeric(risk(fit_k))
        list(
            x1 = as.numeric(x1), x2 = as.numeric(x2), y = as.numeric(y),
            fitted_k = fitted_k, risk_k = risk_k, k = k
        )
    }
    """
    r_func = ro.r(r_code)
    result = r_func(mstop, nu, n, seed)
    return {
        "x1": np.asarray(result[0], dtype=np.float64),
        "x2": np.asarray(result[1], dtype=np.float64),
        "y": np.asarray(result[2], dtype=np.float64),
        "fitted_k": np.asarray(result[3], dtype=np.float64),
        "risk_k": np.asarray(result[4], dtype=np.float64),
        "k": int(result[5][0]),
    }


@pytest.mark.r_parity
def test_gamboost_slice_fitted():
    r = _run_r_gamboost_slice(50, 0.1)
    data = {"y": r["y"], "x1": r["x1"], "x2": r["x2"]}
    model = mboost.gamboost(
        "y ~ bbs(x1, df=4, knots=20) + bols(x2)", data=data,
        family=Gaussian(), control=mboost.boost_control(50, 0.1),
    )
    k = r["k"]
    model_k = model[:k]
    np.testing.assert_allclose(model_k.fitted_, r["fitted_k"], rtol=rtol_gam, atol=atol,
                               err_msg=f"gamboost model[:{k}].fitted_ does not match R")


@pytest.mark.r_parity
def test_gamboost_slice_risk():
    r = _run_r_gamboost_slice(50, 0.1)
    data = {"y": r["y"], "x1": r["x1"], "x2": r["x2"]}
    model = mboost.gamboost(
        "y ~ bbs(x1, df=4, knots=20) + bols(x2)", data=data,
        family=Gaussian(), control=mboost.boost_control(50, 0.1),
    )
    k = r["k"]
    model_k = model[:k]
    np.testing.assert_allclose(model_k.risk_, r["risk_k"], rtol=rtol_gam, atol=atol,
                               err_msg=f"gamboost model[:{k}].risk_ does not match R")


@pytest.mark.r_parity
def test_gamboost_slice_mstop():
    r = _run_r_gamboost_slice(50, 0.1)
    data = {"y": r["y"], "x1": r["x1"], "x2": r["x2"]}
    model = mboost.gamboost(
        "y ~ bbs(x1, df=4, knots=20) + bols(x2)", data=data,
        family=Gaussian(), control=mboost.boost_control(50, 0.1),
    )
    k = r["k"]
    model_k = model[:k]
    assert model_k.mstop == k, f"gamboost model[:{k}].mstop should be {k}, got {model_k.mstop}"


def test_mstop_function():
    risk_path = np.array([10.0, 8.0, 5.0, 3.0, 4.0, 6.0])
    assert np.argmin(risk_path[1:]) + 1 == 3
