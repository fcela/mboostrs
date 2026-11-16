"""R-parity tests for glmboost predict()."""

import mboostrs as mboost
import numpy as np
import pytest
from mboostrs.families import Binomial, Gaussian

rtol = 1e-10


def _run_r_glmboost_predict(mstop, nu, n=200, seed=42):
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
        pred_link <- as.numeric(stats::predict(fit, type = "link"))
        pred_response <- as.numeric(stats::predict(fit, type = "response"))
        pred_k <- as.numeric(stats::predict(fit[as.integer(mstop/2)], type = "link"))
        fitted_k <- as.numeric(stats::predict(fit[as.integer(mstop/2)], type = "link"))
        list(
            x1 = as.numeric(x1), x2 = as.numeric(x2), y = as.numeric(y),
            pred_link = pred_link, pred_response = pred_response,
            pred_k = pred_k, fitted_k = fitted_k,
            k = as.integer(mstop/2)
        )
    }
    """
    r_func = ro.r(r_code)
    result = r_func(mstop, nu, n, seed)
    return {
        "x1": np.asarray(result[0], dtype=np.float64),
        "x2": np.asarray(result[1], dtype=np.float64),
        "y": np.asarray(result[2], dtype=np.float64),
        "pred_link": np.asarray(result[3], dtype=np.float64),
        "pred_response": np.asarray(result[4], dtype=np.float64),
        "pred_k": np.asarray(result[5], dtype=np.float64),
        "fitted_k": np.asarray(result[6], dtype=np.float64),
        "k": int(result[7][0]),
    }


@pytest.mark.r_parity
def test_glmboost_predict_link():
    mstop_val = 50
    nu_val = 0.1
    r = _run_r_glmboost_predict(mstop_val, nu_val)

    data = {"y": r["y"], "x1": r["x1"], "x2": r["x2"]}
    model = mboost.glmboost(
        "y ~ x1 + x2", data=data, family=Gaussian(),
        control=mboost.boost_control(mstop_val, nu_val),
    )

    pred = model.predict(type="link")
    np.testing.assert_allclose(pred, r["pred_link"], rtol=rtol,
                               err_msg="predict(type='link') does not match R")


@pytest.mark.r_parity
def test_glmboost_predict_response_gaussian():
    mstop_val = 50
    nu_val = 0.1
    r = _run_r_glmboost_predict(mstop_val, nu_val)

    data = {"y": r["y"], "x1": r["x1"], "x2": r["x2"]}
    model = mboost.glmboost(
        "y ~ x1 + x2", data=data, family=Gaussian(),
        control=mboost.boost_control(mstop_val, nu_val),
    )

    pred = model.predict(type="response")
    np.testing.assert_allclose(pred, r["pred_response"], rtol=rtol,
                               err_msg="predict(type='response') does not match R for Gaussian")


@pytest.mark.r_parity
def test_glmboost_predict_mstop():
    mstop_val = 50
    nu_val = 0.1
    r = _run_r_glmboost_predict(mstop_val, nu_val)

    data = {"y": r["y"], "x1": r["x1"], "x2": r["x2"]}
    model = mboost.glmboost(
        "y ~ x1 + x2", data=data, family=Gaussian(),
        control=mboost.boost_control(mstop_val, nu_val),
    )

    k = r["k"]
    pred = model.predict(mstop=k, type="link")
    np.testing.assert_allclose(pred, r["pred_k"], rtol=rtol,
                               err_msg=f"predict(mstop={k}) does not match R")


@pytest.mark.r_parity
def test_glmboost_predict_newdata():
    mstop_val = 50
    nu_val = 0.1
    n = 200

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
        newdata <- data.frame(x1 = c(0.5, -0.3), x2 = c(1.0, 0.2))
        pred_new <- as.numeric(stats::predict(fit, newdata = newdata, type = "link"))
        list(
            x1 = as.numeric(x1), x2 = as.numeric(x2), y = as.numeric(y),
            pred_new = pred_new
        )
    }
    """
    r_func = ro.r(r_code)
    result = r_func(mstop_val, nu_val, n, 42)

    data = {
        "y": np.asarray(result[2], dtype=np.float64),
        "x1": np.asarray(result[0], dtype=np.float64),
        "x2": np.asarray(result[1], dtype=np.float64),
    }
    model = mboost.glmboost(
        "y ~ x1 + x2", data=data, family=Gaussian(),
        control=mboost.boost_control(mstop_val, nu_val),
    )

    newdata = {"x1": np.array([0.5, -0.3]), "x2": np.array([1.0, 0.2])}
    pred = model.predict(newdata=newdata, type="link")
    r_pred_new = np.asarray(result[3], dtype=np.float64)
    np.testing.assert_allclose(pred, r_pred_new, rtol=1e-8,
                               err_msg="predict(newdata=...) does not match R")


@pytest.mark.r_parity
def test_glmboost_predict_binomial_response():
    import rpy2.robjects as ro

    r_code = """
    function(mstop, nu, n, seed) {
        set.seed(seed)
        x1 <- rnorm(n)
        x2 <- rnorm(n)
        p <- 1 / (1 + exp(-(0.5 * x1 + 0.3 * x2)))
        y <- factor(rbinom(n, 1, p))
        df <- data.frame(x1 = x1, x2 = x2, y = y)
        fit <- mboost::glmboost(
            y ~ x1 + x2, data = df,
            family = mboost::Binomial(), center = TRUE,
            control = mboost::boost_control(mstop = mstop, nu = nu)
        )
        pred_link <- as.numeric(stats::predict(fit, type = "link"))
        pred_response <- as.numeric(stats::predict(fit, type = "response"))
        list(
            x1 = as.numeric(x1), x2 = as.numeric(x2),
            y_numeric = as.numeric(y) - 1,
            pred_link = pred_link, pred_response = pred_response
        )
    }
    """
    r_func = ro.r(r_code)
    result = r_func(50, 0.1, 200, 42)

    y_m1 = 2.0 * np.asarray(result[2], dtype=np.float64) - 1.0
    data = {"y": y_m1, "x1": np.asarray(result[0], dtype=np.float64),
            "x2": np.asarray(result[1], dtype=np.float64)}
    model = mboost.glmboost(
        "y ~ x1 + x2", data=data, family=Binomial(),
        control=mboost.boost_control(50, 0.1),
    )

    pred_response = model.predict(type="response")
    r_pred_response = np.asarray(result[4], dtype=np.float64)
    np.testing.assert_allclose(pred_response, r_pred_response, rtol=1e-8,
                               err_msg="Binomial predict(type='response') does not match R")


def test_glmboost_weighted_centering_consistency():
    """Weighted fits must center training and new data identically.

    Regression test for an unweighted column-mean used at fit time vs a weighted
    mean at predict time: in-sample predict (== fitted) must equal predict() on
    the same rows passed as newdata, for non-uniform weights.
    """
    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.normal(2.0, 3.0, n)
    x2 = rng.normal(-1.0, 2.0, n)
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + rng.normal(0.0, 1.0, n)
    w = rng.uniform(0.1, 5.0, n)  # non-uniform weights
    data = {"y": y, "x1": x1, "x2": x2}

    model = mboost.glmboost(
        "y ~ x1 + x2", data=data, family=Gaussian(), weights=w,
        control=mboost.boost_control(300, 0.1),
    )

    in_sample = np.asarray(model.predict(type="link")).ravel()
    newdata = np.asarray(model.predict(newdata=data, type="link")).ravel()
    np.testing.assert_allclose(
        newdata, in_sample, rtol=0, atol=1e-9,
        err_msg="weighted glmboost: newdata predict must equal in-sample fitted",
    )
