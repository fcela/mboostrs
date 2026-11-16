"""R-parity tests for gamboost predict()."""

import mboostrs as mboost
import numpy as np
import pytest
from mboostrs.families import Gaussian

rtol = 1e-6
atol = 1e-9


def _run_r_gamboost_predict(mstop, nu, n=100, seed=42):
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
    pred_link <- as.numeric(stats::predict(fit, type = "link"))
    pred_response <- as.numeric(stats::predict(fit, type = "response"))
    newdata <- data.frame(x1 = c(0.3, 0.7), x2 = c(-0.5, 1.0))
    pred_new <- as.numeric(stats::predict(fit, newdata = newdata, type = "link"))
    k <- as.integer(mstop / 2)
    pred_k <- as.numeric(stats::predict(fit[k], type = "link"))
    list(
        x1 = as.numeric(x1), x2 = as.numeric(x2), y = as.numeric(y),
        pred_link = pred_link, pred_response = pred_response,
        pred_k = pred_k, k = k, pred_new = pred_new
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
        "k": int(result[6][0]),
        "pred_new": np.asarray(result[7], dtype=np.float64),
    }


@pytest.mark.r_parity
def test_gamboost_predict_link():
    r = _run_r_gamboost_predict(50, 0.1)
    data = {"y": r["y"], "x1": r["x1"], "x2": r["x2"]}
    model = mboost.gamboost(
        "y ~ bbs(x1, df=4, knots=20) + bols(x2)", data=data,
        family=Gaussian(), control=mboost.boost_control(50, 0.1),
    )
    pred = model.predict(type="link")
    np.testing.assert_allclose(pred, r["pred_link"], rtol=rtol, atol=atol,
                               err_msg="gamboost predict(type='link') does not match R")


@pytest.mark.r_parity
def test_gamboost_predict_response_gaussian():
    r = _run_r_gamboost_predict(50, 0.1)
    data = {"y": r["y"], "x1": r["x1"], "x2": r["x2"]}
    model = mboost.gamboost(
        "y ~ bbs(x1, df=4, knots=20) + bols(x2)", data=data,
        family=Gaussian(), control=mboost.boost_control(50, 0.1),
    )
    pred = model.predict(type="response")
    np.testing.assert_allclose(pred, r["pred_response"], rtol=rtol, atol=atol,
                               err_msg="gamboost predict(type='response') does not match R for Gaussian")


@pytest.mark.r_parity
def test_gamboost_predict_mstop():
    r = _run_r_gamboost_predict(50, 0.1)
    data = {"y": r["y"], "x1": r["x1"], "x2": r["x2"]}
    model = mboost.gamboost(
        "y ~ bbs(x1, df=4, knots=20) + bols(x2)", data=data,
        family=Gaussian(), control=mboost.boost_control(50, 0.1),
    )
    k = r["k"]
    pred = model.predict(mstop=k, type="link")
    np.testing.assert_allclose(pred, r["pred_k"], rtol=rtol, atol=atol,
                               err_msg=f"gamboost predict(mstop={k}) does not match R")


@pytest.mark.r_parity
def test_gamboost_predict_newdata():
    r = _run_r_gamboost_predict(50, 0.1)
    data = {"y": r["y"], "x1": r["x1"], "x2": r["x2"]}
    model = mboost.gamboost(
        "y ~ bbs(x1, df=4, knots=20) + bols(x2)", data=data,
        family=Gaussian(), control=mboost.boost_control(50, 0.1),
    )
    newdata = {"x1": np.array([0.3, 0.7]), "x2": np.array([-0.5, 1.0])}
    pred = model.predict(newdata=newdata, type="link")
    np.testing.assert_allclose(pred, r["pred_new"], rtol=1e-5, atol=atol,
                               err_msg="gamboost predict(newdata=...) does not match R")
