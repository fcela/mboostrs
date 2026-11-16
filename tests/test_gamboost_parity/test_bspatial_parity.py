"""R-parity tests for bspatial (tensor-product P-spline base-learner).

Tests that our Rust/Python bspatial implementation matches R mboost's
bspatial for fitted values, risk path, selected learners, and predictions.
"""

import mboostrs as mboost
import numpy as np
import pytest
from mboostrs.families import Gaussian
from mboostrs.learners import Bspatial

rtol = 1e-5  # residual df→lambda difference on the Kronecker penalty vs R
atol = 1e-8


def _run_r_bspatial(mstop, nu, n=200, seed=42):
    """Fit gamboost with bspatial in R and return results."""
    import rpy2.robjects as ro

    r_code = """
    function(mstop, nu, n, seed) {
        library(mboost)
        set.seed(seed)
        x1 <- runif(n, 0, 1)
        x2 <- runif(n, 0, 1)
        y <- sin(2 * pi * x1) * cos(2 * pi * x2) + rnorm(n, sd = 0.3)
        df <- data.frame(x1 = x1, x2 = x2, y = y)
        fit <- gamboost(
            y ~ bspatial(x1, x2, df = 6, knots = 10),
            data = df, family = Gaussian(),
            control = boost_control(mstop = mstop, nu = nu)
        )
        list(
            x1 = as.numeric(x1),
            x2 = as.numeric(x2),
            y = as.numeric(y),
            fitted = as.numeric(stats::predict(fit, type = "link")),
            risk = as.numeric(risk(fit)),
            selected = as.integer(selected(fit)),
            offset = as.numeric(fit$offset)
        )
    }
    """
    r_func = ro.r(r_code)
    result = r_func(mstop, nu, n, seed)
    return {
        "x1": np.asarray(result[0], dtype=np.float64),
        "x2": np.asarray(result[1], dtype=np.float64),
        "y": np.asarray(result[2], dtype=np.float64),
        "fitted": np.asarray(result[3], dtype=np.float64),
        "risk": np.asarray(result[4], dtype=np.float64),
        "selected": np.asarray(result[5], dtype=np.int32) - 1,  # 0-based
        "offset": float(result[6][0]),
    }


@pytest.mark.r_parity
def test_bspatial_fit_parity():
    """bspatial fitted values and risk path must match R."""
    mstop_val = 50
    nu_val = 0.1

    r = _run_r_bspatial(mstop_val, nu_val)
    data = {"y": r["y"], "x1": r["x1"], "x2": r["x2"]}

    model = mboost.gamboost(
        "y ~ bspatial(x1, x2, df=6, knots=10)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(mstop_val, nu_val),
    )

    np.testing.assert_allclose(
        model.fitted_, r["fitted"], rtol=rtol, atol=atol,
        err_msg="bspatial fitted values do not match R",
    )
    np.testing.assert_allclose(
        model.risk_, r["risk"], rtol=rtol, atol=atol,
        err_msg="bspatial risk path does not match R",
    )
    np.testing.assert_allclose(
        model.offset_, r["offset"], rtol=rtol, atol=atol,
        err_msg="bspatial offset does not match R",
    )
    assert model._selected == r["selected"].tolist(), (
        f"bspatial selected learners differ: Python={model._selected}, R={r['selected'].tolist()}"
    )


@pytest.mark.r_parity
def test_bspatial_predict_parity():
    """bspatial predict on new data must match R."""
    import rpy2.robjects as ro

    r_code = """
    function(mstop, nu, n, seed) {
        library(mboost)
        set.seed(seed)
        x1 <- runif(n, 0, 1)
        x2 <- runif(n, 0, 1)
        y <- sin(2 * pi * x1) * cos(2 * pi * x2) + rnorm(n, sd = 0.3)
        df <- data.frame(x1 = x1, x2 = x2, y = y)
        fit <- gamboost(
            y ~ bspatial(x1, x2, df = 6, knots = 10),
            data = df, family = Gaussian(),
            control = boost_control(mstop = mstop, nu = nu)
        )
        newdata <- data.frame(x1 = c(0.2, 0.5, 0.8), x2 = c(0.3, 0.6, 0.9))
        pred_new <- as.numeric(stats::predict(fit, newdata = newdata, type = "link"))
        list(
            x1 = as.numeric(x1),
            x2 = as.numeric(x2),
            y = as.numeric(y),
            new_x1 = c(0.2, 0.5, 0.8),
            new_x2 = c(0.3, 0.6, 0.9),
            pred_new = pred_new
        )
    }
    """
    r_func = ro.r(r_code)
    result = r_func(50, 0.1, 200, 42)

    x1 = np.asarray(result[0], dtype=np.float64)
    x2 = np.asarray(result[1], dtype=np.float64)
    y = np.asarray(result[2], dtype=np.float64)
    new_x1 = np.asarray(result[3], dtype=np.float64)
    new_x2 = np.asarray(result[4], dtype=np.float64)
    pred_new_r = np.asarray(result[5], dtype=np.float64)

    data = {"y": y, "x1": x1, "x2": x2}
    model = mboost.gamboost(
        "y ~ bspatial(x1, x2, df=6, knots=10)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(50, 0.1),
    )

    newdata = {"x1": new_x1, "x2": new_x2}
    py_pred = model.predict(newdata=newdata, type="link")

    np.testing.assert_allclose(
        py_pred, pred_new_r, rtol=rtol, atol=atol,
        err_msg="bspatial predict on new data does not match R",
    )


@pytest.mark.r_parity
def test_bspatial_explicit_lambda_parity():
    """bspatial with explicit lambda must match R."""
    import rpy2.robjects as ro

    r_code = """
    function(mstop, nu, lambda, n, seed) {
        library(mboost)
        set.seed(seed)
        x1 <- runif(n, 0, 1)
        x2 <- runif(n, 0, 1)
        y <- sin(2 * pi * x1) * cos(2 * pi * x2) + rnorm(n, sd = 0.3)
        df <- data.frame(x1 = x1, x2 = x2, y = y)
        fit <- gamboost(
            y ~ bspatial(x1, x2, lambda = lambda),
            data = df, family = Gaussian(),
            control = boost_control(mstop = mstop, nu = nu)
        )
        list(
            x1 = as.numeric(x1),
            x2 = as.numeric(x2),
            y = as.numeric(y),
            fitted = as.numeric(stats::predict(fit, type = "link")),
            risk = as.numeric(risk(fit))
        )
    }
    """
    r_func = ro.r(r_code)

    for lam in [0.1, 1.0]:
        result = r_func(50, 0.1, lam, 200, 42)
        x1 = np.asarray(result[0], dtype=np.float64)
        x2 = np.asarray(result[1], dtype=np.float64)
        y = np.asarray(result[2], dtype=np.float64)
        fitted_r = np.asarray(result[3], dtype=np.float64)
        risk_r = np.asarray(result[4], dtype=np.float64)

        data = {"y": y, "x1": x1, "x2": x2}
        model = mboost.gamboost(
            "y ~ bspatial(x1, x2)",
            data=data,
            family=Gaussian(),
            control=mboost.boost_control(50, 0.1),
            learners=[Bspatial("x1", "x2", lambda_value=lam)],
        )

        np.testing.assert_allclose(
            model.fitted_, fitted_r, rtol=rtol, atol=atol,
            err_msg=f"bspatial lambda={lam} fitted values do not match R",
        )
        np.testing.assert_allclose(
            model.risk_, risk_r, rtol=rtol, atol=atol,
            err_msg=f"bspatial lambda={lam} risk path does not match R",
        )


@pytest.mark.r_parity
def test_bspatial_with_bols_mixed():
    """bspatial mixed with bols must match R."""
    import rpy2.robjects as ro

    r_code = """
    function(mstop, nu, n, seed) {
        library(mboost)
        set.seed(seed)
        x1 <- runif(n, 0, 1)
        x2 <- runif(n, 0, 1)
        z <- rnorm(n)
        y <- sin(2 * pi * x1) * cos(2 * pi * x2) + 0.5 * z + rnorm(n, sd = 0.3)
        df <- data.frame(x1 = x1, x2 = x2, z = z, y = y)
        fit <- gamboost(
            y ~ bspatial(x1, x2, df = 6, knots = 10) + bols(z),
            data = df, family = Gaussian(),
            control = boost_control(mstop = mstop, nu = nu)
        )
        list(
            x1 = as.numeric(x1),
            x2 = as.numeric(x2),
            z = as.numeric(z),
            y = as.numeric(y),
            fitted = as.numeric(stats::predict(fit, type = "link")),
            risk = as.numeric(risk(fit)),
            selected = as.integer(selected(fit)),
            offset = as.numeric(fit$offset)
        )
    }
    """
    r_func = ro.r(r_code)
    result = r_func(100, 0.1, 200, 42)

    x1 = np.asarray(result[0], dtype=np.float64)
    x2 = np.asarray(result[1], dtype=np.float64)
    z = np.asarray(result[2], dtype=np.float64)
    y = np.asarray(result[3], dtype=np.float64)
    fitted_r = np.asarray(result[4], dtype=np.float64)
    risk_r = np.asarray(result[5], dtype=np.float64)
    selected_r = np.asarray(result[6], dtype=np.int32) - 1
    offset_r = float(result[7][0])

    data = {"y": y, "x1": x1, "x2": x2, "z": z}
    model = mboost.gamboost(
        "y ~ bspatial(x1, x2, df=6, knots=10) + bols(z)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(100, 0.1),
    )

    np.testing.assert_allclose(
        model.fitted_, fitted_r, rtol=rtol, atol=atol,
        err_msg="bspatial+bols mixed fitted values do not match R",
    )
    np.testing.assert_allclose(
        model.risk_, risk_r, rtol=rtol, atol=atol,
        err_msg="bspatial+bols mixed risk path does not match R",
    )
    np.testing.assert_allclose(
        model.offset_, offset_r, rtol=rtol, atol=atol,
        err_msg="bspatial+bols mixed offset does not match R",
    )
    assert model._selected == selected_r.tolist(), (
        f"bspatial+bols selected differ: Python={model._selected}, R={selected_r.tolist()}"
    )
