"""R-parity tests for bbs out-of-domain (OOD) extrapolation.

When x falls outside boundary_knots, R's bbs uses linear extrapolation
based on the derivative of the B-spline at the boundary.  Our Rust
implementation must match this exactly.
"""

import mboostrs as mboost
import numpy as np
import pytest
from mboostrs.families import Gaussian

rtol = 1e-6
atol = 1e-9


def _run_r_bbs_ood(mstop, nu, n=100, seed=42):
    """Fit bbs in R with some x outside [0,1] and return predictions."""
    import rpy2.robjects as ro

    r_code = """
    function(mstop, nu, n, seed) {
        library(mboost)
        set.seed(seed)
        x1 <- runif(n, 0, 1)
        y <- sin(2 * pi * x1) + rnorm(n, sd = 0.3)
        df <- data.frame(x1 = x1, y = y)
        fit <- gamboost(
            y ~ bbs(x1, df = 4, knots = 20),
            data = df, family = Gaussian(),
            control = boost_control(mstop = mstop, nu = nu)
        )
        # Training predictions
        pred_train <- as.numeric(stats::predict(fit, type = "link"))
        # OOD newdata: x1 values outside [0,1]
        newdata <- data.frame(x1 = c(-0.5, -0.2, 1.2, 1.5))
        pred_ood <- as.numeric(stats::predict(fit, newdata = newdata, type = "link"))
        list(
            x1 = as.numeric(x1),
            y = as.numeric(y),
            fitted = pred_train,
            ood_x = c(-0.5, -0.2, 1.2, 1.5),
            ood_pred = pred_ood,
            risk = as.numeric(risk(fit)),
            offset = as.numeric(fit$offset)
        )
    }
    """
    r_func = ro.r(r_code)
    result = r_func(mstop, nu, n, seed)
    return {
        "x1": np.asarray(result[0], dtype=np.float64),
        "y": np.asarray(result[1], dtype=np.float64),
        "fitted": np.asarray(result[2], dtype=np.float64),
        "ood_x": np.asarray(result[3], dtype=np.float64),
        "ood_pred": np.asarray(result[4], dtype=np.float64),
        "risk": np.asarray(result[5], dtype=np.float64),
        "offset": float(result[6][0]),
    }


@pytest.mark.r_parity
def test_bbs_ood_training_parity():
    """Training fitted values must match R when some x are near boundary."""
    mstop_val = 50
    nu_val = 0.1

    r = _run_r_bbs_ood(mstop_val, nu_val)
    data = {"y": r["y"], "x1": r["x1"]}

    model = mboost.gamboost(
        "y ~ bbs(x1, df=4, knots=20)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(mstop_val, nu_val),
    )

    np.testing.assert_allclose(
        model.fitted_, r["fitted"], rtol=rtol, atol=atol,
        err_msg="bbs training fitted values do not match R",
    )
    np.testing.assert_allclose(
        model.risk_, r["risk"], rtol=rtol, atol=atol,
        err_msg="bbs risk path does not match R",
    )


@pytest.mark.r_parity
def test_bbs_ood_extrapolation_parity():
    """Predictions for x outside boundary_knots must match R's linear extrapolation."""
    mstop_val = 50
    nu_val = 0.1

    r = _run_r_bbs_ood(mstop_val, nu_val)
    data = {"y": r["y"], "x1": r["x1"]}

    model = mboost.gamboost(
        "y ~ bbs(x1, df=4, knots=20)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(mstop_val, nu_val),
    )

    # Predict on OOD x values
    newdata = {"x1": r["ood_x"]}
    py_pred = model.predict(newdata=newdata, type="link")

    np.testing.assert_allclose(
        py_pred, r["ood_pred"], rtol=rtol, atol=atol,
        err_msg="bbs OOD extrapolation predictions do not match R",
    )


@pytest.mark.r_parity
def test_bbs_ood_extrapolation_far():
    """Predictions far outside boundary_knots must remain linear (not explode)."""
    import rpy2.robjects as ro

    r_code = """
    function(mstop, nu, seed) {
        library(mboost)
        set.seed(seed)
        x1 <- runif(80, 0, 1)
        y <- sin(2 * pi * x1) + rnorm(80, sd = 0.3)
        df <- data.frame(x1 = x1, y = y)
        fit <- gamboost(
            y ~ bbs(x1, df = 4, knots = 20),
            data = df, family = Gaussian(),
            control = boost_control(mstop = mstop, nu = nu)
        )
        newdata <- data.frame(x1 = c(-2.0, -1.0, 2.0, 3.0))
        pred_ood <- as.numeric(stats::predict(fit, newdata = newdata, type = "link"))
        list(
            x1 = as.numeric(x1),
            y = as.numeric(y),
            ood_x = c(-2.0, -1.0, 2.0, 3.0),
            ood_pred = pred_ood
        )
    }
    """
    r_func = ro.r(r_code)
    result = r_func(50, 0.1, 42)

    x1 = np.asarray(result[0], dtype=np.float64)
    y = np.asarray(result[1], dtype=np.float64)
    ood_x = np.asarray(result[2], dtype=np.float64)
    ood_pred_r = np.asarray(result[3], dtype=np.float64)

    data = {"y": y, "x1": x1}
    model = mboost.gamboost(
        "y ~ bbs(x1, df=4, knots=20)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(50, 0.1),
    )

    newdata = {"x1": ood_x}
    py_pred = model.predict(newdata=newdata, type="link")

    np.testing.assert_allclose(
        py_pred, ood_pred_r, rtol=rtol, atol=atol,
        err_msg="bbs far-OOD extrapolation predictions do not match R",
    )
