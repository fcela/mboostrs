"""R-parity tests for glmboost Huber + bols."""

import mboostrs as mboost
import numpy as np
import pytest
from mboostrs.families import Huber

rtol = 1e-10


def _run_r_huber(mstop, nu, n=200, seed=42):
    """Generate data in R and fit glmboost with Huber family."""
    import rpy2.robjects as ro

    r_code = """
    function(mstop, nu, n, seed) {
        set.seed(seed)
        x1 <- rnorm(n)
        x2 <- rnorm(n)
        y <- 1.5 * x1 + 0.8 * x2 + rnorm(n, sd = 0.5)
        fit <- mboost::glmboost(
            y ~ x1 + x2,
            data = data.frame(x1 = x1, x2 = x2, y = y),
            family = mboost::Huber(),
            center = TRUE,
            control = mboost::boost_control(mstop = mstop, nu = nu)
        )
        cc_nooff <- stats::coef(fit, off2int = FALSE)
        off <- attr(cc_nooff, "offset")
        list(
            x1 = as.numeric(x1),
            x2 = as.numeric(x2),
            y = as.numeric(y),
            fitted = as.numeric(stats::predict(fit, type = "link")),
            risk = as.numeric(mboost::risk(fit)),
            coef = as.numeric(cc_nooff),
            offset = as.numeric(off)
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
        "coef": np.asarray(result[5], dtype=np.float64),
        "offset": float(result[6][0]) if len(result[6]) > 0 else 0.0,
    }


@pytest.mark.r_parity
def test_huber_bols_parity():
    mstop_val = 50
    nu_val = 0.1

    r = _run_r_huber(mstop_val, nu_val)

    data = {"y": r["y"], "x1": r["x1"], "x2": r["x2"]}
    model = mboost.glmboost(
        "y ~ x1 + x2",
        data=data,
        family=Huber(),
        control=mboost.boost_control(mstop_val, nu_val),
    )

    np.testing.assert_allclose(
        model.risk_, r["risk"], rtol=rtol,
        err_msg="risk path does not match R"
    )

    np.testing.assert_allclose(
        model.fitted_, r["fitted"], rtol=rtol,
        err_msg="fitted values do not match R"
    )
