"""R-parity tests for glmboost Quantile + bols."""

import mboostrs as mboost
import numpy as np
import pytest
from mboostrs.families import Quantile

rtol = 1e-10


def _run_r_quantreg(mstop, nu, n=200, seed=42):
    """Generate data in R and fit glmboost with QuantReg family."""
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
            family = mboost::QuantReg(),
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
def test_quantile_bols_parity():
    mstop_val = 50
    nu_val = 0.1

    r = _run_r_quantreg(mstop_val, nu_val)

    data = {"y": r["y"], "x1": r["x1"], "x2": r["x2"]}
    model = mboost.glmboost(
        "y ~ x1 + x2",
        data=data,
        family=Quantile(tau=0.5, qoffset=0.5, response_scale=None),
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


def _run_r_quantreg_weighted(mstop, nu, tau, n=180, seed=11):
    """QuantReg under NON-UNIFORM case weights.

    R's QuantReg offset is quantile(y[rep(1:n, w)], qoffset) — the weighted
    (replicate) type-7 quantile. The old core ignored weights for this offset,
    so weighted fits diverged from R (offset off by ~1.5e-3, propagating into
    fitted/risk). qoffset defaults to 0.5 and is independent of tau.
    """
    import rpy2.robjects as ro

    r_code = """
    function(mstop, nu, tau, n, seed) {
        set.seed(seed)
        x1 <- rnorm(n); x2 <- rnorm(n)
        y <- 1 + 0.8 * x1 - 0.5 * x2 + rnorm(n)
        w <- sample(1:4, n, replace = TRUE)
        fit <- mboost::glmboost(
            y ~ x1 + x2, data = data.frame(x1 = x1, x2 = x2, y = y),
            weights = w, family = mboost::QuantReg(tau = tau),
            center = TRUE, control = mboost::boost_control(mstop = mstop, nu = nu))
        list(x1 = as.numeric(x1), x2 = as.numeric(x2), y = as.numeric(y),
             w = as.numeric(w),
             fitted = as.numeric(stats::predict(fit, type = "link")),
             risk = as.numeric(mboost::risk(fit)),
             offset = as.numeric(attr(stats::coef(fit, off2int = FALSE), "offset")))
    }
    """
    res = ro.r(r_code)(mstop, nu, tau, n, seed)
    return {
        "x1": np.asarray(res[0], dtype=np.float64),
        "x2": np.asarray(res[1], dtype=np.float64),
        "y": np.asarray(res[2], dtype=np.float64),
        "w": np.asarray(res[3], dtype=np.float64),
        "fitted": np.asarray(res[4], dtype=np.float64),
        "risk": np.asarray(res[5], dtype=np.float64),
        "offset": float(res[6][0]),
    }


@pytest.mark.r_parity
@pytest.mark.parametrize("tau", [0.5, 0.25, 0.75])
def test_quantile_bols_weighted_parity(tau):
    mstop_val, nu_val = 80, 0.1
    r = _run_r_quantreg_weighted(mstop_val, nu_val, tau)

    data = {"y": r["y"], "x1": r["x1"], "x2": r["x2"]}
    model = mboost.glmboost(
        "y ~ x1 + x2", data=data,
        family=Quantile(tau=tau, qoffset=0.5, response_scale=None),
        weights=r["w"], control=mboost.boost_control(mstop_val, nu_val),
    )

    np.testing.assert_allclose(
        model.offset_, r["offset"], atol=1e-12,
        err_msg="weighted QuantReg offset does not match R",
    )
    np.testing.assert_allclose(
        model.risk_, r["risk"], rtol=1e-9, atol=1e-9,
        err_msg="weighted QuantReg risk path does not match R",
    )
    np.testing.assert_allclose(
        model.fitted_, r["fitted"], rtol=1e-9, atol=1e-9,
        err_msg="weighted QuantReg fitted values do not match R",
    )
