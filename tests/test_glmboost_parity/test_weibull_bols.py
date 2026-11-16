"""R-parity tests for glmboost Weibull + bols.

Tolerances reflect the only genuine source of divergence from R: the sigma
nuisance is calibrated by Brent minimization vs R's optimize(), differing at
~optimizer-tolerance level (achieved: risk ~2e-7, fitted ~1e-6, offset exact).
  - risk: rtol=1e-5
  - fitted: rtol=1e-4, atol=1e-5 (atol covers near-zero linear predictors)
  - offset: rtol=1e-6
"""

import mboostrs as mboost
import numpy as np
import pytest
from mboostrs.families import Weibull

RTOL_RISK = 1e-5
RTOL_FITTED = 1e-4
ATOL_FITTED = 1e-5
RTOL_OFFSET = 1e-6


def _run_r_weibull(mstop, nu, n=200, seed=42):
    """Generate survival data in R and fit glmboost with Weibull family."""
    import rpy2.robjects as ro

    r_code = """
    function(mstop, nu, n, seed) {
        set.seed(seed)
        x1 <- rnorm(n)
        x2 <- rnorm(n)
        eta <- 0.3 * x1 + 0.1 * x2
        time <- rweibull(n, shape = 1, scale = exp(eta))
        event <- rbinom(n, 1, 0.7)
        df <- data.frame(time = time, event = event, x1 = x1, x2 = x2)
        fit <- mboost::glmboost(
            survival::Surv(time, event) ~ x1 + x2,
            data = df,
            family = mboost::Weibull(),
            center = TRUE,
            control = mboost::boost_control(mstop = mstop, nu = nu)
        )
        cc_nooff <- stats::coef(fit, off2int = FALSE)
        off <- attr(cc_nooff, "offset")
        list(
            x1 = as.numeric(x1),
            x2 = as.numeric(x2),
            time = as.numeric(time),
            event = as.numeric(event),
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
        "time": np.asarray(result[2], dtype=np.float64),
        "event": np.asarray(result[3], dtype=np.float64),
        "fitted": np.asarray(result[4], dtype=np.float64),
        "risk": np.asarray(result[5], dtype=np.float64),
        "coef": np.asarray(result[6], dtype=np.float64),
        "offset": float(result[7][0]) if len(result[7]) > 0 else 0.0,
    }


@pytest.mark.r_parity
def test_weibull_bols_risk():
    r = _run_r_weibull(50, 0.1)

    data = {"time": r["time"], "event": r["event"], "x1": r["x1"], "x2": r["x2"]}
    model = mboost.glmboost(
        "Surv(time, event) ~ x1 + x2",
        data=data,
        family=Weibull(),
        control=mboost.boost_control(50, 0.1),
    )

    np.testing.assert_allclose(
        model.risk_, r["risk"], rtol=RTOL_RISK,
        err_msg="Weibull risk path does not match R"
    )


@pytest.mark.r_parity
def test_weibull_bols_fitted():
    r = _run_r_weibull(50, 0.1)

    data = {"time": r["time"], "event": r["event"], "x1": r["x1"], "x2": r["x2"]}
    model = mboost.glmboost(
        "Surv(time, event) ~ x1 + x2",
        data=data,
        family=Weibull(),
        control=mboost.boost_control(50, 0.1),
    )

    np.testing.assert_allclose(
        model.fitted_, r["fitted"], rtol=RTOL_FITTED, atol=ATOL_FITTED,
        err_msg="Weibull fitted values do not match R"
    )


@pytest.mark.r_parity
def test_weibull_bols_offset():
    r = _run_r_weibull(50, 0.1)

    data = {"time": r["time"], "event": r["event"], "x1": r["x1"], "x2": r["x2"]}
    model = mboost.glmboost(
        "Surv(time, event) ~ x1 + x2",
        data=data,
        family=Weibull(),
        control=mboost.boost_control(50, 0.1),
    )

    np.testing.assert_allclose(
        model.offset_, r["offset"], rtol=RTOL_OFFSET,
        err_msg="Weibull offset does not match R"
    )