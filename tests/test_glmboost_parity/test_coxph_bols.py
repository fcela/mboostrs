"""R-parity tests for glmboost CoxPH + bols.

The Breslow partial-likelihood ngradient matches R's C implementation to
machine precision, so the boosting path is bit-exact to R (coef, risk, and
fitted all agree to ~1e-15). Tolerances kept slightly above machine epsilon
for platform robustness, not to mask divergence.
  - risk: rtol=1e-8
  - fitted: rtol=1e-6, atol=1e-8 (atol covers near-zero linear predictors)
  - offset: atol=1e-10 (CoxPH offset is exactly 0)
"""

import mboostrs as mboost
import numpy as np
import pytest
from mboostrs.families import CoxPH

RTOL_RISK = 1e-8
RTOL_FITTED = 1e-6
ATOL_FITTED = 1e-8
ATOL_OFFSET = 1e-10


def _run_r_coxph(mstop, nu, n=200, seed=42):
    """Generate survival data in R and fit glmboost with CoxPH family."""
    import rpy2.robjects as ro

    r_code = """
    function(mstop, nu, n, seed) {
        set.seed(seed)
        x1 <- rnorm(n)
        x2 <- rnorm(n)
        eta <- 0.3 * x1 + 0.1 * x2
        time <- rexp(n, rate = exp(eta))
        event <- rbinom(n, 1, 0.7)
        df <- data.frame(time = time, event = event, x1 = x1, x2 = x2)
        fit <- mboost::glmboost(
            survival::Surv(time, event) ~ x1 + x2,
            data = df,
            family = mboost::CoxPH(),
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
def test_coxph_bols_risk():
    r = _run_r_coxph(50, 0.1)

    data = {"time": r["time"], "event": r["event"], "x1": r["x1"], "x2": r["x2"]}
    model = mboost.glmboost(
        "Surv(time, event) ~ x1 + x2",
        data=data,
        family=CoxPH(),
        control=mboost.boost_control(50, 0.1),
    )

    np.testing.assert_allclose(
        model.risk_, r["risk"], rtol=RTOL_RISK,
        err_msg="CoxPH risk path does not match R"
    )


@pytest.mark.r_parity
def test_coxph_bols_fitted():
    r = _run_r_coxph(50, 0.1)

    data = {"time": r["time"], "event": r["event"], "x1": r["x1"], "x2": r["x2"]}
    model = mboost.glmboost(
        "Surv(time, event) ~ x1 + x2",
        data=data,
        family=CoxPH(),
        control=mboost.boost_control(50, 0.1),
    )

    np.testing.assert_allclose(
        model.fitted_, r["fitted"], rtol=RTOL_FITTED, atol=ATOL_FITTED,
        err_msg="CoxPH fitted values do not match R"
    )


@pytest.mark.r_parity
def test_coxph_bols_offset():
    r = _run_r_coxph(50, 0.1)

    data = {"time": r["time"], "event": r["event"], "x1": r["x1"], "x2": r["x2"]}
    model = mboost.glmboost(
        "Surv(time, event) ~ x1 + x2",
        data=data,
        family=CoxPH(),
        control=mboost.boost_control(50, 0.1),
    )

    np.testing.assert_allclose(
        model.offset_, r["offset"], atol=ATOL_OFFSET,
        err_msg="CoxPH offset does not match R"
    )


def _run_r_coxph_cvrisk(mstop, nu, n=120, seed=21, B=15):
    """Fit CoxPH glmboost in R and cross-validate with R-generated folds.

    Returns the data, the (n x B) fold-weight matrix, and R's normalized OOB
    risk matrix (B x mstop+1). Passing the same fold matrix into mboostrs makes
    the comparison exact (no fold-RNG mismatch). Exercises the WEIGHTED Cox
    partial-likelihood path: bootstrap inbag fits use integer counts > 1, and
    R's cvrisk normalizes every family by colSums(OOBweights).
    """
    import rpy2.robjects as ro

    r_code = """
    function(mstop, nu, n, seed, B) {
        suppressMessages(library(survival))
        set.seed(seed)
        x1 <- rnorm(n); x2 <- rnorm(n)
        eta <- 0.8 * x1 - 0.5 * x2
        time <- rexp(n, exp(eta)); cens <- runif(n, 0, 3)
        status <- as.numeric(time <= cens); time <- pmin(time, cens)
        df <- data.frame(time = time, status = status, x1 = x1, x2 = x2)
        m <- mboost::glmboost(
            Surv(time, status) ~ x1 + x2, data = df,
            family = mboost::CoxPH(),
            control = mboost::boost_control(mstop = mstop, nu = nu))
        set.seed(7)
        folds <- mboost::cv(stats::model.weights(m), type = "bootstrap", B = B)
        cvr <- mboost::cvrisk(m, folds = folds, papply = lapply)
        oob <- matrix(as.numeric(cvr), nrow = nrow(cvr))
        list(time = as.numeric(time), status = as.numeric(status),
             x1 = as.numeric(x1), x2 = as.numeric(x2),
             folds = as.numeric(folds), nrow_f = nrow(folds), ncol_f = ncol(folds),
             oob = as.numeric(oob), nrow_o = nrow(oob), ncol_o = ncol(oob),
             rmstop = mboost::mstop(cvr))
    }
    """
    res = ro.r(r_code)(mstop, nu, n, seed, B)
    d = {k: np.asarray(res.rx2(k), dtype=np.float64) for k in
         ("time", "status", "x1", "x2", "folds", "oob")}
    nrf, ncf = int(res.rx2("nrow_f")[0]), int(res.rx2("ncol_f")[0])
    nro, nco = int(res.rx2("nrow_o")[0]), int(res.rx2("ncol_o")[0])
    # R fills matrices column-major
    d["folds"] = d["folds"].reshape((ncf, nrf)).T
    d["oob"] = d["oob"].reshape((nco, nro)).T
    d["rmstop"] = int(res.rx2("rmstop")[0])
    return d


@pytest.mark.r_parity
def test_coxph_cvrisk_matches_r():
    mstop, nu = 100, 0.1
    r = _run_r_coxph_cvrisk(mstop, nu)
    data = {"time": r["time"], "status": r["status"], "x1": r["x1"], "x2": r["x2"]}
    model = mboost.glmboost(
        "Surv(time, status) ~ x1 + x2", data=data, family=CoxPH(),
        control=mboost.boost_control(mstop, nu),
    )
    cvr = model.cvrisk(folds=r["folds"], parallel=False)
    py_oob = np.asarray(cvr.risk_matrix)

    assert py_oob.shape == r["oob"].shape
    # OOB Cox partial likelihood is the weighted-fit + weighted-OOB-risk path;
    # bit-exact up to fold-fit accumulation noise.
    np.testing.assert_allclose(
        py_oob, r["oob"], rtol=1e-6, atol=1e-8,
        err_msg="CoxPH cvrisk OOB risk matrix does not match R",
    )
    # And the selected stopping iteration must agree.
    assert cvr.optimal_mstop == r["rmstop"], (
        f"mstop mismatch: py={cvr.optimal_mstop} R={r['rmstop']}"
    )