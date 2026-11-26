"""R-parity tests for the LogNormalLSS survival family.

Log-normal accelerated-failure-time model with a right-censored
``Surv(time, status)`` response: mu (identity link, log-location) and sigma
(log link, scale). Exercises the 2-column response path through the LSS engine.

Requires R packages mboost + gamboostLSS + survival; skipped otherwise.
"""

import mboostrs as mboost  # noqa: F401
import numpy as np
import pytest
from mboostrs.gamlss import LogNormalLSS, gamboostLSS, glmboostLSS
from conftest import r_call

RTOL = 1e-6
ATOL = 1e-7
PARAMS = ("mu", "sigma")


def _r_glm(mstop, nu, n=200, seed=9):
    import rpy2.robjects as ro
    f = ro.r("""
    function(mstop, nu, n, seed) {
        if (!requireNamespace("gamboostLSS", quietly = TRUE)) stop("no gamboostLSS")
        suppressMessages(library(gamboostLSS)); suppressMessages(library(survival))
        set.seed(seed)
        x1 <- rnorm(n); x2 <- rnorm(n)
        t <- exp(1 + 0.5 * x1 + exp(0.2 * x2) * rnorm(n)); st <- rbinom(n, 1, 0.7)
        m <- glmboostLSS(Surv(t, st) ~ x1 + x2,
                         data = data.frame(t = t, st = st, x1 = x1, x2 = x2),
                         families = LogNormalLSS(),
                         control = boost_control(mstop = mstop, nu = nu))
        list(t = as.numeric(t), st = as.numeric(st),
             x1 = as.numeric(x1), x2 = as.numeric(x2),
             mu = as.numeric(predict(m, parameter = "mu", type = "link")),
             sigma = as.numeric(predict(m, parameter = "sigma", type = "link")))
    }
    """)
    r = r_call(f, mstop, nu, n, seed, packages=("gamboostLSS",))
    return {k: np.asarray(r.rx2(k), dtype=np.float64)
            for k in ("t", "st", "x1", "x2", "mu", "sigma")}


def _r_gam(mstop, nu, n=200, seed=9):
    import rpy2.robjects as ro
    f = ro.r("""
    function(mstop, nu, n, seed) {
        if (!requireNamespace("gamboostLSS", quietly = TRUE)) stop("no gamboostLSS")
        suppressMessages(library(gamboostLSS)); suppressMessages(library(survival))
        set.seed(seed)
        x <- runif(n, -1, 1)
        t <- exp(1 + sin(2 * x) + exp(0.2 * x) * rnorm(n)); st <- rbinom(n, 1, 0.7)
        m <- gamboostLSS(Surv(t, st) ~ bbs(x, knots = 20, df = 4),
                         data = data.frame(t = t, st = st, x = x),
                         families = LogNormalLSS(),
                         control = boost_control(mstop = mstop, nu = nu))
        list(t = as.numeric(t), st = as.numeric(st), x = as.numeric(x),
             mu = as.numeric(predict(m, parameter = "mu", type = "link")),
             sigma = as.numeric(predict(m, parameter = "sigma", type = "link")))
    }
    """)
    r = r_call(f, mstop, nu, n, seed, packages=("gamboostLSS",))
    return {k: np.asarray(r.rx2(k), dtype=np.float64)
            for k in ("t", "st", "x", "mu", "sigma")}


@pytest.mark.r_parity
def test_lognormallss_glmboostlss_parity():
    mstop, nu = 100, 0.1
    r = _r_glm(mstop, nu)
    m = glmboostLSS("Surv(t, st) ~ x1 + x2",
                    data={"t": r["t"], "st": r["st"], "x1": r["x1"], "x2": r["x2"]},
                    families=LogNormalLSS(), control=mboost.boost_control(mstop, nu),
                    response_scale=None)
    for p in PARAMS:
        np.testing.assert_allclose(
            m.predict(parameter=p, type="link"), r[p], rtol=RTOL, atol=ATOL,
            err_msg=f"LogNormalLSS {p} (link) does not match R")


@pytest.mark.r_parity
def test_lognormallss_gamboostlss_bbs_parity():
    mstop, nu = 100, 0.1
    r = _r_gam(mstop, nu)
    m = gamboostLSS("Surv(t, st) ~ bbs(x, knots=20, df=4)",
                    data={"t": r["t"], "st": r["st"], "x": r["x"]},
                    families=LogNormalLSS(), control=mboost.boost_control(mstop, nu),
                    response_scale=None)
    for p in PARAMS:
        np.testing.assert_allclose(
            m.predict(parameter=p, type="link"), r[p], rtol=RTOL, atol=ATOL,
            err_msg=f"LogNormalLSS bbs {p} (link) does not match R")
