"""R-parity tests for the StudentTLSS family (3 parameters).

StudentTLSS exercises the 3-parameter cyclic path: mu (identity), sigma (log,
scale), df (log, degrees of freedom). The df gradient uses digamma and the
offsets are computed sequentially (each parameter's offset uses the already-
computed offsets of earlier parameters), matching R's gamboostLSS.

Requires R packages mboost + gamboostLSS; skipped otherwise.
"""

import mboostrs as mboost  # noqa: F401
import numpy as np
import pytest
from mboostrs.gamlss import StudentTLSS, gamboostLSS, glmboostLSS
from conftest import r_call

RTOL = 1e-6
ATOL = 1e-7
PARAMS = ("mu", "sigma", "df")


def _r_glm(mstop, nu, n=200, seed=3):
    import rpy2.robjects as ro
    f = ro.r("""
    function(mstop, nu, n, seed) {
        if (!requireNamespace("gamboostLSS", quietly = TRUE)) stop("no gamboostLSS")
        suppressMessages(library(gamboostLSS))
        set.seed(seed)
        x1 <- rnorm(n); x2 <- rnorm(n); x3 <- rnorm(n)
        y <- (1 + 0.5 * x1) + exp(0.2 * x2) * rt(n, df = exp(1 + 0.3 * x3))
        m <- glmboostLSS(y ~ x1 + x2 + x3,
                         data = data.frame(y = y, x1 = x1, x2 = x2, x3 = x3),
                         families = StudentTLSS(),
                         control = boost_control(mstop = mstop, nu = nu))
        list(x1 = as.numeric(x1), x2 = as.numeric(x2), x3 = as.numeric(x3),
             y = as.numeric(y),
             mu = as.numeric(predict(m, parameter = "mu", type = "link")),
             sigma = as.numeric(predict(m, parameter = "sigma", type = "link")),
             df = as.numeric(predict(m, parameter = "df", type = "link")))
    }
    """)
    r = r_call(f, mstop, nu, n, seed, packages=("gamboostLSS",))
    return {k: np.asarray(r.rx2(k), dtype=np.float64)
            for k in ("x1", "x2", "x3", "y", "mu", "sigma", "df")}


def _r_gam(mstop, nu, n=200, seed=3):
    import rpy2.robjects as ro
    f = ro.r("""
    function(mstop, nu, n, seed) {
        if (!requireNamespace("gamboostLSS", quietly = TRUE)) stop("no gamboostLSS")
        suppressMessages(library(gamboostLSS))
        set.seed(seed)
        x <- runif(n, -1, 1)
        y <- (1 + sin(2 * x)) + exp(0.2 * x) * rt(n, df = 5)
        m <- gamboostLSS(y ~ bbs(x, knots = 20, df = 4),
                         data = data.frame(y = y, x = x), families = StudentTLSS(),
                         control = boost_control(mstop = mstop, nu = nu))
        list(x = as.numeric(x), y = as.numeric(y),
             mu = as.numeric(predict(m, parameter = "mu", type = "link")),
             sigma = as.numeric(predict(m, parameter = "sigma", type = "link")),
             df = as.numeric(predict(m, parameter = "df", type = "link")))
    }
    """)
    r = r_call(f, mstop, nu, n, seed, packages=("gamboostLSS",))
    return {k: np.asarray(r.rx2(k), dtype=np.float64)
            for k in ("x", "y", "mu", "sigma", "df")}


@pytest.mark.r_parity
def test_studenttlss_glmboostlss_parity():
    mstop, nu = 100, 0.1
    r = _r_glm(mstop, nu)
    m = glmboostLSS("y ~ x1 + x2 + x3",
                    data={"y": r["y"], "x1": r["x1"], "x2": r["x2"], "x3": r["x3"]},
                    families=StudentTLSS(), control=mboost.boost_control(mstop, nu))
    for p in PARAMS:
        np.testing.assert_allclose(
            m.predict(parameter=p, type="link"), r[p], rtol=RTOL, atol=ATOL,
            err_msg=f"StudentTLSS {p} (link) does not match R")


@pytest.mark.r_parity
def test_studenttlss_gamboostlss_bbs_parity():
    mstop, nu = 100, 0.1
    r = _r_gam(mstop, nu)
    m = gamboostLSS("y ~ bbs(x, knots=20, df=4)", data={"y": r["y"], "x": r["x"]},
                    families=StudentTLSS(), control=mboost.boost_control(mstop, nu))
    for p in PARAMS:
        np.testing.assert_allclose(
            m.predict(parameter=p, type="link"), r[p], rtol=RTOL, atol=ATOL,
            err_msg=f"StudentTLSS bbs {p} (link) does not match R")
