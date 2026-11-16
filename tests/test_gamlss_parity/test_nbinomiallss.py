"""R-parity tests for the NBinomialLSS family (overdispersed counts).

mu (log link, mean) and sigma (log link, size/dispersion); sigma gradient uses
digamma. Requires R packages mboost + gamboostLSS; skipped otherwise.
"""

import mboostrs as mboost  # noqa: F401
import numpy as np
import pytest
from mboostrs.gamlss import NBinomialLSS, gamboostLSS, glmboostLSS
from conftest import r_call

RTOL = 1e-6
ATOL = 1e-7
PARAMS = ("mu", "sigma")


def _r_glm(mstop, nu, n=200, seed=7):
    import rpy2.robjects as ro
    f = ro.r("""
    function(mstop, nu, n, seed) {
        if (!requireNamespace("gamboostLSS", quietly = TRUE)) stop("no gamboostLSS")
        suppressMessages(library(gamboostLSS))
        set.seed(seed)
        x1 <- rnorm(n); x2 <- rnorm(n)
        y <- rnbinom(n, mu = exp(1 + 0.3 * x1), size = exp(0.5 + 0.2 * x2))
        m <- glmboostLSS(y ~ x1 + x2, data = data.frame(y = y, x1 = x1, x2 = x2),
                         families = NBinomialLSS(),
                         control = boost_control(mstop = mstop, nu = nu))
        list(x1 = as.numeric(x1), x2 = as.numeric(x2), y = as.numeric(y),
             mu = as.numeric(predict(m, parameter = "mu", type = "link")),
             sigma = as.numeric(predict(m, parameter = "sigma", type = "link")))
    }
    """)
    r = r_call(f, mstop, nu, n, seed, packages=("gamboostLSS",))
    return {k: np.asarray(r.rx2(k), dtype=np.float64)
            for k in ("x1", "x2", "y", "mu", "sigma")}


def _r_gam(mstop, nu, n=200, seed=7):
    import rpy2.robjects as ro
    f = ro.r("""
    function(mstop, nu, n, seed) {
        if (!requireNamespace("gamboostLSS", quietly = TRUE)) stop("no gamboostLSS")
        suppressMessages(library(gamboostLSS))
        set.seed(seed)
        x <- runif(n, -1, 1)
        y <- rnbinom(n, mu = exp(1 + sin(2 * x)), size = exp(0.5 + 0.3 * x))
        m <- gamboostLSS(y ~ bbs(x, knots = 20, df = 4),
                         data = data.frame(y = y, x = x), families = NBinomialLSS(),
                         control = boost_control(mstop = mstop, nu = nu))
        list(x = as.numeric(x), y = as.numeric(y),
             mu = as.numeric(predict(m, parameter = "mu", type = "link")),
             sigma = as.numeric(predict(m, parameter = "sigma", type = "link")))
    }
    """)
    r = r_call(f, mstop, nu, n, seed, packages=("gamboostLSS",))
    return {k: np.asarray(r.rx2(k), dtype=np.float64) for k in ("x", "y", "mu", "sigma")}


@pytest.mark.r_parity
def test_nbinomiallss_glmboostlss_parity():
    mstop, nu = 100, 0.1
    r = _r_glm(mstop, nu)
    m = glmboostLSS("y ~ x1 + x2", data={"y": r["y"], "x1": r["x1"], "x2": r["x2"]},
                    families=NBinomialLSS(), control=mboost.boost_control(mstop, nu))
    for p in PARAMS:
        np.testing.assert_allclose(
            m.predict(parameter=p, type="link"), r[p], rtol=RTOL, atol=ATOL,
            err_msg=f"NBinomialLSS {p} (link) does not match R")


@pytest.mark.r_parity
def test_nbinomiallss_gamboostlss_bbs_parity():
    mstop, nu = 100, 0.1
    r = _r_gam(mstop, nu)
    m = gamboostLSS("y ~ bbs(x, knots=20, df=4)", data={"y": r["y"], "x": r["x"]},
                    families=NBinomialLSS(), control=mboost.boost_control(mstop, nu))
    for p in PARAMS:
        np.testing.assert_allclose(
            m.predict(parameter=p, type="link"), r[p], rtol=RTOL, atol=ATOL,
            err_msg=f"NBinomialLSS bbs {p} (link) does not match R")
