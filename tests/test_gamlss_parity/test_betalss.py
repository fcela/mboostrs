"""R-parity tests for the BetaLSS family (bounded (0,1) response).

BetaLSS uses a logit link for mu (mean) and a log link for phi (precision);
both gradients use digamma. Adds link-function diversity (logit) to the LSS
parity coverage.

Requires R packages mboost + gamboostLSS; skipped otherwise.
"""

import mboostrs as mboost  # noqa: F401
import numpy as np
import pytest
from mboostrs.gamlss import BetaLSS, gamboostLSS, glmboostLSS
from conftest import r_call

RTOL = 1e-6
ATOL = 1e-7
PARAMS = ("mu", "phi")


def _r_glm(mstop, nu, n=200, seed=5):
    import rpy2.robjects as ro
    f = ro.r("""
    function(mstop, nu, n, seed) {
        if (!requireNamespace("gamboostLSS", quietly = TRUE)) stop("no gamboostLSS")
        suppressMessages(library(gamboostLSS))
        set.seed(seed)
        x1 <- rnorm(n); x2 <- rnorm(n)
        mu <- plogis(0.3 + 0.5 * x1); phi <- exp(2 + 0.3 * x2)
        y <- rbeta(n, mu * phi, (1 - mu) * phi)
        m <- glmboostLSS(y ~ x1 + x2, data = data.frame(y = y, x1 = x1, x2 = x2),
                         families = BetaLSS(),
                         control = boost_control(mstop = mstop, nu = nu))
        list(x1 = as.numeric(x1), x2 = as.numeric(x2), y = as.numeric(y),
             mu = as.numeric(predict(m, parameter = "mu", type = "link")),
             phi = as.numeric(predict(m, parameter = "phi", type = "link")))
    }
    """)
    r = r_call(f, mstop, nu, n, seed, packages=("gamboostLSS",))
    return {k: np.asarray(r.rx2(k), dtype=np.float64)
            for k in ("x1", "x2", "y", "mu", "phi")}


def _r_gam(mstop, nu, n=200, seed=5):
    import rpy2.robjects as ro
    f = ro.r("""
    function(mstop, nu, n, seed) {
        if (!requireNamespace("gamboostLSS", quietly = TRUE)) stop("no gamboostLSS")
        suppressMessages(library(gamboostLSS))
        set.seed(seed)
        x <- runif(n, -1, 1)
        mu <- plogis(0.3 + sin(2 * x)); phi <- exp(2 + 0.3 * x)
        y <- rbeta(n, mu * phi, (1 - mu) * phi)
        m <- gamboostLSS(y ~ bbs(x, knots = 20, df = 4),
                         data = data.frame(y = y, x = x), families = BetaLSS(),
                         control = boost_control(mstop = mstop, nu = nu))
        list(x = as.numeric(x), y = as.numeric(y),
             mu = as.numeric(predict(m, parameter = "mu", type = "link")),
             phi = as.numeric(predict(m, parameter = "phi", type = "link")))
    }
    """)
    r = r_call(f, mstop, nu, n, seed, packages=("gamboostLSS",))
    return {k: np.asarray(r.rx2(k), dtype=np.float64) for k in ("x", "y", "mu", "phi")}


@pytest.mark.r_parity
def test_betalss_glmboostlss_parity():
    mstop, nu = 100, 0.1
    r = _r_glm(mstop, nu)
    m = glmboostLSS("y ~ x1 + x2", data={"y": r["y"], "x1": r["x1"], "x2": r["x2"]},
                    families=BetaLSS(), control=mboost.boost_control(mstop, nu))
    for p in PARAMS:
        np.testing.assert_allclose(
            m.predict(parameter=p, type="link"), r[p], rtol=RTOL, atol=ATOL,
            err_msg=f"BetaLSS {p} (link) does not match R")


@pytest.mark.r_parity
def test_betalss_gamboostlss_bbs_parity():
    mstop, nu = 100, 0.1
    r = _r_gam(mstop, nu)
    m = gamboostLSS("y ~ bbs(x, knots=20, df=4)", data={"y": r["y"], "x": r["x"]},
                    families=BetaLSS(), control=mboost.boost_control(mstop, nu))
    for p in PARAMS:
        np.testing.assert_allclose(
            m.predict(parameter=p, type="link"), r[p], rtol=RTOL, atol=ATOL,
            err_msg=f"BetaLSS bbs {p} (link) does not match R")
