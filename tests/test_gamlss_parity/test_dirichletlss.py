"""R-parity tests for the DirichletLSS family (compositional response).

K concentration parameters ``alpha_k = exp(eta_k)`` (log links) for a K-column
composition response given as ``cbind(y1, ..., yK) ~ ...``. Matches R
``gamboostLSS::DirichletLSS(K)`` value-for-value.

Requires R packages mboost + gamboostLSS; skipped otherwise.
"""

import mboostrs as mboost  # noqa: F401
import numpy as np
import pytest
from mboostrs.gamlss import DirichletLSS, gamboostLSS, glmboostLSS
from conftest import r_call

RTOL = 1e-6
ATOL = 1e-7
PARAMS = ("a1", "a2", "a3")


def _r_glm(mstop, nu, n=150, seed=2):
    import rpy2.robjects as ro
    f = ro.r("""
    function(mstop, nu, n, seed) {
        if (!requireNamespace("gamboostLSS", quietly = TRUE)) stop("no gamboostLSS")
        suppressMessages(library(gamboostLSS))
        set.seed(seed)
        x <- rnorm(n)
        a1 <- exp(0.5 + 0.3 * x); a2 <- exp(0.2); a3 <- exp(0.4 - 0.2 * x)
        Y <- t(apply(cbind(a1, a2, a3), 1, function(a) { g <- rgamma(3, a); g / sum(g) }))
        colnames(Y) <- c("y1", "y2", "y3"); dat <- data.frame(Y, x = x)
        m <- glmboostLSS(cbind(y1, y2, y3) ~ x, data = dat, families = DirichletLSS(K = 3),
                         control = boost_control(mstop = mstop, nu = nu))
        list(y1 = Y[, 1], y2 = Y[, 2], y3 = Y[, 3], x = x,
             a1 = as.numeric(predict(m, parameter = "a1", type = "link")),
             a2 = as.numeric(predict(m, parameter = "a2", type = "link")),
             a3 = as.numeric(predict(m, parameter = "a3", type = "link")))
    }
    """)
    r = r_call(f, mstop, nu, n, seed, packages=("gamboostLSS",))
    return {k: np.asarray(r.rx2(k), dtype=np.float64)
            for k in ("y1", "y2", "y3", "x", "a1", "a2", "a3")}


def _r_gam(mstop, nu, n=150, seed=2):
    import rpy2.robjects as ro
    f = ro.r("""
    function(mstop, nu, n, seed) {
        if (!requireNamespace("gamboostLSS", quietly = TRUE)) stop("no gamboostLSS")
        suppressMessages(library(gamboostLSS))
        set.seed(seed)
        x <- runif(n, -1, 1)
        a1 <- exp(0.5 + sin(2 * x)); a2 <- exp(0.2); a3 <- exp(0.4 - 0.3 * x)
        Y <- t(apply(cbind(a1, a2, a3), 1, function(a) { g <- rgamma(3, a); g / sum(g) }))
        colnames(Y) <- c("y1", "y2", "y3"); dat <- data.frame(Y, x = x)
        m <- gamboostLSS(cbind(y1, y2, y3) ~ bbs(x, knots = 20, df = 4), data = dat,
                         families = DirichletLSS(K = 3),
                         control = boost_control(mstop = mstop, nu = nu))
        list(y1 = Y[, 1], y2 = Y[, 2], y3 = Y[, 3], x = x,
             a1 = as.numeric(predict(m, parameter = "a1", type = "link")),
             a2 = as.numeric(predict(m, parameter = "a2", type = "link")),
             a3 = as.numeric(predict(m, parameter = "a3", type = "link")))
    }
    """)
    r = r_call(f, mstop, nu, n, seed, packages=("gamboostLSS",))
    return {k: np.asarray(r.rx2(k), dtype=np.float64)
            for k in ("y1", "y2", "y3", "x", "a1", "a2", "a3")}


@pytest.mark.r_parity
def test_dirichletlss_glmboostlss_parity():
    mstop, nu = 100, 0.1
    r = _r_glm(mstop, nu)
    m = glmboostLSS("cbind(y1, y2, y3) ~ x",
                    data={"y1": r["y1"], "y2": r["y2"], "y3": r["y3"], "x": r["x"]},
                    families=DirichletLSS(K=3), control=mboost.boost_control(mstop, nu))
    for p in PARAMS:
        np.testing.assert_allclose(
            m.predict(parameter=p, type="link"), r[p], rtol=RTOL, atol=ATOL,
            err_msg=f"DirichletLSS {p} (link) does not match R")


@pytest.mark.r_parity
def test_dirichletlss_gamboostlss_bbs_parity():
    mstop, nu = 100, 0.1
    r = _r_gam(mstop, nu)
    m = gamboostLSS("cbind(y1, y2, y3) ~ bbs(x, knots=20, df=4)",
                    data={"y1": r["y1"], "y2": r["y2"], "y3": r["y3"], "x": r["x"]},
                    families=DirichletLSS(K=3), control=mboost.boost_control(mstop, nu))
    for p in PARAMS:
        np.testing.assert_allclose(
            m.predict(parameter=p, type="link"), r[p], rtol=RTOL, atol=ATOL,
            err_msg=f"DirichletLSS bbs {p} (link) does not match R")
