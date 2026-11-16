"""R-parity tests for the ZIPoLSS family (zero-inflated Poisson).

mu (log link, Poisson mean) and sigma (logit link, zero-inflation probability),
wrapping gamlss.dist's ZIP. Matches R ``gamboostLSS::ZIPoLSS()`` value-for-value:
offsets are the gamlss initials (mu = log(mean y), sigma = qlogis(0.1)), the
gradients are ZIP's dldm/dldd times the link derivatives, and the risk is the
ZIP negative log-likelihood.

Requires R packages mboost + gamboostLSS; skipped otherwise.
"""

import mboostrs as mboost  # noqa: F401
import numpy as np
import pytest
from mboostrs.gamlss import ZIPoLSS, gamboostLSS, glmboostLSS
from conftest import r_call

RTOL = 1e-6
ATOL = 1e-7
PARAMS = ("mu", "sigma")


def _r_glm(mstop, nu, n=200, seed=3):
    import rpy2.robjects as ro
    f = ro.r("""
    function(mstop, nu, n, seed) {
        if (!requireNamespace("gamboostLSS", quietly = TRUE)) stop("no gamboostLSS")
        suppressMessages(library(gamboostLSS))
        set.seed(seed)
        x1 <- rnorm(n); x2 <- rnorm(n)
        mu <- exp(0.5 + 0.4 * x1); pz <- plogis(-1 + 0.5 * x2)
        z <- rbinom(n, 1, pz); y <- ifelse(z == 1, 0, rpois(n, mu))
        m <- glmboostLSS(y ~ x1 + x2, data = data.frame(y = y, x1 = x1, x2 = x2),
                         families = ZIPoLSS(),
                         control = boost_control(mstop = mstop, nu = nu))
        list(y = as.numeric(y), x1 = as.numeric(x1), x2 = as.numeric(x2),
             mu = as.numeric(predict(m, parameter = "mu", type = "link")),
             sigma = as.numeric(predict(m, parameter = "sigma", type = "link")))
    }
    """)
    r = r_call(f, mstop, nu, n, seed, packages=("gamboostLSS",))
    return {k: np.asarray(r.rx2(k), dtype=np.float64)
            for k in ("y", "x1", "x2", "mu", "sigma")}


def _r_gam(mstop, nu, n=200, seed=3):
    import rpy2.robjects as ro
    f = ro.r("""
    function(mstop, nu, n, seed) {
        if (!requireNamespace("gamboostLSS", quietly = TRUE)) stop("no gamboostLSS")
        suppressMessages(library(gamboostLSS))
        set.seed(seed)
        x <- runif(n, -1, 1)
        mu <- exp(0.5 + sin(2 * x)); pz <- plogis(-1 + x)
        z <- rbinom(n, 1, pz); y <- ifelse(z == 1, 0, rpois(n, mu))
        m <- gamboostLSS(y ~ bbs(x, knots = 20, df = 4),
                         data = data.frame(y = y, x = x), families = ZIPoLSS(),
                         control = boost_control(mstop = mstop, nu = nu))
        list(y = as.numeric(y), x = as.numeric(x),
             mu = as.numeric(predict(m, parameter = "mu", type = "link")),
             sigma = as.numeric(predict(m, parameter = "sigma", type = "link")))
    }
    """)
    r = r_call(f, mstop, nu, n, seed, packages=("gamboostLSS",))
    return {k: np.asarray(r.rx2(k), dtype=np.float64) for k in ("y", "x", "mu", "sigma")}


@pytest.mark.r_parity
def test_zipolss_glmboostlss_parity():
    mstop, nu = 100, 0.1
    r = _r_glm(mstop, nu)
    m = glmboostLSS("y ~ x1 + x2",
                    data={"y": r["y"], "x1": r["x1"], "x2": r["x2"]},
                    families=ZIPoLSS(), control=mboost.boost_control(mstop, nu))
    for p in PARAMS:
        np.testing.assert_allclose(
            m.predict(parameter=p, type="link"), r[p], rtol=RTOL, atol=ATOL,
            err_msg=f"ZIPoLSS {p} (link) does not match R")


@pytest.mark.r_parity
def test_zipolss_gamboostlss_bbs_parity():
    mstop, nu = 100, 0.1
    r = _r_gam(mstop, nu)
    m = gamboostLSS("y ~ bbs(x, knots=20, df=4)",
                    data={"y": r["y"], "x": r["x"]},
                    families=ZIPoLSS(), control=mboost.boost_control(mstop, nu))
    for p in PARAMS:
        np.testing.assert_allclose(
            m.predict(parameter=p, type="link"), r[p], rtol=RTOL, atol=ATOL,
            err_msg=f"ZIPoLSS bbs {p} (link) does not match R")
