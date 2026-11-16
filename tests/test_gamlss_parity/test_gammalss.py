"""R-parity tests for the GammaLSS family (glmboostLSS + gamboostLSS).

Verifies the cyclic LSS fit with the Gamma location-shape family against R's
``gamboostLSS`` value-for-value. GammaLSS uses log links for both mu (mean) and
sigma (shape); its sigma gradient needs the digamma function and its offsets are
Brent minima of the per-parameter risk (matching R's ``optimize``).

Requires R packages mboost + gamboostLSS; skipped otherwise.
"""

import mboostrs as mboost  # noqa: F401
import numpy as np
import pytest
from mboostrs.gamlss import GammaLSS, gamboostLSS, glmboostLSS
from conftest import r_call

RTOL = 1e-6
ATOL = 1e-7


def _r_glm(mstop, nu, n=200, seed=1):
    import rpy2.robjects as ro
    f = ro.r("""
    function(mstop, nu, n, seed) {
        if (!requireNamespace("gamboostLSS", quietly = TRUE)) stop("no gamboostLSS")
        suppressMessages(library(gamboostLSS))
        set.seed(seed)
        x1 <- rnorm(n); x2 <- rnorm(n)
        mu <- exp(0.5 + 0.3 * x1); sig <- exp(0.2 + 0.2 * x2)
        y <- rgamma(n, shape = sig, scale = mu / sig)
        m <- glmboostLSS(y ~ x1 + x2, data = data.frame(y = y, x1 = x1, x2 = x2),
                         families = GammaLSS(),
                         control = boost_control(mstop = mstop, nu = nu))
        list(x1 = as.numeric(x1), x2 = as.numeric(x2), y = as.numeric(y),
             mu = as.numeric(predict(m, parameter = "mu", type = "link")),
             sigma = as.numeric(predict(m, parameter = "sigma", type = "link")),
             sigma_r = as.numeric(predict(m, parameter = "sigma", type = "response")))
    }
    """)
    r = r_call(f, mstop, nu, n, seed, packages=("gamboostLSS",))
    return {k: np.asarray(r.rx2(k), dtype=np.float64)
            for k in ("x1", "x2", "y", "mu", "sigma", "sigma_r")}


def _r_gam(mstop, nu, n=200, seed=1):
    import rpy2.robjects as ro
    f = ro.r("""
    function(mstop, nu, n, seed) {
        if (!requireNamespace("gamboostLSS", quietly = TRUE)) stop("no gamboostLSS")
        suppressMessages(library(gamboostLSS))
        set.seed(seed)
        x <- runif(n, -1, 1)
        mu <- exp(0.5 + 0.5 * sin(2 * x)); sig <- exp(0.3 + 0.3 * x)
        y <- rgamma(n, shape = sig, scale = mu / sig)
        m <- gamboostLSS(y ~ bbs(x, knots = 20, df = 4),
                         data = data.frame(y = y, x = x), families = GammaLSS(),
                         control = boost_control(mstop = mstop, nu = nu))
        list(x = as.numeric(x), y = as.numeric(y),
             mu = as.numeric(predict(m, parameter = "mu", type = "link")),
             sigma = as.numeric(predict(m, parameter = "sigma", type = "link")))
    }
    """)
    r = r_call(f, mstop, nu, n, seed, packages=("gamboostLSS",))
    return {k: np.asarray(r.rx2(k), dtype=np.float64) for k in ("x", "y", "mu", "sigma")}


@pytest.mark.r_parity
def test_gammalss_glmboostlss_parity():
    mstop, nu = 100, 0.1
    r = _r_glm(mstop, nu)
    m = glmboostLSS("y ~ x1 + x2", data={"y": r["y"], "x1": r["x1"], "x2": r["x2"]},
                    families=GammaLSS(), control=mboost.boost_control(mstop, nu))
    np.testing.assert_allclose(m.predict(parameter="mu", type="link"), r["mu"],
                               rtol=RTOL, atol=ATOL, err_msg="GammaLSS mu (link)")
    np.testing.assert_allclose(m.predict(parameter="sigma", type="link"), r["sigma"],
                               rtol=RTOL, atol=ATOL, err_msg="GammaLSS sigma (link)")
    np.testing.assert_allclose(m.predict(parameter="sigma", type="response"), r["sigma_r"],
                               rtol=RTOL, atol=ATOL, err_msg="GammaLSS sigma (response)")


@pytest.mark.r_parity
def test_gammalss_gamboostlss_bbs_parity():
    mstop, nu = 100, 0.1
    r = _r_gam(mstop, nu)
    m = gamboostLSS("y ~ bbs(x, knots=20, df=4)", data={"y": r["y"], "x": r["x"]},
                    families=GammaLSS(), control=mboost.boost_control(mstop, nu))
    np.testing.assert_allclose(m.predict(parameter="mu", type="link"), r["mu"],
                               rtol=RTOL, atol=ATOL, err_msg="GammaLSS bbs mu (link)")
    np.testing.assert_allclose(m.predict(parameter="sigma", type="link"), r["sigma"],
                               rtol=RTOL, atol=ATOL, err_msg="GammaLSS bbs sigma (link)")
