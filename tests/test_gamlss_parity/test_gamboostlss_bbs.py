"""R-parity tests for gamboostLSS with smooth (bbs) base-learners.

Verifies the cyclic LSS fit with bbs learners against R's
``gamboostLSS::gamboostLSS(y ~ bbs(x), families = GaussianLSS())`` value-for-
value on the link and response scales (in-sample predictions).

Requires R packages mboost + gamboostLSS; skipped otherwise.
"""

import mboostrs as mboost  # noqa: F401  (ensures the extension is importable)
import numpy as np
import pytest
from mboostrs.gamlss import GaussianLSS, gamboostLSS
from conftest import r_call

RTOL = 1e-7
ATOL = 1e-7


def _run_r(mstop, nu, n=200, seed=1):
    import rpy2.robjects as ro

    r_func = ro.r("""
    function(mstop, nu, n, seed) {
        if (!requireNamespace("gamboostLSS", quietly = TRUE))
            stop("gamboostLSS not installed")
        suppressMessages(library(gamboostLSS))
        set.seed(seed)
        x <- runif(n)
        mu_true <- 2 + sin(2 * pi * x)
        sigma_true <- exp(0.3 + 0.5 * x)
        y <- rnorm(n, mean = mu_true, sd = sigma_true)
        m <- gamboostLSS(
            y ~ bbs(x, knots = 20, df = 4),
            data = data.frame(y = y, x = x),
            families = GaussianLSS(),
            control = boost_control(mstop = mstop, nu = nu))
        list(
            x = as.numeric(x), y = as.numeric(y),
            mu_link = as.numeric(predict(m, parameter = "mu", type = "link")),
            sigma_link = as.numeric(predict(m, parameter = "sigma", type = "link")),
            sigma_resp = as.numeric(predict(m, parameter = "sigma", type = "response"))
        )
    }
    """)
    res = r_call(r_func, mstop, nu, n, seed, packages=("gamboostLSS",))
    return {k: np.asarray(res.rx2(k), dtype=np.float64)
            for k in ("x", "y", "mu_link", "sigma_link", "sigma_resp")}


@pytest.mark.r_parity
def test_gamboostlss_bbs_parity():
    mstop, nu = 100, 0.1
    r = _run_r(mstop, nu)
    data = {"y": r["y"], "x": r["x"]}

    m = gamboostLSS(
        "y ~ bbs(x, knots=20, df=4)", data=data, families=GaussianLSS(),
        control=mboost.boost_control(mstop, nu), response_scale=None,
    )

    np.testing.assert_allclose(
        m.predict(parameter="mu", type="link"), r["mu_link"], rtol=RTOL, atol=ATOL,
        err_msg="gamboostLSS mu (link) does not match R",
    )
    np.testing.assert_allclose(
        m.predict(parameter="sigma", type="link"), r["sigma_link"], rtol=RTOL, atol=ATOL,
        err_msg="gamboostLSS sigma (link) does not match R",
    )
    np.testing.assert_allclose(
        m.predict(parameter="sigma", type="response"), r["sigma_resp"], rtol=RTOL, atol=ATOL,
        err_msg="gamboostLSS sigma (response) does not match R",
    )


@pytest.mark.r_parity
def test_gamboostlss_bbs_per_param_mstop():
    """Per-parameter mstop (R's mstop<- vector) with bbs learners."""
    import rpy2.robjects as ro

    mstop, nu = 80, 0.1
    r = _run_r(mstop, nu)
    data = {"y": r["y"], "x": r["x"]}

    r_func = ro.r("""
    function(y, x, mstop, nu, m_mu, m_sigma) {
        suppressMessages(library(gamboostLSS))
        m <- gamboostLSS(
            y ~ bbs(x, knots = 20, df = 4),
            data = data.frame(y = y, x = x),
            families = GaussianLSS(),
            control = boost_control(mstop = mstop, nu = nu))
        mstop(m) <- c(mu = m_mu, sigma = m_sigma)
        list(
            mu_link = as.numeric(predict(m, parameter = "mu", type = "link")),
            sigma_link = as.numeric(predict(m, parameter = "sigma", type = "link")))
    }
    """)
    res = r_func(ro.FloatVector(r["y"]), ro.FloatVector(r["x"]), mstop, nu, 60, 30)
    r_mu = np.asarray(res.rx2("mu_link"), dtype=np.float64)
    r_sigma = np.asarray(res.rx2("sigma_link"), dtype=np.float64)

    m = gamboostLSS(
        "y ~ bbs(x, knots=20, df=4)", data=data, families=GaussianLSS(),
        control=mboost.boost_control(mstop, nu), mstop={"mu": 60, "sigma": 30},
        response_scale=None,
    )
    np.testing.assert_allclose(
        m.predict(parameter="mu", type="link"), r_mu, rtol=RTOL, atol=ATOL,
        err_msg="gamboostLSS per-param mstop mu does not match R",
    )
    np.testing.assert_allclose(
        m.predict(parameter="sigma", type="link"), r_sigma, rtol=RTOL, atol=ATOL,
        err_msg="gamboostLSS per-param mstop sigma does not match R",
    )
