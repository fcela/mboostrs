"""R-parity tests for glmboostLSS with the GaussianLSS family.

Verifies the cyclic LSS fit (mboostrs_core::lss) against R's
``gamboostLSS::glmboostLSS(..., families = GaussianLSS())`` value-for-value:
per-parameter fitted predictors on the link scale (mu, eta_sigma) and the
response scale (sigma = exp(eta_sigma)).

Requires R packages mboost + gamboostLSS; skipped (via --run-r-parity gating and
an import guard) otherwise.
"""

import mboostrs as mboost  # noqa: F401  (ensures the extension is importable)
import numpy as np
import pytest
from mboostrs.gamlss import GaussianLSS, glmboostLSS
from conftest import r_call

RTOL = 1e-8
ATOL = 1e-8


def _run_r(mstop, nu, n=200, seed=1):
    import rpy2.robjects as ro

    r_func = ro.r("""
    function(mstop, nu, n, seed) {
        if (!requireNamespace("gamboostLSS", quietly = TRUE))
            stop("gamboostLSS not installed")
        suppressMessages(library(gamboostLSS))
        set.seed(seed)
        x1 <- rnorm(n); x2 <- rnorm(n)
        mu_true <- 1 + 0.8 * x1
        sigma_true <- exp(0.3 * x2)
        y <- rnorm(n, mean = mu_true, sd = sigma_true)
        m <- glmboostLSS(
            y ~ x1 + x2, data = data.frame(y = y, x1 = x1, x2 = x2),
            families = GaussianLSS(),
            control = boost_control(mstop = mstop, nu = nu))
        list(
            x1 = as.numeric(x1), x2 = as.numeric(x2), y = as.numeric(y),
            mu_link = as.numeric(predict(m, parameter = "mu", type = "link")),
            sigma_link = as.numeric(predict(m, parameter = "sigma", type = "link")),
            sigma_resp = as.numeric(predict(m, parameter = "sigma", type = "response"))
        )
    }
    """)
    res = r_call(r_func, mstop, nu, n, seed, packages=("gamboostLSS",))
    return {k: np.asarray(res.rx2(k), dtype=np.float64)
            for k in ("x1", "x2", "y", "mu_link", "sigma_link", "sigma_resp")}


@pytest.mark.r_parity
def test_gaussianlss_glmboostlss_parity():
    mstop, nu = 100, 0.1
    r = _run_r(mstop, nu)
    data = {"y": r["y"], "x1": r["x1"], "x2": r["x2"]}

    m = glmboostLSS(
        "y ~ x1 + x2", data=data, families=GaussianLSS(),
        control=mboost.boost_control(mstop, nu),
    )

    np.testing.assert_allclose(
        m.predict(parameter="mu", type="link"), r["mu_link"], rtol=RTOL, atol=ATOL,
        err_msg="GaussianLSS mu (link) does not match R glmboostLSS",
    )
    np.testing.assert_allclose(
        m.predict(parameter="sigma", type="link"), r["sigma_link"], rtol=RTOL, atol=ATOL,
        err_msg="GaussianLSS eta_sigma (link) does not match R glmboostLSS",
    )
    np.testing.assert_allclose(
        m.predict(parameter="sigma", type="response"), r["sigma_resp"], rtol=RTOL, atol=ATOL,
        err_msg="GaussianLSS sigma (response) does not match R glmboostLSS",
    )


def _run_r_per_param_mstop(max_mstop, nu, mu_m, sigma_m, n=200, seed=1):
    import rpy2.robjects as ro

    r_func = ro.r("""
    function(max_mstop, nu, mu_m, sigma_m, n, seed) {
        if (!requireNamespace("gamboostLSS", quietly = TRUE))
            stop("gamboostLSS not installed")
        suppressMessages(library(gamboostLSS))
        set.seed(seed)
        x1 <- rnorm(n); x2 <- rnorm(n)
        y <- rnorm(n, mean = 1 + 0.8 * x1, sd = exp(0.3 * x2))
        m <- glmboostLSS(
            y ~ x1 + x2, data = data.frame(y = y, x1 = x1, x2 = x2),
            families = GaussianLSS(),
            control = boost_control(mstop = max_mstop, nu = nu))
        mstop(m) <- c(mu = mu_m, sigma = sigma_m)
        list(
            x1 = as.numeric(x1), x2 = as.numeric(x2), y = as.numeric(y),
            mu_link = as.numeric(predict(m, parameter = "mu", type = "link")),
            sigma_link = as.numeric(predict(m, parameter = "sigma", type = "link"))
        )
    }
    """)
    res = r_call(r_func, max_mstop, nu, mu_m, sigma_m, n, seed, packages=("gamboostLSS",))
    return {k: np.asarray(res.rx2(k), dtype=np.float64)
            for k in ("x1", "x2", "y", "mu_link", "sigma_link")}


@pytest.mark.r_parity
def test_gaussianlss_per_param_mstop_parity():
    # R's `mstop(m) <- c(mu=.., sigma=..)` per-parameter budget matches our
    # cyclic engine (parameter freezes once its budget is hit).
    mu_m, sigma_m = 50, 20
    r = _run_r_per_param_mstop(60, 0.1, mu_m, sigma_m)
    data = {"y": r["y"], "x1": r["x1"], "x2": r["x2"]}
    m = glmboostLSS(
        "y ~ x1 + x2", data=data, families=GaussianLSS(),
        control=mboost.boost_control(60, 0.1), mstop={"mu": mu_m, "sigma": sigma_m},
    )
    assert m.mstop == {"mu": mu_m, "sigma": sigma_m}
    np.testing.assert_allclose(
        m.predict(parameter="mu", type="link"), r["mu_link"], rtol=RTOL, atol=ATOL,
        err_msg="per-parameter mstop mu does not match R",
    )
    np.testing.assert_allclose(
        m.predict(parameter="sigma", type="link"), r["sigma_link"], rtol=RTOL, atol=ATOL,
        err_msg="per-parameter mstop sigma does not match R",
    )


@pytest.mark.r_parity
def test_gaussianlss_predict_newdata_consistency():
    # predict(newdata=training) must reproduce the stored fitted predictors.
    mstop, nu = 50, 0.1
    r = _run_r(mstop, nu, seed=7)
    data = {"y": r["y"], "x1": r["x1"], "x2": r["x2"]}
    m = glmboostLSS("y ~ x1 + x2", data=data, families=GaussianLSS(),
                    control=mboost.boost_control(mstop, nu))
    for param in ("mu", "sigma"):
        fitted = m.predict(parameter=param, type="link")
        on_newdata = m.predict(newdata=data, parameter=param, type="link")
        np.testing.assert_allclose(fitted, on_newdata, rtol=1e-12, atol=1e-12)
