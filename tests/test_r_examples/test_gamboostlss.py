"""Parity tests for gamboostLSS examples from R's gamboostLSS package.

Source: ``example("glmboostLSS", package = "gamboostLSS")`` — NBinomialLSS
on simulated overdispersed count data.

And the gamboostLSS Tutorial vignette — GaussianLSS on toy data with
per-parameter formulas.

Gated by ``@pytest.mark.r_parity``; requires R + rpy2 + gamboostLSS.
"""

import mboostrs as mboost
import numpy as np
import pytest
from mboostrs.gamlss import GaussianLSS, NBinomialLSS, glmboostLSS
from conftest import r_call

RTOL = 1e-6
ATOL = 1e-7


# ── NBinomialLSS (from glmboostLSS example) ───────────────────────────

def _run_r_nbinomiallss(mstop, seed=1907):
    """R's glmboostLSS with NBinomialLSS on simulated count data.

    This is the canonical example from ``example("glmboostLSS")``.
    """
    import rpy2.robjects as ro

    r_code = """
    function(mstop, seed) {
        if (!requireNamespace("gamboostLSS", quietly = TRUE))
            stop("gamboostLSS not installed")
        suppressMessages(library(gamboostLSS))
        set.seed(seed)
        x1 <- rnorm(1000); x2 <- rnorm(1000)
        x3 <- rnorm(1000); x4 <- rnorm(1000)
        x5 <- rnorm(1000); x6 <- rnorm(1000)
        mu    <- exp(1.5 + 1 * x1 + 0.5 * x2 - 0.5 * x3 - 1 * x4)
        sigma <- exp(-0.4 * x3 - 0.2 * x4 + 0.2 * x5 + 0.4 * x6)
        y <- numeric(1000)
        for(i in 1:1000)
            y[i] <- rnbinom(1, size = sigma[i], mu = mu[i])
        dat <- data.frame(x1, x2, x3, x4, x5, x6, y)
        model <- glmboostLSS(y ~ ., families = NBinomialLSS(), data = dat,
                             control = boost_control(mstop = mstop), center = TRUE)
        list(
            x1 = as.numeric(x1), x2 = as.numeric(x2),
            x3 = as.numeric(x3), x4 = as.numeric(x4),
            x5 = as.numeric(x5), x6 = as.numeric(x6),
            y  = as.numeric(y),
            mu    = as.numeric(predict(model, parameter = "mu", type = "link")),
            sigma = as.numeric(predict(model, parameter = "sigma", type = "link"))
        )
    }
    """
    res = r_call(ro.r(r_code), mstop, seed, packages=("gamboostLSS",))

    return {
        k: np.asarray(res.rx2(k), dtype=np.float64)
        for k in ("x1", "x2", "x3", "x4", "x5", "x6", "y", "mu", "sigma")
    }


@pytest.mark.r_parity
class TestNBinomialLSSEample:
    """``glmboostLSS(y ~ ., families = NBinomialLSS())`` — the canonical example."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.r = _run_r_nbinomiallss(mstop=200)
        self.data = {k: self.r[k] for k in ("x1", "x2", "x3", "x4", "x5", "x6", "y")}

    def test_mu_parity(self):
        m = glmboostLSS("y ~ x1 + x2 + x3 + x4 + x5 + x6",
                        data=self.data, families=NBinomialLSS(),
                        control=mboost.boost_control(200, 0.1))
        np.testing.assert_allclose(
            m.predict(parameter="mu", type="link"), self.r["mu"],
            rtol=RTOL, atol=ATOL,
            err_msg="NBinomialLSS mu (link) mismatch",
        )

    def test_sigma_parity(self):
        m = glmboostLSS("y ~ x1 + x2 + x3 + x4 + x5 + x6",
                        data=self.data, families=NBinomialLSS(),
                        control=mboost.boost_control(200, 0.1))
        np.testing.assert_allclose(
            m.predict(parameter="sigma", type="link"), self.r["sigma"],
            rtol=RTOL, atol=ATOL,
            err_msg="NBinomialLSS sigma (link) mismatch",
        )


# ── GaussianLSS (from gamboostLSS Tutorial) ───────────────────────────

def _run_r_gaussianlss_toy(mstop, seed=1907):
    """R's glmboostLSS with GaussianLSS on toy data.

    From the gamboostLSS Tutorial vignette.
    """
    import rpy2.robjects as ro

    r_code = """
    function(mstop, seed) {
        if (!requireNamespace("gamboostLSS", quietly = TRUE))
            stop("gamboostLSS not installed")
        suppressMessages(library(gamboostLSS))
        set.seed(seed)
        n <- 150
        x1 <- rnorm(n); x2 <- rnorm(n); x3 <- rnorm(n)
        toydata <- data.frame(x1 = x1, x2 = x2, x3 = x3)
        toydata$y <- rnorm(n, mean = 1 + 2 * x1 - x2,
                                 sd = exp(0.5 - 0.25 * x1 + 0.5 * x3))
        model <- glmboostLSS(y ~ x1 + x2 + x3, data = toydata,
                             families = GaussianLSS(),
                             control = boost_control(mstop = mstop))
        list(
            x1 = as.numeric(x1), x2 = as.numeric(x2), x3 = as.numeric(x3),
            y  = as.numeric(toydata$y),
            mu    = as.numeric(predict(model, parameter = "mu", type = "link")),
            sigma = as.numeric(predict(model, parameter = "sigma", type = "link"))
        )
    }
    """
    res = r_call(ro.r(r_code), mstop, seed, packages=("gamboostLSS",))

    return {
        k: np.asarray(res.rx2(k), dtype=np.float64)
        for k in ("x1", "x2", "x3", "y", "mu", "sigma")
    }


@pytest.mark.r_parity
class TestGaussianLSSTutorial:
    """``glmboostLSS(y ~ x1 + x2 + x3, families = GaussianLSS())`` — Tutorial toy data."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.r = _run_r_gaussianlss_toy(mstop=100)
        self.data = {k: self.r[k] for k in ("x1", "x2", "x3", "y")}

    def test_mu_parity(self):
        m = glmboostLSS("y ~ x1 + x2 + x3", data=self.data,
                        families=GaussianLSS(),
                        control=mboost.boost_control(100, 0.1))
        np.testing.assert_allclose(
            m.predict(parameter="mu", type="link"), self.r["mu"],
            rtol=RTOL, atol=ATOL,
            err_msg="GaussianLSS toy mu (link) mismatch",
        )

    def test_sigma_parity(self):
        m = glmboostLSS("y ~ x1 + x2 + x3", data=self.data,
                        families=GaussianLSS(),
                        control=mboost.boost_control(100, 0.1))
        np.testing.assert_allclose(
            m.predict(parameter="sigma", type="link"), self.r["sigma"],
            rtol=RTOL, atol=ATOL,
            err_msg="GaussianLSS toy sigma (link) mismatch",
        )


# ── Per-parameter formulas (from glmboostLSS example) ──────────────────

def _run_r_nbinomiallss_per_param(mstop_mu, mstop_sigma, seed=1907):
    """R's glmboostLSS with different formulas per parameter."""
    import rpy2.robjects as ro

    r_code = """
    function(mstop_mu, mstop_sigma, seed) {
        if (!requireNamespace("gamboostLSS", quietly = TRUE))
            stop("gamboostLSS not installed")
        suppressMessages(library(gamboostLSS))
        set.seed(seed)
        x1 <- rnorm(1000); x2 <- rnorm(1000)
        x3 <- rnorm(1000); x4 <- rnorm(1000)
        x5 <- rnorm(1000); x6 <- rnorm(1000)
        mu    <- exp(1.5 + 1 * x1 + 0.5 * x2 - 0.5 * x3 - 1 * x4)
        sigma <- exp(-0.4 * x3 - 0.2 * x4 + 0.2 * x5 + 0.4 * x6)
        y <- numeric(1000)
        for(i in 1:1000)
            y[i] <- rnbinom(1, size = sigma[i], mu = mu[i])
        dat <- data.frame(x1, x2, x3, x4, x5, x6, y)
        model <- glmboostLSS(formula = list(mu = y ~ x1 + x2 + x3 + x4,
                                            sigma = y ~ x3 + x4 + x5 + x6),
                             families = NBinomialLSS(), data = dat,
                             control = boost_control(mstop = mstop_mu),
                             center = TRUE)
        mstop(model) <- c(mu = mstop_mu, sigma = mstop_sigma)
        list(
            x1 = as.numeric(x1), x2 = as.numeric(x2),
            x3 = as.numeric(x3), x4 = as.numeric(x4),
            x5 = as.numeric(x5), x6 = as.numeric(x6),
            y  = as.numeric(y),
            mu    = as.numeric(predict(model, parameter = "mu", type = "link")),
            sigma = as.numeric(predict(model, parameter = "sigma", type = "link"))
        )
    }
    """
    res = r_call(ro.r(r_code), mstop_mu, mstop_sigma, seed, packages=("gamboostLSS",))

    return {
        k: np.asarray(res.rx2(k), dtype=np.float64)
        for k in ("x1", "x2", "x3", "x4", "x5", "x6", "y", "mu", "sigma")
    }


@pytest.mark.r_parity
def test_nbinomiallss_per_param_formulas():
    """Different formulas for mu and sigma, with per-parameter mstop."""
    r = _run_r_nbinomiallss_per_param(200, 150)
    data = {k: r[k] for k in ("x1", "x2", "x3", "x4", "x5", "x6", "y")}

    m = glmboostLSS(
        {"mu": "y ~ x1 + x2 + x3 + x4",
         "sigma": "y ~ x3 + x4 + x5 + x6"},
        data=data, families=NBinomialLSS(),
        control=mboost.boost_control(200, 0.1),
        mstop={"mu": 200, "sigma": 150},
    )

    np.testing.assert_allclose(
        m.predict(parameter="mu", type="link"), r["mu"],
        rtol=RTOL, atol=ATOL,
        err_msg="NBinomialLSS per-param mu mismatch",
    )
    np.testing.assert_allclose(
        m.predict(parameter="sigma", type="link"), r["sigma"],
        rtol=RTOL, atol=ATOL,
        err_msg="NBinomialLSS per-param sigma mismatch",
    )
