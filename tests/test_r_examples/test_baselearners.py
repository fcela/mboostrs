"""Parity tests for base-learner examples from R mboost's baselearners.Rd.

Source: ``example("baselearners", package = "mboost")`` — tests multi-learner
gamboost (bbs + bols with factors), bspatial surfaces, and brandom.

Gated by ``@pytest.mark.r_parity``; requires R + rpy2 + mboost.
"""

import mboostrs as mboost
import numpy as np
import pytest
from mboostrs.families import Gaussian

RTOL = 1e-5
ATOL = 1e-6


def _run_r_multi_learner(mstop, seed=290875):
    """R's multi-learner gamboost: bbs(x1) + bbs(x2) + bols(x2).

    Uses continuous predictors only — R's factor encoding (dummy columns)
    is not replicated by the Python formula parser.
    """
    import rpy2.robjects as ro

    r_code = """
    function(mstop, seed) {
        suppressMessages(library(mboost))
        set.seed(seed)
        n <- 100
        x1 <- rnorm(n)
        x2 <- rnorm(n) + 0.25 * x1
        y <- 3 * sin(x1) + x2^2 + rnorm(n)
        mod <- gamboost(y ~ bbs(x1, knots = 20, df = 4) +
                            bbs(x2, knots = 5, df = 5) +
                            bols(x2),
                        control = boost_control(mstop = mstop))
        list(
            x1 = as.numeric(x1), x2 = as.numeric(x2),
            y  = as.numeric(y),
            fitted = as.numeric(predict(mod, type = "link")),
            risk   = as.numeric(risk(mod)),
            offset = as.numeric(mod$offset)
        )
    }
    """
    res = ro.r(r_code)(mstop, seed)
    return {
        "x1": np.asarray(res[0], dtype=np.float64),
        "x2": np.asarray(res[1], dtype=np.float64),
        "y": np.asarray(res[2], dtype=np.float64),
        "fitted": np.asarray(res[3], dtype=np.float64),
        "risk": np.asarray(res[4], dtype=np.float64),
        "offset": float(res[5][0]),
    }


def _run_r_bspatial(mstop, seed=1907):
    """R's bspatial example: sin(x1)*sin(x2) surface."""
    import rpy2.robjects as ro

    r_code = """
    function(mstop, seed) {
        suppressMessages(library(mboost))
        set.seed(seed)
        x1 <- runif(250, -pi, pi)
        x2 <- runif(250, -pi, pi)
        y <- sin(x1) * sin(x2) + rnorm(250, sd = 0.4)
        mod <- gamboost(y ~ bspatial(x1, x2, knots = 12),
                        control = boost_control(mstop = mstop))
        list(
            x1 = as.numeric(x1), x2 = as.numeric(x2),
            y  = as.numeric(y),
            fitted = as.numeric(predict(mod, type = "link")),
            risk   = as.numeric(risk(mod)),
            offset = as.numeric(mod$offset)
        )
    }
    """
    res = ro.r(r_code)(mstop, seed)
    return {
        "x1": np.asarray(res[0], dtype=np.float64),
        "x2": np.asarray(res[1], dtype=np.float64),
        "y": np.asarray(res[2], dtype=np.float64),
        "fitted": np.asarray(res[3], dtype=np.float64),
        "risk": np.asarray(res[4], dtype=np.float64),
        "offset": float(res[5][0]),
    }


def _run_r_brandom(mstop, seed=1907):
    """R's brandom example: random intercept for 10 groups."""
    import rpy2.robjects as ro

    r_code = """
    function(mstop, seed) {
        suppressMessages(library(mboost))
        set.seed(seed)
        id <- factor(rep(1:10, each = 5))
        y <- rnorm(50, mean = rep(rnorm(10), each = 5), sd = 0.1)
        mod <- gamboost(y ~ brandom(id),
                        control = boost_control(mstop = mstop))
        list(
            id  = as.numeric(id),
            y   = as.numeric(y),
            fitted = as.numeric(predict(mod, type = "link")),
            risk   = as.numeric(risk(mod)),
            offset = as.numeric(mod$offset)
        )
    }
    """
    res = ro.r(r_code)(mstop, seed)
    return {
        "id": np.asarray(res[0], dtype=np.float64).astype(int),
        "y": np.asarray(res[1], dtype=np.float64),
        "fitted": np.asarray(res[2], dtype=np.float64),
        "risk": np.asarray(res[3], dtype=np.float64),
        "offset": float(res[4][0]),
    }


# ── Multi-learner (bbs + bols) ─────────────────────────────────────────

@pytest.mark.r_parity
class TestMultiLearnerGamboost:
    """``gamboost(y ~ bbs(x1) + bbs(x2) + bols(x3) + bols(x4))``."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.r = _run_r_multi_learner(mstop=100)
        self.data = {
            "y": self.r["y"], "x1": self.r["x1"], "x2": self.r["x2"],
        }

    def test_fitted(self):
        m = mboost.gamboost(
            "y ~ bbs(x1, knots=20, df=4) + bbs(x2, knots=5, df=5) + bols(x2)",
            data=self.data, family=Gaussian(),
            control=mboost.boost_control(100, 0.1),
        )
        np.testing.assert_allclose(m.fitted_, self.r["fitted"], rtol=RTOL, atol=ATOL,
                                   err_msg="multi-learner fitted mismatch")

    def test_risk(self):
        m = mboost.gamboost(
            "y ~ bbs(x1, knots=20, df=4) + bbs(x2, knots=5, df=5) + bols(x2)",
            data=self.data, family=Gaussian(),
            control=mboost.boost_control(100, 0.1),
        )
        np.testing.assert_allclose(m.risk_, self.r["risk"], rtol=RTOL, atol=ATOL,
                                   err_msg="multi-learner risk mismatch")


# ── bspatial ───────────────────────────────────────────────────────────

@pytest.mark.r_parity
class TestBspatialGamboost:
    """``gamboost(y ~ bspatial(x1, x2, knots=12))``."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.r = _run_r_bspatial(mstop=100)
        self.data = {"y": self.r["y"], "x1": self.r["x1"], "x2": self.r["x2"]}

    def test_fitted(self):
        m = mboost.gamboost(
            "y ~ bspatial(x1, x2, knots=12)",
            data=self.data, family=Gaussian(),
            control=mboost.boost_control(100, 0.1),
        )
        np.testing.assert_allclose(m.fitted_, self.r["fitted"], rtol=RTOL, atol=ATOL,
                                   err_msg="bspatial fitted mismatch")

    def test_risk(self):
        m = mboost.gamboost(
            "y ~ bspatial(x1, x2, knots=12)",
            data=self.data, family=Gaussian(),
            control=mboost.boost_control(100, 0.1),
        )
        np.testing.assert_allclose(m.risk_, self.r["risk"], rtol=RTOL, atol=ATOL,
                                   err_msg="bspatial risk mismatch")


# ── brandom ────────────────────────────────────────────────────────────

@pytest.mark.r_parity
class TestBrandomGamboost:
    """``gamboost(y ~ brandom(id))`` — random intercept base-learner."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.r = _run_r_brandom(mstop=100)
        self.data = {"y": self.r["y"], "id": self.r["id"]}

    def test_fitted(self):
        m = mboost.gamboost(
            "y ~ brandom(id)",
            data=self.data, family=Gaussian(),
            control=mboost.boost_control(100, 0.1),
        )
        np.testing.assert_allclose(m.fitted_, self.r["fitted"], rtol=RTOL, atol=ATOL,
                                   err_msg="brandom fitted mismatch")

    def test_risk(self):
        m = mboost.gamboost(
            "y ~ brandom(id)",
            data=self.data, family=Gaussian(),
            control=mboost.boost_control(100, 0.1),
        )
        np.testing.assert_allclose(m.risk_, self.r["risk"], rtol=RTOL, atol=ATOL,
                                   err_msg="brandom risk mismatch")
