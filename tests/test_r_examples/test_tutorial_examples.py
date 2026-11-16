"""Parity tests for examples from R mboost's tutorial vignette.

Source: ``mboost_tutorial.Rnw`` — bodyfat data (smooth effects via gamboost),
cyclic P-splines, and the AIC-based mstop selection via cvrisk.

Gated by ``@pytest.mark.r_parity``; requires R + rpy2 + mboost + TH.data.
"""

import mboostrs as mboost
import numpy as np
import pytest
from mboostrs.families import Gaussian
from conftest import r_call

RTOL = 1e-5
ATOL = 1e-6


# ── bodyfat smooth effects ────────────────────────────────────────────

def _run_r_bodyfat_gamboost():
    """R's gamboost with bbs smooth effects on bodyfat (requires TH.data)."""
    import rpy2.robjects as ro

    r_code = """
    function() {
        suppressMessages(library(mboost))
        if (!requireNamespace("TH.data", quietly = TRUE))
            stop("TH.data not installed")
        data("bodyfat", package = "TH.data")
        gam1 <- gamboost(DEXfat ~ bbs(hipcirc) + bbs(kneebreadth) + bbs(anthro3a),
                         data = bodyfat)
        list(
            hipcirc     = as.numeric(bodyfat$hipcirc),
            kneebreadth = as.numeric(bodyfat$kneebreadth),
            anthro3a    = as.numeric(bodyfat$anthro3a),
            y           = as.numeric(bodyfat$DEXfat),
            fitted      = as.numeric(predict(gam1, type = "link")),
            risk        = as.numeric(risk(gam1)),
            offset      = as.numeric(gam1$offset)
        )
    }
    """
    res = r_call(ro.r(r_code), packages=("mboost", "TH.data"))

    return {
        "hipcirc": np.asarray(res[0], dtype=np.float64),
        "kneebreadth": np.asarray(res[1], dtype=np.float64),
        "anthro3a": np.asarray(res[2], dtype=np.float64),
        "y": np.asarray(res[3], dtype=np.float64),
        "fitted": np.asarray(res[4], dtype=np.float64),
        "risk": np.asarray(res[5], dtype=np.float64),
        "offset": float(res[6][0]),
    }


@pytest.mark.r_parity
class TestBodyfatGamboostSmooth:
    """``gamboost(DEXfat ~ bbs(hipcirc) + bbs(kneebreadth) + bbs(anthro3a))``."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.r = _run_r_bodyfat_gamboost()
        self.data = {
            "y": self.r["y"],
            "hipcirc": self.r["hipcirc"],
            "kneebreadth": self.r["kneebreadth"],
            "anthro3a": self.r["anthro3a"],
        }

    def test_fitted(self):
        m = mboost.gamboost(
            "y ~ bbs(hipcirc) + bbs(kneebreadth) + bbs(anthro3a)",
            data=self.data, family=Gaussian(),
            control=mboost.boost_control(100, 0.1),
        )
        np.testing.assert_allclose(m.fitted_, self.r["fitted"], rtol=RTOL, atol=ATOL,
                                   err_msg="bodyfat smooth fitted mismatch")

    def test_risk(self):
        m = mboost.gamboost(
            "y ~ bbs(hipcirc) + bbs(kneebreadth) + bbs(anthro3a)",
            data=self.data, family=Gaussian(),
            control=mboost.boost_control(100, 0.1),
        )
        np.testing.assert_allclose(m.risk_, self.r["risk"], rtol=RTOL, atol=ATOL,
                                   err_msg="bodyfat smooth risk mismatch")


# ── Cyclic P-splines ──────────────────────────────────────────────────

def _run_r_cyclic_splines(mstop, seed=1907):
    """R's P-spline gamboost from the tutorial vignette (non-cyclic).

    Note: cyclic P-splines (``cyclic=TRUE``) are not yet supported by the
    mboostrs Python formula parser, so we test the non-cyclic variant on
    the same data.
    """
    import rpy2.robjects as ro

    r_code = """
    function(mstop, seed) {
        suppressMessages(library(mboost))
        set.seed(seed)
        true_f <- function(x) cos(x) + 0.25 * sin(4 * x)
        x <- runif(150, 0, 2 * pi)
        y <- rnorm(150, mean = true_f(x), sd = 0.1)
        mod <- gamboost(y ~ bbs(x, knots = 20),
                        control = boost_control(mstop = mstop))
        list(
            x      = as.numeric(x),
            y      = as.numeric(y),
            fitted = as.numeric(predict(mod, type = "link")),
            risk   = as.numeric(risk(mod)),
            offset = as.numeric(mod$offset)
        )
    }
    """
    res = ro.r(r_code)(mstop, seed)
    return {
        "x": np.asarray(res[0], dtype=np.float64),
        "y": np.asarray(res[1], dtype=np.float64),
        "fitted": np.asarray(res[2], dtype=np.float64),
        "risk": np.asarray(res[3], dtype=np.float64),
        "offset": float(res[4][0]),
    }


@pytest.mark.r_parity
class TestCyclicSplines:
    """P-spline ``bbs(x, knots=20)`` on periodic data (non-cyclic variant).

    Cyclic P-splines (``cyclic=TRUE``) are not yet supported by the mboostrs
    Python formula parser. We test the non-cyclic variant on the same data
    from the tutorial vignette.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.r = _run_r_cyclic_splines(mstop=500)
        self.data = {"y": self.r["y"], "x": self.r["x"]}

    def test_fitted(self):
        m = mboost.gamboost(
            "y ~ bbs(x, knots=20)",
            data=self.data, family=Gaussian(),
            control=mboost.boost_control(500, 0.1),
        )
        np.testing.assert_allclose(m.fitted_, self.r["fitted"], rtol=RTOL, atol=ATOL,
                                   err_msg="P-spline fitted mismatch")

    def test_risk(self):
        m = mboost.gamboost(
            "y ~ bbs(x, knots=20)",
            data=self.data, family=Gaussian(),
            control=mboost.boost_control(500, 0.1),
        )
        np.testing.assert_allclose(m.risk_, self.r["risk"], rtol=RTOL, atol=ATOL,
                                   err_msg="P-spline risk mismatch")


# ── bspatial surface (from tutorial) ──────────────────────────────────

def _run_r_bspatial_tutorial(mstop, seed=1907):
    """R's bspatial from the tutorial vignette."""
    import rpy2.robjects as ro

    r_code = """
    function(mstop, seed) {
        suppressMessages(library(mboost))
        set.seed(seed)
        x1 <- runif(250, -pi, pi)
        x2 <- runif(250, -pi, pi)
        y <- sin(x1) * sin(x2) + rnorm(250, sd = 0.4)
        mod <- gamboost(y ~ bspatial(x1, x2, knots = list(x1 = 12, x2 = 12)),
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


@pytest.mark.r_parity
class TestBspatialTutorial:
    """``gamboost(y ~ bspatial(x1, x2, knots=list(x1=12, x2=12)))`` from tutorial."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.r = _run_r_bspatial_tutorial(mstop=100)
        self.data = {"y": self.r["y"], "x1": self.r["x1"], "x2": self.r["x2"]}

    def test_fitted(self):
        m = mboost.gamboost(
            "y ~ bspatial(x1, x2, knots=12)",
            data=self.data, family=Gaussian(),
            control=mboost.boost_control(100, 0.1),
        )
        np.testing.assert_allclose(m.fitted_, self.r["fitted"], rtol=RTOL, atol=ATOL,
                                   err_msg="bspatial tutorial fitted mismatch")

    def test_risk(self):
        m = mboost.gamboost(
            "y ~ bspatial(x1, x2, knots=12)",
            data=self.data, family=Gaussian(),
            control=mboost.boost_control(100, 0.1),
        )
        np.testing.assert_allclose(m.risk_, self.r["risk"], rtol=RTOL, atol=ATOL,
                                   err_msg="bspatial tutorial risk mismatch")
