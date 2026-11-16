"""Parity tests for the gamboost() examples from R mboost's gamboost.Rd.

Source: ``example("gamboost", package = "mboost")`` — the cars data and the
artificial sinus transformation example.

Gated by ``@pytest.mark.r_parity``; requires R + rpy2 + mboost.
"""

import mboostrs as mboost
import numpy as np
import pytest
from mboostrs.families import Gaussian

RTOL = 1e-6
ATOL = 1e-9


def _run_r_gamboost_cars(mstop):
    """R's gamboost on *cars* with ``bbs(speed, dfbase=4)``."""
    import rpy2.robjects as ro

    r_code = """
    function(mstop) {
        suppressMessages(library(mboost))
        data(cars, package = "datasets")
        fit <- gamboost(dist ~ speed, data = cars, dfbase = 4,
                        control = boost_control(mstop = mstop))
        list(
            speed  = as.numeric(cars$speed),
            dist   = as.numeric(cars$dist),
            fitted = as.numeric(predict(fit, type = "link")),
            risk   = as.numeric(risk(fit)),
            offset = as.numeric(fit$offset)
        )
    }
    """
    res = ro.r(r_code)(mstop)
    return {
        "speed": np.asarray(res[0], dtype=np.float64),
        "dist": np.asarray(res[1], dtype=np.float64),
        "fitted": np.asarray(res[2], dtype=np.float64),
        "risk": np.asarray(res[3], dtype=np.float64),
        "offset": float(res[4][0]),
    }


def _run_r_gamboost_sinus(mstop, seed=42):
    """R's gamboost on artificial sinus data."""
    import rpy2.robjects as ro

    r_code = """
    function(mstop, seed) {
        suppressMessages(library(mboost))
        set.seed(seed)
        x <- sort(runif(100)) * 10
        y <- sin(x) + rnorm(length(x), sd = 0.25)
        fit <- gamboost(y ~ x,
                        control = boost_control(mstop = mstop))
        list(
            x      = as.numeric(x),
            y      = as.numeric(y),
            fitted = as.numeric(predict(fit, type = "link")),
            risk   = as.numeric(risk(fit)),
            offset = as.numeric(fit$offset)
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


# ── Cars data ──────────────────────────────────────────────────────────

@pytest.mark.r_parity
class TestGamboostCars:
    """``gamboost(dist ~ speed, cars, dfbase = 4, mstop = 50)``."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.r = _run_r_gamboost_cars(mstop=50)
        self.data = {"dist": self.r["dist"], "speed": self.r["speed"]}

    def test_fitted(self):
        m = mboost.gamboost("dist ~ bbs(speed, df=4)", data=self.data,
                            family=Gaussian(),
                            control=mboost.boost_control(50, 0.1))
        np.testing.assert_allclose(m.fitted_, self.r["fitted"], rtol=RTOL, atol=ATOL,
                                   err_msg="gamboost cars fitted mismatch")

    def test_risk(self):
        m = mboost.gamboost("dist ~ bbs(speed, df=4)", data=self.data,
                            family=Gaussian(),
                            control=mboost.boost_control(50, 0.1))
        np.testing.assert_allclose(m.risk_, self.r["risk"], rtol=RTOL, atol=ATOL,
                                   err_msg="gamboost cars risk mismatch")

    def test_offset(self):
        m = mboost.gamboost("dist ~ bbs(speed, df=4)", data=self.data,
                            family=Gaussian(),
                            control=mboost.boost_control(50, 0.1))
        np.testing.assert_allclose(m.offset_, self.r["offset"], rtol=RTOL, atol=ATOL,
                                   err_msg="gamboost cars offset mismatch")


# ── Sinus transformation ───────────────────────────────────────────────

@pytest.mark.r_parity
class TestGamboostSinus:
    """``gamboost(y ~ x, sinus data, mstop = 500)``."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.r = _run_r_gamboost_sinus(mstop=500)
        self.data = {"y": self.r["y"], "x": self.r["x"]}

    def test_fitted(self):
        m = mboost.gamboost("y ~ bbs(x)", data=self.data, family=Gaussian(),
                            control=mboost.boost_control(500, 0.1))
        np.testing.assert_allclose(m.fitted_, self.r["fitted"], rtol=RTOL, atol=ATOL,
                                   err_msg="gamboost sinus fitted mismatch")

    def test_risk(self):
        m = mboost.gamboost("y ~ bbs(x)", data=self.data, family=Gaussian(),
                            control=mboost.boost_control(500, 0.1))
        np.testing.assert_allclose(m.risk_, self.r["risk"], rtol=RTOL, atol=ATOL,
                                   err_msg="gamboost sinus risk mismatch")
