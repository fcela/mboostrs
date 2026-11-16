"""Parity tests for the glmboost() examples from R mboost's glmboost.Rd.

Source: ``example("glmboost", package = "mboost")`` — the cars data
examples testing Gaussian (centered / uncentered), Laplace, and Huber families.

All tests are gated by ``@pytest.mark.r_parity`` and require R + rpy2 +
mboost installed.  Data is generated inside R so that RNG sequences match
exactly, then re-fitted with the mboostrs Python API.
"""

import mboostrs as mboost
import numpy as np
import pytest
from mboostrs.families import Gaussian, Huber, Laplace

RTOL = 1e-6
ATOL = 1e-7


def _run_r_cars_glmboost(family_name, mstop, center=True):
    """Run R's mboost glmboost on the built-in *cars* dataset.

    Returns a dict with ``speed``, ``dist``, ``fitted``, ``coef``, ``risk``,
    and ``offset`` arrays.
    """
    import rpy2.robjects as ro

    family_r = {
        "Gaussian": "Gaussian()",
        "Laplace": "Laplace()",
        "Huber": "Huber()",
    }[family_name]

    r_code = f"""
    function(mstop, center) {{
        suppressMessages(library(mboost))
        data(cars, package = "datasets")
        fit <- glmboost(dist ~ speed, data = cars,
                        control = boost_control(mstop = mstop),
                        family = {family_r},
                        center = center)
        cf <- coef(fit, off2int = TRUE)
        list(
            speed    = as.numeric(cars$speed),
            dist     = as.numeric(cars$dist),
            fitted   = as.numeric(predict(fit, type = "link")),
            coef     = as.numeric(cf),
            risk     = as.numeric(risk(fit)),
            offset   = as.numeric(fit$offset)
        )
    }}
    """
    r_func = ro.r(r_code)
    res = r_func(mstop, center)
    return {
        "speed": np.asarray(res[0], dtype=np.float64),
        "dist": np.asarray(res[1], dtype=np.float64),
        "fitted": np.asarray(res[2], dtype=np.float64),
        "coef": np.asarray(res[3], dtype=np.float64),
        "risk": np.asarray(res[4], dtype=np.float64),
        "offset": float(res[5][0]),
    }


# ── Gaussian (centered) ────────────────────────────────────────────────

@pytest.mark.r_parity
class TestGlmboostCarsGaussian:
    """``glmboost(dist ~ speed, cars, family = Gaussian(), center = TRUE)``."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.r = _run_r_cars_glmboost("Gaussian", mstop=2000, center=True)
        self.data = {"dist": self.r["dist"], "speed": self.r["speed"]}

    def test_fitted(self):
        m = mboost.glmboost("dist ~ speed", data=self.data, family=Gaussian(),
                            control=mboost.boost_control(2000, 0.1))
        np.testing.assert_allclose(m.fitted_, self.r["fitted"], rtol=RTOL, atol=ATOL,
                                   err_msg="Gaussian fitted mismatch")

    def test_risk(self):
        m = mboost.glmboost("dist ~ speed", data=self.data, family=Gaussian(),
                            control=mboost.boost_control(2000, 0.1))
        np.testing.assert_allclose(m.risk_, self.r["risk"], rtol=RTOL, atol=ATOL,
                                   err_msg="Gaussian risk mismatch")

    def test_offset(self):
        m = mboost.glmboost("dist ~ speed", data=self.data, family=Gaussian(),
                            control=mboost.boost_control(2000, 0.1))
        np.testing.assert_allclose(m.offset_, self.r["offset"], rtol=1e-6,
                                   err_msg="Gaussian offset mismatch")

    def test_coef_near_ols(self):
        """At mstop=2000 the boosted coefficients should be close to OLS."""
        m = mboost.glmboost("dist ~ speed", data=self.data, family=Gaussian(),
                            control=mboost.boost_control(2000, 0.1))
        cf = m.coef()
        slope = cf.get("speed", cf.get("speed", 0))
        # R's lm slope is ~3.9324; check we're in the right ballpark
        assert abs(slope - 3.93) < 0.5, f"slope={slope} far from OLS ~3.93"


# ── Laplace ────────────────────────────────────────────────────────────

@pytest.mark.r_parity
class TestGlmboostCarsLaplace:
    """``glmboost(dist ~ speed, cars, family = Laplace())``."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.r = _run_r_cars_glmboost("Laplace", mstop=1000, center=True)
        self.data = {"dist": self.r["dist"], "speed": self.r["speed"]}

    def test_fitted(self):
        m = mboost.glmboost("dist ~ speed", data=self.data, family=Laplace(),
                            control=mboost.boost_control(1000, 0.1))
        np.testing.assert_allclose(m.fitted_, self.r["fitted"], rtol=1e-6, atol=1e-6,
                                   err_msg="Laplace fitted mismatch")


# ── Huber ──────────────────────────────────────────────────────────────

@pytest.mark.r_parity
class TestGlmboostCarsHuber:
    """``glmboost(dist ~ speed, cars, family = Huber())``."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.r = _run_r_cars_glmboost("Huber", mstop=1000, center=True)
        self.data = {"dist": self.r["dist"], "speed": self.r["speed"]}

    def test_fitted(self):
        m = mboost.glmboost("dist ~ speed", data=self.data, family=Huber(),
                            control=mboost.boost_control(1000, 0.1))
        np.testing.assert_allclose(m.fitted_, self.r["fitted"], rtol=1e-5, atol=1e-5,
                                   err_msg="Huber fitted mismatch")
