"""R-parity tests for glmboost GaussReg + bols."""

import mboostrs as mboost
import numpy as np
import pytest
from mboostrs.families import GaussReg

rtol = 1e-8


def _run_r_gaussreg(mstop, nu, n=100, seed=42):
    """Generate data in R and fit glmboost with GaussReg family."""
    import rpy2.robjects as ro

    r_code = """
    function(mstop, nu, n, seed) {
        set.seed(seed)
        x1 <- rnorm(n)
        x2 <- rnorm(n)
        y <- 1 + 0.5*x1 - 0.3*x2 + rnorm(n)
        data <- data.frame(x1 = x1, x2 = x2, y = y)
        fit <- mboost::glmboost(
            y ~ x1 + x2,
            data = data,
            family = mboost::GaussReg(),
            center = TRUE,
            control = mboost::boost_control(mstop = mstop, nu = nu)
        )
        list(
            x1 = as.numeric(x1),
            x2 = as.numeric(x2),
            y = as.numeric(y),
            fitted = as.numeric(stats::predict(fit, type = "link")),
            risk = as.numeric(mboost::risk(fit)),
            offset = as.numeric(fit$offset),
            selected = as.integer(fit$xselect())
        )
    }
    """
    r_func = ro.r(r_code)
    result = r_func(mstop, nu, n, seed)

    return {
        "x1": np.asarray(result[0], dtype=np.float64),
        "x2": np.asarray(result[1], dtype=np.float64),
        "y": np.asarray(result[2], dtype=np.float64),
        "fitted": np.asarray(result[3], dtype=np.float64),
        "risk": np.asarray(result[4], dtype=np.float64),
        "offset": float(result[5][0]) if len(result[5]) > 0 else 0.0,
        "selected": [int(s) - 1 for s in result[6]],
    }


@pytest.mark.r_parity
class TestGaussRegBols:
    """GaussReg family parity against R mboost (should match Gaussian exactly)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.r = _run_r_gaussreg(mstop=100, nu=0.1, n=200, seed=42)
        self.data = {
            "y": self.r["y"],
            "x1": self.r["x1"],
            "x2": self.r["x2"],
        }
        self.mstop = 100
        self.nu = 0.1

    def test_risk_parity(self):
        model = mboost.glmboost(
            "y ~ x1 + x2",
            data=self.data,
            family=GaussReg(),
            control=mboost.boost_control(self.mstop, self.nu),
        )
        np.testing.assert_allclose(
            model.risk_,
            self.r["risk"],
            rtol=rtol,
            err_msg="GaussReg risk path does not match R",
        )

    def test_fitted_parity(self):
        model = mboost.glmboost(
            "y ~ x1 + x2",
            data=self.data,
            family=GaussReg(),
            control=mboost.boost_control(self.mstop, self.nu),
        )
        np.testing.assert_allclose(
            model.fitted_,
            self.r["fitted"],
            rtol=rtol,
            err_msg="GaussReg fitted values do not match R",
        )

    def test_offset_parity(self):
        model = mboost.glmboost(
            "y ~ x1 + x2",
            data=self.data,
            family=GaussReg(),
            control=mboost.boost_control(self.mstop, self.nu),
        )
        np.testing.assert_allclose(
            model.offset_,
            self.r["offset"],
            rtol=1e-6,
            err_msg="GaussReg offset does not match R",
        )

    def test_selected_parity(self):
        model = mboost.glmboost(
            "y ~ x1 + x2",
            data=self.data,
            family=GaussReg(),
            control=mboost.boost_control(self.mstop, self.nu),
        )
        assert list(model._selected) == self.r["selected"]

    def test_matches_gaussian(self):
        """GaussReg should produce identical results to Gaussian."""
        gr_model = mboost.glmboost(
            "y ~ x1 + x2",
            data=self.data,
            family=GaussReg(),
            control=mboost.boost_control(self.mstop, self.nu),
        )
        g_model = mboost.glmboost(
            "y ~ x1 + x2",
            data=self.data,
            family=mboost.Gaussian(),
            control=mboost.boost_control(self.mstop, self.nu),
        )
        np.testing.assert_allclose(
            gr_model.risk_,
            g_model.risk_,
            rtol=1e-12,
            err_msg="GaussReg and Gaussian risk must be identical",
        )
        np.testing.assert_allclose(
            gr_model.fitted_,
            g_model.fitted_,
            rtol=1e-12,
            err_msg="GaussReg and Gaussian fitted must be identical",
        )