"""R-parity tests for glmboost NBinomial + bols."""

import mboostrs as mboost
import numpy as np
import pytest
from mboostrs.families import NBinomial

rtol = 1e-6


def _run_r_nbinomial(mstop, nu, n=200, seed=42):
    """Generate data in R and fit glmboost with NBinomial family."""
    import rpy2.robjects as ro

    r_code = """
    function(mstop, nu, n, seed) {
        set.seed(seed)
        x1 <- rnorm(n)
        x2 <- rnorm(n)
        mu <- exp(0.5 + 0.3 * x1 - 0.2 * x2)
        y <- rnbinom(n, size = 5, mu = mu)
        fit <- mboost::glmboost(
            y ~ x1 + x2,
            data = data.frame(x1 = x1, x2 = x2, y = y),
            family = mboost::NBinomial(),
            center = TRUE,
            control = mboost::boost_control(mstop = mstop, nu = nu)
        )
        cc <- stats::coef(fit, off2int = FALSE)
        off <- attr(cc, "offset")
        list(
            x1 = as.numeric(x1),
            x2 = as.numeric(x2),
            y = as.numeric(y),
            fitted = as.numeric(stats::predict(fit, type = "link")),
            risk = as.numeric(mboost::risk(fit)),
            offset = as.numeric(off),
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
class TestNBinomialBols:
    """NBinomial family parity against R mboost."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.r = _run_r_nbinomial(mstop=100, nu=0.1, n=200, seed=42)
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
            family=NBinomial(),
            control=mboost.boost_control(self.mstop, self.nu),
        )
        # ABSOLUTE risk parity. The NBinomial plloss now includes the log(y!) =
        # lgamma(y+1) normalizing constant R uses, so absolute risk matches R
        # (previously the two differed by a y-only constant ~171.94 and only the
        # increments could be compared). The residual is the per-step Brent
        # sigma re-calibration vs R's optimize (~9e-8 relative); rtol=1e-5 +
        # atol=1e-3 keeps a wide platform margin while still catching the old
        # constant offset by ~5 orders of magnitude.
        np.testing.assert_allclose(
            model.risk_,
            self.r["risk"],
            rtol=1e-5,
            atol=1e-3,
            err_msg="NBinomial risk path does not match R",
        )

    def test_fitted_parity(self):
        model = mboost.glmboost(
            "y ~ x1 + x2",
            data=self.data,
            family=NBinomial(),
            control=mboost.boost_control(self.mstop, self.nu),
        )
        np.testing.assert_allclose(
            model.fitted_,
            self.r["fitted"],
            rtol=rtol,
            err_msg="NBinomial fitted values do not match R",
        )

    def test_offset_parity(self):
        model = mboost.glmboost(
            "y ~ x1 + x2",
            data=self.data,
            family=NBinomial(),
            control=mboost.boost_control(self.mstop, self.nu),
        )
        np.testing.assert_allclose(
            model.offset_,
            self.r["offset"],
            rtol=1e-4,
            err_msg="NBinomial offset does not match R",
        )

    def test_selected_parity(self):
        model = mboost.glmboost(
            "y ~ x1 + x2",
            data=self.data,
            family=NBinomial(),
            control=mboost.boost_control(self.mstop, self.nu),
        )
        assert list(model._selected) == self.r["selected"]


@pytest.mark.r_parity
class TestNBinomialBolsShortMstop:
    """NBinomial at low mstop where sigma calibration might differ."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.r = _run_r_nbinomial(mstop=20, nu=0.1, n=100, seed=123)
        self.data = {
            "y": self.r["y"],
            "x1": self.r["x1"],
            "x2": self.r["x2"],
        }

    def test_risk_parity_short(self):
        model = mboost.glmboost(
            "y ~ x1 + x2",
            data=self.data,
            family=NBinomial(),
            control=mboost.boost_control(20, 0.1),
        )
        # ABSOLUTE risk parity (the lgamma(y+1) constant is now included, see
        # test_risk_parity). At low mstop the per-step Brent sigma recalibration
        # vs R's optimize is a bit more sensitive (~9e-8 relative), so rtol=5e-5
        # + atol=1e-3 keeps a safe platform margin.
        np.testing.assert_allclose(
            model.risk_,
            self.r["risk"],
            rtol=5e-5,
            atol=1e-3,
            err_msg="NBinomial risk path (short mstop) does not match R",
        )

    def test_fitted_parity_short(self):
        model = mboost.glmboost(
            "y ~ x1 + x2",
            data=self.data,
            family=NBinomial(),
            control=mboost.boost_control(20, 0.1),
        )
        np.testing.assert_allclose(
            model.fitted_,
            self.r["fitted"],
            rtol=1e-5,
            err_msg="NBinomial fitted (short mstop) does not match R (tolerant for sigma calibration diffs)",
        )