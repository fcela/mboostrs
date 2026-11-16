"""R-parity tests for glmboost AdaExp + bols."""

import mboostrs as mboost
import numpy as np
import pytest
from mboostrs.families import AdaExp

# AdaExp has no nuisance parameter (pure exponential loss), so the fit is
# bit-exact to R: measured risk rel ~3e-16, fitted ~2e-14, offset exact. The
# tolerance sits just above machine epsilon for platform robustness, not to
# absorb any real divergence. (Was 1e-6, which was needlessly loose.)
rtol = 1e-10


def _run_r_adaexp(mstop, nu, n=100, seed=42):
    """Generate data in R and fit glmboost with AdaExp family."""
    import rpy2.robjects as ro

    r_code = """
    function(mstop, nu, n, seed) {
        set.seed(seed)
        x1 <- rnorm(n)
        x2 <- rnorm(n)
        eta <- 0.5*x1 - 0.3*x2
        p <- 1/(1 + exp(-eta))
        y_num <- ifelse(runif(n) < p, 1, -1)
        y <- as.factor(ifelse(y_num == 1, "yes", "no"))
        data <- data.frame(x1 = x1, x2 = x2, y = y)
        fit <- mboost::glmboost(
            y ~ x1 + x2,
            data = data,
            family = mboost::AdaExp(),
            center = TRUE,
            control = mboost::boost_control(mstop = mstop, nu = nu)
        )
        list(
            x1 = as.numeric(x1),
            x2 = as.numeric(x2),
            y_num = as.numeric(y_num),
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
class TestAdaExpBols:
    """AdaExp family parity against R mboost."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.r = _run_r_adaexp(mstop=50, nu=0.1, n=200, seed=42)
        self.data = {
            "y": self.r["y"],
            "x1": self.r["x1"],
            "x2": self.r["x2"],
        }

    def test_risk_parity(self):
        model = mboost.glmboost(
            "y ~ x1 + x2",
            data=self.data,
            family=AdaExp(),
            control=mboost.boost_control(50, 0.1),
        )
        np.testing.assert_allclose(
            model.risk_,
            self.r["risk"],
            rtol=rtol,
            err_msg="AdaExp risk path does not match R",
        )

    def test_fitted_parity(self):
        model = mboost.glmboost(
            "y ~ x1 + x2",
            data=self.data,
            family=AdaExp(),
            control=mboost.boost_control(50, 0.1),
        )
        np.testing.assert_allclose(
            model.fitted_,
            self.r["fitted"],
            rtol=rtol,
            err_msg="AdaExp fitted values do not match R",
        )

    def test_offset_parity(self):
        model = mboost.glmboost(
            "y ~ x1 + x2",
            data=self.data,
            family=AdaExp(),
            control=mboost.boost_control(50, 0.1),
        )
        np.testing.assert_allclose(
            model.offset_,
            self.r["offset"],
            rtol=1e-10,  # offset matches R exactly (measured rel 0.0)
            err_msg="AdaExp offset does not match R",
        )

    def test_selected_parity(self):
        model = mboost.glmboost(
            "y ~ x1 + x2",
            data=self.data,
            family=AdaExp(),
            control=mboost.boost_control(50, 0.1),
        )
        assert list(model._selected) == self.r["selected"]


@pytest.mark.r_parity
class TestAdaExpBolsShortMstop:
    """AdaExp at low mstop."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.r = _run_r_adaexp(mstop=20, nu=0.1, n=100, seed=123)
        self.data = {
            "y": self.r["y"],
            "x1": self.r["x1"],
            "x2": self.r["x2"],
        }

    def test_risk_parity_short(self):
        model = mboost.glmboost(
            "y ~ x1 + x2",
            data=self.data,
            family=AdaExp(),
            control=mboost.boost_control(20, 0.1),
        )
        np.testing.assert_allclose(
            model.risk_,
            self.r["risk"],
            rtol=rtol,
            err_msg="AdaExp risk (short mstop) does not match R",
        )

    def test_fitted_parity_short(self):
        model = mboost.glmboost(
            "y ~ x1 + x2",
            data=self.data,
            family=AdaExp(),
            control=mboost.boost_control(20, 0.1),
        )
        np.testing.assert_allclose(
            model.fitted_,
            self.r["fitted"],
            rtol=rtol,
            err_msg="AdaExp fitted (short mstop) does not match R",
        )