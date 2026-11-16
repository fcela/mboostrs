"""R-parity tests for glmboost AUC + bols (Gate 1: R oracle).

See crates/mboostrs-core/src/families.rs
(auc_ngradient_matches_finite_difference_of_unstandardized_loss) for Gate 2,
the R-independent finite-difference gradient safety net -- this file only
checks that the end-to-end fit (which includes the per-iteration f/sd(f)
standardization) matches R's mboost::AUC() numerically.
"""

import numpy as np
import pytest
from mboostrs.families import AUC

import mboostrs as mboost

# The AUC pseudo-residual is a smoothed, standardized surrogate (not an exact
# closed-form update like AdaExp's exponential loss), so R parity is measured
# rather than assumed bit-exact; 1e-6 comfortably separates "matches R" from
# "coincidentally close".
rtol = 1e-6
atol = 1e-8


def _run_r_auc(mstop, nu, n=200, seed=42):
    """Generate data in R and fit glmboost with the AUC family."""
    import rpy2.robjects as ro

    r_code = """
    function(mstop, nu, n, seed) {
        set.seed(seed)
        x1 <- rnorm(n)
        x2 <- rnorm(n)
        eta <- 0.5*x1 - 0.3*x2
        p <- 1/(1 + exp(-eta))
        y_num <- ifelse(runif(n) < p, 1, 0)
        y <- as.factor(ifelse(y_num == 1, "yes", "no"))
        data <- data.frame(x1 = x1, x2 = x2, y = y)
        fit <- mboost::glmboost(
            y ~ x1 + x2,
            data = data,
            family = mboost::AUC(),
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
class TestAucBols:
    """AUC family parity against R mboost."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.r = _run_r_auc(mstop=50, nu=0.1, n=200, seed=42)
        self.data = {
            "y": self.r["y"],
            "x1": self.r["x1"],
            "x2": self.r["x2"],
        }

    def _fit(self):
        return mboost.glmboost(
            "y ~ x1 + x2",
            data=self.data,
            family=AUC(),
            control=mboost.boost_control(50, 0.1),
        )

    def test_risk_parity(self):
        model = self._fit()
        np.testing.assert_allclose(
            model.risk_,
            self.r["risk"],
            rtol=rtol,
            atol=atol,
            err_msg="AUC risk path does not match R",
        )

    def test_fitted_parity(self):
        model = self._fit()
        np.testing.assert_allclose(
            model.fitted_,
            self.r["fitted"],
            rtol=rtol,
            atol=atol,
            err_msg="AUC fitted values do not match R",
        )

    def test_offset_parity(self):
        model = self._fit()
        np.testing.assert_allclose(
            model.offset_,
            self.r["offset"],
            rtol=rtol,
            atol=atol,
            err_msg="AUC offset does not match R",
        )
        assert model.offset_ == 0.0

    def test_selected_parity(self):
        model = self._fit()
        assert list(model._selected) == self.r["selected"]


@pytest.mark.r_parity
class TestAucBolsShortMstop:
    """AUC at a different mstop/seed, to catch fixture-specific coincidences."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.r = _run_r_auc(mstop=20, nu=0.15, n=150, seed=123)
        self.data = {
            "y": self.r["y"],
            "x1": self.r["x1"],
            "x2": self.r["x2"],
        }

    def test_risk_parity_short(self):
        model = mboost.glmboost(
            "y ~ x1 + x2",
            data=self.data,
            family=AUC(),
            control=mboost.boost_control(20, 0.15),
        )
        np.testing.assert_allclose(
            model.risk_,
            self.r["risk"],
            rtol=rtol,
            atol=atol,
            err_msg="AUC risk (short mstop) does not match R",
        )

    def test_fitted_parity_short(self):
        model = mboost.glmboost(
            "y ~ x1 + x2",
            data=self.data,
            family=AUC(),
            control=mboost.boost_control(20, 0.15),
        )
        np.testing.assert_allclose(
            model.fitted_,
            self.r["fitted"],
            rtol=rtol,
            atol=atol,
            err_msg="AUC fitted (short mstop) does not match R",
        )


@pytest.mark.r_parity
def test_auc_accepts_non_unit_weights():
    """`Auc` now implements a principled *weighted* gradient (see the `Auc`
    doc comment in families.rs) instead of rejecting non-unit weights --
    this is a deliberate mboostrs extension, NOT R parity: R's own weighted
    AUC ngradient is broken (returns a compacted vector of length sum(w),
    not n), so there is no R numerical target to compare against here. The
    R-independent correctness oracle for the weighted gradient lives in
    Rust (families.rs:
    auc_ngradient_matches_finite_difference_of_weighted_unstandardized_loss,
    auc_zero_one_weights_match_unit_weight_subset). This test only checks
    that non-unit case weights, generated from the same R fixture used for
    unit-weight parity above, are accepted end-to-end and produce a finite
    fit -- not that the result matches R."""
    r = _run_r_auc(mstop=10, nu=0.1, n=60, seed=7)
    data = {"y": r["y"], "x1": r["x1"], "x2": r["x2"]}
    weights = np.ones(60)
    weights[::3] = 2.0
    model = mboost.glmboost(
        "y ~ x1 + x2",
        data=data,
        family=AUC(),
        weights=weights,
        control=mboost.boost_control(10, 0.1),
    )
    assert np.all(np.isfinite(model.fitted_))
    assert np.all(np.isfinite(model.risk_))
