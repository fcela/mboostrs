"""R-parity tests for glmboost Gehan + bols (Gate 1: R oracle).

See crates/mboostrs-core/src/families.rs
(gehan_ngradient_matches_finite_difference_of_risk,
gehan_ngradient_sign_matches_aft_ranking_direction) for Gate 2, the
R-independent finite-difference gradient safety net.

The Gehan offset is unidentifiable: the loss depends only on pairwise
residual differences, so a constant shift of every fitted value cancels and
the risk is flat in it (verified directly against R -- identical risk at
several offset values). mboostrs therefore defines offset() == 0.0 rather
than reproducing R's arbitrary optimize()-on-a-flat-function value. All
comparisons below are consequently offset-invariant:
  - risk path: matches R exactly (unaffected by the offset choice).
  - selected: matches R exactly.
  - fitted values: compared *centered* (`fitted - mean(fitted)`), never raw
    -- raw fitted differs from R by the (arbitrary) offset constant, which
    is expected and not a bug.
  - negative_gradient: compared element-wise against R's
    `Gehan()@ngradient(y, f, w)` directly, at a fixed random `f` -- the
    strongest, most direct gradient oracle, and offset-invariant by
    construction (it doesn't go through any fit at all).
"""

import numpy as np
import pytest
from mboostrs._rust import family_diagnostics
from mboostrs.families import Gehan

import mboostrs as mboost

RTOL_RISK = 1e-8
ATOL_RISK = 1e-8
RTOL_CENTERED_FITTED = 1e-6
ATOL_CENTERED_FITTED = 1e-8
RTOL_NGRAD = 1e-8
ATOL_NGRAD = 1e-8


def _run_r_gehan(mstop, nu, n=200, seed=42):
    """Generate survival data in R and fit glmboost with the Gehan family."""
    import rpy2.robjects as ro

    r_code = """
    function(mstop, nu, n, seed) {
        suppressMessages(library(survival))
        set.seed(seed)
        x1 <- rnorm(n)
        x2 <- rnorm(n)
        eta <- 0.3 * x1 + 0.1 * x2
        time <- rexp(n, rate = exp(eta))
        event <- rbinom(n, 1, 0.7)
        df <- data.frame(time = time, event = event, x1 = x1, x2 = x2)
        fit <- mboost::glmboost(
            Surv(time, event) ~ x1 + x2,
            data = df,
            family = mboost::Gehan(),
            center = TRUE,
            control = mboost::boost_control(mstop = mstop, nu = nu)
        )
        list(
            x1 = as.numeric(x1),
            x2 = as.numeric(x2),
            time = as.numeric(time),
            event = as.numeric(event),
            fitted = as.numeric(stats::predict(fit, type = "link")),
            risk = as.numeric(mboost::risk(fit)),
            selected = as.integer(fit$xselect())
        )
    }
    """
    r_func = ro.r(r_code)
    result = r_func(mstop, nu, n, seed)

    return {
        "x1": np.asarray(result[0], dtype=np.float64),
        "x2": np.asarray(result[1], dtype=np.float64),
        "time": np.asarray(result[2], dtype=np.float64),
        "event": np.asarray(result[3], dtype=np.float64),
        "fitted": np.asarray(result[4], dtype=np.float64),
        "risk": np.asarray(result[5], dtype=np.float64),
        "selected": [int(s) - 1 for s in result[6]],
    }


def _run_r_gehan_ngradient(n=80, data_seed=11, f_seed=99):
    """Pull R's Gehan()@ngradient(y, f, w) at a fixed random f, independent
    of any fit -- the strongest, most direct gradient oracle."""
    import rpy2.robjects as ro

    r_code = """
    function(n, data_seed, f_seed) {
        suppressMessages(library(survival))
        set.seed(data_seed)
        time <- rexp(n, rate = 1)
        event <- rbinom(n, 1, 0.7)
        y <- Surv(time, event)
        set.seed(f_seed)
        f <- rnorm(n, sd = 1.5)
        w <- rep(1, n)
        fam <- mboost::Gehan()
        ng <- fam@ngradient(y, f, w)
        risk <- fam@risk(y, f, w)
        list(
            time = as.numeric(time),
            event = as.numeric(event),
            f = as.numeric(f),
            ng = as.numeric(ng),
            risk = as.numeric(risk)
        )
    }
    """
    r_func = ro.r(r_code)
    result = r_func(n, data_seed, f_seed)
    return {
        "time": np.asarray(result[0], dtype=np.float64),
        "event": np.asarray(result[1], dtype=np.float64),
        "f": np.asarray(result[2], dtype=np.float64),
        "ng": np.asarray(result[3], dtype=np.float64),
        "risk": float(result[4][0]),
    }


@pytest.mark.r_parity
class TestGehanBols:
    """Gehan family fit parity against R mboost (offset-invariant)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.r = _run_r_gehan(mstop=50, nu=0.1, n=200, seed=42)
        self.data = {
            "time": self.r["time"],
            "event": self.r["event"],
            "x1": self.r["x1"],
            "x2": self.r["x2"],
        }

    def _fit(self):
        return mboost.glmboost(
            "Surv(time, event) ~ x1 + x2",
            data=self.data,
            family=Gehan(),
            control=mboost.boost_control(50, 0.1),
        )

    def test_risk_parity(self):
        model = self._fit()
        np.testing.assert_allclose(
            model.risk_,
            self.r["risk"],
            rtol=RTOL_RISK,
            atol=ATOL_RISK,
            err_msg="Gehan risk path does not match R",
        )

    def test_selected_parity(self):
        model = self._fit()
        assert list(model._selected) == self.r["selected"]

    def test_centered_fitted_parity(self):
        # NEVER compare raw fitted values -- they differ from R by the
        # (arbitrary) offset constant. Only the centered path is meaningful.
        model = self._fit()
        ours_centered = model.fitted_ - model.fitted_.mean()
        r_centered = self.r["fitted"] - self.r["fitted"].mean()
        np.testing.assert_allclose(
            ours_centered,
            r_centered,
            rtol=RTOL_CENTERED_FITTED,
            atol=ATOL_CENTERED_FITTED,
            err_msg="Gehan centered fitted values do not match R",
        )

    def test_offset_is_zero_not_r_arbitrary_value(self):
        # mboostrs deliberately does not reproduce R's arbitrary offset (see
        # module docstring); it defines offset() == 0.0 instead.
        model = self._fit()
        assert model.offset_ == 0.0


@pytest.mark.r_parity
class TestGehanBolsShortMstop:
    """Gehan at a different mstop/seed, to catch fixture-specific coincidences."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.r = _run_r_gehan(mstop=20, nu=0.15, n=150, seed=123)
        self.data = {
            "time": self.r["time"],
            "event": self.r["event"],
            "x1": self.r["x1"],
            "x2": self.r["x2"],
        }

    def test_risk_parity_short(self):
        model = mboost.glmboost(
            "Surv(time, event) ~ x1 + x2",
            data=self.data,
            family=Gehan(),
            control=mboost.boost_control(20, 0.15),
        )
        np.testing.assert_allclose(
            model.risk_,
            self.r["risk"],
            rtol=RTOL_RISK,
            atol=ATOL_RISK,
            err_msg="Gehan risk (short mstop) does not match R",
        )

    def test_centered_fitted_parity_short(self):
        model = mboost.glmboost(
            "Surv(time, event) ~ x1 + x2",
            data=self.data,
            family=Gehan(),
            control=mboost.boost_control(20, 0.15),
        )
        ours_centered = model.fitted_ - model.fitted_.mean()
        r_centered = self.r["fitted"] - self.r["fitted"].mean()
        np.testing.assert_allclose(
            ours_centered,
            r_centered,
            rtol=RTOL_CENTERED_FITTED,
            atol=ATOL_CENTERED_FITTED,
            err_msg="Gehan centered fitted (short mstop) does not match R",
        )


@pytest.mark.r_parity
def test_gehan_ngradient_matches_r_element_wise():
    """Direct gradient oracle: mboostrs negative_gradient vs R's
    Gehan()@ngradient(y, f, w) at a fixed random f, independent of any fit.
    """
    r = _run_r_gehan_ngradient(n=80, data_seed=11, f_seed=99)
    y = np.concatenate([r["time"], r["event"]])
    w = np.ones_like(r["time"])

    _, grad, risk = family_diagnostics("gehan", {}, y, r["f"], w)
    grad = np.asarray(grad)

    np.testing.assert_allclose(
        grad,
        r["ng"],
        rtol=RTOL_NGRAD,
        atol=ATOL_NGRAD,
        err_msg="Gehan negative_gradient does not match R's Gehan()@ngradient element-wise",
    )
    np.testing.assert_allclose(
        risk,
        r["risk"],
        rtol=RTOL_RISK,
        atol=ATOL_RISK,
        err_msg="Gehan risk does not match R's Gehan()@risk",
    )


@pytest.mark.r_parity
def test_gehan_accepts_non_unit_weights():
    """`Gehan` now implements a principled *weighted* gradient (see the
    `Gehan` doc comment in families.rs) instead of rejecting non-unit
    weights -- a deliberate mboostrs extension, NOT R parity: R's own
    weighted Gehan ngradient is broken the same way AUC's is (a compacted
    vector of length sum(w), not n), so there is no R numerical target to
    compare against here. The R-independent correctness oracle lives in
    Rust (families.rs:
    gehan_ngradient_matches_finite_difference_of_weighted_risk,
    gehan_zero_one_weights_match_unit_weight_subset). This test only checks
    that non-unit case weights, generated from the same R fixture used for
    unit-weight parity above, are accepted end-to-end and produce a finite
    fit -- not that the result matches R."""
    r = _run_r_gehan(mstop=10, nu=0.1, n=60, seed=7)
    data = {"time": r["time"], "event": r["event"], "x1": r["x1"], "x2": r["x2"]}
    weights = np.ones(60)
    weights[::3] = 2.0
    model = mboost.glmboost(
        "Surv(time, event) ~ x1 + x2",
        data=data,
        family=Gehan(),
        weights=weights,
        control=mboost.boost_control(10, 0.1),
    )
    assert np.all(np.isfinite(model.fitted_))
    assert np.all(np.isfinite(model.risk_))
