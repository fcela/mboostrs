"""R-parity tests for glmboost PropOdds + bols."""

import mboostrs as mboost
import numpy as np
import pytest
from mboostrs.families import PropOdds

rtol = 1e-5


def _run_r_propodds(mstop, nu, n_levels=3, n=200, seed=42):
    """Generate data in R and fit glmboost with PropOdds family."""
    import rpy2.robjects as ro

    r_code = """
    function(mstop, nu, n_levels, n, seed) {
        set.seed(seed)
        x1 <- rnorm(n)
        x2 <- rnorm(n)
        eta <- 0.5 * x1 - 0.3 * x2
        K <- n_levels
        cutpoints <- seq(0, by = 1.5, length.out = K - 1)
        probs <- matrix(0, nrow = n, ncol = K)
        for (i in 1:n) {
            cum <- numeric(K)
            cum[1] <- 1/(1 + exp(-(cutpoints[1] + eta[i])))
            for (k in 2:(K-1)) {
                cum[k] <- 1/(1 + exp(-(cutpoints[k] + eta[i])))
            }
            cum[K] <- 1
            p <- diff(c(0, cum))
            probs[i, ] <- p
        }
        y_int <- apply(probs, 1, function(p) sample(1:K, 1, prob=p))
        y <- as.ordered(y_int)
        fit <- mboost::glmboost(
            y ~ x1 + x2,
            data = data.frame(x1 = x1, x2 = x2, y = y),
            family = mboost::PropOdds(),
            center = TRUE,
            control = mboost::boost_control(mstop = mstop, nu = nu)
        )
        cc <- stats::coef(fit)
        resp <- stats::predict(fit, type = "response")  # n x K
        list(
            x1 = as.numeric(x1),
            x2 = as.numeric(x2),
            y_int = as.integer(y_int),
            fitted = as.numeric(stats::predict(fit, type = "link")),
            risk = as.numeric(mboost::risk(fit)),
            offset = as.numeric(fit$offset),
            selected = as.integer(fit$xselect()),
            response = as.numeric(resp),
            resp_ncol = as.integer(ncol(resp)),
            coef_x1 = as.numeric(cc["x1"]),
            coef_x2 = as.numeric(cc["x2"])
        )
    }
    """
    r_func = ro.r(r_code)
    result = r_func(mstop, nu, n_levels, n, seed)

    n_obs = len(np.asarray(result[0]))
    k_cat = int(result[8][0])
    # R fills the n x K response matrix column-major.
    response = np.asarray(result[7], dtype=np.float64).reshape((n_obs, k_cat), order="F")

    return {
        "x1": np.asarray(result[0], dtype=np.float64),
        "x2": np.asarray(result[1], dtype=np.float64),
        "y": np.asarray(result[2], dtype=np.float64),
        "fitted": np.asarray(result[3], dtype=np.float64),
        "risk": np.asarray(result[4], dtype=np.float64),
        "offset": float(result[5][0]) if len(result[5]) > 0 else 0.0,
        "selected": [int(s) - 1 for s in result[6]],
        "response": response,
        "coef_x1": float(result[9][0]),
        "coef_x2": float(result[10][0]),
    }


@pytest.mark.r_parity
class TestPropOddsBols:
    """PropOdds family parity against R mboost."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.r = _run_r_propodds(mstop=100, nu=0.1, n_levels=3, n=200, seed=42)
        self.data = {
            "y": self.r["y"],
            "x1": self.r["x1"],
            "x2": self.r["x2"],
        }
        self.mstop = 100
        self.nu = 0.1
        self.n_levels = 3

    def test_risk_parity(self):
        model = mboost.glmboost(
            "y ~ x1 + x2",
            data=self.data,
            family=PropOdds(n_levels=self.n_levels),
            control=mboost.boost_control(self.mstop, self.nu),
        )
        np.testing.assert_allclose(
            model.risk_,
            self.r["risk"],
            rtol=1e-5,
            atol=1e-4,
            err_msg="PropOdds risk path does not match R",
        )

    def test_response_parity(self):
        # The real test: per-category probabilities P(Y=k) must match R's
        # predict(type="response") VALUE-FOR-VALUE. These are shift-invariant, so
        # they match R despite the link's identifiability constant (below).
        model = mboost.glmboost(
            "y ~ x1 + x2",
            data=self.data,
            family=PropOdds(n_levels=self.n_levels),
            control=mboost.boost_control(self.mstop, self.nu),
        )
        resp = model.predict(type="response")
        assert resp.shape == self.r["response"].shape, (
            f"response shape {resp.shape} != R {self.r['response'].shape}"
        )
        np.testing.assert_allclose(
            resp.sum(axis=1), 1.0, atol=1e-10,
            err_msg="PropOdds per-category probabilities must sum to 1",
        )
        np.testing.assert_allclose(
            resp, self.r["response"], rtol=1e-4, atol=1e-5,
            err_msg="PropOdds per-category response does not match R",
        )

    def test_coef_parity(self):
        model = mboost.glmboost(
            "y ~ x1 + x2",
            data=self.data,
            family=PropOdds(n_levels=self.n_levels),
            control=mboost.boost_control(self.mstop, self.nu),
        )
        cc = model.coef()
        np.testing.assert_allclose(cc.get("x1"), self.r["coef_x1"], rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(cc.get("x2"), self.r["coef_x2"], rtol=1e-4, atol=1e-5)

    def test_link_is_constant_shift_from_r(self):
        # PropOdds identifiability: the linear predictor is defined only up to a
        # location shift (R parameterizes thresholds via delta/d2s + BFGS; we use
        # direct coordinate-descent). So the absolute link differs from R by a
        # CONSTANT — verify exactly that (constant, not just correlated), which
        # together with response parity proves the fitted model is equivalent.
        model = mboost.glmboost(
            "y ~ x1 + x2",
            data=self.data,
            family=PropOdds(n_levels=self.n_levels),
            control=mboost.boost_control(self.mstop, self.nu),
        )
        diff = model.fitted_ - self.r["fitted"]
        assert np.std(diff) < 1e-4, (
            f"PropOdds link should differ from R by a constant; std={np.std(diff)}"
        )

    def test_selected_parity(self):
        model = mboost.glmboost(
            "y ~ x1 + x2",
            data=self.data,
            family=PropOdds(n_levels=self.n_levels),
            control=mboost.boost_control(self.mstop, self.nu),
        )
        agreement = np.mean(
            np.array(model._selected) == np.array(self.r["selected"])
        )
        assert agreement > 0.8, (
            f"PropOdds variable selection agreement with R too low: {agreement}"
        )


@pytest.mark.r_parity
class TestPropOddsBolsShortMstop:
    """PropOdds at low mstop."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.r = _run_r_propodds(mstop=20, nu=0.1, n_levels=3, n=100, seed=123)
        self.data = {
            "y": self.r["y"],
            "x1": self.r["x1"],
            "x2": self.r["x2"],
        }

    def test_risk_decreases(self):
        model = mboost.glmboost(
            "y ~ x1 + x2",
            data=self.data,
            family=PropOdds(n_levels=3),
            control=mboost.boost_control(20, 0.1),
        )
        diffs = np.diff(model.risk_)
        assert np.all(diffs <= 1e-10), "PropOdds risk should be non-increasing"

    def test_response_parity_short(self):
        model = mboost.glmboost(
            "y ~ x1 + x2",
            data=self.data,
            family=PropOdds(n_levels=3),
            control=mboost.boost_control(20, 0.1),
        )
        np.testing.assert_allclose(
            model.predict(type="response"), self.r["response"],
            rtol=1e-4, atol=1e-5,
            err_msg="PropOdds response (short mstop) does not match R",
        )


@pytest.mark.r_parity
class TestPropOdds4Classes:
    """PropOdds with 4 ordered classes."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.r = _run_r_propodds(mstop=50, nu=0.1, n_levels=4, n=200, seed=77)
        self.data = {
            "y": self.r["y"],
            "x1": self.r["x1"],
            "x2": self.r["x2"],
        }

    def test_risk_decreases_4class(self):
        model = mboost.glmboost(
            "y ~ x1 + x2",
            data=self.data,
            family=PropOdds(n_levels=4),
            control=mboost.boost_control(50, 0.1),
        )
        diffs = np.diff(model.risk_)
        assert np.all(diffs <= 1e-10), "PropOdds risk (4 class) should be non-increasing"

    def test_response_parity_4class(self):
        model = mboost.glmboost(
            "y ~ x1 + x2",
            data=self.data,
            family=PropOdds(n_levels=4),
            control=mboost.boost_control(50, 0.1),
        )
        resp = model.predict(type="response")
        assert resp.shape == self.r["response"].shape
        np.testing.assert_allclose(resp.sum(axis=1), 1.0, atol=1e-10)
        np.testing.assert_allclose(
            resp, self.r["response"], rtol=1e-4, atol=1e-5,
            err_msg="PropOdds response (4 class) does not match R",
        )