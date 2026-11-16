"""Parity tests for GaussReg and AdaExp families against R mboost."""

import mboostrs as mboost
import numpy as np
import pytest

pytestmark = pytest.mark.r_parity

ro = pytest.importorskip("rpy2.robjects")
from rpy2.robjects.packages import importr

r_mboost = importr("mboost")
r_base = importr("base")
r_stats = importr("stats")


def _r_vec(arr):
    return ro.FloatVector(arr.tolist())


class TestGaussRegParity:
    """GaussReg should produce identical results to Gaussian (and R's GaussReg)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        ro.r("set.seed(42)")
        np.random.seed(42)
        self.n = 100
        x1 = np.random.uniform(size=self.n)
        x2 = np.random.uniform(size=self.n)
        self.y = 2 * x1 - x2 + np.random.normal(size=self.n, scale=0.5)
        self.data = {"y": self.y, "x1": x1, "x2": x2}
        ro.globalenv["x1"] = _r_vec(x1)
        ro.globalenv["x2"] = _r_vec(x2)
        ro.globalenv["y"] = _r_vec(self.y)

    def test_gaussreg_equals_gaussian(self):
        ctrl = mboost.boost_control(mstop=50, nu=0.1)
        mod_gauss = mboost.glmboost(
            "y ~ x1 + x2", data=self.data, control=ctrl, family=mboost.Gaussian()
        )
        mod_gaussreg = mboost.glmboost(
            "y ~ x1 + x2", data=self.data, control=ctrl, family=mboost.GaussReg()
        )
        np.testing.assert_allclose(mod_gauss.fitted_, mod_gaussreg.fitted_, atol=1e-15)
        np.testing.assert_allclose(mod_gauss.risk_, mod_gaussreg.risk_, atol=1e-15)

    def test_gaussreg_matches_r(self):
        ro.r(
            'mod_r <- glmboost(y ~ x1 + x2, family=GaussReg(), control=boost_control(mstop=50, nu=0.1))'
        )
        r_risk = np.array(ro.r("risk(mod_r)"))
        r_fitted = np.array(ro.r("fitted(mod_r)"))
        r_offset = float(ro.r("mod_r$offset")[0])

        ctrl = mboost.boost_control(mstop=50, nu=0.1)
        mod = mboost.glmboost(
            "y ~ x1 + x2", data=self.data, control=ctrl, family=mboost.GaussReg()
        )
        np.testing.assert_allclose(mod.risk_, r_risk, rtol=1e-8)
        np.testing.assert_allclose(mod.fitted_, r_fitted, rtol=1e-8)
        np.testing.assert_allclose(mod.offset_, r_offset, rtol=1e-8)


class TestAdaExpParity:
    """AdaExp family parity tests."""

    @pytest.fixture(autouse=True)
    def setup(self):
        ro.r("set.seed(42)")
        np.random.seed(42)
        self.n = 200
        # R generates factor from runif, we need to match exactly
        ro.r(
            """
set.seed(42)
n <- 200
x1 <- rnorm(n)
x2 <- rnorm(n)
eta <- 0.5 * x1 + 0.3 * x2
y <- factor(ifelse(runif(n) < 1/(1+exp(-eta)), 1, -1))
df <- data.frame(x1=x1, x2=x2, y=y)
            """
        )
        # Get the -1/+1 coding R uses internally
        self.y_coded = np.array(ro.r("as.numeric(y) - 1")) * 2 - 1
        # Wait - R factor converts to 1/2 then check_y maps to -1/+1
        # Let me get the actual internal coding
        r_y_numeric = np.array(ro.r("as.numeric(as.character(y))"), dtype=np.float64)
        self.y_coded = r_y_numeric
        self.x1 = np.array(ro.r("x1"), dtype=np.float64)
        self.x2 = np.array(ro.r("x2"), dtype=np.float64)
        self.data = {"y": self.y_coded, "x1": self.x1, "x2": self.x2}

    def test_adaexp_risk_parity(self):
        ro.r(
            'fit <- glmboost(y ~ x1 + x2, data=df, family=AdaExp(), control=boost_control(mstop=50, nu=0.1))'
        )
        r_risk = np.array(ro.r("risk(fit)"))
        r_offset = float(ro.r("fit$offset")[0])

        ctrl = mboost.boost_control(mstop=50, nu=0.1)
        mod = mboost.glmboost(
            "y ~ x1 + x2", data=self.data, control=ctrl, family=mboost.AdaExp()
        )
        np.testing.assert_allclose(mod.offset_, r_offset, rtol=1e-6)
        np.testing.assert_allclose(mod.risk_, r_risk, rtol=1e-6)

    def test_adaexp_fitted_parity(self):
        ro.r(
            'fit <- glmboost(y ~ x1 + x2, data=df, family=AdaExp(), control=boost_control(mstop=50, nu=0.1))'
        )
        r_fitted = np.array(ro.r("fitted(fit)"), dtype=np.float64)

        ctrl = mboost.boost_control(mstop=50, nu=0.1)
        mod = mboost.glmboost(
            "y ~ x1 + x2", data=self.data, control=ctrl, family=mboost.AdaExp()
        )
        np.testing.assert_allclose(mod.fitted_, r_fitted, rtol=1e-6)

    def test_adaexp_selected_parity(self):
        ro.r(
            'fit <- glmboost(y ~ x1 + x2, data=df, family=AdaExp(), control=boost_control(mstop=50, nu=0.1))'
        )
        # Returning fit$xselect() straight to rpy2 corrupts its converter
        # ("bad INTSXP vector" -> R aborts the process); materialise it as a
        # plain integer vector inside R first.
        r_selected = list(ro.r("as.integer(unlist(fit$xselect()))"))
        # R is 1-based with the intercept at index 1; our `_selected` is 0-based
        # with the intercept at 0, so R index - 1 lines up with `_selected`.
        r_selected_0based = [s - 1 for s in r_selected]

        ctrl = mboost.boost_control(mstop=50, nu=0.1)
        mod = mboost.glmboost(
            "y ~ x1 + x2", data=self.data, control=ctrl, family=mboost.AdaExp()
        )
        assert list(mod._selected) == r_selected_0based
