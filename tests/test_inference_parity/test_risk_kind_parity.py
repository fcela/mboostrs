"""Parity tests for risk='oobag' and risk='none' against R mboost."""

import mboostrs as mboost
import numpy as np
import pytest

pytestmark = pytest.mark.r_parity

ro = pytest.importorskip("rpy2.robjects")
from rpy2.robjects.packages import importr

r_mboost = importr("mboost")
r_base = importr("base")


def _r_vec(arr):
    return ro.FloatVector(arr.tolist())


def _r_int_vec(arr):
    return ro.IntVector(arr.tolist())


class TestGLMBoostRiskNone:
    """risk='none' should return same risk vector as risk='inbag' (R semantics)."""

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

    def test_none_equals_inbag_risk(self):
        ctrl_none = mboost.boost_control(mstop=50, nu=0.1, risk="none")
        mod_none = mboost.glmboost(
            "y ~ x1 + x2", data=self.data, control=ctrl_none, family=mboost.Gaussian()
        )
        ctrl_inbag = mboost.boost_control(mstop=50, nu=0.1, risk="inbag")
        mod_inbag = mboost.glmboost(
            "y ~ x1 + x2", data=self.data, control=ctrl_inbag, family=mboost.Gaussian()
        )
        np.testing.assert_allclose(mod_none.risk_, mod_inbag.risk_, atol=1e-10)

    def test_none_risk_matches_r(self):
        ro.r(
            'mod_none <- glmboost(y ~ x1 + x2, control=boost_control(mstop=50, nu=0.1, risk="none"), family=Gaussian())'
        )
        r_risk = np.array(ro.r("risk(mod_none)"))
        ctrl = mboost.boost_control(mstop=50, nu=0.1, risk="none")
        mod = mboost.glmboost(
            "y ~ x1 + x2", data=self.data, control=ctrl, family=mboost.Gaussian()
        )
        np.testing.assert_allclose(mod.risk_, r_risk, rtol=1e-6)

    def test_none_risk_positive_and_decreasing(self):
        ctrl = mboost.boost_control(mstop=50, nu=0.1, risk="none")
        mod = mboost.glmboost(
            "y ~ x1 + x2", data=self.data, control=ctrl, family=mboost.Gaussian()
        )
        assert mod.risk_[0] > 0
        assert mod.risk_[-1] < mod.risk_[0]


class TestGLMBoostRiskOobag:
    """risk='oobag' uses observations with weight=0 for risk."""

    @pytest.fixture(autouse=True)
    def setup(self):
        ro.r("set.seed(42)")
        np.random.seed(42)
        self.n = 100
        x1 = np.random.uniform(size=self.n)
        x2 = np.random.uniform(size=self.n)
        self.y = 2 * x1 - x2 + np.random.normal(size=self.n, scale=0.5)
        self.data = {"y": self.y, "x1": x1, "x2": x2}
        # Same subsampling indices for both Python and R
        np.random.seed(99)
        self.oob_idx = np.sort(np.random.choice(self.n, 30, replace=False))
        ro.globalenv["x1"] = _r_vec(x1)
        ro.globalenv["x2"] = _r_vec(x2)
        ro.globalenv["y"] = _r_vec(self.y)

    def test_oobag_uniform_weights_zero_risk(self):
        ctrl = mboost.boost_control(mstop=50, nu=0.1, risk="oobag")
        mod = mboost.glmboost(
            "y ~ x1 + x2", data=self.data, control=ctrl, family=mboost.Gaussian()
        )
        np.testing.assert_allclose(mod.risk_, 0.0, atol=1e-15)

    def test_oobag_with_subsampling_parity(self):
        w = np.ones(self.n)
        w[self.oob_idx] = 0.0
        ro.globalenv["w"] = _r_vec(w)

        ctrl = mboost.boost_control(mstop=50, nu=0.1, risk="oobag")
        mod = mboost.glmboost(
            "y ~ x1 + x2",
            data=self.data,
            control=ctrl,
            weights=w,
            family=mboost.Gaussian(),
        )
        ro.r(
            'mod_r <- glmboost(y ~ x1 + x2, control=boost_control(mstop=50, nu=0.1, risk="oobag"), weights=w, family=Gaussian())'
        )
        r_risk = np.array(ro.r("risk(mod_r)"))
        np.testing.assert_allclose(mod.risk_, r_risk, rtol=1e-6)

    def test_oobag_with_subsampling_positive_decreasing(self):
        w = np.ones(self.n)
        w[self.oob_idx] = 0.0
        ctrl = mboost.boost_control(mstop=50, nu=0.1, risk="oobag")
        mod = mboost.glmboost(
            "y ~ x1 + x2",
            data=self.data,
            control=ctrl,
            weights=w,
            family=mboost.Gaussian(),
        )
        assert mod.risk_[0] > 0
        assert mod.risk_[-1] < mod.risk_[0]


class TestGAMBoostRiskNone:
    """risk='none' for gamboost should also return inbag risk values."""

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

    def test_none_equals_inbag_risk(self):
        ctrl_none = mboost.boost_control(mstop=20, nu=0.1, risk="none")
        mod_none = mboost.gamboost(
            "y ~ bbs(x1, df=4) + bbs(x2, df=4)",
            data=self.data,
            control=ctrl_none,
            family=mboost.Gaussian(),
        )
        ctrl_inbag = mboost.boost_control(mstop=20, nu=0.1, risk="inbag")
        mod_inbag = mboost.gamboost(
            "y ~ bbs(x1, df=4) + bbs(x2, df=4)",
            data=self.data,
            control=ctrl_inbag,
            family=mboost.Gaussian(),
        )
        np.testing.assert_allclose(mod_none.risk_, mod_inbag.risk_, atol=1e-10)

    def test_none_risk_matches_r(self):
        ro.r(
            'mod_none <- gamboost(y ~ bbs(x1, df=4) + bbs(x2, df=4), control=boost_control(mstop=20, nu=0.1, risk="none"), family=Gaussian())'
        )
        r_risk = np.array(ro.r("risk(mod_none)"))
        ctrl = mboost.boost_control(mstop=20, nu=0.1, risk="none")
        mod = mboost.gamboost(
            "y ~ bbs(x1, df=4) + bbs(x2, df=4)",
            data=self.data,
            control=ctrl,
            family=mboost.Gaussian(),
        )
        np.testing.assert_allclose(mod.risk_, r_risk, rtol=1e-5)
