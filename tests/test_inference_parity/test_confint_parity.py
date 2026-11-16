"""Tests for confint() — bootstrap confidence intervals."""

import mboostrs as mboost
import numpy as np


def _make_gaussian_data(n=100, seed=42):
    np.random.seed(seed)
    x1 = np.random.uniform(size=n)
    x2 = np.random.uniform(size=n)
    y = 2*x1 - 1*x2 + np.random.normal(size=n, scale=0.5)
    return {"y": y, "x1": x1, "x2": x2}


class TestGLMBoostConfint:
    def test_confint_basic(self):
        data = _make_gaussian_data()
        ctrl = mboost.boost_control(mstop=50, nu=0.1)
        mod = mboost.glmboost("y ~ x1 + x2", data=data, control=ctrl, family=mboost.Gaussian())
        ci = mod.confint(B=10, B_mstop=0, seed=123)
        assert "lower" in ci
        assert "upper" in ci
        assert "level" in ci
        assert ci["level"] == 0.95
        assert isinstance(ci["lower"], dict)
        assert isinstance(ci["upper"], dict)

    def test_confint_coverage(self):
        data = _make_gaussian_data()
        ctrl = mboost.boost_control(mstop=50, nu=0.1)
        mod = mboost.glmboost("y ~ x1 + x2", data=data, control=ctrl, family=mboost.Gaussian())
        ci = mod.confint(B=50, B_mstop=0, seed=42)
        # True coefficient of x1 is 2.0; CI should contain it
        if "x1" in ci["lower"]:
            assert ci["lower"]["x1"] < 2.0 < ci["upper"]["x1"], \
                f"CI for x1: [{ci['lower']['x1']:.3f}, {ci['upper']['x1']:.3f}] doesn't contain 2.0"

    def test_confint_custom_level(self):
        data = _make_gaussian_data()
        ctrl = mboost.boost_control(mstop=50, nu=0.1)
        mod = mboost.glmboost("y ~ x1 + x2", data=data, control=ctrl, family=mboost.Gaussian())
        ci_90 = mod.confint(level=0.9, B=10, B_mstop=0, seed=42)
        assert ci_90["level"] == 0.9

    def test_confint_which(self):
        data = _make_gaussian_data()
        ctrl = mboost.boost_control(mstop=50, nu=0.1)
        mod = mboost.glmboost("y ~ x1 + x2", data=data, control=ctrl, family=mboost.Gaussian())
        ci = mod.confint(parm="x1", B=10, B_mstop=0, seed=42)
        assert "x1" in ci["lower"]
        assert "x2" not in ci["lower"]

    def test_confint_boot_coefs_shape(self):
        data = _make_gaussian_data()
        ctrl = mboost.boost_control(mstop=50, nu=0.1)
        mod = mboost.glmboost("y ~ x1 + x2", data=data, control=ctrl, family=mboost.Gaussian())
        ci = mod.confint(B=20, B_mstop=0, seed=42)
        assert ci["boot_coefs"].shape[0] == 20


class TestGAMBoostConfint:
    def test_confint_basic(self):
        np.random.seed(42)
        n = 100
        x1 = np.random.uniform(size=n)
        y = 3*np.sin(2*np.pi*x1) + np.random.normal(size=n, scale=0.5)
        data = {"y": y, "x1": x1}

        ctrl = mboost.boost_control(mstop=30, nu=0.1)
        mod = mboost.gamboost("y ~ bbs(x1, df=4)", data=data, control=ctrl, family=mboost.Gaussian())
        ci = mod.confint(B=5, B_mstop=0, seed=42)
        assert "lower" in ci
        assert "upper" in ci
        assert ci["level"] == 0.95
