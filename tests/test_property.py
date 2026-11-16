"""Property tests: determinism, NaN/Inf guards, API contracts."""

import mboostrs as mboost
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from mboostrs.families import (
    Binomial,
    Gaussian,
)


class TestDeterminism:
    """Same input + seed -> identical output."""

    def test_glmboost_deterministic(self):
        rng = np.random.default_rng(123)
        n = 80
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        y = 1.5 * x1 - 0.8 * x2 + rng.normal(scale=0.3, size=n)
        data = {"y": y, "x1": x1, "x2": x2}

        ctrl = mboost.boost_control(mstop=50, nu=0.1)
        mod1 = mboost.glmboost("y ~ x1 + x2", data=data, control=ctrl, family=Gaussian())
        mod2 = mboost.glmboost("y ~ x1 + x2", data=data, control=ctrl, family=Gaussian())

        np.testing.assert_array_equal(mod1.fitted_, mod2.fitted_)
        np.testing.assert_array_equal(mod1.risk_, mod2.risk_)
        np.testing.assert_array_equal(mod1.selected, mod2.selected)
        np.testing.assert_array_equal(mod1.coef(), mod2.coef())

    def test_gamboost_deterministic(self):
        rng = np.random.default_rng(456)
        n = 80
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        y = 2 * x1 + np.sin(2 * np.pi * x2) + rng.normal(scale=0.3, size=n)
        data = {"y": y, "x1": x1, "x2": x2}

        ctrl = mboost.boost_control(mstop=30, nu=0.1)
        mod1 = mboost.gamboost(
            "y ~ bbs(x1, df=4) + bbs(x2, df=4)", data=data, control=ctrl, family=Gaussian()
        )
        mod2 = mboost.gamboost(
            "y ~ bbs(x1, df=4) + bbs(x2, df=4)", data=data, control=ctrl, family=Gaussian()
        )

        np.testing.assert_array_equal(mod1.fitted_, mod2.fitted_)
        np.testing.assert_array_equal(mod1.risk_, mod2.risk_)
        np.testing.assert_array_equal(mod1.selected, mod2.selected)


class TestNaNInfGuards:
    """NaN / Inf in y or weights should raise."""

    def test_glmboost_nan_y(self):
        rng = np.random.default_rng(789)
        n = 50
        x1 = rng.normal(size=n)
        y = np.ones(n)
        y[5] = np.nan
        data = {"y": y, "x1": x1}
        with pytest.raises(ValueError, match="NaN"):
            mboost.glmboost("y ~ x1", data=data, family=Gaussian())

    def test_glmboost_inf_y(self):
        rng = np.random.default_rng(789)
        n = 50
        x1 = rng.normal(size=n)
        y = np.ones(n)
        y[5] = np.inf
        data = {"y": y, "x1": x1}
        with pytest.raises(ValueError, match="Inf"):
            mboost.glmboost("y ~ x1", data=data, family=Gaussian())

    def test_glmboost_nan_weights(self):
        rng = np.random.default_rng(789)
        n = 50
        x1 = rng.normal(size=n)
        y = 1.5 * x1 + rng.normal(scale=0.3, size=n)
        w = np.ones(n)
        w[3] = np.nan
        data = {"y": y, "x1": x1}
        with pytest.raises(ValueError, match="NaN"):
            mboost.glmboost("y ~ x1", data=data, weights=w, family=Gaussian())

    def test_glmboost_negative_weights(self):
        rng = np.random.default_rng(789)
        n = 50
        x1 = rng.normal(size=n)
        y = 1.5 * x1 + rng.normal(scale=0.3, size=n)
        w = np.ones(n)
        w[3] = -1.0
        data = {"y": y, "x1": x1}
        with pytest.raises(ValueError, match="non-negative"):
            mboost.glmboost("y ~ x1", data=data, weights=w, family=Gaussian())

    def test_gamboost_nan_y(self):
        rng = np.random.default_rng(789)
        n = 50
        x1 = rng.uniform(size=n)
        y = np.ones(n)
        y[5] = np.nan
        data = {"y": y, "x1": x1}
        with pytest.raises(ValueError, match="NaN"):
            mboost.gamboost("y ~ bbs(x1, df=4)", data=data, family=Gaussian())

    def test_glmboost_nan_x(self):
        rng = np.random.default_rng(789)
        n = 50
        x1 = rng.normal(size=n)
        x1[7] = np.nan
        y = 1.5 * x1 + rng.normal(scale=0.3, size=n)
        data = {"y": y, "x1": x1}
        with pytest.raises(ValueError, match="NaN"):
            mboost.glmboost("y ~ x1", data=data, family=Gaussian())


class TestAPIContracts:
    """Basic API contract tests (shapes, types, invariants)."""

    def test_glmboost_fitted_shape(self):
        rng = np.random.default_rng(42)
        n = 80
        x1 = rng.normal(size=n)
        y = 1.5 * x1 + rng.normal(scale=0.3, size=n)
        data = {"y": y, "x1": x1}
        mod = mboost.glmboost("y ~ x1", data=data, family=Gaussian())
        assert mod.fitted_.shape == (n,)
        assert mod.risk_.shape == (mod.mstop + 1,)
        assert len(mod.selected) == mod.mstop
        assert mod.offset_ == float(np.mean(y))

    def test_gamboost_fitted_shape(self):
        rng = np.random.default_rng(42)
        n = 80
        x1 = rng.uniform(size=n)
        y = 2 * x1 + rng.normal(scale=0.3, size=n)
        data = {"y": y, "x1": x1}
        mod = mboost.gamboost(
            "y ~ bbs(x1, df=4)", data=data, family=Gaussian(),
            control=mboost.boost_control(mstop=30, nu=0.1),
        )
        assert mod.fitted_.shape == (n,)
        assert mod.risk_.shape == (mod.mstop + 1,)
        assert len(mod.selected) == mod.mstop

    def test_glmboost_risk_nonincreasing(self):
        rng = np.random.default_rng(42)
        n = 80
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        y = 1.5 * x1 - 0.8 * x2 + rng.normal(scale=0.3, size=n)
        data = {"y": y, "x1": x1, "x2": x2}
        mod = mboost.glmboost(
            "y ~ x1 + x2", data=data, family=Gaussian(),
            control=mboost.boost_control(mstop=100, nu=0.1),
        )
        for i in range(1, len(mod.risk_)):
            assert mod.risk_[i] <= mod.risk_[i - 1] + 1e-10

    @given(
        seed=st.integers(min_value=0, max_value=2**31 - 1),
        n=st.integers(min_value=20, max_value=100),
        p=st.integers(min_value=1, max_value=5),
        mstop=st.integers(min_value=5, max_value=100),
        nu=st.floats(min_value=0.05, max_value=0.3),
    )
    @settings(max_examples=30, deadline=None)
    def test_glmboost_risk_nonincreasing_property(self, seed, n, p, mstop, nu):
        """Genuinely property-based (hypothesis-generated) version of
        test_glmboost_risk_nonincreasing above: Theorem 1 (monotone risk
        decrease) should hold for every valid (n, p, mstop, nu), not just the
        one fixed example."""
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, p))
        beta = rng.normal(scale=1.0, size=p)
        y = X @ beta + rng.normal(scale=0.5, size=n)
        data = {f"x{j}": X[:, j] for j in range(p)}
        data["y"] = y
        formula = "y ~ " + " + ".join(f"x{j}" for j in range(p))
        mod = mboost.glmboost(formula, data=data, family=Gaussian(),
                               control=mboost.boost_control(mstop=mstop, nu=nu))
        diffs = np.diff(mod.risk_)
        assert np.all(diffs <= 1e-9), f"risk increased: {diffs[diffs > 1e-9]}"

    def test_glmboost_slicing_reduces_mstop(self):
        rng = np.random.default_rng(42)
        n = 80
        x1 = rng.normal(size=n)
        y = 1.5 * x1 + rng.normal(scale=0.3, size=n)
        data = {"y": y, "x1": x1}
        mod = mboost.glmboost("y ~ x1", data=data, family=Gaussian())
        sub = mod[:30]
        assert sub.mstop == 30
        assert len(sub.selected) == 30

    def test_gamboost_slicing_reduces_mstop(self):
        rng = np.random.default_rng(42)
        n = 80
        x1 = rng.uniform(size=n)
        y = 2 * x1 + rng.normal(scale=0.3, size=n)
        data = {"y": y, "x1": x1}
        mod = mboost.gamboost(
            "y ~ bbs(x1, df=4)", data=data, family=Gaussian(),
            control=mboost.boost_control(mstop=50, nu=0.1),
        )
        sub = mod[:20]
        assert sub.mstop == 20
        assert len(sub.selected) == 20

    def test_predict_link_response_gaussian(self):
        rng = np.random.default_rng(42)
        n = 80
        x1 = rng.normal(size=n)
        y = 1.5 * x1 + rng.normal(scale=0.3, size=n)
        data = {"y": y, "x1": x1}
        mod = mboost.glmboost("y ~ x1", data=data, family=Gaussian())
        pred_link = mod.predict(type="link")
        pred_resp = mod.predict(type="response")
        np.testing.assert_allclose(pred_link, pred_resp, atol=1e-15)

    def test_glmboost_predict_path_matches_per_k_predict(self):
        rng = np.random.default_rng(7)
        n, p = 100, 8
        X = rng.normal(size=(n, p))
        beta = np.zeros(p)
        beta[[1, 4]] = [1.5, -0.8]
        y = X @ beta + rng.normal(scale=0.4, size=n)
        data = {f"x{j}": X[:, j] for j in range(p)}
        data["y"] = y
        formula = "y ~ " + " + ".join(f"x{j}" for j in range(p))
        mod = mboost.glmboost(formula, data=data, family=Gaussian(),
                              control=mboost.boost_control(mstop=60, nu=0.1))

        grid = [1, 5, 20, 40, 60]
        batch = mod.predict_path(grid, type="link")
        assert batch.shape == (len(grid), n)
        for i, k in enumerate(grid):
            single = mod.predict(mstop=k, type="link")
            np.testing.assert_allclose(batch[i], single, atol=1e-10)

        # Also check newdata + response scale.
        newdata = {f"x{j}": X[:, j] + 0.05 for j in range(p)}
        batch_nd = mod.predict_path(grid, newdata=newdata, type="response")
        for i, k in enumerate(grid):
            single = mod.predict(newdata=newdata, mstop=k, type="response")
            np.testing.assert_allclose(batch_nd[i], single, atol=1e-10)

    def test_predict_link_response_binomial(self):
        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.normal(size=n)
        eta = 1.5 * x1
        prob = 1.0 / (1.0 + np.exp(-eta))
        y = (rng.random(size=n) < prob).astype(float)
        data = {"y": y, "x1": x1}
        mod = mboost.glmboost(
            "y ~ x1", data=data, family=Binomial(),
            control=mboost.boost_control(mstop=50, nu=0.1),
        )
        pred_link = mod.predict(type="link")
        pred_resp = mod.predict(type="response")
        # mboost Binomial: P(Y=1|eta) = sigmoid(2*eta) due to -1/+1 coding
        expected = 1.0 / (1.0 + np.exp(-2.0 * pred_link))
        np.testing.assert_allclose(pred_resp, expected, atol=1e-12)

    def test_glmboost_predict_newdata_shape(self):
        rng = np.random.default_rng(42)
        n = 80
        x1 = rng.normal(size=n)
        y = 1.5 * x1 + rng.normal(scale=0.3, size=n)
        data = {"y": y, "x1": x1}
        mod = mboost.glmboost("y ~ x1", data=data, family=Gaussian())
        newdata = {"x1": rng.normal(size=20)}
        pred = mod.predict(newdata=newdata)
        assert pred.shape == (20,)


class TestBoostControlValidation:
    """boost_control parameter validation."""

    def test_mstop_zero(self):
        with pytest.raises(ValueError):
            mboost.boost_control(mstop=0, nu=0.1)

    def test_nu_zero(self):
        with pytest.raises(ValueError):
            mboost.boost_control(mstop=100, nu=0.0)

    def test_nu_negative(self):
        with pytest.raises(ValueError):
            mboost.boost_control(mstop=100, nu=-0.1)

    def test_nu_above_one(self):
        with pytest.raises(ValueError):
            mboost.boost_control(mstop=100, nu=1.5)

    def test_nu_exactly_one(self):
        ctrl = mboost.boost_control(mstop=100, nu=1.0)
        assert ctrl[1] == 1.0

    def test_invalid_risk_string(self):
        with pytest.raises(ValueError):
            mboost.boost_control(mstop=100, nu=0.1, risk="invalid")
