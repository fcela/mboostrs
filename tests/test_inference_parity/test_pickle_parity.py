"""Tests for pickle serialization of fitted models."""

import io
import pickle

import mboostrs as mboost
import numpy as np


def _make_gaussian_data(n=100, seed=42):
    np.random.seed(seed)
    x1 = np.random.uniform(size=n)
    x2 = np.random.uniform(size=n)
    y = 2*x1 - 1*x2 + np.random.normal(size=n, scale=0.5)
    return {"y": y, "x1": x1, "x2": x2}


class TestGLMBoostPickle:
    def test_pickle_roundtrip(self):
        data = _make_gaussian_data()
        ctrl = mboost.boost_control(mstop=50, nu=0.1)
        mod = mboost.glmboost("y ~ x1 + x2", data=data, control=ctrl, family=mboost.Gaussian())

        buf = io.BytesIO()
        pickle.dump(mod, buf)
        buf.seek(0)
        mod2 = pickle.load(buf)

        np.testing.assert_array_equal(mod2.fitted_, mod.fitted_)
        np.testing.assert_array_equal(mod2.risk_, mod.risk_)
        assert mod2.mstop == mod.mstop
        assert mod2.selected == mod.selected
        assert mod2.offset_ == mod.offset_
        assert mod2._formula == mod._formula

    def test_pickle_predict_after_load(self):
        data = _make_gaussian_data()
        ctrl = mboost.boost_control(mstop=50, nu=0.1)
        mod = mboost.glmboost("y ~ x1 + x2", data=data, control=ctrl, family=mboost.Gaussian())

        buf = io.BytesIO()
        pickle.dump(mod, buf)
        buf.seek(0)
        mod2 = pickle.load(buf)

        # After load, predict on training data should work (data is dropped, but predict without newdata uses stored fitted)
        pred = mod2.predict()
        np.testing.assert_array_equal(pred, mod.fitted_)

    def test_pickle_coef_after_load(self):
        data = _make_gaussian_data()
        ctrl = mboost.boost_control(mstop=50, nu=0.1)
        mod = mboost.glmboost("y ~ x1 + x2", data=data, control=ctrl, family=mboost.Gaussian())

        buf = io.BytesIO()
        pickle.dump(mod, buf)
        buf.seek(0)
        mod2 = pickle.load(buf)

        coefs1 = mod.coef()
        coefs2 = mod2.coef()
        assert set(coefs1.keys()) == set(coefs2.keys())
        for k in coefs1:
            np.testing.assert_allclose(coefs2[k], coefs1[k], rtol=1e-10)

    def test_pickle_varimp_after_load(self):
        data = _make_gaussian_data()
        ctrl = mboost.boost_control(mstop=50, nu=0.1)
        mod = mboost.glmboost("y ~ x1 + x2", data=data, control=ctrl, family=mboost.Gaussian())

        buf = io.BytesIO()
        pickle.dump(mod, buf)
        buf.seek(0)
        mod2 = pickle.load(buf)

        vi1 = mod.varimp()
        vi2 = mod2.varimp()
        for name in vi1["reduction"]:
            np.testing.assert_allclose(vi2["reduction"][name], vi1["reduction"][name], rtol=1e-10)

    def test_pickle_binomial_family(self):
        np.random.seed(42)
        n = 100
        x1 = np.random.uniform(size=n)
        y = np.random.binomial(1, 0.5, size=n).astype(float)
        data = {"y": y, "x1": x1}

        ctrl = mboost.boost_control(mstop=30, nu=0.1)
        mod = mboost.glmboost("y ~ x1", data=data, control=ctrl, family=mboost.Binomial())

        buf = io.BytesIO()
        pickle.dump(mod, buf)
        buf.seek(0)
        mod2 = pickle.load(buf)

        assert isinstance(mod2._family, mboost.Binomial)
        np.testing.assert_array_equal(mod2.fitted_, mod.fitted_)


class TestGAMBoostPickle:
    def test_pickle_roundtrip(self):
        np.random.seed(42)
        n = 100
        x1 = np.random.uniform(size=n)
        y = 3*np.sin(2*np.pi*x1) + np.random.normal(size=n, scale=0.5)
        data = {"y": y, "x1": x1}

        ctrl = mboost.boost_control(mstop=50, nu=0.1)
        mod = mboost.gamboost("y ~ bbs(x1, df=4)", data=data, control=ctrl, family=mboost.Gaussian())

        buf = io.BytesIO()
        pickle.dump(mod, buf)
        buf.seek(0)
        mod2 = pickle.load(buf)

        np.testing.assert_array_equal(mod2.fitted_, mod.fitted_)
        np.testing.assert_array_equal(mod2.risk_, mod.risk_)
        assert mod2.mstop == mod.mstop
        assert mod2.selected == mod.selected

    def test_pickle_predict_after_load(self):
        np.random.seed(42)
        n = 100
        x1 = np.random.uniform(size=n)
        y = 3*np.sin(2*np.pi*x1) + np.random.normal(size=n, scale=0.5)
        data = {"y": y, "x1": x1}

        ctrl = mboost.boost_control(mstop=50, nu=0.1)
        mod = mboost.gamboost("y ~ bbs(x1, df=4)", data=data, control=ctrl, family=mboost.Gaussian())

        buf = io.BytesIO()
        pickle.dump(mod, buf)
        buf.seek(0)
        mod2 = pickle.load(buf)

        # predict() without newdata should still work (uses stored fitted)
        pred = mod2.predict()
        np.testing.assert_array_equal(pred, mod.fitted_)

    def test_pickle_slicing_after_load(self):
        np.random.seed(42)
        n = 100
        x1 = np.random.uniform(size=n)
        y = 3*np.sin(2*np.pi*x1) + np.random.normal(size=n, scale=0.5)
        data = {"y": y, "x1": x1}

        ctrl = mboost.boost_control(mstop=50, nu=0.1)
        mod = mboost.gamboost("y ~ bbs(x1, df=4)", data=data, control=ctrl, family=mboost.Gaussian())

        buf = io.BytesIO()
        pickle.dump(mod, buf)
        buf.seek(0)
        mod2 = pickle.load(buf)

        # Slicing should still work
        mod_sliced = mod2[:25]
        assert mod_sliced.mstop == 25

    def test_pickle_coef_after_load(self):
        np.random.seed(42)
        n = 100
        x1 = np.random.uniform(size=n)
        y = 3*np.sin(2*np.pi*x1) + np.random.normal(size=n, scale=0.5)
        data = {"y": y, "x1": x1}

        ctrl = mboost.boost_control(mstop=50, nu=0.1)
        mod = mboost.gamboost("y ~ bbs(x1, df=4)", data=data, control=ctrl, family=mboost.Gaussian())

        buf = io.BytesIO()
        pickle.dump(mod, buf)
        buf.seek(0)
        mod2 = pickle.load(buf)

        coefs1 = mod.coef()
        coefs2 = mod2.coef()
        assert set(coefs1.keys()) == set(coefs2.keys())
