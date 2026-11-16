"""Verify model[:k] slicing correctness across all state attributes."""

import mboostrs as mboost
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

rng = np.random.default_rng(20250609)
n = 50
x1 = rng.normal(2.0, 3.0, n)
x2 = rng.normal(-1.0, 2.0, n)
y = 1.0 + 2.0 * x1 - 0.5 * x2 + rng.normal(0, 1, n)
data = {"y": y, "x1": x1, "x2": x2}

MSTOP = 100
K = 37


def test_glmboost_slice_risk():
    model = mboost.glmboost(
        "y ~ x1 + x2", data=data, family=mboost.Gaussian(),
        control=mboost.boost_control(MSTOP, 0.1),
    )
    sliced = model[:K]
    assert len(sliced.risk_) == K + 1
    assert_allclose(sliced.risk_, model.risk_[:K + 1], atol=1e-15)


def test_glmboost_slice_fitted():
    model = mboost.glmboost(
        "y ~ x1 + x2", data=data, family=mboost.Gaussian(),
        control=mboost.boost_control(MSTOP, 0.1),
    )
    sliced = model[:K]
    model_k = mboost.glmboost(
        "y ~ x1 + x2", data=data, family=mboost.Gaussian(),
        control=mboost.boost_control(K, 0.1),
    )
    assert_allclose(sliced.fitted(), model_k.fitted(), atol=1e-12)


def test_glmboost_slice_selected():
    model = mboost.glmboost(
        "y ~ x1 + x2", data=data, family=mboost.Gaussian(),
        control=mboost.boost_control(MSTOP, 0.1),
    )
    sliced = model[:K]
    assert len(sliced.selected) == K
    assert_array_equal(sliced.selected, model.selected[:K])


def test_glmboost_slice_coef():
    model = mboost.glmboost(
        "y ~ x1 + x2", data=data, family=mboost.Gaussian(),
        control=mboost.boost_control(MSTOP, 0.1),
    )
    sliced = model[:K]
    model_k = mboost.glmboost(
        "y ~ x1 + x2", data=data, family=mboost.Gaussian(),
        control=mboost.boost_control(K, 0.1),
    )
    sc = sliced.coef()
    kc = model_k.coef()
    for name in sc:
        assert_allclose(sc[name], kc[name], atol=1e-12)


def test_glmboost_slice_predict_after_slice():
    """Predict after slicing should match a fresh fit at that mstop."""
    model = mboost.glmboost(
        "y ~ x1 + x2", data=data, family=mboost.Gaussian(),
        control=mboost.boost_control(MSTOP, 0.1),
    )
    sliced = model[:K]
    model_k = mboost.glmboost(
        "y ~ x1 + x2", data=data, family=mboost.Gaussian(),
        control=mboost.boost_control(K, 0.1),
    )
    pred_sliced = sliced.predict(type="link")
    pred_k = model_k.predict(type="link")
    assert_allclose(pred_sliced, pred_k, atol=1e-12)


def test_gamboost_slice_risk():
    model = mboost.gamboost(
        "y ~ bbs(x1, df=4) + bbs(x2, df=4)", data=data,
        family=mboost.Gaussian(),
        control=mboost.boost_control(MSTOP, 0.1),
    )
    sliced = model[:K]
    assert len(sliced.risk_) == K + 1
    assert_allclose(sliced.risk_, model.risk_[:K + 1], atol=1e-15)


def test_gamboost_slice_selected():
    model = mboost.gamboost(
        "y ~ bbs(x1, df=4) + bbs(x2, df=4)", data=data,
        family=mboost.Gaussian(),
        control=mboost.boost_control(MSTOP, 0.1),
    )
    sliced = model[:K]
    assert len(sliced.selected) == K
    assert_array_equal(sliced.selected, model.selected[:K])


def test_gamboost_slice_predict():
    model = mboost.gamboost(
        "y ~ bbs(x1, df=4) + bbs(x2, df=4)", data=data,
        family=mboost.Gaussian(),
        control=mboost.boost_control(MSTOP, 0.1),
    )
    sliced = model[:K]
    model_k = mboost.gamboost(
        "y ~ bbs(x1, df=4) + bbs(x2, df=4)", data=data,
        family=mboost.Gaussian(),
        control=mboost.boost_control(K, 0.1),
    )
    assert_allclose(sliced.fitted(), model_k.fitted(), atol=1e-10)
