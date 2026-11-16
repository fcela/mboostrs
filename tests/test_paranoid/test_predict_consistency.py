"""Verify predict(newdata=training_data) matches fitted() for all families."""

import mboostrs as mboost
import numpy as np
import pytest
from numpy.testing import assert_allclose

rng = np.random.default_rng(20250609)
n = 100
x1 = rng.normal(2.0, 3.0, n)
x2 = rng.normal(-1.0, 2.0, n)
w = rng.uniform(0.5, 3.0, n)

BASE_DATA = {"x1": x1, "x2": x2}


def _check(model, data, atol=1e-12):
    fitted = np.asarray(model.fitted()).ravel()
    pred = np.asarray(model.predict(newdata=data, type="link")).ravel()
    assert_allclose(pred, fitted, atol=atol)


@pytest.mark.parametrize("family,maker,y", [
    ("Gaussian", mboost.Gaussian, lambda: rng.normal(0, 1, n)),
    ("GaussReg", mboost.GaussReg, lambda: rng.normal(0, 1, n)),
    ("Poisson", mboost.Poisson, lambda: rng.poisson(5, n).astype(float)),
    ("GammaReg", mboost.GammaReg, lambda: rng.gamma(2, 2, n)),
    ("Huber", mboost.Huber, lambda: rng.normal(0, 1, n)),
    ("Laplace", mboost.Laplace, lambda: rng.normal(0, 1, n)),
    ("Expectile", lambda: mboost.Expectile(tau=0.5), lambda: rng.normal(0, 1, n)),
    ("Quantile", lambda: mboost.Quantile(tau=0.5), lambda: rng.normal(0, 1, n)),
])
def test_glmboost_predict_consistency(family, maker, y):
    y_arr = y()
    data = dict(BASE_DATA, y=np.asarray(y_arr, dtype=np.float64))
    model = mboost.glmboost(
        "y ~ x1 + x2", data=data, family=maker(),
        control=mboost.boost_control(50, 0.1),
    )
    _check(model, data)


@pytest.mark.parametrize("family,maker,y", [
    ("Gaussian", mboost.Gaussian, lambda: rng.normal(0, 1, n)),
    ("GaussReg", mboost.GaussReg, lambda: rng.normal(0, 1, n)),
    ("Poisson", mboost.Poisson, lambda: rng.poisson(5, n).astype(float)),
    ("GammaReg", mboost.GammaReg, lambda: rng.gamma(2, 2, n)),
    ("Huber", mboost.Huber, lambda: rng.normal(0, 1, n)),
    ("Laplace", mboost.Laplace, lambda: rng.normal(0, 1, n)),
    ("Expectile", lambda: mboost.Expectile(tau=0.5), lambda: rng.normal(0, 1, n)),
    ("Quantile", lambda: mboost.Quantile(tau=0.5), lambda: rng.normal(0, 1, n)),
])
def test_gamboost_predict_consistency(family, maker, y):
    y_arr = y()
    data = dict(BASE_DATA, y=np.asarray(y_arr, dtype=np.float64))
    model = mboost.gamboost(
        "y ~ bbs(x1, df=4) + bbs(x2, df=4)", data=data, family=maker(),
        control=mboost.boost_control(50, 0.1),
    )
    _check(model, data, atol=1e-10)


def test_glmboost_predict_weighted():
    y = rng.normal(0, 1, n)
    data = dict(BASE_DATA, y=y)
    model = mboost.glmboost(
        "y ~ x1 + x2", data=data, family=mboost.Gaussian(), weights=w,
        control=mboost.boost_control(50, 0.1),
    )
    _check(model, data)


def test_gamboost_predict_weighted():
    y = rng.normal(0, 1, n)
    data = dict(BASE_DATA, y=y)
    model = mboost.gamboost(
        "y ~ bbs(x1, df=4) + bbs(x2, df=4)", data=data,
        family=mboost.Gaussian(), weights=w,
        control=mboost.boost_control(50, 0.1),
    )
    _check(model, data, atol=1e-10)


def test_glmboost_predict_binomial():
    rngb = np.random.default_rng(20250610)
    n = 50
    x1 = rngb.normal(size=n)
    logit = 0.5 * x1
    p = 1 / (1 + np.exp(-logit))
    y = rngb.binomial(1, p).astype(float)
    data = {"y": y, "x1": x1}
    model = mboost.glmboost(
        "y ~ x1", data=data, family=mboost.Binomial(),
        control=mboost.boost_control(50, 0.1),
    )
    _check(model, data)


def test_gamboost_predict_binomial():
    rngb = np.random.default_rng(20250610)
    n = 50
    x1 = rngb.normal(size=n)
    logit = 0.5 * x1
    p = 1 / (1 + np.exp(-logit))
    y = rngb.binomial(1, p).astype(float)
    data = {"y": y, "x1": x1}
    model = mboost.gamboost(
        "y ~ bbs(x1, df=4)", data=data, family=mboost.Binomial(),
        control=mboost.boost_control(50, 0.1),
    )
    _check(model, data)


def test_glmboost_predict_binomial_weighted():
    rngb = np.random.default_rng(20250610)
    n = 50
    x1 = rngb.normal(size=n)
    logit = 0.5 * x1
    p = 1 / (1 + np.exp(-logit))
    y = rngb.binomial(1, p).astype(float)
    w = rngb.uniform(0.5, 3.0, n)
    data = {"y": y, "x1": x1}
    model = mboost.glmboost(
        "y ~ x1", data=data, family=mboost.Binomial(), weights=w,
        control=mboost.boost_control(50, 0.1),
    )
    _check(model, data)


def test_glmboost_predict_adaexp():
    rngb = np.random.default_rng(20250610)
    n = 50
    x1 = rngb.normal(size=n)
    logit = 0.5 * x1
    p = 1 / (1 + np.exp(-logit))
    y = 2 * rngb.binomial(1, p).astype(float) - 1
    data = {"y": y, "x1": x1}
    model = mboost.glmboost(
        "y ~ x1", data=data, family=mboost.AdaExp(),
        control=mboost.boost_control(50, 0.1),
    )
    _check(model, data)


def test_gamboost_predict_adaexp():
    rngb = np.random.default_rng(20250610)
    n = 50
    x1 = rngb.normal(size=n)
    logit = 0.5 * x1
    p = 1 / (1 + np.exp(-logit))
    y = 2 * rngb.binomial(1, p).astype(float) - 1
    data = {"y": y, "x1": x1}
    model = mboost.gamboost(
        "y ~ bbs(x1, df=4)", data=data, family=mboost.AdaExp(),
        control=mboost.boost_control(50, 0.1),
    )
    _check(model, data)


def test_glmboost_predict_response_gaussian():
    rngr = np.random.default_rng(20250611)
    n = 30
    x = rngr.normal(size=n)
    y = 1.0 + 0.5 * x + rngr.normal(scale=0.2, size=n)
    data = {"y": y, "x": x}
    model = mboost.glmboost(
        "y ~ x", data=data, family=mboost.Gaussian(),
        control=mboost.boost_control(20, 0.1),
    )
    pred_link = model.predict(type="link")
    pred_resp = model.predict(type="response")
    assert_allclose(pred_resp, pred_link, atol=1e-15)


def test_nbinomial_predict_consistency():
    rngn = np.random.default_rng(20250612)
    n = 50
    x1 = rngn.normal(size=n)
    mu = np.exp(0.5 + 0.3 * x1)
    y = rngn.negative_binomial(5, 5 / (5 + mu)).astype(float)
    data = {"y": y, "x1": x1}
    model = mboost.glmboost(
        "y ~ x1", data=data, family=mboost.NBinomial(),
        control=mboost.boost_control(50, 0.1),
    )
    _check(model, data)
