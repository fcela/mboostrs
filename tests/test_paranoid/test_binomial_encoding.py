"""Verify binomial response encoding: 0/1 vs -1/+1."""

import mboostrs as mboost
import numpy as np
from numpy.testing import assert_allclose

rng = np.random.default_rng(20250609)
n = 50
x1 = rng.normal(size=n)
logit = 0.5 * x1
p = 1 / (1 + np.exp(-logit))


def test_binomial_01_vs_11():
    y_01 = rng.binomial(1, p).astype(float)
    y_11 = 2.0 * y_01 - 1.0

    model_01 = mboost.glmboost(
        "y ~ x1", data={"y": y_01, "x1": x1}, family=mboost.Binomial(),
        control=mboost.boost_control(50, 0.1),
    )
    model_11 = mboost.glmboost(
        "y ~ x1", data={"y": y_11, "x1": x1}, family=mboost.Binomial(),
        control=mboost.boost_control(50, 0.1),
    )
    assert_allclose(model_01.fitted(), model_11.fitted(), atol=1e-12)
    assert_allclose(model_01.risk_, model_11.risk_, atol=1e-12)
    y_01 = rng.binomial(1, p).astype(float)
    y_11 = 2.0 * y_01 - 1.0

    model_01 = mboost.glmboost(
        "y ~ x1", data={"y": y_01, "x1": x1}, family=mboost.AdaExp(),
        control=mboost.boost_control(50, 0.1),
    )
    model_11 = mboost.glmboost(
        "y ~ x1", data={"y": y_11, "x1": x1}, family=mboost.AdaExp(),
        control=mboost.boost_control(50, 0.1),
    )
    assert_allclose(model_01.fitted(), model_11.fitted(), atol=1e-12)
    assert_allclose(model_01.risk_, model_11.risk_, atol=1e-12)


def test_binomial_predict_01_vs_11():
    y_01 = rng.binomial(1, p).astype(float)
    y_11 = 2.0 * y_01 - 1.0

    model_01 = mboost.glmboost(
        "y ~ x1", data={"y": y_01, "x1": x1}, family=mboost.Binomial(),
        control=mboost.boost_control(50, 0.1),
    )
    model_11 = mboost.glmboost(
        "y ~ x1", data={"y": y_11, "x1": x1}, family=mboost.Binomial(),
        control=mboost.boost_control(50, 0.1),
    )
    pred_01 = model_01.predict(type="response")
    pred_11 = model_11.predict(type="response")
    assert_allclose(pred_01, pred_11, atol=1e-12)


def test_binomial_gamboost_01_vs_11():
    y_01 = rng.binomial(1, p).astype(float)
    y_11 = 2.0 * y_01 - 1.0

    model_01 = mboost.gamboost(
        "y ~ bbs(x1, df=4)", data={"y": y_01, "x1": x1},
        family=mboost.Binomial(),
        control=mboost.boost_control(50, 0.1),
    )
    model_11 = mboost.gamboost(
        "y ~ bbs(x1, df=4)", data={"y": y_11, "x1": x1},
        family=mboost.Binomial(),
        control=mboost.boost_control(50, 0.1),
    )
    assert_allclose(model_01.fitted(), model_11.fitted(), atol=1e-10)
