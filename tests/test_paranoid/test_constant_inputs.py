"""Verify crash-free behavior for degenerate inputs."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import mboostrs as mboost


def test_constant_y_glmboost():
    n = 30
    y = np.full(n, 5.0)
    x = np.random.default_rng(42).normal(size=n)
    data = {"y": y, "x": x}
    model = mboost.glmboost(
        "y ~ x", data=data, family=mboost.Gaussian(),
        control=mboost.boost_control(50, 0.1),
    )
    f = model.fitted()
    assert np.all(np.isfinite(f))
    assert_allclose(f, f[0], atol=1e-10)


def test_constant_y_with_weights():
    n = 30
    rng = np.random.default_rng(43)
    y = np.full(n, 3.0)
    x = rng.normal(size=n)
    w = rng.uniform(0.5, 3.0, n)
    data = {"y": y, "x": x}
    model = mboost.glmboost(
        "y ~ x", data=data, family=mboost.Gaussian(),
        weights=w,
        control=mboost.boost_control(50, 0.1),
    )
    assert np.all(np.isfinite(model.fitted()))


def test_single_observation():
    data = {"y": np.array([1.0]), "x": np.array([0.5])}
    model = mboost.glmboost(
        "y ~ x", data=data, family=mboost.Gaussian(),
        control=mboost.boost_control(10, 0.1),
    )
    assert np.all(np.isfinite(model.fitted()))


def test_two_observations():
    data = {"y": np.array([1.0, 2.0]), "x": np.array([0.5, 1.5])}
    model = mboost.glmboost(
        "y ~ x", data=data, family=mboost.Gaussian(),
        control=mboost.boost_control(10, 0.1),
    )
    assert np.all(np.isfinite(model.fitted()))


def test_constant_predictor():
    rng = np.random.default_rng(45)
    n = 30
    y = rng.normal(size=n)
    x = np.zeros(n)
    data = {"y": y, "x": x}
    model = mboost.glmboost(
        "y ~ x", data=data, family=mboost.Gaussian(),
        control=mboost.boost_control(50, 0.1),
    )
    assert np.all(np.isfinite(model.fitted()))


def test_tiny_mstop():
    rng = np.random.default_rng(46)
    n = 20
    data = {"y": rng.normal(size=n), "x": rng.normal(size=n)}
    model = mboost.glmboost(
        "y ~ x", data=data, family=mboost.Gaussian(),
        control=mboost.boost_control(1, 0.1),
    )
    assert len(model.fitted()) == n
    assert len(model.risk_) == 2  # initial + 1 iteration
    assert len(model.selected) == 1


def test_mstop_zero():
    rng = np.random.default_rng(47)
    data = {"y": rng.normal(size=10), "x": rng.normal(size=10)}
    with pytest.raises(ValueError, match="mstop must be > 0"):
        mboost.glmboost(
            "y ~ x", data=data, family=mboost.Gaussian(),
            control=mboost.boost_control(0, 0.1),
        )


def test_zero_weights_all():
    """An all-zero weight vector is rejected rather than fitted.

    The object it used to produce reported a completed path over an empty
    sample: every learner's normal equations are empty, the risk is
    identically zero at every iteration, and the fitted values are all zero.
    Those are finite numbers that mean nothing, so the failure is moved to
    the call.
    """
    n = 20
    rng = np.random.default_rng(49)
    y = rng.normal(10.0, 2.0, n)
    x = rng.normal(size=n)
    data = {"y": y, "x": x}
    with pytest.raises(ValueError, match="weights are all zero"):
        mboost.glmboost(
            "y ~ x", data=data, family=mboost.Gaussian(),
            weights=np.zeros(n),
            control=mboost.boost_control(50, 0.1),
        )
    with pytest.raises(ValueError, match="weights are all zero"):
        mboost.gamboost(
            "y ~ bbs(x, df=4)", data=data, family=mboost.Gaussian(),
            weights=np.zeros(n),
            control=mboost.boost_control(50, 0.1),
        )


def test_constant_y_gamboost():
    n = 20
    y = np.full(n, 5.0)
    rng = np.random.default_rng(50)
    x = rng.normal(size=n)
    data = {"y": y, "x": x}
    model = mboost.gamboost(
        "y ~ bbs(x, df=4)", data=data, family=mboost.Gaussian(),
        control=mboost.boost_control(20, 0.1),
    )
    assert np.all(np.isfinite(model.fitted()))
