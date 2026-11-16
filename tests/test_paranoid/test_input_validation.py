"""Input validation: the exception type and message, not merely that one fires.

Every check here used to be `pytest.raises(Exception)`, which passes on any
failure at all -- including a `TypeError` from a refactor that broke the call,
or an `AttributeError` from a missing method. A test that cannot tell a
rejection from a crash is not testing the rejection.

The types and message fragments below were measured against the code rather
than guessed, so a change that keeps the rejection but loses the diagnostic
fails here.
"""

import mboostrs as mboost
import numpy as np
import pytest

rng = np.random.default_rng(20250609)
n = 30
x1 = rng.normal(size=n)
y = rng.normal(size=n)


def test_nan_in_response():
    data = {"y": np.array([1.0, np.nan, 3.0]), "x1": np.array([0.5, 1.0, 1.5])}
    with pytest.raises(ValueError, match=r"Column 'y' contains 1 NaN"):
        mboost.glmboost(
            "y ~ x1", data=data, family=mboost.Gaussian(),
            control=mboost.boost_control(10, 0.1),
        )


def test_inf_in_response():
    data = {"y": np.array([1.0, np.inf, 3.0]), "x1": np.array([0.5, 1.0, 1.5])}
    with pytest.raises(ValueError, match=r"Column 'y' contains 1 Inf"):
        mboost.glmboost(
            "y ~ x1", data=data, family=mboost.Gaussian(),
            control=mboost.boost_control(10, 0.1),
        )


def test_nan_in_predictor():
    data = {"y": np.array([1.0, 2.0, 3.0]), "x1": np.array([0.5, np.nan, 1.5])}
    with pytest.raises(ValueError, match=r"Column 'x1' contains 1 NaN"):
        mboost.glmboost(
            "y ~ x1", data=data, family=mboost.Gaussian(),
            control=mboost.boost_control(10, 0.1),
        )


def test_mismatched_lengths():
    data = {"y": np.array([1.0, 2.0, 3.0]), "x1": np.array([0.5, 1.0])}
    with pytest.raises(ValueError, match="rows but y has"):
        mboost.glmboost(
            "y ~ x1", data=data, family=mboost.Gaussian(),
            control=mboost.boost_control(10, 0.1),
        )


def test_negative_weights():
    w = np.array([1.0, -0.5, 3.0])
    data = {"y": np.array([1.0, 2.0, 3.0]), "x1": np.array([0.5, 1.0, 1.5])}
    with pytest.raises(ValueError, match="weights must be non-negative"):
        mboost.glmboost(
            "y ~ x1", data=data, family=mboost.Gaussian(), weights=w,
            control=mboost.boost_control(10, 0.1),
        )


def test_wrong_weight_length():
    w = np.ones(5)
    data = {"y": np.array([1.0, 2.0, 3.0]), "x1": np.array([0.5, 1.0, 1.5])}
    with pytest.raises(ValueError, match="weights length 5 != n=3"):
        mboost.glmboost(
            "y ~ x1", data=data, family=mboost.Gaussian(), weights=w,
            control=mboost.boost_control(10, 0.1),
        )


def test_negative_y_poisson():
    """Poisson requires non-negative y."""
    data = {"y": np.array([-1.0, 2.0, 3.0]), "x1": np.array([0.5, 1.0, 1.5])}
    with pytest.raises(ValueError, match="Poisson.*y must be >= 0"):
        mboost.glmboost(
            "y ~ x1", data=data, family=mboost.Poisson(),
            control=mboost.boost_control(10, 0.1),
        )


def test_zero_y_gamma():
    """Gamma requires positive y."""
    data = {"y": np.array([0.0, 2.0, 3.0]), "x1": np.array([0.5, 1.0, 1.5])}
    with pytest.raises(ValueError, match="GammaReg.*y must be > 0"):
        mboost.glmboost(
            "y ~ x1", data=data, family=mboost.GammaReg(),
            control=mboost.boost_control(10, 0.1),
        )


def test_empty_data():
    with pytest.raises(KeyError, match="is required by the model"):
        mboost.glmboost(
            "y ~ x1", data={}, family=mboost.Gaussian(),
            control=mboost.boost_control(10, 0.1),
        )


def test_nonexistent_column():
    data = {"y": y, "x1": x1}
    with pytest.raises(KeyError, match=r"column 'x2' is required"):
        mboost.glmboost(
            "y ~ x2", data=data, family=mboost.Gaussian(),
            control=mboost.boost_control(10, 0.1),
        )


def test_gamboost_empty_formula():
    # `y ~ 1` is the offset-only model: no learner is selected (#357).
    data = {"y": y, "x1": x1}
    fit = mboost.gamboost(
        "y ~ 1", data=data, family=mboost.Gaussian(),
        control=mboost.boost_control(10, 0.1),
    )
    assert list(fit.selected) == []
