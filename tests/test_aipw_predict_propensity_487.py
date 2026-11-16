"""AIPWResult.predict_propensity: the all-rows propensity model of an aipw fit (issue #487)."""

import dataclasses

import numpy as np
import polars as pl
import pytest

from mboostrs.causal import CausalSpec, aipw

FAST = dict(smooth=True, folds=5, tune=False, mstop=100)


def _spec():
    return CausalSpec(
        treatment="a", outcome="y", covariates=("x1", "x2"), target_population="simulated",
        unit="person", time_zero="treatment assignment", follow_up="one year",
        adjustment_rationale="x1 and x2 cause both treatment and outcome",
    )


def _data(n=300, seed=487):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=(2, n))
    a = rng.binomial(1, 1 / (1 + np.exp(-(0.4 * x1 - 0.6 * x2))))
    y = 1.0 * a + x1 - 0.5 * x2 + rng.normal(size=n)
    return pl.DataFrame({"a": a, "y": y, "x1": x1, "x2": x2})


@pytest.fixture(scope="module")
def fitted():
    data = _data()
    return data, aipw(data, _spec(), **FAST)


def test_values_lie_inside_propensity_clip(fitted):
    data, result = fitted
    p = result.predict_propensity(data)
    lo, hi = result.diagnostics["propensity_clip"]
    assert isinstance(p, np.ndarray)
    assert p.shape == (data.height,)
    assert np.all(p >= lo) and np.all(p <= hi)


def test_training_rows_correlate_with_cross_fitted_propensity(fitted):
    data, result = fitted
    p = result.predict_propensity(data)
    r = np.corrcoef(p, result.propensity)[0, 1]
    assert r > 0.9


def test_missing_covariate_column_is_named(fitted):
    data, result = fitted
    newdata = data.drop("x2")
    with pytest.raises(ValueError, match="x2"):
        result.predict_propensity(newdata)


def test_result_without_stored_model_raises(fitted):
    data, result = fitted
    bare = dataclasses.replace(result, _propensity_model=None)
    with pytest.raises(ValueError, match="no stored propensity model"):
        bare.predict_propensity(data)
