"""The two-part outcome model of aipw for a zero-inflated outcome.

The fits here use a small synthetic population with a thin-overlap tail: above a
covariate cut-off, few rows are treated, so the sparse-arm check flags a bin.
They check the model's selection, its refusals and its formula record, and the
two-part helpers that aipw calls. The simulation study with its tables is in the
PR body and the causal topic, not in these tests.
"""

import warnings

import numpy as np
import polars as pl
import pytest

from mboostrs import _rust, causal
from mboostrs.causal import CausalSpec, aipw


def _spec(**overrides):
    values = dict(
        treatment="treated",
        outcome="earnings",
        covariates=("x1", "x2"),
        estimand="ate",
        outcome_type="continuous",
        time_zero="assignment",
        follow_up="one year after assignment",
        adjustment_rationale="x1 and x2 drive both assignment and earnings",
        outcome_missing="error",
    )
    values.update(overrides)
    return CausalSpec(**values)


def _tail_population(n=1500, seed=3, zeros=True):
    """A population whose treated rows are scarce above x1 = 1.2, with an outcome
    that is zero for some rows (``zeros=True``) and strictly positive otherwise."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    eta_treat = -0.5 + 0.8 * x1 + 0.3 * x2 - 6.0 * (x1 > 1.2)
    treated = (rng.random(n) < 1 / (1 + np.exp(-eta_treat))).astype(float)
    log_level = 1.0 + 0.5 * x1 + 0.2 * x2 + 0.2 * treated
    earnings = np.exp(log_level + 0.3 * rng.normal(size=n))
    if zeros:
        employed = rng.random(n) < 1 / (1 + np.exp(-(0.5 + 0.8 * x1 + 0.2 * treated)))
        earnings = np.where(employed, earnings, 0.0)
    return pl.DataFrame({"treated": treated, "earnings": earnings, "x1": x1, "x2": x2})


def _fit(frame, **kwargs):
    options = dict(smooth=True, tune=False, mstop=60, folds=3, seed=7)
    options.update(kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return aipw(frame, _spec(), **options)


def test_two_part_helpers_compute_the_formulas():
    s2 = _rust.log_residual_variance(
        np.array([1.0, 3.0, 5.0]), np.array([0.0, 3.0, 4.0]), np.array([1.0, 1.0, 2.0]))
    assert s2 == pytest.approx(0.75)
    mean = _rust.two_part_mean(np.array([0.5, 1.0]), np.array([2.0, 0.0]), 0.5)
    assert mean == pytest.approx([0.5 * np.exp(2.25), np.exp(0.25)])
    with pytest.raises(ValueError):
        _rust.two_part_mean(np.array([0.5]), np.array([2.0]), -1.0)


def test_two_part_is_refused_for_a_binary_outcome():
    frame = _tail_population().with_columns(
        (pl.col("earnings") > 0).cast(pl.Float64).alias("earnings"))
    with pytest.raises(ValueError, match="continuous outcome"):
        aipw(frame, _spec(outcome_type="binary"), outcome_model="two_part")


def test_two_part_is_refused_for_negative_values_and_for_no_zeros():
    negative = _tail_population().with_columns(pl.col("earnings") - 1.0)
    with pytest.raises(ValueError, match="negative values"):
        aipw(negative, _spec(), outcome_model="two_part")
    positive = _tail_population(zeros=False)
    with pytest.raises(ValueError, match="no zero values"):
        aipw(positive, _spec(), outcome_model="two_part")


def test_two_part_is_refused_with_treatment_interactions():
    with pytest.raises(ValueError, match="two-part model has no treatment interactions"):
        aipw(_tail_population(), _spec(), outcome_model="two_part", outcome_interactions=True)


def test_two_part_fit_records_its_model_and_both_formulas():
    result = _fit(_tail_population(), outcome_model="two_part")
    assert result.diagnostics["outcome_model"] == "two_part"
    choice = result.diagnostics["outcome_model_choice"]
    assert choice["requested"] == "two_part" and choice["used"] == "two_part"
    assert choice["outcome_interactions"] is False
    terms = result.nuisance["terms"]["outcome"]
    assert "treated" in terms and "by=" not in terms
    assert np.isfinite(result.estimate)
    assert np.all(np.isfinite(result.mu0)) and np.all(np.isfinite(result.mu1))
    assert set(result.nuisance["selected_mstop"][0]) >= {"propensity", "employment", "log_outcome"}
    assert "two-part" in result.describe()


@pytest.fixture
def above_threshold(monkeypatch):
    """Treat the 1,500-row fixture as larger than the row threshold of the
    small-sample rule, so ``auto`` applies the sparse-arm rule."""
    monkeypatch.setattr(causal, "_AUTO_POOLED_MAX_ROWS", 0)


def test_auto_keeps_the_pooled_model_for_a_flagged_zero_inflated_outcome(above_threshold):
    # Issue #456: auto never selects the two-part model, so the flagged bin of a
    # zero-inflated outcome gets the pooled model with treatment interactions.
    result = _fit(_tail_population(), outcome_model="auto")
    sparse = result.diagnostics["sparse_arms"]
    assert sparse["flagged_bins"], "the fixture must flag a thin propensity bin"
    assert result.diagnostics["outcome_model"] == "pooled"
    assert result.diagnostics["outcome_model_choice"]["outcome_interactions"] is True
    assert result.diagnostics["outcome_model_choice"]["requested"] == "auto"


def test_auto_keeps_pooled_interactions_for_a_positive_outcome(above_threshold):
    # Without zeros the auto rule is unchanged: the flagged bin selects the pooled model
    # with treatment interactions, as before this option existed.
    result = _fit(_tail_population(zeros=False), outcome_model="auto")
    assert result.diagnostics["sparse_arms"]["flagged_bins"]
    assert result.diagnostics["outcome_model"] == "pooled"
    assert result.diagnostics["outcome_model_choice"]["outcome_interactions"] is True


def test_explicit_separate_and_pooled_requests_are_honoured_for_zero_inflated_outcomes():
    frame = _tail_population()
    separate = _fit(frame, outcome_model="separate")
    pooled = _fit(frame, outcome_model="pooled")
    assert separate.diagnostics["outcome_model"] == "separate"
    assert pooled.diagnostics["outcome_model"] == "pooled"
    assert separate.diagnostics["outcome_model_choice"]["requested"] == "separate"
