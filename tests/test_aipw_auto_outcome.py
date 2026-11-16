"""aipw(outcome_model="auto"): the sparse-arm check picks the outcome model.

The sparse design has a tail (z above 1) where treated rows are about 0.2% of
all rows, and the propensity deciles of that tail hold fewer than 20 treated
rows, so the check flags a bin. The balanced design has no tail and no flag.
Fits use n = 2,000 and few seeds so the file runs in well under a minute.
"""

import inspect
import warnings

import numpy as np

import mboostrs as mb
from mboostrs import causal

N = 2000
FAST = dict(smooth=True, folds=5, tune=False, mstop=200)


def expit(x):
    return 1 / (1 + np.exp(-x))


def sparse_design(seed, *, effect):
    """Treatment is almost absent above z = 1. The outcome baseline grows with z,
    and the effect, when present, is half the baseline (so it is largest in the
    tail where treated rows are scarce)."""
    rng = np.random.default_rng(seed)
    x1, z = rng.normal(size=(2, N))
    baseline = 3000 * np.exp(0.5 * z)
    logit = -0.2 + 0.3 * x1 - 0.8 * z - 40.0 * np.maximum(z - 1.0, 0)
    a = rng.binomial(1, expit(logit))
    tau = 0.5 * baseline if effect else np.zeros(N)
    y = baseline + 300 * x1 + rng.normal(scale=500, size=N) + a * tau
    return {"a": a, "y": y, "x1": x1, "z": z}, tau


def balanced_design(seed):
    """Treatment depends on x1 and z over their whole range; no tail."""
    rng = np.random.default_rng(seed)
    x1, z = rng.normal(size=(2, N))
    a = rng.binomial(1, expit(0.4 * x1 - 0.3 * z))
    y = x1 + 0.5 * z + a * (1 + 0.5 * x1) + rng.normal(size=N)
    return {"a": a, "y": y, "x1": x1, "z": z}


def spec():
    return mb.CausalSpec(
        treatment="a", outcome="y", covariates=("x1", "z"), target_population="simulated",
        unit="person", time_zero="treatment assignment", follow_up="one year",
        adjustment_rationale="x1 and z cause both treatment and outcome")


def sparse_messages(caught):
    return [str(w.message) for w in caught if "propensity bin" in str(w.message)]


def fit(data, **options):
    # These tests are about the "auto" rule, which is no longer the default.
    options.setdefault("outcome_model", "auto")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = mb.aipw(data, spec(), **options)
    return result, sparse_messages(caught)


def test_outcome_model_defaults_to_cv_and_auto_is_requested_by_name():
    assert inspect.signature(mb.aipw).parameters["outcome_model"].default == "cv"


def test_auto_picks_pooled_with_interactions_on_the_sparse_design():
    data, _ = sparse_design(0, effect=True)
    result, messages = fit(data, **FAST)
    choice = result.diagnostics["outcome_model_choice"]
    assert choice["requested"] == "auto" and choice["used"] == "pooled"
    assert choice["outcome_interactions"] is True
    assert result.diagnostics["outcome_model"] == "pooled"
    assert result.nuisance["outcome_model"] == "pooled"
    assert result.nuisance["outcome_interactions"] is True
    assert "by=a" in result.nuisance["terms"]["outcome"]
    assert result.diagnostics["sparse_arms"]["flagged_bins"]
    # The check flagged a bin, so the warning is issued and names the model used.
    assert len(messages) == 1
    assert "outcome_model='pooled' with outcome_interactions=True" in messages[0]
    assert "chosen by outcome_model='auto'" in messages[0]
    assert "of all rows" in messages[0] and "of the all rows" not in messages[0]


def test_auto_picks_separate_models_on_a_balanced_design():
    result, messages = fit(balanced_design(0), **FAST)
    choice = result.diagnostics["outcome_model_choice"]
    assert choice["requested"] == "auto" and choice["used"] == "separate"
    assert choice["outcome_interactions"] is False
    assert result.diagnostics["sparse_arms"]["flagged_bins"] == []
    assert messages == []
    assert result.nuisance["terms"]["outcome"] == result.nuisance["terms"]["propensity"]
    assert "a" not in result.nuisance["terms"]["outcome"].split(" + ")


def test_auto_with_interactions_requested_uses_the_pooled_model():
    result, _ = fit(balanced_design(1), outcome_interactions=True, **FAST)
    choice = result.diagnostics["outcome_model_choice"]
    assert choice["used"] == "pooled" and choice["outcome_interactions"] is True
    assert "needs the pooled model" in choice["reason"]


def test_explicit_separate_with_interactions_is_refused():
    with np.testing.assert_raises_regex(ValueError, "applies to outcome_model='pooled' or 'auto'"):
        mb.aipw(balanced_design(1), spec(), outcome_model="separate",
                outcome_interactions=True, **FAST)


def test_unknown_outcome_model_names_auto():
    with np.testing.assert_raises_regex(ValueError, "'auto'"):
        mb.aipw(balanced_design(1), spec(), outcome_model="shared", **FAST)


def test_explicit_separate_keeps_its_warning_and_names_the_model():
    data, _ = sparse_design(0, effect=True)
    result, messages = fit(data, outcome_model="separate", **FAST)
    choice = result.diagnostics["outcome_model_choice"]
    assert choice["requested"] == "separate" and choice["used"] == "separate"
    assert result.diagnostics["sparse_arms"]["warned"] is True
    assert len(messages) == 1
    assert "outcome_model='separate' (one outcome model per arm)" in messages[0]
    assert "chosen by outcome_model='auto'" not in messages[0]
    assert "Pass outcome_model='pooled'" in messages[0]


def test_explicit_pooled_without_interactions_does_not_warn():
    data, _ = sparse_design(0, effect=True)
    result, messages = fit(data, outcome_model="pooled", **FAST)
    assert messages == []
    assert result.diagnostics["outcome_model_choice"]["used"] == "pooled"
    assert result.diagnostics["outcome_model_choice"]["outcome_interactions"] is False


def test_no_effect_sparse_design_interval_covers_zero_in_most_seeds():
    covered = []
    for seed in range(5):
        data, _ = sparse_design(seed, effect=False)
        result, _ = fit(data, **FAST)
        assert result.diagnostics["outcome_model"] == "pooled"
        lower, upper = result.interval
        covered.append(lower <= 0 <= upper)
    assert sum(covered) >= 4, covered


def test_describe_states_the_model_and_the_reason():
    data, _ = sparse_design(0, effect=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = mb.aipw(data, spec(), outcome_model="auto", **FAST)
    text = result.describe()
    assert ("Outcome model: pooled with treatment interactions, chosen by outcome_model='auto' "
            "because the sparse-arm check flagged propensity bin(s) [") in text
    balanced, _ = fit(balanced_design(0), **FAST)
    assert ("Outcome model: separate, chosen by outcome_model='auto' because the sparse-arm "
            "check flagged no propensity bin") in balanced.describe()


def test_confounding_benchmark_refits_with_the_resolved_model_and_max_mstop(monkeypatch):
    data, _ = sparse_design(0, effect=True)
    result, _ = fit(data, max_mstop=300, **FAST)
    assert result.nuisance["max_mstop"] == 300 and result.nuisance["mstop"] == 200
    calls = []
    original = causal.aipw

    def recording(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(causal, "aipw", recording)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = causal.confounding_benchmark(data, result, output="records")
    assert len(rows) == 2 and len(calls) == 2
    for call in calls:
        assert call["outcome_model"] == "pooled"
        assert call["outcome_interactions"] is True
        assert call["mstop"] == 200
        assert call["max_mstop"] == 300
