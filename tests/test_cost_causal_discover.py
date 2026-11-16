"""estimate_cost covers aipw, effect_model, ci_test, iv_effect,
confounding_benchmark and discover, and each estimate equals the fits the call
makes. discover records the measured seconds of each stage."""
from __future__ import annotations

import importlib
import warnings

import numpy as np
import polars as pl
import pytest

import mboostrs as mb
from mboostrs.cost import CALLS, estimate_cost

explore = importlib.import_module("mboostrs.explore")
discover_module = importlib.import_module("mboostrs.discover")


@pytest.fixture(autouse=True)
def quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def spec(**extra):
    fields = dict(
        treatment="t", outcome="y", covariates=("x", "z"), estimand="ate",
        outcome_type="continuous", target_population="p", unit="u",
        time_zero="t0", follow_up="1y", treatment_contrast=(0, 1),
        adjustment_rationale="r")
    fields.update(extra)
    return mb.CausalSpec(**fields)


def causal_data(n=300, seed=0, missing=False):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    t = (rng.random(n) < 1 / (1 + np.exp(-x))).astype(float)
    y = x + z + 2 * t + rng.normal(size=n)
    if missing:
        y = np.where(rng.random(n) < 0.15, np.nan, y)
    return {"t": t, "y": y, "x": x, "z": z, "inst": (rng.random(n) < 0.5).astype(float)}


class FitCounter:
    """Counts model fits as the cost estimate counts them: the fits of each
    tune_mstop call (its folds plus the full-data fit) and each direct
    gamboost or glmboost call of the causal code."""

    def __init__(self, monkeypatch):
        self.fits = 0
        original_tune = explore.tune_mstop
        counter = self

        def tune(*args, **kwargs):
            result = original_tune(*args, **kwargs)
            counter.fits += result.fits + 1
            return result

        monkeypatch.setattr(explore, "tune_mstop", tune)
        for name in ("gamboost", "glmboost"):
            original = getattr(mb, name)

            def wrapper(*args, _original=original, **kwargs):
                counter.fits += 1
                return _original(*args, **kwargs)

            monkeypatch.setattr(mb, name, wrapper)


def test_calls_list_names_the_new_calls():
    for name in ("aipw", "effect_model", "ci_test", "iv_effect",
                 "confounding_benchmark", "discover"):
        assert name in CALLS


# ---- aipw ----------------------------------------------------------------

@pytest.mark.parametrize("options", [
    dict(folds=3, outcome_model="separate"),
    dict(folds=3, outcome_model="pooled"),
    dict(folds=3, outcome_model="separate", tune=False),
    dict(folds=3, outcome_model="separate", tune=False, repeats=2),
    dict(folds=3, outcome_model="pooled", inner_folds=3),
    dict(folds=3, outcome_model="separate", propensity_learner="auto"),
])
def test_aipw_estimate_equals_the_fits(monkeypatch, options):
    data = causal_data()
    cost = estimate_cost("aipw", data, spec(), **options)
    counter = FitCounter(monkeypatch)
    mb.aipw(data, spec(), **options)
    assert cost.fits == counter.fits
    assert cost.seconds_per_fit > 0


def test_aipw_estimate_counts_the_response_model(monkeypatch):
    data = causal_data(missing=True)
    options = dict(folds=3, outcome_model="separate", outcome_missing="ipw")
    cost = estimate_cost("aipw", data, spec(), **options)
    counter = FitCounter(monkeypatch)
    mb.aipw(data, spec(), **options)
    assert cost.fits == counter.fits


def test_aipw_estimate_reads_defaults_from_the_signature(monkeypatch):
    import inspect

    from mboostrs import causal

    data = causal_data()
    default_tune = inspect.signature(causal.aipw).parameters["tune"].default
    assert default_tune is True
    untuned = estimate_cost("aipw", data, spec(), folds=3, outcome_model="separate",
                            tune=False)
    tuned = estimate_cost("aipw", data, spec(), folds=3, outcome_model="separate")
    assert tuned.fits == untuned.fits * 6


def test_aipw_auto_is_an_upper_bound(monkeypatch):
    data = causal_data()
    cost = estimate_cost("aipw", data, spec(), folds=3, outcome_model="auto")
    counter = FitCounter(monkeypatch)
    mb.aipw(data, spec(), folds=3, outcome_model="auto")
    assert counter.fits <= cost.fits
    assert "upper bound" in cost.explanation


def test_aipw_default_counts_every_cv_candidate(monkeypatch):
    # The default outcome_model is "cv", which fits every candidate.
    data = causal_data()
    cost = estimate_cost("aipw", data, spec(), folds=3)
    counter = FitCounter(monkeypatch)
    mb.aipw(data, spec(), folds=3)
    assert counter.fits <= cost.fits
    assert "every candidate" in cost.explanation
    pooled = estimate_cost("aipw", data, spec(), folds=3, outcome_model="pooled")
    assert cost.fits > pooled.fits


def test_aipw_cv_outcome_model_counts_every_candidate():
    data = causal_data()
    separate = estimate_cost("aipw", data, spec(), folds=3, outcome_model="separate",
                             tune=False)
    pooled = estimate_cost("aipw", data, spec(), folds=3, outcome_model="pooled",
                           tune=False)
    # The value need not exist in the call yet; the plan counts it generically.
    from mboostrs import _cost_plans

    count, text = _cost_plans._outcome_models("cv", data, spec())
    assert count == 3
    assert "every candidate" in text
    assert separate.fits - pooled.fits == 3  # one more outcome model on each of 3 folds


# ---- effect_model --------------------------------------------------------

@pytest.mark.parametrize("options", [
    dict(folds=3, terms="smooth"),
    dict(folds=3, terms="stumps"),
    dict(folds=3, terms="joint"),
    dict(folds=3, terms="smooth", interval_method="repeats", interval_repeats=2),
])
def test_effect_model_estimate_equals_the_fits(monkeypatch, options):
    data = causal_data()
    cost = estimate_cost("effect_model", data, spec(), **options)
    counter = FitCounter(monkeypatch)
    mb.effect_model(data, spec(), **options)
    assert cost.fits == counter.fits


def test_effect_model_auto_is_exact_or_one_above(monkeypatch):
    data = causal_data()
    cost = estimate_cost("effect_model", data, spec(), folds=3)
    counter = FitCounter(monkeypatch)
    model = mb.effect_model(data, spec(), folds=3)
    assert counter.fits in (cost.fits, cost.fits - 1)
    assert (counter.fits == cost.fits) == (model.terms == "smooth")


def test_effect_model_with_supplied_nuisances_skips_them(monkeypatch):
    data = causal_data()
    result = mb.aipw(data, spec(), folds=3, outcome_model="separate")
    cost = estimate_cost("effect_model", data, spec(), nuisance=result, folds=3,
                         terms="smooth")
    counter = FitCounter(monkeypatch)
    mb.effect_model(data, spec(), nuisance=result, folds=3, terms="smooth")
    assert cost.fits == counter.fits


# ---- ci_test -------------------------------------------------------------

@pytest.mark.parametrize("given", [("z",), ("z", "w")])
def test_ci_test_estimate_equals_the_fits(monkeypatch, given):
    data = causal_data()
    data["w"] = data["x"] * 0.5 + np.random.default_rng(3).normal(size=len(data["x"]))
    cost = estimate_cost("ci_test", data, "x", "y", given, folds=3)
    counter = FitCounter(monkeypatch)
    mb.ci_test(data, "x", "y", given, folds=3)
    assert cost.fits == counter.fits
    assert cost.fits > 0


def test_ci_test_with_no_conditioning_set_fits_nothing():
    data = causal_data()
    cost = estimate_cost("ci_test", data, "x", "y", (), folds=3)
    assert cost.fits == 0
    assert cost.estimated_seconds == 0.0


def test_ci_test_estimate_runs_no_regression(monkeypatch):
    data = causal_data()
    counter = FitCounter(monkeypatch)
    estimate_cost("ci_test", data, "x", "y", ("z",), folds=3)
    assert counter.fits == 1  # the single timed fit


# ---- iv_effect -----------------------------------------------------------

@pytest.mark.parametrize("options", [
    dict(folds=3),
    dict(folds=3, tune=False),
    dict(folds=3, inner_folds=3),
])
def test_iv_effect_estimate_equals_the_fits(monkeypatch, options):
    data = causal_data()
    cost = estimate_cost("iv_effect", data, spec(), "inst", **options)
    counter = FitCounter(monkeypatch)
    mb.iv_effect(data, spec(), "inst", **options)
    assert cost.fits == counter.fits


def test_iv_effect_estimate_counts_the_response_model(monkeypatch):
    data = causal_data(missing=True)
    options = dict(folds=3, outcome_missing="ipw")
    cost = estimate_cost("iv_effect", data, spec(), "inst", **options)
    counter = FitCounter(monkeypatch)
    mb.iv_effect(data, spec(), "inst", **options)
    assert cost.fits == counter.fits


# ---- confounding_benchmark -----------------------------------------------

@pytest.mark.parametrize("tune", [True, False])
def test_confounding_benchmark_estimate_equals_the_fits(monkeypatch, tune):
    data = causal_data()
    result = mb.aipw(data, spec(), folds=3, outcome_model="separate", tune=tune)
    cost = estimate_cost("confounding_benchmark", data, result)
    counter = FitCounter(monkeypatch)
    mb.confounding_benchmark(data, result)
    assert cost.fits == counter.fits


def test_confounding_benchmark_estimate_follows_the_variables(monkeypatch):
    data = causal_data()
    result = mb.aipw(data, spec(), folds=3, outcome_model="separate", tune=False)
    both = estimate_cost("confounding_benchmark", data, result)
    one = estimate_cost("confounding_benchmark", data, result, variables=["x"])
    assert both.fits == 2 * one.fits


# ---- discover ------------------------------------------------------------

def discover_frame(n=400, seed=1):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-2, 2, n)
    x2 = rng.uniform(-2, 2, n)
    y = np.sin(2 * x1) + 1.5 * x2 + rng.normal(scale=0.3, size=n)
    return pl.DataFrame({"x1": x1, "x2": x2, "y": y})


def test_discover_estimate_equals_the_fits_of_each_stage(monkeypatch):
    frame = discover_frame()
    # max_mstop=100 keeps the step search to its first grid, which is what
    # the estimate counts; each further doubling is another tune_mstop call.
    cost = estimate_cost("discover", frame, "y", seed=0, max_mstop=100)
    events = []

    original_compare = discover_module.compare_specifications
    original_tune = discover_module.tune_mstop
    original_gamboost = discover_module.gamboost

    def compare(*args, **kwargs):
        rows = original_compare(*args, **kwargs)
        events.append(("compare", rows[-1]["cumulative fit count"]))
        return rows

    def tune(*args, **kwargs):
        result = original_tune(*args, **kwargs)
        events.append(("tune", result.fits + 1))
        return result

    def gamboost(*args, **kwargs):
        events.append(("gamboost", 1))
        return original_gamboost(*args, **kwargs)

    monkeypatch.setattr(discover_module, "compare_specifications", compare)
    monkeypatch.setattr(discover_module, "tune_mstop", tune)
    monkeypatch.setattr(discover_module, "gamboost", gamboost)
    result = mb.discover(frame, "y", seed=0, max_mstop=100)
    assert result.formula.count("bbs") == 2  # both terms kept: one elimination round
    kinds = [kind for kind, _ in events]
    assert kinds == ["compare", "gamboost", "gamboost", "tune", "gamboost"]
    actual = dict(zip(("mean effects", "interactions", "spread", "tune", "final model"),
                      (count for _, count in events), strict=True))
    assert cost.stages == actual
    assert cost.fits == sum(actual.values())
    assert cost.seconds_per_fit > 0
    assert "first pass" in cost.explanation


def test_discover_estimate_follows_the_switches():
    frame = discover_frame()
    full = estimate_cost("discover", frame, "y", seed=0)
    bare = estimate_cost("discover", frame, "y", seed=0, interactions=False, spread=False)
    assert bare.stages["interactions"] == 0 and bare.stages["spread"] == 0
    assert bare.fits == full.fits - 2


def test_discover_records_the_seconds_of_each_stage():
    result = mb.discover(discover_frame(), "y", seed=0)
    seconds = result.stage_seconds
    for stage in ("mean effects", "interactions", "spread", "tune", "refit"):
        assert seconds[stage] >= 0
    assert "predict" not in seconds  # no newdata
    assert seconds["mean effects"] > 0
    assert result.to_dict()["stage_seconds"] == seconds
    assert f"seconds={sum(seconds.values()):.1f}" in repr(result)


def test_discover_records_prediction_seconds_with_newdata():
    frame = discover_frame()
    result = mb.discover(frame, "y", seed=0, newdata=frame.head(20))
    assert result.stage_seconds["predict"] > 0
