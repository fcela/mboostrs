"""Mean comparisons, tuning, planning and returned fits share the chosen step."""

import importlib

import numpy as np
import polars as pl
import pytest

import mboostrs as mb


def frame():
    rng = np.random.default_rng(679)
    x = rng.uniform(-1, 1, 180)
    mean = np.exp(2 + 1.2 * x)
    return pl.DataFrame({"x": x, "y": rng.negative_binomial(3, 3 / (3 + mean))})


@pytest.mark.parametrize("step", ["fixed", "line_search"])
def test_comparison_step_reaches_each_actual_evaluation(step):
    data = frame()
    specifications = ["bols(x)", "bbs(x, df=4)"]
    rows = mb.compare_specifications(specifications, data, "y", family=mb.NBinomial(),
        folds=3, mstop=30, nu=.02, seed=679, step=step, include_evaluations=True)
    for rhs, row in zip(specifications, rows, strict=True):
        direct = mb.cv_risk(rhs, data, "y", family=mb.NBinomial(), folds=3, mstop=30,
                            nu=.02, seed=679, step=step, return_result=True)
        assert row["step"] == step
        np.testing.assert_array_equal(row["evaluation"].fold_scores, direct.fold_scores)


@pytest.mark.parametrize("family,requested,expected", [
    (mb.NBinomial(), "auto", "line_search"), (mb.NBinomial(), "fixed", "fixed"),
    (mb.Poisson(), "auto", "fixed"), (mb.Gaussian(), "auto", "fixed")])
def test_discovery_uses_one_step_for_comparison_tuning_cost_and_final_fit(monkeypatch, family, requested, expected):
    module = importlib.import_module("mboostrs.discover")
    compare, tune, cost = module.compare_specifications, module.tune_mstop, module.estimate_cost
    comparisons, tuning, planning = [], [], []

    def observed_compare(*args, **kwargs):
        comparisons.append(kwargs["step"])
        return compare(*args, **kwargs)

    def observed_tune(*args, **kwargs):
        tuning.append(kwargs["step"])
        return tune(*args, **kwargs)

    def observed_cost(call, *args, **kwargs):
        if call in ("compare_specifications", "tune_mstop"):
            planning.append(kwargs["step"])
        return cost(call, *args, **kwargs)

    monkeypatch.setattr(module, "compare_specifications", observed_compare)
    monkeypatch.setattr(module, "tune_mstop", observed_tune)
    monkeypatch.setattr(module, "estimate_cost", observed_cost)
    result = mb.discover(frame(), "y", family=family, seed=679, spread=False,
                         interactions=False, mandatory_terms=["x"], max_mstop=100,
                         mean_step=requested)
    assert comparisons and tuning and planning
    assert set(comparisons + tuning + planning) == {expected}
    assert result.model._control.step == expected
    assert result.provenance["mean_step"] == expected
    assert result.provenance["mean_step_requested"] == requested


def test_mean_screening_refusal_avoids_fit(monkeypatch):
    module = importlib.import_module("mboostrs.discover")
    analysis = module._Analysis.__new__(module._Analysis)
    analysis.mean_terms, analysis.response, analysis.family = ["bols(x)"], "y", mb.NBinomial()
    analysis.complete, analysis.na_action = frame(), "omit"
    analysis._resolved_mean_step = "line_search"
    analysis._admit = lambda *args, **kwargs: False
    monkeypatch.setattr(module, "gamboost", lambda *args, **kwargs: pytest.fail("budget refusal must avoid screening fit"))
    assert analysis._fit_mean() is None


@pytest.mark.parametrize("value", ["unknown", None, 1])
def test_invalid_step_is_rejected(value):
    with pytest.raises(ValueError, match="mean_step"):
        mb.discover(frame(), "y", mean_step=value)


def test_default_mean_step_is_fixed_until_the_line_search_is_promoted():
    # The line search is opt-in: #679's confirmation did not pass its gates.
    import inspect
    module = importlib.import_module("mboostrs.discover")
    assert inspect.signature(module.discover).parameters["mean_step"].default == "fixed"
