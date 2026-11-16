"""``discover(mean_budget=...)``: the opt-in rate-scaled mean budget of #679.

The default ``"fixed"`` must reproduce the step counts and the output of the
discover procedure that existed before the option, byte for byte; only
``"rate_scaled"`` changes the mean model's screening and tuning step counts.
"""

import importlib
import inspect

import numpy as np
import polars as pl
import pytest

import mboostrs as mb


def frame():
    rng = np.random.default_rng(679)
    n = 240
    x1, x2 = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
    mean = np.exp(.3 + .8 * np.sin(np.pi * x1) - .4 * x2)
    return pl.DataFrame({"x1": x1, "x2": x2, "y": rng.poisson(mean).astype(float)})


def output(result, newdata):
    """Everything discover returns except wall-clock times and their cost estimates."""
    log = [{k: v for k, v in entry.items() if k != "estimated_seconds"}
           for entry in result.log if entry["stage"] != "estimate"]
    columns = {c: result.predictions[c].to_numpy().tobytes() for c in result.predictions.columns}
    return dict(formula=result.formula, family=result.family, mstop=result.mstop, log=log,
                fitted=np.asarray(result.model.predict(newdata), dtype=float).tobytes(),
                predictions=columns)


def observed_steps(monkeypatch, **options):
    """Run discover on the fixture and record the step counts of its mean fits."""
    module = importlib.import_module("mboostrs.discover")
    compare, tune = module.compare_specifications, module.tune_mstop
    comparisons, tuning = [], []

    def observed_compare(*args, **kwargs):
        comparisons.append(kwargs["mstop"])
        return compare(*args, **kwargs)

    def observed_tune(*args, **kwargs):
        tuning.append((kwargs["start"], kwargs["max_mstop"]))
        return tune(*args, **kwargs)

    monkeypatch.setattr(module, "compare_specifications", observed_compare)
    monkeypatch.setattr(module, "tune_mstop", observed_tune)
    data = frame()
    result = mb.discover(data, "y", seed=3, **options)
    return result, comparisons, tuning


def test_default_mean_budget_is_fixed():
    module = importlib.import_module("mboostrs.discover")
    assert inspect.signature(module.discover).parameters["mean_budget"].default == "fixed"


def test_default_uses_the_unscaled_step_counts(monkeypatch):
    module = importlib.import_module("mboostrs.discover")
    result, comparisons, tuning = observed_steps(monkeypatch)
    assert result.family in ("Poisson", "NBinomial")
    assert comparisons and set(comparisons) == {module.SCREEN_MSTOP} == {200}
    assert tuning[0] == (module.TUNE_START, module.DEFAULT_MAX_MSTOP) == (100, 400)
    assert result.provenance["mean_budget"] == "fixed"


def test_default_output_is_byte_identical_to_explicit_fixed():
    data = frame()
    newdata = data[180:].drop("y")
    default = mb.discover(data[:180], "y", newdata=newdata, seed=3)
    explicit = mb.discover(data[:180], "y", newdata=newdata, seed=3, mean_budget="fixed")
    assert output(default, newdata) == output(explicit, newdata)


def test_rate_scaled_multiplies_count_step_counts_by_five(monkeypatch):
    result, comparisons, tuning = observed_steps(monkeypatch, mean_budget="rate_scaled")
    assert set(comparisons) == {1000}
    assert tuning[0] == (500, 2000)
    assert result.provenance["mean_budget"] == "rate_scaled"


def test_rate_scaled_respects_an_explicit_max_mstop(monkeypatch):
    _, _, tuning = observed_steps(monkeypatch, mean_budget="rate_scaled", max_mstop=300)
    assert tuning[0] == (300, 300)


def test_rate_scaled_leaves_a_gaussian_fit_unchanged():
    rng = np.random.default_rng(679)
    x = rng.uniform(-1, 1, 200)
    data = pl.DataFrame({"x": x, "y": np.sin(np.pi * x) + rng.normal(0, .5, 200)})
    newdata = data[150:].drop("y")
    default = mb.discover(data[:150], "y", newdata=newdata, seed=3, spread=False)
    scaled = mb.discover(data[:150], "y", newdata=newdata, seed=3, spread=False,
                         mean_budget="rate_scaled")
    assert output(default, newdata) == output(scaled, newdata)


@pytest.mark.parametrize("value", ["adaptive", None, 5])
def test_invalid_mean_budget_is_rejected(value):
    with pytest.raises(ValueError, match="mean_budget"):
        mb.discover(frame(), "y", mean_budget=value)
