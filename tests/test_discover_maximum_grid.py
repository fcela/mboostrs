"""The discovery cap reaches the public tuner instead of its 10k default."""
import importlib
from types import SimpleNamespace

import numpy as np
import pytest

import mboostrs as mb


@pytest.mark.parametrize("ceiling", [800, 12800, 25600])
def test_requested_ceiling_reaches_actual_tuner_and_preflight(monkeypatch, ceiling):
    module = importlib.import_module("mboostrs.discover")
    actual_grids, preflight = [], []
    original_cost = module.estimate_cost

    def bounded_tuner(formula, data, *, start, max_mstop=10000, **kwargs):
        actual = min(start, max_mstop)
        actual_grids.append(actual)
        return SimpleNamespace(mstop=actual, at_boundary=True)

    def cost(function, *args, **kwargs):
        if function == "tune_mstop":
            preflight.append(min(kwargs["start"], kwargs.get("max_mstop", 10000)))
        return original_cost(function, *args, **kwargs)

    monkeypatch.setattr(module, "tune_mstop", bounded_tuner)
    monkeypatch.setattr(module, "estimate_cost", cost)
    rng = np.random.default_rng(619212)
    x = rng.uniform(-1, 1, 200)
    result = mb.discover({"x": x, "y": np.sin(np.pi * x) + rng.normal(0, .05, len(x))},
                         "y", seed=3, folds=3, interactions=False, spread=False,
                         max_mstop=ceiling)
    assert max(actual_grids) == ceiling and result.mstop == ceiling
    assert max(preflight) == ceiling
    assert any(str(ceiling) in row["reason"] for row in result.log if row["stage"] == "tune")
