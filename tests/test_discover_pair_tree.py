"""Public discovery checks for discontinuous continuous-pair alternatives."""
import numpy as np
import polars as pl
import pytest

import mboostrs as mb


def _step(seed=912, n=450):
    rng = np.random.default_rng(seed)
    x, z, junk = rng.uniform(-1, 1, (3, n))
    y = 3*np.sign(x)*np.sign(z) + rng.normal(0, .3, n)
    return pl.DataFrame(dict(x=x, z=z, junk=junk, y=y))


def test_sharp_pair_can_use_a_tree_and_predict_new_rows():
    result = mb.discover(_step(), "y", folds=3, seed=7, max_mstop=800, spread=False)
    assert any(r["term"].startswith("btree(") and set(r["variables"]) == {"x", "z"}
               for r in result.interactions)
    holdout = _step(seed=913, n=1000)
    prediction = result.model.predict(holdout.drop("y"), type="response", extrapolation="clip")
    assert np.mean((prediction-holdout["y"].to_numpy())**2) < .8
    log = [r for r in result.log if r["stage"] == "interaction form" and r["adopted"]]
    assert len(log) >= 1 and log[0]["change"] < -2*log[0]["se"]
    assert log[0]["reference_terms"]


@pytest.mark.parametrize("rules", [
    {"forbidden_pairs": [("x", "z")]},
    {"heredity": "strong"},
    {"candidates": {"mu": ["x", "junk"]}},
    {"interactions": False},
    {"budget_seconds": 0},
])
def test_tree_forms_respect_the_same_public_constraints(rules):
    result = mb.discover(_step(), "y", folds=3, seed=7, max_mstop=200,
                         spread=False, **rules)
    assert not any(r["term"].startswith("btree(") and set(r["variables"]) == {"x", "z"}
                   for r in result.interactions)
