"""Discover's report of the spread columns it did not test, and its extrapolation
warning (issue #447, items 1/4 and 6/16)."""

import numpy as np
import polars as pl

import mboostrs as mb


def _frame(seed=0, n=300):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    h = np.exp(rng.normal(5.0, 1.0, n))
    g = rng.choice(["a", "b", "c"], n)
    sd = 0.15 + 0.6 * np.sqrt(np.exp(5.0) / h)
    y = 0.6 * x + sd * rng.normal(size=n)
    return pl.DataFrame({"x": x, "h": h, "g": g, "y": y})


def test_spread_stage_names_the_categorical_column_it_did_not_test():
    result = mb.discover(_frame(), "y", seed=0, interactions=False)

    stage = [e for e in result.log if e["stage"] == "spread"]
    assert stage, "the spread stage writes a log entry"
    reason = stage[0]["reason"]
    assert "Not tested as spread candidates: g (" in reason
    assert "categorical" in reason
    assert "A spread driver among them cannot be nominated." in reason
    # The numeric columns are tested and not named as untested.
    assert "Not tested as spread candidates: x" not in reason
    assert "Not tested as spread candidates: h" not in reason


def test_spread_stage_says_every_predictor_was_tested_when_none_is_skipped():
    rng = np.random.default_rng(5)
    n = 300
    x = rng.uniform(-2, 2, n)
    h = np.exp(rng.normal(5.0, 1.0, n))
    y = 0.6 * x + rng.normal(0, 0.5, n)
    frame = pl.DataFrame({"x": x, "h": h, "y": y})
    result = mb.discover(frame, "y", seed=0, interactions=False)

    stage = [e for e in result.log if e["stage"] == "spread"]
    assert "Every predictor in the data was tested as a spread candidate." in stage[0]["reason"]


def test_real_extrapolation_in_discover_names_discover_once():
    rng = np.random.default_rng(1)
    n = 300
    x = rng.uniform(0, 1, n)
    x[:6] = [5.0, 6.0, -4.0, 7.0, 8.0, 9.0]  # rare extreme rows, held out in some folds
    z = rng.normal(size=n)
    y = np.sin(3 * x) + 0.3 * z + rng.normal(0, 0.3, n)
    frame = pl.DataFrame({"x": x, "z": z, "y": y})

    result = mb.discover(frame, "y", seed=0, folds=5, interactions=False)

    named = [w for w in result.warnings if "outside the range" in w]
    assert len(named) == 1
    assert "Pass discover(extrapolation='allow')" in named[0]
    assert "compare_specifications(" not in named[0]
