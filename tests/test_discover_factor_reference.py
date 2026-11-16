"""A treatment-coded interaction needs its reference-level curve after elimination."""
import numpy as np
import polars as pl
import pytest

import mboostrs as mb


def opposed(seed=618206, *, labels=("a", "b"), logged=False):
    rng = np.random.default_rng(seed)
    axis = rng.uniform(0, 4, 250) if logged else rng.uniform(-1, 1, 250)
    curve_axis = axis / 2 - 1 if logged else axis
    x = np.repeat(np.exp(axis) if logged else axis, 2)
    sign = np.tile([1, -1], len(axis))
    z = np.repeat(rng.uniform(-1, 1, len(axis)), 2)
    y = np.repeat(2 * np.sin(np.pi * curve_axis), 2) * sign + .5 * z + rng.normal(0, .2, len(x))
    return pl.DataFrame({"x": x, "g": np.tile(labels, len(axis)), "z": z, "y": y})


@pytest.mark.parametrize("labels", [("a", "b"), ("zeta", "alpha")])
def test_discovery_represents_the_reference_curve_when_marginal_effect_is_zero(labels):
    result = mb.discover(opposed(labels=labels), "y", seed=7, spread=False, folds=3, max_mstop=800)
    assert any(set(r["variables"]) == {"x", "g"} for r in result.interactions)
    grid = np.linspace(-.9, .9, 100)
    for label, sign in zip(labels, [1, -1], strict=True):
        prediction = result.model.predict({"x": grid, "z": np.zeros(len(grid)),
                                           "g": np.repeat(label, len(grid))})
        assert np.mean((prediction - sign * 2 * np.sin(np.pi * grid)) ** 2) < .15
    adopted = next(r for r in result.log if r["stage"] == "interactions" and r["adopted"])
    assert "bbs(x, df=4)" in adopted["reference_terms"]


def test_log_scale_reference_is_replayed_on_the_original_input_scale():
    result = mb.discover(opposed(logged=True), "y", seed=7, spread=False, folds=3, max_mstop=800)
    assert "bbs(log(x), df=4)" in result.formula
    axis = np.linspace(.1, 3.9, 100)
    for label, sign in (("a", 1), ("b", -1)):
        prediction = result.model.predict({"x": np.exp(axis), "z": np.zeros(len(axis)),
                                           "g": np.repeat(label, len(axis))})
        assert np.mean((prediction - sign * 2 * np.sin(np.pi * (axis / 2 - 1))) ** 2) < .15


def test_reference_completion_does_not_override_strong_heredity():
    result = mb.discover(opposed(), "y", seed=7, spread=False, folds=3, heredity="strong")
    assert not any(set(r["variables"]) == {"x", "g"} for r in result.interactions)
    assert any("heredity strong" in r["reason"] for r in result.log)
