"""Binary scalar coding must not remove the inactive level's curve."""
import numpy as np
import polars as pl
import pytest

import mboostrs as mb


def opposed(encoding, *, logged=False):
    rng = np.random.default_rng(618212)
    axis = rng.uniform(0, 4, 250) if logged else rng.uniform(-1, 1, 250)
    original = np.exp(axis) if logged else axis
    curve = axis / 2 - 1 if logged else axis
    b = np.tile([0, 1], len(axis))
    variants = {"indicator": b, "boolean": b == 1, "signed": 1 - 2 * b, "shifted": b + 2}
    x = np.repeat(original, 2)
    z = np.repeat(rng.uniform(-1, 1, len(axis)), 2)
    y = np.repeat(2 * np.sin(np.pi * curve), 2) * (1 - 2 * b) + .5 * z + rng.normal(0, .2, len(x))
    return pl.DataFrame({"x": x, "g": variants[encoding], "z": z, "y": y})


@pytest.mark.parametrize("encoding, values", [("indicator", [0, 1]), ("boolean", [False, True])])
def test_opposed_indicator_curves_are_recovered_without_a_marginal_main_effect(encoding, values):
    result = mb.discover(opposed(encoding), "y", seed=7, folds=3, spread=False, max_mstop=800)
    grid = np.linspace(-.9, .9, 100)
    for value, sign in zip(values, [1, -1], strict=True):
        prediction = result.model.predict({"x": grid, "g": np.repeat(value, len(grid)), "z": np.zeros(len(grid))})
        assert np.mean((prediction - sign * 2 * np.sin(np.pi * grid)) ** 2) < .15


def test_a_signed_by_still_represents_both_signs():
    result = mb.discover(opposed("signed"), "y", seed=7, folds=3, spread=False, max_mstop=800)
    grid = np.linspace(-.9, .9, 100)
    for value in [1, -1]:
        prediction = result.model.predict({"x": grid, "g": np.repeat(value, len(grid)), "z": np.zeros(len(grid))})
        assert np.mean((prediction - value * 2 * np.sin(np.pi * grid)) ** 2) < .15


def test_log_scale_indicator_reference_is_on_the_original_input_scale():
    result = mb.discover(opposed("indicator", logged=True), "y", seed=7, folds=3, spread=False, max_mstop=800)
    axis = np.linspace(.1, 3.9, 100)
    prediction = result.model.predict({"x": np.exp(axis), "g": np.zeros(len(axis)), "z": np.zeros(len(axis))})
    assert np.mean((prediction - 2 * np.sin(np.pi * (axis / 2 - 1))) ** 2) < .15


def test_strong_heredity_still_blocks_a_pair_without_its_main_terms():
    result = mb.discover(opposed("indicator"), "y", seed=7, folds=3, spread=False, heredity="strong")
    assert not any(set(r["variables"]) == {"x", "g"} for r in result.interactions)
