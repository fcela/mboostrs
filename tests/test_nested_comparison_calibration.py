"""The paired standard error of a nested (mstop="cv") comparison under the null.

`tools/simulate_nested_comparison.py` measures the calibration over 1,000
replicates; `compare_specifications` documents the result. This runs 150 of
those replicates on fixed seeds, so a change that breaks the pairing (for
example, candidates scored on different folds, which inflates the standard
error, or inner tuning that leaks held-out rows, which biases the change) moves
the statistic out of the range the documentation describes.
"""

import importlib.util
import pathlib

import numpy as np

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "tools" / "simulate_nested_comparison.py"


def _simulation():
    spec = importlib.util.spec_from_file_location("simulate_nested_comparison", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_paired_statistic_is_near_standard_under_the_null():
    sim = _simulation()
    z = np.array([sim.replicate(1000 + r, beta=0.0) for r in range(150)])
    assert abs(z.mean()) < 0.6
    assert 0.8 < z.std(ddof=1) < 1.5
    assert np.mean(z < -2) <= 0.06


def test_a_real_effect_moves_the_statistic():
    sim = _simulation()
    z = np.array([sim.replicate(5000 + r, beta=0.3) for r in range(40)])
    assert np.median(z) < -2
