"""Entry-point validation of autospec and discover count arguments (#822)."""

import numpy as np
import polars as pl
import pytest

import mboostrs as mb


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(0)
    x = rng.normal(size=120)
    return pl.DataFrame({"x": x, "y": 2 * x + rng.normal(size=120)})


@pytest.mark.parametrize("name, value", [
    ("mstop", 2.7), ("mstop", 2.0), ("mstop", True), ("mstop", "5"),
    ("budget", 1.5), ("budget", None), ("budget", False),
])
def test_autospec_rejects_non_integers(frame, name, value):
    with pytest.raises(TypeError, match=name):
        mb.autospec(frame, "y", **{name: value})


@pytest.mark.parametrize("name, value", [("mstop", 0), ("mstop", -3), ("budget", -3)])
def test_autospec_rejects_non_positive(frame, name, value):
    with pytest.raises(ValueError, match=name):
        mb.autospec(frame, "y", **{name: value})


def test_autospec_accepts_numpy_integers(frame):
    rec = mb.autospec(frame, "y", mstop=np.int64(50), budget=np.int32(0))
    assert rec.mstop == 50


@pytest.mark.parametrize("value", [2.5, 3.0, True, "3", None])
def test_discover_rejects_non_integer_folds(frame, value):
    with pytest.raises(TypeError, match="folds"):
        mb.discover(frame, "y", folds=value)


@pytest.mark.parametrize("value", [1, 0, -2])
def test_discover_rejects_small_folds(frame, value):
    with pytest.raises(ValueError, match="folds"):
        mb.discover(frame, "y", folds=value)


@pytest.mark.parametrize("value, error", [(2.5, TypeError), (0, ValueError)])
def test_discover_validates_max_mstop(frame, value, error):
    with pytest.raises(error, match="max_mstop"):
        mb.discover(frame, "y", max_mstop=value)
