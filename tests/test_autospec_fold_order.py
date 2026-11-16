"""autospec moves a rolling-origin FoldScheme to row order as a whole (#61).

A time column makes autospec cross-validate on a rolling origin. The scheme
numbers rows in time order, and autospec moved its weight matrix and judge
mask to the frame's row order but passed the original scheme to the refits
that read it (cvrisk, the mstop probe). When the rows were not sorted by time
those refits trained on rows later than the ones they scored, and the same
data in another row order gave another recommendation: bols(t) at mstop 100
sorted, bbs(t) at mstop 77 shuffled.
"""

import datetime

import numpy as np
import polars as pl

import mboostrs as mb
from mboostrs.autospec import _scheme_rows


def _frame(n=240, seed=0):
    rng = np.random.default_rng(seed)
    t = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(n)]
    x, z = rng.uniform(-1, 1, n), rng.normal(size=n)
    y = np.sin(3 * x) + 0.01 * np.arange(n) + 0.5 * z + rng.normal(0, 0.3, n)
    return pl.DataFrame({"t": t, "x": x, "z": z, "y": y}), rng


def test_the_recommendation_does_not_depend_on_row_order():
    data, rng = _frame()
    shuffled = data[rng.permutation(data.height)]
    a = mb.autospec(data, "y", budget=40, seed=0)
    b = mb.autospec(shuffled, "y", budget=40, seed=0)
    assert "rolling-origin" in a.report()
    assert (a.formula, a.mstop) == (b.formula, b.mstop)


def test_scheme_rows_moves_every_per_row_array():
    scheme = mb.rolling_folds(40, B=4)
    order = np.random.default_rng(1).permutation(40)
    moved = _scheme_rows(scheme, order)
    np.testing.assert_array_equal(moved.weights, scheme.weights[order])
    np.testing.assert_array_equal(moved.held_out, scheme.held_out[order])
    np.testing.assert_array_equal(moved.judge, scheme.judge[order])
    assert moved.scheme == scheme.scheme


def test_a_supplied_scheme_is_subset_with_the_incomplete_rows():
    data, _ = _frame(n=200)
    x = data["x"].to_numpy().copy()
    x[[3, 150]] = np.nan
    data = data.with_columns(pl.Series("x", x))
    rec = mb.autospec(data, "y", budget=20, seed=0, folds=mb.rolling_folds(200, B=4))
    assert rec.mstop > 0
