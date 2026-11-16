"""Regression tests for #265: ``Btree`` default ``selection="sse"``.

The default was ``"ctree"`` until #265. Evidence (California housing spatial
holdout, #263): a joint depth-6 tree scores 0.2638 with ``"sse"`` vs 0.2705
with ``"ctree"`` at the same depth and budget, and autospec's own
residual-tree policy already used ``"sse"``. The ``"ctree"`` path is kept for
R parity (see ``tests/test_gamboost_parity/test_btree_parity.py``, which pins
it explicitly).
"""

import numpy as np
import pytest

import mboostrs as mb
from mboostrs.learners import Btree


def _hole_dgp(seed):
    """Non-monotone 'hole' DGP: y depends on |x1|, whose linear association
    with y is ~0, so ctree's first-split statistic cannot see it."""
    rng = np.random.default_rng(seed)
    n = 400
    x1 = rng.uniform(-2, 2, n)
    x2 = rng.uniform(-2, 2, n)
    y = (np.abs(x1) > 1).astype(float) + rng.normal(0, 0.1, n)
    return {"x1": x1, "x2": x2, "y": y}


def _fit(selection, data):
    return mb.gamboost(
        "y ~ btree(x1, x2, maxdepth=2)", data, family=mb.Gaussian(),
        control=mb.boost_control(50, 0.3),
        learners=[Btree(["x1", "x2"], maxdepth=2, selection=selection)])


def test_btree_default_selection_is_sse():
    assert Btree(["x"]).selection == "sse"


def test_explicit_ctree_selection_still_accepted():
    assert Btree(["x"], selection="ctree").selection == "ctree"
    fit = _fit("ctree", _hole_dgp(7))
    risk = np.asarray(fit.risk)
    assert risk[-1] < risk[0]  # it still learns


def test_invalid_selection_still_rejected():
    with pytest.raises(ValueError, match="selection must be"):
        Btree(["x"], selection="cart")


def test_sse_beats_ctree_on_nonmonotone_effect():
    """ctree's linear association statistic is near zero for the hole DGP,
    so it wastes splits on the noise column; sse finds x1. Guarded over
    three seeds (verified 2026-10-05: sse wins on all three)."""
    for seed in (7, 11, 23):
        data = _hole_dgp(seed)
        r_sse = float(np.asarray(_fit("sse", data).risk)[-1])
        r_ct = float(np.asarray(_fit("ctree", data).risk)[-1])
        assert r_sse < r_ct, f"seed {seed}: sse={r_sse:.3f} ctree={r_ct:.3f}"


def test_formula_without_selection_matches_explicit_sse():
    """A formula ``btree(...)`` with no ``selection=`` must take the new
    default path, i.e. behave exactly like explicit ``selection="sse"``."""
    data = _hole_dgp(7)
    default_fit = mb.gamboost(
        "y ~ btree(x1, x2, maxdepth=2)", data, family=mb.Gaussian(),
        control=mb.boost_control(50, 0.3))
    explicit_fit = mb.gamboost(
        "y ~ btree(x1, x2, maxdepth=2, selection=sse)", data,
        family=mb.Gaussian(), control=mb.boost_control(50, 0.3))
    np.testing.assert_allclose(np.asarray(default_fit.risk),
                               np.asarray(explicit_fit.risk), rtol=1e-12)
