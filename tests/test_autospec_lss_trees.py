"""Distributional candidates that additive terms cannot represent.

The heteroscedasticity screen regresses log(residual^2) on each column
linearly, so a spread that depends on a product of two predictors left every
slope near zero and the mean model stood. When the screen finds no column,
GaussianLSS is offered with a joint tree over the numeric predictors in its
scale equation, and the held-out log score decides with the usual gates.
`lss_trees` adds a joint tree to every parameter of an adopted
distributional model, adopted only when it wins held out.
"""

import numpy as np
import polars as pl
import pytest

import mboostrs as mb


def _product_spread(n=2000, seed=0, sd_varies=True):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (n, 4))
    if sd_varies:
        mu = 2 * x[:, 0] * x[:, 1] + x[:, 2]
        sd = np.exp(-0.7 + 0.8 * x[:, 0] * x[:, 1])
    else:
        # An additive mean the mean model fits, so its residual spread is the
        # constant 0.5 (a misfitted product in the mean would leave a spread
        # that does vary, and a scale tree would rightly win).
        mu = np.sin(2 * x[:, 0]) + x[:, 2]
        sd = np.full(n, 0.5)
    y = mu + sd * rng.standard_normal(n)
    return pl.DataFrame({**{f"x{i}": x[:, i] for i in range(4)}, "y": y})


def test_a_product_spread_is_found_by_the_tree_scale_candidate():
    rec = mb.autospec(_product_spread(), "y", seed=0, budget=60)
    assert type(rec.family).__name__ == "GaussianLSS"
    assert "btree(" in rec.formula["sigma"]


def test_a_constant_spread_keeps_the_mean_family():
    rec = mb.autospec(_product_spread(sd_varies=False), "y", seed=0, budget=60)
    assert type(rec.family).__name__ in ("Gaussian", "GaussReg")


def test_lss_trees_records_each_depth_and_adopts_only_a_held_out_win():
    rec = mb.autospec(_product_spread(), "y", seed=0, budget=60, lss_trees=True)
    record = rec.evidence["lss_joint_tree"]
    depths = [row["depth"] for row in record["rows"]]
    assert depths == [3, 6]
    for row in record["rows"]:
        if row["status"] == "scored":
            assert row["passes"] == (row["improvement"] > row["bar"]
                                     and row["improvement"] > 2 * row["se"])
    if record["adopted"] is not None:
        tree = f"maxdepth={record['adopted']}"
        assert all(tree in formula for formula in rec.formula.values())


def test_lss_trees_is_validated():
    with pytest.raises(ValueError, match="lss_trees"):
        mb.autospec(_product_spread(n=200), "y", lss_trees="always")
