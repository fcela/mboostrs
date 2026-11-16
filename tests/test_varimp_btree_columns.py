"""Regression tests for #509: a joint ``btree``'s reported input columns.

A joint tree is labelled by its first column only (``"x (btree)"``), but it
may split on every column in its spec. ``_learner_columns`` used to parse the
label, so ``varimp()["variable_names"]`` reported one column for a tree that
reads several -- and the benchmark's input-count metric inherited the
undercount. ``variable_names`` must list every candidate column; the new
``split_columns`` key narrows that to the columns the fitted trees actually
split on (from ``btree_splits()``).
"""

import numpy as np

import mboostrs as mb


def _repro_fit():
    rng = np.random.default_rng(509)
    x, z = rng.normal(size=200), rng.normal(size=200)
    return mb.gamboost(
        "y ~ btree(x, z, maxdepth=2, selection=sse)",
        data={"x": x, "z": z, "y": 2 * (x > 0) + 3 * (z > 0)},
        control=mb.boost_control(20, nu=.05),
    )


def test_joint_btree_variable_names_lists_all_candidate_columns():
    imp = _repro_fit().varimp()
    assert len(imp["variable_names"]) == 1
    (columns,) = imp["variable_names"].values()
    assert sorted(columns.split(",")) == ["x", "z"]


def test_joint_btree_split_columns_match_btree_splits():
    model = _repro_fit()
    imp = model.varimp()
    (label,) = imp["split_columns"].keys()
    split_vars = {s["variable"] for s in model.btree_splits()[label]["splits"]}
    assert sorted(imp["split_columns"][label].split(",")) == sorted(split_vars)
    # The repro DGP puts signal on both columns; the fitted trees split on both.
    assert split_vars == {"x", "z"}


def test_split_columns_agree_with_variable_names_for_scalar_learners():
    rng = np.random.default_rng(509)
    x = rng.normal(size=200)
    model = mb.gamboost(
        "y ~ bols(x) + bbs(x, df=4)",
        data={"x": x, "y": 2 * x + rng.normal(size=200)},
        control=mb.boost_control(20, nu=.1),
    )
    imp = model.varimp()
    for label in imp["variable_names"]:
        assert imp["split_columns"][label] == imp["variable_names"][label]


def test_unused_btree_column_not_counted_as_split_input():
    # z carries no signal: the tree may read it, but splits should land on x.
    rng = np.random.default_rng(11)
    x, z = rng.normal(size=400), rng.normal(size=400)
    model = mb.gamboost(
        "y ~ btree(x, z, maxdepth=1, selection=sse)",
        data={"x": x, "z": z, "y": (x > 0).astype(float)},
        control=mb.boost_control(50, nu=.1),
    )
    imp = model.varimp()
    (label,) = imp["variable_names"].keys()
    assert sorted(imp["variable_names"][label].split(",")) == ["x", "z"]
    assert imp["split_columns"][label] == "x"
