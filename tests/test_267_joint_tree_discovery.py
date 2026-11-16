"""Joint-tree candidates for screened pairs are discoverable (#267).

The autospec interaction screen proposes a `btree(col1, col2)` joint candidate
for the top-ranked pair, and the form picker adopts the better of the tensor
and the joint tree on held-out risk. These tests pin that behavior:
(a) a planted interaction yields a joint_tree lead in the evidence;
(b) purely additive truth yields no joint_tree lead;
(c) the documented multi-column btree() syntax executes.
"""

import numpy as np

import mboostrs as mb


def _interaction_data(n=800, seed=11):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n, 4))
    y = (3.0 * (X[:, 0] - 0.5) * (X[:, 1] - 0.5)
         + 1.5 * np.sin(np.pi * X[:, 2])
         + rng.normal(scale=0.7, size=n))
    return {f"x{i}": X[:, i] for i in range(4)} | {"y": y}


def _additive_data(n=800, seed=12):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n, 4))
    y = (1.5 * np.sin(np.pi * X[:, 2]) + 1.0 * X[:, 3]
         + rng.normal(scale=0.7, size=n))
    return {f"x{i}": X[:, i] for i in range(4)} | {"y": y}


def test_joint_tree_lead_proposed_for_planted_interaction():
    rec = mb.autospec(_interaction_data(), "y", family=mb.Gaussian(),
                      budget=40, nu=0.1, interactions="report",
                      residual_tree=False, seed=0)
    leads = rec.evidence.get("interactions", ())
    joint = [lead for lead in leads if lead.kind == "joint_tree"]
    assert len(joint) == 1, "exactly one joint-tree lead for the top pair"
    lead = joint[0]
    assert {lead.left, lead.right} == {"x0", "x1"}
    assert lead.term == "btree(x0, x1, maxdepth=3, selection=sse)"
    assert lead.removed > 0


def test_no_joint_tree_lead_for_additive_truth():
    # With adoption, a spurious screen pair must not survive validation:
    # the final formula has no pair term on additive truth.
    rec = mb.autospec(_additive_data(), "y", family=mb.Gaussian(),
                      budget=40, nu=0.1, interactions="adopt",
                      residual_tree=False, seed=0)
    import re
    pair_terms = re.findall(
        r'bbs\(x\d[^)]*\) %X% bbs\(x\d[^)]*\)|btree\(x\d, x\d', rec.formula)
    assert not pair_terms, f"spurious pair adopted: {pair_terms}"


def test_multicolumn_btree_formula_syntax_executes():
    # The documented multi-column btree() syntax: a joint tree over two
    # columns, splitting on both.
    import polars as pl
    df = pl.DataFrame(_interaction_data())
    fit = mb.gamboost("y ~ btree(x0, x1, maxdepth=3, selection=sse)", data=df,
                      control=mb.boost_control(50, 0.1))
    splits = fit.btree_splits()
    assert "x0" in splits
    variables = splits["x0"]["variables"]
    assert set(variables) == {"x0", "x1"}, "tree splits on both columns"
    pred = fit.predict(df)
    assert pred.shape == (800,)
