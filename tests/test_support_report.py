"""support_report(): joint support of new rows and unseen levels (issue #67).

A row inside every predictor's training range can lie where the training
data has no rows -- a high x1 with a low x2 when the two move together. The
report gives each new row's Mahalanobis distance from the training rows, the
share of training rows at least that far, and unseen categorical levels.
"""

import numpy as np
import pytest

import mboostrs as mb
from conftest import HAS_R, HAS_RPY2


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 1, n)
    x2 = x1 + rng.normal(0, 0.03, n)
    g = rng.choice(list("abc"), n)
    return {"x1": x1, "x2": x2, "g": g, "y": x1 + x2 + (g == "b") + rng.normal(0, 0.2, n)}


NEW = {"x1": np.array([0.5, 0.9]), "x2": np.array([0.5, 0.1]), "g": np.array(["a", "z"])}


@pytest.mark.parametrize("fit", ["gamboost", "glmboost"])
def test_inside_the_marginals_but_off_the_cloud(fit):
    d = _data()
    model = (mb.gamboost("y ~ bbs(x1) + bbs(x2) + bols(g)", data=d) if fit == "gamboost"
             else mb.glmboost("y ~ x1 + x2 + g", data=d))
    first, second = model.support_report(NEW, output="records")
    assert not first["outside_joint_support"] and first["unseen_levels"] == ""
    assert second["outside_joint_support"] and second["training_share_beyond"] == 0.0
    assert second["unseen_levels"] == "g=z"
    # Inside both marginal ranges, so the range report finds nothing.
    if fit == "gamboost":
        assert all(r["n_outside"] == 0 for r in model.range_report(
            {"x1": NEW["x1"], "x2": NEW["x2"], "g": np.array(["a", "a"])}, output="records"))


def test_the_share_matches_a_direct_computation():
    d = _data()
    model = mb.glmboost("y ~ x1 + x2", data=d)
    rows = model.support_report(NEW, output="records")
    X = np.column_stack([d["x1"], d["x2"]])
    centre, inv = X.mean(axis=0), np.linalg.inv(np.cov(X, rowvar=False))
    train_d = np.einsum("ij,jk,ik->i", X - centre, inv, X - centre)
    for row, x in zip(rows, np.column_stack([NEW["x1"], NEW["x2"]]), strict=True):
        dist = (x - centre) @ inv @ (x - centre)
        assert row["distance_sq"] == pytest.approx(dist, rel=1e-9)
        assert row["training_share_beyond"] == pytest.approx(np.mean(train_d >= dist))


@pytest.mark.r_parity
@pytest.mark.skipif(not (HAS_R and HAS_RPY2), reason="requires R + rpy2")
def test_r_and_python_agree():
    import pandas as pd
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    d = pd.DataFrame(_data())
    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv["d"] = ro.conversion.py2rpy(d)
        ro.globalenv["nd"] = ro.conversion.py2rpy(pd.DataFrame(NEW))
    r = ro.r("r <- mboostrs::support_report(mboostrs::glmboost(y ~ x1 + x2 + g, data = d), nd); "
             "list(r$distance_sq, r$training_share_beyond)")
    rows = mb.glmboost("y ~ x1 + x2 + g", data=dict(_data())).support_report(NEW, output="records")
    np.testing.assert_allclose(np.asarray(r[0]), [row["distance_sq"] for row in rows], rtol=1e-9)
    np.testing.assert_allclose(np.asarray(r[1]), [row["training_share_beyond"] for row in rows])
