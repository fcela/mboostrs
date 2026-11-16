"""Adaptive-Huber cvrisk against mboost (see the Huber parity notes).

mboost's Huber stores the last fitted predictor in its closure and cvrisk
reuses one family object across folds, so each fold's first delta comes from
the previous fold's final fit and its cvrisk depends on the fold order (about
1% here). mboostrs starts every fold afresh, which equals mboost refitted
fold by fold with a fresh Huber().
"""

import numpy as np
import pandas as pd
import pytest

import mboostrs as mb
from conftest import HAS_R, HAS_RPY2


@pytest.mark.r_parity
@pytest.mark.skipif(not (HAS_R and HAS_RPY2), reason="requires R + rpy2")
def test_matches_mboost_refitted_fold_by_fold():
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri, pandas2ri
    from rpy2.robjects.conversion import localconverter

    rng = np.random.default_rng(42)
    n = 300
    d = pd.DataFrame({"x": rng.uniform(-1, 1, n), "z": rng.normal(size=n)})
    d["y"] = np.sin(2 * d.x) + 0.5 * d.z + 0.4 * rng.standard_t(3, n)
    folds = mb.cv_weights(n, "kfold", 4, seed=5)
    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv["d"] = ro.conversion.py2rpy(d)
    with localconverter(ro.default_converter + numpy2ri.converter):
        ro.globalenv["f"] = folds
    fresh = np.asarray(ro.r(
        "suppressMessages(library(mboost)); r <- sapply(1:ncol(f), function(j) { "
        "m <- mboost::gamboost(y ~ bbs(x) + bols(z), data = d, family = mboost::Huber(), "
        "weights = f[, j], control = mboost::boost_control(mstop = 40, risk = 'oobag')); "
        "mboost::risk(m) / sum(f[, j] == 0) }); rowMeans(r)"))
    fit = mb.gamboost("y ~ bbs(x) + bols(z)", data=d, family=mb.Huber(),
                      control=mb.boost_control(40))
    ours = np.asarray(fit.cvrisk(folds=folds).risk_matrix).mean(axis=0)
    np.testing.assert_allclose(ours, fresh, atol=1e-10)
    assert "adaptive huber cvrisk starts each fold afresh" in fit.parity_notes()["name"].to_list()
