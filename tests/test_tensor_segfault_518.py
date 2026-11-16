"""Regression test for #518: tensor-product term segfault during learner preparation.

Fitting `bbs(x1) %X% bbs(x2)` segfaulted because learner preparation runs on
rayon's global pool while faer's divide-and-conquer SVD spawned nested
parallelism on the 2MB worker stacks — a stack overflow, not a tensor-layout
bug. The fix makes faer run serially on rayon worker threads
(`setup_parallelism` in crates/mboostrs-core/src/learners.rs). This test fits
the issue's repro model; a segfault here fails the test process outright.
"""

import numpy as np
import polars as pl

import mboostrs as mb
from mboostrs._autospec import _control
from mboostrs.autospec import _infer_family


def test_tensor_product_bbs_does_not_segfault():
    rng = np.random.default_rng(3)
    n = 900
    x1 = rng.uniform(-2, 2, n)
    x2 = rng.uniform(-2, 2, n)
    g = rng.choice(["a", "b", "c"], n)
    y = (
        np.sin(2 * x1)
        + 1.5 * x1 * x2
        + 0.9 * x2 * (g == "a")
        + rng.normal(0, 0.3, n)
    )
    frame = pl.DataFrame({"y": y, "x1": x1, "x2": x2, "g": g})

    fit = mb.gamboost(
        "y ~ bbs(x1, df=4) + bbs(x2, df=4) + bols(g) + bbs(x1) %X% bbs(x2)",
        data=frame,
        family=_infer_family(frame, "y"),
        control=_control(5000, 0.1),
    )

    preds = np.asarray(fit.predict(), dtype=float)
    assert preds.shape == (n,)
    assert np.all(np.isfinite(preds))
