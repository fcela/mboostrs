"""A refit-mode cross-validation that reuses its fold fits gives the fresh result.

The conformal tuning (#578) calls cvrisk(preprocessing="refit") at growing grid
ends, passing one cache. Each call must match a fresh cross-validation at the
same grid end, because a boosting path is prefix-consistent.
"""

import numpy as np

import mboostrs as mb
from mboostrs.cvrisk import cv_weights
from mboostrs.families import Quantile


def _model(n=400, seed=2):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    y = np.sin(x) + (0.3 + 0.5 * np.abs(x)) * rng.standard_normal(n)
    return mb.gamboost("y ~ bbs(x, df=4)", {"x": x, "y": y}, family=Quantile(tau=0.1))


def test_cached_refit_cvrisk_matches_a_fresh_one_at_each_grid_end():
    model = _model()
    folds = cv_weights(400, "kfold", 4, seed=3)
    cache = {}
    probe = model.continue_to(60)
    first = probe.cvrisk(folds=folds, preprocessing="refit", _path_cache=cache)
    for end in (120, 240):
        probe = probe.continue_to(end)
        cached = probe.cvrisk(folds=folds, preprocessing="refit", _path_cache=cache)
        fresh = model.continue_to(end).cvrisk(folds=folds, preprocessing="refit")
        np.testing.assert_array_equal(np.asarray(cached.aggregate), np.asarray(fresh.aggregate))
        assert cached.optimal_mstop == fresh.optimal_mstop
    assert len(np.asarray(first.aggregate)) == 61
