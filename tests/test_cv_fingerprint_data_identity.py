"""CV results pair only when they share folds and data (issue #63).

The fold fingerprint used to digest the fold matrix, evaluation mask and
weights only, so the same folds applied to a different response paired and
`compare_cv` reported a paired comparison between unrelated data sets.
"""

import numpy as np
import pytest

import mboostrs as mb
from mboostrs.cvrisk import compare_cv


def _data(seed, binary=False):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=120)
    y = x + rng.normal(size=120)
    if binary:
        y = (y > 0).astype(float)
    return {"x": x, "z": rng.normal(size=120), "y": y}


FOLDS = mb.cv_weights(120, type="kfold", B=5, seed=3)


def test_same_folds_on_different_responses_do_not_pair():
    a = mb.glmboost("y ~ x", data=_data(0), control=mb.boost_control(30)).cvrisk(folds=FOLDS)
    b = mb.glmboost("y ~ x", data=_data(1), control=mb.boost_control(30)).cvrisk(folds=FOLDS)
    assert a.fold_fingerprint != b.fold_fingerprint
    with pytest.raises(ValueError, match="different folds"):
        compare_cv(a, b, mstop=20)


@pytest.mark.parametrize("binary", [False, True])
def test_glm_and_gam_on_the_same_data_still_pair(binary):
    data = _data(0, binary=binary)
    family = mb.Binomial() if binary else mb.Gaussian()
    a = mb.glmboost("y ~ x + z", data=data, family=family,
                    control=mb.boost_control(30)).cvrisk(folds=FOLDS)
    b = mb.gamboost("y ~ bbs(x) + bbs(z)", data=data, family=family,
                    control=mb.boost_control(30)).cvrisk(folds=FOLDS)
    assert a.fold_fingerprint == b.fold_fingerprint
    assert compare_cv(a, b, mstop=20).inferential
