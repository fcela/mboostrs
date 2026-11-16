"""`extract("lambda")` reports the smoothing parameter of every block learner.

A learner specified by `df` calibrates its lambda; R's
`extract(fit, what = "lambda")` returns that value, and so does mboostrs. The
R agreement is pinned in `test_inference_parity/test_extract_coef_parity.py`;
this file checks that no block learner reports None without R.
"""

import math

import numpy as np
import pytest

import mboostrs as mb


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 200
    return {"x": rng.uniform(size=n), "w": rng.uniform(size=n),
            "g": rng.integers(0, 6, n), "y": rng.normal(size=n),
            "u": np.zeros(n)}


def _neighbours(k):
    K = np.zeros((k, k))
    for i in range(k - 1):
        K[i, i + 1] = K[i + 1, i] = -1.0
    np.fill_diagonal(K, -K.sum(axis=1))
    return K


@pytest.mark.parametrize("term", [
    "bbs(x, df=4)",
    "bns(x, df=4)",
    "brandom(g, df=4)",
    "bkernel(x, df=4)",
    "bmrf(g, K=K_regions, df=3)",
    "buser(u, X=DX, K=DK, df=4)",
    "bbs(x, df=3) %X% bbs(w, df=3)",
])
def test_every_block_learner_reports_its_lambda(data, term):
    basis = mb.learner_design("bbs(x, df=4)", data)
    fit = mb.gamboost(f"y ~ {term}", data=data,
                      env={"K_regions": _neighbours(6), "DX": basis.design,
                           "DK": basis.penalty},
                      control=mb.boost_control(mstop=20))
    (value,) = fit.extract("lambda").values()
    assert value is not None and math.isfinite(value) and value > 0


def test_no_parity_note_claims_the_lambda_is_unavailable():
    from mboostrs._parity_notes import LEARNER_NOTES

    names = [note.name for notes in LEARNER_NOTES.values() for note in notes]
    assert not any("extract('lambda')" in name for name in names)
