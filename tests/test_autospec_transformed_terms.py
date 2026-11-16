"""autospec must route every learner on a transformed column around the strict parser (#320).

The skewed-predictor rule fits ``bbs(log(x))``; the shape probe then compares it
with ``bols(log(x))`` and ``btree(x)``. When the straight line on the log scale
won, autospec sent the rebuilt formula to ``parse_gamboost_formula``, which does
not read transforms, and raised ``cannot parse formula term: 'bols(log(x))'``
after spending its budget. It surfaced on 1 of 8 Mroz labour-force splits.
"""

import numpy as np
import polars as pl
import pytest

import mboostrs as mb
from mboostrs.autospec import _reads_transformed


@pytest.mark.parametrize("term, expected", [
    ("bbs(log(x), df=4.0)", True),
    ("bols(log(x))", True),
    ("bols(sqrt(income))", True),
    ("btree( log(x))", True),
    ("bbs(x, df=4.0)", False),
    ("bols(logx)", False),
    ("bols(catalog)", False),
])
def test_the_predicate_reads_any_learner_on_a_transform(term, expected):
    assert _reads_transformed(term) is expected


def test_autospec_survives_a_linear_log_shape():
    rng = np.random.default_rng(3)
    n = 400
    x = rng.lognormal(0, 1.2, n)
    z = rng.uniform(-2, 2, n)
    frame = pl.DataFrame({"y": 0.8 * np.log(x) + np.sin(z) + rng.normal(0, .5, n),
                          "x": x, "z": z})
    rec = mb.autospec(frame, "y", budget=200, seed=0)
    # This seed is the one where the probe prefers the line on the log scale.
    assert "bols(log(x))" in rec.formula
    model = rec.fit(frame)
    assert np.isfinite(np.asarray(model.predict(frame), dtype=float)).all()
