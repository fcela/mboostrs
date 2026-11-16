"""Small non-negative integer counts are measurements, not levels (#313).

`kid5` (children under five) and `ment` (articles mentored) are counts: a
`brandom` term fits one coefficient per value seen in training and
`predict()` raises on a new one, so a test row with ``kid5 = 4`` could not
be scored. The count reading is structural -- non-negative whole numbers
from 0 or 1 -- and comes before both the few-levels reading and
`label_evidence`, while a genuine label column (region codes that are
neither 0/1-based nor contiguous) keeps its random effect.
"""

import numpy as np
import polars as pl

import mboostrs as mb


def _frame(seed=0, n=500):
    """`kid5` 0..3 and `ment` 0..10 are counts; `region` codes are labels."""
    rng = np.random.default_rng(seed)
    kid5 = rng.integers(0, 4, n)                       # 0..3
    ment = rng.poisson(3.0, n)                        # Poisson counts
    region_index = rng.integers(0, 4, n)
    region = np.array([11, 27, 53, 98])[region_index]
    y = (0.4 * kid5 + 0.05 * ment
         + np.array([0.0, 0.5, -0.4, 0.2])[region_index]
         + rng.normal(size=n))
    return pl.DataFrame({"y": y, "kid5": kid5, "ment": ment, "region": region})


# ─── the count reading ───────────────────────────────────────────────


def test_count_columns_keep_a_measurement_term():
    frame = _frame()
    kinds = {c.name: c for c in mb.classify_columns(frame, response="y")}

    kid5 = kinds["kid5"]
    assert kid5.kind == "continuous"
    assert kid5.term == "bols(kid5)"
    assert "read as a count" in kid5.reason
    assert any(term == "brandom(kid5, df=4.0)"
               and "predict() fails on a value not seen in training" in assumption
               for term, assumption in kid5.alternatives)

    ment = kinds["ment"]
    assert ment.kind == "continuous"
    assert ment.term == "bbs(ment, df=4.0)"
    assert "read as a count" in ment.reason
    assert any(term == "brandom(ment, df=4.0)"
               and "predict() fails on a value not seen in training" in assumption
               for term, assumption in ment.alternatives)

    rhs = mb.suggest_terms(frame, response="y")
    assert "bols(kid5)" in rhs
    assert "bbs(ment, df=4.0)" in rhs


def test_label_codes_that_are_not_counts_keep_brandom():
    frame = _frame()
    kinds = {c.name: c for c in mb.classify_columns(frame, response="y")}
    region = kinds["region"]
    assert region.kind == "categorical"
    assert region.term == "brandom(region, df=4.0)"


def test_a_count_column_is_not_overridden_by_the_label_reading():
    """`label_evidence` reads these `ment` values as labels -- one unrelated
    mean response per value -- and the count reading wins anyway."""
    rng = np.random.default_rng(1)
    n = 1200
    ment = rng.poisson(6.0, n)
    y = rng.normal(size=ment.max() + 1)[ment] + rng.normal(scale=0.5, size=n)
    frame = pl.DataFrame({"y": y, "ment": ment})

    evidence = mb.label_evidence("ment", frame["ment"].to_numpy(),
                                 frame["y"].to_numpy())
    assert evidence is not None and evidence.labels

    kind = mb.classify_columns(frame, response="y")[0]
    assert kind.kind == "continuous"
    assert kind.term == "bbs(ment, df=4.0)"


def test_a_string_column_of_count_like_codes_is_still_levels():
    """The count reading is about numbers: a string column whose labels
    look like counts is a factor whatever its codes say."""
    rng = np.random.default_rng(2)
    n = 300
    frame = pl.DataFrame({
        "y": rng.normal(size=n),
        "kid5": np.array(["0", "1", "2", "3"])[rng.integers(0, 4, n)],
    })
    kind = mb.classify_columns(frame, response="y")[0]
    assert kind.kind == "categorical"
    assert kind.term == "brandom(kid5, df=4.0)"


# ─── the model the reading builds ────────────────────────────────────


def test_autospec_predicts_on_a_count_the_fit_never_saw():
    frame = _frame()
    rec = mb.autospec(frame, "y", budget=60, seed=0)
    assert "brandom(region" in rec.formula
    assert "brandom(kid5" not in rec.formula
    assert "brandom(ment" not in rec.formula

    model = rec.fit(frame)
    rng = np.random.default_rng(3)
    new = pl.DataFrame({
        "kid5": rng.integers(0, 7, 40),       # 4, 5 and 6 unseen in training
        "ment": rng.integers(0, 15, 40),      # counts slightly past the fitted range
        "region": np.array([11, 27, 53, 98])[rng.integers(0, 4, 40)],
    })
    prediction = np.asarray(model.predict(newdata=new), dtype=float)
    assert prediction.shape == (40,)
    assert np.all(np.isfinite(prediction))
