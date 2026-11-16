"""Discover does not nominate a pair of two brandom factors (issue #447, item 7).

A factor with more than ten levels is a brandom term. A pair of two such terms,
or of one with a two-valued label column, is a brandom tensor and is not
nominated. A factor paired with a numeric column is still nominated.
"""

import numpy as np
import polars as pl

import mboostrs as mb
from mboostrs.discover import FACTOR_PAIR_NOTE
from mboostrs.explore import pair_term_for


def _factor_frame(seed=0, n=600):
    rng = np.random.default_rng(seed)
    g = rng.choice([f"R{i:02d}" for i in range(12)], n)
    b = rng.choice(["yes", "no"], n)
    x = rng.normal(size=n)
    signed = np.where(b == "yes", 1.0, -1.0)
    effect = signed * np.isin(g, ["R00", "R01", "R02"])
    y = effect + 0.5 * x + rng.normal(0, 0.3, n)
    return pl.DataFrame({"g": g, "b": b, "x": x, "y": y})


def _numeric_pair_frame(seed=0, n=600):
    rng = np.random.default_rng(seed)
    g = rng.choice([f"R{i:02d}" for i in range(12)], n)
    x = rng.normal(size=n)
    slope = np.where(np.isin(g, ["R00", "R01", "R02"]), 1.5, 0.0)
    y = slope * x + 0.3 * x + rng.normal(0, 0.3, n)
    return pl.DataFrame({"g": g, "x": x, "y": y})


def test_the_factor_pair_term_is_a_brandom_tensor():
    # The screen nominates this pair; the test is about discover's handling of it.
    term = pair_term_for("g", "b", data=_factor_frame())

    assert term == "brandom(g) %X% brandom(b)"


def test_discover_does_not_nominate_a_pair_of_two_brandom_factors():
    result = mb.discover(_factor_frame(), "y", seed=0, spread=False)

    candidates = [e["candidate"] for e in result.log
                  if e["stage"] == "interactions" and e["candidate"]]
    assert all("brandom(g) %X% brandom(b)" not in c for c in candidates)
    assert "brandom(g) %X% brandom(b)" not in result.formula


def test_discover_report_states_that_factor_pairs_are_not_searched():
    result = mb.discover(_factor_frame(), "y", seed=0, spread=False)

    notes = [e["reason"] for e in result.log
             if e["stage"] == "interactions" and e["candidate"] is None]
    assert FACTOR_PAIR_NOTE in notes
    assert "Interactions between two brandom factors are not searched" in FACTOR_PAIR_NOTE
    assert "bols(g, by=b)" in FACTOR_PAIR_NOTE


def test_a_factor_paired_with_a_numeric_column_is_still_nominated():
    result = mb.discover(_numeric_pair_frame(), "y", seed=0, spread=False)

    pair_entries = [e for e in result.log
                    if e["stage"] == "interactions" and e["candidate"]
                    and "x" in e["candidate"] and "g" in e["candidate"]]
    assert pair_entries, "the factor-by-numeric pair is tried in the interaction stage"
    assert "brandom(g, by=x)" in pair_entries[0]["candidate"]


def test_the_hand_written_factor_cell_term_fits_and_predicts():
    frame = _factor_frame()
    model = mb.gamboost("y ~ bols(x) + bols(g) + bols(b) + bols(g, by=b)", frame,
                        family=mb.Gaussian(), control=mb.boost_control(200, 0.1))

    assert np.all(np.isfinite(np.asarray(model.predict(frame.head(5)))))
