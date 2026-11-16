"""`pair_term_for` reads a string column as labels when given the frame (#336).

Without `levels=`, a column described by neither `levels` nor `groups` was
assumed continuous, so ``pair_term_for("x", "b")`` built a spline surface
over a string column's factor codes and the fit died with
``could not convert string to float``. With ``data=`` the frame itself types
each margin: two levels switch a smooth on, more get a random slope.
"""

import numpy as np
import polars as pl

import mboostrs as mb


def _frame(levels, seed=0, n=300):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return pl.DataFrame({
        "y": 1.5 * x + rng.normal(size=n),
        "x": x,
        "b": rng.choice(list(levels), n),
    })


def test_a_two_level_string_column_switches_a_smooth_on():
    term = mb.pair_term_for("x", "b", data=_frame("lh"))
    assert term == "bbs(x, by=b)"


def test_a_many_level_string_column_gets_a_random_slope():
    term = mb.pair_term_for("x", "b", data=_frame("news"))
    assert "brandom(b" in term
    assert "bbs(b" not in term


def test_the_typed_pair_terms_fit_the_frame_they_came_from():
    for levels in ("lh", "news"):
        frame = _frame(levels)
        term = mb.pair_term_for("x", "b", data=frame)
        model = mb.gamboost(f"y ~ {term}", data=frame,
                            control=mb.boost_control(30, 0.1))
        assert np.all(np.isfinite(np.asarray(model.predict(), dtype=float)))


def test_levels_already_given_take_precedence_over_the_frame():
    frame = _frame("news")
    term = mb.pair_term_for("x", "b", levels={"x": 300, "b": 2}, data=frame)
    assert term == "bbs(x, by=b)"


def test_without_data_the_caller_still_describes_the_columns():
    """`data=None` keeps the old contract: an undescribed column is assumed
    continuous, so the caller must pass `levels=`/`groups=` for a label
    column or build the term from a frame with `data=`."""
    assert mb.pair_term_for("x", "b") == (
        "bbs(x, df=2.5, knots=5) %X% bbs(b, df=2.5, knots=5)")
