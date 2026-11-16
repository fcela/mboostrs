"""A categorical column's learner is named by the term as written, at any level count.

Before this change a two-level factor under ``bols(f)`` was reported as
``"bols(f)"`` while a factor with three or more levels, under the same
spelling, was reported as ``"f"``. The label now follows the formula term for
every level count, in ``base_learner_names()``, ``coef()``, ``selected`` and
the LSS models.
"""

import warnings

import numpy as np
import polars as pl
import pytest

import mboostrs as mb
from mboostrs.transforms import factor_term_labels


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(7)
    n = 300
    x = rng.uniform(0, 1, n)
    two = rng.choice(["a", "b"], n)
    three = rng.choice(["a", "b", "c"], n)
    y = np.sin(2 * x) + (two == "b") * 0.5 + (three == "c") * 0.8 + rng.normal(0, 0.2, n)
    return pl.DataFrame({"x": x, "two": two, "three": three, "y": y})


def _fit(formula, frame, mstop=20):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mb.gamboost(formula, frame, control=mb.boost_control(mstop=mstop))


@pytest.mark.parametrize("rhs, expected", [
    ("bbs(x, df=4) + bols(two)", "bols(two)"),
    ("bbs(x, df=4) + bols(three)", "bols(three)"),
    ("bbs(x, df=4) + brandom(three)", "brandom(three)"),
    ("bbs(x, df=4) + brandom(two)", "brandom(two)"),
    ("bbs(x, df=4) + three", "three"),
    ("bbs(x, df=4) + two", "two"),
    ("bbs(x, df=4) + bols(three, df=3)", "bols(three, df=3)"),
])
def test_factor_learner_is_named_by_the_term_as_written(frame, rhs, expected):
    model = _fit(f"y ~ {rhs}", frame)
    assert expected in model.base_learner_names()
    assert "x" in model.base_learner_names()


def test_two_and_three_level_bols_are_named_alike(frame):
    two = _fit("y ~ bbs(x, df=4) + bols(two)", frame).base_learner_names()
    three = _fit("y ~ bbs(x, df=4) + bols(three)", frame).base_learner_names()
    assert two[1] == "bols(two)"
    assert three[1] == "bols(three)"


def test_numeric_learners_keep_their_column_name(frame):
    model = _fit("y ~ bbs(x, df=4) + bols(three)", frame)
    assert model.base_learner_names()[0] == "x"


def test_coef_and_selected_use_the_term_as_written(frame):
    model = _fit("y ~ bbs(x, df=4) + bols(three)", frame, mstop=60)
    coefficients = model.coef()
    assert "bols(three)" in coefficients
    assert "three" not in coefficients
    assert "bols(three)" in set(model.selected)


def test_which_accepts_the_reported_name(frame):
    model = _fit("y ~ bbs(x, df=4) + bols(three)", frame)
    effects = model.partial_effects(which="bols(three)")
    assert effects is not None


def test_bare_name_and_bols_on_three_levels_fit_the_same_model(frame):
    bare = _fit("y ~ bbs(x, df=4) + three", frame)
    explicit = _fit("y ~ bbs(x, df=4) + bols(three)", frame)
    np.testing.assert_allclose(np.asarray(bare.predict(frame)).ravel(),
                               np.asarray(explicit.predict(frame)).ravel(),
                               rtol=0, atol=1e-12)


def test_factor_term_labels_maps_each_categorical_column_to_its_term(frame):
    formula = "y ~ bbs(x, df=4) + bols(two) + brandom(three, lambda=0)"
    assert factor_term_labels(formula, frame) == {
        "two": "bols(two)",
        "three": "brandom(three, lambda=0)",
    }


def test_factor_term_labels_leaves_out_a_column_named_twice(frame):
    # One written term cannot label two learners, so the column keeps its name.
    labels = factor_term_labels("y ~ bols(three) + brandom(three)", frame)
    assert "three" not in labels


def test_factor_term_labels_leaves_out_a_column_under_another_learner(frame):
    labels = factor_term_labels("y ~ bmrf(three) + bols(two)", frame)
    assert "three" not in labels
    assert labels == {"two": "bols(two)"}


def test_factor_term_labels_ignores_numeric_columns(frame):
    assert factor_term_labels("y ~ bols(x)", frame) == {}


def test_lss_factor_learners_are_named_by_the_term_as_written(frame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = mb.gamboostLSS(
            {"mu": "y ~ bbs(x, df=4) + bols(two) + bols(three)",
             "sigma": "y ~ bbs(x, df=4) + bols(three)"},
            frame, families=mb.GaussianLSS(), control=mb.boost_control(mstop=10))
    names = model.base_learner_names()
    assert names["mu"] == ["x", "bols(two)", "bols(three)"]
    assert names["sigma"] == ["x", "bols(three)"]


def test_which_still_takes_the_column_name_of_a_factor_learner(frame):
    # Code written against the old label (`which="three"`) keeps selecting the learner.
    model = _fit("y ~ bbs(x, df=4) + brandom(three)", frame)
    by_column = model.partial_effects(which="three")
    by_label = model.partial_effects(which="brandom(three)")
    assert list(by_column) == list(by_label)
    with pytest.raises(ValueError, match="Unknown base-learner"):
        model.partial_effects(which="nonsense")
