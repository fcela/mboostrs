"""A Polars Enum column is a factor, encoded the way a String or Categorical is.

An Enum used to fail in the formula encoder (``bols(f)`` raised "not numbers")
because the dtype name check in ``transforms._is_categorical`` compared the
printed dtype, ``Enum(categories=[...])``, against a bare name.
"""

import numpy as np
import polars as pl
import pytest

import mboostrs as mb
from mboostrs import transforms

LEVELS = ["a", "b", "c", "d"]


@pytest.fixture(scope="module")
def frames():
    rng = np.random.default_rng(0)
    n = 200
    f = rng.choice(LEVELS, n)
    x = rng.normal(size=n)
    base = pl.DataFrame({"y": x + (f == "a"), "x1": x, "f": f})
    enum = base.with_columns(pl.col("f").cast(pl.Enum(LEVELS)))
    categorical = base.with_columns(pl.col("f").cast(pl.Categorical))
    newdata = pl.DataFrame({"x1": np.linspace(-2, 2, 25),
                            "f": (["a", "b", "c", "d"] * 7)[:25]})
    return {
        "string": base,
        "enum": enum,
        "categorical": categorical,
        "newdata_string": newdata,
        "newdata_enum": newdata.with_columns(pl.col("f").cast(pl.Enum(LEVELS))),
    }


CONTROL = mb.boost_control(30, 0.1)


def _assert_same_coef(a, b):
    assert list(a) == list(b)
    for key in a:
        np.testing.assert_allclose(np.asarray(a[key]), np.asarray(b[key]),
                                   rtol=1e-12, atol=1e-12, err_msg=key)


def test_gamboost_enum_matches_string(frames):
    formula = "y ~ bbs(x1) + bols(f)"
    fit_s = mb.gamboost(formula, data=frames["string"], control=CONTROL)
    fit_e = mb.gamboost(formula, data=frames["enum"], control=CONTROL)
    _assert_same_coef(fit_s.coef(), fit_e.coef())
    np.testing.assert_allclose(fit_s.fitted(), fit_e.fitted(), rtol=1e-12)
    np.testing.assert_allclose(
        fit_s.predict(newdata=frames["newdata_string"]),
        fit_e.predict(newdata=frames["newdata_enum"]), rtol=1e-12)


def test_glmboost_enum_matches_string(frames):
    formula = "y ~ x1 + f"
    fit_s = mb.glmboost(formula, data=frames["string"], control=CONTROL)
    fit_e = mb.glmboost(formula, data=frames["enum"], control=CONTROL)
    _assert_same_coef(fit_s.coef(), fit_e.coef())
    np.testing.assert_allclose(
        fit_s.predict(newdata=frames["newdata_string"]),
        fit_e.predict(newdata=frames["newdata_enum"]), rtol=1e-12)


def test_enum_matches_categorical(frames):
    formula = "y ~ bbs(x1) + bols(f)"
    fit_c = mb.gamboost(formula, data=frames["categorical"], control=CONTROL)
    fit_e = mb.gamboost(formula, data=frames["enum"], control=CONTROL)
    _assert_same_coef(fit_c.coef(), fit_e.coef())


def test_enum_levels_follow_declared_order(frames):
    reordered = frames["string"].with_columns(
        pl.col("f").cast(pl.Enum(["d", "c", "b", "a"])))
    assert transforms._levels(reordered, "f") == ("d", "c", "b", "a")
    # A String or Categorical column keeps the sorted order.
    assert transforms._levels(frames["string"], "f") == tuple(LEVELS)
    assert transforms._levels(frames["categorical"], "f") == tuple(LEVELS)


def test_enum_declared_unobserved_level_is_not_a_fitted_level(frames):
    # "e" is declared in the Enum but absent from the training rows: it adds
    # no level, so the fit is the same as the four-level String fit.
    declared = frames["string"].with_columns(
        pl.col("f").cast(pl.Enum(LEVELS + ["e"])))
    assert transforms._levels(declared, "f") == tuple(LEVELS)
    fit_s = mb.gamboost("y ~ bbs(x1) + bols(f)", data=frames["string"], control=CONTROL)
    fit_d = mb.gamboost("y ~ bbs(x1) + bols(f)", data=declared, control=CONTROL)
    _assert_same_coef(fit_s.coef(), fit_d.coef())


def test_enum_unseen_level_refused_like_categorical(frames):
    formula = "y ~ bbs(x1) + bols(f)"
    fit_s = mb.gamboost(formula, data=frames["string"], control=CONTROL)
    fit_e = mb.gamboost(
        formula,
        data=frames["string"].with_columns(pl.col("f").cast(pl.Enum(LEVELS + ["e"]))),
        control=CONTROL)
    unseen_string = pl.DataFrame({"x1": [0.0, 1.0], "f": ["a", "e"]})
    unseen_enum = unseen_string.with_columns(pl.col("f").cast(pl.Enum(LEVELS + ["e"])))
    unseen_categorical = unseen_string.with_columns(pl.col("f").cast(pl.Categorical))

    with pytest.raises(ValueError, match="not present when the model was fitted") as cat_err:
        fit_s.predict(newdata=unseen_categorical)
    with pytest.raises(ValueError, match="not present when the model was fitted") as enum_err:
        fit_e.predict(newdata=unseen_enum)
    assert str(enum_err.value) == str(cat_err.value)
