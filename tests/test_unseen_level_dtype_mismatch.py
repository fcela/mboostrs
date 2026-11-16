"""A factor column whose levels differ from the fitted ones only in dtype is
refused under every ``unseen_levels=`` policy (issue #334).

``unseen_levels="average"`` is for a level the fit never saw — a new group.
A column fitted as strings '1', '2', '3' that arrives as Int64 1, 2, 3 has
every level unseen and no new group among them, so the policy silently
predicted every row at the average group. The mismatch is now refused with
the cast that fixes it, under "error" (whose generic message named the
levels but not the dtype) and under "average"/"zero_effect" alike; a
genuinely new level keeps the policy.
"""

import re

import numpy as np
import polars as pl
import pytest

import mboostrs as mb

CONTROL = mb.boost_control(60, 0.1)
HINT = re.escape("cast it, e.g. pl.col('f').cast(pl.Utf8), before predicting")


def _string_frame(seed=0, n=60):
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "y": rng.normal(size=n),
        "f": pl.Series(["1", "2", "3"] * (n // 3), dtype=pl.Utf8),
        "x": rng.normal(size=n),
    })


def _int_newdata():
    return pl.DataFrame({"f": pl.Series([1, 2, 3], dtype=pl.Int64),
                         "x": [0.0, 0.1, 0.2]})


@pytest.mark.parametrize("policy", [None, "average", "zero_effect"])
def test_string_fitted_int_arriving_is_refused_under_every_policy(policy):
    """The repro of #334: under "average" every row was silently predicted
    at the average group instead of refusing."""
    fit = mb.gamboost("y ~ brandom(f) + bbs(x)", data=_string_frame(),
                      control=CONTROL)
    options = {} if policy is None else {"unseen_levels": policy}
    with pytest.raises(ValueError, match=HINT) as excinfo:
        fit.predict(_int_newdata(), **options)
    message = str(excinfo.value)
    assert "column 'f' was fitted as strings ['1', '2', '3']" in message
    assert "arrives as Int64 [1, 2, 3]" in message
    assert isinstance(excinfo.value, mb.MboostrsValueError)
    assert getattr(excinfo.value, "code", None) == "unseen_level"


@pytest.mark.parametrize("policy", [None, "average", "zero_effect"])
def test_int_fitted_string_arriving_is_refused_under_every_policy(policy):
    """An object column of Python ints is categorical, so the fit records
    integer levels; the same labels as strings are then every level unseen
    and, under "average", silently averaged."""
    rng = np.random.default_rng(1)
    n = 60
    data = {"y": rng.normal(size=n),
            "f": np.array([1, 2, 3] * (n // 3), dtype=object),
            "x": rng.normal(size=n)}
    fit = mb.gamboost("y ~ brandom(f) + bbs(x)", data=data, control=CONTROL)
    newdata = {"f": np.array(["1", "2", "3"], dtype=object),
               "x": np.array([0.0, 0.1, 0.2])}
    options = {} if policy is None else {"unseen_levels": policy}
    with pytest.raises(
            ValueError,
            match=re.escape("cast it, e.g. pl.col('f').cast(pl.Int64), "
                             "before predicting")) as excinfo:
        fit.predict(newdata, **options)
    message = str(excinfo.value)
    assert "column 'f' was fitted as integers [1, 2, 3]" in message
    assert "arrives as strings ['1', '2', '3']" in message
    assert getattr(excinfo.value, "code", None) == "unseen_level"


def test_a_genuinely_new_level_still_averages():
    """A new string level among the fitted ones is what the policy is for:
    it averages under "average" and only the default refuses it."""
    fit = mb.gamboost("y ~ brandom(f) + bbs(x)", data=_string_frame(seed=2),
                      control=CONTROL)
    newdata = pl.DataFrame({"f": ["1", "2", "3", "4"],
                            "x": [0.5, 0.5, 0.5, 0.5]})
    predicted = np.asarray(fit.predict(newdata, unseen_levels="average"))
    assert np.isfinite(predicted).all()
    # The new group predicts at the mean of the fitted levels' predictions
    # (same x in every row), the documented meaning of "average".
    assert predicted[3] == pytest.approx(np.mean(predicted[:3]), abs=1e-9)
    per_level = np.asarray([
        float(np.ravel(fit.predict(pl.DataFrame({"f": [level], "x": [0.5]}),
                                     unseen_levels="average"))[0])
        for level in ("1", "2", "3")])
    np.testing.assert_allclose(predicted[:3], per_level, rtol=0, atol=1e-10)
    with pytest.raises(ValueError, match="'4'"):
        fit.predict(newdata)


def test_the_cast_the_error_names_restores_the_prediction():
    fit = mb.gamboost("y ~ brandom(f) + bbs(x)", data=_string_frame(seed=3),
                      control=CONTROL)
    casted = _int_newdata().with_columns(pl.col("f").cast(pl.Utf8))
    strings = pl.DataFrame({"f": ["1", "2", "3"], "x": [0.0, 0.1, 0.2]})
    np.testing.assert_allclose(np.asarray(fit.predict(casted)),
                               np.asarray(fit.predict(strings)),
                               rtol=0, atol=1e-12)


@pytest.mark.parametrize("policy", [None, "average"])
def test_gamboostlss_predict_refuses_the_mismatch_too(policy):
    fit = mb.gamboostLSS({"mu": "y ~ brandom(f) + bbs(x)", "sigma": "y ~ bols(x)"},
                         data=_string_frame(seed=4), families=mb.GaussianLSS(),
                         control=mb.boost_control(30))
    options = {} if policy is None else {"unseen_levels": policy}
    with pytest.raises(ValueError, match=HINT):
        fit.predict(_int_newdata(), type="link", **options)
