"""A model returned by `mb.fit` predicts an unseen group code as the average group.

`mb.fit` reads a whole-number column with few levels, or one whose values
behave as labels, as a grouping and fits it with `brandom`. On new rows such a
column routinely carries codes the training rows did not (a new store, a new
source video), and a model fitted unattended used to refuse them. Its
`predict` now uses the model's policy, `"average"`, recorded in the trail; an
explicit `unseen_levels="error"` still refuses.
"""

import numpy as np
import polars as pl
import pytest

import mboostrs as mb


def _grouped(seed=0, n=900):
    rng = np.random.default_rng(seed)
    code = rng.integers(0, 6, n)
    x = rng.uniform(-1, 1, n)
    effect = np.array([0.0, 1.0, -1.0, 0.5, 2.0, -0.5])
    y = np.sin(2 * x) + effect[code] + 0.3 * rng.standard_normal(n)
    labels = np.array([f"store_{c}" for c in code])
    return pl.DataFrame({"x": x, "code": labels, "y": y, "level": code})


def test_fit_predicts_an_unseen_code_as_the_average_group():
    data = _grouped()
    train = data.filter(pl.col("level") < 5).drop("level")
    fit = mb.fit(train, "y", seed=0)
    assert "brandom(code" in fit.trail.final["formula"]
    assert fit.trail.final["unseen_levels"] == "average"
    new = data.filter(pl.col("level") == 5).drop("level")
    predicted = np.asarray(fit.predict(new), dtype=float)
    assert np.all(np.isfinite(predicted))
    with pytest.raises(Exception, match="unknown|not observed|unseen"):
        fit.predict(new, unseen_levels="error")


def test_a_model_fitted_directly_still_refuses_an_unseen_code():
    data = _grouped()
    train = data.filter(pl.col("level") < 5).drop("level")
    model = mb.gamboost("y ~ bbs(x) + brandom(code)", train, family=mb.Gaussian())
    with pytest.raises(Exception, match="unknown|not observed|unseen"):
        model.predict(data.filter(pl.col("level") == 5).drop("level"))


def test_a_missing_category_on_new_rows_is_predicted_as_the_average_group():
    # OpenML adult: a categorical column with no missing value in training and
    # None in test rows. The unseen-level path sorted the new levels, and a
    # None does not compare with strings.
    data = _grouped()
    train = data.filter(pl.col("level") < 5).drop("level")
    model = mb.gamboost("y ~ bbs(x) + brandom(code)", train, family=mb.Gaussian())
    new = data.filter(pl.col("level") == 5).drop("level").with_columns(
        pl.when(pl.col("x") > 0).then(None).otherwise(pl.col("code")).alias("code"))
    predicted = np.asarray(model.predict(new, unseen_levels="average"), dtype=float)
    assert np.all(np.isfinite(predicted))
    with pytest.raises(Exception, match="None"):
        model.predict(new)
