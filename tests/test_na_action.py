"""Missing-value policy: na_action='error' (default) vs 'omit'.

The default is fail-closed (verified by the C.1 probe): any non-finite input
raises, naming the column and row. `na_action='omit'` drops rows with a missing
value in a model variable and records how many.
"""

import numpy as np
import polars as pl
import pytest

import mboostrs as mb


def _data_with_holes(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = 1.5 * x - 0.8 * z + rng.normal(scale=0.2, size=n)
    x = x.copy()
    x[5] = np.nan
    y = y.copy()
    y[10] = np.inf
    return {"x": x, "z": z, "y": y}


def test_default_still_raises_on_missing():
    data = _data_with_holes()
    with pytest.raises(ValueError, match="contains"):
        mb.glmboost("y ~ x + z", data=data, family=mb.Gaussian(),
                    control=mb.boost_control(20, 0.1))


@pytest.mark.parametrize("engine,formula", [
    ("glmboost", "y ~ x + z"),
    ("gamboost", "y ~ bbs(x, df=4) + bols(z)"),
])
def test_omit_drops_exactly_the_bad_rows(engine, formula):
    data = _data_with_holes()
    fit = getattr(mb, engine)(
        formula, data=data, family=mb.Gaussian(),
        control=mb.boost_control(50, 0.1), na_action="omit",
    )
    assert fit._na_dropped == [5, 10]
    assert len(fit.fitted_) == 198
    assert np.all(np.isfinite(fit.fitted_))


def test_omit_equals_a_manual_pre_filter():
    rng = np.random.default_rng(2)
    n = 200
    x = rng.normal(size=n)
    x[3] = np.nan
    y = 2.0 * x + rng.normal(scale=0.1, size=n)
    keep = np.isfinite(x)

    omitted = mb.glmboost("y ~ x", data={"x": x, "y": y}, family=mb.Gaussian(),
                          control=mb.boost_control(80, 0.1), na_action="omit")
    manual = mb.glmboost("y ~ x", data={"x": x[keep], "y": y[keep]},
                         family=mb.Gaussian(), control=mb.boost_control(80, 0.1))
    np.testing.assert_array_equal(omitted.fitted_, manual.fitted_)


def test_unused_column_with_missing_values_does_not_force_drops():
    rng = np.random.default_rng(1)
    n = 100
    data = {
        "x": rng.normal(size=n),
        "z": rng.normal(size=n),
        "unused": np.full(n, np.nan),
        "y": rng.normal(size=n),
    }
    fit = mb.glmboost("y ~ x + z", data=data, family=mb.Gaussian(),
                      control=mb.boost_control(20, 0.1), na_action="omit")
    assert fit._na_dropped == []
    assert len(fit.fitted_) == n


def test_clean_data_with_omit_drops_nothing():
    rng = np.random.default_rng(4)
    n = 100
    data = {"x": rng.normal(size=n), "y": rng.normal(size=n)}
    fit = mb.glmboost("y ~ x", data=data, family=mb.Gaussian(),
                      control=mb.boost_control(20, 0.1), na_action="omit")
    assert fit._na_dropped == []


def test_omit_aligns_weights():
    rng = np.random.default_rng(6)
    n = 100
    x = rng.normal(size=n)
    x[7] = np.nan
    y = 2.0 * x + rng.normal(scale=0.1, size=n)
    w = rng.uniform(0.5, 2.0, size=n)
    fit = mb.glmboost("y ~ x", data={"x": x, "y": y}, family=mb.Gaussian(),
                      control=mb.boost_control(50, 0.1), weights=w, na_action="omit")
    assert len(fit.fitted_) == 99


def test_omit_records_count_in_summary():
    data = _data_with_holes()
    fit = mb.glmboost("y ~ x + z", data=data, family=mb.Gaussian(),
                      control=mb.boost_control(20, 0.1), na_action="omit")
    info = fit.summary(as_dict=True)
    assert info["n_omitted"] == 2
    assert info["n"] == 198


def test_omit_works_on_polars_frames():
    data = pl.DataFrame(_data_with_holes())
    fit = mb.glmboost("y ~ x + z", data=data, family=mb.Gaussian(),
                      control=mb.boost_control(30, 0.1), na_action="omit")
    assert len(fit.fitted_) == 198


def test_missing_factor_level_is_omitted():
    rng = np.random.default_rng(8)
    n = 150
    g = np.array(["a", "b", "c", None], dtype=object)[rng.integers(0, 4, n)]
    x = rng.normal(size=n)
    y = x + rng.normal(scale=0.2, size=n)
    fit = mb.gamboost("y ~ brandom(g) + bols(x)", data={"g": g, "x": x, "y": y},
                      family=mb.Gaussian(), control=mb.boost_control(50, 0.1),
                      na_action="omit")
    n_missing = int(np.sum([v is None for v in g]))
    assert len(fit._na_dropped) == n_missing


def test_invalid_na_action_is_rejected():
    data = {"x": np.arange(10.0), "y": np.arange(10.0)}
    with pytest.raises(ValueError, match="na_action must be"):
        mb.glmboost("y ~ x", data=data, family=mb.Gaussian(),
                    control=mb.boost_control(10, 0.1), na_action="drop")


def test_all_rows_missing_is_rejected():
    data = {"x": np.full(20, np.nan), "y": np.arange(20.0)}
    with pytest.raises(ValueError, match="every row"):
        mb.glmboost("y ~ x", data=data, family=mb.Gaussian(),
                    control=mb.boost_control(10, 0.1), na_action="omit")


@pytest.mark.parametrize("kind", ["dict", "pandas", "polars"])
def test_omit_drops_the_same_rows_for_every_frame_type(kind):
    """pandas frames have `.filter` too, which selects columns: the row filter
    must not take the polars branch for them."""
    import pandas as pd
    import polars as pl

    rng = np.random.default_rng(0)
    x = rng.normal(size=100)
    y = x + rng.normal(size=100)
    x[:5] = np.nan
    columns = {"x1": x, "y": y}
    data = {"dict": columns, "pandas": pd.DataFrame(columns),
            "polars": pl.DataFrame(columns)}[kind]
    fit = mb.gamboost("y ~ bbs(x1)", data=data, na_action="omit",
                      control=mb.boost_control(mstop=20))
    assert len(np.asarray(fit.predict())) == 95


def _holes(seed=0, n=400):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.uniform(-2, 2, n), rng.uniform(-2, 2, n)
    y = np.sin(x1) + 0.5 * x2 + rng.normal(0, 0.3, n)
    x1 = x1.copy()
    x1[rng.uniform(size=n) < 0.15] = np.nan
    return {"x1": x1, "x2": x2, "y": y}


def test_impute_keeps_every_row_and_predicts_rows_with_missing_values(tmp_path):
    data = _holes()
    fit = mb.gamboost("y ~ bbs(x1) + bbs(x2)", data=data, na_action="impute",
                      control=mb.boost_control(100, 0.1))
    assert len(fit.fitted()) == 400
    assert "x1_missing" in fit.base_learner_names()
    new = {"x1": np.array([np.nan, 0.0]), "x2": np.array([0.0, 0.0])}
    pred = np.asarray(fit.predict(newdata=new)).ravel()
    assert np.all(np.isfinite(pred))
    # A missing x1 is predicted at the training median of x1 plus the
    # indicator's effect.
    median = float(np.nanmedian(data["x1"]))
    at_median = np.asarray(fit.predict(newdata={"x1": np.array([median]),
                                                "x2": np.array([0.0])})).ravel()[0]
    indicator = np.asarray(fit.predict(newdata=new, which="x1_missing")).ravel()[0]
    assert pred[0] == pytest.approx(at_median + indicator)
    fit.cvrisk(folds=mb.cv_weights(400, "kfold", 3, seed=0))
    path = tmp_path / "m.json"
    mb.save_model(fit, str(path))
    np.testing.assert_allclose(np.asarray(mb.load_model(str(path)).predict(newdata=new)).ravel(), pred)


def test_impute_drops_a_missing_response_and_works_for_glmboost():
    data = _holes()
    data["y"] = data["y"].copy()
    data["y"][:5] = np.nan
    fit = mb.glmboost("y ~ x1 + x2", data=data, na_action="impute",
                      control=mb.boost_control(100, 0.1))
    assert len(fit.fitted()) == 395
    assert "x1_missing" in fit.coef()


def _labels_with_missing(n=300, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.choice(["a", "b", "c"], n).astype(object)
    x = rng.normal(size=n)
    y = x + np.array([{"a": 0.0, "b": 1.0, "c": -1.0}[v] for v in g])
    y = y + rng.normal(scale=0.3, size=n)
    g[:30] = None
    y[:30] += 2.0
    return {"y": y, "x": x, "g": g}


@pytest.mark.parametrize("fit", [
    lambda d: mb.glmboost("y ~ x + g", data=d, na_action="impute",
                          control=mb.boost_control(500, 0.1)),
    lambda d: mb.gamboost("y ~ bbs(x) + bols(g)", data=d, na_action="impute",
                          control=mb.boost_control(500, 0.1)),
    lambda d: mb.gamboost("y ~ bbs(x) + brandom(g)", data=d, na_action="impute",
                          control=mb.boost_control(500, 0.1)),
])
def test_impute_gives_missing_labels_a_level_of_their_own(fit):
    model = fit(_labels_with_missing())
    new = {"x": np.zeros(4), "g": np.array(["a", "b", "c", None], dtype=object)}
    prediction = model.predict(newdata=new)
    assert prediction[3] == pytest.approx(2.0, abs=0.3)
    assert prediction[1] - prediction[0] == pytest.approx(1.0, abs=0.2)


def test_the_missing_level_is_never_the_reference_level():
    model = mb.glmboost("y ~ x + g", data=_labels_with_missing(), na_action="impute")
    assert {"gb", "gc"} <= set(model.coef())
    assert "ga" not in model.coef()


def test_missing_labels_under_na_action_error_name_the_column():
    with pytest.raises(ValueError, match="column 'g' has 30 missing"):
        mb.glmboost("y ~ x + g", data=_labels_with_missing())


def test_impute_predicts_a_missing_value_in_a_column_complete_in_training():
    """A new row can miss a value the training rows all had: it is predicted
    at the training median, as a row at that median is."""
    import numpy as np
    import polars as pl

    import mboostrs as mb

    rng = np.random.default_rng(0)
    x, z = rng.normal(size=300), rng.normal(size=300)
    train = pl.DataFrame({"x": x, "z": z, "y": x + rng.normal(size=300)})
    model = mb.tune_mstop("y ~ .", train, na_action="impute").model
    new = pl.DataFrame({"x": [None, float(np.median(x))], "z": [0.2, 0.2]})
    predicted = np.asarray(model.predict(newdata=new)).ravel()
    assert np.isfinite(predicted).all()
    assert predicted[0] == predicted[1]
