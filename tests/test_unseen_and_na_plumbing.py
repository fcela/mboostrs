"""Regression tests for unseen_levels, na_action, and hygiene fixes."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import mboostrs as mb


def _cat_data(n: int = 240, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    grp = rng.choice(["north", "south", "east", "west"], size=n)
    eff = {"north": -0.6, "south": 0.4, "east": 0.2, "west": 0.0}
    y = np.sin(x) + np.array([eff[g] for g in grp]) + rng.normal(0.0, 0.4 + 0.2 * np.abs(x), size=n)
    return pl.DataFrame({"x": x, "grp": grp, "y": y})


def test_conformal_interval_unseen_levels_gamboost():
    train = _cat_data(240, seed=1)
    test = pl.DataFrame({"x": [0.0, 0.5, -0.5], "grp": ["north", "island", "west"]})
    fit = mb.gamboost(
        "y ~ bbs(x, df=4) + brandom(grp, df=2)",
        train,
        family=mb.Gaussian(),
        control=mb.boost_control(60, 0.1),
    )
    with pytest.raises(ValueError, match="unseen"):
        mb.conformal_interval(fit, test, level=0.9, method="split", seed=0)

    ci = mb.conformal_interval(
        fit, test, level=0.9, method="split", seed=0, unseen_levels="average"
    )
    assert ci.lower.shape == (3,)
    assert np.all(np.isfinite(ci.lower))
    assert np.all(np.isfinite(ci.upper))
    assert np.all(ci.lower <= ci.upper)
    assert np.all(np.isfinite(ci.fitted_mean))

    ci_method = fit.conformal_interval(
        test, level=0.9, method="split", seed=0, unseen_levels="average"
    )
    np.testing.assert_allclose(ci_method.lower, ci.lower)
    np.testing.assert_allclose(ci_method.upper, ci.upper)

    with pytest.raises(ValueError, match="unseen_levels"):
        mb.conformal_interval(fit, test, unseen_levels="bogus")


def test_conformal_interval_unseen_levels_lss_and_heldout():
    train = _cat_data(240, seed=2)
    cal = _cat_data(80, seed=3)
    test = pl.DataFrame({"x": [0.2, -0.3], "grp": ["north", "island"]})
    fit = mb.gamboostLSS(
        {"mu": "y ~ bbs(x, df=4) + brandom(grp, df=2)", "sigma": "y ~ bbs(x, df=4)"},
        train,
        families=mb.GaussianLSS(),
        control=mb.boost_control(50, 0.1),
    )
    ci_split = mb.conformal_interval(
        fit, test, level=0.9, seed=0, unseen_levels="average"
    )
    assert np.all(np.isfinite(ci_split.lower))
    assert np.all(np.isfinite(ci_split.upper))

    ci_cal = mb.conformal_interval(
        fit, test, level=0.9, calibration=cal, unseen_levels="average"
    )
    assert np.all(np.isfinite(ci_cal.lower))
    assert np.all(np.isfinite(ci_cal.upper))

    params = fit.predict_parameters(test, unseen_levels="average")
    assert set(params) == {"mu", "sigma"}
    assert np.all(np.isfinite(params["mu"]))


def test_lss_cvrisk_accepts_unseen_levels_cyclic_and_noncyclic():
    data = _cat_data(180, seed=4)
    for method in ("cyclic", "noncyclic"):
        fit = mb.gamboostLSS(
            {"mu": "y ~ bbs(x, df=4) + brandom(grp, df=2)", "sigma": "y ~ bbs(x, df=4)"},
            data,
            families=mb.GaussianLSS(),
            control=mb.boost_control(30, 0.1),
            method=method,
        )
        cv = fit.cvrisk(B=3, seed=1, length_out=3, unseen_levels="average")
        assert np.all(np.isfinite(cv.aggregate))

        with pytest.raises(ValueError, match="unseen_levels"):
            fit.cvrisk(B=3, seed=1, length_out=3, unseen_levels="invalid")


def test_discover_unseen_levels_produces_predictions():
    train = _cat_data(240, seed=5)
    test = pl.DataFrame({"x": [0.1, -0.4, 0.8], "grp": ["north", "island", "east"]})
    res = mb.discover(
        train,
        "y",
        newdata=test,
        seed=0,
        folds=3,
        interactions=False,
        spread=False,
        unseen_levels="average",
    )
    assert res.predictions is not None
    assert res.predictions.height == 3
    assert np.all(np.isfinite(res.predictions["mean"].to_numpy()))
    assert np.all(np.isfinite(res.predictions["lower"].to_numpy()))
    assert np.all(np.isfinite(res.predictions["upper"].to_numpy()))

    with pytest.raises(ValueError, match="unseen_levels"):
        mb.discover(train, "y", unseen_levels="invalid")


def test_fit_and_autospec_na_action_impute_and_hygiene_string_column():
    rng = np.random.default_rng(7)
    n = 180
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x2[:15] = np.nan
    cat = rng.choice(["a", "b", "c"], size=n)
    y = 1.5 * x1 + 0.5 * np.nan_to_num(x2) + rng.normal(0.0, 0.3, size=n)
    train = pl.DataFrame({"x1": x1, "x2": x2, "cat": cat, "y": y})
    test = pl.DataFrame({"x1": [0.0, 1.0], "x2": [np.nan, 0.5], "cat": ["a", "b"]})

    # String column with 3 levels must NOT be misdiagnosed as discrete-under-float.
    rec = mb.autospec(train, "y", budget=0, na_action="impute")
    assert not any(
        f.column == "cat" and f.kind == "discrete-under-float" for f in rec.hygiene
    )

    fitted = mb.fit(train, "y", budget="quick", seed=0, na_action="impute")
    preds = fitted.predict(test, type="response")
    assert preds.shape == (2,)
    assert np.all(np.isfinite(preds))

    with pytest.raises(ValueError, match="missing values"):
        mb.autospec(train, "y", budget=0, na_action="error")
