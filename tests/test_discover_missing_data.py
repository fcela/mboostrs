"""mb.discover and screen_interactions with missing values, skewed covariates and sparse labels."""

import importlib
import types

import numpy as np
import polars as pl
import pytest

import mboostrs as mb
from mboostrs.terms import screen_interactions

# `mboostrs.discover` is the function; the module holds the stages to patch.
discover_module = importlib.import_module("mboostrs.discover")


def _missing_planted(seed=0, n=300):
    """Spread driven by x, an interaction of z with a label g, and NaN in x and z."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    z = rng.uniform(-2, 2, n)
    g = rng.integers(0, 2, n).astype(float)
    y = np.sin(x) + 1.5 * z * g + rng.normal(0, 0.5, n) * np.exp(0.5 * x)
    x_obs = x.copy()
    x_obs[rng.choice(n, 30, replace=False)] = np.nan
    z_obs = z.copy()
    z_obs[rng.choice(n, 30, replace=False)] = np.nan
    return pl.DataFrame({"x": x_obs, "z": z_obs, "g": g, "y": y})


def _stage_errors(result):
    return [e for e in result.log if e["reason"].startswith("error")]


@pytest.fixture(scope="module")
def missing_result():
    return mb.discover(_missing_planted(), "y", seed=0)


def test_nan_in_spread_driver_and_interaction_partner_runs_every_stage(missing_result):
    assert _stage_errors(missing_result) == []
    stages = {e["stage"] for e in missing_result.log}
    assert {"mean effects", "interactions", "spread", "tune", "refit"} <= stages
    candidates = [e for e in missing_result.log
                  if e["stage"] == "interactions" and e["candidate"] is not None]
    assert candidates, "the interaction stage evaluated no candidate"
    assert missing_result.model is not None and missing_result.mstop >= 1


def test_nan_is_missing_so_the_impute_fits_are_kept(missing_result):
    # A NaN is a missing value. When it is not counted, discover switches
    # "impute" to "omit" for data it takes to be complete, and says so in the log.
    setup = [e["reason"] for e in missing_result.log if e["stage"] == "setup"]
    assert not any("na_action='omit'" in reason for reason in setup)


def _skewed(seed=3, n=300):
    rng = np.random.default_rng(seed)
    skew = rng.lognormal(0, 1.2, n)
    x = rng.uniform(-2, 2, n)
    y = 1.5 * np.log(skew) + np.sin(x) + rng.normal(0, 0.4, n)
    return pl.DataFrame({"skew": skew, "x": x, "y": y})


@pytest.fixture(scope="module")
def skewed_result():
    new = pl.DataFrame({"skew": [1.0, 3.0, 20.0], "x": [0.0, 0.5, 1.0]})
    return mb.discover(_skewed(), "y", newdata=new, seed=0, interactions=False, spread=False)


def test_skewed_positive_covariate_is_drafted_on_the_log_scale(skewed_result):
    assert "bbs(log(skew), df=4.0)" in skewed_result.formula
    assert "bbs(x, df=4.0)" in skewed_result.formula
    logged = [e for e in skewed_result.log if e["stage"] == "draft" and e["candidate"] == "log(skew)"]
    assert len(logged) == 1
    assert "sample skewness" in logged[0]["reason"]
    record = [e for e in skewed_result.mean_effects if e["variables"] == ["skew"]]
    assert record and record[0]["transform"] == "log"


def test_log_drafted_covariate_contrast_is_on_the_original_scale(skewed_result):
    # y = 1.5 log(skew) + f(x): moving skew from 1 to 3 changes the mean by 1.5 log 3.
    contrast = skewed_result.contrast("skew", 1.0, 3.0)
    assert contrast.scale == "link"
    assert abs(contrast.estimate - 1.5 * np.log(3.0)) < 0.4
    assert contrast.lower is not None and contrast.lower < contrast.estimate < contrast.upper


def test_predictions_take_original_scale_values(skewed_result):
    mean = skewed_result.predictions["mean"].to_list()
    assert mean[0] < mean[1] < mean[2]
    assert abs((mean[2] - mean[0]) - 1.5 * np.log(20.0)) < 0.8


def test_positive_covariate_without_skew_is_not_logged():
    rng = np.random.default_rng(1)
    n = 300
    u = rng.uniform(1, 3, n)
    y = np.sin(u) + rng.normal(0, 0.3, n)
    result = mb.discover(pl.DataFrame({"u": u, "y": y}), "y", seed=0,
                         interactions=False, spread=False)
    assert not [e for e in result.log if e["candidate"] == "log(u)"]
    assert "log(u)" not in result.formula


def test_four_level_string_column_is_drafted_as_bols():
    rng = np.random.default_rng(2)
    n = 300
    x = rng.uniform(-2, 2, n)
    lev = rng.choice(["north", "south", "east", "west"], n)
    effect = {"north": 0.0, "south": 1.0, "east": -1.0, "west": 0.5}
    y = np.sin(x) + np.array([effect[v] for v in lev]) + rng.normal(0, 0.4, n)
    result = mb.discover(pl.DataFrame({"x": x, "lev": lev, "y": y}), "y", seed=0,
                         interactions=False, spread=False)
    drafted = [e for e in result.log if e["stage"] == "draft"]
    assert any(e["candidate"] == "bols(lev)" for e in drafted)
    assert "brandom(lev" not in drafted[0]["candidate"]
    assert "brandom(lev" not in result.formula


def test_tuning_boundary_keeps_the_term_set_and_logs_it(monkeypatch):
    frame = _missing_planted()
    baseline = mb.discover(frame, "y", seed=0, spread=False, interactions=False)

    calls = []

    def still_falling(formula, data, **kwargs):
        calls.append(kwargs["start"])
        return types.SimpleNamespace(mstop=kwargs["start"], at_boundary=True)

    monkeypatch.setattr(discover_module, "tune_mstop", still_falling)
    result = mb.discover(frame, "y", seed=0, spread=False, interactions=False)

    assert calls == [100, 200, 400]
    assert result.formula == baseline.formula
    assert result.mstop == 400
    tune = [e for e in result.log if e["stage"] == "tune"]
    assert len(tune) == 1 and tune[0]["reason"].startswith("boundary")
    assert any("boundary" in w for w in result.warnings)


def test_tuning_stops_when_the_risk_turns(monkeypatch):
    frame = _missing_planted()
    calls = []

    def turns_at_200(formula, data, **kwargs):
        calls.append(kwargs["start"])
        at_boundary = kwargs["start"] < 200
        return types.SimpleNamespace(mstop=kwargs["start"] - 1, at_boundary=at_boundary)

    monkeypatch.setattr(discover_module, "tune_mstop", turns_at_200)
    result = mb.discover(frame, "y", seed=0, spread=False, interactions=False)
    assert calls == [100, 200]
    assert result.mstop == 199
    tune = [e for e in result.log if e["stage"] == "tune"]
    assert tune and not tune[0]["reason"].startswith("boundary")


def test_distributional_tuning_boundary_doubles_the_grid_and_keeps_the_terms(monkeypatch):
    rng = np.random.default_rng(0)
    n = 300
    x = rng.uniform(-2, 2, n)
    z = rng.uniform(-2, 2, n)
    y = np.sin(x) + 0.5 * z + rng.normal(0, 0.4, n) * np.exp(0.8 * x)
    x_missing = x.copy()
    x_missing[rng.choice(n, 20, replace=False)] = np.nan
    frame = pl.DataFrame({"x": x_missing, "z": z, "y": y})

    grids = []
    real_gamboostLSS = discover_module.gamboostLSS

    def record(formulas, data, **kwargs):
        grids.append(kwargs.get("mstop"))
        return real_gamboostLSS(formulas, data, **kwargs)

    monkeypatch.setattr(discover_module, "LSS_GRID_MSTOP", 20)
    monkeypatch.setattr(discover_module, "gamboostLSS", record)
    monkeypatch.setattr(discover_module, "is_at_boundary", lambda optimum, end: True)
    result = mb.discover(frame, "y", seed=0, interactions=False)

    assert result.spread_formula == "bbs(x, df=4)"
    assert grids[:3] == [20, 40, 80]
    assert result.formula == "bbs(x, df=4.0) + bbs(z, df=4.0)"
    tune = [e for e in result.log if e["stage"] == "tune"]
    assert len(tune) == 1 and tune[0]["reason"].startswith("boundary")
    assert any(w.startswith("boundary") for w in result.warnings)


def test_screen_interactions_with_nan_and_impute_runs():
    rng = np.random.default_rng(4)
    n = 300
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    c = rng.normal(size=n)
    residual = a * b + rng.normal(0, 0.3, n)
    a_missing = a.copy()
    a_missing[[5, 17]] = np.nan
    ranked = screen_interactions({"a": a_missing, "b": b, "c": c}, residual, na_action="impute")
    assert {(p.left, p.right) for p in ranked} == {("a", "b"), ("a", "c"), ("b", "c")}
    assert ranked[0].left == "a" and ranked[0].right == "b"


def test_screen_interactions_default_names_the_column_and_remedy():
    rng = np.random.default_rng(4)
    n = 300
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    residual = a * b + rng.normal(0, 0.3, n)
    a_missing = a.copy()
    a_missing[5] = np.nan
    with pytest.raises(ValueError, match=r"column 'a' has 1 missing value\(s\).*first at row 5") as info:
        screen_interactions({"a": a_missing, "b": b}, residual)
    assert "na_action='impute'" in str(info.value)
    assert "na_action='omit'" in str(info.value)


def test_screen_interactions_omit_drops_the_rows():
    rng = np.random.default_rng(5)
    n = 300
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    residual = a * b + rng.normal(0, 0.3, n)
    a_missing = a.copy()
    a_missing[:10] = np.nan
    ranked = screen_interactions({"a": a_missing, "b": b}, residual, na_action="omit")
    assert len(ranked) == 1 and ranked[0].removed > 0


def test_screen_interactions_rejects_an_unknown_na_action():
    with pytest.raises(ValueError, match="na_action must be 'error', 'omit', 'impute', 'impute_model' or 'impute_model_split'"):
        screen_interactions({"a": np.zeros(10), "b": np.zeros(10)}, np.zeros(10), na_action="drop")


def test_tensor_margin_with_nan_names_the_column_and_remedy():
    rng = np.random.default_rng(6)
    n = 200
    x = rng.uniform(-2, 2, n)
    z = rng.uniform(-2, 2, n)
    z[3] = np.nan
    y = np.sin(x) + z + rng.normal(0, 0.3, n)
    data = pl.DataFrame({"x": x, "z": z, "y": y})
    with pytest.raises(mb.MboostrsValueError, match="margin 'z' has 1 missing value") as info:
        mb.gamboost("y ~ bbs(x, df=4) + bbs(x, df=2.5, knots=5) %X% bbs(z, df=2.5, knots=5)", data)
    assert "na_action='impute'" in str(info.value)
