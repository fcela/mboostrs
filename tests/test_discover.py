"""mb.discover: planted structure is recovered, null data returns nothing, budgets are logged."""

import json

import numpy as np
import polars as pl
import pytest

import mboostrs as mb
from mboostrs.discover import _label_curve


def _planted(seed=0, n=300):
    """Two smooth effects (x, z), one z-by-g interaction, a log-scale driver (x), two nulls."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    z = rng.uniform(-2, 2, n)
    g = rng.integers(0, 2, n).astype(float)
    n1 = rng.normal(size=n)
    n2 = rng.normal(size=n)
    y = np.sin(x) + 1.5 * z * g + rng.normal(0, 0.5, n) * np.exp(0.5 * x)
    return pl.DataFrame({"x": x, "z": z, "g": g, "n1": n1, "n2": n2, "y": y})


def _permuted(seed=0, n=300):
    """The same covariates with the response shuffled: no structure at all."""
    df = _planted(seed, n)
    return df.with_columns(pl.col("y").shuffle(seed=seed + 1))


@pytest.fixture(scope="module")
def planted():
    df = _planted()
    new = _planted(seed=9, n=12).drop("y")
    # Preserve the 300-row selection fixture; reserve additional independent
    # calibration rows before the analysis rather than reusing its responses.
    return mb.discover(df, "y", newdata=new, calibration=_planted(seed=8, n=150), seed=0)


@pytest.fixture(scope="module")
def permuted():
    df = _permuted()
    new = _planted(seed=9, n=5).drop("y")
    return mb.discover(df, "y", newdata=new, calibration=_permuted(seed=8, n=150), seed=0)


def test_planted_structure_is_recovered(planted):
    variables = {v for e in planted.mean_effects for v in e["variables"]}
    assert {"x", "z"} <= variables
    assert not variables & {"n1", "n2"}
    assert any(set(e["variables"]) == {"z", "g"} for e in planted.interactions)
    assert [d["variables"] for d in planted.spread_drivers] == [["x"]]
    assert planted.spread_drivers[0]["direction"] == "increasing"
    assert planted.spread_formula is not None


def test_permuted_response_returns_nothing(permuted):
    assert permuted.mean_effects == []
    assert permuted.interactions == []
    assert permuted.spread_drivers == []
    assert permuted.formula == "1"
    assert permuted.spread_formula is None
    assert permuted.predictions.height == 5
    assert permuted.predictions["mean"].is_finite().all()
    # No spread model was adopted, so there is no fitted scale to report.
    assert permuted.predictions.columns == ["row", "mean", "lower", "upper"]


def test_predictions_have_intervals_for_newdata(planted):
    preds = planted.predictions
    # The planted data have a spread driver, so the location-scale fit adds
    # its fitted scale beside the mean (issue #589).
    assert preds.columns == ["row", "mean", "sigma", "lower", "upper"]
    assert preds.height == 12
    assert preds["row"].to_list() == list(range(12))
    assert preds["lower"].is_finite().all() and preds["upper"].is_finite().all()
    assert (preds["lower"] <= preds["upper"]).all()
    assert (preds["sigma"] > 0).all()


def test_to_dict_is_json_serializable(planted, permuted):
    for result in (planted, permuted):
        record = json.loads(json.dumps(result.to_dict()))
        assert record["provenance"]["seed"] == 0
        assert "mboostrs_version" in record["provenance"]
        assert record["formula"] == result.formula


def test_budget_skipping_is_logged():
    result = mb.discover(_planted(n=120), "y", seed=0, budget_seconds=0.0)
    skipped = [r for r in result.log if "skipped" in r["reason"]]
    assert any(r["stage"] == "mean effects" for r in skipped)
    assert all("budget_seconds=0" in r["reason"] for r in skipped)
    # The mean stage was skipped, so the draft terms, nulls included, were kept.
    assert "n1" in result.formula and "n2" in result.formula


def test_include_arrays_adds_the_training_fit(planted):
    record = json.loads(json.dumps(planted.to_dict(include_arrays=True)))
    assert len(record["fitted"]) == 300
    assert "fitted" not in planted.to_dict()


def test_contrast_interval_brackets_the_estimate(planted):
    contrast = planted.contrast("x", -1.5, 1.5)
    assert contrast.lower <= contrast.estimate <= contrast.upper
    assert contrast.upper > contrast.lower
    assert contrast.estimate > 0


def test_contrast_of_an_unselected_variable_is_zero(planted):
    contrast = planted.contrast("n1", -1.0, 1.0)
    assert contrast.estimate == 0.0
    assert contrast.lower == 0.0 and contrast.upper == 0.0
    assert "not selected" in contrast.note


def test_gam_refit_contrast_path():
    result = mb.discover(_planted(), "y", seed=0, spread=False, interactions=False)
    assert result.spread_formula is None
    assert type(result.refit).__name__ == "GamRefit"
    contrast = result.contrast("z", -1.0, 1.0)
    assert contrast.scale == "link"
    assert contrast.lower <= contrast.estimate <= contrast.upper
    assert contrast.estimate > 0


def test_small_data_lowers_the_fold_count():
    result = mb.discover(_planted(n=60), "y", seed=0, interactions=False, spread=False)
    assert result.folds == 3
    assert any(r["stage"] == "setup" and "lowered" in r["reason"] for r in result.log)


def test_binary_response_is_fitted_as_binomial():
    rng = np.random.default_rng(3)
    x = rng.uniform(-2, 2, 200)
    z = rng.uniform(-2, 2, 200)
    eta = 1.5 * x
    y = (rng.uniform(size=200) < 1 / (1 + np.exp(-eta))).astype(float)
    result = mb.discover(pl.DataFrame({"x": x, "z": z, "y": y}), "y", seed=0)
    assert result.family == "Binomial"
    assert result.spread_formula is None
    assert "x" in {v for e in result.mean_effects for v in e["variables"]}


def test_warnings_are_recorded_once_each(planted):
    assert isinstance(planted.warnings, list)
    assert len(set(planted.warnings)) == len(planted.warnings)


def test_missing_response_column_raises():
    with pytest.raises(ValueError, match="not a column"):
        mb.discover(_planted(n=40), "outcome", seed=0)


def test_bad_level_raises():
    with pytest.raises(ValueError, match="level must be between 0 and 1"):
        mb.discover(_planted(n=40), "y", level=1.5)


def test_too_few_rows_raises():
    with pytest.raises(ValueError, match="at least 30 rows"):
        mb.discover(_planted(n=20), "y")


@pytest.mark.parametrize(("curve", "expected"), [
    (np.linspace(0, 1, 25), "increasing"),
    (np.linspace(1, 0, 25), "decreasing"),
    (np.linspace(0, 1, 25) + 0.01 * np.sin(np.arange(25) * 2.0), "increasing"),
    ((np.linspace(-1, 1, 25)) ** 2, "u_shaped"),
    (-(np.linspace(-1, 1, 25)) ** 2, "inverted_u"),
    (np.sin(np.linspace(0, 4 * np.pi, 25)), "nonmonotone"),
    (np.r_[np.zeros(12), np.ones(13)], "threshold_increase"),
    (np.r_[np.zeros(12), -np.ones(13)], "threshold_decrease"),
])
def test_shape_labels_read_the_curve(curve, expected):
    assert _label_curve(np.asarray(curve, dtype=float)) == expected
