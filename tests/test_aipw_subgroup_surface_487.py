"""AIPWResult.subgroup_effect and effect_surface (issue #487, items 3 and 4)."""

import numpy as np
import polars as pl
import pytest

import mboostrs as mb
from mboostrs.causal import CausalSpec, aipw

FAST = dict(smooth=False, folds=5, tune=False, mstop=60)


def _spec():
    return CausalSpec(
        treatment="a", outcome="y", covariates=("x1", "x2"), target_population="simulated",
        unit="person", time_zero="treatment assignment", follow_up="one year",
        adjustment_rationale="x1 and x2 cause both treatment and outcome",
    )


def _data(n=400, seed=487):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=(2, n))
    a = rng.binomial(1, 1 / (1 + np.exp(-(0.4 * x1 - 0.6 * x2))))
    y = (1.0 + 1.5 * (x1 > 0)) * a + x1 - 0.5 * x2 + rng.normal(size=n)
    return pl.DataFrame({
        "a": a, "y": y, "x1": x1, "x2": x2,
        "grp": np.where(x1 > 0, "high", "low"),
        "_aipw_score": np.zeros(n),
    })


@pytest.fixture(scope="module")
def fitted():
    data = _data()
    return data, aipw(data, _spec(), **FAST)


def test_full_subgroup_reproduces_ate_and_se(fitted):
    data, result = fitted
    row = result.subgroup_effect(np.ones(data.height, dtype=bool), output="records")[0]
    assert row["n"] == data.height
    assert row["estimate"] == pytest.approx(result.estimate, rel=1e-12)
    assert row["standard_error"] == pytest.approx(result.standard_error, rel=1e-12)
    assert (row["lower"], row["upper"]) == pytest.approx(result.interval, rel=1e-12)


def test_full_subgroup_reproduces_ate_with_weights_and_clusters():
    data = _data(n=300, seed=3)
    n = data.height
    weights = np.random.default_rng(1).integers(1, 4, n).astype(float)
    groups = np.repeat(np.arange(60), 5)
    result = aipw(data, _spec(), weights=weights, folds=mb.grouped_folds(groups, B=5),
                  smooth=False, tune=False, mstop=60)
    row = result.subgroup_effect(np.ones(n, dtype=bool), output="records")[0]
    assert row["estimate"] == pytest.approx(result.estimate, rel=1e-12)
    assert row["standard_error"] == pytest.approx(result.standard_error, rel=1e-12)
    assert (row["lower"], row["upper"]) == pytest.approx(result.interval, rel=1e-12)


def test_column_levels_match_masks(fitted):
    data, result = fitted
    by_column = {r["level"]: r for r in result.subgroup_effect("grp", output="records")}
    assert set(by_column) == {"high", "low"}
    mask = (data["grp"] == "high").to_numpy()
    by_mask = result.subgroup_effect(mask, output="records")[0]
    assert by_mask["level"] is True
    for key in ("n", "estimate", "standard_error", "lower", "upper"):
        assert by_mask[key] == pytest.approx(by_column["high"][key], rel=1e-12)
    assert by_column["high"]["estimate"] == pytest.approx(
        result.scores[mask].mean(), rel=1e-12)
    assert by_column["high"]["estimate"] > by_column["low"]["estimate"]


def test_level_widens_interval(fitted):
    _, result = fitted
    narrow = result.subgroup_effect("grp", level=0.8, output="records")[0]
    wide = result.subgroup_effect("grp", level=0.99, output="records")[0]
    assert wide["upper"] - wide["lower"] > narrow["upper"] - narrow["lower"]


def test_errors_name_the_argument(fitted):
    data, result = fitted
    with pytest.raises(ValueError, match="nope"):
        result.subgroup_effect("nope")
    with pytest.raises(ValueError, match="one entry per training row"):
        result.subgroup_effect(np.ones(3, dtype=bool))
    with pytest.raises(ValueError, match="selects no rows"):
        result.subgroup_effect(np.zeros(data.height, dtype=bool))


def test_overlap_result_is_refused():
    data = _data(n=200, seed=5)
    overlap = aipw(data, _spec(), target="overlap", **FAST)
    with pytest.raises(ValueError, match="aipw\\(target='all'\\)"):
        overlap.subgroup_effect("grp")
    with pytest.raises(ValueError, match="aipw\\(target='all'\\)"):
        overlap.effect_surface("bols(x1)")


def test_effect_surface_recovers_heterogeneity(fitted):
    data, result = fitted
    model = result.effect_surface("bbs(x1, df=4) + bols(x2)", mstop=150)
    grid = pl.DataFrame({"x1": [-1.5, 1.5], "x2": [0.0, 0.0]})
    low, high = np.asarray(model.predict(grid), dtype=float).ravel()
    assert high - low > 0.3  # the true effect steps up by 1.5 at x1 = 0; a smooth fit blurs it
    assert np.all(data["_aipw_score"].to_numpy() == 0)  # the user's column is untouched


def test_effect_surface_accepts_tilde_and_rejects_unknown_columns(fitted):
    _, result = fitted
    assert result.effect_surface("~ bols(x1)", mstop=20) is not None
    with pytest.raises(ValueError, match="names no column"):
        result.effect_surface("bols(zzz)")
