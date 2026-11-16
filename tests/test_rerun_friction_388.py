"""Defects found by the claims-discovery benchmark rerun (issue #388, items 7-18).

Each test reproduces what an analyst met: missing-value indicators and
sensitivity grids, turn flags, candidate comparisons, conformal intervals with
tensor terms, and the warnings and summaries that cross-validation and
validate() print.
"""

import warnings

import numpy as np
import polars as pl
import pytest

import mboostrs as mb
from mboostrs import explore
from mboostrs._extrapolation import ExtrapolationWarning
from mboostrs._refit import refit_spec


@pytest.fixture(scope="module")
def incomplete():
    """x has missing values; z and g are complete. y depends on x and z."""
    rng = np.random.default_rng(3)
    n = 600
    x = rng.uniform(0, 1, n)
    z = rng.uniform(0, 1, n)
    y = np.sin(3 * x) + (0.3 + z) * rng.normal(0, 1, n)
    return pl.DataFrame({
        "x": np.where(rng.uniform(size=n) < 0.1, np.nan, x),
        "z": z,
        "y": y,
    })


@pytest.fixture(scope="module")
def roof():
    """A distributional fit whose mu formula has a categorical term."""
    rng = np.random.default_rng(5)
    n = 500
    x = rng.uniform(0, 1, n)
    z = rng.uniform(0, 1, n)
    roof_class = rng.choice(["slate", "tile", "thatch"], n)
    sigma = np.exp(0.5 * (roof_class == "thatch") + 0.3 * np.sin(4 * z))
    y = np.sin(3 * x) + sigma * rng.normal(0, 0.3, n)
    return pl.DataFrame({"x": x, "z": z, "roof_class": roof_class, "y": y})


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


# ─── item 7: generated missing-value indicators ───────────────────────


def test_sensitivity_builds_the_indicator_of_an_imputed_glmboost(incomplete):
    model = _quiet(mb.glmboost, "y ~ x + z", data=incomplete, na_action="impute",
                   control=mb.boost_control(mstop=60))
    rows = _quiet(mb.sensitivity, model, incomplete.drop("y"))
    assert sorted(row["variable"] for row in rows) == ["x", "z"]


def test_sensitivity_builds_the_indicator_when_the_name_is_taken(incomplete):
    """The caller's own `x_missing` column forces the indicator to `x_missing_`."""
    data = incomplete.with_columns(x_missing=pl.Series(
        np.random.default_rng(9).uniform(size=incomplete.height) < 0.5).cast(pl.Float64))
    model = _quiet(mb.gamboost, "y ~ bbs(x) + bols(z)", data=data, na_action="impute",
                   control=mb.boost_control(mstop=60))
    assert any(step.name == "x_missing_" for step in model._transform_recipe.steps)
    rows = _quiet(mb.sensitivity, model, data.drop("y"))
    assert sorted(row["variable"] for row in rows) == ["x", "z"]


def test_a_generated_indicator_error_says_it_is_generated(incomplete):
    model = _quiet(mb.glmboost, "y ~ x + z", data=incomplete, na_action="impute",
                   control=mb.boost_control(mstop=20))
    message = explore._missing_predictor_message(model, ["x_missing"])
    assert "missing-value indicator" in message
    assert "generates from 'x'" in message


def test_the_indicator_rule_is_documented_where_imputation_is_documented():
    assert "<x>_missing_" in mb.gamboost.__doc__
    assert "<x>_missing_" in mb.glmboost.__doc__
    assert "<x>_missing_" in mb.transforms.impute_missing.__doc__


# ─── item 8: distributional sensitivity grid and term names ───────────


def test_sigma_sensitivity_uses_the_mu_columns_when_the_data_has_them(roof):
    lss = _quiet(mb.gamboostLSS,
                 {"mu": "y ~ bbs(x, df=4) + bols(roof_class)", "sigma": "y ~ bbs(z, df=4)"},
                 data=roof, family=mb.GaussianLSS(), control=mb.boost_control(40, 0.1))
    rows = _quiet(mb.sensitivity, lss, roof.drop("y"), parameter="sigma")
    assert [row["variable"] for row in rows] == ["z"]


def test_missing_mu_column_is_named_as_the_term_the_user_wrote(roof):
    lss = _quiet(mb.gamboostLSS,
                 {"mu": "y ~ bbs(x, df=4) + bols(roof_class)", "sigma": "y ~ bbs(z, df=4)"},
                 data=roof, family=mb.GaussianLSS(), control=mb.boost_control(40, 0.1))
    with pytest.raises(ValueError) as excinfo:
        _quiet(mb.sensitivity, lss, roof.select(["z"]), parameter="sigma")
    message = str(excinfo.value)
    assert "bols(roof_class)" in message
    assert "brandom" not in message


# ─── item 9: turns need both legs above 5% of the range ───────────────


def test_a_small_reversal_on_a_monotone_curve_is_not_a_turn():
    curve = np.linspace(0.0, 10.0, 60)
    curve[30] -= 0.2  # a 2% of the range dip inside a rise
    assert explore._has_turn(curve) is False


def test_a_rise_then_fall_larger_than_the_tolerance_is_a_turn():
    curve = np.concatenate([np.linspace(0, 10, 30), np.linspace(10, 4, 30)])
    assert explore._has_turn(curve) is True


def test_a_large_rise_with_a_small_fall_is_not_a_turn():
    curve = np.concatenate([np.linspace(0, 10, 30), np.linspace(10, 9.7, 30)])
    assert explore._has_turn(curve) is False


def test_a_plateau_at_the_peak_is_still_a_turn():
    curve = np.concatenate([np.linspace(0, 10, 20), np.full(10, 10.0),
                            np.linspace(10, 0, 20)])
    assert explore._has_turn(curve) is True


def test_a_flat_curve_has_no_turn():
    assert explore._has_turn(np.zeros(20)) is False


# ─── item 10: a failing candidate is recorded, not fatal ─────────────


@pytest.fixture(scope="module")
def plain():
    rng = np.random.default_rng(12)
    n = 240
    x = rng.uniform(0, 1, n)
    z = rng.uniform(0, 1, n)
    y = np.sin(3 * x) + 0.5 * z + rng.normal(0, 0.3, n)
    return {"x": x, "z": z, "y": y}


def test_a_failing_candidate_gets_an_error_row_and_the_rest_are_compared(plain):
    rows = _quiet(mb.compare_specifications, ["bbs(x, df=3)", "bbs(nope, df=3)",
                                              "bbs(x, df=3) + bols(z)"],
                  plain, "y", folds=3, mstop=40)
    assert [row["specification"] for row in rows] == [
        "bbs(x, df=3)", "bbs(nope, df=3)", "bbs(x, df=3) + bols(z)"]
    failed = rows[1]
    assert "bbs(nope, df=3)" in failed["error"]
    assert np.isnan(failed["out-of-fold risk"]) and np.isnan(failed["change se"])
    assert np.isfinite(rows[0]["out-of-fold risk"])
    assert np.isfinite(rows[2]["out-of-fold risk"])
    assert rows[0]["error"] is None


def test_the_comparison_raises_only_when_every_candidate_fails(plain):
    with pytest.raises(ValueError, match="no specification could be scored"):
        _quiet(mb.compare_specifications, ["bbs(nope1, df=3)", "bbs(nope2, df=3)"],
               plain, "y", folds=3, mstop=20)


# ─── items 11 and 12: candidate forms and row fields ──────────────────


@pytest.fixture(scope="module")
def sigma_data():
    rng = np.random.default_rng(21)
    n = 300
    x = rng.uniform(0, 1, n)
    z = rng.uniform(0, 1, n)
    y = np.sin(3 * x) + (0.2 + 0.5 * z) * rng.normal(0, 1, n)
    return {"x": x, "z": z, "y": y}


def test_a_mapping_of_right_hand_sides_is_scored_like_a_mapping_of_formulas(sigma_data):
    rhs_map = {"constant": {"mu": "bbs(x, df=3)", "sigma": "1"},
               "varying": {"mu": "bbs(x, df=3)", "sigma": "bbs(z, df=3)"}}
    formula_map = {"constant": {"mu": "y ~ bbs(x, df=3)", "sigma": "y ~ 1"},
                   "varying": {"mu": "y ~ bbs(x, df=3)", "sigma": "y ~ bbs(z, df=3)"}}
    by_rhs = _quiet(mb.compare_specifications, rhs_map, sigma_data, "y",
                    family=mb.GaussianLSS(), folds=3, mstop=30)
    by_formula = _quiet(mb.compare_specifications, formula_map, sigma_data, "y",
                        family=mb.GaussianLSS(), folds=3, mstop=30)
    assert [row["out-of-fold risk"] for row in by_rhs] == pytest.approx(
        [row["out-of-fold risk"] for row in by_formula])


def test_every_row_carries_rhs_and_specification(plain, sigma_data):
    rows = _quiet(mb.compare_specifications, ["bbs(x, df=3)", "bbs(x, df=3) + bols(z)"],
                  plain, "y", folds=3, mstop=30)
    for row in rows:
        assert isinstance(row["rhs"], str)
        assert row["specification"] == row["rhs"]

    rows = _quiet(mb.compare_specifications,
                  {"constant": {"mu": "bbs(x, df=3)", "sigma": "1"}},
                  sigma_data, "y", family=mb.GaussianLSS(), folds=3, mstop=30)
    assert rows[0]["rhs"] == {"mu": "bbs(x, df=3)", "sigma": "1"}
    assert rows[0]["specification"] == "constant"


# ─── item 13: tensor terms in the distributional refit ────────────────


@pytest.fixture(scope="module")
def tensor_data():
    rng = np.random.default_rng(11)
    n = 400
    x = rng.uniform(0, 1, n)
    z = rng.uniform(0, 1, n)
    y = np.sin(3 * x) * z + (0.2 + 0.3 * z) * rng.normal(0, 1, n)
    return pl.DataFrame({"x": x, "z": z, "y": y})


def test_distributional_conformal_accepts_a_tensor_term(tensor_data):
    lss = _quiet(mb.gamboostLSS,
                 {"mu": "y ~ bbs(x, df=3) %X% bbs(z, df=3)", "sigma": "y ~ bbs(z, df=3)"},
                 data=tensor_data, family=mb.GaussianLSS(),
                 control=mb.boost_control(40, 0.1))
    interval = _quiet(mb.conformal_interval, lss, newdata=tensor_data.drop("y"),
                      method="distributional", seed=2)
    assert interval.lower.shape == (tensor_data.height,)
    assert np.all(np.isfinite(interval.lower))


def test_a_tensor_refit_subset_keeps_each_design_row_with_its_data(tensor_data):
    """Refitting on a permutation of all rows reproduces the fit row for row."""
    model = _quiet(mb.gamboost, "y ~ bols(x) + bbs(x, df=3) %X% bols(z)",
                   data=tensor_data, control=mb.boost_control(30, 0.1))
    rows = np.random.default_rng(0).permutation(tensor_data.height)
    rebuilt = _quiet(lambda: refit_spec(model).subset(rows).build())
    np.testing.assert_allclose(rebuilt.fitted(), np.asarray(model.fitted())[rows],
                               atol=1e-8)


# ─── items 14, 15, 16: what conformal_interval reports ────────────────


@pytest.fixture(scope="module")
def conformal_fit(plain):
    return _quiet(mb.gamboost, "y ~ bbs(x, df=3) + bols(z)", data=plain,
                  control=mb.boost_control(60, 0.1))


def test_fitted_mean_is_the_full_model_prediction(conformal_fit, plain):
    interval = _quiet(mb.conformal_interval, conformal_fit, level=0.9, seed=1)
    np.testing.assert_allclose(interval.fitted_mean,
                               np.asarray(conformal_fit.predict(type="response")).ravel())


def test_fitted_mean_follows_newdata(conformal_fit, plain):
    new = {"x": plain["x"][:25], "z": plain["z"][:25]}
    interval = _quiet(mb.conformal_interval, conformal_fit, new, level=0.9, seed=1)
    np.testing.assert_allclose(
        interval.fitted_mean,
        np.asarray(conformal_fit.predict(new, type="response")).ravel())


def test_to_frame_shares_prediction_interval_names(conformal_fit):
    frame = _quiet(mb.conformal_interval, conformal_fit, level=0.9, seed=1).to_frame()
    for name in ("predicted", "lower", "upper", "width", "median", "level", "fitted_mean"):
        assert name in frame.columns
    np.testing.assert_array_equal(np.asarray(frame["median"]), np.asarray(frame["predicted"]))
    np.testing.assert_allclose(np.asarray(frame["level"]), 0.9)


def test_guarantee_text_describes_the_band_in_words(roof):
    gam = _quiet(mb.gamboost, "y ~ bbs(z, df=3)", data=roof,
                 control=mb.boost_control(40, 0.1))
    lss = _quiet(mb.gamboostLSS,
                 {"mu": "y ~ bbs(x, df=3)", "sigma": "y ~ bbs(z, df=3)"},
                 data=roof, family=mb.GaussianLSS(), control=mb.boost_control(40, 0.1))
    for interval in (
            _quiet(mb.conformal_interval, gam, level=0.9, seed=1),
            _quiet(mb.conformal_interval, gam, level=0.9, seed=1, method="cqr"),
            _quiet(mb.conformal_interval, lss, level=0.9, seed=1, method="distributional")):
        assert "[" not in interval.guarantee
        assert "\n" not in interval.guarantee
        assert "coverage" in interval.guarantee


# ─── item 17: one extrapolation summary per routine call ──────────────


@pytest.fixture(scope="module")
def outlier_data():
    """Row 0 is far outside the rest of x, so the fold that holds it out predicts outside."""
    rng = np.random.default_rng(31)
    n = 200
    x = rng.uniform(0, 10, n)
    x[0] = 100.0
    y = np.sin(x / 3) + rng.normal(0, 0.2, n)
    return {"x": x, "y": y}


def _extrapolation_warnings(record):
    return [w for w in record if issubclass(w.category, ExtrapolationWarning)]


def test_cv_risk_issues_one_summary_naming_folds_and_learners(outlier_data):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        explore.cv_risk("bbs(x, df=4)", outlier_data, "y", folds=5, mstop=30, seed=1)
    found = _extrapolation_warnings(record)
    assert len(found) == 1
    assert "1 fold prediction" in str(found[0].message)
    assert "1 learner(s)" in str(found[0].message)


def test_compare_specifications_issues_one_summary_for_all_candidates(outlier_data):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        mb.compare_specifications(["bbs(x, df=4)", "bbs(x, df=6)", "bbs(x, df=3)"],
                                  outlier_data, "y", folds=5, mstop=30, seed=1)
    found = _extrapolation_warnings(record)
    assert len(found) == 1
    assert "3 fold prediction" in str(found[0].message)


# ─── item 18: validate groups unused columns into one issue ───────────


def test_validate_lists_all_unused_columns_in_one_issue(incomplete):
    model = _quiet(mb.glmboost, "y ~ z", data=incomplete.drop_nulls(),
                   control=mb.boost_control(mstop=20))
    newdata = incomplete.drop_nulls().head(5).select(["z"]).with_columns(
        junk_a=pl.lit(1.0), junk_b=pl.lit(2.0), junk_c=pl.lit(3.0))
    issues = mb.validate(model, newdata, output="records")
    extra = [issue for issue in issues if issue["kind"] == "extra_column"]
    assert len(extra) == 1
    assert extra[0]["count"] == 3
    for name in ("junk_a", "junk_b", "junk_c"):
        assert name in extra[0]["message"]
