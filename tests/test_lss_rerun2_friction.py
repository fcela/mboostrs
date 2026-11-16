"""Distributional-model fixes from the rerun 2 claim-cost friction log.

Covers: `tune_mstop` on a mapping specification returns the fitted model
(item 3); `gam_refit` names the terms that share unpenalized directions
(item 8); `conformal_interval(method="distributional")` re-estimates fitted
transforms on the proper-training rows (item 11); `gamlss_refit` supports
intercept-only parameters (item 13); `cvrisk` accepts a flat grid of step
counts (item 14).
"""

import numpy as np
import pytest

import mboostrs as mb
from mboostrs.conformal import _preprocessing_refit_spec
from mboostrs.gamlss import _lss_grid_points
from mboostrs.selection import _null_sharing_message


def _frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = 1.0 + 0.5 * x + np.exp(0.3 * z) * rng.normal(size=n)
    return {"y": y, "x": x, "z": z}


# ---------------------------------------------------------------- item 3


def test_tune_mstop_returns_the_distributional_model_at_the_chosen_step():
    data = _frame()
    formulas = {"mu": "y ~ bbs(x)", "sigma": "y ~ bols(z)"}
    tuning = mb.tune_mstop(formulas, data, mb.GaussianLSS(), folds=3, start=40,
                           nu=0.1, max_mstop=160, seed=2)
    assert tuning.model is not None
    assert type(tuning.model).__name__ == "LSSModel"
    reference = mb.gamboostLSS(formulas, data=data, family=mb.GaussianLSS(),
                               control=mb.boost_control(tuning.mstop, 0.1))
    np.testing.assert_allclose(np.asarray(tuning.model.predict(parameter="mu")).ravel(),
                               np.asarray(reference.predict(parameter="mu")).ravel(),
                               rtol=0, atol=1e-10)


# ---------------------------------------------------------------- item 8


def test_gam_refit_names_the_two_terms_with_identical_columns():
    data = _frame()
    data["z"] = data["x"].copy()
    data["w"] = np.random.default_rng(5).normal(size=data["x"].size)
    model = mb.gamboost("y ~ bols(x) + bbs(z) + bbs(w)", data=data,
                        control=mb.boost_control(60, 0.1))
    with pytest.warns(RuntimeWarning, match="unpenalized direction") as record:
        mb.gam_refit(model, terms=["bols(x)", "bbs(z)", "bbs(w)"])
    message = str(record[0].message)
    assert "'bols(x)'" in message and "'bbs(z)'" in message
    assert "'bbs(w)'" not in message.split("Remedy")[0]
    assert "drop one of the named terms" in message
    assert "drop the main effect" not in message


def test_null_sharing_message_drops_the_main_effect_a_tensor_contains():
    terms = ("bols(x)", "bbs(x) %X% bbs(z)")
    message = _null_sharing_message(2, terms, terms)
    assert "drop the main effect 'bols(x)'" in message
    assert "'bbs(x) %X% bbs(z)' already contains its linear part" in message
    assert "held 2 unpenalized direction(s)" in message


def test_null_sharing_message_with_no_named_terms_gives_the_duplicate_column_remedy():
    message = _null_sharing_message(1, (), ())
    assert "held 1 unpenalized direction(s)" in message
    assert "drop one of them, or drop a duplicate column" in message


# --------------------------------------------------------------- item 11


def _impute_lss_fit(seed=0, n=600):
    rng = np.random.default_rng(seed)
    ave = rng.lognormal(1.0, 0.4, n)
    income = rng.lognormal(0.5, 0.5, n)
    households = rng.lognormal(3.0, 0.8, n)
    y = 2 * np.log(ave) + np.log(income) + 0.3 * rng.normal(size=n)
    ave[rng.uniform(size=n) < 0.1] = np.nan
    data = {"y": y, "ave_occup": ave, "median_income": income, "households": households}
    fit = mb.gamboostLSS(
        {"mu": "y ~ bbs(log(ave_occup)) + bbs(log(median_income))",
         "sigma": "y ~ bbs(log(households))"},
        data=data, family=mb.GaussianLSS(), control=mb.boost_control(80, 0.1),
        na_action="impute")
    return data, fit


def test_impute_records_a_median_for_every_numeric_column_including_complete_ones():
    _, fit = _impute_lss_fit()
    imputed = {s.name for s in fit._transform_recipe.steps if s.kind == "impute"}
    assert imputed == {"ave_occup", "median_income", "households"}
    # A new row missing a column that was complete in training gets its median.
    new = {"ave_occup": np.array([20.0]), "median_income": np.array([np.nan]),
           "households": np.array([100.0])}
    assert np.isfinite(fit.predict(newdata=new, parameter="mu")).all()


def test_predict_refuses_a_missing_value_no_fill_was_recorded_for_without_suggesting_impute():
    from types import SimpleNamespace

    from mboostrs.transforms import TransformRecipe, _Step, refuse_unfilled_missing

    recipe = TransformRecipe((_Step(name="x", kind="impute", sources=("x",),
                                    expression="x", params=(0.0,)),))
    model = SimpleNamespace(_formula="y ~ bbs(x) + bbs(q)")
    data = {"x": np.array([1.0, np.nan]), "q": np.array([0.5, np.nan])}
    with pytest.raises(ValueError, match="Column 'q' contains 1 NaN") as caught:
        refuse_unfilled_missing(model, recipe, data)
    message = str(caught.value)
    assert "na_action='impute', but it records no fill for 'q'" in message
    assert "pass na_action" not in message
    # A column the recipe fills is not refused, and a model with no imputation
    # steps is not checked (the generic message applies there).
    refuse_unfilled_missing(model, recipe, {"x": np.array([np.nan]), "q": np.array([0.5])})
    refuse_unfilled_missing(SimpleNamespace(_formula="y ~ bbs(x)"), None, data)


def test_conformal_refusal_does_not_name_a_median_for_a_complete_column():
    rng = np.random.default_rng(3)
    n = 300
    ave = rng.lognormal(1.0, 0.4, n)
    households = rng.lognormal(3.0, 0.8, n)
    z = rng.uniform(0.5, 2.0, n)
    y = np.log(ave) + z + 0.2 * rng.normal(size=n)
    ave[rng.uniform(size=n) < 0.1] = np.nan
    data = {"y": y, "ave_occup": ave, "households": households, "z": z}
    fit = mb.gamboostLSS({"mu": "y ~ bbs(log(ave_occup)) %X% bbs(z) + bbs(log(households))",
                          "sigma": "y ~ bols(z)"},
                         data=data, family=mb.GaussianLSS(),
                         control=mb.boost_control(40, 0.1), na_action="impute")
    with pytest.raises(NotImplementedError) as caught:
        mb.conformal_interval(fit, method="distributional", level=0.9, seed=1)
    message = str(caught.value)
    assert "impute(households)" not in message
    assert "did not keep them" in message


def test_distributional_conformal_works_with_impute_and_log():
    _, fit = _impute_lss_fit()
    interval = mb.conformal_interval(fit, method="distributional", level=0.9, seed=1)
    assert np.isfinite(interval.lower).all() and np.isfinite(interval.upper).all()
    assert "impute(ave_occup)" in interval.calibration_source
    assert "re-estimated on the proper-training rows" in interval.calibration_source
    new = {"ave_occup": np.array([np.nan, 20.0]),
           "median_income": np.array([3.0, 3.0]),
           "households": np.array([100.0, 100.0])}
    scored = mb.conformal_interval(fit, newdata=new, method="distributional", level=0.9, seed=1)
    assert np.isfinite(scored.lower).all() and np.isfinite(scored.upper).all()


def test_the_refit_estimates_the_median_from_its_own_rows():
    data, fit = _impute_lss_fit()
    rows = np.arange(300)
    subset_median = float(np.nanmedian(np.asarray(data["ave_occup"])[rows]))
    full_median = float(np.nanmedian(np.asarray(data["ave_occup"])))
    assert abs(subset_median - full_median) > 1e-6
    spec = _preprocessing_refit_spec(fit, [s for s in fit._transform_recipe.steps
                                            if s.kind == "impute"])
    refit = spec.subset(rows, refit_preprocessing=True).build()
    impute = [s for s in refit._transform_recipe.steps
              if s.kind == "impute" and s.name == "ave_occup"]
    assert impute[0].params[0] == pytest.approx(subset_median)


def test_a_transform_whose_input_is_missing_is_the_only_one_named():
    from types import SimpleNamespace

    _, fit = _impute_lss_fit()
    stranded = SimpleNamespace(kind="log", name="log__orphan", sources=("orphan",))
    with pytest.raises(NotImplementedError) as caught:
        _preprocessing_refit_spec(fit, [stranded])
    message = str(caught.value)
    assert "log(log__orphan)" in message
    assert "'orphan'" in message
    assert "ave_occup" not in message


def test_a_model_without_retained_inputs_refuses_naming_the_reason():
    rng = np.random.default_rng(3)
    n = 300
    x = rng.uniform(0.5, 2.0, n)
    z = rng.uniform(0.5, 2.0, n)
    w = rng.uniform(1.0, 3.0, n)
    y = np.log(x) + z + 0.2 * rng.normal(size=n)
    data = {"y": y, "x": x, "z": z, "w": w}
    fit = mb.gamboostLSS({"mu": "y ~ bbs(x) %X% bbs(z) + bbs(log(w))",
                          "sigma": "y ~ bols(z)"},
                         data=data, family=mb.GaussianLSS(),
                         control=mb.boost_control(40, 0.1))
    with pytest.raises(NotImplementedError, match="did not keep them"):
        mb.conformal_interval(fit, method="distributional", level=0.9, seed=1)


# --------------------------------------------------------------- item 13


def test_gamlss_refit_keeps_an_intercept_only_parameter_and_says_so():
    data = _frame()
    fit = mb.gamboostLSS({"mu": "y ~ 1", "sigma": "y ~ bols(z)"}, data=data,
                         family=mb.GaussianLSS(), control=mb.boost_control(100, 0.1))
    for which in ("selected", "all"):
        refit = mb.gamlss_refit(fit, which=which)
        assert refit.intercept_only == ("mu",)
        assert refit.terms["mu"] == ()
        assert set(refit.intercepts) == {"mu", "sigma"}
        assert refit.intercepts["mu"]["se"] > 0


def test_gamlss_refit_intercept_only_for_both_parameters_matches_the_sample_moments():
    data = _frame()
    y = np.asarray(data["y"])
    fit = mb.gamboostLSS("y ~ 1", data=data, family=mb.GaussianLSS(),
                         control=mb.boost_control(100, 0.1))
    refit = mb.gamlss_refit(fit, which="all")
    assert refit.intercept_only == ("mu", "sigma")
    assert refit.intercepts["mu"]["estimate"] == pytest.approx(y.mean(), abs=1e-3)
    assert refit.intercepts["mu"]["se"] == pytest.approx(y.std() / np.sqrt(y.size), rel=0.02)
    assert refit.intercepts["sigma"]["estimate"] == pytest.approx(np.log(y.std()), abs=1e-2)


def test_gamlss_refit_accepts_an_intercept_term_next_to_a_smooth():
    data = _frame()
    fit = mb.gamboostLSS({"mu": "y ~ 1 + bbs(x)", "sigma": "y ~ bols(z)"}, data=data,
                         family=mb.GaussianLSS(), control=mb.boost_control(100, 0.1))
    refit = mb.gamlss_refit(fit, which="selected")
    assert refit.intercept_only == ()
    assert refit.terms["mu"] == ("bbs(x)",)


# --------------------------------------------------------------- item 14


def test_cvrisk_accepts_a_flat_list_as_a_common_grid():
    data = _frame(n=300)
    fit = mb.gamboostLSS({"mu": "y ~ bbs(x)", "sigma": "y ~ bols(z)"}, data=data,
                         family=mb.GaussianLSS(), control=mb.boost_control(300, 0.1))
    folds = mb.cv_weights(300, type="kfold", B=3, seed=1)
    result = fit.cvrisk(folds=folds, grid=[100, 250, 300])
    assert result.grid == [[100, 100], [250, 250], [300, 300]]
    assert list(result.optimal_mstop) in ([100, 100], [250, 250], [300, 300])


def test_a_grid_point_with_the_wrong_length_names_make_grid_lss():
    with pytest.raises(ValueError, match="make_grid_lss") as caught:
        _lss_grid_points([100, (250, 250, 250)], 2)
    assert "one step count per distribution parameter" in str(caught.value)
    assert "(100, 100)" in str(caught.value)


def test_lss_grid_points_keeps_tuples_and_flat_counts_apart():
    assert _lss_grid_points([100, 250], 2) == [[100, 100], [250, 250]]
    assert _lss_grid_points([(100, 50), (250, 120)], 2) == [[100, 50], [250, 120]]
    assert _lss_grid_points(np.array([7, 9]), 3) == [[7, 7, 7], [9, 9, 9]]
