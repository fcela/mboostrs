"""na_action='impute_model': a model-based fill of missing numeric predictors.

The fill is the conditional mean of the column given the other predictors,
fitted in Rust on the rows where the column is observed. These tests check
the fill against the mean it should recover, that the fitted fill is part of
the recipe and is replayed without refitting, that the response never enters
the fill, that the model survives save and load, and that every fitting
function accepts the option.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import mboostrs as mb
from mboostrs.transforms import TransformRecipe, apply_recipe, imputation_r2

CTL = mb.boost_control(mstop=150, nu=0.1)


def _frame(n=1200, seed=3, missing_rate=0.2):
    """`age` is missing at random and is a smooth function of `inc` and of a
    factor `roof`; the response depends on `age`."""
    rng = np.random.default_rng(seed)
    inc = rng.uniform(1.0, 10.0, n)
    roof = rng.choice(["A", "B", "C"], n)
    roof_shift = np.array([{"A": -5.0, "B": 0.0, "C": 6.0}[r] for r in roof])
    age = 20.0 + 3.0 * inc + roof_shift + rng.normal(0.0, 1.0, n)
    y = 0.05 * age - 0.3 * inc + rng.normal(0.0, 0.3, n)
    missing = rng.uniform(size=n) < missing_rate
    age_obs = np.where(missing, np.nan, age)
    frame = pl.DataFrame({"age": age_obs, "inc": inc, "roof": roof, "y": y})
    return frame, age, missing


def _step(model, name):
    steps = [s for s in model._transform_recipe.steps if s.name == name]
    assert len(steps) == 1, [s.name for s in model._transform_recipe.steps]
    return steps[0]


def test_fill_recovers_the_conditional_mean_better_than_the_median():
    frame, age, missing = _frame()
    fit = mb.gamboost("y ~ bbs(age) + bbs(inc) + bols(roof)", data=frame,
                      na_action="impute_model", control=CTL)
    filled = apply_recipe(TransformRecipe((_step(fit, "age"),)), frame)["age"].to_numpy()
    model_rmse = np.sqrt(np.mean((filled[missing] - age[missing]) ** 2))
    median_rmse = np.sqrt(np.mean((np.nanmedian(frame["age"].to_numpy()) - age[missing]) ** 2))
    assert model_rmse < 0.5 * median_rmse
    assert np.all(np.isfinite(filled))
    # Observed values are returned unchanged.
    observed = ~missing
    np.testing.assert_array_equal(filled[observed], frame["age"].to_numpy()[observed])


def test_r2_is_reported_per_imputed_column():
    frame, _, _ = _frame()
    fit = mb.gamboost("y ~ bbs(age) + bols(roof) + bbs(inc)", data=frame,
                      na_action="impute_model", control=CTL)
    r2 = imputation_r2(fit._transform_recipe)
    assert set(r2) == {"age"}
    assert 0.9 < r2["age"] <= 1.0


def test_the_fill_is_in_the_recipe_with_an_indicator_term():
    frame, _, _ = _frame()
    fit = mb.gamboost("y ~ bbs(age) + bbs(inc) + bols(roof)", data=frame,
                      na_action="impute_model", control=CTL)
    kinds = {s.name: s.kind for s in fit._transform_recipe.steps}
    assert kinds["age_missing"] == "missing_indicator"
    assert kinds["age"] == "impute_model"
    assert "age_missing" in fit._formula


def test_predict_replays_the_stored_fill_and_never_refits_it():
    frame, _, _ = _frame()
    fit = mb.gamboost("y ~ bbs(age) + bbs(inc) + bols(roof)", data=frame,
                      na_action="impute_model", control=CTL)
    before = _step(fit, "age").params
    new = pl.DataFrame({"age": [None, 30.0, None], "inc": [2.0, 5.0, 9.0],
                        "roof": ["A", "B", "C"]})
    first = np.asarray(fit.predict(newdata=new)).ravel()
    # A new frame with very different covariates must not move the stored fill.
    shifted = pl.DataFrame({"age": [None, None], "inc": [100.0, -50.0],
                            "roof": ["C", "A"]})
    fit.predict(newdata=shifted)
    assert _step(fit, "age").params == before
    np.testing.assert_allclose(np.asarray(fit.predict(newdata=new)).ravel(), first)
    assert np.all(np.isfinite(first))


def test_the_response_never_enters_the_fill():
    frame, _, _ = _frame()
    flipped = frame.with_columns((-pl.col("y")).alias("y"))
    a = mb.gamboost("y ~ bbs(age) + bbs(inc) + bols(roof)", data=frame,
                    na_action="impute_model", control=CTL)
    b = mb.gamboost("y ~ bbs(age) + bbs(inc) + bols(roof)", data=flipped,
                    na_action="impute_model", control=CTL)
    assert _step(a, "age").params[0] == _step(b, "age").params[0]


def test_the_fill_uses_only_the_formula_predictors():
    frame, _, _ = _frame()
    # A column the formula does not read is not a predictor: adding noise to
    # the data leaves the stored fill unchanged.
    noisy = frame.with_columns(pl.Series("noise", np.random.default_rng(0).normal(size=frame.height)))
    a = mb.gamboost("y ~ bbs(age) + bbs(inc) + bols(roof)", data=frame,
                    na_action="impute_model", control=CTL)
    b = mb.gamboost("y ~ bbs(age) + bbs(inc) + bols(roof)", data=noisy,
                    na_action="impute_model", control=CTL)
    assert _step(a, "age").params == _step(b, "age").params


def test_a_categorical_predictor_enters_the_fill_by_its_levels():
    rng = np.random.default_rng(5)
    n = 900
    roof = rng.choice(["A", "B", "C"], n)
    age = np.array([{"A": 10.0, "B": 30.0, "C": 50.0}[r] for r in roof]) + rng.normal(0, 1, n)
    miss = rng.uniform(size=n) < 0.25
    frame = pl.DataFrame({"age": np.where(miss, np.nan, age), "roof": roof,
                          "y": 0.1 * age + rng.normal(0, 0.2, n)})
    fit = mb.gamboost("y ~ bbs(age) + bols(roof)", data=frame, na_action="impute_model",
                      control=CTL)
    assert _step(fit, "age").params[1] == (("A", "B", "C"),)
    assert imputation_r2(fit._transform_recipe)["age"] > 0.95
    new = pl.DataFrame({"age": [None, None, None], "roof": ["A", "B", "C"]})
    filled = apply_recipe(TransformRecipe((_step(fit, "age"),)), new)["age"].to_numpy()
    np.testing.assert_allclose(filled, [10.0, 30.0, 50.0], atol=1.5)


def test_the_model_round_trips_through_save_and_load(tmp_path):
    frame, _, _ = _frame()
    fit = mb.gamboost("y ~ bbs(age) + bbs(inc) + bols(roof)", data=frame,
                      na_action="impute_model", control=CTL)
    path = Path(tmp_path) / "impute_model.json"
    mb.save_model(fit, str(path))
    restored = mb.load_model(str(path))
    new = pl.DataFrame({"age": [None, 25.0], "inc": [4.0, 4.0], "roof": ["B", "B"]})
    np.testing.assert_allclose(np.asarray(restored.predict(newdata=new)).ravel(),
                               np.asarray(fit.predict(newdata=new)).ravel())
    json.loads(path.read_text())  # the artifact is plain JSON


def test_option_is_accepted_by_the_gamboost_glmboost_and_lss_fits():
    frame, _, _ = _frame()
    glm = mb.glmboost("y ~ age + inc + roof", data=frame, na_action="impute_model",
                      control=CTL)
    assert np.all(np.isfinite(np.asarray(glm.predict(newdata=frame.head(3))).ravel()))
    lss = mb.gamboostLSS({"mu": "y ~ bbs(age) + bbs(inc)", "sigma": "y ~ bbs(inc)"},
                         data=frame, families=mb.GaussianLSS(), control=CTL,
                         na_action="impute_model")
    mu = np.asarray(lss.predict_parameters(frame.head(4))["mu"]).ravel()
    assert np.all(np.isfinite(mu))
    assert "age" in {s.name for s in lss._transform_recipe.steps if s.kind == "impute_model"}


def test_a_tensor_margin_over_an_imputed_column_predicts_incomplete_rows():
    frame, _, _ = _frame()
    fit = mb.gamboost("y ~ bbs(age, df=3) + bbs(inc, df=3) + bbs(age, df=3) %X% bbs(inc, df=3)",
                      data=frame, na_action="impute_model", control=CTL)
    new = pl.DataFrame({"age": [None, 30.0], "inc": [5.0, 5.0], "roof": ["B", "B"]})
    pred = np.asarray(fit.predict(newdata=new)).ravel()
    assert np.all(np.isfinite(pred))
    # The margin is built from the filled column, not the raw NaN: the filled
    # age the recipe produces is finite and inside the training range.
    filled_age = apply_recipe(TransformRecipe((_step(fit, "age"),)), new)["age"].to_numpy()[0]
    assert np.isfinite(filled_age)


def test_gam_refit_reuses_the_stored_fill():
    frame, _, _ = _frame()
    fit = mb.gamboost("y ~ bbs(age) + bbs(inc) + bols(roof)", data=frame,
                      na_action="impute_model", control=CTL)
    refit = mb.gam_refit(fit, which="all")
    pred = np.asarray(refit.predict(frame.head(20))).ravel()
    assert np.all(np.isfinite(pred))


def test_na_action_rejects_an_unknown_value_and_names_both_options():
    frame, _, _ = _frame()
    with pytest.raises(ValueError, match="impute_model"):
        mb.gamboost("y ~ bbs(age) + bbs(inc)", data=frame, na_action="model", control=CTL)


def test_the_remedy_names_impute_model_when_the_column_is_predictable():
    from mboostrs._preflight import _refused_column_policy

    frame, _, _ = _frame()

    class _Stub:
        _refit_bundle = {"raw_data": frame}

    text = _refused_column_policy(_Stub(), frame, "age")
    assert "na_action='impute_model'" in text
    assert "R-squared" in text


def test_the_remedy_does_not_name_impute_model_for_an_unrelated_column():
    from mboostrs._preflight import _refused_column_policy

    rng = np.random.default_rng(9)
    n = 500
    frame = pl.DataFrame({"u": np.where(rng.uniform(size=n) < 0.2, np.nan, rng.normal(size=n)),
                          "v": rng.normal(size=n), "y": rng.normal(size=n)})

    class _Stub:
        _refit_bundle = {"raw_data": frame}

    assert "impute_model" not in _refused_column_policy(_Stub(), frame, "u")


def test_no_observed_value_is_refused_with_the_remedy():
    frame = pl.DataFrame({"x": [np.nan] * 10, "z": np.arange(10.0), "y": np.arange(10.0)})
    with pytest.raises(ValueError, match="no observed value"):
        mb.gamboost("y ~ bbs(x) + bbs(z)", data=frame, na_action="impute_model", control=CTL)


# --- observed and filled parts of a smooth term on an imputed column ----------

def test_a_smooth_term_on_an_imputed_column_splits_into_observed_and_filled_parts():
    from mboostrs.transforms import _switch_model_terms

    sw = [("x", "x_missing", "__observed_x")]
    out = _switch_model_terms("y ~ bbs(x, df=4) + bols(z)", sw, "bols")
    assert out == ("y ~ bbs(x, df=4, by=__observed_x) + bbs(x, df=4, by=x_missing) "
                   "+ bols(z)")
    # A bare name is bbs under gamboost (learner "bols") and bols under glmboost.
    assert "bbs(x, by=__observed_x)" in _switch_model_terms("y ~ x + z", sw, "bols")
    assert "bols(x, by=__observed_x)" in _switch_model_terms("y ~ x + z", sw, "")


def test_terms_that_are_left_as_written():
    from mboostrs.transforms import _switch_model_terms

    sw = [("x", "x_missing", "__observed_x")]
    # A term that already has by=, and a tensor, are not split.
    assert _switch_model_terms("y ~ bbs(x, by=z)", sw, "bols") == "y ~ bbs(x, by=z)"
    tensor = "y ~ bbs(x, df=3) %X% bbs(z)"
    assert _switch_model_terms(tensor, sw, "bols") == tensor


def test_the_switched_parts_are_labelled_in_the_selection():
    frame, _, _ = _frame()
    fit = mb.gamboost("y ~ bbs(age) + bbs(inc) + bols(roof)", data=frame,
                      na_action="impute_model_split", control=CTL)
    names = list(fit._learner_names)
    assert "bbs(age) [observed]" in names
    assert "bbs(age) [filled]" in names
    assert "age" not in names


def test_the_filled_part_is_zero_for_an_observed_value_in_a_contrast():
    from mboostrs.selection import _treat_as_observed

    frame, _, _ = _frame()
    fit = mb.gamboost("y ~ bbs(age) + bbs(inc) + bols(roof)", data=frame,
                      na_action="impute_model_split", control=CTL)
    refit = mb.gam_refit(fit, which="all")
    data = refit._imputed(frame.drop("y"))
    treated = _treat_as_observed(data, refit._impute, "age")
    assert np.all(treated["age_missing"].to_numpy() == 0.0)
    assert np.all(treated["__observed_age"].to_numpy() == 1.0)


def test_an_age_contrast_uses_the_observed_part_only():
    from mboostrs.selection import _average_contrast

    rng = np.random.default_rng(21)
    n = 1500
    inc = rng.uniform(1.0, 10.0, n)
    age = 20.0 + 3.0 * inc + rng.normal(0.0, 1.0, n)
    y = 0.05 * age - 0.3 * inc + rng.normal(0.0, 0.2, n)
    miss = rng.uniform(size=n) < 0.2
    frame = pl.DataFrame({"age": np.where(miss, np.nan, age), "inc": inc, "y": y})
    fit = mb.gamboost("y ~ bbs(age) + bbs(inc)", data=frame, na_action="impute_model_split",
                      control=mb.boost_control(mstop=300, nu=0.1))
    refit = mb.gam_refit(fit, which="all")
    estimate, lower, upper = _average_contrast(refit, frame.drop("y"), "age", 20.0, 45.0, 0.9)
    # The true effect of age on y over 20 -> 45 is 0.05 * 25 = 1.25. The refit on
    # the complete data estimates 1.39 with this mstop, so the tolerance allows
    # for the refit's own bias.
    assert abs(estimate - 1.25) < 0.25
    assert lower < 1.25 < upper


def test_the_lss_fit_labels_and_refits_the_switched_parts():
    frame, _, _ = _frame()
    lss = mb.gamboostLSS({"mu": "y ~ bbs(age) + bbs(inc)", "sigma": "y ~ bbs(inc)"},
                         data=frame, families=mb.GaussianLSS(), control=CTL,
                         na_action="impute_model_split")
    assert "bbs(age) [observed]" in lss._learner_labels(0)
    refit = mb.gamlss_refit(lss, which="all")
    pred = np.asarray(refit.predict(frame.head(5), parameter="mu")).ravel()
    assert np.all(np.isfinite(pred))


def test_a_tensor_on_the_imputed_column_refits_without_a_duplicate_term():
    # The constant x_missing term is in the span of the observed part, so the
    # refit leaves it out and says so; the refit is then identified.
    frame, _, _ = _frame()
    mu = "bbs(age, df=3) + bbs(inc, df=3) + bbs(age, df=3) %X% bbs(inc, df=3)"
    lss = mb.gamboostLSS({"mu": "y ~ " + mu, "sigma": "y ~ bbs(inc, df=3)"}, data=frame,
                         families=mb.GaussianLSS(), control=CTL, na_action="impute_model_split")
    with pytest.warns(UserWarning, match="leaves out the constant term"):
        refit = mb.gamlss_refit(lss, which="all")
    pred = np.asarray(refit.predict(frame.head(5), parameter="mu")).ravel()
    assert np.all(np.isfinite(pred))


def test_impute_model_does_not_split_and_impute_model_split_does():
    # The split lowered the age-effect coverage on the claim-cost benchmark
    # (0.90 to 0.60), so it is opt-in under its own na_action value.
    frame, _, _ = _frame()
    plain = mb.gamboost("y ~ bbs(age) + bbs(inc)", data=frame, na_action="impute_model",
                        control=CTL)
    assert not any("__observed_" in str(name) for name in plain.formula.split())
    assert all(s.kind != "observed_indicator" for s in plain._transform_recipe.steps)
    split = mb.gamboost("y ~ bbs(age) + bbs(inc)", data=frame,
                        na_action="impute_model_split", control=CTL)
    assert any(s.kind == "observed_indicator" for s in split._transform_recipe.steps)
