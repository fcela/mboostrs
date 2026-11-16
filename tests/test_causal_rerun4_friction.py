"""Friction items from the causal benchmark rerun 4 (issue #444).

One group of tests per item: ci_test missing outcomes (1), the dependent status
of ci_test (2), categorical columns in aipw (3), the smooth argument (4), the
return type of EffectModel.predict (5), a fixed step count per nuisance (6), the
heterogeneity warnings (7), and repeated warnings (8).
"""

import warnings

import numpy as np
import pytest

import mboostrs as mb
from mboostrs._extrapolation import ExtrapolationWarning, reset_shown_summaries


@pytest.fixture(autouse=True)
def _fresh_summary_state():
    reset_shown_summaries()
    yield
    reset_shown_summaries()


def _spec(covariates=("x1", "x2")):
    return mb.CausalSpec(
        treatment="a", outcome="y", covariates=tuple(covariates),
        time_zero="assignment", follow_up="one year",
        adjustment_rationale="the covariates are the common causes",
    )


def _confounded(n=500, seed=1, region=False):
    """Treatment and outcome both depend on x1 and x2; the effect is 1 + x1.

    With ``region`` a four-level string column enters the treatment and the
    outcome."""
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=(2, n))
    data = {"x1": x1, "x2": x2}
    shift = 0.0
    if region:
        levels = np.array(["north", "south", "east", "west"], dtype=object)
        labels = levels[rng.integers(0, 4, n)]
        shift = np.where(labels == "north", 0.6, np.where(labels == "west", -0.4, 0.0))
        data["region"] = labels.tolist()
    a = rng.binomial(1, 1 / (1 + np.exp(-(0.8 * x1 - 0.5 * x2 + shift))))
    y = x1 + 0.5 * x2 + a * (1 + x1) + rng.normal(size=n)
    data.update({"a": a, "y": y})
    return data


# Item 1: ci_test with outcome_missing.

def _missing_outcome_data(seed=3, n=600, dependent=False):
    """y is observed with a probability that depends on z, so dropping the
    unobserved rows conditions on that observation. Without ``dependent`` x is
    independent of y given z; with it, y also depends on x."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    x = rng.normal(size=n) + 0.5 * z
    y = z + (x if dependent else 0.0) + rng.normal(size=n)
    observed = rng.binomial(1, 1 / (1 + np.exp(-(0.5 + z))))
    return {"x": x, "y": np.where(observed == 1, y, np.nan), "z": z}


def test_ci_test_refuses_a_missing_outcome_and_names_both_remedies():
    data = _missing_outcome_data()
    missing = int(np.isnan(data["y"]).sum())
    with pytest.raises(ValueError) as caught:
        mb.ci_test(data, "x", "y", given=("z",), folds=4)
    message = str(caught.value)
    assert f"{missing} missing values" in message
    assert "outcome_missing='ipw'" in message
    assert "drop" in message
    assert "bias" in message
    assert "post-treatment" in message


def test_ci_test_ipw_keeps_every_row_and_reports_the_missing_count():
    data = _missing_outcome_data()
    missing = int(np.isnan(data["y"]).sum())
    result = mb.ci_test(data, "x", "y", given=("z",), folds=4, outcome_missing="ipw")
    assert result.status == "tested"
    assert result.skipped is False
    assert result.outcome_missing == "ipw"
    assert result.n_missing_outcome == missing
    assert result.n == len(data["y"])
    assert any("weights each observed row" in note for note in result.notes)


def test_ci_test_ipw_with_a_complete_outcome_matches_the_default():
    data = _missing_outcome_data()
    data["y"] = np.nan_to_num(data["y"], nan=0.0)
    default = mb.ci_test(data, "x", "y", given=("z",), folds=4)
    ipw = mb.ci_test(data, "x", "y", given=("z",), folds=4, outcome_missing="ipw")
    assert ipw.statistic == default.statistic
    assert ipw.p_value == default.p_value
    assert ipw.n_missing_outcome == 0


def test_ci_test_ipw_detects_a_dependence_among_the_rows_with_an_outcome():
    dependent = mb.ci_test(_missing_outcome_data(dependent=True), "x", "y", given=("z",),
                           folds=4, outcome_missing="ipw")
    independent = mb.ci_test(_missing_outcome_data(), "x", "y", given=("z",), folds=4,
                             outcome_missing="ipw")
    assert dependent.p_value < 0.01
    assert independent.p_value > 0.05


def test_ci_test_still_refuses_a_missing_x_under_ipw():
    data = _missing_outcome_data()
    data["x"] = np.where(np.arange(data["x"].size) < 5, np.nan, data["x"])
    with pytest.raises(ValueError, match="'x'"):
        mb.ci_test(data, "x", "y", given=("z",), folds=4, outcome_missing="ipw")


def test_ci_test_rejects_an_unknown_outcome_missing_value():
    with pytest.raises(ValueError, match="outcome_missing must be 'error' or 'ipw'"):
        mb.ci_test(_missing_outcome_data(), "x", "y", given=("z",), outcome_missing="drop")


# Item 2: an exact linear dependence is reported as dependent, not as a test.

def _dependent_data(n=400, seed=4):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    return {"x": rng.normal(size=n), "y": rng.normal(size=n), "z": z,
            "w": 2.0 * z + 1.0}


def test_an_exactly_dependent_x_is_flagged_as_dependent_and_not_tested():
    result = mb.ci_test(_dependent_data(), "w", "y", given=("z",), folds=4)
    assert result.status == "dependent"
    assert result.skipped is True
    assert np.isnan(result.p_value)
    assert np.isnan(result.statistic)
    text = result.describe()
    assert text.startswith("UNDETERMINED")
    assert "no test was run" in text
    assert "placeholder" not in text
    assert repr(result).startswith("CITest(status='dependent'")


def test_an_exactly_dependent_y_is_flagged_as_dependent():
    data = _dependent_data()
    data["y"] = 3.0 * data["z"] - 2.0
    result = mb.ci_test(data, "x", "y", given=("z",), folds=4)
    assert result.status == "dependent"
    assert result.skipped is True
    assert "y" in result.describe()


def test_a_tested_result_says_so_and_does_not_read_as_undetermined():
    rng = np.random.default_rng(6)
    data = {"x": rng.normal(size=300), "y": rng.normal(size=300), "z": rng.normal(size=300)}
    result = mb.ci_test(data, "x", "y", given=("z",), folds=4)
    assert result.status == "tested"
    assert result.skipped is False
    assert not result.describe().startswith("UNDETERMINED")
    assert repr(result).startswith("CITest(status='tested'")


# Item 3: aipw encodes a categorical baseline itself on every path.

@pytest.mark.parametrize("smooth", [False, True])
@pytest.mark.parametrize("tune", [False, True])
def test_aipw_runs_with_a_four_level_string_column(smooth, tune):
    data = _confounded(n=400, region=True)
    result = mb.aipw(data, _spec(("x1", "x2", "region")), smooth=smooth, tune=tune,
                     folds=4, mstop=100, max_mstop=400, seed=1)
    assert np.isfinite(result.estimate)
    assert np.isfinite(result.standard_error)


# Item 4: the smooth argument reaches the fitted propensity.

def test_smooth_changes_the_fitted_propensity_under_propensity_learner_auto():
    data = _confounded(n=600)
    kwargs = dict(tune=False, mstop=150, folds=4, seed=2, propensity_learner="auto")
    smooth_fit = mb.aipw(data, _spec(), smooth=True, **kwargs)
    linear_fit = mb.aipw(data, _spec(), smooth=False, **kwargs)
    assert not np.allclose(smooth_fit.propensity, linear_fit.propensity)
    deviances_smooth = smooth_fit.diagnostics["propensity_learner"]["out_of_fold_deviance"]
    deviances_linear = linear_fit.diagnostics["propensity_learner"]["out_of_fold_deviance"]
    assert deviances_smooth["smooth+trees"] != deviances_linear["smooth+trees"]


def test_the_linear_propensity_formula_is_spelled_with_bols():
    result = mb.aipw(_confounded(n=300), _spec(), smooth=False, tune=False, mstop=80,
                     folds=3, seed=2)
    assert result.nuisance["terms"]["propensity"] == "bols(x1) + bols(x2)"


# Item 5: the return type of EffectModel.predict.

@pytest.fixture(scope="module")
def effect_fit():
    rng = np.random.default_rng(12)
    n = 900
    x1, x2, x3 = rng.uniform(-1.5, 1.5, (3, n))
    a = rng.binomial(1, 1 / (1 + np.exp(-(0.8 * x1 + 0.6 * x2)))).astype(float)
    tau = 1 + 0.8 * x1
    y = np.sin(x1) + 0.7 * x2 + tau * a + rng.normal(0, 0.5, n)
    data = {"a": a, "y": y, "x1": x1, "x2": x2, "x3": x3}
    fit = mb.effect_model(data, _spec(("x1", "x2", "x3")), seed=0, terms="smooth",
                          learner="r")
    new = {"x1": x1[:40], "x2": x2[:40], "x3": x3[:40]}
    return fit, new


def test_predict_without_interval_returns_an_array_by_default(effect_fit):
    fit, new = effect_fit
    values = fit.predict(new)
    assert isinstance(values, np.ndarray)
    assert values.shape == (40,)


def test_predict_numpy_output_is_the_same_array(effect_fit):
    fit, new = effect_fit
    np.testing.assert_allclose(fit.predict(new, output="numpy"), fit.predict(new))


def test_predict_frame_output_has_an_estimate_column_equal_to_the_array(effect_fit):
    pytest.importorskip("polars")
    fit, new = effect_fit
    frame = fit.predict(new, output="polars")
    assert "estimate" in frame.columns
    np.testing.assert_allclose(frame["estimate"].to_numpy(), fit.predict(new))


def test_predict_with_interval_returns_a_frame_with_an_estimate_column(effect_fit):
    pytest.importorskip("polars")
    fit, new = effect_fit
    frame = fit.predict(new, interval=True)
    assert "estimate" in frame.columns
    np.testing.assert_allclose(frame["estimate"].to_numpy(), fit.predict(new))


def test_predict_with_interval_and_numpy_output_returns_arrays_by_name(effect_fit):
    fit, new = effect_fit
    arrays = fit.predict(new, interval=True, output="numpy")
    assert isinstance(arrays, dict)
    assert arrays["estimate"].shape == (40,)
    assert np.all(arrays["lower"] <= arrays["upper"])


def test_predict_refuses_an_unknown_output(effect_fit):
    fit, new = effect_fit
    with pytest.raises(ValueError, match="output must be"):
        fit.predict(new, output="json")


# Item 6: one step count per nuisance, refit without a second tuned pass.

def test_the_tuned_step_summary_refits_each_nuisance_at_its_fixed_steps():
    data = _confounded(n=300)
    tuned = mb.aipw(data, _spec(), tune=True, mstop=60, max_mstop=240, folds=3,
                    seed=5, propensity_learner="smooth", outcome_model="auto")
    summary = tuned.diagnostics["nuisance_mstop_summary"]
    assert summary and all(isinstance(v, int) and v >= 1 for v in summary.values())
    fixed = mb.aipw(data, _spec(), tune=False, mstop=summary, folds=3, seed=5,
                    propensity_learner="smooth", outcome_model="auto")
    for name, steps in summary.items():
        record = fixed.diagnostics["nuisance_mstop"][name]
        assert record["first_grid"] == steps
        assert set(record["mstop_by_fold"]) == {steps}


def test_a_step_mapping_needs_tune_false():
    with pytest.raises(ValueError, match="tune=False"):
        mb.aipw(_confounded(n=200), _spec(), tune=True, mstop={"propensity": 50},
                folds=3)


def test_a_step_mapping_names_only_known_nuisances():
    with pytest.raises(ValueError, match="unknown nuisance"):
        mb.aipw(_confounded(n=200), _spec(), tune=False, mstop={"propensity_model": 50},
                folds=3)


def test_a_step_mapping_must_cover_every_nuisance_the_call_fits():
    with pytest.raises(ValueError, match="no step count for the 'outcome' nuisance"):
        mb.aipw(_confounded(n=200), _spec(), tune=False, mstop={"propensity": 50},
                folds=3, outcome_model="pooled")


# Item 7: the heterogeneity warnings point to each other.

def test_effect_modifiers_describe_says_what_its_tests_are_about():
    rng = np.random.default_rng(14)
    n = 800
    x1, x2 = rng.normal(size=(2, n))
    a = rng.binomial(1, 1 / (1 + np.exp(-(0.6 * x1)))).astype(float)
    data = {"a": a, "y": x1 + a * (1 + x1) + rng.normal(size=n), "x1": x1, "x2": x2}
    fit = mb.aipw(data, _spec(), tune=False, mstop=120, folds=4, seed=3)
    text = mb.effect_modifiers(fit, data).describe()
    assert "average effect changes along a covariate" in text
    assert "R-learner gate" in text


def test_the_constant_effect_warning_names_effect_modifiers():
    rng = np.random.default_rng(21)
    n = 700
    x1, x2 = rng.normal(size=(2, n))
    a = rng.binomial(1, 1 / (1 + np.exp(-(0.6 * x1)))).astype(float)
    data = {"a": a, "y": x1 + 1.0 * a + rng.normal(size=n), "x1": x1, "x2": x2}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mb.effect_model(data, _spec(), seed=0, terms="smooth", learner="r")
    texts = [str(w.message) for w in caught if "does not beat a constant effect" in str(w.message)]
    if not texts:
        pytest.skip("this sample does not trigger the constant-effect warning")
    assert "effect_modifiers" in texts[0]
    assert "low power" in texts[0]


# Item 8: repeated warnings.

def _extrapolating_data(n=200, seed=5):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    z[0] = 50.0  # the only row far above the rest; its fold trains without it
    return {"x": rng.normal(size=n), "y": rng.normal(size=n), "z": z}


def test_ci_test_extrapolation_summary_is_shown_once_across_calls():
    data = _extrapolating_data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for seed in range(3):
            mb.ci_test(data, "x", "y", given=("z",), folds=2, seed=seed)
    extrapolation = [w for w in caught if issubclass(w.category, ExtrapolationWarning)]
    assert len(extrapolation) == 1


def test_collect_warnings_shows_a_repeated_warning_once_with_its_count():
    data = _extrapolating_data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with mb.collect_warnings():
            for seed in range(3):
                mb.ci_test(data, "x", "y", given=("z",), folds=2, seed=seed)
    extrapolation = [w for w in caught if issubclass(w.category, ExtrapolationWarning)]
    assert len(extrapolation) == 1
    assert "raised 3 times" in str(extrapolation[0].message)
    assert extrapolation[0].message.rows  # the rows of the first call are kept


def test_collect_warnings_passes_a_single_warning_through_unchanged():
    data = _extrapolating_data()
    with warnings.catch_warnings(record=True) as plain:
        warnings.simplefilter("always")
        mb.ci_test(data, "x", "y", given=("z",), folds=2, seed=0)
    with warnings.catch_warnings(record=True) as collected:
        warnings.simplefilter("always")
        with mb.collect_warnings():
            mb.ci_test(data, "x", "y", given=("z",), folds=2, seed=0)
    assert [str(w.message) for w in collected] == [str(w.message) for w in plain]
