"""Issues #487 (items 1 and 2) and #484 (S1): the defaults smooth=False and
outcome_model="cv", the small-sample rule of outcome_model="auto", and
outcome_model="cv"."""

import inspect
import warnings

import numpy as np
import polars as pl
import pytest

import mboostrs as mb
from mboostrs import causal

FAST = dict(tune=False, mstop=60, folds=3, seed=5)


def expit(x):
    return 1 / (1 + np.exp(-x))


def _spec(outcome_type="continuous"):
    return mb.CausalSpec(
        treatment="a", outcome="y", covariates=("x1", "x2"), outcome_type=outcome_type,
        target_population="simulated", unit="person", time_zero="treatment assignment",
        follow_up="one year", adjustment_rationale="x1 and x2 cause treatment and outcome")


def _data(n, seed=1, kind="continuous"):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=(2, n))
    a = rng.binomial(1, expit(0.5 * x1 - 0.3 * x2)).astype(float)
    if kind == "binary":
        y = rng.binomial(1, expit(0.4 * x1 + 0.5 * a)).astype(float)
    elif kind == "zero_inflated":
        employed = rng.random(n) < expit(0.5 + 0.5 * x1 + 0.3 * a)
        y = np.where(employed, np.exp(1.0 + 0.4 * x1 + 0.2 * a + 0.3 * rng.normal(size=n)), 0.0)
    else:
        y = x1 + 0.5 * x2 + a + rng.normal(size=n)
    return pl.DataFrame({"a": a, "y": y, "x1": x1, "x2": x2})


def _fit(data, outcome_type="continuous", **kwargs):
    options = dict(FAST)
    options.update(kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mb.aipw(data, _spec(outcome_type), **options)


# --- defaults (issue 487 item 1, revised) ---

def test_aipw_and_iv_effect_default_to_linear_nuisances():
    assert inspect.signature(mb.aipw).parameters["smooth"].default is False
    assert inspect.signature(mb.iv_effect).parameters["smooth"].default is False


def test_aipw_defaults_to_the_cv_outcome_model():
    assert inspect.signature(mb.aipw).parameters["outcome_model"].default == "cv"


def test_aipw_default_fits_linear_terms_and_smooth_true_fits_spline_terms():
    data = _data(400)
    default = _fit(data)
    assert default.nuisance["smooth"] is False
    assert "bbs" not in default.nuisance["terms"]["propensity"]
    spline = _fit(data, smooth=True)
    assert "bbs(x1" in spline.nuisance["terms"]["propensity"]


def test_default_fit_requests_cv_and_equals_the_explicit_arguments():
    data = _data(400)
    default = _fit(data)
    explicit = _fit(data, smooth=False, outcome_model="cv")
    assert default.diagnostics["outcome_model_choice"]["requested"] == "cv"
    assert default.diagnostics["outcome_model_cv"] is not None
    assert default.estimate == explicit.estimate


def test_iv_effect_default_is_linear_and_smooth_true_uses_spline_terms():
    rng = np.random.default_rng(2)
    n = 600
    x1, x2, z = rng.normal(size=(3, n))
    d = rng.binomial(1, expit(0.3 * x1 + 1.5 * z)).astype(float)
    y = 2.0 * d + x1 + rng.normal(size=n)
    frame = pl.DataFrame({"a": d, "y": y, "x1": x1, "x2": x2, "z": z})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default = mb.iv_effect(frame, _spec(), "z", tune=False, mstop=60, folds=3)
        linear = mb.iv_effect(frame, _spec(), "z", tune=False, mstop=60, folds=3, smooth=False)
        spline = mb.iv_effect(frame, _spec(), "z", tune=False, mstop=60, folds=3, smooth=True)
    assert default.estimate == linear.estimate
    assert default.estimate != spline.estimate


# --- small-sample rule of auto (issue 484, S1) ---

def test_threshold_is_the_largest_small_draw_of_the_measurement():
    assert causal._AUTO_POOLED_MAX_ROWS == 1500


@pytest.mark.parametrize("n", [600, 1500])
def test_auto_uses_pooled_without_interactions_up_to_the_threshold(n):
    result = _fit(_data(n), outcome_model="auto")
    choice = result.diagnostics["outcome_model_choice"]
    assert choice["requested"] == "auto" and choice["used"] == "pooled"
    assert choice["outcome_interactions"] is False
    assert "at most 1500" in choice["reason"]
    assert "pooled" in result.describe()


def test_auto_applies_the_sparse_arm_rule_above_the_threshold():
    result = _fit(_data(1501), outcome_model="auto")
    choice = result.diagnostics["outcome_model_choice"]
    assert choice["used"] == "separate"
    assert "flagged no propensity bin" in choice["reason"]


def test_auto_with_requested_interactions_stays_pooled_with_interactions_when_small():
    result = _fit(_data(600), outcome_model="auto", outcome_interactions=True)
    choice = result.diagnostics["outcome_model_choice"]
    assert choice["used"] == "pooled" and choice["outcome_interactions"] is True


# --- outcome_model="cv" (issue 487 item 2) ---

def test_cv_reports_every_candidate_and_uses_the_lowest_loss():
    data = _data(500)
    result = _fit(data, outcome_model="cv")
    cv = result.diagnostics["outcome_model_cv"]
    assert set(cv["candidates"]) == {"pooled", "separate"}
    assert cv["chosen"] == min(cv["loss"], key=cv["loss"].get)
    assert all(np.isfinite(v) for v in cv["loss"].values())
    choice = result.diagnostics["outcome_model_choice"]
    assert choice["requested"] == "cv" and choice["used"] == result.diagnostics["outcome_model"]
    assert "lowest out-of-fold" in choice["reason"]
    assert "outcome_model='cv'" in result.describe()
    # The estimate is that of a fit that requests the chosen model by name.
    named = _fit(data, outcome_model=cv["chosen"])
    assert result.estimate == pytest.approx(named.estimate, rel=1e-12)
    np.testing.assert_allclose(result.mu0, named.mu0, rtol=1e-12)


def test_cv_loss_is_the_mean_squared_error_of_the_cross_fitted_means():
    data = _data(400, seed=4)
    result = _fit(data, outcome_model="cv")
    cv = result.diagnostics["outcome_model_cv"]
    for label in cv["candidates"]:
        named = _fit(data, outcome_model=label)
        mu = np.where(named.treatment == 1, named.mu1, named.mu0)
        assert cv["loss"][label] == pytest.approx(
            np.mean((named.outcome - mu) ** 2), rel=1e-9)


def test_cv_includes_two_part_for_a_zero_inflated_outcome():
    result = _fit(_data(500, kind="zero_inflated"), outcome_model="cv")
    cv = result.diagnostics["outcome_model_cv"]
    assert "two_part" in cv["candidates"]
    assert set(cv["loss"]) == set(cv["candidates"])


def test_cv_works_for_a_binary_outcome_and_scores_the_deviance():
    result = _fit(_data(500, kind="binary"), outcome_type="binary", outcome_model="cv")
    cv = result.diagnostics["outcome_model_cv"]
    assert "two_part" not in cv["candidates"]
    assert "deviance" in cv["loss_name"]
    assert all(v > 0 for v in cv["loss"].values())


def test_cv_with_repeats_scores_the_averaged_cross_fitted_means():
    result = _fit(_data(300), outcome_model="cv", repeats=2)
    assert set(result.diagnostics["outcome_model_cv"]["loss"]) >= {"pooled", "separate"}


def test_non_cv_fits_report_no_cv_record():
    assert _fit(_data(300), outcome_model="auto").diagnostics["outcome_model_cv"] is None


def test_invalid_outcome_model_message_names_cv():
    with pytest.raises(ValueError, match="'cv'"):
        _fit(_data(200), outcome_model="bogus")


def test_cv_step_summary_can_be_passed_back_to_a_cv_call():
    data = _data(300, seed=3, kind="zero_inflated")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = mb.aipw(data, _spec(), folds=3, seed=5, mstop=30)
        summary = first.diagnostics["nuisance_mstop_summary"]
        # Every candidate's nuisances have a step count, not only the chosen one's.
        assert {"outcome", "mu0", "mu1", "employment", "log_outcome"} <= set(summary)
        again = mb.aipw(data, _spec(), folds=3, seed=5, mstop=summary, tune=False)
    assert again.diagnostics["outcome_model_cv"]["candidates"] == \
        first.diagnostics["outcome_model_cv"]["candidates"]
    assert np.isfinite(again.estimate)
