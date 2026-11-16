"""Causal rerun 5 friction (issue 484): iv_effect missing outcomes, ci_test
exact dependence and warning names, repeated warnings within one call, and the
post-treatment check of causal_trap_report.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest

import mboostrs as mb
from mboostrs._extrapolation import ExtrapolationWarning, reset_shown_summaries
from mboostrs._warning_collect import once_per_call, warning_key


def _iv_spec(**extra):
    return mb.CausalSpec(
        treatment="enrolled", outcome="earn", covariates=("x",), estimand="ate",
        outcome_type="continuous", target_population="applicants", unit="applicant",
        time_zero="intake", follow_up="one year", treatment_contrast=(0, 1),
        adjustment_rationale="x causes enrolment and earnings", **extra)


def _iv_data(seed, n=2500, missing=True):
    """Instrument z moves enrolment; the true effect of enrolment is 2. The
    outcome is missing at random given x and enrolment when ``missing``."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.integers(0, 2, n).astype(float)
    u = rng.normal(size=n)
    d = (0.9 * z + 0.6 * x + 0.8 * u + rng.normal(size=n) > 0.5).astype(float)
    y = 2.0 * d + 1.5 * x + u + rng.normal(size=n)
    if missing:
        p_obs = 1 / (1 + np.exp(-(0.8 + 0.6 * x - 0.9 * d)))
        y = np.where(rng.random(n) < p_obs, y, np.nan)
    return {"enrolled": d, "earn": y, "z": z, "x": x}


# -- 1. iv_effect and outcome_missing -----------------------------------------

def test_iv_effect_refuses_a_missing_outcome_and_names_the_argument_it_accepts():
    data = _iv_data(0)
    with pytest.raises(ValueError, match="outcome_missing='ipw'") as info:
        mb.iv_effect(data, _iv_spec(), "z", folds=4, mstop=100)
    assert "outcome_missing" in str(info.value)


def test_iv_effect_ipw_recovers_the_effect_on_missing_at_random_outcomes():
    data = _iv_data(1)
    fit = mb.iv_effect(data, _iv_spec(), "z", folds=4, seed=1, mstop=100,
                       outcome_missing="ipw")
    assert abs(fit.estimate - 2.0) < 0.8
    assert fit.nuisance["outcome_missing"] == "ipw"
    assert fit.nuisance["response_model"] == "fitted"
    assert fit.nuisance["n_missing_outcome"] == int(np.isnan(data["earn"]).sum())
    assert fit.n == 2500
    assert np.all(np.isfinite(fit.influence))


def test_iv_effect_reads_outcome_missing_from_the_spec_when_the_call_omits_it():
    data = _iv_data(2)
    fit = mb.iv_effect(data, _iv_spec(outcome_missing="ipw"), "z", folds=4, seed=2,
                       mstop=100)
    assert fit.nuisance["outcome_missing"] == "ipw"


def test_iv_effect_ipw_without_missing_rows_is_the_complete_data_estimate():
    data = _iv_data(3, missing=False)
    plain = mb.iv_effect(data, _iv_spec(), "z", folds=4, seed=3, mstop=100)
    ipw = mb.iv_effect(data, _iv_spec(), "z", folds=4, seed=3, mstop=100,
                       outcome_missing="ipw")
    assert ipw.estimate == plain.estimate
    assert ipw.nuisance["response_model"] is None
    assert ipw.nuisance["n_missing_outcome"] == 0


def test_iv_effect_outcome_missing_argument_is_checked():
    with pytest.raises(ValueError, match="outcome_missing must be"):
        mb.iv_effect(_iv_data(4), _iv_spec(), "z", folds=4, mstop=100,
                     outcome_missing="drop")


# -- 2 and 3. ci_test exact dependence and column names ------------------------

def _identity_data(seed=5, n=800):
    rng = np.random.default_rng(seed)
    age = rng.uniform(20, 60, n).round()
    education = rng.integers(8, 18, n).astype(float)
    experience = age - education - 6
    y = 0.5 * education + 0.02 * age + rng.normal(size=n)
    return {"age": age, "education": education, "experience": experience, "y": y,
            "x": rng.normal(size=n)}


def test_an_x_that_is_an_identity_of_the_conditioning_set_is_not_tested_and_is_nan():
    result = mb.ci_test(_identity_data(), "experience", "y", given=("age", "education"))
    assert result.status == "dependent"
    assert result.skipped is True
    assert np.isnan(result.p_value)
    assert np.isnan(result.statistic)
    assert np.isnan(result.residual_correlation)


def test_the_identity_note_names_every_column_in_it_and_the_one_ci_test_suggests():
    result = mb.ci_test(_identity_data(), "experience", "y", given=("age", "education"))
    note = next(n for n in result.notes if n.startswith("'experience' is an exact"))
    assert "(experience = -6 * intercept +1 * age -1 * education)" in note
    assert "one of 'age', 'education' or 'experience' to leave the set" in note
    assert "suggests dropping 'education'" in note
    assert "no test was run" in note
    assert "p_value is 1" not in note


def test_an_identity_in_y_names_the_column_to_drop():
    data = _identity_data()
    data["y"] = 3.0 * data["age"] - 2.0 * data["education"] + 1.0
    result = mb.ci_test(data, "x", "y", given=("age", "education"))
    assert result.status == "dependent"
    assert np.isnan(result.p_value)
    note = next(n for n in result.notes if n.startswith("'y' is an exact"))
    assert "suggests dropping 'education'" in note


def test_a_negligible_term_is_not_reported_as_part_of_the_identity():
    # experience is an exact combination of age and education; x is unrelated and
    # enters the identity only at rounding level, so it is not named in it.
    result = mb.ci_test(_identity_data(), "y", "experience",
                        given=("age", "education", "x"))
    note = next(n for n in result.notes if n.startswith("'experience' is an exact"))
    assert "* x" not in note
    assert "suggests dropping 'education'" in note


def test_the_describe_text_of_a_skipped_test_says_no_test_was_run():
    result = mb.ci_test(_identity_data(), "experience", "y", given=("age", "education"))
    text = result.describe()
    assert text.startswith("UNDETERMINED")
    assert "no test was run" in text
    assert "placeholder" not in text


def test_ci_test_warnings_name_the_conditioning_column_not_the_alias():
    rng = np.random.default_rng(9)
    m = 1200
    z = rng.normal(size=m)
    w = rng.normal(size=m)
    w[0] = 60.0  # outside the training range of the fold that holds this row out
    x = z + rng.normal(size=m)
    y = 2 * z + rng.normal(size=m) + 0.5 * np.sin(w)
    reset_shown_summaries()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mb.ci_test({"x": x, "y": y, "z": z, "w": w}, "x", "y", given=("w",))
    texts = [str(item.message) for item in caught]
    extrapolation = [t for t in texts if "outside the range" in t]
    assert extrapolation, texts
    assert any("bbs(w)" in t and "covariate w" in t for t in extrapolation)
    assert not any("__ci_" in t for t in texts)


# -- 5. repeated warnings within one call -------------------------------------

def test_a_warning_repeated_within_one_call_is_issued_once_with_its_count():
    @once_per_call
    def routine():
        for steps in (1600, 1600, 1600):
            warnings.warn(f"the cross-validated risk was still falling at {steps} steps",
                          RuntimeWarning, stacklevel=2)
        return "done"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert routine() == "done"
    texts = [str(item.message) for item in caught]
    assert len(texts) == 1
    assert "raised 3 times in this call" in texts[0]
    assert "still falling at 1600 steps" in texts[0]


def test_a_warning_raised_once_in_a_call_keeps_its_text_and_has_no_count():
    @once_per_call
    def routine():
        warnings.warn("a single notice about 12 rows", UserWarning, stacklevel=2)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        routine()
    assert [str(item.message) for item in caught] == ["a single notice about 12 rows"]


def test_once_per_call_keeps_the_signature_and_the_warning_category():
    @once_per_call
    def routine(a, *, b=2):
        """Doc."""
        warnings.warn(ExtrapolationWarning("beyond the range", rows=[]), stacklevel=2)
        return a + b

    assert routine.__doc__ == "Doc."
    assert routine(1, b=3) == 4
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        routine(1)
    assert caught[0].category is ExtrapolationWarning


def test_an_aipw_call_issues_no_two_warnings_with_the_same_wording():
    rng = np.random.default_rng(12)
    n = 600
    x = rng.normal(size=n)
    a = (rng.uniform(size=n) < 1 / (1 + np.exp(-x))).astype(float)
    y = a + x + rng.normal(size=n)
    spec = mb.CausalSpec(
        treatment="a", outcome="y", covariates=("x",), estimand="ate",
        target_population="sample", unit="person", time_zero="baseline",
        follow_up="one year", adjustment_rationale="x causes a and y")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mb.aipw({"a": a, "y": y, "x": x}, spec, folds=4, seed=3, smooth=True,
                mstop=20, max_mstop=40, tune=True)
    keys = [warning_key(str(item.message)) for item in caught]
    assert len(keys) == len(set(keys))


# -- 6. causal_trap_report and post-treatment columns ------------------------

def _trap_spec(**overrides):
    fields = {
        "treatment": "a",
        "outcome": "y",
        "covariates": ("x1",),
        "time_zero": "admission",
        "follow_up": "30 days",
        "adjustment_rationale": "x1 is measured at baseline and precedes a",
    }
    fields.update(overrides)
    return mb.CausalSpec(**fields)


def _trap_frame(n=300, seed=3):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    a = (rng.uniform(size=n) < 0.5).astype(float)
    y = 1.0 * a + 0.5 * x1 + rng.normal(scale=0.5, size=n)
    return {"x1": x1, "a": a, "y": y}


def test_a_post_treatment_column_the_frame_lacks_is_not_a_blocking_finding():
    report = mb.causal_trap_report(_trap_frame(),
                                   _trap_spec(post_treatment=("certificate",)))
    assert "columns_absent" not in {f.name for f in report.findings}
    assert report.blocking == ()


def test_a_covariate_the_frame_lacks_is_still_blocking_when_post_treatment_is_set():
    report = mb.causal_trap_report(_trap_frame(),
                                   _trap_spec(covariates=("x1", "absent"),
                                              post_treatment=("certificate",)))
    finding = next(f for f in report.findings if f.name == "columns_absent")
    assert finding.severity == "blocking"
    assert "absent" in finding.message
    assert "certificate" not in finding.message


# -- 4. terms="smooth" reports the gains as terms="auto" does -----------------

def _effect_data(seed=21, n=400):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    a = (rng.uniform(size=n) < 1 / (1 + np.exp(-x1))).astype(float)
    y = a * (1 + 0.5 * x2) + x1 + rng.normal(size=n)
    return {"a": a, "y": y, "x1": x1, "x2": x2}


def _effect_spec():
    return mb.CausalSpec(
        treatment="a", outcome="y", covariates=("x1", "x2"),
        time_zero="assignment", follow_up="one year",
        adjustment_rationale="the covariates are the common causes")


def test_effect_model_smooth_reports_the_gain_over_constant_and_its_standard_error():
    fit = mb.effect_model(_effect_data(), _effect_spec(), seed=0, terms="smooth",
                          folds=3, max_mstop=40)
    selection = fit.diagnostics["terms_selection"]
    assert selection["rule"] == "requested"
    assert set(selection["gain_over_constant"]) == {"smooth"}
    assert set(selection["gain_se"]) == {"smooth"}
    assert np.isfinite(selection["gain_over_constant"]["smooth"])
    assert np.isfinite(selection["gain_se"]["smooth"])


# -- 5. a warning repeated within one effect_model call is issued once --------

def test_effect_model_issues_each_warning_text_once_per_call():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mb.effect_model(_effect_data(), _effect_spec(), seed=0, terms="smooth",
                        folds=3, max_mstop=40)
    texts = [str(item.message) for item in caught]
    assert len([t for t in texts if "still falling" in t]) == 1
    assert len(texts) == len(set(texts))


# -- 9. aipw states its default interval level --------------------------------

def test_aipw_documents_its_default_level_of_0_95():
    assert "The default is 0.95" in mb.aipw.__doc__
    assert "level=0.9" in mb.aipw.__doc__
    topic = Path(mb.__file__).parent / "agent_docs" / "topics" / "causal-average.md"
    assert "defaults to 0.95" in topic.read_text()
