"""Causal tools: ci_test conditioning, extrapolation summary, CausalSpec defaults,
causal_trap_report with outcome_missing, tune_mstop flat-curve warning."""

import warnings

import numpy as np
import pytest

import mboostrs as mb
from mboostrs import _rust
from mboostrs._extrapolation import ExtrapolationWarning
from mboostrs.causal import (
    DEFAULT_OVERLAP_BOUNDS,
    FROM_SPEC,
    CausalSpec,
    _sparse_arm_message,
    _spec_default,
    ci_test,
)
from mboostrs.explore import tune_mstop


def _spec(**overrides):
    fields = dict(
        treatment="a", outcome="y", covariates=("x1", "x2"), estimand="ate",
        target_population="simulated sample", unit="person",
        time_zero="assignment", follow_up="one period",
        adjustment_rationale="x1 and x2 are measured before treatment",
    )
    fields.update(overrides)
    return CausalSpec(**fields)


def _collinear_frame(n=400, seed=3):
    rng = np.random.default_rng(seed)
    age = rng.integers(20, 60, n).astype(float)
    education = rng.integers(8, 20, n).astype(float)
    experience = age - education - 6
    x = rng.normal(size=n) + 0.02 * age
    y = 0.5 * x + 0.1 * age + rng.normal(size=n)
    return {"x": x, "y": y, "age": age, "education": education, "experience": experience}


# Item 2: exact linear dependence in the conditioning set.

def test_linear_dependence_reports_the_column_and_its_coefficients():
    n = 6
    age = np.array([30.0, 41.0, 25.0, 52.0, 38.0, 44.0])
    education = np.array([12.0, 16.0, 9.0, 12.0, 14.0, 10.0])
    experience = age - education - 6
    design = np.concatenate([np.ones(n), age, education, experience])
    found = _rust.linear_dependence(design, n, 4, 1e-10)
    assert [column for column, _ in found] == [3]
    coefficients = dict(found[0][1])
    assert coefficients[1] == pytest.approx(1.0)
    assert coefficients[2] == pytest.approx(-1.0)
    assert coefficients[0] == pytest.approx(-6.0)


def test_a_collinear_conditioning_column_is_dropped_with_a_note():
    data = _collinear_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ci_test(data, "x", "y", given=("age", "education", "experience"))
    assert result.dropped == ("experience",)
    assert result.given == ("age", "education", "experience")
    assert len(result.notes) == 1
    note = result.notes[0]
    assert "'experience' was dropped" in note
    assert "age" in note and "education" in note
    assert "experience" in result.describe()
    assert np.isfinite(result.p_value)


def test_x_that_is_a_combination_of_the_conditioning_set_gives_p_one():
    data = _collinear_frame()
    result = ci_test(data, "experience", "y", given=("age", "education"))
    assert np.isnan(result.p_value)
    assert np.isnan(result.statistic)
    assert any(note.startswith("'experience' is an exact linear combination")
               for note in result.notes)


def test_a_categorical_conditioning_column_is_accepted():
    rng = np.random.default_rng(11)
    n = 300
    region = rng.choice(["north", "south", "west"], n)
    z = rng.normal(size=n) + (region == "west")
    x = z + rng.normal(size=n)
    y = 2 * z + rng.normal(size=n)
    data = {"x": x, "y": y, "region": region}
    result = ci_test(data, "x", "y", given=("region",))
    assert result.dropped == ()
    assert result.n == n
    assert np.isfinite(result.statistic)


def test_a_repeated_conditioning_column_is_refused():
    data = _collinear_frame()
    with pytest.raises(ValueError, match="twice"):
        ci_test(data, "x", "y", given=("age", "age"))


# Item 3: one extrapolation warning per call, under the user's column name.

def test_extrapolation_is_summarised_once_under_the_user_column_name():
    rng = np.random.default_rng(5)
    n = 200
    z = rng.normal(size=n)
    z[0] = 50.0  # the only row above the rest; its fold trains without it
    x = rng.normal(size=n)
    y = x + rng.normal(size=n)
    data = {"x": x, "y": y, "z": z}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ci_test(data, "x", "y", given=("z",), folds=2, seed=0)
    extrapolation = [w for w in caught if issubclass(w.category, ExtrapolationWarning)]
    assert len(extrapolation) == 1, [str(w.message) for w in extrapolation]
    message = str(extrapolation[0].message)
    assert "__ci_given" not in message
    assert "z" in message
    assert "ci_test(extrapolation='allow')" in message


def test_extrapolation_error_refuses_and_allow_is_silent():
    rng = np.random.default_rng(5)
    n = 200
    z = rng.normal(size=n)
    z[0] = 50.0
    data = {"x": rng.normal(size=n), "y": rng.normal(size=n), "z": z}
    with pytest.raises(ValueError, match="z"):
        ci_test(data, "x", "y", given=("z",), folds=2, seed=0, extrapolation="error")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ci_test(data, "x", "y", given=("z",), folds=2, seed=0, extrapolation="allow")
    assert not [w for w in caught if issubclass(w.category, ExtrapolationWarning)]


def test_an_unknown_extrapolation_policy_is_refused():
    with pytest.raises(ValueError, match="extrapolation must be one of"):
        ci_test(_collinear_frame(), "x", "y", given=("age",), extrapolation="sometimes")


def test_a_two_level_categorical_x_is_tested_as_one_indicator():
    # Issue #429 item 5: a categorical x is tested, not refused. Two levels give
    # one indicator, so the test is the binary-column test with one function.
    rng = np.random.default_rng(0)
    x = np.array(["a", "b"] * 20)
    data = {"x": x, "y": np.arange(40.0) + (x == "b") * 3.0 + rng.normal(size=40) * 0.1}
    test = ci_test(data, "x", "y")
    assert test.functions == 1


# Item 2 (core): the warm-start check and an almost exact fit.

def test_a_collinear_fit_on_12000_rows_resumes_without_a_warm_start_error():
    # experience is an exact function of age and education, so the boosted fit
    # reaches a risk near zero. On 12,000 rows the incremental risk drifts from
    # a direct recomputation by about 2e-8, which the check used to refuse.
    rng = np.random.default_rng(0)
    n = 12000
    age = rng.integers(20, 60, n).astype(float)
    education = rng.integers(8, 20, n).astype(float)
    experience = age - education - 6
    data = {"t": experience, "age": age, "education": education}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tuning = tune_mstop("t ~ bols(age) + bols(education)", data, folds=5,
                            start=100, nu=0.3, seed=0)
    assert tuning.mstop >= 1


# Item 6: CausalSpec carries propensity_clip, outcome_missing and overlap_bounds.

def test_spec_defaults_match_the_estimators_defaults():
    spec = _spec()
    assert spec.propensity_clip == (0.01, 0.99)
    assert spec.outcome_missing == "error"
    assert spec.overlap_bounds == DEFAULT_OVERLAP_BOUNDS


def test_spec_fields_are_validated_and_normalised():
    spec = _spec(propensity_clip=[0.05, 0.95], overlap_bounds=[0.1, 0.9],
                 outcome_missing="ipw")
    assert spec.propensity_clip == (0.05, 0.95)
    assert spec.overlap_bounds == (0.1, 0.9)
    assert _spec(propensity_clip=None).propensity_clip is None
    with pytest.raises(ValueError, match="propensity_clip"):
        _spec(propensity_clip=(0.5, 0.2))
    with pytest.raises(ValueError, match="outcome_missing"):
        _spec(outcome_missing="drop")
    with pytest.raises(ValueError, match="overlap bounds"):
        _spec(overlap_bounds=(0.0, 0.5))


def test_spec_default_gives_the_explicit_argument_precedence():
    spec = _spec(propensity_clip=(0.05, 0.95))
    assert _spec_default(spec, "propensity_clip", FROM_SPEC) == (0.05, 0.95)
    assert _spec_default(spec, "propensity_clip", (0.01, 0.99)) == (0.01, 0.99)
    assert _spec_default(spec, "propensity_clip", None) is None
    assert _spec_default(None, "propensity_clip", FROM_SPEC) == (0.01, 0.99)


def test_overlap_report_reads_the_spec_bounds_and_an_explicit_bound_wins():
    rng = np.random.default_rng(4)
    n = 300
    x1 = rng.normal(size=n)
    x2 = rng.uniform(-1, 1, n)
    a = rng.binomial(1, 1 / (1 + np.exp(-1.5 * x1))).astype(float)
    y = 1.0 * a + x1 + x2 + rng.normal(size=n)
    data = {"a": a, "y": y, "x1": x1, "x2": x2}
    spec = _spec(treatment="a", outcome="y", overlap_bounds=(0.3, 0.7))
    with warnings.catch_warnings():
        # The sparse-arm warning is the subject of its own test; this one is noise.
        warnings.simplefilter("ignore")
        result = mb.aipw(data, spec, folds=3, mstop=40, tune=False, seed=1)
    from_spec = mb.overlap_report(result, data=data)
    explicit = mb.overlap_report(result, data=data, bounds=(0.02, 0.98))
    assert from_spec.bounds == (0.3, 0.7)
    assert explicit.bounds == (0.02, 0.98)
    assert from_spec.share_outside >= explicit.share_outside
    bare = mb.overlap_report(np.asarray(result.propensity), covariates=("x1",), data=data)
    assert bare.bounds == DEFAULT_OVERLAP_BOUNDS


# Item 4: causal_trap_report and outcome_missing; top-level export.

def test_causal_trap_report_is_exported_at_the_top_level():
    from mboostrs.causal import causal_trap_report

    assert mb.causal_trap_report is causal_trap_report


def _frame_with_missing_outcome(n=200, seed=8):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    a = rng.binomial(1, 0.5, n).astype(float)
    y = a + x1 + rng.normal(size=n)
    y[:30] = np.nan
    return {"a": a, "y": y, "x1": x1}


def test_trap_report_states_the_subpopulation_for_the_default_policy():
    frame = _frame_with_missing_outcome()
    report = mb.causal_trap_report(frame, _spec(treatment="a", outcome="y",
                                                covariates=("x1",)))
    names = {f.name for f in report.findings}
    assert "outcome_missing" in names
    assert "outcome_missing_ipw" not in names
    message = next(f.message for f in report.findings if f.name == "outcome_missing")
    assert "subpopulation" in message


def test_trap_report_says_all_rows_under_outcome_missing_ipw():
    frame = _frame_with_missing_outcome()
    spec = _spec(treatment="a", outcome="y", covariates=("x1",), outcome_missing="ipw")
    report = mb.causal_trap_report(frame, spec)
    names = {f.name for f in report.findings}
    assert "outcome_missing" not in names
    finding = next(f for f in report.findings if f.name == "outcome_missing_ipw")
    assert finding.severity == "advisory"
    assert "all 200 rows" in finding.message
    assert "subpopulation" not in finding.message


def test_trap_report_argument_overrides_the_spec():
    frame = _frame_with_missing_outcome()
    spec = _spec(treatment="a", outcome="y", covariates=("x1",), outcome_missing="ipw")
    report = mb.causal_trap_report(frame, spec, outcome_missing="error")
    assert "outcome_missing" in {f.name for f in report.findings}


# Item 8: tune_mstop reports a flat curve.

def _flat_and_strong_data(seed=0, n=300):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    noise = rng.normal(size=n)
    strong = 3.0 * x + rng.normal(size=n) * 0.2
    return {"x": x, "noise": noise, "strong": strong}


def test_tune_mstop_warns_when_the_curve_is_flat():
    data = _flat_and_strong_data()
    with pytest.warns(RuntimeWarning, match="close to a constant"):
        tuning = tune_mstop("noise ~ bols(x)", data, folds=3, start=20, nu=0.1, seed=1)
    assert tuning.mstop >= 1


def test_tune_mstop_is_silent_on_a_curve_with_a_real_signal():
    data = _flat_and_strong_data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tune_mstop("strong ~ bols(x)", data, folds=3, start=20, nu=0.1, seed=1)
    assert not [w for w in caught if "close to a constant" in str(w.message)]


def test_tune_mstop_flat_tolerance_none_turns_the_check_off():
    data = _flat_and_strong_data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tune_mstop("noise ~ bols(x)", data, folds=3, start=20, nu=0.1, seed=1,
                   flat_tolerance=None)
    assert not [w for w in caught if "close to a constant" in str(w.message)]


# Item 7: sparse-arm wording names the target without an article.

def test_sparse_arm_message_says_of_all_rows():
    report = {
        "target": "all rows", "target_share": 0.05, "min_rows_per_arm": 20,
        "bins": [{
            "bin": 0, "propensity_min": 0.01, "propensity_max": 0.02,
            "n_treated": 3, "n_control": 900, "share_of_target": 0.08,
            "sparse_arms": ["treated"], "covariate_span": {"x1": [-2.0, -1.0]},
        }],
    }
    text = _sparse_arm_message(report, ("x1",), "separate", False, "auto")
    assert "of all rows" in text
    assert "of the all rows" not in text
