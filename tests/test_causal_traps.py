"""The causal-trap battery: one synthetic scenario per design failure.

Each test builds the smallest frame that trips one trap, so a finding that
stops firing names the check that regressed rather than the dataset that
happened to change.
"""

import numpy as np
import pytest

import mboostrs as mb


def _spec(**overrides):
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


def _frame(n=300, seed=3, **overrides):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    a = (rng.uniform(size=n) < 0.5).astype(float)
    y = 1.0 * a + 0.5 * x1 + rng.normal(scale=0.5, size=n)
    frame = {"x1": x1, "a": a, "y": y}
    frame.update(overrides)
    return frame


def _names(report):
    return {f.name for f in report.findings}


def test_a_well_formed_design_raises_only_the_timing_finding():
    report = mb.causal_trap_report(_frame(), _spec())
    assert _names(report) == {"timing_unverifiable"}
    assert report.blocking == ()
    assert not report.clean


def test_a_timestamp_column_clears_the_timing_finding():
    frame = _frame(visit=np.arange("2020-01-01", "2020-11-06", dtype="datetime64[D]"))
    report = mb.causal_trap_report(frame, _spec())
    assert "timing_unverifiable" not in _names(report)
    assert report.clean


def test_columns_the_frame_lacks_are_blocking():
    report = mb.causal_trap_report(_frame(), _spec(covariates=("x1", "absent")))
    finding = next(f for f in report.findings if f.name == "columns_absent")
    assert finding.severity == "blocking"
    assert "absent" in finding.message


def test_a_constant_treatment_blocks_the_contrast():
    frame = _frame()
    frame["a"] = np.zeros(len(frame["a"]))
    report = mb.causal_trap_report(frame, _spec())
    assert "no_treatment_variation" in _names(report)
    assert "no_treatment_variation" in {f.name for f in report.blocking}


def test_a_contrast_with_no_rows_blocks():
    frame = _frame()
    report = mb.causal_trap_report(frame, _spec(treatment_contrast=(0, 7)))
    finding = next(f for f in report.findings if f.name == "contrast_absent")
    assert finding.severity == "blocking"
    assert "7" in finding.message


def test_a_three_level_treatment_narrows_the_population():
    frame = _frame()
    frame["a"] = np.repeat([0.0, 1.0, 2.0], 100)
    report = mb.causal_trap_report(frame, _spec())
    finding = next(f for f in report.findings if f.name == "extra_treatment_levels")
    assert finding.severity == "advisory"
    assert "2.0" in finding.message


def test_a_tiny_arm_blocks_the_estimate():
    frame = _frame()
    frame["a"] = np.concatenate([np.zeros(297), np.ones(3)])
    report = mb.causal_trap_report(frame, _spec())
    finding = next(f for f in report.findings if f.name == "arm_too_small")
    assert finding.severity == "blocking"
    assert "3 rows" in finding.message


def test_missing_outcomes_shrink_the_stated_population():
    frame = _frame()
    frame["y"][:60] = np.nan
    report = mb.causal_trap_report(frame, _spec())
    finding = next(f for f in report.findings if f.name == "outcome_missing")
    assert finding.severity == "advisory"
    assert "60 rows" in finding.message


def test_a_missing_adjustment_column_changes_exchangeability():
    frame = _frame()
    frame["x1"][:90] = np.nan
    report = mb.causal_trap_report(frame, _spec())
    finding = next(f for f in report.findings if f.name == "covariate_missing")
    assert "30.0%" in finding.message


def test_a_level_seen_under_one_arm_only_is_reported():
    n = 300
    rng = np.random.default_rng(5)
    frame = {
        "x1": np.concatenate([np.zeros(n // 2), np.ones(n - n // 2)]),
        "a": np.concatenate([np.zeros(n // 2), np.ones(n - n // 2)]),
        "y": rng.normal(size=n),
    }
    report = mb.causal_trap_report(frame, _spec())
    finding = next(f for f in report.findings if f.name == "positivity_by_level")
    assert finding.severity == "advisory"
    assert "one treatment arm" in finding.message


def test_missing_treatment_values_are_reported_as_a_population_change():
    frame = _frame()
    frame["a"][:45] = np.nan
    report = mb.causal_trap_report(frame, _spec())
    finding = next(f for f in report.findings if f.name == "treatment_missing")
    assert "45 rows" in finding.message


def test_the_report_serializes_with_severities_counted():
    report = mb.causal_trap_report(_frame(), _spec(covariates=("x1", "gone")))
    record = report.to_dict()
    assert record["n_blocking"] == len(report.blocking) >= 1
    assert any(entry["severity"] == "blocking" for entry in record["findings"])


@pytest.mark.parametrize("kwargs", [
    {"min_per_arm": 0},
    {"min_arm_share": 0.0},
    {"min_arm_share": 1.0},
    {"max_missing_share": 1.5},
])
def test_thresholds_outside_their_range_are_rejected(kwargs):
    with pytest.raises(ValueError):
        mb.causal_trap_report(_frame(), _spec(), **kwargs)
