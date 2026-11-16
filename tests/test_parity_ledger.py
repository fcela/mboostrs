"""Tests for the scheduled parity evidence collector."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "docs/schemas/parity-result.schema.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(ROOT))

from tools.parity_ledger import ComparisonSummary, _measure  # noqa: E402
from tools.run_parity import focused_nodes, validate_ledger  # noqa: E402


def test_parity_plugin_emits_results_errors_tolerances_and_versions(tmp_path):
    test_file = tmp_path / "test_synthetic_parity.py"
    test_file.write_text(
        """import numpy as np
import pytest

def test_numeric_parity():
    np.testing.assert_allclose([1.0, 2.0], [1.0, 2.02], rtol=0.02, atol=0.001)

@pytest.mark.skip(reason="synthetic skip")
def test_skipped_parity():
    pass
""",
        encoding="utf-8",
    )
    ledger = tmp_path / "parity.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "tools.parity_ledger",
            f"--parity-ledger={ledger}",
            "--parity-scope=test",
            str(test_file),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    assert payload["kind"] == "mboostrs-parity-result"
    assert payload["scope"] == "test"
    assert payload["summary"]["selected"] == 2
    assert payload["versions"]["python"]["pytest"]
    by_status = {row["status"]: row for row in payload["results"]}
    measured = by_status["passed"]
    assert measured["measurement_status"] == "measured"
    assert measured["comparison_count"] == 1
    assert abs(measured["measured_error"]["absolute"] - 0.02) < 1e-12
    assert measured["tolerance"] == {"atol": 0.001, "rtol": 0.02}
    assert by_status["skipped"]["measurement_status"] == "not_emitted"


def test_focused_manifest_nodes_collect_and_ledger_validator_accepts_minimal_contract():
    nodes = focused_nodes()
    assert len(nodes) == 12
    collection = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *nodes],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert collection.returncode == 0, collection.stdout + collection.stderr
    collected = {line for line in collection.stdout.splitlines() if line.startswith("tests/")}
    assert set(nodes) <= collected

    results = [
        {
            "node_id": node,
            "status": "passed",
            "duration_seconds": 0.01,
            "measurement_status": "not_emitted",
            "comparison_count": 0,
            "measured_error": {"absolute": None, "relative": None, "normalized_to_tolerance": None},
            "tolerance": {"atol": None, "rtol": None},
            "validation_ids": [],
            "compared_outputs": [],
        }
        for node in nodes
    ]
    validate_ledger(
        {
            "schema_version": 1,
            "kind": "mboostrs-parity-result",
            "started_at": "2026-07-14T00:00:00Z",
            "finished_at": "2026-07-14T00:00:01Z",
            "scope": "focused",
            "pytest_exit_status": 0,
            "summary": {"selected": len(results), "status_counts": {"passed": len(results)}},
            "commit": "abc123",
            "platform": {},
            "versions": {"python": {"rpy2": None}, "r": {"R": None}},
            "results": results,
        },
        scope="focused",
        require_dependencies=False,
    )


def _validate_result_row(row: dict) -> None:
    # $ref targets like "#/$defs/nonNegativeMeasurement" only resolve against
    # the full document, so wrap the row rather than validating it in isolation.
    payload = {
        "schema_version": 1,
        "kind": "mboostrs-parity-result",
        "started_at": "2026-07-14T00:00:00Z",
        "finished_at": "2026-07-14T00:00:01Z",
        "scope": "test",
        "pytest_exit_status": 0,
        "summary": {"selected": 1, "status_counts": {row["status"]: 1}},
        "commit": None,
        "platform": {},
        "versions": {},
        "results": [row],
    }
    Draft202012Validator(SCHEMA).validate(payload)


def test_an_infinite_worst_case_survives_a_later_smaller_measurement():
    """A comparison node's worst measurement must not be silently replaced by
    a later, smaller one just because the worst case was non-finite."""
    summary = ComparisonSummary()
    exact_mismatch = _measure(
        np.array([1.0, 2.0]), np.array([1.0, 2.001]), rtol=0, atol=0, kind="assert_array_equal"
    )
    summary.observe(exact_mismatch)
    assert summary.normalized_error == float("inf")

    tiny_pass = _measure(np.array([1.0]), np.array([1.0 + 1e-9]), rtol=1e-6, atol=1e-9, kind="assert_allclose")
    summary.observe(tiny_pass)
    assert summary.normalized_error == float("inf"), "the infinite worst case must not be overwritten"

    row = summary.to_dict()
    assert row["measured_error"]["normalized_to_tolerance"] == "Infinity"
    _validate_result_row(
        {
            "node_id": "synthetic",
            "status": "failed",
            "duration_seconds": 0.0,
            "validation_ids": [],
            "compared_outputs": [],
            **row,
        }
    )


def test_a_nan_mismatch_is_not_silently_dropped_by_a_finite_max():
    """One side NaN where a real value was expected is the worst possible
    outcome, not an average of the other (agreeing) elements."""
    measurement = _measure(
        np.array([np.nan, 1.0]), np.array([5.0, 1.0000001]), rtol=1e-6, atol=1e-9, kind="assert_allclose"
    )
    assert measurement is not None
    assert np.isnan(measurement["normalized_error"])

    summary = ComparisonSummary()
    summary.observe(measurement)
    row = summary.to_dict()
    assert row["measured_error"]["normalized_to_tolerance"] == "NaN"
    _validate_result_row(
        {
            "node_id": "synthetic",
            "status": "failed",
            "duration_seconds": 0.0,
            "validation_ids": [],
            "compared_outputs": [],
            **row,
        }
    )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_a_value_agreeing_with_itself_is_an_exact_pass_not_an_error(value):
    """NaN-vs-NaN (asserted by na_action/missing-data tests) and inf-vs-inf
    (same sign) are legitimate exact passes, not infinite/NaN errors."""
    measurement = _measure(np.array([value]), np.array([value]), rtol=1e-6, atol=1e-9, kind="assert_allclose")
    assert measurement is not None
    assert measurement["normalized_error"] == 0.0
    assert measurement["max_absolute_error"] == 0.0

    summary = ComparisonSummary()
    summary.observe(measurement)
    row = summary.to_dict()
    assert row["measured_error"]["normalized_to_tolerance"] == 0.0
    _validate_result_row(
        {
            "node_id": "synthetic",
            "status": "passed",
            "duration_seconds": 0.0,
            "validation_ids": [],
            "compared_outputs": [],
            **row,
        }
    )


def test_a_legitimate_nan_vs_nan_pass_does_not_fail_the_parity_gate(tmp_path):
    """End-to-end regression: a test that legitimately compares NaN to NaN
    through an instrumented assertion must not be rejected by validate_ledger
    with an "incomplete error/tolerance" error -- it is an exact pass."""
    test_file = tmp_path / "test_synthetic_nan_parity.py"
    test_file.write_text(
        """import numpy as np

def test_nan_matches_nan():
    np.testing.assert_allclose([np.nan, 1.0], [np.nan, 1.0], rtol=1e-6, atol=1e-9)
""",
        encoding="utf-8",
    )
    ledger = tmp_path / "parity.json"
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest", "-q", "-p", "tools.parity_ledger",
            f"--parity-ledger={ledger}", "--parity-scope=test", str(test_file),
        ],
        cwd=ROOT, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    Draft202012Validator(SCHEMA).validate(payload)
    row = payload["results"][0]
    assert row["measurement_status"] == "measured"
    assert row["measured_error"]["normalized_to_tolerance"] == 0.0


def test_vector_tolerance_is_applied_per_element():
    """Enhancement-hunt 20.2. A per-element tolerance was reduced to its
    maximum before `normalized_error` was computed, so every element was judged
    against the loosest entry in the vector and the ledger reported a ratio
    smaller than the one the assertion applied."""
    actual = np.array([1.0 + 1e-6, 2.0])
    desired = np.array([1.0, 2.0])
    atol = np.array([1e-9, 1e-3])

    measurement = _measure(actual, desired, rtol=0.0, atol=atol, kind="allclose")
    # The first element misses its own 1e-9 tolerance by a factor of 1000.
    assert measurement["normalized_error"] == pytest.approx(1000.0, rel=1e-6)
    # Under the old max-first reduction this was 1e-3, i.e. "well inside".
    assert measurement["normalized_error"] > 1.0

    # Both ends of the vector are recorded, so a reader can tell that the
    # reported `atol` is the loosest of several rather than the only one.
    assert measurement["atol"] == 1e-3
    assert measurement["atol_min"] == 1e-9


def test_scalar_tolerance_records_no_range():
    measurement = _measure(
        np.array([1.0]), np.array([1.0]), rtol=0.0, atol=1e-3, kind="allclose"
    )
    assert "atol_min" not in measurement
    assert "rtol_min" not in measurement
