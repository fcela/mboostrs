"""Tests for auditable benchmark scenarios and structured results."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benches.benchmark_contract import (  # noqa: E402
    BenchmarkContractError,
    correctness_digest,
    data_digest,
    load_scenarios,
    measurement,
    regression_report,
    validate_result,
)


def _result_payload() -> dict:
    shared, error = correctness_digest(
        {"fitted": np.array([1.0, 2.0])},
        {"fitted": np.array([1.0, 2.0 + 1e-10])},
        rtol=1e-6,
        atol=1e-9,
    )
    rows = [
        measurement(
            name,
            times,
            output_digest=data_digest({"name": name}),
            comparison_digest=shared,
        )
        for name, times in [("mboostrs", [0.9, 1.0, 1.1]), ("R mboost", [1.8, 2.0, 2.2])]
    ]
    return {
        "schema_version": 1,
        "kind": "mboostrs-benchmark-result",
        "created_at": "2026-07-14T00:00:00Z",
        "manifest_digest": data_digest({"manifest": 1}),
        "metadata": {
            "commit": "abc",
            "platform": {},
            "versions": {},
            "compiler_flags": {},
            "threads": {},
        },
        "warmup_policy": {"iterations": 1, "description": "one untimed call"},
        "results": [
            {
                "scenario_id": "fit.scalar.gaussian.end_to_end",
                "boundary": "end_to_end",
                "thread_mode": "single",
                "seed": 42,
                "parameters": {"n": 2},
                "input_digest": data_digest({"x": np.array([1.0, 2.0])}),
                "compared_outputs": ["fitted"],
                "tolerance": {"atol": 1e-9, "rtol": 1e-6},
                "correctness": {"status": "passed", **error},
                "measurements": rows,
            }
        ],
        "regressions": [],
    }


def test_scenario_manifest_covers_every_required_benchmark_dimension():
    manifest = load_scenarios()
    tags = {tag for row in manifest["scenarios"] for tag in row["coverage_tags"]}
    required = {
        "packed_scalar",
        "gaussian_gram",
        "general_block",
        "monotone",
        "tree",
        "survival",
        "multinomial",
        "lss_cyclic",
        "lss_noncyclic",
        "cv",
        "stability_selection",
        "parallel_single",
        "parallel_default",
        "prediction",
        "path_slicing",
        "serialization",
        "diagnostics",
        "cholesky",
        "indefinite_lu",
        "instrumentation_overhead",
        "scaling_n",
        "scaling_candidates",
        "scaling_basis_width",
        "scaling_mstop",
        "scaling_folds",
        "scaling_threads",
        "allocation_count",
    }
    assert required <= tags
    by_id = {row["id"]: row for row in manifest["scenarios"]}
    assert by_id["cv.scalar.serial.end_to_end"]["status"] == "implemented"
    assert by_id["cv.scalar.default.end_to_end"]["status"] == "implemented"


def test_correctness_must_pass_before_result_is_accepted():
    payload = _result_payload()
    validate_result(payload)

    mismatched = copy.deepcopy(payload)
    mismatched["results"][0]["measurements"][1]["correctness_digest"] = (
        "mboostrs-cmp-v1:" + data_digest({"different": True})
    )
    with pytest.raises(BenchmarkContractError, match="mismatched correctness digests"):
        validate_result(mismatched)


def test_a_correctness_digest_must_come_from_a_real_comparison():
    """Enhancement-hunt 20.10. `validate_result` required only that a
    scenario's digests be *equal*, which two copies of any arbitrary string
    satisfy — so a fabricated `sha256:000...0` published a competitive claim
    that validated cleanly. The tag makes a produced digest distinguishable
    from any other string. It closes the accidental cases (a stubbed runner, a
    copied digest, a comparison dropped in a refactor); it is not cryptographic
    and does not stop deliberate forgery."""
    fabricated = copy.deepcopy(_result_payload())
    fake = "sha256:" + "0" * 64
    for row in fabricated["results"][0]["measurements"]:
        row["correctness_digest"] = fake
    with pytest.raises(BenchmarkContractError, match="neither correctness_digest"):
        validate_result(fabricated)


def test_a_single_implementation_row_declares_its_evidence_as_a_self_check():
    """`correctness_digest` used to mean a validated comparison midpoint in one
    runner and the raw output digest in two others, under one field name. The
    two tags say which."""
    from benches.benchmark_contract import (
        CORRECTNESS_DIGEST_TAG,
        SELF_CHECK_DIGEST_TAG,
        self_check_digest,
    )

    digest = self_check_digest({"fitted": [1.0]}, invariant="round-trip reproduces a fresh run")
    assert digest.startswith(f"{SELF_CHECK_DIGEST_TAG}:")
    assert not digest.startswith(f"{CORRECTNESS_DIGEST_TAG}:")

    # The invariant is hashed in, so two rows agreeing on a digest also agree
    # on what was verified.
    other = self_check_digest({"fitted": [1.0]}, invariant="a different invariant")
    assert other != digest

    with pytest.raises(BenchmarkContractError, match="non-empty invariant"):
        self_check_digest({"fitted": [1.0]}, invariant="")

    accepted = copy.deepcopy(_result_payload())
    for row in accepted["results"][0]["measurements"]:
        row["correctness_digest"] = digest
    validate_result(accepted)

    with pytest.raises(BenchmarkContractError, match="failed correctness validation"):
        correctness_digest({"x": [1.0]}, {"x": [2.0]}, rtol=1e-9, atol=0.0)


def test_measured_allocations_require_method_and_scope_provenance():
    with pytest.raises(BenchmarkContractError, match="method and scope"):
        measurement(
            "mboostrs",
            [0.9, 1.0, 1.1],
            output_digest=data_digest({"output": 1}),
            comparison_digest=data_digest({"comparison": 1}),
            allocation_count=42,
        )

    row = measurement(
        "mboostrs",
        [0.9, 1.0, 1.1],
        output_digest=data_digest({"output": 1}),
        comparison_digest=data_digest({"comparison": 1}),
        allocation_count=42,
        allocation_method="test allocator",
        allocation_scope="one call",
    )
    assert row["allocations"] == {
        "status": "measured",
        "count": 42,
        "method": "test allocator",
        "scope": "one call",
    }


def test_result_schema_and_nonblocking_regression_report(tmp_path):
    payload = _result_payload()
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    validate_result(json.loads(path.read_text(encoding="utf-8")))

    slower = copy.deepcopy(payload)
    slower["results"][0]["measurements"][0]["summary"]["median_seconds"] = 1.2
    reports = regression_report(slower, payload)
    assert len(reports) == 1
    assert reports[0]["change_fraction"] == pytest.approx(0.2)
    assert reports[0]["blocking"] is False
