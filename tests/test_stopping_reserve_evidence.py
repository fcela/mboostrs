"""Failure and integrity checks for the retained #541 research evidence."""
import copy
from pathlib import Path

import pytest

from benches.report_stopping_reserve import assess, load_evidence


@pytest.fixture(scope="module")
def evidence():
    return load_evidence(Path(__file__).resolve().parents[1] /
                         "docs/book/benchmarks/competitive_refresh")


def test_retained_candidate_fails_its_frozen_gate(evidence):
    summary = assess(evidence)
    assert len(summary["draws"]) == 12
    assert not summary["failures"]
    assert not summary["clears_experimental_gate"]
    cancer = next(r for r in summary["means"] if r["task"] == "cancer")
    assert cancer["mean_loss"]["baseline"] == pytest.approx(.06908294)
    assert cancer["paired_difference"] > 0


@pytest.mark.parametrize("mutation, message", [
    ("missing", "missing draw"), ("duplicate", "duplicate draw"),
    ("misaligned", "invalid paired"), ("wrong_score", "disagree"),
])
def test_inconsistent_evidence_is_rejected(evidence, mutation, message):
    document = copy.deepcopy(evidence)
    if mutation == "missing":
        document["results"].pop()
    elif mutation == "duplicate":
        document["results"].append(document["results"][0])
    elif mutation == "misaligned":
        document["results"][0]["results"]["reserve"]["row_loss"].pop()
    else:
        document["results"][0]["results"]["reserve"]["scores"]["loss"] += .1
    with pytest.raises(ValueError, match=message):
        assess(document)


def test_failed_job_remains_visible_and_cannot_pass(evidence):
    document = copy.deepcopy(evidence)
    first = document["results"][0]
    document["results"][0] = {"job": [first["task"], first["seed"]],
                              "_status": "timeout", "failure": {"timeout_s": 600}}
    summary = assess(document)
    assert len(summary["failures"]) == 1
    assert not summary["clears_experimental_gate"]
    assert not summary["means"][0]["complete"]
