"""Every fresh family and behavioral gate must pass independently."""
import copy

import pytest

from benches.bench_binary_reference import CASES
from benches.report_binary_reference import assess


@pytest.fixture
def documents():
    specs = [[c, s] for c in CASES for s in range(6)]
    rows = [{"case": c, "seed": s, "_status": "success", "mse": 1.,
             "input_digest": f"data-{c}-{s}", "prediction_digest": f"prediction-{c}-{s}",
             "interactions": [], "reference_curve_range": 1. if c in ("indicator", "boolean") else None}
            for c, s in specs]
    before = {"protocol": {"specs": specs, "document_sha256": "same"}, "results": rows}
    after = copy.deepcopy(before)
    for r in after["results"]:
        if r["case"] in ("indicator", "boolean", "shifted", "logged"):
            r["mse"] = .5
    return before, after


def test_complete_nonregression_and_behavioral_checks_pass(documents):
    result = assess(*documents)
    assert result["gate_passed"] and len(result["families"]) == 9 and len(result["paired"]) == 54


@pytest.mark.parametrize("mutation", ["regression", "categorical_change", "null_adoption", "flat_reference", "failure"])
def test_improvements_cannot_compensate_a_failed_gate(documents, mutation):
    rows = documents[1]["results"]
    if mutation == "regression":
        for r in rows:
            if r["case"] == "smooth":
                r["mse"] = 1.1
    elif mutation == "categorical_change":
        next(r for r in rows if r["case"] == "categorical")["prediction_digest"] = "changed"
    elif mutation == "null_adoption":
        next(r for r in rows if r["case"] == "null")["interactions"] = [{"variables": ["g", "x"]}]
    elif mutation == "flat_reference":
        next(r for r in rows if r["case"] == "indicator")["reference_curve_range"] = 0.
    else:
        rows[0]["_status"] = "timeout"
    assert not assess(*documents)["gate_passed"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "data", "nonfinite", "family"])
def test_incomplete_or_incompatible_evidence_is_rejected(documents, mutation):
    before, after = documents
    if mutation == "missing":
        after["results"].pop()
    elif mutation == "duplicate":
        after["results"].append(copy.deepcopy(after["results"][0]))
    elif mutation == "data":
        after["results"][0]["input_digest"] = "changed"
    elif mutation == "nonfinite":
        after["results"][0]["mse"] = float("nan")
    else:
        for d in (before, after):
            d["protocol"]["specs"] = [s for s in d["protocol"]["specs"] if s[0] != "null"]
            d["results"] = [r for r in d["results"] if r["case"] != "null"]
    with pytest.raises(ValueError):
        assess(before, after)
