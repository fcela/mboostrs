"""Schema recovery cannot hide changed predictions or omitted controls."""
import copy

import pytest

from benches.bench_factor_reference import CASES, data
from benches.benchmark_contract import data_digest
from benches.report_factor_schema_replay import assess


@pytest.fixture
def documents():
    rows = []
    for case in CASES:
        for seed in range(6):
            train, test = data(case, seed)
            rows.append({"case": case, "seed": seed, "_status": "success", "mse": 1.,
                         "input_digest": data_digest([train.to_dict("list"), test.to_dict("list")]),
                         "prediction_digest": f"prediction-{case}-{seed}"})
    document = {"protocol": {"document_sha256": "same"}, "results": rows}
    return document, copy.deepcopy(document)


def test_recovered_failures_are_separate_from_numerical_changes(documents):
    before, after = documents
    for seed in (0, 1, 3):
        before["results"][6 * 6 + seed] = {"job": ["logged", seed], "_status": "crash"}
    summary = assess(before, after)
    assert summary["schema_replay_passed"]
    assert len(summary["unchanged_successful_predictions"]) == 39
    assert len(summary["recovered_failures"]) == 3
    assert summary["original_factor_reference_confirmation_gate_passed"] is False


@pytest.mark.parametrize("mutation", ["prediction", "data", "missing"])
def test_changed_or_missing_evidence_is_rejected(documents, mutation):
    if mutation == "prediction":
        documents[1]["results"][0]["prediction_digest"] = "changed"
    elif mutation == "data":
        documents[1]["results"][0]["input_digest"] = "changed"
    else:
        documents[1]["results"].pop()
    with pytest.raises(ValueError):
        assess(*documents)
