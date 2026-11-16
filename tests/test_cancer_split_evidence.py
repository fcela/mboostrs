"""Reject incomplete, leaked, or numerically inconsistent split evidence."""
import copy
import math

import pytest

from benches.report_cancer_splits_541 import assess


@pytest.fixture
def evidence():
    methods = ["autospec", "linear_small", "linear_tuned"]
    results = {}
    for method, p in zip(methods, (.8, .7, .9), strict=True):
        results[method] = {"predictions": [p, 1-p], "row_loss": [-math.log(p)]*2,
                           "scores": {"loss": -math.log(p), "brier": (1-p)**2,
                                      "auc": 1., "calibrated_loss": .1},
                           "top_five_loss_share": 1.}
    row = {"seed": 1, "_status": "success", "indices": [[0], [1], [2], [3, 4]],
           "autospec": {"fits": 20, "mstop": 100}, "test_response": [1, 0], "results": results,
           "prevalence": {"test": .5}}
    return {"protocol": {"seeds": [1, 2], "methods": methods,
                          "autospec": {"budget": 32}, "target_excess_pct": 5},
            "results": [row, {**copy.deepcopy(row), "seed": 2}]}


def test_paired_comparisons_keep_both_references(evidence):
    summary = assess(evidence)
    simple, tuned = summary["comparisons"]
    assert simple["paired_difference"]["mean"] == pytest.approx(math.log(.7/.8))
    assert tuned["paired_difference"]["mean"] == pytest.approx(math.log(.9/.8))
    assert simple["within_5pct"] and not tuned["within_5pct"]
    assert len(summary["observations_by_excess_loss"]) == 2
    assert summary["observations_by_excess_loss"][0]["test_appearances"] == 2


@pytest.mark.parametrize("mutation, message", [
    ("missing", "missing planned"), ("duplicate", "duplicate result"),
    ("leakage", "disjoint"), ("predictions", "stored losses"),
    ("score", "mean loss"), ("budget", "budget exceeded"),
])
def test_invalid_evidence_cannot_be_ranked(evidence, mutation, message):
    row = evidence["results"][0]
    if mutation == "missing":
        evidence["results"].pop()
    elif mutation == "duplicate":
        evidence["results"].append(row)
    elif mutation == "leakage":
        row["indices"][0] = [3]
    elif mutation == "predictions":
        row["results"]["autospec"]["predictions"].reverse()
    elif mutation == "score":
        row["results"]["autospec"]["scores"]["loss"] += 1
    else:
        row["autospec"]["fits"] = 33
    with pytest.raises(ValueError, match=message):
        assess(evidence)


def test_failure_is_visible_and_prevents_passing_even_when_successes_win(evidence):
    evidence["results"][1] = {"job": [2], "_status": "timeout"}
    summary = assess(evidence)
    assert len(summary["failures"]) == 1
    assert summary["completed"] == 1
    assert all(not row["within_5pct"] for row in summary["comparisons"])
