"""Scientific completeness and identity gates for the full predictive refresh."""
import copy

import pytest

from benches.report_competitive_refresh import assess, draw_scores, substitute


def _row(task, seed, method, tier, value):
    return {"task": task, "seed": seed, "method": method, "tier": tier,
            "_status": "success", "input_digest": f"data-{task}-{seed}",
            "partition_digest": f"split-{task}-{seed}",
            "results": [{"tier": tier, "scores": {"loss": value}}]}


@pytest.fixture
def documents():
    current, reference = [], []
    for task in ("a", "b"):
        for seed in range(3):
            current.append(_row(task, seed, "mboost_auto", "small", .72 if task == "a" else 1.2))
            for method, small, tuned in (("linear", 1., .95), ("catboost", .8, .7),
                                          ("mboost_auto", 1.1, 1.05)):
                for tier, value in (("small", small), ("tuned", tuned)):
                    reference.append(_row(task, seed, method, tier, value))
    def document(rows):
        return {"protocol": {"tier": "small", "specs": [
            [r["task"], r["seed"], r["method"], r["tier"]] for r in rows]}, "results": rows}
    return document(current), document(reference)


def test_gap_uses_strongest_complete_tier_and_distinguishes_simple_reference(documents):
    result = assess(*documents)
    a, b = result["gaps"]
    assert a["best_method"] == "catboost" and a["best_tier"] == "tuned"
    assert a["excess_pct"] == pytest.approx(100 * (.72 / .7 - 1))
    assert a["within_5pct"] and a["D11"] == 3
    assert not b["within_5pct"] and b["D11"] == 1
    assert result["passing_tasks"] == 1 and not result["all_tasks_within_5pct"]
    assert len(result["paired"]) == 6


@pytest.mark.parametrize("mutation, message", [
    ("missing_current", "missing planned"), ("missing_reference", "missing planned"),
    ("duplicate", "duplicate result"), ("partition", "mismatch"),
    ("nonfinite", "non-finite"),
])
def test_invalid_evidence_is_rejected(documents, mutation, message):
    current, reference = documents
    if mutation == "missing_current":
        current["results"].pop()
    elif mutation == "missing_reference":
        reference["results"].pop()
    elif mutation == "duplicate":
        current["results"].append(copy.deepcopy(current["results"][0]))
    elif mutation == "partition":
        current["results"][0]["partition_digest"] = "different"
    else:
        current["results"][0]["results"][0]["scores"]["loss"] = float("nan")
    with pytest.raises(ValueError, match=message):
        assess(current, reference)


def test_failure_keeps_task_unresolved_even_if_successful_subset_wins(documents):
    current, reference = documents
    original = current["results"][0]
    current["results"][0] = {"job": [original[k] for k in ("task", "seed", "method", "tier")],
                              "_status": "timeout", "failure": {"timeout_s": 240}}
    result = assess(current, reference)
    failed = result["gaps"][0]
    assert failed["completed"] == 2 and failed["failed"] == 1
    assert failed["excess_pct"] is None and failed["D11"] is None
    assert not failed["within_5pct"]
    assert len(result["failures"]) == 1


def _frozen(document):
    document["protocol"].update(id="competitive-refresh/1", document_sha256="p",
                                reference_sha256="r", scorecard_sha256="s", budget=32,
                                timeout_seconds=240, workers=1)
    return document


def _rerun(seeds, value=.5):
    rows = [_row("a", seed, "mboost_auto", "small", value) for seed in seeds]
    return _frozen({"protocol": {"tier": "small", "specs": [
        [r["task"], r["seed"], r["method"], r["tier"]] for r in rows]}, "results": rows})


def test_task_rerun_replaces_only_its_rows(documents):
    current, reference = documents
    base = _frozen(current)
    merged, tasks = substitute(base, _rerun(range(3)))
    assert tasks == ["a"] and len(base["results"]) == 6
    result, before = assess(merged, reference), assess(base, reference)
    assert result["gaps"][0]["current_loss"] == pytest.approx(.5)
    assert result["gaps"][1] == before["gaps"][1]
    draws = draw_scores(merged, reference, result["gaps"], tasks)
    assert [d["seed"] for d in draws] == [0, 1, 2]
    assert draws[0]["minus_best"]["loss"] == pytest.approx(.5 - .7)
    assert draws[0]["minus_simple"]["loss"] == pytest.approx(.5 - 1.)


def test_task_rerun_must_match_frozen_jobs(documents):
    base = _frozen(documents[0])
    with pytest.raises(ValueError, match="specs differ"):
        substitute(base, _rerun(range(2)))
    rerun = _rerun(range(3))
    rerun["protocol"]["budget"] = 128
    with pytest.raises(ValueError, match="budget"):
        substitute(base, rerun)
