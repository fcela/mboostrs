"""The 5% comparison requires every planned draw and matched inputs."""
import copy
import json
from pathlib import Path

import pytest

from benches.report_pair_form_stopping import assess

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "docs/book/benchmarks/pair_form_stopping"


@pytest.fixture
def documents():
    return [json.loads(p.read_text()) for p in (
        DIRECTORY / "before.json", DIRECTORY / "after.json",
        ROOT / "docs/benchmarks/competitive_shootout_results.json")]


def test_published_summary_is_reproducible(documents):
    result = assess(*documents)
    assert result == json.loads((DIRECTORY / "summary.json").read_text())
    assert len(result["paired"]) == 15
    assert all(r["fits_unchanged"] for r in result["paired"])
    assert all(r["prediction_unchanged"] for r in result["paired"]
               if r["task"] != "joint_boundary")
    joint = next(r for r in result["supplemental_best_any_tier"] if r["task"] == "joint_boundary")
    assert joint["best_method"] == "catboost" and joint["best_tier"] == "tuned"
    assert joint["within_5pct"]


@pytest.mark.parametrize("mutation, message", [
    ("missing", "missing"), ("duplicate", "duplicate"),
    ("partition", "inputs"), ("comparator", "methods are missing"),
])
def test_incompatible_or_incomplete_evidence_is_refused(documents, mutation, message):
    if mutation == "missing":
        documents[1]["results"].pop()
    elif mutation == "duplicate":
        documents[1]["results"].append(copy.deepcopy(documents[1]["results"][0]))
    elif mutation == "partition":
        documents[1]["results"][0]["partition_digest"] = "different"
    else:
        documents[2]["results"] = [r for r in documents[2]["results"]
                                    if r.get("method") != "catboost"]
    with pytest.raises(ValueError, match=message):
        assess(*documents)


def test_a_failed_draw_keeps_its_task_unresolved(documents):
    row = documents[1]["results"][0]
    row.update(_status="timeout", loss=None, failure={"timeout_s": 240})
    result = assess(*documents)
    gap = next(g for g in result["gaps"] if g["task"] == row["task"])
    assert len(result["failures"]) == 1
    assert gap["excess_pct"] is None
    assert not gap["within_5pct"]
    assert not result["all_tasks_within_5pct"]
