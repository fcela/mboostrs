"""Encoding reports retain complete paired draws and do not pool type views."""
import copy

import numpy as np
import pytest

from benches.bench_encoding_tracks import ENCODINGS, METHODS
from benches.report_encoding_tracks import compare_phases, index, scorecards, summarize


@pytest.fixture
def document():
    specs = [[t, s, m] for t in ENCODINGS for s in range(8) for m in METHODS]
    rows = [{"task": t, "seed": s, "method": m, "_status": "success",
             "mse": .5 if m.startswith("mboostrs") else 2. if m == "ridge" else 1.,
             "input_digest": f"encoded-{t}-{s}", "latent_digest": f"latent-{s}",
             "prediction_digest": f"prediction-{t}-{s}-{m}"} for t, s, m in specs]
    return {"protocol": {"id": "encoding-tracks/1", "specs": specs, "document_sha256": "protocol",
                         "scorecard_sha256": "card", "guide_sha256": "guide"}, "results": rows}


def test_primary_se_uses_eight_draws_per_encoding(document):
    for r in document["results"]:
        if r["method"] == "catboost":
            r["mse"] = .7 + .05 * r["seed"]
    summary = summarize(document)
    assert len(summary["gaps"]) == 10
    gap = summary["gaps"][0]
    expected = 100 * (.5 / (.7 + .05 * np.arange(8)) - 1)
    assert gap["best_competitor"] == "catboost" and len(gap["paired"]) == 8
    assert gap["gap_pct"] == pytest.approx(expected.mean())
    assert gap["gap_se_pct"] == pytest.approx(expected.std(ddof=1) / np.sqrt(8))


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "nonfinite", "encoded", "latent", "plan"])
def test_missing_or_incompatible_views_are_rejected(document, mutation):
    if mutation == "missing":
        document["results"].pop()
    elif mutation == "duplicate":
        document["results"].append(copy.deepcopy(document["results"][0]))
    elif mutation == "nonfinite":
        document["results"][0]["mse"] = float("nan")
    elif mutation == "encoded":
        document["results"][0]["input_digest"] = "changed"
    elif mutation == "latent":
        for r in document["results"]:
            if r["task"] == "indicator" and r["seed"] == 0:
                r["latent_digest"] = "different-latent-rows"
    else:
        document["protocol"]["specs"].pop()
    with pytest.raises(ValueError):
        index(document)


def test_failed_cells_are_not_success_only_means(document):
    for r in document["results"]:
        if r["task"] == "indicator" and r["seed"] == 0 and r["method"] in ("catboost", "mboostrs_discover"):
            r["_status"] = "timeout"
    summary = summarize(document)
    gap = next(g for g in summary["gaps"] if g["task"] == "indicator" and g["track"] == "discovery")
    assert gap["completed"] == 7 and gap["gap_pct"] is None
    assert "catboost" not in gap["complete_competitors"]
    assert gap["d11_score"] == 1 and not summary["all_jobs_complete"]


def test_no_scorecard_pooling_or_unmeasured_grade(document):
    cards = scorecards(summarize(document), "source")
    assert len(cards["profiles"]) == 10 and cards["aggregate"] is None
    for p in cards["profiles"]:
        assert len(p["dimensions"]) == 12 and p["aggregate"] is None
        assert sum(d["status"] == "unmeasured" for d in p["dimensions"]) == 11
        assert next(d for d in p["dimensions"] if d["id"] == "D11")["score"] == 3


def test_paired_repair_retains_each_encoding_and_exact_peers(document):
    after = copy.deepcopy(document)
    for r in after["results"]:
        if r["task"] == "indicator" and r["method"] == "mboostrs_discover":
            r["mse"] = .25
            r["prediction_digest"] = "new-indicator-prediction"
    result = compare_phases(document, after)
    assert result["competitor_predictions_replayed"] == 200
    row = next(r for r in result["changes"] if r["task"] == "indicator" and r["track"] == "discovery")
    assert row["change_mse"] == -.25 and row["change_se"] == 0. and row["change_pct"] == -50.


@pytest.mark.parametrize("mutation", ["peer", "protocol"])
def test_changed_peer_or_protocol_cannot_support_a_repair_claim(document, mutation):
    after = copy.deepcopy(document)
    if mutation == "peer":
        next(r for r in after["results"] if r["method"] == "catboost")["prediction_digest"] = "changed"
    else:
        after["protocol"]["document_sha256"] = "changed"
    with pytest.raises(ValueError):
        compare_phases(document, after)
