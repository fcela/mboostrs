"""Paired reporting keeps tracks, complete draws and comparator selection explicit."""
import copy

import numpy as np
import pytest

from benches.bench_predictive_tracks import METHODS, TASKS
from benches.report_predictive_tracks import index, mean_se, scorecards, summarize


@pytest.fixture
def document():
    specs = [[task, seed, method] for task in TASKS for seed in range(8) for method in METHODS]
    rows = [{"task": t, "seed": s, "method": m, "_status": "success",
             "mse": .5 if m.startswith("mboostrs") else 2. if m == "ridge" else 1.,
             "input_digest": f"data-{t}-{s}"} for t, s, m in specs]
    return {"protocol": {"specs": specs}, "results": rows}


def test_primary_gap_is_the_mean_of_paired_ratios_and_se(document):
    for r in document["results"]:
        if r["method"] == "catboost":
            r["mse"] = .6 + .1 * r["seed"]
    summary = summarize(document)
    gap = summary["gaps"][0]
    assert gap["best_competitor"] == "catboost"
    ratios = 100 * (.5 / (.6 + .1 * np.arange(8)) - 1)
    assert gap["gap_pct"] == pytest.approx(ratios.mean())
    assert gap["gap_se_pct"] == pytest.approx(ratios.std(ddof=1) / np.sqrt(8))
    assert len(gap["paired"]) == 8
    assert gap["d11_score"] == 3


def test_failed_peer_is_excluded_and_failed_mboost_is_not_a_success_only_mean(document):
    for r in document["results"]:
        if r["task"] == "smooth" and r["seed"] == 0 and r["method"] in ("catboost", "mboostrs_discover"):
            r["_status"] = "timeout"
    summary = summarize(document)
    gap = next(g for g in summary["gaps"] if g["task"] == "smooth" and g["track"] == "discovery")
    assert "catboost" not in gap["complete_competitors"]
    assert gap["gap_pct"] is None and gap["paired"] == [] and gap["d11_score"] == 1
    cell = next(c for c in summary["cells"] if c["task"] == "smooth" and c["method"] == "mboostrs_discover")
    assert cell["completed"] == 7 and cell["mean_mse"] is None


@pytest.mark.parametrize("mutation, message", [("omit", "missing planned"), ("duplicate", "duplicate"),
    ("data", "different"), ("nonfinite", "non-finite"), ("plan", "required")])
def test_invalid_evidence_cannot_silently_change_the_battery(document, mutation, message):
    if mutation == "omit":
        document["results"].pop()
    elif mutation == "duplicate":
        document["results"].append(copy.deepcopy(document["results"][0]))
    elif mutation == "data":
        document["results"][0]["input_digest"] = "different"
    elif mutation == "nonfinite":
        document["results"][0]["mse"] = float("nan")
    else:
        document["protocol"]["specs"].pop()
    with pytest.raises(ValueError, match=message):
        index(document)


def test_absence_of_all_complete_tree_peers_remains_unmeasured(document):
    for r in document["results"]:
        if r["method"] in ("catboost", "lightgbm", "xgboost", "histgb"):
            r["_status"] = "crash"
    assert all(g["gap_pct"] is None and g["within_1pct"] is None for g in summarize(document)["gaps"])


def test_profiles_measure_only_d11_and_do_not_pool_tracks(document):
    for r in document["results"]:
        if r["method"] == "mboostrs_discover" and r["task"] == "factor":
            r["mse"] = 3.
    card = scorecards(summarize(document), "revision")
    assert len(card["profiles"]) == 6 and card["aggregate"] is None
    for profile in card["profiles"]:
        assert len(profile["dimensions"]) == 12 and profile["aggregate"] is None
        assert sum(d["status"] == "unmeasured" for d in profile["dimensions"]) == 11
        d11 = next(d for d in profile["dimensions"] if d["id"] == "D11")
        assert d11["score"] == (1 if profile["task"] == "factor" and profile["track"] == "discovery" else 3)


def test_uncertainty_requires_multiple_finite_observations():
    with pytest.raises(ValueError):
        mean_se([1.])


def paired_documents(document):
    before = copy.deepcopy(document)
    before["protocol"].update(id="same", document_sha256="protocol", scorecard_sha256="card", guide_sha256="guide")
    for row in before["results"]:
        row["prediction_digest"] = f"prediction-{row['task']}-{row['seed']}-{row['method']}"
    return before, copy.deepcopy(before)


def test_paired_repair_change_uses_every_draw_and_keeps_competitors_fixed(document):
    from benches.report_predictive_tracks import compare_phases
    before, after = paired_documents(document)
    for row in after["results"]:
        if row["task"] == "factor" and row["method"] == "mboostrs_discover":
            row["mse"] = .25
            row["prediction_digest"] = "changed-factor"
    comparison = compare_phases(before, after)
    assert comparison["competitor_predictions_replayed"] == 120
    factor = next(r for r in comparison["changes"] if r["task"] == "factor" and r["track"] == "discovery")
    assert factor["change_mse"] == -.25 and factor["change_se"] == 0.
    assert factor["change_pct"] == -50. and factor["unchanged_predictions"] == 0


@pytest.mark.parametrize("mutation", ["competitor", "protocol"])
def test_changed_competitor_or_protocol_prevents_a_paired_repair_claim(document, mutation):
    from benches.report_predictive_tracks import compare_phases
    before, after = paired_documents(document)
    if mutation == "competitor":
        next(r for r in after["results"] if r["method"] == "catboost")["prediction_digest"] = "different"
    else:
        after["protocol"]["document_sha256"] = "different"
    with pytest.raises(ValueError):
        compare_phases(before, after)
