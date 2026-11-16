"""Complete paired D11 profiles and guide-compliant gap tables."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from great_tables import GT

from benches.bench_encoding_tracks import DIRECTORY, METHODS, ROOT, ENCODINGS as TASKS

TRACKS = {"known_specification": "mboostrs_known", "discovery": "mboostrs_discover"}
COMPETITORS = ("catboost", "lightgbm", "xgboost", "histgb")


def mean_se(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("at least two finite paired observations are required")
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def index(document):
    expected = {(task, seed, method) for task in TASKS for seed in range(8) for method in METHODS}
    if {tuple(s) for s in document["protocol"]["specs"]} != expected or len(document["protocol"]["specs"]) != len(expected):
        raise ValueError("protocol omits a required encoding, method or seed")
    rows = {}
    for row in document["results"]:
        key = ((row["task"], row["seed"], row["method"]) if "task" in row else tuple(row["job"]))
        if key not in expected or key in rows:
            raise ValueError("unexpected or duplicate method/draw")
        if row["_status"] == "success" and (row.get("mse") is None or not np.isfinite(row["mse"])):
            raise ValueError("successful method has non-finite MSE")
        rows[key] = row
    if set(rows) != expected:
        raise ValueError("missing planned method/draw; retain explicit failures")
    for task in TASKS:
        for seed in range(8):
            successful = [rows[task, seed, m] for m in METHODS if rows[task, seed, m]["_status"] == "success"]
            if len({r["input_digest"] for r in successful}) > 1:
                raise ValueError("methods saw different development/test rows")
    for seed in range(8):
        views = [r for (task, s, method), r in rows.items() if s == seed and r["_status"] == "success"]
        if len({r["latent_digest"] for r in views}) > 1:
            raise ValueError("encoded views do not share the same latent rows")
    return rows


def summarize(document):
    rows = index(document)
    cells, gaps = [], []
    for task in TASKS:
        for method in METHODS:
            good = [rows[task, seed, method] for seed in range(8) if rows[task, seed, method]["_status"] == "success"]
            complete = len(good) == 8
            mean, se = mean_se([r["mse"] for r in good]) if complete else (None, None)
            cells.append({"task": task, "method": method, "planned": 8, "completed": len(good),
                          "mean_mse": mean, "mse_se": se})
        complete_peers = [c for c in cells if c["task"] == task and c["method"] in COMPETITORS and c["completed"] == 8]
        best = min(complete_peers, key=lambda c: c["mean_mse"]) if complete_peers else None
        for track, method in TRACKS.items():
            current = next(c for c in cells if c["task"] == task and c["method"] == method)
            gap = {"task": task, "track": track, "method": method,
                   "best_competitor": None if best is None else best["method"],
                   "complete_competitors": [c["method"] for c in complete_peers],
                   "completed": current["completed"], "gap_pct": None, "gap_se_pct": None,
                   "difference_mse": None, "difference_se": None, "paired": [],
                   "ridge_improvement": None, "ridge_improvement_se": None,
                   "d11_score": None, "within_1pct": None, "unresolved_at_2se": None}
            complete_ridge = all(rows[task, seed, "ridge"]["_status"] == "success" for seed in range(8))
            if current["completed"] == 8 and complete_ridge:
                improvement = [rows[task, seed, "ridge"]["mse"] - rows[task, seed, method]["mse"] for seed in range(8)]
                gain, se = mean_se(improvement)
                gap.update(ridge_improvement=gain, ridge_improvement_se=se,
                           d11_score=3 if gain > 2 * se else 2 if gain > 0 else 1)
            elif current["completed"] < 8:
                gap["d11_score"] = 1
            if current["completed"] == 8 and best is not None:
                paired = [{"seed": seed, "mboost_mse": rows[task, seed, method]["mse"],
                           "competitor_mse": rows[task, seed, best["method"]]["mse"]} for seed in range(8)]
                for p in paired:
                    p.update(gap_pct=100 * (p["mboost_mse"] / p["competitor_mse"] - 1),
                             difference_mse=p["mboost_mse"] - p["competitor_mse"])
                ratio, ratio_se = mean_se([p["gap_pct"] for p in paired])
                delta, delta_se = mean_se([p["difference_mse"] for p in paired])
                gap.update(gap_pct=ratio, gap_se_pct=ratio_se, difference_mse=delta,
                           difference_se=delta_se, paired=paired, within_1pct=ratio <= 1,
                           unresolved_at_2se=abs(ratio) <= 2 * ratio_se)
            gaps.append(gap)
    failures = [{"task": t, "seed": s, "method": m, **r} for (t, s, m), r in rows.items() if r["_status"] != "success"]
    return {"cells": cells, "gaps": gaps, "failures": failures,
            "all_jobs_complete": not failures, "all_measured_gaps_within_1pct": all(g["within_1pct"] is True for g in gaps)}



def compare_phases(before, after):
    for field in ("id", "specs", "document_sha256", "scorecard_sha256", "guide_sha256"):
        if before["protocol"][field] != after["protocol"][field]:
            raise ValueError("paired phase protocol changed")
    old, new = index(before), index(after)
    changed_peers = []
    replayed_peers = 0
    for key in old:
        b, a = old[key], new[key]
        if b["_status"] == a["_status"] == "success":
            if b["input_digest"] != a["input_digest"]:
                raise ValueError("paired phase rows changed")
            if key[2] in (*COMPETITORS, "ridge"):
                if b["prediction_digest"] != a["prediction_digest"]:
                    changed_peers.append(key)
                else:
                    replayed_peers += 1
    if changed_peers:
        raise ValueError("unchanged competitors failed exact paired replay")
    changes = []
    for task in TASKS:
        for track, method in TRACKS.items():
            pairs = [(old[task, seed, method], new[task, seed, method]) for seed in range(8)]
            good = [(b, a) for b, a in pairs if b["_status"] == a["_status"] == "success"]
            complete = len(good) == 8
            delta, delta_se = mean_se([a["mse"] - b["mse"] for b, a in good]) if complete else (None, None)
            pct, pct_se = mean_se([100 * (a["mse"] / b["mse"] - 1) for b, a in good]) if complete else (None, None)
            changes.append({"task": task, "track": track, "completed": len(good),
                "before_mse": float(np.mean([b["mse"] for b, a in good])) if complete else None,
                "after_mse": float(np.mean([a["mse"] for b, a in good])) if complete else None,
                "change_mse": delta, "change_se": delta_se, "change_pct": pct, "change_pct_se": pct_se,
                "unchanged_predictions": sum(b["prediction_digest"] == a["prediction_digest"] for b, a in good)})
    return {"changes": changes, "competitor_predictions_replayed": replayed_peers}


def headline(summary):
    records = []
    for track, mboost in TRACKS.items():
        for method in (mboost, *COMPETITORS, "ridge"):
            row = {"track": track, "method": method}
            for task in TASKS:
                cell = next(c for c in summary["cells"] if c["task"] == task and c["method"] == method)
                peers = [c["mean_mse"] for c in summary["cells"] if c["task"] == task
                         and c["method"] in (mboost, *COMPETITORS, "ridge") and c["mean_mse"] is not None]
                if cell["mean_mse"] is None:
                    row[task] = f"incomplete ({cell['completed']}/8)"
                else:
                    mean = f"{cell['mean_mse']:.3f}"
                    if round(cell["mean_mse"], 3) == round(min(peers), 3):
                        mean = f"<strong>{mean}</strong>"
                    row[task] = f"{mean} ({cell['mse_se']:.3f})"
            records.append(row)
    frame = pd.DataFrame(records)
    return GT(frame, rowname_col="method", groupname_col="track").tab_header(
        title="Predictor encoding: every method in both information tracks",
        subtitle="Mean outer-test MSE (SE), squared response units; lower is better within each encoding/track"
    ).tab_spanner("Encoding", list(TASKS)).fmt(lambda value: value, columns=list(TASKS)).tab_source_note(
        "Eight independent latent draws; SE=sample SD/sqrt(8). The five encodings reuse the same rows and are not forty independent samples. Best displayed means are bold.").tab_source_note(
        "Competitors see data only and share predictions across tracks. Four configurations are validation-tuned per method; Ridge is the prespecified reference. Incomplete cells have no successful-only mean.")


def gap_table(summary):
    frame = pd.DataFrame(summary["gaps"])[["track", "task", "best_competitor", "completed", "gap_pct", "gap_se_pct",
                                          "difference_mse", "difference_se", "within_1pct", "unresolved_at_2se"]]
    return GT(frame, rowname_col="task", groupname_col="track").tab_header(
        title="Encoding tracks: paired gap to the strongest complete named competitor",
        subtitle="Percent gap and paired Monte Carlo SE; negative favors mboostrs; target at most +1%"
    ).tab_spanner("Gap (%)", ["gap_pct", "gap_se_pct"]).tab_spanner("Paired MSE difference", ["difference_mse", "difference_se"]).fmt_number(
        columns=["gap_pct", "gap_se_pct"], decimals=2).fmt_number(columns=["difference_mse", "difference_se"], decimals=4).tab_source_note(
        "The competitor is selected retrospectively from mean test MSE. SE treats this selected winner as fixed and does not include method-selection uncertainty; it is descriptive.").tab_source_note(
        "Each row uses all eight matching draws. A point gap above 1% is filed even when within two SE of zero. Tracks and processes are never pooled.")


def scorecards(summary, revision):
    template = json.loads((ROOT / "docs/scorecard/report-template.json").read_text())
    profiles = []
    for gap in summary["gaps"]:
        profile = json.loads(json.dumps(template))
        profile.update(library="mboostrs", mode="automated pipeline", model_revision=revision,
                       benchmark_protocol_id="encoding-tracks/1", task=gap["task"], track=gap["track"])
        d11 = next(d for d in profile["dimensions"] if d["id"] == "D11")
        if gap["d11_score"] is not None:
            d11.update(score=gap["d11_score"], status="scored", evidence=["summary.json", "../PROTOCOL.md"],
                       justification=(f"Frozen D11 rule: Ridge minus mboostrs mean MSE {gap['ridge_improvement']}, "
                                      f"paired SE {gap['ridge_improvement_se']}; {gap['completed']}/8 successful draws. "
                                      "Focused generator and unequal search designs cannot establish levels 4/5. "
                                      "Known supplied terms earn no discovery credit; other dimensions are unmeasured."))
        profiles.append(profile)
    return {"scorecard_id": template["scorecard_id"], "assessment_timing": "rules frozen before outcomes",
            "profiles": profiles, "aggregate": None}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--phase", required=True)
    args = parser.parse_args()
    document = json.loads((DIRECTORY / f"{args.phase}.json").read_text())
    if document["protocol"]["document_sha256"] != hashlib.sha256((DIRECTORY / "PROTOCOL.md").read_bytes()).hexdigest():
        raise ValueError("frozen protocol changed")
    hashes = {r["_identity"]["components"]["mboostrs"]["hash"] for r in document["results"]}
    if len(hashes) != 1:
        raise ValueError("loaded library identity differs within the phase")
    summary = summarize(document)
    summary.update(reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   protocol_sha256=document["protocol"]["document_sha256"])
    directory = DIRECTORY / args.phase
    directory.mkdir(exist_ok=True)
    (directory / "summary.json").write_text(json.dumps(summary, indent=1, allow_nan=False) + "\n")
    (directory / "scorecard.json").write_text(json.dumps(scorecards(summary, document["protocol"]["library_source_revision"]), indent=1) + "\n")
    (directory / "headline.html").write_text(headline(summary).as_raw_html())
    (directory / "gaps.html").write_text(gap_table(summary).as_raw_html())
    if args.phase == "after":
        before = json.loads((DIRECTORY / "baseline.json").read_text())
        comparison = compare_phases(before, document)
        (DIRECTORY / "paired_improvement.json").write_text(json.dumps(comparison, indent=1, allow_nan=False) + "\n")
    print(json.dumps(summary["gaps"], indent=1))


if __name__ == "__main__":
    main()
