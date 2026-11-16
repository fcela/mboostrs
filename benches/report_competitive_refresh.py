"""Assess the complete predictive refresh without fitting or selecting models."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

from docs.book.cases.bench_tables import table

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "docs/book/benchmarks/competitive_refresh"


def key(row):
    return (row["task"], row["seed"], row["method"], row["tier"]) if "task" in row else tuple(row["job"][:4])


def indexed(document):
    planned = {tuple(s[:4]) for s in document["protocol"]["specs"]}
    if len(planned) != len(document["protocol"]["specs"]):
        raise ValueError("duplicate planned job")
    rows = {}
    for row in document["results"]:
        k = key(row)
        if k not in planned or k in rows:
            raise ValueError("unexpected or duplicate result")
        rows[k] = row
    if set(rows) != planned:
        raise ValueError("missing planned results; keep explicit failure rows")
    return rows


def score(row):
    if row["_status"] != "success":
        return None
    matches = [r for r in row["results"] if r["tier"] == row["tier"]]
    if len(matches) != 1:
        raise ValueError("missing or duplicate scored tier")
    value = matches[0]["scores"]["loss"]
    if value is None or not np.isfinite(value):
        raise ValueError("successful result has non-finite loss")
    return float(value)


def assess(document, reference):
    fresh, historical = indexed(document), indexed(reference)
    planned = list(fresh)
    tasks = sorted({k[0] for k in planned})
    tier = document["protocol"]["tier"]
    if {k[2] for k in planned} != {"mboost_auto"} or {k[3] for k in planned} != {tier}:
        raise ValueError("refresh method/tier differs from protocol")
    groups = defaultdict(list)
    failures, paired = [], []
    for k, row in fresh.items():
        groups[k[0], "mboostrs autospec (current)", k[3], "current"].append(row)
        if score(row) is None:
            failures.append({"task": k[0], "seed": k[1], "status": row["_status"],
                             "failure": row.get("failure")})
    for k, row in historical.items():
        if k[0] not in tasks:
            continue
        current = fresh.get((k[0], k[1], "mboost_auto", tier))
        if current is None:
            raise ValueError("refresh omitted an original outer draw")
        if score(row) is not None and score(current) is not None:
            if any(row[d] != current[d] for d in ("input_digest", "partition_digest")):
                raise ValueError(f"input/partition mismatch: {k}")
        groups[k[0], k[2], k[3], "historical"].append(row)
    cells = []
    for (task, method, cell_tier, measurement), rows in groups.items():
        expected = sum(k[0] == task for k in fresh)
        good = [r for r in rows if score(r) is not None]
        values = [score(r) for r in good]
        cells.append({"task": task, "method": method, "tier": cell_tier,
                      "measurement": measurement, "planned": expected, "n": len(good),
                      "failed": expected - len(good),
                      "loss": float(np.mean(values)) if values else None,
                      "loss_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                      "complete": len(rows) == expected and len(good) == expected})
    gaps = []
    for task in tasks:
        candidates = [c for c in cells if c["task"] == task and c["complete"]
                      and c["measurement"] == "historical" and not c["method"].startswith("mboost")]
        if not candidates:
            raise ValueError(f"no complete competitor for {task}")
        best = min(candidates, key=lambda c: c["loss"])
        current = next(c for c in cells if c["task"] == task and c["measurement"] == "current")
        baseline = next(c for c in cells if c["task"] == task and c["method"] == "linear"
                        and c["tier"] == tier)
        excess = 100 * (current["loss"] / best["loss"] - 1) if current["complete"] else None
        grade = (3 if current["loss"] < baseline["loss"] else 1
                 ) if current["complete"] and baseline["complete"] else None
        gaps.append({"task": task, "current_tier": tier, "completed": current["n"],
                     "planned": current["planned"], "failed": current["failed"],
                     "current_loss": current["loss"] if current["complete"] else None,
                     "loss_sd": current["loss_sd"] if current["complete"] else None,
                     "best_method": best["method"], "best_tier": best["tier"],
                     "best_loss": best["loss"], "excess_pct": excess,
                     "within_5pct": excess is not None and excess <= 5,
                     "simple_loss": baseline["loss"] if baseline["complete"] else None,
                     "simple_tier": tier,
                     "D11": grade})
        for c in (c for c in cells if c["task"] == task):
            c["excess_pct"] = 100 * (c["loss"] / best["loss"] - 1) if c["complete"] else None
        for k, row in fresh.items():
            if k[0] != task or score(row) is None:
                continue
            simple = historical[task, k[1], "linear", tier]
            old = historical[task, k[1], "mboost_auto", tier]
            paired.append({"task": task, "seed": k[1], "current_loss": score(row),
                           "simple_loss": score(simple), "original_autospec_loss": score(old),
                           "excess_over_simple": score(row) - score(simple) if score(simple) is not None else None,
                           "change_from_original": score(row) - score(old) if score(old) is not None else None})
    return {"tier": tier, "planned_jobs": len(fresh), "failures": failures,
            "cells": cells, "gaps": gaps, "paired": paired,
            "all_tasks_within_5pct": all(g["within_5pct"] for g in gaps),
            "passing_tasks": sum(g["within_5pct"] for g in gaps), "planned_tasks": len(tasks)}


BINARY_SCORES = ("loss", "brier", "auc", "calibrated_loss")


def substitute(document, rerun):
    """Replace the base artifact's rows for the rerun's tasks with its rows.

    The rerun must plan exactly the base artifact's jobs for those tasks under
    the same frozen protocol, scorecard and historical reference; the base
    artifact file itself is left unchanged as the earlier measurement.
    """
    for field in ("id", "tier", "document_sha256", "reference_sha256", "scorecard_sha256",
                  "budget", "timeout_seconds", "workers"):
        if rerun["protocol"][field] != document["protocol"][field]:
            raise ValueError(f"rerun protocol differs: {field}")
    tasks = {s[0] for s in rerun["protocol"]["specs"]}
    planned = [s for s in document["protocol"]["specs"] if s[0] in tasks]
    if sorted(map(list, rerun["protocol"]["specs"])) != sorted(map(list, planned)):
        raise ValueError("rerun specs differ from the base artifact's jobs for its tasks")
    merged = dict(document)
    merged["results"] = [r for r in document["results"] if key(r)[0] not in tasks] + rerun["results"]
    return merged, sorted(tasks)


def draw_scores(fresh_document, reference, gaps, tasks):
    """Per-draw binary scores against the simple reference and the best comparator."""
    fresh, historical = indexed(fresh_document), indexed(reference)
    tier = fresh_document["protocol"]["tier"]
    rows = []
    for gap in (g for g in gaps if g["task"] in tasks):
        for k, row in sorted((k, r) for k, r in fresh.items() if k[0] == gap["task"]):
            def scores(r):
                if r["_status"] != "success":
                    return None
                match = next(x for x in r["results"] if x["tier"] == r["tier"])
                return {s: match["scores"].get(s) for s in BINARY_SCORES}
            def minus(a, b):
                if a is None or b is None:
                    return None
                return {s: a[s] - b[s] if a[s] is not None and b[s] is not None else None
                        for s in BINARY_SCORES}
            current = scores(row)
            simple = scores(historical[gap["task"], k[1], "linear", tier])
            best = scores(historical[gap["task"], k[1], gap["best_method"], gap["best_tier"]])
            rows.append({
                "task": gap["task"], "seed": k[1], "status": row["_status"],
                "failure": row.get("failure"), "current": current,
                "simple_reference": {"method": "linear", "tier": tier, "scores": simple},
                "best_comparator": {"method": gap["best_method"], "tier": gap["best_tier"],
                                    "scores": best},
                "minus_simple": minus(current, simple), "minus_best": minus(current, best)})
    return rows


def per_task_table(summary):
    frame = pl.DataFrame(summary["cells"]).drop("complete", "planned")
    return table(frame, group="task", title="Current autospec and the frozen competitive study",
                 subtitle="Mean test MSE or binary log loss; lower is better; historical methods retain both search tiers",
                 spanners={"Prediction": ["loss", "loss_sd", "excess_pct"],
                           "Accounting": ["tier", "measurement", "n", "failed"]},
                 better={"loss": "min"}, formats={"excess_pct": 1},
                 notes=("Ten real panels: three overlapping outer splits; three synthetic panels: five independent draws. SD is descriptive spread.",
                        "The best complete historical non-mboostrs method/tier is a retrospective test-selected oracle reference. Inputs and partitions match.",
                        "Current failures remain visible. Means on incomplete tasks condition on successful jobs and cannot pass the 5% gate.",
                        "Historical results and search budgets differ from the current pipeline; no contemporary speed ratio is inferred."))


def headline(summary):
    grouped = defaultdict(list)
    for cell in summary["cells"]:
        grouped[cell["method"], cell["tier"], cell["measurement"]].append(cell)
    rows = []
    for (method, tier, measurement), cells in grouped.items():
        complete = [c for c in cells if c["complete"]]
        excess = [c["excess_pct"] for c in complete]
        rows.append({"method": method, "tier": tier, "measurement": measurement,
                     "complete_tasks": len(complete), "incomplete_tasks": len(cells) - len(complete),
                     "within_5pct": sum(c["excess_pct"] <= 5 for c in complete),
                     "median_excess_pct": float(np.median(excess)) if excess else None,
                     "worst_excess_pct": max(excess) if excess else None})
    return table(pl.DataFrame(rows), title="Predictive battery: current autospec and every recorded method",
                 subtitle="Complete-task counts and excess test loss in percent; more passing tasks and lower excess are better",
                 spanners={"Evidence": ["tier", "measurement", "complete_tasks", "incomplete_tasks"],
                           "Competitor gaps": ["within_5pct", "median_excess_pct", "worst_excess_pct"]},
                 better={"within_5pct": "max"}, formats={"within_5pct": 0, "median_excess_pct": 1,
                                                        "worst_excess_pct": 1},
                 notes=(f"{summary['planned_tasks']} tasks; each task requires all three real-data splits or five synthetic draws before passing.",
                        "The reference is the best complete recorded non-mboostrs method/tier on each task, selected retrospectively by test loss.",
                        "Median and worst excess condition on complete tasks. Consult incomplete-task counts; missing tasks never pass.",
                        "Current and historical measurements retain separate labels. Counts are descriptive across this battery, not a weighted scorecard grade."))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--tier", choices=("small", "tuned"), default="small")
    parser.add_argument("--rerun", default=None,
                        help="artifact from refresh_competitive --tasks whose rows replace "
                             "the base artifact's rows for those tasks")
    args = parser.parse_args()
    document = json.loads((DIRECTORY / f"{args.tier}.json").read_text())
    base_revision, rerun_tasks, rerun_record = document["commit"], [], None
    if args.rerun is not None:
        rerun_path = DIRECTORY / args.rerun
        rerun = json.loads(rerun_path.read_text())
        document, rerun_tasks = substitute(document, rerun)
        document["commit"] = rerun["commit"]
        rerun_record = {"artifact": args.rerun, "tasks": rerun_tasks,
                        "sha256": hashlib.sha256(rerun_path.read_bytes()).hexdigest(),
                        "source_revision": rerun["commit"], "build_info": rerun["build_info"],
                        "base_artifact": f"{args.tier}.json", "base_source_revision": base_revision}
    reference_path = ROOT / document["protocol"]["reference"]
    if hashlib.sha256(reference_path.read_bytes()).hexdigest() != document["protocol"]["reference_sha256"]:
        raise ValueError("historical reference changed")
    for path, field in ((DIRECTORY / "PROTOCOL.md", "document_sha256"),
                        (ROOT / "docs/scorecard/scorecard.md", "scorecard_sha256")):
        if hashlib.sha256(path.read_bytes()).hexdigest() != document["protocol"][field]:
            raise ValueError(f"frozen protocol/scorecard changed: {path}")
    reference = json.loads(reference_path.read_text())
    summary = assess(document, reference)
    summary["assessment"] = {"reporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             "protocol": document["protocol"], "source_revision": document["commit"]}
    if rerun_record is not None:
        summary["assessment"]["rerun"] = rerun_record
        summary["rerun_draws"] = draw_scores(document, reference, summary["gaps"], rerun_tasks)
    (DIRECTORY / f"{args.tier}_summary.json").write_text(json.dumps(summary, indent=1, allow_nan=False) + "\n")
    (DIRECTORY / f"{args.tier}_headline.html").write_text(headline(summary).as_raw_html())
    (DIRECTORY / f"{args.tier}_per_task.html").write_text(per_task_table(summary).as_raw_html())
    template = json.loads((ROOT / "docs/scorecard/report-template.json").read_text())
    template.update(library="mboostrs", model_revision=document["commit"],
                    benchmark_protocol_id="competitive-refresh/1", mode="automated pipeline",
                    task_profiles=[{
                        "task": g["task"], "dimension": "D11", "score": g["D11"],
                        "status": "scored" if g["D11"] is not None else "unmeasured",
                        "evidence": [f"{args.tier}_summary.json", "PROTOCOL.md",
                                     *([args.rerun] if g["task"] in rerun_tasks else [])],
                        "justification": (
                            "Incomplete outer evidence or simple reference; no grade assigned."
                            if g["D11"] is None else
                            f"All {g['planned']} outer jobs completed. Mean test loss {g['current_loss']:.6g} "
                            f"{'improves on' if g['D11'] == 3 else 'fails to beat'} the prespecified "
                            f"same-tier Ridge/logistic reference {g['simple_loss']:.6g}; paired draws "
                            "and descriptive spread are reported. Historical comparators, unequal "
                            "search budgets and known tasks cannot establish level 4 or 5.")}
                        for g in summary["gaps"]])
    # The overall D11 remains unmeasured: differing per-task anchors must not
    # be averaged into a seemingly complete canonical grade.
    audit = next(d for d in template["dimensions"] if d["id"] == "D10")
    if all(r.get("_identity", {}).get("components", {}).get("mboostrs", {}).get("hash")
           for r in document["results"]):
        audit.update(status="scored", score=5, evidence=[f"{args.tier}.json", f"{args.tier}_summary.json"],
                     justification="This instrument retains code, environment, loaded package/binary and input/partition/output hashes, "
                     "explicit failed-job records and a reproducible assessment. It is not a general end-user workflow score.")
    template["scorecard_sha256"] = document["protocol"]["scorecard_sha256"]
    (DIRECTORY / f"{args.tier}_scorecard.json").write_text(json.dumps(template, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("passing_tasks", "planned_tasks", "all_tasks_within_5pct")}))


if __name__ == "__main__":
    main()
