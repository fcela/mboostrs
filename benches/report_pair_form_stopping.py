"""Validate and report paired replay results; never fit or select a model."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "docs/book/benchmarks/pair_form_stopping"
sys.path.insert(0, str(ROOT / "docs/book/cases"))
from bench_tables import table  # noqa: E402


def read(name):
    return json.loads((DIRECTORY / name).read_text())


def assess(before, after, reference):
    """Pair every planned draw; refuse incompatible or selectively missing evidence."""
    specs = [tuple(s) for s in before["protocol"]["specs"]]
    if specs != [tuple(s) for s in after["protocol"]["specs"]] or len(set(specs)) != len(specs):
        raise ValueError("planned draws differ or contain duplicates")
    if before["protocol"]["document_sha256"] != after["protocol"]["document_sha256"]:
        raise ValueError("protocol changed between phases")
    indexed = []
    failures = []
    for phase, document in (("before", before), ("after", after)):
        rows = {}
        for row in document["results"]:
            key = (row["task"], row["seed"]) if "task" in row else tuple(row["job"])
            if key in rows or key not in specs:
                raise ValueError("unexpected or duplicate result")
            rows[key] = row
            if (row["_status"] != "success" or row.get("loss") is None
                    or not np.isfinite(row["loss"])):
                failures.append({"phase": phase, "task": key[0], "seed": key[1],
                                 "status": row["_status"], "failure": row.get("failure")})
        if set(rows) != set(specs):
            raise ValueError("planned results are missing; retain explicit failure rows")
        indexed.append(rows)
    b, a = indexed
    paired = []
    headline = []
    for task in sorted({s[0] for s in specs}):
        seeds = [s[1] for s in specs if s[0] == task]
        for phase, rows in (("mboostrs before", b), ("mboostrs after", a)):
            good = [rows[task, seed] for seed in seeds
                    if rows[task, seed]["_status"] == "success"
                    and rows[task, seed].get("loss") is not None]
            values = [r["loss"] for r in good]
            headline.append({"task": task, "method": phase, "n": len(good),
                             "loss": float(np.mean(values)) if values else None,
                             "loss_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                             "failures": len(seeds) - len(good), "reference_period": "current"})
        refs = [r for r in reference["results"] if r.get("task") == task
                and r.get("seed") in seeds and r.get("tier") == "small"
                and not r.get("method", "").startswith("mboost")]
        expected_methods = {s[2] for s in reference["protocol"]["specs"]
                            if s[0] == task and not s[2].startswith("mboost")}
        if {r["method"] for r in refs} != expected_methods:
            raise ValueError(f"comparator methods are missing: {task}")
        for row in refs:
            key = (task, row["seed"])
            for current in (b[key], a[key]):
                if current["_status"] != "success":
                    continue
                if any(current[k] != row[k] for k in ("input_digest", "partition_digest")):
                    raise ValueError(f"comparator inputs/partitions differ: {key}")
        for method in sorted({r["method"] for r in refs}):
            selected = [r for r in refs if r["method"] == method]
            if len(selected) != len(seeds) or {r["seed"] for r in selected} != set(seeds):
                raise ValueError(f"missing or duplicate reference draws: {task}/{method}")
            good = [r for r in selected if r["_status"] == "success"]
            values = [next(x["scores"]["loss"] for x in r["results"] if x["tier"] == "small")
                      for r in good]
            headline.append({"task": task, "method": method, "n": len(good),
                             "loss": float(np.mean(values)) if values else None,
                             "loss_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                             "failures": len(seeds) - len(good), "reference_period": "historical"})
        if not refs:
            raise ValueError(f"no comparator evidence: {task}")
        for seed in seeds:
            x, y = b[task, seed], a[task, seed]
            if x["_status"] != "success" or y["_status"] != "success":
                continue
            if any(x[k] != y[k] for k in ("input_digest", "partition_digest")):
                raise ValueError("before/after inputs differ")
            paired.append({"task": task, "seed": seed, "before": x["loss"],
                           "after": y["loss"], "change_pct": 100 * (y["loss"] / x["loss"] - 1),
                           "before_mstop": x["mstop"], "after_mstop": y["mstop"],
                           "fits_unchanged": x["fits"] == y["fits"],
                           "prediction_unchanged": x["prediction_digest"] == y["prediction_digest"]})
    gaps = []
    for task in sorted({s[0] for s in specs}):
        complete = [h for h in headline if h["task"] == task and not h["failures"]]
        candidates = [h for h in complete if h["reference_period"] == "historical"]
        best = min(candidates, key=lambda h: h["loss"]) if candidates else None
        for row in (h for h in headline if h["task"] == task):
            row["excess_pct"] = (100 * (row["loss"] / best["loss"] - 1)
                                 if best and not row["failures"] else None)
        current = next(h for h in headline if h["task"] == task and h["method"] == "mboostrs after")
        gaps.append({"task": task, "best_competitor": best["method"] if best else None,
                     "excess_pct": current["excess_pct"],
                     "within_5pct": current["excess_pct"] is not None and current["excess_pct"] <= 5})
    # Supplemental comparison against the strongest recorded search tier,
    # keeping the frozen small-tier primary comparison above unchanged.
    strongest = []
    for task in sorted({s[0] for s in specs}):
        seeds = {s[1] for s in specs if s[0] == task}
        groups = defaultdict(list)
        for row in reference["results"]:
            if (row.get("task") != task or row.get("seed") not in seeds
                    or row.get("method", "").startswith("mboost")):
                continue
            for result in row.get("results", []):
                groups[row["method"], result["tier"]].append((row, result))
        complete = {}
        for key, values in groups.items():
            if len(values) != len(seeds) or {r["seed"] for r, _ in values} != seeds:
                continue
            if any(r["_status"] != "success" for r, _ in values):
                continue
            for row, _ in values:
                current = a[task, row["seed"]]
                if current["_status"] == "success" and any(
                        current[k] != row[k] for k in ("input_digest", "partition_digest")):
                    raise ValueError("supplemental comparator inputs differ")
            complete[key] = float(np.mean([v["scores"]["loss"] for _, v in values]))
        best = min(complete, key=complete.get)
        current = next(h for h in headline if h["task"] == task and h["method"] == "mboostrs after")
        excess = None if current["failures"] else 100 * (current["loss"] / complete[best] - 1)
        strongest.append({"task": task, "best_method": best[0], "best_tier": best[1],
                          "best_loss": complete[best], "after_loss": current["loss"],
                          "excess_pct": excess, "within_5pct": excess is not None and excess <= 5})
    return {"headline": headline, "paired": paired, "failures": failures,
            "gaps": gaps, "all_tasks_within_5pct": all(g["within_5pct"] for g in gaps),
            "supplemental_best_any_tier": strongest}


def headline_table(summary):
    return table(pl.DataFrame(summary["headline"]), group="task",
                 title="Pair-form stopping: prediction on independent draws",
                 subtitle="Test MSE and excess over the best complete historical comparator; lower is better",
                 better={"loss": "min"}, formats={"loss_sd": 4, "excess_pct": 1},
                 labels={"loss": "Mean MSE", "loss_sd": "Draw SD", "n": "Completed",
                         "failures": "Failed", "reference_period": "Measurement",
                         "excess_pct": "Excess (%)"},
                 spanners={"Prediction": ["loss", "loss_sd", "excess_pct"],
                           "Accounting": ["n", "failures", "reference_period"]},
                 notes=("Five paired draws per task; SD is across independent draws. Best MSE is bold.",
                        "Historical comparators: original small-tier shootout, identical input and partition hashes. "
                        "Search budgets differ; these are pipeline comparisons. Test-selected best is a retrospective oracle.",
                        "Before and after rerun only autospec. None of these known generators is independent confirmation."))


def main():
    before, after = read("before.json"), read("after.json")
    reference_path = ROOT / after["protocol"]["comparator_reference"]
    digest = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    for document in (before, after):
        if document["protocol"]["comparator_sha256"] != digest:
            raise ValueError("historical comparator artifact changed")
        if document["protocol"]["scorecard_sha256"] != hashlib.sha256(
                (ROOT / "docs/scorecard/scorecard.md").read_bytes()).hexdigest():
            raise ValueError("scorecard changed; an explicit reassessment is required")
        if document["protocol"]["document_sha256"] != hashlib.sha256(
                (DIRECTORY / "PROTOCOL.md").read_bytes()).hexdigest():
            raise ValueError("frozen protocol content changed")
    summary = assess(before, after, json.loads(reference_path.read_text()))
    (DIRECTORY / "summary.json").write_text(json.dumps(summary, indent=1, allow_nan=False) + "\n")
    (DIRECTORY / "headline.html").write_text(headline_table(summary).as_raw_html())
    card = json.loads((ROOT / "docs/scorecard/report-template.json").read_text())
    card.update(library="mboostrs", model_revision=after["commit"],
                benchmark_protocol_id="pair-form-stopping/1", mode="automated pipeline",
                aggregate=None)
    for d in card["dimensions"]:
        if d["id"] == "D10":
            d.update(status="scored", score=5, evidence=["before.json", "after.json", "PROTOCOL.md"],
                     justification="Artifacts retain code/environment, loaded binary, input/partition and prediction hashes; "
                     "the recorded runner reproduces the procedure and the reporter verifies paired identities. "
                     "This is evidence for this instrument, not a general end-user workflow score.")
        elif d["id"] == "D11":
            targets = [h for h in summary["headline"] if h["method"] == "mboostrs after"]
            baselines = {h["task"]: h for h in summary["headline"] if h["method"] == "linear"}
            useful = not summary["failures"] and all(
                not baselines[t["task"]]["failures"] and t["loss"] < baselines[t["task"]]["loss"]
                for t in targets)
            if useful:
                d.update(status="scored", score=3, evidence=["summary.json", "PROTOCOL.md"],
                         justification="All three completed outer task batteries improve over the prespecified Ridge "
                         "baseline; per-draw changes, spread and failures are reported. Known Gaussian generators, "
                         "historical comparators and unequal tuning budgets preclude level 4 or an independent confirmation claim.")
    card["model_identity"] = {"source_commit": after["commit"],
                              "working_tree": after["working_tree"],
                              "job_package_hash": after["results"][0]["_identity"]["components"]["mboostrs"]["hash"]}
    card["assessment_identity"] = {
        "reporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "artifact_sha256": {name: hashlib.sha256((DIRECTORY / name).read_bytes()).hexdigest()
                            for name in ("before.json", "after.json", "summary.json")},
        "comparator_sha256": digest,
    }
    (DIRECTORY / "scorecard.json").write_text(json.dumps(card, indent=1) + "\n")
    print(json.dumps({"gaps": summary["gaps"], "failures": len(summary["failures"])}))


if __name__ == "__main__":
    main()
