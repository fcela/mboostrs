"""Summarize the frozen stopping-reserve experiment without selecting a rule."""
from __future__ import annotations

import json
import gzip
import hashlib
from pathlib import Path

import numpy as np

DIRECTORY = Path("docs/book/benchmarks/competitive_refresh")


def load_evidence(directory=DIRECTORY):
    """Read retained compressed evidence, checking both archive and JSON bytes."""
    manifest = json.loads((directory / "stopping_reserve_archive.json").read_text())
    compressed = (directory / manifest["archive"]).read_bytes()
    if hashlib.sha256(compressed).hexdigest() != manifest["archive_sha256"]:
        raise ValueError("archive hash mismatch")
    raw = gzip.decompress(compressed)
    if hashlib.sha256(raw).hexdigest() != manifest["uncompressed_sha256"]:
        raise ValueError("uncompressed evidence hash mismatch")
    return json.loads(raw)


def assess(document):
    protocol = document["protocol"]
    expected = {(task, seed) for task in protocol["tasks"] for seed in protocol["seeds"]}
    observed = set()
    draws, failures = [], []
    for row in document["results"]:
        key = tuple(row.get("job", ())) if "task" not in row else (row["task"], row["seed"])
        if key not in expected or key in observed:
            raise ValueError("unexpected or duplicate draw")
        observed.add(key)
        if row["_status"] != "success":
            failures.append(row)
            continue
        methods = row["results"]
        losses = {name: np.asarray(value["row_loss"], dtype=float)
                  for name, value in methods.items()}
        if any(a.ndim != 1 or a.size == 0 or not np.isfinite(a).all()
               for a in losses.values()) or len({a.shape for a in losses.values()}) != 1:
            raise ValueError("invalid paired row losses")
        for name, values in losses.items():
            if not np.isclose(values.mean(), methods[name]["scores"]["loss"], atol=1e-12, rtol=0):
                raise ValueError("row losses disagree with reported score")
        delta = losses["reserve"] - losses["baseline"]
        if not np.isfinite(delta).all() or delta.size == 0:
            raise ValueError("invalid paired row losses")
        draws.append({"task": key[0], "seed": key[1], "reserve_stop": row["reserve_stop"],
                      "baseline_stop": row["baseline_config"]["mstop"],
                      "paired_loss_difference": float(delta.mean()),
                      "scores": {k: {m: v["scores"][m] for m in
                                      ("loss", "brier", "auc", "calibrated_loss")}
                                 for k, v in methods.items()}, "prevalence": row["prevalence"]})
    if observed != expected:
        raise ValueError("missing draw records")
    means = []
    for task in protocol["tasks"]:
        rows = [r for r in draws if r["task"] == task]
        complete = len(rows) == len(protocol["seeds"])
        values = {method: float(np.mean([r["scores"][method]["loss"] for r in rows]))
                  for method in ("baseline", "reserve", "linear")} if complete else None
        means.append({"task": task, "complete": complete, "mean_loss": values,
                      "paired_difference": values["reserve"]-values["baseline"] if complete else None,
                      "excess_over_linear_pct": 100*(values["reserve"]/values["linear"]-1)
                      if complete else None})
    passed = not failures and all(r["paired_difference"] <= 0 for r in means)
    passed = passed and next(r for r in means if r["task"] == "cancer")["excess_over_linear_pct"] <= 5
    return {"protocol": protocol, "draws": draws, "means": means, "failures": failures,
            "clears_experimental_gate": passed,
            "interpretation": "Descriptive paired splits; cancer rows overlap. No default change."}


def main():
    raw = DIRECTORY / "stopping_reserve.json"
    summary = assess(json.loads(raw.read_text()) if raw.exists() else load_evidence())
    (DIRECTORY / "stopping_reserve_summary.json").write_text(
        json.dumps(summary, indent=1, allow_nan=False) + "\n")
    import polars as pl
    from docs.book.cases.bench_tables import table

    records = []
    for task in summary["protocol"]["tasks"]:
        draws = [r for r in summary["draws"] if r["task"] == task]
        for method in ("baseline", "reserve", "linear"):
            if not draws:
                continue
            records.append({"task": task, "method": method, "completed": len(draws),
                            **{metric: float(np.mean([r["scores"][method][metric] for r in draws]))
                               for metric in ("loss", "brier", "auc", "calibrated_loss")}})
    gt = table(pl.DataFrame(records), title="Stopping reserve: cancer and three binary controls",
               subtitle="Three draws per task; log loss in nats per observation and Brier score: "
                        "lower is better; AUC: higher is better",
               group="task",
               spanners={"Raw test scores": ["loss", "brier", "auc"],
                         "Separate calibration": ["calibrated_loss"]},
               better={"loss": "min", "brier": "min", "auc": "max", "calibrated_loss": "min"},
               formats={"loss": 4, "brier": 4, "auc": 4, "calibrated_loss": 4, "completed": 0},
               notes=("Each row reports a mean across three paired draws; cancer splits overlap.",
                      "Per-draw values and descriptive limits are retained in stopping_reserve_summary.json.",
                      "baseline: original autospec; reserve: separate stopping reserve; linear: tuned logistic reference.",
                      "Calibrated loss uses separate calibration rows; the primary metric is raw loss.",
                      "Source: stopping_reserve.json.gz; runner: benches/cancer_stopping_reserve.py."))
    (DIRECTORY / "stopping_reserve_headline.html").write_text("\n".join(line.rstrip() for line in gt.as_raw_html().splitlines()) + "\n")
    print(json.dumps({"means": summary["means"], "passes": summary["clears_experimental_gate"]}, indent=2))


if __name__ == "__main__":
    main()
