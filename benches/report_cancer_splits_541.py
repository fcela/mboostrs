"""Descriptive paired summaries for the prespecified #541 repeated splits."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DIRECTORY = Path("docs/book/benchmarks/competitive_refresh")
METRICS = ("loss", "brier", "auc", "calibrated_loss")


def spread(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("empty or non-finite measurements")
    return {"mean": float(values.mean()), "sd": float(values.std(ddof=1)) if values.size > 1 else None,
            "min": float(values.min()), "median": float(np.median(values)),
            "max": float(values.max())}


def assess(document):
    protocol = document["protocol"]
    expected = set(protocol["seeds"])
    methods = protocol["methods"]
    if len(expected) != len(protocol["seeds"]):
        raise ValueError("duplicate planned seed")
    observed, good, failures = set(), [], []
    observations = defaultdict(lambda: defaultdict(list))
    for row in document["results"]:
        seed = row["seed"] if "seed" in row else row["job"][0]
        if seed not in expected or seed in observed:
            raise ValueError("unexpected or duplicate result")
        observed.add(seed)
        if row["_status"] != "success":
            failures.append(row)
            continue
        parts = row["indices"]
        flat = [i for part in parts for i in part]
        if len(parts) != 4 or len(flat) != len(set(flat)) or not all(parts):
            raise ValueError("partitions must be nonempty and disjoint")
        if row["autospec"]["fits"] > protocol["autospec"]["budget"]:
            raise ValueError("fit budget exceeded")
        y = np.asarray(row["test_response"], dtype=float)
        if y.shape != (len(parts[3]),) or not np.isin(y, [0, 1]).all():
            raise ValueError("invalid test response")
        for method in methods:
            result = row["results"][method]
            p = np.asarray(result["predictions"], dtype=float)
            losses = np.asarray(result["row_loss"], dtype=float)
            if p.shape != y.shape or losses.shape != y.shape or not np.isfinite(p).all():
                raise ValueError("misaligned predictions or losses")
            if np.any((p < 0) | (p > 1)):
                raise ValueError("invalid probabilities")
            clipped = np.clip(p, 1e-12, 1-1e-12)
            computed = -y*np.log(clipped)-(1-y)*np.log1p(-clipped)
            if not np.allclose(computed, losses, rtol=0, atol=1e-12):
                raise ValueError("stored losses differ from predictions")
            scores = result["scores"]
            if not all(np.isfinite(scores[m]) for m in METRICS):
                raise ValueError("non-finite score")
            if not np.isclose(losses.mean(), scores["loss"], rtol=0, atol=1e-12):
                raise ValueError("mean loss differs from score")
            for index, value in zip(parts[3], losses, strict=True):
                observations[index][method].append(float(value))
        good.append(row)
    if observed != expected:
        raise ValueError("missing planned result; retain explicit failure rows")
    complete = len(good) == len(expected)
    method_summary = {method: {metric: spread([r["results"][method]["scores"][metric]
                                               for r in good]) for metric in METRICS}
                      for method in methods} if good else {}
    comparisons = []
    for reference in methods[1:]:
        deltas = [r["results"]["autospec"]["scores"]["loss"] -
                  r["results"][reference]["scores"]["loss"] for r in good]
        relative = (100*(method_summary["autospec"]["loss"]["mean"] /
                         method_summary[reference]["loss"]["mean"]-1)) if good else None
        comparisons.append({"reference": reference, "complete": complete,
                            "paired_difference": spread(deltas) if deltas else None,
                            "autospec_wins": sum(d < 0 for d in deltas), "n": len(good),
                            "excess_pct": relative,
                            "within_5pct": complete and relative <= protocol["target_excess_pct"]})
    per_observation = []
    for index, values in sorted(observations.items()):
        means = {method: float(np.mean(losses)) for method, losses in values.items()}
        per_observation.append({"row": index, "test_appearances": len(values["autospec"]),
                                "mean_loss": means,
                                "difference_from_tuned": means["autospec"]-means["linear_tuned"]})
    draws = [{"seed": r["seed"], "autospec": r["autospec"], "prevalence": r["prevalence"],
              "results": {method: {"scores": {m: r["results"][method]["scores"][m] for m in METRICS},
                                    "top_five_loss_share": r["results"][method]["top_five_loss_share"]}
                          for method in methods}} for r in good]
    diagnostics = {}
    if good:
        diagnostics = {
            "selected_mstop": spread([r["autospec"]["mstop"] for r in good]),
            "search_fits": spread([r["autospec"]["fits"] for r in good]),
            "top_five_loss_share": {
                method: spread([r["results"][method]["top_five_loss_share"] for r in good])
                for method in methods},
            "unique_test_observations": len(per_observation),
            "test_appearances_per_observed_row": spread([
                r["test_appearances"] for r in per_observation]),
        }
        selected = [r for r in good if r["autospec"].get("selected_validation_loss") is not None]
        if selected:
            diagnostics["test_minus_selected_validation_loss"] = spread([
                r["results"]["autospec"]["scores"]["loss"] -
                r["autospec"]["selected_validation_loss"] for r in selected])
    return {"protocol": protocol, "source_revision": document.get("commit"),
            "build_info": document.get("build_info"), "planned": len(expected),
            "completed": len(good), "failures": failures, "method_summary": method_summary,
            "comparisons": comparisons, "draws": draws,
            "diagnostics": diagnostics,
            "observations_by_excess_loss": sorted(per_observation,
                key=lambda r: r["difference_from_tuned"], reverse=True),
            "interpretation": "Overlapping splits of a fixed panel; descriptive spread, no IID standard errors. "
                              "Additional splits do not replace the original acceptance battery."}


def main():
    document = json.loads((DIRECTORY / "cancer_splits_541.json").read_text())
    planned = json.loads((DIRECTORY / "cancer_splits_541.planned.json").read_text())
    if document["protocol"] != planned["protocol"]:
        raise ValueError("executed protocol differs from frozen plan")
    summary = assess(document)
    (DIRECTORY / "cancer_splits_541_summary.json").write_text(
        json.dumps(summary, indent=1, allow_nan=False) + "\n")
    from docs.book.cases.bench_tables import table
    import polars as pl

    records = []
    for method, scores in summary["method_summary"].items():
        records.append({"method": method, "completed": summary["completed"],
                        "log_loss": scores["loss"]["mean"], "split_sd": scores["loss"]["sd"],
                        "brier": scores["brier"]["mean"], "auc": scores["auc"]["mean"],
                        "calibrated_log_loss": scores["calibrated_loss"]["mean"]})
    if records:
        gt = table(pl.DataFrame(records), title="Cancer: single-holdout autospec and logistic references",
                   subtitle=f"{summary['completed']} of {summary['planned']} overlapping random splits; "
                            "log loss in nats per observation and Brier score: lower is better; AUC: higher is better",
                   spanners={"Raw test scores": ["log_loss", "split_sd", "brier", "auc"],
                             "Separate calibration": ["calibrated_log_loss"]},
                   better={"log_loss": "min", "brier": "min", "auc": "max", "calibrated_log_loss": "min"},
                   formats={"log_loss": 4, "split_sd": 4, "brier": 4, "auc": 4,
                            "calibrated_log_loss": 4, "completed": 0},
                   notes=("Split SD is descriptive standard deviation across overlapping splits, not a standard error.",
                          "linear_small uses the first frozen penalty; linear_tuned selects among four on validation rows.",
                          "Calibration is logistic calibration on separate calibration rows; primary loss is uncalibrated.",
                          "All scores use the same partitions within each split. Incomplete evidence cannot pass the target.",
                          "Source: cancer_splits_541.json; runner: benches/bench_cancer_splits_541.py."))
        (DIRECTORY / "cancer_splits_541_headline.html").write_text("\n".join(line.rstrip() for line in gt.as_raw_html().splitlines()) + "\n")
    print(json.dumps({key: summary[key] for key in ("planned", "completed", "comparisons")}, indent=2))


if __name__ == "__main__":
    main()
