"""Per-family frozen non-regression gate for the factor dictionary repair."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from great_tables import GT

from benches.bench_factor_reference_v2 import CASES, DIRECTORY


def assess(before, after):
    expected = {(case, seed) for case in CASES for seed in range(6)}
    indexes = []
    for document in (before, after):
        if ({tuple(s) for s in document["protocol"]["specs"]} != expected
                or len(document["protocol"]["specs"]) != len(expected)):
            raise ValueError("protocol omits a required control family/draw")
        rows = {}
        for row in document["results"]:
            key = (row["case"], row["seed"]) if "case" in row else tuple(row["job"])
            if key not in expected or key in rows:
                raise ValueError("unexpected or duplicate control")
            if row["_status"] == "success" and (row.get("mse") is None or not np.isfinite(row["mse"])):
                raise ValueError("successful control has non-finite MSE")
            rows[key] = row
        if set(rows) != expected:
            raise ValueError("missing planned control")
        indexes.append(rows)
    if before["protocol"]["document_sha256"] != after["protocol"]["document_sha256"]:
        raise ValueError("control protocol changed")
    paired, failures = [], []
    for key in sorted(expected):
        b, a = (rows[key] for rows in indexes)
        if b["_status"] != "success" or a["_status"] != "success":
            failures.append({"case": key[0], "seed": key[1], "before": b["_status"], "after": a["_status"]})
            continue
        if b["input_digest"] != a["input_digest"]:
            raise ValueError("paired control rows differ")
        paired.append({"case": key[0], "seed": key[1], "before": b["mse"], "after": a["mse"],
                       "change": a["mse"] - b["mse"],
                       "prediction_unchanged": b["prediction_digest"] == a["prediction_digest"],
                       "before_interactions": len(b["interactions"]), "after_interactions": len(a["interactions"]),
                       "reference_curve_range": a["reference_curve_range"]})
    families = []
    for case in CASES:
        rows = [r for r in paired if r["case"] == case]
        complete = len(rows) == 6
        before_mean = float(np.mean([r["before"] for r in rows])) if complete else None
        after_mean = float(np.mean([r["after"] for r in rows])) if complete else None
        ratio = after_mean / before_mean if complete else None
        families.append({"case": case, "completed": len(rows), "before_mse": before_mean, "after_mse": after_mean,
                         "ratio": ratio, "change": float(np.mean([r["change"] for r in rows])) if complete else None,
                         "change_se": float(np.std([r["change"] for r in rows], ddof=1) / np.sqrt(6)) if complete else None,
                         "passes_1_01": complete and ratio <= 1.01})
    null_rows = [r for r in paired if r["case"] == "null"]
    null_ok = len(null_rows) == 6 and sum(r["after_interactions"] for r in null_rows) <= sum(r["before_interactions"] for r in null_rows)
    numeric = [r for r in paired if r["case"] == "numeric"]
    numeric_ok = len(numeric) == 6 and all(r["prediction_unchanged"] for r in numeric)
    opposed_ok = all(next(f for f in families if f["case"] == case)["ratio"] is not None
                     and next(f for f in families if f["case"] == case)["ratio"] < 1
                     and all(r["reference_curve_range"] is not None and r["reference_curve_range"] > 1e-8 for r in paired if r["case"] == case)
                     for case in ("opposed", "relabelled"))
    return {"families": families, "paired": paired, "failures": failures,
            "numeric_predictions_identical": numeric_ok, "null_adoptions_not_increased": null_ok,
            "opposed_curves_recovered": opposed_ok,
            "gate_passed": not failures and all(f["passes_1_01"] for f in families) and null_ok and numeric_ok and opposed_ok}


def control_table(summary):
    return GT(pd.DataFrame(summary["families"]), rowname_col="case").tab_header(
        title="Fresh controls for the factor-reference dictionary",
        subtitle="Test MSE, after/before mean ratio and paired change SE; lower is better; limit 1.01"
    ).tab_spanner("Prediction", ["before_mse", "after_mse", "ratio"]).tab_spanner("Paired change", ["change", "change_se"]).fmt_number(
        columns=["before_mse", "after_mse", "ratio", "change", "change_se"], decimals=4).tab_source_note(
        "Six independent new draws per family. SE is sample SD of the paired after-minus-before MSE divided by sqrt(6). Every family must pass; improvements do not compensate regressions.").tab_source_note(
        "This is a parent non-regression gate, not a best-competitor target or calibrated FDR claim. Null-adoption and exact numeric-indicator prediction checks are separate gates.")


def main():
    before, after = (json.loads((DIRECTORY / f"{phase}.json").read_text()) for phase in ("before", "after"))
    digest = hashlib.sha256((DIRECTORY / "PROTOCOL.md").read_bytes()).hexdigest()
    if before["protocol"]["document_sha256"] != digest:
        raise ValueError("frozen protocol changed")
    summary = assess(before, after)
    (DIRECTORY / "summary.json").write_text(json.dumps(summary, indent=1, allow_nan=False) + "\n")
    (DIRECTORY / "controls.html").write_text(control_table(summary).as_raw_html())
    print(json.dumps({k: v for k, v in summary.items() if k not in ("paired",)}, indent=1))


if __name__ == "__main__":
    main()
