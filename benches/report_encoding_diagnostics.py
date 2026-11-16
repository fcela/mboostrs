"""Verify matched path diagnostics and observed discovery-grid contracts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from great_tables import GT

from benches.bench_encoding_tracks import DIRECTORY, ENCODINGS
from benches.report_encoding_tracks import mean_se


def indexed(document, expected):
    rows = {}
    for row in document["results"]:
        if "task" in row:
            key = (row["task"], row["seed"], row.get("method", "mboostrs_known"))
        else:
            job = row["job"]
            key = (*job, "mboostrs_known") if len(job) == 2 else tuple(job[-3:])
        if key not in expected or key in rows:
            raise ValueError("unexpected or duplicate diagnostic draw")
        rows[key] = row
    if set(rows) != expected:
        raise ValueError("missing diagnostic draw")
    return rows


def conditioning(document, baseline, primary_summary):
    expected = {(task, seed, "mboostrs_known") for task in ("indicator", "shifted") for seed in range(8)}
    rows = indexed(document, expected)
    reference = {(r["task"], r["seed"], r["method"]): r for r in baseline["results"] if "task" in r}
    peers = {g["task"]: g["best_competitor"] for g in primary_summary["gaps"]}
    failures, groups, replays = [], [], []
    for key, row in rows.items():
        if row["_status"] != "success":
            failures.append({"key": key, "status": row["_status"]})
            continue
        risk = np.asarray(row["risk"])
        if len(risk) != row["largest"] + 1 or not np.isfinite(risk).all():
            raise ValueError("incomplete or non-finite diagnostic risk path")
        if len(row["choices"]) != 2 or {c["pool"] for c in row["choices"]} != {"prefix1600", "longer_search"}:
            raise ValueError("missing diagnostic candidate pool")
        for choice in row["choices"]:
            limit = min(1600, row["largest"]) if choice["pool"] == "prefix1600" else row["largest"]
            stop = 1 + int(np.argmin(risk[1:limit + 1]))
            if choice["mstop"] != stop or choice["validation_mse"] != risk[stop]:
                raise ValueError("diagnostic choice differs from its positive CV minimum")
        prefix = next(c for c in row["choices"] if c["pool"] == "prefix1600")
        original = reference[key]
        if (row["input_digest"] != original["input_digest"] or row["latent_digest"] != original["latent_digest"]
                or any(prefix[f] != original[f] for f in ("mse", "prediction_digest"))):
            raise ValueError("diagnostic prefix does not exactly replay the original known fit")
        replays.append(key)
    for task in ("indicator", "shifted"):
        good = [r for (t, s, m), r in sorted(rows.items()) if t == task and r["_status"] == "success"]
        for pool in ("prefix1600", "longer_search"):
            choices = [next(c for c in r["choices"] if c["pool"] == pool) for r in good]
            complete = len(good) == 8
            mse, se = mean_se([c["mse"] for c in choices]) if complete else (None, None)
            gaps = [100 * (c["mse"] / reference[task, r["seed"], peers[task]]["mse"] - 1)
                    for r, c in zip(good, choices, strict=True)]
            gap, gap_se = mean_se(gaps) if complete else (None, None)
            groups.append({"task": task, "pool": pool, "completed": len(good), "mean_mse": mse,
                           "mse_se": se, "gap_pct": gap, "gap_se_pct": gap_se,
                           "mstop_by_seed": [c["mstop"] for c in choices]})
    return {"groups": groups, "exact_prefix_replays": replays, "failures": failures}


def cap_contract(before_high, after_high, after_low, primary_after):
    high_keys = {("shifted", s, "mboostrs_discover") for s in range(8)}
    low_keys = {(t, s, m) for t in ENCODINGS for s in range(8) for m in ("mboostrs_known", "mboostrs_discover")}
    before, after, low = indexed(before_high, high_keys), indexed(after_high, high_keys), indexed(after_low, low_keys)
    reference = {(r["task"], r["seed"], r["method"]): r for r in primary_after["results"] if "task" in r}
    failures, replays, high_rows = [], [], []
    for key, row in low.items():
        if row["_status"] != "success":
            failures.append({"key": key, "phase": "low", "status": row["_status"]})
        elif any(row[f] != reference[key][f] for f in ("input_digest", "latent_digest", "prediction_digest", "mse")):
            raise ValueError("cap repair changed an original 1600-limit prediction")
        else:
            replays.append(key)
    for key in sorted(high_keys):
        b, a = before[key], after[key]
        if b["_status"] != "success" or a["_status"] != "success":
            failures.append({"key": key, "phase": "high", "before": b["_status"], "after": a["_status"]})
            continue
        if not b["tuning_stages"] or not a["tuning_stages"]:
            raise ValueError("high-cap contract has no actual observed tuner returns")
        if b["input_digest"] != a["input_digest"] or b["formula"] != a["formula"]:
            raise ValueError("high-cap data or selected dictionary differs")
        high_rows.append({"seed": key[1], "before_stop": b["mstop"], "after_stop": a["mstop"],
                          "before_mse": b["mse"], "after_mse": a["mse"],
                          "before_mismatch": any(s["declared_grid"] != s["actual_largest"] for s in b["tuning_stages"]),
                          "after_mismatch": any(s["declared_grid"] != s["actual_largest"] for s in a["tuning_stages"]),
                          "after_maximum_forwarded": all(s["public_maximum"] == 25600 for s in a["tuning_stages"]),
                          "before_largest": max(s["actual_largest"] for s in b["tuning_stages"]),
                          "after_largest": max(s["actual_largest"] for s in a["tuning_stages"])})
    return {"low_cap_exact_replays": replays, "high_rows": high_rows, "failures": failures,
            "passed": not failures and len(replays) == 80 and len(high_rows) == 8
                and all(not r["after_mismatch"] and r["after_maximum_forwarded"] for r in high_rows)}


def main():
    def read(path):
        return json.loads(path.read_text())
    baseline, after = read(DIRECTORY / "baseline.json"), read(DIRECTORY / "after.json")
    cdir = DIRECTORY / "conditioning"
    cdoc = read(cdir / "results.json")
    if cdoc["protocol"]["document_sha256"] != hashlib.sha256((cdir / "PROTOCOL.md").read_bytes()).hexdigest():
        raise ValueError("conditioning protocol changed")
    c = conditioning(cdoc, baseline, read(DIRECTORY / "baseline/summary.json"))
    c["reporter_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (cdir / "summary.json").write_text(json.dumps(c, indent=1, allow_nan=False) + "\n")
    capdir = DIRECTORY / "cap_contract"
    docs = [read(capdir / name) for name in ("before_high.json", "after_high.json", "after_low.json")]
    expected = hashlib.sha256((capdir / "PROTOCOL.md").read_bytes()).hexdigest()
    if any(d["protocol"]["document_sha256"] != expected for d in docs):
        raise ValueError("cap contract protocol changed")
    cap = cap_contract(*docs, after)
    repeats = 0
    for name, current in zip(("before_high.json", "after_high.json", "after_low.json"), docs, strict=True):
        old = read(capdir / "initial_observer" / name)
        for b, a in zip(old["results"], current["results"], strict=True):
            if any(a[f] != b[f] for f in ("task", "seed", "method", "input_digest", "prediction_digest", "mse")):
                raise ValueError("observer correction changed a captured prediction")
            repeats += 1
    cap["initial_capture_exact_replays"] = repeats
    cap["reporter_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (capdir / "summary.json").write_text(json.dumps(cap, indent=1, allow_nan=False) + "\n")
    frame = pd.DataFrame(c["groups"])
    frame["mstop_by_seed"] = frame["mstop_by_seed"].map(lambda values: ", ".join(map(str, values)))
    html = GT(frame, rowname_col="pool", groupname_col="task").tab_header(
        title="Known-specification stopping-range diagnostic",
        subtitle="Mean outer-test MSE and paired gap percent; lower is better; longer search is diagnostic"
    ).fmt_number(columns=["mean_mse", "mse_se"], decimals=4).fmt_number(columns=["gap_pct", "gap_se_pct"], decimals=2).tab_source_note(
        "Eight paired known draws per encoding; all sixteen original prefixes replay exactly. Reference is retrospective CatBoost with a different search budget; no default is selected from these outer scores.")
    (cdir / "comparison.html").write_text(html.as_raw_html())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    with plt.rc_context({"font.size": 9}):
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        for row in cdoc["results"]:
            if row["_status"] != "success":
                continue
            indicator = row["task"] == "indicator"
            risk = np.asarray(row["risk"])
            ax.plot(np.arange(1, len(risk)), risk[1:],
                    color="#22577a" if indicator else "#bb4d00",
                    linestyle="-" if indicator else "--", linewidth=.8, alpha=.55)
        ax.axvline(1600, color="#444444", linestyle=":", linewidth=1.2)
        ax.set(xscale="log", xlabel="Boosting updates (log scale)",
               ylabel="Cross-validated MSE (squared response units)",
               title="Known specification: paired CV paths")
        ax.legend(handles=[Line2D([0], [0], color="#22577a", label="0/1 indicator: eight draws"),
                           Line2D([0], [0], color="#bb4d00", linestyle="--", label="2/3 shifted: eight draws"),
                           Line2D([0], [0], color="#444444", linestyle=":", label="Original 1,600-update limit")],
                  frameon=False)
        fig.tight_layout()
        fig.savefig(cdir / "cv_paths.svg")
        fig.savefig(cdir / "cv_paths.png", dpi=180)
        plt.close(fig)
    print(json.dumps({"conditioning": c["groups"], "cap_passed": cap["passed"],
                      "low_replays": len(cap["low_cap_exact_replays"]), "observer_replays": repeats}, indent=1))


if __name__ == "__main__":
    main()
