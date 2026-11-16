"""Score one analyst run against the planted truth. Prints the measurable scorecard metrics as JSON."""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

STUDY = Path(sys.argv[1])
RUN = Path(sys.argv[2])
TRUTH_DIR = STUDY / "truth" if (STUDY / "truth").is_dir() else STUDY  # released layout keeps truth at the root
TRUTH = json.loads((TRUTH_DIR / "truth.json").read_text())
PLANTED = ["x1", "x2", "x4", "x5"]
PAIR = "x6:x7"
NULL_MEAN = ["x3"] + [f"x{j}" for j in range(9, 21)]
NO_CONTRAST = {"x6", "x7"}  # interaction-only inputs: no main-effect contrast truth


def rank(a):
    order = np.argsort(a)
    r = np.empty(len(a))
    r[order] = np.arange(len(a))
    return r


def spearman(a, b):
    return float(np.corrcoef(rank(np.asarray(a)), rank(np.asarray(b)))[0, 1])


def d1(findings):
    eff = findings["effects"]
    selected = {k for k, v in eff.items() if v["selected"]}
    pair_sel = findings.get("pairs", {}).get(PAIR, {}).get("selected", False)
    tp = len([k for k in PLANTED if k in selected]) + int(pair_sel)
    fp = len([k for k in selected if k not in PLANTED])
    n_sel = len(selected) + int(pair_sel)
    return {"tpr": tp / 5, "fdr": fp / n_sel if n_sel else 0.0,
            "selected": sorted(selected) + ([PAIR] if pair_sel else []),
            "false_positives": sorted(k for k in selected if k not in PLANTED)}


def d4(findings, ds):
    contr = TRUTH["datasets"][ds]["contrasts"]
    eff = findings["effects"]
    cov, widths = [], []
    for k, c in contr.items():
        if k in NO_CONTRAST or k not in eff:
            continue
        e = eff[k]
        cov.append(e["lower90"] <= c <= e["upper90"])
        if k in PLANTED:
            widths.append(e["upper90"] - e["lower90"])
    return {"contrast_coverage90": float(np.mean(cov)), "n_inputs_scored": len(cov),
            "median_width_planted": float(np.median(widths)) if widths else None}


def d9(run, ds="main"):
    pred = pl.read_csv(run / f"predictions_{ds}.csv")
    truth = pl.read_csv(TRUTH_DIR / f"{ds}_test_truth.csv")
    j = truth.join(pred, on="row_id", how="left")
    assert j.null_count().sum_horizontal().item() == 0, "missing predictions"
    y = j["y"].to_numpy()
    lo, hi = j["lower90"].to_numpy(), j["upper90"].to_numpy()
    half = (hi - lo) / 2
    return {"predictive_coverage90": float(np.mean((y >= lo) & (y <= hi))),
            "spearman_halfwidth_vs_sigma": spearman(half, j["sigma"].to_numpy())}


def d5(run):
    log = pl.read_csv(run / "fit_log.csv").filter(pl.col("dataset") == "main")
    s = log["seconds"].to_numpy()
    return {"n_fits_main": int(len(s)), "median_s": float(np.median(s)),
            "p90_s": float(np.quantile(s, 0.9)), "libraries": sorted(set(log["library"].to_list()))}


if __name__ == "__main__":
    out = {"run": RUN.name}
    f = {ds: json.loads((RUN / f"findings_{ds}.json").read_text()) for ds in ["main", "small", "messy"]}
    out["D1_main"] = d1(f["main"])
    out["D4_main"] = d4(f["main"], "main")
    out["D6_small"] = {**{"D1": d1(f["small"])}, **{"D4": d4(f["small"], "small")}}
    out["D6_messy"] = {**{"D1": d1(f["messy"])}, **{"D4": d4(f["messy"], "messy")}}
    out["D9_main"] = d9(RUN)
    out["D5_main"] = d5(RUN)
    print(json.dumps(out, indent=2))
