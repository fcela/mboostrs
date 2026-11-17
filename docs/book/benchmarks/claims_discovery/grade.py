"""Score analyst pipeline outputs against the planted truth. Pre-registered metrics (PROTOCOL.md)."""

import json
import re
from pathlib import Path

import numpy as np
import polars as pl

COVARIATES = [
    "longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms",
    "population", "households", "median_income", "median_house_value",
    "ave_rooms", "ave_bedrooms", "ave_occup", "roof_class",
]
MEAN_ACTIVE = {"median_income", "housing_median_age", "ave_occup", "roof_class", "latitude", "longitude"}
PAIRS = {frozenset(("latitude", "longitude")), frozenset(("housing_median_age", "median_income"))}
SCALE = {"households"}
PROXIES = {"population", "total_rooms", "total_bedrooms"}
SHAPES = {
    "median_income": {"decreasing"},
    "housing_median_age": {"increasing", "threshold_increase"},
    "ave_occup": {"increasing"},
    "roof_class": {"categorical"},
}
QUERIES = [f"Q{i}" for i in range(1, 9)]


def canon(name) -> set[str]:
    """Map a claimed variable label to the covariate names it refers to."""
    s = str(name).strip().lower()
    s = re.sub(r"^(bbs|bols|bspatial|s|te|f|l|bs|cr|c)\((.*)\)$", r"\2", s)
    if s in COVARIATES:
        return {s}
    hits = {c for c in COVARIATES if re.search(rf"(?<![a-z]){re.escape(c)}(?![a-z])", s)}
    if not hits and ("lat" in s or "lon" in s or "location" in s or "spatial" in s):
        hits = {c for c, k in (("latitude", "lat"), ("longitude", "lon")) if k in s}
        if not hits:
            hits = {"latitude", "longitude"}
    return hits


def claims(f: dict) -> dict:
    mean = set()
    shapes = {}
    for m in f.get("mean_effects") or []:
        vs = canon(m.get("variable", ""))
        mean |= vs
        for v in vs:
            shapes[v] = str(m.get("shape", "")).lower()
    pairs = set()
    for it in f.get("interactions") or []:
        vs = it.get("variables") or []
        names = [canon(v) for v in vs]
        flat = set().union(*names) if names else set()
        if len(vs) == 1 and len(flat) == 2:
            pairs.add(frozenset(flat))
        elif len(names) >= 2:
            for a in names[0]:
                for b in names[1]:
                    if a != b:
                        pairs.add(frozenset((a, b)))
    scale = set()
    for v in f.get("variance_drivers") or []:
        scale |= canon(v.get("variable", ""))
    return {"mean": mean, "shapes": shapes, "pairs": pairs, "scale": scale}


def rates(claimed: set, truth: set) -> tuple[float, float]:
    tp = len(claimed & truth)
    tpr = tp / len(truth) if truth else float("nan")
    fdr = (len(claimed) - tp) / len(claimed) if claimed else 0.0
    return tpr, fdr


def score_discovery(f: dict | None, kind: str) -> dict:
    if f is None:
        return {"ok": False, "mean_tpr": 0.0, "mean_fdr": 0.0, "pair_tpr": 0.0, "pair_fdr": 0.0,
                "scale_tpr": 0.0, "scale_fdr": 0.0, "n_false": 0, "shape_correct": float("nan"),
                "proxy_scale": 0, "n_mean": 0, "n_pairs": 0, "n_scale": 0, "decoy_value": False}
    c = claims(f)
    if kind == "null":
        n_false = len(c["mean"]) + len(c["pairs"]) + len(c["scale"])
        return {"ok": True, "n_false": n_false, "n_mean": len(c["mean"]), "n_pairs": len(c["pairs"]),
                "n_scale": len(c["scale"]), "decoy_value": "median_house_value" in c["mean"]}
    mt, mf = rates(c["mean"], MEAN_ACTIVE)
    pt, pf = rates(c["pairs"], PAIRS)
    st, sf = rates(c["scale"], SCALE)
    graded = [v for v in SHAPES if v in c["shapes"]]
    shape_ok = np.mean([c["shapes"][v] in SHAPES[v] for v in graded]) if graded else float("nan")
    return {
        "ok": True, "mean_tpr": mt, "mean_fdr": mf, "pair_tpr": pt, "pair_fdr": pf,
        "scale_tpr": st, "scale_fdr": sf, "shape_correct": shape_ok,
        "proxy_scale": len(c["scale"] & PROXIES), "n_mean": len(c["mean"]), "n_pairs": len(c["pairs"]),
        "n_scale": len(c["scale"]), "n_false": (len(c["mean"] - MEAN_ACTIVE) + len(c["pairs"] - PAIRS) + len(c["scale"] - SCALE)),
        "decoy_value": "median_house_value" in c["mean"],
    }


def _num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def score_queries(f: dict | None, truth: dict) -> list[dict]:
    rows = []
    eq = (f or {}).get("effect_queries") or {}
    for q in QUERIES:
        t = truth[q]
        e = eq.get(q) or {}
        est, lo, hi = _num(e.get("estimate")), _num(e.get("lower90")), _num(e.get("upper90"))
        has_int = lo is not None and hi is not None
        rows.append({
            "query": q, "truth": t, "estimate": est, "lower90": lo, "upper90": hi,
            "has_estimate": est is not None, "has_interval": has_int,
            "covered": bool(has_int and min(lo, hi) <= t <= max(lo, hi)),
            "width": abs(hi - lo) if has_int else None,
            "error": (est - t) if est is not None else None,
        })
    return rows


def score_predictions(pred: pl.DataFrame | None, tt: pl.DataFrame) -> dict:
    """Prediction metrics against the true mean f and a fresh draw y.

    tt holds one row per test row: row_id, f, sd, y,
    age_missing and households (for the coverage-by-quartile check).
    """
    if pred is None:
        return {"ok": False}
    j = tt.join(pred, on="row_id", how="left")
    if j["mean"].null_count() > 0:
        return {"ok": False}
    y, f = j["y"].to_numpy(), j["f"].to_numpy()
    m = j["mean"].to_numpy()
    lo, hi = j["lower90"].to_numpy().astype(float), j["upper90"].to_numpy().astype(float)
    has_pi = np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
    out = {"ok": True, "rmse_f": float(np.sqrt(np.mean((m - f) ** 2))), "rmse_y": float(np.sqrt(np.mean((m - y) ** 2))), "has_pi": bool(has_pi)}
    if has_pi:
        inside = (y >= lo) & (y <= hi)
        a = 0.1
        wink = (hi - lo) + (2 / a) * (lo - y) * (y < lo) + (2 / a) * (y - hi) * (y > hi)
        q = np.quantile(j["households"].to_numpy(), [0.25, 0.5, 0.75])
        bins = np.digitize(j["households"].to_numpy(), q)
        out.update(
            cov90=float(inside.mean()), width90=float(np.mean(hi - lo)), winkler=float(np.mean(wink)),
            **{f"cov_hh_q{k + 1}": float(inside[bins == k].mean()) for k in range(4)},
            **{f"width_hh_q{k + 1}": float(np.mean((hi - lo)[bins == k])) for k in range(4)},
            cov_age_missing=float(inside[j["age_missing"].to_numpy()].mean()),
        )
    return out


def load_outputs(out_dir: Path):
    f = p = None
    try:
        f = json.loads((out_dir / "findings.json").read_text())
    except Exception:  # noqa: BLE001
        pass
    try:
        p = pl.read_parquet(out_dir / "predictions.parquet").select(
            pl.col("row_id").cast(pl.Int64), pl.col("mean").cast(pl.Float64),
            pl.col("lower90").cast(pl.Float64), pl.col("upper90").cast(pl.Float64))
    except Exception:  # noqa: BLE001
        pass
    return f, p


def grade_library(bench: Path, lib: str, reps: list[str]) -> dict:
    """All per-replicate metrics for one library, as three Polars frames."""
    disc, qrows, prows = [], [], []
    for rep in reps:
        rd = bench / "truth" / "reps" / rep
        truth = json.loads((rd / "truth.json").read_text())
        kind = truth["kind"]
        f, p = load_outputs(bench / "eval" / lib / rep)
        disc.append({"library": lib, "rep": rep, "kind": kind, **score_discovery(f, kind)})
        for r in score_queries(f, truth["queries"]):
            qrows.append({"library": lib, "rep": rep, "kind": kind, **r})
        tt = pl.read_parquet(rd / "test_truth.parquet").with_columns(pl.col("row_id").cast(pl.Int64))
        prows.append({"library": lib, "rep": rep, "kind": kind, **score_predictions(p, tt)})
    return {
        "discovery": pl.DataFrame(disc, infer_schema_length=None),
        "queries": pl.DataFrame(qrows, infer_schema_length=None),
        "predictions": pl.DataFrame(prows, infer_schema_length=None),
    }
