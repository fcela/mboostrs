"""Grade one analyst's outputs on every replicate against the hidden truth.

python grade.py --eval <bench>/eval/<lib> --hidden <hidden dir>
Prints a JSON summary and writes <eval>/scores.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

QUERIES = ["ATE", "ATT", "CATE_educ_lt12", "CATE_educ_ge16", "CATE_prior_q1", "CATE_prior_q4",
           "CATE_age_lt25", "EMP_ATE"]
COLUMNS = ["age", "education", "experience", "ethnicity", "smsa", "region", "parttime",
           "prior_earnings", "months_unemployed", "distance_km", "caseworker_note_len", "enrolled",
           "certificate", "responded", "hours_post", "earnings_post"]
# Roles that describe the same causal position under the brief's vocabulary.
EQUIVALENT = {"instrument": {"instrument", "treatment_only"}, "treatment_only": {"instrument", "treatment_only"}}
CONFOUNDER_GROUPS = [["prior_earnings"], ["education"], ["months_unemployed"], ["age", "experience"],
                     ["region"], ["smsa"]]


def load(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def grade_queries(f: dict, truth: dict) -> dict:
    out = {}
    eq = (f or {}).get("effect_queries", {}) or {}
    for q in QUERIES:
        r = eq.get(q) or {}
        est, lo, hi = num(r.get("estimate")), num(r.get("lower90")), num(r.get("upper90"))
        t = truth[q]
        out[q] = {
            "estimate": est, "truth": t,
            "error": None if est is None else est - t,
            "covered": lo is not None and hi is not None and lo <= t <= hi,
            "width": None if lo is None or hi is None else hi - lo,
            "excludes_zero": lo is not None and hi is not None and (lo > 0 or hi < 0),
        }
    return out


def grade_roles(f: dict, key: dict) -> dict:
    roles = (f or {}).get("roles", {}) or {}
    correct = {}
    for col in COLUMNS:
        want = key["roles"][col]
        got = str(roles.get(col, "")).strip().lower()
        ok = got in EQUIVALENT.get(want, {want})
        if col == "age" and got in {"confounder"}:
            ok = True
        correct[col] = ok
    adj = set((f or {}).get("adjustment_set", []) or [])
    bad_adjust = sorted(adj & set(key["never_adjust"]))
    missing_conf = [g for g in CONFOUNDER_GROUPS if not (adj & set(g))]
    return {"accuracy": float(np.mean(list(correct.values()))), "correct": correct,
            "adjusts_post_treatment": bad_adjust, "missing_confounders": missing_conf,
            "adjustment_set_valid": not bad_adjust and not missing_conf}


def grade_modifiers(f: dict, key: dict, null: bool) -> dict:
    claims = [m for m in (f or {}).get("effect_modifiers", []) or [] if isinstance(m, dict)]
    names = [str(m.get("variable", "")) for m in claims]
    material = key["effect_modifiers"]
    if null:
        return {"claimed": names, "false_claims": len(names)}
    tp = [n for n in names if n in material]
    fp = [n for n in names if n in key["effect_modifiers_none"]
          or n not in material and n not in key["effect_modifiers_neutral"]]
    dir_ok = []
    for m in claims:
        n = m.get("variable")
        if n in material and n != "parttime":
            dir_ok.append(m.get("direction") == material[n])
    return {"claimed": names, "tpr": len(set(tp)) / len(material),
            "fdr": len(fp) / max(1, len(tp) + len(fp)), "false": fp,
            "direction_correct": float(np.mean(dir_ok)) if dir_ok else None}


def grade_cate(pred_path: Path, truth_path: Path) -> dict:
    try:
        p = pl.read_parquet(pred_path)
        t = pl.read_parquet(truth_path)
        d = t.join(p, on="row_id", how="left")
        cate, est = d["cate"].to_numpy(), d["cate_right"].to_numpy().astype(float)
        lo, hi = d["lower90"].to_numpy().astype(float), d["upper90"].to_numpy().astype(float)
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)[:300]}
    ok = np.isfinite(est)
    k = int(round(0.2 * len(cate)))
    chosen = np.argsort(-np.where(ok, est, -np.inf))[:k]
    oracle = np.argsort(-cate)[:k]
    cov = np.isfinite(lo) & np.isfinite(hi) & (lo <= cate) & (cate <= hi)
    return {
        "pehe": float(np.sqrt(np.mean((est[ok] - cate[ok]) ** 2))) if ok.any() else None,
        "missing": int((~ok).sum()),
        "coverage": float(cov.mean()),
        "mean_width": float(np.nanmean(hi - lo)) if np.isfinite(hi - lo).any() else None,
        "policy_value": float(cate[chosen].mean()),
        "policy_oracle": float(cate[oracle].mean()),
        "policy_random": float(cate.mean()),
        "policy_regret_share": (float((cate[oracle].mean() - cate[chosen].mean())
                                      / (cate[oracle].mean() - cate.mean()))
                                if cate[oracle].mean() > cate.mean() else None),
    }


def grade_rep(ev: Path, rd: Path, key: dict) -> dict:
    f = load(ev / "findings.json")
    truth = load(rd / "truth.json")
    null = bool(truth["null"])
    res = {"ran": f is not None, "null": null}
    res["queries"] = grade_queries(f, truth)
    res["roles"] = grade_roles(f, key)
    res["modifiers"] = grade_modifiers(f, key, null)
    ch = (f or {}).get("channels", {}) or {}
    want = not null
    res["channels_correct"] = (bool(ch.get("employment")) == want) + (bool(ch.get("earnings_if_employed")) == want)
    lim = json.dumps((f or {}).get("limited_overlap", []) or []).lower()
    res["overlap_named"] = "prior_earnings" in lim
    if null:
        res["null_false_claims"] = (sum(r["excludes_zero"] for r in res["queries"].values())
                                    + res["modifiers"]["false_claims"]
                                    + 2 - res["channels_correct"])
    else:
        res["cate"] = grade_cate(ev / "cate_predictions.parquet", rd / "test_truth.parquet")
    return res


def summarise(per: dict) -> dict:
    alt = [r for k, r in per.items() if not r["null"] and r["ran"]]
    nul = [r for k, r in per.items() if r["null"]]
    s = {"ran": sum(r["ran"] for r in per.values()), "replicates": len(per)}
    if alt:
        s["coverage"] = {q: float(np.mean([r["queries"][q]["covered"] for r in alt])) for q in QUERIES}
        s["coverage_all_queries"] = float(np.mean([r["queries"][q]["covered"] for r in alt for q in QUERIES]))
        s["rmse"] = {q: float(np.sqrt(np.mean([r["queries"][q]["error"] ** 2 for r in alt
                                               if r["queries"][q]["error"] is not None])))
                     if any(r["queries"][q]["error"] is not None for r in alt) else None for q in QUERIES}
        s["role_accuracy"] = float(np.mean([r["roles"]["accuracy"] for r in alt]))
        s["adjustment_valid_share"] = float(np.mean([r["roles"]["adjustment_set_valid"] for r in alt]))
        s["modifier_tpr"] = float(np.mean([r["modifiers"]["tpr"] for r in alt]))
        s["modifier_fdr"] = float(np.mean([r["modifiers"]["fdr"] for r in alt]))
        s["channels_correct"] = float(np.mean([r["channels_correct"] for r in alt])) / 2
        s["overlap_named"] = float(np.mean([r["overlap_named"] for r in alt]))
        cs = [r["cate"] for r in alt if "pehe" in r.get("cate", {}) and r["cate"]["pehe"] is not None]
        if cs:
            s["pehe"] = float(np.mean([c["pehe"] for c in cs]))
            s["cate_coverage"] = float(np.mean([c["coverage"] for c in cs]))
            s["policy_regret_share"] = float(np.mean([c["policy_regret_share"] for c in cs]))
    null_claims = [r.get("null_false_claims", 0) if r["ran"] else 0 for r in nul]
    s["null_false_claims"] = float(np.mean(null_claims)) if nul else None
    s["null_ran"] = sum(r["ran"] for r in nul)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--hidden", required=True)
    a = ap.parse_args()
    ev, hidden = Path(a.eval), Path(a.hidden)
    key = json.loads((hidden / "key.json").read_text())
    per = {}
    for rd in sorted((hidden / "reps").iterdir()):
        if rd.name == "dev":
            continue
        per[rd.name] = grade_rep(ev / rd.name, rd, key)
    out = {"summary": summarise(per), "per_replicate": per}
    dev = hidden / "reps" / "dev"
    if (ev / "dev").exists():
        out["dev"] = grade_rep(ev / "dev", dev, key)
    (ev / "scores.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main()
