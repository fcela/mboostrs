"""Measurements by scorecard dimension, as defined in PROTOCOL.md."""
import json
import pathlib
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

HERE = pathlib.Path(__file__).resolve().parent
rows = [json.loads(line) for line in (HERE / "results.jsonl").read_text().splitlines()]
ARMS = ("mboostrs", "catboost", "lightgbm", "xgboost", "ebm", "reference")
SEEDS = (5000, 5001, 5002)
synth = {t["id"]: t for t in json.load(open(HERE / "data/manifest_synth.json"))["tasks"]}
by = {(str(r["id"]), r["seed"], r["arm"]): r for r in rows}
real_ids = sorted({str(r["id"]) for r in rows if not str(r["id"]).startswith("S")}, key=int)
kind = {str(r["id"]): r.get("kind") for r in rows if r.get("kind")}
ok = lambda r: r is not None and r["status"] == "ok"  # noqa: E731
rep = {}


def task_loss(t, arm, seeds):
    rs = [by.get((t, s, arm)) for s in seeds]
    return float(np.mean([r["loss"] for r in rs])) if all(map(ok, rs)) else None


def boot(v, n=10000):
    rng = np.random.default_rng(0)
    v = np.asarray(v)
    m = v[rng.integers(0, len(v), (n, len(v)))].mean(axis=1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


# ---- D11 -----------------------------------------------------------------
def d11(tasks, seeds_of):
    L = {t: {a: task_loss(t, a, seeds_of(t)) for a in ARMS} for t in tasks}
    out = {}
    for a in ARMS:
        exc, gap, comp = [], [], 0
        for t in tasks:
            la, lr = L[t][a], L[t]["reference"]
            done = [v for v in L[t].values() if v is not None]
            if la is None:
                continue
            comp += 1
            if lr is not None:
                exc.append(la / lr - 1)
            gap.append(la / min(done) - 1)
        out[a] = dict(complete_tasks=comp, of=len(tasks),
                      median_excess_over_reference=round(float(np.median(exc)), 4) if exc else None,
                      beats_reference=int(sum(e < 0 for e in exc)),
                      median_gap_to_best_complete=round(float(np.median(gap)), 4) if gap else None,
                      within_5pct_of_best=int(sum(g <= 0.05 for g in gap)),
                      more_than_10pct_behind_best=int(sum(g > 0.10 for g in gap)))
    paired = {}
    for a in ARMS:
        if a == "mboostrs":
            continue
        r = [np.log(L[t]["mboostrs"] / L[t][a]) for t in tasks if L[t]["mboostrs"] and L[t][a]]
        if len(r) > 1:
            paired[f"mboostrs_vs_{a}"] = dict(n=len(r), mboostrs_better=int(sum(x < 0 for x in r)),
                                              mboostrs_worse=int(sum(x > 0 for x in r)),
                                              geometric_mean_ratio=round(float(np.exp(np.mean(r))), 4),
                                              median_ratio=round(float(np.exp(np.median(r))), 4),
                                              mean_log_ratio_ci95=boot(r), wilcoxon_p=float(wilcoxon(r).pvalue))
    return dict(by_arm=out, paired=paired)


rep["D11"] = {
    "real_regression": d11([t for t in real_ids if kind[t] == "regression"], lambda t: SEEDS),
    "real_binary": d11([t for t in real_ids if kind[t] == "binary"], lambda t: SEEDS),
}
fam = defaultdict(list)
for t in synth:
    fam[t.split("_")[0]].append(t)
rep["D11"]["planted_by_task"] = {
    f: {a: (round(float(np.mean(v)), 4) if all(x is not None for x in v) else None)
        for a in ARMS for v in [[task_loss(t, a, (5000,)) for t in ts]]}
    for f, ts in sorted(fam.items())}

# ---- D1b -----------------------------------------------------------------
d1b = {}
for a in ARMS + ("mboostrs_native",):
    src = "mboostrs" if a == "mboostrs_native" else a
    hit, tot, fd, found_n, silent, s4 = defaultdict(int), defaultdict(int), 0, 0, 0, 0
    other = defaultdict(lambda: [0, 0])
    for t, task in synth.items():
        r = by.get((t, 5000, src))
        if not ok(r):
            continue
        if a == "mboostrs_native":
            found = set(r.get("native_claim") or [])
        else:
            found = {c for c, v in r["permutation"].items() if v["found"]}
        truth = task["truth"]
        for c, typ in truth.items():
            if typ in ("scale_only", "proxy"):
                other[typ][0] += c in found
                other[typ][1] += 1
            else:
                tot[typ] += 1
                hit[typ] += c in found
        counted = {c for c in found if truth.get(c) not in ("scale_only", "proxy")}
        found_n += len(counted)
        fd += sum(c not in truth for c in counted)
        if t.startswith("S4"):
            s4 += 1
            silent += not found
    d1b[a] = dict(tpr_by_type={k: f"{hit[k]}/{tot[k]}" for k in sorted(tot)},
                  tpr_overall=round(sum(hit.values()) / max(sum(tot.values()), 1), 3),
                  fdr=round(fd / found_n, 3) if found_n else None,
                  null_task_silent=f"{silent}/{s4}",
                  found_scale_only=f"{other['scale_only'][0]}/{other['scale_only'][1]}",
                  found_proxy=f"{other['proxy'][0]}/{other['proxy'][1]}")
known_pairs = {}
for a in ARMS:
    h = n = 0
    for r in rows:
        if r["arm"] == a and ok(r) and r.get("effects"):
            for e in r["effects"]:
                n += 1
                h += e["recovered"]
    known_pairs[a] = f"{h}/{n}" + (f" ({h / n:.1%})" if n else "")
common = defaultdict(lambda: [0, 0])
for t in real_ids:
    for s in SEEDS:
        rs = {a: by.get((t, s, a)) for a in ARMS}
        if all(map(ok, rs.values())) and rs["mboostrs"].get("effects"):
            for a in ARMS:
                for e in rs[a]["effects"]:
                    common[a][0] += e["recovered"]
                    common[a][1] += 1
d1b["known_effects_real_each_arm_completed"] = known_pairs
d1b["known_effects_real_all_arms_completed"] = {a: f"{h}/{n}" for a, (h, n) in common.items()}
rep["D1b"] = d1b

# ---- D9 ------------------------------------------------------------------
d9 = {}
for a in ARMS:
    cov, wid, s3, ece = [], [], [], []
    for r in rows:
        if r["arm"] != a or not ok(r):
            continue
        iv = r.get("interval")
        if isinstance(iv, dict):
            cov.append(iv["coverage"])
            wid.append(iv["median_width_over_sd"])
            if "coverage_by_x2_tercile" in iv:
                s3.append(iv["coverage_by_x2_tercile"])
        if "ece" in r:
            ece.append(r["ece"])
    d9[a] = dict(interval_runs=len(cov), offered=bool(cov) or a != "ebm",
                 median_coverage=round(float(np.median(cov)), 3) if cov else None,
                 share_within_085_095=round(float(np.mean([(0.85 <= c <= 0.95) for c in cov])), 3) if cov else None,
                 share_below_085=round(float(np.mean([c < 0.85 for c in cov])), 3) if cov else None,
                 median_width_over_sd=round(float(np.nanmedian(np.where(np.isfinite(wid), wid, np.nan))), 3) if wid else None,
                 nonfinite_width_runs=int(np.sum(~np.isfinite(wid))) if wid else 0,
                 s3_coverage_by_x2_tercile=[round(float(x), 3) for x in np.mean(s3, axis=0)] if s3 else None,
                 binary_median_ece=round(float(np.median(ece)), 4) if ece else None)
rep["D9"] = d9

# ---- D5, D6 --------------------------------------------------------------
d5, d6 = {}, {}
small = [t for t in real_ids if (by.get((t, 5000, "reference")) or {}).get("n_train", 1e9) < 400]
for a in ARMS:
    sec = [r["fit_predict_seconds"] for r in rows if r["arm"] == a and ok(r) and "fit_predict_seconds" in r]
    d5[a] = dict(runs=len(sec), median_fit_predict_seconds=round(float(np.median(sec)), 3),
                 p90=round(float(np.percentile(sec, 90)), 3))
    runs = [by.get((t, s, a)) for t in real_ids for s in SEEDS]
    fails = defaultdict(int)
    for r in runs:
        if not ok(r):
            fails[(r or {"status": "missing"})["status"][:70]] += 1
    exc = []
    for t in small:
        la, lr = task_loss(t, a, SEEDS), task_loss(t, "reference", SEEDS)
        if la and lr:
            exc.append(la / lr - 1)
    d6[a] = dict(real_runs=len(runs), completed=int(sum(map(ok, runs))),
                 completion_rate=round(float(np.mean([ok(r) for r in runs])), 3),
                 top_failures=dict(sorted(fails.items(), key=lambda kv: -kv[1])[:4]),
                 small_n_tasks=len(exc),
                 small_n_median_excess_over_reference=round(float(np.median(exc)), 4) if exc else None)
s5 = rep["D11"]["planted_by_task"]["S5"]
for a in ARMS:
    d6[a]["S5_loss"] = s5[a]
rep["D5"], rep["D6"] = d5, d6
sess = HERE / "sessions.jsonl"
if sess.exists():
    rep["D5"]["session"] = {json.loads(line)["arm"]: {k: json.loads(line)[k] for k in ("median_seconds", "p90_seconds", "total_seconds")}
                            for line in sess.read_text().splitlines()}
json.dump(rep, open(HERE / "analysis.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
