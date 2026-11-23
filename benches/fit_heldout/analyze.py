"""Analyses A, B, C and the secondaries of PREREGISTRATION_v2.md."""
import json
import pathlib
import sys

import numpy as np
from scipy.stats import wilcoxon

HERE = pathlib.Path(__file__).resolve().parent
rows = [json.loads(line) for line in (HERE / sys.argv[1]).read_text().splitlines()]
SEEDS = tuple(int(s) for s in sys.argv[2].split(",")) if len(sys.argv) > 2 else (2000, 2001, 2002)
by = {(r["id"], r["seed"], r["arm"]): r for r in rows}
ids = sorted({r["id"] for r in rows})
names = {r["id"]: r.get("name") for r in rows if r.get("name")}
known = json.load(open(HERE / "known_effects.json"))
tier = sorted(int(k) for k in known)
scan = json.load(open(HERE / "onceover_scan.json"))
flagged = {int(k) for k, v in scan.items() if v.get("identity_with_target") or v.get("leakage")}


def ok(r):
    return r is not None and r["status"] == "ok"


def ratios(arm, subset=None):
    out = {}
    for d in ids if subset is None else subset:
        a = [by.get((d, s, arm)) for s in SEEDS]
        b = [by.get((d, s, "old")) for s in SEEDS]
        if all(map(ok, a)) and all(map(ok, b)):
            out[d] = float(np.mean([np.log(x["loss"] / y["loss"]) for x, y in zip(a, b, strict=True)]))
    return out


def boot(v, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(v)
    m = v[rng.integers(0, len(v), (n, len(v)))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def stats(r):
    v = np.array(list(r.values()))
    if len(v) < 2:
        return {"n": len(v)}
    lo, hi = boot(v)
    return dict(n=len(v), mean_r=round(float(v.mean()), 4), ci95=(round(lo, 4), round(hi, 4)),
                wilcoxon_p=float(wilcoxon(v).pvalue), better=int((v < 0).sum()),
                worse=int((v > 0).sum()), median_ratio=round(float(np.exp(np.median(v))), 4),
                geometric_mean_ratio=round(float(np.exp(v.mean())), 4))


def fails(arm, subset=None):
    return [(d, s, by[(d, s, arm)]["status"][:120]) for d in (ids if subset is None else subset)
            for s in SEEDS if (d, s, arm) in by and not ok(by[(d, s, arm)])]


def effects(arm):
    hit = tot = 0
    missed = []
    for d in tier:
        for s in SEEDS:
            a, b = by.get((d, s, arm)), by.get((d, s, "old"))
            if not (ok(a) and ok(b)):
                continue
            for e in a["effects"]:
                tot += 1
                hit += e["recovered"]
                if not e["recovered"]:
                    missed.append((names.get(d), s, e["variable"], round(e["change"], 6)))
    return hit, tot, missed


report = {}
for arm in ("new", "quick"):
    rA = ratios(arm)
    A = stats(rA)
    fa, fo = fails(arm), fails("old")
    A.update(failures_arm=len(fa), failures_old=len(fo))
    A["holds"] = bool(A["mean_r"] < 0 and A["ci95"][1] < 0 and A["wilcoxon_p"] < 0.05
                      and len(fa) <= len(fo))
    B = stats(ratios(arm, tier))
    B["holds"] = bool(B.get("mean_r", 1) < 0 and B["ci95"][1] < 0 and B["wilcoxon_p"] < 0.05)
    h_a, t_a, miss_a = effects(arm)
    h_o, t_o, miss_o = effects("old")
    C = dict(recovered_arm=h_a, recovered_old=h_o, pairs=t_a,
             rate_arm=round(h_a / t_a, 4), rate_old=round(h_o / t_o, 4))
    C["holds"] = bool(C["rate_arm"] >= C["rate_old"] - 0.05)
    t_arm = {k[:2]: r["seconds"] for k, r in by.items() if k[2] == arm and ok(r)}
    t_old = {k[:2]: r["seconds"] for k, r in by.items() if k[2] == "old" and ok(r)}
    common = [k for k in t_arm if k in t_old and t_old[k] > 0]
    src = {d: ("UCI" if d >= 100000 and d < 200000 else "mb" if d >= 200000 else "OpenML") for d in ids}
    report[arm] = dict(
        A_all=A, B_tier=B, C_known_effects=C,
        secondary=dict(
            uci_only=stats({d: v for d, v in rA.items() if src[d] == "UCI"}),
            openml_only=stats({d: v for d, v in rA.items() if src[d] == "OpenML"}),
            onceover_flagged=stats({d: v for d, v in rA.items() if d in flagged}),
            onceover_unflagged=stats({d: v for d, v in rA.items() if d not in flagged}),
            more_than_2pct_worse={f"{d} {names.get(d)}": round(float(np.exp(v)), 4)
                                  for d, v in sorted(rA.items(), key=lambda t: -t[1]) if np.exp(v) > 1.02},
            median_time_ratio=round(float(np.median([t_arm[k] / t_old[k] for k in common])), 2),
            median_seconds=(float(np.median(list(t_old.values()))), float(np.median(list(t_arm.values())))),
            p95_seconds=float(np.percentile(list(t_arm.values()), 95)),
            failures_only_arm=sorted({(d, s) for d, s, _ in fa} - {(d, s) for d, s, _ in fo}),
            failures_only_old=sorted({(d, s) for d, s, _ in fo} - {(d, s) for d, s, _ in fa}),
            missed_effects_arm=miss_a, missed_effects_old=miss_o),
        per_dataset={f"{d} {names.get(d)}": round(float(np.exp(v)), 4)
                     for d, v in sorted(rA.items(), key=lambda t: t[1])})
json.dump(report, open(HERE / "analysis_v2.json", "w"), indent=1, default=str)
for arm, rep in report.items():
    print(arm, "A", rep["A_all"])
    print(arm, "B", rep["B_tier"])
    print(arm, "C", rep["C_known_effects"])
    for k in ("uci_only", "openml_only", "onceover_flagged", "onceover_unflagged"):
        print("   ", k, rep["secondary"][k])
    for k in ("more_than_2pct_worse", "median_time_ratio", "median_seconds", "p95_seconds",
              "failures_only_arm", "failures_only_old", "missed_effects_arm", "missed_effects_old"):
        print("   ", k, rep["secondary"][k])
