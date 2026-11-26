"""Version-2 analysis fixed in PROTOCOL_v2.md.

python -I analyze_v2.py results.jsonl results_v2.jsonl HELDOUT_BEFORE HELDOUT_AFTER OUT_DIR

Runs the version-1 analysis on the version-2 results (OUT_DIR/analysis.json,
tables.md) and adds OUT_DIR/v2_comparison.json and v2_tables.md: the paired
version-1 -> version-2 change of every arm, the `lss_trees` decision, and the
held-out mean-accuracy check of `mb.fit`.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

import analyze

RNG = np.random.default_rng(20261124)
TIE = 0.01


def per_task(path, metric="crps"):
    """{(kind, task): {arm: mean over complete seeds}}."""
    runs = analyze.load(path)
    seeds = defaultdict(set)
    values = defaultdict(lambda: defaultdict(dict))
    for r in runs:
        seeds[(r["kind"], r["task"])].add(int(r["seed"]))
        if analyze.ok(r) and r["scores"].get(metric) is not None:
            values[(r["kind"], r["task"])][r["arm"]][int(r["seed"])] = r["scores"][metric]
    out = defaultdict(dict)
    for key, arms in values.items():
        for arm, by_seed in arms.items():
            if len(by_seed) == len(seeds[key]):
                out[key][arm] = float(np.mean(list(by_seed.values())))
    return out


def seconds(path):
    runs = analyze.load(path)
    by = defaultdict(list)
    for r in runs:
        if analyze.ok(r):
            by[r["arm"]].append(r["seconds"])
    return {arm: float(np.median(v)) for arm, v in by.items()}


def paired(a, b, kind):
    """Log ratio b / a over tasks of `kind` both complete."""
    keys = [k for k in a if k[0] == kind and k in b]
    return keys, np.log([b[k] / a[k] for k in keys]) if keys else np.array([])


def summary(lr):
    if lr.size == 0:
        return None
    draws = RNG.integers(0, lr.size, (10_000, lr.size))
    means = lr[draws].mean(axis=1)
    nonzero = lr[np.abs(lr) > 1e-12]
    return {
        "tasks": int(lr.size),
        "ratio": float(np.exp(lr.mean())),
        "ci95": [float(np.exp(np.quantile(means, 0.025))), float(np.exp(np.quantile(means, 0.975)))],
        "better": int((lr < -math.log1p(TIE)).sum()),
        "worse": int((lr > math.log1p(TIE)).sum()),
        "tied": int((np.abs(lr) <= math.log1p(TIE)).sum()),
        "wilcoxon_p": float(stats.wilcoxon(nonzero).pvalue) if nonzero.size >= 5 else None,
    }


def heldout(before_path, after_path):
    def load(path):
        out = {}
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            out[(r["id"], r["seed"])] = r
        return out
    before, after = load(before_path), load(after_path)
    keys = sorted(set(before) & set(after))
    ok = [k for k in keys if before[k].get("loss") and after[k].get("loss")]
    by_dataset = defaultdict(list)
    for k in ok:
        by_dataset[k[0]].append(math.log(after[k]["loss"] / before[k]["loss"]))
    lr = np.array([np.mean(v) for v in by_dataset.values()])
    failed_before = sum(1 for k in keys if not before[k].get("loss"))
    failed_after = sum(1 for k in keys if not after[k].get("loss"))
    fixed = sorted({k[0] for k in keys if not before[k].get("loss") and after[k].get("loss")})
    broken = sorted({k[0] for k in keys if before[k].get("loss") and not after[k].get("loss")})
    sec_b = [before[k]["seconds"] for k in ok if before[k].get("seconds")]
    sec_a = [after[k]["seconds"] for k in ok if after[k].get("seconds")]
    out = {"runs": len(keys), "failed_before": failed_before, "failed_after": failed_after,
           "datasets_fixed": fixed, "datasets_broken": broken,
           "loss_ratio": summary(lr),
           "median_seconds_before": float(np.median(sec_b)) if sec_b else None,
           "median_seconds_after": float(np.median(sec_a)) if sec_a else None}
    s = out["loss_ratio"]
    out["mean_accuracy_holds"] = bool(s and s["ratio"] <= 1.01 and s["ci95"][0] < 1.01)
    return out


def main(v1, v2, before, after, out_dir):
    out = Path(out_dir)
    analyze.main(v2, out)
    a1, a2 = per_task(v1), per_task(v2)
    arms = sorted({arm for task in a2.values() for arm in task})
    comparison = {"v1_to_v2": {}, "seconds_v1": seconds(v1), "seconds_v2": seconds(v2)}
    for arm in arms:
        comparison["v1_to_v2"][arm] = {}
        for kind in ("synth", "real"):
            keys = [k for k in a1 if k[0] == kind and arm in a1[k] and arm in a2.get(k, {})]
            lr = np.log([a2[k][arm] / a1[k][arm] for k in keys]) if keys else np.array([])
            comparison["v1_to_v2"][arm][kind] = summary(lr)
    # The lss_trees decision rule.
    decision = {}
    for kind in ("synth", "real"):
        keys = [k for k in a2 if k[0] == kind and "mboostrs.fit" in a2[k]
                and "mboostrs.fit_lss_trees" in a2[k]]
        lr = np.log([a2[k]["mboostrs.fit_lss_trees"] / a2[k]["mboostrs.fit"] for k in keys])
        decision[kind] = summary(lr)
    sec = comparison["seconds_v2"]
    time_ratio = (sec.get("mboostrs.fit_lss_trees", float("nan"))
                  / sec.get("mboostrs.fit", float("nan")))
    real, synth = decision["real"], decision["synth"]
    adopt = bool(real and synth and real["ratio"] < 0.99 and real["ci95"][1] < 1.0
                 and synth["ratio"] <= 1.01 and time_ratio < 2.0)
    comparison["lss_trees_decision"] = {**decision, "median_time_ratio": time_ratio,
                                        "adopt_true": adopt}
    comparison["heldout"] = heldout(before, after)
    (out / "v2_comparison.json").write_text(json.dumps(comparison, indent=1))
    # Only version-2 results are reported (PROTOCOL_v2.md); the version-1
    # pairing stays in v2_comparison.json as a record, not in the tables.
    d = comparison["lss_trees_decision"]
    lines = ["## lss_trees decision\n"]
    for kind in ("synth", "real"):
        s = d[kind]
        if s:
            lines.append(f"- {kind}: ratio {s['ratio']:.3f} ({s['ci95'][0]:.3f}–{s['ci95'][1]:.3f}), "
                         f"{s['better']} better, {s['tied']} tied, {s['worse']} worse over {s['tasks']} tasks")
    lines.append(f"- median time ratio {d['median_time_ratio']:.2f}; adopt `lss_trees=True`: {d['adopt_true']}")
    h = comparison["heldout"]
    s = h["loss_ratio"]
    lines += ["", "## Held-out mean accuracy of mb.fit\n",
              f"- runs {h['runs']}; failed before {h['failed_before']}, after {h['failed_after']}",
              f"- datasets fixed: {h['datasets_fixed']}; datasets newly failing: {h['datasets_broken']}"]
    if s:
        lines.append(f"- loss ratio after/before {s['ratio']:.4f} ({s['ci95'][0]:.4f}–{s['ci95'][1]:.4f}), "
                     f"{s['better']} better, {s['tied']} tied, {s['worse']} worse over {s['tasks']} datasets")
    lines.append(f"- median seconds before {h['median_seconds_before']}, after {h['median_seconds_after']}")
    lines.append(f"- mean accuracy holds: {h['mean_accuracy_holds']}")
    (out / "v2_tables.md").write_text("\n".join(lines) + "\n")
    # For the maintainer, not the book: each arm's version-2 CRPS against its
    # version-1 CRPS on the same task instances.
    pair = ["## Version 1 to version 2, per arm\n",
            "Geometric-mean CRPS ratio, version 2 over version 1, on tasks complete in "
            "both (below 1 is better in version 2).\n",
            "| Arm | Planted: ratio (95% interval) | better / tied / worse | Real: ratio (95% interval) | better / tied / worse |",
            "|---|---|---|---|---|"]
    for arm in arms:
        row = comparison["v1_to_v2"][arm]
        cells = []
        for kind in ("synth", "real"):
            s = row[kind]
            cells += (["–", "–"] if s is None else
                      [f"{s['ratio']:.3f} ({s['ci95'][0]:.3f}–{s['ci95'][1]:.3f})",
                       f"{s['better']} / {s['tied']} / {s['worse']}"])
        pair.append(f"| {arm} | " + " | ".join(cells) + " |")
    (out / "v1_vs_v2.md").write_text("\n".join(pair) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:6])
