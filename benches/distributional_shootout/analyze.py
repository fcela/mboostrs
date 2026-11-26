"""Analysis fixed in PROTOCOL.md. python -I analyze.py results.jsonl OUT_DIR

Writes OUT_DIR/analysis.json and OUT_DIR/tables.md.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

FOCAL = ("mboostrs.LSS_select", "mboostrs.fit")
# Deviation 1 (DEVIATIONS.md): ranks and gaps use the arms that completed at
# least this share of their runs, so one arm that fails almost everywhere
# does not shrink the comparison to the few tasks it finished.
RANK_MIN_COMPLETION = 0.9
TIE = 0.01
BOOT = 10_000


def load(path):
    runs = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    # The last record of a job wins (a resumed run can append a rerun).
    latest = {}
    for r in runs:
        latest[(r["kind"], r["task"], int(r["seed"]), r["arm"])] = r
    return list(latest.values())


def ok(r):
    s = r.get("scores") or {}
    return r.get("status") == "ok" and s.get("crps") is not None and math.isfinite(s["crps"])


def task_table(runs, kind):
    """{task: {arm: {seed: run}}} for one kind, ok runs only, plus all arms and seeds."""
    table = defaultdict(lambda: defaultdict(dict))
    arms, seeds = set(), defaultdict(set)
    for r in runs:
        if r["kind"] != kind:
            continue
        arms.add(r["arm"])
        seeds[r["task"]].add(int(r["seed"]))
        if ok(r):
            table[r["task"]][r["arm"]][int(r["seed"])] = r
    return table, sorted(arms), seeds


def complete(table, seeds, task, arm):
    return len(table[task].get(arm, {})) == len(seeds[task])


def mean_over_seeds(table, task, arm, key, ref=None):
    vals = []
    for seed, r in table[task][arm].items():
        v = r["scores"].get(key)
        if v is None:
            return None
        if ref is not None:
            base = table[task][ref].get(seed)
            if base is None:
                return None
            v = 1 - v / base["scores"]["crps"]
        vals.append(v)
    return float(np.mean(vals)) if vals else None


def bootstrap_mean(x, rng):
    x = np.asarray(x)
    draws = rng.integers(0, len(x), (BOOT, len(x)))
    means = x[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def analyse_kind(runs, kind):
    table, arms, seeds = task_table(runs, kind)
    tasks = sorted(seeds)
    out = {"tasks": len(tasks), "arms": arms}
    # Completion.
    total = defaultdict(int)
    good = defaultdict(int)
    failures = defaultdict(list)
    for r in runs:
        if r["kind"] != kind:
            continue
        total[r["arm"]] += 1
        if ok(r):
            good[r["arm"]] += 1
        else:
            failures[r["arm"]].append({"task": r["task"], "seed": r["seed"],
                                       "status": r.get("status"),
                                       "error": (r.get("error") or r.get("stderr") or "")[:300]})
    out["completion"] = {a: {"ok": good[a], "runs": total[a]} for a in arms}
    out["failures"] = dict(failures)
    # Per-task mean CRPS.
    crps = {t: {a: mean_over_seeds(table, t, a, "crps") for a in arms if complete(table, seeds, t, a)}
            for t in tasks}
    # 1. Ranks on tasks every arm completed.
    ranked = [a for a in arms if good[a] >= RANK_MIN_COMPLETION * total[a]]
    out["ranked_arms"] = ranked
    full = [t for t in tasks if all(a in crps[t] for a in ranked)]
    ranks = defaultdict(list)
    for t in full:
        order = stats.rankdata([crps[t][a] for a in ranked])
        for a, rk in zip(ranked, order, strict=True):
            ranks[a].append(float(rk))
    out["rank_tasks"] = len(full)
    out["mean_rank"] = {a: float(np.mean(ranks[a])) if ranks[a] else None for a in arms}
    # 2. Skill over climatology and gap to best.
    skill, gap = {}, {}
    for a in arms:
        s = [mean_over_seeds(table, t, a, "crps", ref="reference.climatology") for t in tasks
             if complete(table, seeds, t, a) and complete(table, seeds, t, "reference.climatology")]
        s = [v for v in s if v is not None]
        skill[a] = {"mean": float(np.mean(s)) if s else None,
                    "median": float(np.median(s)) if s else None, "tasks": len(s)}
        best = {t: min(crps[t][b] for b in ranked) for t in full}
        g = [crps[t][a] / best[t] - 1 for t in full if a in crps[t]]
        gap[a] = {"median": float(np.median(g)) if g else None,
                  "mean": float(np.mean(g)) if g else None,
                  "best_count": int(sum(1 for t in full if a in crps[t] and crps[t][a] == best[t]))}
    out["skill_over_climatology"] = skill
    out["gap_to_best"] = gap
    # 3. Paired comparisons.
    rng = np.random.default_rng(20261123)
    paired = {}
    for focal in FOCAL:
        if focal not in arms:
            continue
        paired[focal] = {}
        for other in arms:
            if other == focal:
                continue
            common = [t for t in tasks if focal in crps[t] and other in crps[t]]
            if not common:
                continue
            lr = np.log([crps[t][focal] / crps[t][other] for t in common])
            lo, hi = bootstrap_mean(lr, rng)
            nonzero = lr[np.abs(lr) > 1e-12]
            p = float(stats.wilcoxon(nonzero).pvalue) if len(nonzero) >= 5 else None
            paired[focal][other] = {
                "tasks": len(common),
                "mean_log_ratio": float(lr.mean()), "ci95": [lo, hi],
                "geometric_mean_ratio": float(np.exp(lr.mean())),
                "wins": int((lr < -math.log1p(TIE)).sum()),
                "losses": int((lr > math.log1p(TIE)).sum()),
                "ties": int((np.abs(lr) <= math.log1p(TIE)).sum()),
                "wilcoxon_p": p}
    out["paired"] = paired
    # 4. Planted tasks: excess over the truth by generator and size.
    if kind == "synth":
        excess = {}
        for t in tasks:
            excess[t] = {a: mean_over_seeds(table, t, a, "excess_crps") for a in arms
                         if complete(table, seeds, t, a)}
        out["excess_over_truth"] = excess
        qerr = {}
        for t in tasks:
            qerr[t] = {a: mean_over_seeds(table, t, a, "quantile_error") for a in arms
                       if complete(table, seeds, t, a)}
        out["quantile_error"] = qerr
    # 5. Secondary per-arm summaries over all ok runs.
    keys = ["quantile_calibration_mae", "quantile_calibration_max", "coverage_50", "coverage_80",
            "coverage_90", "coverage_98", "width_90_over_sd", "coverage_90_tercile_max_dev"]
    secondary = {}
    for a in arms:
        rs = [r for t in tasks for r in table[t].get(a, {}).values()]
        row = {}
        for k in keys:
            v = [r["scores"][k] for r in rs if r["scores"].get(k) is not None]
            row[k] = float(np.mean(v)) if v else None
        secs = [r["seconds"] for r in rs]
        row["seconds_median"] = float(np.median(secs)) if secs else None
        row["seconds_p90"] = float(np.quantile(secs, 0.9)) if secs else None
        row["seconds_total"] = float(np.sum(secs)) if secs else None
        nn = [r["scores"].get("nll_nonfinite_rows", 0) for r in rs if "nll_nonfinite_rows" in r["scores"]]
        row["runs_with_zero_density_rows"] = int(sum(1 for v in nn if v > 0)) if nn else None
        secondary[a] = row
    out["secondary"] = secondary
    # Log score among parametric arms: mean rank on tasks all of them completed.
    parametric = [a for a in arms if all(
        "nll" in r["scores"] for t in tasks for r in table[t].get(a, {}).values())
        and any(table[t].get(a) for t in tasks)]
    nll = {t: {a: mean_over_seeds(table, t, a, "nll") for a in parametric
               if complete(table, seeds, t, a)} for t in tasks}
    nfull = [t for t in tasks if len(nll[t]) == len(parametric) and
             all(v is not None for v in nll[t].values())]
    nranks = defaultdict(list)
    for t in nfull:
        order = stats.rankdata([nll[t][a] for a in parametric])
        for a, rk in zip(parametric, order, strict=True):
            nranks[a].append(float(rk))
    out["log_score_mean_rank"] = {"tasks": len(nfull),
                                  "ranks": {a: float(np.mean(nranks[a])) for a in parametric if nranks[a]}}
    # Point accuracy: RMSE relative to the best arm per task.
    rm = {t: {a: mean_over_seeds(table, t, a, "rmse_mean") for a in arms if a in crps[t]} for t in full}
    out["rmse_gap_to_best_median"] = {
        a: float(np.median([rm[t][a] / min(rm[t][b] for b in ranked) - 1
                            for t in full if a in rm[t]])) if full and a in ranked else None
        for a in arms}
    out["per_task_crps"] = crps
    return out


def fmt_skill(v):
    return "–" if v is None else (f"{v:.3f}" if v > -10 else "< −10")


def fmt(v, d=3):
    return "–" if v is None else f"{v:.{d}f}"


def tables(analysis):
    lines = []
    for kind, title in (("synth", "Planted-truth tasks"), ("real", "OpenML-CTR23")):
        a = analysis.get(kind)
        if not a:
            continue
        lines.append(f"## {title}\n")
        lines.append(f"{a['tasks']} tasks. Mean rank, gap to best and best-on count use the "
                     f"{a['rank_tasks']} tasks completed by every arm that finished at least "
                     f"{RANK_MIN_COMPLETION:.0%} of its runs; other arms show a dash.\n")
        lines.append("| Arm | Completed | Mean rank | Skill, median | Skill, mean | Median gap to best | Best on | Calibration MAE | Cov 50 | Cov 90 | Cov 98 | Median s |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        order = sorted(a["arms"], key=lambda x: a["mean_rank"].get(x) or 1e9)
        for arm in order:
            c = a["completion"][arm]
            s = a["secondary"][arm]
            lines.append(
                f"| {arm} | {c['ok']}/{c['runs']} | {fmt(a['mean_rank'][arm], 2)} | "
                f"{fmt(a['skill_over_climatology'][arm]['median'])} | "
                f"{fmt_skill(a['skill_over_climatology'][arm]['mean'])} | "
                f"{fmt(a['gap_to_best'][arm]['median'])} | {a['gap_to_best'][arm]['best_count']} | "
                f"{fmt(s['quantile_calibration_mae'])} | {fmt(s['coverage_50'])} | "
                f"{fmt(s['coverage_90'])} | {fmt(s['coverage_98'])} | {fmt(s['seconds_median'], 1)} |")
        lines.append("")
        for focal, rows in a["paired"].items():
            lines.append(f"### {focal} against each arm\n")
            lines.append("| Other arm | Tasks | CRPS ratio (geometric mean) | 95% interval | Wins | Ties | Losses | Wilcoxon p |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for other, p in sorted(rows.items(), key=lambda kv: kv[1]["mean_log_ratio"]):
                lo, hi = (math.exp(v) for v in p["ci95"])
                lines.append(f"| {other} | {p['tasks']} | {p['geometric_mean_ratio']:.3f} | "
                             f"{lo:.3f}–{hi:.3f} | {p['wins']} | {p['ties']} | {p['losses']} | "
                             f"{fmt(p['wilcoxon_p'], 4)} |")
            lines.append("")
        if kind == "synth":
            lines.append("### Excess CRPS over the true distribution, by task\n")
            arms = order
            lines.append("| Task | " + " | ".join(arms) + " |")
            lines.append("|---|" + "---|" * len(arms))
            for t, row in sorted(a["excess_over_truth"].items()):
                lines.append(f"| {t} | " + " | ".join(fmt(row.get(x)) for x in arms) + " |")
            lines.append("")
        lr = a["log_score_mean_rank"]
        lines.append(f"### Log score, parametric arms (mean rank over {lr['tasks']} tasks)\n")
        for arm, rk in sorted(lr["ranks"].items(), key=lambda kv: kv[1]):
            zero = a["secondary"][arm]["runs_with_zero_density_rows"]
            lines.append(f"- {arm}: {rk:.2f} (runs with a zero-density test row: {zero})")
        lines.append("")
    return "\n".join(lines)


def main(path, out_dir):
    runs = load(path)
    analysis = {kind: analyse_kind(runs, kind) for kind in ("synth", "real")
                if any(r["kind"] == kind for r in runs)}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(analysis, indent=1))
    (out / "tables.md").write_text(tables(analysis))
    print(tables(analysis))


if __name__ == "__main__":
    main(*sys.argv[1:3])
