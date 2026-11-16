"""Annotated tables and scientific figures from the frozen study artifacts."""
from __future__ import annotations

import html
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import polars as pl
from great_tables import GT

from docs.book.cases.bench_tables import bold_best

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "docs/benchmarks"
LABELS = {"linear": "Ridge / logistic", "mboost_additive": "mboostrs additive",
          "mboost_hybrid": "mboostrs hybrid", "mboost_auto": "mboostrs autospec",
          "ebm": "EBM", "catboost": "CatBoost", "lightgbm": "LightGBM", "xgboost": "XGBoost"}


def rows(document):
    points, failures = [], []
    for row in document["results"]:
        if row.get("_status") != "success":
            failures.append(row)
            continue
        for result in row["results"]:
            scores, size, explanation = result["scores"], result["structure"], result["explanation"]
            points.append({"task": row["task"], "seed": row["seed"], "method": row["method"],
                           "tier": result["tier"], "loss": scores["loss"],
                           "seconds": result["workflow_seconds"],
                           "explain_seconds": explanation["seconds"],
                           "native_explain_seconds": explanation["native_decomposition"]["seconds"],
                           "used_inputs": size["used_inputs"], "native_terms": size["native_terms"],
                           "bytes": size.get("gzip_pickle_bytes"),
                           "rss_mb": row["peak_process_rss_bytes"] / 1024**2,
                           "coverage90": scores.get("coverage90"), "width90_sd": scores.get("width90_sd"),
                           "auc": scores.get("auc"), "calibrated_loss": scores.get("calibrated_loss"),
                           "curve_nrmse": explanation.get("curve_nrmse"),
                           "recall_at3": explanation.get("driver_group_recall_at3"),
                           "prediction_ms": 1000 * float(np.median(scores["prediction_seconds"]))})
    return points, failures


def aggregate(points):
    groups = defaultdict(list)
    for point in points:
        groups[(point["task"], point["method"], point["tier"])].append(point)
    cells = []
    for (task, method, tier), values in groups.items():
        cell = {"task": task, "method": method, "tier": tier, "n": len(values)}
        for key in values[0]:
            if key in ("task", "method", "tier", "seed"):
                continue
            valid = [v[key] for v in values if v[key] is not None]
            cell[key] = float(np.mean(valid)) if valid else None
        cell["loss_sd"] = float(np.std([v["loss"] for v in values], ddof=1)) if len(values) > 1 else None
        cell["complete"] = len(values) == (5 if task in ("smooth_sparse", "joint_boundary", "correlated_proxies") else 3)
        cells.append(cell)
    references = {}
    for cell in cells:
        if cell["complete"] and not cell["method"].startswith("mboost"):
            key = (cell["task"], cell["tier"])
            references[key] = min(references.get(key, float("inf")), cell["loss"])
    for cell in cells:
        reference = references.get((cell["task"], cell["tier"]))
        cell["excess_pct"] = None if reference is None else 100 * (cell["loss"] / reference - 1)
        cell["competitive"] = bool(cell["complete"] and cell["excess_pct"] is not None
                                   and cell["excess_pct"] <= 5)
        cell["material_gap"] = bool(cell["complete"] and cell["excess_pct"] is not None
                                    and cell["excess_pct"] > 10)
        cell["pareto"] = False
    # All three axes have common engineering units. Native term counts and
    # leaf/coefficient units are deliberately excluded from this comparison.
    for cell in cells:
        if not cell["complete"] or cell.get("bytes") is None:
            continue
        peers = [p for p in cells if p["task"] == cell["task"] and p["tier"] == cell["tier"]
                 and p["complete"] and p.get("bytes") is not None]
        axes = ("loss", "seconds", "bytes")
        cell["pareto"] = not any(all(p[k] <= cell[k] for k in axes)
                                  and any(p[k] < cell[k] for k in axes) for p in peers)
    return cells


def table(records, title, subtitle, *, rowname=None, decimals=3, note="", better=None):
    data = pl.DataFrame(records, strict=False)
    obj = GT(data, rowname_col=rowname).tab_header(title=title, subtitle=subtitle)
    numeric = [n for n, t in data.schema.items() if t in (pl.Float64, pl.Float32)]
    if numeric:
        obj = obj.fmt_number(columns=numeric, decimals=decimals)
    if better:
        obj = bold_best(obj, data, better, decimals=decimals)
    if rowname:
        obj = obj.tab_stubhead(label="Method")
    if note:
        obj = obj.tab_source_note(source_note=note)
    return obj.as_raw_html()


REPORT_HEADER = """# Competitive modelling study: findings and gaps

Eight model families were measured on one frozen protocol: 720 predictive
workflows over ten real-data panels and three generated panels with known
response functions, plus 140 causal-effect workflows over four designs with
seven independent draws each. The [frozen protocol](competitive_shootout_protocol.md)
fixed the tasks, the candidate counts, the timing controls and the grading
thresholds before any outcome was read.

Two thresholds separate the results. A method is *competitive* on a panel when
its mean test loss is at most 5% above the best complete non-mboostrs
competitor on that panel, and a *material gap* when it exceeds 10%. These are
engineering acceptance thresholds, not significance tests: three splits with
overlapping training sets describe these panels, they do not support a claim
about a wider population.

## Summary

Read the first table row by row. One row is one dataset. One column is one
method. A cell says how much worse that method's test error is than the best
of the other libraries on that same dataset: "within 5%" means the difference
is too small to matter in practice, "+67%" means it is two thirds worse.
Compare methods down a column, never across a row, because each row has its
own response scale.

CatBoost is the most accurate method measured here. It is within 5% of the best
competitor on 8 of the 10 real datasets, and on the remaining 2 it is 17% and
31% behind. The mboostrs hybrid, which fits a smooth curve for each column and
then one tree that may use several columns at once, is the best mboostrs
configuration: within 5% on 2 datasets and 6% to 24% behind on the rest, for a
typical penalty of 8.4%. The mboostrs additive model, which fits one smooth
curve per column and no interactions, is typically 46.5% behind; adding the
single joint tree is what recovers most of that, so the missing piece was
interaction capacity rather than the settings search.

Neither mboostrs nor any competitor is best everywhere. CatBoost is behind on
the cancer dataset, where a linear score on 30 features is already enough, and
on the bike-rental forecasting panel, where EBM wins. Ridge wins nothing but
that cancer panel. EBM is within 5% on 4 of the 10 datasets and behind on 2.
These are screening results over thirteen datasets with at most five draws each;
they describe these datasets and do not rank the libraries in general.

## The dimensions that are not accuracy

Accuracy is one column of the trade-off. The second table records what each
method costs and how much of the data it looks at. CatBoost reads 9.8 of the 10
columns on a typical dataset; the mboostrs hybrid reads 1, because its joint
tree is the only retained term and that term can see every column at once. The
`columns actually used` count is an upper bound for any term with access to the
full table, so it measures how much of the table the retained specification
touches, not how much each column contributes.

Explanation time is the cost of one common workload: draw a curve per input
column. EBM is fastest at 0.027 s because it already stores per-feature terms.
mboostrs is second at 0.040 s for the additive model and 0.150 s for the hybrid.
CatBoost needs 0.592 s because a per-column curve requires partial-dependence
style integration across combinations. On a table with many categorical columns
the gap widens further.

Speed does not order the methods the way accuracy does. Ridge fits in 0.111 s
and is unusable on 8 datasets. EBM needs 39.7 s, more than CatBoost and thirty
times LightGBM, while being within 5% on only 4 datasets. LightGBM fits in
1.276 s and reaches a 6.8% median penalty. These are elapsed times from one
machine at one thread, not a controlled implementation speed ratio.

Uncertainty is uniform. Every method's 90% split-conformal interval covers
between 0.89 and 0.91 on the regression panels, so conformal calibration
is not a differentiator in this study; interval width is, and mboostrs hybrid
produces widths within about 10% of the tree boosters on most panels.

## What the study closed

The study found a representational gap in `effect_model` and it is now closed.
The public estimator compared a smooth dictionary against a univariate-stumps
dictionary, and its automatic mode picked the lower cross-validated R-loss.
Neither dictionary can represent a conditional effect that switches on within
the range of another modifier, so a panel whose effect is
`1 + 1.5 * sign(x0 * x1)` was estimated at PEHE 1.55 with no available
alternative.

`effect_model` now builds a third candidate, a depth-3 `btree` over all
modifiers pooled with `selection=sse`, and takes it only when its
cross-validated R-loss is at least 5% below the best additive candidate. On
seven confirmation draws (seeds 7 to 13), held out from the three draws used to
choose the depth, the panel that needs the interaction improved from PEHE 1.548
to 1.255 and selected the joint dictionary in 7 of 7 draws, while its
cross-validated advantage was 27% to 39%. The three panels that do not need an
interaction selected exactly the same dictionary as before in all 21 draws, and
their PEHE values are unchanged to three decimals.

Two limits on that result. The 5% selection margin was set after seeing these
confirmation draws, so the confirmation validates the regression control on the
other panels but is not an independent test of the margin itself. And a joint
tree remains a black box: it fixes the estimation failure without producing a
readable one-curve-per-modifier decomposition. The predictions improved; the
interpretability that motivated an additive dictionary in the first place did
not change.

## Prioritized gaps

1. **Representation: joint effects in `effect_model` — closed, with the
   interpretability limit above.** PEHE 1.548 to 1.255 on the joint panel.
2. **Prediction: hybrid stays about 8% above CatBoost on real panels.** Median
   excess loss is 8.4% and it holds material gaps on 4 panels, concentrated in
   the high-dimensional tabular panels (concrete, bike future) rather than in
   the low-dimensional ones. The joint tree closes the additive gap but not the
   distance to a strong GBDT. Acceptance test: median excess loss at or below
   5% on real panels with no panel above the 10% threshold.
3. **Search: autospec does not reach the capacity it already has.** Autospec is
   competitive on 2 panels with a 38.3% median excess loss at 11.0 seconds, while
   the hand-built hybrid reaches 8.4% at 6.9 seconds. It misses `joint_boundary`
   entirely (1.68 loss, a 314% gap) where hybrid is 0.676. Its own search never
   selects the representation that fixes that panel. Acceptance test: autospec's
   selected model on `joint_boundary` reaches the hybrid's loss.
4. **Search: autospec wall-clock budgeting.** Autospec costs roughly 100 seconds
   on `housing_iid` against 1.8 for hybrid, and all six of its superconductivity
   workflows hit the 240-second ceiling, which is also all six of its completed
   panels lost. Acceptance test: no autospec workflow exceeds the tuned-search
   budget of the method it is searching over, or reports why it cannot.
5. **Interpretability: no native exact decomposition for the joint tree.** A
   `btree` term is not a set of per-modifier curves, and the marginal-curve
   figures show flat curves for a model whose surface is correct. Acceptance
   test: a documented way to read a `btree` term's conditional effect.
6. **Process: the study cannot separate the additive failure modes.** The
   primary marginal-curve workload scored `mboost_additive` and Ridge
   identically on `joint_boundary` (9.14 loss each), because both predict flat
   marginals. The two-variable surface probe below separates them: additive
   scores surface NRMSE 1.000, indistinguishable from a constant prediction,
   where hybrid reaches 0.145. Any future study should grade a joint surface
   directly rather than infer it from marginal curves.

## Evidence

| Artifact | Contents |
| --- | --- |
| [competitive_shootout_protocol.md](competitive_shootout_protocol.md) | frozen design, thresholds, timing controls |
| [competitive_shootout_results.json](competitive_shootout_results.json) | 720 predictive workflows, per-job identities and digests |
| [competitive_shootout_causal.json](competitive_shootout_causal.json) | 140 causal-effect workflows |
| [competitive_shootout_causal_before.json](competitive_shootout_causal_before.json) | confirmation draws, pre-remediation |
| [competitive_shootout_causal_after.json](competitive_shootout_causal_after.json) | confirmation draws, post-remediation |
| [competitive_shootout_joint_surface.json](competitive_shootout_joint_surface.json) | exploratory two-variable surface probe |
| [competitive_shootout.html](competitive_shootout.html) | annotated report with figures |
| [competitive_shootout_scores.csv](competitive_shootout_scores.csv) | every score as one row |
| [competitive_shootout_summary.json](competitive_shootout_summary.json) | aggregated cells and headline counts |

Every artifact records the commit, the native build digest, the package and
extension byte hashes, and the thread environment, so a rerun can be checked
against the run that produced these numbers. One CPU model, Apple M4.
"""

def md_table(records, title, subtitle, *, rowname, decimals=3, note="", better=None):
    """GitHub-readable rendering of the same records `table` annotates.

    `competitive_shootout.html` is the annotated artifact, but GitHub drops
    inline styles from a raw .html file, so the report committed to the
    repository is markdown. Both renderings come from one record list.
    """
    if not records:
        return f"### {title}\n\n_No rows._\n"
    better = better or {}
    columns = [name for name in records[0] if name != rowname]
    numeric = [name for name in columns
               if all(r[name] is None or isinstance(r[name], (int, float)) and not isinstance(r[name], bool)
                      for r in records)]
    best = {}
    for name, direction in better.items():
        if name not in numeric:
            continue
        present = [r[name] for r in records if r[name] is not None]
        if not present:
            continue
        best[name] = min(present) if direction == "min" else max(present)

    def cell(name, value):
        if value is None:
            return "—"
        if isinstance(value, bool):
            return "yes" if value else "no"
        mark = "**" if name in best and value == best[name] else ""
        # Counts and term counts are integers; a fractional count reads as a
        # measurement that was not taken.
        if isinstance(value, int):
            return f"{mark}{value}{mark}"
        if isinstance(value, float):
            return f"{mark}{value:.{decimals}f}{mark}"
        return str(value)

    head = "| " + " | ".join([rowname, *columns]) + " |"
    rule = "| " + " | ".join(["---", *("---:" if n in numeric else "---" for n in columns)]) + " |"
    body = [f"| {r[rowname]} | " + " | ".join(cell(n, r[n]) for n in columns) + " |" for r in records]
    out = f"### {title}\n\n_{subtitle}_\n\n{head}\n{rule}\n" + "\n".join(body) + "\n"
    if note:
        out += f"\n{note}\n"
    return out


def validate_pairing(document):
    """Reject an incomplete run or different inputs being ranked as paired."""
    expected = document["protocol"]["specs"]
    if len(document["results"]) != len(expected):
        raise ValueError("the evidence does not account for every planned job")
    identities = defaultdict(set)
    for row in document["results"]:
        if row.get("_status") == "success":
            identities[(row["task"], row["seed"])].add((row["input_digest"], row["partition_digest"]))
            if len(row["results"]) != 1 or row["results"][0]["tier"] != row["tier"]:
                raise ValueError("a job returned the wrong search tier")
    if any(len(value) != 1 for value in identities.values()):
        raise ValueError("models did not receive identical inputs and partitions")


def figures(cells, raw_rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = dict(zip(LABELS, plt.get_cmap("tab10").colors, strict=False))
    synthetic = {"smooth_sparse", "joint_boundary", "correlated_proxies"}
    plots = []
    for task in ("housing_iid", "housing_spatial", "bike_future", "superconductivity"):
        subset = [c for c in cells if c["task"] == task and c["tier"] == "tuned" and c["complete"]]
        if not subset:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for c in subset:
            ax.scatter(c["seconds"], c["excess_pct"], color=colors[c["method"]], s=50)
            ax.annotate(LABELS[c["method"]], (c["seconds"], c["excess_pct"]),
                        xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax.axhline(5, color="grey", linestyle="--", label="5% competitiveness threshold")
        ax.set_xscale("log")
        ax.set(xlabel="Whole-workflow elapsed seconds (log scale; lower is better)",
               ylabel="Test loss above best non-mboostrs competitor (%)",
               title=f"{task.replace('_', ' ')}: quality and workflow cost\nThree paired splits; tuned search")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = DEST / f"shootout_{task}.svg"
        fig.savefig(path)
        plt.close(fig)
        plots.append(path.name)
    for task in sorted(synthetic - {"correlated_proxies"}):
        selected = [r for r in raw_rows if r.get("task") == task and r.get("seed") == 0
                    and r.get("tier") == "tuned" and r.get("_status") == "success"]
        if not selected:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
        truth_drawn = False
        for row in selected:
            e = row["results"][0]["explanation"]
            for j, ax in enumerate(axes):
                name = f"x{j}"
                curve = e["curves"].get(name)
                if curve is None:
                    continue
                ax.plot(curve["grid"], curve["effect"], label=LABELS[row["method"]],
                        color=colors[row["method"]], linewidth=1)
                if not truth_drawn and name in e.get("truth_curves", {}):
                    ax.plot(curve["grid"], e["truth_curves"][name], color="black",
                            linestyle="--", linewidth=2, label="Known response function")
                ax.set(xlabel=f"{name} (generated input units)",
                       ylabel="Centred mean response (response units)")
            truth_drawn = True
        axes[-1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
        fig.suptitle(f"{task.replace('_', ' ')}: common marginal-response curves, draw 0")
        fig.tight_layout()
        path = DEST / f"shootout_{task}_curves.svg"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        plots.append(path.name)
    return plots


SYNTHETIC_TASKS = {"smooth_sparse", "joint_boundary", "correlated_proxies"}


def penalty_matrix(cells):
    """One row per dataset: how far each method's loss sits above the best competitor.

    A per-dataset matrix answers the question a headline count cannot: on which
    datasets does a method lose, and by how much. `best` means within the 5%
    competitiveness threshold, not strictly the lowest loss, because a method
    3% behind is as usable as one 0% behind.
    """
    by_task = defaultdict(dict)
    for cell in cells:
        if cell["tier"] == "tuned":
            by_task[cell["task"]][cell["method"]] = cell
    methods = [m for m in LABELS if any(m in t for t in by_task.values())]
    records = []
    for task in sorted(by_task):
        complete = {m: c for m, c in by_task[task].items() if c["complete"]}
        reference = min((c["loss"] for m, c in complete.items()
                         if not m.startswith("mboost")), default=None)
        record = {"dataset": task.replace("_", " ")}
        for method in methods:
            cell = complete.get(method)
            if cell is None:
                record[LABELS[method]] = "incomplete"
            elif reference is None:
                record[LABELS[method]] = "—"
            else:
                pct = 100 * (cell["loss"] / reference - 1)
                record[LABELS[method]] = "within 5%" if pct <= 5 else f"+{pct:.0f}%"
        records.append(record)
    return records


def build_blocks():
    """Collect every section once, as data, for both output renderings."""
    document = json.loads((DEST / "competitive_shootout_results.json").read_text())
    causal = json.loads((DEST / "competitive_shootout_causal.json").read_text())
    surface_path = DEST / "competitive_shootout_joint_surface.json"
    surface = json.loads(surface_path.read_text()) if surface_path.exists() else None
    validate_pairing(document)
    points, failures = rows(document)
    cells = aggregate(points)
    plot_paths = figures(cells, document["results"])
    synthetic = {"smooth_sparse", "joint_boundary", "correlated_proxies"}
    real = [c for c in cells if c["task"] not in synthetic and c["tier"] == "tuned"]
    headline = []
    for method, label in LABELS.items():
        subset = [c for c in real if c["method"] == method and c["complete"]]
        headline.append({"method": label, "complete_tasks": len(subset),
                         "competitive_tasks": sum(c["competitive"] for c in subset),
                         "material_gaps": sum(c["material_gap"] for c in subset),
                         "median_excess_pct": float(np.median([c["excess_pct"] for c in subset])) if subset else None,
                         "median_workflow_s": float(np.median([c["seconds"] for c in subset])) if subset else None,
                         "median_explain_s": float(np.median([c["explain_seconds"] for c in subset])) if subset else None,
                         "median_used_inputs": float(np.median([c["used_inputs"] for c in subset])) if subset else None,
                         "median_pickle_MiB": float(np.median([c["bytes"] for c in subset])) / 1024**2 if subset else None})
    blocks = [("h1", "Competitive modelling study"),
              ("p", "Frozen protocol; one CPU thread; three paired outer splits on ten real-data panels, "
                    "five independent draws per effect-recovery panel, seven per causal panel. "
                    "All figures are descriptive screening evidence."),
              ("table", penalty_matrix(cells),
               dict(title="Accuracy on every dataset: how far behind the best competitor each method sits",
                    subtitle="Percentage is the excess test loss over the best complete non-mboostrs "
                             "method on that dataset; 'within 5%' counts as competitive",
                    rowname="dataset",
                    note="Read a row for one dataset and a column for one method. Generated panels "
                         "have five draws, real datasets three paired splits. 'incomplete' means at "
                         "least one split failed or timed out; the value is not inferred. '—' means no "
                         "competitor completed on that panel, so no reference exists. Datasets with "
                         "different response scales cannot be compared across rows, only within one. "
                         "Source: competitive_shootout_results.json.")),
              ("table", [{"method": h["method"],
                          "datasets_won_or_tied": h["competitive_tasks"],
                          "datasets_behind": h["material_gaps"],
                          "median_behind_pct": h["median_excess_pct"],
                          "fit_and_select_s": h["median_workflow_s"],
                          "explanations_s": h["median_explain_s"],
                          "columns_actually_used": h["median_used_inputs"],
                          "model_size_MiB": h["median_pickle_MiB"]} for h in headline],
               dict(title="The other dimensions: cost, size and how much of the table each method reads",
                    subtitle="10 real datasets; time and size are medians over completed datasets, "
                             "lower is better, more columns used is worse",
                    rowname="method",
                    better={"median_behind_pct": "min", "fit_and_select_s": "min",
                            "explanations_s": "min", "columns_actually_used": "min",
                            "model_size_MiB": "min"},
                    note="'Columns actually used' counts inputs entering retained terms, so a model "
                         "that ignores half the table scores lower. A joint tree can read every "
                         "column, so its count is an upper bound. Fit time covers searching over "
                         "settings, refitting, calibration and building explanations. Explanation "
                         "time is the cost of the common workload that draws a curve per column. "
                         "Model size is the stored model including retained training state, not the "
                         "count of mathematical parameters. These are engineering costs on one machine "
                         "at one thread, not a controlled implementation speed benchmark. "
                         "Source: competitive_shootout_results.json.")),
              ("table", headline,
               dict(title="Real-data comparison: tuned workflows",
                    subtitle="10 panels; lower loss, elapsed seconds and storage are better; "
                             "competitive = at most 5% excess loss",
                    rowname="method",
                    better={"competitive_tasks": "max", "median_excess_pct": "min",
                            "median_workflow_s": "min", "median_explain_s": "min"},
                    note="Reference: best complete non-mboostrs competitor on each panel. "
                         "Complete means all three splits succeeded. Medians condition on complete panels; "
                         "consult failures. Workflow includes search, refit, calibration, explanations and "
                         "storage reporting. Input counts are inputs of retained learners; joint-tree counts "
                         "are upper bounds. Pickle bytes include retained state, not just parameters. "
                         "Source: competitive_shootout_results.json.")),
              ("h2", "Quality, cost and response curves")]
    blocks += [("figure", p) for p in plot_paths]
    for tier in ("small", "tuned"):
        blocks.append(("h2", f"{tier.capitalize()} search: per-task results"))
        for task in dict.fromkeys(c["task"] for c in cells):
            selected = [c for c in cells if c["task"] == task and c["tier"] == tier]
            record = [{"method": LABELS[c["method"]], "loss": c["loss"], "loss_sd": c["loss_sd"],
                       "n": c["n"], "excess_pct": c["excess_pct"], "workflow_s": c["seconds"],
                       "explanation_s": c["explain_seconds"], "used_inputs": c["used_inputs"],
                       "terms": c["native_terms"], "peak_RSS_MiB": c["rss_mb"],
                       "quality_cost_storage_front": c["pareto"],
                       "coverage90": c["coverage90"], "width90_SD": c["width90_sd"]}
                      for c in selected]
            if record:
                blocks.append(("table", record,
                               dict(title=task.replace("_", " "),
                                    subtitle="MSE on the declared response scale, or binary log loss; lower is better",
                                    rowname="method", better={"loss": "min", "workflow_s": "min"},
                                    note="Loss SD is descriptive split spread, not an independent-sample SE. "
                                         "n reports successful splits/draws. Terms are method-specific native "
                                         "additive components; no term count is assigned to black-box ensembles. "
                                         "Coverage is empirical 90% predictive interval coverage; width is "
                                         "measured in training-response standard deviations. Small = first "
                                         "configuration; tuned = best of four, or the separately labelled "
                                         "autospec system.")))
    recovery = [{"task": c["task"], "method": LABELS[c["method"]], "curve_NRMSE": c["curve_nrmse"],
                 "group_recall_at3": c["recall_at3"], "used_inputs": c["used_inputs"],
                 "native_terms": c["native_terms"], "native_decomp_s": c["native_explain_seconds"],
                 "workflow_s": c["seconds"]} for c in cells if c["task"] in synthetic and c["tier"] == "tuned"]
    blocks += [("h2", "Known effects and explanations"),
               ("table", recovery,
                dict(title="Effect recovery on independent generated data",
                     subtitle="Five draws; lower normalized curve RMSE and seconds are better; "
                              "higher driver-group recall is better",
                     rowname="method",
                     note="Curve error compares the same marginal-response workload against evaluator-only "
                          "truth. It is unavailable for correlated proxies because individual interventions "
                          "leave joint support. Correlated inputs are permuted together for group recovery. "
                          "Exact native decompositions also exist for the tree boosters via SHAP; "
                          "reconstruction alone does not establish a causal explanation."))]
    causal_groups = defaultdict(list)
    for row in causal["results"]:
        if row.get("_status") == "success":
            causal_groups[(row["scenario"], row["method"])].append(row)
    causal_table = []
    for (scenario, method), values in causal_groups.items():
        def avg(key, values=values):
            v = [r["scores"][key] for r in values if r["scores"][key] is not None]
            return float(np.mean(v)) if v else None
        causal_table.append({"scenario": scenario, "method": method, "n": len(values),
                             "PEHE": avg("pehe"), "ATE_bias": avg("ate_error"),
                             "ATE_RMSE": float(np.sqrt(np.mean([r["scores"]["ate_error"]**2 for r in values]))),
                             "ATE_coverage95": avg("ate_covered95"),
                             "workflow_s": float(np.mean([r["seconds"] for r in values]))})
    blocks += [("h2", "Causal-effect screening"),
               ("table", causal_table,
                dict(title="Causal effects under declared adjustment designs",
                     subtitle="Seven independent draws; lower PEHE, absolute bias, RMSE and seconds are better",
                     rowname="method", better={"PEHE": "min"},
                     note="PEHE is root mean squared conditional-effect error; ATE is population average "
                          "effect, truth = 1. Only AIPW and grf provide the reported ATE intervals. "
                          "Missing entries are unsupported estimands, not zero errors. The "
                          "omitted-confounder panel is deliberately unidentified. Seven draws are "
                          "insufficient to establish precise interval calibration. "
                          "Source: competitive_shootout_causal.json."))]
    if surface is not None:
        probe = [{"method": LABELS.get(r["method"], r["method"]),
                  "surface_NRMSE": r["surface_nrmse"],
                  "predicted_surface_min": r["predicted_surface_range"][0],
                  "predicted_surface_max": r["predicted_surface_range"][1],
                  "exploratory": r["exploratory"]}
                 for r in surface["results"] if r.get("_status") == "success"]
        truth_range = next(r["truth_surface_range"] for r in surface["results"]
                           if r.get("_status") == "success")
        blocks += [("h2", "Joint-threshold surface probe"),
                   ("table", probe,
                    dict(title="Two-variable response surface at draw 0",
                         subtitle=f"Truth spans [{truth_range[0]:.2f}, {truth_range[1]:.2f}]; lower "
                                  "normalized surface RMSE is better; predicted range shows how much "
                                  "signal the model holds",
                         rowname="method", better={"surface_NRMSE": "min"},
                         note="Exploratory follow-up on one draw, graded on the two-variable surface the "
                              "primary marginal curves cannot see. The primary joint_boundary loss ranks "
                              "mboost_additive and linear equally because both predict flat marginals, so "
                              "it cannot separate them. Truth range is a constant of the task and is "
                              "identical for every method. "
                              "Source: competitive_shootout_joint_surface.json."))]
        probe_failures = [r for r in surface["results"] if r.get("_status") != "success"]
        if probe_failures:
            blocks.append(("table", [{"method": str(r.get("job")), "detail": str(r.get("failure"))[:400]}
                                     for r in probe_failures],
                           dict(title="Failed surface probe jobs",
                                subtitle="No failed method is silently removed", rowname="method")))
    fail_table = []
    causal_failures = [r for r in causal["results"] if r.get("_status") != "success"]
    for row in failures + causal_failures:
        job = row.get("job", [])
        fail_table.append({"task": job[0] if job else "unknown", "seed": job[1] if job else None,
                           "method": job[2] if job else "unknown", "tier": job[3] if len(job) > 3 else "unknown",
                           "status": row.get("_status"), "detail": str(row.get("failure", row.get("error")))[:700]})
    blocks.append(("h2", "Failures and provenance"))
    blocks.append(("table", fail_table,
                   dict(title="All failed workflows",
                        subtitle="No failed method is silently removed", rowname="task"))
                  if fail_table else ("p", "No workflow failed."))
    provenance = {k: document[k] for k in
                  ("commit", "build_info", "platform", "competitor_versions", "thread_environment")}
    blocks += [("p", "Full inputs, partitions, code, native-build and prediction digests are retained in "
                    "[predictive evidence](competitive_shootout_results.json), "
                    "[causal evidence](competitive_shootout_causal.json) and "
                    "[surface probe evidence](competitive_shootout_joint_surface.json)."),
               ("pre", json.dumps(provenance, indent=2))]
    return blocks, points, cells, headline, causal_table, failures


def remediation_blocks():
    """The confirmation study for the `effect_model` joint-dictionary change.

    Reported only when both confirmation artifacts are present, so the report
    never claims a remediation without the numbers behind it.
    """
    before_path = DEST / "competitive_shootout_causal_before.json"
    after_path = DEST / "competitive_shootout_causal_after.json"
    if not (before_path.exists() and after_path.exists()):
        return []
    def summarize(path):
        rows = [r for r in json.loads(path.read_text())["results"]
                if r.get("_status") == "success"]
        picks, pehe, advantage = defaultdict(Counter), defaultdict(lambda: defaultdict(list)), defaultdict(list)
        for r in rows:
            picks[r["scenario"]][r["terms"]] += 1
            pehe[r["scenario"]][r["terms"]].append(r["pehe"])
            cv = r.get("cv_rloss") or {}
            if "joint" in cv and len(cv) > 1:
                best_additive = min(v for k, v in cv.items() if k != "joint")
                advantage[r["scenario"]].append(1 - cv["joint"] / best_additive)
        return picks, pehe, advantage
    before = summarize(before_path)
    after = summarize(after_path)
    records = []
    for scenario in before[0]:
        record = {"scenario": scenario}
        for label, index in (("before", 0), ("after", 1)):
            picks, pehe, _ = (before, after)[index]
            chosen = picks[scenario].most_common(1)[0][0]
            record[f"{label}_dictionary"] = ", ".join(f"{t} x{n}" for t, n in picks[scenario].most_common())
            record[f"{label}_PEHE"] = float(np.sqrt(np.mean(np.square(pehe[scenario][chosen]))))
        adv = after[2][scenario]
        record["joint_cv_advantage_pct"] = (f"{min(adv)*100:.1f} to {max(adv)*100:.1f}"
                                           if adv else "not a candidate")
        records.append(record)
    return [("h2", "Joint-dictionary remediation: confirmation on held-out draws"),
            ("table", records,
             dict(title="effect_model automatic candidate selection, before and after",
                  subtitle="Seven draws (seeds 7 to 13) not used to choose the tree depth; "
                           "lower PEHE is better",
                  rowname="scenario", better={"after_PEHE": "min"},
                  note="The dictionary column counts how often each candidate was selected. PEHE is "
                       "root mean squared conditional-effect error for the selected dictionary, so the "
                       "before and after values are directly comparable. Joint CV advantage is how much "
                       "lower the joint candidate's cross-validated R-loss was than the best additive "
                       "candidate's; where it is absent, the joint candidate was not in the pool. "
                       "The 5 percent selection margin was fixed after these draws were seen, so these "
                       "draws confirm that the other panels keep their dictionary, and do not "
                       "independently validate the margin itself. "
                       "Source: competitive_shootout_causal_before.json and "
                       "competitive_shootout_causal_after.json."))]


def render():
    blocks, points, cells, headline, causal_table, failures = build_blocks()
    sections = []
    for block in blocks + remediation_blocks():
        kind = block[0]
        if kind in ("h1", "h2", "p", "pre"):
            tag = {"h1": "h1", "h2": "h2", "p": "p", "pre": "pre"}[kind]
            sections.append(f"<{tag}>{html.escape(block[1])}</{tag}>")
        elif kind == "figure":
            sections.append(f"<figure><img src='{block[1]}' style='max-width:100%' alt='{html.escape(block[1])}'>"
                            "<figcaption>Source: frozen study artifact. Curves describe predictions, not "
                            "identified interventions; a one-variable curve can conceal a joint effect."
                            "</figcaption></figure>")
        else:
            sections.append(table(block[1], **block[2]))
    (DEST / "competitive_shootout.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Competitive modelling study</title>"
        "<style>body{font:16px system-ui,sans-serif;max-width:1450px;margin:auto;padding:24px}"
        "pre{white-space:pre-wrap}h2{margin-top:3rem}</style>"
        + html.escape("<h1>Competitive modelling study</h1>") + "\n"
        + "<p>Findings, closed gaps and priorities: see the markdown report "
          "<a href='competitive_shootout_report.md'>competitive_shootout_report.md</a>. "
          "The tables and figures below are the evidence behind it.</p>"
        + "\n".join(sections))
    markdown = [REPORT_HEADER,
                "\nThe tables below are generated from the artifacts, so they cannot drift from the "
                "measurements. The annotated HTML rendering, with the great_tables spanners, notes and "
                "figures in place, is [competitive_shootout.html](competitive_shootout.html).\n"]
    for block in blocks + remediation_blocks():
        kind = block[0]
        if kind == "h1":
            continue
        if kind == "h2":
            markdown.append(f"\n## {block[1]}\n")
        elif kind == "p":
            markdown.append(f"\n{block[1]}\n")
        elif kind == "pre":
            markdown.append("\n```json\n" + block[1] + "\n```\n")
        elif kind == "figure":
            markdown.append(f"\n![{block[1]}]({block[1]})\n")
        else:
            markdown.append("\n" + md_table(block[1], **block[2]))
    (DEST / "competitive_shootout_report.md").write_text("".join(markdown))
    (DEST / "competitive_shootout_summary.json").write_text(json.dumps({"headline": headline,
            "cells": cells, "causal": causal_table, "failure_count": len(failures)}, indent=1, allow_nan=False) + "\n")
    # A score column can be absent for every early task and present later, so
    # the whole column has to be read before the types are fixed.
    pl.DataFrame(points, strict=False,
                 infer_schema_length=None).write_csv(DEST / "competitive_shootout_scores.csv")


if __name__ == "__main__":
    render()
