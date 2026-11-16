"""Versioned stopping-rule ablation for issue #509.

The historical shootout uses single-split cvrisk(preprocessing="refit") at
1200 iterations. Version 2 preserves that baseline, adds a matched native
fixed-preprocessing baseline for growth/patience, and compares fold-local
fixed/growing/structured-validation routes for deployment-facing conclusions.
The native family loss is MSE for Gaussian and log loss for Binomial;
fold-local routes use MSE/log loss through the public Rust-backed score path.
Only comparisons within a route isolate stopping. Patience is unsupported by
the fold-local API and remains a native-route diagnostic.

IID validation uses repeated K-fold; forecasting uses whole-time-unit rolling
origins; spatial/grouped panels keep dependency units together. Calibration
and test frames are scored after development-only selection. Paired assessment
seeds, rather than inner folds, are the descriptive replication units. No
public defaults change, and outer losses never choose a rule or a regime.

Prepare: python -m benches.stopping_ablation_509 --prepare
Run: OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     python -m benches.stopping_ablation_509 --run --workers 3
Report: python -m benches.stopping_ablation_509 --report
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import json
import math
import resource
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benches._jobs import run_jobs  # noqa: E402
from benches.benchmark_contract import data_digest  # noqa: E402
from benches.shootout_data import CACHE, TASKS, payload, prepare  # noqa: E402
from benches.shootout_models import (
    GRIDS,
    Fitted,
    formula,  # noqa: E402
    loss as task_loss,
)
from benches.study_evidence import evidence_document, write_evidence  # noqa: E402

ARCHIVE_DIR = ROOT / "docs/benchmarks/stopping_ablation_509"
OUT_DIR = ARCHIVE_DIR / "v2"
OUT = OUT_DIR / "results.json.gz"
REPORT = OUT_DIR / "README.md"

#: The status-quo fixed budget; also the first grid of every growing rule.
BUDGET = 1200
#: The largest grid the growing rules may reach; a still-falling minimum there
#: is recorded as an unresolved boundary rather than grown past.
CAP = 2400
#: The prespecified patience window, matching the competitors' early stopping.
PATIENCE = 60
KFOLD, REPEATS = 5, 2
FOLD_SEED_OFFSET = 555
EXPERIMENT_VERSION = "stopping-ablation/2"
RULES = (
    "fixed_argmin",
    "fixed_preprocessing",
    "boundary_growth",
    "patience_60",
    "repeated_cv_growth",
    "fold_local_fixed",
    "fold_local_growth",
    "fold_local_cv_growth",
)
COMPARISONS = (
    ("preprocessing", "fixed_preprocessing", "fixed_argmin"),
    ("native_growth", "boundary_growth", "fixed_preprocessing"),
    ("native_patience", "patience_60", "boundary_growth"),
    ("native_validation", "repeated_cv_growth", "boundary_growth"),
    ("fold_local_growth", "fold_local_growth", "fold_local_fixed"),
    ("fold_local_validation", "fold_local_cv_growth", "fold_local_growth"),
)
DEFAULT_TASKS = ("cancer", "bike_iid", "bike_future", "concrete", "joint_boundary", "smooth_sparse", "housing_iid")
DEFAULT_METHODS = ("mboost_hybrid", "mboost_additive")
THREADS = {
    k: "1"
    for k in (
        "RAYON_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
}


def _float(value):
    """A JSON-safe float: non-finite values become None, never NaN."""
    value = float(value)
    return value if math.isfinite(value) else None


def _curve(values):
    """A risk curve as a JSON list, kept finite and compact.

    Six significant digits keep the versioned artifact small while leaving
    the curve's shape, minima and boundary band fully legible.
    """
    out = []
    for value in np.asarray(values, dtype=np.float64).ravel():
        out.append(None if not math.isfinite(value) else float(f"{value:.6g}"))
    return out


def _captured(call, *args, **kwargs):
    """Run `call`, returning its result and the warnings it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = call(*args, **kwargs)
    messages = [f"{entry.category.__name__}: {entry.message}" for entry in caught]
    return result, messages


def read_run(path=OUT):
    """Read the full, losslessly compressed evidence document."""
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def compress_run(source, target=OUT):
    """Compress the exact JSON bytes, retaining a digest of the original."""
    source, target = Path(source), Path(target)
    temporary = target.with_suffix(target.suffix + ".tmp")
    digest = hashlib.sha256()
    with source.open("rb") as original, temporary.open("wb") as destination:
        # A stable gzip header makes publication deterministic.
        with gzip.GzipFile(filename="", mode="wb", fileobj=destination, mtime=0) as compressed:
            while block := original.read(1024 * 1024):
                digest.update(block)
                compressed.write(block)
    temporary.replace(target)
    write_evidence(
        {
            "file": target.name,
            "compression": "gzip",
            "uncompressed_sha256": digest.hexdigest(),
            "uncompressed_bytes": source.stat().st_size,
        },
        target.parent / "manifest.json",
    )


def write_run(document):
    """Write complete JSON evidence, then publish its lossless compressed copy."""
    original = OUT.with_suffix("")
    write_evidence(document, original)
    compress_run(original)
    original.unlink()


def stopping_runs(rhs, dev, family, nu, split_weights, repeated_scheme, *, budget=BUDGET, cap=CAP, patience=PATIENCE):
    """Every rule's mstop choice for one configuration, with its evidence.

    Rules share the specification and development rows. Their preprocessing,
    validation schemes and grid limits are explicit in each record.
    Each record carries the chosen iteration (never 0), the aggregate risk
    curve over the evaluated grid, the boundary state, the config-selection
    score at the chosen iteration, the runtime, the model fitted at the chosen
    iteration, and any boundary or flat-curve warning the library raised.
    """
    import mboostrs as mb
    from mboostrs.cvrisk import is_at_boundary

    records = {}

    for rule, preprocessing in (("fixed_argmin", "refit"), ("fixed_preprocessing", "fixed")):
        begin = time.perf_counter()
        model = mb.gamboost(rhs, dev, family=family, control=mb.boost_control(int(budget), nu=nu))
        cv, messages = _captured(model.cvrisk, folds=split_weights, preprocessing=preprocessing)
        stop = int(cv.optimal_mstop)
        records[rule] = {
            "mstop": stop,
            "at_boundary": bool(is_at_boundary(stop, budget)),
            "grid_ends": [int(budget)],
            "largest": int(budget),
            "aggregate": _curve(cv.aggregate),
            "fold_risk_curves": [_curve(row) for row in cv.risk_matrix],
            "fold_scores": _curve(np.asarray(cv.risk_matrix)[:, stop]),
            "selection_loss": _float(cv.aggregate[stop]),
            "fold_fits": 1,
            "total_fits": 2,
            "extended": False,
            "seconds": time.perf_counter() - begin,
            "warnings": messages,
            "model": model.with_mstop(stop),
            "preprocessing": preprocessing,
            "score": "family",
            "validation": "single_split",
        }

    response_score = "log_loss" if type(family).__name__ == "Binomial" else "mse"
    routes = (
        ("boundary_growth", split_weights, None, False, cap),
        ("patience_60", split_weights, int(patience), False, cap),
        ("repeated_cv_growth", repeated_scheme, None, False, cap),
        ("fold_local_fixed", split_weights, None, True, budget),
        ("fold_local_growth", split_weights, None, True, cap),
        ("fold_local_cv_growth", repeated_scheme, None, True, cap),
    )
    for rule, folds, rule_patience, fold_local, limit in routes:
        begin = time.perf_counter()
        tuning, messages = _captured(
            mb.tune_mstop,
            rhs,
            dev,
            family=family,
            folds=folds,
            start=int(budget),
            max_mstop=int(limit),
            nu=nu,
            patience=rule_patience,
            fold_local=fold_local,
            score=response_score if fold_local else "family",
            unseen_levels="zero_effect",
            extrapolation="allow",
        )
        fold_values = np.asarray(tuning.fold_scores)
        full_curves = fold_values.ndim == 2
        records[rule] = {
            "mstop": int(tuning.mstop),
            "at_boundary": bool(tuning.at_boundary),
            "grid_ends": [int(end) for end in tuning.grid_ends],
            "largest": int(tuning.largest),
            "aggregate": _curve(tuning.risk),
            "fold_scores": _curve(fold_values[:, tuning.mstop] if full_curves else fold_values),
            "fold_risk_curves": ([_curve(row) for row in fold_values] if full_curves else None),
            "selection_loss": _float(tuning.risk[tuning.mstop]),
            "fold_fits": int(tuning.fits),
            "total_fits": int(tuning.total_fits),
            "extended": bool(tuning.extended),
            "seconds": time.perf_counter() - begin,
            "warnings": messages,
            "model": tuning.model,
            "preprocessing": "fold_local" if fold_local else "fixed",
            "score": response_score if fold_local else "family",
            "validation": "single_split" if folds is split_weights else repeated_scheme.scheme,
        }
    return records


def configuration_set(
    method,
    frames,
    cats,
    kind,
    seed,
    *,
    budget=BUDGET,
    cap=CAP,
    patience=PATIENCE,
    k=KFOLD,
    repeats=REPEATS,
    design="iid",
    groups=None,
):
    """Run every configuration of one method grid under every rule.

    Returns per-configuration rule records (still holding fitted models) plus,
    per rule, the tier choices the rule's own development-side score makes:
    the small tier is the grid's first configuration, the tuned tier is the
    configuration whose selected-iteration score is lowest. Ties keep the
    lowest configuration index, matching the shootout's `min` selection.
    """
    train, val, cal, test = frames
    dev = pl.concat([train, val])
    split_weights = np.r_[np.ones(len(train)), np.zeros(len(val))][:, None]
    scheme = validation_scheme(len(dev), seed, design=design, groups=groups, k=k, repeats=repeats)
    names = [c for c in dev.columns if c != "y"]
    levels = {c: sorted(dev[c].unique().to_list()) for c in cats}
    outputs = []
    for index, config in enumerate(GRIDS[method]):
        rhs, terms = formula(dev, cats, config["df"], config.get("depth"))
        records = stopping_runs(
            rhs, dev, _family(kind), config["nu"], split_weights, scheme, budget=budget, cap=cap, patience=patience
        )
        for record in records.values():
            wrapped = Fitted(record.pop("model"), method, kind, cats, names, levels, dict(config), terms)
            record["cal_loss"] = _float(task_loss(cal["y"].to_numpy(), wrapped.predict(cal), kind))
            record["test_loss"] = _float(task_loss(test["y"].to_numpy(), wrapped.predict(test), kind))
        outputs.append({"method": method, "config_index": index, "config": _plain_config(config), "rules": records})
    tiers = {}
    for rule in RULES:
        scores = [row["rules"][rule]["selection_loss"] for row in outputs]
        chosen = min(range(len(scores)), key=lambda i: (scores[i] is None, scores[i] if scores[i] is not None else 0.0))
        tiers[rule] = {"small": 0, "tuned": int(chosen)}
    return outputs, tiers, scheme


def validation_scheme(n, seed, *, design="iid", groups=None, k=KFOLD, repeats=REPEATS):
    """Dependency-matched inner folds, keeping horizon masks and time units intact."""
    from mboostrs.folds import FoldScheme, grouped_folds, repeated_kfold_folds, rolling_folds

    if design == "iid":
        return repeated_kfold_folds(n, k=int(k), repeats=int(repeats), seed=int(seed) + FOLD_SEED_OFFSET)
    if groups is None or np.asarray(groups).shape != (n,):
        raise ValueError(f"{design} validation needs one dependency unit per development row")
    groups = np.asarray(groups)
    if design in ("spatial", "grouped"):
        return grouped_folds(groups, B=int(k), repeats=int(repeats), seed=int(seed) + FOLD_SEED_OFFSET)
    if design != "temporal":
        raise ValueError(f"unsupported dependency design: {design}")
    # Fold whole days rather than rows: adjacent hourly observations on the
    # same date must never straddle a training/forecast boundary.
    levels, inverse = np.unique(groups, return_inverse=True)
    origins = rolling_folds(len(levels), B=int(k) * int(repeats))
    held = origins.held_out[inverse]
    return FoldScheme(
        weights=origins.weights[inverse],
        held_out=held,
        judge=held,
        scheme=origins.scheme + "; whole time units",
        ordered=True,
        groups=groups,
        times=groups,
        base_weights=np.ones(n),
    )


def fold_manifest(scheme):
    """Serializable membership, including rows neither trained on nor judged."""
    return {
        "scheme": scheme.scheme,
        "seed": scheme.seed,
        "ordered": scheme.ordered,
        "repeats": scheme.repeats,
        "groups": None if scheme.groups is None else scheme.groups.astype(str).tolist(),
        "times": None if scheme.times is None else scheme.times.astype(str).tolist(),
        "folds": [
            {
                "training_rows": np.flatnonzero(scheme.weights[:, b] > 0).tolist(),
                "training_weights": scheme.weights[scheme.weights[:, b] > 0, b].tolist(),
                "held_out_rows": np.flatnonzero(scheme.held_out[:, b]).tolist(),
                "judge_rows": np.flatnonzero(
                    scheme.held_out[:, b] if scheme.judge is None else scheme.judge[:, b]
                ).tolist(),
            }
            for b in range(scheme.weights.shape[1])
        ],
    }


def _family(kind):
    import mboostrs as mb

    return mb.Binomial() if kind == "binary" else mb.Gaussian()


def _plain_config(config):
    return {key: (float(value) if isinstance(value, float) else value) for key, value in config.items()}


def run_job(spec):
    """One task/draw: every method grid, every configuration, every rule."""
    task, seed, methods, source_hash = spec[0], int(spec[1]), spec[2], spec[3]
    import mboostrs  # noqa: F401

    data = payload(task, seed)
    train, val, cal, test = data["frames"]
    cats, kind = data["categorical"], data["kind"]
    begin = time.perf_counter()
    method_results = []
    scheme = None
    design = TASKS[task][1]
    auxiliary_groups = data["metadata"].get("auxiliary", {}).get("groups")
    groups = None
    if design != "iid" and auxiliary_groups is not None:
        groups = np.asarray(auxiliary_groups)[data["indices"][0] + data["indices"][1]]
        if design == "temporal":
            groups = groups.astype("datetime64[D]")
    for method in methods.split(","):
        outputs, tiers, scheme = configuration_set(
            method, data["frames"], cats, kind, seed, design=design, groups=groups
        )
        method_results.append(
            {
                "method": method,
                "configurations": outputs,
                "tiers": {
                    rule: {
                        "small": {
                            "config_index": tiers[rule]["small"],
                            "cal_loss": outputs[tiers[rule]["small"]]["rules"][rule]["cal_loss"],
                            "test_loss": outputs[tiers[rule]["small"]]["rules"][rule]["test_loss"],
                            "mstop": outputs[tiers[rule]["small"]]["rules"][rule]["mstop"],
                        },
                        "tuned": {
                            "config_index": tiers[rule]["tuned"],
                            "cal_loss": outputs[tiers[rule]["tuned"]]["rules"][rule]["cal_loss"],
                            "test_loss": outputs[tiers[rule]["tuned"]]["rules"][rule]["test_loss"],
                            "mstop": outputs[tiers[rule]["tuned"]]["rules"][rule]["mstop"],
                        },
                    }
                    for rule in RULES
                },
            }
        )
    assert scheme is not None
    return {
        "task": task,
        "seed": seed,
        "kind": kind,
        "rules": list(RULES),
        "experiment_version": EXPERIMENT_VERSION,
        "design": design,
        "budget": BUDGET,
        "cap": CAP,
        "patience": PATIENCE,
        "repeated_folds": {"k": KFOLD, "repeats": REPEATS, "fold_seed": seed + FOLD_SEED_OFFSET},
        "source_hash": source_hash,
        "input_digest": data_digest(
            {"columns": train.columns, "frames": [f.to_dict(as_series=False) for f in data["frames"]]}
        ),
        "partition_digest": data_digest(data["indices"]),
        "fold_scheme": fold_manifest(scheme),
        "fold_scheme_digest": data_digest(fold_manifest(scheme)),
        "single_split": {
            "training_rows": list(range(len(train))),
            "judge_rows": list(range(len(train), len(train) + len(val))),
        },
        "missing_cells": int(sum(f[c].null_count() for f in (train, val) for c in f.columns)),
        "rows": {k: len(f) for k, f in zip(("fit", "validation", "calibration", "test"), data["frames"], strict=True)},
        "method_results": method_results,
        "job_seconds": time.perf_counter() - begin,
        "peak_rss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        ),
    }


def paired_differences(rows, reference="fixed_argmin", rules=None):
    """Pair rule and reference losses within task, seed, method and tier.

    A positive difference means the rule has higher assessment loss. Relative
    change uses the reference loss as denominator; a zero denominator is null.
    """
    out = []
    for row in rows:
        for method in row["method_results"]:
            if reference not in method["tiers"]:
                continue
            for tier in ("small", "tuned"):
                base = method["tiers"][reference][tier]
                for rule in row["rules"] if rules is None else rules:
                    if rule == reference or rule not in method["tiers"]:
                        continue
                    side = method["tiers"][rule][tier]
                    loss, baseline = side["test_loss"], base["test_loss"]
                    out.append(
                        {
                            "task": row["task"],
                            "seed": row["seed"],
                            "method": method["method"],
                            "tier": tier,
                            "rule": rule,
                            "reference": reference,
                            "test_loss": loss,
                            "paired_test_loss": baseline,
                            "difference": None if loss is None or baseline is None else loss - baseline,
                            "relative": None if loss is None or baseline in (None, 0.0) else loss / baseline - 1,
                            "mstop": side["mstop"],
                            "fixed_mstop": base["mstop"],
                            "config_index": side["config_index"],
                            "fixed_config_index": base["config_index"],
                        }
                    )
    return out


def configuration_differences(rows):
    """Tuned minus first-configuration loss under each rule, including harms."""
    out = []
    for row in rows:
        for method in row["method_results"]:
            for rule, tiers in method["tiers"].items():
                small, tuned = tiers["small"], tiers["tuned"]
                loss, baseline = tuned["test_loss"], small["test_loss"]
                out.append(
                    {
                        "task": row["task"],
                        "seed": row["seed"],
                        "method": method["method"],
                        "tier": "tuned vs small",
                        "rule": rule,
                        "reference": "small",
                        "test_loss": loss,
                        "paired_test_loss": baseline,
                        "difference": None if loss is None or baseline is None else loss - baseline,
                        "relative": None if loss is None or baseline in (None, 0.0) else loss / baseline - 1,
                    }
                )
    return out


def grouped_summary(pairs):
    """Descriptive medians, full ranges and worst-seed regressions; no CIs."""
    keys = ("task", "method", "tier", "rule", "reference")
    groups = {}
    for pair in pairs:
        groups.setdefault(tuple(pair[k] for k in keys), []).append(pair)
    out = []
    for key, records in groups.items():
        valid = [r for r in records if r["relative"] is not None]
        worst = max(valid, key=lambda r: r["relative"]) if valid else None
        values = [r["relative"] for r in valid]
        out.append(
            {
                **dict(zip(keys, key, strict=True)),
                "n": len(records),
                "n_finite": len(valid),
                "median_relative": None if not values else float(np.median(values)),
                "min_relative": None if not values else min(values),
                "max_relative": None if not values else max(values),
                "worst_seed": None if worst is None else worst["seed"],
                "worst_reference_loss": None if worst is None else worst["paired_test_loss"],
                "worst_test_loss": None if worst is None else worst["test_loss"],
                "seeds": [r["seed"] for r in records],
            }
        )
    return out


def summary_document(document):
    rows = [r for r in document["results"] if "method_results" in r]
    comparisons = {}
    for label, rule, reference in COMPARISONS:
        pairs = paired_differences(rows, reference=reference, rules=[rule])
        comparisons[label] = {"pairs": pairs, "summary": grouped_summary(pairs)}
    configurations = configuration_differences(rows)
    historical = paired_differences(rows)
    return {
        "experiment_version": document.get("experiment_version", "stopping-ablation/1"),
        "relative_denominator": "reference test loss within the same task/seed/method/tier",
        "interpretation": "descriptive paired assessment seeds; no confidence intervals",
        "comparisons": comparisons,
        "historical": {"pairs": historical, "summary": grouped_summary(historical)},
        "configuration_selection": {"pairs": configurations, "summary": grouped_summary(configurations)},
    }


def _percent(value):
    return "null" if value is None else f"{value:+.1%}"


def _summary_table(records):
    lines = [
        "| panel | method | tier | rule / reference | finite seeds / all | median change | range | worst seed | reference loss → rule loss at worst seed |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        losses = (
            "null"
            if r["worst_reference_loss"] is None
            else f"{r['worst_reference_loss']:.6g} → {r['worst_test_loss']:.6g}"
        )
        lines.append(
            f"| {r['task']} | {r['method']} | {r['tier']} | {r['rule']} / {r['reference']}"
            f" | {r['n_finite']}/{r['n']} | {_percent(r['median_relative'])}"
            f" | {_percent(r['min_relative'])} to {_percent(r['max_relative'])}"
            f" | {r['worst_seed']} | {losses} |"
        )
    return lines + [""]


def summarize(document):
    """Generate every numeric summary from the recorded assessment losses."""
    summary = summary_document(document)
    rows = [r for r in document["results"] if "method_results" in r]
    lines = [
        "# Stopping-rule ablation (issue #509)",
        "",
        f"Version `{summary['experiment_version']}`. Produced {document['created_at']}"
        f" from commit `{document['commit']}`. [Full results](results.json.gz),"
        " [machine-readable summaries](summary.json).",
        "",
        "This run replaces the stopping conclusions of the October 12 v1 run."
        " The original results are retained unchanged in [the archive](../results.json);"
        " its claims are audited in [the correction report](../audit_v1.md).",
        "",
        "Eight prespecified rules use the same development rows and first budget."
        " The historical `fixed_argmin` uses refitted preprocessing."
        " `fixed_preprocessing` supplies the matched native baseline for"
        " `boundary_growth` and `patience_60`. Deployment-facing comparisons"
        " use `fold_local_fixed`, `fold_local_growth`, and `fold_local_cv_growth`,"
        " whose recipes are fitted on each fold's training rows.",
        "",
        "Negative change favors the rule. Every percentage divides by the named"
        " reference's assessment loss on the same panel, method, tier and seed."
        " Each table reports the median of those paired percentages, their full"
        " range and the seed with the largest increase. These are descriptive"
        " summaries, with no confidence intervals. Real-data assessment splits"
        " overlap and temporal origins share training history; they are not"
        " independent experiments. Synthetic seeds are independent draws."
        " Calibration/test scores never choose the stopping rule or configuration.",
        "",
    ]
    labels = {
        "preprocessing": "Preprocessing routes",
        "native_growth": "Native budget growth",
        "native_patience": "Native patience",
        "native_validation": "Native validation schemes",
        "fold_local_growth": "Fold-local budget growth",
        "fold_local_validation": "Fold-local validation schemes",
    }
    for name, contents in summary["comparisons"].items():
        lines += [f"## {labels[name]}", ""]
        if name.startswith("native") or name == "preprocessing":
            lines += [
                "Native fixed-preprocessing comparisons are diagnostic."
                " Their held-out rows participate in basis construction;"
                " deployment conclusions use the fold-local tables below.",
                "",
            ]
        lines += _summary_table(contents["summary"])
    lines += [
        "## Configuration selection",
        "",
        "The tuned configuration minimizes development-side risk among four"
        " candidates. The small tier uses the first candidate. Positive"
        " assessment changes show where configuration selection was harmful.",
        "",
    ]
    lines += _summary_table(summary["configuration_selection"]["summary"])
    lines += [
        "## Boundaries and validation",
        "",
        "A boundary flag means the minimum remains in the library's boundary"
        " band; reaching the cap leaves it unresolved. Counts refer to the"
        " configuration selected for that tier, over the displayed seeds.",
        "",
        "| panel | method | tier | rule | selected boundaries / seeds | extensions / seeds |",
        "|---|---|---|---|---|---|",
    ]
    groups = {}
    for row in rows:
        for method in row["method_results"]:
            for rule, tiers in method["tiers"].items():
                for tier, pick in tiers.items():
                    record = method["configurations"][pick["config_index"]]["rules"][rule]
                    groups.setdefault((row["task"], method["method"], tier, rule), []).append(record)
    for key, records in groups.items():
        n = len(records)
        boundaries = sum(r["at_boundary"] for r in records)
        extensions = sum(r.get("extended", False) for r in records)
        lines.append(f"| {' | '.join(key)} | {boundaries}/{n} | {extensions}/{n} |")
    lines += ["", "| panel | seed | dependency design | inner validation |", "|---|---|---|---|"]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['seed']} | {row.get('design', 'legacy')}"
            f" | {row.get('fold_scheme', {}).get('scheme', 'not recorded')} |"
        )
    lines += [
        "",
        "## Protocol notes",
        "",
        "- All routes score MSE for Gaussian or log loss for Binomial."
        " Fold-local routes use `tune_mstop`'s Rust-backed response path."
        " Contrasts isolate stopping within a preprocessing route.",
        "- Patience is unsupported with fold-local preprocessing by the"
        " existing public API. Its native result remains diagnostic; no"
        " fold-local patience result or new stopping subsystem is claimed.",
        "- IID panels use 2 × 5 random folds. `bike_future` uses 10"
        " expanding rolling origins over whole dates, with explicit"
        " held-out/judge masks. Later unused rows do not enter a fold's"
        " score. These origins are not repeated random K-fold partitions."
        " Grouped/spatial designs use whole-group folds when included.",
        "- Every record includes aggregate risk, grid ends, selected-step"
        " fold scores and warnings. Fold-local and fixed-budget records"
        " also include every fold's full risk curve; native growing"
        " `tune_mstop` exposes only the aggregate and selected-step scores.",
        "- Absence of missing cells does not establish fold-local"
        " preprocessing. Spline boundaries and factor vocabularies can"
        " still depend on held-out rows on the native route.",
        "- The original `benches/verify_selection/`,"
        " `benches/path_diagnostics_509.py` and"
        " `docs/benchmarks/path_diagnostics_509.json` are present on main."
        " The v1 claim that they were absent from every branch is withdrawn."
        " Their oracle path minima remain exploratory diagnostics.",
        "- No public defaults, tree topology or tree-AIC support changed."
        " Tree-form and adaptive-degrees-of-freedom research remain separate."
        " This ablation does not rerun the competitor shootout or establish"
        " the original 4–5% excess-loss target.",
        "",
    ]
    return "\n".join(lines)


def audit_v1():
    """Preserve the original run and derive corrections from its recorded losses."""
    source = ARCHIVE_DIR / "results.json"
    document = json.loads(source.read_text())
    summary = summary_document(document)
    summary["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    write_evidence(summary, ARCHIVE_DIR / "audit_v1.json")
    lines = [
        "# Superseded stopping-ablation findings",
        "",
        f"The unchanged [v1 artifact](results.json) was recorded {document['created_at']}"
        f" from `{document['commit']}`. Its dependency/build identities remain in that file."
        " This audit supersedes the numerical conclusions in PR #521 and the"
        " October 12 issue progress report; it does not modify the experiment record.",
        "",
        "The historical refitted baseline and native growing routes have different"
        " preprocessing. Their loss differences combine preprocessing, stopping and"
        " configuration changes. They cannot identify a budget-growth effect."
        " Random inner folds on `bike_future` also target random-row prediction"
        " rather than forecasting. Use [version 2](v2/README.md) for matched-route"
        " and rolling-origin comparisons.",
        "",
        "## Historical comparisons",
        "",
        "The blanket claims that growth never materially worsens loss and that"
        " patience never improves on growth are withdrawn. The following tables"
        " use paired per-seed relative changes, with the named reference loss as"
        " denominator. A median improvement can coexist with a harmful seed.",
        "",
    ]
    lines += _summary_table(summary["historical"]["summary"])
    lines += ["## Patience comparisons", ""]
    lines += _summary_table(summary["comparisons"]["native_patience"]["summary"])
    lines += [
        "## Configuration selection",
        "",
        "The claim that every mboost configuration choice was benign is"
        " withdrawn. Tuned-minus-small changes include harmful seeds inside"
        " this experiment's own mboost grids.",
        "",
    ]
    lines += _summary_table(summary["configuration_selection"]["summary"])
    lines += [
        "## Diagnostic artifacts",
        "",
        "The claim that the original diagnostics were unrecoverable from"
        " every branch is withdrawn. Commit `da27cd9` contains"
        " `benches/verify_selection/`, `benches/path_diagnostics_509.py` and"
        " `docs/benchmarks/path_diagnostics_509.json`, now on main. Their"
        " presence does not validate an oracle stopping rule: a minimum"
        " selected by scanning an assessment path remains exploratory.",
        "",
    ]
    (ARCHIVE_DIR / "audit_v1.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--audit-v1", action="store_true")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    tasks = args.tasks.split(",")
    if args.audit_v1:
        audit_v1()
        return
    if args.prepare:
        for task in tasks:
            prepare(task)
            print("prepared", task, flush=True)
    if args.report:
        document = read_run()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(summarize(document) + "\n")
        write_evidence(summary_document(document), OUT_DIR / "summary.json")
        print("wrote", REPORT, flush=True)
        return
    if not args.run:
        return
    specs = []
    for task in tasks:
        source, _, _ = TASKS[task]
        source_hash = (
            "synthetic-v1"
            if source == "synthetic"
            else hashlib.sha256(
                (CACHE / f"{task}.parquet").read_bytes() + (CACHE / f"{task}.json").read_bytes()
            ).hexdigest()
        )
        for seed in range(args.seeds or (5 if source == "synthetic" else 3)):
            specs.append([task, seed, args.methods, source_hash])
    protocol = {
        "document": "docs/benchmarks/stopping_ablation_509/v2/README.md",
        "experiment_version": EXPERIMENT_VERSION,
        "issue": 509,
        "rules": list(RULES),
        "budget": BUDGET,
        "cap": CAP,
        "patience": PATIENCE,
        "repeated_folds": {"k": KFOLD, "repeats": REPEATS},
        "fold_seed_offset": FOLD_SEED_OFFSET,
        "primary_regression": "MSE on the declared response scale",
        "primary_binary": "log loss",
        "pairing": "within panel/seed/method/tier; contrasts defined in comparisons",
        "comparisons": [list(pair) for pair in COMPARISONS],
        "fold_local_scores": {"regression": "mse", "binary": "log_loss"},
        "temporal_validation": "10 expanding rolling origins over whole dates",
        "status_quo": "shootout_models.fit_model single-split cvrisk argmin",
        "timeout_seconds": 3600,
        "specs": specs,
    }
    document = evidence_document(
        protocol,
        {"real_outer_splits": 3, "synthetic_draws": 5},
        [
            ROOT / "benches" / name
            for name in ("stopping_ablation_509.py", "shootout_models.py", "shootout_data.py", "shootout_metrics.py")
        ],
    )
    document["experiment_version"] = EXPERIMENT_VERSION
    document["uncertainty"] = {
        "real": "paired differences across overlapping outer splits/origins; descriptive, no CI",
        "synthetic": "paired differences across independent draws",
        "timing": "whole-rule wall seconds; no controlled speed ratio",
    }
    rows = run_jobs(
        "stopping_ablation_509", specs, CACHE / "jobs_509_v2", workers=args.workers, timeout=3600, env=THREADS
    )
    document.update(results=rows, completed_at=datetime.datetime.now(datetime.UTC).isoformat())
    write_run(document)
    failed = [row for row in rows if row.get("_status") != "success"]
    if failed or len(rows) != len(specs):
        raise RuntimeError(f"incomplete experiment: {len(failed)} failed, {len(rows)}/{len(specs)} recorded")
    REPORT.write_text(summarize(document) + "\n")
    write_evidence(summary_document(document), OUT_DIR / "summary.json")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
