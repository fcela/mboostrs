"""Run the frozen multi-objective competitive study in isolated serial jobs.

Prepare: python -m benches.bench_competitive_shootout --prepare
Run: OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     python -m benches.bench_competitive_shootout --run
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import os
import resource
import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benches.benchmark_contract import data_digest  # noqa: E402
from benches.shootout_data import CACHE, TASKS, payload, prepare  # noqa: E402
from benches.shootout_metrics import evaluate, explanations, structure  # noqa: E402
from benches.shootout_models import METHODS, clean, search  # noqa: E402

OUT = ROOT / "docs/benchmarks/competitive_shootout_results.json"
THREADS = {k: "1" for k in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                            "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}


def run_job(spec):
    """One task/draw/method; test outcomes enter only the final evaluator."""
    task, seed, method, tier, *_ = spec
    # Import model libraries before the clock; process start/imports remain
    # visible in the supervisor's elapsed time and peak process memory.
    if method == "catboost":
        import catboost  # noqa: F401
    elif method == "lightgbm":
        import lightgbm  # noqa: F401
    elif method == "xgboost":
        import xgboost  # noqa: F401
    elif method == "ebm":
        import interpret.glassbox  # noqa: F401
    import mboostrs  # noqa: F401

    data = payload(task, int(seed))
    train, val, cal, test = data["frames"]
    cats, kind = data["categorical"], data["kind"]
    begin = time.perf_counter()
    outputs = search(method, train, val, cats, kind, int(seed), tier=tier)
    dev = pl.concat([train, val])
    dev, cal, test = clean(dev, cal, test, cats=cats)
    results = []
    for tier, model, costs in outputs:
        scores, predictions = evaluate(model, cal, test, dev)
        explanation = explanations(model, dev, task, int(seed))
        size = structure(model)
        seconds = (costs["search_seconds"] + costs["refit_seconds"]
                   + scores["calibration_prediction_seconds"] + explanation["seconds"]
                   + size["report_storage_seconds"])
        results.append({"tier": tier, "scores": scores, "explanation": explanation,
                        "structure": size, "costs": costs, "workflow_seconds": seconds,
                        "selected_config": model.config,
                        "prediction_digest": data_digest(predictions)})
    return {"task": task, "seed": int(seed), "method": method, "tier": tier, "kind": kind,
            "results": results, "input_digest": data_digest({
                "columns": train.columns,
                "frames": [f.to_dict(as_series=False) for f in data["frames"]]}),
            "partition_digest": data_digest(data["indices"]),
            "sizes": {k: len(f) for k, f in zip(("fit", "validation", "calibration", "test"),
                                                data["frames"], strict=True)},
            "task_metadata": {k: v for k, v in data["metadata"].items() if k != "auxiliary"},
            "auxiliary_digest": data_digest(data["metadata"].get("auxiliary", {})),
            "job_workflow_seconds": time.perf_counter() - begin,
            "peak_process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                                           * (1 if sys.platform == "darwin" else 1024))}


def main():
    import importlib.metadata
    import subprocess

    from benches._jobs import run_jobs
    from benches.study_evidence import evidence_document, write_evidence

    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()
    tasks = args.tasks.split(",")
    if args.prepare:
        for task in tasks:
            prepare(task)
            print("prepared", task, flush=True)
    if not args.run:
        return
    specs = []
    for i, task in enumerate(tasks):
        source_hash = ("synthetic-v1" if TASKS[task][0] == "synthetic" else
                       hashlib.sha256((CACHE / f"{task}.parquet").read_bytes()
                                      + (CACHE / f"{task}.json").read_bytes()).hexdigest())
        for seed in range(args.seeds or (5 if TASKS[task][0] == "synthetic" else 3)):
            methods = args.methods.split(",")
            shift = (i + seed) % len(methods)
            methods = methods[shift:] + methods[:shift]
            specs.extend((task, seed, method, tier, source_hash)
                         for method in methods for tier in ("small", "tuned"))
    protocol = {"document": "docs/benchmarks/competitive_shootout_protocol.md",
                "document_sha256": hashlib.sha256((ROOT / "docs/benchmarks/competitive_shootout_protocol.md")
                                                   .read_bytes()).hexdigest(),
                "primary_regression": "MSE on the declared response scale",
                "primary_binary": "log loss", "competitive_excess_loss": .05,
                "gap_excess_loss": .1, "timeout_seconds": 240,
                "max_development_rows": 4000, "max_calibration_rows": 1000,
                "max_test_rows": 2000, "specs": [list(s) for s in specs],
                "order": "serial rotated method order", "workflow_warmup": 0,
                "prediction_warmup": 1, "prediction_repeats": 7,
                "data_excluded_from_clock": "imports and input loading"}
    document = evidence_document(protocol, {"real_outer_splits": 3, "synthetic_draws": 5},
                                 [ROOT / "benches" / name for name in (
                                     "bench_competitive_shootout.py", "shootout_models.py",
                                     "shootout_data.py", "shootout_metrics.py")])
    document["uncertainty"] = {"real": "paired descriptive split spread; no IID SE or population CI",
                               "synthetic": "independent-draw Monte Carlo variation",
                               "timing": "whole-workflow descriptive seconds; no controlled speed ratio"}
    document["competitor_versions"] = {name: importlib.metadata.version(name) for name in
                                       ("catboost", "xgboost", "lightgbm", "interpret", "scikit-learn")}
    if sys.platform == "darwin":
        document["platform"]["cpu_model"] = os.environ.get("MBOOSTRS_BENCH_CPU_MODEL")
        if document["platform"]["cpu_model"] is None:
            try:
                document["platform"]["cpu_model"] = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            except subprocess.SubprocessError:
                document["platform"]["cpu_model"] = None
    rows = run_jobs("bench_competitive_shootout", specs, CACHE / "jobs", workers=1,
                    timeout=240, env=THREADS)
    document.update(results=rows, completed_at=datetime.datetime.now(datetime.UTC).isoformat(),
                    memory={"status": "measured", "unit": "fresh process peak RSS, includes imports"})
    write_evidence(document, OUT)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
