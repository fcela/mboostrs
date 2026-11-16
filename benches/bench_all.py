#!/usr/bin/env python3
"""Run fair, correctness-gated mboostrs/R benchmark comparisons.

The runner times the same semantic boundary on both sides. Fit scenarios start
with already-materialized language-native data and end with a fitted model. CV
scenarios start with an already-fitted model plus identical folds and end with
the CV result; neither side includes the initial fit. Raw repetitions and full
machine provenance are written to JSON rather than copied into prose.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

try:
    from benches.benchmark_contract import (
        BenchmarkContractError,
        correctness_digest,
        data_digest,
        load_scenarios,
        machine_metadata,
        measurement,
        regression_report,
        validate_result,
    )
except ModuleNotFoundError:  # direct ``python benches/bench_all.py`` execution
    from benchmark_contract import (  # type: ignore[no-redef]
        BenchmarkContractError,
        correctness_digest,
        data_digest,
        load_scenarios,
        machine_metadata,
        measurement,
        regression_report,
        validate_result,
    )


def _python_timings(function: Callable[[], Any], warmup: int, repetitions: int) -> list[float]:
    for _ in range(warmup):
        function()
    timings = []
    for _ in range(repetitions):
        gc.collect()
        started = time.perf_counter()
        function()
        timings.append(time.perf_counter() - started)
    return timings


def _r_timings(ro: Any, expression: str, warmup: int, repetitions: int) -> list[float]:
    timer = ro.r(
        f"""
        function(warmup, repetitions) {{
            benchmark_call <- function() {{ {expression} }}
            if (warmup > 0) for (i in seq_len(warmup)) invisible(benchmark_call())
            vapply(seq_len(repetitions), function(i) {{
                gc(verbose = FALSE)
                unname(system.time(invisible(benchmark_call()))[["elapsed"]])
            }}, numeric(1))
        }}
        """
    )
    values = np.asarray(timer(warmup, repetitions), dtype=np.float64)
    if np.any(values <= 0):
        raise BenchmarkContractError(
            "R timer resolution produced a non-positive repetition; use a larger scenario instead of publishing it"
        )
    return values.tolist()


def _python_allocation_count(function: Callable[[], Any]) -> tuple[int, str, str]:
    """Count one warmed call's Python, native-extension, and Rust allocations."""
    try:
        import memray
    except ImportError as exc:  # pragma: no cover - exercised by benchmark environments
        raise BenchmarkContractError(
            "cross-language allocation evidence requires Memray; install mboostrs[benchmark]"
        ) from exc

    deallocators = {
        memray.AllocatorType.PYMALLOC_FREE,
        memray.AllocatorType.FREE,
        memray.AllocatorType.MUNMAP,
    }

    def tracked_count(call: Callable[[], Any]) -> int:
        with TemporaryDirectory(prefix="mboostrs-memray-") as directory:
            capture = Path(directory) / "allocations.bin"
            with memray.Tracker(capture, trace_python_allocators=True):
                retained_result = call()
            # Keep the result alive until tracking stops so its construction is
            # measured even when the caller would otherwise discard it.
            del retained_result
            records = memray.FileReader(capture).get_allocation_records()
            return sum(
                record.n_allocations
                for record in records
                if record.allocator not in deallocators
            )

    # Tracker entry/exit has a small stable footprint of its own. Subtract an
    # empty capture made through the identical context so the published count
    # describes the frontend call rather than the profiling harness.
    harness_count = tracked_count(lambda: None)
    call_count = tracked_count(function)
    return (
        max(0, call_count - harness_count),
        f"Memray {memray.__version__} ALL_ALLOCATIONS minus empty-tracker baseline",
        "one warmed frontend call; system and CPython allocator events across all threads, "
        "including Rust/native extensions",
    )


def _r_allocation_count(ro: Any, expression: str, *, parallel: bool = False) -> tuple[int, str, str]:
    """Count R allocator events, including fork workers for parallel CV."""
    if parallel:
        papply_definition = """
            bench_profiled_mclapply <- function(
                X, FUN, ..., mc.preschedule = TRUE, mc.set.seed = TRUE,
                mc.silent = FALSE, mc.cores = getOption("mc.cores", 2L),
                mc.cleanup = TRUE, mc.allow.recursive = TRUE, affinity.list = NULL
            ) {
                result <- parallel::mclapply(
                    seq_along(X),
                    function(index) {
                        child_path <- file.path(
                            profile_dir,
                            paste0("worker-", Sys.getpid(), "-", index, ".out")
                        )
                        utils::Rprofmem(child_path, threshold = 0)
                        on.exit(utils::Rprofmem(NULL), add = TRUE)
                        FUN(X[[index]], ...)
                    },
                    mc.preschedule = mc.preschedule,
                    mc.set.seed = mc.set.seed,
                    mc.silent = mc.silent,
                    mc.cores = mc.cores,
                    mc.cleanup = mc.cleanup,
                    mc.allow.recursive = mc.allow.recursive,
                    affinity.list = affinity.list
                )
                names(result) <- names(X)
                result
            }
        """
    else:
        papply_definition = ""

    value = ro.r(
        f"""
        (function() {{
            if (!capabilities("profmem"))
                stop("R was built without memory-profiling support")
            profile_dir <- tempfile("mboostrs-rprofmem-")
            dir.create(profile_dir)
            on.exit(unlink(profile_dir, recursive = TRUE), add = TRUE)
            {papply_definition}
            parent_path <- file.path(profile_dir, "parent.out")
            utils::Rprofmem(parent_path, threshold = 0)
            on.exit(utils::Rprofmem(NULL), add = TRUE)
            profiled_value <- {expression}
            utils::Rprofmem(NULL)
            profile_files <- list.files(profile_dir, full.names = TRUE)
            event_count <- sum(vapply(
                profile_files,
                function(path) length(readLines(path, warn = FALSE)),
                integer(1)
            ))
            invisible(profiled_value)
            as.numeric(event_count)
        }})()
        """
    )
    count = float(np.asarray(value, dtype=np.float64)[0])
    if not np.isfinite(count) or count < 0:
        raise BenchmarkContractError("Rprofmem did not return a finite allocation count")
    scope = "one warmed frontend call; R allocator events in the parent"
    if parallel:
        scope += " and all fork workers"
    scope += "; excludes allocations made outside R's allocator API"
    return int(count), "Rprofmem threshold=0", scope


def _install_r_frame(ro: Any, name: str, columns: dict[str, np.ndarray]) -> None:
    ro.globalenv[name] = ro.r("data.frame")(
        **{column: ro.FloatVector(np.asarray(values, dtype=np.float64)) for column, values in columns.items()}
    )


def _r_fit_outputs(ro: Any, fit_expression: str) -> dict[str, np.ndarray]:
    result = ro.r(
        f"""
        local({{
            fit <- {fit_expression}
            list(
                fitted = as.numeric(stats::predict(fit, type = "link")),
                risk = as.numeric(mboost::risk(fit))
            )
        }})
        """
    )
    return {
        "fitted": np.asarray(result[0], dtype=np.float64),
        "risk": np.asarray(result[1], dtype=np.float64),
    }


def _fit_scenario(row: dict[str, Any], mb: Any, pl: Any, ro: Any, warmup: int, repetitions: int) -> dict:
    parameters = row["parameters"]
    n = int(parameters["n"])
    mstop = int(parameters["mstop"])
    candidates = int(parameters["candidates"])
    rng = np.random.default_rng(row["seed"])
    family = "gaussian" if ".gaussian." in row["id"] else "poisson"

    if row["id"].startswith("fit.scalar"):
        matrix = rng.normal(size=(n, candidates))
        beta = rng.normal(scale=0.4, size=candidates)
        y = matrix @ beta + rng.normal(scale=0.5, size=n)
        columns = {f"x{index + 1}": matrix[:, index] for index in range(candidates)} | {"y": y}
        formula = "y ~ " + " + ".join(f"x{index + 1}" for index in range(candidates))
        def py_fit():
            return mb.glmboost(
                formula,
                data=py_data,
                family=mb.Gaussian(),
                control=mb.boost_control(mstop=mstop, nu=0.1),
            )
        r_fit = (
            f"mboost::glmboost({formula}, data = df_bench, family = mboost::Gaussian(), "
            f"center = TRUE, control = mboost::boost_control(mstop = {mstop}, nu = 0.1))"
        )
        rtol, atol = 1e-10, 1e-10
    else:
        x = rng.uniform(-3.0, 3.0, size=n)
        z = rng.uniform(-3.0, 3.0, size=n)
        eta = np.sin(1.2 * x) + 0.5 * np.cos(z)
        if family == "poisson":
            y = rng.poisson(np.exp(0.5 + 0.25 * eta)).astype(np.float64)
            py_family, r_family = mb.Poisson(), "Poisson"
        else:
            y = eta + rng.normal(scale=0.25, size=n)
            py_family, r_family = mb.Gaussian(), "Gaussian"
        columns = {"x": x, "z": z, "y": y}
        formula = "y ~ bbs(x, df=4) + bbs(z, df=4)"
        def py_fit():
            return mb.gamboost(
                formula,
                data=py_data,
                family=py_family,
                control=mb.boost_control(mstop=mstop, nu=0.1),
            )
        r_fit = (
            f"mboost::gamboost({formula}, data = df_bench, family = mboost::{r_family}(), "
            f"control = mboost::boost_control(mstop = {mstop}, nu = 0.1))"
        )
        rtol, atol = 1e-6, 1e-8

    py_data = pl.DataFrame(columns)
    _install_r_frame(ro, "df_bench", columns)
    python_model = py_fit()
    python_outputs = {
        "fitted": np.asarray(python_model.fitted_, dtype=np.float64),
        "risk": np.asarray(python_model.risk_, dtype=np.float64),
    }
    r_outputs = _r_fit_outputs(ro, r_fit)
    shared_digest, error = correctness_digest(python_outputs, r_outputs, rtol=rtol, atol=atol)
    python_times = _python_timings(py_fit, warmup, repetitions)
    r_times = _r_timings(ro, r_fit, warmup, repetitions)
    python_allocations = _python_allocation_count(py_fit)
    r_allocations = _r_allocation_count(ro, r_fit)
    return _scenario_result(
        row,
        columns,
        python_outputs,
        r_outputs,
        shared_digest,
        error,
        python_times,
        r_times,
        python_allocations,
        r_allocations,
        rtol,
        atol,
    )


def _cv_scenario(row: dict[str, Any], mb: Any, pl: Any, ro: Any, warmup: int, repetitions: int) -> dict:
    parameters = row["parameters"]
    n = int(parameters["n"])
    candidates = int(parameters["candidates"])
    folds_count = int(parameters["folds"])
    mstop = int(parameters["mstop"])
    rng = np.random.default_rng(row["seed"])
    matrix = rng.normal(size=(n, candidates))
    beta = rng.normal(scale=0.4, size=candidates)
    y = matrix @ beta + rng.normal(scale=0.5, size=n)
    columns = {f"x{index + 1}": matrix[:, index] for index in range(candidates)} | {"y": y}
    formula = "y ~ " + " + ".join(f"x{index + 1}" for index in range(candidates))
    py_data = pl.DataFrame(columns)
    _install_r_frame(ro, "df_bench", columns)
    ro.r(
        f"""
        fit_bench <- mboost::glmboost(
            {formula}, data = df_bench, family = mboost::Gaussian(), center = TRUE,
            control = mboost::boost_control(mstop = {mstop}, nu = 0.1))
        set.seed({row['seed']})
        folds_bench <- mboost::cv(model.weights(fit_bench), type = "kfold", B = {folds_count})
        """
    )
    folds = np.asarray(ro.r("as.matrix(folds_bench)"), dtype=np.float64)
    python_model = mb.glmboost(
        formula,
        data=py_data,
        family=mb.Gaussian(),
        control=mb.boost_control(mstop=mstop, nu=0.1),
    )
    parallel = row["thread_mode"] == "default"

    def python_call():
        return python_model.cvrisk(folds=folds, parallel=parallel)

    if parallel:
        r_call = "mboost::cvrisk(fit_bench, folds = folds_bench, papply = parallel::mclapply)"
        r_allocation_call = (
            "mboost::cvrisk(fit_bench, folds = folds_bench, papply = bench_profiled_mclapply)"
        )
    else:
        r_call = "mboost::cvrisk(fit_bench, folds = folds_bench, papply = lapply)"
        r_allocation_call = r_call
    python_cv = python_call()
    r_cv = ro.r(
        f"local({{ value <- {r_call}; list(risk = as.matrix(value), aggregate = colSums(value, na.rm = TRUE)) }})"
    )
    python_outputs = {
        "risk": np.asarray(python_cv.risk_matrix, dtype=np.float64),
        "aggregate": np.asarray(python_cv.aggregate, dtype=np.float64),
    }
    r_outputs = {
        "risk": np.asarray(r_cv[0], dtype=np.float64),
        "aggregate": np.asarray(r_cv[1], dtype=np.float64),
    }
    rtol, atol = 1e-6, 1e-9
    shared_digest, error = correctness_digest(python_outputs, r_outputs, rtol=rtol, atol=atol)
    python_times = _python_timings(python_call, warmup, repetitions)
    r_times = _r_timings(ro, r_call, warmup, repetitions)
    python_allocations = _python_allocation_count(python_call)
    r_allocations = _r_allocation_count(ro, r_allocation_call, parallel=parallel)
    inputs = {**columns, "folds": folds}
    return _scenario_result(
        row,
        inputs,
        python_outputs,
        r_outputs,
        shared_digest,
        error,
        python_times,
        r_times,
        python_allocations,
        r_allocations,
        rtol,
        atol,
    )


def _scenario_result(
    row: dict[str, Any],
    inputs: dict[str, Any],
    python_outputs: dict[str, np.ndarray],
    r_outputs: dict[str, np.ndarray],
    shared_digest: str,
    error: dict[str, float],
    python_times: list[float],
    r_times: list[float],
    python_allocations: tuple[int, str, str],
    r_allocations: tuple[int, str, str],
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    return {
        "scenario_id": row["id"],
        "boundary": row["boundary"],
        "boundary_description": (
            "language-native frontend call with data/control already materialized"
            if row["operation"].endswith("fit")
            else "CV call on an already-fitted model with identical folds; initial fit excluded"
        ),
        "thread_mode": row["thread_mode"],
        "seed": row["seed"],
        "parameters": row["parameters"],
        "input_digest": data_digest(inputs),
        "compared_outputs": sorted(python_outputs),
        "tolerance": {"rtol": rtol, "atol": atol},
        "correctness": {"status": "passed", **error},
        "measurements": [
            measurement(
                "mboostrs",
                python_times,
                output_digest=data_digest(python_outputs),
                comparison_digest=shared_digest,
                allocation_count=python_allocations[0],
                allocation_method=python_allocations[1],
                allocation_scope=python_allocations[2],
            ),
            measurement(
                "R mboost",
                r_times,
                output_digest=data_digest(r_outputs),
                comparison_digest=shared_digest,
                allocation_count=r_allocations[0],
                allocation_method=r_allocations[1],
                allocation_scope=r_allocations[2],
            ),
        ],
    }


def _markdown_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# Auditable mboostrs benchmark result",
        "",
        f"Commit: `{payload['metadata']['commit'] or 'unknown'}`",
        "",
        "| Scenario | Boundary | Threads | mboostrs median | R median | Ratio | mboostrs allocs | R allocs |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for result in payload["results"]:
        by_name = {row["implementation"]: row for row in result["measurements"]}
        rust = by_name["mboostrs"]["summary"]["median_seconds"]
        r_time = by_name["R mboost"]["summary"]["median_seconds"]
        rust_allocations = by_name["mboostrs"]["allocations"]["count"]
        r_allocations = by_name["R mboost"]["allocations"]["count"]
        lines.append(
            f"| `{result['scenario_id']}` | `{result['boundary']}` | `{result['thread_mode']}` | "
            f"{rust * 1000:.3f} ms | {r_time * 1000:.3f} ms | {r_time / rust:.2f}x | "
            f"{rust_allocations:,} | {r_allocations:,} |"
        )
    lines.extend(
        [
            "",
            "A ratio is emitted only after the compared outputs pass the recorded tolerance "
            "and share one correctness digest.",
            "Allocation counts are implementation-specific evidence, not a ratio: Memray records "
            "system plus CPython allocators for mboostrs, while Rprofmem records R allocator events.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("benches/results/local-benchmark.json"))
    parser.add_argument("--scenario", action="append", help="implemented cross-language scenario ID (repeatable)")
    parser.add_argument("--quick", action="store_true", help="use one warmup and three measured repetitions")
    parser.add_argument(
        "--thread-mode",
        choices=("single", "default"),
        default="single",
        help="select implemented scenarios with one matching thread mode",
    )
    parser.add_argument("--list", action="store_true", help="list scenario coverage and exit")
    parser.add_argument("--baseline", type=Path, help="optional prior result for non-blocking >10% reports")
    args = parser.parse_args()
    manifest = load_scenarios()
    if args.list:
        for row in manifest["scenarios"]:
            print(f"{row['status']:<11} {row['boundary']:<10} {row['thread_mode']:<7} {row['id']}")
        return 0

    selected_ids = args.scenario or [
        row["id"]
        for row in manifest["scenarios"]
        if row["runner"] == "cross_language"
        and row["status"] == "implemented"
        and row["thread_mode"] == args.thread_mode
    ]
    by_id = {row["id"]: row for row in manifest["scenarios"]}
    invalid = [
        scenario
        for scenario in selected_ids
        if scenario not in by_id or by_id[scenario]["status"] != "implemented"
    ]
    if invalid:
        raise BenchmarkContractError(f"scenarios are unknown or not implemented: {invalid}")

    selected_modes = {by_id[scenario]["thread_mode"] for scenario in selected_ids}
    if len(selected_modes) != 1:
        raise BenchmarkContractError("single/default scenarios must run in separate processes")
    selected_mode = selected_modes.pop()

    # Rayon reads its global pool configuration on first use. Separate process
    # runs prevent a serial pool from contaminating default-parallel evidence.
    if selected_mode == "single":
        os.environ["RAYON_NUM_THREADS"] = "1"
    elif os.environ.get("RAYON_NUM_THREADS") is not None:
        raise BenchmarkContractError("default thread-mode evidence requires RAYON_NUM_THREADS to be unset")
    import polars as pl
    import rpy2.robjects as ro

    import mboostrs as mb

    ro.r("suppressPackageStartupMessages(library(mboost))")

    warmup = 1 if args.quick else manifest["defaults"]["warmup"]
    # 7, not 3: below seven the bootstrap percentile interval collapses to
    # the min/max range while still being labelled CI95, and --quick would
    # otherwise be one flag away from writing a degenerate interval into a
    # published artifact with nothing marking it.
    repetitions = 7 if args.quick else manifest["defaults"]["repetitions"]
    results = []
    for scenario_id in selected_ids:
        row = by_id[scenario_id]
        print(f"Running {scenario_id} ...", flush=True)
        if scenario_id.startswith("cv."):
            result = _cv_scenario(row, mb, pl, ro, warmup, repetitions)
        else:
            result = _fit_scenario(row, mb, pl, ro, warmup, repetitions)
        results.append(result)

    payload = {
        "schema_version": 1,
        "kind": "mboostrs-benchmark-result",
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "manifest_digest": data_digest(manifest),
        "metadata": machine_metadata(),
        "warmup_policy": {
            "iterations": warmup,
            "description": "untimed calls per implementation before independent wall-clock repetitions",
        },
        "results": results,
        "regressions": [],
    }
    if args.baseline:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        payload["regressions"] = regression_report(
            payload, baseline, manifest["defaults"]["regression_report_threshold"]
        )
    validate_result(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    summary = _markdown_summary(payload)
    args.output.with_suffix(".md").write_text(summary, encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
