#!/usr/bin/env python3
"""Isolated stability-selection serial/default and thread-scaling evidence."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    from benches.benchmark_contract import (
        ROOT,
        BenchmarkContractError,
        data_digest,
        load_scenarios,
        machine_metadata,
        measurement,
        self_check_digest,
        validate_result,
    )
except ModuleNotFoundError:
    from benchmark_contract import (  # type: ignore[no-redef]
        ROOT,
        BenchmarkContractError,
        data_digest,
        load_scenarios,
        machine_metadata,
        measurement,
        self_check_digest,
        validate_result,
    )


def _worker(parameters: dict[str, Any], seed: int, warmup: int, repetitions: int) -> dict[str, Any]:
    import mboostrs as mb

    rng = np.random.default_rng(seed)
    n = int(parameters["n"])
    candidates = int(parameters["candidates"])
    matrix = rng.normal(size=(n, candidates))
    beta = np.zeros(candidates)
    beta[:5] = [1.2, -0.9, 0.7, -0.5, 0.3]
    y = matrix @ beta + rng.normal(scale=0.7, size=n)
    data = {f"x{index}": matrix[:, index] for index in range(candidates)} | {"y": y}
    formula = "y ~ " + " + ".join(f"x{index}" for index in range(candidates))
    model = mb.glmboost(
        formula,
        data=data,
        control=mb.boost_control(int(parameters["mstop"]), 0.1),
    )

    def operation():
        return mb.stabsel(
            model,
            q=int(parameters["q"]),
            PFER=float(parameters["PFER"]),
            B=int(parameters["subsamples"]),
            verbose=False,
        )

    for _ in range(warmup):
        operation()
    raw_seconds = []
    result = None
    for _ in range(repetitions):
        gc.collect()
        started = time.perf_counter()
        result = operation()
        raw_seconds.append(time.perf_counter() - started)
    if result is None:
        raise RuntimeError("stability-selection worker produced no result")
    outputs = {
        "phat": result.phat,
        "max": result.max_,
        "selected": np.asarray(result.selected, dtype=np.int64),
        "cutoff": result.cutoff,
        "q": result.q,
        "PFER": result.PFER,
    }
    return {
        "raw_seconds": raw_seconds,
        "input_digest": data_digest(data),
        "output_digest": data_digest(outputs),
        "selected": result.selected,
        "phat_shape": list(result.phat.shape),
    }


def _isolated_worker(
    parameters: dict[str, Any], seed: int, warmup: int, repetitions: int, threads: int | None
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--parameters",
        json.dumps(parameters, separators=(",", ":")),
        "--seed",
        str(seed),
        "--warmup",
        str(warmup),
        "--repetitions",
        str(repetitions),
    ]
    environment = dict(os.environ)
    if threads is None:
        environment.pop("RAYON_NUM_THREADS", None)
    else:
        environment["RAYON_NUM_THREADS"] = str(threads)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"thread worker failed:\n{completed.stdout}\n{completed.stderr}")
    return json.loads(completed.stdout)


def _result(row: dict[str, Any], implementation: str, worker: dict[str, Any], shared_digest: str) -> dict[str, Any]:
    return {
        "scenario_id": row["id"],
        "boundary": row["boundary"],
        "thread_mode": row["thread_mode"],
        "seed": row["seed"],
        "parameters": row["parameters"],
        "input_digest": worker["input_digest"],
        "compared_outputs": ["phat", "max", "selected", "cutoff", "q", "PFER"],
        "tolerance": {"atol": 0.0, "rtol": 0.0},
        "correctness": {"status": "passed", "max_absolute_error": 0.0, "max_normalized_error": 0.0},
        "measurements": [
            measurement(
                implementation,
                worker["raw_seconds"],
                output_digest=worker["output_digest"],
                comparison_digest=shared_digest,
            )
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("benches/results/thread-scaling.json"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--parameters", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--warmup", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--repetitions", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if None in (args.parameters, args.seed, args.warmup, args.repetitions):
            parser.error("worker mode requires parameters, seed, warmup, and repetitions")
        print(json.dumps(_worker(json.loads(args.parameters), args.seed, args.warmup, args.repetitions)))
        return 0

    manifest = load_scenarios()
    rows = {row["id"]: row for row in manifest["scenarios"] if row["runner"] == "thread_scaling"}
    serial_row = rows["stabsel.scalar.serial.end_to_end"]
    default_row = rows["stabsel.scalar.default.end_to_end"]
    scaling_row = rows["scaling.thread_count.core"]
    warmup = 1 if args.quick else manifest["defaults"]["warmup"]
    # 7, not 3: below seven the bootstrap percentile interval collapses to
    # the min/max range while still being labelled CI95, and --quick would
    # otherwise be one flag away from writing a degenerate interval into a
    # published artifact with nothing marking it.
    repetitions = 7 if args.quick else manifest["defaults"]["repetitions"]

    serial = _isolated_worker(serial_row["parameters"], serial_row["seed"], warmup, repetitions, 1)
    default = _isolated_worker(default_row["parameters"], default_row["seed"], warmup, repetitions, None)
    if serial["input_digest"] != default["input_digest"] or serial["output_digest"] != default["output_digest"]:
        raise BenchmarkContractError("serial/default stability-selection outputs differ")
    # The check above is stricter than a tolerance comparison: the outputs must
    # be bit-identical, which is what `atol: 0.0, rtol: 0.0` in this runner's
    # rows records. Tagging it as a self-check rather than reusing the raw
    # output digest is what lets a reader of the artifact tell this evidence
    # from `bench_all.py`'s two-implementation comparison.
    raw_shared = serial["output_digest"]
    shared_digest = self_check_digest(
        {"outputs": raw_shared},
        invariant="stability-selection outputs bit-identical across thread counts",
    )
    results = [
        _result(serial_row, "mboostrs-threads-1", serial, shared_digest),
        _result(default_row, "mboostrs-threads-default", default, shared_digest),
    ]

    scaling_workers = []
    for threads in scaling_row["parameters"]["threads"]:
        worker = _isolated_worker(
            scaling_row["parameters"], scaling_row["seed"], warmup, repetitions, int(threads)
        )
        if worker["output_digest"] != raw_shared:
            raise BenchmarkContractError(f"thread-count {threads} changed stability-selection outputs")
        scaling_workers.append((threads, worker))
    scaling_result = {
        **_result(scaling_row, "placeholder", scaling_workers[0][1], shared_digest),
        "measurements": [
            measurement(
                f"mboostrs-threads-{threads}",
                worker["raw_seconds"],
                output_digest=worker["output_digest"],
                comparison_digest=shared_digest,
            )
            for threads, worker in scaling_workers
        ],
    }
    results.append(scaling_result)
    payload = {
        "schema_version": 1,
        "kind": "mboostrs-benchmark-result",
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "manifest_digest": data_digest(manifest),
        "metadata": machine_metadata(),
        "warmup_policy": {"iterations": warmup, "description": "fresh-process untimed calls before repetitions"},
        "results": results,
        "regressions": [],
    }
    validate_result(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote serial/default and {len(scaling_workers)}-point thread scaling to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
