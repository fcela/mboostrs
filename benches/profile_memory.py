#!/usr/bin/env python3
"""Measure real execution-stage shares, retained caches, and process peak RSS."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator

try:
    from benches.benchmark_contract import ROOT, data_digest, machine_metadata
except ModuleNotFoundError:  # direct execution
    from benchmark_contract import ROOT, data_digest, machine_metadata  # type: ignore[no-redef]

SCHEMA = ROOT / "docs/schemas/profile-memory-result.schema.json"
PHASE_FIELDS = {
    "basis": "basis_ns",
    "factorization": "factorization_ns",
    "binding": "binding_ns",
    "core_setup": "setup_ns",
    "cross_gram": "cross_gram_ns",
    "iterations": "iteration_ns",
    "reconstruction": "reconstruction_ns",
}


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _worker(route: str, n: int, learners: int, mstop: int, seed: int) -> dict[str, Any]:
    import mboostrs as mb

    rng = np.random.default_rng(seed)
    baseline_rss = _peak_rss_bytes()
    data: dict[str, np.ndarray] = {}
    terms = []
    signal = np.zeros(n)
    for index in range(learners):
        x = rng.uniform(-3.0, 3.0, size=n)
        data[f"x{index}"] = x
        terms.append(f"bbs(x{index}, df=4)")
        signal += np.sin((index + 1) * x) / learners
    if route == "gaussian_gram":
        y = signal + rng.normal(scale=0.25, size=n)
        family = mb.Gaussian()
        risk = "inbag"
    elif route == "general_block":
        y = rng.poisson(np.exp(0.4 + 0.2 * signal)).astype(np.float64)
        family = mb.Poisson()
        risk = "inbag"
    else:
        raise ValueError(f"unsupported profile route: {route}")
    data["y"] = y
    formula = "y ~ " + " + ".join(terms)
    model = mb.gamboost(
        formula,
        data=data,
        family=family,
        control=mb.boost_control(mstop=mstop, nu=0.1, risk=risk, trace=True),
    )
    report = model.execution_report()
    if report is None or report.instrumentation != "summary":
        raise RuntimeError("profile worker did not receive summary instrumentation")
    phases_ns = {name: int(getattr(report, field) or 0) for name, field in PHASE_FIELDS.items()}
    measured_total = sum(phases_ns.values())
    if measured_total <= 0:
        raise RuntimeError("profile worker measured no execution time")
    return {
        "route": route,
        "n": n,
        "learners": learners,
        "basis_width": report.total_basis_columns,
        "mstop": mstop,
        "seed": seed,
        "input_digest": data_digest(data),
        "output_digest": data_digest({"fitted": model.fitted_, "risk": model.risk_}),
        "peak_rss_bytes": _peak_rss_bytes(),
        "baseline_rss_bytes": baseline_rss,
        "incremental_peak_rss_bytes": max(0, _peak_rss_bytes() - baseline_rss),
        "factor_cache_bytes": report.factor_cache_bytes,
        "gram_cache_bytes": report.gram_cache_bytes,
        "phases_ns": phases_ns,
        "phase_share": {name: value / measured_total for name, value in phases_ns.items()},
        "measured_total_ns": measured_total,
        "core_total_ns": report.total_ns,
        "execution_report": report.to_dict(),
    }


def _run_worker(route: str, n: int, learners: int, mstop: int, seed: int) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--route",
        route,
        "--n",
        str(n),
        "--learners",
        str(learners),
        "--mstop",
        str(mstop),
        "--seed",
        str(seed),
    ]
    environment = {**os.environ, "RAYON_NUM_THREADS": "1"}
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
        raise RuntimeError(f"profile worker failed:\n{completed.stdout}\n{completed.stderr}")
    return json.loads(completed.stdout)


def _validate(payload: dict[str, Any]) -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)
    for result in payload["results"]:
        share = sum(result["phase_share"].values())
        if not np.isclose(share, 1.0, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"phase shares for {result['route']} sum to {share}, not one")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("benches/results/profile-memory.json"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--route", choices=("gaussian_gram", "general_block"), help=argparse.SUPPRESS)
    parser.add_argument("--n", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--learners", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--mstop", type=int, default=200, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=20260715, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.route is None or args.n is None or args.learners is None:
            parser.error("worker mode requires route, n, and learners")
        print(json.dumps(_worker(args.route, args.n, args.learners, args.mstop, args.seed)))
        return 0

    grid = (
        [("gaussian_gram", 1000, 2), ("gaussian_gram", 5000, 10), ("general_block", 5000, 2)]
        if args.quick
        else [
            ("gaussian_gram", n, learners)
            for n in (1000, 5000, 10000)
            for learners in (2, 10)
        ]
        + [("general_block", n, 2) for n in (1000, 5000, 10000)]
    )
    results = []
    for index, (route, n, learners) in enumerate(grid):
        print(f"Measuring {route} n={n} learners={learners} ...", file=sys.stderr, flush=True)
        results.append(_run_worker(route, n, learners, args.mstop, args.seed + index))
    payload = {
        "schema_version": 1,
        "kind": "mboostrs-profile-memory-result",
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "metadata": machine_metadata(),
        "measurement": {
            "profile": "exclusive wall-clock phases from ExecutionReport summary instrumentation",
            "rss": "fresh-process ru_maxrss; bytes on macOS and KiB normalized to bytes elsewhere",
            "thread_mode": "single",
        },
        "results": results,
    }
    _validate(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(results)} measured profile/memory rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
