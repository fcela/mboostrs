"""Machine-readable benchmark contracts shared by runners and documentation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_PATH = ROOT / "benches/scenarios.json"
RESULT_SCHEMA_PATH = ROOT / "docs/schemas/benchmark-result.schema.json"

BOUNDARIES = {"core", "binding", "end_to_end"}
THREAD_MODES = {"single", "default"}
STATUSES = {"implemented", "planned"}


class BenchmarkContractError(RuntimeError):
    """Raised when a benchmark claim lacks required provenance or correctness."""


def _command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def machine_metadata() -> dict[str, Any]:
    """Collect the reproducibility metadata required for a published timing."""
    import mboostrs as mb

    worktree_status = _command_output(["git", "status", "--short"])
    cpu = platform.processor() or None
    if not cpu and Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                cpu = line.partition(":")[2].strip()
                break
    commit = os.environ.get("GITHUB_SHA") or _command_output(["git", "rev-parse", "HEAD"])
    working_tree = {
        "dirty": bool(worktree_status),
        "status_digest": data_digest(worktree_status) if worktree_status else None,
        "commit_source": "working directory",
    }
    if not commit:
        # A run from a frozen copy of the runners outside the checkout has no
        # git working directory; the installed build records the revision it
        # was compiled from and whether that tree was dirty.
        build = dict(mb.build_info())
        commit = build.get("git_revision")
        working_tree = {"dirty": bool(build.get("git_dirty")), "status_digest": None,
                        "commit_source": "mboostrs build_info"}
    return {
        "commit": commit,
        "working_tree": working_tree,
        "native_build": dict(mb.build_info()),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "cpu": cpu,
            "python": platform.python_version(),
        },
        "versions": {
            "mboostrs": _package_version("mboostrs"),
            "numpy": _package_version("numpy"),
            "rpy2": _package_version("rpy2"),
            "rustc": _command_output(["rustc", "--version"]),
            "R": _command_output(["Rscript", "-e", "cat(as.character(getRversion()))"]),
            "mboost": _command_output(
                [
                    "Rscript",
                    "-e",
                    'if (requireNamespace("mboost", quietly=TRUE)) cat(as.character(packageVersion("mboost")))',
                ]
            ),
        },
        "compiler_flags": {
            "RUSTFLAGS": os.environ.get("RUSTFLAGS"),
            "CARGO_PROFILE": os.environ.get("CARGO_PROFILE", "release"),
        },
        "threads": {
            "logical_cpus": os.cpu_count(),
            "r_mclapply_default_cores": _command_output(
                ["Rscript", "-e", 'cat(getOption("mc.cores", 2L))']
            ),
            "RAYON_NUM_THREADS": os.environ.get("RAYON_NUM_THREADS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
    }


def load_scenarios(path: Path = SCENARIOS_PATH) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise BenchmarkContractError("unsupported benchmark scenario manifest schema")
    defaults = manifest.get("defaults", {})
    if not isinstance(defaults.get("warmup"), int) or defaults["warmup"] < 0:
        raise BenchmarkContractError("benchmark warmup must be a non-negative integer")
    # At n = 3 the bootstrap percentile interval is degenerate: the median of a
    # 3-element resample can take only 3 values, so the reported "CI95" is
    # exactly the min/max range in every trial while being labelled a 95%
    # interval — and the artifact already reports min and max separately. Seven
    # is the floor at which the interval is narrower than the range.
    if not isinstance(defaults.get("repetitions"), int) or defaults["repetitions"] < 7:
        raise BenchmarkContractError("benchmark repetitions must be at least seven")
    rows = manifest.get("scenarios", [])
    ids = [row.get("id") for row in rows]
    if not rows or len(ids) != len(set(ids)) or any(not value for value in ids):
        raise BenchmarkContractError("benchmark scenario IDs must be non-empty and unique")
    for row in rows:
        if row.get("boundary") not in BOUNDARIES:
            raise BenchmarkContractError(f"invalid boundary for {row['id']}: {row.get('boundary')}")
        if row.get("thread_mode") not in THREAD_MODES:
            raise BenchmarkContractError(f"invalid thread mode for {row['id']}: {row.get('thread_mode')}")
        if row.get("status") not in STATUSES:
            raise BenchmarkContractError(f"invalid status for {row['id']}: {row.get('status')}")
        if not row.get("coverage_tags"):
            raise BenchmarkContractError(f"scenario {row['id']} has no coverage tags")
        if row.get("cross_language") and row["boundary"] == "core":
            raise BenchmarkContractError(f"cross-language scenario {row['id']} cannot claim a Rust-only core boundary")
    return manifest


def _hash_value(digest: Any, value: Any) -> None:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"array\0")
        digest.update(array.dtype.str.encode())
        digest.update(json.dumps(array.shape).encode())
        digest.update(array.tobytes())
    elif isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value, key=str):
            digest.update(str(key).encode())
            digest.update(b"\0")
            _hash_value(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(b"sequence\0")
        for item in value:
            _hash_value(digest, item)
    elif isinstance(value, np.generic):
        _hash_value(digest, value.item())
    else:
        digest.update(json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode())
        digest.update(b"\0")


def data_digest(value: Any) -> str:
    """Hash inputs or outputs without lossy string formatting."""
    digest = hashlib.sha256()
    _hash_value(digest, value)
    return f"sha256:{digest.hexdigest()}"


#: Prefix on a digest produced by `correctness_digest`: a validated comparison
#: between two implementations' outputs.
CORRECTNESS_DIGEST_TAG = "mboostrs-cmp-v1"

#: Prefix on a digest produced by `self_check_digest`: a single implementation
#: whose outputs were validated against an internal invariant — a pickle
#: round-trip reproducing a fresh prediction, or bit-identical results across
#: thread counts — rather than against a second implementation.
#:
#: The two tags exist because `correctness_digest` in the artifact used to mean
#: three different things under one name: a validated comparison midpoint in
#: `bench_all.py`, and the raw output digest in `bench_python.py` and
#: `bench_thread_scaling.py`. All three runners do verify correctness, by three
#: different mechanisms; a reader of the artifact could not tell which. Now the
#: tag says.
SELF_CHECK_DIGEST_TAG = "mboostrs-self-v1"

DIGEST_TAGS = (CORRECTNESS_DIGEST_TAG, SELF_CHECK_DIGEST_TAG)


def self_check_digest(outputs: Mapping[str, Any], *, invariant: str) -> str:
    """Digest for a single-implementation row, naming the invariant checked.

    `invariant` is a short description of what was validated before this was
    called — the runner performs the check; this records that it claims to have.
    The invariant text is hashed in, so two rows agreeing on a digest also agree
    on what was verified.
    """
    if not invariant:
        raise BenchmarkContractError("self_check_digest requires a non-empty invariant")
    return f"{SELF_CHECK_DIGEST_TAG}:{data_digest({'outputs': outputs, 'invariant': invariant})}"


def correctness_digest(
    outputs: Mapping[str, Sequence[float] | np.ndarray],
    reference: Mapping[str, Sequence[float] | np.ndarray],
    *,
    rtol: float,
    atol: float,
) -> tuple[str, dict[str, float]]:
    """Validate numerical equivalence and return one shared comparison digest.

    The digest hashes the midpoint of the already-validated output pair, and
    carries the `mboostrs-cmp-v1:` tag so `validate_result` can tell it from
    any other string. Both timing rows receive the same digest.

    What that check stops: a stubbed runner, a digest copied from somewhere
    else, a scenario whose comparison was dropped in a refactor — the accidental
    cases. It is not cryptographic and does not stop deliberate forgery: anyone
    writing the artifact by hand can write the tag too. The previous
    formulation, that "a comparison cannot be accepted unless correctness was
    checked first", claimed more than the code did — `validate_result` only
    required the two rows' digests to be *equal*, which two copies of any
    arbitrary string satisfy.
    """
    if set(outputs) != set(reference):
        raise BenchmarkContractError(
            f"compared output fields differ: {sorted(outputs)} != {sorted(reference)}"
        )
    agreed: dict[str, np.ndarray] = {}
    worst_absolute = 0.0
    worst_normalized = 0.0
    for name in sorted(outputs):
        actual = np.asarray(outputs[name], dtype=np.float64)
        desired = np.asarray(reference[name], dtype=np.float64)
        if actual.shape != desired.shape:
            raise BenchmarkContractError(f"output {name} shape mismatch: {actual.shape} != {desired.shape}")
        try:
            np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol, equal_nan=True)
        except AssertionError as exc:
            raise BenchmarkContractError(f"output {name} failed correctness validation: {exc}") from exc
        difference = np.abs(actual - desired)
        allowed = atol + rtol * np.abs(desired)
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = np.where(allowed > 0, difference / allowed, np.where(difference == 0, 0, np.inf))
        if difference.size:
            worst_absolute = max(worst_absolute, float(np.nanmax(difference)))
            worst_normalized = max(worst_normalized, float(np.nanmax(normalized)))
        agreed[name] = (actual + desired) / 2.0
    comparison = {
        "outputs": agreed,
        "rtol": rtol,
        "atol": atol,
    }
    tagged = f"{CORRECTNESS_DIGEST_TAG}:{data_digest(comparison)}"
    return tagged, {
        "max_absolute_error": worst_absolute,
        "max_normalized_error": worst_normalized,
    }


def timing_summary(raw_seconds: Sequence[float]) -> dict[str, float]:
    """Return robust dispersion and a deterministic bootstrap median interval."""
    values = np.asarray(raw_seconds, dtype=np.float64)
    if values.ndim != 1 or values.size < 3 or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise BenchmarkContractError("timings must contain at least three positive finite repetitions")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    rng = np.random.default_rng(20260714)
    samples = rng.choice(values, size=(2000, values.size), replace=True)
    boot = np.median(samples, axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return {
        "median_seconds": median,
        "mad_seconds": mad,
        "ci95_low_seconds": float(low),
        "ci95_high_seconds": float(high),
        "min_seconds": float(np.min(values)),
        "max_seconds": float(np.max(values)),
    }


def paired_ratios(
    arms: Mapping[str, Any],
    *,
    rounds: int,
    warmup: int = 1,
) -> dict[str, Any]:
    """Time two arms alternating in one process, and summarise the *ratios*.

    The protocol from `.remember/done/performance/stabsel-perf-baseline-2026-09-04.md`,
    which is the only one in this repository shown to give a stable number.
    The session before it measured a 6% speedup where it expected several-fold
    and could not tell signal from noise, because repeated runs of the same
    benchmark varied by 3x. Running both arms alternating and reporting the
    median of the *paired* ratios brought that to 1-5% on every shape: machine
    drift -- thermal, another process, the page cache -- moves both arms of a
    pair together and cancels in the ratio, while it does not cancel between
    two separately timed batches.

    It lived in a document and in two scripts that are not in `benches/`, so
    every runner had to reimplement it or go without. This is the shared form.

    `arms` maps a name to a zero-argument callable. Exactly two are expected:
    the ratio is `second / first`, so name the baseline first and read a ratio
    above one as "the second is slower".

    Each callable's return value is digested and the two are required to
    agree, because a speed comparison between arms that computed different
    things is not a measurement of anything. Pass values that compare
    equal under `data_digest` -- the fitted path, the selected components,
    the risk -- not a model object.
    """
    if len(arms) != 2:
        raise BenchmarkContractError(
            f"a paired comparison needs exactly two arms, got {len(arms)}"
        )
    if rounds < 3:
        raise BenchmarkContractError(
            f"a paired comparison needs at least three rounds, got {rounds}"
        )

    import time

    names = list(arms)
    for _ in range(max(0, warmup)):
        for name in names:
            arms[name]()

    seconds: dict[str, list[float]] = {name: [] for name in names}
    digests: dict[str, set[str]] = {name: set() for name in names}
    for _ in range(rounds):
        # Alternating *within* a round, not arm-by-arm across rounds: a batch
        # of one arm followed by a batch of the other is exactly the design
        # whose 3x spread this replaces.
        for name in names:
            started = time.perf_counter()
            output = arms[name]()
            seconds[name].append(time.perf_counter() - started)
            digests[name].add(data_digest(output))

    for name in names:
        if len(digests[name]) != 1:
            raise BenchmarkContractError(
                f"arm {name!r} did not produce the same output on every "
                f"round, so its timings are not repetitions of one thing"
            )
    first, second = names
    if digests[first] != digests[second]:
        raise BenchmarkContractError(
            f"arms {first!r} and {second!r} computed different results; a "
            f"speed ratio between them measures nothing"
        )

    ratios = [
        second_seconds / first_seconds
        for first_seconds, second_seconds
        in zip(seconds[first], seconds[second], strict=True)
    ]
    values = np.asarray(ratios, dtype=np.float64)
    return {
        "protocol": "paired-alternating",
        "arms": [first, second],
        "rounds": rounds,
        "warmup": warmup,
        "ratio": {
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
            "spread": float(values.max() / values.min()),
        },
        "paired_ratios": [float(value) for value in ratios],
        "seconds": {name: [float(v) for v in seconds[name]] for name in names},
        "output_digest": digests[first].pop(),
    }


def measurement(
    implementation: str,
    raw_seconds: Sequence[float],
    *,
    output_digest: str,
    comparison_digest: str,
    peak_rss_bytes: int | None = None,
    allocation_count: int | None = None,
    allocation_method: str | None = None,
    allocation_scope: str | None = None,
) -> dict[str, Any]:
    if allocation_count is not None and (not allocation_method or not allocation_scope):
        raise BenchmarkContractError(
            "measured allocation counts require an explicit method and scope"
        )
    return {
        "implementation": implementation,
        "raw_seconds": [float(value) for value in raw_seconds],
        "summary": timing_summary(raw_seconds),
        "peak_rss": {
            "status": "measured" if peak_rss_bytes is not None else "not_measured",
            "bytes": peak_rss_bytes,
        },
        "allocations": {
            "status": "measured" if allocation_count is not None else "not_measured",
            "count": allocation_count,
            "method": allocation_method,
            "scope": allocation_scope,
        },
        "raw_output_digest": output_digest,
        "correctness_digest": comparison_digest,
    }


def validate_result(payload: dict[str, Any]) -> None:
    schema = json.loads(RESULT_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)
    for result in payload["results"]:
        measurements = result["measurements"]
        digests = {row["correctness_digest"] for row in measurements}
        if len(measurements) > 1 and len(digests) != 1:
            raise BenchmarkContractError(
                f"scenario {result['scenario_id']} has mismatched correctness digests; timing comparison rejected"
            )
        for digest in digests:
            if not any(str(digest).startswith(f"{tag}:") for tag in DIGEST_TAGS):
                raise BenchmarkContractError(
                    f"scenario {result['scenario_id']} carries a correctness_digest that "
                    f"neither correctness_digest() nor self_check_digest() produced "
                    f"(no {' or '.join(DIGEST_TAGS)} tag); timing comparison rejected"
                )

        if result["boundary"] not in BOUNDARIES or result["thread_mode"] not in THREAD_MODES:
            raise BenchmarkContractError(f"scenario {result['scenario_id']} has an invalid boundary/thread mode")
        for row in measurements:
            allocations = row["allocations"]
            if allocations["status"] == "measured" and (
                not allocations.get("method") or not allocations.get("scope")
            ):
                raise BenchmarkContractError(
                    f"scenario {result['scenario_id']} has an allocation count without method/scope provenance"
                )


def regression_report(
    current: dict[str, Any], baseline: dict[str, Any], threshold: float = 0.10
) -> list[dict[str, Any]]:
    """Report, but do not fail on, stable median regressions over ``threshold``."""
    baseline_rows = {
        (result["scenario_id"], row["implementation"]): row
        for result in baseline["results"]
        for row in result["measurements"]
    }
    reports = []
    for result in current["results"]:
        for row in result["measurements"]:
            key = (result["scenario_id"], row["implementation"])
            old = baseline_rows.get(key)
            if old is None:
                continue
            old_median = old["summary"]["median_seconds"]
            new_median = row["summary"]["median_seconds"]
            change = new_median / old_median - 1.0
            if math.isfinite(change) and change > threshold:
                reports.append(
                    {
                        "scenario_id": result["scenario_id"],
                        "implementation": row["implementation"],
                        "baseline_seconds": old_median,
                        "current_seconds": new_median,
                        "change_fraction": change,
                        "blocking": False,
                    }
                )
    return reports
