"""Pytest plugin that writes a node-level mboostrs parity ledger.

Load with ``-p tools.parity_ledger --parity-ledger=PATH``. The plugin is
deliberately independent of pytest-json-report: scheduled parity must still
produce an artifact when a test fails or the optional reporting package is not
installed.
"""

from __future__ import annotations

import functools
import importlib.metadata
import json
import math
import os
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "python/mboostrs/capability_manifest.json"
SCHEMA_VERSION = 1


@dataclass
class ComparisonSummary:
    comparison_count: int = 0
    measured: bool = False
    max_absolute_error: float | None = None
    max_relative_error: float | None = None
    normalized_error: float | None = None
    atol: float | None = None
    rtol: float | None = None
    #: Set only when the assertion passed a per-element tolerance, so a reader
    #: can tell one tolerance from the loosest of several.
    atol_min: float | None = None
    rtol_min: float | None = None
    comparison: str | None = None

    def observe(self, measurement: dict[str, float | str | None]) -> None:
        self.comparison_count += 1
        normalized = measurement["normalized_error"]
        if not isinstance(normalized, float):
            return
        self.measured = True
        # `self.normalized_error is None` means "nothing observed yet" -- once
        # set, it is always a real float (possibly inf/nan), never reset to
        # None, so a non-finite worst case can never look unset to a later,
        # smaller measurement. NaN is treated as worse than anything, since it
        # means a comparison could not even be evaluated (an unrescued
        # mismatch); plain `>=` would silently never select a NaN here,
        # because every IEEE754 comparison against NaN is False.
        if self.normalized_error is None or _at_least_as_bad(normalized, self.normalized_error):
            self.max_absolute_error = float(measurement["max_absolute_error"])
            self.max_relative_error = float(measurement["max_relative_error"])
            self.normalized_error = normalized
            self.atol = float(measurement["atol"])
            self.rtol = float(measurement["rtol"])
            atol_min = measurement.get("atol_min")
            rtol_min = measurement.get("rtol_min")
            self.atol_min = None if atol_min is None else float(atol_min)
            self.rtol_min = None if rtol_min is None else float(rtol_min)
            self.comparison = str(measurement["comparison"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "measurement_status": "measured" if self.measured else "not_emitted",
            "comparison_count": self.comparison_count,
            "comparison": self.comparison,
            "measured_error": {
                "absolute": _json_safe(self.max_absolute_error),
                "relative": _json_safe(self.max_relative_error),
                "normalized_to_tolerance": _json_safe(self.normalized_error),
            },
            "tolerance": {
                "atol": _json_safe(self.atol),
                "rtol": _json_safe(self.rtol),
                **({"atol_min": _json_safe(self.atol_min)}
                   if self.atol_min is not None else {}),
                **({"rtol_min": _json_safe(self.rtol_min)}
                   if self.rtol_min is not None else {}),
            },
        }


@dataclass
class LedgerState:
    output: Path
    started_at: str = field(default_factory=lambda: _now())
    current_nodeid: str | None = None
    selected: list[str] = field(default_factory=list)
    reports: dict[str, dict[str, Any]] = field(default_factory=dict)
    comparisons: dict[str, ComparisonSummary] = field(default_factory=dict)
    originals: dict[str, Any] = field(default_factory=dict)

    def comparison(self, actual: Any, desired: Any, *, rtol: Any, atol: Any, kind: str) -> None:
        if self.current_nodeid is None:
            return
        measurement = _measure(actual, desired, rtol=rtol, atol=atol, kind=kind)
        if measurement is not None:
            self.comparisons.setdefault(self.current_nodeid, ComparisonSummary()).observe(measurement)


_STATE: LedgerState | None = None


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _at_least_as_bad(candidate: float, current: float) -> bool:
    """Is `candidate` at least as bad a normalized error as `current`?

    NaN means a comparison produced a mismatch that could not even be
    evaluated (see `_measure`'s `equal_nonfinite` handling) and is always the
    worst possible outcome, ahead of any finite value or plain `inf`.
    """
    if math.isnan(candidate):
        return True
    if math.isnan(current):
        return False
    return candidate >= current


def _json_safe(value: float | None) -> float | str | None:
    """JSON has no token for inf/nan; encode them as the strings it can hold
    instead of collapsing them to `null`, which already means "not measured"."""
    if value is None or math.isfinite(value):
        return value
    if math.isnan(value):
        return "NaN"
    return "Infinity" if value > 0 else "-Infinity"


def _measure(actual: Any, desired: Any, *, rtol: Any, atol: Any, kind: str) -> dict[str, float | str | None] | None:
    try:
        actual_array, desired_array = np.broadcast_arrays(np.asarray(actual), np.asarray(desired))
        with np.errstate(invalid="ignore"):
            difference = np.abs(actual_array - desired_array).astype(float, copy=False)
        desired_abs = np.abs(desired_array).astype(float, copy=False)
        # NaN vs NaN is a legitimate pass in this package (na_action / missing-
        # data tests assert it), but `actual == desired` is False for NaN, so
        # without the explicit `both_nan` term below only inf-vs-same-sign-inf
        # would be rescued, and NaN-vs-NaN would report as an infinite error.
        both_nan = np.isnan(actual_array) & np.isnan(desired_array)
        equal_nonfinite = (
            (actual_array == desired_array) & ~np.isfinite(actual_array) & ~np.isfinite(desired_array)
        ) | both_nan
        difference = np.where(equal_nonfinite, 0.0, difference)
        # Keep the tolerances as arrays. `normalized_error` is `difference /
        # (atol + rtol*|desired|)` and must divide each element by the
        # tolerance that element was actually judged against. Reducing a vector
        # tolerance to its maximum first would divide every element by the
        # loosest entry in it, reporting a ratio smaller than the one the
        # assertion applied. The scalars below are only for reporting.
        atol_array = np.asarray(atol, dtype=float)
        rtol_array = np.asarray(rtol, dtype=float)
        atol_value = float(np.max(atol_array))
        rtol_value = float(np.max(rtol_array))
        atol_min = float(np.min(atol_array))
        rtol_min = float(np.min(rtol_array))
    except (TypeError, ValueError, OverflowError):
        return None
    if difference.size == 0:
        max_absolute = 0.0
        max_relative = 0.0
        normalized = 0.0
    else:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            relative = np.where(desired_abs > 0.0, difference / desired_abs, difference)
            allowed = atol_array + rtol_array * desired_abs
            normalized_array = np.where(allowed > 0.0, difference / allowed, np.where(difference == 0.0, 0.0, np.inf))
        # np.max (not nanmax): a NaN entry here is a real mismatch (one side
        # NaN, the other not -- the both-NaN case was already zeroed out
        # above), and must propagate as the worst possible result rather than
        # being skipped in favor of whatever the largest *finite* value was.
        max_absolute = float(np.max(difference))
        max_relative = float(np.max(relative))
        normalized = float(np.max(normalized_array))
    return {
        "comparison": kind,
        "max_absolute_error": max_absolute,
        "max_relative_error": max_relative,
        "normalized_error": normalized,
        "atol": atol_value,
        "rtol": rtol_value,
        # Present only when the assertion passed a per-element tolerance, so a
        # reader can tell "one tolerance" from "the loosest of several".
        **({"atol_min": atol_min} if atol_min != atol_value else {}),
        **({"rtol_min": rtol_min} if rtol_min != rtol_value else {}),
    }


def _command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _r_versions() -> dict[str, str | None]:
    package_names = ("mboost", "gamboostLSS", "stabs", "survival", "partykit", "TH.data")
    expression = (
        'cat(paste0("R=", as.character(getRversion()), "\\n")); '
        f"for (p in c({', '.join(json.dumps(name) for name in package_names)})) "
        'cat(paste0(p, "=", '
        'if (requireNamespace(p, quietly=TRUE)) as.character(packageVersion(p)) else "missing", "\\n"))'
    )
    output = _command_output(["Rscript", "-e", expression])
    versions: dict[str, str | None] = {"R": None, **dict.fromkeys(package_names)}
    if output is None:
        return versions
    for line in output.splitlines():
        name, separator, value = line.partition("=")
        if separator and name in versions:
            versions[name] = None if value == "missing" else value
    return versions


def _commit() -> str | None:
    return os.environ.get("GITHUB_SHA") or _command_output(["git", "rev-parse", "HEAD"])


def _validation_by_node() -> dict[str, list[dict[str, Any]]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    result: dict[str, list[dict[str, Any]]] = {}
    for record in manifest["validations"]:
        node_id = record["node_id"]
        if node_id.startswith("tests/"):
            result.setdefault(node_id, []).append(record)
    return result


def _metadata() -> dict[str, Any]:
    return {
        "commit": _commit(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
        },
        "versions": {
            "python": {
                "mboostrs": _package_version("mboostrs"),
                "jsonschema": _package_version("jsonschema"),
                "numpy": _package_version("numpy"),
                "pytest": _package_version("pytest"),
                "rpy2": _package_version("rpy2"),
            },
            "r": _r_versions(),
        },
        "ci": {
            "event": os.environ.get("GITHUB_EVENT_NAME"),
            "repository": os.environ.get("GITHUB_REPOSITORY"),
            "run_id": os.environ.get("GITHUB_RUN_ID"),
            "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "ref": os.environ.get("GITHUB_REF"),
        },
    }


def _install_wrappers(state: LedgerState) -> None:
    state.originals = {
        "assert_allclose": np.testing.assert_allclose,
        "assert_array_equal": np.testing.assert_array_equal,
        "allclose": np.allclose,
    }

    @functools.wraps(np.testing.assert_allclose)
    def assert_allclose(actual, desired, rtol=1e-7, atol=0, *args, **kwargs):
        state.comparison(actual, desired, rtol=rtol, atol=atol, kind="numpy.testing.assert_allclose")
        return state.originals["assert_allclose"](actual, desired, rtol, atol, *args, **kwargs)

    @functools.wraps(np.testing.assert_array_equal)
    def assert_array_equal(actual, desired, *args, **kwargs):
        state.comparison(actual, desired, rtol=0.0, atol=0.0, kind="numpy.testing.assert_array_equal")
        return state.originals["assert_array_equal"](actual, desired, *args, **kwargs)

    @functools.wraps(np.allclose)
    def allclose(actual, desired, rtol=1e-5, atol=1e-8, *args, **kwargs):
        state.comparison(actual, desired, rtol=rtol, atol=atol, kind="numpy.allclose")
        return state.originals["allclose"](actual, desired, rtol, atol, *args, **kwargs)

    np.testing.assert_allclose = assert_allclose
    np.testing.assert_array_equal = assert_array_equal
    np.allclose = allclose


def _restore_wrappers(state: LedgerState) -> None:
    if state.originals:
        np.testing.assert_allclose = state.originals["assert_allclose"]
        np.testing.assert_array_equal = state.originals["assert_array_equal"]
        np.allclose = state.originals["allclose"]


def pytest_addoption(parser):
    group = parser.getgroup("mboostrs parity ledger")
    group.addoption("--parity-ledger", metavar="PATH", help="write node-level parity evidence as JSON")
    group.addoption("--parity-scope", default="unspecified", help="scope label stored in the parity artifact")


def pytest_configure(config):
    global _STATE
    output = config.getoption("--parity-ledger")
    if output:
        _STATE = LedgerState(Path(output).resolve())
        _install_wrappers(_STATE)


def pytest_collection_finish(session):
    if _STATE is not None:
        _STATE.selected = [item.nodeid for item in session.items]


def pytest_runtest_call(item):
    if _STATE is not None:
        _STATE.current_nodeid = item.nodeid


def pytest_runtest_teardown(item, nextitem):
    if _STATE is not None:
        _STATE.current_nodeid = None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if _STATE is None:
        return
    existing = _STATE.reports.setdefault(item.nodeid, {"duration_seconds": 0.0})
    existing["duration_seconds"] += report.duration
    if report.when == "call" or (report.when == "setup" and report.skipped):
        status = report.outcome
        if hasattr(report, "wasxfail"):
            status = "xfailed" if report.skipped else "xpassed"
        existing["status"] = status
        if report.failed:
            existing["failure"] = str(report.longrepr)[:8000]
        elif report.skipped:
            existing["reason"] = str(report.longrepr)[:2000]


def pytest_sessionfinish(session, exitstatus):
    if _STATE is None:
        return
    validations = _validation_by_node()
    results = []
    for node_id in _STATE.selected:
        report = _STATE.reports.get(node_id, {})
        comparison = _STATE.comparisons.get(node_id, ComparisonSummary()).to_dict()
        contracts = validations.get(node_id, [])
        result = {
            "node_id": node_id,
            "status": report.get("status", "not_run"),
            "duration_seconds": round(float(report.get("duration_seconds", 0.0)), 9),
            **comparison,
            "validation_ids": [record["id"] for record in contracts],
            "compared_outputs": sorted({output for record in contracts for output in record["compared_outputs"]}),
            "declared_tolerances": [
                {"validation_id": record["id"], "atol": record["atol"], "rtol": record["rtol"]} for record in contracts
            ],
        }
        if "failure" in report:
            result["failure"] = report["failure"]
        if "reason" in report:
            result["reason"] = report["reason"]
        results.append(result)
    counts: dict[str, int] = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "mboostrs-parity-result",
        "started_at": _STATE.started_at,
        "finished_at": _now(),
        "scope": session.config.getoption("--parity-scope"),
        "pytest_exit_status": int(exitstatus),
        "summary": {"selected": len(results), "status_counts": counts},
        **_metadata(),
        "results": results,
    }
    _STATE.output.parent.mkdir(parents=True, exist_ok=True)
    _STATE.output.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def pytest_unconfigure(config):
    global _STATE
    if _STATE is not None:
        _restore_wrappers(_STATE)
        _STATE = None
