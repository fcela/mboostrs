#!/usr/bin/env python3
"""Run an mboostrs parity scope and validate its evidence artifact."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

ROOT = Path(__file__).resolve().parents[1]
FOCUSED = ROOT / "tests/parity_focus.json"
RESULT_SCHEMA = ROOT / "docs/schemas/parity-result.schema.json"
FULL_MINIMUM = 232
REQUIRED_R_PACKAGES = ("mboost", "gamboostLSS", "stabs", "survival", "partykit", "TH.data")


class LedgerError(RuntimeError):
    """Raised when pytest did not produce the promised evidence contract."""


def focused_nodes() -> list[str]:
    payload = json.loads(FOCUSED.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise LedgerError("tests/parity_focus.json has an unsupported schema")
    nodes = [row["node_id"] for row in payload.get("tests", [])]
    if not nodes or len(nodes) != len(set(nodes)):
        raise LedgerError("focused parity nodes must be non-empty and unique")
    return nodes


def pytest_command(scope: str, output: Path, extra: list[str]) -> list[str]:
    selection = ["tests", "-m", "r_parity"] if scope == "full" else focused_nodes()
    return [
        sys.executable,
        "-m",
        "pytest",
        "--run-r-parity",
        "-p",
        "tools.parity_ledger",
        f"--parity-ledger={output}",
        f"--parity-scope={scope}",
        *selection,
        *extra,
    ]


def validate_ledger(payload: dict[str, Any], *, scope: str, require_dependencies: bool) -> None:
    schema = json.loads(RESULT_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)
    required = {
        "schema_version",
        "kind",
        "scope",
        "summary",
        "commit",
        "platform",
        "versions",
        "results",
    }
    missing = required - set(payload)
    if missing:
        raise LedgerError(f"parity ledger lacks fields: {sorted(missing)}")
    if payload["schema_version"] != 1 or payload["kind"] != "mboostrs-parity-result":
        raise LedgerError("parity ledger schema/kind mismatch")
    if payload["scope"] != scope:
        raise LedgerError(f"parity scope mismatch: expected {scope}, found {payload['scope']}")
    results = payload["results"]
    expected = FULL_MINIMUM if scope == "full" else len(focused_nodes())
    if len(results) < expected:
        raise LedgerError(f"parity ledger selected {len(results)} tests; expected at least {expected}")
    node_ids = [row.get("node_id") for row in results]
    if len(node_ids) != len(set(node_ids)):
        raise LedgerError("parity ledger contains duplicate node IDs")
    for row in results:
        row_required = {
            "node_id",
            "status",
            "duration_seconds",
            "measurement_status",
            "comparison_count",
            "measured_error",
            "tolerance",
            "validation_ids",
            "compared_outputs",
        }
        if missing_row := row_required - set(row):
            raise LedgerError(f"result {row.get('node_id')} lacks fields: {sorted(missing_row)}")
        if row["measurement_status"] == "measured":
            if row["measured_error"]["absolute"] is None or row["tolerance"]["rtol"] is None:
                raise LedgerError(f"measured result has incomplete error/tolerance: {row['node_id']}")
    if require_dependencies:
        r_versions = payload["versions"]["r"]
        missing_r = [name for name in REQUIRED_R_PACKAGES if not r_versions.get(name)]
        if not r_versions.get("R") or missing_r:
            raise LedgerError(f"required R parity dependencies are missing: {missing_r}")
        if not payload["versions"]["python"].get("rpy2"):
            raise LedgerError("required Python parity dependency rpy2 is missing")


def summary_markdown(payload: dict[str, Any]) -> str:
    counts = payload["summary"]["status_counts"]
    results = payload["results"]
    measured = sum(row["measurement_status"] == "measured" for row in results)
    failures = [row for row in results if row["status"] in {"failed", "xpassed", "not_run"}]
    skips = [row for row in results if row["status"] in {"skipped", "xfailed"}]
    r_versions = payload["versions"]["r"]
    lines = [
        "# mboostrs parity evidence",
        "",
        f"- Scope: `{payload['scope']}`",
        f"- Commit: `{payload.get('commit') or 'unknown'}`",
        f"- Selected: {len(results)}",
        f"- Statuses: {', '.join(f'{name}={count}' for name, count in sorted(counts.items()))}",
        f"- Numeric measurement coverage: {measured}/{len(results)} tests",
        f"- R: {r_versions.get('R') or 'missing'}; mboost: {r_versions.get('mboost') or 'missing'}; "
        f"gamboostLSS: {r_versions.get('gamboostLSS') or 'missing'}",
    ]
    if failures:
        lines.extend(["", "## Failures", *[f"- `{row['node_id']}`: {row['status']}" for row in failures]])
    if skips:
        lines.extend(["", "## Skips / expected failures", *[f"- `{row['node_id']}`: {row['status']}" for row in skips]])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("full", "focused"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-dependencies", action="store_true")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER, help="arguments after -- are passed to pytest")
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    extra = args.pytest_args[1:] if args.pytest_args[:1] == ["--"] else args.pytest_args
    command = pytest_command(args.scope, output, extra)
    print("Running:", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if not output.is_file():
        print("parity ledger was not produced", file=sys.stderr)
        return completed.returncode or 2
    try:
        payload = json.loads(output.read_text(encoding="utf-8"))
        validate_ledger(payload, scope=args.scope, require_dependencies=args.require_dependencies)
    except (json.JSONDecodeError, LedgerError, SchemaError, ValidationError) as exc:
        print(f"invalid parity ledger: {exc}", file=sys.stderr)
        return completed.returncode or 2
    summary = summary_markdown(payload)
    summary_path = output.with_name("parity-summary.md")
    summary_path.write_text(summary, encoding="utf-8")
    print(summary)
    if github_summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(github_summary).open("a", encoding="utf-8") as stream:
            stream.write(summary)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
