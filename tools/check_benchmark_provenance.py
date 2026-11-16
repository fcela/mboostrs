#!/usr/bin/env python3
"""Check that a published timing says where it came from.

P3.5: "Refresh benchmark artifacts after current changes; older July artifacts
cannot support current headlines. Pin thread count, CPU target, versions,
dirty-tree state, input and output digests, warmup, repetitions, uncertainty,
and memory scope."

The repository already has that contract. `docs/schemas/benchmark-result.schema.json`
requires commit, platform, versions, compiler flags, threads, warmup policy,
input digests and per-measurement correctness, and
`benches/benchmark_contract.py` builds and validates it.

The eight artifacts the book actually renders carry **none of it**. Not a
commit, not a date, not the versions of the libraries being compared. One of
them carries a `timing_note` saying two configurations were re-measured
because "their first measurement overlapped another job" -- the drift the
paired protocol exists to cancel, noticed by hand and recorded in prose.

So a reader cannot tell whether a published speedup was measured on the code
they have, and neither can we: the numbers are undated.

Re-running them is a machine-dependent job that has to happen on one quiet
box, so this does not invent the missing fields. It records the gap per
artifact and fails when it grows: a new artifact must carry provenance, and an
existing one may only gain fields, never lose them.

Run from the repository root:

    python tools/check_benchmark_provenance.py
    python tools/check_benchmark_provenance.py --report
    python tools/check_benchmark_provenance.py --update
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "docs" / "book" / "cases"
BASELINE = ROOT / "docs" / "book" / "benchmark_provenance_baseline.json"

#: What a published timing has to say about itself, and why each one matters.
#: The names match `docs/schemas/benchmark-result.schema.json` where they
#: overlap; this is the subset a rendered artifact needs to be re-runnable.
FIELDS = {
    "created_at": "when it was measured, so a reader can tell whether it "
                  "describes the code they have",
    "commit": "which commit, and whether the tree was dirty",
    "platform": "the machine and CPU, since a ratio is not portable",
    "versions": "the versions compared -- a library comparison without them "
                "cannot be repeated",
    "threads": "the thread count, which changes every number here",
    "protocol": "sample cap, tuning budget, metric, split, and search, held "
                "fixed across the arms",
    "repeats": "how many repetitions the summary is over",
    "uncertainty": "a spread or interval, not a bare point estimate",
    "row_identity": "each result row's measurement time and build fingerprint, "
                    "so assembling cached rows cannot relabel when, or under "
                    "which build, a number was measured",
}


def artifacts() -> list[pathlib.Path]:
    return sorted(ARTIFACTS.glob("bench_*.json"))


def _filled(value) -> bool:
    return value is not None and value != "" and value != {} and value != []


def present_fields(path: pathlib.Path) -> set[str]:
    """Which provenance fields an artifact carries with a value.

    A field counts only in the artifact's top-level provenance record (the
    ``provenance`` object, or the top level when there is none) and only when
    it holds a value: a key found somewhere deep in a result row, or an empty
    one, says nothing about the artifact. ``row_identity`` counts when every
    result row carries its own fingerprint and measurement time, which the
    job harness (`benches/_jobs.py`) writes.
    """
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    if not isinstance(document, dict):
        return set()
    record = document.get("provenance")
    if not isinstance(record, dict):
        record = document
    found = {key for key in FIELDS if key != "row_identity" and _filled(record.get(key))}
    rows = document.get("results")
    if isinstance(rows, list) and rows and all(
            isinstance(row, dict) and isinstance(row.get("_identity"), dict)
            and row["_identity"].get("fingerprint") and row["_identity"].get("measured_at")
            for row in rows):
        found.add("row_identity")
    return found


def last_commit(path: pathlib.Path) -> dict[str, str]:
    """When the artifact last changed, from git.

    Not the same claim as `created_at` -- it is an upper bound on when the
    numbers were measured, since a file can be committed later than it was
    written. It is recoverable, which an invented measurement date is not, and
    it answers the question a reader actually has: can this describe the code
    I have?
    """
    import subprocess

    result = subprocess.run(
        ["git", "log", "-1", "--format=%H %ad", "--date=short", "--",
         str(path.relative_to(ROOT))],
        capture_output=True, text=True, cwd=ROOT,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return {}
    commit, _, date = result.stdout.strip().partition(" ")
    return {"last_commit": commit[:12], "last_committed": date}


def audit() -> dict[str, list[str]]:
    """`artifact -> sorted missing fields`."""
    return {
        path.name: sorted(set(FIELDS) - present_fields(path))
        for path in artifacts()
    }


def load_baseline() -> dict[str, list[str]]:
    if not BASELINE.exists():
        return {}
    return json.loads(BASELINE.read_text(encoding="utf-8"))["missing"]


def write_baseline(current: dict[str, list[str]]) -> None:
    total = sum(len(missing) for missing in current.values())
    BASELINE.write_text(
        json.dumps({
            "artifacts": len(current),
            "missing_fields": total,
            "missing": current,
            # Recovered from git rather than invented, and labelled as what it
            # is. The artifacts carry no measurement date; this is the latest
            # date each could have been measured.
            "last_change": {name: last_commit(ARTIFACTS / name)
                            for name in current},
        }, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true",
                        help="print the gap per artifact and stop")
    parser.add_argument("--update", action="store_true",
                        help="record the current gap as the new baseline")
    arguments = parser.parse_args()

    current = audit()
    if arguments.report:
        for name, missing in current.items():
            when = last_commit(ARTIFACTS / name).get("last_committed", "?")
            print(f"{name}: last changed {when}; {len(missing)} missing — "
                  f"{', '.join(missing) or 'none'}")
        total = sum(len(m) for m in current.values())
        print(f"{len(current)} artifact(s), {total} missing field(s) of "
              f"{len(current) * len(FIELDS)}")
        return 0
    if arguments.update:
        write_baseline(current)
        print(f"recorded {BASELINE.relative_to(ROOT)}")
        return 0

    baseline = load_baseline()
    problems = []
    for name, missing in current.items():
        allowed = set(baseline.get(name, []))
        if name not in baseline:
            if missing:
                problems.append(
                    f"{name} is new and carries no {', '.join(missing)}. A "
                    f"published timing has to say where it came from; see "
                    f"benches/benchmark_contract.py."
                )
            continue
        grown = sorted(set(missing) - allowed)
        if grown:
            problems.append(
                f"{name} lost {', '.join(grown)}, which it used to record")
    for name in baseline:
        if name not in current:
            problems.append(f"{name} is in the baseline and no longer exists; "
                            f"run --update")

    if problems:
        print("Benchmark provenance check failed:", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1
    total = sum(len(m) for m in current.values())
    print(f"Benchmark provenance: {len(current)} artifact(s), {total} "
          f"missing field(s), none worse than recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
