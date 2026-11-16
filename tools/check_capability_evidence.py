#!/usr/bin/env python3
"""Check that the capability ledger's evidence still exists.

P1.1: "the exact test node and last successful artifact ... Make stale
`last_reviewed` dates and tests that no longer collect fail CI."

Every row in the capability tables is backed by `validation_ids`, and every
validation record names a `node_id` -- a pytest node on the Python side, an R
script on the R side. That link is the difference between a status the ledger
asserts and a status something checks. Nothing verified the link, so a renamed
test, a deleted parametrisation, or a moved file would leave the ledger
claiming evidence that no longer runs, and the claim would keep being printed
in the capability tables and the review packet.

Four things are checked:

* every Python `node_id` still collects under pytest, run once for all of
  them rather than per node;
* every R `node_id` names a file that exists and carries the matching
  `#@validates <id>` marker on the script itself;
* every `#@validates` marker in an R script names a validation the manifest
  records, so evidence cannot be claimed from the test side either;
* `last_reviewed` is not older than the last commit that changed the
  manifest. A ledger edited after its own review date is describing a state
  nobody looked at;
* no operation is marked `verified` on the strength of the *other frontend*.
  `GaussianKnownVarLSS` claimed a verified R fit and a verified R prediction,
  and the only supporting script compares `predict(type="link")` against three
  numbers copied from a Python run. That is a real check -- the two frontends
  assemble the same call -- and it is not what `verified` means, which is a
  numerical comparison against an outside reference. The Python side of the
  same family said `smoke`, and an R frontend cannot be better verified than
  the implementation it is checked against.

The date rule falls back to an age limit where git history is unavailable, so
a shallow checkout still fails on a genuinely old review rather than passing
because it cannot tell.

Run from the repository root:

    python tools/check_capability_evidence.py
    python tools/check_capability_evidence.py --show
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "python" / "mboostrs" / "capability_manifest.json"
R_TESTS = ROOT / "mboostrs" / "tests"

#: A review older than this is stale whatever the git history says. Used when
#: git cannot be reached; the commit-date rule is the sharper one.
MAX_REVIEW_AGE_DAYS = 180


def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def collected_nodes() -> set[str] | None:
    """Every pytest node id, or `None` if collection itself failed."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         "--no-header", "-p", "no:cacheprovider"],
        capture_output=True, text=True, cwd=ROOT,
    )
    if result.returncode != 0:
        return None
    return {line.strip() for line in result.stdout.splitlines()
            if "::" in line}


def check_python_nodes(records: list[dict], show: bool) -> list[str]:
    wanted = [record for record in records
              if record["node_id"].startswith("tests/")]
    collected = collected_nodes()
    if collected is None:
        return ["pytest collection failed, so the ledger's evidence cannot "
                "be checked; fix collection first"]
    missing = [record for record in wanted
               if record["node_id"] not in collected]
    if show and not missing:
        print(f"  {len(wanted)} python node(s) collect")
    return [
        f"validation `{record['id']}` cites "
        f"{record['node_id']}, which pytest does not collect. The ledger "
        f"claims evidence that no longer runs."
        for record in missing
    ]


def r_markers() -> dict[str, list[str]]:
    markers: dict[str, list[str]] = {}
    for path in sorted(R_TESTS.glob("*.R")):
        relative = str(path.relative_to(ROOT))
        for name in re.findall(r"#@validates\s+([\w.]+)",
                               path.read_text(encoding="utf-8")):
            markers.setdefault(name, []).append(relative)
    return markers


def check_rust_nodes(records: list[dict], show: bool) -> list[str]:
    """Rust evidence, as `<file>::<test module>`.

    The core crate is published on its own, so a contract it enforces can only
    be checked where the contract lives: a Python test cannot show that
    `QuantReg::try_new` rejects a `tau`, because the Python constructor
    rejects it first.
    """
    wanted = [record for record in records
              if record["node_id"].startswith("crates/")]
    problems = []
    for record in wanted:
        source, _, module = record["node_id"].partition("::")
        path = ROOT / source
        if not module:
            problems.append(
                f"validation `{record['id']}` names {record['node_id']}, "
                f"which does not say which test module")
            continue
        if not path.is_file():
            problems.append(
                f"validation `{record['id']}` cites {source}, which does not "
                f"exist")
            continue
        if f"mod {module} {{" not in path.read_text(encoding="utf-8"):
            problems.append(
                f"validation `{record['id']}` cites {record['node_id']}, and "
                f"{source} has no `mod {module}`")
    if show and not problems:
        print(f"  {len(wanted)} Rust node(s) name a test module that exists")
    return problems


def check_r_nodes(records: list[dict], show: bool) -> list[str]:
    wanted = [record for record in records
              if not record["node_id"].startswith(("tests/", "crates/"))]
    markers = r_markers()
    problems = []
    for record in wanted:
        path = ROOT / record["node_id"]
        if not path.exists():
            problems.append(
                f"validation `{record['id']}` cites {record['node_id']}, "
                f"which does not exist")
            continue
        files = markers.get(record["id"], [])
        if record["node_id"] not in files:
            problems.append(
                f"{record['node_id']} does not carry `#@validates "
                f"{record['id']}`, so the script and the ledger disagree "
                f"about what it is evidence for"
            )
    if show and not problems:
        print(f"  {len(wanted)} R node(s) exist and are marked")
    return problems


def check_orphan_markers(records: list[dict], show: bool) -> list[str]:
    known = {record["id"] for record in records}
    orphans = sorted(set(r_markers()) - known)
    if show and not orphans:
        print(f"  {len(r_markers())} R marker(s) match a recorded validation")
    return [
        f"`#@validates {name}` appears in {', '.join(r_markers()[name])} and "
        f"is not a validation the manifest records"
        for name in orphans
    ]


#: An oracle naming one of these is an outside reference: another package,
#: a numerical library, or a derivation. An oracle that names the sibling
#: frontend is not, however carefully the comparison is done.
EXTERNAL_ORACLE = (
    "mboost", "gamboostLSS", "gamlss", "scipy", "SciPy", "fields",
    "analytical", "finite-difference", "invariant", "identity",
)

#: The sibling frontend, spelled as the oracle field spells it.
SIBLING_ORACLE = ("mboostrs Python frontend", "mboostrs R frontend")


def check_verified_has_an_outside_oracle(document: dict,
                                         show: bool) -> list[str]:
    """`verified` is a comparison against something other than ourselves."""
    records = {record["id"]: record for record in document["validations"]}
    problems = []
    checked = 0
    # `families` and `learners` are lists of rows; `methods` is keyed by
    # frontend and carries no per-row support block.
    for table in document["tables"].values():
        for row in (table if isinstance(table, list) else []):
            for frontend, support in row.get("support", {}).items():
                ids = support.get("validation_ids", [])
                cited = [records[name] for name in ids if name in records]
                for operation, status in support.get("operations", {}).items():
                    if status != "verified":
                        continue
                    checked += 1
                    outside = [
                        record for record in cited
                        if any(word in record["oracle"]
                               for word in EXTERNAL_ORACLE)
                        and not any(word in record["oracle"]
                                    for word in SIBLING_ORACLE)
                    ]
                    if not outside:
                        problems.append(
                            f"{row.get('name', '?')} marks {frontend} "
                            f"{operation} `verified`, and none of its "
                            f"validations names an oracle outside mboostrs. "
                            f"Cross-frontend agreement is `smoke`."
                        )
    if show and not problems:
        print(f"  {checked} `verified` status(es) cite an outside oracle")
    return problems


def manifest_commit_date() -> datetime.date | None:
    result = subprocess.run(
        ["git", "log", "-1", "--format=%ad", "--date=short", "--",
         str(MANIFEST.relative_to(ROOT))],
        capture_output=True, text=True, cwd=ROOT,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return datetime.date.fromisoformat(result.stdout.strip())


def check_review_date(document: dict, show: bool) -> list[str]:
    reviewed = datetime.date.fromisoformat(document["last_reviewed"])
    changed = manifest_commit_date()
    if changed is not None:
        if reviewed < changed:
            return [
                f"the ledger was last reviewed on {reviewed} and last changed "
                f"on {changed}. Re-review it and set `last_reviewed`, or the "
                f"tables state a status nobody has looked at since it moved."
            ]
        if show:
            print(f"  reviewed {reviewed}, last changed {changed}")
        return []

    age = (datetime.date.today() - reviewed).days
    if age > MAX_REVIEW_AGE_DAYS:
        return [f"the ledger was last reviewed {age} days ago (limit "
                f"{MAX_REVIEW_AGE_DAYS}); git history was unavailable to "
                f"compare against its last change"]
    if show:
        print(f"  reviewed {reviewed} ({age} days ago; no git history)")
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true",
                        help="print each check as it passes")
    arguments = parser.parse_args()

    document = manifest()
    records = document["validations"]
    problems = (check_python_nodes(records, arguments.show)
                + check_rust_nodes(records, arguments.show)
                + check_r_nodes(records, arguments.show)
                + check_orphan_markers(records, arguments.show)
                + check_verified_has_an_outside_oracle(document,
                                                       arguments.show)
                + check_review_date(document, arguments.show))

    if problems:
        print("Capability evidence check failed:", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1
    print(f"Capability evidence: {len(records)} validation(s), every node "
          f"resolves, reviewed {document['last_reviewed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
