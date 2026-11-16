#!/usr/bin/env python3
"""Verify that the real-data refresh kept its comparison rows unchanged.

Run after ``python benches/refresh_autospec.py --studies realdata``. The
refresh intentionally remeasures only autospec variants; other methods
retain their original results and provenance. This check catches a partial
or mislabeled assembly before the book consumes it.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/book/cases/bench_realdata.json"
REFRESHED = ROOT / "docs/book/cases/bench_realdata_preexploration.json"


def validate() -> dict[str, int]:
    """Return fresh autospec status counts or raise for an invalid assembly."""
    source_bytes = SOURCE.read_bytes()
    original = json.loads(source_bytes)
    refreshed = json.loads(REFRESHED.read_text())
    details = refreshed["protocol"]["refresh"]
    if details["source"] != SOURCE.name:
        raise ValueError("refresh names a different source artifact")
    if details["source_sha256"] != hashlib.sha256(source_bytes).hexdigest():
        raise ValueError("source bytes changed after the refresh began")
    if refreshed["original_provenance"] != original["provenance"]:
        raise ValueError("the original provenance was changed")

    before = original["results"]
    after = refreshed["results"]
    if len(after) != len(before):
        raise ValueError(f"expected {len(before)} result rows, found {len(after)}")
    statuses = Counter()
    refreshed_count = 0
    for index, (old, row) in enumerate(zip(before, after, strict=True)):
        old_key = (old["data_id"], old["method"], old["seed"])
        new_key = (row["data_id"], row["method"], row["seed"])
        if old_key != new_key:
            raise ValueError(f"row {index} changed its dataset, method or seed")
        if "autospec" in old["method"]:
            refreshed_count += 1
            if row.get("refresh_status") != "current autospec":
                raise ValueError(f"autospec row {index} is not marked as remeasured")
            identity = row.get("_identity", {})
            if not identity.get("fingerprint") or not identity.get("measured_at"):
                raise ValueError(f"autospec row {index} lacks job provenance")
            status = row.get("_status")
            if status not in {"success", "fit_failure", "crash", "timeout"}:
                raise ValueError(f"autospec row {index} has invalid status {status!r}")
            statuses[status] += 1
        elif row != {**old, "refresh_status": "retained original measurement"}:
            raise ValueError(f"comparison row {index} was changed")
    if details["rows_remeasured"] != refreshed_count:
        raise ValueError("protocol's refreshed-row count differs from results")
    return dict(statuses)


if __name__ == "__main__":
    print(json.dumps({"refresh_status": validate()}, sort_keys=True))
