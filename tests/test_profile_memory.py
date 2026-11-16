"""Contract tests for measured stage-profile and memory artifacts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]


def test_profile_memory_quick_artifact_has_measured_shares_rss_and_cache(tmp_path):
    output = tmp_path / "profile-memory.json"
    completed = subprocess.run(
        [sys.executable, "benches/profile_memory.py", "--quick", "--output", str(output)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    schema = json.loads((ROOT / "docs/schemas/profile-memory-result.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(payload)
    assert len(payload["results"]) == 3
    for row in payload["results"]:
        assert abs(sum(row["phase_share"].values()) - 1.0) < 1e-12
        assert row["peak_rss_bytes"] >= row["baseline_rss_bytes"]
        assert row["factor_cache_bytes"] > 0
    gram = next(row for row in payload["results"] if row["route"] == "gaussian_gram")
    general = next(row for row in payload["results"] if row["route"] == "general_block")
    assert gram["gram_cache_bytes"] > 0
    assert general["gram_cache_bytes"] == 0


def test_committed_allocation_probe_matches_schema():
    payload = json.loads((ROOT / "docs/book/data/allocation-2026-07-15.json").read_text(encoding="utf-8"))
    schema = json.loads((ROOT / "docs/schemas/allocation-result.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(payload)
    assert payload["packed_scalar"]["allocations"] > 0
    assert payload["gaussian_gram"]["allocations"] > payload["packed_scalar"]["allocations"]
