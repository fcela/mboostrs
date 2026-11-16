"""Contract test for isolated stability-selection thread evidence."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benches.benchmark_contract import load_scenarios, validate_result  # noqa: E402


def test_thread_scaling_runner_is_correctness_stable_and_completes_manifest(tmp_path):
    output = tmp_path / "thread-scaling.json"
    completed = subprocess.run(
        [sys.executable, "benches/bench_thread_scaling.py", "--quick", "--output", str(output)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    validate_result(payload)
    assert len(payload["results"]) == 3
    digests = {
        measurement["correctness_digest"]
        for result in payload["results"]
        for measurement in result["measurements"]
    }
    assert len(digests) == 1
    assert len(payload["results"][-1]["measurements"]) == 4
    assert all(row["status"] == "implemented" for row in load_scenarios()["scenarios"])
