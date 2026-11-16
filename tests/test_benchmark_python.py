"""Integration test for Python-only benchmark scenario evidence."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benches.benchmark_contract import load_scenarios, validate_result  # noqa: E402


def test_python_scenario_runner_covers_every_implemented_python_row(tmp_path):
    output = tmp_path / "python-benchmark.json"
    completed = subprocess.run(
        [sys.executable, "benches/bench_python.py", "--quick", "--output", str(output)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    validate_result(payload)
    expected = {
        row["id"]
        for row in load_scenarios()["scenarios"]
        if row["runner"] == "python" and row["status"] == "implemented"
    }
    assert {row["scenario_id"] for row in payload["results"]} == expected
    assert len(expected) == 20
