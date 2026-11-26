"""Integrity tests for the unified benchmark registry and CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "benches/registry.json"
VALID_TIERS = {"headline", "domain", "workflow", "calibration", "ablation"}
VALID_MODES = {"full", "mboostrs_only", "recorded_external"}


def test_benchmark_registry_paths_and_schema():
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    studies = data["studies"]
    assert studies, "registry must list at least one study"

    seen_ids = set()
    for study in studies:
        sid = study["id"]
        assert sid not in seen_ids, f"duplicate benchmark id: {sid}"
        seen_ids.add(sid)
        assert study["tier"] in VALID_TIERS, f"invalid tier on {sid}: {study['tier']}"
        assert study["refresh_mode"] in VALID_MODES, f"invalid mode on {sid}"

        runner = study.get("runner_script")
        if study["refresh_mode"] == "recorded_external":
            assert runner is None
        else:
            assert runner and (ROOT / runner).exists(), f"missing runner for {sid}: {runner}"
            assert study["command"], f"missing command for runnable study {sid}"

        for artifact in study.get("artifacts", ()):
            assert (ROOT / artifact).exists(), f"missing artifact for {sid}: {artifact}"
        for page in study.get("pages", ()):
            assert (ROOT / page).exists(), f"missing book page for {sid}: {page}"


def test_benchmark_runner_cli_list_status_and_dry_run():
    for args in (
        ["list"],
        ["list", "--tier", "headline"],
        ["status"],
        ["run", "distributional_comparison", "--dry-run", "--update-book"],
        ["update-book", "distributional_shootout", "--dry-run"],
    ):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "benches/run.py"), *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"benches/run.py {args} failed: {proc.stderr}"
