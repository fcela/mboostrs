import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "release_notes.py"

CHANGELOG = """# Changelog

## [0.2.0] - pending

### Added

- New feature.

### Detailed notes

Long text that must not appear.

## [Unreleased]

## [0.1.3] - 2026-06-25

### Fixed

- Old fix.
"""


def run(tmp_path, version):
    path = tmp_path / "CHANGELOG.md"
    path.write_text(CHANGELOG, encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(SCRIPT), version, str(path)],
        capture_output=True,
        text=True,
    )


def test_top_section_without_detailed_notes(tmp_path):
    r = run(tmp_path, "0.2.0")
    assert r.returncode == 0
    assert r.stdout.startswith("## [0.2.0] - pending")
    assert "New feature." in r.stdout
    assert "Detailed notes" not in r.stdout
    assert "Long text" not in r.stdout
    assert "Old fix." not in r.stdout


def test_older_section_and_missing_version(tmp_path):
    r = run(tmp_path, "0.1.3")
    assert "Old fix." in r.stdout and "New feature." not in r.stdout
    assert run(tmp_path, "9.9.9").returncode == 1
