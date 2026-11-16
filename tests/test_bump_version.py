"""The bump command keeps every version location in agreement (#236).

`docs/book/check_docs.py` already fails the build when the locations
disagree; these tests cover the other half — that the bump rewrites all of
them to the new value, refuses a no-op, and refuses a malformed version —
so a future edit to the tool cannot leave one file behind silently.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOCATIONS = (
    "Cargo.toml",
    "pyproject.toml",
    "python/mboostrs/__init__.py",
    "mboostrs/DESCRIPTION",
)
#: The same patterns the guard in `docs/book/check_docs.py` matches; the
#: fixture asserts with the guard's eyes, not the tool's.
GUARD_PATTERNS = {
    "Cargo.toml": r'^version\s*=\s*"([^"]+)"',
    "pyproject.toml": r'^version\s*=\s*"([^"]+)"',
    "python/mboostrs/__init__.py": r'^__version__\s*=\s*"([^"]+)"',
    "mboostrs/DESCRIPTION": r"^Version:\s*(\S+)",
}


@pytest.fixture
def repo_copy(tmp_path):
    """A minimal tree with the four real files and the real tool."""
    for relative in LOCATIONS:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / relative, destination)
    (tmp_path / "tools").mkdir()
    shutil.copy(ROOT / "tools/bump_version.py", tmp_path / "tools/bump_version.py")
    return tmp_path


def guard_versions(root):
    return {
        name: re.search(pattern, (root / name).read_text(), re.MULTILINE).group(1)
        for name, pattern in GUARD_PATTERNS.items()
    }


def test_bump_rewrites_every_location(repo_copy):
    result = subprocess.run(
        [sys.executable, str(repo_copy / "tools/bump_version.py"), "9.9.9"],
        capture_output=True, text=True, cwd=repo_copy,
    )
    assert result.returncode == 0, result.stderr

    versions = guard_versions(repo_copy)
    assert set(versions.values()) == {"9.9.9"}
    assert "9.9.9" in result.stdout


def test_bump_keeps_the_rest_of_each_file(repo_copy):
    before = {
        name: (repo_copy / name).read_text() for name in LOCATIONS
    }
    subprocess.run(
        [sys.executable, str(repo_copy / "tools/bump_version.py"), "9.9.9"],
        capture_output=True, text=True, cwd=repo_copy, check=True,
    )
    for name, old in before.items():
        new = (repo_copy / name).read_text()
        old_lines = [
            line for line in old.splitlines()
            if not re.search(GUARD_PATTERNS[name], line)
        ]
        new_lines = [
            line for line in new.splitlines()
            if not re.search(GUARD_PATTERNS[name], line)
        ]
        assert old_lines == new_lines, f"{name} changed beyond the version line"


def test_bump_refuses_a_no_op(repo_copy):
    current = next(iter(guard_versions(repo_copy).values()))
    result = subprocess.run(
        [sys.executable, str(repo_copy / "tools/bump_version.py"), current],
        capture_output=True, text=True, cwd=repo_copy,
    )
    assert result.returncode != 0
    assert "nothing to do" in result.stderr
    assert set(guard_versions(repo_copy).values()) == {current}


def test_bump_refuses_a_malformed_version(repo_copy):
    result = subprocess.run(
        [sys.executable, str(repo_copy / "tools/bump_version.py"), "next"],
        capture_output=True, text=True, cwd=repo_copy,
    )
    assert result.returncode != 0
    assert "not a version" in result.stderr


def test_a_missing_line_fails_the_bump(repo_copy):
    """A pattern that finds nothing must exit loudly, not skip the file."""
    init = repo_copy / "python/mboostrs/__init__.py"
    text = init.read_text().splitlines()
    init.write_text(
        "\n".join(
            line for line in text if "__version__" not in line
        ) + "\n"
    )
    result = subprocess.run(
        [sys.executable, str(repo_copy / "tools/bump_version.py"), "9.9.9"],
        capture_output=True, text=True, cwd=repo_copy,
    )
    assert result.returncode != 0
    assert "matched 0 lines" in result.stderr


def test_the_repository_files_still_agree():
    """The guard's own invariant, checked the way the guard checks it."""
    assert len(set(guard_versions(ROOT).values())) == 1


def test_the_tool_documents_itself():
    """`--help` names the four files, so the tool is discoverable."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/bump_version.py"), "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "version" in result.stdout
