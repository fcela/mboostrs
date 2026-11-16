#!/usr/bin/env python3
"""Set the package version in the four files that carry it.

The guard already exists: `docs/book/check_docs.py` reads the version from
`Cargo.toml`, `pyproject.toml`, `python/mboostrs/__init__.py` and
`mboostrs/DESCRIPTION`, fails the build when they disagree, and CI runs it.
What was missing is the one command that keeps them agreeing in the first
place: a release bump edited four files by hand, and any missed file turned
CI red for a reason that has nothing to do with the change (#236).

    python tools/bump_version.py 0.2.0
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: (file, regex, replacement) for every location the guard in
#: `docs/book/check_docs.py` reads. The patterns here are the same ones the
#: guard matches, so a pattern that stops finding the line fails the bump
#: loudly instead of silently editing nothing while CI stays green.
LOCATIONS = (
    (ROOT / "Cargo.toml",
     r'^version\s*=\s*"[^"]+"',
     'version = "{version}"'),
    (ROOT / "pyproject.toml",
     r'^version\s*=\s*"[^"]+"',
     'version = "{version}"'),
    (ROOT / "python/mboostrs/__init__.py",
     r'^__version__\s*=\s*"[^"]+"',
     '__version__ = "{version}"'),
    (ROOT / "mboostrs/DESCRIPTION",
     r"^Version:\s*\S+",
     "Version: {version}"),
)

VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-+][\w.]+)?$")


def read_current_version():
    """The version of the first location, which all others must match."""
    for path, pattern, _ in LOCATIONS:
        match = re.search(pattern, path.read_text(encoding="utf-8"), re.MULTILINE)
        if match is None:
            sys.exit(f"error: {pattern} found nothing in {path}")
        return match.group(0)
    raise AssertionError


def write_version(new_version):
    """Rewrite every location; return the list of files written."""
    written = []
    for path, pattern, replacement in LOCATIONS:
        text = path.read_text(encoding="utf-8")
        new_text, count = re.subn(
            pattern, replacement.format(version=new_version), text, count=1,
            flags=re.MULTILINE,
        )
        if count != 1:
            sys.exit(f"error: {pattern} matched {count} lines in {path}")
        path.write_text(new_text, encoding="utf-8")
        written.append(path)
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Set the package version in every file that carries it."
    )
    parser.add_argument("version", help="new version, e.g. 0.2.0")
    args = parser.parse_args(argv)

    if not VERSION_RE.match(args.version):
        sys.exit(f"error: {args.version!r} is not a version like 0.2.0")
    current = read_current_version()
    if args.version in current:
        sys.exit(f"error: already at {args.version}; nothing to do")

    for path in write_version(args.version):
        print(f"bumped {path.relative_to(ROOT)}")
    print(f"{current} -> {args.version}")


if __name__ == "__main__":
    main()
