"""Print the release notes for one version from CHANGELOG.md.

Usage: python tools/release_notes.py VERSION [CHANGELOG]

VERSION has no leading "v". The section starts at the heading "## [VERSION]"
and ends at the next "## " heading or at a "### Detailed notes" subsection,
whichever comes first. Exit status 1 and no output when the heading is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path


def extract(version: str, text: str) -> str | None:
    lines = text.splitlines()
    prefix = f"## [{version}]"
    start = None
    for i, line in enumerate(lines):
        if line == prefix or line.startswith(prefix + " "):
            start = i
            break
    if start is None:
        return None
    out = [lines[start]]
    for line in lines[start + 1 :]:
        if line.startswith("## ") or line.strip().lower() == "### detailed notes":
            break
        out.append(line)
    return "\n".join(out).rstrip() + "\n"


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    path = Path(argv[2] if len(argv) > 2 else "CHANGELOG.md")
    notes = extract(argv[1], path.read_text(encoding="utf-8"))
    if notes is None:
        return 1
    sys.stdout.write(notes)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
