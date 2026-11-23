#!/usr/bin/env python3
"""Check book prose for leaked internal development artifacts.

Published documentation should not leak issue/PR tracker numbers, private
repository names, raw git commit hashes, or local filesystem paths. Prose
clarity and scientific style are governed by `docs/WRITING_GUIDE.md` rather
than English phrase blacklists.

    python docs/book/check_register.py            # fail on any finding
    python docs/book/check_register.py --report   # counts per page
    python docs/book/check_register.py --show PAGE
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

BOOK = Path(__file__).resolve().parent

#: Objective internal-artifact leak patterns scanned in prose (code cells,
#: fenced blocks, math, and link targets are excluded).
RULES = {
    "tracker reference": re.compile(
        r"(?<![\w/&])#\d{2,5}\b|\b(?:PR|MR)\s+#?\d+\b|\bissue\s+#?\d+\b|\bpull request\b|mboostrs-dev",
        re.I,
    ),
    "commit hash": re.compile(
        r"\b(?:commit|merge|main|origin/main|rev(?:ision)?)\s+`?[0-9a-f]{7,40}\b|`[0-9a-f]{7,12}`"
    ),
    "local path": re.compile(
        r"(?:/workspace/|/private/tmp/|/Users/[A-Za-z0-9_.-]+/|/home/[A-Za-z0-9_.-]+/)"
    ),
}

FENCE_RE = re.compile(r"(```|~~~).*?\1", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
MATH_RE = re.compile(r"\$\$.*?\$\$|\$[^$\n]+\$", re.DOTALL)
MYST_LABEL_RE = re.compile(r"^\([^)]*\)=\s*$", re.MULTILINE)
LINK_TARGET_RE = re.compile(r"\]\([^)]*\)")


def toc_files() -> list[Path]:
    config = yaml.safe_load((BOOK / "myst.yml").read_text(encoding="utf-8"))

    def walk(entries, out):
        for entry in entries:
            if "file" in entry:
                out.append(BOOK / entry["file"])
            walk(entry.get("children", []), out)
        return out

    return walk(config["project"]["toc"], [])


def prose(path: Path) -> str:
    if path.suffix == ".ipynb":
        notebook = json.loads(path.read_text(encoding="utf-8"))
        text = "\n\n".join(
            "".join(c["source"])
            for c in notebook["cells"]
            if c["cell_type"] == "markdown"
        )
    else:
        text = path.read_text(encoding="utf-8")
        if text.startswith("---\n"):
            text = text.split("\n---\n", 1)[-1]
    text = FENCE_RE.sub("", text)
    text = HTML_COMMENT_RE.sub("", text)
    text = MYST_LABEL_RE.sub("", text)
    text = MATH_RE.sub("", text)
    text = LINK_TARGET_RE.sub("]", text)
    return text


def findings(path: Path) -> list[tuple[str, str]]:
    text = prose(path)
    out: list[tuple[str, str]] = []
    for name in ("commit hash", "local path"):
        for match in RULES[name].finditer(text):
            line_start = text.rfind("\n", 0, match.start()) + 1
            line_end = text.find("\n", match.end())
            line = text[line_start:line_end if line_end >= 0 else None].strip()
            out.append((name, f"[{match.group(0)}] {line[:150]}"))
    stripped = INLINE_CODE_RE.sub("", text)
    for match in RULES["tracker reference"].finditer(stripped):
        line_start = stripped.rfind("\n", 0, match.start()) + 1
        line_end = stripped.find("\n", match.end())
        line = stripped[line_start:line_end if line_end >= 0 else None].strip()
        out.append(("tracker reference", f"[{match.group(0)}] {line[:150]}"))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--show", metavar="PAGE", help="list the findings of one page")
    arguments = parser.parse_args()

    if arguments.show:
        for name, line in findings(BOOK / arguments.show):
            print(f"{name:18} {line}")
        return 0

    per_page = {}
    for path in toc_files():
        if path.exists():
            found = findings(path)
            if found:
                per_page[path.relative_to(BOOK).as_posix()] = found

    total = sum(len(v) for v in per_page.values())
    if arguments.report:
        kinds = Counter(name for v in per_page.values() for name, _ in v)
        for name, count in kinds.most_common():
            print(f"{count:5}  {name}")
        print()
        for page, found in sorted(per_page.items(), key=lambda kv: -len(kv[1])):
            print(f"{len(found):5}  {page}")
        return 0

    if per_page:
        print(f"Register check: {total} leaked artifact(s) on {len(per_page)} page(s)", file=sys.stderr)
        for page, found in sorted(per_page.items()):
            for name, line in found:
                print(f"- {page}: {name}: {line}", file=sys.stderr)
        return 1
    print("Register check: no leaked tracker references, commit hashes, or local paths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
