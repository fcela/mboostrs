#!/usr/bin/env python3
"""Verify `examples/python/` and book notebooks against the public Python API.

What this checks:

* **Every `mb.<name>` resolves** against `mboostrs.__all__` in both code and
  markdown cells across `examples/python/` and every notebook in `docs/book/myst.yml`.
* **No stored outputs in `examples/python/`**: standalone examples are verified
  by executing them in CI (`docs/book/execute_examples.sh`), not by static
  saved outputs. Book notebooks are governed by `docs/book/check_docs.py`.

Run from the repository root:

    python tools/check_examples.py
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "python"

sys.path.insert(0, str(ROOT / "python"))

import mboostrs as mb  # noqa: E402


def notebooks() -> list[pathlib.Path]:
    """The standalone examples plus every notebook listed in `docs/book/myst.yml`."""
    import yaml

    pages = sorted(EXAMPLES.glob("*.ipynb")) if EXAMPLES.is_dir() else []
    book = ROOT / "docs" / "book"
    config = yaml.safe_load((book / "myst.yml").read_text(encoding="utf-8"))

    def walk(entries, out):
        for entry in entries:
            if "file" in entry:
                out.append(entry["file"])
            if "children" in entry:
                walk(entry["children"], out)
        return out

    pages += [book / page for page in walk(config["project"]["toc"], [])
              if page.endswith(".ipynb")]
    return pages


def cells(notebook: dict, kind: str) -> str:
    return "\n".join("".join(c["source"]) for c in notebook["cells"]
                     if c["cell_type"] == kind)


def check(path: pathlib.Path) -> list[str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = cells(notebook, "code")
    prose = cells(notebook, "markdown")
    problems: list[str] = []

    public = set(mb.__all__)
    for name in sorted(set(re.findall(r"\bmb\.(\w+)", code + "\n" + prose))):
        if name not in public and not name.startswith("_"):
            problems.append(f"`mb.{name}` is not a public name")

    in_examples = EXAMPLES in path.parents
    stored = sum(1 for c in notebook["cells"]
                 if c["cell_type"] == "code" and c.get("outputs"))
    if stored and in_examples:
        problems.append(
            f"{stored} cell(s) carry stored outputs. Keep `examples/python/` "
            f"notebooks output-free; `execute_examples.sh` verifies them."
        )

    return problems


def main() -> int:
    if not EXAMPLES.is_dir():
        print(f"no {EXAMPLES.relative_to(ROOT)} to check")
        return 0
    failed = 0
    for path in notebooks():
        problems = check(path)
        if problems:
            failed += 1
            print(f"\n=== {path.relative_to(ROOT)}")
            for problem in problems:
                print(f"  - {problem}")
    if failed:
        print(f"\n{failed} notebook(s) need attention")
        return 1
    print(f"{len(notebooks())} notebooks (examples + book ToC): names "
          f"resolve, no stored outputs in examples/python")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
