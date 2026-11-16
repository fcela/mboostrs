"""Compile every code cell in the book. Catches a syntax error introduced by
an edit made after the notebook was last executed, which is otherwise only
found by a full build."""
import ast
import json
import pathlib
import re
import sys

import yaml

# Relative to this file, so the check works from the repository root and
# from `docs/book`, where build-html.sh runs it.
BOOK = pathlib.Path(__file__).resolve().parent


def public_case_pages():
    config = yaml.safe_load((BOOK / "myst.yml").read_text())
    pages = set()

    def walk(entries):
        for entry in entries:
            relative = entry.get("file", "")
            if relative.startswith("cases/") and not relative.endswith("overview.ipynb"):
                pages.add(BOOK / relative)
            walk(entry.get("children", []))

    walk(config["project"]["toc"])
    return pages


PUBLIC_CASES = public_case_pages()

# `print(frame)` writes polars' text rendering, whose box-drawing characters
# (U+250C and friends) have no LaTeX definition, so pdflatex stops with
# "Unicode character not set up for use with LaTeX" and produces no PDF. The
# HTML build renders the same output without complaint, so the failure only
# appears in the PDF. `display(frame)` emits an HTML table instead, which both
# outputs handle.
PRINTED_FRAME = re.compile(
    r"^\s*print\(\s*(?:"
    r"pl\.DataFrame\b"          # print(pl.DataFrame(...))
    r"|pd\.DataFrame\b"
    r"|[A-Za-z_][\w.]*\.(?:select|head|tail|sort|filter|with_columns|group_by)\b"
    r")"
)

bad = []
printed_frames = []
dense_case_cells = []
uncommented_case_cells = []
for p in sorted(BOOK.rglob("*.ipynb")):
    if "_build" in str(p) or "exports" in str(p):
        continue
    nb = json.load(open(p))
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if not src.strip():
            continue
        try:
            ast.parse(src)
        except SyntaxError as e:
            bad.append(f"{p.relative_to(BOOK)} cell {i}: {e.msg} (line {e.lineno})")

        for line in src.splitlines():
            if PRINTED_FRAME.match(line):
                printed_frames.append(
                    f"{p.relative_to(BOOK)} cell {i}: {line.strip()[:70]}")

        if p in PUBLIC_CASES:
            lines = src.splitlines()
            if len(lines) > 35:
                dense_case_cells.append(
                    f"{p.relative_to(BOOK)} cell {i}: {len(lines)} lines"
                )
            if (len(lines) > 25
                    and not any(line.strip().startswith("#") for line in lines)):
                uncommented_case_cells.append(
                    f"{p.relative_to(BOOK)} cell {i}: {len(lines)} lines"
                )

for line in bad:
    print("  SYNTAX", line)
for line in printed_frames:
    print("  PRINTED FRAME", line)
for line in dense_case_cells:
    print("  DENSE CASE CELL", line)
for line in uncommented_case_cells:
    print("  UNCOMMENTED CASE CELL", line)
if printed_frames:
    print("\n  Use display(...) rather than print(...) for a data frame: the\n"
          "  text rendering's box-drawing characters have no LaTeX definition\n"
          "  and stop the PDF build, which the HTML build does not catch.")
if dense_case_cells:
    print("\n  Split case-study cells before 35 lines. Keep one analytical step\n"
          "  per cell and introduce the next step in prose.")
if uncommented_case_cells:
    print("\n  Case-study cells longer than 25 lines need an intent comment that\n"
          "  explains an analytical choice or intermediate result.")
print(f"{len(bad)} cells with syntax errors, "
      f"{len(printed_frames)} printed data frames, "
      f"{len(dense_case_cells)} dense case cells, "
      f"{len(uncommented_case_cells)} uncommented case cells")
sys.exit(1 if (bad or printed_frames or dense_case_cells
               or uncommented_case_cells) else 0)
