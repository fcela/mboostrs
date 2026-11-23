"""Compile every code cell in the book and catch printed DataFrames that break PDF.

Checks:
1. Every Python code cell parses with `ast.parse`.
2. No cell prints a Polars/Pandas DataFrame directly via `print(...)`, whose
   box-drawing characters break `pdflatex`; use `display(...)` instead.
"""
import ast
import json
import pathlib
import re
import sys

BOOK = pathlib.Path(__file__).resolve().parent

# `print(frame)` writes polars' text rendering, whose box-drawing characters
# (U+250C and friends) have no LaTeX definition, so pdflatex stops with
# "Unicode character not set up for use with LaTeX" and produces no PDF.
PRINTED_FRAME = re.compile(
    r"^\s*print\(\s*(?:"
    r"pl\.DataFrame\b"
    r"|pd\.DataFrame\b"
    r"|[A-Za-z_][\w.]*\.(?:select|head|tail|sort|filter|with_columns|group_by)\b"
    r")"
)

bad = []
printed_frames = []
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

for line in bad:
    print("  SYNTAX", line)
for line in printed_frames:
    print("  PRINTED FRAME", line)
if printed_frames:
    print("\n  Use display(...) rather than print(...) for a data frame: the\n"
          "  text rendering's box-drawing characters have no LaTeX definition\n"
          "  and stop the PDF build, which the HTML build does not catch.")
print(f"{len(bad)} cells with syntax errors, {len(printed_frames)} printed data frames")
sys.exit(1 if (bad or printed_frames) else 0)
