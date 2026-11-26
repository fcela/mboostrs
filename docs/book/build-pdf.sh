#!/usr/bin/env bash
# Build mboostrs book as PDF via MyST + LaTeX.
# Requires: myst (>=1.0), rsvg-convert, pdflatex (TeX Live)
set -euo pipefail

cd "$(dirname "$0")"

# Polars prints a "shape: (rows, cols)" caption and a row of column dtypes in
# both its text and HTML reprs. MyST converts the HTML repr into the LaTeX
# tables in the PDF, so both end up in the printed document: the shape line
# lands as a stray sentence before every table, and the dtype row as the
# table's first data row. Neither is content. These variables are the
# environment form of pl.Config.set_tbl_hide_dataframe_shape /
# set_tbl_hide_column_data_types, and the executing kernels inherit them, so
# no notebook has to set them itself.
export POLARS_FMT_TABLE_HIDE_DATAFRAME_SHAPE_INFORMATION=1
export POLARS_FMT_TABLE_HIDE_COLUMN_DATA_TYPES=1

# shellcheck source=kernel-env.sh
source ./kernel-env.sh

# The book's string pool exceeds TeX Live defaults (~469k strings against a
# 500k cap and growing with each chapter), so pdflatex dies with "TeX capacity
# exceeded [number of strings]". If the user keeps raised limits in
# ~/texmf/web2c/texmf.cnf, put that directory first on TEXMFCNF so kpathsea
# finds it. The format file must then be regenerated for the new values:
#   fmtutil-user --byfmt pdflatex
if [ -d "$HOME/texmf/web2c" ]; then
  export TEXMFCNF="$HOME/texmf/web2c:${TEXMFCNF:-}"
fi

# Step 0: Honor an explicitly selected or already-active environment. Fall
# back to the contributor-guide environment only for a direct local build.
if [ -n "${MBOOSTRS_DOCS_VENV:-}" ]; then
  # shellcheck disable=SC1090
  source "$MBOOSTRS_DOCS_VENV/bin/activate"
elif [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$HOME/venv/lab/bin/activate" ]; then
  source "$HOME/venv/lab/bin/activate"
fi
if [ -n "${PYTHON:-}" ]; then
  PYTHON_BIN="$PYTHON"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
else
  PYTHON_BIN=python3
fi

# check_cells.py catches the two failures that reach only this build: a
# syntax error introduced after a notebook was last executed, and a
# printed data frame, whose box-drawing characters stop pdflatex.
"$PYTHON_BIN" check_cells.py
"$PYTHON_BIN" check_docs.py
"$PYTHON_BIN" check_runtime.py

# Step 1: Regenerate figures (SVG, PNG)
echo ":: Generating figures..."
"$PYTHON_BIN" generate_figures.py

# Step 1b: Pre-render mermaid diagrams to SVG
echo ":: Pre-rendering mermaid diagrams..."
"$PYTHON_BIN" render_mermaid.py

# Step 2: Clear stale execution cache (unless called right after build-html.sh
# with MBOOSTRS_REUSE_EXEC_CACHE=1) and generate LaTeX via MyST.
# `--tex` (not `--pdf`) so MyST only emits the LaTeX export defined in
# myst.yml (project.exports, output: _build/tex/mboostrs.tex) and never
# writes its own unpatched PDF render to exports/.
if [ "${MBOOSTRS_REUSE_EXEC_CACHE:-0}" = "1" ]; then
  echo ":: Reusing fresh execution cache from HTML build..."
  rm -rf _build/temp/ _build/tex/
else
  echo ":: Clearing execution cache..."
  rm -rf _build/execute/ _build/temp/ _build/tex/
fi
echo ":: Generating LaTeX via MyST..."
mkdir -p _build
if ! myst build --tex --execute --strict --ci > _build/pdf-myst.log 2>&1; then
  echo "ERROR: MyST TeX build failed; see _build/pdf-myst.log"
  tail -40 _build/pdf-myst.log
  exit 1
fi
# MyST can exit 0 after a notebook cell raises (issue #178), leaving the
# traceback in the book. Match the full execution report (#179) and print
# the following lines so the failure is visible in the build log.
if grep -Fq "An exception occurred during code execution" _build/pdf-myst.log; then
  echo "ERROR: a notebook cell raised during execution; see _build/pdf-myst.log"
  grep -F -A20 "An exception occurred during code execution" _build/pdf-myst.log | head -60
  exit 1
fi
echo ":: MyST log: _build/pdf-myst.log"

TEX_DIR="_build/tex"
if [ ! -f "$TEX_DIR/mboostrs.tex" ]; then
  echo "ERROR: $TEX_DIR/mboostrs.tex not found; see _build/pdf-myst.log"
  exit 1
fi
echo ":: Using LaTeX source: $TEX_DIR"

# Step 4: Convert SVGs to PDF and patch references.
# Mermaid diagrams are skipped here -- their SVG node labels are HTML
# <foreignObject> content that rsvg-convert drops, rendering as empty
# boxes. render_mermaid.py pre-renders a same-stem PNG (via the same
# puppeteer pipeline mmdc uses for the SVG) that keeps the text; copy
# that in under the hashed name MyST gave the SVG so patch_tex.py's
# rewrite (.svg -> .png for mermaid figures) resolves.
echo ":: Converting SVGs to PDF..."
for svg in "$TEX_DIR"/files/*.svg; do
  [ -f "$svg" ] || continue
  base=$(basename "$svg" .svg)
  if [[ "$base" == mermaid-* ]]; then
    stem=$(echo "$base" | sed -E 's/^(mermaid-[0-9a-f]+)-.*/\1/')
    src_png="figures/${stem}.png"
    if [ -f "$src_png" ]; then
      cp "$src_png" "${svg%.svg}.png"
    elif ! grep -q '<foreignObject' "$svg"; then
      # Sequence diagrams use ordinary SVG <text> nodes, which librsvg keeps.
      rsvg-convert -f pdf -o "${svg%.svg}.pdf" "$svg"
    else
      echo "ERROR: no PNG for $svg (expected $src_png); refusing a diagram with missing text."
      exit 1
    fi
  else
    rsvg-convert -f pdf -o "${svg%.svg}.pdf" "$svg"
  fi
done

echo ":: Patching LaTeX preamble and SVG references..."
"$PYTHON_BIN" patch_tex.py "${TEX_DIR}"

# Step 5: Build PDF with pdflatex + bibtex + two resolving passes.
# Every renderer exit status is authoritative. Logs are retained in the TeX
# directory and unresolved references are checked after the final pass.
rm -f "$TEX_DIR/mboostrs.pdf"
echo ":: Running pdflatex (pass 1)..."
cd "$TEX_DIR"
pdflatex -halt-on-error -interaction=nonstopmode -shell-escape mboostrs.tex </dev/null | tail -10
echo ":: Running bibtex..."
bibtex mboostrs </dev/null | tail -10
echo ":: Running pdflatex (pass 2)..."
pdflatex -halt-on-error -interaction=nonstopmode -shell-escape mboostrs.tex </dev/null | tail -10
echo ":: Running pdflatex (pass 3)..."
pdflatex -halt-on-error -interaction=nonstopmode -shell-escape mboostrs.tex </dev/null | tail -10
if grep -Eq 'undefined references|Reference .* undefined|Citation .* undefined' mboostrs.log; then
  echo "ERROR: unresolved LaTeX references or citations remain after the final pass."
  grep -E 'undefined references|Reference .* undefined|Citation .* undefined' mboostrs.log | tail -20
  exit 1
fi
cd -

# Step 6: Verify the PDF was actually (re)built, then publish it.
if [ ! -s "$TEX_DIR/mboostrs.pdf" ]; then
  echo "ERROR: $TEX_DIR/mboostrs.pdf missing or empty after pdflatex; see the log above."
  exit 1
fi
mkdir -p exports
cp "$TEX_DIR/mboostrs.pdf" exports/mboostrs.pdf
echo ":: Built exports/mboostrs.pdf"
