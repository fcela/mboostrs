#!/usr/bin/env bash
# Build the mboostrs book as an HTML website (with optional PDF).
#
# All chapter sources are .ipynb (executable notebooks).
# MyST executes code cells during build; merge-outputs.py then patches
# the HTML to inline SVG chart references that the client renderer
# cannot fetch from the static build.
#
# Requires: myst, rsvg-convert, pdflatex/bibtex, and the Python dependencies in
# requirements.txt. Set MBOOSTRS_DOCS_VENV to activate a specific environment.
#
# Usage:
#   bash build-html.sh                      # local preview (root-relative paths)
#   BASE_URL=/mboostrs bash build-html.sh   # GitHub Pages deployment
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

# ── Environment ──────────────────────────────────────────────────────────────
if [ -n "${MBOOSTRS_DOCS_VENV:-}" ]; then
  # shellcheck disable=SC1090
  source "$MBOOSTRS_DOCS_VENV/bin/activate"
fi
if [ -n "${PYTHON:-}" ]; then
  PYTHON_BIN="$PYTHON"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
else
  PYTHON_BIN=python3
fi

# BASE_URL controls MyST's asset path prefix.
#   (unset, empty) — root-relative paths, works for local preview
#   /mboostrs      — GitHub Pages project site (https://fcela.github.io/mboostrs)
export BASE_URL="${BASE_URL:-}"

# ── Clean ────────────────────────────────────────────────────────────────────
myst clean --all --yes

# ── Prerequisites ────────────────────────────────────────────────────────────
# Vendor the site template locally to avoid a network fetch on every build.
BOOK_THEME_REV="45706bcb9ac02b6713270785e3fc22eda348ed3b"
BOOK_THEME_ARCHIVE_URL="https://github.com/myst-templates/book-theme/archive/${BOOK_THEME_REV}.zip"
if [ ! -d book-theme ]; then
  echo ":: Fetching book-theme template at ${BOOK_THEME_REV} (one time)..."
  curl -sSL -o /tmp/book-theme.zip \
    "$BOOK_THEME_ARCHIVE_URL"
  mkdir -p book-theme-tmp && unzip -q /tmp/book-theme.zip -d book-theme-tmp
  mv "book-theme-tmp/book-theme-${BOOK_THEME_REV}" book-theme && rmdir book-theme-tmp
  rm /tmp/book-theme.zip
fi

"$PYTHON_BIN" generate_custom_css.py
"$PYTHON_BIN" check_cells.py
"$PYTHON_BIN" check_docs.py
"$PYTHON_BIN" check_runtime.py
"$PYTHON_BIN" generate_figures.py

# Pre-warm shared scikit-learn datasets so parallel MyST kernels do not race
# when downloading and unlinking the same archive (e.g. cal_housing.tgz).
"$PYTHON_BIN" -c "from sklearn.datasets import fetch_california_housing; fetch_california_housing()" >/dev/null 2>&1 || true

# ── HTML build ───────────────────────────────────────────────────────────────
mkdir -p _build
myst build --html --execute --strict --ci 2>&1 | tee _build/html-myst.log
myst_status=${PIPESTATUS[0]}
# MyST reports a cell that raised in its log but can still exit 0 (issue
# #178); the traceback would then be published.
if [ "$myst_status" -ne 0 ] || grep -q "An exception occurred during" _build/html-myst.log; then
  echo "ERROR: MyST HTML build failed or a notebook cell raised; see _build/html-myst.log"
  exit 1
fi

# Patch: inline SVG file references + strip text/plain fallbacks in HTML pages.
# MyST stores some Altair outputs as {hash, path} refs; the client renderer
# cannot resolve these from a static build, showing "Loading..." indefinitely.
"$PYTHON_BIN" merge-outputs.py

# ── PDF build ────────────────────────────────────────────────────────────────
echo ":: Building PDF..."
bash ./build-pdf.sh

# ── Stage outputs ────────────────────────────────────────────────────────────
# The notebook bundle is regenerated here rather than committed stale. It was
# a month older than the chapters it contained and no page linked to it, so it
# was a third state of the book that no reader could reach and no build
# refreshed.
"$PYTHON_BIN" generate_notebooks.py

mkdir -p _build/html/exports
cp exports/mboostrs.pdf _build/html/exports/
cp exports/notebooks.zip _build/html/exports/

echo "=== Build complete (BASE_URL='${BASE_URL:-<root>}')"
echo "  local preview:  python -m http.server 3027 --directory _build/html --bind 127.0.0.1"
echo "  deploy (gh-pages): BASE_URL=/mboostrs ${0} && rsync ..."
