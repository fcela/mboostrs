#!/usr/bin/env bash
# Execute every public example notebook in an isolated temporary output tree.
#
# Two of the eight notebooks (02_r_parity, 03_performance) import rpy2 and need
# a working R installation. The documentation CI job does not install R, so those
# two are skipped there rather than failing the build — the same treatment the
# pytest suite gives its `r_parity` marker, which is deselected by default on CI.
# The count check below still accounts for every notebook, so a skip is reported
# rather than silently reducing coverage.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUTPUT_DIR=$(mktemp -d "${TMPDIR:-/tmp}/mboostrs-examples.XXXXXX")
trap 'rm -rf "$OUTPUT_DIR"' EXIT
if [ -n "${PYTHON:-}" ]; then
  PYTHON_BIN="$PYTHON"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
else
  PYTHON_BIN=python3
fi

if "$PYTHON_BIN" -c "import rpy2" >/dev/null 2>&1; then
  HAVE_RPY2=1
else
  HAVE_RPY2=0
  echo "note: rpy2 is not importable; notebooks requiring R will be skipped."
fi

count=0
executed=0
skipped=0
for notebook in "$ROOT"/examples/python/*.ipynb; do
  count=$((count + 1))
  if [ "$HAVE_RPY2" -eq 0 ] && grep -q "rpy2" "$notebook"; then
    echo "skipped (needs R): $(basename "$notebook")"
    skipped=$((skipped + 1))
    continue
  fi
  "$PYTHON_BIN" -m jupyter nbconvert \
    --to notebook \
    --execute "$notebook" \
    --output-dir "$OUTPUT_DIR" \
    --ExecutePreprocessor.timeout=300
  executed=$((executed + 1))
done

if [ "$count" -ne 8 ]; then
  echo "ERROR: expected 8 public example notebooks, found $count" >&2
  exit 1
fi
echo "Executed $executed public example notebooks successfully ($skipped skipped, needs R)."
