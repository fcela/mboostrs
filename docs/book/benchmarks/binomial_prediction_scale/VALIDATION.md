# Binomial scoring validation

## Scope

Issue #607 originally attributed the cancer losses and coefficient correlation
to a Binomial gradient/link defect. The original notebook clipped predict()'s
default half-log-odds output as probabilities. The audit reruns the same
recorded splits and fixed settings with both scales from identical fitted
models. It makes no engine, estimator-default or stopping-policy change.
The correctly scored autospec gap in #541 is a separate unresolved issue.

This is a diagnostic correction, not a complete competitive benchmark:
there is no data/y-only discovery track, competitor tuning comparison,
independent dataset battery, performance scorecard or 1% target claim. The
three cancer splits overlap on the same data; they are not independent Monte
Carlo draws. Per-split losses are shown without inferential standard errors.

## Results

For 200 updates, the invalid link-clipped losses 0.254925/0.397894/0.327920 become
response log losses 0.102662/0.131175/0.103217. At 500 updates the valid losses are
0.085291/0.108485/0.090664. Every fixed competitor outcome is retained; the first
split still favors the tree controls. All 15 final model/split records have
input/prediction digests and warnings. Response predictions agree with
`sigmoid(2*link)` to the asserted 1e-14 tolerance.

The seed 0 coefficient probe reproduces correlation 0.21925. The raw-column
logistic reference reaches its 2000-iteration limit and emits a convergence
warning. Its coefficients cannot be treated as a validated MLE, and it uses
a different regularization regime from the 500-step boosted fit. Doubling
boosted coefficients puts them on log-odds units without changing correlation.

An independent nonseparable 4,000-row, three-predictor logistic process has a warning-
free nearly unpenalized reference converging in 8 iterations. At 5000 updates the
maximum slope difference is 2.2023e-8 and probability difference 2.1493e-8;
training log loss agrees to reported precision. The 100/500/5000/20000 paths
are all retained, not selected by a test score.

Live R mboost 2.9.11 fits the exact three cancer training splits with 30 `bbs`
terms, mstop 200 and nu=0.1. Maximum held-out probability differences are
1.2895e-8/5.6782e-9/2.3814e-8; maximum link differences 1.2859e-7/1.6904e-8/
1.1988e-7. Entire training risk paths differ by at most 3.5791e-6. Diagnostic
checks require probability/link differences < 1e-6 and risk differences < 1e-4;
these are numerical comparison bounds, not a hypothesis test.

Two existing parity tests also pass: a live R glmboost Binomial/bols case
at rtol 1e-10, and the weighted spline GAM fixture at its existing rtol 5e-4.
Run with:

```bash
PYTHONPATH=python:. .venv/bin/python -m pytest \
  tests/test_glmboost_parity/test_binomial_bols.py \
  tests/test_gamboost_parity/test_weighted_fixtures_gam.py \
  -k binomial --run-r-parity -q
```

## Notebook and provenance

The source protocol was committed in `e8f5311` before measurement. results.json
records script/protocol hashes, interpreter/platform/package versions,
compiled build information, warnings and complete numerical outcomes. Rust
sources and Cargo manifests/lock match the freshly built release extension
from `b089d48`; the audit uses repository Python 3.12 and isolated worktree sources.
No shared editable extension or user workspace edit was changed.

historical_cells.json preserves the original source cells and outputs, with an
invalid-for-probability-scoring label. The notebook explicitly distinguishes
its newly executed cancer section and Gaussian control from retained original
Quine, large-n timing and all-column tree outputs. Those unrelated output
objects are verified unchanged after normalizing notebook serialization.
Source and result records for the cancer correction are declared as downloadable
notebook artifacts inside the book root.

Three development-instrument failures are retained: a scalar-zero unselected
coefficient extraction assumption; an invisible R availability result; and a
GT rendering limitation for a list-valued coefficient column. Their fixes
change reporting/probing only. All final settings and seeds are unchanged;
no failed statistical trial or unfavorable model row was excluded.

The corrected setup/cancer/Gaussian section executes 8 code cells (maximum 25
lines) with 5 HTML GT tables and no error outputs. Original Quine, large-n and
all-column tree output objects remain unchanged. Notebook format is upgraded
from 4.4 to 4.5 with stable cell IDs. The setup now displays warnings.
The complete book source checker passes 178 ToC pages; 23 navigation/ZIP/surface
tests pass. Complete Ruff and changed-file whitespace checks pass. No full
statistical suite is repeated for this diagnostic-only correction.

The displayed diagnostic tables use Polars data frames.
