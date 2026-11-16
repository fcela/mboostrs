# Stopping-rule ablation (issue #509)

Version 2 compares stopping rules on matched preprocessing routes and uses
rolling-origin inner validation for forecasting. Its descriptive conclusions
replace the stopping claims from the October 12 experiment. No library default
or boosting algorithm changes in this experiment.

[Annotated headline table](v2/headline.html) compares both methods on all
seven panels with loss units, medians, ranges and marked best values.

- [Current report](v2/README.md), [results](v2/results.json.gz), and
  [computed per-seed comparisons and summaries](v2/summary.json).
- [Correction report](audit_v1.md) and [computed audit](audit_v1.json) for the
  superseded run, including individual harmful seeds and their denominators.
- [Original results](results.json), retained byte for byte. The
  [original report](v1/README.md) is retained as a superseded record; its
  causal attributions and universal claims are withdrawn.

[Validation record](VALIDATION.md) includes the full-suite failures and the
focused passing checks.

[Research handoff](HANDOFF.md) records the remaining topology and tree-AIC
experiments, their scope and promotion criteria.

## Experimental design

The historical single-split fixed-budget rule uses refitted preprocessing.
A second fixed-budget control uses the native fixed-preprocessing route,
matching native growth and patience. These native comparisons are diagnostic:
held-out rows can still affect basis construction. Deployment-facing comparisons
use fold-local fixed-budget, budget-growth and multiple-fold growth rules. All
routes score mean squared error for Gaussian responses and log loss for binary
responses. Calibration and test rows enter only after development-side choices.

IID panels use two repeated five-fold partitions. The forecasting panel uses
ten expanding rolling origins over whole dates. Every origin keeps explicit
held-out and judge masks, so future rows beyond its horizon remain unused.
Grouped and spatial designs use whole-group folds when included. Inner folds
are not independent assessment replicates. Real-data outer splits overlap;
reported medians, ranges and worst seeds are descriptive, without confidence
intervals. Synthetic seeds are independent data draws.

The existing API supports fold-local growth but refuses fold-local patience.
Patience therefore remains an explicitly labelled native-route diagnostic.
The experiment changes no public defaults and supplies no recommendation to
replace them. Tree topology, split-gain diagnostics and tree degrees of freedom
require separate controlled research; these results do not establish the
original competitor-relative predictive-loss target.

## Reproduction

Use the repository-local Python environment with the release extension:

```bash
.venv/bin/maturin develop --release
OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m benches.stopping_ablation_509 --run --workers 3
.venv/bin/python -m benches.stopping_ablation_509 --report
.venv/bin/python -m benches.stopping_ablation_509_table
.venv/bin/python -m benches.stopping_ablation_509 --audit-v1
.venv/bin/python -m pytest tests/test_stopping_ablation_509.py
```

The 25 jobs cover seven panels, both four-configuration mboost grids, three
outer splits on each real panel and five draws on each synthetic panel.
Results include source/build identities, dependency versions, data/partition
fingerprints, complete fold membership and judge masks, aggregate curves,
selected iterations, grid limits, boundary warnings, per-rule wall time and
peak process memory. Fold-local and fixed-budget routes retain full fold-level
curves. Native growing routes retain the public aggregate and selected-step
fold scores. Timings measured with concurrent jobs are whole-workflow costs,
not controlled speed comparisons.

The full evidence is lossless gzip-compressed JSON. Its manifest records the
uncompressed SHA-256 and byte count. The numerical run used `ebe1e45`; publication
adds compression and presentation without changing those measurements. To inspect
the complete JSON directly:

```bash
gzip -dc docs/benchmarks/stopping_ablation_509/v2/results.json.gz > /tmp/stopping-509-v2.json
```
