# KDD scorecard study: mboostrs-only run and free-choice run

Pre-registered before any analyst run. The battery, the output contract and the
scoring map below are fixed; the truth specification in `truth.json` is released
with the results, after scoring. `PREREG_HASHES.txt` records the SHA-256 of every
data file, the truth file and the generator, so the released files can be checked
against what was registered.

## Question

Grade mboostrs on the knowledge-discovery scorecard (`docs/scorecard/scorecard.md`,
dimensions D1–D10) against an analyst that may use any Python library except
mboostrs. Two analyst runs receive the same brief and differ only in the library
rule. The run with mboostrs is `run_A`; the free run is `run_B`.

## Battery

One response `y` and twenty candidate inputs `x1`…`x20`. The response is
`f(x) + sigma(x) * eps`, `eps ~ N(0,1)`, with

- `f(x) = 1.0*x1 + 0.5*x2 + 2*sin(x4) + 1.5*(x5 > 0) + 1.0*x6*x7`
- `sigma(x) = 0.5 * exp(0.4 * x8)`
- `x4 ~ Uniform(-3, 3)`; all other inputs `~ N(0,1)`

Planted structure: main effects of `x1`, `x2`, `x4`, `x5`; interaction `x6:x7`
with no main effect of `x6` or `x7`; `x8` drives the noise scale and has no mean
effect; `x3` and `x9`…`x20` are noise.

| Dataset | Train rows | Features | Test rows | Purpose |
|---|---|---|---|---|
| `main` | 2000 | clean | 1000 | D1, D4, D9 |
| `small` | 200 | clean | 1000 | D6 (small n) |
| `messy` | 2000 | 15% MCAR missing in every input, a `region` column (no effect), 1% of `y` shifted by ±10 | 1000 (clean `y`, same missingness rule) | D6 (messy) |

Truth targets, computed on the clean training inputs of each dataset:

- Contrast `c_j`: the mean change in `f` when `x_j` moves from its 25th to its
  75th training percentile, averaged over training rows, other inputs held at
  their observed values. Planted main effects have `c_j` ≠ 0; `x6` and `x7` have
  no main effect and are not scored on contrast; `x8`, `x3` and `x9`…`x20` have
  `c_j = 0`.
- Selected: `x1`, `x2`, `x4`, `x5` and the pair `x6:x7`. Variance driver: `x8`.

## Analyst deliverables

For each dataset `D` in {main, small, messy}, the analyst writes into its run
directory:

1. `findings_D.json`:
   ```json
   {
     "dataset": "D",
     "effects": {"x1": {"selected": true, "estimate": 1.2, "lower90": 0.9, "upper90": 1.5}, "...": {}},
     "pairs": {"x6:x7": {"selected": true}},
     "variance_drivers": ["x8"]
   }
   ```
   `effects` has one entry for each of `x1`…`x20`. `estimate`, `lower90` and
   `upper90` are the contrast `c_j` and its 90% interval. `selected` is a
   decision that the variable has a mean effect.
2. `predictions_D.csv`: columns `row_id, lower90, median, upper90` for every row
   of the test features for `D`. The interval is the 90% predictive interval for
   `y`.
3. `fit_log.csv`: one row per model fit, columns `dataset, library, call, n_rows,
   seconds`.
4. `run_all.py`: regenerates every deliverable from the data directory alone.
5. A final message with the analysis report: findings, uncertainty, caveats, and
   next checks. The report is the final message, not a file.

## Library rules

- **run_A (mboostrs):** mboostrs, and the libraries mboostrs itself imports for
  data handling: numpy, polars, pandas. Direct imports of sklearn, statsmodels,
  lightgbm, xgboost, catboost, pygam, interpret or econml are not allowed. Reading
  the mboostrs agent skill documents is allowed. The `mboostrs.sklearn` adapter
  is part of mboostrs and is allowed.
- **run_B (free):** any Python library in the venv at
  `scratchpad/study/venv_free`, except mboostrs and `mboostrs.sklearn`. The venv
  does not contain mboostrs.

## Budgets and fairness

- Same brief, same data paths, same deliverables, same cap of 60 Python
  executions per run.
- Analysts may not open files outside their run directory and the data directory.
  The truth directory is not readable from either run; the analysts were not given
  its path.
- Manual tuning of hyperparameters is allowed and must be written in the analysis
  script, not made by hand between runs.

## Scoring map

| Dimension | Measure | Source |
|---|---|---|
| D1 | TPR over the five planted structures; FDR over selected items, on `main` | `findings_main.json` vs truth |
| D4 | Coverage of the 90% contrast intervals over all twenty inputs (target 0.90); median interval width for planted inputs | `findings_main.json` vs truth |
| D5 | Median and 90th percentile of logged fit seconds on `main` | `fit_log.csv` |
| D6 | D1 TPR/FDR and contrast coverage on `small` and `messy` | `findings_small.json`, `findings_messy.json` |
| D9 | Coverage of the 90% predictive intervals on test rows (target 0.90); Spearman correlation between interval half-width and the true `sigma` | `predictions_main.csv` vs `main_test_truth.csv` |
| D10 | Regenerate deliverables with `run_all.py`; agreement of findings to 1e-6 relative; completeness of `fit_log.csv` | rerun in a clean directory |
| D2, D3, D7, D8 | Judged against the scorecard anchors from the report, the deliverables and the run transcript | written justification in `SCORES.md` |

D5 is scored from logged fit times on `main` (n = 2000, 20 inputs). The
scorecard's standard 50k×20 analyst session is not run in this study. The score
for D5 is therefore reported as a proxy.

Weights: the study is fully automated, so the scorecard's fully-automated weights
(§6.2) are used for the weighted profile. Per-dimension scores are always reported.

## Notebook

Every study produces a notebook in the benchmarks section of the book,
`docs/book/benchmarks/`. The notebook shows the work and documents the findings:
the battery, the two runs, the scoring code and its output, the score sheet, and
the limits stated in `SCORES.md`. It reads the released artifacts in this
directory and recomputes the scores. The notebook is listed in `docs/book/myst.yml`
under Part V. It is committed to git with its outputs, so the results can be read
on GitHub.

## Known limitations

- The isolation of the truth file is by path and instruction, not by sandbox.
- One planted battery, one seed per dataset.
- The free run's library choice is the analyst's. The library names are recorded.
