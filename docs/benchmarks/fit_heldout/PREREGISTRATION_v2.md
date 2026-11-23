# Pre-registration v2: `fit` with a fit budget and a time limit

Written 2026-11-21 before any run of this study. The first study
(PREREGISTRATION.md) failed its criterion 3: the new default exceeded 600 s on
one 1,025-column table. The change under test now adds a wall-clock limit per
budget setting (60 / 120 / 1200 s) with a fallback to the budget-0
specification. Those limits were chosen from the first study's run times, so
the first study's seeds are not reused.

## Arms

- **old**: HEAD 12c87a3 `python/mboostrs/_procedure/_fit.py`, verbatim (`old_fit.py`).
- **new**: working-tree `mb.fit(train, "y", seed=s)` (budget `"standard"`: 32 fits, 120 s).
- **quick** (secondary): working-tree `mb.fit(..., budget="quick")` (16 fits, 60 s).

## Data

95 datasets, none used to choose the budget mapping:

- 36 OpenML datasets from the first study (manifest.json);
- 56 UCI datasets fetched with `ucimlrepo` (manifest_uci.json): every
  importable tabular dataset with one target, 300 to 500,000 rows and 2 to
  300 features, whose target is binary or (when UCI lists a regression task)
  numeric with more than 10 values, minus duplicates of the OpenML set and of
  the selection panels;
- 3 `mboostrs.datasets` tables with published additive-model analyses
  (Munich rent, India stunting, bodyfat; manifest_mb.json).

**Well-studied tier**: 18 of these datasets whose effects are documented in a
cited paper. `known_effects.json` lists 38 effect directions with their
citations; each was checked against the marginal Spearman sign on the full
data, and all 38 agree, so no target is coded backwards.

**onceover strata**: `onceover_scan.json`, computed before any run, flags
datasets where onceover finds an exact identity between the target and a
column (17) or a leakage finding (5). The flags only stratify reporting.

## Splits, scoring, failures

As in v1, with seeds 2000, 2001, 2002: 80/20 random split (stratified for
binary), at most 4,000 training and 2,000 test rows, test MSE or clipped log
loss on the response (the mean for distributional models), one thread, 600 s
per run; an error, timeout or non-finite loss is a failure of that arm.

**Effect direction.** For each known effect, every test row's value of that
column is set to the training 10th percentile and then to the 90th, and the
mean change in prediction is recorded. The effect is **recovered** in a run
when the change has the published sign and its magnitude exceeds 1e-8 times
the training standard deviation of `y` (binary: of the predicted
probability scale, 1e-8). A column the model does not use gives a change of
exactly 0 and is not recovered.

## Primary analyses

**A. Prediction, all datasets.** With `r` = mean over seeds of
`log(L_new / L_old)` over datasets where both arms complete all seeds: mean
`r` < 0 with the 95% bootstrap upper bound < 0 (10,000 resamples of datasets,
seed 0); two-sided Wilcoxon p < 0.05; new fails no more runs than old.

**B. Prediction, well-studied tier.** Criteria 1 and 2 of A on the 18 tier
datasets.

**C. Known effects.** Over (effect, seed) pairs where both arms completed,
the new arm's recovery rate is at least the old arm's minus 5 percentage
points (non-inferiority). The old arm keeps every column by construction, so
this tests whether the search drops or reverses documented effects.

## Secondary, reported regardless

UCI-only and OpenML-only versions of A; A within and outside the onceover
flags; datasets more than 2% worse; median time ratio; failures with
messages; quick against old for A, B and C; which known effects each arm
misses.
