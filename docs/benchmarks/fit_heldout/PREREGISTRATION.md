# Pre-registration: does the `fit` budget change improve held-out loss?

Written 2026-11-20 before any run of this study. Nothing below is changed
after results are read; any deviation is reported as a deviation.

## Claim tested

`mb.fit(data, "y")` after the change (budget `"standard"` = 32 model fits
passed to autospec) has lower held-out loss than `mb.fit(data, "y")` before
the change (HEAD 12c87a3, which passed no budget, so autospec ran at 0).

## Arms

- **old**: the HEAD 12c87a3 source of `python/mboostrs/_procedure/_fit.py`,
  extracted verbatim with `git show` and called as `fit(train, "y", seed=s)`.
- **new**: the working-tree `mb.fit(train, "y", seed=s)` (default budget
  `"standard"`).
- **quick** (secondary): working-tree `mb.fit(train, "y", seed=s, budget="quick")`.

No family is passed; each arm infers it, as an agent's bare call would. The
same native extension serves all arms.

## Data

36 OpenML datasets read from the local scikit-learn cache, none of which was
used to choose the budget mapping: every cached dataset that loads, minus the
selection panels (OpenML 44959 concrete, 44963 protein, 44964
superconductivity, 44979 diamonds, 31 credit) and two duplicates (45402 =
44985 space_ga, 197 = 44978 cpu_activity). Targets with more than two values
and numeric type are regression; two-class targets are binary (the second
class in sorted order is 1). Multiclass datasets are excluded. The manifest is
`data/manifest.json`.

## Splits and scoring

Seeds 1000, 1001, 1002 (not used before). Per seed: a random 80/20 split
(stratified for binary); training rows are subsampled to at most 4,000 and
test rows to at most 2,000. Loss is test MSE (regression) or test log loss with
probabilities clipped to [1e-12, 1 - 1e-12] (binary), on `type="response"`
predictions. One thread per process. Per-run wall-clock limit 600 s; a run
that exceeds it, raises, or returns a non-finite loss is a failure of that arm.

## Primary analysis

For each dataset where both old and new complete all three seeds,
`r = mean over seeds of log(L_new / L_old)`. The claim holds if all three:

1. the mean of `r` over datasets is below 0, and the upper end of its 95%
   percentile bootstrap interval (10,000 resamples of datasets, seed 0) is
   below 0;
2. a two-sided Wilcoxon signed-rank test of `r` against 0 has p < 0.05;
3. the new arm fails on no more (dataset, seed) runs than the old arm.

## Secondary, reported regardless

- Number of datasets with `exp(r) > 1.02` (new more than 2% worse) and the
  worst dataset.
- Median wall-clock ratio new/old.
- The same primary statistics for quick against old.
- Failures by arm with their error messages.
