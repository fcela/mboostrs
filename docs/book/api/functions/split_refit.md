# `split_refit`

Defined in `mboostrs.split_refit`; exported as `mboostrs.split_refit`.


*Function*

```python
split_refit(
    formula,
    data,
    family=None,
    select='deselect',
    repeats=1,
    fraction=0.5,
    level=0.9,
    seed=0,
    contrasts=None
)
```

Choose terms on one random half of the rows and estimate them by REML on
the other half, optionally over several random splits.

**Parameters**

- `formula` (str or dict): A `gamboost` formula, `"y ~ bbs(x1, df=4) + bbs(x2, df=4)"`, or a
  mapping of distribution parameter names to formulas for a
  distributional family (`family=` then names one, such as
  `mb.GaussianLSS()`).
- `data`: A dict of columns, a Polars or a pandas frame.
- `family` (optional): The family, as for `gamboost` or `gamboostLSS`.
- `select` ({"deselect", "stabsel"} or callable): How the selection half chooses terms. `"deselect"` tunes the boosted
  fit on that half by 5-fold cross-validation and keeps the terms whose
  share of the risk reduction is at least 1% (`mboostrs.deselect`).
  `"stabsel"` keeps the terms whose selection frequency over subsamples
  passes `cutoff=0.75` under an expected-false-learners bound of
  `PFER=1`, with `stabsel`'s default subsampling
  (`mboostrs.stabsel`); it needs a `gamboost` formula. A callable
  `select(data)` receives the selection half and returns the formula
  terms to keep, as strings, or for a mapping `{parameter: [terms]}` or
  a list of `(parameter, term)` pairs. Returned terms must be terms of
  the formula, written as they are there.
- `repeats` (int): Number of random splits. Each repeat draws its own split and seed.
- `fraction` (float): Share of the rows in the selection half; the rest are the inference
  half. The refit sees only the inference half.
- `level` (float): Coverage of the intervals.
- `seed` (int): Seeds the splits, the fold assignments of the selection, and the
  coefficient draws. The same seed reproduces the same result.
- `contrasts` (list of (variable, reference, alternative), optional): Average contrasts to report. For each repeat the contrast is the mean,
  over the inference rows, of the refit's change in the predictor when
  `variable` changes from `reference` to `alternative` and every
  other column keeps its value in that row. For a `gamboost` refit the
  predictor is on the link scale, and the interval comes from the refit's
  coefficient covariance. For a distributional refit it is the
  response-scale mean of the first distribution parameter, and the
  interval is the quantiles of posterior draws. A variable that no term
  the split kept reads contributes exactly zero in that split.

**Returns**

SplitRefitResult
    `terms` and `contrasts` hold the aggregate over repeats; `splits`
    holds one record per repeat, with the rows used, the terms chosen, the
    term p-values and each contrast's per-split interval.

Aggregation over repeats (`repeats > 1`)
    Selection frequency is the share of repeats that chose a term or read a
    contrast variable. Each contrast's estimate and interval bounds are the
    medians over repeats of the per-split values. Each p-value is aggregated
    by `meinshausen_buhlmann_p` with `gamma = 0.5`: twice the median
    of the per-split p-values, capped at 1. A repeat that did not choose a
    term counts as p = 1 for that term. With one repeat the per-split
    p-value is reported unchanged, and the intervals are those of the split.

Limitations
-----------
The intervals are valid for a fixed split. Selective inference, which
conditions on the selection event of the whole boosting path (the
polyhedral method of Lee et al., 2016), is not implemented here. Each
refit uses `fraction` of the rows, so a split interval is wider than one
from a refit on all rows. A term that `gam_refit` cannot represent (`btree`,
`bmono`, `bkernel`) makes the refit raise, naming the term. A contrast
variable read by two or more refit terms raises, because the interval of
their sum needs the covariance between those terms.

The coverage of the procedure is a property of the whole procedure, the
split, the selection rule and the refit, and it is checked by simulation in
`tests/test_split_refit.py`, not by the intervals alone.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/split_refit.py` lines 386-522](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/split_refit.py#L386-L522)
