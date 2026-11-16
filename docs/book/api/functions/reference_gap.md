# `reference_gap`

Defined in `mboostrs.reference_gap`; exported as `mboostrs.reference_gap`.


*Function*

```python
reference_gap(
    formula_or_model,
    data,
    family=None,
    folds=5,
    fold_scheme=None,
    time=None,
    reference='trees',
    maxdepth=4,
    max_mstop=5000,
    seed=0,
    candidates=None,
    top=5,
    lags=(1, 3, 24),
    nu=0.1,
    lag_columns=None,
    reference_columns='all',
    exclude_columns=None,
    extrapolation='warn',
    time_budget: float | None = 600
) -> ReferenceGapResult
```

Compare a specification with a tree reference and rank candidate revisions.

Two boosted models are scored on the same held-out rows. The specification
is the formula (or the formula of a fitted `gamboost` model). The
reference is a single `btree` term with depth `maxdepth` and the same
family and step length. By default (`reference_columns="all"`) it is
given every predictor column of `data` besides the response, the time
column and `exclude_columns`, and, with `time=`, the lag and
rolling-mean features described below; `reference_columns="formula"`
gives it only the covariates the formula reads. Each model's step count is
the one that minimises its mean held-out loss, found on the folds. The gap
is the mean over folds of the specification's loss minus the reference's,
and its standard error is taken across folds. The result lists the columns
the reference used and those among them that the formula does not use.

The function then ranks terms that could close the gap, whether or not the
gap is above two standard errors (`material` reports that). The screen
runs on the specification's out-of-fold working residual: the negative
gradient of the family's loss at the out-of-fold linear predictor, divided
by the square root of the loss's curvature where that is positive. For a
Gaussian fit this is the ordinary residual `y - fitted`, so its ranking
is the residual mean-square ranking. For other families the ranking is by
the family loss a term removes, not by squared error on the response scale.
Interaction pairs of the formula's covariates are screened with
`mboostrs.screen_interactions`. A numeric pair becomes
`bbs(a) %X% bbs(b)`; a numeric and a factor become
`bbs(x, by=f, df=6)`; two factors are not turned into a term. A column of
`data` that the formula does not use is screened as a main effect and
becomes `bbs(col, df=4)` (`bols(col)` for a factor). With `time=`,
every floating-point column with more than two distinct values gets lagged and
rolling-mean features (`lags` steps back, and a rolling mean over
`max(lags)` rows), screened together with the other columns, and the
candidates are `bbs(<feature>, df=4)`. The best `top` candidates are
each added to the specification and refitted once, and the candidates with
a positive gain are then added together in one more refit.

**Parameters**

- `formula_or_model` (str or fitted gamboost model): The specification, as `"y ~ bbs(x, df=4) + ..."`. A fitted
  `gamboost` model supplies its formula and, when `family` is None,
  its family.
- `data` (polars or pandas DataFrame, or a mapping of column to values): The training rows. Every column the formula reads must be present and
  complete: missing values raise, as they do in `gamboost`.
- `family` (Family, optional): The loss. `None` infers it from the response as `autospec` does.
- `folds` (int, default 5): Number of folds. Without `time=` the folds are a random k-fold split
  with `seed`. With `time=` they are rolling-origin blocks: fold `b`
  trains on the rows before its block and is scored on the block, so
  no fold trains on later rows. The first block starts after half the
  rows.
- `fold_scheme` (FoldScheme, optional): An explicit scheme from `mboostrs.folds`, such as
  `grouped_folds` or `blocked_folds`. Overrides `folds` and
  `time` for the split. Its rows must match `data` in order. With
  `time=` and lags, rows in the first `max(lags)` of time order are
  removed from the analysis, and the scheme is restricted to the rows
  that remain.
- `time` (str, optional): Name of a timestamp column (numeric or datetime). The folds become
  time-ordered, and lag and rolling-mean candidates are built in time
  order.
- `reference` ({"trees"}, default "trees"): The flexible reference. `"trees"` is the `btree` term described
  above.
- `maxdepth` (int, default 4): Depth of the reference tree.
- `max_mstop` (int, default 5000): Largest step count searched for each model. A best step count within
  5% of it is reported as unfinished tuning.
- `seed` (int, default 0): Seed of the random k-fold split and of the residual screen.
- `candidates` (list of str, optional): Terms to measure in place of the screened candidates, for example
  `["bbs(TEMP_lag1, df=4)"]`. Without this argument the screened
  candidates are measured.
- `top` (int, default 5): Number of candidates measured, one refit each, whatever the gap.
  `0` skips candidate measurement.
- `lags` (tuple of int, default (1, 3, 24)): Lag steps, and the rolling-mean window is `max(lags)`. Used only with
  `time=`.
- `nu` (float, default 0.1): Step length of both models.
- `lag_columns` (list of str, optional): Columns to lag. `None` lags every floating-point column with more
  than two distinct values, other than the response and the time column.
  Integer-typed columns and binary flags are not lagged.
- `reference_columns` ({"all", "formula"} or list of str, default "all"): The columns the reference `btree` sees besides the formula's
  covariates. `"all"` adds every other predictor column of `data`
  (columns with missing values, a single value, a date type or more than
  50 levels are skipped and noted) and, with `time=`, the lag and
  rolling features. `"formula"` is the earlier behaviour: only the
  covariates the formula reads. A list names the extra columns.
- `exclude_columns` (list of str, optional): Columns that are never given to the reference, such as weight, offset
  or identifier columns.
- `extrapolation` ({"warn", "clip", "error", "allow"}, default "warn"): Checked and stored for the same calls as `discover`. Every score
  here is on rows the models were fitted on, so no extrapolation policy
  acts on it; the setting has no effect on the numbers returned.
- `time_budget` (float or None, default 600): Wall-clock seconds for the whole call. The budget is checked between
  the computation stages, and the tree reference scan — the most
  expensive stage — runs with the remaining budget as a hard timeout:
  exceeding it raises `TimeoutError` naming the stage instead of
  hanging silently. `None` disables the budget (not recommended).

**Returns**

ReferenceGapResult
    The gap, its standard error, the two losses, the candidate table, the
    share of the gap the candidates close together, and the checks in
    `notes`. `describe()` says what to try next.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`
- `TypeError`

**Examples**

Measure the gap for a model on random folds, then read the result:

```python
res = reference_gap("y ~ bbs(x1, df=4) + bols(g)", data, folds=5, seed=1)
print(res.describe())
res.candidates          # ranked candidate terms, with measured gains
```


Source: [`python/mboostrs/reference_gap.py` lines 726-1082](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/reference_gap.py#L726-L1082)
