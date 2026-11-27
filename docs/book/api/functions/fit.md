# `fit`

Defined in `mboostrs._procedure._fit`; exported as `mboostrs.fit`.


*Function*

```python
fit(
    data: Any,
    response: str,
    family: Any = None,
    folds: Any = None,
    groups: Any = None,
    time: str | None = None,
    seed: int = 0,
    budget: str = 'standard',
    uncertainty: str = 'none',
    na_action: str | None = None,
    **kwargs: Any
) -> Fit
```

Fit one model and return it with the trail of how it was chosen.

`fit` uses `autospec` (with guarded interaction adoption and
`residual_tree="auto"` by default) to choose the specification and
returns the fitted model with a trail containing the input declarations,
specification decision, active final formula (terms carrying at least 1%
of risk reduction or effect variance, or intercept-only when the CV curve
is flat), family, and stopping iteration.

The returned model predicts unseen categorical or `brandom` levels as
the average group by default (override per call with
`unseen_levels="error"`). On tables with more than 3,000 rows and no
explicit `folds` or `groups`, specification search runs on a
deterministic 3,000-row subsample (pinning the response extremes to
preserve support) and the final model is fitted on all rows.

**Parameters**

- `data` (DataFrame or dict): Polars/pandas frame or mapping of column name to array.
- `response` (str): The response column.
- `family` (Family, optional): Pinned family, passed to `autospec`.
- `folds` (optional): Fold scheme or weight matrix, passed to `autospec`.
- `groups` (optional): Declared grouping column(s), passed to `autospec`.
- `time` (str, optional): Time column name ordering the observations. When `folds` is omitted,
  constructs rolling-origin folds via `mboostrs.folds.rolling_folds`
  and defaults `fit.predict_interval(...)` to `method="adaptive"`.
- `seed` (int, default 0): The single seed for every random stream.
- `budget` ({"quick", "standard", "thorough"}): The number of model fits and the wall-clock time the specification
  search is given: 16 fits and 60 s, 64 fits and 120 s, 512 fits and
  1200 s. Each fit is one cross-validated or held-out evaluation of a
  candidate family, learner, interaction or stopping depth. With fewer
  fits more decisions fall back to the dtype-driven default learner for
  each column.

  Neither limit is exact. autospec checks the fit count before each
  stage starts, so a stage in progress can exceed it. The time is
  checked before each model fit starts, so the search can overrun by the
  length of one fit. When the time runs out the search keeps what it
  measured if the probe in progress allows it, and otherwise returns the
  default specification (one default learner per column, nothing
  measured). The trail's `budget` records both limits and the fits
  spent, and `degradations` says when a limit stopped the search and
  which of those outcomes occurred.
- `uncertainty` ({"none", "stability", "bootstrap", "conformal"}): Post-fit uncertainty quantification computed and attached to the
  returned `Fit` (`fit.stability`, `fit.bootstrap`, or
  `fit.conformal`) and recorded in `trail.uncertainty`.
- `na_action` ({"impute", "impute_model", "impute_model_split", "omit", "error"}, optional): Missing-value policy passed through to `autospec` and the final fit.
  `None` (default) uses `"impute"` so incomplete predictors are
  median/mode-filled with missingness indicators and prediction handles
  missing values cleanly; pass `"omit"` or `"error"` to override.
- `**kwargs`: Passed through to `autospec`. `interactions` defaults to
  `"adopt"` and `residual_tree` defaults to `"auto"`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/_procedure/_fit.py` lines 749-1001](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_procedure/_fit.py#L749-L1001)
