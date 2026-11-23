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
    time: Any = None,
    seed: int = 0,
    budget: str = 'standard',
    uncertainty: str = 'none',
    na_action: str | None = None,
    **kwargs: Any
) -> Fit
```

Fit one model and return it with the trail of how it was chosen.

`fit` uses `autospec` to choose the specification and returns the
fitted model with a trail containing the input declarations, specification
decision, final formula, family and stopping iteration.

**Parameters**

- `data` (DataFrame or dict): Polars/pandas frame or mapping of column name to array.
- `response` (str): The response column.
- `family` (Family, optional): Pinned family, passed to `autospec`.
- `folds` (optional): Fold scheme or weight matrix, passed to `autospec`.
- `groups, time` (optional): Declared dependence structure. `groups` is passed to `autospec`;
  `time` currently raises `NotImplementedError`.
- `seed` (int, default 0): The single seed for every random stream.
- `budget` ({"quick", "standard", "thorough"}): The number of model fits and the wall-clock time the specification
  search is given: 16 fits and 60 s, 32 fits and 120 s, 512 fits and
  1200 s. Each fit is one cross-validated or held-out evaluation of a
  candidate family, learner, interaction or stopping depth. With fewer
  fits more decisions fall back to the dtype-driven default learner for
  each column. `"standard"` took a median of 4.5 s on benchmark
  panels of up to 4,000 rows at one thread; `"quick"` took 0.45 s and
  lost up to 45% in test loss on some of them.

  Neither limit is exact. autospec checks the fit count before each
  stage starts, so a stage in progress can exceed it (on 36 held-out
  OpenML datasets the search spent up to 60 fits at 32 and 45 at 16).
  The time is checked before each model fit starts, so the search can
  overrun by the length of one fit. When the time runs out the search
  keeps what it measured if the probe in progress allows it, and
  otherwise returns the default specification (one default learner per
  column, nothing measured). The trail's `budget` records both limits
  and the fits spent, and `degradations` says when a limit stopped the
  search and which of those outcomes occurred.
- `uncertainty` ({"none", "stability", "bootstrap"}): `"none"` is currently supported. `"stability"` and
  `"bootstrap"` raise `NotImplementedError`.
- `na_action` ({"impute", "impute_model", "impute_model_split", "omit", "error"}, optional): Missing-value policy passed through to `autospec` and the final fit.
  `None` (default) drops incomplete rows when missing values are present.
- `**kwargs`: Passed through to `autospec`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`
- `NotImplementedError`

Source: [`python/mboostrs/_procedure/_fit.py` lines 97-230](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_procedure/_fit.py#L97-L230)
