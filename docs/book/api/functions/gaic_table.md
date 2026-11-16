# `gaic_table`

Defined in `mboostrs.model_selection`; exported as `mboostrs.gaic_table`.


*Function*

```python
gaic_table(
    *models,
    k: float | tuple = 2.0,
    labels: list[str] | None = None,
    as_frame: bool | None = None,
    output: str | None = None,
    common_sample: bool = False
) -> list[dict[str, Any]] | Any
```

Compare same-sample fits using an explicitly named stopping criterion.

**Parameters**

- `*models`: Fitted model objects (GLMBoostModel, GAMBoostModel, or LSSModel).
- `k` (float or tuple): Penalty multiplier. If a float, use that value. If a tuple
  `(k1, "logn")`, compute GAIC with penalty `k1` and also
  `log(n)` (BIC-like).
- `labels` (list of str, optional): Custom model labels. Defaults to formula/family.
- `as_frame` (bool, optional): Superseded by `output=`. `True` returns a data frame, `False`
  returns records. Leaving it unset — the default — returns records,
  which is what this function has always returned.
- `output` ({"auto", "pandas", "polars", "records"}, optional): Output format. Overrides `as_frame` if both are given.
- `common_sample` (bool, default False): Score every model on the rows all of them can use. A model drops the
  rows its `na_action` removes, so two candidates can keep different
  rows; with `common_sample=True` the intersection of the kept rows is
  taken, every model is refitted on it, and the criteria are computed
  there. Rows are never dropped silently: a warning names the rows each
  candidate loses, and each row reports `common sample rows`,
  `common sample excluded rows` (how many of its own rows the common
  sample removed) and `common sample excluded by` (which other
  candidates dropped them). With the default `False`, models on
  different rows raise `ValueError`.

**Returns**

DataFrame or list of dict
    Gaussian rows use `mboost_corrected`, a per-observation criterion
    with a fixed penalty; k must be 2. Likelihood rows use `GAIC(k)`.
    LSS rows use `heuristic_GAIC(k)`: their penalty counts updates plus
    one, not effective df. These three definitions cannot be ranked
    together. Rows report criterion, normalization, df convention and the
    evaluated mstop; risk and logLik refer to that same step.

    Models fitted to different rows, response values, row order or
    effective weights raise `ValueError`. Models that share their rows
    but differ in response transform (`log(y)` against `y`) or weight
    policy are returned unsorted, each row with a `comparison` column
    giving the reason, because their likelihoods are over different scales.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`
- `NotImplementedError`

Source: [`python/mboostrs/model_selection.py` lines 239-382](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/model_selection.py#L239-L382)
