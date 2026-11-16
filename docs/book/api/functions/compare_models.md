# `compare_models`

Defined in `mboostrs.model_selection`; exported as `mboostrs.compare_models`.


*Function*

```python
compare_models(
    *models,
    metrics: tuple[str, ...] = ('risk', 'logLik', 'GAIC', 'mstop', 'selected_learners'),
    labels: list[str] | None = None,
    sort: str | None = None,
    cv: list | None = None,
    as_frame: bool | None = None,
    output: str | None = None,
    common_sample: bool = False
) -> list[dict[str, Any]] | Any
```

Compare models across multiple metrics.

**Parameters**

- `*models`: Fitted model objects.
- `metrics` (tuple of str): Metrics to evaluate: "risk", "logLik", "GAIC", "AIC", "mstop", "n",
  "selected_learners".

  `logLik` follows the family's likelihood contract and is unavailable
  for non-likelihood objectives. Legacy `GAIC`/`AIC` column names
  carry the selected criterion value; always read the accompanying
  `criterion`, `normalization` and `df_convention`. Gaussian
  corrected criteria and LSS update-count heuristics are not on a
  shared GAIC scale. When a criterion is requested, all metrics in a
  row refer to its selected stopping point.

  `selected_learners` is the number of distinct base learners the
  path updated, up to the model's own mstop; a distributional model
  adds the counts of its parameters.

  `risk` is withheld (`None`, with a `risk_unavailable` reason on
  every row) when the models do not share one loss: different families,
  or one family with different settings, such as two quantile levels.
  Risks in different units cannot be ranked.
- `labels` (list of str, optional): Custom model labels.
- `sort` (str, optional): A column to order the rows by, best first: smallest first for every
  metric except `logLik`, largest first. Rows whose value is missing
  go last. Sorting by `risk` when it is withheld, or by `GAIC` or
  `AIC` when the rows use different criteria, raises `ValueError`.
  Unset keeps the order the models were given in.
- `cv` (list of CvRiskResult or None, optional): One cross-validation result per model, or `None` for a model without
  one. Adds `cv_mstop` and `cv_risk`: the stopping point the
  cross-validated risk prefers and the risk there. All supplied results
  must carry the same non-empty `fold_fingerprint`, so every model is
  scored on the same folds.
- `as_frame` (bool, optional): Superseded by `output=`. `True` returns a data frame, `False`
  returns records. Leaving it unset — the default — returns records,
  which is what this function has always returned.
- `output` ({"auto", "pandas", "polars", "records"}, optional): Output format. Overrides `as_frame` if both are given.
- `common_sample` (bool, default False): Score every model on the rows all of them can use, refitting each on
  the intersection of their kept rows, as `gaic_table` does. The
  excluded rows are reported per row. Cannot be combined with `cv=`
  when rows are excluded, because the cross-validation results were
  computed on the full sample.

**Returns**

DataFrame or list of dict
    One row per model with requested metrics. Models on different rows,
    responses or effective weights raise `ValueError`. Models that share
    their rows but differ in response transform or weight policy keep the
    given order, each row carries a `comparison` reason, and `sort=`
    raises, since the rows are not on one scale.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/model_selection.py` lines 453-598](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/model_selection.py#L453-L598)
