# `time_residual_check`

Defined in `mboostrs.residual_checks`; exported as `mboostrs.time_residual_check`.


*Function*

```python
time_residual_check(
    model,
    time,
    max_lag: int | None = None,
    alpha: float = 0.05,
    permutations: int = 999,
    draws: int = 200,
    seed: int = 0
) -> RevisionFinding
```

Test whether residuals are correlated across ordered time points.

The residuals are ordered by `time` and averaged within each distinct
time value, so the series has one point per time value and a lag is a step
between consecutive time values. The statistic is the Ljung-Box sum of the
squared sample autocorrelations over lags `1..max_lag`. Its reference is a
permutation distribution: the series is shuffled and the statistic
recomputed, which is exact when the residuals are exchangeable under no
serial dependence. The chi-squared p-value is reported as well.

**Parameters**

- `model` (glmboost or gamboost fit)
- `time` (str or array): Column name of the model's data, or one numeric time per row.
- `max_lag` (int, optional): Highest lag in the statistic. Defaults to `10 * log10(m)`, rounded
  down, where `m` is the number of distinct time values, as R's
  `acf` does.
- `alpha, permutations, draws, seed`: As in `cluster_residual_check`.

**Returns**

RevisionFinding
    `evidence` holds the autocorrelations by lag, the white-noise band
    `1.96 / sqrt(m)` and the lags whose autocorrelation lies outside it.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

**Notes**

A time term already in the formula is noted in the revision text, because
a remaining dependence then points at the error process rather than at
a missing smooth.

Source: [`python/mboostrs/residual_checks.py` lines 384-515](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/residual_checks.py#L384-L515)
