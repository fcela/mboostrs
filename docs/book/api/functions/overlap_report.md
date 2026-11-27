# `overlap_report`

Defined in `mboostrs.causal`; exported as `mboostrs.overlap_report`.


*Function*

```python
overlap_report(source, data=None, bounds=from_spec, *, covariates=None)
```

Summarize where the estimated propensity leaves the overlap bounds.

**Parameters**

- `source` (array, AIPWResult or EffectModel): Cross-fitted propensities `P(treated | x)` for each row, or a result
  that carries them. For a result, the covariates default to its
  specification's and the estimate is reported next to the restricted
  one.
- `data` (mapping or data frame, optional): The rows' covariate columns. Without it the covariate regions are not
  computed.
- `bounds` (tuple, optional): `(lower, upper)`: rows with propensity below `lower` or above
  `upper` are outside. Defaults to the `overlap_bounds` of the
  result's spec, and to `(0.02, 0.98)` for a bare array or an
  `EffectModel` without one. `(0.02, 0.98)` is a convention,
  not a theorem; state the bounds used.
- `covariates` (sequence of str, optional): Columns of `data` to summarize when `source` is a bare array. A result
  uses its own covariates.

**Returns**

OverlapReport
    `share_outside` is the fraction of rows outside the bounds.
    `regions` locates them by covariate. `estimate_inside` is the
    estimand of the result restricted to the rows inside the bounds: the
    mean augmented score for an ATE, the treated-row ratio for an ATT, and
    the mean effect over the fitting rows for an EffectModel. For
    `aipw(target="overlap")` it is None, since the overlap scores are not
    ATE scores.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/causal.py` lines 3166-3244](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L3166-L3244)
