# `prediction_contrast`

Defined in `mboostrs.effects`; exported as `mboostrs.prediction_contrast`.


*Function*

```python
prediction_contrast(
    model,
    data,
    variable,
    reference,
    alternative,
    type='response',
    parameter=None,
    weights=None,
    observation_ids=None,
    population='supplied rows',
    extrapolation='error',
    mstop=None,
    cluster=None,
    level=0.95,
    n_draws=1000,
    seed=0,
    joint_support='report',
    joint_threshold=1.0,
    support_columns=None
)
```

Average a full-model prediction change over the supplied rows.

Two copies of `data` are made. `variable` is set to `reference` in
one and `alternative` in the other; every other predictor retains each
row's observed value. The returned estimate is the weighted mean of the
row-wise full-prediction differences. On a nonidentity link, response-scale
contrasts transform the complete predictor before subtraction rather than
exponentiating or transforming an isolated component.

`cluster` names a cluster label per supplied row (a column's values, an
array of length n). When it is given, the result carries a cluster-level
uncertainty: the cluster-robust standard error of the weighted mean and a
cluster-bootstrap percentile interval at `level`, from `n_draws`
resamples of clusters with `seed`. The clusters are the sampling units,
and the fitted model is held fixed.

`joint_support` checks the counterfactual combinations. Setting
`variable` to `reference` keeps every other predictor at its row value,
and that combination may never occur in the training data (a categorical
level seen only with other covariate values, or a high `x1` with a low
correlated `x2`). A row is jointly supported when the training data has a
row in the same categorical cell within `joint_threshold` training
standard deviations on the numeric predictors the model reads. The policy
`"report"` records the supported share in `support["joint"]`;
`"warn"` also warns, `"error"` raises, and `"off"` skips the check.
`support_columns` names the conditioning columns, as in
`prediction_profile`. The check costs one pass over the training rows
per standardization row and per counterfactual setting; pass
`joint_support="off"` for very large data.

This is standardization over `population`, not a causal effect. A causal
reading additionally needs a well-defined intervention, baseline adjustment
set, exchangeability, positivity, consistency and an interference design.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/effects.py` lines 986-1080](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/effects.py#L986-L1080)
