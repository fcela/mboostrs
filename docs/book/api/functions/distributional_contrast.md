# `distributional_contrast`

Defined in `mboostrs.effects`; exported as `mboostrs.distributional_contrast`.


*Function*

```python
distributional_contrast(
    model,
    data,
    variable,
    reference,
    alternative,
    quantity,
    value=None,
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
    support_columns=None,
    refit=None
)
```

Contrast a fitted LSS mean, quantile, or exceedance probability.

All response-scale distribution parameters are predicted under the two
settings before the requested functional is evaluated. A quantile can
therefore change through location, scale, or shape; an exceedance
probability can change even when the mean does not. Exceedance uses the
strict convention `P(Y > value) = 1 - F(value)`, including discrete and
mixed families.

The averaging, weighting, support and causal limits are those of
`prediction_contrast`. `value` is the probability for a quantile,
the response threshold for an exceedance probability, and must be omitted
for a mean.

`refit` is a `mboostrs.gamlss_refit` result for the same model. Its
coefficient draws (`n_draws` of them, from `seed`) give one contrast per
draw, and the interval is their `level` percentile range. This is the
refit's approximate posterior, which includes the coefficient uncertainty of
the distributional fit; the point estimate still comes from `model`. It
cannot be combined with `cluster=`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/effects.py` lines 1316-1403](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/effects.py#L1316-L1403)
