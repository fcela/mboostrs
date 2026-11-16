# `accumulated_local_effect`

Defined in `mboostrs.effects`; exported as `mboostrs.accumulated_local_effect`.


*Function*

```python
accumulated_local_effect(
    model,
    data,
    variable,
    bins=10,
    type='response',
    parameter=None,
    weights=None,
    observation_ids=None,
    population='supplied rows',
    extrapolation='error',
    mstop=None
)
```

Estimate a centered univariate accumulated local effect.

Weighted empirical quantiles partition `variable`. Within each interval,
the model predicts its resident rows at the interval's lower and upper
boundaries while keeping their other predictors unchanged. Weighted local
differences are accumulated across intervals and centered to weighted mean
zero over `population`. This avoids the global predictor combinations of
partial dependence, but sparse intervals and extrapolated input rows remain
visible in the returned counts and support policy.

The interval effect is placed at its midpoint by accumulating prior local
changes plus half the current one. This is a finite-bin approximation to
the ALE integral. It describes the fitted model and carries no sampling,
model-selection, or causal uncertainty.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/effects.py` lines 1406-1527](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/effects.py#L1406-L1527)
