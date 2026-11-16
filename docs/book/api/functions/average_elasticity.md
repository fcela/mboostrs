# `average_elasticity`

Defined in `mboostrs.effects`; exported as `mboostrs.average_elasticity`.


*Function*

```python
average_elasticity(
    model,
    data,
    variable,
    kind='elasticity',
    log_step=0.001,
    parameter=None,
    weights=None,
    observation_ids=None,
    population='supplied rows',
    extrapolation='error',
    mstop=None
)
```

Average a finite elasticity or semi-elasticity of response prediction.

`kind="elasticity"` computes `d log(prediction) / d log(variable)`;
both the predictor and prediction must be positive. `kind="semi_elasticity"`
computes `d log(prediction) / d variable` and only the prediction must be
positive. Evaluation points are multiplicatively separated by `log_step`
for elasticity and use the corresponding local additive separation for the
semi-elasticity. Training-boundary rows use one-sided intervals.

The result describes the fitted response surface over `population`. It is
not a coefficient when the response link is nonlinear, and it is not a
causal effect without an identified intervention.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/effects.py` lines 1200-1313](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/effects.py#L1200-L1313)
