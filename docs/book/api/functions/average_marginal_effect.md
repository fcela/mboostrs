# `average_marginal_effect`

Defined in `mboostrs.effects`; exported as `mboostrs.average_marginal_effect`.


*Function*

```python
average_marginal_effect(
    model,
    data,
    variable,
    step=None,
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

Average row-specific finite derivatives of the full fitted prediction.

Other predictors retain each row's observed value. Interior rows use a
centered finite difference; rows within half a step of the training boundary
use the available one-sided interval. The denominator is each row's actual
upper-minus-lower value, so boundary truncation does not silently change the
derivative scale. `step` defaults to one ten-thousandth of the nonzero
training span and is recorded in the result.

On a nonidentity link, response derivatives transform the complete predictor
before differencing. The returned average standardizes a fitted derivative
over `population`; it is neither a coefficient nor a causal effect by
itself. `cluster`, `level`, `n_draws` and `seed` give the
cluster-level uncertainty of the average, as in `prediction_contrast`;
without `cluster` the average has no sampling uncertainty attached.
`joint_support` checks the evaluation points against the training
combinations, as in `prediction_contrast`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/effects.py` lines 1083-1197](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/effects.py#L1083-L1197)
