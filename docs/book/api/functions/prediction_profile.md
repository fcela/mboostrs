# `prediction_profile`

Defined in `mboostrs.effects`; exported as `mboostrs.prediction_profile`.


*Function*

```python
prediction_profile(
    model,
    data,
    variable,
    grid=None,
    points=25,
    type='response',
    parameter=None,
    weights=None,
    observation_ids=None,
    population='supplied rows',
    extrapolation='error',
    retain_ice=True,
    support_threshold=2.5,
    support_columns=None,
    mstop=None
)
```

Weighted partial dependence and optional individual conditional curves.

At each grid value, `variable` is replaced for every supplied row while
all other predictors retain their row values. The weighted average is the
partial-dependence curve; the prediction matrix contains individual
conditional-expectation (ICE) curves. The default grid uses weighted
empirical quantiles and therefore remains dense where the declared
population has observations.

`support_fraction_far` is the weighted fraction of counterfactual rows
whose replaced value lies more than `support_threshold` residual standard
deviations from a weighted linear regression of `variable` on other
finite numeric columns. By default those are the supplied columns the
fitted model's formula reads; a column the model does not use is left
out, because it stays fixed while `variable` is replaced and, if it is
derived from `variable` (an indicator of old buildings beside the
construction year), it would report combinations as unsupported that the
model never sees. `support_columns=` names the conditioning columns
explicitly. A formula with a `.` term reads every column, so then every
supplied column is used. `support["conditioning_columns"]` and
`support["excluded_columns"]` record the choice. This detects linear
joint-support problems; nonlinear and categorical dependence may be
missed. ALE is preferable when global replacements create unsupported
combinations.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`
- `TypeError`

Source: [`python/mboostrs/effects.py` lines 1627-1740](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/effects.py#L1627-L1740)
