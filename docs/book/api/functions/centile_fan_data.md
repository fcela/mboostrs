# `centile_fan_data`

Defined in `mboostrs.diagnostics`; exported as `mboostrs.centile_fan_data`.


*Function*

```python
centile_fan_data(
    model,
    xvar,
    probs=(0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975),
    split_by=None,
    newdata=None,
    mstop=None,
    output='auto'
)
```

Observed responses and fitted centiles ordered by one numeric covariate.

`probs` must contain 0.5 and symmetric lower/upper pairs so a fan has an
unambiguous median and nested central bands. `split_by` retains a group
label for faceting; it does not refit separate models.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `TypeError`
- `ValueError`

Source: [`python/mboostrs/diagnostics.py` lines 616-667](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/diagnostics.py#L616-L667)
