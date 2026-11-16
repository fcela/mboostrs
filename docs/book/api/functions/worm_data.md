# `worm_data`

Defined in `mboostrs.diagnostics`; exported as `mboostrs.worm_data`.


*Function*

```python
worm_data(
    model,
    xvar=None,
    n_intervals=4,
    newdata=None,
    mstop=None,
    randomized=True,
    random_state=None,
    output='auto'
)
```

Return worm-plot deviation data.

Bins observations by `xvar` into `n_intervals` groups, then
computes expected vs theoretical normal quantiles within each bin.

Source: [`python/mboostrs/diagnostics.py` lines 392-432](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/diagnostics.py#L392-L432)
