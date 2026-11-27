# `deviance_profile`

Defined in `mboostrs.diagnostics`; exported as `mboostrs.deviance_profile`.


*Function*

```python
deviance_profile(
    model,
    xvar=None,
    bins=10,
    newdata=None,
    mstop=None,
    weights=None,
    output='auto'
)
```

Bin joint deviance contributions along a numeric covariate.

LSS deviance is joint across parameters and cannot be assigned to a single
parameter update. The profile therefore localizes observations where the
fitted distribution performs poorly; it is not a per-term attribution.
Equal-count bins keep the support per row comparable.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `TypeError`
- `ValueError`

Source: [`python/mboostrs/diagnostics.py` lines 670-725](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/diagnostics.py#L670-L725)
