# `residual_frame`

Defined in `mboostrs.diagnostics`; exported as `mboostrs.residual_frame`.


*Function*

```python
residual_frame(
    model,
    kind='response',
    newdata=None,
    mstop=None,
    randomized=True,
    random_state=None,
    output='auto'
)
```

Diagnostic residual table for any model.

`kind` is one of the kinds `mboostrs._residuals` defines --
`"response"`, `"pearson"`, `"working"`, `"deviance"`,
`"martingale"` -- plus `"quantile"` for randomized quantile residuals,
which are uniform under a correct fit and so are the one kind comparable
across families.

The arithmetic is the same as `residuals` on the model. The two used
to disagree: for a Gaussian fit this divided by the residual RMS while the
method did not, and both called the result Pearson.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/diagnostics.py` lines 240-296](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/diagnostics.py#L240-L296)
