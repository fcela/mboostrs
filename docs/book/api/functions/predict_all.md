# `predict_all`

Defined in `mboostrs.diagnostics`; exported as `mboostrs.predict_all`.


*Function*

```python
predict_all(
    model,
    newdata=None,
    type='link',
    mstop=None,
    include_response=True,
    output='auto',
    extrapolation='warn'
)
```

Return a table of fitted values (and response for in-sample).

For LSS models, one column per parameter plus `y`.
For standard models, `fitted` plus `y`. `extrapolation` is as in
the model's `predict`; a linear model has no range to check and ignores it.

Source: [`python/mboostrs/diagnostics.py` lines 47-80](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/diagnostics.py#L47-L80)
