# `pit`

Defined in `mboostrs.diagnostics`; exported as `mboostrs.pit`.


*Function*

```python
pit(model, newdata=None, mstop=None, randomized=True, random_state=None)
```

Probability integral transform for LSS models with CDF support.

Standard (non-LSS) models do not have a full distribution CDF and raise
`NotImplementedError`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `NotImplementedError`

Source: [`python/mboostrs/diagnostics.py` lines 189-202](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/diagnostics.py#L189-L202)
