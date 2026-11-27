# `deviance_increment`

Defined in `mboostrs.diagnostics`; exported as `mboostrs.deviance_increment`.


*Function*

```python
deviance_increment(model, newdata=None)
```

Per-observation deviance contributions.

For LSS families with `deviance_increment`, delegates to the family.
For supported standard exponential-family models, dispatches on family
name and computes the corresponding deviance contribution. Families whose
loss is not a likelihood deviance raise `NotImplementedError`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `NotImplementedError`

Source: [`python/mboostrs/diagnostics.py` lines 154-182](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/diagnostics.py#L154-L182)
