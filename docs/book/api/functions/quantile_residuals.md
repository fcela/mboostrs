# `quantile_residuals`

Defined in `mboostrs.diagnostics`; exported as `mboostrs.quantile_residuals`.


*Function*

```python
quantile_residuals(model, newdata=None, mstop=None, randomized=True, random_state=None)
```

Randomized quantile residuals.

For LSS families with CDF support, this is `Phi^{-1}(F(y))`. For
standard Gaussian models, it reduces to normalized response residuals.
For other standard families, raises `NotImplementedError`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `NotImplementedError`

Source: [`python/mboostrs/diagnostics.py` lines 187-211](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/diagnostics.py#L187-L211)
