# `quantile_boost`

Defined in `mboostrs.quantile`; exported as `mboostrs.quantile_boost`.


*Function*

```python
quantile_boost(
    formula: str,
    data: Any,
    quantiles: Any = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95),
    weights: Any = None,
    folds: Any = 5,
    max_mstop: int = 500,
    nu: float = 0.3,
    bins: int | str | None = 'auto',
    response_scale: str | float | None = 'auto',
    seed: int = 42,
    na_action: str = 'impute'
) -> MultiQuantileModel
```

Simultaneously CV-tune and fit multiple conditional quantile models in one Rust pass.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/quantile.py` lines 324-612](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/quantile.py#L324-L612)
