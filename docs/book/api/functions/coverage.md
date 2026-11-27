# `coverage`

Defined in `mboostrs.diagnostics`; exported as `mboostrs.coverage`.


*Function*

```python
coverage(observed, lower, upper)
```

Compute empirical coverage of prediction intervals.

Returns `(coverage_rate, avg_width)` using inclusive bounds. Inputs
must be non-empty one-dimensional arrays of identical shape. Observed
values must be finite; unbounded intervals may use negative infinity for
lower bounds and positive infinity for upper bounds. NaN or reversed
bounds are refused. Coverage on training observations does not estimate
future-outcome coverage.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/diagnostics.py` lines 840-866](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/diagnostics.py#L840-L866)
