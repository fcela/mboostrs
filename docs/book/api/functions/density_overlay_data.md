# `density_overlay_data`

Defined in `mboostrs.diagnostics`; exported as `mboostrs.density_overlay_data`.


*Function*

```python
density_overlay_data(
    model,
    newdata=None,
    mstop=None,
    grid=None,
    n_grid=200,
    weights=None,
    output='auto'
)
```

Observed and average fitted response distributions on one grid.

The fitted curve is the weighted average of the row-specific conditional
densities or masses, not the density evaluated at average parameters.
This distinction matters whenever scale, skewness, or tail shape varies by
row. The observed continuous curve is a Gaussian kernel estimate; for a
discrete response it is the weighted empirical frequency.

The overlay is descriptive in-sample unless `newdata` is held out. It
can reveal a marginal mismatch while hiding offsetting conditional errors;
use `worm_data` or `q_stats` against relevant covariates for
those errors.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `TypeError`
- `NotImplementedError`
- `ValueError`

Source: [`python/mboostrs/diagnostics.py` lines 453-591](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/diagnostics.py#L453-L591)
