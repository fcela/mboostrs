# `outlier_frame`

Defined in `mboostrs.diagnostics`; exported as `mboostrs.outlier_frame`.


*Function*

```python
outlier_frame(model, newdata=None, mstop=None, output='auto')
```

Observations ranked by absolute residual, largest first.

Records are ranked before the requested frame is built, so row order is
consistent across pandas and Polars output.

Source: [`python/mboostrs/diagnostics.py` lines 801-811](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/diagnostics.py#L801-L811)
