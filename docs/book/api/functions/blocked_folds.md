# `blocked_folds`

Defined in `mboostrs.folds`; exported as `mboostrs.blocked_folds`.


*Function*

```python
blocked_folds(n, B=5, *, weights=None) -> FoldScheme
```

Split rows into contiguous held-out blocks.

For data with local dependence -- a time series, a spatial transect --
where neighbouring rows are nearly the same observation. A random split
holds out a row whose immediate neighbours are in the training data, which
for smooth serial dependence is close to holding out nothing.

Training may include rows after a held-out block, so these folds measure
prediction across local blocks, not forecasting into future periods.
Use `rolling_folds` to evaluate forecasting.

Source: [`python/mboostrs/folds.py` lines 691-715](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/folds.py#L691-L715)
