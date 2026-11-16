# `repeated_kfold_folds`

Defined in `mboostrs.folds`; exported as `mboostrs.repeated_kfold_folds`.


*Function*

```python
repeated_kfold_folds(n, k=10, repeats=5, *, seed=42, weights=None) -> FoldScheme
```

`repeats` independent k-fold partitions, side by side.

One k-fold split has a variance of its own: the risk it reports depends on
which partition was drawn, and with a small sample that dependence can be
larger than the difference between two models. Repeating the split and
averaging over partitions reduces it. The columns are all `k * repeats`
folds, so the aggregate is over every fold of every repeat.

Source: [`python/mboostrs/folds.py` lines 929-963](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/folds.py#L929-L963)
