# `stratified_folds`

Defined in `mboostrs.folds`; exported as `mboostrs.stratified_folds`.


*Function*

```python
stratified_folds(y, B=10, *, seed=42, weights=None) -> FoldScheme
```

K-fold splits that preserve the class proportions of `y`.

A plain 10-fold split of 200 rows carrying 12 positives leaves folds with
0, 1, 2 or 3 of them; a fold holding none produces an out-of-fold risk
that estimates nothing, and the spread across folds measures the split
rather than the model. Here each class is partitioned separately and the
pieces are combined, so every fold holds out as near as possible the same
share of each class.

`y` is the stratifying variable: the response for a classification fit,
but any label vector works -- a treatment arm, a site.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/folds.py` lines 563-609](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/folds.py#L563-L609)
