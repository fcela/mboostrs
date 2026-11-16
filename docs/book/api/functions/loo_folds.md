# `loo_folds`

Defined in `mboostrs.folds`; exported as `mboostrs.loo_folds`.


*Function*

```python
loo_folds(n, *, weights=None) -> FoldScheme
```

Leave-one-out: fold `i` holds out row `i` alone (`n` folds).

mboost's `cv(type = "kfold", B = n)`. Every row is judged by a fit to
all the others, so the estimate of prediction error has little bias, at
the cost of `n` fits -- one per row -- and a high variance between
neighbouring rows' contributions. A row with zero model weight is out of
the sample and is held out by no fold. For grouped data use
`grouped_folds`, which leaves out whole groups.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/folds.py` lines 718-743](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/folds.py#L718-L743)
