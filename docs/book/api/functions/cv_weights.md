# `cv_weights`

Defined in `mboostrs.cvrisk`; exported as `mboostrs.cv_weights`.


*Function*

```python
cv_weights(n, type='bootstrap', B=25, prob=0.5, seed=42, weights=None)
```

Generate cross-validation fold weights.

**Parameters**

- `n` (int): Number of observations.
- `type` (str): Resampling type ("bootstrap", "kfold", "subsampling").
- `B` (int): Number of folds / bootstrap replications.
- `prob` (float): Proportion of observations per fold (bootstrap/sub-sampling).
- `seed` (int): Random seed.
- `weights` (array-like or None): Observation weights (passed through to folds).

**Returns**

np.ndarray
    (n, B) matrix of fold membership weights.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `RuntimeError`

Source: [`python/mboostrs/cvrisk.py` lines 932-958](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/cvrisk.py#L932-L958)
