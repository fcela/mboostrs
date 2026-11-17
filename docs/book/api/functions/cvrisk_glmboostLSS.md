# `cvrisk_glmboostLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.cvrisk_glmboostLSS`.


*Function*

```python
cvrisk_glmboostLSS(
    formula,
    data,
    families=None,
    family=None,
    control=None,
    folds=None,
    grid=None,
    weight_policy='upstream',
    nu=None,
    stabilization=None,
    weights=None
)
```

Cross-validated risk for a linear GAMLSS over a grid of per-parameter
mstop combinations (R's `cvrisk.mboostLSS` for `glmboostLSS`).

**Parameters**

- `formula, data, families, control` (as in :func:`glmboostLSS`.)
- `folds` ((n, B) array of in-bag weights (one column per fold), a): `mboostrs.folds.FoldScheme`, or an integer fold count. An
  integer `k` builds k-fold weights with
  `mb.cv_weights(n, type='kfold', B=k)` (seed 42). Required.
- `grid` (list of per-parameter mstop vectors (each length ``family.n_params``).): The out-of-bag risk is returned for each grid point.
- `Returns a dict with ``risk`` (B x len(grid) per-fold OOB risk), ``grid``,`
- `and ``mstop`` (the grid point minimizing the mean OOB risk).`
- ```LSSModel.cvrisk()`` is the model-based interface: it obtains the`
- `specification and default grid from the fitted model. This function`
- `remains available when folds or a grid must be supplied explicitly.`
- ```nu=`` sets per-parameter learning rates, upstream's`
- ```boost_control(nu = c(...))`` vector: ``None`` (default) keeps the`
- `control's scalar, a float recycles it, a list is in family order, and a`
- ```{param_name: float}`` dict must name every parameter.`
- `A lower rate slows updates to that parameter relative to the others. If`
- `the fitted mean and scale curves show different convergence, compare rate`
- `choices on the same folds with the joint score; the score alone does not`
- `identify which parameter accounts for residual structure.`

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `RuntimeError`
- `ValueError`
- `TypeError`

Source: [`python/mboostrs/gamlss.py` lines 6729-6866](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L6729-L6866)
