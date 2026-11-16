# `cvrisk_gamboostLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.cvrisk_gamboostLSS`.


*Function*

```python
cvrisk_gamboostLSS(
    formula,
    data,
    families=None,
    family=None,
    control=None,
    folds=None,
    grid=None,
    bins=None,
    env=None,
    weight_policy='upstream',
    nu=None,
    stabilization=None,
    weights=None,
    na_action='error',
    _search=None
)
```

Cross-validated risk for a mixed structured `gamboostLSS` model.

`folds` is an `(n, B)` matrix of in-bag weights, a
`mboostrs.folds.FoldScheme`, or an integer fold count. An integer
`k` builds k-fold weights with `mb.cv_weights(n, type='kfold', B=k)`
(seed 42), so `folds=5` is enough; it is required.

`na_action` is the argument of `gamboostLSS`, with the same values
and meaning. `"omit"` drops a row with a missing value in a column any
parameter formula reads, from the data, the weights and the fold matrix
together, so every fold fit and every held-out score sees the same rows.
`"error"` (default) refuses a missing value as the fit does.

Same interface and return value as `cvrisk_glmboostLSS`, but for smooth
Learners are prepared against each fold's weights, so centering and
`df -> lambda` are recomputed while fold-invariant spline templates remain
cached.

Returns a dict with `risk` (B x len(grid) per-fold OOB risk), `grid`,
and `mstop` (the grid point minimizing the mean OOB risk).

`bins=` applies the same global quantile-bin count as
`gamboostLSS`, and `LSSModel.cvrisk()` inherits the value its fit
was given. It is the argument that decides whether the per-fold rebuild
accumulates a bin-weight vector or a full `X'WX`.


`LSSModel.cvrisk()` is the model-based interface: it obtains the
specification and default grid from the fitted model. This function
remains available when folds or a grid must be supplied explicitly.

`nu=` sets per-parameter learning rates, upstream's
`boost_control(nu = c(...))` vector: `None` (default) keeps the
control's scalar, a float recycles it, a list is in family order, and a
`{param_name: float}` dict must name every parameter.
A lower rate slows updates to that parameter relative to the others. If
the fitted mean and scale curves show different convergence, compare rate
choices on the same folds with the joint score; the score alone does not
identify which parameter accounts for residual structure.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `RuntimeError`
- `ValueError`
- `TypeError`

Source: [`python/mboostrs/gamlss.py` lines 6867-7055](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L6867-L7055)
