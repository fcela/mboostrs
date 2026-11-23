# `stabsel_gamboostLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.stabsel_gamboostLSS`.


*Function*

```python
stabsel_gamboostLSS(
    formula,
    data,
    families=None,
    family=None,
    control=None,
    folds=None,
    q=None,
    PFER=None,
    cutoff=None,
    B=50,
    sampling_type='SS',
    assumption='unimodal',
    seed=42,
    weights=None,
    bins=None,
    env=None,
    weight_policy='upstream',
    nu=None,
    stabilization=None,
    fit_budget=None,
    na_action='error'
)
```

Stability selection for a mixed structured `gamboostLSS` model.

`na_action` is the argument of `gamboostLSS`, with the same values
and meaning. `"omit"` drops a row with a missing value in a column any
parameter formula reads from the data, the weights and any supplied fold
matrix, so every subsample refers to the same rows.

Same interface and error-control math as `stabsel_glmboostLSS`, but for
structured learners. It runs non-cyclic `gamboostLSS` on each subsample
and tracks every combined `"<learner>.<parameter>"` component. Learners
are rebuilt with the subsample weights, including centering and
`df -> lambda`.

As in `stabsel_glmboostLSS`, `control`'s `mstop` bounds each
subsample's path rather than fixing it: a fold stops once it has selected
`q` distinct components.

`B` is the number of initial subsamples or complementary pairs used in
the error bound. Under `sampling_type="SS"`, each pair fits both halves,
for `2 * B` fits; `sampling_type="MB"` fits `B` subsamples. The
result reports the actual number of fits as `n_fits`.

`bins=` applies the same global quantile-bin count as
`gamboostLSS`, and `LSSModel.stabsel()` inherits the value its fit
was given. This is the single largest lever on the cost of the run: each
fold rebuilds every learner against its own weights, and a binned rebuild
accumulates a bin-weight vector where the exact one copies an `n x p`
design and forms `X'WX` on it.


`LSSModel.stabsel()` is the model-based interface and uses the fitted
object's specification. This function remains available when folds must
be supplied explicitly.

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
- `TypeError`
- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 7386-7574](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L7386-L7574)
