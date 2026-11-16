# `stabsel_glmboostLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.stabsel_glmboostLSS`.


*Function*

```python
stabsel_glmboostLSS(
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
    weight_policy='upstream',
    nu=None,
    stabilization=None,
    fit_budget=None
)
```

Stability selection for a linear GAMLSS (R's stabsel.mboostLSS).

Runs non-cyclic glmboostLSS on `B` half-samples, plus their exact
complements under `sampling_type="SS"`, tracking how often each *combined*
base-learner (`"<predictor>.<parameter>"`) is selected, and returns the
learners whose maximum selection frequency reaches the error-control cutoff.

The error-control parameters (cutoff/q/PFER) use the same stabs math as
`mboostrs.stabsel.stabsel`. `folds` (an `n x B` in-bag weight
matrix of *initial* subsamples) may be supplied to reproduce a specific
subsampling; under `"SS"` the exact complements are appended here, so
`B` columns produce `2 * B` fits and pairing is never inferred from an
even column count.

`control`'s `mstop` is an upper bound on each subsample's path, not the
number of iterations every subsample runs: a fold stops once it has
selected `q` distinct components, because nothing it selects after that
reaches the reported frequencies. Raising `mstop` therefore costs nothing
on folds that reach `q` early, and only guards against folds that do not
reach it at all -- which is warned about when it happens.

`B` is the number of initial subsamples or complementary pairs used in
the error bound. Under `sampling_type="SS"`, each pair fits both halves,
for `2 * B` fits; `sampling_type="MB"` fits `B` subsamples. The
result reports the actual number of fits as `n_fits`.

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

Source: [`python/mboostrs/gamlss.py` lines 7203-7368](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L7203-L7368)
