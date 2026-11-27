# `glmboostLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.glmboostLSS`.


*Function*

```python
glmboostLSS(
    formula,
    data,
    families=None,
    family=None,
    control=None,
    offset=None,
    mstop=None,
    method='cyclic',
    weights=None,
    env=None,
    weight_policy='upstream',
    stabilization=None,
    nu=None,
    na_action='error',
    response_scale='auto'
)
```

Fit a linear GAMLSS by component-wise boosting.

Workflow, pitfalls and worked analyses: `mboostrs.agent_guide()`, or
`python -m mboostrs guide index` from a shell. After a fit, run
`mboostrs.reference_gap` to measure what a flexible model still gains.

`method="cyclic"` (default) updates every parameter each iteration;
`method="noncyclic"` updates only the single parameter that most reduces
the risk each step (R's noncyclic, Thomas et al. 2018), so `mstop` is a
single total-step budget and the per-parameter mstop argument is ignored.

**Parameters**

- `formula` (str | dict): One formula shared by every parameter, or `{param_name: formula}` to
  give parameters different predictor sets. The response must be the same
  across parameters.
- `data` (mapping / DataFrame)
- `families` (LSS family instance, optional): The family fitted to the response. Defaults to `GaussianLSS()`.
- `family` (LSS family instance, optional): Alias for `families=`. Pass only one of these arguments.
- `control` (boost_control tuple (default ``boost_control(100, 0.1)``).)
- `offset` (optional list of per-parameter offset overrides.)
- `mstop` (optional per-parameter step budget (R's mstop vector). ``None`` uses): `control`'s scalar mstop for every parameter; an int sets the same for
  all; a list (family order) or `{param_name: int}` dict sets per
  parameter. Cyclic: parameter k takes exactly its mstop steps.
- `weights` (array-like of length n, optional): Observation weights. Must be non-negative. Default is equal weights.
- `stabilization` ({"none", "MAD", "L2"}, optional): Partial-gradient stabilization, upstream's `stabilize_ngradient`.
  `None` (default) uses the family's own `stabilization=` setting —
  upstream gamboostLSS sets it on the family constructor
  (`GaussianLSS(stabilization="MAD")`); passing one of the three
  values here overrides the family for this fit, an mboostrs extension.
- ```nu=`` sets per-parameter learning rates, upstream's`
- ```boost_control(nu = c(...))`` vector: ``None`` (default) keeps the`
- `control's scalar, a float recycles it, a list is in family order, and a`
- ```{param_name: float}`` dict must name every parameter.`
- `A lower rate slows updates to that parameter relative to the others. If`
- `the fitted mean and scale curves show different convergence, compare rate`
- `choices on the same folds with the joint score; the score alone does not`
- `identify which parameter accounts for residual structure.`
- ```na_action`` handles missing values in the columns any parameter`
- `formula reads, as in :func:`mboostrs.gamboost`: ``"error"`` (default)`
- `refuses them, ``"omit"`` drops those rows, and ``"impute"`` fills each`
- `numeric predictor with its training median plus a missing-value`
- `indicator term in every parameter that reads it (a categorical`
- `predictor gets a ``"(missing)"`` level). The imputation is part of the`
- `model's predictor recipe, so ``predict`` handles new rows with missing`
- `values the same way.`

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `TypeError`
- `RuntimeError`
- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 6000-6242](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L6000-L6242)
