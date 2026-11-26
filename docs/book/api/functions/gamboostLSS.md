# `gamboostLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.gamboostLSS`.


*Function*

```python
gamboostLSS(
    formula,
    data,
    families=None,
    family=None,
    control=None,
    offset=None,
    mstop=None,
    weights=None,
    method='cyclic',
    bins=None,
    env=None,
    weight_policy='upstream',
    stabilization=None,
    nu=None,
    na_action='error',
    response_scale='auto'
)
```

Fit a distributional additive model with linear, spline, and structured learners.

Workflow, pitfalls and worked analyses: `mboostrs.agent_guide()`, or
`python -m mboostrs guide index` from a shell. After a fit, run
`mboostrs.reference_gap` to measure what a flexible model still gains.

Each distribution parameter owns a component-wise learner competition. A
single formula is reused for every parameter; a mapping keyed by family
parameter names supplies distinct formulas. `method="cyclic"` updates one
learner in every active parameter per cycle. `method="noncyclic"` chooses
one parameter/learner pair per global boosting step.

`bins=` sets a global quantile-bin count for every `bols`/`bbs`/
`bspatial`/`brandom` term that does not carry its own. An LSS fit
prepares the candidate set once per distribution parameter, so binning
saves a multiple of what it saves on `gamboost`: a 300,000-row,
six-spline, two-parameter fit measured 2.9 s exact against 0.44 s at
`bins=100`, and its stability selection 82 s against 1.0 s, on the same
selected set. Terms with no binned design keep their exact storage and are
named in a warning; see the `bins` note on `mboostrs.gamboost`.

Partial-gradient stabilization is set on the family
(`GaussianLSS(stabilization="MAD")`, upstream's spelling); passing
`stabilization=` to this fit overrides the family for this fit only.
`nu=` sets per-parameter learning rates, upstream's
`boost_control(nu = c(...))` vector: `None` (default) keeps the
control's scalar, a float recycles, a list is family order, and a
`{param_name: float}` dict names every parameter.
A lower rate slows updates to that parameter relative to the others. If
the fitted mean and scale curves show different convergence, compare rate
choices on the same folds with the joint score; the score alone does not
identify which parameter accounts for residual structure.

`na_action` handles missing values in the columns any parameter
formula reads, as in `mboostrs.gamboost`: `"error"` (default)
refuses them, `"omit"` drops those rows, and `"impute"` fills each
numeric predictor with its training median plus a missing-value
indicator term in every parameter that reads it (a categorical
predictor gets a `"(missing)"` level). The imputation is part of the
model's predictor recipe, so `predict` handles new rows with missing
values the same way.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `RuntimeError`
- `TypeError`
- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 6498-6730](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L6498-L6730)
