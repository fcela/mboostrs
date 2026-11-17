# `GLMBoostModel`

Defined in `mboostrs.glmboost`; exported as `mboostrs.GLMBoostModel`.


*Class*

```python
GLMBoostModel(
    formula: str,
    family,
    control: tuple,
    response_name: str,
    learner_names: list[str],
    fitted: numpy.ndarray,
    risk: numpy.ndarray,
    selected: list[int],
    coefficients: list[numpy.ndarray],
    offset: float,
    nu: float,
    data=None,
    weights: numpy.ndarray | None = None,
    predictor_names: list[str] | None = None,
    centering_means: list[float] | None = None,
    nuisance_vec: list[float] | None = None,
    nuisance_vec_path: list[list[float]] | None = None,
    learner_store=None,
    execution_report: dict | mboostrs.report.ExecutionReport | None = None
)
```

Fitted gradient boosting model.

Source: [`python/mboostrs/glmboost.py` lines 97-1728](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/glmboost.py#L97-L1728)

## Members

| Member | Kind | Summary |
|---|---|---|
| `path_frame` | Method | Return one structured row per boosting iteration. |
| `family` | Property | The response family the model was fitted with. |
| `formula` | Property | The formula as it was written. |
| `fitted_` | Property | The fitted linear predictor. |
| `fitted` | Method | Fitted values, optionally at a given mstop. |
| `coefficients_` | Property |  |
| `concurvity` | Method | Concurvity of the fitted terms, as mgcv's ``concurvity`` measures |
| `variable_names` | Method | Names of all variables used in the model (response + predictors). |
| `base_learner_names` | Method | Names of all base-learners (including intercept if present). |
| `model_matrix` | Method | Return the design matrix (n × p) for the specified predictors. |
| `nuisance` | Method | The family's nuisance parameter at the model's step count, as R's |
| `predict` | Method | Predicted values, optionally decomposed by learner. |
| `extrapolation_report` | Method | Predictors with rows of `newdata` outside their training range. |
| `predict_path` | Method | Predicted values at every `mstop` in `grid`, in one pass. |
| `prediction_contributions` | Method | Per-learner prediction contributions. |
| `cvrisk` | Method | Cross-validated risk along the path. |
| `coef` | Method | Extract coefficients per predictor. |
| `residuals` | Method | Residuals of the named kind, or a refusal saying why there are none. |
| `residual_kinds` | Method | The residual kinds defined for this model's family. |
| `extract` | Method | Extract model components, matching R's extract.glmboost. |
| `varimp` | Method | Variable importance based on in-bag risk reduction per predictor. |
| `coef_path` | Method | Compute cumulative coefficient paths over boosting iterations. |
| `plot_coef_path` | Method | Plot coefficient paths. |
| `plot` | Method | Plot coefficient paths. |
| `confint` | Method | Bootstrap confidence intervals for coefficients. |
| `aic` | Method | Compute AIC for each boosting iteration. |
| `hatvalues` | Method | Per-observation leverage: the hat-matrix diagonal at the current |
| `logLik` | Method | The fitted model's log-likelihood, or a refusal naming the reason. |
| `negative_risk` | Method | The negated empirical risk at the current mstop. |
| `likelihood_contract` | Method | What this family's risk is: a likelihood, a loss, or neither. |
| `to_dict` | Method | The ``summary(as_dict=True)`` record plus a ``"provenance"`` entry: |
| `summary` | Method | Summarise the fitted model. |
| `update` | Method | Update the model by changing mstop or other parameters. |
| `continue_to` | Method | Return the model grown to a larger ``mstop``. |

(glmboostmodel-path_frame)=
## `GLMBoostModel.path_frame`

*Method* of `GLMBoostModel`

```python
GLMBoostModel.path_frame(output='auto')
```

Return one structured row per boosting iteration.

`output` may be `"auto"`, `"pandas"`, `"polars"`, or
`"records"`. Rows are derived on demand, so fitting overhead is
unchanged when this method is unused.

Source: [`python/mboostrs/glmboost.py` lines 149-164](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/glmboost.py#L149-L164)

(glmboostmodel-family)=
## `GLMBoostModel.family`

*Property* of `GLMBoostModel`

```python
GLMBoostModel.family
```

The response family the model was fitted with.

Source: [`python/mboostrs/glmboost.py` lines 169-172](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/glmboost.py#L169-L172)

(glmboostmodel-formula)=
## `GLMBoostModel.formula`

*Property* of `GLMBoostModel`

```python
GLMBoostModel.formula  # -> str
```

The formula as it was written.

Source: [`python/mboostrs/glmboost.py` lines 174-178](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/glmboost.py#L174-L178)

(glmboostmodel-fitted_)=
## `GLMBoostModel.fitted_`

*Property* of `GLMBoostModel`

```python
GLMBoostModel.fitted_  # -> np.ndarray
```

The fitted linear predictor.

Read-only: this is the fit's own array, not a copy. See
`mboostrs._state`.

Source: [`python/mboostrs/glmboost.py` lines 180-189](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/glmboost.py#L180-L189)

(glmboostmodel-fitted)=
## `GLMBoostModel.fitted`

*Method* of `GLMBoostModel`

```python
GLMBoostModel.fitted(mstop=None) -> ndarray
```

Fitted values, optionally at a given mstop.

mstop: int or None. If None, returns fitted values at the full mstop.
       If given, recomputes fitted values at iteration `mstop`.

Source: [`python/mboostrs/glmboost.py` lines 191-199](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/glmboost.py#L191-L199)

(glmboostmodel-coefficients_)=
## `GLMBoostModel.coefficients_`

*Property* of `GLMBoostModel`

```python
GLMBoostModel.coefficients_  # -> dict[str, float]
```

*No description in the docstring.*

Source: [`python/mboostrs/glmboost.py` lines 203-211](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/glmboost.py#L203-L211)

(glmboostmodel-concurvity)=
## `GLMBoostModel.concurvity`

*Method* of `GLMBoostModel`

```python
GLMBoostModel.concurvity(pairwise=False, output='auto')
```

Concurvity of the fitted terms, as mgcv's `concurvity` measures
it.

Concurvity is the additive-model form of collinearity: how far one
term's function space reaches into another's, which is when partial
effects stop being separately identified -- the same fit can be
redistributed between the terms, so their coefficients and the
selection that chose them are unstable (Wood, *Generalized Additive
Models*, 2nd ed., section 5.6.2). A glmboost term is one column, so
its function space is the multiples of that column and each row of
the default table is ordinary collinearity: the term's centered
column measured against the intercept and every other centered
column (mgcv's `concurvity(full = TRUE)`). Each row gives three
values from 0 (the column is orthogonal to the others) to 1 (it lies
in their span), where near-collinear columns can overshoot 1
slightly from the numerics: `worst`, the largest fraction of the
column's squared norm the other terms can reproduce over every
coefficient vector; `observed`, the same fraction at the fitted
coefficients (NaN for a term that was never selected); and
`estimate`, an average over coefficient directions. For a single
column the three coincide up to rounding.

With `pairwise=True` the rows give mgcv's
`concurvity(full = FALSE)`: one row per ordered pair, where
`term` is measured against `against` alone, no intercept
involved. Each column is first centered, as mgcv measures its
constrained smooths, so the intercept learner (fitted whenever the
formula has an intercept) is constant after centering and reports
NaN in the default table, and every pairwise row naming it too.

**Parameters**

- `pairwise` (bool): Term-against-the-rest rows (the default) or one row per
  ordered pair of terms.
- `output` ({"auto", "records", "polars", "pandas", "frame"}): The return format; see `mboostrs.glmboost` for the
  convention.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/glmboost.py` lines 230-279](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/glmboost.py#L230-L279)

(glmboostmodel-variable_names)=
## `GLMBoostModel.variable_names`

*Method* of `GLMBoostModel`

```python
GLMBoostModel.variable_names()
```

Names of all variables used in the model (response + predictors).

Source: [`python/mboostrs/glmboost.py` lines 284-295](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/glmboost.py#L284-L295)

(glmboostmodel-base_learner_names)=
## `GLMBoostModel.base_learner_names`

*Method* of `GLMBoostModel`

```python
GLMBoostModel.base_learner_names()
```

Names of all base-learners (including intercept if present).

Source: [`python/mboostrs/glmboost.py` lines 297-299](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/glmboost.py#L297-L299)

(glmboostmodel-model_matrix)=
## `GLMBoostModel.model_matrix`

*Method* of `GLMBoostModel`

```python
GLMBoostModel.model_matrix(which=None)
```

Return the design matrix (n × p) for the specified predictors.

**Parameters**

- `which` (str, int, or list thereof, optional): Predictor name(s) or index. None = all predictors.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/glmboost.py` lines 303-320](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/glmboost.py#L303-L320)

(glmboostmodel-nuisance)=
## `GLMBoostModel.nuisance`

*Method* of `GLMBoostModel`

```python
GLMBoostModel.nuisance()
```

The family's nuisance parameter at the model's step count, as R's
`nuisance()` returns it (the size of `NBinomial`, `Var = mu + mu**2 / sigma`, the shape of
`GammaReg`, the thresholds of `PropOdds`); None when there is none.

A family with no nuisance parameter stores an empty path entry. That
is `None`, not a zero-length array: `gamboost` already returned
`None`, and `nuisance()[0]` on an empty array raises `IndexError`.

Source: [`python/mboostrs/glmboost.py` lines 324-342](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/glmboost.py#L324-L342)

(glmboostmodel-predict)=
## `GLMBoostModel.predict`

*Method* of `GLMBoostModel`

```python
GLMBoostModel.predict(
    newdata=None,
    mstop=None,
    type=<object object>,
    which=None,
    aggregate='sum',
    offset=None
)
```

Predicted values, optionally decomposed by learner.

**Parameters**

- `newdata` (dict or DataFrame, optional): New observations. If None, predicts on the training data.
- `offset` (number or array-like, optional): The offset these rows start from. A model fitted with a
  per-observation offset needs the offset of every *new* row, so
  predicting `newdata` requires this argument; the training rows
  reuse the fitted vector.
- `mstop` (int, optional): Truncate the boosting path at this iteration.
- `type` ({"link", "response", "class", "log_odds"}, default "link"): Link-scale or response-scale predictions. The default is the link
  scale (R `mboost` parity): the first default-scale call on a
  non-identity-link family warns once per fitted model, naming the
  scale, because a log-mean is not a mean. `"log_odds"` is the
  link scale times 2 for `Binomial`, whose link is half the
  log-odds; with `which=` it gives each learner's contribution to
  the log-odds, the scale other libraries report.
  `"class"` returns 0/1 class codes for Binomial, AdaExp,
  HingeLoss and AUC. A score on the decision boundary returns 0.
- `which` (int, str, or list thereof, optional): If given, return only the contribution of the specified learner(s).
  Integers are learner indices; strings are learner names.
- `aggregate` ({"sum", "cumsum", "none"}): "sum" returns the aggregated contribution.
  "cumsum" returns cumulative contributions at each step.
  "none" returns per-step contributions.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/glmboost.py` lines 345-468](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/glmboost.py#L345-L468)

(glmboostmodel-extrapolation_report)=
## `GLMBoostModel.extrapolation_report`

*Method* of `GLMBoostModel`

```python
GLMBoostModel.extrapolation_report(newdata, output='auto')
```

Predictors with rows of `newdata` outside their training range.

A GLM has no spline learners, so every term is linear. One row per
predictor with values outside its training range, with the columns of
`GAMBoostModel.extrapolation_report`; `kind` is `"linear"`. A
linear term continues the fitted line. `predict()` does not clip it, so
this report is the only check that the line is being used where the
data supports it. `worst_distance` is in training ranges.

Needs the training data the model stores; a model restored from a
pickle carries none and raises here.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/glmboost.py` lines 470-501](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/glmboost.py#L470-L501)

(glmboostmodel-predict_path)=
## `GLMBoostModel.predict_path`

*Method* of `GLMBoostModel`

```python
GLMBoostModel.predict_path(grid, *, newdata=None, type='link')
```

Predicted values at every `mstop` in `grid`, in one pass.

Equivalent to ``[self.predict(newdata=newdata, mstop=k, type=type)
for k in grid]`` but shares one incremental sweep over the boosting
path instead of redoing the full design-application loop per grid
point — the difference matters for sweeps with many grid points
(path plots, manual early-stopping inspection) on models with many
candidate predictors. Only the `aggregate="sum"`/no-`which`
case is supported; use `predict()` per grid point for per-learner
contributions.

**Parameters**

- `grid` (sequence of int): `mstop` values to evaluate. Need not be sorted; values above
  `self.mstop` are clamped to the full path.
- `newdata` (dict or DataFrame, optional): New observations. If None, predicts on the training data.
- `type` ({"link", "response"}): Link-scale or response-scale predictions.

**Returns**

np.ndarray of shape (len(grid), n)

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/glmboost.py` lines 503-555](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/glmboost.py#L503-L555)

(glmboostmodel-prediction_contributions)=
## `GLMBoostModel.prediction_contributions`

*Method* of `GLMBoostModel`

```python
GLMBoostModel.prediction_contributions(
    newdata=None,
    mstop=None,
    which=None,
    aggregate='sum'
)
```

Per-learner prediction contributions.

**Parameters**

- `newdata` (dict or DataFrame, optional)
- `mstop` (int, optional)
- `which` (int, str, or list thereof, optional): Restrict to specific learner(s).
- `aggregate` ({"sum", "cumsum", "none"})

**Returns**

ndarray
    For `aggregate="sum"`: 1-D array of length n.
    For `aggregate="cumsum"`: 2-D array of shape (mstop, n).
    For `aggregate="none"`: 2-D array of shape (mstop, n).

Source: [`python/mboostrs/glmboost.py` lines 683-707](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/glmboost.py#L683-L707)

(glmboostmodel-cvrisk)=
## `GLMBoostModel.cvrisk`

*Method* of `GLMBoostModel`

```python
GLMBoostModel.cvrisk(
    folds=None,
    type='bootstrap',
    B=25,
    prob=0.5,
    seed=42,
    parallel=True,
    at_mstop=None,
    preprocessing='fixed',
    _inner=False
)
```

Cross-validated risk along the path.

`at_mstop=k` judges the path truncated at `k` steps: the returned
curve, fold rows and grid end there, and the boundary warning is
skipped because the step count is the caller's choice, not a
search's output -- fixing the step answers the question the
warning asks. Pass it when comparing designs or specifications at
a prespecified step count; leave it unset when tuning.

`preprocessing="fixed"` reuses the full-data transformations and
design, matching mboost. `"refit"` learns predictor transformations
on each fold's positive-weight training rows and predicts its held-out
rows through that recipe. Refit folds run sequentially; fitting,
prediction and family scoring use Rust. The result records the mode.

**Parameters**

- `folds` (FoldScheme or array_like of shape (n, B), optional): A `FoldScheme` or an `(n, B)` matrix of in-bag weights,
  used as given; a scheme also carries its judge mask. `None`
  generates `B` folds of resampling type `type`.
- `type` ({"bootstrap", "kfold", "subsampling"}, default "bootstrap"): Resampling used when `folds` is `None`.
- `B` (int, default 25): Number of generated folds when `folds` is `None`.
- `prob` (float, default 0.5): Share of observations drawn per fold for `"subsampling"`.
- `seed` (int, default 42): Random seed for generated folds.
- `parallel` (bool, default True): Run folds in parallel in the Rust engine.
- `at_mstop` (int, optional): Judge the path truncated at this many steps; the curve and grid end
  there and the boundary warning is skipped.
- `preprocessing` ({"fixed", "refit"}, default "fixed"): Reuse the full-data recipe, or learn it within each training fold.

**Returns**

CvRiskResult
    Out-of-fold risk matrix of shape `(n_folds, n_grid)`, the
    aggregate curve over the grid, and `optimal_mstop`.

**Raises**

- `ValueError`: If the fold matrix does not match the number of observations, or
the model keeps no data to refit from.
- `RuntimeError`: If the Rust engine is not available.

Source: [`python/mboostrs/glmboost.py` lines 782-919](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/glmboost.py#L782-L919)

(glmboostmodel-coef)=
## `GLMBoostModel.coef`

*Method* of `GLMBoostModel`

```python
GLMBoostModel.coef(which=None, off2int=False, aggregate='sum')
```

Extract coefficients per predictor.

which: predictor name(s) or index/indices. None = all non-zero.
off2int: if True, add offset to intercept coefficient.
aggregate: "sum" (default) sums per-step coefs; "none" returns raw per-step list.
Returns dict mapping predictor name -> coefficient value.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `TypeError`
- `ValueError`

Source: [`python/mboostrs/glmboost.py` lines 921-989](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/glmboost.py#L921-L989)

(glmboostmodel-residuals)=
## `GLMBoostModel.residuals`

*Method* of `GLMBoostModel`

```python
GLMBoostModel.residuals(type='response', newdata=None, mstop=None)
```

Residuals of the named kind, or a refusal saying why there are none.

**Parameters**

- `type` ({"response", "pearson", "working", "deviance", "martingale"}): `"response"` is `y - mu` with `mu` on the response's own
  scale. `"pearson"` is `(y - mu) / sqrt(V(mu))`, available only
  for families that declare a variance function. `"working"` is the
  fitted family's negative gradient -- what the next boosting step
  regresses on -- and is available for every family. `"deviance"`
  is the signed square root of each observation's deviance
  contribution. `"martingale"` is for `CoxPH`.
- `If a residual kind is undefined for the family, the method raises`
- ``NotImplementedError` and names the available kinds. Use`
- ``residual_kinds()` to inspect them.`
- `See :meth:`residual_kinds` for what this family offers, and`
- `:func:`mboostrs.quantile_residuals` for a kind that is comparable`
- `across families because it is uniform under a correct fit.`

Source: [`python/mboostrs/glmboost.py` lines 998-1022](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/glmboost.py#L998-L1022)

(glmboostmodel-residual_kinds)=
## `GLMBoostModel.residual_kinds`

*Method* of `GLMBoostModel`

```python
GLMBoostModel.residual_kinds()
```

The residual kinds defined for this model's family.

Source: [`python/mboostrs/glmboost.py` lines 1024-1028](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/glmboost.py#L1024-L1028)

(glmboostmodel-extract)=
## `GLMBoostModel.extract`

*Method* of `GLMBoostModel`

```python
GLMBoostModel.extract(what, which=None)
```

Extract model components, matching R's extract.glmboost.

what: one of "design", "coefficients", "residuals", "variable.names",
      "offset", "nuisance", "weights", "control".
      Aliases accepted: "coef", "coefs", "coefficient" -> "coefficients";
      "x", "model_matrix" -> "design"; "w" -> "weights".
which: learner name(s) or zero-based index, including the intercept.
       None selects the used learners, plus the intercept when centered.
"design" returns a dict of the fitted design columns, centered using
the training means when center=True, with a column of ones for the
intercept. "variable.names" returns their names in design order.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/glmboost.py` lines 1030-1093](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/glmboost.py#L1030-L1093)

(glmboostmodel-varimp)=
## `GLMBoostModel.varimp`

*Method* of `GLMBoostModel`

```python
GLMBoostModel.varimp(percent=True)
```

Variable importance based on in-bag risk reduction per predictor.

Returns a dict with keys:
  reduction: dict of predictor_name -> risk reduction
  selfreq: dict of predictor_name -> selection frequency
With `percent=True` (the default) each reduction is divided by the
sum of their absolute values, so the reductions are shares summing
to 1. `percent=False` returns the in-bag risk reduction per
observation, which is what mboost's `varimp()` returns.

When several learners share a name, their reductions and selection
frequencies are summed under that name; the value is the predictor's
total, not one learner's.

Source: [`python/mboostrs/glmboost.py` lines 1095-1121](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/glmboost.py#L1095-L1121)

(glmboostmodel-coef_path)=
## `GLMBoostModel.coef_path`

*Method* of `GLMBoostModel`

```python
GLMBoostModel.coef_path(which=None)
```

Compute cumulative coefficient paths over boosting iterations.

**Parameters**

- `which` (list of str, optional): Predictor names. If None, includes all non-zero predictors.

**Returns**

dict
    Mapping predictor name -> 1D array of cumulative coefficients
    over iterations 0..mstop.

Source: [`python/mboostrs/glmboost.py` lines 1123-1155](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/glmboost.py#L1123-L1155)

(glmboostmodel-plot_coef_path)=
## `GLMBoostModel.plot_coef_path`

*Method* of `GLMBoostModel`

```python
GLMBoostModel.plot_coef_path(which=None)
```

Plot coefficient paths.

**Parameters**

- `which` (list of str, optional): Which predictors to plot.

**Returns**

alt.Chart

Source: [`python/mboostrs/glmboost.py` lines 1157-1171](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/glmboost.py#L1157-L1171)

(glmboostmodel-plot)=
## `GLMBoostModel.plot`

*Method* of `GLMBoostModel`

```python
GLMBoostModel.plot(which=None)
```

Plot coefficient paths.

**Parameters**

- `which` (list of str, optional): Which predictors to plot.

**Returns**

alt.Chart

Source: [`python/mboostrs/glmboost.py` lines 1173-1185](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/glmboost.py#L1173-L1185)

(glmboostmodel-confint)=
## `GLMBoostModel.confint`

*Method* of `GLMBoostModel`

```python
GLMBoostModel.confint(
    parm=None,
    level=0.95,
    B=1000,
    B_mstop=25,
    seed=None,
    n_jobs=1,
    groups=None,
    method='bootstrap'
)
```

Bootstrap confidence intervals for coefficients.

parm: predictor name(s) to compute CIs for. None = all used.
level: confidence level (default 0.95).
B: number of bootstrap replications.
B_mstop: number of inner CV folds for mstop selection (0 = skip).
seed: random seed for reproducibility.
method: "bootstrap". "refit" intervals are currently available through
    GAMBoostModel.confint on a gamboost fit; glmboost names this limit.
n_jobs: bootstrap worker threads (1 = sequential, -1 = all CPUs). Results
    are independent of n_jobs (each replicate has its own seeded RNG).
    Most effective with B_mstop=0; when B_mstop>0 the inner CV is already
    parallel (rayon), so n_jobs>1 mainly contends and adds little.
groups: a column name or one label per row. When given, every
    bootstrap sample (and every inner fold of the `B_mstop` tuning)
    draws whole groups with replacement instead of single rows, and a
    row's weight in a sample is its group's draw count times its
    model weight. Use it when rows are repeated measurements of a
    unit, since resampling such rows separately treats them as
    independent and gives intervals that are too narrow.
Returns a dict with 'lower', 'upper', 'level', 'boot_coefs', and
'resampled' ('rows' or 'groups').

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`
- `NotImplementedError`

Source: [`python/mboostrs/glmboost.py` lines 1191-1345](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/glmboost.py#L1191-L1345)

(glmboostmodel-aic)=
## `GLMBoostModel.aic`

*Method* of `GLMBoostModel`

```python
GLMBoostModel.aic(method='corrected', k=2.0)
```

Compute AIC for each boosting iteration.

Returns an AicResult with AIC values, optimal mstop, and df trace.
Requires that the model was fitted with inbag risk and that all
learners have hat matrices (true for all GLMBoost linear learners).

`method='corrected'` (default) is the upstream per-observation
Gaussian stopping criterion, not a total GAIC. `'gmdl'` is also
Gaussian-only. `'classical'` uses total likelihood risk and refuses
Gaussian squared-error loss. `k` is positive and applies only to
the classical method. Results identify the criterion, normalization
and conditional hat-trace df; the trace excludes model-search df.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `NotImplementedError`
- `ValueError`

Source: [`python/mboostrs/glmboost.py` lines 1423-1504](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/glmboost.py#L1423-L1504)

(glmboostmodel-hatvalues)=
## `GLMBoostModel.hatvalues`

*Method* of `GLMBoostModel`

```python
GLMBoostModel.hatvalues()
```

Per-observation leverage: the hat-matrix diagonal at the current
mstop, as a 1-d array of length n.

Refuses when there is no hat matrix to take a diagonal of -- a `btree`
or `bmono` learner among the selected ones, or a model with no
retained learner store. It used to return `None` for both, so the
caller had a missing value and no reason for it.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`
- `NotImplementedError`

Source: [`python/mboostrs/glmboost.py` lines 1506-1538](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/glmboost.py#L1506-L1538)

(glmboostmodel-loglik)=
## `GLMBoostModel.logLik`

*Method* of `GLMBoostModel`

```python
GLMBoostModel.logLik()
```

The fitted model's log-likelihood, or a refusal naming the reason.

Available only where the fitted loss *is* a log-likelihood. `Binomial`,
`Poisson`, `NBinomial`, `PropOdds` and the AFT families minimise the
exact negative log-likelihood, so this is `-risk`. `Gaussian` minimises the
residual sum of squares, which is not the Gaussian NLL -- no variance,
no `2*pi`, and a factor of two out -- so the log-likelihood is
reconstructed with `sigma^2 = RSS/n`, the MLE the loss implies but
does not carry.

For a robust or ranking loss -- `Laplace`, `Quantile`, `Expectile`,
`Huber`, `AdaExp`, `HingeLoss`, `AUC`, `Gehan`, `Cindex` -- there is no
probability model, so this method raises `NotImplementedError` for
them. `GammaReg` does not refuse. Its risk omits `log(y)` on each row,
and `logLik` adds that term back, so the number is the gamma density
at the fitted mean and the `nuisance()` shape. `negative_risk()`
leaves those terms out. `Multinomial` refuses, because its reported
risk omits the reference class; see `likelihood_contract`.

Unlike `mboost`, this method reports a log-likelihood only when the
family defines one. The Gaussian value includes the normalizing
terms implied by its fitted variance. Use `negative_risk` for
the negated empirical risk convention used by `mboost`.

Source: [`python/mboostrs/glmboost.py` lines 1540-1567](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/glmboost.py#L1540-L1567)

(glmboostmodel-negative_risk)=
## `GLMBoostModel.negative_risk`

*Method* of `GLMBoostModel`

```python
GLMBoostModel.negative_risk()
```

The negated empirical risk at the current mstop.

This matches `mboost`'s `logLik` convention, and for most families is
the quantity the fit minimised with its sign flipped. `Multinomial` is
the exception: its reported risk omits the reference class, so
`negative_risk()` is the mboost-comparable *reported* value while the
fit actually descends the full categorical NLL (see
`likelihood_contract`). It is a log-likelihood
only for the families where the risk happens to be one; for the rest it
is a negated robust or ranking loss, and differences between two of
them are not likelihood ratios. Use it to compare two fits of the *same
family* on the *same data*, and `logLik` when a log-likelihood is
what the calculation needs.

Source: [`python/mboostrs/glmboost.py` lines 1569-1584](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/glmboost.py#L1569-L1584)

(glmboostmodel-likelihood_contract)=
## `GLMBoostModel.likelihood_contract`

*Method* of `GLMBoostModel`

```python
GLMBoostModel.likelihood_contract()
```

What this family's risk is: a likelihood, a loss, or neither.

Returns the `mboostrs._likelihood.FamilyLikelihood` row, whose
fields say whether the loss comes from a probability model, whether
`-risk` is the complete log-likelihood, what measure that likelihood is
taken with respect to, and which residual kinds are defined.

Source: [`python/mboostrs/glmboost.py` lines 1586-1596](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/glmboost.py#L1586-L1596)

(glmboostmodel-to_dict)=
## `GLMBoostModel.to_dict`

*Method* of `GLMBoostModel`

```python
GLMBoostModel.to_dict(*, include_arrays=False)
```

The `summary(as_dict=True)` record plus a `"provenance"` entry:
the library build and environment, and the fingerprint of the
training data. `include_arrays` is accepted for the uniform export
signature; the summary holds no per-row arrays.

Source: [`python/mboostrs/glmboost.py` lines 1598-1605](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/glmboost.py#L1598-L1605)

(glmboostmodel-summary)=
## `GLMBoostModel.summary`

*Method* of `GLMBoostModel`

```python
GLMBoostModel.summary(as_dict=False, cv=None)
```

Summarise the fitted model.

Returns a `mboostrs.summary.Summary`, which prints as
fixed-width text and renders as a table in a notebook. Use
`as_dict=True` for the machine-readable form.

**Parameters**

- `as_dict` (bool): If True, return a JSON-serializable dict instead of the Summary.
- `cv` (CvRiskResult, optional): The cross-validation result for this model. Supplying it adds a
  block comparing the iteration the model stopped at with the one
  cross-validation prefers. A model keeps no record of how it was
  tuned, so this is not inferred.

Source: [`python/mboostrs/glmboost.py` lines 1607-1665](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/glmboost.py#L1607-L1665)

(glmboostmodel-update)=
## `GLMBoostModel.update`

*Method* of `GLMBoostModel`

```python
GLMBoostModel.update(mstop=None, **kwargs)
```

Update the model by changing mstop or other parameters.

With only `mstop`, returns the model sliced to it, or grown to it
through `continue_to` when it is larger than the current one.
With keyword arguments, fits the model again with those arguments
replaced and every other argument as the original fit had it.

**Parameters**

- `mstop` (int, optional): New stopping iteration. Without keyword arguments it slices or
  grows the existing model; with them it is the refit's `mstop`,
  which otherwise stays at the current one.
- `kwargs` (dict): Arguments to change in the refit: the control fields `nu`,
  `risk`, `trace`, `step` and `step_iterations`, a whole
  `control=` record, or the fit arguments `family`, `weights`,
  `offset`, `center`, `subsample`, `subsample_seed` and
  `weight_policy`. Any other name raises `TypeError`.

**Returns**

GLMBoostModel
    Updated model.

Source: [`python/mboostrs/glmboost.py` lines 1667-1701](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/glmboost.py#L1667-L1701)

(glmboostmodel-continue_to)=
## `GLMBoostModel.continue_to`

*Method* of `GLMBoostModel`

```python
GLMBoostModel.continue_to(mstop: int) -> GLMBoostModel
```

Return the model grown to a larger `mstop`.

Equivalent to a cold `glmboost` with `mstop=N` to machine precision,
so `fit.continue_to(N)[:fit.mstop]` reproduces `fit`. See
`GAMBoostModel.continue_to` for the full contract. A linear fit
has no resume path, so this refits from iteration zero: it saves the
re-specification, not the computation.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/glmboost.py` lines 1703-1724](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/glmboost.py#L1703-L1724)
