# `GAMBoostModel`

Defined in `mboostrs.gamboost`; exported as `mboostrs.GAMBoostModel`.


*Class*

```python
GAMBoostModel(
    formula: str,
    family,
    control: tuple,
    response_name: str,
    learner_names: list[str],
    learner_types: list[str],
    fitted: numpy.ndarray,
    risk: numpy.ndarray,
    selected: list[int],
    coefficients: list[numpy.ndarray],
    offset: float,
    nu: float,
    data=None,
    weights: numpy.ndarray | None = None,
    learners_spec: list | None = None,
    rust_idx_to_formula_idx: list[int] | None = None,
    formula_idx_to_rust_idx: dict | None = None,
    buser_centering_means: dict | None = None,
    family_sigma: list[float] | None = None,
    family_sigma_path: list[list[float]] | None = None,
    learner_store=None,
    execution_report: dict | mboostrs.report.ExecutionReport | None = None,
    residual: numpy.ndarray | None = None
)
```

Fitted generalized additive boosting model.

Source: [`python/mboostrs/gamboost.py` lines 884-4897](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L884-L4897)

## Members

| Member | Kind | Summary |
|---|---|---|
| `path_frame` | Method | Return one structured row per boosting iteration. |
| `penalty_drift` | Method | Each penalized learner's degrees of freedom along the fitted path. |
| `concurvity` | Method | Concurvity of the fitted terms, as mgcv's ``concurvity`` measures it. |
| `family` | Property | The response family the model was fitted with. |
| `formula` | Property | The formula as it was written (``y ~ .`` stays ``y ~ .``). |
| `fitted_` | Property | The fitted linear predictor. |
| `fitted` | Method | Fitted values, optionally at a given mstop. |
| `aic` | Method | Compute AIC for each boosting iteration. |
| `hatvalues` | Method | Per-observation leverage: the hat-matrix diagonal at the current |
| `range_report` | Method | Each learner's training range against `newdata`. |
| `extrapolation_report` | Method | Terms and covariates with rows outside their training range. |
| `variable_names` | Method | Names of all variables used in the model (response + predictors). |
| `base_learner_names` | Method | Names of all base learners. |
| `selection_counts` | Method | One row per base learner with the boosting updates it received. |
| `model_matrix` | Method | Return the design matrix for the specified base-learner(s). |
| `nuisance` | Method | Return nuisance parameters (family-specific). |
| `predict` | Method | Predicted values, optionally decomposed by learner. |
| `predict_over_mstop` | Method | Predictions at several path positions, in one pass over the basis. |
| `predict_over_grid` | Method | Predictions at each value of one variable, other columns held per row. |
| `prediction_contributions` | Method | Per-learner prediction contributions. |
| `binning_report` | Method | What binning each learner actually got, and what it cost. |
| `check_binning` | Method | Refit one subsample with binning on and off. |
| `cvrisk` | Method | Cross-validate the fitted model's risk path. |
| `coef` | Method | Extract coefficients per base-learner. |
| `btree_splits` | Method | The cutpoints the fitted btree learners actually used. |
| `btree_partition_table` | Method | The fitted btree partition as a Polars frame, one row per region. |
| `plot_btree_partition` | Method | Plot a btree term's fitted partition. |
| `residuals` | Method | Residuals of the named kind, or a refusal saying why there are none. |
| `residual_kinds` | Method | The residual kinds defined for this model's family. |
| `extract` | Method | Extract model components, matching R's extract.mboost/glmboost. |
| `varimp` | Method | Variable importance based on in-bag risk reduction per base-learner. |
| `learner_index` | Method | Index of a base learner, disambiguating learners that share a name. |
| `partial_effects` | Method | Compute per-learner partial effects (contribution to the additive predictor). |
| `plot_partial_effects` | Method | Plot partial effects. |
| `plot` | Method | Plot partial effects. |
| `confint` | Method | Confidence intervals for partial effects. |
| `logLik` | Method | The fitted model's log-likelihood, or a refusal naming the reason. |
| `negative_risk` | Method | The negated empirical risk at the current mstop. |
| `likelihood_contract` | Method | What this family's risk is: a likelihood, a loss, or neither. |
| `to_dict` | Method | The ``summary(as_dict=True)`` record plus a ``"provenance"`` entry: |
| `summary` | Method | Summarise the fitted model. |
| `update` | Method | Update the model by changing mstop or other parameters. |
| `continue_to` | Method | Return the model grown to a larger ``mstop``. |

(gamboostmodel-path_frame)=
## `GAMBoostModel.path_frame`

*Method* of `GAMBoostModel`

```python
GAMBoostModel.path_frame(output='auto')
```

Return one structured row per boosting iteration.

Source: [`python/mboostrs/gamboost.py` lines 967-977](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L967-L977)

(gamboostmodel-penalty_drift)=
## `GAMBoostModel.penalty_drift`

*Method* of `GAMBoostModel`

```python
GAMBoostModel.penalty_drift(output='auto')
```

Each penalized learner's degrees of freedom along the fitted path.

A learner's `df=` (or `lambda=`) sets the flexibility of one
boosting step. Selected `m` times at step `nu`, its accumulated
contribution is the operator `I - (I - nu S)^m` applied on its basis,
whose df `sum_k 1 - (1 - nu s_k)^m` (over the eigenvalues `s_k`
of the smoother `S`) grows with `m` towards the rank of the basis,
the df of the unpenalized fit (Schulte and Rügamer, 2025). Columns:
`learner`, `selections`, `single_step_df` (`tr(S)`),
`boosted_df`, `unpenalized_df` and `drift_share`, the part of
the gap from `single_step_df` to `unpenalized_df` that boosting
has used. Each learner is measured on its own basis; overlap between
learners is ignored. Trees and monotone splines are omitted.

Source: [`python/mboostrs/gamboost.py` lines 1021-1039](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L1021-L1039)

(gamboostmodel-concurvity)=
## `GAMBoostModel.concurvity`

*Method* of `GAMBoostModel`

```python
GAMBoostModel.concurvity(pairwise=False, output='auto')
```

Concurvity of the fitted terms, as mgcv's `concurvity` measures it.

Concurvity is the additive-model form of collinearity: how far one
term's function space reaches into another's, which is when partial
effects stop being separately identified -- the same fit can be
redistributed between the terms, so their shapes, intervals and the
selection that chose them are unstable (Wood, *Generalized Additive
Models*, 2nd ed., section 5.6.2). Each row of the default table
measures one term against the rest of the model (mgcv's
`concurvity(full = TRUE)`) and gives three values from 0
(orthogonal) to 1 (the term lies in the others' span), where
near-collinear terms can overshoot 1 slightly from the numerics:
`worst`, the largest fraction of a term function's squared norm
the other terms can reproduce over every coefficient vector;
`observed`, the same fraction at the fitted coefficients (NaN for
a term that was never selected); `estimate`, an average over
coefficient directions. `worst` is a pessimistic bound -- mgcv
reports high values between unrelated rich bases -- so compare
magnitudes against the other rows rather than against 0.

With `pairwise=True` the rows give mgcv's
`concurvity(full = FALSE)`: one row per ordered pair, where
`term` is measured against `against` alone, no intercept
involved. Terms are reduced by the constant direction before
measuring, as mgcv measures its constrained smooths, so a constant
term reports NaN. Trees and `%O%` Kronecker learners are omitted.

**Parameters**

- `pairwise` (bool): Term-against-the-rest rows (the default) or one row per
  ordered pair of terms.
- `output` ({"auto", "records", "polars", "pandas", "frame"}): The return format; see `mboostrs.gamboost` for the
  convention.

Source: [`python/mboostrs/gamboost.py` lines 1041-1081](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L1041-L1081)

(gamboostmodel-family)=
## `GAMBoostModel.family`

*Property* of `GAMBoostModel`

```python
GAMBoostModel.family
```

The response family the model was fitted with.

Source: [`python/mboostrs/gamboost.py` lines 1083-1086](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L1083-L1086)

(gamboostmodel-formula)=
## `GAMBoostModel.formula`

*Property* of `GAMBoostModel`

```python
GAMBoostModel.formula  # -> str
```

The formula as it was written (`y ~ .` stays `y ~ .`).
`variable_names` lists the columns it expanded to.

Source: [`python/mboostrs/gamboost.py` lines 1088-1093](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L1088-L1093)

(gamboostmodel-fitted_)=
## `GAMBoostModel.fitted_`

*Property* of `GAMBoostModel`

```python
GAMBoostModel.fitted_  # -> np.ndarray
```

The fitted linear predictor.

Read-only: this is the fit's own array, not a copy. See
`mboostrs._state`.

Source: [`python/mboostrs/gamboost.py` lines 1095-1111](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L1095-L1111)

(gamboostmodel-fitted)=
## `GAMBoostModel.fitted`

*Method* of `GAMBoostModel`

```python
GAMBoostModel.fitted(mstop=None) -> ndarray
```

Fitted values, optionally at a given mstop.

mstop: int or None. If None, returns fitted values at the full mstop.
       If given, recomputes fitted values at iteration `mstop`.

Source: [`python/mboostrs/gamboost.py` lines 1113-1121](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L1113-L1121)

(gamboostmodel-aic)=
## `GAMBoostModel.aic`

*Method* of `GAMBoostModel`

```python
GAMBoostModel.aic(method='corrected', k=2.0)
```

Compute AIC for each boosting iteration.

Returns an AicResult; refuses if a required hat matrix is unavailable.
`'corrected'` (default) is the upstream per-observation Gaussian
stopping criterion, not a total GAIC; `'gmdl'` is Gaussian-only.
`'classical'` uses total likelihood risk and refuses Gaussian
squared-error loss. Positive `k` applies only to the classical
method. Results identify the criterion, normalization and conditional
hat-trace df, which excludes model-search df. For a `btree` term the
trace is likewise conditional on each step's fitted partition, so it
excludes split-search df as well.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `NotImplementedError`
- `ValueError`

Source: [`python/mboostrs/gamboost.py` lines 1125-1223](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L1125-L1223)

(gamboostmodel-hatvalues)=
## `GAMBoostModel.hatvalues`

*Method* of `GAMBoostModel`

```python
GAMBoostModel.hatvalues()
```

Per-observation leverage: the hat-matrix diagonal at the current
mstop, as a 1-d array of length n.

A deserialized model rebuilds its learner store from the stored
specifications, so this works after a pickle round trip. It refuses
when there is no hat matrix to take a diagonal of -- a `bmono`
learner among the selected ones -- rather than returning `None` and
leaving the caller to guess why. A `btree` learner among the
selected ones does not refuse: each tree step contributes the
averaging matrix of the partition it was fitted with, conditional
on that partition, and the diagonal does not count the search that
chose the splits -- the convention the conditional hat trace
already applies to a spline learner's data-chosen knots and to
learner selection itself.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `NotImplementedError`
- `ValueError`

Source: [`python/mboostrs/gamboost.py` lines 1225-1272](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L1225-L1272)

(gamboostmodel-range_report)=
## `GAMBoostModel.range_report`

*Method* of `GAMBoostModel`

```python
GAMBoostModel.range_report(newdata=None, output='auto')
```

Each learner's training range against `newdata`.

One row per spline learner and per `by=` variable of a spline learner,
and per linear (`bols`) term on a numeric column: the interval it was
fitted over, the range it is being asked about, how many rows fall each
side, and the worst distance in training ranges -- the number that
separates a boundary case from a different question. `term` says which
kind of row it is. A spline basis is extrapolated by its edge slope,
which the fit does not constrain, and `predict(extrapolation=)` acts on
those rows. A linear term continues the line the fit estimated, so it is
reported here but not warned about or clipped; the distance still says
how far the prediction relies on that line holding. With no `newdata`
the report reads off the ranges of the training data.

`gamboostLSS` prediction and R `predict()` apply the same
`extrapolation=` policy to spline learners.

Source: [`python/mboostrs/gamboost.py` lines 1519-1541](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L1519-L1541)

(gamboostmodel-extrapolation_report)=
## `GAMBoostModel.extrapolation_report`

*Method* of `GAMBoostModel`

```python
GAMBoostModel.extrapolation_report(newdata, output='auto')
```

Terms and covariates with rows outside their training range.

One row per term and covariate that has at least one row of `newdata`
outside its training range. The columns are `parameter` (always
`"f"` for this model), `term` (the learner, such as `bbs(x)` or
`bbs(x, by=z)`), `covariate` (the column whose values left the
range), `kind` (`"spline basis"` or `"linear"`), `n_below` and
`n_above` (rows below and above the training range), `n_outside`,
`train_low` and `train_high` (the training range), `new_low` and
`new_high` (the range of `newdata` on that covariate), and
`worst_distance` (the farthest outside row, in training ranges; 1
means a whole training range beyond the edge).

A covariate that stays in range is not listed, so an empty frame means
`newdata` lies inside every training range. `predict(extrapolation=)`
acts on the spline rows: `"clip"` clamps the covariate and `by=`
values listed here.

Source: [`python/mboostrs/gamboost.py` lines 1543-1568](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L1543-L1568)

(gamboostmodel-variable_names)=
## `GAMBoostModel.variable_names`

*Method* of `GAMBoostModel`

```python
GAMBoostModel.variable_names()
```

Names of all variables used in the model (response + predictors).

Source: [`python/mboostrs/gamboost.py` lines 1765-1779](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L1765-L1779)

(gamboostmodel-base_learner_names)=
## `GAMBoostModel.base_learner_names`

*Method* of `GAMBoostModel`

```python
GAMBoostModel.base_learner_names(disambiguate=True)
```

Names of all base learners.

A learner is stored under the column it reads, and two learners can read
one column: `bols(x) + bbs(x, df=1, center=True)` is the standard way
to split a term into a linear part and a deviation from it, and both
halves are named `"x"`. Where a name repeats, the learner type is
appended — `"x (bols)"`, `"x (bbs)"` — so the list identifies the
learners rather than the columns. `learner_index` accepts that
form, which is how such a name reaches `which=`.

**Parameters**

- `disambiguate` (bool): Append the learner type to names that more than one learner shares.
  `coef`, `partial_effects` and `varimp` are keyed
  by these names. `False` returns the stored column names, which
  `selected` reports.

**Returns**

list of str

Source: [`python/mboostrs/gamboost.py` lines 1781-1808](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L1781-L1808)

(gamboostmodel-selection_counts)=
## `GAMBoostModel.selection_counts`

*Method* of `GAMBoostModel`

```python
GAMBoostModel.selection_counts(output='auto')
```

One row per base learner with the boosting updates it received.

`selected` is keyed by the column a learner reads, and two
learners can read one column -- ``bols(x) + bbs(x, df=1,
center=True)`` is the standard linear-plus-deviation split, and
counting `selected` merges the halves. This table counts per
learner, names each row with the disambiguated form of
`base_learner_names`, and includes the learners that were
never selected with zero updates, so they remain distinguishable from
learners that are absent from the model.

**Parameters**

- `output` ({"auto", "pandas", "polars", "records"}): Frame backend for the table, or `"records"` for the raw rows.

**Returns**

DataFrame or list of dict
    One row per learner, most updates first: `learner`
    (disambiguated name), `column` (the column it reads),
    `updates` (selections received), `share` (updates over all
    boosting steps).

Source: [`python/mboostrs/gamboost.py` lines 1878-1919](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L1878-L1919)

(gamboostmodel-model_matrix)=
## `GAMBoostModel.model_matrix`

*Method* of `GAMBoostModel`

```python
GAMBoostModel.model_matrix(which=None)
```

Return the design matrix for the specified base-learner(s).

**Parameters**

- `which` (str, int, or list thereof, optional): Base-learner name(s) or index. None = all selected learners.

Source: [`python/mboostrs/gamboost.py` lines 1923-1931](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L1923-L1931)

(gamboostmodel-nuisance)=
## `GAMBoostModel.nuisance`

*Method* of `GAMBoostModel`

```python
GAMBoostModel.nuisance()
```

Return nuisance parameters (family-specific).

The family's nuisance parameter at the model's step count, as R's
`nuisance()` returns it: the size of `NBinomial` (`Var = mu + mu**2 / sigma`), the shape
of `GammaReg` and the parametric survival families, the thresholds
of `PropOdds`. None for a family that has none.

Source: [`python/mboostrs/gamboost.py` lines 1935-1947](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L1935-L1947)

(gamboostmodel-predict)=
## `GAMBoostModel.predict`

*Method* of `GAMBoostModel`

```python
GAMBoostModel.predict(
    newdata=None,
    mstop=None,
    type=<object object>,
    which=None,
    aggregate='sum',
    buser_newdata=None,
    extrapolation='warn',
    unseen_levels=None,
    offset=None
)
```

Predicted values, optionally decomposed by learner.
Usage:

```python
import mboostrs as mb
fit = mb.gamboost("y ~ bbs(x, df=4)", data)
fit.predict(newdata=data, type="response")
fit.predict(data, mstop=100, which="x (bbs)")  # one learner's contribution
```


**Parameters**

- `newdata` (dict or DataFrame, optional): New observations. If None, predicts on the training data.
- `offset` (number or array-like, optional): The offset these rows start from. A model fitted with a
  per-observation offset needs the offset of every *new* row, so
  predicting `newdata` requires this argument; the training rows
  reuse the fitted vector.
- `mstop` (int, optional): Truncate the boosting path at this iteration.
- `type` ({"link", "response", "class", "log_odds"}): Link-scale or response-scale predictions. `"log_odds"` is the
  link scale times 2 for `Binomial`, whose link is half the
  log-odds; with `which=` it gives each learner's contribution to
  the log-odds, the scale other libraries report.
  `"class"` returns 0/1 class codes for Binomial, AdaExp,
  HingeLoss and AUC. A score on the decision boundary returns 0.
- `which` (int, str, or list thereof, optional): If given, return only the contribution of the specified learner(s).
  Strings are learner names.
- `aggregate` ({"sum", "cumsum", "none"}): "sum" returns the aggregated contribution.
  "cumsum" returns cumulative contributions at each step.
  "none" returns per-step contributions.
- `buser_newdata` (dict, optional): Passed through for Buser learners (see main predict).
- `extrapolation` ({"warn", "error", "allow", "clip"}, default "warn"): How to treat rows that leave a spline learner's training range.
  The check covers the covariate and the `by=` variable of each
  spline term, since the term's effect is the basis times the
  `by=` value. The warning names the terms, the covariate, the rows
  each side, and the worst distance in training ranges. Excursions
  below 5% of the span do not warn, but `error` rejects and
  `clip` clamps every value strictly outside the interval.
  `extrapolation_report(newdata)` lists the same excursions as a
  table. The check covers univariate spline learners, both coordinates
  of a `bspatial` term, and the B-spline marginals of a `%X%`
  term, in full-model predictions, in `which=` contributions and in
  Multinomial predictions. `btree`, `bkernel`, `buser` and
  `brad` terms have no training-range boundary to check, so this
  policy does not apply to them (`validate()` lists them).
- `unseen_levels` ({"error", "average", "zero_effect"}, optional): `None` (default) uses the model's policy: `"error"`, except
  for a model returned by `mb.fit`, which predicts an unseen
  level as the average group.
  What a term over a categorical column does with a level in
  `newdata` that the fit never saw, such as a group held out of a
  grouped cross-validation fold. `"error"` refuses the prediction.
  `"average"` predicts the row as a member of an average group:
  each term that reads the column contributes the mean of its
  contributions over the fitted levels. For `brandom` that is the
  mean level coefficient -- near zero for a random intercept, and for
  a random slope (`brandom(g, by=x)`) the part of the average slope
  the group term absorbed during boosting, which a new group's
  prediction needs. A `bols` over a factor, a factor `by=`, a
  `%X%` factor margin and a `btree` split on the column average
  the same way (a tree over the levels on each side of each split).
  `"zero_effect"` gives the level a zero coefficient, the value
  the ridge penalty shrinks every level towards (a `btree` sends it
  to the branch that held more training weight); it is the
  prediction `cvrisk` makes for a group its fold holds out, where
  that group's rows enter the fit with zero weight.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamboost.py` lines 1951-2173](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L1951-L2173)

(gamboostmodel-predict_over_mstop)=
## `GAMBoostModel.predict_over_mstop`

*Method* of `GAMBoostModel`

```python
GAMBoostModel.predict_over_mstop(
    mstops,
    newdata=None,
    type='link',
    extrapolation='warn',
    output='auto',
    unseen_levels=None
)
```

Predictions at several path positions, in one pass over the basis.

`predict(newdata=, mstop=k)` builds every learner's new-data basis, so
calling it once per grid point rebuilds them each time. Measured on
20,000 rows and eight `bbs` learners, six positions cost 52.1 ms
against 8.7 for one -- six sets of B-spline evaluations to change
nothing but how many path steps were summed.

This walks the path once, carrying the coefficient sums forward from
one position to the next, and evaluates each basis once for the grid.

**Parameters**

- `mstops` (sequence of int): Path positions, ascending. The grid is walked once, so each
  position continues the previous one's sums.
- `newdata` (frame or dict, optional): Where to predict. Defaults to the training data.
- `type` ({"link", "response"}): As in `predict`.
- `unseen_levels` ({"error", "average", "zero_effect"}, optional): `None` (default) uses the model's policy: `"error"`, except
  for a model returned by `mb.fit`, which predicts an unseen
  level as the average group.
  As in `predict`.

**Returns**

A `(len(mstops), n)` array under `output="numpy"`, otherwise a frame
with one column per position named `mstop_<k>`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamboost.py` lines 2362-2454](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L2362-L2454)

(gamboostmodel-predict_over_grid)=
## `GAMBoostModel.predict_over_grid`

*Method* of `GAMBoostModel`

```python
GAMBoostModel.predict_over_grid(
    variable,
    grid,
    newdata=None,
    mstop=None,
    type='link',
    extrapolation='warn',
    unseen_levels=None,
    offset=None,
    output='auto'
)
```

Predictions at each value of one variable, other columns held per row.

For each grid value, `variable` is set to that value in every row of
`newdata`, and the model predicts those rows. Every other column keeps
each row's own value. The result equals one `predict` call per grid
value on the replaced frame. The difference is the work: a term that does
not read `variable` is evaluated once, and a term that reads it is
evaluated once per grid value, not once per boosting step.

**Parameters**

- `variable` (str): Column whose value is replaced. It may be a covariate of a spline or
  linear term, or the `by=` variable of one.
- `grid` (sequence): Values to set. Numeric values are used as given; a factor column
  takes its level labels.
- `newdata` (frame or dict, optional): Rows to predict. Defaults to the training data.
- `mstop` (int, optional): Truncate the boosting path at this iteration, as in `predict`.
- `type` ({"link", "response"}): As in `predict`.
- `extrapolation` ({"warn", "error", "allow", "clip"}, default "warn"): Applied to the replaced values exactly as `predict` applies it:
  a grid value outside a spline term's training range warns, errors,
  is kept, or is clamped.
- `unseen_levels` ({"error", "average", "zero_effect"}, optional): `None` (default) uses the model's policy: `"error"`, except
  for a model returned by `mb.fit`, which predicts an unseen
  level as the average group.
  As in `predict`.
- `offset` (number or array-like, optional): As in `predict`.
- `output` (str, default "auto"): `"numpy"` returns the array described below. Other values return a
  frame with one column per grid value, named `variable=value`.

**Returns**

ndarray
    Under `output="numpy"`, a `(len(grid), n)` array: row `k` holds
    the prediction of every row with `variable` set to `grid[k]`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamboost.py` lines 2456-2564](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L2456-L2564)

(gamboostmodel-prediction_contributions)=
## `GAMBoostModel.prediction_contributions`

*Method* of `GAMBoostModel`

```python
GAMBoostModel.prediction_contributions(
    newdata=None,
    mstop=None,
    which=None,
    aggregate='sum',
    buser_newdata=None,
    unseen_levels=None,
    extrapolation='warn'
)
```

Per-learner prediction contributions.

**Parameters**

- `newdata` (dict or DataFrame, optional)
- `mstop` (int, optional)
- `which` (int, str, or list thereof, optional): Restrict to specific learner(s).
- `aggregate` ({"sum", "cumsum", "none"})
- `buser_newdata` (dict, optional)
- `unseen_levels` ({"error", "average", "zero_effect"}, optional): `None` (default) uses the model's policy: `"error"`, except
  for a model returned by `mb.fit`, which predicts an unseen
  level as the average group.
  As in `predict`.
- `extrapolation` ({"warn", "error", "allow", "clip"}, default "warn"): As in `predict`: the contributions of spline, `bspatial`
  and `%X%` margin terms are checked against their training ranges
  before they are evaluated.

**Returns**

ndarray
    For `aggregate="sum"`: 1-D array of length n.
    For `aggregate="cumsum"`: 2-D array of shape (mstop, n).
    For `aggregate="none"`: 2-D array of shape (mstop, n).

Source: [`python/mboostrs/gamboost.py` lines 2663-2705](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L2663-L2705)

(gamboostmodel-binning_report)=
## `GAMBoostModel.binning_report`

*Method* of `GAMBoostModel`

```python
GAMBoostModel.binning_report(output='auto')
```

What binning each learner actually got, and what it cost.

`bins=k` is a request. What a learner ends up with is the smaller of
`k` and the column's distinct values, so `bins=50` on a column taking
seven values bins nothing and says nothing -- this reports the
effective count, the ties collapsed, and the learners `bins=` did not
reach at all.

One row per learner. The retained storage the fit used is on
`execution_report().retained_learner_bytes`; the *saving* against
exact storage is not a subtraction this can do -- it would need the
exact fit -- so `check_binning()` reports both, on a subsample, along
with how far the two fits diverged.

Source: [`python/mboostrs/gamboost.py` lines 2904-2951](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L2904-L2951)

(gamboostmodel-check_binning)=
## `GAMBoostModel.check_binning`

*Method* of `GAMBoostModel`

```python
GAMBoostModel.check_binning(n=2000, seed=42, mstop=None)
```

Refit one subsample with binning on and off.

Returns the maximum fitted-value discrepancy relative to the exact
refit's residual standard deviation, plus the first selection-path
divergence. This answers whether the chosen bin count is adequate for
this model rather than relying on a generic recommendation.

The comparison applies to this model and the selected subsample.
Agreement supports the bin count for this fit; other data or model
specifications can produce a different discrepancy.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamboost.py` lines 2953-3027](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L2953-L3027)

(gamboostmodel-cvrisk)=
## `GAMBoostModel.cvrisk`

*Method* of `GAMBoostModel`

```python
GAMBoostModel.cvrisk(
    folds=None,
    type='bootstrap',
    B=25,
    prob=0.5,
    seed=42,
    parallel=True,
    at_mstop=None,
    preprocessing='fixed',
    _inner=False,
    _extend=None,
    _path_cache=None
)
```

Cross-validate the fitted model's risk path.

`at_mstop=k` judges the path truncated at `k` steps: the returned
curve, fold rows and grid end there, and the boundary warning is
skipped because the step count is the caller's choice, not a
search's output -- fixing the step answers the question the
warning asks. Pass it when comparing designs or specifications at
a prespecified step count; leave it unset when tuning.

`preprocessing="fixed"` reuses the full-data transformations and
spline knots, matching mboost. `"refit"` learns transformations,
factor encodings and spline boundaries on each fold's training rows.
Held-out rows use that fitted recipe, with zero effects for unseen
group levels. Refit folds run sequentially through the native fit,
prediction and family-score APIs. The result records the mode.

`_extend` (internal, used by `tune_mstop`) is a mapping of
`max_mstop`, `patience`, `boundary_share` and
`max_extensions`: the grid starts at the model's `mstop` and grows
by resuming each fold's path while the held-out risk is still falling.
The call then returns `(result, grid_ends, at_boundary)`.

**Parameters**

- `folds` (int, FoldScheme or array_like of shape (n, B), optional): An integer is a k-fold split with that many folds (at least 2),
  drawn with `seed`. A `FoldScheme` or an `(n, B)` matrix
  of in-bag weights is used as given; a scheme also carries its
  judge mask. `None` generates `B` folds of resampling type
  `type`.
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

- `ValueError`: If `folds` is an integer below 2, the fold matrix does not match
the number of observations, or the model keeps no data to refit
from.
- `RuntimeError`: If the Rust engine is not available.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `NotImplementedError`

Source: [`python/mboostrs/gamboost.py` lines 3029-3218](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L3029-L3218)

(gamboostmodel-coef)=
## `GAMBoostModel.coef`

*Method* of `GAMBoostModel`

```python
GAMBoostModel.coef(which=None, off2int=False, aggregate='sum')
```

Extract coefficients per base-learner.

which: base-learner name(s) or index/indices. None = all used.
off2int: ignored with warning for gamboost (matching R's behavior;
    R warns "Arguments off2int unknown" for gamboost).
aggregate: "sum" (default) sums per-step coefs; "none" returns raw per-step lists.
Returns dict mapping learner name -> coefficient value or array.
The offset is always included as result["(offset)"].

Source: [`python/mboostrs/gamboost.py` lines 3219-3308](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L3219-L3308)

(gamboostmodel-btree_splits)=
## `GAMBoostModel.btree_splits`

*Method* of `GAMBoostModel`

```python
GAMBoostModel.btree_splits()
```

The cutpoints the fitted btree learners actually used.

One entry per btree learner that was selected at least once. Numeric
splits name the variable and the cutpoint with the number of boosting
steps that used it; categorical splits name the variable and the
level sets. The variables follow the tree's own input order, so index
`i` in a step's decoded nodes is `variables[i]`. A tree's splits
are empirical cutpoints learned by the fit. Use them to propose
hinge, threshold or other simpler candidate terms when examining a tree.

Source: [`python/mboostrs/gamboost.py` lines 3340-3384](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L3340-L3384)

(gamboostmodel-btree_partition_table)=
## `GAMBoostModel.btree_partition_table`

*Method* of `GAMBoostModel`

```python
GAMBoostModel.btree_partition_table(term=None)
```

The fitted btree partition as a Polars frame, one row per region.

The companion to `btree_splits`: the cutpoints there name the
boundaries, and here they become the `<column>_lower` and
`<column>_upper` bounds of each region with the term's fitted
`value` at the region centre.

**Parameters**

- `term` (str or int, optional): Which btree term to read; the only one when there is exactly one.

**Returns**

polars.DataFrame

Source: [`python/mboostrs/gamboost.py` lines 3386-3403](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L3386-L3403)

(gamboostmodel-plot_btree_partition)=
## `GAMBoostModel.plot_btree_partition`

*Method* of `GAMBoostModel`

```python
GAMBoostModel.plot_btree_partition(term=None, grid=60, data=None, at=None)
```

Plot a btree term's fitted partition.

A heatmap over the two columns the term splits on, or a step plot for a
one-column term; see `mboostrs.plot.plot_btree_partition`.

**Returns**

alt.Chart

Source: [`python/mboostrs/gamboost.py` lines 3405-3418](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L3405-L3418)

(gamboostmodel-residuals)=
## `GAMBoostModel.residuals`

*Method* of `GAMBoostModel`

```python
GAMBoostModel.residuals(type='response', newdata=None, mstop=None)
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

Source: [`python/mboostrs/gamboost.py` lines 3420-3444](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L3420-L3444)

(gamboostmodel-residual_kinds)=
## `GAMBoostModel.residual_kinds`

*Method* of `GAMBoostModel`

```python
GAMBoostModel.residual_kinds()
```

The residual kinds defined for this model's family.

Source: [`python/mboostrs/gamboost.py` lines 3446-3450](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L3446-L3450)

(gamboostmodel-extract)=
## `GAMBoostModel.extract`

*Method* of `GAMBoostModel`

```python
GAMBoostModel.extract(what, which=None)
```

Extract model components, matching R's extract.mboost/glmboost.

what: one of "design", "penalty", "lambda", "df", "coefficients",
      "residuals", "offset", "weights".
      Aliases accepted: "coef", "coefs", "coefficient" -> "coefficients";
      "x", "model_matrix" -> "design"; "w" -> "weights";
      "penalties" -> "penalty"; "lambdas" -> "lambda"; "dfs" -> "df".
which: base-learner name(s) or index. None = all used learners.
Returns dict mapping learner name -> extracted value.

"design" is the design on every training row, which is mboost's
`extract(..., expand = TRUE)`; mboost's default reports a factor
learner's design on its distinct rows. A `bols(x)` reports the
block `[1, x]` and "penalty" reports K without lambda, as mboost
does.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`
- `TypeError`

Source: [`python/mboostrs/gamboost.py` lines 3452-3585](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L3452-L3585)

(gamboostmodel-varimp)=
## `GAMBoostModel.varimp`

*Method* of `GAMBoostModel`

```python
GAMBoostModel.varimp(percent=True)
```

Variable importance based on in-bag risk reduction per base-learner.

Returns a dict with keys:
  reduction: dict of learner_name -> risk reduction
  selfreq: dict of learner_name -> selection frequency
  variable_names: dict of learner_name -> variable name
  split_columns: dict of learner_name -> columns actually used in splits
With `percent=True` (the default) each reduction is divided by the
sum of their absolute values, so the reductions are shares summing
to 1. `percent=False` returns the in-bag risk reduction per
observation, which is what mboost's `varimp()` returns.

Each base-learner has its own entry, as in mboost. Learners that
share a column (`bbs(x) + bmono(x)`) are keyed by
`base_learner_names` (`"x (bbs)"`, `"x (bmono)"`); sum the
entries whose `variable_names` agree for a variable's total.

`variable_names` lists every candidate input column a learner may
read -- for a joint `btree` that is the whole column list, not the
single column the label is named for. `split_columns` narrows that
to the columns the fitted trees actually split on (from
`btree_splits`); for non-tree learners the two agree. Counting
inputs a model retains should use `split_columns`: a joint tree
that never split on a column does not use it.

Source: [`python/mboostrs/gamboost.py` lines 3601-3667](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L3601-L3667)

(gamboostmodel-learner_index)=
## `GAMBoostModel.learner_index`

*Method* of `GAMBoostModel`

```python
GAMBoostModel.learner_index(name, kind=None)
```

Index of a base learner, disambiguating learners that share a name.

`bols(x) + bbs(x, df=1, center=True)` puts two learners on column
`x`, so the name alone does not identify one. `kind` selects between
them by learner type — `"bols"`, `"bbs"`, `"btree"` and so on.

**Parameters**

- `name` (str): Column the learner is built on. The disambiguated form
  `base_learner_names` returns for a shared name — `"x (bols)"`
  — is also accepted, and carries the kind with it.
- `kind` (str, optional): Learner type, matched case-insensitively against the class name.
  Required when more than one learner shares the name.

**Returns**

int
    Index usable as `partial_effects(which=...)`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamboost.py` lines 3669-3742](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L3669-L3742)

(gamboostmodel-partial_effects)=
## `GAMBoostModel.partial_effects`

*Method* of `GAMBoostModel`

```python
GAMBoostModel.partial_effects(
    which=None,
    n_points=None,
    newdata=None,
    unseen_levels=None,
    scale='link',
    extrapolation='allow',
    _prepared=False,
    _buser=None,
    mstop=None
)
```

Compute per-learner partial effects (contribution to the additive predictor).

**Parameters**

- `which` (str, int, or list of those, optional): Base-learner(s) to compute effects for. If None, all selected
  base-learners.

  A name two learners share (the decomposition idiom
  `bols(x) + bbs(x, df=1, center=True)`) selects both, keyed
  `"x (bols)"` and `"x (bbs)"`; that form, or the learner's index,
  selects one.
- `n_points` (int, optional): For smooth (bbs/bmono/bspatial) effects, evaluate on an equispaced
  grid of this many points. If None, evaluate on training/newdata.
- `newdata` (dict or DataFrame, optional): Data to evaluate effects on. If None, uses training data.
- `unseen_levels` ({"error", "average", "zero_effect"}, optional): `None` (default) uses the model's policy: `"error"`, except
  for a model returned by `mb.fit`, which predicts an unseen
  level as the average group.
  As in `predict`.
- `extrapolation` ({"allow", "warn", "error", "clip"}, default "allow"): As in `predict`, for `newdata` outside a spline learner's
  training range. The default stays silent, because an effect curve
  is often drawn on purpose beyond the data; `predict(which=...)`
  passes its own policy (default `"warn"`). The `%X%` margins
  are not checked here, because this method takes the `%X%` designs
  it is given rather than building them from `newdata`.
- `scale` ({"link", "log_odds"}, default "link"): `"link"` reports effects on the family's link scale, as mboost
  does: for `Binomial` that is half the log-odds (the response is
  `1 / (1 + exp(-2 * eta))`). `"log_odds"` multiplies a
  `Binomial` effect by 2, the scale EBM, XGBoost, mgcv and other
  libraries report and the one whose exponential is an odds ratio.
- `mstop` (int, optional): Truncate the boosting path at this iteration: only steps before
  it contribute to each effect, so the result equals the effect of
  a model refit with `mstop` steps. If None, the full fitted
  path contributes.

**Returns**

dict
    Mapping base-learner name -> (x_values, effect_values) pairs.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamboost.py` lines 3744-4070](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L3744-L4070)

(gamboostmodel-plot_partial_effects)=
## `GAMBoostModel.plot_partial_effects`

*Method* of `GAMBoostModel`

```python
GAMBoostModel.plot_partial_effects(which=None, n_points=100)
```

Plot partial effects.

**Parameters**

- `which` (list of str, optional): Which base-learners to plot.
- `n_points` (int): Number of grid points for smooth effects.

**Returns**

alt.Chart | alt.ConcatChart

Source: [`python/mboostrs/gamboost.py` lines 4072-4088](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L4072-L4088)

(gamboostmodel-plot)=
## `GAMBoostModel.plot`

*Method* of `GAMBoostModel`

```python
GAMBoostModel.plot(which=None, n_points=100)
```

Plot partial effects.

**Parameters**

- `which` (list of str, optional): Which base-learners to plot.
- `n_points` (int): Number of grid points for smooth effects.

**Returns**

alt.Chart | alt.ConcatChart

Source: [`python/mboostrs/gamboost.py` lines 4090-4104](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L4090-L4104)

(gamboostmodel-confint)=
## `GAMBoostModel.confint`

*Method* of `GAMBoostModel`

```python
GAMBoostModel.confint(
    parm=None,
    level=0.95,
    B=1000,
    B_mstop=25,
    seed=None,
    newdata=None,
    n_jobs=1,
    groups=None,
    centre=False,
    method='bootstrap'
)
```

Confidence intervals for partial effects.

method="bootstrap" (default): bootstrap intervals for partial effects.
These under-cover on a boosted fit (74% empirical coverage at nominal
90% on the book's additive benchmark): resampling reproduces the
shrinkage that early stopping induces instead of correcting it, so
every call on this path warns and points at method="refit".
method="refit": refit the selected terms jointly by REML
(`mboostrs.selection.gam_refit`) and return its Bayesian
pointwise bands, which average close to nominal. Terms the refit
cannot represent are held at their boosted contribution
(`unsupported="hold"`) and listed in the result's `held` entry;
a family the refit does not support is refused by name.

parm: base-learner name(s) or index. None = all used. For
method="refit", names are resolved to their formula terms.
level: confidence level (default 0.95).
B: number of bootstrap replications (bootstrap method only).
B_mstop: number of inner CV folds for mstop selection (0 = skip).
seed: random seed.
newdata: optional data for prediction grid.
n_jobs: bootstrap worker threads (1 = sequential, -1 = all CPUs). Results
    are independent of n_jobs (each replicate has its own seeded RNG).
    Most effective with B_mstop=0; when B_mstop>0 the inner CV is already
    parallel (rayon), so n_jobs>1 mainly contends and adds little.
groups: a column name or one label per row. When given, every
    bootstrap sample (and every inner fold of the `B_mstop` tuning)
    draws whole groups with replacement instead of single rows, and a
    row's weight in a sample is its group's draw count times its
    model weight. Use it when rows are repeated measurements of a
    unit -- the points of one curve, the visits of one patient --
    since resampling such rows separately treats them as independent
    and gives intervals that are too narrow.
centre: when True, each replicate's effect is centred over the rows
    of `newdata` (its mean there subtracted) before the quantiles are
    taken, so the interval is that of the centred effect
    `f(x) - mean f`. A partial effect's level is not identified: the
    offset and the other terms can absorb a constant, and a `bbs`
    term (`center=False`, the default) carries its own. Use
    `centre=True` to compare an effect with a curve known up to a
    constant, such as a simulation's true component centred over the
    same rows; comparing the default interval with such a curve
    mixes in an arbitrary shift. On a three-term additive simulation
    (400 rows, ten seeds, nominal 0.80) the bands covered the true
    components 0.38 of the time when compared with the components as
    written, whose means over the range are not zero, and 0.89 when
    both were centred over the grid. `boot_pred` stays uncentred.
Returns a dict with 'lower', 'upper', 'level', 'centred', 'boot_pred', and the
counts describing how many replicates the interval was actually built
from ('B', 'n_used', 'n_failed', 'n_untuned', and 'n_at_mstop_cap',
the replicates whose inner cross-validation chose the model's largest
`mstop` with the risk still falling; 10% or more of them warns), and
'resampled' ('rows' or 'groups').

A replicate that does not select a learner contributes effect zero
for that learner; these zero values are included in the interval.

With method="refit" the returned dict instead carries 'lower',
'upper', 'estimate' (each keyed by formula term), 'level', 'method'
('"refit"'), the refit's 'edf' per term, 'held' (terms the refit could
not represent, kept at their boosted contribution), and
'conditional_on_selection'.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`
- `TypeError`

Source: [`python/mboostrs/gamboost.py` lines 4171-4397](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L4171-L4397)

(gamboostmodel-loglik)=
## `GAMBoostModel.logLik`

*Method* of `GAMBoostModel`

```python
GAMBoostModel.logLik()
```

The fitted model's log-likelihood, or a refusal naming the reason.

Available only where the fitted loss is a log-likelihood. `Binomial`,
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

Source: [`python/mboostrs/gamboost.py` lines 4523-4550](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L4523-L4550)

(gamboostmodel-negative_risk)=
## `GAMBoostModel.negative_risk`

*Method* of `GAMBoostModel`

```python
GAMBoostModel.negative_risk()
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

Source: [`python/mboostrs/gamboost.py` lines 4552-4567](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L4552-L4567)

(gamboostmodel-likelihood_contract)=
## `GAMBoostModel.likelihood_contract`

*Method* of `GAMBoostModel`

```python
GAMBoostModel.likelihood_contract()
```

What this family's risk is: a likelihood, a loss, or neither.

Returns the `mboostrs._likelihood.FamilyLikelihood` row, whose
fields say whether the loss comes from a probability model, whether
`-risk` is the complete log-likelihood, what measure that likelihood is
taken with respect to, and which residual kinds are defined.

Source: [`python/mboostrs/gamboost.py` lines 4569-4579](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L4569-L4579)

(gamboostmodel-to_dict)=
## `GAMBoostModel.to_dict`

*Method* of `GAMBoostModel`

```python
GAMBoostModel.to_dict(*, include_arrays=False)
```

The `summary(as_dict=True)` record plus a `"provenance"` entry:
the library build and environment, and the fingerprint of the
training data. `include_arrays` is accepted for the uniform export
signature; the summary holds no per-row arrays.

Source: [`python/mboostrs/gamboost.py` lines 4581-4588](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L4581-L4588)

(gamboostmodel-summary)=
## `GAMBoostModel.summary`

*Method* of `GAMBoostModel`

```python
GAMBoostModel.summary(as_dict=False, cv=None)
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

Source: [`python/mboostrs/gamboost.py` lines 4590-4658](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L4590-L4658)

(gamboostmodel-update)=
## `GAMBoostModel.update`

*Method* of `GAMBoostModel`

```python
GAMBoostModel.update(mstop=None, **kwargs)
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
  `offset`, `bins`, `subsample`, `subsample_seed` and
  `weight_policy`. Any other name raises `TypeError`.

**Returns**

GAMBoostModel
    Updated model.

Source: [`python/mboostrs/gamboost.py` lines 4660-4694](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L4660-L4694)

(gamboostmodel-continue_to)=
## `GAMBoostModel.continue_to`

*Method* of `GAMBoostModel`

```python
GAMBoostModel.continue_to(mstop: int) -> GAMBoostModel
```

Return the model grown to a larger `mstop`.

`fit` fixes `mstop` up front, but the interactive loop is fit,
inspect the risk path, discover the optimum is at the boundary, extend.
`continue_to(N)` does that without re-specifying the model.

The returned model is the one a cold `fit` with `mstop=N` would
produce, to machine precision — boosting is deterministic given the
data, so the first `self.mstop` iterations are identical either way.
Slicing it back reproduces the current model: `fit.continue_to(N)[:m]`
equals `fit` for `m == fit.mstop`.

Growing to `N <= self.mstop` slices instead. Growing below 1 is an
error.

Note
----
Where the fit kept the gradient it left off on, this **resumes** it and
does not recompute the first `self.mstop` iterations. That covers
every family whose calibration is a pure function of `(y, f, w)`.
A fit that took the Gaussian Gram route keeps per-learner projections
rather than a length-n residual, and `PropOdds`, `Multinomial`, the
survival families and binned learners have no resume path, so those
refit from iteration zero. The equivalence guarantee above holds on
both routes; only the cost differs.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamboost.py` lines 4696-4759](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L4696-L4759)
