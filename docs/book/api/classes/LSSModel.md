# `LSSModel`

Defined in `mboostrs.gamlss`; exported as `mboostrs.LSSModel`.


*Class*

```python
LSSModel(
    family,
    nu,
    risk,
    params,
    data,
    response_name=None,
    weights=None,
    formula=None,
    control=None,
    offset=None,
    method='cyclic',
    model_kind='glmboostLSS',
    execution_report=None,
    selection_order=None,
    bins=None,
    env=None,
    nu_per_param=None,
    stabilization=None
)
```

Fitted gamboostLSS/glmboostLSS model (one predictor per parameter).

Source: [`python/mboostrs/gamlss.py` lines 3034-5626](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L3034-L5626)

## Members

| Member | Kind | Summary |
|---|---|---|
| `fit_warnings` | Property | Warnings the fit showed, as ``warnings.WarningMessage`` objects. |
| `execution_report` | Method | Return aggregate numerical-route metadata for this LSS fit. |
| `path_frame` | Method | Return globally ordered parameter-update rows for this LSS fit. |
| `risk_` | Property | The joint risk after each iteration. |
| `mstop` | Property |  |
| `stopping_report` | Method | Where the boosting path stopped, and whether that looks finished. |
| `stopping_budget` | Method | How far this model was boosted, as |
| `selected_terms` | Method | One row per boosting update, in the order the fit made them: |
| `predict_parameters` | Method | Return response-scale predictions by distribution parameter. |
| `selected` | Property | Selected base-learners at each iteration, as ``{parameter: path}``. |
| `fitted` | Method | In-sample fitted values, mirroring ``predict(newdata=None, ...)``. |
| `params` | Method | Predicted LSS parameters for all family parameters. |
| `predict_all` | Method | All-parameter prediction table, analogous to GAMLSS ``predictAll``. |
| `coef_all` | Method | All-parameter coefficient table, analogous to GAMLSS ``coefAll``. |
| `risk` | Property | The joint risk after each iteration (each full cycle for a cyclic |
| `parameter_risk` | Method | The joint risk after each parameter's own updates, as |
| `model_weights` | Method | Observation weights used for fitting. |
| `deviance_increment` | Method | Per-observation deviance contribution for supported LSS families. |
| `centiles` | Method | Distribution centile table for fitted LSS parameters. |
| `prediction_interval` | Method | Central prediction interval table for the fitted LSS distribution. |
| `predint` | Method | Alias for :meth:`prediction_interval`, matching R ``predint`` naming. |
| `PI` | Method | Alias for :meth:`prediction_interval`, matching R ``PI`` naming. |
| `pit` | Method | Probability integral transform values for fitted LSS distributions. |
| `quantile_residuals` | Method | Randomized quantile residuals for fitted LSS distributions. |
| `residual_frame` | Method | Diagnostic residual table for LSS models. |
| `worm_data` | Method | Worm-plot data. See :func:`mboostrs.worm_data`. |
| `calibration_frame` | Method | Calibration table. See :func:`mboostrs.calibration_frame`. |
| `q_stats` | Method | Binned quantile-residual moment summaries. |
| `density_overlay_data` | Method | Observed and fitted marginal-distribution curves as a table. |
| `centile_fan_data` | Method | Observed responses and fitted centiles ordered by ``xvar``. |
| `deviance_profile` | Method | Binned joint deviance contributions along a numeric covariate. |
| `worm_plot` | Method | Worm plot(s) of (randomized) quantile residuals (R's ``gamlss::wp``). |
| `logLik` | Method | Joint log-likelihood at the current LSS fit. |
| `negative_risk` | Method | The negated joint risk at the current fit. |
| `parity_notes` | Method | Where this fit differs from what its names refer to. |
| `likelihood_contract` | Method | What this family's risk is: a likelihood, a loss, or neither. |
| `gaic` | Method | Update-count penalized likelihood heuristic for LSS models. |
| `llm_explain_prompt` | Method | Render this fit as a prompt for a language model. |
| `llm_describe` | Method | Send this fit's description to a language model and return the reply. |
| `to_dict` | Method | The ``summary(as_dict=True)`` record plus a ``"provenance"`` entry: |
| `summary` | Method | Summarise the fitted LSS model. |
| `cvrisk` | Method | Cross-validated risk for this model's own specification. |
| `stabsel` | Method | Stability selection for this model's own specification. |
| `profile_mstop` | Method | Refit over a grid of stopping points and report `metric` at each. |
| `control` | Property | The `boost_control` this model was fitted with. |
| `offset` | Property | Per-parameter offsets, as ``{parameter: value}``. |
| `plot_risk` | Method | Plot the joint risk path. |
| `plot_selected` | Method | Plot selection frequency per base learner and parameter. |
| `plot_density` | Method | Overlay the observed and average fitted response distributions. |
| `plot_centile_fan` | Method | Plot nested fitted centile bands against ``xvar``. |
| `plot_deviance_profile` | Method | Plot binned joint deviance along a covariate or row order. |
| `continue_to` | Method | Return the model grown to a larger `mstop`, per parameter. |
| `varimp` | Method | Variable importance from in-bag risk reduction, per parameter. |
| `base_learner_names` | Method | Base-learner names, as a `{parameter: [name, ...]}` mapping. |
| `bnames` | Method | Alias for :meth:`base_learner_names`. |
| `variable_names` | Method | Distinct predictor names, in formula order. |
| `selected_which` | Method | Selected learner names per parameter, optionally filtered. |
| `fitted_` | Property | Fitted link-scale values per parameter, as a mapping. |
| `offset_` | Property | Per-parameter offsets, as a mapping. See also :attr:`offset`. |
| `save` | Method | Write a portable JSON artifact. See :func:`mboostrs.save_model`. |
| `with_mstop` | Method | Return a copy refit to ``k`` boosting iterations. |
| `update` | Method | Refit the stored LSS model specification. |
| `extrapolation_report` | Method | Terms and covariates with rows outside their training range, by parameter. |
| `mean` | Method | Expected response under the fitted distribution. |
| `predict` | Method | Predictions for one distribution parameter, or for all of them. |
| `partial_effects` | Method | Return formula-ordered learner contributions for one LSS parameter. |
| `coef` | Method | Per-parameter aggregated coefficients (sum of nu*coef per learner). |
| `residual_kinds` | Method | Residual kinds defined for the fitted LSS family. |
| `residuals` | Method | Residuals of the named kind for an LSS parameter, or a refusal. |

(lssmodel-fit_warnings)=
## `LSSModel.fit_warnings`

*Property* of `LSSModel`

```python
LSSModel.fit_warnings  # -> tuple
```

Warnings the fit showed, as `warnings.WarningMessage` objects.

Empty for a quiet fit. The warnings are also emitted normally during
the fit call itself, so reading this attribute afterwards is an
alternative to wrapping the call with `warnings.catch_warnings`.
A warning your filters suppress is not recorded: under Python's
default filters that includes `ResourceWarning` and a warning the
same line already showed once.

Not persisted: `save_model`/`load_model` restore the model without
the fit-time warnings, because the artifact format carries plain data
only and the warning types are interpreter-specific.

Source: [`python/mboostrs/gamlss.py` lines 3040-3055](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L3040-L3055)

(lssmodel-execution_report)=
## `LSSModel.execution_report`

*Method* of `LSSModel`

```python
LSSModel.execution_report() -> mboostrs.report.ExecutionReport | None
```

Return aggregate numerical-route metadata for this LSS fit.

Source: [`python/mboostrs/gamlss.py` lines 3153-3155](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L3153-L3155)

(lssmodel-path_frame)=
## `LSSModel.path_frame`

*Method* of `LSSModel`

```python
LSSModel.path_frame(output='auto')
```

Return globally ordered parameter-update rows for this LSS fit.

Source: [`python/mboostrs/gamlss.py` lines 3157-3169](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L3157-L3169)

(lssmodel-risk_)=
## `LSSModel.risk_`

*Property* of `LSSModel`

```python
LSSModel.risk_  # -> np.ndarray
```

The joint risk after each iteration.

Read-only: this is the fit's own array, not a copy. See
`mboostrs._state`.

Source: [`python/mboostrs/gamlss.py` lines 3171-3180](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L3171-L3180)

(lssmodel-mstop)=
## `LSSModel.mstop`

*Property* of `LSSModel`

```python
LSSModel.mstop
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 3182-3189](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L3182-L3189)

(lssmodel-stopping_report)=
## `LSSModel.stopping_report`

*Method* of `LSSModel`

```python
LSSModel.stopping_report() -> dict
```

Where the boosting path stopped, and whether that looks finished.

Same report as `ModelAPIMixin.stopping_report`: best in-sample
risk step, risk there versus at the end, whether the risk rose after
its best step, and a same-family GLM gap when one can be fit cheaply.
Reads only the stored joint risk path; nothing is refit.

Source: [`python/mboostrs/gamlss.py` lines 3191-3201](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L3191-L3201)

(lssmodel-stopping_budget)=
## `LSSModel.stopping_budget`

*Method* of `LSSModel`

```python
LSSModel.stopping_budget()
```

How far this model was boosted, as
`mboostrs._model_api.StoppingBudget`: per-parameter update
counts, their total (the budget a non-cyclic refit takes) and whether
the parameters were updated cyclically or non-cyclically.

Source: [`python/mboostrs/gamlss.py` lines 3203-3213](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L3203-L3213)

(lssmodel-selected_terms)=
## `LSSModel.selected_terms`

*Method* of `LSSModel`

```python
LSSModel.selected_terms(output='auto')
```

One row per boosting update, in the order the fit made them:
`order` (1-based, across all parameters), `parameter`,
`iteration` (1-based, within the parameter) and `learner`. A
cyclic fit updates every parameter once per iteration, in parameter
order; a non-cyclic fit records the parameter it chose at each update.

Source: [`python/mboostrs/gamlss.py` lines 3215-3236](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L3215-L3236)

(lssmodel-predict_parameters)=
## `LSSModel.predict_parameters`

*Method* of `LSSModel`

```python
LSSModel.predict_parameters(
    newdata=None,
    type=<object object>,
    extrapolation='warn',
    unseen_levels='error'
) -> dict
```

Return response-scale predictions by distribution parameter.

An LSS model returns `{parameter: array}`. A standard model has
one linear predictor and returns it as `{"f": array}`.

The default is the response/natural scale: each parameter comes
back as the quantity it is named for — an `NBinomialLSS` `sigma`
is the size, a `GaussianLSS` `sigma` a standard deviation, a
logit parameter a probability — and no link-scale warning fires for
an omitted `type`. Pass `type="link"` for the linear predictors;
`predict()` keeps the link scale as its own (warning) default.
`extrapolation` and `unseen_levels` are as in `predict`;
pass `extrapolation="clip"` to hold covariates at the training range
and `unseen_levels="average"` to score categories absent from training.

Source: [`python/mboostrs/gamlss.py` lines 3238-3259](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L3238-L3259)

(lssmodel-selected)=
## `LSSModel.selected`

*Property* of `LSSModel`

```python
LSSModel.selected
```

Selected base-learners at each iteration, as `{parameter: path}`.

The call forms `fit.selected(parameter=...)` and
`fit.selected(merge=True)` still work and warn; they are
`fit.selected_which(parameter=...)` and
`fit.selected_which(merge=True)`.

Source: [`python/mboostrs/gamlss.py` lines 3333-3346](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L3333-L3346)

(lssmodel-fitted)=
## `LSSModel.fitted`

*Method* of `LSSModel`

```python
LSSModel.fitted(parameter=None, mstop=None, type='link')
```

In-sample fitted values, mirroring `predict(newdata=None, ...)`.

Source: [`python/mboostrs/gamlss.py` lines 3363-3365](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L3363-L3365)

(lssmodel-params)=
## `LSSModel.params`

*Method* of `LSSModel`

```python
LSSModel.params(
    newdata=None,
    type='response',
    mstop=None,
    buser_newdata=None,
    unseen_levels='error',
    extrapolation='warn'
)
```

Predicted LSS parameters for all family parameters.

`type="response"` returns distribution parameters such as `sigma` on
their natural scale. `type="link"` returns the linear predictors.
`unseen_levels` is as in `predict`: how a row whose
`brandom` level the fit never saw is predicted — `"average"` as a
member of an average group, `"zero_effect"` with a zero coefficient,
`"error"` (the default) refused. `extrapolation` is as in
`predict`; pass `"clip"` to hold covariates at the training
range instead of continuing the spline edge slope.

Source: [`python/mboostrs/gamlss.py` lines 3415-3435](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L3415-L3435)

(lssmodel-predict_all)=
## `LSSModel.predict_all`

*Method* of `LSSModel`

```python
LSSModel.predict_all(
    newdata=None,
    type='response',
    mstop=None,
    output='auto',
    include_response=True,
    buser_newdata=None,
    unseen_levels='error',
    extrapolation='warn'
)
```

All-parameter prediction table, analogous to GAMLSS `predictAll`.

**Parameters**

- `output` ({"auto", "pandas", "polars", "records", "dict", "dataframe"}): `"dataframe"` is a deprecated alias for `"auto"`.
- `unseen_levels` ({"average", "zero_effect", "error"}): As in `predict`; `"average"` predicts a group the fit
  never saw as the average group.
- `extrapolation` ({"warn", "clip", "error", "allow"}, default "warn"): As in `predict`, applied to every parameter's column.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 3437-3490](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L3437-L3490)

(lssmodel-coef_all)=
## `LSSModel.coef_all`

*Method* of `LSSModel`

```python
LSSModel.coef_all(output='auto')
```

All-parameter coefficient table, analogous to GAMLSS `coefAll`.

**Parameters**

- `output` ({"auto", "pandas", "polars", "records", "dict", "dataframe"}): `"dataframe"` is a deprecated alias for `"auto"`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 3493-3518](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L3493-L3518)

(lssmodel-risk)=
## `LSSModel.risk`

*Property* of `LSSModel`

```python
LSSModel.risk
```

The joint risk after each iteration (each full cycle for a cyclic
fit, each update for a non-cyclic one), as a read-only array.

A property, like `risk_` and like the standard classes' `risk`.
Calling it, `fit.risk()`, returns the same values as a fresh writable
ndarray. The joint negative log-likelihood after each parameter's own
updates, which gamboostLSS's `risk()` reports, is
`parameter_risk`.

Source: [`python/mboostrs/gamlss.py` lines 3521-3534](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L3521-L3534)

(lssmodel-parameter_risk)=
## `LSSModel.parameter_risk`

*Method* of `LSSModel`

```python
LSSModel.parameter_risk(merge=False)
```

The joint risk after each parameter's own updates, as
gamboostLSS's `risk(merge = FALSE)` and `risk(merge = TRUE)`.

`merge=False` returns a dict keyed by parameter name; each array
starts with the offset-only risk and then holds the joint risk after
each update of that parameter. A cyclic fit's last parameter entry
per cycle is the cycle's risk (`risk`). `merge=True` returns
one array and the parameter each entry belongs to: for a cyclic fit
the per-parameter arrays interleaved by position, for a non-cyclic
fit the offset-only risk once per parameter followed by every step's
risk in order.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 3536-3575](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L3536-L3575)

(lssmodel-model_weights)=
## `LSSModel.model_weights`

*Method* of `LSSModel`

```python
LSSModel.model_weights()
```

Observation weights used for fitting.

Source: [`python/mboostrs/gamlss.py` lines 3587-3592](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L3587-L3592)

(lssmodel-deviance_increment)=
## `LSSModel.deviance_increment`

*Method* of `LSSModel`

```python
LSSModel.deviance_increment(newdata=None)
```

Per-observation deviance contribution for supported LSS families.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `NotImplementedError`

Source: [`python/mboostrs/gamlss.py` lines 3605-3614](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L3605-L3614)

(lssmodel-centiles)=
## `LSSModel.centiles`

*Method* of `LSSModel`

```python
LSSModel.centiles(probs=(0.025, 0.5, 0.975), newdata=None, mstop=None)
```

Distribution centile table for fitted LSS parameters.

Returns one row per observation, in fitting order, as a frame with an
`index` column followed by one column per requested probability. A
column is named for the probability as a **percentage**, with any
decimal point replaced by an underscore: `probs=[0.5]` gives
`q50`, `probs=[0.025, 0.975]` gives `q2_5` and `q97_5`.

Select by name rather than position; converting the frame to a bare
array puts `index` in column 0, where a caller reading column 0 as
the first centile gets row numbers instead.

Every family that exposes `quantile()` supports this, which is all of
them except `DirichletLSS` and `PIGLSS`; the rest raise
NotImplementedError naming the family.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `NotImplementedError`
- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 3616-3655](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L3616-L3655)

(lssmodel-prediction_interval)=
## `LSSModel.prediction_interval`

*Method* of `LSSModel`

```python
LSSModel.prediction_interval(
    level=0.95,
    newdata=None,
    mstop=None,
    unseen_levels='error',
    group_sd=None
)
```

Central prediction interval table for the fitted LSS distribution.

`unseen_levels` is as in `predict`; the interval is for the
conditional distribution at the point prediction the policy makes,
and does not integrate the uncertainty of a group the fit never saw.

`unseen_levels="marginal"` changes that: rows whose group the fit
never saw get the predictive distribution convolved with the
assumed group effect, `group_sd=` being its standard deviation on
the location scale. For a Gaussian location-scale fit the
convolution is closed form: the scale of an unseen-group row widens
from `sigma(x)` to `sqrt(sigma(x)^2 + group_sd^2)`. The
group-effect distribution is an assumption the caller states, not
something the fitted brandom coefficients supply — they are shrunk
by design, and the model refuses to guess their dispersion.
Rows whose group the fit saw keep their conditional interval.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`
- `NotImplementedError`

Source: [`python/mboostrs/gamlss.py` lines 3734-3800](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L3734-L3800)

(lssmodel-predint)=
## `LSSModel.predint`

*Method* of `LSSModel`

```python
LSSModel.predint(level=0.95, newdata=None, mstop=None, unseen_levels='error')
```

Alias for `prediction_interval`, matching R `predint` naming.

Source: [`python/mboostrs/gamlss.py` lines 3861-3864](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L3861-L3864)

(lssmodel-pi)=
## `LSSModel.PI`

*Method* of `LSSModel`

```python
LSSModel.PI(level=0.95, newdata=None, mstop=None, unseen_levels='error')
```

Alias for `prediction_interval`, matching R `PI` naming.

Source: [`python/mboostrs/gamlss.py` lines 3866-3869](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L3866-L3869)

(lssmodel-pit)=
## `LSSModel.pit`

*Method* of `LSSModel`

```python
LSSModel.pit(
    newdata=None,
    mstop=None,
    randomized=True,
    random_state=None,
    unseen_levels='error'
)
```

Probability integral transform values for fitted LSS distributions.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `NotImplementedError`

Source: [`python/mboostrs/gamlss.py` lines 3871-3894](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L3871-L3894)

(lssmodel-quantile_residuals)=
## `LSSModel.quantile_residuals`

*Method* of `LSSModel`

```python
LSSModel.quantile_residuals(
    newdata=None,
    mstop=None,
    randomized=True,
    random_state=None,
    unseen_levels='error'
)
```

Randomized quantile residuals for fitted LSS distributions.

For continuous families this is `Phi^-1(F(y))`. For discrete
families, `randomized=True` draws uniformly between
`F(y - 1)` and `F(y)`; `randomized=False` uses the midpoint of
that interval for deterministic diagnostics.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `NotImplementedError`

Source: [`python/mboostrs/gamlss.py` lines 3896-3912](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L3896-L3912)

(lssmodel-residual_frame)=
## `LSSModel.residual_frame`

*Method* of `LSSModel`

```python
LSSModel.residual_frame(
    kind='quantile',
    newdata=None,
    mstop=None,
    randomized=True,
    random_state=None,
    output='auto',
    unseen_levels='error'
)
```

Diagnostic residual table for LSS models.

`kind="quantile"` returns randomized quantile residuals and requires a
family CDF. `kind="response"` returns the location response residual
`y - mu` and is refused for survival and multivariate responses,
exactly as `residuals` refuses them.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`
- `NotImplementedError`

Source: [`python/mboostrs/gamlss.py` lines 3914-3968](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L3914-L3968)

(lssmodel-worm_data)=
## `LSSModel.worm_data`

*Method* of `LSSModel`

```python
LSSModel.worm_data(
    xvar=None,
    n_intervals=4,
    newdata=None,
    mstop=None,
    randomized=True,
    random_state=None,
    output='auto'
)
```

Worm-plot data. See `mboostrs.worm_data`.

The table `worm_plot` draws from, for a caller who wants the
numbers rather than the figure.

Source: [`python/mboostrs/gamlss.py` lines 3970-3982](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L3970-L3982)

(lssmodel-calibration_frame)=
## `LSSModel.calibration_frame`

*Method* of `LSSModel`

```python
LSSModel.calibration_frame(xvar=None, bins=10, newdata=None, mstop=None, output='auto')
```

Calibration table. See `mboostrs.calibration_frame`.

Reports the location parameter's mean per bin, under the name
`location_mean`: on a distributional fit there is no single fitted
vector, and the first parameter is the mean only for some families.

Source: [`python/mboostrs/gamlss.py` lines 3984-3994](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L3984-L3994)

(lssmodel-q_stats)=
## `LSSModel.q_stats`

*Method* of `LSSModel`

```python
LSSModel.q_stats(
    xvar=None,
    n_intervals=10,
    newdata=None,
    mstop=None,
    randomized=True,
    random_state=None,
    output='auto'
)
```

Binned quantile-residual moment summaries.

A data-table analogue of the descriptive part of GAMLSS `Q.stats`.
It uses the same equal-count bins and sample-variance convention as
`mboostrs.q_stats`.

Source: [`python/mboostrs/gamlss.py` lines 3996-4024](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L3996-L4024)

(lssmodel-density_overlay_data)=
## `LSSModel.density_overlay_data`

*Method* of `LSSModel`

```python
LSSModel.density_overlay_data(
    newdata=None,
    mstop=None,
    grid=None,
    n_grid=200,
    weights=None,
    output='auto'
)
```

Observed and fitted marginal-distribution curves as a table.

Source: [`python/mboostrs/gamlss.py` lines 4026-4034](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L4026-L4034)

(lssmodel-centile_fan_data)=
## `LSSModel.centile_fan_data`

*Method* of `LSSModel`

```python
LSSModel.centile_fan_data(
    xvar,
    probs=(0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975),
    split_by=None,
    newdata=None,
    mstop=None,
    output='auto'
)
```

Observed responses and fitted centiles ordered by `xvar`.

Source: [`python/mboostrs/gamlss.py` lines 4036-4045](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L4036-L4045)

(lssmodel-deviance_profile)=
## `LSSModel.deviance_profile`

*Method* of `LSSModel`

```python
LSSModel.deviance_profile(
    xvar=None,
    bins=10,
    newdata=None,
    mstop=None,
    weights=None,
    output='auto'
)
```

Binned joint deviance contributions along a numeric covariate.

Source: [`python/mboostrs/gamlss.py` lines 4047-4055](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L4047-L4055)

(lssmodel-worm_plot)=
## `LSSModel.worm_plot`

*Method* of `LSSModel`

```python
LSSModel.worm_plot(
    xvar=None,
    n_intervals=4,
    newdata=None,
    mstop=None,
    randomized=True,
    random_state=None
)
```

Worm plot(s) of (randomized) quantile residuals (R's `gamlss::wp`).

Thin wrapper around `mboostrs.plot.plot_worm`; see there for
the full description of the panels.

Source: [`python/mboostrs/gamlss.py` lines 4057-4069](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L4057-L4069)

(lssmodel-loglik)=
## `LSSModel.logLik`

*Method* of `LSSModel`

```python
LSSModel.logLik()
```

Joint log-likelihood at the current LSS fit.

A distributional family fits every parameter of a named distribution by
minimising its negative log-likelihood, so unlike the standard
families' losses this risk drops no constants and `-risk` is the
log-likelihood exactly.

For the survival families the density is over `log(T)`, not `T`:
the `1/t` Jacobian is absent, as it is in the standard AFT families.
The number is a proper log-likelihood, but not one comparable with a
fit of the untransformed time.

Returns a plain `float`. Effective degrees of freedom for the
adaptive LSS fit are not implemented. The update count used by
`gaic` is a separate heuristic and must not be reported as
likelihood degrees of freedom.

Source: [`python/mboostrs/gamlss.py` lines 4071-4094](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L4071-L4094)

(lssmodel-negative_risk)=
## `LSSModel.negative_risk`

*Method* of `LSSModel`

```python
LSSModel.negative_risk()
```

The negated joint risk at the current fit.

Equal to `logLik` for every distributional family, because their
risk is the joint negative log-likelihood. It exists so that a caller
writing family-agnostic code can ask for the quantity the fit minimised
without asserting it is a likelihood; on a standard model the two
differ.

Source: [`python/mboostrs/gamlss.py` lines 4096-4105](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L4096-L4105)

(lssmodel-parity_notes)=
## `LSSModel.parity_notes`

*Method* of `LSSModel`

```python
LSSModel.parity_notes(*, include_global=False, output='auto')
```

Where this fit differs from what its names refer to.

See `GLMBoostModel.parity_notes`. A distributional fit collects
notes from its family and from every learner kind in its formulas.

Source: [`python/mboostrs/gamlss.py` lines 4107-4132](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L4107-L4132)

(lssmodel-likelihood_contract)=
## `LSSModel.likelihood_contract`

*Method* of `LSSModel`

```python
LSSModel.likelihood_contract()
```

What this family's risk is: a likelihood, a loss, or neither.

Returns the `mboostrs._likelihood.FamilyLikelihood` row.

Source: [`python/mboostrs/gamlss.py` lines 4134-4141](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L4134-L4141)

(lssmodel-gaic)=
## `LSSModel.gaic`

*Method* of `LSSModel`

```python
LSSModel.gaic(k=2.0)
```

Update-count penalized likelihood heuristic for LSS models.

The historical name is retained for compatibility. The penalty is
the number of boosting updates across parameters plus one, not
effective degrees of freedom. Repeated selection of the same learner
increases it at every step. It is not a normalized AIC or BIC and
does not estimate the complexity of adaptive distributional fitting.
`gaic_table` labels it `lss_update_count_heuristic` and refuses
to rank it against standard-model criteria. Prefer held-out scoring
for family selection.

Source: [`python/mboostrs/gamlss.py` lines 4143-4160](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L4143-L4160)

(lssmodel-llm_explain_prompt)=
## `LSSModel.llm_explain_prompt`

*Method* of `LSSModel`

```python
LSSModel.llm_explain_prompt(**kwargs)
```

Render this fit as a prompt for a language model.

One block per distribution parameter. See
`mboostrs.llm_explain_prompt` for the arguments (`parameter`,
`which`, `top`, `n_points`, `dataset_description`, `output`,
...).

Source: [`python/mboostrs/gamlss.py` lines 4162-4172](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L4162-L4172)

(lssmodel-llm_describe)=
## `LSSModel.llm_describe`

*Method* of `LSSModel`

```python
LSSModel.llm_describe(**kwargs)
```

Send this fit's description to a language model and return the reply.

Makes a network request. See `mboostrs.llm_describe` for the
arguments (`llm`, `base_url`, `api_key`, `client`, `reasoning`,
plus everything `mboostrs.llm_explain_prompt` takes).

Source: [`python/mboostrs/gamlss.py` lines 4174-4183](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L4174-L4183)

(lssmodel-to_dict)=
## `LSSModel.to_dict`

*Method* of `LSSModel`

```python
LSSModel.to_dict(*, include_arrays=False)
```

The `summary(as_dict=True)` record plus a `"provenance"` entry:
the library build and environment, and the fingerprint of the
training data. `include_arrays` is accepted for the uniform export
signature; the summary holds no per-row arrays. A model loaded from
an artifact keeps no training frame, so its provenance has no
`"data"` entry.

Source: [`python/mboostrs/gamlss.py` lines 4185-4194](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L4185-L4194)

(lssmodel-summary)=
## `LSSModel.summary`

*Method* of `LSSModel`

```python
LSSModel.summary(as_dict=False, cv=None)
```

Summarise the fitted LSS model.

Returns a `mboostrs.summary.Summary`, which prints as
fixed-width text and renders as a table in a notebook, with one
selection table per distribution parameter. Use `as_dict=True` for
the machine-readable form.

Source: [`python/mboostrs/gamlss.py` lines 4196-4232](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L4196-L4232)

(lssmodel-cvrisk)=
## `LSSModel.cvrisk`

*Method* of `LSSModel`

```python
LSSModel.cvrisk(
    folds=None,
    grid=None,
    B=5,
    seed=42,
    length_out=10,
    unseen_levels='error',
    **kwargs
)
```

Cross-validated risk for this model's own specification.

Dispatches on the fitting function that produced the model, so
`fit.cvrisk()` replaces
`mb.cvrisk_gamboostLSS(formula, data, families, ..., folds=...)` and
cannot disagree with the fit about which of the two it is.

If `folds` or `grid` is omitted, the method builds them from this
model's weights, step count and the supplied seed.

`folds` defaults to `B`-fold in-bag weights drawn from this model's
own weights, and `grid` to the **diagonal**: every parameter given the
same budget, `length_out` points up to the fitted `mstop`. The
diagonal is not a shortcut but the cheap direction -- the engine reads
every diagonal point off one fitted path per fold, where each
off-diagonal point needs its own fit, because a cyclic path is not
prefix-consistent. `mb.diagonal_grid` builds the same thing, and
`LssCvRiskResult.grid_cost()` reports what a given grid cost.

**Parameters**

- `folds` (FoldScheme or array_like of shape (n, B), optional): In-bag weight matrix or scheme. `None` draws `B`-fold k-fold
  weights from the model's weights with `seed`.
- `grid` (sequence, optional): Stopping budgets to evaluate. `None` uses the diagonal grid.
- `B` (int, default 5): Number of folds when `folds` is `None`.
- `seed` (int, default 42): Seed for generated folds.
- `length_out` (int, default 10): Number of diagonal grid points when `grid` is `None`.
- `unseen_levels` ({"error", "average", "zero_effect"}, default "error"): Policy for factor or group levels held out completely in a fold.
  Passed to held-out prediction in non-cyclic cross-validation.
- `**kwargs`: Overrides of the arguments the model is refitted with, passed to
  `cvrisk_gamboostLSS` or `cvrisk_glmboostLSS`.

**Returns**

LssCvRiskResult
    Cross-validated risk over the grid.

**Raises**

- `ValueError`: If `folds` is not a two-dimensional matrix or `unseen_levels`
is invalid.

Source: [`python/mboostrs/gamlss.py` lines 4293-4386](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L4293-L4386)

(lssmodel-stabsel)=
## `LSSModel.stabsel`

*Method* of `LSSModel`

```python
LSSModel.stabsel(**kwargs)
```

Stability selection for this model's own specification.

Replaces `mb.stabsel_glmboostLSS` / `mb.stabsel_gamboostLSS`; see
those for the error-control parameters.

The model's own weights go with it. Without that the subsamples were
drawn with unit weights whatever the fit used, so a weighted model's
selection frequencies described a sample the caller never fitted --
and nothing said so. `cvrisk` has always passed them, by building its
folds from `model_weights()`; this path built its own inside the entry
point and had no way to receive them.

Source: [`python/mboostrs/gamlss.py` lines 4537-4558](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L4537-L4558)

(lssmodel-profile_mstop)=
## `LSSModel.profile_mstop`

*Method* of `LSSModel`

```python
LSSModel.profile_mstop(grid=None, metric='risk')
```

Refit over a grid of stopping points and report `metric` at each.

Source: [`python/mboostrs/gamlss.py` lines 4560-4564](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L4560-L4564)

(lssmodel-control)=
## `LSSModel.control`

*Property* of `LSSModel`

```python
LSSModel.control
```

The `boost_control` this model was fitted with.

Source: [`python/mboostrs/gamlss.py` lines 4566-4569](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L4566-L4569)

(lssmodel-offset)=
## `LSSModel.offset`

*Property* of `LSSModel`

```python
LSSModel.offset
```

Per-parameter offsets, as `{parameter: value}`.

Same mapping as `offset_`, with one value per distribution
parameter.

Source: [`python/mboostrs/gamlss.py` lines 4571-4578](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L4571-L4578)

(lssmodel-plot_risk)=
## `LSSModel.plot_risk`

*Method* of `LSSModel`

```python
LSSModel.plot_risk(**kwargs)
```

Plot the joint risk path.

Source: [`python/mboostrs/gamlss.py` lines 4580-4584](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L4580-L4584)

(lssmodel-plot_selected)=
## `LSSModel.plot_selected`

*Method* of `LSSModel`

```python
LSSModel.plot_selected(**kwargs)
```

Plot selection frequency per base learner and parameter.

Source: [`python/mboostrs/gamlss.py` lines 4586-4590](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L4586-L4590)

(lssmodel-plot_density)=
## `LSSModel.plot_density`

*Method* of `LSSModel`

```python
LSSModel.plot_density(**kwargs)
```

Overlay the observed and average fitted response distributions.

Source: [`python/mboostrs/gamlss.py` lines 4592-4596](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L4592-L4596)

(lssmodel-plot_centile_fan)=
## `LSSModel.plot_centile_fan`

*Method* of `LSSModel`

```python
LSSModel.plot_centile_fan(xvar, **kwargs)
```

Plot nested fitted centile bands against `xvar`.

Source: [`python/mboostrs/gamlss.py` lines 4598-4602](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L4598-L4602)

(lssmodel-plot_deviance_profile)=
## `LSSModel.plot_deviance_profile`

*Method* of `LSSModel`

```python
LSSModel.plot_deviance_profile(**kwargs)
```

Plot binned joint deviance along a covariate or row order.

Source: [`python/mboostrs/gamlss.py` lines 4604-4608](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L4604-L4608)

(lssmodel-continue_to)=
## `LSSModel.continue_to`

*Method* of `LSSModel`

```python
LSSModel.continue_to(mstop)
```

Return the model grown to a larger `mstop`, per parameter.

The standard models' `continue_to` grows an existing path; an LSS path
cannot be extended in place, because each parameter has its own risk
path and the cyclic schedule interleaves them — which is why
`mstop<-` for the LSS classes in R delegates to `update()` too. This
refits, and is exact rather than warm-started.

`mstop` is either a scalar, broadcast to every parameter, or a mapping
from parameter name to iteration count.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 4610-4649](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L4610-L4649)

(lssmodel-varimp)=
## `LSSModel.varimp`

*Method* of `LSSModel`

```python
LSSModel.varimp(percent=True, parameter=None)
```

Variable importance from in-bag risk reduction, per parameter.

The standard models attribute each iteration's risk drop to the learner
selected at that iteration. An LSS fit interleaves updates across
parameters and, in a cyclic fit, records risk once per *cycle* rather
than once per update — so this reads `path_frame()`, which already
resolves that indexing and the per-parameter learner names, and sums
its `risk_change` per learner.

Results are keyed by `"<learner>.<parameter>"`, the same combined name
stability selection uses. Returns a dict with `reduction` and
`selfreq`, matching the standard models' shape. `parameter=`
restricts to one distribution parameter.

Requires `method="noncyclic"`. A cyclic fit records risk once per
cycle rather than once per update, so there is nothing to attribute
per learner; it raises `NotImplementedError` saying so rather than
returning a table in which the dominant predictor scores zero.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `NotImplementedError`

Source: [`python/mboostrs/gamlss.py` lines 4651-4695](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L4651-L4695)

(lssmodel-base_learner_names)=
## `LSSModel.base_learner_names`

*Method* of `LSSModel`

```python
LSSModel.base_learner_names(parameter=None)
```

Base-learner names, as a `{parameter: [name, ...]}` mapping.

`parameter=` restricts to one distribution parameter and returns a
plain list, matching what the standard models return.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 4703-4723](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L4703-L4723)

(lssmodel-bnames)=
## `LSSModel.bnames`

*Method* of `LSSModel`

```python
LSSModel.bnames(parameter=None)
```

Alias for `base_learner_names`.

Source: [`python/mboostrs/gamlss.py` lines 4725-4727](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L4725-L4727)

(lssmodel-variable_names)=
## `LSSModel.variable_names`

*Method* of `LSSModel`

```python
LSSModel.variable_names(parameter=None)
```

Distinct predictor names, in formula order.

Source: [`python/mboostrs/gamlss.py` lines 4729-4737](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L4729-L4737)

(lssmodel-selected_which)=
## `LSSModel.selected_which`

*Method* of `LSSModel`

```python
LSSModel.selected_which(which=None, parameter=None, merge=False)
```

Selected learner names per parameter, optionally filtered.

`which` accepts a learner name or a list of them. Selections are
returned as `{parameter: [name, ...]}`, or a plain list when
`parameter=` names one. `merge=True` flattens the mapping to one list
of parameter-qualified names such as `"x.mu"`, which is the form
that lets two parameters' paths be counted together.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 4739-4770](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L4739-L4770)

(lssmodel-fitted_)=
## `LSSModel.fitted_`

*Property* of `LSSModel`

```python
LSSModel.fitted_
```

Fitted link-scale values per parameter, as a mapping.

The standard models' `fitted_` is one array; an LSS fit has one per
distribution parameter, so this mirrors `fitted()`'s shape.

Source: [`python/mboostrs/gamlss.py` lines 4772-4786](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L4772-L4786)

(lssmodel-offset_)=
## `LSSModel.offset_`

*Property* of `LSSModel`

```python
LSSModel.offset_
```

Per-parameter offsets, as a mapping. See also `offset`.

Source: [`python/mboostrs/gamlss.py` lines 4788-4794](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L4788-L4794)

(lssmodel-save)=
## `LSSModel.save`

*Method* of `LSSModel`

```python
LSSModel.save(path, *, indent=None)
```

Write a portable JSON artifact. See `mboostrs.save_model`.

Load it with `mboostrs.load_model`. A distributional artifact
carries one learner block per distribution parameter and the family's
own state; the Rust and R readers score linear and additive models
only and refuse it by name.

Source: [`python/mboostrs/gamlss.py` lines 4796-4806](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L4796-L4806)

(lssmodel-with_mstop)=
## `LSSModel.with_mstop`

*Method* of `LSSModel`

```python
LSSModel.with_mstop(k)
```

Return a copy refit to `k` boosting iterations.

Equivalent to `model.update(mstop=k)` but more readable.
Scalar `k` is broadcast to all parameters.

Source: [`python/mboostrs/gamlss.py` lines 4808-4814](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L4808-L4814)

(lssmodel-update)=
## `LSSModel.update`

*Method* of `LSSModel`

```python
LSSModel.update(mstop=None, weights=None, oobweights=None, risk=None, trace=None)
```

Refit the stored LSS model specification.

Supports changing `mstop`, `weights`, `risk`, and `trace`. The
latter two are forwarded to `boost_control`; when omitted, the
original fit's settings are carried over.

`oobweights` is unsupported because the LSS fit functions do not
take a separate out-of-bag weight vector. Use `risk="oobag"` with
zero weights on held-out rows instead.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `NotImplementedError`
- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 4845-4993](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L4845-L4993)

(lssmodel-extrapolation_report)=
## `LSSModel.extrapolation_report`

*Method* of `LSSModel`

```python
LSSModel.extrapolation_report(newdata, output='auto')
```

Terms and covariates with rows outside their training range, by parameter.

Columns are those of `GAMBoostModel.extrapolation_report`, with
`parameter` naming the distribution parameter whose predictor holds the
term. Only spline terms appear: `predict(extrapolation=)` applies to
spline learners, and a parameter whose predictor is linear has no rows.

Source: [`python/mboostrs/gamlss.py` lines 5075-5106](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5075-L5106)

(lssmodel-mean)=
## `LSSModel.mean`

*Method* of `LSSModel`

```python
LSSModel.mean(
    newdata=None,
    mstop=None,
    buser_newdata=None,
    unseen_levels='error',
    extrapolation='warn'
)
```

Expected response under the fitted distribution.

For a zero-inflated family this is the mixture mean, not the mean of
the count component: `ZIPoLSS` returns `(1 - sigma) * mu` and
`ZINBLSS` returns `(1 - nu) * mu`. `predict(type="response")`
still returns the parameters themselves.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `TypeError`

Source: [`python/mboostrs/gamlss.py` lines 5170-5198](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5170-L5198)

(lssmodel-predict)=
## `LSSModel.predict`

*Method* of `LSSModel`

```python
LSSModel.predict(
    newdata=None,
    parameter=None,
    type=<object object>,
    mstop=None,
    buser_newdata=None,
    unseen_levels='error',
    extrapolation='warn'
)
```

Predictions for one distribution parameter, or for all of them.

`type="link"` is the linear predictor. `type="response"` applies
each parameter's link, so for `ZIPoLSS` `mu` is the Poisson mean
of the count component and `sigma` is the extra-zero probability —
not the expected count. `type="mean"` is that expected response
(`(1 - sigma) * mu` for `ZIPoLSS`); it returns one array and
ignores `parameter`.

Returns an array when `parameter` names a parameter or `type` is
`"mean"`, otherwise a dict keyed by parameter name. `unseen_levels` is as in
`GAMBoostModel.predict`: `"average"` predicts a `brandom`
level the fit never saw as an average group, and `"zero_effect"`
with a zero coefficient, instead of refusing it.

`extrapolation` is as in `GAMBoostModel.predict`, applied to
every parameter's spline learners (`bbs`, `bns`, `bmono`,
constrained splines), to both coordinates of `bspatial`, and to the
B-spline margins of `%X%` terms. `btree`, `bkernel`, `buser`
and `brad` terms have no training-range boundary, so the policy does
not apply to them. `"warn"` (default) warns when a new value lies
materially outside a learner's training range, `"error"` refuses any
value outside it, `"clip"` holds the effect flat at the boundary, and
`"allow"` continues the basis silently. A spline basis continues at
its edge slope outside the boundary knots, which the fit does not
constrain, and a scale or shape parameter extrapolated that way can
reach implausible values quickly.

**Parameters**

- `newdata` (dict or DataFrame, optional): Rows to predict. `None` predicts the training rows.
- `parameter` (str, optional): One distribution parameter (for example `"mu"`). `None` returns
  every parameter. Not allowed with `type="mean"`.
- `type` ({"link", "response", "mean"}, default "link"): `"link"` is each parameter's linear predictor, `"response"`
  applies the inverse link, and `"mean"` is the expected response
  E[Y]. Leaving `type` unset predicts on the link scale and
  emits a one-time `UserWarning` per model when that scale is not
  the response scale.
- `mstop` (int or sequence of int, optional): Truncate the boosting path. A scalar is a per-parameter budget for
  a cyclic model and a global update budget for a non-cyclic one;
  `None` uses the fitted stopping iterations.
- `buser_newdata` (dict, optional): Design matrices for `buser` learners, which cannot be rebuilt
  from a column of `newdata`.
- `unseen_levels` ({"error", "average", "zero_effect"}, default "error"): Treatment of categorical levels absent from the fit, as in
  `GAMBoostModel.predict`.
- `extrapolation` ({"warn", "error", "allow", "clip"}, default "warn"): Treatment of rows outside a spline learner's training range.

**Returns**

ndarray or dict of ndarray
    Shape `(n,)` arrays, one value per row of `newdata`. An array
    when `parameter` is given or `type="mean"`; otherwise a dict
    keyed by parameter name.

**Raises**

- `ValueError`: If `type` or `extrapolation` is not one of the listed values,
or `parameter` is combined with `type="mean"`.

Source: [`python/mboostrs/gamlss.py` lines 5200-5329](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5200-L5329)

(lssmodel-partial_effects)=
## `LSSModel.partial_effects`

*Method* of `LSSModel`

```python
LSSModel.partial_effects(
    parameter,
    which=None,
    newdata=None,
    mstop=None,
    buser_newdata=None,
    unseen_levels='error',
    extrapolation='allow'
)
```

Return formula-ordered learner contributions for one LSS parameter.

Each effect excludes the parameter offset. Adding the returned effect
vectors to that offset reconstructs the parameter's link predictor.
`mstop` follows `predict`: a scalar is a per-parameter budget for
cyclic models and a global update budget for noncyclic models.
`unseen_levels` is as in `predict`.

`extrapolation` is as in `predict`, applied to the spline and
`bspatial` terms of `newdata` before their effects are evaluated.
It defaults to `"allow"`, as `GAMBoostModel.partial_effects`
does. The `%X%` margins are not checked here: this method reads the
raw `newdata` and does not rebuild the `%X%` designs that
`predict` builds. Under `"clip"` the effect is evaluated at the
clamped values, while the returned coordinates are the values supplied.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `NotImplementedError`
- `IndexError`
- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 5331-5435](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5331-L5435)

(lssmodel-coef)=
## `LSSModel.coef`

*Method* of `LSSModel`

```python
LSSModel.coef(parameter=None)
```

Per-parameter aggregated coefficients (sum of nu*coef per learner).

Returns a mapping from each parameter to learner coefficients, or
one parameter's mapping when `parameter` is given. Coefficients sum
the updates after multiplying each by that parameter's learning rate;
the intercept includes the shifts induced by centered predictors.

Source: [`python/mboostrs/gamlss.py` lines 5437-5497](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5437-L5497)

(lssmodel-residual_kinds)=
## `LSSModel.residual_kinds`

*Method* of `LSSModel`

```python
LSSModel.residual_kinds() -> tuple[str, ...]
```

Residual kinds defined for the fitted LSS family.

`working` is always present (the negative gradient exists for every
parameter); `response` needs an uncensored univariate response and a
response-scale `mu`; `pearson` needs a declared variance function,
which only `GaussianLSS` and `PoissonLSS` have.

Source: [`python/mboostrs/gamlss.py` lines 5499-5507](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5499-L5507)

(lssmodel-residuals)=
## `LSSModel.residuals`

*Method* of `LSSModel`

```python
LSSModel.residuals(parameter=None, type='response', newdata=None, mstop=None)
```

Residuals of the named kind for an LSS parameter, or a refusal.

`type` is checked against the family's declared kinds, so a kind that
is undefined raises with the alternative instead of returning a
different quantity under the requested name:

- `"response"`: `y - mu` with `mu` on the location's response
  scale. Only defined for the location parameter, and refused for
  survival (censored) and multivariate (composition) responses.
- `"pearson"`: `(y - mu) / sqrt(V(mu))`, only where a variance
  function is declared: `GaussianLSS` divides by `sigma`,
  `PoissonLSS` by `sqrt(mu)`.
- `"working"`: the fitted negative gradient for the named parameter
  -- the quantity the next boosting step actually regresses on,
  computed by the same core routine used by the fitting loop.

`newdata` and `mstop` behave as for `predict`. When
`parameter` is `None` the result is a dict ``{location_name:
residuals}``; otherwise a plain array.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`
- `NotImplementedError`

Source: [`python/mboostrs/gamlss.py` lines 5509-5620](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5509-L5620)
