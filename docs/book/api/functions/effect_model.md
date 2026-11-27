# `effect_model`

Defined in `mboostrs.causal`; exported as `mboostrs.effect_model`.


*Function*

```python
effect_model(
    data,
    spec: mboostrs.causal.CausalSpec,
    nuisance=None,
    modifiers=None,
    terms='auto',
    folds=5,
    seed=42,
    df=4,
    propensity_clip=from_spec,
    max_mstop=None,
    outcome_missing=from_spec,
    weights=None,
    overlap_bounds=from_spec,
    learner='r_winsorized',
    extrapolation='warn',
    flat_tolerance=0.01,
    propensity_learner='smooth',
    interval_method='auto',
    interval_repeats=5,
    bootstrap_draws=200
)
```

Estimate the conditional average treatment effect `tau(x)`.

The R-learner (Nie and Wager, 2021) fits `tau` by minimizing
`sum w (y - m(x) - (t - e(x)) tau(x))^2`, where `m(x) = E[y | x]` and
`e(x) = P(t = 1 | x)` are predicted out of fold. This equals a weighted
regression of the pseudo-outcome `(y - m)/(t - e)` with weights
`w (t - e)^2`. Because `y - m` and `t - e` are residuals, a
confounder's effect on the outcome does not leak into `tau` when the
nuisances are accurate.

Intervals. With `terms="smooth"` the second stage is a penalized
additive model (`mboostrs.selection.gam_refit`) whose smoothing is
chosen by REML. The pseudo-outcome's error variance is proportional to
`1 / (t - e)^2`, so the weights `(t - e)^2` make the fit weighted least
squares with the correct error variance, and the coefficient covariance
is the one that weighting implies. `predict(newdata, interval=True)`
returns, per row, the estimate `tau(x)`, its standard error from the
full coefficient covariance (the covariance between the smooth terms is
included) and a pointwise interval. The interval covers the sampling
variation of the second stage given the cross-fitted nuisances: it treats
`m` and `e` as fixed at their out-of-fold predictions. It does not
cover error in those nuisances, the choice of modifiers or `terms`, or
the bias a penalized smooth carries where the true effect bends sharply.
Its coverage is near nominal when the additive form holds; the test suite
checks it on simulated effects. The `"stumps"` and `"joint"`
candidates are boosted fits, which have no coefficient covariance, so
their per-row intervals are refused.

`interval_method="second_stage_bootstrap"` keeps the pseudo-outcome and
the nuisance predictions fixed and resamples the fitting rows with
replacement `bootstrap_draws` times. Each resample is a multinomial draw
of the rows, used as prior weights on the full-sample spline basis, and the
second stage is refitted on it with its smoothing chosen by REML. The
interval is the percentile interval of the draws, and the estimate is the
full-sample fit. It captures the variation of the second stage, including
its smoothing choice, at the cost of one penalized refit per draw; it does
not refit the nuisances, so it accepts `nuisance=`. `"auto"`, the
default, is this method for a smooth second stage and `"refit"` for the
boosted candidates.

`interval_method="repeats"` refits the nuisances and the second stage on
`interval_repeats` split seeds (the primary fit is one of them). At each
row it takes the estimate and standard error from every fit and combines
them by the median rule of Chernozhukov et al. (2018): the estimate is the
median over fits, and the variance is the median of ``se_r^2 + (tau_r -
estimate)^2``. The second term adds the split-to-split spread to each fit's
own sampling variance. `interval_method="bootstrap"` resamples the rows
with replacement `bootstrap_draws` times, refits the whole pipeline on
each resample, and reports the full-sample estimate with the percentile
interval of the refitted effects. Both methods refit the nuisances, so they
refuse `nuisance=`, and both need the smooth second stage. Neither removes
the bias of the nuisances or of the penalized smooth.

Checks. The constant effect, `tau` equal to the ATE, is scored on the same
folds (`constant_rloss`; `diagnostics["constant_effect"]` gives its
difference from the chosen candidate and that difference's standard error).
A warning is issued when the chosen candidate does not beat the constant by
more than its standard error, and when a boosted candidate is tuned to fewer
than five steps, which makes it close to a constant. A warning is also issued
when the largest absolute fitted effect over the fitting rows exceeds 10 times
the interquartile range of the observed outcome. Extrapolation warnings from
the fold fits are reported once for the call.

Bounded second stage. The R-learner pseudo-outcome `(y - m) / (t - e)`
has no bound: where `|t - e|` is small it is a large multiple of the
residual. With `learner="r_winsorized"` (the default) it is clipped to its
1st and 99th percentiles over the rows with positive weight before the
second stage, so that no row's pseudo-outcome lies beyond those percentiles.
The clipping does not drop rows, and the bounds are reported in
`diagnostics["pseudo_bounds"]`. Clipping also shrinks the effects of the rows
in the tails when those tails are not heavy: on a design with light tails the
per-row 90% coverage falls by one to two points. `learner="r"` keeps the
unbounded pseudo-outcome.

Bounded modifiers. The smooth candidate fits each modifier's spline over its
central range, the 1st and 99th percentiles of the fitting rows, so its
equally spaced knots fall over the rows and not over a few extreme values.
Rows outside a range are left out of the fit, and predictions outside it are
held at the range edge (`extrapolation="clip"`, the default of
`EffectModel.predict` for `terms="smooth"`). A positive modifier whose
99th percentile is at least 3 times its median is entered as its logarithm,
which spreads its long right tail over the basis. The chosen scales and ranges
are in `diagnostics["smooth_scale"]`. The boosted candidates use the raw
columns.

**Parameters**

- `data` (mapping or data frame): Outcome, treatment and covariate columns. Covariates may be categorical
  (strings or factors); they are read as factors, as the boosting formulas
  read them. A missing categorical value is an error.
- `spec` (CausalSpec): The treatment, outcome, covariates and assumptions.
- `nuisance` (AIPWResult, optional): Reuse the cross-fitted propensity, outcome and response models of
  `aipw` on the same rows (`m = e mu1 + (1 - e) mu0`). By default
  they are cross-fitted here with smooth terms and a tuned step count.
  The result's `outcome_missing` and weights must match these arguments.
- `modifiers` (sequence of str, optional): The covariates `tau` may depend on; all of `spec.covariates` by
  default.
- `terms` ({"auto", "smooth", "stumps", "joint"}): `"smooth"` gives each modifier a penalized smooth (a linear term for
  a modifier with fewer than ten distinct values), with REML smoothing
  and per-row intervals; `"stumps"` a one-split tree for effects that
  jump at a threshold; `"joint"` one depth-3 tree that may split on any
  modifier. The joint tree is the only candidate that can represent an
  effect of two modifiers at once, such as `1 + 1.5 * (x0 * x1 > 0)`,
  for which each modifier's own effect is flat; it selects its split by
  squared error because a modifier whose joint effect is strong can have
  a marginal association near zero. `"auto"` scores every applicable
  candidate by its out-of-fold R-loss, on the same folds, and then
  restricts the choice to the candidates that beat a constant effect by
  more than their standard error and whose tuned step count is not close
  to zero (a boosted fit that stops within 5 steps is close to a
  constant). Among those it keeps the lowest R-loss, except that the joint
  tree must be at least 5 percent lower than the best additive candidate
  to be preferred. A choice that beats `"smooth"` by no more than one
  standard error of their loss difference is replaced by `"smooth"`. If
  no candidate is eligible, `"smooth"` is returned and the constant-effect
  warning says so. `diagnostics["terms_selection"]` gives the gains, the
  flags and the rule that decided the choice, and `EffectModel.describe`
  states it. The candidates are not mixed in one model. The joint candidate
  needs two or more modifiers.
- `folds` (int): Outer cross-fitting folds for the nuisances and the folds that score
  and tune `tau`, stratified by treatment.
- `df` (float): The `df` of the boosted smooths in the nuisance models. The smooth
  candidate's basis and smoothing come from the REML refit, which does
  not read `df`.
- `outcome_missing` ({"error", "ipw"}, optional): When the call does not pass it, the `outcome_missing` of the
  `CausalSpec` is used, which defaults to `"error"`: an observed
  outcome is required in every row. `"ipw"` treats the outcome as
  missing at random given the covariates and treatment: the outcome mean
  is fitted on observed rows, and the R-loss weight of each row is
  multiplied by `R / pi`, where `pi` is the cross-fitted response
  propensity. The effect is then estimated for all rows, as in
  `aipw`.
- `weights` (array-like, optional): Observation weights read as frequencies. They enter the nuisance fits
  and the R-loss.
- `propensity_clip` (tuple or None, optional): Bounds for `e(x)` and, under `outcome_missing="ipw"`, for the
  response propensity `pi`. When the call does not pass it, the
  `propensity_clip` of the `CausalSpec` is used, which defaults
  to `(0.01, 0.99)`, the same as `aipw`, so the two estimators
  weight rows alike. An argument that is passed always wins. The count of
  clipped rows is reported. `None` disables clipping.
- `extrapolation` ({"warn", "error", "allow", "clip"}, default "warn"): The policy for the cross-fitted nuisance predictions of a row outside a
  spline's training range, as in `aipw`. The warning names the
  nuisance model and this argument.
- `flat_tolerance` (float or None, default 0.01): The relative gap at which a tuned nuisance's cross-validated curve
  counts as flat (see `aipw`). `None` turns that check off. The
  second stage's own tuning keeps the default check of
  `mboostrs.explore.tune_mstop`.
- `max_mstop` (int, optional): The largest step count searched by every boosted fit: the outer
  cross-validated tuning of each nuisance model and the tuning of a boosted
  `tau` candidate. `None` (the default) is 1600, the cap that
  `aipw` uses. Before 1600 was the default the cap was 10,000, and a
  propensity model reached it in every fold, so the nuisance tuning took
  most of the fit time.
- `overlap_bounds` (tuple): Propensity bounds `(lower, upper)` that define the overlap region.
  Rows outside them are reported in `diagnostics["overlap"]` and, for
  new rows, flagged by `EffectModel.predict`.
- `learner` ({"r_winsorized", "r"}): The pseudo-outcome the second stage fits. `"r_winsorized"` (the
  default) clips it as described above. `"r"` uses the unbounded
  pseudo-outcome; where overlap is thin, a flexible second stage can
  then place effects far outside the outcome's range.
- `propensity_learner` ({"smooth", "auto", "smooth+trees"}): Which propensity formula is fitted. `"smooth"`, the default, fits the
  base formula only. `"auto"` compares the base formula with the base plus
  a depth-one tree per numeric covariate, on the same folds, and keeps the
  one with the lower out-of-fold binomial deviance. `"smooth+trees"` fits
  the tree formula only. The formulas and the deviances are in
  `diagnostics["propensity_learner"]`.
  The base formula of `aipw` is the one its `smooth` argument
  builds, linear by default; here it is the spline dictionary.
- `interval_method` ({"auto", "refit", "second_stage_bootstrap", "repeats", "bootstrap"}): How `predict(..., interval=True)` builds the per-row interval. `"auto"`,
  the default, is `"second_stage_bootstrap"` for a smooth second stage and
  `"refit"` for the boosted candidates, as described under Intervals.
  `"repeats"` and `"bootstrap"` refit the whole pipeline and need the
  smooth second stage.
- `interval_repeats` (int): With `interval_method="repeats"`, the number of split seeds, the primary
  fit included. The fit costs that many times the single fit. Ignored otherwise.
- `bootstrap_draws` (int): With `interval_method="bootstrap"`, the number of row resamples, each
  refitted in full; with `"second_stage_bootstrap"`, the number of
  resamples of the second stage alone. Ignored otherwise.

**Returns**

EffectModel

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `TypeError`
- `ValueError`

Source: [`python/mboostrs/causal.py` lines 4159-4747](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L4159-L4747)
