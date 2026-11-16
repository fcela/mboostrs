# `effect_modifiers`

Defined in `mboostrs.effect_modifiers`; exported as `mboostrs.effect_modifiers`.


*Function*

```python
effect_modifiers(
    result,
    data,
    covariates=None,
    method='blp',
    groups=4,
    level=0.9,
    overlap_bounds=(0.02, 0.98),
    adjust='holm',
    gate=True,
    gate_seed=42
)
```

Test whether the effect changes along each covariate, using the AIPW scores.

Rows are kept when their cross-fitted propensity lies inside
`overlap_bounds`. For each covariate the function fits two tests on those
rows: a best linear predictor (BLP) slope, the regression of the score on
the covariate (standardized for a numeric covariate), and GATES, the mean
score in `groups` quantile groups of the covariate with the top-minus-bottom
contrast. Both use heteroskedasticity-robust (HC1) standard errors. A joint
regression on all covariates is also fit; its projection ranks rows (see
`EffectModifiers.projection` and `mboostrs.rank_by_effect`).

**Parameters**

- `result` (AIPWResult): The result of `mboostrs.aipw`. Its per-row scores, cross-fitted
  propensities and observation weights (when present) are used.
- `data` (mapping or data frame): The rows the result was fit on, with the covariates as columns.
- `covariates` (sequence of str, optional): The covariates to test. Defaults to the covariates of `result.spec`. A
  name that is not a column of `data` is read as a transform expression
  with the transforms of the model formulas: `log(x)`, `log1p(x)`,
  `sqrt(x)`, `exp(x)`, `scale(x)`, `center(x)`, `ecdf(x)`,
  `pmin(x, bound)`, `pmax(x, bound)` and `I(...)` arithmetic on
  columns. `covariates=["log(prior_earnings)"]` tests the effect along
  log prior earnings. Reports name the expression as written. A fitted
  transform (`scale`, `center`, `ecdf`) keeps the parameters of
  the fitting rows when a projection scores new rows. An unsupported
  expression is an error that lists the supported transforms.
- `method` ({"blp", "gates"}): Which test decides `decision`, `direction` and `estimate`. Both
  tests are always reported.
- `groups` (int): Number of quantile groups for GATES on a numeric covariate.
- `level` (float): Confidence level of the intervals. The decision threshold is
  `1 - level` on the adjusted p-value.
- `overlap_bounds` (tuple): `(lower, upper)` propensity bounds of the rows used.
- `adjust` ({"holm", "none"}): Multiplicity adjustment across the covariates tested by the call. Holm
  controls the family-wise error rate over that set.
- `gate` (bool): With `True` (the default) the function first runs an R-learner gate: it
  fits the smooth second stage of `mboostrs.effect_model` on the
  tested covariates, using the cross-fitted nuisances of `result` and
  folds stratified by treatment, and compares its out-of-fold R-loss with
  that of a constant effect. The gate passes when the gain exceeds its
  standard error. When it fails, every `decision` is False,
  `selected()` is empty and `message` states the reason; the score
  tests are still reported, with their own result in `test_decision`.
  The gate's gain, standard error and outcome are in
  `diagnostics["gate"]`. The gate needs the treatment and outcome
  columns in `data` and covariates that are in `result.spec.covariates`
  (the columns an expression reads). `gate=False` skips it and decides
  on the score tests alone. The score regressions
  treat the scores as fixed and so omit the error of the nuisance fits; on
  data with no effect modification they can report a modifier, and the
  gate removes that claim. Cost: on a 2,000-row simulation with five
  covariates, after `mboostrs.aipw` with its defaults, the
  gate took 0.28 s of CPU time (median of five draws; about 1 s of wall time on a
  loaded machine), against 0.34 s for the `aipw` call and 0.001 s for the score
  tests. It reuses the nuisances of `result` and fits one penalized second stage
  on five folds, without the second-stage bootstrap that `effect_model` runs for
  intervals. On 40 simulated draws with no effect (2,000 rows, five covariates) the
  score tests alone claimed a modifier in 4 and the gate in none; on 20 draws with an effect that
  rises with one covariate the gate passed in all 20 and kept that covariate.
  The gate has the low power of the R-loss comparison, so a real but
  small modifier can fail it.
- `gate_seed` (int): Seed of the gate's folds.

**Returns**

EffectModifiers

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `TypeError`
- `ValueError`

**Notes**

The regressions use the observation weights of the result, as the estimate
is a weighted mean of the scores. The standard errors treat the scores as
fixed: they do not include the variability of the nuisance fits. The
covariates are not chosen from these results, so a covariate selected after
looking at them does not carry the nominal error rate.

Source: [`python/mboostrs/effect_modifiers.py` lines 478-726](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/effect_modifiers.py#L478-L726)
