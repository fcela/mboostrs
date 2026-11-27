# `iv_effect`

Defined in `mboostrs.causal`; exported as `mboostrs.iv_effect`.


*Function*

```python
iv_effect(
    data,
    spec: mboostrs.causal.CausalSpec,
    instrument,
    folds=5,
    seed=0,
    smooth=False,
    df=4,
    mstop=200,
    max_mstop=None,
    tune=True,
    inner_folds=5,
    level=0.95,
    weak_threshold=10.0,
    outcome_missing=from_spec,
    propensity_clip=from_spec
)
```

Cross-fitted partially linear instrumental-variable estimate for a binary treatment.

The model is `Y = theta D + g(X) + U` with a binary treatment `D`,
baseline covariates `X` and an instrument `Z` that is related to `D`
but reaches `Y` only through `D`. Three boosted nuisance models are
fitted out of fold on the covariates: the outcome mean `E[Y|X]`, the
treatment mean `E[D|X]` and the instrument mean `E[Z|X]`. Each
observation's residuals are taken from the fold that did not fit them
(`Y - m(X)`, `D - e(X)`, `Z - r(X)`), and the estimate is the ratio of
residual covariances, `sum(Z~ Y~) / sum(Z~ D~)` (DML-IV, Chernozhukov et
al., 2018). With the covariates linear, this is the two-stage least squares
estimate of `theta` on the covariates.

The estimand is a weighted average of the effect of `D` among compliers,
the units whose treatment the instrument switches on, under monotonicity.
With an effect that does not vary across units it is that effect. The
estimand is the complier effect whatever `spec.estimand` says; the field
is not used.

A missing outcome is handled as in `aipw`: with
`outcome_missing="ipw"` a response model for `P(outcome observed | X, D, Z)`
is cross-fitted on the same folds, the outcome mean is fitted on the observed
rows weighted by `1 / pi`, and each observed row's moment is multiplied by
`1 / pi`, where `pi` is clipped to `propensity_clip`. This assumes the
outcome is missing at random given the covariates, the treatment and the
instrument. Rows with a missing outcome get weight zero.

Uncertainty is the sandwich variance from the influence function
`W Z~ (Y~ - theta D~) / mean(W Z~ D~)`, with `W` the inverse response
weight (one when every outcome is observed) and a normal interval at
`level`. It treats the nuisance predictions as fixed and the rows as
independent draws, and it does not include uncertainty from the choice of
fold split.

The first stage is the squared correlation of `Z~` and `D~`, the partial
R2 of the treatment on the instrument after the covariates, and the partial
F `n R2 / (1 - R2)` from the same residuals. When F is below
`weak_threshold` (10 by default) the result is marked
`weak_instrument` and a RuntimeWarning is raised: a weak instrument makes
the ratio biased toward the confounded regression and the interval too
narrow.

**Parameters**

- `data` (mapping or data frame): Treatment, outcome, instrument and covariate columns. The outcome must
  be numeric; it may hold NaN or None only under `outcome_missing="ipw"`.
- `spec` (CausalSpec): Names the treatment (binary, read through `treatment_contrast`), the
  outcome and the covariates. `adjustment_rationale`, `time_zero` and
  `follow_up` are required as for every causal spec, and the
  `estimand` field is not used.
- `instrument` (str): The instrument column. It may be continuous or binary (0/1). A
  categorical instrument is refused.
- `folds` (int or FoldScheme): Cross-fitting folds, stratified on the treatment. A FoldScheme must hold
  every row out exactly once.
- `seed` (int): Seed of the fold split and of the inner step-count searches.
- `smooth, df, mstop, max_mstop, tune, inner_folds`: Nuisance-model options with the meanings they have in `aipw`:
  `smooth` (default False) fits linear terms only; `smooth=True` fits a
  penalized spline for each numeric covariate with at least ten distinct
  values and a linear term for the others. `mstop` is the first step grid
  of each nuisance and `tune` chooses its step count by inner
  cross-validation.
- `level` (float): Coverage of the interval, between 0 and 1.
- `weak_threshold` (float): The first-stage partial F below which the instrument is reported weak.
- `outcome_missing` ({"error", "ipw"}, optional): What a missing outcome (NaN) means. `"error"` refuses a missing outcome
  and names this argument. `"ipw"` keeps every row through the inverse
  response weighting described above. When the call does not pass it, the
  `outcome_missing` of the `CausalSpec` is used.
- `propensity_clip` (tuple or None, optional): Bounds for the response probability `pi` under `outcome_missing="ipw"`.
  When the call does not pass it, the `propensity_clip` of the spec is used.

**Returns**

IVResult
    `estimate`, `standard_error`, `interval`, the first-stage
    `first_stage_r2` and `first_stage_f`, `weak_instrument`, the
    per-row residuals and influence values, and the nuisance step counts.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `TypeError`
- `ValueError`

Source: [`python/mboostrs/causal.py` lines 6278-6543](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L6278-L6543)
