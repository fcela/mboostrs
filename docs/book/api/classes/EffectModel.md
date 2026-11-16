# `EffectModel`

Defined in `mboostrs.causal`; exported as `mboostrs.EffectModel`.


*Class*

```python
EffectModel(
    model: object,
    spec: mboostrs.causal.CausalSpec,
    modifiers: tuple[str, ...],
    terms: str,
    cv_rloss: dict,
    tau: numpy.ndarray,
    propensity: numpy.ndarray,
    outcome_mean: numpy.ndarray,
    diagnostics: dict,
    propensity_model: object = None,
    overlap_bounds: tuple = (0.02, 0.98),
    constant_rloss: float = nan,
    scale_plan: dict = <factory>,
    interval_method: str = 'refit',
    interval_fits: tuple = (),
    second_stage: object = None
) -> None
```

A conditional average treatment effect `tau(x)`, fitted by the
R-learner.

`terms` names the candidate whose cross-validated R-loss was lowest,
and `cv_rloss` holds every candidate's score. `tau` holds the
in-sample predictions and `ate_plugin` their unweighted mean; use
`aipw` for an average effect with an interval.

`model` is the fitted `tau`. For `terms="smooth"` it is a
`mboostrs.selection.GamRefit`: a penalized additive model whose
smoothing parameters are chosen by REML, so `predict(..., interval=True)`
gives a per-row interval. For `"stumps"` and `"joint"` it is a boosted
model, which has no coefficient covariance and therefore no per-row
interval.

`propensity_model` is a boosted model of `e(x)` fitted on all rows; it
flags new rows whose estimated propensity lies outside `overlap_bounds`
when `predict` is called. `propensity` holds the cross-fitted
propensities of the fitting rows as an array; calling it, as
`propensity(newdata)`, returns `e(x)` for new rows. `overlap` in `diagnostics` summarizes
the cross-fitted propensities of the fitting rows.

The `smooth` and `stumps` candidates are additive in the modifiers:
each modifier contributes its own terms and none of them can depend on
another. The `joint` candidate is a single tree over all modifiers, so
each modifier's share of it depends on the others; a fitted `joint`
model reports a combined `partial_effects` row rather than one row per
modifier.

Source: [`python/mboostrs/causal.py` lines 3240-3567](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L3240-L3567)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`predict`](#effectmodel-predict) | Method | ``tau(x)`` for new rows. |
| [`ate_plugin`](#effectmodel-ate_plugin) | Property | The unweighted mean of the fitted ``tau(x)`` over the fitting rows. |
| [`overlap_report`](#effectmodel-overlap_report) | Method | The overlap summary of the fitting rows; see :func:`overlap_report`. |
| [`to_dict`](#effectmodel-to_dict) | Method | A JSON-safe record; per-row arrays only on request. |
| [`describe`](#effectmodel-describe) | Method | The fitted effect, how it was chosen, and what it assumes. |

(effectmodel-predict)=
## `EffectModel.predict`

*Method* of `EffectModel`

```python
EffectModel.predict(
    newdata,
    interval=False,
    level=0.95,
    output='auto',
    extrapolation=None
)
```

`tau(x)` for new rows.

The return type is set by `output` and `interval`:

* `interval=False, output="auto"` (the default) returns a NumPy array
  of the estimates, as earlier versions did.
* `interval=False` with `output="numpy"` returns the same array.
* `interval=False` with `output="polars"`, `"pandas"` or
  `"records"` returns a frame, or a list of dicts, with one column,
  `estimate`.
* `interval=True` returns a frame by default, with `estimate`,
  `se`, `lower`, `upper` (the pointwise interval at `level`),
  `outside_overlap`, a boolean flag that is true where the estimated
  propensity lies outside `overlap_bounds`, and `outside_range`, true
  where a modifier lies outside the training range of its spline. The
  interval needs `terms="smooth"`. With `output="numpy"` it returns a
  dict that maps each of those names to a NumPy array; the
  `outside_overlap` entry is None when the propensity is not available.

Every frame output has an `estimate` column, so code that reads
`pred["estimate"]` works with `interval=True` and with
`output="polars"`. Indexing the array of the default return by name
does not work; use `output="polars"` for named columns.

How the interval is built follows `interval_method` of
`effect_model`. `"refit"` uses the second stage's covariance
given the nuisances. `"repeats"` takes the estimate and standard error
at each row from every refit over a different split seed, and combines
them by the median rule of Chernozhukov et al. (2018): the estimate is
the median over refits, and the variance is the median of
`se^2 + (estimate_r - estimate)^2`. With `"repeats"`, the array
returned without `interval` is the same median. `"bootstrap"`
reports the full-sample estimate with the percentile interval of the
refits on resampled rows. For `"repeats"` and `"bootstrap"`,
`outside_range` is the flag of the full-sample fit.

`extrapolation` is passed to the fitted `tau` model, which applies
it to the modifiers that have a spline basis (see
`mboostrs.selection.GamRefit.predict`). `"clip"`, the default
for `terms="smooth"`, holds each modifier at its central range edge,
the 1st or 99th percentile of the fitting rows, so the effect is flat
beyond the range the spline was fitted on. The default for the boosted
candidates is `"warn"`, which predicts and warns once. `"error"`
refuses; `"allow"` is silent and continues the spline's edge slope.

The modifier columns are required. The covariates of the propensity
model are required only to check overlap; without them the effects are
returned, with no flag and a warning.

A row outside the bounds still gets an effect; its estimate is an
extrapolation. The call warns once with the number of such rows.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`
- `NotImplementedError`

Source: [`python/mboostrs/causal.py` lines 3358-3476](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L3358-L3476)

(effectmodel-ate_plugin)=
## `EffectModel.ate_plugin`

*Property* of `EffectModel`

```python
EffectModel.ate_plugin  # -> float
```

The unweighted mean of the fitted `tau(x)` over the fitting rows.

Each row counts once; rows are not weighted by overlap (the R-loss
weights `e(x)(1 - e(x))` are not applied). The value is an estimate
of the population average effect only as far as the fitted `tau` is
accurate. It has no standard error. Use `aipw` for an average
effect with an interval.

Source: [`python/mboostrs/causal.py` lines 3478-3488](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L3478-L3488)

(effectmodel-overlap_report)=
## `EffectModel.overlap_report`

*Method* of `EffectModel`

```python
EffectModel.overlap_report(bounds=None, data=None)
```

The overlap summary of the fitting rows; see `overlap_report`.

Source: [`python/mboostrs/causal.py` lines 3490-3493](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L3490-L3493)

(effectmodel-to_dict)=
## `EffectModel.to_dict`

*Method* of `EffectModel`

```python
EffectModel.to_dict(*, include_arrays=False)
```

A JSON-safe record; per-row arrays only on request.

Source: [`python/mboostrs/causal.py` lines 3495-3501](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L3495-L3501)

(effectmodel-describe)=
## `EffectModel.describe`

*Method* of `EffectModel`

```python
EffectModel.describe() -> str
```

The fitted effect, how it was chosen, and what it assumes.

Source: [`python/mboostrs/causal.py` lines 3507-3567](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L3507-L3567)
