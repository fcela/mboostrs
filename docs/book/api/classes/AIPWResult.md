# `AIPWResult`

Defined in `mboostrs.causal`; exported as `mboostrs.AIPWResult`.


*Class*

```python
AIPWResult(
    estimate: float,
    standard_error: float | None,
    interval: tuple[float, float] | None,
    level: float,
    spec: mboostrs.causal.CausalSpec,
    treatment: numpy.ndarray,
    outcome: numpy.ndarray,
    propensity: numpy.ndarray,
    mu0: numpy.ndarray,
    mu1: numpy.ndarray,
    scores: numpy.ndarray,
    influence: numpy.ndarray,
    fold_id: numpy.ndarray,
    diagnostics: dict,
    nuisance: dict,
    fold_ids: numpy.ndarray | None = None,
    observed: numpy.ndarray | None = None,
    response_propensity: numpy.ndarray | None = None,
    weights: numpy.ndarray | None = None,
    ato_companion: dict | None = None,
    _propensity_model: object = None,
    _data: object = None,
    _groups: object = None,
    _ordered: bool = False
) -> None
```

Cross-fitted AIPW estimate with reconstructible row contributions.

Source: [`python/mboostrs/causal.py` lines 453-917](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L453-L917)

## Members

| Member | Kind | Summary |
|---|---|---|
| `to_frame` | Method | One row per observation, sufficient to reconstruct the estimate. |
| `to_dict` | Method | A JSON-safe record; the per-row contribution tracks only on request. |
| `predict_propensity` | Method | Estimated P(treatment = treated \| covariates) of each row of ``newdata``. |
| `subgroup_effect` | Method | Average treatment effect within subgroups, with a standard error and interval. |
| `effect_surface` | Method | Doubly robust second stage: regress the augmented scores on covariates. |
| `balance_frame` | Method | Standardized mean differences before and after treatment weighting. |
| `overlap_frame` | Method | Row-level propensity support and estimand-specific analysis weights. |
| `target` | Property | The population the estimate describes: ``"all"`` (every row, or |
| `describe` | Method | Estimate, target, interval, overlap and the assumptions it depends on. |
| `digest` | Method | Return a concise, deterministic summary card of this AIPW estimate. |

(aipwresult-to_frame)=
## `AIPWResult.to_frame`

*Method* of `AIPWResult`

```python
AIPWResult.to_frame(output='auto')
```

One row per observation, sufficient to reconstruct the estimate.

Source: [`python/mboostrs/causal.py` lines 529-542](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L529-L542)

(aipwresult-to_dict)=
## `AIPWResult.to_dict`

*Method* of `AIPWResult`

```python
AIPWResult.to_dict(*, include_arrays=False)
```

A JSON-safe record; the per-row contribution tracks only on request.

The scalar estimate, standard error, interval, level and specification
are always present; the per-row propensity, means, scores, influence
and fold ids are gated behind `include_arrays`.

Source: [`python/mboostrs/causal.py` lines 544-558](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L544-L558)

(aipwresult-predict_propensity)=
## `AIPWResult.predict_propensity`

*Method* of `AIPWResult`

```python
AIPWResult.predict_propensity(newdata)
```

Estimated P(treatment = treated | covariates) of each row of `newdata`.

The model is the propensity model that `aipw` fitted on all rows,
with the same formula, smooth flag, step count and tuning as its
cross-fitted propensity fits. The values are clipped to the
`propensity_clip` of the fit, and the result is a NumPy array with one
value per row, in row order. `newdata` needs every covariate of the
specification; a missing covariate raises `ValueError` naming it. Use
`propensity` for the training rows, where the estimate was computed;
use this method for new rows or to check the model on any rows.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/causal.py` lines 568-597](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L568-L597)

(aipwresult-subgroup_effect)=
## `AIPWResult.subgroup_effect`

*Method* of `AIPWResult`

```python
AIPWResult.subgroup_effect(by, *, level=None, output='auto')
```

Average treatment effect within subgroups, with a standard error and interval.

Each subgroup's effect is the weighted mean of the augmented (AIPW)
scores of its rows, `scores`. Its standard error is computed
as `aipw` computes the standard error of the overall estimate,
with the scores of the subgroup as influence values: frequency weights
give `sum(w (score - mean)^2) / (W (W - 1))`, and a grouped fold
scheme gives the cluster-robust sandwich over the clusters present in
the subgroup, with a Student-t interval on `clusters - 1` degrees of
freedom. A subgroup that holds every row therefore reproduces
`estimate` and `standard_error`. The standard error is None
for a subgroup with total weight of one or less or fewer than two
clusters, and for an ordered fold scheme.

**Parameters**

- `by` (str or array-like): A boolean mask over the training rows (one subgroup, the rows where
  it is true), the name of a column of the training data (one subgroup
  per distinct value), or an array of labels with one entry per
  training row. Missing values in the labels are refused.
- `level` (float, optional): Interval level; default is `level`, the level `aipw` used.
- `output` ({"auto", "pandas", "polars", "records"}): Result format.

**Returns**

One row per subgroup with `level` (the subgroup value; `True` for a
mask), `n` (rows), `weight` (total observation weight), `estimate`,
`standard_error`, `lower` and `upper`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

**Notes**

The subgroup effects use nuisance models fitted on all rows. The
variance does not account for choosing subgroups after looking at the
data; a subgroup effect is as credible as the subgroup is prespecified.
The method needs an ATE result (`target='all'`). With
`repeat_aggregation='median'` the overall estimate is the median over
repeats, and a subgroup holding every row gives the mean of the
averaged scores, which differs from it.

Source: [`python/mboostrs/causal.py` lines 606-699](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L606-L699)

(aipwresult-effect_surface)=
## `AIPWResult.effect_surface`

*Method* of `AIPWResult`

```python
AIPWResult.effect_surface(formula, *, mstop=200, nu=0.1, **gamboost_args)
```

Doubly robust second stage: regress the augmented scores on covariates.

Fits `gamboost` with the augmented scores `scores` as the
response and `formula` as the right-hand side, on the training data
and the observation weights of the `aipw` fit. The expected score of
a row given its covariates is the conditional average treatment effect,
so `predict` of the returned model estimates the effect as a function
of the covariates (the DR-learner of Kennedy, 2023). The scores are
stored under a generated column name that differs from every column of
the data.

**Parameters**

- `formula` (str): The right-hand side, for example `"bbs(age, df=4) + bols(sex)"`;
  a leading `"~"` is accepted. Every variable must be a column of
  the training data.
- `mstop, nu` (int, float): Boosting steps and step length of the second stage. Choose `mstop`
  by cross-validation (`mb.cvrisk`) for a tuned surface.
- `**gamboost_args`: Passed to `mboostrs.gamboost` (`family` defaults to Gaussian).

**Returns**

The fitted `GAMBoostModel`; `model.predict(newdata)` gives the
estimated conditional effect.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

**Notes**

The scores are noisy pseudo-outcomes: their mean given the covariates
is the effect, but each score has a variance far above that of an
outcome, so the surface is estimated with wide uncertainty. Intervals
or standard errors from the second-stage model treat the scores as
fixed observed data and ignore the error of the nuisance models that
produced them and of the choice of `mstop`; they understate the
uncertainty. The surface describes the effect only through the
covariates in `formula`. The method needs an ATE result
(`target='all'`).

Source: [`python/mboostrs/causal.py` lines 701-767](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L701-L767)

(aipwresult-balance_frame)=
## `AIPWResult.balance_frame`

*Method* of `AIPWResult`

```python
AIPWResult.balance_frame(output='auto')
```

Standardized mean differences before and after treatment weighting.

Source: [`python/mboostrs/causal.py` lines 769-771](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L769-L771)

(aipwresult-overlap_frame)=
## `AIPWResult.overlap_frame`

*Method* of `AIPWResult`

```python
AIPWResult.overlap_frame(output='auto')
```

Row-level propensity support and estimand-specific analysis weights.

Plot or summarize this table by treatment arm. A propensity histogram
can show whether the two arms occupy the same covariate-support region;
the weight column shows which observations dominate the requested ATE
or ATT. Propensities used for estimation include the explicitly
recorded clipping policy, when one was requested.

Source: [`python/mboostrs/causal.py` lines 773-802](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L773-L802)

(aipwresult-target)=
## `AIPWResult.target`

*Property* of `AIPWResult`

```python
AIPWResult.target
```

The population the estimate describes: `"all"` (every row, or
every treated row for the ATT) or `"overlap"` (the overlap-weighted
ATE). Set by `aipw(target=...)`.

Source: [`python/mboostrs/causal.py` lines 804-809](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L804-L809)

(aipwresult-describe)=
## `AIPWResult.describe`

*Method* of `AIPWResult`

```python
AIPWResult.describe() -> str
```

Estimate, target, interval, overlap and the assumptions it depends on.

Source: [`python/mboostrs/causal.py` lines 837-857](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L837-L857)

(aipwresult-digest)=
## `AIPWResult.digest`

*Method* of `AIPWResult`

```python
AIPWResult.digest() -> str
```

Return a concise, deterministic summary card of this AIPW estimate.

Source: [`python/mboostrs/causal.py` lines 889-917](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L889-L917)
