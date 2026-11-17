# `DistributionEvaluationResult`

Defined in `mboostrs.distribution_evaluation`; exported as `mboostrs.DistributionEvaluationResult`.


*Class*

```python
DistributionEvaluationResult(
    family: str,
    response: str,
    data_role: str,
    data_fingerprint: str,
    observed: numpy.ndarray,
    weights: numpy.ndarray,
    parameters: dict[str, numpy.ndarray],
    density_or_mass: numpy.ndarray,
    negative_log_score: numpy.ndarray,
    deviance: numpy.ndarray,
    pit: numpy.ndarray,
    crps: numpy.ndarray | None,
    metrics: dict,
    intervals: tuple[dict, ...],
    exceedances: tuple[dict, ...],
    assumptions: tuple[str, ...]
) -> None
```

Row-level distribution scores and aggregate predictive diagnostics.

Source: [`python/mboostrs/distribution_evaluation.py` lines 34-129](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/distribution_evaluation.py#L34-L129)

## Members

| Member | Kind | Summary |
|---|---|---|
| `to_dict` | Method | A JSON-safe record; the per-row score arrays and distribution |
| `to_frame` | Method | One row per evaluated outcome with parameters and scores. |
| `interval_frame` | Method | One row per requested central interval level. |
| `exceedance_frame` | Method | One row per requested strict exceedance threshold. |
| `describe` | Method | State the evaluated distribution, data role, scores, and scope. |

(distributionevaluationresult-to_dict)=
## `DistributionEvaluationResult.to_dict`

*Method* of `DistributionEvaluationResult`

```python
DistributionEvaluationResult.to_dict(*, include_arrays=False)
```

A JSON-safe record; the per-row score arrays and distribution
parameters only on request, the small metrics always.

Source: [`python/mboostrs/distribution_evaluation.py` lines 84-93](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/distribution_evaluation.py#L84-L93)

(distributionevaluationresult-to_frame)=
## `DistributionEvaluationResult.to_frame`

*Method* of `DistributionEvaluationResult`

```python
DistributionEvaluationResult.to_frame(output='auto')
```

One row per evaluated outcome with parameters and scores.

Source: [`python/mboostrs/distribution_evaluation.py` lines 95-112](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/distribution_evaluation.py#L95-L112)

(distributionevaluationresult-interval_frame)=
## `DistributionEvaluationResult.interval_frame`

*Method* of `DistributionEvaluationResult`

```python
DistributionEvaluationResult.interval_frame(output='auto')
```

One row per requested central interval level.

Source: [`python/mboostrs/distribution_evaluation.py` lines 114-116](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/distribution_evaluation.py#L114-L116)

(distributionevaluationresult-exceedance_frame)=
## `DistributionEvaluationResult.exceedance_frame`

*Method* of `DistributionEvaluationResult`

```python
DistributionEvaluationResult.exceedance_frame(output='auto')
```

One row per requested strict exceedance threshold.

Source: [`python/mboostrs/distribution_evaluation.py` lines 118-120](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/distribution_evaluation.py#L118-L120)

(distributionevaluationresult-describe)=
## `DistributionEvaluationResult.describe`

*Method* of `DistributionEvaluationResult`

```python
DistributionEvaluationResult.describe() -> str
```

State the evaluated distribution, data role, scores, and scope.

Source: [`python/mboostrs/distribution_evaluation.py` lines 122-129](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/distribution_evaluation.py#L122-L129)
