# `ElasticityResult`

Defined in `mboostrs.effects`; exported as `mboostrs.ElasticityResult`.


*Class*

```python
ElasticityResult(
    estimate: float,
    individual: numpy.ndarray,
    lower_value: numpy.ndarray,
    upper_value: numpy.ndarray,
    lower_prediction: numpy.ndarray,
    upper_prediction: numpy.ndarray,
    weights: numpy.ndarray,
    observation_ids: numpy.ndarray,
    variable: str,
    kind: str,
    parameter: str | None,
    population: str,
    log_step: float,
    support: dict,
    uncertainty: str = 'none',
    causal: bool = False
) -> None
```

Average and row-specific finite log-response derivatives.

Source: [`python/mboostrs/effects.py` lines 710-790](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/effects.py#L710-L790)

## Members

| Member | Kind | Summary |
|---|---|---|
| `to_frame` | Method | One row per finite elasticity or semi-elasticity. |
| `to_dict` | Method | A JSON-safe record; the per-row evidence only on request. |
| `describe` | Method | State the logarithmic derivative, averaging population, and limits. |

(elasticityresult-to_frame)=
## `ElasticityResult.to_frame`

*Method* of `ElasticityResult`

```python
ElasticityResult.to_frame(output='auto')
```

One row per finite elasticity or semi-elasticity.

Source: [`python/mboostrs/effects.py` lines 748-762](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/effects.py#L748-L762)

(elasticityresult-to_dict)=
## `ElasticityResult.to_dict`

*Method* of `ElasticityResult`

```python
ElasticityResult.to_dict(*, include_arrays=False)
```

A JSON-safe record; the per-row evidence only on request.

Source: [`python/mboostrs/effects.py` lines 770-777](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/effects.py#L770-L777)

(elasticityresult-describe)=
## `ElasticityResult.describe`

*Method* of `ElasticityResult`

```python
ElasticityResult.describe() -> str
```

State the logarithmic derivative, averaging population, and limits.

Source: [`python/mboostrs/effects.py` lines 779-790](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/effects.py#L779-L790)
