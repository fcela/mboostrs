# `ContrastResult`

Defined in `mboostrs.effects`; exported as `mboostrs.ContrastResult`.


*Class*

```python
ContrastResult(
    estimate: float,
    individual: numpy.ndarray,
    reference_prediction: numpy.ndarray,
    alternative_prediction: numpy.ndarray,
    weights: numpy.ndarray,
    observation_ids: numpy.ndarray,
    variable: str,
    reference: object,
    alternative: object,
    scale: str,
    parameter: str | None,
    population: str,
    support: dict,
    uncertainty: str = 'none',
    causal: bool = False,
    dependence: dict | None = None
) -> None
```

A standardized full-model prediction contrast and its row contributions.

Source: [`python/mboostrs/effects.py` lines 524-611](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/effects.py#L524-L611)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`interval`](#contrastresult-interval) | Property | ``(lower, upper)`` of the cluster-bootstrap interval, or None. |
| [`to_frame`](#contrastresult-to_frame) | Method | One row per standardized prediction contrast. |
| [`to_dict`](#contrastresult-to_dict) | Method | A JSON-safe record; the per-row evidence only on request. |
| [`describe`](#contrastresult-describe) | Method | State the estimand, scale, averaging population and uncertainty limit. |

(contrastresult-interval)=
## `ContrastResult.interval`

*Property* of `ContrastResult`

```python
ContrastResult.interval
```

`(lower, upper)` of the cluster-bootstrap interval, or None.

Source: [`python/mboostrs/effects.py` lines 563-568](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/effects.py#L563-L568)

(contrastresult-to_frame)=
## `ContrastResult.to_frame`

*Method* of `ContrastResult`

```python
ContrastResult.to_frame(output='auto')
```

One row per standardized prediction contrast.

Source: [`python/mboostrs/effects.py` lines 570-581](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/effects.py#L570-L581)

(contrastresult-to_dict)=
## `ContrastResult.to_dict`

*Method* of `ContrastResult`

```python
ContrastResult.to_dict(*, include_arrays=False)
```

A JSON-safe record; the per-row evidence only on request.

Source: [`python/mboostrs/effects.py` lines 590-597](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/effects.py#L590-L597)

(contrastresult-describe)=
## `ContrastResult.describe`

*Method* of `ContrastResult`

```python
ContrastResult.describe() -> str
```

State the estimand, scale, averaging population and uncertainty limit.

Source: [`python/mboostrs/effects.py` lines 599-611](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/effects.py#L599-L611)
