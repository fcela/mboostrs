# `MarginalEffectResult`

Defined in `mboostrs.effects`; exported as `mboostrs.MarginalEffectResult`.


*Class*

```python
MarginalEffectResult(
    estimate: float,
    individual: numpy.ndarray,
    lower_value: numpy.ndarray,
    upper_value: numpy.ndarray,
    lower_prediction: numpy.ndarray,
    upper_prediction: numpy.ndarray,
    weights: numpy.ndarray,
    observation_ids: numpy.ndarray,
    variable: str,
    scale: str,
    parameter: str | None,
    population: str,
    step: float,
    support: dict,
    uncertainty: str = 'none',
    causal: bool = False,
    dependence: dict | None = None
) -> None
```

Average and row-specific finite derivatives of the full prediction.

Source: [`python/mboostrs/effects.py` lines 614-707](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/effects.py#L614-L707)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`interval`](#marginaleffectresult-interval) | Property | ``(lower, upper)`` of the cluster-bootstrap interval, or None. |
| [`to_frame`](#marginaleffectresult-to_frame) | Method | One row per local finite derivative. |
| [`to_dict`](#marginaleffectresult-to_dict) | Method | A JSON-safe record; the per-row evidence only on request. |
| [`describe`](#marginaleffectresult-describe) | Method | State the derivative target, scale, step, population, and limits. |

(marginaleffectresult-interval)=
## `MarginalEffectResult.interval`

*Property* of `MarginalEffectResult`

```python
MarginalEffectResult.interval
```

`(lower, upper)` of the cluster-bootstrap interval, or None.

Source: [`python/mboostrs/effects.py` lines 657-662](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/effects.py#L657-L662)

(marginaleffectresult-to_frame)=
## `MarginalEffectResult.to_frame`

*Method* of `MarginalEffectResult`

```python
MarginalEffectResult.to_frame(output='auto')
```

One row per local finite derivative.

Source: [`python/mboostrs/effects.py` lines 664-677](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/effects.py#L664-L677)

(marginaleffectresult-to_dict)=
## `MarginalEffectResult.to_dict`

*Method* of `MarginalEffectResult`

```python
MarginalEffectResult.to_dict(*, include_arrays=False)
```

A JSON-safe record; the per-row evidence only on request.

Source: [`python/mboostrs/effects.py` lines 686-693](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/effects.py#L686-L693)

(marginaleffectresult-describe)=
## `MarginalEffectResult.describe`

*Method* of `MarginalEffectResult`

```python
MarginalEffectResult.describe() -> str
```

State the derivative target, scale, step, population, and limits.

Source: [`python/mboostrs/effects.py` lines 695-707](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/effects.py#L695-L707)
