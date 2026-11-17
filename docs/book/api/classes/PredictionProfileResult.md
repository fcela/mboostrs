# `PredictionProfileResult`

Defined in `mboostrs.effects`; exported as `mboostrs.PredictionProfileResult`.


*Class*

```python
PredictionProfileResult(
    variable: str,
    grid: numpy.ndarray,
    average: numpy.ndarray,
    ice: numpy.ndarray | None,
    weights: numpy.ndarray,
    observation_ids: numpy.ndarray,
    scale: str,
    parameter: str | None,
    population: str,
    support_fraction_far: numpy.ndarray,
    support: dict
) -> None
```

A weighted partial-dependence curve with optional ICE predictions.

Source: [`python/mboostrs/effects.py` lines 884-983](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/effects.py#L884-L983)

## Members

| Member | Kind | Summary |
|---|---|---|
| `to_frame` | Method | One row per grid value with its average and support diagnostic. |
| `to_dict` | Method | A JSON-safe record; the curve always, ICE rows only on request. |
| `ice_frame` | Method | One row per observation-grid prediction when ICE was retained. |
| `describe` | Method | State scale, population, retained rows, and support diagnostic. |

(predictionprofileresult-to_frame)=
## `PredictionProfileResult.to_frame`

*Method* of `PredictionProfileResult`

```python
PredictionProfileResult.to_frame(output='auto')
```

One row per grid value with its average and support diagnostic.

Source: [`python/mboostrs/effects.py` lines 928-939](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/effects.py#L928-L939)

(predictionprofileresult-to_dict)=
## `PredictionProfileResult.to_dict`

*Method* of `PredictionProfileResult`

```python
PredictionProfileResult.to_dict(*, include_arrays=False)
```

A JSON-safe record; the curve always, ICE rows only on request.

Source: [`python/mboostrs/effects.py` lines 948-954](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/effects.py#L948-L954)

(predictionprofileresult-ice_frame)=
## `PredictionProfileResult.ice_frame`

*Method* of `PredictionProfileResult`

```python
PredictionProfileResult.ice_frame(output='auto')
```

One row per observation-grid prediction when ICE was retained.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/effects.py` lines 956-973](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/effects.py#L956-L973)

(predictionprofileresult-describe)=
## `PredictionProfileResult.describe`

*Method* of `PredictionProfileResult`

```python
PredictionProfileResult.describe() -> str
```

State scale, population, retained rows, and support diagnostic.

Source: [`python/mboostrs/effects.py` lines 975-983](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/effects.py#L975-L983)
