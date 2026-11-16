# `ALE1DResult`

Defined in `mboostrs.effects`; exported as `mboostrs.ALE1DResult`.


*Class*

```python
ALE1DResult(
    variable: str,
    edges: numpy.ndarray,
    grid: numpy.ndarray,
    effect: numpy.ndarray,
    local_change: numpy.ndarray,
    counts: numpy.ndarray,
    weight_totals: numpy.ndarray,
    bin_index: numpy.ndarray,
    observation_effect: numpy.ndarray,
    weights: numpy.ndarray,
    observation_ids: numpy.ndarray,
    scale: str,
    parameter: str | None,
    population: str,
    support: dict
) -> None
```

A centered univariate accumulated-local-effect curve and its support.

Source: [`python/mboostrs/effects.py` lines 793-881](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/effects.py#L793-L881)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`to_frame`](#ale1dresult-to_frame) | Method | One row per local interval with its effect and empirical support. |
| [`to_dict`](#ale1dresult-to_dict) | Method | A JSON-safe record; per-bin curves always, per-row evidence on request. |
| [`observation_frame`](#ale1dresult-observation_frame) | Method | One row per standardization observation and its assigned ALE value. |
| [`describe`](#ale1dresult-describe) | Method | State scale, population, centering and the sparsest local interval. |

(ale1dresult-to_frame)=
## `ALE1DResult.to_frame`

*Method* of `ALE1DResult`

```python
ALE1DResult.to_frame(output='auto')
```

One row per local interval with its effect and empirical support.

Source: [`python/mboostrs/effects.py` lines 835-845](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/effects.py#L835-L845)

(ale1dresult-to_dict)=
## `ALE1DResult.to_dict`

*Method* of `ALE1DResult`

```python
ALE1DResult.to_dict(*, include_arrays=False)
```

A JSON-safe record; per-bin curves always, per-row evidence on request.

Source: [`python/mboostrs/effects.py` lines 853-859](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/effects.py#L853-L859)

(ale1dresult-observation_frame)=
## `ALE1DResult.observation_frame`

*Method* of `ALE1DResult`

```python
ALE1DResult.observation_frame(output='auto')
```

One row per standardization observation and its assigned ALE value.

Source: [`python/mboostrs/effects.py` lines 861-870](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/effects.py#L861-L870)

(ale1dresult-describe)=
## `ALE1DResult.describe`

*Method* of `ALE1DResult`

```python
ALE1DResult.describe() -> str
```

State scale, population, centering and the sparsest local interval.

Source: [`python/mboostrs/effects.py` lines 872-881](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/effects.py#L872-L881)
