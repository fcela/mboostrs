# `MultiQuantileModel`

Defined in `mboostrs.quantile`; exported as `mboostrs.MultiQuantileModel`.


*Class*

```python
MultiQuantileModel(
    quantiles: tuple[float, ...],
    models: dict[float, mboostrs.gamboost.GAMBoostModel],
    optimal_mstop: dict[float, int],
    cv_risk: dict[float, numpy.ndarray],
    formula: str,
    response: str
) -> None
```

Simultaneously fitted conditional quantile models sharing one learner basis.

Source: [`python/mboostrs/quantile.py` lines 133-321](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/quantile.py#L133-L321)

## Members

| Member | Kind | Summary |
|---|---|---|
| `predict` | Method | Predict all quantile levels on ``newdata``, rearranging crossings by default. |
| `predict_mean` | Method | Estimate conditional mean via probability-weighted integration over rearranged quantiles. |
| `predict_interval` | Method | Predict central quantile intervals, optionally calibrated via CQR (Romano et al., 2019). |
| `crps` | Method | Mean pinball-integrated CRPS via trapezoidal rule across the fitted quantile grid. |
| `to_dict` | Method | Return a JSON-safe summary dictionary of the multi-quantile model. |
| `digest` | Method | Return a concise, deterministic summary card of this MultiQuantileModel. |

(multiquantilemodel-predict)=
## `MultiQuantileModel.predict`

*Method* of `MultiQuantileModel`

```python
MultiQuantileModel.predict(
    newdata: Any = None,
    rearrange: bool = True,
    unseen_levels: str = 'average',
    output: str = 'auto'
) -> mboostrs.rearrange.RearrangedQuantiles | numpy.ndarray
```

Predict all quantile levels on `newdata`, rearranging crossings by default.

Source: [`python/mboostrs/quantile.py` lines 170-186](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/quantile.py#L170-L186)

(multiquantilemodel-predict_mean)=
## `MultiQuantileModel.predict_mean`

*Method* of `MultiQuantileModel`

```python
MultiQuantileModel.predict_mean(
    newdata: Any = None,
    unseen_levels: str = 'average'
) -> ndarray
```

Estimate conditional mean via probability-weighted integration over rearranged quantiles.

Source: [`python/mboostrs/quantile.py` lines 188-199](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/quantile.py#L188-L199)

(multiquantilemodel-predict_interval)=
## `MultiQuantileModel.predict_interval`

*Method* of `MultiQuantileModel`

```python
MultiQuantileModel.predict_interval(
    newdata: Any,
    level: float = 0.9,
    calibration: Any = None,
    unseen_levels: str = 'average'
) -> dict[str, Any]
```

Predict central quantile intervals, optionally calibrated via CQR (Romano et al., 2019).

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/quantile.py` lines 201-243](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/quantile.py#L201-L243)

(multiquantilemodel-crps)=
## `MultiQuantileModel.crps`

*Method* of `MultiQuantileModel`

```python
MultiQuantileModel.crps(
    newdata: Any,
    y_true: Any = None,
    unseen_levels: str = 'average'
) -> float
```

Mean pinball-integrated CRPS via trapezoidal rule across the fitted quantile grid.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/quantile.py` lines 245-266](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/quantile.py#L245-L266)

(multiquantilemodel-to_dict)=
## `MultiQuantileModel.to_dict`

*Method* of `MultiQuantileModel`

```python
MultiQuantileModel.to_dict() -> dict[str, Any]
```

Return a JSON-safe summary dictionary of the multi-quantile model.

Source: [`python/mboostrs/quantile.py` lines 268-279](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/quantile.py#L268-L279)

(multiquantilemodel-digest)=
## `MultiQuantileModel.digest`

*Method* of `MultiQuantileModel`

```python
MultiQuantileModel.digest() -> str
```

Return a concise, deterministic summary card of this MultiQuantileModel.

Source: [`python/mboostrs/quantile.py` lines 288-321](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/quantile.py#L288-L321)
