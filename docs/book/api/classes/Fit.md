# `Fit`

Defined in `mboostrs._procedure._fit`; exported as `mboostrs.Fit`.


*Class*

```python
Fit(
    model: Any,
    trail: mboostrs._procedure._trail.DecisionTrail,
    stability: Any = None,
    bootstrap: Any = None,
    conformal: Any = None,
    time: str | None = None
)
```

A fitted model wrapped with the trail that produced it.

`model` is the underlying fitted model (a `GAMBoostModel` or
`LSSModel`), `trail` is the `mboostrs._procedure._trail.DecisionTrail`
explaining how it was chosen, and optional uncertainty summaries are
attached to `stability`, `bootstrap`, or `conformal` depending on
`uncertainty=`. `predict` delegates to the model and `predict_interval`
computes conformal prediction intervals (defaulting to `"adaptive"` when
fitted with `time=`).

Source: [`python/mboostrs/_procedure/_fit.py` lines 43-214](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_procedure/_fit.py#L43-L214)

## Members

| Member | Kind | Summary |
|---|---|---|
| `predict` | Method | Predict with the wrapped model. See ``model.predict``. |
| `predict_interval` | Method | Compute conformal prediction intervals on ``newdata``. |
| `to_dict` | Method | The wrapping record: the decision trail as a JSON-safe dict. |
| `reproduce` | Method | Replay this fit's decision trail on ``data`` and verify agreement. |
| `digest` | Method | Return a concise, deterministic summary card of this Fit. |

(fit-predict)=
## `Fit.predict`

*Method* of `Fit`

```python
Fit.predict(newdata: Any, **kwargs: Any) -> Any
```

Predict with the wrapped model. See `model.predict`.

Defaults `extrapolation="clip"` so spline terms are held flat outside
their training boundary knots on automated fits (linear `bols` terms
still extrapolate linearly). For distributional (`LSSModel`) fits,
defaults `type="mean"` when neither `type` nor `parameter` is
specified so `fit.predict(newdata)` returns 1D response-scale mean
predictions uniformly across families. Pass an explicit `type=`,
`parameter=`, or `extrapolation=` to override.

Source: [`python/mboostrs/_procedure/_fit.py` lines 72-92](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_procedure/_fit.py#L72-L92)

(fit-predict_interval)=
## `Fit.predict_interval`

*Method* of `Fit`

```python
Fit.predict_interval(
    newdata: Any = None,
    level: float = 0.9,
    method: str | None = None,
    **kwargs: Any
) -> Any
```

Compute conformal prediction intervals on `newdata`.

Defaults `method="adaptive"` when `fit(..., time=...)` was used
and `method="split"` otherwise.

Source: [`python/mboostrs/_procedure/_fit.py` lines 94-124](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_procedure/_fit.py#L94-L124)

(fit-to_dict)=
## `Fit.to_dict`

*Method* of `Fit`

```python
Fit.to_dict(*, include_arrays: bool = False) -> dict[str, Any]
```

The wrapping record: the decision trail as a JSON-safe dict.

Source: [`python/mboostrs/_procedure/_fit.py` lines 126-129](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_procedure/_fit.py#L126-L129)

(fit-reproduce)=
## `Fit.reproduce`

*Method* of `Fit`

```python
Fit.reproduce(data: Any, **kwargs: Any) -> Any
```

Replay this fit's decision trail on `data` and verify agreement.

Source: [`python/mboostrs/_procedure/_fit.py` lines 131-135](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_procedure/_fit.py#L131-L135)

(fit-digest)=
## `Fit.digest`

*Method* of `Fit`

```python
Fit.digest() -> str
```

Return a concise, deterministic summary card of this Fit.

Source: [`python/mboostrs/_procedure/_fit.py` lines 143-214](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_procedure/_fit.py#L143-L214)
