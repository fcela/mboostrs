# `Fit`

Defined in `mboostrs._procedure._fit`; exported as `mboostrs.Fit`.


*Class*

```python
Fit(model: Any, trail: mboostrs._procedure._trail.DecisionTrail, stability: Any = None)
```

A fitted model wrapped with the trail that produced it.

`model` is the underlying fitted model (a `GAMBoostModel` or
`LSSModel`), `trail` is the `mboostrs._procedure._trail.DecisionTrail`
explaining how it was chosen, and `stability` is currently `None`
because selection-aware uncertainty is not yet supported.
`predict` delegates to the model, so a `Fit` can stand in wherever the
model itself was used.

Source: [`python/mboostrs/_procedure/_fit.py` lines 16-45](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_procedure/_fit.py#L16-L45)

## Members

| Member | Kind | Summary |
|---|---|---|
| `predict` | Method | Predict with the wrapped model. See ``model.predict``. |
| `to_dict` | Method | The wrapping record: the decision trail as a JSON-safe dict. |

(fit-predict)=
## `Fit.predict`

*Method* of `Fit`

```python
Fit.predict(newdata: Any, **kwargs: Any) -> Any
```

Predict with the wrapped model. See `model.predict`.

Source: [`python/mboostrs/_procedure/_fit.py` lines 32-34](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_procedure/_fit.py#L32-L34)

(fit-to_dict)=
## `Fit.to_dict`

*Method* of `Fit`

```python
Fit.to_dict(*, include_arrays: bool = False) -> dict[str, Any]
```

The wrapping record: the decision trail as a JSON-safe dict.

Source: [`python/mboostrs/_procedure/_fit.py` lines 36-39](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_procedure/_fit.py#L36-L39)
