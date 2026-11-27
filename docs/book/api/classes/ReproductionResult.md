# `ReproductionResult`

Defined in `mboostrs._procedure._reproduce`; exported as `mboostrs.ReproductionResult`.


*Class*

```python
ReproductionResult(
    fit: mboostrs._procedure._fit.Fit,
    report: str,
    matched: bool,
    differences: tuple[str, ...],
    prediction_sha256: str,
    original_trail: dict[str, typing.Any],
    reproduced_trail: dict[str, typing.Any]
) -> None
```

Outcome of replaying a saved `DecisionTrail` on a dataset.

Source: [`python/mboostrs/_procedure/_reproduce.py` lines 77-115](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_procedure/_reproduce.py#L77-L115)

## Members

| Member | Kind | Summary |
|---|---|---|
| `model` | Property |  |
| `trail` | Property |  |
| `predict` | Method |  |
| `to_dict` | Method |  |

(reproductionresult-model)=
## `ReproductionResult.model`

*Property* of `ReproductionResult`

```python
ReproductionResult.model  # -> Any
```

*No description in the docstring.*

Source: [`python/mboostrs/_procedure/_reproduce.py` lines 89-91](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_procedure/_reproduce.py#L89-L91)

(reproductionresult-trail)=
## `ReproductionResult.trail`

*Property* of `ReproductionResult`

```python
ReproductionResult.trail  # -> DecisionTrail
```

*No description in the docstring.*

Source: [`python/mboostrs/_procedure/_reproduce.py` lines 93-95](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_procedure/_reproduce.py#L93-L95)

(reproductionresult-predict)=
## `ReproductionResult.predict`

*Method* of `ReproductionResult`

```python
ReproductionResult.predict(newdata: Any, **kwargs: Any) -> Any
```

*No description in the docstring.*

Source: [`python/mboostrs/_procedure/_reproduce.py` lines 97-98](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_procedure/_reproduce.py#L97-L98)

(reproductionresult-to_dict)=
## `ReproductionResult.to_dict`

*Method* of `ReproductionResult`

```python
ReproductionResult.to_dict() -> dict[str, Any]
```

*No description in the docstring.*

Source: [`python/mboostrs/_procedure/_reproduce.py` lines 100-108](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_procedure/_reproduce.py#L100-L108)
