# `Probing`

Defined in `mboostrs.selection`; exported as `mboostrs.Probing`.


*Class*

```python
Probing(
    selected: tuple,
    stop: int | None,
    first_shadow: str | None,
    formula: str,
    model: object
) -> None
```

Result of `probe`: selected terms, the first shadow, and its entry iteration.

Source: [`python/mboostrs/selection.py` lines 90-115](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/selection.py#L90-L115)

## Members

| Member | Kind | Summary |
|---|---|---|
| `variables` | Property | The data columns the selected terms read, in formula order. |
| `to_dict` | Method | A JSON-safe record (the model itself is not exported). |

(probing-variables)=
## `Probing.variables`

*Property* of `Probing`

```python
Probing.variables
```

The data columns the selected terms read, in formula order.

Source: [`python/mboostrs/selection.py` lines 106-109](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/selection.py#L106-L109)

(probing-to_dict)=
## `Probing.to_dict`

*Method* of `Probing`

```python
Probing.to_dict(*, include_arrays=False)
```

A JSON-safe record (the model itself is not exported).

Source: [`python/mboostrs/selection.py` lines 111-115](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/selection.py#L111-L115)
