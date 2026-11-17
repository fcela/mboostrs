# `Deselection`

Defined in `mboostrs.selection`; exported as `mboostrs.Deselection`.


*Class*

```python
Deselection(
    kept: tuple,
    dropped: tuple,
    shares: dict,
    threshold: float,
    formula: str | dict,
    model: object
) -> None
```

The terms `deselect` kept and dropped, and the refitted model.

Source: [`python/mboostrs/selection.py` lines 47-87](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/selection.py#L47-L87)

## Members

| Member | Kind | Summary |
|---|---|---|
| `variables` | Property | The data columns the kept terms read, in formula order; for a |
| `to_dict` | Method | A JSON-safe record (the model itself is not exported). |

(deselection-variables)=
## `Deselection.variables`

*Property* of `Deselection`

```python
Deselection.variables
```

The data columns the kept terms read, in formula order; for a
distributional model, `{parameter: columns}`.

Source: [`python/mboostrs/selection.py` lines 67-76](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/selection.py#L67-L76)

(deselection-to_dict)=
## `Deselection.to_dict`

*Method* of `Deselection`

```python
Deselection.to_dict(*, include_arrays=False)
```

A JSON-safe record (the model itself is not exported).

Source: [`python/mboostrs/selection.py` lines 78-87](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/selection.py#L78-L87)
