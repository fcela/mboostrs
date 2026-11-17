# `PathRow`

Defined in `mboostrs.report`; exported as `mboostrs.PathRow`.


*Class*

```python
PathRow(
    iteration: int,
    cycle: int | None,
    parameter_index: int | None,
    parameter: str | None,
    learner_index: int,
    learner: str,
    risk: float | None,
    risk_change: float | None,
    coefficient_norm: float,
    scaled_coefficient_norm: float,
    nuisance: tuple[float, ...] = ()
) -> None
```

One learner update in a fitted boosting path.

Mirrors the core's `mboostrs_core::report::PathRow` field-for-field, and
adds the two resolved names the core row cannot carry (it holds indices,
not the name tables) plus `scaled_coefficient_norm`, which needs `nu` from
the control object. `parameter_index` and `parameter` are deliberately
separate fields: the index is the schema, the name is the enrichment.
R's `path_frame()` emits the same columns in the same order.

Source: [`python/mboostrs/report.py` lines 14-46](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/report.py#L14-L46)

## Members

| Member | Kind | Summary |
|---|---|---|
| `to_dict` | Method | Return a row mapping suitable for JSON or a data frame. |

(pathrow-to_dict)=
## `PathRow.to_dict`

*Method* of `PathRow`

```python
PathRow.to_dict(*, include_arrays: bool = False) -> dict[str, Any]
```

Return a row mapping suitable for JSON or a data frame.

`include_arrays` is accepted for the uniform export signature; a
path row holds no arrays.

Source: [`python/mboostrs/report.py` lines 38-46](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/report.py#L38-L46)
