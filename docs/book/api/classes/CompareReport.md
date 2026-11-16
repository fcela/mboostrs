# `CompareReport`

Defined in `mboostrs.compare`; exported as `mboostrs.CompareReport`.


*Class*

```python
CompareReport(spec: mboostrs.compare.CompareSpec)
```

Structured comparison report between Python and R.

**Attributes**

- `spec` (dict): The model specification used.
- `r_available` (bool): Whether R + rpy2 were available to run the comparison.
- `py_result` (dict): Python-side results (fitted, risk, coef, selected).
- `r_result` (dict or None): R-side results, or None if R unavailable.
- `diffs` (dict): Numerical diffs for each compared quantity.
- `max_abs_diff` (dict): Maximum absolute difference for each quantity.
- `status` (str): "ok", "r_unavailable", "error", or "mismatch".
- `error` (str or None): Error message if status is "error".

Source: [`python/mboostrs/compare.py` lines 110-167](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/compare.py#L110-L167)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`to_dict`](#comparereport-to_dict) | Method | The report as a mapping. ``diffs`` holds per-quantity summary |
| [`to_json`](#comparereport-to_json) | Method |  |
| [`summary`](#comparereport-summary) | Method | One-line summary string. |

(comparereport-to_dict)=
## `CompareReport.to_dict`

*Method* of `CompareReport`

```python
CompareReport.to_dict(*, include_arrays: bool = False) -> dict[str, Any]
```

The report as a mapping. `diffs` holds per-quantity summary
numbers, not arrays, so `include_arrays` changes nothing here.

Source: [`python/mboostrs/compare.py` lines 143-153](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/compare.py#L143-L153)

(comparereport-to_json)=
## `CompareReport.to_json`

*Method* of `CompareReport`

```python
CompareReport.to_json(indent: int = 2) -> str
```

*No description in the docstring.*

Source: [`python/mboostrs/compare.py` lines 155-156](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/compare.py#L155-L156)

(comparereport-summary)=
## `CompareReport.summary`

*Method* of `CompareReport`

```python
CompareReport.summary() -> str
```

One-line summary string.

Source: [`python/mboostrs/compare.py` lines 158-167](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/compare.py#L158-L167)
