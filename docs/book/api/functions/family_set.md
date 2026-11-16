# `family_set`

Defined in `mboostrs.model_selection`; exported as `mboostrs.family_set`.


*Function*

```python
family_set(support: str) -> list[str]
```

Return the candidate LSS family names for one response support.

`support` is a key of `FAMILY_SETS`: `"realline"` (the whole real
line), `"realplus"` (positive continuous), `"real0to1"` (proportions in
the open unit interval), `"counts"` (non-negative integers) or
`"zeroplus"` (non-negative continuous with a point mass at zero).

The result is a new list, so changing it does not change `FAMILY_SETS`.
Pass it as `families=` to `scan_lss_families`, or read it before
scanning to see which families a support covers.

**Raises**

- `ValueError`: If `support` is not one of the keys of `FAMILY_SETS`. The message
lists the valid supports.

Source: [`python/mboostrs/model_selection.py` lines 1074-1097](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/model_selection.py#L1074-L1097)
