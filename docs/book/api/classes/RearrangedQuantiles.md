# `RearrangedQuantiles`

Defined in `mboostrs.rearrange`; exported as `mboostrs.RearrangedQuantiles`.


*Class*

```python
RearrangedQuantiles(
    kind: str,
    levels: tuple,
    predictions: Any,
    unsorted: Any,
    crossing_rows: int,
    crossing_share: float,
    pair_crossings: Any,
    max_crossing: float,
    n: int
) -> None
```

Sorted predictions at several levels and the crossings found before sorting.

**Attributes**

- `kind` ({"quantile", "expectile"}): The family kind of the fitted models.
- `levels` (tuple of float): The `tau` of each column, increasing.
- `predictions` (DataFrame): Predictions after sorting, one column per level, named `q10`,
  `q50`, ... for quantiles and `e10`, ... for expectiles. Every
  row is non-decreasing from left to right.
- `unsorted` (DataFrame): The predictions of each model before sorting, same columns.
- `crossing_rows` (int): Number of rows where some level was predicted above the next higher
  level.
- `crossing_share` (float): `crossing_rows` divided by the number of rows.
- `pair_crossings` (DataFrame): One row per pair of adjacent levels: `lower` and `upper` (the
  two `tau` values), `n_crossing` (rows where the lower level was
  predicted strictly above the upper level) and `share` (that count
  over the number of rows).
- `max_crossing` (float): Largest amount by which a lower level was predicted above the next
  higher level, in the units of the response; 0 when nothing crossed.
- `n` (int): Number of rows.

Source: [`python/mboostrs/rearrange.py` lines 63-124](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/rearrange.py#L63-L124)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`to_dict`](#rearrangedquantiles-to_dict) | Method | A plain-Python record of the crossing report. |

(rearrangedquantiles-to_dict)=
## `RearrangedQuantiles.to_dict`

*Method* of `RearrangedQuantiles`

```python
RearrangedQuantiles.to_dict(*, include_arrays=False) -> dict
```

A plain-Python record of the crossing report.

The scalar summary and the per-pair crossing counts are always
included; the sorted and unsorted predictions (one list per level) only
with `include_arrays=True`.

Source: [`python/mboostrs/rearrange.py` lines 106-124](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/rearrange.py#L106-L124)
