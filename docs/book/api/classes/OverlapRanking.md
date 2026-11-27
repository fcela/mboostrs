# `OverlapRanking`

Defined in `mboostrs.causal`; exported as `mboostrs.OverlapRanking`.


*Class*

```python
OverlapRanking(
    chosen: tuple,
    outside: tuple,
    gain: float,
    gain_interval: tuple | None,
    level: float | None,
    share: float,
    bounds: tuple,
    shortfall: int,
    rows: tuple,
    tied_rows: int = 0,
    tie_break: str = 'row order'
) -> None
```

The top group of rows by estimated effect, ranked inside the overlap region.

`chosen` holds the positions in `newdata` of the top group, best first.
`outside` holds the positions whose propensity lies outside `bounds`;
they are listed here and are never in `chosen`. `gain` is the mean
estimated effect over the chosen rows. `gain_interval` is its interval at
`level`, computed from the covariance of the fitted effect model; it is
`None` when the effect has no coefficient covariance (a boosted
candidate, or a plain frame of estimates). `shortfall` counts the places
the top group could not fill because fewer rows lie inside the overlap
than `share` asks for. `rows` holds one record per row of `newdata`
with its estimate, interval, propensity, overlap flag, rank among the
overlap rows and whether it is chosen.

Source: [`python/mboostrs/causal.py` lines 4957-5020](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L4957-L5020)

## Members

| Member | Kind | Summary |
|---|---|---|
| `table` | Method | One row per row of ``newdata``; see the ``rows`` field. The |
| `to_dict` | Method | A JSON-safe record; the per-row records only on request. |
| `describe` | Method | The top group, its gain, and the rows left out for lack of overlap. |

(overlapranking-table)=
## `OverlapRanking.table`

*Method* of `OverlapRanking`

```python
OverlapRanking.table(output='auto')
```

One row per row of `newdata`; see the `rows` field. The
`tied_at_boundary` column marks the tied rows.

Source: [`python/mboostrs/causal.py` lines 4990-4993](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L4990-L4993)

(overlapranking-to_dict)=
## `OverlapRanking.to_dict`

*Method* of `OverlapRanking`

```python
OverlapRanking.to_dict(*, include_arrays=False)
```

A JSON-safe record; the per-row records only on request.

Source: [`python/mboostrs/causal.py` lines 4995-4999](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L4995-L4999)

(overlapranking-describe)=
## `OverlapRanking.describe`

*Method* of `OverlapRanking`

```python
OverlapRanking.describe() -> str
```

The top group, its gain, and the rows left out for lack of overlap.

Source: [`python/mboostrs/causal.py` lines 5001-5020](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L5001-L5020)
