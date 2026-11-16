# `OverlapReport`

Defined in `mboostrs.causal`; exported as `mboostrs.OverlapReport`.


*Class*

```python
OverlapReport(
    bounds: tuple,
    n: int,
    n_below: int,
    n_above: int,
    regions: tuple,
    estimand: str | None,
    estimate: float | None,
    standard_error: float | None,
    estimate_inside: float | None,
    standard_error_inside: float | None,
    n_inside: int
) -> None
```

Rows whose estimated propensity lies outside `bounds`, where they
are, and the estimate among the rows inside the bounds.

`regions` has one entry per covariate passed with `data`. Numeric
entries give the `region` (span of values) where most rows are outside,
with the share outside inside and outside it; categorical entries list
each level with its share outside. `estimate` and `standard_error` are
the full-population estimate of a result (`None` for a bare propensity
array), and `estimate_inside` the same estimand restricted to rows whose
propensity is inside `bounds`. The restricted standard error treats the
inside rows as an independent sample and does not account for the
cross-fitting or for choosing the bounds after seeing the propensities.

Source: [`python/mboostrs/causal.py` lines 3035-3102](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L3035-L3102)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`n_outside`](#overlapreport-n_outside) | Property |  |
| [`share_outside`](#overlapreport-share_outside) | Property |  |
| [`to_dict`](#overlapreport-to_dict) | Method | A JSON-safe record of the counts, regions and estimates. |
| [`describe`](#overlapreport-describe) | Method | The share outside the bounds, the flagged regions and both estimates. |

(overlapreport-n_outside)=
## `OverlapReport.n_outside`

*Property* of `OverlapReport`

```python
OverlapReport.n_outside  # -> int
```

*No description in the docstring.*

Source: [`python/mboostrs/causal.py` lines 3063-3065](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L3063-L3065)

(overlapreport-share_outside)=
## `OverlapReport.share_outside`

*Property* of `OverlapReport`

```python
OverlapReport.share_outside  # -> float
```

*No description in the docstring.*

Source: [`python/mboostrs/causal.py` lines 3067-3069](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L3067-L3069)

(overlapreport-to_dict)=
## `OverlapReport.to_dict`

*Method* of `OverlapReport`

```python
OverlapReport.to_dict(*, include_arrays=False)
```

A JSON-safe record of the counts, regions and estimates.

Source: [`python/mboostrs/causal.py` lines 3071-3076](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L3071-L3076)

(overlapreport-describe)=
## `OverlapReport.describe`

*Method* of `OverlapReport`

```python
OverlapReport.describe() -> str
```

The share outside the bounds, the flagged regions and both estimates.

Source: [`python/mboostrs/causal.py` lines 3083-3102](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L3083-L3102)
