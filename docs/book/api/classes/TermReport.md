# `TermReport`

Defined in `mboostrs.explore`; exported as `mboostrs.TermReport`.


*Class*

```python
TermReport(
    formula: str,
    response: str,
    baseline: float,
    score: str = 'family',
    fits: int = 0,
    contributions: list = <factory>,
    shapes: list = <factory>,
    levels: dict = <factory>,
    groups: tuple = (),
    candidates: int = 0
) -> None
```

Held-out contribution and alternative-shape scores for each term.

Source: [`python/mboostrs/explore.py` lines 2072-2119](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/explore.py#L2072-L2119)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`to_dict`](#termreport-to_dict) | Method | A JSON-safe record of what each term is worth, and in what shape. |

(termreport-to_dict)=
## `TermReport.to_dict`

*Method* of `TermReport`

```python
TermReport.to_dict(*, include_arrays=False)
```

A JSON-safe record of what each term is worth, and in what shape.

Source: [`python/mboostrs/explore.py` lines 2117-2119](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/explore.py#L2117-L2119)
