# `InfluenceFinding`

Defined in `mboostrs.residual_checks`; exported as `mboostrs.InfluenceFinding`.


*Class*

```python
InfluenceFinding(
    check: str,
    flagged: bool,
    pattern: str,
    statistic: str,
    value: float,
    units: str,
    p_value: float | None,
    reference: str,
    false_alarm_rate: float,
    false_alarm_draws: int,
    assumption: str,
    alternatives: tuple[str, ...],
    revision: str,
    held_out_check: str,
    evidence: dict[str, typing.Any] = <factory>,
    rows: tuple[int, ...] = (),
    cooks: numpy.ndarray | None = None,
    loo_residual: numpy.ndarray | None = None,
    studentized: numpy.ndarray | None = None,
    leverage: numpy.ndarray | None = None,
    refits: tuple[dict, ...] = ()
) -> None
```

The influence check, with the per-row measures and exact deletion refits.

`rows` are the flagged row positions (zero-based, in data order).
`cooks`, `loo_residual`, `studentized` and `leverage` hold one value
per row. `refits` holds one record per refitted row: the row, and the
largest and mean change in fitted values on the rows that were kept.

Source: [`python/mboostrs/residual_checks.py` lines 107-122](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/residual_checks.py#L107-L122)
