# `RevisionFinding`

Defined in `mboostrs.residual_checks`; exported as `mboostrs.RevisionFinding`.


*Class*

```python
RevisionFinding(
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
    evidence: dict[str, typing.Any] = <factory>
) -> None
```

One residual check: what it found, the evidence, and the revision.

`flagged` is the decision at `alpha`. `statistic`, `value` and
`units` name the number the decision rests on; `p_value` and
`reference` say how it is judged. `false_alarm_rate` is the share of
`false_alarm_draws` simulated null datasets, with the same design, on
which the same rule fires. `revision` is the change to try when
`flagged` is true, and a statement that no change is indicated when it
is not.

Source: [`python/mboostrs/residual_checks.py` lines 39-104](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/residual_checks.py#L39-L104)

## Members

| Member | Kind | Summary |
|---|---|---|
| `summary` | Property | The finding as a readable block: pattern, evidence, guide. |

(revisionfinding-summary)=
## `RevisionFinding.summary`

*Property* of `RevisionFinding`

```python
RevisionFinding.summary  # -> str
```

The finding as a readable block: pattern, evidence, guide.

Source: [`python/mboostrs/residual_checks.py` lines 83-98](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/residual_checks.py#L83-L98)
