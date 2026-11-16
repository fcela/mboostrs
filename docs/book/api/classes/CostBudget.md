# `CostBudget`

Defined in `mboostrs.policy`; exported as `mboostrs.CostBudget`.


*Class*

```python
CostBudget(
    seconds: float | None = None,
    fits: int | None = None,
    memory_mb: float | None = None
) -> None
```

Limits on a procedure's fit count, run time and memory. `None` means no limit on that field.

A budget covers everything the procedure does -- probes, fold fits,
grid sweeps, the final fit -- not one stage of it. Counting model fits
is the portable measure; the seconds and memory ceilings are checked
between stages, since neither is observable from inside a fit.

Source: [`python/mboostrs/policy.py` lines 80-109](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/policy.py#L80-L109)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`to_dict`](#costbudget-to_dict) | Method | The limits, as a JSON-safe record. |

(costbudget-to_dict)=
## `CostBudget.to_dict`

*Method* of `CostBudget`

```python
CostBudget.to_dict(*, include_arrays=False)
```

The limits, as a JSON-safe record.

Source: [`python/mboostrs/policy.py` lines 103-109](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/policy.py#L103-L109)
