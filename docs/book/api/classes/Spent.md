# `Spent`

Defined in `mboostrs.policy`; exported as `mboostrs.Spent`.


*Class*

```python
Spent(
    seconds: float = 0.0,
    fits: int = 0,
    memory_mb: float | None = None,
    budget: mboostrs.policy.CostBudget | None = None,
    stages: list = <factory>
) -> None
```

Fit, time and memory charges recorded during a procedure.

Source: [`python/mboostrs/policy.py` lines 112-169](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/policy.py#L112-L169)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`record`](#spent-record) | Method | Add one stage's cost, and return ``self`` for chaining. |
| [`exceeded`](#spent-exceeded) | Method | The name of the limit passed, or ``None`` while inside budget. |
| [`remaining`](#spent-remaining) | Method | Remaining allowance for each budget limit; ``None`` where unlimited. |
| [`to_dict`](#spent-to_dict) | Method | Spent against budget, with the stage log. |

(spent-record)=
## `Spent.record`

*Method* of `Spent`

```python
Spent.record(label, *, seconds=0.0, fits=0, memory_mb=None)
```

Add one stage's cost, and return `self` for chaining.

Source: [`python/mboostrs/policy.py` lines 123-131](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/policy.py#L123-L131)

(spent-exceeded)=
## `Spent.exceeded`

*Method* of `Spent`

```python
Spent.exceeded()
```

The name of the limit passed, or `None` while inside budget.

Source: [`python/mboostrs/policy.py` lines 133-144](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/policy.py#L133-L144)

(spent-remaining)=
## `Spent.remaining`

*Method* of `Spent`

```python
Spent.remaining()
```

Remaining allowance for each budget limit; `None` where unlimited.

Source: [`python/mboostrs/policy.py` lines 146-157](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/policy.py#L146-L157)

(spent-to_dict)=
## `Spent.to_dict`

*Method* of `Spent`

```python
Spent.to_dict(*, include_arrays=False)
```

Spent against budget, with the stage log.

Source: [`python/mboostrs/policy.py` lines 159-169](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/policy.py#L159-L169)
