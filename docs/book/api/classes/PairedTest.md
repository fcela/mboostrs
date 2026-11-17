# `PairedTest`

Defined in `mboostrs.policy`; exported as `mboostrs.PairedTest`.


*Class*

```python
PairedTest(
    rung: str,
    baseline_rhs: str,
    candidate_rhs: str,
    baseline: float,
    candidate: float,
    gain: float,
    se: float,
    t: float,
    threshold_se: float,
    passed: bool,
    worse_folds: int,
    n_folds: int,
    adopted: bool,
    reason: str
) -> None
```

One rung's decision: the paired fold comparison behind an adoption.

Source: [`python/mboostrs/policy.py` lines 280-320](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/policy.py#L280-L320)

## Members

| Member | Kind | Summary |
|---|---|---|
| `to_dict` | Method | A JSON-safe record of the decision. |

(pairedtest-to_dict)=
## `PairedTest.to_dict`

*Method* of `PairedTest`

```python
PairedTest.to_dict(*, include_arrays=False)
```

A JSON-safe record of the decision.

Source: [`python/mboostrs/policy.py` lines 299-315](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/policy.py#L299-L315)
