# `NegativeControlResult`

Defined in `mboostrs.causal`; exported as `mboostrs.NegativeControlResult`.


*Class*

```python
NegativeControlResult(
    kind: str,
    rationale: str,
    expected_effect: float,
    effect: mboostrs.causal.AIPWResult
) -> None
```

A prespecified negative-control estimate and its expected null value.

Source: [`python/mboostrs/causal.py` lines 883-952](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L883-L952)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`interval_excludes_expected`](#negativecontrolresult-interval_excludes_expected) | Property | Whether the reported interval excludes the prespecified null. |
| [`to_dict`](#negativecontrolresult-to_dict) | Method | The falsification verdict and its expected-null justification. |
| [`describe`](#negativecontrolresult-describe) | Method | Interpret the falsification result without treating a null as proof. |

(negativecontrolresult-interval_excludes_expected)=
## `NegativeControlResult.interval_excludes_expected`

*Property* of `NegativeControlResult`

```python
NegativeControlResult.interval_excludes_expected  # -> bool | None
```

Whether the reported interval excludes the prespecified null.

Source: [`python/mboostrs/causal.py` lines 902-908](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L902-L908)

(negativecontrolresult-to_dict)=
## `NegativeControlResult.to_dict`

*Method* of `NegativeControlResult`

```python
NegativeControlResult.to_dict(*, include_arrays=False)
```

The falsification verdict and its expected-null justification.

The nested effect record keeps its scalar estimate, standard error and
interval; its per-row contribution arrays are gated by
`include_arrays`, as in `AIPWResult.to_dict`.

Source: [`python/mboostrs/causal.py` lines 910-922](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L910-L922)

(negativecontrolresult-describe)=
## `NegativeControlResult.describe`

*Method* of `NegativeControlResult`

```python
NegativeControlResult.describe() -> str
```

Interpret the falsification result without treating a null as proof.

Source: [`python/mboostrs/causal.py` lines 931-952](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L931-L952)
