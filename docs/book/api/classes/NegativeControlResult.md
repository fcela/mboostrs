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

Source: [`python/mboostrs/causal.py` lines 920-989](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L920-L989)

## Members

| Member | Kind | Summary |
|---|---|---|
| `interval_excludes_expected` | Property | Whether the reported interval excludes the prespecified null. |
| `to_dict` | Method | The falsification verdict and its expected-null justification. |
| `describe` | Method | Interpret the falsification result without treating a null as proof. |

(negativecontrolresult-interval_excludes_expected)=
## `NegativeControlResult.interval_excludes_expected`

*Property* of `NegativeControlResult`

```python
NegativeControlResult.interval_excludes_expected  # -> bool | None
```

Whether the reported interval excludes the prespecified null.

Source: [`python/mboostrs/causal.py` lines 939-945](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L939-L945)

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

Source: [`python/mboostrs/causal.py` lines 947-959](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L947-L959)

(negativecontrolresult-describe)=
## `NegativeControlResult.describe`

*Method* of `NegativeControlResult`

```python
NegativeControlResult.describe() -> str
```

Interpret the falsification result without treating a null as proof.

Source: [`python/mboostrs/causal.py` lines 968-989](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L968-L989)
