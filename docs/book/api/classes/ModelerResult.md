# `ModelerResult`

Defined in `mboostrs.modeler`; exported as `mboostrs.ModelerResult`.


*Class*

```python
ModelerResult(
    spec: str,
    family: object,
    mstop: int,
    groups: tuple,
    transcript: mboostrs.modeler.Transcript,
    ledger: mboostrs.modeler.AssumptionLedger,
    refusals: list,
    fits: int,
    causal_result: object = None,
    trap_report: object = None,
    judgment_calls: int = 0,
    model: object = None,
    stop_reason: str = 'round_limit',
    locked_columns: tuple = (),
    budget_report: dict | None = None,
    action_evidence: tuple = ()
) -> None
```

Model and specification selected by the loop, with the decision record.

Source: [`python/mboostrs/modeler.py` lines 676-719](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/modeler.py#L676-L719)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`to_dict`](#modelerresult-to_dict) | Method | Editable specification and audit trail, excluding model internals. |

(modelerresult-to_dict)=
## `ModelerResult.to_dict`

*Method* of `ModelerResult`

```python
ModelerResult.to_dict(*, include_arrays=False)
```

Editable specification and audit trail, excluding model internals.

Source: [`python/mboostrs/modeler.py` lines 697-714](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/modeler.py#L697-L714)
