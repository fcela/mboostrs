# `CausalDraft`

Defined in `mboostrs.modeler`; exported as `mboostrs.CausalDraft`.


*Class*

```python
CausalDraft(
    spec: object,
    ledger: mboostrs.modeler.AssumptionLedger,
    questions: list,
    unresolved: list,
    problems: list
) -> None
```

A proposed causal design: a spec a person has not yet confirmed.

`spec` is `None` when the proposal failed the mechanical checks in
`problems`; `questions` and `unresolved` travel with the draft so
the confirming person sees what the elicitation could not settle. The
draft's ledger entry is `llm-proposed-unconfirmed` until
`AssumptionLedger.confirm` marks it confirmed.

Source: [`python/mboostrs/modeler.py` lines 535-556](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/modeler.py#L535-L556)
