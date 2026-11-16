# `elicit_causal_spec`

Defined in `mboostrs.modeler`; exported as `mboostrs.elicit_causal_spec`.


*Function*

```python
elicit_causal_spec(frame, judgment, *, domain_context=None)
```

Interview a judgment layer for a causal design and validate it.

The judge receives the variable digest and the schema above and returns
a proposed treatment, outcome, timing, adjustment set and rationale.
Mechanical checks then run against the frame: every named column must
exist, the treatment must be binary for the supported estimators, the
proposal must state timing and a rationale, and
`mboostrs.CausalSpec`'s own validation must pass. A proposal
that fails any check returns `CausalDraft(spec=None, problems=[...])`
— it is never silently repaired.

Returns a `CausalDraft`. Confirming the design is a separate
human act: `draft.ledger.confirm(draft.ledger.entries[0].statement)`
upgrades the draft's entry to `llm-proposed-confirmed`, and the
person's own additions are recorded as `modeler-stated`.

**Parameters**

- `frame` (mapping of column name to array)
- `judgment` (callable): `judgment(digest) -> dict`; see `LLMJudge`.
- `domain_context` (str, optional): What the columns and the study mean, sent with the digest.

Source: [`python/mboostrs/modeler.py` lines 574-657](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/modeler.py#L574-L657)
