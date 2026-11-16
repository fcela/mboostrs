# `AssumptionLedger`

Defined in `mboostrs.modeler`; exported as `mboostrs.AssumptionLedger`.


*Class*

```python
AssumptionLedger(entries: list = <factory>) -> None
```

The causal statements a run depends on, tagged by who vouched for them.

An LLM-proposed statement is recorded `llm-proposed-unconfirmed` and
stays that way until a person confirms it; the loop never upgrades
provenance on its own. Unresolved items belong in the ledger as
`unresolved` so outputs can say what was never settled.

Source: [`python/mboostrs/modeler.py` lines 125-155](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/modeler.py#L125-L155)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`add`](#assumptionledger-add) | Method |  |
| [`confirm`](#assumptionledger-confirm) | Method | Mark the first matching unconfirmed LLM proposal as confirmed. |
| [`to_list`](#assumptionledger-to_list) | Method |  |

(assumptionledger-add)=
## `AssumptionLedger.add`

*Method* of `AssumptionLedger`

```python
AssumptionLedger.add(statement, provenance, *, source='', round=None)
```

*No description in the docstring.*

Source: [`python/mboostrs/modeler.py` lines 137-140](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/modeler.py#L137-L140)

(assumptionledger-confirm)=
## `AssumptionLedger.confirm`

*Method* of `AssumptionLedger`

```python
AssumptionLedger.confirm(statement)
```

Mark the first matching unconfirmed LLM proposal as confirmed.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `KeyError`

Source: [`python/mboostrs/modeler.py` lines 142-148](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/modeler.py#L142-L148)

(assumptionledger-to_list)=
## `AssumptionLedger.to_list`

*Method* of `AssumptionLedger`

```python
AssumptionLedger.to_list()
```

*No description in the docstring.*

Source: [`python/mboostrs/modeler.py` lines 150-155](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/modeler.py#L150-L155)
