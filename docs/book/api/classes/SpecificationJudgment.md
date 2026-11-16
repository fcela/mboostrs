# `SpecificationJudgment`

Defined in `mboostrs.modeler`; exported as `mboostrs.SpecificationJudgment`.


*Class*

```python
SpecificationJudgment(
    drop: tuple[str, ...] = (),
    add: tuple[str, ...] = (),
    lock: tuple[str, ...] = (),
    replace: dict = <factory>,
    monotone: dict = <factory>,
    flags: tuple = (),
    assumptions: tuple = (),
    reasoning: str = '',
    estimand_note: str = '',
    evidence_id: str | None = None
) -> None
```

Validated editable proposal; predictive evidence decides adoption.

Replacements name existing terms exactly. Monotone proposals require an
explicit direction; interactions involving that column must be resolved
before a univariate constraint can describe its complete effect.

Source: [`python/mboostrs/modeler.py` lines 186-285](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/modeler.py#L186-L285)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`to_dict`](#specificationjudgment-to_dict) | Method | The editable JSON proposal accepted by the judgment interface. |
| [`from_mapping`](#specificationjudgment-from_mapping) | Method |  |

(specificationjudgment-to_dict)=
## `SpecificationJudgment.to_dict`

*Method* of `SpecificationJudgment`

```python
SpecificationJudgment.to_dict(*, include_arrays=False)
```

The editable JSON proposal accepted by the judgment interface.

Source: [`python/mboostrs/modeler.py` lines 229-239](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/modeler.py#L229-L239)

(specificationjudgment-from_mapping)=
## `SpecificationJudgment.from_mapping`

*Method* of `SpecificationJudgment`

```python
SpecificationJudgment.from_mapping(proposal)
```

*No description in the docstring.*

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/modeler.py` lines 241-285](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/modeler.py#L241-L285)
