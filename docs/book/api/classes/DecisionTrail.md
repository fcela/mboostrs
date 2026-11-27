# `DecisionTrail`

Defined in `mboostrs._procedure._trail`; exported as `mboostrs.DecisionTrail`.


*Class*

```python
DecisionTrail(
    schema_version: str = '1.0',
    provenance: dict[str, typing.Any] = <factory>,
    data: dict[str, typing.Any] = <factory>,
    folds: dict[str, typing.Any] = <factory>,
    criterion: str = 'oof_log_score',
    steps: list[dict[str, typing.Any]] = <factory>,
    final: dict[str, typing.Any] = <factory>,
    uncertainty: dict[str, typing.Any] = <factory>,
    budget: dict[str, typing.Any] = <factory>
) -> None
```

A builder for the record in section 5 of the design note.

Source: [`python/mboostrs/_procedure/_trail.py` lines 191-378](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_procedure/_trail.py#L191-L378)

## Members

| Member | Kind | Summary |
|---|---|---|
| `start` | Method | Open a trail with the data, folds and provenance. |
| `add_step` | Method | Append one decision. Extra keyword arguments become its fields. |
| `set_final` | Method | Record the final specification, and the model's unseen-level policy |
| `set_uncertainty` | Method |  |
| `set_budget` | Method | Record the budget setting, its fit and time limits and the fits spent. |
| `to_dict` | Method | The trail as plain Python types, JSON-safe (NaN/inf become null). |
| `to_json` | Method |  |
| `from_dict` | Method | Rebuild a trail from :meth:`to_dict` output or a JSON string. |
| `validate` | Method | Check the record against the schema, raising on failure. |

(decisiontrail-start)=
## `DecisionTrail.start`

*Method* of `DecisionTrail`

```python
DecisionTrail.start(
    data: Any,
    response: str,
    seed: int,
    folds_info: dict[str, typing.Any] | None = None,
    column_kinds: dict[str, str] | None = None,
    folds: Any = None
) -> DecisionTrail
```

Open a trail with the data, folds and provenance.

`folds_info` is used as given; when it is `None` the record is
derived from `folds` (a `mboostrs.folds.FoldScheme`, a weight
matrix, or `None` for the default scheme), which is what `fit`
does.

Source: [`python/mboostrs/_procedure/_trail.py` lines 222-273](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_procedure/_trail.py#L222-L273)

(decisiontrail-add_step)=
## `DecisionTrail.add_step`

*Method* of `DecisionTrail`

```python
DecisionTrail.add_step(step: int | str, decision: str, **fields: Any) -> DecisionTrail
```

Append one decision. Extra keyword arguments become its fields.

Source: [`python/mboostrs/_procedure/_trail.py` lines 275-280](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_procedure/_trail.py#L275-L280)

(decisiontrail-set_final)=
## `DecisionTrail.set_final`

*Method* of `DecisionTrail`

```python
DecisionTrail.set_final(
    formula: str,
    family: Any,
    mstop: int,
    unseen_levels: str | None = None
) -> DecisionTrail
```

Record the final specification, and the model's unseen-level policy
when it is not the library default `"error"`.

Source: [`python/mboostrs/_procedure/_trail.py` lines 282-291](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_procedure/_trail.py#L282-L291)

(decisiontrail-set_uncertainty)=
## `DecisionTrail.set_uncertainty`

*Method* of `DecisionTrail`

```python
DecisionTrail.set_uncertainty(method: str = 'none', **fields: Any) -> DecisionTrail
```

*No description in the docstring.*

Source: [`python/mboostrs/_procedure/_trail.py` lines 293-305](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_procedure/_trail.py#L293-L305)

(decisiontrail-set_budget)=
## `DecisionTrail.set_budget`

*Method* of `DecisionTrail`

```python
DecisionTrail.set_budget(
    setting: str = 'standard',
    spent: float = 0.0,
    degradations: Any = (),
    limit: int | None = None,
    seconds_limit: float | None = None
) -> DecisionTrail
```

Record the budget setting, its fit and time limits and the fits spent.

Source: [`python/mboostrs/_procedure/_trail.py` lines 307-320](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_procedure/_trail.py#L307-L320)

(decisiontrail-to_dict)=
## `DecisionTrail.to_dict`

*Method* of `DecisionTrail`

```python
DecisionTrail.to_dict(*, include_arrays: bool = False) -> dict[str, Any]
```

The trail as plain Python types, JSON-safe (NaN/inf become null).

Source: [`python/mboostrs/_procedure/_trail.py` lines 323-338](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_procedure/_trail.py#L323-L338)

(decisiontrail-to_json)=
## `DecisionTrail.to_json`

*Method* of `DecisionTrail`

```python
DecisionTrail.to_json(indent: int = 2) -> str
```

*No description in the docstring.*

Source: [`python/mboostrs/_procedure/_trail.py` lines 340-341](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_procedure/_trail.py#L340-L341)

(decisiontrail-from_dict)=
## `DecisionTrail.from_dict`

*Method* of `DecisionTrail`

```python
DecisionTrail.from_dict(value: dict[str, typing.Any] | str) -> DecisionTrail
```

Rebuild a trail from `to_dict` output or a JSON string.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `TypeError`

Source: [`python/mboostrs/_procedure/_trail.py` lines 343-363](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_procedure/_trail.py#L343-L363)

(decisiontrail-validate)=
## `DecisionTrail.validate`

*Method* of `DecisionTrail`

```python
DecisionTrail.validate() -> bool
```

Check the record against the schema, raising on failure.

Source: [`python/mboostrs/_procedure/_trail.py` lines 366-378](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_procedure/_trail.py#L366-L378)
