# `ColumnKind`

Defined in `mboostrs.terms`; exported as `mboostrs.ColumnKind`.


*Class*

```python
ColumnKind(
    name: str,
    levels: int,
    kind: str,
    term: str,
    dtype: str = 'numeric',
    level_names: tuple | None = None,
    ordered: bool = True,
    n_missing: int = 0,
    unit: str | None = None,
    reason: str = '',
    alternatives: tuple[tuple[str, str], ...] = ()
) -> None
```

What a column looks like, the learner that follows, and what else it
could be.

`term` is a recommendation, and a recommendation with no stated
alternative is an instruction. `alternatives` names the other learners
that fit the same column and says what each one *assumes* -- the estimand
it targets and what it gives up -- so the choice can be made rather than
accepted. Everything here is a string the caller may edit before it
reaches `gamboost`.

Source: [`python/mboostrs/terms.py` lines 346-399](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/terms.py#L346-L399)

## Members

| Member | Kind | Summary |
|---|---|---|
| `explain` | Method | The recommendation, its reason, and the alternatives, as text. |

(columnkind-explain)=
## `ColumnKind.explain`

*Method* of `ColumnKind`

```python
ColumnKind.explain() -> str
```

The recommendation, its reason, and the alternatives, as text.

Source: [`python/mboostrs/terms.py` lines 384-399](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/terms.py#L384-L399)
