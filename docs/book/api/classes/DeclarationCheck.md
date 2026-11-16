# `DeclarationCheck`

Defined in `mboostrs.declarations`; exported as `mboostrs.DeclarationCheck`.


*Class*

```python
DeclarationCheck(
    declaration: str,
    kind: str,
    variable: str,
    test: str | None,
    verdict: str,
    agreement: str,
    evidence: dict = <factory>,
    reason: str = ''
) -> None
```

The test of one declaration against the fit.

`verdict` is `"agrees"`, `"contradicts"` or `"inconclusive"`. The
rule that maps measurements to a verdict is in `evidence["rule"]`.
`agreement` is one sentence, phrased as agreement or disagreement with the
fitted association. `evidence` holds the measured numbers.

Source: [`python/mboostrs/declarations.py` lines 92-123](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/declarations.py#L92-L123)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`to_dict`](#declarationcheck-to_dict) | Method | A JSON-safe record. The average curve of a shape test is an array; |

(declarationcheck-to_dict)=
## `DeclarationCheck.to_dict`

*Method* of `DeclarationCheck`

```python
DeclarationCheck.to_dict(*, include_arrays=False) -> dict
```

A JSON-safe record. The average curve of a shape test is an array;
it is left out unless `include_arrays=True`.

Source: [`python/mboostrs/declarations.py` lines 114-120](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/declarations.py#L114-L120)
