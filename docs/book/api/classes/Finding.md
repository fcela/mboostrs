# `Finding`

Defined in `mboostrs.insights`; exported as `mboostrs.Finding`.


*Class*

```python
Finding(
    finding: str,
    effect_size: float,
    uncertainty: float,
    basis: str = 'measured',
    evidence: dict = <factory>,
    next_action: str = ''
) -> None
```

One thing the discovery pipeline found, and what to do about it.

`basis` says what kind of number `effect_size` is, following the
`mboostrs.explore.Suggestion` convention: `"measured"` means the
effect was fitted and scored on held-out folds; `"detected"` means
structure was found in a residual without any model of it fitted. A
detected signal says something is there; it does not say which learner
will capture it.

Source: [`python/mboostrs/insights.py` lines 50-75](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/insights.py#L50-L75)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`to_dict`](#finding-to_dict) | Method | A JSON-safe record of the finding and its evidence. |

(finding-to_dict)=
## `Finding.to_dict`

*Method* of `Finding`

```python
Finding.to_dict(*, include_arrays=False)
```

A JSON-safe record of the finding and its evidence.

Source: [`python/mboostrs/insights.py` lines 73-75](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/insights.py#L73-L75)
