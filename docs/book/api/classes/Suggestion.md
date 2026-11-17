# `Suggestion`

Defined in `mboostrs.explore`; exported as `mboostrs.Suggestion`.


*Class*

```python
Suggestion(
    action: str,
    term: str,
    expected_gain: float,
    reason: str,
    basis: str = 'measured',
    specification: str = '',
    evidence: dict = <factory>
) -> None
```

One change worth trying, and the evidence for it.

`basis` says what kind of number `expected_gain` is, and the two are not
interchangeable. `"measured"` means the change was fitted and scored on
held-out folds, so the gain is what that change did. `"detected"` means
something was found in the residual — joint structure in a pair of columns,
say — without any model of it being fitted. A detected signal says there is
something there; it does not say which learner will capture it, and the
wrong learner can make the fit worse than leaving it out.

Source: [`python/mboostrs/explore.py` lines 2621-2654](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/explore.py#L2621-L2654)

## Members

| Member | Kind | Summary |
|---|---|---|
| `to_dict` | Method | A JSON-safe record of the change and the evidence for it. |

(suggestion-to_dict)=
## `Suggestion.to_dict`

*Method* of `Suggestion`

```python
Suggestion.to_dict(*, include_arrays=False)
```

A JSON-safe record of the change and the evidence for it.

Source: [`python/mboostrs/explore.py` lines 2652-2654](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/explore.py#L2652-L2654)
