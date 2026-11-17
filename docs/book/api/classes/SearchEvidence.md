# `SearchEvidence`

Defined in `mboostrs.search_session`; exported as `mboostrs.SearchEvidence`.


*Class*

```python
SearchEvidence(
    action: str,
    status: str,
    evidence_id: str,
    state_id: str,
    fit_charges: int,
    seconds: float,
    summary: dict,
    value: object = None
) -> None
```

One action's result and its position in the search's audit trail.

Source: [`python/mboostrs/search_session.py` lines 48-68](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/search_session.py#L48-L68)

## Members

| Member | Kind | Summary |
|---|---|---|
| `to_dict` | Method | Audit metadata; the arbitrary action value is retrieved separately. |

(searchevidence-to_dict)=
## `SearchEvidence.to_dict`

*Method* of `SearchEvidence`

```python
SearchEvidence.to_dict(*, include_arrays=False)
```

Audit metadata; the arbitrary action value is retrieved separately.

Source: [`python/mboostrs/search_session.py` lines 61-68](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/search_session.py#L61-L68)
