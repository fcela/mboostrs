# `SelectionReport`

Defined in `mboostrs.equivalence`; exported as `mboostrs.SelectionReport`.


*Class*

```python
SelectionReport(
    equivalence: mboostrs.equivalence.EquivalenceSets,
    substitution: mboostrs.equivalence.SubstitutionSets | None,
    dropped_with_substitute: tuple,
    dropped_alone: tuple,
    scope: str = 'Predictive substitution in the rows and resamples used. A group names learners that carry the same information there; it does not identify which member is causal, and it is not a causal claim.'
) -> None
```

Equivalence sets, substitution sets, and the deselected learners they explain.

`dropped_with_substitute` lists each learner that `mboostrs.deselect`
dropped while a learner in its equivalence group was kept. Such a drop
removes one of two interchangeable predictors, not the information they
share. A dropped learner with no kept equivalent is listed in
`dropped_alone`.

Source: [`python/mboostrs/equivalence.py` lines 156-182](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/equivalence.py#L156-L182)

## Members

| Member | Kind | Summary |
|---|---|---|
| `to_dict` | Method | A JSON-safe record. ``include_arrays`` is accepted for the shared |

(selectionreport-to_dict)=
## `SelectionReport.to_dict`

*Method* of `SelectionReport`

```python
SelectionReport.to_dict(*, include_arrays=False)
```

A JSON-safe record. `include_arrays` is accepted for the shared
export interface; these results hold no arrays, so it changes nothing.

Source: [`python/mboostrs/equivalence.py` lines 173-182](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/equivalence.py#L173-L182)
