# `EquivalenceSets`

Defined in `mboostrs.equivalence`; exported as `mboostrs.EquivalenceSets`.


*Class*

```python
EquivalenceSets(
    groups: tuple,
    overlap: dict,
    threshold: float,
    measured: tuple = (),
    scope: str = 'Predictive substitution in the rows and resamples used. A group names learners that carry the same information there; it does not identify which member is causal, and it is not a causal claim.'
) -> None
```

Groups of learners whose design spaces overlap at or above a threshold.

`groups` holds the connected components of the pairs whose overlap is at
least `threshold`, each with two or more learners. Components can chain:
if A overlaps B and B overlaps C, the three form one group even when A and
C fall below the threshold, so read `overlap` for the pair values.

Source: [`python/mboostrs/equivalence.py` lines 47-90](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/equivalence.py#L47-L90)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`group_of`](#equivalencesets-group_of) | Method | The group containing ``name``, or None when it has no equivalent. |
| [`to_dict`](#equivalencesets-to_dict) | Method | A JSON-safe record. ``include_arrays`` is accepted for the shared |

(equivalencesets-group_of)=
## `EquivalenceSets.group_of`

*Method* of `EquivalenceSets`

```python
EquivalenceSets.group_of(name)
```

The group containing `name`, or None when it has no equivalent.

Source: [`python/mboostrs/equivalence.py` lines 71-76](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/equivalence.py#L71-L76)

(equivalencesets-to_dict)=
## `EquivalenceSets.to_dict`

*Method* of `EquivalenceSets`

```python
EquivalenceSets.to_dict(*, include_arrays=False)
```

A JSON-safe record. `include_arrays` is accepted for the shared
export interface; these results hold no arrays, so it changes nothing.

Source: [`python/mboostrs/equivalence.py` lines 78-90](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/equivalence.py#L78-L90)
