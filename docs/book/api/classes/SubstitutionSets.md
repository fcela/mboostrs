# `SubstitutionSets`

Defined in `mboostrs.equivalence`; exported as `mboostrs.SubstitutionSets`.


*Class*

```python
SubstitutionSets(
    groups: tuple,
    pairs: tuple,
    subsamples: int,
    min_share: float,
    max_joint: float,
    min_overlap: float | None = None,
    scope: str = 'Predictive substitution in the rows and resamples used. A group names learners that carry the same information there; it does not identify which member is causal, and it is not a causal claim.'
) -> None
```

Substitute pairs and the groups they form, from a stability selection run.

A pair is a substitute pair when both learners reach `min_share` of the
subsamples, and they are selected together in at most `max_joint` times
the smaller of their two frequencies. Groups are the connected components of
the substitute pairs, so a group can chain as described for
`EquivalenceSets`.

Source: [`python/mboostrs/equivalence.py` lines 106-153](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/equivalence.py#L106-L153)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`group_of`](#substitutionsets-group_of) | Method | The substitution group containing ``name``, or None. |
| [`to_dict`](#substitutionsets-to_dict) | Method | A JSON-safe record. ``include_arrays`` is accepted for the shared |

(substitutionsets-group_of)=
## `SubstitutionSets.group_of`

*Method* of `SubstitutionSets`

```python
SubstitutionSets.group_of(name)
```

The substitution group containing `name`, or None.

Source: [`python/mboostrs/equivalence.py` lines 132-137](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/equivalence.py#L132-L137)

(substitutionsets-to_dict)=
## `SubstitutionSets.to_dict`

*Method* of `SubstitutionSets`

```python
SubstitutionSets.to_dict(*, include_arrays=False)
```

A JSON-safe record. `include_arrays` is accepted for the shared
export interface; these results hold no arrays, so it changes nothing.

Source: [`python/mboostrs/equivalence.py` lines 139-153](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/equivalence.py#L139-L153)
