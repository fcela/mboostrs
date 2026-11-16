# `selection_report`

Defined in `mboostrs.equivalence`; exported as `mboostrs.selection_report`.


*Function*

```python
selection_report(
    model,
    deselection=None,
    stability=None,
    equivalence_threshold=0.9,
    min_share=0.1,
    max_joint=0.2,
    min_overlap=0.5
)
```

Equivalence sets, substitution sets and deselection, read together.

**Parameters**

- `model` (GAMBoostModel or GLMBoostModel): The fit the learners belong to.
- `deselection` (Deselection, optional): A result of `mboostrs.deselect` on `model`. Its dropped
  learners are classified against the equivalence sets.
- `stability` (StabSelResult, optional): A stability selection run on `model`, for the substitution sets. The
  substitution pairs are screened by the overlap in the equivalence sets.
- `equivalence_threshold, min_share, max_joint, min_overlap`: Passed to `equivalence_sets` and `substitution_sets`.

**Returns**

SelectionReport

Source: [`python/mboostrs/equivalence.py` lines 413-465](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/equivalence.py#L413-L465)
