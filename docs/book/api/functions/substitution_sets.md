# `substitution_sets`

Defined in `mboostrs.equivalence`; exported as `mboostrs.substitution_sets`.


*Function*

```python
substitution_sets(
    stability,
    min_share=0.1,
    max_joint=0.2,
    equivalence=None,
    min_overlap=0.5
)
```

Learners that stability selection picks in turn rather than together.

**Parameters**

- `stability` (StabSelResult): A result of `mboostrs.stabsel` on a `GAMBoostModel` or
  `GLMBoostModel`. Its `fit_selected` holds each subsample's
  selection.
- `min_share` (float): A learner enters the search when it is selected in at least this share
  of the subsamples.
- `max_joint` (float): A pair is a substitute pair when the share of subsamples selecting both
  is at most `max_joint` times the smaller of the two individual shares.
  Two independent learners selected in shares `f` and `g` are selected
  together in about `f * g` of the subsamples, which is `max(f, g)`
  times `min(f, g)`. A `max_joint` well below the larger share
  therefore separates alternation from chance co-selection.
- `equivalence` (EquivalenceSets, optional): When given, a pair also has to overlap at least `min_overlap` in the
  fit's design spaces. Without this, two unrelated learners that the path
  rarely selects together can join a group only because one of them is
  rarely selected at all, so a chain of unrelated learners can form.
- `min_overlap` (float): The overlap a pair needs when `equivalence` is given. Unrelated
  learners measured 0.16 to 0.3 in the simulation behind this module, and
  a correlated proxy (correlation about 0.89) measured 0.79 to 0.82.

**Returns**

SubstitutionSets

**Raises**

- `ValueError`: When the result carries no per-subsample selections, as LSS stability
results do not.

Source: [`python/mboostrs/equivalence.py` lines 321-410](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/equivalence.py#L321-L410)
