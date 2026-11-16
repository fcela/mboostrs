# `equivalence_sets`

Defined in `mboostrs.equivalence`; exported as `mboostrs.equivalence_sets`.


*Function*

```python
equivalence_sets(model, *, threshold=0.9)
```

Learners whose design spaces overlap at or above `threshold`.

**Parameters**

- `model` (GAMBoostModel or GLMBoostModel): A fitted model. The overlap is measured on the rows the fit used, over
  every learner with a design, whether or not the boosting path selected
  it.
- `threshold` (float): Overlap at or above which two learners are grouped, in (0, 1]. The
  core's concurvity measure reports 1 for a learner that another learner
  reproduces exactly. On a simulated fit with 300 rows, unrelated terms
  measured 0.15 to 0.3 and a near-copy of a variable about 0.99. On few
  rows, unrelated terms can also reach the threshold; a warning names
  the wide basis when that is possible.

**Returns**

EquivalenceSets

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`
- `TypeError`

**Notes**

The overlap is a pessimistic bound: it is the largest fraction of one
learner's function space that the other can reproduce over every
coefficient vector, so a rich basis such as a P-spline can score high
against a linear term that lies in its span. Read the pair values, not
only the groups.

Source: [`python/mboostrs/equivalence.py` lines 247-318](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/equivalence.py#L247-L318)
