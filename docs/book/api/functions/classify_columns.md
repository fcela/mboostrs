# `classify_columns`

Defined in `mboostrs.terms`; exported as `mboostrs.classify_columns`.


*Function*

```python
classify_columns(data, response=None, df=4.0, smooth='bbs')
```

Describe each predictor and name the learner suggested for it.

**Parameters**

- `data` (dict or DataFrame)
- `response` (str, optional): Column to leave out. When it is numeric, it is also used to read a
  whole-number column with more than `CATEGORICAL_MAX_LEVELS` values:
  see `label_evidence`. A column whose values behave as labels gets
  `brandom`; one that behaves as a measurement keeps the smooth.
  A column that reads as a count (non-negative whole numbers from 0
  or 1, at least three of them; see `count_like`) keeps a measurement
  term whatever the response says, because its effect has to extend
  to a count the fit never saw.
- `df` (float): Degrees of freedom for the penalised learners.
- `smooth` ({"bbs", "btree", "hinge", "menu"}): The learner for a continuous column. `"bbs"` is a P-spline;
  `"btree"` is a stump, which fits a threshold where a spline would
  smooth across it; `"hinge"` places a kink where the data bends,
  keeping both sides straight; `"menu"` offers `bols`, `bbs` and
  `btree` for the column and leaves the choice to boosting's
  selection step.

**Returns**

list of ColumnKind

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/terms.py` lines 434-472](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/terms.py#L434-L472)
