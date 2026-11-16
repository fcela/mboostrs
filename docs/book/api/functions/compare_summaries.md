# `compare_summaries`

Defined in `mboostrs.summary`; exported as `mboostrs.compare_summaries`.


*Function*

```python
compare_summaries(*models, labels=None, cv=None)
```

Put several fitted models side by side as one `Summary`.

Each model becomes a column: the first table holds the scalar facts, the
second how many updates each base learner received under each model. A
learner absent from a model shows an empty cell, so the rows line up
across models fitted on different formulas.

**Parameters**

- `*models`: Fitted models. They need not be the same kind; rows that do not apply
  to a model are left blank.
- `labels` (list of str, optional): Column headings. Defaults to "model 1", "model 2", ...
- `cv` (list, optional): One `mboostrs.cvrisk.CvRiskResult` per model, or None for a
  model that has none. Adds the cross-validated risk and optimal
  iteration to the facts table.

**Returns**

Summary

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

**Notes**

The models must be fitted to the same rows, response and effective
weights, or this raises `ValueError`: their risks and criteria would
describe different data. Models that share their data but differ in
response scale (`log(y)` against `y`) or weight policy are shown
side by side with a note that they are not ranked against each other.
The risks are comparable across models only when the family and the data
are the same; a Gaussian risk and a Binomial risk are different losses.
`cv` results must all come from the same folds.

Source: [`python/mboostrs/summary.py` lines 674-793](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/summary.py#L674-L793)
