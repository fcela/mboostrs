# `holdout`

Defined in `mboostrs.explore`; exported as `mboostrs.holdout`.


*Function*

```python
holdout(data, fraction=0.2, *, groups=None, time=None, seed=0)
```

Split a frame into a search part and a test part.

The search part is where specifications are compared and tuned; the test
part is scored once, at the end, by predicting it from a model fitted on
the search part. `fraction` is the test part's share of the rows, or of
the groups under `groups=`.

The split follows the evaluation task the scores are meant to describe:

* With neither `groups` nor `time`, rows are dealt at random under
  `seed`: the test part measures prediction for new observations like
  the sampled ones.
* `groups=` (a column name, or one group code per row) deals whole
  groups to one side: the test part holds groups the search part never
  saw, so it measures prediction in new groups.
* `time=` (a column name, or one value per row) splits on the time
  axis: the test part is the most recent `fraction` of the rows, so it
  measures prediction of the future.

`groups` and `time` name different tasks and cannot be combined.

Returns `(search, test)`, each the same kind of frame as `data`. The
test part is remembered by its content: an evaluation assistant
(`evaluate_specification`, `compare_specifications`,
`term_report`, `tune_mstop`) later run on exactly those rows
warns, from the second specification onward, how many have used them.
A hand-written predict-and-score loop is not intercepted; score the test
part once and keep the result with the search that produced it.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/explore.py` lines 1223-1293](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/explore.py#L1223-L1293)
