# `grouped_folds`

Defined in `mboostrs.folds`; exported as `mboostrs.grouped_folds`.


*Function*

```python
grouped_folds(groups, B=10, *, seed=42, weights=None, repeats=1) -> FoldScheme
```

K-fold splits that keep each group entirely on one side.

For repeated measurements, clustered sampling, or any design where rows
within a group share more than the model explains. Splitting such rows at
random leaves a group's own members in the training data whenever one of
them is held out, so the held-out row is partly predicted by itself and
the cross-validated risk is optimistic.

Groups are dealt to folds largest-first, which keeps the fold sizes close
when the groups are of unequal size; the seed orders groups of equal size.

With `repeats` greater than one the groups are dealt that many times,
each deal with the next draw of the seed's generator, and the deals are
stacked side by side as `repeated_kfold_folds` stacks row partitions: every
group is held out once in each deal. With few groups, which groups share a
fold changes a comparison by as much as the difference between two
specifications; scoring over several deals averages that out.
`compare_specifications` and `term_report` compute their paired standard
errors within each deal, since the deals reuse the same groups.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/folds.py` lines 618-688](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/folds.py#L618-L688)
