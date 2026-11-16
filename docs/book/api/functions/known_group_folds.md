# `known_group_folds`

Defined in `mboostrs.folds`; exported as `mboostrs.known_group_folds`.


*Function*

```python
known_group_folds(groups, B=10, *, seed=42, weights=None) -> FoldScheme
```

K-fold splits in which every held-out row has a group-mate in training.

The complement of `grouped_folds`. Rows are dealt to folds one group
at a time, so the members of one group land in different folds, and each
held-out row keeps at least one group-mate among its fold's training rows.
The score then measures prediction for a group the model has already
observed, the setting of a `brandom` or `bmrf` effect estimated for clusters
that are in the data. `grouped_folds` measures prediction for a
group the model has not seen.

Each group needs at least two rows with positive weight, so that a
held-out row has a training sibling; a group with one row is refused.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/folds.py` lines 881-926](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/folds.py#L881-L926)
