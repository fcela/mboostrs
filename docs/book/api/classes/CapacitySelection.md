# `CapacitySelection`

Defined in `mboostrs.capacity`; exported as `mboostrs.CapacitySelection`.


*Class*

```python
CapacitySelection(
    selected: str,
    formula: str,
    rhs: str,
    tolerance: float,
    rungs: tuple[dict[str, typing.Any], ...],
    screened_pairs: tuple[dict[str, typing.Any], ...],
    model: object,
    selected_estimator: str = 'boosting'
) -> None
```

The capacity ladder `select_capacity` measured, and what it kept.

`rungs` is one record per estimator/specification evaluated, in increasing
capacity: `label` (`"intercept"`, `"base"`, `"+tensors"`,
`"+tree"`), `estimator`, `rhs`,
`formula`, the held-out `score`, its marginal standard error
`fold_se` across folds, `fold_scores` (each fold's score at the step
count the rung was scored at, in fold order -- the paired differences
against the best rung are what `tolerance` reads), `n_folds`, the
`mstop` the rung was scored at (its tuned step count, the shared budget
for `budget="shared"`, or `None` for a direct REML fit, which has no
step count), whether the rung was `selected`, and a `note`. With more
than one `tree_depth` there is one `"+tree"` record per depth, in the
order given.
`screened_pairs` lists the pairs the residual screen proposed (the ones
with a positive held-out gain, up to `pairs=`), each with the mean
squared error it removed out of sample.

`model` is the selected estimator refitted on all rows. A boosting fit
uses its rung's tuned step count under `budget="tuned"` and the
comparison's `mstop` under `budget="shared"`; a direct REML fit has no
boosting step count. `selected_estimator` identifies the fitted
estimator.

Source: [`python/mboostrs/capacity.py` lines 204-271](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/capacity.py#L204-L271)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`variables`](#capacityselection-variables) | Property | The data columns the selected formula reads, in first-seen order. |
| [`to_dict`](#capacityselection-to_dict) | Method | A JSON-safe record; the model itself is not exported. |

(capacityselection-variables)=
## `CapacitySelection.variables`

*Property* of `CapacitySelection`

```python
CapacitySelection.variables
```

The data columns the selected formula reads, in first-seen order.

A `%X%` term names both of its margins.

Source: [`python/mboostrs/capacity.py` lines 251-262](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/capacity.py#L251-L262)

(capacityselection-to_dict)=
## `CapacitySelection.to_dict`

*Method* of `CapacitySelection`

```python
CapacitySelection.to_dict(*, include_arrays: bool = False)
```

A JSON-safe record; the model itself is not exported.

Source: [`python/mboostrs/capacity.py` lines 264-271](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/capacity.py#L264-L271)
