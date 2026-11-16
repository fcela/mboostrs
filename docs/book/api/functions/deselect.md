# `deselect`

Defined in `mboostrs.selection`; exported as `mboostrs.deselect`.


*Function*

```python
deselect(model, threshold=0.01, *, folds=5, seed=0, mstop=None)
```

Drop base learners with a small share of the risk reduction, then refit.

**Parameters**

- `model` (GAMBoostModel, GLMBoostModel or LSSModel): A fit at its tuned step count (for example `tune_mstop(...).model`).
  A distributional model must be fitted with `method="noncyclic"`,
  the setting Strömer et al. (2022) propose deselection for: each step
  updates one learner of one parameter, so each step's risk reduction
  belongs to one learner. Shares are computed over all parameters
  together, a parameter left with no term is refitted as intercept
  only (`y ~ 1`), and the refit's total step count is tuned by
  `LSSModel.cvrisk`.
- `threshold` (float): A learner whose share of the total in-bag risk reduction is below this
  is dropped. Strömer et al. (2022) use 0.01.
- `folds` (int or fold matrix): Cross-validation for the refit's `mstop`: an int is random k-fold on
  `seed`; pass `grouped_folds(...)` for grouped data.
- `mstop` (int, optional): The refit's largest step count; defaults to twice the larger of the
  model's control `mstop` and its current `mstop`.

**Returns**

Deselection
    `kept`, `dropped`, the per-learner `shares`, the reduced
    `formula` and the refitted `model`.

A learner's share is its accumulated in-bag risk reduction, so a term that
moved the fit a little in many steps can fall below the threshold even
when the true effect is non-zero but small; the threshold trades those
losses against the noise terms removed.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/selection.py` lines 243-321](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/selection.py#L243-L321)
