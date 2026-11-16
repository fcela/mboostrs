# `rank_by_effect`

Defined in `mboostrs.causal`; exported as `mboostrs.rank_by_effect`.


*Function*

```python
rank_by_effect(effect, newdata, *, share=0.2, propensity=None, bounds=(0.02, 0.98))
```

Rank new rows by estimated effect, using only rows with overlap.

Propensity values near 0 or 1 indicate that one treatment arm has few
comparable rows. Effect estimates in those regions rely on extrapolation.
This function ranks rows inside `bounds` and lists rows outside the
overlap region separately.

**Parameters**

- `effect` (EffectModel, BLPProjection or data frame): An `EffectModel`, a `mboostrs.effect_modifiers.BLPProjection`
  from `mboostrs.effect_modifiers.EffectModifiers.projection` (its
  linear projection of the effect on the covariates), or a frame of
  estimated effects with an `estimate` column, one row per row of
  `newdata` (for example the output of `EffectModel.predict`). A
  projection or a frame needs `propensity`.
- `newdata` (mapping or data frame): The rows to rank. For an `EffectModel` it needs the modifiers and, when
  `propensity` is not given, every covariate of the propensity model.
- `share` (float): The top group holds `round(share * n)` rows, where `n` counts every
  row of `newdata`, so the default ranks the top fifth of the rows. If
  fewer rows lie inside `bounds`, the group is the inside rows and
  `shortfall` says how many places are unfilled.
- `propensity` (array-like, optional): The propensity of each row of `newdata`. By default an
  `EffectModel` computes it from its propensity model.
- `bounds` (tuple): `(lower, upper)`: a row is inside the overlap region when its
  propensity lies in the closed interval. The default `(0.02, 0.98)`
  matches `overlap_report`.

**Returns**

OverlapRanking
    `chosen` and `outside` are positions in `newdata`. The gain
    interval is the mean effect's own interval (level 0.95), computed
    from the full coefficient covariance, and is available only for an
    `EffectModel` with `terms="smooth"`.

Ties: rows with exactly equal estimates are ordered by interval width (the
span of `lower` to `upper`, or `se` when only that is available),
narrower first, and then by row position. A discrete modifier gives many
rows the same estimate, so the boundary can split a tie. `tied_rows`
counts the rows at the cut that share its estimate, and `tie_break` names
the rule; both appear in `OverlapRanking.describe`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/causal.py` lines 4714-4868](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L4714-L4868)
