# `influence_check`

Defined in `mboostrs.residual_checks`; exported as `mboostrs.influence_check`.


*Function*

```python
influence_check(
    model,
    alpha: float = 0.05,
    refit_top: int = 5,
    draws: int = 200,
    seed: int = 0
) -> InfluenceFinding
```

Use leverage and deletion refits to assess changes associated with individual rows.

The measures come from the hat-matrix diagonal, with no refit: Cook's
distance `D_i = r_i^2 h_i / (df sigma^2 (1 - h_i)^2)` (the size of the
change in every fitted value when row `i` is deleted), the leave-one-out
residual `r_i / (1 - h_i)` and the externally studentized deleted
residual `t_i`, where `r_i` is the response residual, `h_i` the
leverage, `df` the trace of the hat matrix and `sigma` the residual
standard deviation. A row is flagged when the two-sided Student-t p-value
of `t_i`, on `n - df - 1` degrees of freedom and Bonferroni-adjusted
over the rows, is below `alpha`. The flag uses `t_i` because its
reference is exact for a linear smoother with normal errors, where a
Cook's distance reference is not.

The top `refit_top` flagged rows are then removed one at a time and the
model is refitted with the same arguments, so the approximation is checked
against an exact deletion. The refit's change in fitted values on the
remaining rows is reported per row.

**Parameters**

- `model` (glmboost or gamboost fit with Gaussian (normal-errors) family): The leave-one-out formulas are for a Gaussian linear smoother. Other
  families are refused by name.
- `alpha` (float): Bonferroni-adjusted level of the flag.
- `refit_top` (int): How many flagged rows to refit without, largest Cook's distance first.
- `draws, seed`: Null datasets for the false-alarm rate, and the random seed.

**Returns**

InfluenceFinding
    `rows` are the flagged positions. `refits` gives the exact deletion
    change for each refitted row.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`
- `NotImplementedError`

**Notes**

The false-alarm rate simulates residuals with variance
`sigma^2 (1 - h_i)` and treats them as independent, because only the
hat-matrix diagonal is available. The Student-t reference is exact for a
linear smoother with normal errors and approximate for a boosting fit, so
the simulated rate is the number to read, not the nominal `alpha`.

Source: [`python/mboostrs/residual_checks.py` lines 523-680](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/residual_checks.py#L523-L680)
