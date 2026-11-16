# `sensitivity`

Defined in `mboostrs.explore`; exported as `mboostrs.sensitivity`.


*Function*

```python
sensitivity(
    model,
    data,
    variables=None,
    points=25,
    quantiles=(0.1, 0.9),
    parameter=None
)
```

How much the fitted function moves across each variable's range.

The predictors are the fitted model's own -- the columns its terms name
-- never every column in `data`. For each one the fitted partial effect
is evaluated on a grid with the other predictors held at their means, and
three numbers are reported: the total rise and fall of the effect, its
slope between the given quantiles, and whether it turns. `turns` is
true when the effect rises and then falls (or the reverse), with each of
the two moves larger than 5% of the effect's range. Smaller reversals,
such as spline wiggles on a nearly monotone curve, are not turns. The
flag is what distinguishes a monotone effect that can be summarised by a
single number from one that cannot.

A categorical predictor is gridded over its actual levels, not a numeric
interpolation between codes ("level 0.37" is not a category). For an
unordered one the slope and the turn count are `nan`: without a
declared order there is no scale to take a slope along, and the
"direction" of the effect would depend on an arbitrary level ordering. A
datetime predictor is a numeric time axis and is reported in its unit.

A distributional model (`gamboostLSS()` or `glmboostLSS()`) has a
separate predictor set for each distribution parameter, so it is measured
one parameter at a time. `parameter` names that parameter (`"mu"`,
`"sigma"`, ...); the default is the family's first parameter. The effect
is the parameter's linear predictor on its link scale (the log scale for
a log-link parameter such as `sigma`), and each row names the parameter
it was measured on. `parameter` is refused for a gamboost()/glmboost()
model, which has one predictor set.

The columns that the other parameters' formulas read are held at their
means (or most frequent levels) on the grid, as the non-gridded columns
of the measured parameter are, so a parameter is measured whatever the
other formulas contain. Those columns must be in `data`.

A model fitted with `na_action="impute"` has missing-value indicator
columns (`<col>_missing`, or `<col>_missing_` when the data already
has a column of that name). They are built from `<col>` when the grid is
predicted, so `data` supplies only the original columns.

Source: [`python/mboostrs/explore.py` lines 3069-3113](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/explore.py#L3069-L3113)
