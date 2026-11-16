# `heteroscedasticity`

Defined in `mboostrs.explore`; exported as `mboostrs.heteroscedasticity`.


*Function*

```python
heteroscedasticity(model, data=None, *, columns=None, transforms=None) -> dict
```

Measure how residual spread changes across fitted values.

A mean model assumes it does not. When it does, the fit is still the best
guess at the mean and every interval around it is wrong -- too narrow
where the spread is large, too wide where it is small -- so the answer is
a distributional model (`glmboostLSS`/`gamboostLSS`), which gives the
scale its own predictor rather than one number.

The measurement is the slope of `log(residual^2)` on the fitted value.
Under constant variance that slope is zero and `log(residual^2)` is
`log` of a chi-square with one degree of freedom, whose variance is
`pi^2 / 2` whatever the scale of the data. That fixed number gives the
slope a standard error without estimating anything, so `t` below is
comparable across data sets -- which an R-squared here is not: the
irreducible variance of `log(chi^2_1)` is so large that a real and
strong pattern still explains only a few percent of it.

`spread_ratio` is the residual spread at the 90th percentile of the
fitted values against the 10th. It is the number that says whether the
pattern matters in practice, where `t` says whether it is there at all.

The variance claim holds only under the normal-errors contract that
`mboostrs.families.Gaussian` and
`mboostrs.families.GaussReg` fit. A Gamma or Poisson fit is
expected to show a spread that grows with the mean even when it is exactly
right, so the raw residual spread is not a test of heteroscedasticity
there; this diagnostic refuses those families by name rather than return a
number that reads as one.

`by_column` repeats the slope test against each numeric predictor of the
data, whether or not the mean model uses it. A predictor that moves the
spread without moving the mean (a column that enters only the variance,
such as the number of households behind a block-group average) is found
there and not in the fitted-value slope. `columns` names the predictors
to test; by default every column that `mboostrs.classify_columns`
describes as a predictor is a candidate, with the response left out.
`transforms` maps a column to the scale its term is fitted on
(`{"households": "log"}` tests the slope on `log(households)`). A
candidate that cannot be tested is listed in `untested_columns` with the
reason. `tested_columns` lists the columns that have a row in
`by_column`, and each row gives the `transform` it was tested on.

Each column also has `curve_removed`: the mean held-out reduction of
squared log-squared-residual error from eight quantile-bin means over
five fixed half-splits (seeds 0 through 4). Edges and means use only each
fit half. A positive value nominates a possible varying scale even when
its signed slope cancels, as for a symmetric U-shaped pattern. The five
splits overlap; this number is a screening score, not a significance
test or an independent-fold standard error. Discovery still requires its
held-out Gaussian location-scale comparison before adopting the term.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/explore.py` lines 2670-2759](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/explore.py#L2670-L2759)
