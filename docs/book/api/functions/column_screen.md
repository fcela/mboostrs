# `column_screen`

Defined in `mboostrs.terms`; exported as `mboostrs.column_screen`.


*Function*

```python
column_screen(data, response, columns=None, *, output='auto')
```

Scatterplot diagnostics of the response against each column.

For each numeric, boolean or categorical column (a categorical column is
read through its level codes), the Rust core measures how much of the
response's variance a straight line, a curve (means in up to 20 bins of
equal counts, adjusted for the bins' degrees of freedom) and the best
single step each explain, the Pearson and
Spearman correlations, the column's skewness and its distinct values.
These are marginal: a column correlated with another carries some of its
effect, which a model fitted to all columns separates.

Columns
-------
`linear_r2`, `curve_r2`, `step_r2`
    Shares of the response variance explained.
`curvature`
    `curve_r2 - linear_r2`: evidence for a smooth over a line.
`step_share`
    What a single step adds over the line, relative to `curvature`:
    near one or above when a jump carries the curve.
`hint`
    `"none"` (the curve explains under 1%), `"levels"` for a factor,
    `"line"`, `"step"` or `"curve"`, from the shares above; a hint
    for which shapes to test, not a decision.

Returns a data frame (see `output`) sorted by `curve_r2`.

Source: [`python/mboostrs/terms.py` lines 1503-1562](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/terms.py#L1503-L1562)
