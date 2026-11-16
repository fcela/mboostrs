# `acf_residuals`

Defined in `mboostrs.diagnostics`; exported as `mboostrs.acf_residuals`.


*Function*

```python
acf_residuals(
    model,
    max_lag=None,
    powers=(1, 2, 3, 4),
    kind=None,
    newdata=None,
    mstop=None,
    output='auto'
)
```

Compute residual autocorrelation and autocorrelation of selected powers.

The counterpart of `gamlss::acfResid`. Boosting assumes the observations
are independent given the predictors; when they are ordered in time, any
structure the model failed to absorb shows up as correlation between
neighbouring residuals. Checking the *powers* as well as the residuals
themselves separates which moment is still dependent: correlation in `r`
is unmodelled level, in `r**2` unmodelled variance, and in `r**3` and
`r**4` unmodelled skewness and kurtosis.

The result contains the values used in the residual diagnostic panels.

**Parameters**

- `model` (fitted model)
- `max_lag` (int, optional): Highest lag. Defaults to `10 * log10(n)`, as R's `acf` does.
- `powers` (sequence of int): Which powers of the centered residual to examine.
- `kind` (str, optional): Residual type, passed to `residual_frame`. Defaults to
  `"quantile"` where the family supports it and `"response"`
  otherwise, since only quantile residuals are comparable across
  families.
- `output` (str): Frame format, as elsewhere.

**Returns**

frame
    One row per (power, lag): `power`, `lag`, `acf`, `pacf`, and
    `bound` — the two-sided 95% white-noise band `1.96 / sqrt(n)`.
    A value outside the band at a given lag is the signal to look at.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

**Notes**

Order matters and is taken as given: the rows are used in the order they
appear in the data. For an unordered sample the result is meaningless
rather than merely uninformative, so only apply it when the rows have a
sequence.

Source: [`python/mboostrs/diagnostics.py` lines 978-1060](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/diagnostics.py#L978-L1060)
