# `ci_test`

Defined in `mboostrs.causal`; exported as `mboostrs.ci_test`.


*Function*

```python
ci_test(
    data,
    x,
    y,
    given=(),
    folds=5,
    seed=0,
    extrapolation='warn',
    outcome_missing='error'
)
```

Test whether `x` and `y` are independent given `given`.

The generalised covariance measure (Shah and Peters, 2020): regress `x`
on `given` and `y` on `given` with boosted additive models, each
predicted out of fold, and test whether the products of the two residuals
have mean zero. A single covariance misses a dependence that is not
monotone (`y` on `x^2` for a symmetric `x`), so a numeric `x` with
ten or more distinct values is expanded into `x` and hinge functions at
its quartiles, each residualized on `given`, and the four residual
products are tested jointly (a chi-squared test with four degrees of
freedom, the multivariate form of the same test). The statistic is approximately standard normal under
conditional independence when both regressions are accurate; it needs no
linear model and no distributional assumption on `x` or `y`. A
column that predicts the response only through a driver it is correlated
with (a proxy) shows no association once the driver is in `given`.

**Parameters**

- `data` (mapping or data frame)
- `x, y` (str): A numeric column (a binary 0/1 column is fine: its residual is taken
  from its conditional mean), or a categorical column.

  A categorical `x` with more than two levels is tested jointly as a
  factor: each level after the first observed level is one 0/1
  indicator, residualized on `given` like a numeric `x`, and the
  indicators are tested together with a chi-squared test whose degrees
  of freedom equal the number of indicators. A two-level `x` is one
  indicator, tested as a binary column.

  A categorical `y` with two observed levels is one indicator, tested
  as a binary column. A `y` with more than two levels is refused, with
  the per-level route named: pass `data[y] == level` as a 0/1 column for
  each level of interest.
- `given` (sequence of str): Conditioning columns, numeric, boolean, categorical or string. A
  categorical column enters as a factor, coded with treatment contrasts
  as `mboostrs.gamboost` codes a string column. Naming rule: any
  column name is accepted, including one with `=`, a space or a dot.
  The names are never parsed as formula text; each column is copied
  under an internal alias (`__ci_given0`, ...) before the boosted
  regressions are fitted, with its values and type unchanged. `x`,
  `y` and `given` must be distinct names.
- `folds` (int): Cross-fitting folds.
- `extrapolation` ({"warn", "error", "allow", "clip"}): How a held-out row outside the training range of a conditioning
  column is predicted, as in `GAMBoostModel.predict`. The
  regressions' warnings are combined into one per call, which names the
  columns of `given` rather than their internal aliases.
- `outcome_missing` ({"error", "ipw"}): What a missing value of `y` means. `"error"` (the default) refuses
  it, because dropping those rows conditions the test on the rows whose
  outcome was observed, which can bias it when that depends on a
  post-treatment column. `"ipw"` keeps every row and weights each
  observed row by the inverse of a boosted model of P(y observed | given,
  x), clipped to 0.01 to 0.99. The test then assumes the outcome is
  missing at random given `given` and x. Missing values in x or the
  conditioning columns are refused under either setting.

**Returns**

CITest
    `status` is `"dependent"` and `skipped` is True when x or y is an
    exact linear combination of the conditioning set: the test is then not
    run, and `statistic`, `p_value` and `residual_correlation` are NaN.
    The `notes` name the columns of that identity and the one to drop to
    break it (the last conditioning column in the identity). Otherwise
    `status` is `"tested"`. `dropped` names conditioning columns removed
    because they are exact linear combinations of the others, and `notes`
    records each such decision, every regression that reached the step cap,
    and any case in which x or y is itself a combination of the conditioning
    set. The
    `functions` field is the number of residual functions of x that were
    tested: 1 for a numeric x with fewer than ten distinct values, 4 with
    the quartile hinges, and the number of indicators for a categorical x.
    `residual_correlation` is the correlation of the first function of x
    (x itself, or the first indicator of a categorical x) with y's residual.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`
- `TypeError`

Source: [`python/mboostrs/causal.py` lines 5589-5817](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L5589-L5817)
