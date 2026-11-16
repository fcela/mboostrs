# Penalized GAM by REML without boosting

`mb.gam(formula, data, family=..., na_action=..., method="REML")` fits a
penalized additive model from a formula in one step. Every term of the
formula is in the fit. Each smooth has a smoothing parameter, and REML
(restricted maximum likelihood, the criterion that mgcv's
`gam(method="REML")` maximizes) chooses all of them. No boosting runs, no
iteration count (`mstop`) is chosen, and no term is dropped. Use it when the
terms are fixed before the fit: a prespecified model, or the model that
earlier analysis chose on other rows. When the data should choose the terms,
boost first and refit the chosen terms with `gam_refit`; that interval is
conditional on the choice.

| Question | Call | Returns |
|---|---|---|
| Fit the model | `fit = mb.gam("y ~ bbs(x) + bols(g) + bspatial(lon, lat)", data)` | a `GamRefit` |
| Average change when a covariate moves | `fit.contrast("x", 2.5, 6.0, data=rows)` | `EffectContrast`: `estimate`, `se`, `lower`, `upper` (link scale) |
| Same, several covariates moved together | `fit.contrast(("lat", "lon"), (37.8, -122.3), (34.2, -117.4), data=rows)` | as above |
| Prediction with interval | `fit.predict(rows, interval=True, level=0.9)` | `estimate`, `se`, `lower`, `upper`, `outside_range` per row |
| Each term's effect at rows | `fit.effects(rows, level=0.9)` | `{term: (estimate, lower, upper)}` |
| Coefficients, covariance, term tests | `fit.coef()`, `fit.vcov()`, `fit.summary()` | table, matrix, table |
| Two or more distribution parameters | `mb.gamlss({"mu": "y ~ bbs(x)", "sigma": "y ~ bbs(z)"}, data, family=mb.GaussianLSS())` | `GamlssRefit`, with `predict(parameter=...)`, `contrast(..., parameter=...)`, `draws` |

```python
import mboostrs as mb

body = mb.datasets.bodyfat()
fit = mb.gam("DEXfat ~ bbs(hipcirc, knots=6) + bbs(waistcirc, knots=6)", data=body)
change = fit.contrast("hipcirc", 90.0, 110.0, data=body)
print(round(change.estimate, 3), round(change.lower, 3), round(change.upper, 3))
```

## Terms

- `bbs(x, knots=k)`: a P-spline smooth. A bare numeric name `x` means
  `bbs(x)`. The second-difference penalty leaves a straight line unpenalized,
  so the smoothing parameter shrinks the smooth toward a line.
- `bols(g)`: a linear term. For a categorical column it is a factor with one
  coefficient per level after the first, and no penalty. A bare categorical
  name means `bols(g)`.
- `brandom(g)`: one coefficient per level, shrunk toward zero by REML.
- `bspatial(lon, lat, knots=k)`: a smooth surface, the tensor product of two
  P-spline margins with one smoothing parameter each.
- `a %X% b`: a tensor product of two smooths, for example
  `bbs(age, knots=6) %X% bbs(income, knots=6)`. Each margin has its own
  smoothing parameter.
- `by=` with a numeric column: a varying-coefficient term.
- A transform inside a term, such as `bbs(log(income))`, is evaluated on the
  data and on new rows.

Supported only through the boosted refit: `btree`, `bmono`, `bkernel`,
`buser`, `bconstrained`. `mb.gam` refuses them by name.

Write `bbs(x)` alone for a smooth. A `bols(x)` added to a `bbs(x)` over the
same column puts the linear part in both terms. The fit then warns that
unpenalized directions are unidentified and names the terms that share them.

## Families and method

`family` is `"gaussian"`, `"binomial"`, `"poisson"` or `"gamma"`, or a family
object `mb.Gaussian()`, `mb.Binomial()`, `mb.Poisson()`, `mb.GammaReg()`. The
links are the canonical ones: identity, logit, log and log. Binomial takes
`{0, 1}` responses. Negative binomial and Cox survival are not in `mb.gam`;
fit them with `mb.gamboost` and `mb.gam_refit`. `method` accepts only
`"REML"`; any other value raises.

`weights=` gives prior weights per row. `offset=` gives a per-row link-scale
offset such as log exposure; it is added to the linear predictor and not
estimated.

## Missing values

`na_action` decides what happens to a formula column with a missing value.

- `"error"` (default): refuses, naming the column and its first missing row.
- `"omit"`: drops the rows with a missing value in any formula column.
- `"impute"`: fills a numeric column with its training median and adds an
  indicator term `bols(<column>_missing, intercept=False)` for the rows that
  were missing. The fill is replayed on new rows in `predict` and `contrast`.

## Contrasts and intervals

`fit.contrast(variable, from_value, to_value, data=rows)` takes every row of
`data`, sets the covariate to `from_value` in one copy and to `to_value` in
another, leaves every other column at the row's value, and returns the mean of
the difference of the link-scale predictions. `estimate` is that mean.
`se` is its standard error from the full coefficient covariance, so the terms
a covariate enters through are combined with their covariance. A covariate no
term reads has a change of exactly zero with zero standard error. A factor
contrast takes level labels: `fit.contrast("roof", "A", "D", data=rows)`.

The intervals are Bayesian pointwise intervals (Wahba 1983; Nychka 1988).
They treat the smoothing parameters REML chose as known, as
`GamRefit.effects` does. `predict(..., corrected=True)` widens them by the
uncertainty of those smoothing parameters.

`predict(..., extrapolation=)` checks each spline covariate against its
training range. A B-spline continues by its edge slope outside its boundary
knots, which the fit does not constrain. The default `"warn"` predicts and
warns; `"error"` refuses; `"clip"` holds the covariate at the training
boundary; `"allow"` predicts without a message.

## Cost

The fit forms the Gram matrix `X'WX` of the design once (one pass over the
rows, cost proportional to rows times coefficients squared), then runs its
REML iterations on the coefficients only, so the iterations do not depend on
the row count. The total number of coefficients is the sum of the column
counts of all terms. A `bspatial` or `%X%` surface has the product of its
margins' column counts, so `knots=20` (24 columns a margin, 576 for a
surface) is the expensive choice; a margin with 4 to 8 knots is usually
enough for a surface that is not sharply peaked.

Measured on this library's development machine (shared, load average 60 to
200 during the runs), with about 130 coefficients: a fit and predict of
15,480 rows with one term dropped took 0.4 to 1.1 seconds; a fit of 500,000
rows took about 25 seconds and predicting them about 2 seconds; the
distributional fit `mb.gamlss` took about 160 seconds at 500,000 rows.

## Checks

Read `fit.summary()` for the edf of each term. An edf at the number of
columns of the basis means REML left the term unsmoothed; an edf near 1 for
a smooth means it is almost linear. `fit.convergence` reports whether REML
converged; a warning from the fit says when it did not.
