# Missing values

Every fitting function refuses a missing predictor value by default
(`na_action="error"`) and names the column and the first row. Three other
policies are built in, for `gamboost`, `glmboost`, `gamboostLSS` and
`glmboostLSS` alike:

- `na_action="omit"` drops every row with a missing value in a column the
  formula reads (in any parameter formula, for a distributional fit). The
  model cannot predict a new row that has a missing value.
- `na_action="impute"` keeps every row. A numeric predictor with missing
  values is filled with its training median, and a term
  `bols(<x>_missing, intercept=False)` is added for its missing-value
  indicator, so the fit estimates how rows with a missing `x` differ from
  rows at the median. A categorical predictor gets a level of its own,
  `"(missing)"`. Both steps are part of the model's predictor recipe, so
  `predict(newdata=...)` fills new rows the same way and an incomplete test
  row still gets a prediction. Rows whose response is missing are dropped.
- `na_action="impute_model"` keeps every row and the same `<x>_missing`
  indicator term, but fills a numeric predictor with a model-based value
  instead of the median. The fill is the conditional mean of `x` given the
  other predictors in the formula. The Rust core fits it on the training rows
  where `x` is observed, as a penalized additive regression: each numeric
  predictor is a cubic B-spline with 20 interior knots and a second-difference
  penalty, each factor predictor is a level design with a ridge penalty, and
  each other predictor with missing values adds a 0/1 missingness column. The
  smoothing parameter is chosen by generalized cross-validation. The response
  is never a predictor. A categorical predictor with missing values gets the
  `"(missing)"` level, as under `"impute"`. The fill is stored in the recipe,
  so `predict(newdata=...)` applies it unchanged and never refits it.
- `na_action="impute_model_split"` is `"impute_model"` plus a split: a
  smooth or linear term on the filled column is fitted as two terms, and the formula
  still reads `bbs(x)`. The term `bbs(x) [observed]` is fitted on the rows
  where `x` was observed (its design is multiplied by `1 - <x>_missing`). The
  term `bbs(x) [filled]` is fitted on the filled rows (its design is multiplied
  by `<x>_missing`). Both parts appear under these names in `selected` and in
  the refits. The observed parts sum to one minus `<x>_missing`, so with the
  intercept the constant `<x>_missing` term is already in their span. Boosting
  keeps that term; `gam_refit` and `gamlss_refit` leave it out and emit a
  warning that names it, because a refit with both would not be identified.
  The span of the refit's columns is the same either way. An effect query or contrast that sets `x` to a value counts every
  row as observed, so only the observed part contributes. A categorical column
  keeps its `"(missing)"` level and is not split, and a tensor term `a %X% b`
  over the filled column uses the fill without a split.
- The indicator is named `<x>_missing`. If the data already has a column with
  that name, the indicator is named `<x>_missing_` instead, and `selected`
  shows that name. Prediction and `sensitivity` build the indicator from `<x>`,
  so the data passed to them needs only the original column `<x>`.

The indicator term is an ordinary learner: it appears in `selected`,
`varimp()` and the refits (`gam_refit`, `gamlss_refit`,
`confint(method="refit")`), and an indicator that boosting never selects
contributes nothing.

A NaN in a float column is a missing value, the same as a null. The
interaction tensor `a %X% b` takes `na_action` as well. Under `"impute"` a
numeric margin's missing values are filled with its training median, recorded
in the term so `predict` fills new rows the same way. Under `"impute_model"`
a margin is filled from the same model as a predictor, because the model
fill is applied before the tensor is built. A tensor margin gets no indicator
term. A missing margin value under the default `"error"` raises and names the
column. A refit (`gam_refit`, `gamlss_refit`) refuses a numeric column with
missing values and names it, so fit with `"impute"`, `"impute_model"` or
`"omit"` before refitting.

`screen_interactions(data, residual, na_action=...)` takes the same values.
`"error"` (default) raises, naming the column, the number of missing values,
the first row and the remedy. `"impute"` and `"impute_model"` fill a numeric
column with its median, and a column named in `groups=` keeps its missing
values as a group of their own. `"omit"` drops the rows with a missing value
in any screened column or a missing residual. A residual cannot be filled, so
under `"error"` and the two imputing policies it must be finite on every row.

```python
import numpy as np, polars as pl
import mboostrs as mb

rng = np.random.default_rng(0)
n = 500
x = rng.uniform(0, 1, n)
z = rng.uniform(0, 1, n)
y = np.sin(3 * x) + 0.5 * z + rng.normal(0, 0.3, n)
x_obs = np.where(rng.uniform(size=n) < 0.1, np.nan, x)       # 10% missing
df = pl.DataFrame({"x": x_obs, "z": z, "y": y})

fit = mb.gamboost("y ~ bbs(x) + bbs(z)", df, na_action="impute",
                  control=mb.boost_control(mstop=300))
print(fit.selected[:5])                     # learners include 'x_missing'
new = pl.DataFrame({"x": [0.5, None], "z": [0.5, 0.5]})
print(fit.predict(newdata=new))             # the second row is filled and predicted

fit_m = mb.gamboost("y ~ bbs(x) + bbs(z)", df, na_action="impute_model",
                    control=mb.boost_control(mstop=300))
print(fit_m.predict(newdata=new))           # the second row uses the model fill

lss = mb.gamboostLSS({"mu": "y ~ bbs(x) + bbs(z)", "sigma": "y ~ bbs(z)"},
                     data=df, families=mb.GaussianLSS(), na_action="impute",
                     control=mb.boost_control(mstop=300))
print(lss.predict(newdata=new, parameter="mu"))
```

When the indicator method is wrong: it is unbiased for prediction when
whether a value is missing is unrelated to the value itself. When
missingness depends on other covariates (for example a field recorded less
often in one region), the indicator absorbs part of that covariate's
effect. `"impute_model"` removes the part of that bias that the other
covariates explain: when the column is predictable from them, the fill
follows the other covariates instead of sitting at the median. It does not
remove a bias that comes from the unobserved value itself. A conditional-mean
fill is also narrower than the column: on the claim-cost benchmark the fills
of the missing ages had a standard deviation of 9.2 years against 13.3 for
the true ages. On that benchmark, over 10 draws, the effect of roof class
(which carries information about the missing age) was less biased under
`"impute_model"` than under `"impute"` (interval coverage 0.80 against
0.30), and the effect of age was as well estimated (0.90 under both). Under
`"impute_model_split"` the age coverage fell to 0.60, which is why the
split is opt-in. Report how many rows each
policy filled or dropped, and compare the two imputing policies with
`compare_specifications` when the column matters.

Refits and resampling use the fill as follows. For a `gamboost` or
`glmboost` model, `cvrisk` with its default `preprocessing="fixed"` reuses
the fill fitted on all training rows, as `"impute"` reuses its medians;
`preprocessing="refit"` refits the fill inside each training fold. `discover`
runs its stopping-iteration search through `cvrisk`, so it uses the fill
fitted on all rows. `compare_specifications` fits each candidate on each
fold's training rows, so its fill is refitted per fold.
`conformal_interval(method="distributional")` refits the fill on the proper
training rows. `gam_refit` and `gamlss_refit` reuse the fill stored in the
model.
