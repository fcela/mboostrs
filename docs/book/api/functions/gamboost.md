# `gamboost`

Defined in `mboostrs.gamboost`; exported as `mboostrs.gamboost`.


*Function*

```python
gamboost(
    formula: str,
    data,
    family=None,
    control=None,
    weights=None,
    offset=None,
    learners: list | None = None,
    env: dict | None = None,
    subsample: float | None = None,
    subsample_seed: int = 0,
    na_action: str = 'error',
    bins: int | None = None,
    weight_policy: str = 'upstream'
) -> GAMBoostModel
```

Fit a component-wise boosted additive model.

Workflow, pitfalls and worked analyses: `mboostrs.agent_guide()`, or
`python -m mboostrs guide index` from a shell. After a fit, run
`mboostrs.reference_gap` to measure what a flexible model still gains.

Generalises `glmboost` to base learners that are not single
coefficients: P-splines, monotone splines, random effects, spatial
surfaces, trees and user-supplied designs. One learner is selected and
updated per iteration, so the fit stays additive and each term's
contribution can be read off separately.

Selection is fair between learners only when they have equal degrees of
freedom, and that is something the caller arranges rather than something
this function does. A spline with more `df` than a linear term wins
selection partly for being more flexible, not only for fitting better.
`bbs` defaults to `df=4`; `bols` is unpenalized by default and carries the
df its design implies, so `"y ~ bbs(x) + bols(z)"` is *not* an equal-df
menu. Write the `df` you want on every term when the comparison between
them is the point.

**Parameters**

- `formula` (str): Formula whose terms name learners, `"y ~ bbs(x, df=4) + bols(z)"`.
  A bare numeric name becomes `bbs(name)` — a P-spline at the default
  `df=4`, matching R's `gamboost` — so `"y ~ x"` fits a smooth, not
  a line. Write `bols(x)` for a linear term. A bare *categorical* name
  is coded as `brandom(name, lambda=0)`, one unpenalized effect per
  level (the fit `bols(name)` gives), and a `UserWarning` names the
  column; its levels are never numbered, because a smooth over numbered
  levels would fit an order that is not there. `"y ~ ."` expands to
  every column except the response.
- `data` (frame or mapping): Polars or pandas frame, or a mapping of column name to array.
- `family` (Family, optional): The loss to minimise. Defaults to `mboostrs.families.Gaussian`.
- `control` (boost_control or tuple, optional): `boost_control(mstop, nu)`. Defaults to 100 iterations at `nu=0.1`.
- `weights` (array, optional): Non-negative observation weights, one per row. Integer weights are
  read as frequencies -- weight 3 stands for three identical rows -- and
  are used at the scale given. Fractional weights are read as analytic
  weights, where only the ratios carry meaning, and are rescaled to sum
  to the number of weight-positive rows before fitting; see
  `weight_policy`. A zero weight excludes a row while keeping its
  position, so every row-aligned object stays aligned. An all-zero
  vector is rejected: the resulting object would report a completed path
  over an empty sample.
- `offset` (float, optional): A single starting value for the linear predictor. Per-observation
  offsets are not supported; a constant array is read as its one value.
- `learners` (list, optional): Learner objects (`Bbs`, `Bols`, `Bmrf`, ...) given directly
  instead of being parsed from the formula. Use with `"y ~ ."` when a
  learner needs an argument awkward to write in a formula, such as a
  penalty matrix.
- `env` (dict, optional): Names resolved inside formula learner arguments, for example the
  `K` matrix of `bmrf(district, K=K_matrix)`.
- `subsample` (float, optional): Fit on this fraction of rows, drawn once before fitting.
- `subsample_seed` (int): Seed for that draw.
- `na_action` ({"error", "omit", "impute", "impute_model", "impute_model_split"}): What to do with rows holding a missing value: refuse them, drop them,
  or (`"impute"`) fill each missing predictor value with the training
  median and add a `<x>_missing` indicator term, which lets the model
  predict rows with missing values; a missing response still drops its
  row. `"impute_model"` does the same with a model-based fill: each
  numeric predictor with missing values is filled from a penalized
  additive regression on the other predictors, fitted on the training
  rows where it is observed. `"impute_model_split"` also fits a smooth or linear term on a filled
  column as two terms, one on the rows where it was observed and one on the
  filled rows. It is opt-in: on the claim-cost benchmark it lowered the
  coverage of one effect interval from 0.90 to 0.60. If the data already has a column named
  `<x>_missing`, the indicator is named `<x>_missing_` instead. The indicator is built
  from `<x>` at prediction time, so prediction data and
  `sensitivity` need only `<x>`.
- `bins` (int, optional): Global quantile-bin count for supported continuous `bols`/`bbs`
  learners and per-axis count for `bspatial`. A learner's explicit
  `bins=` value takes precedence. Random-intercept learners are always
  stored through their exact discrete bins. Terms with no binned design
  -- a `by=` on any learner, an intercept-carrying `bols`, a
  shape-constrained `bbs`, and every learner class outside those four
  -- keep their exact storage; the first three are named in a warning,
  because those are the ones a caller writing `bins=` expects to be
  binned.
- `weight_policy` ({"upstream", "raw"}): How the scale of `weights` is settled before fitting. `"upstream"`
  (the default) reproduces `mboost`'s `rescale_weights`: integer
  weights pass through and fractional weights are rescaled to sum to the
  number of positive weights. `"raw"` passes the vector through
  unchanged, for a caller whose weight scale is itself meaningful (survey
  weights summing to a population total, for example). The scale is not
  cosmetic -- it enters `X'WX`, the `df -> lambda` calibration, and
  the risk -- so `"raw"` gives a different path from R's on the same
  fractional weights. The chosen policy is on the model as
  `weight_policy` and in `summary()`.

**Returns**

GAMBoostModel
    A spline learner has no single coefficient, so read its fit with
    `partial_effects()` or `mboostrs.plot.plot_partial_effects`
    rather than `coef()`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `RuntimeError`
- `ValueError`
- `NotImplementedError`

**Examples**

```python
>>> import mboostrs as mb
>>> data = mb.datasets.bodyfat()
>>> fit = mb.gamboost("DEXfat ~ bbs(age, df=4) + bbs(hipcirc, df=4)",
...                   data=data, family=mb.Gaussian(),
...                   control=mb.boost_control(200, 0.1))
>>> effects = fit.partial_effects()
```

**See Also**

glmboost : linear terms only, one coefficient each.
gamboostLSS : one additive predictor per distribution parameter.

Source: [`python/mboostrs/gamboost.py` lines 4944-5645](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamboost.py#L4944-L5645)
