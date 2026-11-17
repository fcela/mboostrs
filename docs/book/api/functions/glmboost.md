# `glmboost`

Defined in `mboostrs.glmboost`; exported as `mboostrs.glmboost`.


*Function*

```python
glmboost(
    formula: str,
    data,
    family=None,
    control=None,
    weights=None,
    offset=None,
    center: bool = True,
    subsample: float | None = None,
    subsample_seed: int = 0,
    na_action: str = 'error',
    weight_policy: str = 'upstream'
) -> GLMBoostModel
```

Fit a component-wise boosted linear model.

Workflow, pitfalls and worked analyses: `mboostrs.agent_guide()`, or
`python -m mboostrs guide index` from a shell. After a fit, run
`mboostrs.reference_gap` to measure what a flexible model still gains.

Each predictor becomes one base learner holding a single coefficient. At
every iteration the learner that best fits the current negative gradient
is chosen and its coefficient updated by `nu` times the fitted value, so
the path adds one predictor's worth of signal at a time. Predictors never
selected keep a coefficient of exactly zero, which is what makes the fit
sparse.

**Parameters**

- `formula` (str): Patsy-style formula, `"y ~ x1 + x2"`. `"y ~ ."` expands to every
  column except the response. Transformations such as `log(x)` and
  `I(x**2)` are applied before fitting and replayed on new data.
- `data` (frame or mapping): Polars or pandas frame, or a mapping of column name to array.
- `family` (Family, optional): The loss to minimise. Defaults to `mboostrs.families.Gaussian`
  (squared error). The family also fixes the link and the offset.
- `control` (boost_control or tuple, optional): `boost_control(mstop, nu)`. Defaults to 100 iterations at `nu=0.1`.
  `mstop` is the main tuning parameter; choose it with `cvrisk`
  rather than by eye.
- `weights` (array, optional): Non-negative observation weights, one per row. Integer weights are
  read as frequencies -- weight 3 stands for three identical rows -- and
  are used at the scale given. Fractional weights are read as analytic
  weights, where only the ratios carry meaning, and are rescaled to sum
  to the number of weight-positive rows before fitting; see
  `weight_policy`. A zero weight excludes a row while keeping its
  position, so every row-aligned object stays aligned. An all-zero
  vector is rejected: the resulting object would report a completed path
  over an empty sample.
- `offset` (float, optional): A single starting value for the linear predictor, applied to every
  observation. Defaults to the family's own offset. Per-observation
  offsets are not supported; a constant array is read as its one value.
- `center` (bool, default True): Subtract each predictor's mean before fitting, matching R's
  `glmboost(center = TRUE)`. Centering does not change the solution
  the path converges to, only how fast it gets there; `False`
  reproduces R's uncentered example.
- `subsample` (float, optional): Fit on this fraction of rows, drawn once before fitting.
- `subsample_seed` (int): Seed for that draw.
- `na_action` ({"error", "omit", "impute", "impute_model", "impute_model_split"}): What to do with rows holding a missing value: refuse them, drop them,
  or (`"impute"`) fill each missing predictor value with the training
  median and add a `<x>_missing` indicator term, which lets the model
  predict rows with missing values; a missing response still drops its
  row. `"impute_model"` keeps the indicator and fills each numeric
  predictor from a penalized additive regression on the other predictors
  (see `mboostrs.transforms`). `"impute_model_split"` also fits a smooth or linear term on a filled
  column as two terms, one on the rows where it was observed and one on the
  filled rows. It is opt-in: on the claim-cost benchmark it lowered the
  coverage of one effect interval from 0.90 to 0.60. If the data already has a column named `<x>_missing`, the
  indicator is named `<x>_missing_` instead. The indicator is built
  from `<x>` at prediction time, so prediction data and
  `sensitivity` need only `<x>`. `"error"` names the
  offending column and row; `"omit"` drops them and records how many.
- `weight_policy` ({"upstream", "raw"}): How the scale of `weights` is settled before fitting. `"upstream"`
  (the default) reproduces `mboost`'s `rescale_weights`: integer
  weights pass through and fractional weights are rescaled to sum to the
  number of positive weights. `"raw"` passes the vector through
  unchanged, for a caller whose weight scale is itself meaningful (survey
  weights summing to a population total, for example). The scale is not
  cosmetic -- it enters `X'WX` and the risk -- so `"raw"` gives a
  different path from R's on the same fractional weights. The chosen
  policy is on the model as `weight_policy`.

**Returns**

GLMBoostModel
    Call `summary()` for the fit, `coef()` for the coefficients,
    `predict()` for fitted values, and `cvrisk()` to choose `mstop`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `RuntimeError`
- `ValueError`

**Examples**

```python
>>> import mboostrs as mb
>>> data = mb.datasets.bodyfat()
>>> fit = mb.glmboost("DEXfat ~ age + waistcirc + hipcirc", data=data,
...                   family=mb.Gaussian(),
...                   control=mb.boost_control(100, 0.1))
>>> cv = fit.cvrisk(B=10)
>>> tuned = fit[:cv.optimal_mstop]
```

**See Also**

gamboost : the same algorithm with smooth and other structured learners.
gamboostLSS : one linear predictor per distribution parameter.

Source: [`python/mboostrs/glmboost.py` lines 2417-2814](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/glmboost.py#L2417-L2814)
