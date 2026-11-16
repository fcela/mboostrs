# `cv_risk`

Defined in `mboostrs.explore`; exported as `mboostrs.cv_risk`.


*Function*

```python
cv_risk(
    rhs,
    data,
    response,
    family=None,
    folds=5,
    mstop=500,
    nu=0.1,
    seed=0,
    weights=None,
    score='family',
    na_action='error',
    score_parameter=None,
    observation_ids=None,
    return_result=False,
    retain_arrays=True,
    unseen_levels='average',
    extrapolation='warn',
    step='fixed',
    response_transform=None,
    max_mstop=None,
    mstop_grid=None,
    estimator='boosting',
    **fit_kwargs
)
```

Out-of-fold risk of a right-hand side.

With a distributional family (`GaussianLSS`, ...), `rhs` may instead be a
mapping of parameter name to right-hand side — ``{"mu": "bbs(x)",
"sigma": "1"}`` — or a plain string shared by every parameter, which is
what a single-formula `gamboostLSS` call means. The score is then the
family's own joint proper score, the negative log predictive density
(`score='family'` is required; the single-quantity scores name a
different target), and the result's per-record predictions are one value
per distribution parameter, in the family's `param_names` order. A row
missing in any parameter's formula is dropped on both sides of the fold
under `na_action='omit'`.

A survival family (`CoxPH`, `Weibull`, `Loglog`, `Lognormal`, `Gehan`,
`Cindex`) takes a `Surv(time, status)` response, given as that text or as
a `(time, status)` pair. It is scored by its family loss
(`score='family'`) on the link scale, and each held-out record carries
the `status` as `event`.

With `return_result=True`, a fold whose training or evaluation side is
empty after `na_action='omit'` stays in the result as a record with
`status='failed'` and its error. The result's score is then NaN, because
a mean over the remaining folds would describe fewer folds. The float return
raises instead. Other errors (a response the family cannot read, an
`unseen_levels='error'` refusal) raise in both cases.

The same folds are used for every call with the same `seed`, `folds`
and row count, so two specifications are compared on the same partition.

**Parameters**

- `folds` (int, ndarray or FoldScheme): A fold count, an `(n, B)` in-bag weight matrix, or a scheme from
  `mboostrs.folds`. Pass a scheme to score a specification the way the
  data is actually structured -- grouped, stratified, or in time order.
  A bare matrix contains resampling multipliers: multiply its column
  by `weights` once to obtain training weights. Bootstrap counts
  retain their multiplicities. A `FoldScheme` already contains
  complete training weights and explicit held-out/judge masks; rows
  outside both training and evaluation remain unused.
- `weights` (array-like, optional): Observation weights for training and evaluation. With a constructed
  `FoldScheme`, defaults to its retained `base_weights`; an explicit
  vector must match them and is not multiplied into training twice.
  For a manually constructed scheme without `base_weights`, this
  vector weights evaluation and excludes zero-weight rows, while the
  scheme's training weights are used as supplied.
- `score` ({"family", "mse", "brier", "log_loss", "pinball", "expectile"}): `"family"` evaluates family loss on link predictions. `"mse"`
  evaluates squared error on response predictions; for Binomial this
  is the Brier score, with signed {-1, +1} labels converted to {0, 1}.
  `"brier"` names that probability score explicitly and extends it to
  the sum of squared class-probability errors for Multinomial.
  `"log_loss"` evaluates Bernoulli log loss on response probabilities.
  `"pinball"` and `"expectile"` evaluate their named asymmetric
  response losses at `score_parameter`. Scores outside `"family"`
  require one response prediction per observation and can compare fitted
  families that target the same held-out quantity.
- `score_parameter` (float, optional): Required strictly between zero and one for pinball and expectile loss;
  rejected for other scores so a silently ignored target cannot enter a
  comparison.
- `na_action` ({"error", "omit"}): Passed to the fit. `"omit"` drops rows with missing values in the
  columns the formula uses, on both sides: a held-out row with a missing
  predictor cannot be scored either.
- `observation_ids` (array-like, optional): Unique record identifiers retained with out-of-fold predictions. Row
  positions are used by default. These identify records, not independent
  sampling units; use a grouped `FoldScheme` for clustered data.
- `return_result` (bool, default False): Return an `EvaluationResult` with predictions and provenance.
  The default float preserves the historical `cv_risk` contract.
- `retain_arrays` (bool, default True): When returning a result, retain one record per held-out prediction and
  the training/evaluation indices per fold. Set false for compact
  provenance when row-level arrays are unnecessary.
- `unseen_levels` ({"average", "zero_effect", "error"}, default "average"): How a held-out row is predicted when its `brandom` level does not
  occur in the fold's training rows, which is every row of a held-out
  group under a grouped fold scheme. `"average"` predicts it as a
  member of an average group, the prediction for a new group (see
  `GAMBoostModel.predict`). `"zero_effect"` gives the level a
  zero coefficient, which reproduces what `model.cvrisk` computes on
  the same folds. `"error"` refuses the fold. Each fold record counts
  the rows this applied to as `n_unseen_level_rows`.
- `extrapolation` ({"warn", "error", "allow", "clip"}, default "warn"): How a held-out row outside its fold's training range is predicted,
  as in `GAMBoostModel.predict`. `"warn"` predicts and reports
  the fold rows that left the range, once per call; `"clip"` evaluates
  the spline at the nearest training value; `"error"` refuses; and
  `"allow"` predicts silently.
- `mstop` (int or "cv"): A fixed budget, or `"cv"` to tune it *inside* each outer fold. A
  fixed budget compares specifications at one number of iterations,
  which favours whichever of them happens to suit it; tuning inside the
  fold compares each specification at its own best stop, and the outer
  score stays honest because the tuning never sees the held-out rows.
  It costs an inner cross-validation per outer fold.
- `max_mstop` (int, optional): With `mstop="cv"`, the largest step count the inner tuning may
  reach; the default is 500. The inner search starts at 500 steps, or at
  this cap when it is smaller, and doubles while the inner risk is still
  in the boundary band at the end of the grid, up to the cap. A fold
  whose search reaches the cap with the risk still falling is reported
  in the fold records (`inner_at_boundary`) and the call warns. Passing
  it with a fixed `mstop` is an error.
- `mstop_grid` (sequence of int, optional): The risk along a grid of step counts from one call. Each fold is
  fitted once to `max(mstop_grid)` steps and its path is cut to every
  grid value, so the grid costs the fits of one `cv_risk` call. `mstop`
  is then not used. Supports every `score` on a standard
  single-response family with `na_action='error'`; other combinations
  raise. The return value is a frame with one row per grid value:
  `mstop`, `risk` (the equal-fold mean, the number `cv_risk(mstop=m)`
  returns for that value), `fold_se` (its standard error over folds)
  and `fold_scores` (each fold's score at that step count, in fold
  order, so paired comparisons across specifications can be made on the
  shared folds).
- `estimator` ({"boosting", "reml"}, default "boosting"): `"boosting"` retains the usual fixed-step fit. `"reml"` fits a
  direct Gaussian GAM independently on each training fold; it accepts
  `score="family"` or `"mse"` and does not use boosting-step,
  response-transform, or other boosting fit options.

**Returns**

float
    Arithmetic mean of the fold scores. Each fold score is normalized
    by its evaluation weight total, so folds contribute equally even
    when their sizes differ. An empty effective training or evaluation
    set raises rather than silently removing that fold from the average.
    With `return_result=True` the same number is
    `EvaluationResult.score`, also readable as
    `EvaluationResult.aggregate`, so the float and the result agree.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`
- `NotImplementedError`
- `FoldError`

Source: [`python/mboostrs/explore.py` lines 458-1054](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/explore.py#L458-L1054)
