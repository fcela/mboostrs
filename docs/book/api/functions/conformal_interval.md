# `conformal_interval`

Defined in `mboostrs.conformal`; exported as `mboostrs.conformal_interval`.


*Function*

```python
conformal_interval(
    model,
    newdata=None,
    level=0.9,
    method='split',
    seed=None,
    calib_fraction=0.5,
    n_folds=10,
    mstop=None,
    calibration=None,
    extrapolation='warn',
    score='auto',
    count_reference='zero_inflated_poisson',
    unseen_levels=None
)
```

Distribution-free prediction intervals for a fitted standard model.
Usage:

```python
import mboostrs as mb
fit = mb.gamboost("y ~ bbs(x, df=4)", data, family=mb.Gaussian())
ci = mb.conformal_interval(fit, level=0.9, seed=1)
ci.lower, ci.upper
```


**Parameters**

- `model` (GLMBoostModel, GAMBoostModel or LSSModel): A fitted model. It is used as a *specification*: the method refits it
  on subsets of the training data, so the intervals are calibrated on
  data the corresponding fit never saw.
- `newdata` (frame or dict, optional): Where to predict. Defaults to the training data.
- `level` (float, default 0.9): Nominal marginal coverage. CV+ reports its generally lower
  finite-sample bound separately when folds are balanced.
- `method` ({"split", "cv+", "cqr", "distributional"}, default "split"): See the module docstring. Use `"cqr"` for heteroscedastic noise,
  `"cv+"` when data are scarce, `"split"` when fits are expensive.
  `"split"` stays the default: its band has one width, so it warns
  when a covariate range or a fitted scale region is under-covered,
  and `"cqr"` or `"cv+"` are the remedies. `"cqr"` is the
  slowest of the three: it refits
  two quantile models, each with its own cross-validated stopping
  search, about 3.5 s against 0.01 s for `"split"` at n = 10,000.
  Pass `mstop=` to fix the step count and skip the search.
  For a `glmboostLSS`/`gamboostLSS` fit the method is
  `"distributional"` (the default `"split"` selects it): the fitted
  distribution's central band, from a refit on the proper-training
  rows, is widened by the conformal quantile of how far calibration
  responses fell outside it. Coverage then holds even if the family is
  wrong.
- `seed` (int, optional): Seed for the split or fold assignment, and for the count-rank
  score's tie randomization when the default `score="auto"` resolves
  to it: the calibration rows and the query rows each draw one uniform
  per row from a stream keyed by the seed, so the same rows and seed
  reproduce the interval exactly -- an internal split and its
  equivalent held-out calibration agree only when both carry the same
  seed. `None` (the default) draws and reports a seed in
  `randomization_seed` for the count score, and draws the split or
  fold assignment, so the call is not reproducible until a seed is
  given.
- `calib_fraction` (float, default 0.5): Share of training rows held out for calibration (`"split"`, `"cqr"`).
- `n_folds` (int, default 10): Number of folds (`"cv+"`).
- `mstop` (int or "tune", optional): Iteration to predict at. `None` (default) uses each refit's full
  path, the model's fitted step count; the guarantee then assumes that
  count was not chosen using the calibration rows, which is false if
  the model was tuned by cross-validation on all its training rows.
  `"tune"` re-chooses the step count for every proper-training fit
  by 5-fold cross-validation on that fit's own rows, so no calibration
  response enters the stopping decision. `method="cqr"` always tunes
  its quantile fits this way unless an integer is given. The returned
  `mstop_source` records which applied.
  Inner cross-validation relearns preprocessing within each training
  fold. CV+ and CQR predict held-out and query rows through each
  corresponding refit's own recipe. Response centering/scaling currently
  requires an integer step count (or None for split/CV+): inner CV refuses
  family risks whose response units differ between folds. CQR with either
  transform therefore requires an explicit integer.
- `calibration` (frame or dict, optional): Rows held out before the model's specification, tuning and fit were
  chosen, with the response and every predictor. With
  `method="split"`, the model is used as fitted, without a refit: the
  calibration scores use `score` on these rows. Every
  choice made on the fitting rows (formula search, family, `mstop`)
  is then free of calibration responses, which is what the guarantee
  needs and what a refit of a specification chosen on all rows cannot
  provide. The caller is responsible for these rows never having been
  used to fit or select the model; `calibration_source` records that
  claim.
  For an LSS model, `"split"` and `"distributional"` calibrate its
  fitted central quantiles, applying its existing predictor recipe to
  the held-out rows without learning it again. Neither path refits or
  retunes the model. Distributional calibration also supports a response
  transform by returning the quantiles to the original response scale.
  In that transformed-response case `predicted` is the original-scale
  median clipped to the band, and `fitted_mean` is unavailable: the
  inverse-transformed location is generally not a response mean.
- `extrapolation` ({"warn", "clip", "error", "allow"}, default "warn"): Policy for rows whose spline covariates lie outside the training range
  of the refit or model that predicts them, as in `predict`. It applies
  to the interval's own predictions, the calibration predictions and the
  `fitted_mean`, so they all use one policy. `"clip"` holds such a
  covariate at the nearest training value, which bounds the interval and
  the fitted mean; `"warn"` continues the spline's edge slope.
- `score` (str, default "auto"): Nonconformity score for standard-model split calibration. Automatic
  scoring uses count ranks for standard Poisson/NBinomial fits with an
  untransformed count response, and absolute scoring for other models.
  Its reference and eligibility are learned on proper-training rows.
  A proper-training residual-to-reference variance ratio above 2 uses
  normalized scoring instead. Explicit legacy scores keep their definitions.
  The
  normalized count score divides absolute residuals by
  `sqrt(max(predicted_mean, 1e-12))` and multiplies the calibrated
  correction by that scale at prediction. The quantile score calibrates
  the central Poisson quantile band at the fitted mean. Both require
  nonnegative integer calibration responses and finite nonnegative
  predicted means. They retain marginal conformal validity without
  assuming the Poisson variance model is correct; conditional coverage
  is not guaranteed. Endpoints remain real-valued. Intersect every
  compared method with known count support when measuring count-set
  size. These options require an untransformed response and do not
  apply to CV+, CQR or LSS models. `quantile` uses standardized
  residual units for the normalized score and response units otherwise.
  `count_rank` uses a mean-preserving fitted count CDF and independent
  auxiliary uniforms. Its correction is in probability-rank units, its
  bounds are real and its target is nonnegative integers. The result
  records `support`, `count_size`, `zero_probability`,
  `reference_dispersion` and `randomization_seed`. A singleton below
  zero represents an empty count set. Coverage averages over responses
  and auxiliary randomness; it does not promise conditional coverage.
  Supply `seed` in [0, 2**53-1] to reproduce the randomization.
  `gaussian_normalized` is an experimental GaussianLSS split score:
  absolute residual divided by the fitted positive scale, inverted as
  location plus/minus the calibration cutoff times scale. It requires an
  untransformed response and uses standardized residual units. Its
  calibration is marginal and does not identify an irreducible noise scale.
- `count_reference` ({"zero_inflated_poisson", "poisson"}, default "zero_inflated_poisson"): Reference for count-rank scores. The first fits one zero-inflation
  probability on proper-training rows while retaining the fitted mean.
  The second fixes that probability at zero. Neither assumes that the
  reference distribution is correct for marginal conformal validity.
- `unseen_levels` ({"error", "average", "zero_effect"}, default "error"): Policy for factor or group levels present in `newdata` or
  `calibration` that were absent from training, as in `predict`.

**Returns**

ConformalInterval

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`
- `NotImplementedError`

**Notes**

Coverage is marginal over a fresh exchangeable test point, not training
coverage and not conditional coverage at each x. The default absolute
split score gives constant width; Poisson scores, CV+ and CQR can vary
with x. The specification and mstop
must be fixed without calibration responses: pass `calibration=` with
rows held out before the model was chosen to make that hold by
construction. Read `guarantee`, `guaranteed_coverage`,
`assumptions`, `mstop_source` and `calibration_source` on the
returned object.

`predicted` is the point prediction attached to the interval. For `method="split"`,
`"cqr"` and `"distributional"` it comes from the proper-training refit,
the fit on the rows not used for calibration: the refit's prediction for
split, the midpoint of its two quantile fits for CQR, and its fitted
median for distributional. For `method="cv+"` and for `calibration=`
it is the full model's prediction. `fitted_mean` holds the full model's
prediction at the same rows, whatever the method, so the two can be
compared. With `score="poisson_quantile"`, `predicted` remains the
fitted mean and need not be the centre of the calibrated quantile band.
With a response transform, bounds, calibration scores and `predicted`
use original-response units. Each proper-training fit inverts predictions
with its own fitted response parameters before scoring the raw observed
response. CQR orders the inverse-transformed candidate functions before
calibration. Response centering/scaling and predictor preprocessing are
re-estimated on proper-training rows. `fitted_mean` is None for transformed
fits: the original-response expectation is not in general the inverse of
the fitted location. A standard fit's inverse prediction is a point
function; a distributional fit supplies an original-response median.
Inversion outside a transform's domain, or a nonfinite inverse prediction,
raises ValueError. Standard `predict(type='response')` retains its usual
transformed-response meaning; this conversion belongs to calibration.

Every method treats rows as exchangeable. Rows from the same cluster,
repeated measurements of one unit, and time-ordered rows are not: a
calibration row from a cluster that also appears in training understates
the error for a new cluster, and a later time point is not drawn like an
earlier one. None of the methods here applies to such data, and none
accepts a grouping or time argument.

CV+ needs folds of equal size for its finite-K bound. When the number of
rows is not a multiple of `n_folds`, the remainder (at most
`n_folds - 1` rows, chosen at random with `seed`) is left out of the
calibration and reported as `n_set_aside`.

**Examples**

```python
>>> import numpy as np, mboostrs as mb
>>> rng = np.random.default_rng(0)
>>> x = rng.uniform(-2, 2, 500)
>>> data = {"x": x, "y": np.sin(x) + rng.normal(scale=0.3, size=500)}
>>> fit = mb.gamboost("y ~ bbs(x, df=4)", data=data, family=mb.Gaussian(),
...                   control=mb.boost_control(100, 0.1))
>>> ci = mb.conformal_interval(fit, level=0.9, seed=1)
>>> ci.lower.shape == ci.upper.shape == (500,)
True
```

Source: [`python/mboostrs/conformal.py` lines 1423-1738](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/conformal.py#L1423-L1738)
