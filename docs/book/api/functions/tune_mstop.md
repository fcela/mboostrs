# `tune_mstop`

Defined in `mboostrs.explore`; exported as `mboostrs.tune_mstop`.


*Function*

```python
tune_mstop(
    formula,
    data,
    family=None,
    folds=5,
    start=100,
    nu=None,
    extensions=None,
    max_mstop=10000,
    seed=0,
    step='fixed',
    score='family',
    score_parameter=None,
    unseen_levels='average',
    fold_local=False,
    patience=None,
    fit_budget=None,
    extrapolation='warn',
    flat_tolerance=0.01,
    **fit_kwargs
)
```

Cross-validate the step count, doubling the grid while the risk still falls.
Usage:

```python
import mboostrs as mb
df = mb.datasets.bodyfat()
tuned = mb.tune_mstop("DEXfat ~ bbs(age, df=4)", df)
tuned.mstop  # tuned.at_boundary says whether the risk had turned
```

The model is fitted once per fold at `start` iterations and the
aggregated out-of-fold risk is read at every iteration of that path. If the
minimum lies in the boundary band at the end of the grid (the last
`mboostrs.cvrisk.BOUNDARY_SHARE`, 5%, of its length), the risk had not
turned: the grid is doubled and the sweep repeated until the minimum falls
before the band, never past `max_mstop` (the last grid is cut to it). For
example, a curve lowest at 399 of 400 iterations has not been seen to turn.
A search that stops at its limit with the minimum still in the band warns,
naming the limit that applied (`max_mstop` or `extensions`), because its
`mstop` is where the grid stopped rather than where the data turned.

A grown grid resumes each fold's path where the previous grid stopped: the
default family score, a distributional specification, and a response score
or `fold_local=True` on a model that is not Gaussian Gram-route. A
response-score fold on the Gaussian Gram route keeps no residual to resume
from, and its general-route continuation costs more per step than the Gram
refit, so that fold is refitted from zero at each grid, as the refit search
did.

The fit count is `folds + 1` when every fold resumes: one fit per fold for
the sweep, and one fit on the full data (the final fit at the chosen grid,
or for a distributional specification the fit the search's refit arguments
come from). A Gram-route response fold adds one fit per later grid.
`fit_budget` is checked before any fit against the worst case, which for
a response score counts every grid up to the limit.

**Parameters**

- `formula` (str): A `gamboost` formula.
- `folds` (int, ndarray or FoldScheme): A fold count (random k-fold on `seed`), an `(n, B)` in-bag weight
  matrix, or a scheme from `mboostrs.folds`, such as
  `grouped_folds` for prediction in new groups.
- `start` (int): The first grid's largest iteration.
- `nu` (float, optional): Step length. `None` (default) is the family's own default where it
  sets one (`Quantile`: 0.3), else 0.1.
- `extensions` (int, optional): How many times the grid may double; `None` (default) doubles until
  the risk turns or `max_mstop` is reached.
- `max_mstop` (int): The largest step count searched.
- `step` ({"fixed", "line_search"}): The step rule (see `mboostrs.boost_control`). `"line_search"`
  scales each step by a one-dimensional search, which keeps a Poisson,
  negative binomial or gamma fit of a large response from overflowing
  its inverse link in the first steps.
- `fit_budget` (int, optional): Maximum model fits allowed for the first grid, `folds + 1` (see the
  note above). The call refuses before any fit when the count is larger.
- `**fit_kwargs`: Passed to `gamboost`: `weights`, `env`, `na_action` and so on.
- `score` ({"family", "mse", "brier", "log_loss", "pinball", "expectile"}): Stopping criterion. Response scores fit each fold's recipes on its
  training rows and evaluate every path position on its held-out rows.
  Pinball/expectile require `score_parameter`. Response-score tuning
  currently requires a plain response column, scalar predictions or
  Multinomial probabilities; transformed-response scales are refused.
- `unseen_levels` ({"average", "zero_effect", "error"}): Held-out group policy for response-score paths.
- `fold_local` (bool, default False): Also evaluate family loss through fold-local prediction, using the
  same recipe, held-out mask and unseen-level semantics as `cv_risk`.
  Response scores always use this route. The default family path
  retains the native mboost-compatible CV behavior.
- `patience` (int, optional): Stop once the aggregated held-out risk has not improved for this many
  iterations, growing the grid only to the minimum plus `patience`
  rather than doubling it. The selected `mstop` equals the full grid's
  unless the risk falls below its minimum again more than `patience`
  steps later. `None` (default) uses the doubling rule above. Applies
  to the default family score.
- `extrapolation` ({"warn", "error", "allow", "clip"}, default "warn"): How a held-out row outside its fold's training range is predicted, as
  in `GAMBoostModel.predict`. It applies to the fold-refit
  predictions: `fold_local=True` and the response scores. The default
  family path builds every fold's basis from the full data, so no
  held-out row can leave it and the argument has no effect there.
- `flat_tolerance` (float or None, default 0.01): On the default family path, warn when the chosen step count's
  cross-validated risk is within this relative gap of the intercept-only
  risk at step 0, that is ``|risk(mstop) - risk(0)| <= flat_tolerance *
  |risk(0)|``. Such a curve is flat: the fit is close to a constant, and
  the warning says so. `None` turns the check off.

**Returns**

MstopTuning
    `mstop` and the curve, with `at_boundary` true when the risk was
    still falling at the end of the last grid.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`
- `NotImplementedError`

Source: [`python/mboostrs/explore.py` lines 1424-1648](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/explore.py#L1424-L1648)
