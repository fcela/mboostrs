# `select_capacity`

Defined in `mboostrs.capacity`; exported as `mboostrs.select_capacity`.


*Function*

```python
select_capacity(
    formula,
    data,
    family=None,
    folds=5,
    seed=0,
    mstop=500,
    nu=0.1,
    pairs=3,
    tree=True,
    tree_depth=3,
    budget='tuned',
    max_mstop=None,
    tolerance=2.0,
    score='family',
    extrapolation='warn',
    estimators=('boosting',),
    **fit_kwargs
)
```

Extend a formula with interaction capacity only where it pays.

Three specifications are built on top of `formula` and scored out of
fold on one shared partition:

`base`
    The formula as given.
`+tensors`
    The base plus one penalized tensor per interaction pair the residual
    screen proposed. The screen bins half the rows, takes cell means of
    the base fit's Pearson residual there and measures the mean squared
    error those means remove on the other half, so a pair that fits only
    noise removes nothing out of sample. A pair is proposed only when it
    removes a positive amount, and at most `pairs` of them are taken,
    best first. Each pair enters as
    `bbs(a, df=2.5) %X% bbs(b, df=2.5)` (`TENSOR_MARGINAL_DF`; squared
    it clears the tensor penalty's null space of 4, see the `pitfalls`
    guide topic). A column with two or fewer distinct values cannot be a
    spline margin -- its boundary knots collapse and the fit refuses --
    so pairs over such columns are never proposed.
`+tree`
    The `+tensors` rung plus one joint tree, ``btree(..., maxdepth=
    tree_depth, selection=sse)``, over every column the base formula
    reads, factors included: a factor can be the half of an interaction
    no numeric-only tree can express, which is how the bike panels'
    hour-by-working-day structure hid from the first chooser. The tree
    can split on several columns at once, so it stands for interaction
    structure that the pair screen did not nominate.

The rungs are nested: `base` is inside `+tensors`, which is inside
`+tree`. By default every boosting rung is searched for its own best
step count out of fold and scored there (`budget="tuned"`): the grid
starts at `mstop` steps and doubles while the rung's held-out score is
still falling at its largest point, up to `max_mstop` (four times
`mstop` by default), and the rung is scored at the best point of that
grid, by `mboostrs.cv_risk`, which shares one fold partition
across calls with the same `seed`, `folds` and row count -- so the
scores are paired fold by fold. With `budget="shared"`, every rung
is fitted and scored at
the same `mstop` and `nu`, and a rung that needs more steps than
that to show its advantage is not adopted.

The retained rung is the smallest whose mean score is within
`tolerance` standard errors of the best rung's mean. The standard
error is calculated from paired fold differences on the shared split.
If a difference cannot be paired (one fold or a failed fold), the best
rung's marginal fold spread is used. A larger rung is retained only when
its held-out score improves enough to satisfy this rule.

**Parameters**

- `formula` (str): The base formula, `"y ~ bbs(x1) + bbs(x2) + f"`. Bare columns,
  penalized splines and factor terms are all accepted; the rungs only
  add to it. The response must be a plain column.
- `data` (polars or pandas DataFrame, or a mapping of column to values): The training rows.
- `family` (Family, optional): The family of every fit; `Gaussian()` when omitted.
- `folds` (int, ndarray or FoldScheme, optional): The fold partition, passed to `mboostrs.cv_risk`. Random folds
  favour the `+tree` rung on time-ordered rows -- a tree can memorize
  the neighbours of a held-out row -- so pass a scheme from
  `mboostrs.folds` for grouped or time-ordered data, as the
  `modeling-tactics` guide topic requires of any reference
  comparison.
- `seed` (int): Seed of the default random folds, and of the screen's half-split.
- `mstop` (int): Under `budget="tuned"` (the default), the first grid end of every
  rung's step-count search, which doubles from there while the rung's
  score still falls. Under `budget="shared"`, the step count every
  rung is fitted to, an equal budget, and a rung that needs more steps
  than this to lower held-out risk enough to meet `tolerance` is not
  adopted.
- `nu` (float): Step length of every fit.
- `pairs` (int): How many screened pairs at most enter the `+tensors` rung. `0`
  skips the rung.
- `tree` (bool, default True): Whether to build and score the `+tree` rung.
- `tree_depth` (int or sequence of int): `maxdepth` of the joint tree. A sequence, such as `(3, 6)`,
  scores one `+tree` rung per depth and keeps the smallest depth
  within `tolerance` of the best score.
- `budget` ({"tuned", "shared"}, default "tuned"): Whether each boosting rung is scored at its own out-of-fold-tuned
  step count ("tuned") or every rung at the same `mstop` ("shared").
  Tuning runs on `cv_risk`'s step-count grid, which needs a standard
  single-response family, `na_action="error"` and no per-row offset;
  a call outside that set is scored at the shared budget and says so
  in one warning.
- `max_mstop` (int, optional): The largest step count a `budget="tuned"` search may reach; four
  times `mstop` by default. A rung whose score is still falling there
  is scored and fitted at the cap and the result warns that its
  advantage may be understated.
- `tolerance` (float): How many standard errors, of each smaller rung's paired fold
  difference against the best rung's mean, that rung may sit above
  the best and still be kept. The default `2.0` keeps only what
  clearly pays, the same two-standard-error rule the `modeling-tactics`
  guide topic states for a measured gap. Pairing uses the same folds
  for every rung and removes fold variation shared by the two scores.
  `tolerance=1.0` gives the one-standard-error rule.
- `estimators` (sequence of {"boosting", "reml"}): Estimators compared for the base and tensor forms, in tie-break order.
  The default `("boosting",)` preserves the original ladder. Requesting
  `"reml"` adds direct Gaussian `mb.gam` candidates and an
  intercept-only boosted baseline. Trees remain boosting candidates.
- `score` ({"family", "mse", "brier", "log_loss", "pinball", "expectile"}): The held-out score, passed to `mboostrs.cv_risk`. Every rung
  is scored on one target, so the comparison is on one scale.
- `extrapolation` ({"warn", "allow", "clip", "error"}): The out-of-range policy for fold predictions.
- `**fit_kwargs`: Passed to every fit (`weights`, `na_action` and the rest).

**Returns**

CapacitySelection
    The selected form and estimator, the candidate table with scores,
    standard errors and the step count each boosting rung was scored at,
    the screened pairs, and the selected model refitted on all rows.
    A boosting fit uses its rung's tuned step count; REML estimates its
    smoothness.

**Raises**

- `ValueError`: When the formula's left-hand side is not a plain column, the base
rung's cross-validation failed on every fold, or no rung beyond the
base was requested or buildable.

**Examples**

A smooth-times-step interaction, `2 sin(x1) 1(x2 > 0)`: the tensor of
the `+tensors` rung cannot make the step in its `x2` margin, the
joint tree makes it in two splits, so the tree rung is the one that pays:

>>> import numpy as np, polars as pl, mboostrs as mb
>>> rng = np.random.default_rng(3)
>>> n = 800
>>> x1, x2, x3 = (rng.uniform(-2, 2, n) for _ in range(3))
>>> data = pl.DataFrame({
...     "x1": x1, "x2": x2, "x3": x3,
...     "y": 2.0 * np.sin(x1) * (x2 > 0) + 0.4 * x3
...          + 0.3 * rng.normal(size=n)})
>>> pick = mb.select_capacity("y ~ bbs(x1) + bbs(x2) + bbs(x3)", data,
...                           mstop=300)
>>> pick.selected
'+tree'

A smooth response, where the extra capacity does not pay and the base is
kept:

>>> y = np.sin(data["x1"].to_numpy()) + data["x2"].to_numpy() ** 2
>>> smooth = data.with_columns(y=y + 0.4 * rng.normal(size=n))
>>> pick = mb.select_capacity("y ~ bbs(x1, df=8) + bbs(x2, df=6) + bbs(x3)",
...                           smooth, mstop=300)
>>> pick.selected
'base'

Source: [`python/mboostrs/capacity.py` lines 274-638](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/capacity.py#L274-L638)
