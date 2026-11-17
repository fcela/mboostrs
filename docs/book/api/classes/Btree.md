# `Btree`

Defined in `mboostrs.learners`; exported as `mboostrs.Btree`.


*Class*

```python
Btree(
    names,
    minbucket=4,
    minsplit=10,
    maxdepth=1,
    by=None,
    selection='sse',
    split_method='exact'
)
```

Tree base-learner (R `btree`). Fits decision stumps by default.

At each boosting step, finds the best binary split across all input
columns and returns the node means as the fitted values. This is the
base-learner used by R's `blackboost()`.

In a formula, `btree(lon, lat, income, maxdepth=6, selection=sse)`
makes one tree that can split on all three columns. `btree(.)` uses
all predictors. Separate terms `btree(lon) + btree(lat)` fit an
additive model of two trees. A joint tree of depth greater than one can
represent interactions through successive splits on different columns.

Args:
    names: list of predictor names this tree may split on.
    minbucket: minimum weighted observations in a terminal node (default 4).
    minsplit: minimum weighted observations to attempt a split (default 10).
    maxdepth: maximum tree depth. 1 = stumps (the R `btree` default).
    by: optional varying-coefficient variable z. The term fits f(t) * z
        by least squares (a tree grown on u / z with weights w * z^2).
        For a 0/1 z this is R's btree(by=): a tree grown on the rows
        where z is 1, its prediction multiplied by z.
    selection: how a node chooses the column it splits. `"sse"` (the
        default) takes the split over every column that most reduces the
        squared error (CART): it favours columns with many split points
        and can find a column whose effect is not monotone at the first
        split, where ctree's linear association statistic is near zero
        (e.g. California housing, #263: a joint tree scores 0.2638 with
        `"sse"` vs 0.2705 with `"ctree"` at the same depth/budget).
        `"ctree"` is partykit's conditional inference tree as mboost's
        `btree` runs it (fitted values agree with mboost to 1e-15): the
        column with the largest association
        statistic with the gradient (`n r^2` for a number, the
        between-level sum of squares over the variance for a factor), then
        its best split. `"ctree_pvalue"` compares that statistic's
        chi-squared p-values instead, so a factor with many levels has no
        advantage over a number. Pass `selection="ctree"` explicitly for
        R-parity with mboost's `btree`.

Example: where the two criteria disagree.

`"sse"` evaluates every split; `"ctree"` first asks which column is
linearly associated with the gradient. When the true effect is sharp but
non-monotone, that association can be near zero at the first split, and
`"ctree"` spends the split on a decoy column:

```python
>>> import numpy as np, mboostrs as mb
>>> from mboostrs.learners import Btree
>>> rng = np.random.default_rng(1)
>>> n = 400
>>> x1 = rng.uniform(-2, 2, n)   # the real signal: a "hole", |x1| > 1
>>> x2 = rng.uniform(-2, 2, n)   # weak linear decoy
>>> y = (np.abs(x1) > 1).astype(float) + 0.08 * x2 + rng.normal(0, 0.1, n)
>>> data = {"x1": x1, "x2": x2, "y": y}
>>> kw = dict(family=mb.Gaussian(), control=mb.boost_control(1))
>>> sse = mb.gamboost("y ~ btree(x1, x2, maxdepth=1)", data, **kw,
...                   learners=[Btree(["x1", "x2"], maxdepth=1,
...                                   selection="sse")])
>>> ctree = mb.gamboost("y ~ btree(x1, x2, maxdepth=1)", data, **kw,
...                     learners=[Btree(["x1", "x2"], maxdepth=1,
...                                     selection="ctree")])
>>> grid = {"x1": np.array([-1.5, 1.5]), "x2": np.zeros(2)}
>>> np.round(np.asarray(sse.predict(grid)), 3)    # split x1: found it
array([0.482, 0.553])
>>> np.round(np.asarray(ctree.predict(grid)), 3)  # split x2: missed it
array([0.484, 0.484])
```

When to prefer which: keep the `"sse"` default when the predictors are
mostly numeric and the response may jump -- spatial discontinuities,
thresholds, regime changes. On California housing (#263) a joint depth-6
tree reaches 0.2638 log-RMSE with `"sse"` against 0.2705 with
`"ctree"` at the same depth and budget. Reach for `"ctree"` when the
column set is a noisy mix and you want the association test to refuse
weak splits: `"sse"` always finds *some* split that reduces in-sample
SSE, while `"ctree"` only spends a split where the association clears
the test. Note the test itself is not neutral -- under `"ctree"` a
factor with many levels has an edge over a number; `"ctree_pvalue"`
compares p-values instead and removes it.

Source: [`python/mboostrs/learners.py` lines 600-706](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/learners.py#L600-L706)

## Members

| Member | Kind | Summary |
|---|---|---|
| `name` | Property |  |

(btree-name)=
## `Btree.name`

*Property* of `Btree`

```python
Btree.name
```

*No description in the docstring.*

Source: [`python/mboostrs/learners.py` lines 704-706](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/learners.py#L704-L706)
