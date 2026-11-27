# `screen_discontinuities`

Defined in `mboostrs.terms`; exported as `mboostrs.screen_discontinuities`.


*Function*

```python
screen_discontinuities(
    data,
    target,
    response=None,
    columns=None,
    cells=20,
    directions=16,
    seed=0,
    pairs=None,
    top_k=25,
    all_pairs_max_p=20,
    fallback_top_columns=10,
    min_side=10,
    refine_units=3,
    refine_factor=8,
    type_top=5,
    na_action='error'
)
```

Screen for axis-aligned and diagonal step discontinuities.

The screen identifies the location and orientation of a candidate jump.
It complements `mboostrs.reference_gap`, which measures how much
structure a smooth model leaves unexplained.

Two candidate families are scanned, all fitted as the step indicator
`1{u > c}` against a smooth background by ordinary least squares:

* `axis`: steps in one column, `x_j > c` at `cells` interior
  quantile cutpoints, background `[1, u, u^2]`;
* `diagonal`: steps across a line in two columns,
  `cos(theta) z_j + sin(theta) z_k > c` on standardized columns over
  `directions` evenly spread directions, background `[1, z_j, z_k]`.
  All pairs are scanned while the column count is at most
  `all_pairs_max_p`; beyond that, pairs among the
  `fallback_top_columns` columns whose axis scan scored best.

The rows are split once (`seed`) into a fit half and an evaluation
half. Every candidate is scored on the fit half only -- `jump_fit` and
its t-statistic `t_fit` -- and selection ranks on that; the same step
is then estimated on the untouched evaluation half, giving `jump`,
`se` and `z`. Under the null of no jump anywhere in the candidate
set the winning row's `z` follows a t distribution with
`n_eval - (q + 1)` degrees of freedom (asymptotically standard normal),
so the decision rule `|z| >= 2` with a positive held-out `gain` (the row's
`flagged` property) fires on about 4.6% of null screens regardless of
how many candidates were searched. `t_fit` and `z` are not the same
quantity: ranking by the naive t-statistic of the winning boundary
overstates, which is why selection and reporting use different halves.
The scan uses Rust-accelerated kernels; one unit costs a sort and a
prefix sum.

`type_z` types the top rows globally: the step is refitted against a background
augmented with the smooth terms along the projection (a quadratic and a
cubic in it). `type_ratio` is the local discriminator that drives the
`typed_jump` decision: the ratio of the level gap measured at two
scales (nearest 60 vs nearest 15 evaluation-half points per side),
maximized over a window around the fit-half threshold. A true jump is
scale-invariant (ratio ~1); a steep smooth transition or kink has a gap
that grows with scale (ratio ~2+), so the row is not typed as a jump.
Resolution limit: the small scale sets the resolution -- a transition
narrower than its span types as a jump, honestly, because the data
cannot resolve it. A row with `typed_jump` true is a jump; a flagged
row with a large `type_ratio` asks for a smooth or tensor candidate
instead of a step indicator.

**Parameters**

- `data` (polars or pandas DataFrame, or a mapping of column to values): The rows to screen. Numeric, boolean and categorical columns are read
  through their encodings, as every screen in this module does.
- `target` (array-like): What to scan: the response, or -- the recommended second pass -- the
  response-scale residual of a fitted smooth model, which removes the
  smooth structure so the jump stands alone. Pass residuals on the
  response scale (`y - predict(type="response")`); a link-scale
  residual leaves curvature every column appears to explain.
- `response` (str, optional): The response column, excluded from the predictors.
- `columns` (sequence of str, optional): Screen only these columns; all non-response columns by default.
- `cells` (int): Threshold cutpoints per direction, as interior quantiles of the fit
  half. Lowered automatically so small samples keep cells on both
  sides.
- `directions` (int): Directions per column pair, evenly spread over the half circle.
- `seed` (int): Seed of the fit/evaluation half split and of the tie order.
- `pairs` (sequence of (str, str), optional): Screen only these column pairs (the axis family is unchanged).
- `top_k` (int): Rows returned, best first. `0` returns all.
- `all_pairs_max_p, fallback_top_columns` (int): The pair-scan switch: all pairs while there are at most
  `all_pairs_max_p` columns, otherwise pairs among the best
  `fallback_top_columns` axis columns.
- `min_side` (int): Fewest rows a side of the boundary must hold on each half; a
  candidate that splits off fewer rows is skipped.
- `refine_units, refine_factor` (int): The best `refine_units` diagonal units are re-scanned on a
  threshold grid twice as fine and sub-directions `refine_factor`
  times as fine around the winner. All refinement reads fit-half
  quantities only, so the evaluation-half calibration is untouched.
- `type_top` (int): How many of the best rows get the `type_z` typing refit.
- `na_action` ({"error", "omit", "impute"}): Missing values in a screened column or the target: raise naming the
  column and row (default), drop the rows, or fill a column with its
  fit-half median (so the fill never reads eval-half rows) -- the
  target rows cannot be filled, pass a residual that is finite, or
  drop those rows.

**Returns**

list of DiscontinuityHit
    Best first by `|t_fit|`. Read the winner's `flagged` and
    `typed_jump`, and treat an unflagged screen as a result: it finds no
    screened evidence for a jump in these columns, and the missing structure a
    `reference_gap` still reports belongs to a smooth or interaction
    term, not a step.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

**Examples**

A diagonal jump that no single-column step can express:

```python
>>> import numpy as np, polars as pl, mboostrs as mb
>>> rng = np.random.default_rng(0)
>>> n = 800
>>> s0, s1 = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
>>> data = pl.DataFrame({
...     "s0": s0, "s1": s1, "noise": rng.uniform(0, 1, n),
...     "y": 2.0 * (s1 > 0.45 + 0.35 * s0) + 1.5 * s0
...          + 0.5 * rng.normal(size=n)})
>>> hits = mb.screen_discontinuities(data, data["y"].to_numpy(), seed=0)
>>> hits[0].typed_jump, hits[0].columns
(True, ('s0', 's1'))
```

Source: [`python/mboostrs/terms.py` lines 1078-1514](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/terms.py#L1078-L1514)
