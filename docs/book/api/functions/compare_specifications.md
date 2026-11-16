# `compare_specifications`

Defined in `mboostrs.explore`; exported as `mboostrs.compare_specifications`.


*Function*

```python
compare_specifications(
    specifications,
    data,
    response,
    family=None,
    folds=5,
    mstop=500,
    nu=0.1,
    seed=0,
    weights=None,
    score='family',
    score_parameter=None,
    na_action='error',
    observation_ids=None,
    include_evaluations=False,
    fit_budget=None,
    output='records',
    unseen_levels='average',
    history=None,
    max_mstop=None,
    extrapolation='warn',
    step='fixed'
)
```

Score several right-hand sides on the same folds.
Usage:

```python
import mboostrs as mb
rows = mb.compare_specifications(
    {"base": "bbs(x, df=4)", "smooth": "bbs(x, df=6)"}, df, "y")
rows[0]["specification"]  # the first entry is the reference
```

`step` selects the ordinary boosting step rule (`"fixed"` or
`"line_search"`), forwarded to every candidate's core-backed evaluator.
It uses the same meaning as `cv_risk(step=...)`; the default is unchanged.

The loop's closing step. `term_report` says what is worth changing and
`suggest_specifications` orders the candidates, but neither has fitted the
changed model: a suggestion is a hypothesis until the specification carrying
it is scored. This scores them together, on one partition, so the numbers
are comparable to each other and to the specification they came from.

With a distributional family, a candidate may be a mapping of parameter
name to right-hand side — `{"mu": "bbs(x)", "sigma": "1"}` — or of
parameter name to a whole formula for the response,
`{"mu": "y ~ bbs(x)", "sigma": "y ~ 1"}`. The response is inserted into a
right-hand side, so both forms score the same candidate. The constant-scale
and varying-scale specifications a spread diagnosis proposes can be
compared directly this way. Every candidate is scored by the
family's joint proper score (negative log predictive density) on the same
folds, and a candidate row's `rhs` is still the thing to edit and pass
back. A plain string candidate with a distributional family shares the
right-hand side across parameters, as a single-formula `gamboostLSS`
call does.

**Parameters**

- `specifications` (mapping or sequence): A mapping of name to right-hand side, or a plain sequence of
  right-hand sides, which are then named by the string itself. The first
  entry is the reference the others are compared against. A whole
  formula whose left-hand side is `response` is accepted and read as its
  right-hand side; a different left-hand side raises. In a plain
  sequence, a distributional candidate (a mapping of parameter to
  right-hand side) is named `"mu: ...; sigma: ..."`.
- `output` ({"records", "auto", "polars", "pandas"}): `"records"` -- the default, and what this always returned -- is a
  list of dicts, which stays editable: a row's `rhs` is the string to
  change and pass back in.
- `include_evaluations` (bool, default False): Include each reconstructible `EvaluationResult` in records
  output. Frame output refuses this option because an opaque Python model
  result has no stable tabular representation.
- `fit_budget` (int, optional): Maximum model fits allowed for the complete comparison. The requirement
  is derived before fitting: one fit per outer fold at fixed `mstop`;
  with `mstop="cv"`, the outer fit plus every structure-aware inner
  fold. An inner search that grows past its first grid (see
  `max_mstop`) refits every inner fold at each grid under any score
  other than the default family score, so the requirement counts every
  grid up to `max_mstop`. The comparison refuses before work begins
  when the budget is insufficient.
- `max_mstop` (int, optional): With `mstop="cv"`, the largest step count the inner tuning of each
  candidate may reach; the default is 500. The inner search starts at
  500 steps, or at this cap when it is smaller, and doubles while its
  optimum is in the boundary band at the end of the grid, up to the cap.
  A candidate whose inner search reaches the cap with the risk still
  falling gets a warning naming `max_mstop`, and its row reports
  `inner_at_boundary`. Passing it with a fixed `mstop` is an error.
- `unseen_levels` ({"average", "zero_effect", "error"}, default "average"): Passed to `cv_risk`: how a held-out row is predicted when its
  `brandom` level does not occur in the fold's training rows.
- `extrapolation` ({"warn", "error", "allow", "clip"}, default "warn"): Passed to `cv_risk`: how a held-out row outside its fold's
  training range is predicted. `"warn"` reports the rows in one
  warning for the whole comparison; `"clip"` holds the spline flat at
  the range edge; `"error"` refuses; `"allow"` predicts silently.
- `history` (list of dict, optional): Rows from earlier ``compare_specifications(...,
  include_evaluations=True)`` calls, in the order they were run, on the
  same data, folds, family and score. The comparison then reports the
  whole search: `search optimism` and `competitive set` cover the
  history's candidates together with the current ones, and every row
  gains `search candidates` and `search fits`, the number of
  candidates and model fits across every call, history first. Rows
  without their retained evaluation are refused, as are rows whose data,
  folds, score or evaluation weights do not pair with the current call.
  The same
  specification scored twice counts twice; pass each candidate once.

**Returns**

One row per specification, in the order given. Every row has
`specification` (its name) and `rhs`, the right-hand side it was
scored with: a string for a string candidate, and a dict of parameter to
right-hand side for a mapping candidate. The row also has its out-of-fold
risk, the change from the reference with its paired fold standard error
(`change se`), and the settings the score was computed under. A row is
the evidence *and* the specification, so a comparison can be edited and
rerun without reconstructing what was scored.

A candidate that cannot be scored (for example, a fit refused on its
data) does not stop the comparison. Its row has `error`, a message that
names the candidate and the cause, and NaN in `out-of-fold risk`,
`change` and `change se`; the others are still compared against the
reference, which is the first candidate that scores. The function raises
only when every candidate fails.

`change se` is the standard deviation of the fold-by-fold differences
divided by the square root of the number of folds. On a scheme that stacks
several partitions (`repeated_kfold_folds`, or `grouped_folds` with
`repeats=`) it is computed within each partition and the variances are
averaged, because folds of different partitions reuse the same rows. Even
for one partition it understates how much the change would vary across
data sets, since the training sets of the folds overlap; a
two-standard-error threshold is weaker than a normal table suggests.

With `mstop="cv"` each candidate's step count is tuned inside every
outer fold, so a row compares tuning procedures, and the paired standard
error covers that tuning. The row reports the inner step counts across
the outer folds as `inner_mstop_min`, `inner_mstop_median` and
`inner_mstop_max`, and `inner_at_boundary` is true when any outer
fold's inner search stopped at `max_mstop` with the risk still falling.
With a fixed `mstop` these four fields are None. Its calibration was measured by simulation
(`tools/simulate_nested_comparison.py`: n = 200, five outer and five
inner folds, `bbs(x)` against `bbs(x) + bols(u)`). When `u` carries
no signal, `change / change se` over 1,000 replicates had standard
deviation 1.15 rather than 1; the extra term was judged better by two
standard errors in 2.4% of replicates (Monte Carlo SE 0.5%) and
`|change| > 2 change se` occurred in 6.3% (SE 0.8%), against 4.6% for a
standard normal statistic. With a coefficient of 0.15 on `u` (noise
standard deviation 0.7), the two-standard-error rule detected it in 35%
of 400 replicates. Read `change se` as about 15% too small at this
sample size. Grouped, rolling-origin and repeated schemes are not covered
by this check.

Two fields on every row describe the search rather than one candidate.
`search optimism` is the Tibshirani & Tibshirani (2009) estimate of how
optimistic the winning score is because it is the best of the candidates
compared: for each fold, the winner's score minus the lowest score any
candidate reached on that fold, averaged over folds, in the same units as
the score. A single-candidate comparison, or a winner that won every fold,
has zero optimism; the more candidates tried and the fewer folds the
winner won, the larger it grows. `competitive set` lists, winner
included, every candidate whose out-of-fold risk is within one paired
fold standard error of the winner's: the folds cannot tell those
candidates apart, so preferring the winner among them is a choice the
comparison does not support.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/explore.py` lines 3719-4022](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/explore.py#L3719-L4022)
