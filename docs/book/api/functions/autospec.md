# `autospec`

Defined in `mboostrs.autospec`; exported as `mboostrs.autospec`.


*Function*

```python
autospec(
    data,
    response,
    family=None,
    mstop=100,
    df=4.0,
    smooth='bbs',
    budget=0,
    folds=None,
    seed=0,
    nu=None,
    prior=None,
    monotone=None,
    coordinates=None,
    adjacency=None,
    basis=None,
    groups=None,
    measurements=None,
    treatment=None,
    interactions='report',
    deselect=True,
    step=None,
    residual_tree=False,
    preexploration='native',
    heredity='none',
    forbid=None,
    mandatory=None,
    joint_tree=True,
    exposure=None,
    na_action=None,
    lss_trees='auto'
)
```

Guess a complete boosting specification for `data`.

**Parameters**

- `data` (dict or DataFrame): Polars or pandas frame, or a mapping of column name to array.
- `response` (str): Column holding the response.
- `family` (Family, optional): Use this family instead of inferring one from the response. Whole-number
  values are otherwise read as counts, including values stored as
  floating point; that ambiguous case emits a warning and is recorded in
  `rec.report()`. Declare the meaning of the response:

  * `Gaussian()` for a rounded continuous measurement (values such as
    house prices or ages recorded to the nearest unit);
  * `Poisson()` or `NBinomial()` for counts;
  * `PropOdds()` for ordered classes coded as integers `1..K`
    (`PropOdds(n_levels=K)`) or as an ordered categorical column.
    Ordered categorical responses are inferred as `PropOdds` even
    without this argument; unordered labels are `Multinomial`.
- `exposure` (str, optional): Column of positive exposures (time at risk, area, number of trials)
  for a count response. The response is then modelled as a rate per
  unit of exposure: `log(exposure)` is an offset in every fit, the
  column is not a predictor, and `rec.fit` passes the offset. The
  family is `Poisson()` unless `family=NBinomial()` is given.
  Missing or non-positive exposures raise an error; predictions on new
  rows need their exposures as `offset=` (see `rec.fit`).
- `mstop` (int): The step count in the returned spec. Default 100, matching
  `mboostrs.boost_control`.
- `df` (float): Degrees of freedom for the penalised learners. Default 4, matching
  `bbs` and R `mboost`.
- `smooth` ({"bbs", "btree"}): The learner for a continuous column. `"bbs"` is a P-spline
  (default); `"btree"` fits thresholds instead.
- `budget` (int): How many model fits the probes may spend. `0`, the default, runs
  none of them: the specification is the dtype-driven one this function
  has always returned, at the same cost, plus the hygiene findings,
  which need no fit. Give a budget to have the choices measured --
  the family probe costs `1 + n_folds` fits per candidate family --
  and `rec.report()` then distinguishes what was measured from what
  fell back to a default. A measurement that lands within sampling
  noise of a decision line declines to choose and says so, and fits
  two probes need in common are made once and counted once. The first
  measurement tunes `mstop` for the dtype-driven specification, and the
  family and shape probes compare their candidates at that depth: at
  a default 100 steps a slowly learned signal is still mostly unfitted
  and every candidate scores about the same. The shape probe changes
  one column's learner at a time with the rest of the model held fixed
  at its fitted value, which costs one-term fits. The family probe also
  offers the distributional family the response's support names —
  `BetaLSS` for a response inside (0, 1), `GammaLSS` for a spread
  that grows with the level — and scores it against the surviving mean
  family on held-out log score, adopting it only when it clearly wins;
  a search that adopts one stops at the family, and `rec.fit()` then
  builds the distributional model.
  For Gaussian responses with coordinate-like predictors (such as
  `lon` and `lat`), the final shape comparison also offers one
  joint CART tree over the undeclared continuous predictors, at depths
  3 and 6. Each specification is tuned on the shared folds; replacement
  requires at least 2% less held-out risk. Declared and mandatory terms
  remain unchanged. The comparison costs 3 * (folds + 2) fits; once the
  family is settled those fits are set aside from the earlier probes
  when the budget can also pay for one more sweep, and
  `evidence["joint_tree"]["reserved"]` records the amount.
  `evidence["joint_tree"]` records the candidates,
  their scores and the decision, including a skipped comparison when
  the budget or interaction constraints prevent it. Selection on these
  folds needs a separate test set to measure generalisation.
  When the search has finished, the specification it settled on is
  compared as a whole with two simple models fixed in advance, on the
  same folds and by the same held-out risk:

  * `"linear"`: the starting specification with every one-column
    smooth written as a linear term (`bbs(x, df=4)`, `btree(x)`
    and `bns(x)` become `bols(x)`; a `log`/`sqrt` transform of
    the input is kept). Factor terms (`bols`, `brandom`), the
    caller's declared terms and terms over two or more columns stay as
    written.
  * `"additive"`: the starting specification itself, the one
    `budget=0` returns: a `bbs(x, df=df)` smooth per continuous
    column, `bols`/`brandom` per factor, with the declarations
    applied; with `prior`, the prior's formula.

  Each rival is fitted at the search's step length and judged at its
  own best step count. A rival replaces the searched specification
  only when its held-out risk is lower by more than one standard error
  of the paired fold difference; on a single holdout that standard
  error is undefined and the searched specification stands. The
  comparison costs one sweep (`n_folds` fits) for the searched
  specification and one per rival that differs from it, and runs only
  when the remaining budget pays for it; the search itself does not set
  fits aside for it. `evidence["whole_model"]` records each rival's
  formula, held-out risk (`risk`) and step count (`mstop`), the
  paired difference to the searched specification (`difference`,
  rival minus searched, negative when the rival does better) with its
  standard error (`difference_se`), whether the rival replaced the
  searched specification (`replaced`, `winner`), or why the
  comparison did not run (`reason`). A distributional
  recommendation carries the reason only: its held-out score is a log
  score, not the mean family's risk, and
  `evidence["finished_routes"]` compares it with the finished mean
  model.
- `folds` (ndarray or FoldScheme, optional): Fold weights the probes share, so every comparison is made on the
  same split. When omitted, the frame's own structure names them: a
  date or datetime column gives rolling-origin folds ordered by it
  (the past predicts the future), and columns the specification fits
  as groups -- `brandom` or `bmrf` terms, declared or read from
  the dtype -- give grouped folds that hold whole groups out;
  otherwise 5 random folds on `seed`. A supplied `folds` overrides the
  detection; `groups=` adds its columns to the detected set.
  A design with a single fold (one supplied holdout) carries no
  across-fold variation, so the paired fold tests cannot fire.
  On such a design the probes that would otherwise decide on an
  aggregate margin use the per-row evidence of the holdout
  itself: a shape replacement, a screened pair's form, an
  interaction pair or the residual tree is taken only when its
  margin also clears the paired per-row z bar. The z recorded in
  the evidence is the one the decision used.
  Families whose loss couples rows (CoxPH, AUC, Cindex, Gehan)
  have no per-row losses; their single-holdout decisions keep the
  margin rule alone.
  The distributional family's adoption checks are paired fold
  tests with no per-row form, so on a single fold they cannot
  pass: a distributional family is not adopted on a supplied
  holdout, and the mean family stands.
- `seed` (int): Seed for the default and grouped folds.
- `nu` (float, optional): Step length of every probe and of the recommendation. `None`, the
  default, is 0.3 (`gamboost`'s own default is mboost's 0.1): the
  probes walk each candidate's path to its cross-validated optimum, and
  at 0.1 those paths ran to 10,000 steps, while held-out accuracy on
  development data matched within 0.001 of R^2. It is lengthened to 0.5
  when the cross-validated risk is still falling at the 10,000-step cap
  (strongly collinear columns make component-wise boosting slow); the
  report says so. A value you give is used as given. When you give
  none, the finished specification is also read at a bounded grid of
  other step lengths (0.1, 0.2, 0.5, 0.7 and 1.0, in addition to the
  step length in force) once the search settles, at one fold path per
  extra value charged to the same fit budget, and the recommendation
  takes the step length whose sweep reads the lowest cross-validated
  risk; `evidence["nu_search"]` records the grid, each value's
  chosen step count and risk, and the fits spent.
- `preexploration` ({"native", "neural", "dual", "hessboost", "off"}): Preliminary boosted-tree exploration for budgeted searches. One
  ensemble supplies permutation importance, partial-dependence shape
  diagnostics and interaction proposals; its fit is charged once.
  For Gaussian responses, agreement between marginal and partial
  diagnostics narrows shape comparisons, ordered by importance.
  Incumbents and declared terms are retained; ambiguous shapes stay
  open. Other families use the pair proposals without shape pruning.
  Every replacement still requires the usual held-out comparison.
  `"neural"` fits a seeded Rust MLP (two 32-unit tanh layers),
  using one-hot categories and training-fitted numeric scaling.
  `"dual"` fits both native trees and the net, charges two fits,
  and merges their pair proposals with source labels. Neural subset
  diagnostics extend greedily to at most five inputs; they describe
  total nonadditivity, including lower-order interactions, and do
  not automatically add higher-order terms. Only tree evidence may
  narrow shape comparisons. The default remains `"native"`.
  `"off"` uses the full shape catalogue and residual pair screen.
  `"hessboost"` requires a build with the optional Cargo feature of
  that name; otherwise it raises an error. Native is the default.
  Screening uses the supplied training data; risk curves after this
  adaptive search are descriptive, not unbiased performance estimates.
  Reserve separate test data for evaluation of the whole procedure.
- `prior` (str or mapping, optional): A specification you already believe in, as a formula string or a
  mapping with `formula` and optionally `family` and `mstop`. It
  becomes the baseline: the probes measure against it rather than
  against the dtype guess, each term it names is the incumbent shape,
  and the report says which probe disagreed with it and by how much. A
  probe that finds nothing better leaves your specification alone.
- `monotone` (dict, optional): Mapping of column name to a `Bmono` constraint (`"increasing"`,
  `"decreasing"`, `"convex"`, `"concave"`, `"positive"`,
  `"negative"`). Those columns get `bmono` terms carrying the
  stated constraint. Declared terms are fixed: the shape probe scores
  the alternatives against them and reports what the constraint costs,
  but never replaces them.
- `coordinates` (tuple of str, optional): Two column names that are coordinates. They get one
  `bspatial(col1, col2)` term in place of their two separate terms.
- `adjacency` (dict, optional): Mapping of column name to a neighbourhood penalty matrix `K` in
  graph-Laplacian form (see `mboostrs.learners.Bmrf`). Those
  columns get `bmrf` terms carrying `K`.
- `basis` (dict, optional): Mapping of column name to an `(X, K)` pair. Those columns get
  `buser` terms built from your design and penalty.
- `groups` (str or sequence of str, optional): Columns that identify groups: clinics, districts, subjects. Each gets
  a `brandom` term, and the probes share grouped folds that hold
  whole groups out, so every measurement scores prediction for a
  group the fit has not seen. Columns the dtype rule already fits as
  groups are detected on their own; this declares the ones it reads
  as measurements. Without `folds`, both sets are held out whole.
- `measurements` (str or sequence of str, optional): Columns you know are ordered quantities, whatever the dtype rule
  would do with them: a seasonal coordinate sampled at three
  values, a code whose levels are equally spaced scores. Each gets
  one slope (`bols`), the shape probe prices the smooth and the
  threshold against it, and the columns leave the group-detection
  set. This is the one-word answer to the report's note that a
  few-valued column "is read as groups, not as a measured
  quantity".
- `interactions` ({"report", "adopt"}): What to do with the pairs the interaction screen finds (it runs when
  `budget` allows). `"report"` lists them under
  `evidence["interactions"]` and leaves the formula additive;
  `"adopt"` also adds each proposed pair term (a `%X%` tensor, a
  varying slope, or a switch-on smooth), skipping pairs that record one
  axis twice, and re-tunes `mstop` for the larger formula when budget
  remains.
- `lss_trees` (bool or "auto"): When the search adopts a distributional family, offer a joint tree
  (`btree` over the numeric predictors, depths 3 and 6) added to every
  parameter's formula, so the mean and the spread can depend on
  products and thresholds of predictors that additive terms cannot
  represent. The tree is adopted only when the family's held-out log
  score beats the additive specification by the family choice's margin
  and two paired fold standard errors, and its squared error of the mean
  is not worse; `evidence["lss_joint_tree"]` records each depth's
  comparison. The same setting governs the spread probe's tree
  candidate: when the linear spread screen finds no covariate, GaussianLSS
  is offered a joint depth-3 tree as its scale equation. `"auto"`
  (default) runs each when the budget left pays for it without taking
  the fits the mean route's stopping search needs (the scale candidate
  needs 8 sweeps of folds left); `True` runs them whatever the budget;
  `False` never.
- `residual_tree` (bool or "auto"): After the interaction step, try one more term: a regression tree of
  depth 5 over every numeric and categorical column (``btree(x1, ...,
  xp, maxdepth=5)``; a factor splits into two sets of its levels),
  boosted alongside the other terms. It fits structure
  that the main effects and the adopted pairs do not: interactions of
  three or more columns, and pairs the screen did not propose. The term
  is kept only when it lowers the cross-validated risk by 2% or by two
  paired fold standard errors, the rule adopted pairs must pass, and
  `mstop` is tuned again with it. `evidence["residual_tree"]` records
  the comparison: `held_out_gain` is the fold mean of the incumbent's
  risk minus the tree's risk (positive when the tree lowers it), with
  `held_out_se` its standard error and `z` the paired statistic the
  rule read; when the standard error is undefined (one fold, or a fold
  without a score) `held_out_se` is NaN and `held_out_reason` says
  why. A distributional family records the same fields for its `mu`
  predictor, in the family's held-out loss (negative log-likelihood
  per row). On the mean route `share` gives the part of
  the adopted prediction the tree carries on training rows, as
  `covariance_share` and `absolute_share`. The other terms keep
  their curves. The screen-stage lead is recorded apart from the
  decision, in `evidence["residual_tree_detection"]`: the share of the
  response variance the additive fit leaves, and a depth-3 tree's R^2 on
  that leftover. When the predictors include a coordinates-like pair and
  the folds are not
  already grouped, the adoption comparison uses grouped folds over
  spatial grid cells: random row folds let the tree validate on
  rows neighbouring its training rows, so the gain need not transfer
  to unseen regions. `evidence["residual_tree"]["validation"]` names
  the scheme that decided. The
  term is not an effect curve to read: `rec.fit(data, refit="reml")`
  holds it at its boosted contribution while the other terms are
  refitted. `"auto"` offers the tree only when the training-only
  rule fires: the predictors include a coordinates-like pair,
  or adopted pairs leave residual structure a shallow tree captures.
  The default `False` never offers; `True` always offers.
- `step` ({None, "fixed", "line_search"}): The step rule of every probe and of the recommendation (see
  `mboostrs.boost_control`). `None`, the default, fits with the
  fixed step and, when a fit diverges (a Poisson, negative binomial or
  gamma response in the hundreds overflows its inverse link in the
  first fixed steps), runs again with `"line_search"`; the
  recommendation's control then carries the line search, so
  `rec.fit` replays it, and the report says why. With `budget=0` no
  fit runs to detect a divergence, so a Poisson response takes the line
  search directly, because its fixed step grows with the mean.
- `heredity` ({"none", "weak", "strong"}, default "none"): The marginality rule for adopted interactions (Nelder; Chipman 1996;
  Bien, Taylor & Tibshirani 2013). `"strong"` adopts a pair only when
  both of its main effects are in the model and keeps both through
  deselection; `"weak"` requires one and keeps at least one.
  Adoption is tested while every modelled column still has its main
  term, and deselection runs afterwards, so a pair whose parents are
  both modelled is admitted under `"strong"` too. A parent that is not
  modelled (a constant column, or one the hygiene checks drop) bars
  the pair. `"none"` (the default) adopts on held-out evidence alone.
  The rule in force is stated in `rec.report()` and in
  `evidence["interaction_rule"]`.
- `forbid` (iterable of (str, str), optional): Column pairs the interaction screen never lists or adopts.
- `mandatory` (iterable of str, optional): Columns whose terms deselection never drops.
- `joint_tree` (bool, default True): Offer the coordinate comparisons when undeclared continuous predictors
  include a coordinate-like pair: `bspatial(x, y)` against
  `btree(x, y)` on the shared folds (`evidence["coordinate_surface"]`),
  then the final Gaussian joint-tree comparison
  (`evidence["joint_tree"]`). `False` keeps the search to its
  per-column shapes and explicitly requested pair/residual-tree probes,
  for example when a caller needs an additive specification. Joint
  replacement also skips non-default heredity rules and forbidden pairs
  among its inputs.
- `deselect` (bool or float): After `mstop` is tuned (when `budget` allows), drop each term whose
  share of the tuned fit's in-bag risk reduction is below 1% (or below
  the share given as a number, such as `deselect=0.025`) and tune
  `mstop` again for the smaller formula (Strömer et al. 2022; the rule
  `mboostrs.deselect` applies to a fitted model). A
  cross-validated boosting fit keeps every term selected even once, so
  most noise columns survive it; this removes them. The smaller
  formula is kept only if its cross-validated risk, on the same folds,
  is at most 1% above the full formula's; many small terms can together
  carry real signal. Declared terms
  (`monotone=`, `coordinates=`, `groups=` and the rest) and terms of a
  `prior` are never dropped. `evidence["deselect"]` lists the dropped
  terms, their shares and the held-out loss. The default share is set
  for prediction. A larger one gives a sparser specification for
  identifying drivers: at 2.5%, on a simulated design with six drivers,
  three correlated proxies and 21 noise columns, a 1,000-row binary
  response kept 0.6 non-drivers per fit instead of 4.1 with the same
  driver recall, while test R^2 on seven real data sets fell by up to
  0.01.
- `treatment` (str or CausalSpec, optional): The column a causal question is about. It is left out of the
  predictive specification -- scoring a treatment by its predictive
  association answers a different question from estimating its
  effect -- and the report points at the causal workflow that does
  answer it: `mboostrs.causal` (`CausalSpec`, `aipw`,
  `negative_control`). A full `CausalSpec` must name this call's
  response as its outcome, and its post-treatment columns are left
  out with the treatment.

**Returns**

SpecRecommendation
    It unpacks as the tuple this function has always returned --
    `(formula, learners, family, mstop)` -- and additionally carries the
    data-hygiene findings and, once probes run, the evidence behind each
    choice. `rec.report()` prints the reasoning and `rec.fit(data)`
    fits the specification. The tuple is: a formula string naming the
    terms, the matching `mboostrs.learners.Bbs`/
    `mboostrs.learners.Bols`/`mboostrs.learners.Brandom`
    objects, the response family, and the step count. The first three feed
    `gamboost(formula, data, learners=..., family=...)` directly.

**Raises**

- `ValueError`: If `family` is omitted and the response is a two-level string factor,
which `Binomial` cannot read until the labels are recoded as 0/1.
Also if `prior` is combined with any declaration keyword, or a
declaration names a column the frame does not hold, a column is
declared both a group and a measurement, or `treatment`
names the response, a column the frame does not hold, or a
`CausalSpec` whose outcome is not this call's response.

Source: [`python/mboostrs/autospec.py` lines 147-611](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/autospec.py#L147-L611)
