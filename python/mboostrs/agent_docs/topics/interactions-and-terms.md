# Interactions and term earnings (step 5)

## Step 5 — Residual interactions and term contributions

**`screen_interactions(data, residual, response=..., groups=...)`** ranks
candidate pairs by held-out residual variance they explain: both columns are
binned on half the rows, cell means of the residual are taken there, and the
MSE those means remove is measured on the other half. A pair that fits noise
removes nothing out of sample.

The one rule that is easy to get wrong: **`residual` is on the response
scale** — `y - model.predict(type="response")`. For a Gaussian fit that is
the ordinary residual; for a non-identity link (Binomial, Poisson, Gamma) the
default `predict()` is the link-scale predictor, and subtracting *that* from
the response leaves curvature that every pair of signal columns appears to
explain — you will "discover" interactions that are the link function
(`screen_interactions` expects response-scale residuals). A fixed seed makes
the half-split reproducible, so scores and their order are stable across runs.

Output: `ScreenedPair` rows, `left`, `right`, `removed` — an MSE per
held-out observation, comparable to out-of-fold risk differences — sorted
descending. A score at or below zero is evidence *against* the pair, not a
weak ranking. Pass group columns in `groups=` so each level is its own cell
(quantile-binning group codes merges unrelated groups and hides random
slopes).

Decision: pairs with `removed` clearly above zero (and above the noise of
its neighbours in the list) become interaction candidates for step 6; the
rest do not. The screen nominates; it never adopts.

**The screen found nothing is weak evidence for binary responses.** The
screen proposes the tensor surface, a 6.25-df learner; a bilinear interaction
(a product term) can be real and still only about one standard error under
that surface, so a negative screen on a Binomial response does not rule the
interaction out. The cheap follow-up is the 1-df varying-coefficient check
`bols(x, by=z)` and its mirror `bols(z, by=x)`: each is one degree of
freedom, so it can find a product the tensor misses. Fit both directions
whenever the screen is negative on binary data and score them with
`compare_specifications`.

**The autospec screen feeds Pearson residuals.** When `probe_interactions`
screens a non-Gaussian fit it now passes *Pearson* residuals to
`screen_interactions` (via `terms.pearson_residual`), not the raw
response-scale residual `y - fitted`. The change is in what
`probe_interactions` hands the screen, not in `screen_interactions` itself:
a raw Binomial `y - p` has variance `p(1-p)`, so rows near `p = 0.5`
dominate the screen's MSE, while Pearson residuals put every row on a common
variance scale. `screen_interactions` called directly still expects the
residual you pass it.

**`term_report(formula, data, response=..., family=..., folds=...)`** answers
two questions by refitting on shared folds. *What does each term earn?* —
the model is refitted without it; the rise in out-of-fold error is its
`earns`, with `se`, the paired fold standard error. *Would another shape do
better?* — each term is retried as `bols`, `bbs`, `btree` with the rest held
fixed (`report.shapes`). The cost is `folds × (1 + n_terms × n_shapes)` fits:
a tool for tens of terms, not thousands.

Output: `report.baseline` (the specification's own out-of-fold risk),
`report.contributions` (per term: earns, se, `measurable`), `report.shapes`
(per term: gains as each alternative shape). A `brandom` term scored on
folds that hold its groups out is reported `measurable=False`: what
removing it changes is how the *other* terms are fitted, and these folds
cannot say whether the groups differ at all.

Decision: a term whose `earns` is within two SE of zero cannot be told from a
term that earns nothing on these folds; a shape whose gain is within two SE
of the current shape's score cannot be told from no change. Do not act on
either alone — that is what step 6 is for. Refit and rescreen after adopting
a catch-all term (`btree`) — a tree that only partially captures threshold
structure leaves it in the residuals, and the same screen says how much.

**Identification is a different question.** `earns` measures held-out
predictive contribution, which is not the same as whether an input carries
signal: a CV-tuned fit can keep noise terms whose tiny effects barely move
held-out risk. For "which inputs carry signal" the recommended path is
`mb.stabsel` (see the `uncertainty` topic) — the only selector with an
error-rate guarantee; `deselect` and `probe` carry none. Use `term_report`
to compare shapes and earnings; use `stabsel` to decide which inputs are
real.

## Step 5 — Discontinuity screening

Both screens above name smooth or pair structure. A sharp jump across a
boundary in two columns is the shape they cannot nominate: the smooth fit
absorbs most of it, an axis-aligned threshold on either column alone reads
"no gain" in cross-validation, and no marginal curve shows the line.
`screen_discontinuities(data, residual, response=...)` scans for it
directly: steps in one column and steps across lines in two standardized
columns over 16 directions, all fitted as step indicators against a smooth
background in the Rust core.

The honesty rule is a half split. Candidates are ranked on one half of the
rows; the reported `z` of the winner comes from the other half, which
selection never touched, so under no jump anywhere it is a standard normal
and `|z| >= 2` with a positive held-out `gain` (the row's `flagged`) fires
on about 4.6% of no-jump screens. `type_ratio` separates a jump from a steep smooth crossing by scale invariance: the level gap is measured at two scales (nearest 15 vs 60 points per side); a jump's gap persists (ratio ~1), a steep crossing's gap grows with scale (ratio ~2+) and does not type as a jump (`typed_jump`). A steep rise *does* flag on the raw response —
that is real step-beyond-smooth structure — but it does not type as a
jump, and the residual pass after the smooth fit stays quiet.

Decision: a `typed_jump` row proposes two candidates for the comparison —
the boundary's step indicator (`bols` on the thresholded line) and
`btree(pair)` — and the named indicator usually wins when the boundary is
real. A flagged row with a small `type_z` proposes a smooth or tensor term
instead. No flag means no screened evidence for a jump in these columns, and the
`reference_gap` belongs to smooth or interaction structure: the negative
is a reportable result. See `example:disc-screening` for the worked panel.

## Skewed predictors: log/sqrt vs quantile O-splines vs rank transforms

An equidistant P-spline places knots evenly from the column minimum to its
maximum. On heavily right-skewed predictors (sample skewness > 2), most rows
crowd into the first one or two knot intervals while sparse upper-tail values
consume the rest. Three alternatives exist, each with different geometry:

- **Domain-safe `log` / `sqrt` (default in `autospec` and `fit`).** Places the
  smoothness prior on the multiplicative (`log`) or variance-stabilized (`sqrt`
  when zeros appear) scale, aligning knot density, uniform curvature penalty,
  log-linear null space, and logarithmic boundary extrapolation. Across 22
  skewed real datasets in the held-out suite, domain-safe `log`/`sqrt` cut
  geometric-mean test loss by **10.1%** vs equidistant `bbs` (15/22 wins,
  worst-case 1.08×, zero domain failures). Automatically downgraded whenever
  the full column violates domain positivity.
- **`bbs(x, knot_placement="quantile")` (O-spline).** Places deduplicated
  interior knots at empirical quantiles and penalizes integrated squared second
  derivatives (Wand & Ormerod 2008), preserving exact raw-scale linearity in
  the null space. Prefer this on **signed heavy-tailed predictors** where
  `log`/`sqrt` do not apply or when raw-scale linear extrapolation is desired;
  on positive right-skewed tables the inverse-cube interval-width weight can
  over-smooth the dense bulk (+4.8% geomean vs equidistant across the 22 skewed
  datasets, up to 2.43× on `cpu_activity`).
- **Empirical-CDF / rank transform.** Uniform smoothness on the rank scale with
  flat tails, breaking raw-scale linearity (+9.8% geomean, up to 5.04× on
  `cpu_activity`); reserve for strictly rank-driven or heavily zero-inflated
  curves.

