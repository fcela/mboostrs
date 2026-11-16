# Mean-shrinkage study for issue #679

## Study objective

Confirmation 2 failed one gate: on the nonlinear zero-inflated count panel the
discovered fit's count sets are too wide (excess +1.674%, one-sided 95% upper
+2.734% against a +1% bound). The calibration diagnosis
(`CALIBRATION_DIAGNOSIS.md`) showed the width is compensation for a shrunk
mean: the discovered fit underpredicts by about 1.9 counts in the top fitted
quartile, its conformal factor is 0.0391 against 0.0141 for a tuned fit
through the identical construction, and substituting the true mean into the
same construction gives excess -3.02%. This study measures where the
discovered mean loses accuracy and tests whether a general search rule can
reduce that error.

Four candidate mechanisms were separated by design, on the development
cohorts only (count seeds 6607100-6607109, Gaussian seeds 6293100-6293109):

- (a) the search stops the mean fit too early on this panel shape;
- (b) the tensor/interaction selection distorts the mean surface;
- (c) shrinkage inherent to the boosting step size on a curved log-mean;
- (d) the zero-inflation mass pulling the fitted mean down through the
  family's link or score.

The remedy candidate, fixed before pricing, is the existing opt-in
`discover(mean_budget="rate_scaled")` promoted to the default: screening,
comparison and tuning step counts multiplied by `0.1 / nu` (5 for the count
families at nu = 0.02, 1 for a Gaussian fit), so every family's fits travel
as far along the boosting path (`mstop * nu`) as a Gaussian fit does. The
decision rule, also fixed before running: the reserved confirmation-3 ranges
(count 6799500-6799549, Gaussian 6799600-6799649, test offset +100000) are
drawn only if the remedy passes every development gate under the flipped
default. A 50-draw development extension (6607200-6607249) was designed to
read the efficiency bound at confirmation-grade n; it was not run, because
the remedy failed other development gates first and no efficiency reading
could change the verdict.

## Mechanism evidence

The confirmation-2 candidate's mean-path results, recomputed from its retained
records: the selected family is negative binomial in 50/50 draws; the final
mstop sits at the 400 cap in 47/50. Latent-mean MSE by formula class: 8.26
with the linear term present (33 draws), 11.49 with a tree/tensor term
(3 draws), 20.96 when backward elimination dropped the linear term
(14 draws). The tuned supplied track (prespecified Poisson spline plus
linear term, grid to 400) records MSE 2.345 with chosen mstop median 45.5
and never at its cap.

Path contrast. Fitting fixed specifications on the recorder's 500-row
partition at nu = 0.02 and scoring latent-mean MSE against the true mean:

| specification / family | 50 | 100 | 200 | 400 | 800 | 1600 | 3200 |
|---|---|---|---|---|---|---|---|
| true terms, Poisson | 2.57 | 2.91 | 3.31 | 3.60 | 3.97 | 4.51 | 5.17 |
| true terms, NBinomial | 60.84 | 44.41 | 27.78 | 12.50 | 4.25 | 2.56 | 2.81 |
| discovered spec, NBinomial | 60.84 | 44.41 | 27.50 | 12.30 | 4.38 | 3.24 | 3.81 |
| no linear term, NBinomial | 60.84 | 44.41 | 30.46 | 22.62 | 20.02 | 19.86 | 20.26 |

On the same panel without zero inflation (nonlinear Poisson), the
NBinomial fit converges at once (0.89 at 50 steps, 0.28 at 100). The slow
convergence is specific to the family the search selects on a zero-inflated,
overdispersed response: family inference measures held-out Pearson
dispersion 9.3-9.5 on this panel and selects negative binomial, whose mean
fit needs travel about 32 (mstop about 1600 at nu = 0.02) to reach the
accuracy a Poisson fit has almost immediately. The shipped budget gives a
count fit screening travel 4 and tuning-cap travel 8, against 20 and 40 for
a Gaussian fit at nu = 0.1.

Step size is not the mechanism: at matched travel (`mstop * nu`), nu = 0.05
reproduces the nu = 0.02 path almost exactly (MSE 2.56 against 2.57 at
travel 1; 5.17 against 5.17 at travel 64), while nu = 0.1 diverges
(MSE about 100 on both count panels) — the overshoot instability the
count-rate constant exists to prevent. Specification is not the mechanism
either: replacing the discovered specification's linear term for the second
predictor with a spline changes MSE at 400 steps from 12.50 to 12.30, and
tensor terms are almost never selected on this panel (3/50 in confirmation
2). A missing term is a separate, path-independent loss: with the linear
term absent, MSE stays near 20 at every path length in both families.

The shrinkage profile (true minus fitted mean, by true-mean quintile) is
the flattening signature of an unfinished descent: at 400 steps
-1.45, -0.59, -0.16, +0.72, +5.86 — overpredicting the bottom of the range
and underpredicting the top by about 5.9 counts; at 3200 steps the profile
is flat (+0.11, +0.11, +0.23, +0.46, +1.00).

Backward elimination is a downstream symptom of the same starved path. Its
screening fits run at the screening budget, so a real term whose effect the
short path has not yet expressed can fail the paired 2-SE rule. On seed
6607100 the linear term's leave-one-out change is 0.00110 against
2 SE = 0.00189 at screening risk 3.496 under the fixed budget (the risk pair
3.4960/3.5000 quoted in `PROTOCOL.md`), and it is dropped; at the scaled
screening budget the change is 0.0118 against 0.0093 at risk 3.473, and it
is kept. The interaction runs both ways: on seed 6607103 the fixed budget
keeps the term (0.00133 against 0.00078) while the scaled budget drops it
(0.00774 against 0.00862), because at the longer screening path the first
predictor's spline absorbs part of the second's signal inside the folds
while the paired SE grows about eleven-fold. Elimination has no separate
fix; its inputs are the same starved fits.

Discover contrast on the same draws (latent-mean MSE): fixed budget 13.91
with 10/10 fits at the 400 cap and the linear term dropped in 2/10; scaled
budget 5.08 with 0/10 at the cap (chosen mstop 816-1985) and the term
dropped in 1/10 (seed 6607103, above). On the non-inflated Poisson panel
the two budgets are indistinguishable (0.42 against 0.44; identical fits on
9/10 draws). Per-seed discover time on the zero-inflated panel: 30.4 s
fixed, 65.9 s scaled.

Verdict on the candidates: (a) yes, in cap form — the searchable grid ends
before the cross-validation minimum, rather than cross-validation choosing
an early stop; (b) no; (c) no as stated — the path is a function of travel,
not of step count or step size; (d) only through the selected family's
fitting dynamics — a Poisson mean fit on the same zero-inflated data is
accurate at short path, so the zero mass does not pull the mean down
through the score; it makes the search select the family whose descent is
slow.

## Remedy pricing on development

The full production recorder was run on the development cohorts with
`rate_scaled` as the `discover()` default. For this study the recorder's
routes were pinned explicitly: candidate = production defaults;
guard_only = the production configuration with `mean_budget="rate_scaled"`
pinned (the identity control — its rows are identical to candidate's on
the zero-inflated panel); baseline = the pre-guard ablation with
`mean_step="fixed"` and `mean_budget="fixed"` pinned, the configuration
confirmation 2 evaluated as its candidate. 1497 of 1500 records are ok; the
3 failures are `supplied_fixed` rows raising the fixed-step instability
error on large-count interaction panels, a diagnostic control only.

Passed: deterministic incumbent; the binding panel's mean-error gate
(upper 95% -36.18%, against -15.66% for confirmation 2's candidate on its
cohort); supplied tuned efficiency (count-size upper -0.22%); additive
representation (curve terms in 10/10 draws and 0 interactions on both
nonlinear panels); genuine-interaction retention (8/10, 8/10, 9/10 draws);
false-inclusion McNemar on all null/noise panels; every Gaussian gate —
mean error identical to baseline (upper 0.0) on 9/10 panels and +7.70% on
heavy_tail, scale adoption and noise inclusion unchanged.

Failed:

1. The binding efficiency gate at n = 10: count-size upper 95% +2.127%
   (width upper -1.09%, coverage 0.9008 eligible), against the +1% bound.
   The point estimate from the earlier development screen of the same
   opt-in is -0.014%; ten draws cannot resolve a +1% upper bound for this
   remedy.
2. Three other-panel mean-error gates, paired candidate against the
   fixed-budget ablation: low +33.11%, zero_inflated +38.14%,
   noise_zero_inflated +18.15%, against the +5% bound.

A probe re-ran those panels plus the binding panel on a tree with the
previous default restored, identical draws and pins. It reproduces the
recorded development values for the previous configuration to full
precision (count-size upper +3.258249% against +3.25825% recorded in
`OPTIMIZATION_DEVELOPMENT.md`; width upper -0.388355% against -0.38835%),
and under the previous configuration the three panels pass at exactly 0.0 —
candidate and ablation fits identical. The failures are therefore caused
by the rescale, not by the cohort.

Their anatomy: the candidate's mean latent MSE improves on all three
panels, but the frozen gates measure paired per-draw percent harm, and the
rescale produces a few large ones. On the low-count panel, two draws
(6607101, 6607106) lose every term at the scaled screening/elimination
stage (final formula the intercept alone; MSE 0.0150 and 0.0155 against
0.0109 and 0.0057) while other draws improve up to sixteen-fold. On the
plain zero-inflated panel the candidate is better on 7/10 draws and worse
on 3, the worst a +186% change at an unchanged formula (mstop 2000 against
400, MSE 0.687 against 0.241): the longer tuning path itself, chosen by
cross-validation on the count loss, can walk away from the latent mean.
On the noise panel the candidate is better on 8 of 9 differing draws, with
one +185% draw.

## Verdict

Do not promote. `mean_budget` stays opt-in with default `"fixed"`. The
shrinkage is path-travel starvation of the negative-binomial mean the
search selects on zero-inflated responses, and the scaled budget repairs
it on the binding panel — latent-mean error upper -36.18%, count-size
point estimate at parity with the comparator. But bought uniformly, the
same travel also buys per-draw regressions on panels that were not broken,
and the protocol's other-panel gates are do-no-harm gates: a remedy that
fails them on development does not earn a confirmation draw. The reserved
confirmation-3 ranges (count seeds 6799500-6799549, Gaussian seeds
6799600-6799649) were not touched and remain reserved.

Potential follow-up candidate: extend a tuning grid only while its minimum
remains at the grid's end, then price that rule against the same full gate
set on development draws. The plain zero-inflated panel's regressions came from
the tuning path itself at an unchanged formula, so even that narrower rule
has to clear the same gates; it is recorded here as a direction, not as a
priced candidate.

Follow-on: the starved-fit direction named above was priced in
`STARVED_TRAVEL_STUDY.md` (this directory), against the same gate set on
the same development draws. The rule — raise the tuning cap in travel
units so the grid doubles past 400 steps only while its minimum sits in
the boundary band — repairs the binding panel by more than the blanket
remedy (paired mean-error upper -59.99%) but fails the same do-no-harm
gates: on the plain zero-inflated panel its fits are identical to the
rate-scaled remedy's on all ten draws, including the +186% regression,
because a minimum at the grid's end is a property of the selected
family's descent on zero-inflated data, not a marker of the panels
where travel is missing. It was not shipped; confirmation 3 remains
reserved and untouched.
