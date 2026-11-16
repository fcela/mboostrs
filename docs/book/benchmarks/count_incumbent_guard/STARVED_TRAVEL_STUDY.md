# Starved-fit travel study for issue #679

## Study objective

The mean-shrinkage study (`MEAN_SHRINKAGE_STUDY.md`) isolated the
mechanism — path-travel starvation of the negative-binomial mean the
search selects on zero-inflated responses — and rejected the blanket
remedy (`mean_budget="rate_scaled"` for every count fit) on three
other-panel do-no-harm gates. Its verdict identified a candidate rule: extend a tuning grid only while its
minimum sits at the grid's end. This study defines that rule, prices it
against the same full gate set on development draws, and measures whether
the signal identifies fits that benefit from more updates without harming
other fits.

## The rule

In `discover()` exactly one stage chooses a stopping iteration from a
cross-validated grid: the mean tuning stage. It already doubles its grid
while the cross-validated minimum lies in the last 5% of the grid (the
boundary band), but stops at a cap of 400 steps at the fixed budget —
travel 8 (`mstop * nu`) at the count families' rate of 0.02, against
travel 40 for a Gaussian fit at 0.1. A fit whose minimum still sits in
the boundary band at the cap is the demonstrably starved one; the stage
already records it as an unresolved boundary.

The candidate raises that cap in travel units, and nowhere else:

    cap = max(DEFAULT_MAX_MSTOP * budget_factor, ceil(tune_travel_cap / nu))

with `tune_travel_cap` defaulting to 40, the travel the shipped cap gives
a Gaussian fit. The grid still starts at 100 steps and doubles only while
the minimum is in the boundary band, each doubling budget-admitted as
today; a fit whose minimum is interior never sees a larger grid. At the
fixed budget the count cap becomes 2000 steps, reached only through
boundary doublings; the Gaussian cap is 400 either way; the rate-scaled
cap is 2000 either way. An explicit `max_mstop` still sets the cap
exactly. The rule applies to the tuning grid only. Screening and
comparison fits run at a fixed length (200 steps at the fixed budget) and
choose no stopping iteration — they have no grid and no minimum to read —
and the blanket remedy's low-count regressions came from its scaled
screening stage, so screening stays at the shipped length. Backward
elimination consumes the screening fits, so the candidate's term sets are
the baseline's; the paired comparison isolates the stopping iteration.

The rule touched these candidate fits on the development battery (candidate
fits whose largest tuning grid passed the shipped 400-step cap, of 10 draws
per panel; grids reached in steps):

| panel | extended | grids reached | still at a boundary |
|---|---|---|---|
| nonlinear_zero_inflated (binding) | 10 | 1600, 2000 | 1 |
| zero_inflated | 10 | 800, 1600, 2000 | 5 |
| low | 9 | 800, 1600 | 0 |
| noise_zero_inflated | 6 | 800, 1600, 2000 | 3 |
| interaction_supplied_family | 4 | 800 | 0 |
| original | 2 | 800 | 0 |
| overdispersed | 2 | 800 | 0 |
| nonlinear_poisson | 1 | 800 | 0 |
| nonlinear_interaction_zero_inflated | 1 | 2000 | 1 |
| high, null_poisson, null_zero_inflated, interaction_zero_inflated | 0 | — | 0 |

## Protocol and verification

The full production recorder, on the same development cohorts (count
seeds 6607100-6607109, Gaussian seeds 6293100-6293109, test offset
+100000), with routes pinned explicitly: candidate = production defaults
with the rule as the default behaviour; guard_only = the same
configuration pinned (`mean_step="fixed"`, `mean_budget="fixed"`,
`tune_travel_cap=40`); baseline = the pre-guard ablation with the rule
off (`tune_travel_cap=0`), the configuration confirmation 2 evaluated as
its candidate and the prior study's baseline. Two identity checks pass
before any candidate number is read: the baseline reproduces the prior
study's retained baseline rows exactly on all 130 count rows (status,
formula, mstop, latent-mean MSE, count size, width, coverage), and
guard_only equals candidate on every count row. 1497 of 1500 records are
ok; the 3 failures are the same `supplied_fixed` rows as the prior run,
raising the fixed-step instability error on large-count interaction
panels, a diagnostic control only. Gate definitions and bounds are the
frozen set of the prior study; per-gate values are in
`starved_travel_development_summary.json`.

## Results

Passed:

- Deterministic incumbent: 0 violations.
- The binding panel's mean-error gate, by more than the blanket remedy:
  paired upper 95% -59.99% (blanket: -36.18%). Every draw improves
  (-29.8% to -93.5%); aggregate latent-mean MSE 22.9995 to 6.5856;
  chosen mstop 816-1985 against 400 for every baseline fit. Screening
  and elimination keep the baseline's behaviour, so the gain is the
  stopping iteration's alone.
- Supplied tuned efficiency (count-size upper -0.22%, the prior run's
  value to the digits shown, as it must be: the supplied rows do not
  depend on the routes).
- Additive representation: curve terms in 10/10 draws and 0 interactions
  on both nonlinear panels.
- Genuine-interaction retention: 8/10 draws against the baseline's 8/10
  on both genuine panels (mean-error uppers 0.0% and +0.86%); the
  supplied-family diagnostic holds 9/10 representation.
- False-inclusion McNemar on all null/noise panels (0 added inclusions).
- Every Gaussian gate. The candidate cannot change a Gaussian fit (the
  cap is 400 steps under both configurations), and every Gaussian
  mean-error upper reads exactly 0.0 except heavy_tail at +7.70%, the
  value both prior runs of this recorder record there; it comes from the
  baseline route's pre-guard interaction ablation, not from any
  candidate.

Failed:

1. The plain zero-inflated panel's do-no-harm gate: paired mean-error
   upper +38.14% against the +5% bound — the blanket remedy's value
   exactly, because the candidate's fits on this panel are identical to
   the blanket remedy's on all ten draws. Eight draws improve; two
   regress at unchanged formulas (`bols(x1) + bols(x2)`): seed 6607103,
   +185.6% (mstop 2000 against 400; latent MSE 0.687 against 0.241), and
   seed 6607109, +55.9% (mstop 2000 against 400; 3.383 against 2.170).
2. The noise zero-inflated panel: upper +31.53%. Four draws improve
   -43% to -85%, five are unchanged, and seed 6607109 regresses +185.3%
   (mstop 2000 against 400 at an unchanged formula; 1.221 against
   0.428).
3. The low-count panel at the +5% bound: upper +5.84% (it passes the
   summarizer's +10% reading). The blanket remedy's mechanism does not
   recur — no draw loses a term, because screening is untouched; its
   worst draw there lost every term. The failure is the smaller
   tuning-path kind: several draws a few percent worse on very small
   latent errors, the worst +15.6% (seed 6607102, mstop 1127 against
   400, MSE 0.0062 against 0.0053).
4. The binding efficiency gate at n = 10: count-size point estimate
   +0.43%, one-sided 95% upper +2.75% against the +1% bound (paired SD
   4.01%); width upper -0.70% passes, coverage 0.8999 is eligible. Ten
   draws do not resolve the +1% bound for this candidate either. The
   50-draw development extension designed to read the bound at
   confirmation-grade n was not run: the candidate fails other
   development gates, and no efficiency reading could change the
   verdict — the same reasoning the prior record applied. The reserved
   confirmation-3 ranges (count 6799500-6799549, Gaussian
   6799600-6799649) were not touched.

## Mechanism comparison

The signal does not separate the two kinds of fit. On the plain
zero-inflated panel it is present in its strongest form exactly where it
harms: on seed 6607103 the tuning trace reads minimum-at-the-end at
every grid — chosen mstop 100, 200, 400, 800, 1600, 2000 as the grid
doubles — because cross-validation on the count loss keeps descending
to the ceiling, while the latent mean worsens from 0.241 to 0.687. The
binding panel's traces have the same shape (at the end through 800,
resolving inside the 1600 grid) and accompany the study's largest
gains. A minimum at the grid's end is a property of the selected
family's slow descent on zero-inflated responses — the prior study's
mechanism — not a marker of the panels where travel is missing. Buying
travel conditionally on it buys, on that panel, literally the same fits
as buying it uniformly.

What the conditional rule does avoid is the blanket's screening-stage
harm, and it buys the binding repair at lower cost (mean recorder-row
elapsed on the binding panel 32.7 s per seed against the baseline's
26.2 s; the blanket remedy's discover contrast cost roughly double).
But the do-no-harm gates are failed by the tuning path itself, which is
the part the rule extends.

## Verdict

Do not ship. The candidate implementation (a `tune_travel_cap` argument
to `discover()`, defaulting to the travel ceiling, with regression
tests) was reverted bit-exactly after the run; the tree that carries
this record differs from the branch it started from only in this record
set and the pointer added to `MEAN_SHRINKAGE_STUDY.md`.
`mean_budget` stays opt-in with default `"fixed"`, and the tuning cap
stays 400 steps at the fixed budget. Confirmation 3 stays reserved and
untouched.

A better trigger would use a diagnostic beyond the position of the
count-loss minimum. On the harmed draws the cross-validated count risk
falls monotonically to the travel ceiling while latent-mean error rises.
The binding panel has the same tuning-curve pattern. For this family,
count loss at the grid edge does not identify whether additional updates
improve mean accuracy on zero-inflated data.
