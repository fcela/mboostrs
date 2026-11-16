# Confirmation: the mean-adequacy spread guard on reserved draws (#629)

This decision is frozen before any confirmation observation is generated.
It follows PROTOCOL.md in this directory, the master design in
docs/benchmarks/gap_resolution_design.md ("Sharp means and scale: #629 and
#608"), and the candidate record in PR #775. Raw records stay outside the
repository (#724); the aggregated result is committed as
CONFIRMATION_RESULT.md with confirmation.metadata.json and
confirmation_summary.json.

## Confirmation status

The development inventory (seeds 6293100-6293109, ten draws per process)
established the defect: with curve nominations, discovery's spread stage
adopted a varying scale on the sharp constant-variance controls (axis 3/10,
shifted 2/10) by absorbing mean approximation error left over at the fixed
200-step screening budget. PR #775 merged the candidate: the screening
comparison is now a screen, and a nominated scale term is adopted only if
it wins again in a confirmation comparison whose two specifications each
tune their stopping inside the folds. On the development inventory the
candidate eliminated the sharp constant-variance false adoptions driven by
mean error (axis 3/10 to 0/10, shifted 2/10 to 1/10) and left the genuine
smooth varying-scale panels at 10/10. The one remaining development
adoption (shifted_constant, seed 6293102) is a noise column whose
confirmation change is -0.0066 with SE 0.0016: certified by the
confirmation step itself, statistically detectable and practically zero.
No magnitude floor was introduced; a floor would be a new adoption rule
needing its own freeze.

None of that is confirmation. The reserved seeds 6294100-6294139 have
never been used by any development work. This study runs the frozen
battery on them against the shipped candidate and the full competitor
set, exactly as the issue's acceptance and PR #775's remainder require.

## Candidate and battery

The candidate is discovery with spread on, as shipped on the tree that
runs the recorder (production defaults, including the merged confirmation
step in `_spread()`/`_confirm_spread()`). It is the battery's
`discovery_spread` method, fitted by the frozen generator in
benches/study_sharp_mean_controls.py.

Each draw uses the frozen instruments unchanged: `draw(panel, seed, 1000)`
for the analysis frame and `draw(panel, seed + 100000, 2000)` for the
independent test rows, the common Rust two-fold assignment reserving 500
fitting and 500 calibration rows, and the ten frozen processes (axis and
shifted sharp means crossed with constant and varying scale; smooth
constant, absolute, bell and monotone scale; null constant; heavy-tail
constant variance).

Methods per draw, all recorded with predictions, warnings and failures
retained:

- `discovery_spread`: the candidate.
- `discovery_constant`: discovery with spread off (battery baseline).
- `supplied_constant`, `supplied_varying`: the frozen supplied fixed
  GaussianLSS controls.
- `oracle_residual`: the oracle-mean residual diagnostic. A diagnostic
  lower bound only; never a discovery input or a competitor.
- The five original fixed competitors, fitted by the shared
  `fit_band` instrument in benches/study_discovery_calibration.py:
  `catboost_split`, `catboost_cqr`, `lightgbm_split`, `lightgbm_cqr`,
  `sklearn_cqr`.

Metrics per record are the development record's: latent-mean MSE,
response (point) MSE, interval score, real width, marginal coverage,
true-scale low/high coverage, sharp-boundary coverage within .1 of either
true threshold, the selected family/formula and the scale-adoption
decision. False scale adoption is a selected varying scale in a truly
constant-variance process.

## Acceptance gates

The verdict is read off these gates, frozen here, with Monte Carlo
standard errors reported over the 40 independent draws of each process.
Draws are never pooled across processes.

- G1, completeness. Every panel/seed/method record is present
  (10 x 40 x 10 = 4000) and every candidate record has status ok.
  Failures of other methods are retained and counted, never dropped.
- G2, false adoption (primary). The candidate selects a varying scale on
  zero of the 40 draws of each constant-variance process
  (axis_constant, shifted_constant, smooth_constant, null_constant,
  heavy_tail). Any adoption fails this gate and is reported with its
  confirmation change and SE from the discovery log. Zero means zero:
  the development inventory's one practically-zero certified adoption is
  recorded above precisely so this line is drawn with open eyes.
- G3, genuine-scale recall. The candidate selects a varying scale on at
  least 90% of the 40 draws of each smooth genuine-scale process
  (smooth_absolute, smooth_bell, smooth_monotone), the level the
  development inventory showed at 10/10 per process. Recall on the sharp
  varying-scale processes is reported without a gate: the frozen baseline
  adopts 0/10 there both before and after the candidate, so it is a
  pre-existing separate limitation, not an effect of this rule.
- G4, no mean harm. On each constant-variance process, the candidate's
  paired latent-mean-MSE percent excess over `discovery_constant` on the
  same draws has a one-sided 95% Student-t upper limit of at most 10%.
- G5, competitor comparison (reported, not a closing gate). Per process,
  the candidate is reported alongside all five competitors on latent-mean
  MSE, point MSE, interval score, width and coverage, with coverage
  eligibility [0.88, 0.92] marked and the paired real-width percent
  excess against prespecified LightGBM CQR given with its one-sided 95%
  upper limit. Per the master design, the one-percent efficiency claim
  belongs to #630; this study reports the comparison in full and makes
  no efficiency claim for #629. A competitor that fails on a draw is
  excluded from that draw's paired contrast only, and the exclusion is
  counted in the report.

#629 closes only if G1-G4 pass on the reserved seeds with G5 reported in
full. Any failed gate leaves the issue open, with the failing numbers
stated in CONFIRMATION_RESULT.md.
