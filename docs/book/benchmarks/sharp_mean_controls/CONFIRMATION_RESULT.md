# Confirmation result: the mean-adequacy spread guard passes on reserved draws (#629)

The confirmation frozen in CONFIRMATION_DECISION.md was run on the
reserved seeds 6294100-6294139 (40 draws per process, ten processes,
independent test seed +100000) against the shipped candidate: discovery
with spread on, as merged by PR #775 on main
81479457dc65fcc7c37677f7a362dd3357218b37. All four closing gates pass.
Per the frozen decision, this closes #629.

## Battery

4,000 records (10 processes x 40 draws x 10 methods), 4,000 with status
ok and zero failures in any method. Methods per draw: the candidate
(`discovery_spread`), discovery with spread off, the supplied fixed
GaussianLSS constant/varying controls, the oracle-mean residual
diagnostic, and the five original fixed competitors (`catboost_split`,
`catboost_cqr`, `lightgbm_split`, `lightgbm_cqr`, `sklearn_cqr`, fitted
by the shared `fit_band` instrument). Every candidate fit selected the
Gaussian family. Raw records stay outside the repository (#724);
per-journal SHA-256 digests, package source identities and versions are
in confirmation.metadata.json, and the full per-panel/per-method
aggregates are in confirmation_summary.json.

Before any reserved draw was generated, the runner and the built tree
were exercised on the spent development seeds: they reproduce PR #775's
post-change development figures exactly (axis_constant 0/10 adoptions,
latent-mean MSE 0.342; shifted_constant 1/10, the seed-6293102 noise
column, MSE 0.390).

## Gates

- G1 completeness: 4,000 of 4,000 records present; all 400 candidate
  records ok. Pass.
- G2 false adoption: the candidate selects a varying scale on 0 of 40
  draws on every constant-variance process (axis_constant,
  shifted_constant, smooth_constant, null_constant, heavy_tail) —
  0 of 200 draws. Pass. For contrast, on the development seeds the
  pre-guard procedure adopted on 3/10 (axis) and 2/10 (shifted) draws,
  and the candidate on 0/10 and 1/10.
- G3 genuine-scale recall: 40 of 40 draws on each smooth genuine-scale
  process (smooth_absolute, smooth_bell, smooth_monotone), against the
  frozen 90% line. Pass.
- G4 no mean harm: on every constant-variance process the candidate's
  paired latent-mean-MSE percent excess over discovery with spread off
  is exactly 0 (mean and one-sided 95% upper limit), over 40 of 40 pairs
  with no exclusions: with no scale adopted, the two procedures return
  the same fit there. Pass.
- G5 competitor comparison: reported below in full; no closing gate and
  no efficiency claim is made for #629 (per the master design, the
  one-percent target belongs to #630).

Recall on the sharp varying-scale processes remains 0 of 40 on both
(axis_varying, shifted_varying), reported as frozen in the decision:
the baseline adopts 0/10 there before and after the candidate, so this
is a pre-existing separate limitation, unchanged by the guard.

## Candidate results per process

| process | scale adopted | latent-mean MSE (SE) | interval score | coverage | boundary coverage |
|---|---|---|---|---|---|
| axis_constant | 0/40 | 0.349 (0.026) | 3.312 | 0.899 | 0.798 |
| axis_varying | 0/40 | 0.648 (0.029) | 6.239 | 0.901 | 0.879 |
| shifted_constant | 0/40 | 0.234 (0.019) | 2.876 | 0.898 | 0.834 |
| shifted_varying | 0/40 | 0.402 (0.019) | 5.908 | 0.901 | 0.899 |
| smooth_constant | 0/40 | 0.043 (0.005) | 4.234 | 0.897 | 0.898 |
| smooth_absolute | 40/40 | 0.068 (0.006) | 4.506 | 0.897 | 0.912 |
| smooth_bell | 40/40 | 0.033 (0.004) | 3.298 | 0.897 | 0.900 |
| smooth_monotone | 40/40 | 0.068 (0.006) | 4.639 | 0.897 | 0.898 |
| null_constant | 0/40 | 0.001 (0.000) | 4.147 | 0.897 | 0.898 |
| heavy_tail | 0/40 | 0.034 (0.004) | 4.555 | 0.903 | 0.904 |

## Competitor comparison

The candidate is coverage-eligible ([0.88, 0.92]) on all ten processes,
as is prespecified LightGBM CQR on all ten. The candidate's intervals
are narrower than LightGBM CQR's on every process at that eligibility:
the paired real-width percent excess (candidate minus reference) has
mean and one-sided 95% upper limit below zero throughout — axis_constant
-36.3% (upper -34.2%), shifted_constant -41.6% (-40.5%), axis_varying
-15.7% (-14.2%), shifted_varying -16.4% (-15.0%), heavy_tail -13.4%
(-12.5%), smooth_constant -8.0% (-7.2%), null_constant -5.2% (-4.5%),
smooth_absolute -4.5% (-3.5%), smooth_monotone -4.5% (-3.7%),
smooth_bell -3.6% (-2.6%). No competitor record failed, so no paired
contrast excludes any draw.

The candidate holds the best mean interval score on eight of ten
processes. The exceptions are the two sharp varying-scale processes,
where a competitor or supplied control that does not face the sharp-mean
search problem scores better (axis_varying: catboost_split 6.002 against
the candidate's 6.239; shifted_varying: supplied_varying 5.306 against
5.908) — the same processes where scale recall stays at the frozen
baseline's zero. The full comparison, all five competitors with paired
differences and Monte Carlo standard errors, is in
confirmation_summary.json and the G5 section of the retained gate file.

## Incidents

None. The battery ran in two panel-parallel processes and completed
without interruption; no draw was rerun and no record was regenerated.
