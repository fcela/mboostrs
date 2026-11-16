# Distributional adoption on a single holdout: the NaN standard-error waiver (census C6, C7)

## Evidence summary

The judging-layer census (`docs/design/judging-layer.md`) classifies
the distributional adoption checks as exposed at two sites that share
one construction:

- C6, `_validate_lss`: a distributional family (BetaLSS, GammaLSS or
  GaussianLSS) replaces the mean family when two held-out checks pass
  — its log score is better by the 2% margin
  (`_LSS_LOGSCORE_MARGIN`) and two paired standard errors, and the
  squared error of its mean is not worse by the same rule
  (`_mse_worse`).
- C7, `compare_finished_routes`: the same two checks applied to the
  two finished routes, deciding which route's model is returned.

The paired standard error comes from `_paired_se`, which returns NaN
for a fold-score vector of fewer than two elements. At one fold —
autospec given a single supplied holdout — both standard errors are
therefore NaN, and both clauses were written to waive on a non-finite
SE: the log-score clause read `not np.isfinite(se) or improvement >
2.0 * se`, and `_mse_worse` carries the same waiver. Adoption rested
on the 2% margin alone, on exactly the design where the margin is
read off the rows that also judge it. This study reproduces the
adoption, judges it on fresh data, and applies the census contract
for the family: an uncomputable SE is a refusal, never a waiver.

The mechanism, verified directly against `_paired_se`: a one-element
difference vector returns NaN; a NaN inside the fold differences
propagates to a NaN standard error; an empty comparison is already
guarded at both sites (`_validate_lss` sets the improvement to 0 and
refuses; `compare_finished_routes` returns `None`, and the mean
route stands). A zero-variance fold difference returns a finite
0.0 — the clauses then require the improvement to clear the margin
and 0, which is a degenerate but real proof (the fold differences
are constant), and is unchanged.

One asymmetry matters for the remedy. `_mse_worse` is a veto
predicate shared with the family validators (census C3), where the
waiver can only refuse a candidate, never adopt one; those callers
are out of this study's scope and `_mse_worse` is untouched. Inside
the two adoption checks, however, the mean check is a gate the
candidate must pass, so its standard error follows the adoption
rule, not the veto rule. The house precedent for the adoption rule
already exists in the same file: `_validate_against_gaussian`'s
`require_better` path requires `np.isfinite(se)` outright.

## Audit method

- **Panels.** A Gaussian bell clipped inside (0, 1) — `y = clip(0.5
  + 0.1 x + N(0, 0.1), 0.02, 0.98)`, `x ~ U(-2, 2)` — where the mean
  family (Gaussian) is the truth and BetaLSS is a flexible
  alternative density whose second predictor has no signal to find.
  n = 364 with a single supplied holdout of 91 rows (273 training
  rows), the constructive test's geometry; seeds 8031000–8031039;
  `autospec(data, "y", budget=40, mstop=100, nu=0.1, folds=folds)`.
  Both sides of the family check are genuine log scores here (the
  Gaussian at its in-bag sigma; BetaLSS at its density), so the
  comparison is the check's own quantity.
- **Fresh audit.** For every adoption, both sides were refitted on
  the 273 training rows only, at the step counts the comparison
  used, and read row by row: Gaussian per-row log score at its
  in-bag sigma (the `_gaussian_fold_nll` definition), BetaLSS at
  `-log_pdf` of its per-row parameter predictions. The pair is
  judged on the original 91 holdout rows and on 20,000 fresh draws
  from the same process (stream seed = panel seed + 500,000), as a
  paired per-row z of the log-score difference.
- **C7 directly.** `compare_finished_routes` was called on the
  seed-8031006 panel with the finished routes the family check
  produced there (Gaussian at its tuned 16 steps; BetaLSS at the
  comparison's own optimum), on the single holdout and on the
  default five-fold design.
- **Gamma footprint.** A constant-coefficient-of-variation gamma
  panel (`mu = exp(0.4 + 0.5 x)`, CV = exp(0.1)), same geometry and
  sweep, run only to measure how often the single-holdout adoption
  fires and what the fix changes. Its adoptions are *not* audited
  for worth and are not claimed as waiver cases: the mean side's
  fold score there is the engine loss at the fitted nuisance, on a
  different scale from the distributional log score (GammaReg fold
  scores ≈ 2.0–2.3 against GammaLSS ≈ 1.1–1.4 on the same folds),
  and the same adoption occurs at five folds with a finite standard
  error clearing the bar (seed 8031001: improvement 0.711 nats/row,
  SE 0.177). The waiver is not what carries those adoptions; the
  refusal the fix applies to them at one fold is the contract
  applying where no paired proof exists, nothing more.

## Audit results

**The family check (C6).** On the unfixed code the sweep adopts
BetaLSS in 5 of 40 draws — seeds 8031006, 8031013, 8031024, 8031034,
8031037 — every adoption with `score_difference_se` NaN and the
mean check passed (each adoption's mean excess is within the 2%
margin, so the deciding waiver in all five is the log-score
clause's). The fresh audit refuses all five:

| seed | LSS score | mean score | improvement | per-row z, own holdout | per-row z, 20,000 fresh rows |
|------|-----------|------------|-------------|------------------------|------------------------------|
| 8031006 | −0.7232 | −0.6864 | 0.0368 | 1.04 | −10.30 (mean −0.0205, se 0.0020) |
| 8031013 | −0.7677 | −0.7467 | 0.0210 | 0.87 | −11.98 (mean −0.0201, se 0.0017) |
| 8031024 | −0.8052 | −0.7856 | 0.0196 | 0.56 | −8.94 (mean −0.0216, se 0.0024) |
| 8031034 | −0.8134 | −0.7893 | 0.0241 | 0.92 | −9.50 (mean −0.0155, se 0.0016) |
| 8031037 | −0.8456 | −0.8117 | 0.0340 | 1.60 | −1.92 (mean −0.0033, se 0.0017) |

None of the five clears even its own holdout's per-row bar of 2.0;
on fresh rows all five lose, four of them significantly (z ≤ −8.9).
On the default five-fold design all five panels refuse BetaLSS —
the single-holdout win does not survive a different split of the
same data. The defect is confirmed: the adoptions are real,
repeatable (5 of 40 draws on a truth the candidate cannot improve),
and harmful out of sample in the quantity the check exists to
measure.

**The finished-route comparison (C7).** Called directly on the
seed-8031006 panel, the unfixed comparison returns winner `lss`
with `log_score_check` and `mean_check` both passed and both
standard errors NaN (LSS optimum (13, 13)). Judged on fresh rows
with those exact fits, the adopted route loses: per-row z −12.54
(mean difference −0.0256 nats/row, se 0.0020); on its own holdout
the z is 1.00. On the five-fold design the same call adopts with a
real proof (improvement 0.0235 > margin 0.0160 and > 2 SE 0.0156)
— a legitimate verdict the fix does not disturb (below).

## The fix

In `python/mboostrs/_autospec.py`, at the two adoption sites only:

- `_validate_lss`: the log-score clause now requires
  `np.isfinite(se)` and `improvement > 2.0 * se`; the waiver is
  deleted. The mean check gains a non-finite branch ahead of
  `_mse_worse`: when the squared-error difference's standard error
  cannot be computed, the check fails with
  `lss_mean_check="failed"` and a sentence that says the standard
  error cannot be computed from these folds. (At one fold the
  log-score gate refuses first, so this branch is contract
  alignment for the states where the two standard errors differ in
  finiteness, not the operative refusal.)
- `compare_finished_routes`: `score_ok` requires a finite
  `score_se`; `mean_ok` requires a finite `mse_se` in addition to
  `not _mse_worse(...)`. Both standard errors remain recorded in
  the returned dict, NaN included, with the checks reading
  `failed`.
- `_mse_worse` itself is unchanged, so its veto callers (census
  C3) keep their rule; the recorded unit behaviour
  (`test_the_mean_rule_is_the_squared_error_rule`) is untouched.
- The public `autospec` docstring's single-fold paragraph now
  states the consequence: the distributional adoption checks are
  paired fold tests with no per-row form, so on a single fold they
  cannot pass and the mean family stands.

There is no per-row substitute proof for these checks in the
probes: their held-out score is a log score, not the engine risk
the row losses read — the reason `_single_fold_design` has always
excluded the distributional probes. Refusal is therefore the whole
remedy, and its cost is stated plainly: on a single supplied
holdout a distributional family can no longer be adopted through
either gate, however large its margin. A caller who wants the
distributional question measured supplies at least two folds.

Regression tests in `tests/test_single_holdout_rules.py`: the
recorded seed-8031006 adoption is refused with its NaN standard
error in the evidence; the direct finished-route call refuses on
the same panel; both mean checks refuse when their standard error
is made uncomputable while the log-score check holds a proof (the
state the two gates reach when the standard errors differ in
finiteness); and a moving-precision beta panel is still adopted on
the five-fold design, where the proof exists. The four refusal
tests fail on the unfixed code.

## Evaluation

**Beta sweep, fixed code.** Adoptions fall from 5 of 40 to 0 of
40; the five adopting seeds end with Gaussian, and the other 35
seeds' final families are unchanged.

**Gamma sweep, fixed code.** Adoptions on the single holdout fall
from 26 of 40 to 0 of 40; every former adopter ends with GammaReg,
and the 14 seeds that never adopted keep their exact final family
(8 GammaReg, 6 Gaussian). As recorded above, these adoptions are
not waiver cases — they occur at five folds under a finite
standard error — so this count is the fix's footprint under the
contract, not a defect rate.

**Finished-route comparison, fixed code.** The direct call on the
seed-8031006 panel: at one fold, winner `mean`, both checks
failed, both standard errors recorded NaN. At five folds the
returned record is identical to the unfixed code's — same scores,
same standard errors (0.00782 and 0.000155), same optimum (36, 36),
winner `lss` — multi-fold behaviour is unchanged to all printed
digits.

**Frozen cancer battery (#541).** Reproduced on the unfixed code
before the change and rerun after: bit-identical per draw — the
final formulas, chosen steps (3261 / 3894 / 9989), fits spent and
judged test log losses (0.1244 / 0.0688 / 0.1604; mean 0.1179) all
match to every printed digit. The battery's Binomial route never
reaches either check (no distributional candidate is offered on
it), so the battery stands as a no-regression check on the wider
autospec run.

**Tests.** `tests/test_single_holdout_rules.py`: 15 passed (the
ten existing tests plus the five new ones; the four refusal tests
were verified to fail on the unfixed code). The touched-area
suites (`test_autospec_lss.py`,
`test_autospec_finished_routes_722.py`,
`test_autospec_heteroscedasticity.py`,
`test_compare_specifications_lss.py`, `test_lss_cv_judge.py`): 36
passed. The wider autospec suite (`test_autospec.py`,
`test_autospec_fold_order.py`, `test_autospec_interaction_rules.py`,
`test_autospec_response_semantics.py`,
`test_autospec_transformed_terms.py`,
`test_autospec_zero_inflation.py`, `test_joint_tree_autospec.py`,
`test_spatial_single_holdout.py`, `test_spatial_tree_choice.py`,
`test_bmrf_bspatial_smoke.py`): 194 passed. No failures in any
suite, and no pre-existing failures were encountered.
