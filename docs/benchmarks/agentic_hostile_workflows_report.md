# Agentic discovery benchmark v4 — hostile workflows: graded report

> **Historical grading note.** This report preserves grades assigned under the
> benchmark's local v4 rubric; those grades are not scores under the current
> canonical `mboostrs-scorecard/2.0`. The historical rubric is archived below
> with its criteria. See the [scorecard migration crosswalk](../scorecard/migration.md)
> for the dimension mapping and limits on reuse. In particular, the old D1 real
> case includes held-out predictive accuracy, which belongs to canonical D11;
> the archived grades do not amount to complete validated coverage of version
> 2.0.

Date: 2026-10-12. 18 runs (arms A = mboostrs + skills, B = sklearn/XGBoost,
C = mboostrs no-skills) on 6 cases: missingness, leakage, heteroscedastic,
confounding, sharp_smooth, real (diabetes). Grading per the
[rubric](agentic_hostile_workflows/agentic_hostile_workflows_rubric.md),
truth per `TRUTH_KEY.json` in
`docs/benchmarks/agentic_hostile_workflows/`.

Per project convention all benchmarks are recorded in the docs; this is the
record of the v4 battery. Protocol:
[agentic_hostile_workflows_protocol.md](agentic_hostile_workflows_protocol.md).

Agents ran against main with the pre-fix `screen_discontinuities`
(independent Kimi k3 verification found two merge-blockers in it:
collinear-column crash, steep-smooth typed as jump — addressed in PR #497).
The v4 sharp_smooth case is a MILD steepness (logistic k=6, validated to type
smooth); it does not probe the steep regime the verification flagged.

## Headline

Every arm got every case's core verdict right — 18/18 on the planted
structure, traps avoided. The battery discriminates on **cost, not
correctness**. The skills arm (A) was the slowest on 5 of 6 cases and the most
fit-hungry overall (~686 reported fits vs ~378 for C, ~355 for B), with zero
accuracy advantage. Arm B (sklearn/XGBoost) matched mboostrs on every case,
including detecting the heteroscedasticity it cannot natively model.

## Per-case results (D1 correctness; all other dimensions below)

| Case | A (skills) | B (sklearn/XGB) | C (no skills) |
|---|---|---|---|
| missingness | kink 0.40/+2.4, x2 hump, x3 +0.98; impute+indicators, fill-bias checked vs complete-case ✓ | kink 0.36/+2.4, x2 hump, x3 +1.05; XGB native-NaN discovery + complete-case/imputed sensitivity ✓ | kink 0.40/+2.41±0.12, x2 hump, x3 +1.07±0.08; complete-case + imputation sensitivity ✓ |
| leakage | leak caught+excluded; x0 +2.10, x1 ∩ peak 0.50 ✓ | leak caught 4 ways; x0 +2.08, x1 ∩ peak 0.51; honest CV R² 0.66 vs illusory 0.995 ✓ | leak caught; x0 +2.0, x1 ∩ peak 0.50; noted the include-it-"improves"-OOF trap (16.7 SE) ✓ |
| heteroscedastic | x0 +1.93, x1 hump amp 1.44; σ ratio ≈5.5× in x2; LSS fit, scale effect robust ✓ | mean right; σ≈0.22+1.42·x2 (~5×); explicit: sklearn can't do distributional regression, second-stage variance, point forecasts unaffected ✓ | x0 +2.0, x1 hump amp 1.56; σ≈exp(−1.2+1.7·x2) ~5×; Gaussian LSS; z=+4.97 diagonal candidate correctly left untyped (type_z=0.28) ✓ |
| confounding | x0 +1.58 within groups, g −2.04; Simpson reversal explained ✓ | x0 +1.58 cond., g −2.04; marginal −1.35; associations-only language ✓ | x0 +1.58, g −2.04; reversal confirmed 3 ways ✓ |
| sharp_smooth | steep-smooth, NOT jump: screen type_z≈0, smooth beats step ~13 SE on CV ✓ | sigmoid swing 2.94, center 0.51, width 0.16; sigmoid CV MSE 0.166 vs step 0.327; narrow-band collapse ✓ | steep-continuous, NOT jump: screen quiet raw+residuals; jump indicator adds 0.00013±0.00013 OOF ✓ |
| real (sealed test) | RMSE 58.51 (CV est 53.5) | RMSE 59.74 (CV est 53.99±2.69) | RMSE 59.13 (CV est 54.3±0.4) |

Truth values: missingness kink 0.40/+2.5, x2 sine amp 1.5, x3 +1.0; leakage
x0 +2.0, x1 amp 1.5; heteroscedastic σ=0.25+1.5·x2; confounding within-group
x0 +1.5, g −2.0, marginal x0 −1.35; sharp_smooth 3.0·logistic(6(x0−0.5)) +
1.5·x1, no jump.

Real-data scoring: all three pickles loaded and predicted cleanly through the
venv (mboostrs 0.1.3, sklearn 1.9.1) — no scoring-path issues. Gaps of ~4–6
points between CV estimates and test RMSE are consistent with test-set
sampling noise at n=88 (B pre-quantified ±8 in its report); no suspicious
optimism anywhere. Rank spread ≈1.2 points: effectively a tie.

## Historical local scores (D1/D2/D3/D5/D8/D10; D4/D6/D7/D9 n/a)

All runs: D1=5. D2=5 everywhere — uncertain values were qualified (kink
±0.04–0.05), associations-only language in confounding, CV-vs-test gaps owned
in real, leakage caveats stated. D3=5 everywhere except A-real=4 (260 fits
for a linear+bbs model). D5: 5 except
A-heteroscedastic=4 (gap step skipped and reported) and C-missingness=4. D8:
A runs 4 (150–260 fits; NaN recipe sensible), B and C runs 5
(B-missingness's native-NaN discovery and sensitivity check; C's
docstring-driven screen discovery held up). D10=5 across the board, with
structured sections on methods and ruled-out alternatives.

## Efficiency (issue #492) — wall-clock and reported fits per run

| Case | A time / fits | B time / fits | C time / fits |
|---|---|---|---|
| missingness | 23 min / ~60 | 12 min / ~11 | 10 min / ~42 |
| leakage | 19 min / ~150 | 8 min / ~60 | 11 min / ~31 |
| heteroscedastic | 24 min / ~50 | 13 min / ~24 | 18 min / ~60 |
| confounding | 16 min / ~16 | 3.5 min / ~45 | 19 min / ~110 |
| sharp_smooth | 20 min / ~150 | 12 min / ~60 | 20 min / ~75 |
| real | 25 min / ~260 | 9.5 min / ~155 | 9 min / ~60 |
| **total** | **~127 min / ~686** | **~58 min / ~355** | **~87 min / ~378** |

Fit counts are as each agent counted them (conventions differ: A-leakage
counts ~100 term_report fits; C-heteroscedastic counts CV fold-fits).
The skills arm costs the most on 5 of 6 cases and reaches the same answers.
Versions 2 and 3 showed no correctness difference; v4 also found no
correctness difference and measured higher costs for the skills arm.

## Mechanism notes

- **Discontinuity-screen typing on the mild case.** A and C both ran the
  screen on sharp_smooth; candidates typed smooth (type_z≈0) or stayed quiet.
  This is consistent with the verification finding:
  the failure is at steeper transitions (sigmoid k=8, kink) than the
  battery's k=6. The battery needs a steeper arm to probe the bug.
- **Leakage detection.** All arms caught `post`
  (r=0.9986) with multiple independent checks; C explicitly named the trap
  (including it "improves" OOF risk 16.7 SE).
- **Missingness:** mboostrs' hard NaN refusal forced explicit handling; all
  three mboostrs arms chose defensible recipes (impute+indicators with
  fill-bias audit; complete-case + imputation sensitivity). B used native-NaN
  discovery, imputed quantification and a sensitivity check.
- **Confounding:** all three reported the reversal (marginal −1.35 →
  conditional +1.5–1.6) with no causal overreach. Fastest correct run of the
  battery: B-confounding, 3.5 min.
- **Tooling issue (library-side):** `reference_gap`'s tree reference hung
  twice on the heteroscedastic data (0% CPU on the Python child; killed after
  9+ and 4+ min). A-heteroscedastic skipped the step and reported the
  failure (issue #496).

## Limitations of this grading

- Fit counts are self-reported with mixed conventions; treat totals as
  directional.
- Wall-clock includes agent deliberation, not just compute.
- Agents ran the pre-fix discontinuity screen; the sharp_smooth verdicts don't
  generalize to the steep regime the verification flagged.
- Real-case rank gaps (~1 point) are small relative to test-set variation at
  n=88; the three methods have similar ranks in this battery.
