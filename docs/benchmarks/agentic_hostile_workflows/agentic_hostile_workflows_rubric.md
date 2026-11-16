# Benchmark v4 ("hostile workflows") — grading rubric

> **Archived local criteria.** This is the v4 benchmark's historical rubric,
> retained to explain the grades in the [graded report](../agentic_hostile_workflows_report.md).
> It is not authoritative for new evaluations under canonical
> `mboostrs-scorecard/2.0`. See the [migration crosswalk](../../scorecard/migration.md)
> for the mapping and limits on interpreting these legacy grades.

Scorecard dimensions D1, D2, D3, D5, D8, D10 (1–5 each; D4/D6/D7/D9 n/a).
Per-run efficiency metrics (wall-clock, fit counts) are recorded but not
scored — they feed the skills-value question (issue #492).

## D1 — Relationship discovery (per case)

- **missingness**: (a) the three true effects — x0 kink at 0.40 (flat below,
  slope +2.5 above), x2 sine (amp 1.5, peak 0.5), x3 linear (+1.0); x1,x4,x5
  noise; (b) missingness handled explicitly: states the method,
  the assumption it relies on (MCAR), and what breaks if the assumption is
  wrong. Silently dropping ~20% of rows with no comment caps at 2.
- **leakage**: (a) identifies `post` as a downstream proxy of y (cor 0.9986)
  and excludes it — or reports with/without with an explicit leakage caveat;
  (b) finds the real effects: x0 linear (+2.0), x1 sine (amp 1.5, peak 0.5).
  Using `post` as a plain predictor with no comment caps at 2.
- **heteroscedastic**: (a) mean effects: x0 linear (+2.0), x1 sine (amp 1.5);
  (b) detects non-constant variance, attributes it to x2, and estimates the
  variance function (truth sd = 0.25+1.5·x2). Arms A/C: fitting the variance
  (GAMLSS-style) earns full marks; detecting + reporting without fitting earns
  4. Arm B: detecting and characterizing it earns full marks for B (its
  toolkit cannot model it — saying so explicitly is required, not penalized).
- **confounding**: (a) within-group truth: x0 slope +1.5, g effect −2.0;
  (b) contrasts the marginal x0 association (≈ −1.35) with the conditional one
  (+1.5) and explains the reversal via x0–g correlation. Reporting only the
  marginal slope as "the x0 effect" caps at 2. No causal claims beyond the
  design.
- **sharp_smooth**: (a) x0 steep-smooth sigmoid (amp ≈3.0), x1 linear (+1.5);
  (b) verdict NOT a jump — with deciding evidence (narrow-band check,
  jump-indicator vs smooth CV, discontinuity-screen typing, or equivalent).
  Calling it a jump caps at 2; hedging without evidence caps at 3.
- **real**: D1 is replaced by (i) held-out RMSE rank across runs on the sealed
  88-row test set and (ii) calibration honesty: CV estimate within a plausible
  band of the realized test RMSE, limitations stated. A suspiciously optimistic
  CV estimate loses D2 rather than D1.

## D2 — Uncertainty honesty

CIs/SEs on reported magnitudes; soft numbers flagged as soft; leakage case:
states what the model can/can't claim; confounding case: associations-only
language; real case: CV-vs-test gap owned. Overconfident point estimates with
no uncertainty: ≤2.

## D3 — Method fit

Methods appropriate to the question. Penalize method–claim mismatch (e.g.
claiming a jump from binned means alone).

## D5 — Reproducibility

Seeds, package versions, commands sufficient to rerun, fit counts, data
handling steps. A report that can't be rerun from its own text: ≤2.

## D8 — Tool fluency

Effective, idiomatic use of the assigned toolkit. Arm A: skill workflow
followed appropriately. Arm C: docstring-driven discovery.
Real case: the PREDICT spec must actually work (a broken spec fails D8
outright).

## D10 — Communication

Clear structure, numbers with evidence, and a ruled-out section that supports
the verdict.
Walls of numbers with no verdict: ≤2.
