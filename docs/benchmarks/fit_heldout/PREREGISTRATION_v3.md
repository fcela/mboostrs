# Pre-registration v3

Written 2026-11-21 before any v3 run. Identical to PREREGISTRATION_v2.md
(same arms, 95 datasets, known-effects key, onceover strata, scoring, primary
analyses A, B, C and secondaries) except:

- seeds 3000, 3001, 3002;
- the new and quick arms run the working tree after one more change found in
  the v2 results: autospec takes the line search when the budget ran out
  before the stopping depth was measured, because no fit checked the fixed
  step (student_performance_por seed 2000: test MSE 138.9 against 7.2).

The v2 results stand as the measurement of the code before that change.
