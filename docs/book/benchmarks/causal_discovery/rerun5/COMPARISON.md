# mboostrs rerun 4 against rerun 5

Rerun 5 is the sixth mboostrs analyst, run wheel-only on main 48b7edd (see DEPARTURES.md). Between the two runs: the `causal-roles` guide topic was rewritten around the disjunctive cause criterion (#480), and #460 fixed `aipw`'s two-part default and `effect_model`'s automatic term choice. Both columns are graded by `grade.py` on H1–H6, N1–N2 and S1–S2 with the same harness.

| Measure | Rerun 4 (58d8a51) | Rerun 5 (48b7edd) | Best other library |
|---|---|---|---|
| Valid adjustment set (of 8) | 3 | 6 | EconML 8 |
| Role accuracy | 0.77 | 0.81 | EconML 0.94 |
| ATE 90% coverage | 0.62 | 0.75 | EconML, statsmodels 0.88 |
| ATE RMSE ($) | 2,071 | 2,622 | statsmodels 593 |
| All-query coverage | 0.84 | 0.83 | DoubleML 0.88 |
| False claims, no-effect draws (mean) | 1.0 | 0.5 | EconML, statsmodels 0 |
| PEHE ($), mean over H1–H6 | 2,755 | 1,905 | EconML 1,703 |
| Per-applicant 90% coverage | 0.64 | 0.82 | statsmodels 0.89 |
| Ranking regret share (lower is better) | 1.00 | 0.29 | EconML 0.36 |
| Effect modifiers found | 0.09 | 0.00 | LightGBM 0.22 |

Scorecard (docs/scorecard/scorecard.md), mboostrs only: D9 rose from 2 to 4 (per-applicant coverage inside 0.80–0.97 and small-sample ATE intervals covering). D6 fell from 4 to 2: on the 1,500-row replicate S1 the per-applicant error (PEHE 8,862) is more than twice the full-size error, and the S1 ATE estimate is 9,550 against a truth of 4,182 (covered only by a wide interval). D3 fell from 4.0 to 3.5 (no modifier established). The weighted scores are 3.3 fully automated and 3.3 semi-automated in both runs; EconML 3.3 and 3.2, statsmodels 3.5 and 3.3.

The adjustment-set and ranking gains are the improvements the fixes targeted. The small-sample effect estimates are a new defect (#484).
