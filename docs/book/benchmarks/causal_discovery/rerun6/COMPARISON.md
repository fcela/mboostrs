# mboostrs rerun 5 against rerun 6

Rerun 6 is the seventh mboostrs analyst, run wheel-only on main 26fed4d. Between the runs, #486 added `outcome_missing` to `iv_effect`, made `ci_test` name the column that breaks an exact identity (with a NaN p-value), mapped internal column names in `ci_test` warnings, reported gains for requested `effect_model` terms, issued repeated warnings once per call, and stopped `causal_trap_report` from blocking on absent post-treatment columns. Both columns are graded by `grade.py` on H1–H6, N1–N2 and S1–S2 with the same harness.

| Measure | Rerun 5 (48b7edd) | Rerun 6 (26fed4d) | Best other library |
|---|---|---|---|
| Valid adjustment set (of 8) | 6 | 8 | EconML 8 |
| Role accuracy | 0.81 | 0.75 | EconML 0.94 |
| Effect modifiers found | 0.00 | 0.12 | LightGBM 0.22 |
| False claims, no-effect draws (mean) | 0.5 | 0.0 | EconML, statsmodels 0 |
| ATE 90% coverage | 0.75 | 0.75 | EconML, statsmodels 0.88 |
| ATE RMSE ($) | 2,622 | 2,747 | statsmodels 593 |
| All-query coverage | 0.83 | 0.88 | DoubleML 0.88 |
| PEHE ($), mean over H1–H6 | 1,905 | 1,875 | EconML 1,703 |
| Per-applicant 90% coverage | 0.82 | 0.61 | statsmodels 0.89 |
| Ranking regret share (lower is better) | 0.29 | 0.97 | EconML 0.36 |
| S1 (1,500 rows): ATE estimate / PEHE | 9,550 / 8,862 | 10,889 / 7,607 | truth 4,182 |

Scorecard, mboostrs: D1 rose from 2 to 3 (every adjustment set valid, no false modifier claims), level with EconML. D9 fell from 4 to 2, because per-applicant coverage fell below 0.80. D6 stays at 2 (the S1 blow-up). Weighted: 3.3 fully automated and 3.3 semi-automated (EconML 3.3 and 3.2, statsmodels 3.5 and 3.3).

The per-applicant coverage and the ranking moved with the analyst's own construction: rerun 6 built a doubly robust second stage from `mb.gam` on the augmented scores, whose intervals cover second-stage sampling error only, and ranked inside the overlap region, leaving 540 test applicants unranked. The library has no doubly robust second stage (#487). The small-sample ATE comes from `aipw`'s automatic outcome model (pooled with treatment interactions when the sparse-arm check fires); the measured alternative is in #484 and needs a decision.
