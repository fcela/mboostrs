# mboostrs before and after the #357 fixes

The first mboostrs run of this battery (`runs/mboostrs/`, graded in
`eval/mboostrs/`) used mboostrs as of 2026-10-09. The fixes it prompted
(issue #357 and the follow-up issue #388) were merged, and a new Haiku
analyst, given the same `BRIEF.md` and assignment and no access to the first
run, wrote a new pipeline (`rerun/mboostrs/`). The grader ran it on the same
dev and hidden replicates with the same harness, 4 threads and a 10-minute
limit, against mboostrs at commit 53875da (`rerun/eval/mboostrs/`). The
machine was shared with other jobs (load average 11 to 38), so the run
times below overstate an idle machine.

Both columns are graded by `grade.py` on replicates H1–H8 (H), N1–N2 (null)
and S1–S2 (S). The two pipelines differ in their own choices as well as in
the library version, so a difference between the columns is the combined
effect of both.

| Measure | Before (2026-10-09) | After (2026-10-10) |
|---|---|---|
| runs completed (of 12) | 9 | 12 |
| mean-effect TPR / FDR (H) | 0.88 / 0.02 | 1.00 / 0.00 |
| interaction TPR / FDR (H) | 0.81 / 0.00 | 0.38 / 0.25 |
| spread-driver TPR (H) | 0.88 | 1.00 |
| false claims, null draws (N1, N2) | crash, crash | 1, 0 |
| mean-effect TPR (S) | 0.58 | 0.33 |
| query coverage (H / S) | 0.69 / 0.56 | 0.73 / 0.44 |
| prediction RMSE of f (H / S) | 0.06 / 2.77 | 0.06 / 0.19 |
| 90% PI coverage (H / S) | 0.90 / 0.89 | 0.90 / 0.91 |

Follow-up refits (`refit.fit_predict` on dev, one per dropped column, 10
in all): median 2.6 s after, 7.9 s before. Large n (500,000 rows): 74 s
after, 252 s before.

## Improvements

- Every replicate ran. The first pipeline crashed on both null draws (it
  built the formula `y ~ `) and timed out on H2 (a stalled REML refit, fixed
  in #371).
- Small-sample predictions no longer extrapolate: RMSE of the mean on S fell
  from 2.77 to 0.19, with 90% prediction-interval coverage of 0.91.
- Mean-effect and spread-driver recovery on H reached 1.00 with no false
  mean effects.

## Regressions

- Interactions on H: the age-by-income interaction was found in fewer
  replicates (TPR 0.81 to 0.38) and a false pair was claimed in three (FDR
  0.25). The rerun pipeline screens pairs with `screen_interactions` and
  adopts a product term only above two standard errors; the first pipeline
  used tensor terms, which the first run's `conformal_interval` defect
  pushed the second analyst away from.
- Small samples: the pipeline claims fewer effects at 1,500 rows (mean-effect
  TPR 0.58 to 0.33), and the effect-query intervals cover less (0.56 to
  0.44).
- One null draw (N1) has one false interaction claim.
