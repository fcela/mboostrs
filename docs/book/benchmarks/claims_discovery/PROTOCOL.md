# Claim-cost discovery battery: pre-registered protocol

Written 2026-10-09 before any analyst started. The grader and the key are fixed
at this point; the sha256 of this file is recorded in the report.

## Comparison set

mboostrs 0.1.3, XGBoost 3.4.0, LightGBM 4.7.0, CatBoost 1.2.10, InterpretML EBM
0.7.8, pyGAM 0.12.0, statsmodels 0.15.0. One analyst per library: a Claude Haiku
agent given BRIEF.md verbatim plus a one-paragraph assignment naming the library,
its run directory, and its documentation source. No assists after launch.

## Battery

Covariates: the 20,640 real California 1990 block groups, plus a simulated
`roof_class`. Test rows (25%) drawn with probability rising in income.
Missingness: `housing_median_age` MAR on latitude (4% south / 16% north),
`total_bedrooms`/`ave_bedrooms` 1% MCAR, redrawn per replicate.

Planted mean (y on log scale):
-0.45 log(inc/3.5) + 0.30 s((age-35)/2.5) + 0.30 s((age-35)/2.5) s((3-inc)/0.6)
+ 0.35 log(clip(occ,1,6)/3) + two Gaussian spatial bumps + roof {C: .15, D: .35}.
Noise: Student-t(5) with sd 0.12 + 0.30 sqrt(400/max(households,20)).

Replicates: dev (given), H1–H8 (new noise and missingness), N1–N2 (y has no
relation to any covariate, constant sd), S1–S2 (800 training rows).

Key: mean-active {median_income, housing_median_age, ave_occup, roof_class,
latitude, longitude}; interactions {(latitude, longitude),
(housing_median_age, median_income)}; variance driver {households}.

## Execution

Each delivered `run.py` is run by the grader on dev (seed 1) and on all 12
hidden replicates (seed 1), one library at a time, 4 threads
(`OMP_NUM_THREADS=RAYON_NUM_THREADS=4`), 10-minute limit per run. A failed or
timed-out run scores as no claims, no intervals and no predictions on that
replicate.

## Measurements

- Discovery (H, S): per replicate, TPR and FDR of `mean_effects` over the 13
  covariates; TPR/FDR for interaction pairs; TPR/FDR for variance drivers.
  Households proxies (population, total_rooms, total_bedrooms) count as false
  positives and are also reported separately. Null replicates: number of false
  claims (mean effects + interactions + variance drivers).
- Shapes: share of correct shapes among claimed income, age, occupancy, roof.
- Effect queries (H, S, N): 90% interval coverage of the replicate's true
  average contrast; RMSE of the estimates; mean width. A missing bound counts
  as non-covering.
- Prediction (H): RMSE of `mean` against the true mean f; 90% coverage of y;
  mean width; interval (Winkler) score at alpha = 0.1; coverage by households
  quartile; coverage on rows with missing age.
- Follow-up session (D5): grader calls `refit.fit_predict` on dev, 10 times,
  dropping each of the 10 non-coordinate covariates in turn except
  `roof_class` (median and p90 seconds, 4 threads, sequential).
- Large n (D6): one `fit_predict` on 500,000 rows resampled from dev train with
  jittered numerics; pass if it finishes in 300 s.
- Reproducibility (D10): dev seed 1 run twice by the grader; compare with the
  analyst's `dev_out` and with each other.

## Rubric mapping (scorecard.md anchors, operationalised)

D1: 1 = no data-driven claims; 2 = mean TPR >= 0.75 but mean FDR > 0.25 or
null replicates average > 1 false claim; 3 = mean TPR >= 0.75, FDR <= 0.25,
null false claims <= 1; 4 = 3 plus one of interactions / variance driver
recovered with TPR >= 0.5 and FDR <= 0.5; 5 = 3 plus both.
D4: 1 = no intervals; 2 = coverage < 0.70; 3 = 0.70 <= coverage < 0.82 or
> 0.98; 4 = coverage in [0.82, 0.98]; 5 = 4 plus the report states the
magnitudes on the outcome scale with their intervals.
D5: median follow-up refit > 60 s = 1; 10–60 s = 2; 1–10 s = 3; < 1 s = 4;
5 requires incremental updating.
D6: 1 = a run failed on any replicate; 2 = all ran but small-n FDR > 0.5 or
small-n prediction RMSE > 2x the full-data RMSE; 3 = all ran and small-n
degrades within those limits; 4 = 3 plus large-n pass; 5 = 4 plus the pipeline
emits a shift/extrapolation warning for the test rows.
D9: 1 = no prediction intervals; 2 = overall coverage outside [0.86, 0.94];
3 = overall in [0.86, 0.94]; 4 = 3 plus every households quartile in
[0.84, 0.96]; 5 = 4 plus the pipeline warns where intervals are unreliable.
D10: 1 = repeated runs differ; 2 = identical runs, no tuning log; 3 = identical
and a complete tuning log; 4 = 3 plus the grader's rerun matches `dev_out`;
5 = 4 plus data/code/environment hashes recorded in the outputs.
D2, D3, D7, D8: judged from REPORT.md, FRICTION.md and run.py against the
scorecard anchors, with the evidence quoted. D3 also counts key facts stated
correctly in REPORT.md (income down, age threshold, occupancy up, roof D
highest, spatial pattern, age x income, households spread, house value not a
driver) and false statements.

No weighted aggregate is reported without the per-dimension profile.
