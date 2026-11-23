(record-agent-benchmarks)=
# Agent workflow benchmarks

This record summarizes the experimental protocols, aggregate outcomes, and key lessons from four multi-library studies in which autonomous LLM agents analyzed datasets from a written brief: [](../../benchmarks/hostile_workflows_v4.ipynb), [](../../benchmarks/causal_discovery.ipynb), [](../../benchmarks/claims_discovery.ipynb), and [](../../benchmarks/pm25_forecast_competition.ipynb).

## Hostile workflows (six traps, three arms)

**Design.** Three arms—Arm A (`mboostrs` with bundled agent guide), Arm B (`scikit-learn` + `XGBoost`), and Arm C (`mboostrs` using only public docstrings/`help()`)—each analyzed six datasets (18 runs total): five planted-truth traps (missingness with kink/hump effects, target leakage with $r = 0.9986$, heteroscedasticity with $5\times$ scale swing, Simpson's confounding reversing slope from −1.35 to +1.50, and a steep-but-smooth sigmoid $k = 6$) plus a real diabetes dataset scored on a sealed 88-row test split.

**Outcomes.** All 18 runs identified every planted trap and core effect accurately; on the sealed real test set, RMSEs were effectively tied within sampling noise (58.51 for Arm A, 59.74 for Arm B, 59.13 for Arm C). The arms separated on **analyst cost**, not correctness: Arm A spent ~127 min and ~686 reported fits across the six tasks, versus ~87 min / ~378 fits for Arm C and ~58 min / ~355 fits for Arm B. On the heteroscedastic task, `reference_gap`'s tree reference hung twice and had to be skipped by Arm A.

*Historical rubric note:* The original report assigned scores under a local v4 rubric (archived in `docs/benchmarks/agentic_hostile_workflows/agentic_hostile_workflows_rubric.md`), which is distinct from canonical `mboostrs-scorecard/2.1`.

## Job-training causal battery

**Design.** Five libraries (`mboostrs`, `EconML`, `DoubleML`, `statsmodels`, `LightGBM`) were each given to an autonomous analyst with a written program-evaluation brief: non-randomized job-training enrollment, 16% missing follow-up earnings, and thin propensity overlap above an eligibility cutoff. Each submitted pipeline was evaluated unchanged on hidden replicates (6 signal replicates H1–H6, 2 no-effect draws N1–N2, and 2 small-sample draws S1–S2 at $n = 1{,}500$).

**Outcomes.** Across the primary comparative run and a subsequent `mboostrs` analyst run (using updated causal diagnostics and an analyst-built doubly robust second stage):

- **Identification and null control:** The later `mboostrs` analyst chose a valid adjustment set on 8/8 replicates (matching `EconML`, up from 5/8 in the primary run) with 0.0 false claims on no-effect draws, though covariate role accuracy was 0.75 versus 0.94 for `EconML`.
- **Effect accuracy and coverage:** Average treatment effect (ATE) RMSE was \$2,747 (75% coverage for nominal 90% intervals) versus \$593 (88% coverage) for `statsmodels`. Mean per-applicant PEHE across H1–H6 was \$1,875 (versus \$1,703 for `EconML`), while nominal 90% per-applicant interval coverage was 61–66% (versus 89% for `statsmodels`) because the second-stage GAM omitted first-stage nuisance uncertainty.
- **Small-sample instability:** On S1 ($n = 1{,}500$, true ATE \$4,182), thin-overlap extrapolation produced ATE estimates of \$9,550–\$10,889 (PEHE \$7,607–\$8,862).

## Claim-cost discovery battery

**Design.** Seven libraries (`mboostrs`, `XGBoost`, `LightGBM`, `CatBoost`, InterpretML `EBM`, `pyGAM`, `statsmodels`) were given to seven autonomous analysts on real California census covariates with a simulated claim-cost outcome, then evaluated unchanged on 12 hidden draws (8 full-size H1–H8, 2 null N1–N2, 2 small-sample S1–S2 at $n = 800$–$1{,}500$).

**Outcomes.** After resolving string-factor refit/conformal errors, empty-selection handling, and REML stall issues present in the initial baseline run, `mboostrs` completed 12/12 hidden replicates (up from 9/12), reached 1.00 mean-effect TPR with 0.00 FDR and 1.00 spread-driver TPR on H1–H8, achieved 0.90–0.91 nominal 90% prediction-interval coverage, and reduced small-sample prediction RMSE from 2.77 to 0.19 (with follow-up refit median dropping from 7.9 s to 2.6 s and 500k-row fit time from 252 s to 74 s). Trade-offs remained: interaction TPR on H1–H8 dropped from 0.81 to 0.38 (FDR 0.25) when the analyst switched from tensor terms to `screen_interactions` with a 2-SE product-term gate, small-sample mean-effect TPR was 0.33, and narrow 0.4°-wide spatial bumps were undercovered by every library ($\le 1/8$ replicates).

## Beijing PM2.5 forecast competition

**Design.** Seven methods (`mboostrs`, `CatBoost`, `LightGBM`, `XGBoost`, `pyGAM`, `statsmodels` quantile regression, and a baseline) forecasted hourly Beijing PM2.5 on a time-shifted future-year split (2014 test year, trained on earlier years).

**Outcomes.** Transforming the right-skewed response with `log1p` reduced mean pinball loss across three quantile levels from 21.2 to 14.0 while preserving exact quantile equivariance. The analyst's initial additive `mboostrs` formula reached 2014 $\log(1 + \text{PM2.5})$ RMSE 0.6615 versus 0.6181 for CatBoost and 0.6213 for LightGBM. Adding lagged weather covariates, tensor interactions, and wind-direction varying coefficients (`by=`) selected on the validation year closed 58% of the gap (2014 RMSE 0.636; validation median pinball loss 21.21 vs. 21.27 for CatBoost). Nominal 80% prediction bands covered 0.74 of 2014 hours overall and 0.68 in winter across every library due to temporal distribution shift.

**Audit artifacts.** Protocols, written analyst briefs, grading rubrics, departure logs, and all individual analyst reports are archived verbatim in `docs/benchmarks/agentic_hostile_workflows/`, `docs/book/benchmarks/causal_discovery/`, `docs/book/benchmarks/claims_discovery/`, and `docs/book/benchmarks/pm25_forecast/`.
