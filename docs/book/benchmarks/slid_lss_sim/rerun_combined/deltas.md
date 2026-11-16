# Combined-branch rerun: scorecard 2.1 deltas

Baseline build: `/workspace/agent-test/bench4/site-base/mboostrs/__init__.py` (declared `0ae6f81d`, 260 rows).
Combined build: `/workspace/agent-test/bench4/site-combined/mboostrs/__init__.py` at merge commit `e79fa6140ee013e1f024b4fc296b0deabc1712ca` (260 rows; the 220 competitor rows are reused unchanged from the baseline).

Scored exactly as the baseline (scorecard 2.1 anchors; the five dimensions the study measures keep their baseline integer anchor).

## Headline: CRPS gap to the per-seed best competitor

Negative is better (below the best competitor). Gap in percent; `->` is baseline -> combined; parentheses the change; SE is baseline, rerun SEs are in `deltas.json`.

| Track | CRPS gap % | RMSE gap % |
|---|---|---|
| sim_known | -4.555 -> -4.555 (0.000) [baseline SE 0.158] | -1.019 -> -1.019 (0.000) [baseline SE 0.142] |
| sim_discovery | -4.442 -> -4.442 (0.000) [baseline SE 0.157] | -0.991 -> -0.991 (0.000) [baseline SE 0.137] |
| slid_known | 0.579 -> 0.579 (0.000) [baseline SE 0.128] | 0.752 -> 0.752 (0.000) [baseline SE 0.132] |
| slid_discovery | 34.067 -> 1.148 (-32.919) [baseline SE 0.467] | 28.761 -> 0.848 (-27.913) [baseline SE 0.422] |

## Per-dimension

### D1a. Relationship recovery under a prescribed specification (measured, anchor 3 -> 3)

`rmse_mu` 0.056 -> 0.056 (0.000) [baseline SE 0.003]; `rmse_log_sigma` 0.014 -> 0.014 (0.000) [baseline SE 0.002]; classical comparator `rmse_mu` 0.123 -> 0.123 (0.000) [baseline SE 0.002].

known-spec path untouched by the fixes; identical recovery

### D1b. Relationship discovery from data alone (measured, anchor 3 -> 3)

sim (planted truth): `loc_precision` 0.980 -> 0.980 (0.000) [baseline SE 0.020], `loc_recall` 1.000 -> 1.000 (0.000) [baseline SE 0.000], `scale_precision` 1.000 -> 1.000 (0.000) [baseline SE 0.000], `scale_recall` 1.000 -> 1.000 (0.000) [baseline SE 0.000]; Gaussian seeds 10 -> 10; false discoveries 0, x5-in-loc 0, active-in-scale 0.

The real SLID arm previously returned the intercept-only model (`formula: 1`) on all ten seeds; the combined discovery now returns Gaussian with a non-empty formula on all ten (see `slid_discovery_selections.json`).

sim selection unchanged; the real-arm collapse that previously capped this dimension is gone

### D2. Causal humility and support (unmeasured, anchor None -> None)

Not exercised by this benchmark (`unmeasured`, listed unchanged as the baseline did).

### D3. Explanation quality (unmeasured, anchor None -> None)

Not exercised by this benchmark (`unmeasured`, listed unchanged as the baseline did).

### D4. Importance quantification (unmeasured, anchor None -> None)

Not exercised by this benchmark (`unmeasured`, listed unchanged as the baseline did).

### D5. Experimental iteration speed (measured, anchor 3 -> 3)

| Track | fit_median | fit_p90 | whatif_median |
|---|---|---|---|
| sim_known | 0.279 -> 0.264 (-0.015) | 0.308 -> 0.304 (-0.005) | 0.246 -> 0.228 (-0.018) |
| sim_discovery | 3.341 -> 3.225 (-0.116) | 3.975 -> 3.597 (-0.378) | 2.494 -> 2.364 (-0.130) |
| slid_known | 0.119 -> 0.140 (0.021) | 0.149 -> 0.174 (0.025) | 0.100 -> 0.107 (0.007) |
| slid_discovery | 1.840 -> 1.137 (-0.703) | 1.959 -> 1.206 (-0.753) | 1.106 -> 0.666 (-0.440) |

unchanged; SLID discovery fit and what-if both faster under the Gaussian family

### D6. Robustness across conditions (unmeasured, anchor None -> None)

Not exercised by this benchmark (`unmeasured`, listed unchanged as the baseline did).

### D7. Human-friendliness (unmeasured, anchor None -> None)

Not exercised by this benchmark (`unmeasured`, listed unchanged as the baseline did).

### D8. Agent-friendliness (unmeasured, anchor None -> None)

Not exercised by this benchmark (`unmeasured`, listed unchanged as the baseline did).

### D9. Uncertainty honesty (calibration) (measured, anchor 3 -> 3)

| Track | coverage80_mean | coverage80_gap_pp | coverage80_gap_se_pp | coverage80_width | coverage95_mean | coverage95_gap_pp | coverage95_gap_se_pp | coverage95_width |
|---|---|---|---|---|---|---|---|---|
| sim_known | 0.798 -> 0.798 (0.000) | -0.207 -> -0.207 (0.000) | 0.270 -> 0.270 (0.000) | 2.763 -> 2.763 (0.000) | 0.948 -> 0.948 (0.000) | -0.193 -> -0.193 (0.000) | 0.129 -> 0.129 (0.000) | 4.226 -> 4.226 (0.000) |
| sim_discovery | 0.816 -> 0.816 (0.000) | 1.597 -> 1.597 (0.000) | 0.302 -> 0.302 (0.000) | 2.859 -> 2.859 (0.000) | 0.956 -> 0.956 (0.000) | 0.597 -> 0.597 (0.000) | 0.197 -> 0.197 (0.000) | 4.373 -> 4.373 (0.000) |
| slid_known | 0.812 -> 0.812 (0.000) | 1.186 -> 1.186 (0.000) | 0.503 -> 0.503 (0.000) | 0.992 -> 0.992 (0.000) | 0.955 -> 0.955 (0.000) | 0.481 -> 0.481 (0.000) | 0.188 -> 0.188 (0.000) | 1.517 -> 1.517 (0.000) |
| slid_discovery | 0.716 -> 0.804 (0.088) | -8.406 -> 0.369 (8.774) | 0.572 -> 0.474 (-0.099) | 1.196 -> 0.988 (-0.208) | 0.958 -> 0.950 (-0.008) | 0.825 -> -0.012 (-0.837) | 0.155 -> 0.211 (0.056) | 1.830 -> 1.511 (-0.319) |

calibrated, unchanged; SLID discovery undercoverage repaired

### D10. Reproducibility and auditability (unmeasured, anchor None -> None)

Not exercised by this benchmark (`unmeasured`, listed unchanged as the baseline did).

### D11. Predictive validity (measured, anchor 3 -> 3)

| Track | crps_gap_pct | rmse_gap_pct |
|---|---|---|
| sim_known | -4.555 -> -4.555 (0.000) [baseline SE 0.158] | -1.019 -> -1.019 (0.000) [baseline SE 0.142] |
| sim_discovery | -4.442 -> -4.442 (0.000) [baseline SE 0.157] | -0.991 -> -0.991 (0.000) [baseline SE 0.137] |
| slid_known | 0.579 -> 0.579 (0.000) [baseline SE 0.128] | 0.752 -> 0.752 (0.000) [baseline SE 0.132] |
| slid_discovery | 34.067 -> 1.148 (-32.919) [baseline SE 0.467] | 28.761 -> 0.848 (-27.913) [baseline SE 0.422] |

unchanged; SLID discovery gap closed from +34% to +1.1%

## Fix attribution

| PR | Change | Dimensions |
|---|---|---|
| #585 empty-model guard / family inference | discovery always infers a family and never returns an empty model silently | D1b, D9, D11 |
| #586 intercept-only warning | discovery warns when it can only fit an intercept | D1b, D7 |
| #587 cvrisk folds accepted as int | cvrisk_gamboostLSS(folds=5) no longer raises IndexError | D5, D8 |
| #589 sigma accessor / spread_formula | discovery exposes the fitted scale model and a sigma accessor | D9, D11, D1b |
| #590 response-copy / row-id hygiene | discovery drops identifier columns and the response's own transforms before selecting predictors | D1b, D11 |

The one large move, SLID discovery, is attributable to the combined branch as a whole. #590 (row-id / response-copy hygiene) stops discovery selecting the response's own transform; #585 (family inference) makes the family probe run so it returns a proper model instead of an empty one; #589 exposes the scale model that supports the coverage repair. #586 and #587 change diagnostics and error handling and move no number in this run.

## Friction log

| # | Item | Status | Fix |
|---|---|---|---|
| 1 | no high-level mb.fit discovery front door | open | - |
| 2 | DiscoveryResult has no fitted sigma accessor | resolved | #589 |
| 3 | discover treats every non-response column as a predictor | resolved | #590 |
| 4 | discover silent on an intercept-only result | resolved | #586 |
| 5 | family probe off by default in discovery | resolved | #585 |
| 6 | cvrisk_gamboostLSS(folds=int) raises IndexError | resolved | #587 |
| 7 | with_mstop predictions disagree with a direct refit | open | - |
| 8 | build_info git_revision is checkout HEAD not build SHA | open | - |
| 9 | brandom(language) fold warning names an absent learner | open | - |
| 10 | discover interval level is the only uncertainty knob | open | - |

