# XGBoost PM2.5 quantile forecast, Beijing 2014

## Final model

- Booster: `xgboost` 3.4.0, `objective="reg:quantileerror"`, `quantile_alpha=[0.1, 0.5, 0.9]` (one booster, one tree sequence per quantile).
- Selected settings: `max_depth=4`, `learning_rate=0.1`, `min_child_weight=10`, 775 rounds, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0`, `tree_method="hist"`, `seed=20140101`.
- Features (covariates only): DEWP, TEMP, PRES, Iws, Is, Ir, `cbwd` (native categorical), month, hour, day of year, day of week; lags 1, 3, 24 and trailing means over 6 and 24 rows of TEMP, PRES, Iws, DEWP. Lags and means are taken in row order within each file, so across a gap left by a dropped target row they span the gap.
- Predictions: `q = expm1(booster output)`, clipped at 0. Each row is sorted so q10 <= q50 <= q90. No row needed sorting (`crossings fixed: 0`).

## Target treatment

The target is `log1p(pm2.5)`. The upper tail is heavy (p99 about 400 against a median of about 70), and squared-error or quantile splits on the raw scale are dominated by a few winter peaks. A quantile of a monotone transform equals the transform of the quantile, so `expm1` of the log-scale quantile is the quantile of PM2.5 itself. The objective is therefore still the pinball loss at the requested levels, and it is applied to a scale on which the trees split more evenly.

## Tuning protocol

- Split: fit on 2010–2012, validate on 2013 (time-ordered).
- Grid: `max_depth` in {4, 6, 8} x `learning_rate` in {0.03, 0.1} x `min_child_weight` in {1, 10}. 12 configurations, well under the limit of 40. Tuning time 136 s of the 30-minute budget.
- Each configuration is trained up to 1000 rounds. The round count is chosen from checkpoints every 25 rounds, using `predict(iteration_range=...)`, and scored by mean pinball loss on the original scale (µg/m³) across the three levels.
- Full log: `tuning_log.csv`. Best: config 4, validation mean pinball 14.345 (q10 7.63, q50 21.79, q90 13.62).
- Refit on 2010–2013 with the same 775 rounds.

## Fit time

Final refit: 8.58 s wall-clock (`refit seconds: 8.58`, measured around the refit and prediction call). The full `run.py` took 150.7 s including tuning.

## Friction log

- No exceptions or error messages were raised.
- The validation loss is not the training loss: `reg:quantileerror` optimises the log-scale pinball loss, and the original-scale pinball loss used for selection is not available as a built-in metric. Round-count selection needed a hand-written checkpoint loop over `iteration_range`; I did not use the library's early-stopping option, so the selection is outside the library.
- The 1000-round cap was reached by configs 1 and 2 (depth 4, learning rate 0.03), so their scores are for a truncated fit. This was not a limit of the library, but the cap was set by me and not checked against the optimum.
- Each quantile in `quantile_alpha` is fit by its own trees, so quantile curves can cross. The library does not enforce ordering; sorting is my post-processing. No crossings occurred on the final model.
- Categorical features need `enable_categorical=True` on each `DMatrix` and a pandas `category` column. I set both from the start, so I did not observe the failure without them.
