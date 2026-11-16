# mboostrs: hourly PM2.5 quantiles for Beijing 2014

## Final model

- Library: `mboostrs.gamboost` with family `Quantile(tau)` (pinball loss), one fit per tau in {0.1, 0.5, 0.9}.
- Target: `log1p(pm2.5)`. Predictions are mapped back with `expm1` and clipped at 0. Quantiles are invariant under monotone maps, so each back-transformed prediction is the tau-quantile of pm2.5 under the model.
- Formula (identical for the three taus):
  `y ~ bbs(hour, df=4) + bbs(month, df=4) + bbs(TEMP, df=4) + bbs(PRES, df=4) + bbs(DEWP, df=4) + bbs(Iws, df=4) + bols(cbwd) + bbs(doy, df=4) + bols(Is) + bols(Ir) + bbs(TEMP_r24, df=4) + bbs(PRES_r24, df=4) + bbs(DEWP_r24, df=4) + bbs(Iws_r24, df=4)`
- Features: calendar `doy`, `dow`; trailing 24-hour means (`*_r24`, `min_samples=1`) of TEMP, PRES, DEWP, Iws computed in row order within each file.
- `boost_control(mstop=m_tau, nu=0.1)`. mstop per tau: q10 = 6000, q50 = 4800, q90 = 3200.
- Output: `q10`, `q50`, `q90` sorted per row, because separately fitted quantiles can cross.

## Tuning protocol

- Time-ordered split: fit on 2010-2012, validate on 2013.
- Path: mstop 50 to 6000 (step 100 after the first pass). For each tau, mstop is the argmin of the 2013 pinball loss on the pm2.5 scale.
- Score per configuration: mean of the three pinball losses.
- Configurations: 25 in total, all in `tuning_log.csv`. Stage 1 crossed transform (raw, log1p), rolling means (off, on) and bbs df (4, 6). Stage 2 varied one setting from the stage-1 best (btree, nu=0.05, df 8, df 3). Stage 3 extended the path of the best configuration.
- The first 12 configurations used a path capped at 1000. Every tau chose the cap, so the path was extended. Those rows stay in the log with `mstop_max=1000`.
- Best configuration: config 25 (log1p, rolling, df=4, path to 6000), mean pinball 13.92. Config 19 (same, path to 3000) scored 14.03. The q10 optimum is still at 6000 and is not a settled value.
- Summed fit time across all 25 configurations: about 265 s, within the 30-minute limit.

## Target distribution

pm2.5 has a right tail and a floor at 0. Fitting on `log1p` gave a mean pinball of 14.0 (config 19) against 21.2 to 21.5 on the raw scale at the same covariate and df settings (configs 15 and 16). The log scale reduces the effect of the largest values on the spline coefficients and on the offset. The quantile check loss is still evaluated on the raw scale, so the scores are comparable to the target.

## Refit

- All 2010-2013 rows, the mstop values above, same formula.
- `run.py` wall-clock for three fits and prediction: 52.8 s (script timer); 53.3 s with interpreter start-up.
- `run.py` was run with `PYTHONWARNINGS=ignore`. The warnings are listed below. Test-set extrapolation warnings were not checked.

## Friction log

1. `_workflow.py:191` UserWarning on the 2010-2012 fits: "boosting has undone much of the penalty of 2 learner(s): bbs(Iws_r24): 11.5 of 24 df after 568 selections ... Choose mstop by cross-validation (cv_risk) rather than fitting this long." At mstop 3000 to 6000 the `df=4` on each bbs term stops limiting flexibility, so the tuning path and the mstop choice are tied to this drift.
2. No built-in time-ordered validation path with a quantile score. The validation path loop (`predict_over_mstop`, pinball on the pm2.5 scale) is written by hand. `cvrisk` was not tried with custom folds.
3. `gamboost.py:1951` ExtrapolationWarning on 2013 validation: "bbs(DEWP): 34 below -28 (worst 0.089 training ranges out); bbs(PRES_r24): 17 below 994.375 and 25 above 1041.71". The default extrapolation behaviour is a warning, not a clip or error, so out-of-range covariates are silently continued by the spline's edge slope unless the caller changes the setting.
4. Separate tau fits do not guarantee non-crossing quantiles; `run.py` sorts each row. This is a post-processing step outside the library.
5. Defaults for quantile tuning are not documented in one place: the default `qoffset=0.5` does not match tau, and the default mstop is 100, far below the optimum found here (3200 to 6000 at nu=0.1).
