# Experimental iteration speed (scorecard D5)

Measures D5: how long one calibrated analysis takes. The deliverable is a 90%
prediction interval, end to end, from raw data (fitting, calibration,
prediction). Every method delivers the same thing on the same data.

Scorecard `mboostrs-scorecard/2.1`, content SHA256
`ccfd33a0b7f9c2cbadc910912551273e66046aaf82c36440d2c286df0cc43fad`.

## Data

Training rows: n = 1,000 and n = 10,000. Test rows: 2,000. Structure: the
heteroscedastic process from the D9 study (`sin(2 pi x1) + 0.5 x2`, noise scale
`0.3 + 1.5|x1|`).

## Methods

- `mboostrs_cqr`: `gamboost` with the true structure, `conformal_interval(method="cqr")`.
- `mboostrs_discover`: `mb.discover` with data and response name only.
- `catboost_cqr`: CatBoost quantile regression at 0.05 and 0.95, CQR correction.
- `lightgbm_cqr`: LightGBM quantile regression, CQR correction.

## Measurement

Median wall-clock time of 3 runs per (method, n), covering the call only, run
sequentially on one machine with default thread settings. The best competitor
at each n is the faster of CatBoost CQR and LightGBM CQR. The gap is the
method's median time divided by the best competitor's, minus one.

## Limits

- Three repeats per cell.
- Default threads. LightGBM has a fixed overhead of about 0.9 s at n = 1,000
  with an OpenMP warning, so its n = 1,000 column is not a fair speed measure.
- One machine, one process. No cold-start or memory measurement.
- The mboostrs CQR time includes the CV-tuned step search of the quantile fits.

## Command

    .venv/bin/python docs/book/benchmarks/iteration_speed_d5/run.py \
        docs/book/benchmarks/iteration_speed_d5/results.json

## Profiling (added after the timing run)

One CQR interval at n = 10,000 (heteroscedastic, seed 0) was profiled with
`cProfile`: 3.16 s in total, 7 refits of the quantile models and 7 five-fold
cross-validations, as the tuning grid doubles. Cross-validation (Rust,
`cvrisk_gamboost_specs`) took 1.87 s, and refits (Rust, `fit_gamboost_specs`)
1.23 s. Python overhead is negligible. The totals in `cqr_profile_results.json`
are copied from that single profiled run; they are not a repeated measurement.

Step-floor timing (`cqr_profile_results.json`, median of 3 seeds, n = 10,000):
floor 200 3.23 s, floor 1600 3.16 s.
