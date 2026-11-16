# Prediction-interval calibration study

Measures scorecard dimension D9 (uncertainty calibration) for prediction
intervals. Scorecard `mboostrs-scorecard/2.1`, content SHA256
`ccfd33a0b7f9c2cbadc910912551273e66046aaf82c36440d2c286df0cc43fad`.

## Study objective

This study measures empirical coverage of nominal 90% prediction intervals and
compares their widths with the narrowest intervals available when the true
conditional distribution is known.

## Data generating processes

Each draw has `y = sin(2*pi*x1) + 0.5*x2 + scale(x1) * eps` with `x1`, `x2`
uniform on [-1, 1]. Only `(y, x1, x2)` reach the methods.

| DGP | `eps` | `scale(x1)` |
|---|---|---|
| `homoscedastic` | N(0, 1) | 1 |
| `heteroscedastic` | N(0, 1) | 0.3 + 1.5 abs(x1) |
| `heavy_tailed_t3` | t with 3 df, scaled to unit variance | 1 |

Training rows: 1000 per draw. Test rows: 2000 per draw, from a separate seed.
Seeds: 20 per DGP (0 to 19).

## Tracks

- **Known specification.** mboostrs is given the true mean structure,
  `y ~ bbs(x1, df=4) + bols(x2)`, and the Gaussian family. The three conformal
  methods (`mboostrs_split`, `mboostrs_cqr`, `mboostrs_cv+`) use this model.
- **Discovery.** mboostrs receives only `(y, x1, x2)` and calls `mb.discover`,
  which chooses the formula, family and spread model itself. Its interval is the
  band returned in `predictions`. This track uses all training rows, while the
  conformal methods split them, so the two tracks differ in data use as well as
  in specification.

Competitors receive the same `(y, x1, x2)` with no structure in both tracks.

## Methods

- `catboost_split`: CatBoost mean regression, split conformal on a 50/50 split.
- `catboost_cqr`: CatBoost quantile regression at 0.05 and 0.95, CQR correction.
- `sklearn_gbr_cqr`: scikit-learn gradient boosting with quantile loss, CQR.
- `lightgbm_cqr`: LightGBM quantile regression, CQR. Fixed settings, untuned.
- `lightgbm_split`: LightGBM mean regression, split conformal. Fixed settings, untuned.
- `oracle`: the exact `mean +/- z * scale` interval for the DGP, using the true
  conditional distribution. It is not a method; it bounds the attainable width.

Competitor settings are fixed, not tuned per process. A stronger, tuned
competitor would lower the best-competitor width and raise the mboostrs gap.

## Best competitor and gap (`gap_to_best.py`)

For each process the best competitor is the one with the smallest mean width
among competitors whose mean coverage is within 0.02 of 0.90. That choice uses
test-set widths, so it is a retrospective best, not a prespecified baseline.
The gap is the paired per-seed width ratio minus one, in percent, with its
standard error across seeds. Negative means mboostrs is narrower. The gap to
the oracle is reported as a secondary reference.

## Metrics

- Coverage: share of test responses inside the interval. Reported as the mean
  over seeds, with its gap to 0.90 in percentage points and the standard error
  of that gap across seeds.
- Tercile coverage: coverage among test rows whose true scale is in the lowest
  and highest thirds. This shows whether the interval adapts to variance.
- Mean width, and width relative to the oracle width (percent).
- Spearman correlation between interval width and true scale on test rows.
  Undefined (reported as absent) when the width is constant.

## Reading the results

- D9 score 3 needs coverage near nominal on standard DGPs. "Near" here means
  a gap within about two standard errors of zero, and the per-DGP tables
  decide it.
- D9 score 4 needs adaptive widening under heteroscedasticity, demonstrated.
  Small-sample and extrapolation adaptivity are not part of this study.
- Width relative to the oracle is an efficiency measure, not a calibration
  measure. An interval can be well calibrated and still wider than necessary.

## Run history and limits

The reported run used the final `run.py` in this folder. Changes between
development runs were limited to how a constant-width Spearman correlation is
computed and how SciPy's constant-input warning is suppressed. No DGP, method,
seed or metric changed, and the rerun reproduced `summary.json` byte for byte.

The library was loaded from the editable install of the main checkout on branch
`fix/cqr-step-search-warnings` (commit `4cd69e2`). That commit only changes
warnings raised inside CQR, so the interval values match `origin/main`.

Each run takes about 100 seconds on one machine.
The three DGPs do not cover small n, extrapolation, discrete responses or
distribution shift, so none of those is scored here.

Command:

```bash
.venv/bin/python docs/book/benchmarks/interval_calibration/run.py \
    docs/book/benchmarks/interval_calibration/results.json
.venv/bin/python docs/book/benchmarks/interval_calibration/summarize.py
.venv/bin/python docs/book/benchmarks/interval_calibration/gap_to_best.py
```
