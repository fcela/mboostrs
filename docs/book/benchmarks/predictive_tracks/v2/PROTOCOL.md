# Predictive validity in two tracks

Instrument `predictive-tracks/2`, frozen before this version's outcomes. This study follows
BENCHMARK_GUIDANCE.md and measures D11 of `mboostrs-scorecard/2.1` (SHA256
ccfd33a0b7f9c2cbadc910912551273e66046aaf82c36440d2c286df0cc43fad).
The target is at most 1% excess test MSE against the strongest complete named
competitor. The selected competitor is a retrospective best on test results,
not a prespecified baseline. Ridge is the prespecified simple baseline.

## Data and estimand

The target is the conditional mean of a continuous response for a new IID row.
For each process and seed 0–7, generate independent development and outer-test
rows with NumPy's generator seeded at 539400 + 10000*process_index + seed.
The development set has 800 rows: the first 600 train competitors, the next
200 choose their settings. Selected competitors refit on all 800. Both
mboostrs tracks also use all 800, with five internal folds. The outer test
has 2,000 independent rows. Predictors x1,x2,x3,n0,n1 are independent
Uniform(-1,1); the factor process adds an independent balanced two-level g.
The response is mu + Normal(0,1), independently by row. No response or truth
from the test set reaches a fit, tuning procedure or discovery call.

| Process | Conditional mean mu | Known specification |
| --- | --- | --- |
| smooth | 2 sin(pi*x1) + 0.5*x2 | y ~ bbs(x1, df=4) + bols(x2) |
| interaction | 2 sin(pi*x1*x2) + 0.5*x3 | y ~ bbs(x1, df=4, knots=8) %X% bbs(x2, df=4, knots=8) + bols(x3) |
| factor | 2 sin(pi*x1)*(1 if g=a else -1) + 0.5*x2 | y ~ bbs(x1, df=4) + bbs(x1, df=4, by=g) + bols(x2) |

The factor dictionary contains a baseline smooth and a treatment-coded
difference smooth, each df=4. Together they represent both level curves;
`by=g` alone is zero on the reference level.
The true conditional mean is a secondary oracle reference only. It is not a
competing method and never selects a configuration.

## Tracks and methods

Known specification receives the stated formula and Gaussian family. Tune
its positive stopping iteration through the public tune_mstop API, start100,
max_mstop1600, nu0.1, five folds, score=mse (fold-local prediction); predict
with the returned public model. Discovery calls discover(data,"y",seed=seed,
folds=5,max_mstop=1600), with no formula, family, candidate restriction,
mandatory term or generating truth. All optional discovery stages retain
their defaults; no newdata is passed while selecting the model. Predict
through result.model, using its mu parameter if distributional. Predictions
use the documented clip extrapolation policy in both tracks. Record every
warning and the full discovery decision log, inferred family/formulas and
selected stopping point. These tracks differ in prior structural information.
No discovery/recovery score is inferred from predictive loss.

Competitors receive every predictor and the response, with no known structure,
in both tracks. Their identical fitted prediction is reused across track
reports, explicitly labelled. Fit one model per configuration on rows0:600,
select minimum validation MSE on rows600:800, then refit the selected setting
on all800. Seed every stochastic estimator and use one thread.

- CatBoost RMSE: depth3/6 × iterations400/1000, learning_rate0.05, native factor.
- LightGBM regression: num_leaves7/15 × n_estimators400/1000,
  learning_rate0.05, min_child_samples20, native factor.
- XGBoost squared error: max_depth2/4 × n_estimators400/1000,
  learning_rate0.05, reg_lambda1, tree_method=hist, enable_categorical=True.
- sklearn HistGradientBoosting squared error: max_leaf_nodes7/15 ×
  max_iter400/1000, learning_rate0.05, min_samples_leaf20,
  early_stopping=False, native categorical feature from pandas dtype.
- Ridge: alpha0.1/1/10/100, training-fitted standard scaling for numeric
  predictors and one-hot factor encoding. This is the simple baseline.

All competitors are tuned over these four named configurations. Search designs
and fit counts differ from mboostrs's five-fold stopping and full discovery;
report those differences. No claim of matched compute or exhaustive optimal
competitor tuning is made. Each method/draw is a fresh serial process, one
thread, timeout240s. Record failures as planned outcomes; never omit a seed.
Timings are descriptive on shared hardware, not speed ratios.

## Analysis and scorecard

Report each method's test MSE mean and standard error over eight independent
draws, separately in each track. Among the four named tree competitors with
all eight successful finite outcomes on a process, choose the smallest mean
test MSE. Missing complete competitors are disclosed. For each mboostrs track,
compute eight paired percent gaps 100*(MSE_mboost/MSE_best-1), their mean and
sample-SD/sqrt(8). Negative favors mboostrs. Also report the paired raw loss
difference and its SE. A gap within two SE of zero is unresolved at this
Monte Carlo precision; a point gap >1% is still filed with that qualification.
Do not pool processes or choose a track by its test score.

D11 is assessed separately by track and process. With eight successful draws,
score3 when the mean paired MSE improvement over prespecified Ridge exceeds
two SE; score2 when improvement is positive but does not clear that threshold;
score1 when mean MSE fails to beat Ridge, or a method fails to complete all
planned draws. Score1 for a failed run is evidence of predictive-workflow
failure; no successful-only numerical gap substitutes for its missing mean.
Scores4/5 are unavailable under these unequal search designs and this focused
battery. The remaining eleven dimensions stay unmeasured. No aggregate is
computed. Freeze these anchor applications before outcomes.

## Reproduction and improvement loop

Retain phase-specific raw results, code/protocol/guide/scorecard hashes, loaded
Python/native identity, dependency versions, input/fold/prediction hashes,
settings and logs. Publish an executed notebook with pandas tables in Part V.
File an issue for every mboostrs process/track gap >1% and every failed route,
link related open issues, diagnose causes and preserve negative trials. Each
actual fix has its own branch/MR. Rerun this exact protocol/seeds/competitors
unchanged on fixed code, retaining the baseline and reporting paired change
with SE. Any protocol change starts a distinct version; never overwrite a
baseline. A default-policy change also needs separately frozen fresh controls
before their outcomes. Do not merge without explicit user authorization.

## Instrument correction

Version 1 omitted the factor reference-level smooth from the supplied known
specification. Its full raw results/protocol are retained outside this v2
directory, and its factor known-specification results are invalid for the
intended information contract. The omission was found from the public factor
construction and reproduced on a separate noiseless draw (seed618204): the
reference-level curve MSE was2.084 rather than below0.1. It is a coordinator
formula error, not a library estimation failure.

This version changes only that supplied factor formula. All processes, seeds,
row counts, competitor settings, discovery calls, losses, SEs, anchors and
selection rules remain fixed. Collect every method/draw again under its new
runner identity; no outcome is relabelled as independent confirmation. Raw
version-1 numbers must not be pooled with this valid baseline. The separate
discovery omission is issue #560; diagnose and fix it independently while
keeping this corrected benchmark unchanged. Freeze this complete version
before collecting its baseline or fix outcomes.
