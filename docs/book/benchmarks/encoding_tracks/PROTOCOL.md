# Predictor encoding and predictive validity

Instrument `encoding-tracks/1`, frozen before outcomes for issue #572.
Follow BENCHMARK_GUIDANCE.md. Measure D11 of mboostrs-scorecard/2.1, content
SHA256 ccfd33a0b7f9c2cbadc910912551273e66046aaf82c36440d2c286df0cc43fad.
Other eleven dimensions are unmeasured; no aggregate. Existing predictive-tracks
and factor-reference protocols and artifacts remain immutable.

## Estimand and data

Predict a new IID continuous response's conditional mean. For each replicate
0–7, NumPy generator seed619000+replicate produces 2,800 rows: five independent
Uniform(-1,1) predictors x1,x2,x3,n0,n1; independent binary b; independent
Gaussian noise with standard deviation1 response unit. The mean is
`2*sin(pi*x1)*(1-2*b)+0.5*x2`. Rows0:800 are development and rows800:2800 are
outer test. The first600 development rows train competitors and the next200
choose settings; the chosen setting refits on all800. Both mboostrs tracks
use all800 with five internal folds.

The exact same latent rows, responses and noise are used for five encodings:

| Encoding | Observed g | Known-specification formula |
| --- | --- | --- |
| categorical | a if b=0, b otherwise | y ~ bbs(x1, df=4) + bbs(x1, df=4, by=g) + bols(x2) |
| indicator | b as integers0/1 | same baseline-plus-by formula |
| boolean | b==1 as booleans | same baseline-plus-by formula |
| signed | 1-2*b as integers+1/-1 | y ~ bbs(x1, df=4, by=g) + bols(x2) |
| shifted | 2+b as integers2/3 | same baseline-plus-by formula |

The known dictionaries span the generating functions. In shifted coding the
baseline coefficient function is5*f(x), the by function−2*f(x), where
f(x)=2*sin(pi*x). Their cancellation is a valid parameterization, but finite
coordinate boosting may converge more slowly; that is an estimation/budget
result, not a missing-term instrument error. Do not change formulas after
seeing outcomes. The exact mean is a secondary oracle reference, not a method.

Each encoding has eight independent draws, but encodings are paired views of
the same eight draws. Never pool them as40independent replicates.

## Information tracks and competitors

Known specification receives the corresponding formula and Gaussian family;
tune_mstop(start100,max_mstop1600,nu0.1,folds5,seed=replicate,score=mse) gives
the final public model. Discovery receives only data and response plus the
same resource settings through discover(data,"y",folds5,seed=replicate,
max_mstop1600), with default optional stages. No formula, family, truth or
encoding hint is supplied to discovery. No test rows reach selection.
Predict both through public model APIs, response scale, clip extrapolation;
a distributional discovery uses its mu parameter. Keep formulas, full decision
logs, warnings, tuning paths and all failures.

Competitors receive every predictor and the response with no structure in
both tracks; identical fitted predictions are reused across their track reports.
Use the existing frozen competitor function from bench_predictive_tracks.py:

- CatBoost RMSE: depth3/6 × iterations400/1000, learning_rate0.05;
  native string factor and numeric/boolean inputs as observed.
- LightGBM regression: leaves7/15 × estimators400/1000, rate0.05,
  min_child_samples20; native string factor.
- XGBoost squared error: depth2/4 × estimators400/1000, rate0.05,
  lambda1, hist, enable_categorical=True.
- sklearn HistGradientBoosting squared error: leaves7/15 × iterations400/1000,
  rate0.05, min_samples_leaf20, early_stopping=False.
- Ridge reference: alpha0.1/1/10/100, training-fitted numeric scaling and
  one-hot factor encoding. This is the prespecified simple baseline.

Each four-configuration choice minimizes MSE on the reserved200rows, then
refits on800. Competitors are tuned within these finite grids. Search designs
and fit counts differ from mboostrs; no matched-compute claim or D11level4/5.
All five encodings and all seven methods are run:280jobs per phase, serial,
one thread, timeout240seconds per method/draw. Keep explicit timeout/crash
records; incomplete cells have no successful-only mean. Shared-load timing
is descriptive, not an implementation speed ratio.

## Gaps and scores

Within each encoding, select the strongest tree competitor among the four
with all eight finite successful outcomes, by mean outer-test MSE. This is a
retrospective best, not a prespecified baseline. Report every competitor,
its mean MSE and SE, and missing complete peers. For each track report the
mean of eight paired percent gaps `100*(mboost_mse/best_mse-1)`, SE=sample
SD/sqrt(8), plus paired raw loss difference and SE. Negative favors mboostrs.
The SE treats the chosen winner as fixed, excluding winner-selection
uncertainty. A gap within twoSE of zero is unresolved at this precision; a
point gap above1% is still filed, with that qualification. Target≤1% per cell.
Do not choose a track/encoding policy by its test score or pool scores.

D11 rules are frozen as in predictive-tracks/2, separately per track/encoding:
score3 if mean paired MSE improvement over Ridge exceeds twoSE; score2 if
positive but not above twoSE; score1 if it fails to beat Ridge, or the mboostrs
cell fails to complete all eight draws. If the Ridge reference is incomplete,
its comparative grade remains unmeasured. No higher score under this focused
unequal-search design. No D1a/D1b credit for supplied or merely predictive terms,
and no robustness grade from this limited encoding check.

## Repair and provenance

Baseline imports the categorical-only repaired package from study746ec91
(library content f290194). Freeze source/package/native identities separately
from the runner. Capture protocol/guide/scorecard/code hashes, exact encoded
input and latent-row hashes, prediction hashes, candidate settings, warnings
and logs. A fixed phase reruns this entire unchanged protocol, same rows,
seeds and competitors, into a distinct directory. Verify identical peers and
known fits where the fix cannot affect them; report paired changes and SE.
Each actual fix has a separate branch/MR; do not merge without authorization.

A broader default-policy fix also needs separately frozen fresh controls for
categorical,0/1,boolean,signed and shifted inputs plus additive and null cases.
Existing #568 controls and the exploratory seed618211 are development evidence,
not fresh confirmation. Preserve all failures/negative proposals. Any analytical
protocol change starts another version. Publish an executed pandas-backed
notebook under Part V with retained outputs, explicit limits and all profiles.
