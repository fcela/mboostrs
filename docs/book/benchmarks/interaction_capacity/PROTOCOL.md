# Sharp interactions: learner form and stopping length

Instrument `interaction-capacity/1`, frozen before outcomes on main e314bbf.
This diagnoses the sharp-interaction mechanism behind the historical +249%
chooser gap in #493; it does not reproduce that historical percentage or
measure bike forecasting, cancer classification or causal estimation.
Follow BENCHMARK_GUIDANCE.md. Target: each measured gap at most +1%.

## Prediction and draws

Predict a new IID response mean. Eight independent draws per process, seeds
202610170 + 10000 * process_index + replicate, NumPy default_rng. Each draw
has 2800 rows of five independent Uniform(-1,1) predictors x1,x2,x3,n0,n1.
All responses add independent Normal(0,0.5) noise (response units):

| Process (index) | Conditional mean | Supplied known specification |
| --- | --- | --- |
| axis_xor (0) | 3*sign(x1)*sign(x2) + x3 | y ~ btree(x1,x2,maxdepth=3,selection=sse) + bols(x3) |
| shifted_xor (1) | 3*sign(x1-.25)*sign(x2+.15) + x3 | same as axis_xor |
| smooth_product (2) | 2*sin(pi*x1*x2) + .5*x3 | y ~ bbs(x1,df=4,knots=8) %X% bbs(x2,df=4,knots=8) + bols(x3) |
| additive (3) | 2*sin(pi*x1) + .5*x2 | y ~ bbs(x1,df=4) + bols(x2) |
| null (4) | 0 | y ~ 1 |

Rows 0:800 are development; rows 800:2800 are untouched outer tests. The
known tree contains the true two-variable step structure, with cut points
estimated from training data; no threshold values are supplied. Only the
known track receives variables/functional structure and Gaussian family.
The exact mean is a secondary reference, not a competing method.

## Tracks and competitors

Known: tune_mstop(formula, development, family=Gaussian(), folds=5,
start=100, max_mstop=1600, nu=.1, seed=replicate, score='mse').
Discovery: discover(development, 'y', folds=5, max_mstop=1600,
seed=replicate), all default stages. No formula, family, predictor typing or
truth is supplied. Predict the boosted model on response scale, mu for LSS,
with clip extrapolation. Record all formulas, warnings, stage failures and
complete decision logs. Discovery's optional-stage failures count separately
from whether its final predictions complete; refit refusal is never hidden.

Four validation-tuned tree competitors and a simple Ridge reference receive
only the five predictors and response. Reuse the frozen competitor function
in bench_predictive_tracks.py: CatBoost depth3/6 x iterations400/1000;
LightGBM leaves7/15 x estimators400/1000; XGBoost depth2/4 x estimators400/1000;
HistGradientBoosting leaves7/15 x iterations400/1000. All rates .05,
min leaf size20 where supported; no early stopping; fixed seeds; one thread.
Ridge alpha .1/1/10/100 with train-fitted scaling. Each chooses the minimum
MSE on development rows600:800 after fitting rows0:600, then refits on800.
Five-fold mboostrs searches differ in cost; no matched-compute claim.

Primary phase: 5 processes x 8 draws x 7 methods = 280 isolated jobs, serial,
one thread, 240-second timeout per job. Keep explicit failures and do not
compute successful-only cell means. Run all competitors again after a repair
and verify exact prediction replay. Never select a policy using outer losses.

## Mechanism diagnostic

On axis_xor and shifted_xor only, separately fit the identical smooth tensor
reference (df2.5, knots5 for each main and margin, plus bols(x3)) at validation-
selected ceilings1600 and6400, other settings identical. These 32 diagnostic
jobs ask whether longer paths overcome a fixed smooth basis's step bias.
This dictionary is deliberately not the supplied true step structure. They
are diagnostic counterfactuals, not extra candidates chosen by outer loss.
Preserve all curves, boundaries, settings and failure rows. Do not replace a
failed primary run with a successful diagnostic.

## Independent confirmation before a default change

A candidate learner-form enhancement must preserve constraints, family rules,
weights, caller budget admission and the incumbent if its comparison fails.
Freeze these six additional families before measuring a proposed policy:
linear, curved additive, smooth product, factor-varying curve, null and step
interaction, six draws each. Seeds202710170 + 10000*family_index + replicate,
2200 rows (600 development/1600 test), five independent U(-1,1) predictors.
Means: 1.5*x1+.5*x2; sin(2*x1)+x2^2; 2*sin(2*x1*x2)+.5*x3;
1.5*sin(2*x1)*(1-2*b)+.5*x2 (independent Bernoulli(.5), observed string g);
0; 2*sign(x1+.3)*sign(x2-.2)+.5*x3. Noise SD .6. Discover with folds3,
max_mstop800, default stages. Compare identical old/new rows before policy
promotion; require each family's mean after/before MSE ratio <=1.01 and no
increase in the number of null draws with an adopted interaction. Record
all individual regressions and failures. This finite gate does not establish
false-discovery control or broad robustness. Existing measured cases are
mechanism-development data, not independent confirmation.

## Reporting and scorecard

For each process, retrospective strongest complete tree competitor minimizes
mean outer MSE over all eight draws. Report every method, MSE in squared
response units, SE=sample SD/sqrt(8), complete/failure counts. For each track,
report the mean paired percent gap 100*(mboost_mse/competitor_mse-1), its SE,
and raw paired MSE difference and SE. Negative favors mboostrs. The SE treats
the selected competitor as fixed, excluding winner-selection uncertainty.
A point gap >1% gets an issue even when within two SE of zero. Keep processes,
tracks and confirmation separate. Report paired before/after changes and
exact unaffected prediction replay; no global default from a winning cell.

Measure D11 only, using scorecard2.1. Per process/track score3 if the paired
Ridge-minus-mboost mean MSE exceeds two SE; score2 if positive but less than
two SE; score1 if no improvement or incomplete mboost cell. An incomplete
Ridge leaves comparative assessment unmeasured. No scores4/5 with these
unequal searches and related synthetic processes; eleven other dimensions
stay unmeasured, no aggregate. Null controls may legitimately score1 under
this simple-reference rule. No claim of structure recovery, inference,
causal identification or interval coverage follows from predictive MSE.

Freeze code/protocol/guide/scorecard hashes, source and native identities,
data/prediction digests, settings, runtime, logs and failures. Build a release
extension matching current source; keep all phases immutable. Each defect
gets an issue and each fix a separate MR; do not merge. Publish an executed
pandas-backed notebook with annotated GT tables and figures, registered in
Part V. Any protocol change starts a new version; retain failed instruments.
