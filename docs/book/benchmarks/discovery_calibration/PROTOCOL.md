# Discovery calibration-boundary study

Instrument `discovery-calibration/1`, issue #631. Commit this protocol before
implementation or measurement. Baseline is spread-nomination MR #632 head
105b1b5; the new comparison changes the calibration boundary, not nomination,
adoption thresholds, formula search, stopping grids or refit rules.

## Procedure

When discover is asked to predict with intervals, reserve the first held-out
fold of the existing Rust two-fold random assignment, seed fixed to the call.
Only the other fold reaches family inference, hygiene, column typing,
transform selection, mean/interaction/spread screening, tuning and REML refit.
Use that selected/fitted model directly on the reserved calibration rows;
there is no further split or response-dependent refit. Default fitting without
newdata retains all supplied analysis rows, since it requests no prediction
band. An explicit calibration frame lets the caller supply its own independent
holdout. Preserve row indices and input namespaces. Existing selection minimum
30 complete rows and calibration minimum at the requested level still apply;
reject an inadequate split clearly rather than shrink a minimum silently.

The final model, reported mean and sigma, effect summaries and interval band
all refer to the selection-only fit when intervals are requested. No all-data
model is substituted. Budget decisions use only selection rows and preexisting
rules. Calibration can widen the band's endpoints but cannot change family,
formula, transforms, coefficients, stopping or refit. Support fixed LSS central
bands through existing Rust quantile/score/widening routines and the public
external-calibration interface. No new Python numerical estimator is introduced.

## Acceptance and executable evidence

Trace actual row identities entering discovery, its fits and calibration.
Changing only calibration responses must leave every selection decision,
model prediction, coefficient and stopping value unchanged while changing
calibration scores/bands. Exercise internal/external splits, standard and
GaussianLSS fits, learned imputation/log transforms, missing data, candidate
constraints, budgets, absent predictors/response, minimum sizes and invalid
arguments. Preserve default all-data analysis when no band is requested.
Structured caller partitions must remain explicit; do not silently redraw
external calibration. Marginal exchangeable-row guarantees do not establish
conditional, grouped, temporal or post-selection effect coverage.

## Frozen confirmation battery

Each panel has 50 independent development/test draws, seeds 6311000 through
6311049; test seed is development seed + 50000. Test size is 1000. Different
panels are never pooled for uncertainty. All random draws and methods are fixed;
no test score alters a setting or replaces a draw.

- `absolute`: n=1000, independent x1/x2 uniform[-1,1], mean
  sin(2*pi*x1)+0.5*x2, Gaussian scale 0.3+1.5*abs(x1).
- `constant`: same n/predictors/mean, Gaussian scale 1.
- `null`: n=1000, same predictors, zero mean, Gaussian scale 1.
- `sharp`: n=800, five independent uniform[-1,1] predictors, mean
  3*sign(x1)*sign(x2)+x3, Gaussian scale 0.5. Other two columns are noise.
- `small`: n=80, same predictors/mean as constant, Gaussian scale 1.
- `correlated`: n=1000, x1 and z independent standard normals,
  x2=0.9*x1+sqrt(0.19)*z, mean 0.8*x1+0.4*x2, Gaussian scale 0.5.
- `poisson`: n=1000, independent x1/x2 uniform[-1,1], response Poisson
  with mean exp(0.3+0.6*x1-0.4*x2).
- `heavy_tail`: n=1000, constant-panel mean/predictors, unit-variance t3 noise.

## Methods and information tracks

Discovery receives only data and response, default search settings, call seed
and level 0.90. Compare baseline's existing all-data selection/internal split
with the new boundary on exactly matched draws. The old workflow's calibration
reuse is explicit; empirical coverage cannot establish its conditional guarantee.

Known-specification fits use the same selection/calibration partition as the
new discovery. Supply Gaussian mean bbs(x1,df=4)+bols(x2), sigma bbs(x1,df=4)
only for absolute; null uses an intercept; sharp uses btree(x1,x2,maxdepth=3,
selection=sse)+bols(x3); correlated uses bols(x1)+bols(x2); Poisson uses that
linear formula on its log link. Family is supplied. Fixed mstop=100, nu=0.1.
Calibrate the fitted model directly using the reserved rows. This is a fixed
adequate-specification reference, not an oracle-tuned performance bound.

Five data-only controls use the same Rust partition, including the small panel:
CatBoost split/CQR (500 trees, depth6, learning_rate0.05), LightGBM split/CQR
(300 trees, 15 leaves, learning_rate0.05), sklearn gradient-boosting CQR
(300 trees, depth3, learning_rate0.05). Call seed fixes their random state.
All settings are untuned. Learn fit/cell models only on selection rows; calibrate
only on the reserved rows. Record row counts, identities, warnings and failures.
Pin every method to one thread and disable CatBoost file output.

The baseline discovery's later NumPy split differs from the new Rust split;
this is a whole-procedure comparison, not a causal estimate of splitting alone.
The new known/discovery/control arms share the actual row partition. A subset
unit test compares internal versus explicitly supplied copies of that partition.
No signal/scale form is supplied to discovery or competitors.

## Metrics and reporting

For every panel, method and draw retain coverage, mean width, scaled two-quantile
interval score, mean prediction MSE, coverage in low/high true-noise-scale
thirds, empty-set count, elapsed workflow time, warnings, input/prediction/source
hashes, chosen family/formulas/stopping and available row provenance. The true
conditional variance is evaluator-only (Poisson scale is sqrt(mean)). Constant
scale puts all rows in both thirds. Empty CQR sets have width zero and no
coverage; interval score uses the signed endpoint difference plus 20 times each
miss distance, as in the previous scoring addendum.

Report means and Monte Carlo SE across the 50 independent draws, and paired
new-minus-baseline coverage/width/score/error differences. Use proper-training
row counts beside every information track. Coverage is empirical evidence on
these panels; construction and executable exclusion establish the calibration
boundary. A valid finite-sample guarantee still requires exchangeability and
a fixed prediction rule independent of calibration responses.

For width leadership require mean coverage within 0.02 of nominal, except the
small panel permits an additional 1/(n_calibration+1) of upper conservatism from
finite-sample quantization, equally for every method. The primary gap is paired
percentage width difference to the retrospective lowest-mean-width eligible
COMPETITOR, chosen using test results. If none is eligible report no width
leader, not a relaxed threshold. Report both known and discovery separately,
including outcomes outside eligibility. A sub-1% point estimate is not a
population bound; SE does not adjust for retrospective winner selection.

File every remaining gap above 1%, using existing issues when they already
track it. Include regressions and failed methods. The protocol/rule is not
retuned after inspecting these tests. Preserve failed instruments and each
version if measurement requires correction. This study measures D9 within its
scope; effect inference and the other scorecard dimensions remain unmeasured.
