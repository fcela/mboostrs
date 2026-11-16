# Binomial prediction-scale audit

Instrument binomial-prediction-scale/1. Diagnose issue607 using the original
D5 round3 cancer split seeds0,1,2 and fixed settings. This is a correction of
an existing diagnostic, not a new best-in-class predictive benchmark or a
default-estimator change. It does not measure both information tracks and
cannot receive a predictive scorecard grade or establish a1% competitor gap.

Use sklearn.datasets.load_breast_cancer. Split30% test with random_state=seed,
no stratification, exactly as the existing notebook. Fit the same raw-column
formula y~sum(bbs(xi)) for30predictors, Binomial family,nu.1,mstop200 and500.
Record the log loss from clipping the default link predictions (the original
incorrect calculation) beside loss from explicit type='response' probabilities,
clipping both to[1e-6,1-1e-6]. Record Brier score and AUC on response scale,
input/prediction hashes and all warnings. Compare direct response predictions
with sigmoid(2*link) as an independent scale check. No test value changes any
fit setting. Preserve the original notebook outputs as historical evidence.

Repeat the original fixed competitors: CatBoost iterations200,depth4,rate.05,
random_seed=seed; LightGBM estimators200,max_depth4,rate.05,random_state=seed;
sklearn logistic regression max_iter1000,C1. All use the same raw Xtrain.
Pin threads1 to bound cost; no timing claims or tuning/compute fairness claim.
These are the diagnostic's recorded controls, not a comprehensive competition.

For the original seed0 linear coefficient probe, retain bols(xi),mstop500,
nu.1 and LogisticRegression(C=1e6,max_iter2000). Explicitly record convergence
warnings and both the booster's half-logit coefficients and their doubled
log-odds values. Correlation with a differently regularized finite-path fit
is descriptive, not a proof about a gradient or an MLE on collinear/separable
data. Use a separate well-conditioned nonseparable synthetic linear logistic
example to test the long path against a converged logistic reference, and
R-mboost numerical parity on a fixed public model path before concluding a
family-update defect. No gradient/link/default is changed from this audit.

After measurements, correct the notebook calls and interpretation, preserve
all historical results with an invalid-for-probability-scoring label, and link
complete provenance and results in607. The distinct autospec gap in541 remains
open unless its own protocol is rerun; this scale audit cannot close it.
