# CV+ and CQR predictor preprocessing

This issue #648 diagnostic is frozen before implementation or measurement
on baseline source 4d1f883. It checks numerical prediction coordinates and
fold exclusion. It does not estimate coverage or rank predictive competitors.
The preceding Poisson competitor study and its frozen results are unchanged.

## Reference construction

Fit each specification directly through the public glmboost/gamboost API on
its raw proper-training rows. Predict raw held-out/query rows through that
fit's own recipe. Do not call conformal_interval or its private helpers for
reference predictions or conformal arithmetic.

For CQR, independently fit Quantile(.05) and Quantile(.95), order their
predictions, compute max(lower-y,y-upper), take order statistic
ceil((n_cal+1)*.9), widen and replace an inverted real band by its midpoint,
as the existing core contract specifies. For CV+, use the same seeded NumPy
permutation and balanced four-fold assignment, train on the sorted complement
of each held fold, and combine each held residual with that fold model's
query prediction. Select the existing lower/upper CV+ order-statistic ranks.
Keep the original full-model point prediction, balanced guarantee and
equal-fold remainder rule. Reference arithmetic uses NumPy only in the
diagnostic/tests; production numerics continue to use the existing Rust core.

## Fixed cases

Use NumPy seed636 and n240 independent uniform[-1,1] latent x. Gaussian
responses are .3+.6*x+.2*N(0,1); Poisson responses have mean exp(.3+.6*x).
Four predictor recipes are bare x, scale(x), log(x) with observed x=exp(latent
x), and scale(x) with every thirteenth observed x missing and na_action=impute.
The response is generated before predictor conversion/missingness. Query at
latent x[-.6,0,.6], exponentiated for log; the missing case has the middle
query predictor missing. Add an unused rowid column for boundary tracing.

Cross two frontends, two families, two methods, four predictor recipes and
two query modes (the three query rows or all retained training rows), for64
fixed cases per implementation. Full models have60 iterations/rate.1; request
prefix30, level.9, split/fold seed4 and four CV+ folds. CQR reserves the first
120 permuted rows; CV+ folds hold60 each. Retain actual positional identities,
data/source/native hashes, endpoint/point arrays, correction and guarantees.

## Tuned cases

Add eight Gaussian cases: two frontends, two methods and scale/missing
predictor recipes, query rows only, mstop=tune. Use the existing five-fold
Rust-generated inner weights on each proper-training set and fixed seed4.
The reference explicitly requests preprocessing=refit in inner CV. Preserve
the existing1600-step starting floor, doubling rule, cap10000, boundary
criterion and positive optimum rule. Retain selected stops and inner-fold
preprocessing metadata; no observed outcome changes these settings.

## Acceptance and limits

All72 corrected cases must match independent endpoints and points within
1e-10, including both quantile fits and all CV+ models. Retain every baseline
failure/discrepancy. Unit/integration tests additionally cover unequal folds,
forced query chunking, requested budgets beyond the original path, weight
alignment, fitted imputation/log floors, missing raw inputs or row alignment,
and calibration-predictor/response mutation. Trace actual raw row identities
entering outer and inner fits and each recipe used for prediction. A held-out
row must not enter the preprocessing of the model predicting it. In CQR,
calibration changes cannot change the proper-training prediction functions
or chosen stops; the final correction may change.

Bare-predictor numerical oracle tests and existing CV+ guarantee/remainder
tests must continue to pass. Existing original-response/transform contracts
are not broadened by this predictor-coordinate repair. Unsupported paths
must refuse clearly rather than return predictions in a different coding.
No new Rust numeric estimator or language-specific numerical implementation
is required. Report deterministic agreement, not a coverage improvement or
a new universal prediction-performance claim. Publish a diagnostic notebook,
complete retained evidence and a separate MR for this tracked fix.
