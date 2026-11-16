# Standard conformal response units

This issue #652 diagnostic is specified on baseline source 64a13b1 before
implementation or measurement. It measures numerical agreement with independently
constructed original-response intervals, rather than predictive coverage or
competitor superiority. Previous benchmark archives remain unchanged.

## Target and reference

All calibration observations, point functions, scores and endpoints use the
original observed response Y. Each proper-training fit inverse-transforms its
own response prediction using its own learned response parameters. This point
function need not be the original-response conditional mean. For CQR, invert
each fitted quantile candidate before ordering and calibration; the conformal
guarantee needs fixed candidate functions, not correct conditional quantiles.
Do not inverse-transform the final endpoints of a transformed-unit interval:
that construction uses a different score. Omit fitted_mean for transformed
models because a fitted transformed location need not identify an original
expectation.

References use public direct glmboost/gamboost fits on raw proper-training rows,
public response predictions, and independently written inverse formulas:
center: z + mean(train Y); scale: z * sd(train Y, ddof=1) + mean(train Y);
log: exp(z); log1p: expm1(z); sqrt: z**2; exp: log(z). No reference calls
conformal_interval or a private conformal/response inverse helper. Reference
order statistics use NumPy in the audit only; production arithmetic stays in
the existing Rust core and existing response-transform API.

Split uses absolute residuals and rank ceil((n_cal+1)*.9). CQR uses
max(lower-Y,Y-upper), the same rank and the existing midpoint rule if widening
inverts a band. CV+ uses four equal folds, each held response's residual paired
with its fold model's query prediction, and the existing CV+ endpoint ranks.
Retain the full-model CV+ point function in original units. Preserve remainder,
guarantee, stopping and query-chunking rules.

## Fixed matrix

Generate n240 observations with NumPy seed636: x uniform[-1,1] and
Y=exp(.3+.6*x+.2*N(0,1)). Keep an unused rowid column. Query x[-.6,0,.6].
Cross two frontends, seven response specifications (bare, center, scale, log,
log1p, sqrt, exp), three methods (split, CV+, CQR), and two query modes (three
query rows or all training rows): 84 internal cases. Use rate .1, full path60,
requested prefix30, split/fold seed4, level.9, and half held for split/CQR.
Add 28 externally calibrated split cases: two frontends, seven specifications,
two query modes; train on the same proper-training rows and calibrate on the
same held rows. Total112 fixed cases per implementation. Retain actual row
identities, original-unit predictions/corrections/endpoints, errors, failures,
source/native identities and data hashes. Do not drop exceptions.

## Tuning boundary

Add eight supported tuned cases: two frontends, two methods (CQR, CV+), and
bare/log response, query rows only. Direct reference inner CV explicitly uses
preprocessing='refit'. Preserve the1600-step floor, doubling, cap10000,
boundary test and exact positive argmin. Retain chosen stops and all arrays.
Add twelve deliberate refusals: two frontends, three internal methods, and
center/scale with mstop='tune'. Fold-specific family risks with different
response scales have no common loss target; keep the existing named refusal.
Generic cvrisk loss semantics are outside this repair. This is a documented
unsupported operation, not an accepted numerical discrepancy.

## Acceptance and regression checks

Every supported corrected case must match independent points/endpoints and
corrections within1e-10 original-response units. Preserve every baseline
failure. Verify actual proper-training response parameters, calibration-Y
mutation cannot change proper-training functions or stops, predictor scale
and missing-value recipes compose with response inversion, raw retained rows
remain aligned after omission, unequal CV+ folds and chunk size1 agree, explicit
calibration uses the raw response source, invalid held responses refuse, and
bare-response behavior remains unchanged. No coverage improvement or sampling
uncertainty is inferred from this shared deterministic fixture. Publish complete
evidence, a reproducible notebook and a separate MR addressing #652; keep the
issue open for the explicitly unsupported tuning contract.
