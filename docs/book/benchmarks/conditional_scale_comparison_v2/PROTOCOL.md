# Cross-fitted mean adequacy and spread

Instrument conditional-scale-comparison/2, frozen after version1 development
and before this candidate or any confirmation. Version1 retains zero varying-
scale adoptions on all constant-variance development controls, but detects
only1/10 axis and4/10 shifted genuinely varying sharp controls. It detects all
smooth varying controls. Its frozen gate therefore fails; no product rule or
confirmation changes from that experiment. Keep every version1 outcome.

Change the developmental candidate in two specified ways. On each of the same
five outer fitting/judging folds, select the Gaussian mean stopping count on
that fold's fitting rows only, using five inner folds, rate.1,start100 and the
existing discover max_mstop cap (default400), seed=draw+outer_fold+1. This
explicit proper-training-only tuning replaces the fixed200-step mean in this
candidate; it does not change the baseline screening budget. Retain all tuning
curves, caps, selected stops, costs and boundary warnings. Judge-fold responses
never tune their mean fit.

For nomination, assemble the held-out Gaussian mean predictions across the
outer folds and use their absolute residuals in the existing slope/held-half
curve diagnostic. Keep |t| threshold, curve-gain threshold, tested predictors,
transforms and sigma restrictions unchanged. This prevents a residual screen
from describing only the same-sample errors of an overfitted mean. Reuse those
fold mean fits for the subsequent conditional scale comparison; constant and
all nominated varying sigma candidates still share exactly the same fold
locations. Scale training uses that fold mean's fitting residuals, as in
version1. Scale-only fits remain mu0/sigma200, offset0,rate.1,no stabilization.
The best-candidate/two-SE gate and final joint stopping/refit remain unchanged.

This is development-only code in benches. Production numerics and matching
language bindings, cost admission, supported observation weights, mandatory
terms and restrictions must be implemented and validated before confirmation.
Cross-fitting does not identify irreducible noise; approximation error can
remain in predictive residual spread and must be reported.

All ten processes,1000 analysis/2000 independent test rows, Rust500/500 split,
development6293100–6293109 and evaluation seed+100000 follow version1. Keep all
version1 and original baseline records. Fresh6296100–6296149 remains untouched
until the final production candidate is frozen. Version1 acceptance criteria,
including clustered sharp false-adoption reduction, >=90% true-scale recall,
constant/heavy-tail controls, marginal coverage, all competitors, full arrays
and explicit #630 original two-predictor scope, remain unchanged. No candidate
or threshold is revised from confirmation outcomes.
