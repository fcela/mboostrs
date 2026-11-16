# Calibration-score audit

This addendum is committed before inspecting numerical confirmation summaries
and before generating the additional audit cohort. The candidate estimator,
score, 17-point tail grid, applicability guard, target, comparator and coverage
window are unchanged. The original confirmation remains a primary reported
cohort; its results will not be replaced or selectively pooled.

The original runner retains cutoff, endpoints, nuisance parameters and source
identities but omits full calibration-score vectors. Add diagnostic observers
around the actual native score/quantile/reference/RNG calls. For each fixed
competitor, observe its actual calibration and query predictions and reconstruct
its declared score with the retained calibration responses. Also retain public
model coefficients for the mboostrs fits. Observers forward arguments/results
unchanged and restore all patched functions after each call.

Run a separately frozen50-draw-per-panel audit cohort with seeds6605100–6605149,
independent test seed+100000, the same five processes, partitions, methods and
settings. Keep every original and audited row, including errors and unbounded
sets. Report both cohorts separately. Default promotion now requires the
existing gate to pass on BOTH the original and audit confirmation cohorts.
An extra draw cannot rescue a failed primary cohort. This strengthens the gate;
it does not select a different algorithm or change a threshold from outcomes.

The earlier strict-JSON serialization failures remain separately archived.
The reversible nonfinite-location codec fixes only evidence transport; the
global public export question is tracked independently in #665. No Gaussian
scale rule or mean-fitting algorithm is changed in this audit.
