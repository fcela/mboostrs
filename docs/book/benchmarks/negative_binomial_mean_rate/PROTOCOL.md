# Negative-binomial mean rate

Instrument negative-binomial-mean-rate/1, frozen before candidate implementation
or measurement. Baseline is count-calibration integration0b12cf1. The previous
count confirmation cohorts are development evidence for this candidate and
are not treated as independent confirmation. #660's zero-inflated count
cardinality target remains at most1%, with the real-width target retained.

## Candidate

Change only ordinary discovery's standard NBinomial mean-learning rate from
.02 to.1. Keep Gaussian.1 and every other family's existing.02. The candidate
uses the same rate for screening, CV stopping, cost estimates and final fitting.
No cap, learner family, nomination/adoption threshold, count reference,17-point
tail grid, dispersion applicability guard, conformal order statistic or RNG
rule changes. NBinomial's log-mean gradient is scaled by its fitted shape rather
than directly by a large Poisson mean; this motivates testing the existing
public fitting rate.1. It does not establish universal convergence.

Start with a benches-only profile override, retaining both old and candidate
rates on the same generated input. Freeze any production control policy in
shared Rust with Python/R bindings and regression checks before generating
confirmation; the prototype alone is not a default fix. A failed development
contrast keeps the existing default and all its outcomes.

## Data and methods

Use the same five count processes/generator as count-rank-calibration/1:
low/original/high Poisson, high overdispersed NB, and high zero-inflated Poisson.
Analysis1000/test1000 independent rows, common Rust500/500 fitting/calibration
partition, test seed+100000. New development6607100–6607109; independent
confirmation6608100–6608149 is not generated until production candidate,
runner and complete analysis are additionally committed after development.

Nine fixed methods: baseline data-only discovery with old rates; candidate
ordinary discovery with only NBinomial rate.1; supplied fixed Poisson100/.1;
supplied Poisson tuned fivefold/start100/cap400/nu.02; and all five original
fixed CatBoost split/CQR500trees/depth6/.05, LightGBM split/CQR300trees/15leaves/
.05, sklearn CQR300trees/depth3/.05 controls, one thread and draw seed.
Every method uses actual default count calibration. Supply no family/formula
to either discovery variant. Record inferred family, chosen formula, actual
learning rate, full tuning/adoption logs, stopping count and model coefficients.
Known-spec fits remain information-rich controls, never data-only success.

Calibrate on the same untouched500 rows. Preserve actual native reference,
score, cutoff, RNG and endpoint calls; complete competitor calibration/query
predictions/scores; auxiliary seeds; nuisance parameters; all endpoint/point
arrays; input hashes/partitions; warnings; failures; valid empty/unbounded sets;
software versions and source/native identities. Grade real width, common
nonnegative-integer cardinality, interval score, latent-mean/response MSE and
marginal/true-scale-tercile coverage. Never average a failed method only over
successful draws or omit infinite outcomes.

## Confirmation gate

Primary comparator remains prespecified LightGBM CQR, with coverage eligibility
[.88,.92] at nominal.90. Report all five controls and separate Bonferroni bounds.
Use paired percent excesses and Monte Carlo SE across independent draws,
with point<=1% and one-sided95% Student-t upper<=1% labelled separately.
Do not pool processes sharing streams or choose a confirmation winner.

For zero inflation, candidate data-only and supplied TUNED defaults must both
meet the upper criterion for real width AND count cardinality. The untuned
100-step supplied preset remains a diagnostic limitation, rather than replacing
the issue's properly tuned information track. Also preserve the original-rate
#636 gate for candidate data-only and supplied FIXED defaults on both metrics.
Every candidate-discovery process must be finite and coverage eligible. A failed
gate leaves the old rate/default and #660 open. No control, rate or threshold
is adjusted from confirmation outcomes. Report any high-rate or other stress
regression beside the result; no universal or noise-identification claim follows.
