# Remaining benchmark designs

This design is specified on main cc60781 before new measurement. Previously
published outcomes are development evidence. They cannot become independent
confirmation by renaming or rescoring them. A merged option and a passing
numerical regression test do not establish an efficiency target.

## Efficiency target

For #636 and #630 the primary estimand remains the mean of paired percentage
real-width excesses, `100 * (method_width / reference_width - 1)`, across
independent draws. Count cardinality after the common intersection with
nonnegative integers is a separate co-primary for #649/#650; it does not
replace real width. Coverage eligibility remains [0.88, 0.92] at nominal 0.90.
Report each draw and Monte Carlo standard errors. A point excess of at most 1%
is labelled a point-target result; claiming the population target requires
the one-sided 95% Student-t upper confidence limit to be at most 1%. Neither
criterion is a universal guarantee on other data-generating processes.

Use LightGBM CQR as the primary prespecified reference, as identified in the
previous studies. Retain all five original fixed controls. Report every
coverage-eligible control rather than choosing a new confirmation winner.
Use a Bonferroni one-sided bound across the five controls for a stronger
claim against every eligible control, and label it separately. If a reference
is ineligible or fails, report an unavailable comparison. Never drop failed
method/draw rows to obtain a favourable paired comparison.

## Sharp means and scale: #629 and #608

Before changing a spread rule, retain baseline discovery and freeze a factorial
control family: axis-aligned and shifted sharp pair boundaries with constant
sd 0.5 and with varying sd `0.3+1.5*abs(x1)`. Add smooth-mean constant, absolute,
bell and monotone-scale controls, a null-mean constant control and a heavy-tail
constant-variance control. Use five observed uniform[-1,1] columns; sharp means
are `3*sign(x1-a)*sign(x2-b)+x3`, with `(a,b)=(0,0)` or `(.25,-.2)`.
Smooth means are `sin(2*pi*x1)+.5*x2`; the null mean is zero. Gaussian errors
apply except Student-t3 errors divided by sqrt3 in the heavy-tail control.

Development seeds 6293100–6293109 and confirmation seeds 6294100–6294139 are disjoint
from previous studies; independent test seed is seed + 100000. Use 1,000 analysis
rows and 2,000 test rows, a common 500/500 fitting/calibration split and no
test/calibration feedback into discovery. Include discovery with spread on/off,
a supplied pair-tree mean with the correct linear main term, the same supplied
mean with constant/varying GaussianLSS scale, an oracle-mean residual diagnostic
and all five original competitors. Record false scale attribution, recall of
genuine scale, latent-mean MSE, response MSE, interval score, widths, marginal
coverage, low/high true-scale coverage and coverage near each sharp boundary.
The oracle is a diagnostic lower bound, never a discovery input or competitor.
Do not increase the screening budget or forbid true varying scale to make a
constant-variance control pass.

The original #608 task additionally has ten columns, n = 2,000 and a 70/30 split.
Retest that task separately with fresh draws before changing its claim. Compare
the existing all-column depth6 tree plus all smooth main terms against supplied
pair trees with/without irrelevant mains, and data-only discovery. Keep the
original200-tree CatBoost/LightGBM references and report NRMSE and latent-mean
error. A btree term already fits successive gradient trees; no ensemble rewrite
is inferred from the original single-learner terminology. The threshold for a
new #608 performance claim is at most1% paired NRMSE excess on fresh draws,
with a separately reported upper bound, rather than an interval-width result.

## Gaussian efficiency: #630

Use the same smooth absolute, bell, monotone and constant controls. Hold raw
fit/calibration partitions fixed. Compare baseline discovery, supplied fixed
GaussianLSS, proper-training-only tuned GaussianLSS, and CQR, with the five
competitors. Isolate mean/scale representation, stopping, and calibration in
separate contrasts; do not attribute a joint score gain to variance recovery.
No rule change is adopted until the sharp-mean controls above are available.
Freeze a candidate choice on development and commit it before reading any
confirmation outcomes. If no candidate is ready, confirmation is not run and
the issue stays open. The one-percent target is retained.

## Poisson means and calibration: #636, #649 and #650

The accompanying Poisson protocol freezes new draws and a source-diagnosed
tuning/refit learning-rate consistency correction. It compares fixed means
across absolute, normalized and central-quantile scores, plus the current and
corrected data-only workflows. A candidate family-aware default is represented
prospectively by the rule 'normalized score for a selected standard count mean,
absolute otherwise'; it remains an experimental policy, not a shipped default.
A successful opt-in row does not close #636. Adopting a default requires the
frozen confirmation target and a separate implementation/review decision.
No randomization of the count-quantile ties is introduced in this batch; #650
remains open if its fixed conservative method fails eligibility or efficiency.

## Cross-validation ties: #653

Keep the existing R-compatible exact-computed-minimum rule: exclude iteration0,
ignore nonfinite risks, select the first grid position among exact equal minima.
Approximate numerical equality is not an exact tie. Specify and test this
decision on newly constructed curves spanning tiny/unit/large risk magnitudes,
exact duplicates and distinct adjacent floating-point minima. Do not introduce
a risk tolerance from the observed76-versus1556 example. Preserve those archived
curves and stops. The existing one-standard-error rule is a different,
prespecified statistical choice; report it separately and do not use it to
relabelling the original tuned runs. Model predictions and attained risk are
reported alongside stopping labels. A new 'stable optimum' claim would require
a separately specified policy and tests; exact-minimum semantics do not promise
identical stops for curves differing by roundoff.
