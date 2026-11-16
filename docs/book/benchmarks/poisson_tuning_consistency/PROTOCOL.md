# Poisson tuning and calibration confirmation

Instrument poisson-tuning-consistency/1. Freeze on source cc60781 before
implementation or measurement. Existing source uses nu0.1 for discovery mean
tuning and nu0.02 for its returned non-Gaussian mean fit. The candidate changes
only mean-tuning's explicit nu to the already declared fitting rate. Gaussian
rates, Poisson gradients, family screening, term selection, stopping caps,
calibration scores and default absolute score are unchanged.

## Draws and partitions

Five panels: low/original/high Poisson means a*exp(.3+.6*x1-.4*x2), with
a=.2/1/10; high-rate negative-binomial counts with conditional variance
mu+mu^2/5; and high-rate zero-inflated Poisson with zero probability.3 and
nonzero-component mean mu/.7, retaining unconditional mean mu. Predictors are
independent uniform[-1,1]. Generate1000 analysis and1000 independent test rows.
Development seeds6493100–6493109; confirmation6494100–6494149. Test seed is
seed+100000. Panels share streams and are not pooled for uncertainty. Reuse
the public Rust two-fold partition:500 proper-training and500 calibration rows.

Run development first. The source-based candidate is fixed now; no development
result selects a rate, score, representation or stopping threshold. Freeze the
corrected implementation and runner before confirmation; do not inspect or fit
confirmation inputs before that commit. No failed draw is replaced, no setting
changes during confirmation and no completed evidence is overwritten.

## Methods and information

Current and corrected data-only discovery run only on the same proper-training
rows, without supplied family or formula. Retain selected family/formula,
stopping path, CV settings, warnings and provenance. Their returned mean is
kept fixed across three separate external calibration scores: absolute,
poisson_normalized, poisson_quantile. Retain a selected unsupported model as
an explicit failed score row; do not substitute an oracle Poisson fit.

The supplied track uses the original Poisson log-linear formula
`y ~ bols(x1)+bols(x2)`, nu.1 and mstop100, on the same rows, with all three
scores. Add a supplied proper-training-only tuned model at nu.02 (start100,
cap400, five folds, exact minimum, fixed step) as an explanatory mean control,
also with all three scores. It is not a data-only success. Retain a true-mean
prediction diagnostic without calibrating it or using it for model selection.

The five original controls are CatBoost split/CQR (500 trees, depth6, rate.05),
LightGBM split/CQR (300 trees,15 leaves, rate.05), and sklearn CQR (300 trees,
depth3, rate.05). All use the same fitting/calibration rows and one thread.
No competitor tuning or replacement is allowed. Calibration is applied after
all fitting and selection, and test responses only reach the evaluator.

## Metrics and decisions

Retain actual input hashes, row partitions and every test prediction/endpoint,
calibration correction, model metadata, warning, failure and elapsed workflow
cost. Scores from one fit share its fitting cost; do not count it as three fits.
Report true-mean squared error as well as response-point MSE, interval score,
real interval length, coverage, true-mean-tercile conditional coverage, common
nonnegative-integer count cardinality, empty sets and support-equivalent
coverage. Test n1000, partition cardinalities, source/binary identity, both
old/new calibration exclusion and the actual nu passed into CV and final fit.

LightGBM CQR is the prespecified primary reference, not a winner selected from
confirmation outcomes. A method/reference is efficiency eligible only if mean
coverage is in[.88,.92]. Primary #636 metric remains mean paired percentage real
width excess. For #649/#650 report paired percentage count-size excess as well.
Target is at most1%; report point-target passage separately from a one-sided95%
Student-t upper-bound passage. Always report MC SE, every competitor, both
information tracks, unsupported cases and ineligible outcomes. A default-policy
change is not made in this study. Opt-in improvements alone do not close #636.

For a correction contrast, pair current/corrected discovery within each draw
and score. For fixed-mean calibration contrasts, pair scores from the identical
fit. Report these changes separately from the competitor comparison. Evaluate
all50 confirmation draws per panel; no optional stopping, extrapolation to
unmeasured panels or change in loss/eligibility to meet the target. Fresh draws
can support a bounded measured improvement and expose residual gaps; they do
not establish universal one-percent dominance.
