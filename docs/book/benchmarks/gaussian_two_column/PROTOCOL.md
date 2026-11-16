# Gaussian two-predictor efficiency

Instrument gaussian-two-column/1, frozen before new implementation or draws.
Baseline is count integration0b12cf1. The five-column sharp-control experiments
and Gaussian standardized-score experiment are development evidence, not a
confirmation of the original two-predictor task in #630. Keep them unchanged.

## Processes and data use

Use the original study_nonmonotone_spread.draw generator: two independent
uniform[-1,1] predictors, mean sin(2*pi*x1)+.5*x2, Gaussian errors with constant
sd1, absolute sd.3+1.5*abs(x1), bell sd.3+1.5*exp(-8*x1^2), or monotone
sd exp(.7*x1). The heavy-tail constant-variance control has t3/sqrt3 errors.
Analysis1000 rows, independent test2000 rows, test seed+100000. All methods
share the same Rust500/500 fitting/calibration partition. Only fitting rows
select forms and stopping; no calibration/evaluation response reaches selection.
Development6303100–6303109; fresh confirmation6305100–6305149 remains untouched
until development, the selected candidate and complete runner are committed.

## Fixed workflows

Retain eleven methods: ordinary data-only discover; separate-budget refinement
of that same discovered specification; supplied fixed GaussianLSS100; supplied
diagonal-tuned GaussianLSS; supplied separate-budget GaussianLSS; supplied CQR;
and all five original CatBoost/LightGBM/sklearn split/CQR controls. Discovery
receives no family/formula and uses the actual public defaults: the ordinary
mean ceiling is400, and an adopted LSS model has ceiling1600. This distinction
is clarified before measurement (#674). The refinement
preserves its selected family/formulas. A standard Gaussian discovery result
is reused unchanged: the separate LSS search is inapplicable, not substituted
with an oracle scale formula.

Supplied mean is bbs(x1,df4)+bols(x2), sigma bbs(x1,df4) on genuinely varying
processes and sigma1 on constant/heavy-tail controls. Family GaussianLSS,
rate.1. Fixed has100 updates per parameter; diagonal tune uses five folds,
start100/cap400 and draw seed. Separate-budget tuning uses the public model
cvrisk with the fixed Cartesian grid mu in[25,50,100,200,400] crossed with sigma
in[0,25,50,100,200,400], five Rust folds and draw seed. Select the exact native
minimum on that prespecified grid, then update the model with the corresponding
named per-parameter budgets. Grid order increases mu, then sigma, retaining
the existing exact tie policy. Every non-diagonal point is fitted afresh by
Rust; do not read off-diagonal combinations from a diagonal path. Retain the
risk matrix, aggregate, grid, chosen budgets and actual fit/cost counts.

Supplied CQR fits the given mean formula separately at tau.05/.95,1600 fixed
updates/rate.1 on the500 fitting rows, then calibrates using the common500
calibration rows. Fixed competitors are unchanged: CatBoost500trees/depth6/
rate.05; LightGBM300trees/15leaves/rate.05; sklearn CQR300trees/depth3/rate.05,
one thread and draw seed. Use the legacy central-band distributional score
and absolute standard score throughout; normalized-score development failed
promotion and is not selected here.

## Target and limits

The prespecified #630 workflows are data-only separate-budget refinement and
supplied separate-budget tuning. The original100-step supplied fit and ordinary
diagonal discovery remain visible controls. This decision tests proper-training
model selection rather than demanding that an untuned100-step preset be
optimal, as #630 explicitly distinguishes fixed settings from tuned performance.
It does not relabel the100-step or ordinary default results as fixed.

Primary reference is prespecified LightGBM CQR. Retain all five controls and
separate five-comparator Bonferroni upper bounds. At nominal.90 require coverage
in[.88,.92]. The target is <=1% mean paired percent real-width excess, reported
with Monte Carlo SE, point criterion and one-sided95% Student-t upper criterion.
Require the upper criterion for BOTH target workflows on BOTH bell and monotone
processes, and finite/coverage-eligible candidate intervals on every process.
Do not pool processes sharing streams or choose a new confirmation winner.

No default changes or causal noise-attribution claim follow from development.
A passed fresh gate supports a bounded public-workflow result and recommendation;
a failed gate leaves #630 open. Keep all old presets, failures, empty/unbounded
sets, source/native identities, model choices, calibration score/cutoff calls,
partitions, warnings, costs, full interval/point arrays, latent-mean and response
MSE, real width, interval score and marginal/true-scale-tercile coverage. The
sharp-mean #629 controls and their failures remain separate and cannot be
closed by an efficiency result on smooth two-predictor data.
