# Sharp-mean control development

Instrument sharp-mean-controls/1, specified before measurement or any Gaussian
spread-policy change. This is the development control inventory for #629 and
the prerequisites in docs/benchmarks/gap_resolution_design.md. It does not run
the later efficiency confirmation, all five competitors, or the original
ten-column #608 retest; those remain separate pending experiments. No result
from this inventory can close #629/#630/#608 or establish one-percent efficiency.

Use development seeds6293100–6293109 and the ten processes specified in the
master design: axis/shifted sharp means crossed with constant/varying scale;
smooth constant/absolute/bell/monotone scale; null constant scale; smooth
heavy-tail constant variance. Generate1000 analysis and2000 independent test
rows (test seed+100000), five independent uniform[-1,1] columns named x1,x2,x3,
n0,n1. Mean and scale formulas follow the master design. Smooth/null constant
sd is1; sharp constant sd is.5. Heavy-tail errors are t3/sqrt3.

Use the common Rust two-fold assignment to reserve500 fitting and500 calibration
rows. Baseline methods are discovery with spread on/off (default400 stopping
cap, no extra knowledge), supplied fixed GaussianLSS constant scale and supplied
fixed GaussianLSS varying scale. Supplied sharp mu is
`btree(x1,x2,maxdepth=3,selection=sse)+bols(x3)`; smooth mu is
`bbs(x1,df=4)+bols(x2)`; null mu is1. LSS sigma is1 or `bbs(x1,df=4)`;
100 updates per parameter, nu.1. Supplied models are diagnostic controls, not
data-only success. Do not select the best supplied result using test outcomes.

An oracle-mean residual control subtracts the known generating mean on fitting
rows only, fits an offset-only Gaussian model and calls the existing public
heteroscedasticity diagnostic. Retain its slope and held-out curve nominations.
The oracle never reaches ordinary discovery or changes any decision.

Calibrate every fitted model on the same500 untouched calibration rows. Record
every prediction, interval, warning, failure, selected family/formula/stopping
count and scale-adoption decision. Evaluate latent-mean MSE, response MSE,
interval score, real width, marginal coverage, true-scale low/high coverage,
and sharp-boundary coverage within.1 of either true threshold. False scale
adoption refers to a selected varying scale in a truly constant-variance
process; it is an interpretation diagnostic, not proof of worse prediction.

The later40-draw-per-process confirmation seeds6294100–6294139 remain untouched
until a concrete candidate/control/competitor design is additionally committed.
No Gaussian selection rule, nomination threshold, budget or default is changed
in this development inventory. Retain all failures and MC uncertainty over
independent draws, while never pooling different processes sharing streams.

Decision (#815): a documented limitation is accepted in place of a requirement that the default recover a smooth scale under a sharp mean; see SHARP_SCALE_DECISION.md in this directory.
