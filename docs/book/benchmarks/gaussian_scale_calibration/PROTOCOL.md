# Gaussian scale calibration

Instrument gaussian-scale-calibration/1, frozen before candidate implementation
or new measurement. Baseline is count-calibration integration0b12cf1, which
includes main768519f and leaves Gaussian distributional calibration unchanged.
Previous #562/#630 outcomes and sharp-mean development controls6293100–6293109
are development evidence. Confirmation seeds6294100–6294139 remain unread.

## Candidate and information tracks

Hold each fitted mean and scale fixed. Compare existing central Gaussian
quantile bands with additive residual correction against standardized scores
abs(y-mu)/sigma and intervals mu +/- q*sigma. The latter score and inversion
are implemented once in Rust core and bound to Python and R. Strictly positive
finite scales and finite responses/means are required; invalid inputs fail
explicitly, rather than choosing a floor after observing results. Nonfinite
q follows the existing valid-unbounded conformal convention. The finite-sample
order statistic and nominal90% coverage remain unchanged. This score estimates
predictive residual spread, not an identified irreducible-noise mechanism.

Apply the candidate only to GaussianLSS with untransformed response. Other LSS
families and transformed responses retain their existing distributional band.
Standard Gaussian models retain absolute scoring. Explicit requests for the
existing distributional score retain their definition. Public auto promotion
requires the confirmation gate below; a failed gate leaves the legacy default
unchanged. No spread nomination/adoption threshold or screening budget changes.

Use these fixed tracks on proper-training rows only:
- data-only ordinary discover, default400 cap, seed equal to the draw, no
  supplied family/formula. Reuse the returned fit across both scores; a
  standard Gaussian result gives identical absolute bands in both rows;
- supplied varying GaussianLSS: known mean covariates/form and sigma bbs(x1,
  df4),100 updates per parameter, rate.1; both scores reuse one fit;
- supplied tuned varying GaussianLSS: same formulas, five-fold tune_mstop,
  start100/cap400/rate.1, draw seed; both scores reuse one fit;
- supplied constant GaussianLSS with the same mean formula, sigma1,100
  updates per parameter/rate.1; retain both scores as a diagnostic control;
- discovery spread=False as a mean-only diagnostic; and supplied-mean CQR
  with fixed1600 updates/rate.1, no test-driven stopping choice;
- existing oracle-mean residual diagnostic, labelled and excluded from
  ordinary discovery/efficiency claims; it sees true mean on fitting rows only.

Supplied sharp mean is btree(x1,x2,maxdepth3,selection=sse)+bols(x3), smooth
mean bbs(x1,df4)+bols(x2), null mean1. These supplied models have information
ordinary discovery does not. Keep all five original fixed controls: CatBoost
split/CQR500trees/depth6/rate.05, LightGBM split/CQR300trees/15leaves/rate.05,
sklearn CQR300trees/depth3/rate.05, one thread and draw seed.

## Draws and reporting

Use all ten frozen sharp-mean-controls/1 processes and generation code:
axis/shifted sharp means with constant/varying scale; smooth constant, absolute,
bell and monotone scale; null constant; smooth constant-variance heavy tail.
Analysis1000 rows, independent evaluation2000 rows, evaluation seed+100000,
common Rust-generated500/500 fit/calibration partition. Development reuses
6293100–6293109, explicitly labelled development rather than fresh confirmation.
No calibration or evaluation response selects a fit, score or reference.

Preserve every method/draw, warnings, failures, empty/unbounded sets, all
calibration responses/scores/quantiles and endpoint/point arrays, fitted
mean/scale predictions, selected formulas/family/budgets, actual call inputs,
source/native hashes and runtime versions. Evaluate real width,90% interval
score, latent-mean and response MSE, marginal coverage, true-scale low/high
tercile coverage, and sharp-boundary coverage within.1 of either threshold.
Report varying-scale adoption on truly constant-variance processes and true-
scale recall separately; calibration alone cannot repair #629 attribution.
Keep original development controls and failures, without overwriting them.

After development, commit the candidate and runner before generating any of
6294100–6294139. Keep all40 confirmation draws per process and every method.
Primary reference is prespecified LightGBM CQR; retain every other fixed
control, with separate five-comparator Bonferroni bounds. Mean coverage must
lie in[.88,.92]. Width excess is the mean per-draw100*(width/reference_width-1),
with paired Monte Carlo SE and one-sided95% Student-t upper limit. Do not pool
processes sharing streams or select a new confirmation winner.

Automatic promotion gate: discovery and supplied-fixed normalized bands on
BOTH smooth bell and smooth monotone processes must have eligible coverage
and width-excess one-sided95% upper <=1%; all candidate default rows across
all processes must be finite and coverage eligible. Supplied tuned is an
explanatory control and cannot replace either required track. Failure retains
the existing default, all measured outcomes and the open #630 gap. Do not
change this score, scope, reference, guard or target from confirmation results.
Any later mean-versus-scale candidate for #629 requires a separate protocol
and independent confirmation cohort; these confirmation draws cannot be
reused as fresh evidence for it.

Decision (#815): a documented limitation is accepted in place of a requirement that the default recover a smooth scale under a sharp mean; see ../sharp_mean_controls/SHARP_SCALE_DECISION.md.
