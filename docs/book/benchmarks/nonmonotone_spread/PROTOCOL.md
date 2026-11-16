# Nonmonotone spread nomination study

Instrument `nonmonotone-spread/1`, issue #562. Freeze this document before
measurement or implementation changes. The study tests whether curve-sensitive
nomination recovers varying Gaussian scale missed by a signed slope while avoiding
scale adoption on constant-variance controls. It measures D9 interval calibration;
it makes no claim for other scorecard dimensions.

## Nomination and adoption

Keep the current linear nomination, absolute t >= 2. Add a second route: positive
held-out reduction of mean squared log-squared residual error, averaged over five
half-splits with seeds 0 through 4, using eight quantile bins. Edges and bin means
are learned on each fit half; the Rust main-effect screening kernel evaluates
the held half. The five overlapping splits stabilize a ranking; their variation
is not an inferential standard error. This is a nomination, not a hypothesis
test. Retain discovery's current GaussianLSS comparison against constant scale,
two-change-SE adoption rule, candidate restrictions, cost admission, tuning,
refit and interval calibration. No calibration or test response nominates a
column. Do not alter these settings in response to this study's test scores.

## Processes and partitions

Each independent draw has 1,000 development rows and 2,000 fresh test rows.
Predictors x1 and x2 are independent uniforms on [-1, 1]. The mean is
sin(2*pi*x1) + 0.5*x2. Noise scales:

- `absolute`: 0.3 + 1.5*abs(x1), the original issue process.
- `monotone`: exp(0.7*x1), a previously detectable control.
- `bell`: 0.3 + 1.5*exp(-8*x1*x1), a nonmonotone control of different shape.
- `constant`: 1, Gaussian noise, the negative control.
- `heavy_tail`: 1, unit-variance t3 noise, the normal-errors violation control.

Original seeds 0 through 5 are diagnostic reproduction only. Fresh confirmation
seeds are 562100 through 562119; test seed is training seed + 50000. No seed is
replaced. Capture every warning, failure, elapsed time, chosen mean/spread formula,
spread-stage log, and input/prediction identity. Failure is an incomplete method,
not an omitted draw. Checkpoint completed rows. Do not pool original and fresh
seeds for Monte Carlo uncertainty. A failed rule remains reported; any revised
rule needs a new frozen confirmation batch and versioned protocol.

## Methods and information tracks

Discovery receives only the development data and response name and runs public
`mb.discover` with its defaults (seed fixed to the draw seed, level 0.90). Compare
unaltered main with the proposed nomination change on identical draws.

Known specification receives the true mean formula bbs(x1, df=4) + bols(x2),
Gaussian family and the x1 scale formula for the three varying-scale processes.
Use public gamboostLSS (GaussianLSS), mstop 100 and distributional conformal
intervals, level 0.90, seed fixed to the draw. For the constant/heavy-tail
processes use Gaussian gamboost and split conformal, mstop 100. This track
measures a fixed supplied estimator, not tuned optimal estimation.

Competitors are CatBoost split and CQR (500 trees, depth 6, rate 0.05), LightGBM
CQR and split (300 trees, 15 leaves, rate 0.05), and sklearn gradient-boosting
CQR (300 trees, depth 3, rate 0.05). These are the original D9 fixed settings,
untuned. The same seed+10000 half split supplies 500 fit and 500 calibration
rows to each competitor. Discovery uses the development rows through its
existing internal selection/refit/calibration path; record its actual use and
state the differing training-data use. Pin every method to one thread, disable
CatBoost file writing, measure elapsed whole workflows, and make no low-level
speed claim. Competitors receive no true formula. Record their scores beside
each mboostrs information track, without double-counting their draws.

## Metrics and decisions

Report nominal 90% coverage, coverage in low/high true-scale thirds, mean width,
width rank correlation with true scale, interval score (width plus 20 times
miss distance), and exact-oracle mean width. Constant-scale rank correlation is
undefined and reported absent. The exact central interval is Gaussian except
for heavy_tail, where it uses standardized t3 quantiles.

The primary competitor gap is the paired percentage width ratio to the
retrospective best mean-width competitor whose mean coverage lies within 0.02
of 0.90 on that process. State that this choice uses test results every time.
Report paired Monte Carlo SE across 20 independent draws, plus coverage and
interval-score differences so narrowing cannot conceal undercoverage. Mark
methods outside the same coverage tolerance as ineligible for width leadership.
Report known and discovery separately. All methods and regressions stay visible.

Adoption incidence, true-driver nomination and constant-scale false-adoption
incidence diagnose selection. Nomination should recover the absolute-scale
process, final adoption must still pass the existing held-out score gate, and
constant-scale false adoption must be reported. An efficiency gap above 1%
remains an issue; this experiment does not close #555 unless current evidence
meets its target. A selection improvement alone cannot establish interval
calibration or competitive width. A weaker result is retained, not retuned.
