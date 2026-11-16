# Conditional spread comparison

Instrument conditional-scale-comparison/1, frozen before this prototype and
any new mean-versus-scale policy measurement. Baseline is count integration
0b12cf1. Prior sharp-mean development6293100–6293109 is development evidence;
no outcome from it is independent confirmation. The separate standardized-
score experiment leaves spread selection unchanged and is not this candidate.

## Candidate

Retain existing mean/family/interaction selection, scale nomination (absolute
slope t and positive held-out curve gain), candidate restrictions, SCREEN_MSTOP
200, learning rate.1, five folds, best-candidate rule and paired two-SE gate.
Change only the conditional reference used for the scale comparison. On each
fold's fitting rows, fit the selected mean formula once with ordinary Gaussian
at200 steps. Reuse its identical mean predictions for constant and every
nominated varying-scale candidate, on fitting and judging rows respectively.
No sigma candidate can change that fold's mean fit.

Compute absolute residuals using the shared core. Fit GaussianLSS to those
residuals with location fixed exactly at zero (offset0, mu budget0) and scale
budget200. Use offset log-scale0, rate.1 and no gradient stabilization for
all constant/varying sigma fits. The Gaussian likelihood at location zero is
identical for a signed residual and its absolute value. Score held-out residuals
under each resulting scale and compare paired fold mean NLL. Keep a candidate
only if it is best and beats constant scale by the existing two-SE rule.

This prevents a scale candidate from winning the comparison because its joint
mu fit differs. Residual scale can still contain mean approximation error; no
irreducible-noise or causal variance interpretation follows. Final joint LSS
stopping/refit is unchanged after a genuinely supported adoption. Standard
Gaussian fitting is unchanged when no scale is adopted. No screening-budget
increase or assumed true mechanism enters ordinary discovery.

Before production integration, prototype in benches only through existing
public fitting APIs and shared-core residual/risk primitives. Record the exact
nomination inputs, per-fold means and residuals, fitted scale predictions,
fold identities, risks and decision. The production statistic must live in
Rust core with thin Python/R bindings. Cost admission must include the extra
fold mean fit and all sigma fits; a refusal retains constant scale and the
recorded reason. Preserve mandatory terms and candidate restrictions.

## Controls and confirmation

Use the ten sharp-mean-controls/1 processes with five observed uniform[-1,1]
columns, analysis1000/evaluation2000 rows and evaluation seed+100000. Keep the
same Rust500/500 fitting/calibration partition. Development6293100–6293109;
separate fresh confirmation6296100–6296149 is not generated until the concrete
production candidate and complete runner are committed after development.
Do not use calibration/test responses to nominate, select or tune anything.

Retain ordinary discovery, the conditional candidate, spread=False discovery,
supplied constant/varying GaussianLSS, the labelled oracle-mean residual
control and all five original fixed competitors. Use existing absolute or
central-band calibration; standardization is not part of this candidate.
Report all means, intervals, scores, hashes, models, fold details, warnings,
costs, failures and any unbounded/empty sets. Metrics include latent-mean and
response MSE, real width, interval score, marginal coverage, true-scale-tercile
coverage and sharp-boundary coverage. All controls and failures remain visible.

Primary #629 diagnostic: a seed has a false scale attribution if either sharp
constant-variance process selects varying scale. Cluster the two processes by
seed because their streams overlap; do not treat them as independent. Require
at least50% reduction in this seed-level rate relative to baseline and a
one-sided exact McNemar/binomial p<=.05 on discordant independent seeds.
Also report each process separately. Require true-scale adoption on at least
90% of draws of each genuinely varying process (axis, shifted, absolute, bell,
monotone), and no increased false-adoption count on smooth/null/heavy-tail
constant-variance controls. Report target point estimates and uncertainty;
no zero-population-error claim follows from observing zero errors.

Candidate default intervals must be finite and mean marginal coverage in
[.88,.92] on every process. Compare width against prespecified LightGBM CQR,
retaining all five controls and paired SE/one-sided95% upper bounds plus
five-comparator Bonferroni bounds. A #630 claim additionally requires a
separately frozen original two-predictor bell/monotone comparison; this sharp-
mean study cannot replace that task. Retain the1% real-width target. No
threshold or candidate is changed from confirmation outcomes.
