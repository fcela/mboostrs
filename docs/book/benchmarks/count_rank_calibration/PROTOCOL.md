# Count-rank calibration

Instrument count-rank-calibration/1. This protocol is committed before candidate
implementation or measurement. Baseline is main5c68ea5, including the consistent
mean-tuning correction. Earlier confirmation outcomes are development evidence
for this new candidate, never its fresh confirmation. Issues636,650 and660 retain
their one-percent targets and the coverage window[0.88,0.92].

## Reference distribution and score

Keep the fitted conditional mean mu(x) fixed. On proper-training rows only,
fit one global zero-inflation probability pi for a Poisson reference with
component mean mu/(1-pi), so its marginal mean remains mu. Minimize the weighted
conditional-mean likelihood over[0,1-1e-8], additionally bounded by the existing
exact-count range. Compare the exact pi=0 endpoint with the minimizer. Degenerate
zero means retain pi=0. A reference-fit parameter is a score ingredient, not a
claim that a true zero-generation mechanism has been identified.

For the fitted count CDF F, use the jittered continuous variable Z=Y+U,
U~Uniform[0,1), whose rank is G(Y,U)=F(Y-1)+U*(F(Y)-F(Y-1)). Generate independent
auxiliary streams for calibration and query rows in Rust with ChaCha8, retaining
the realized seed. The same seed and row order reproduce the same results;
query length does not change calibration draws and query chunks use offsets.
Coverage is marginal over observations and this auxiliary randomness.

Choose a(x) from17 equally spaced values in[0,1-level] by minimizing
Q_Z(a+level)-Q_Z(a), ties taking the smallest a. This depends only on the fitted
reference and x. The score is max(a-G,G-a-level). Calibrate its ordinary finite-
sample conformal order statistic q. At a query, return
[Q_Z(a-q)-U,Q_Z(a+level+q)-U], intersected with known nonnegative count support.
Rank bounds outside[0,1] remove the corresponding constraint; upper rank>=1
gives an unbounded upper endpoint even if the fitted reference is degenerate.
Represent an empty support intersection by a singleton strictly below zero,
using the unprojected upper endpoint. This preserves ordered real endpoints
while its nonnegative-integer count set is empty; retain every such set.

The score is fixed before calibration, so exchangeability of augmented
(X,Y,U) gives marginal split-conformal validity even under a wrong reference.
Conditional efficiency depends on the reference; no conditional-coverage
guarantee or identification of a noise mechanism follows from this argument.
All fitting, CDF inversion, scoring, tail allocation and RNG numerics live in
Rust core, with matching thin Python and R interfaces.

## Automatic policy

The candidate public default is score='auto'. For standard Poisson/NBinomial
fits with untransformed, nonnegative integer training responses and split
calibration, use the fitted count-rank score. Compute the proper-training ratio
D=sum(w*(y-mu)^2)/sum(w*(mu+pi*mu^2/(1-pi))). If D>2, use the existing normalized
score instead. Set D=0 for a zero numerator/denominator and infinity for positive
numerator with zero denominator. This is a prespecified applicability guard,
not a statistical test or a variance interpretation. Other models/methods keep
their existing score/routing. Explicit absolute/normalized/central-quantile
requests retain their existing definitions. Explicit count_rank also permits
an uninflated Poisson reference as a mechanism control; it does not use the
automatic applicability fallback. Record the requested/resolved score, fitted
pi, D, support and randomization seed. No calibration response selects these.

Default promotion is gated prospectively: original-rate supplied-fixed and
data-only default rows must each have eligible coverage and one-sided95%
upper limits <=1% for both real-width and count-size excess against the
prespecified reference. Default rows in every panel must have finite width
and eligible marginal coverage. A failed gate leaves the legacy default in
place and the default-policy issue open, while retaining all candidate results.
Do not alter a score, grid, guard, interval representation or gate from these
confirmation outcomes. High-rate supplied-fixed limitations remain reported.

## Draws and comparisons

Use the previous study's five processes: low/original/high Poisson multipliers
.2/1/10, high negative binomial variance mu+mu^2/5, and high zero-inflated
Poisson with probability.3 and component mean mu/.7. Predictors and n1000
analysis/n1000 independent test generation follow study_poisson_tuning_consistency.
Development seeds6603100–6603109; confirmation6604100–6604149; test seed+100000.
Use the same Rust500/500 fit/calibration partition for every method. Panels
share streams and must not be pooled for uncertainty. Freeze the implementation
after development and before generating confirmation inputs; no outcome selects
a nuisance model, score or default rule.

Both information tracks remain: data-only discovery; supplied fixed Poisson
bols(x1)+bols(x2), nu.1,mstop100; supplied properly tuned nu.02, five-fold
start100/cap400. Every fitted mean is reused across absolute, normalized,
central-count quantile, uninflated rank, fitted-zero rank and actual default
candidate rows. Selected LSS models use the existing distributional default;
unsupported explicit standard-count scores remain failed, never substituted.
No successful-only method/draw averages. Keep all five original fixed controls:
CatBoost split/CQR500trees/depth6/rate.05; LightGBM split/CQR300trees/15leaves/
rate.05; sklearn CQR300trees/depth3/rate.05. One thread per model.

LightGBM CQR is the primary prespecified reference. Report every other eligible
control, with the existing five-comparator Bonferroni bound, without choosing
a confirmation winner. Retain original real width as primary636 metric and
common nonnegative-integer cardinality as co-primary650/660 metric. Report
interval score, latent-mean/response point errors, marginal/scale-tercile
coverage, empty sets, cost and every failure. The real interval score uses the
returned ordered real endpoints, including below-zero singleton representations
of empty count sets; count-set emptiness is reported separately. Do not count a
support-width change alone as a count-cardinality improvement.

Archive all inputs/hashes/partitions, fitted-reference parameters, actual scores,
auxiliary seeds, endpoints/points, warnings, source/native identities and failed
attempts. Validation includes independent SciPy/R CDF/likelihood references,
inversion of the score set on a dense integer grid, boundaries/zero means,
empty and unbounded sets, repeatability/chunking, weights, fixed path prefixes,
internal/external calibration equivalence, calibration-mutation exclusion,
unchanged explicitly requested legacy scores and both language interfaces.

References: Dunn and Smyth(1996), Randomized quantile residuals,
https://gksmyth.github.io/pubs/residual.pdf; Chernozhukov,Wuthrich and Zhu(2021),
Distributional conformal prediction, https://pmc.ncbi.nlm.nih.gov/articles/PMC8640792/.
The discrete-jitter extension and score-set inversion above are stated explicitly
rather than attributing a discrete theorem to a continuous-outcome result.
