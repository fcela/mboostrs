# Poisson conformal bands and count support

This protocol for issue #636 is frozen before implementation or measurement.
Source baseline is bd5a358. It has two parts: a measurement audit of the
previous Poisson panel, and fresh development/confirmation draws comparing
three calibration scores with each fitted mean held fixed.

## Response support and metrics

The response consists of nonnegative integer counts. For every method, retain
the original real endpoints and intersect its prediction set with that known
support: L=max(0,ceil(lower)), U=floor(upper), cardinality=max(U-L+1,0).
An inverted interval, or U<L after intersection, is empty. Do not replace an
empty count set with a singleton. Verify that intersecting with support never
changes coverage for any observed nonnegative integer response, including
integer boundary cases. Apply this operation identically to all methods.

Primary efficiency is mean count-set cardinality (number of possible counts),
not real length. Report raw real length, nonnegative-support real length and
integer endpoint span separately; they are different estimands. Retain the
previous real-endpoint interval score and point MSE, which are also different
estimands. A Poisson mean and a CQR midpoint are distinct point predictors.
Report marginal coverage, coverage within low/high true-mean thirds, empty
sets and workflow/calibration elapsed time. No support treatment is applied
to mboostrs alone. The previous 4.429% real-width result remains unchanged.

## Audit of prior draws

Reproduce the previous Poisson panel, seeds6311000–6311049, with all seven
original methods (known Poisson, discovery, five competitors), settings,
1000-row development/test sizes and seed+50000 independent test stream from
`study_discovery_calibration.py`. Record endpoint arrays, point predictions,
actual fit/calibration row identities and source/native/data hashes. Compare
raw summary and prediction digests with the archived fixed records; retain
any mismatch and explain it before attributing a change to support metrics.
This audit is retrospective and cannot establish prospective improvement.

## Calibration scores

Absolute scores remain the existing default. Add opt-in split scores:

- poisson_normalized: |y-mu|/sqrt(max(mu,1e-12)); query band
  mu +/- q*sqrt(max(mu,1e-12)). The floor is fixed before measurement.
- poisson_quantile: max(Q_.05(mu)-y, y-Q_.95(mu)) at level .90;
  query band [Q_.05(mu)-q, Q_.95(mu)+q]. General levels use alpha/2 and
  1-alpha/2. Reuse core Poisson quantiles and conformal correction routines.

Both are fixed score functions applied to a mean learned without calibration
responses. They retain the exchangeable-row marginal conformal guarantee
whether or not the Poisson variance model is correct; conditional coverage
and efficiency do not follow from this guarantee. Require nonnegative integer
calibration observations and finite nonnegative predicted means. Support
intersection is evaluator-only, leaving public real endpoints unchanged.
Keep numerics in Rust core and expose the same score options through thin
Python and R bindings. Defaults, CV+ and CQR remain unchanged; reject these
score options outside standard-model split calibration explicitly.

## Fresh draws and information tracks

Development seeds6361000–6361019 and confirmation seeds6362000–6362049 are
disjoint from prior work. Confirmation settings and reporting are frozen now;
no confirmation result changes a score, stopping rule or method. Each panel
has independent uniform[-1,1] x1,x2 and Poisson response with mean
a*exp(.3+.6*x1-.4*x2), a=0.2 (low), 1 (original) or 10 (high).
Use 1000 development rows, 1000 independent test rows at seed+50000, and the
existing Rust two-fold partition, first held-out fold reserved for calibration.
Panels share streams and must not be pooled for uncertainty.

Known-specification track: Poisson with `bols(x1)+bols(x2)`, mstop100, nu.1.
Data-only track: `discover` on only the same proper-training rows, ordinary
defaults, no supplied family/forms. Keep each selected mean fit fixed across
the three calibration scores and record its family/formula/mstop. No test or
calibration response enters discovery, tuning, imputation or model choice.
Record an unsupported selected model or negative mean as a failure; do not
substitute a supplied Poisson fit. Selection cost is shared and recorded once;
each score's calibration/prediction cost is recorded separately.

Five data-only controls are unchanged: CatBoost split/CQR (500 trees, depth6,
rate.05), LightGBM split/CQR (300 trees,15 leaves,rate.05), sklearn gradient
boosting CQR (300 trees,depth3,rate.05). Every method uses the same actual
proper-training/calibration partition; one thread, untuned settings.

## Reporting and decision

Retain all methods, failures, warnings, source hashes, inputs, outputs and
partitions. Summarize means and Monte Carlo SE across independent draws,
paired with the absolute-score reference from the same track. Show every
method's headline scores in an opening annotated GT table. A comparator is
eligible for count-size leadership only when its mean coverage is within .02
of .90; do not relax this for discreteness. If none is eligible, report no
leader. Select the lowest mean count-size eligible competitor retrospectively
and explicitly state the resulting SE does not adjust for winner selection.
Report paired percentage count-size excess and real-width excess separately.
Target <=1% eligible excess for both information tracks; a point estimate
and MC SE do not establish a universal population bound. Any unresolved gap
or regression remains an issue even when a new option is useful.

Development is diagnostic, not a selection stage that picks the best score
for confirmation. All three scores and all panels are confirmed. Unit and
integration tests must verify independent numerical references, invalid
inputs, zero/extreme means, ties, support boundaries and empty sets,
internal/external split equivalence, fixed mstop prefixes, and unchanged
mean predictions/coefficients after calibration-response mutation. R and
Python use different internal RNGs but must agree on supplied calibration
rows. No full-book PDF or universal competitor claim is required.

References: Romano, Patterson and Candes (2019),
https://arxiv.org/abs/1905.03222 (CQR and locally adaptive scores).
