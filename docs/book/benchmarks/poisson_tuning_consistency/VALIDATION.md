# Prospective tuning-study validation

The protocol/design freeze is 5042d81. Baseline instrument 2068228 retains the
unchanged discovery implementation; correction/confirmation freeze 5f72afe
aligns both stopping and cost preflight with the existing fitted rate.
Development completed before confirmation. No development or confirmation
result selected a new rate, score, stopping threshold or representation.
No Poisson gradient, Gaussian spread rule or calibration default changed.

## Evidence

Development has 850 complete records. Confirmation has 4250 records on 250 new
independent draws: 50 in each of five separately summarized regimes. All 17
methods per draw, both information tracks, five fixed controls, exact input
hashes/partitions, test truth, prediction arrays, mean-tuning curves, model
choices, source/native identities and warnings are retained. Six score rows
are unsupported: discovery selects an LSS model on overdispersed seed 6494126
in each implementation; each appears as three failed standard-count-score
rows. No successful-only estimate or replacement is reported for those tracks.

LightGBM CQR was prespecified, not selected on confirmation. The target remains
at most 1% paired real-width excess, with point-target and one-sided 95% upper
limit passage distinguished and coverage eligibility fixed at [0.88, 0.92].
Additional bounds cover every eligible fixed control using a Bonferroni
adjustment across five comparators for each specified method/regime.

On the high-rate true-Poisson panel, default discovery width excess changes
from 23.47073% to -6.34448% (MC SE 0.49647 points, upper -5.51212%), corrected
coverage 0.90064; true-mean MSE changes 14.36470 to 0.27162. Corrected stopping
has zero one-update fits versus 24/50 before. Both real width and count size
pass the bounded comparison here. Original-rate default width excess 3.34882%
(upper 6.06945%) fails. Normalized original-rate options do not change the
default-policy acceptance in #636. The primary low-rate reference is outside
coverage eligibility. Zero inflation retains an efficiency gap, filed as #660.
Central quantile conservatism and #650 remain visible.

The sharp-mean development inventory has 500 successful records:ten draws for
each of ten processes and five instrument rows per draw. Its initial 85-record
attempt, including 17 oracle API-keyword failures, is retained separately.
Successful initial model arrays reproduce within 1e-10 after correcting only
the oracle call. False scale adoptions are 3/10 and 2/10 on axis/shifted constant
controls, and 0/10 on both sharp varying counterparts; smooth varying controls
adopt 10/10. These development controls neither change a rule nor confirm an
efficiency target. Their held-back confirmation inputs remain untouched.

## Integrity and checks

All ten raw archives use XZ envelopes. Decompression reproduces the measured
JSONL bytes exactly; unchanged phase metadata and manifests retain the original
gzip digest and both published/uncompressed digests. The conversion changes
no number, record, failure or interpretation. Independent rescoring verifies
5562 prediction records with maximum discrepancy 0.0, count-support coverage
identity, 100 diagnostic-only rows and all 23 retained failed rows.

The final discovery/calibration/instrument test run passes 369 cases with 6
routine warnings. Ruff passes on all changed implementation, runner, reporting
and test files. Source/cell checks cover 185 ToC notebooks and 2295 Rust items;
runtime probes pass with the isolated source-matched release extension.
The executed notebook has 9 code cells and 7 GT tables, with all measured prose
expressions verified by a focused executed strict MyST build. Its download
bundle includes every declared archive, protocol, summary and validation file.

No full Python suite or R parity suite was rerun for this Python argument-
consistency correction; native numerical source and the shared binary are
unchanged. The separate #653 decision MR checks Python/Rust selection and
R mboost 2.9.11 semantics without changing any archived curve or tie threshold.
Workflow costs are descriptive: panel jobs ran concurrently, with each model
restricted to one thread. No controlled speed ratio is claimed.
