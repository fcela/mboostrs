# Competitive predictive refresh

Instrument `competitive-refresh/1`, frozen before running on 2026-10-14.
This refresh measures the current autospec pipeline on every task in the
original competitive shootout, after the pair-form replay fix in MR #537.
It preserves all original results and row identities. This is development
evidence on known tasks, not independent confirmation on held-back families.

The battery consists of ten real-data panels with three paired outer splits
each and three generated panels with five independent draws each: 45 jobs per
search tier. The small tier uses autospec budget 32 and the tuned tier budget
128. Both use the original runner's Gaussian/Binomial response semantics,
nu=0.1, initial mstop=200, one supplied training/validation fold,
interactions="adopt" and residual_tree="auto". Autospec receives no formula or
truth. Training, validation, calibration and test partitions, row limits,
preprocessing, explanations and scoring are the original runner's own code.
Run tiers separately, recording a complete manifest for each before fitting.
An unexecuted tier is explicitly unmeasured, never inferred from another tier.

Run jobs serially in fresh processes, with one thread and a 240-second timeout
per job. Preserve every failure and do not restart a job solely because an
observation request expires. Cache reuse requires the runner, environment,
loaded package, extension and job identity to match. Each output is separate
from the original shootout artifact. Do not edit library code during a run.

The primary metric is mean squared error for regression and log loss for
binary outcomes, lower is better. Report per-task means, descriptive split
standard deviations and each paired draw. Require every planned draw for a
task to pass its 5% gate. Compare with every complete non-mboostrs method from
the original study, retaining both small and tuned searches. For the overall
gap, use the best complete method/tier on each task. That selection uses test
loss retrospectively and is an oracle comparison, not a prospective baseline.
Verify input and partition digests against all comparator rows. Preserve the
original artifact content hash and never relabel historical timings as fresh
implementation speed ratios. Different search budgets compare pipelines.

The target remains at most 5% excess loss on every task, with incomplete tasks
unresolved. Report results and denominator separately for each current tier;
do not choose a tier per task using its test error. Preserve the regression
and binary outcome families, IID, temporal and spatial splits, and all known
failures in the report. These predictive panels do not cover the full causal,
distributional, uncertainty or analyst-workflow requirements of the active goal.

Use scorecard `mboostrs-scorecard/2.1` and record its SHA256. The instrument
measures D11 predictive validity and D10 auditability. Report the remaining
ten dimensions as unmeasured and no weighted aggregate. Assign task-level
D11=3 when all outer draws complete and mean loss improves over the
prespecified complete Ridge/logistic reference; assign D11=1 when it fails to
beat that reference. An incomplete task remains unmeasured. This separates a
simple-reference score from the stronger 5% competitor gate. Known tasks,
unequal tuning budgets and historical comparators do not support D11=4 or 5.
Assign instrument-level D10=5 only when the artifacts retain code/environment,
loaded binary, data, partitions and predictions hashes and the reporter can
reproduce the assessment. Numerical results do not grade explanation quality,
causal support or analyst usability.

Reproduction from the repository environment:

```bash
.venv/bin/python -m benches.refresh_competitive --tier small
.venv/bin/python -m benches.refresh_competitive --tier tuned
```

Subsequent diagnosis and fixes require separate issue and MR records. Freeze
new confirmation draws and settings before promoting a policy change; retain
the refresh as the pre-fix comparison instead of overwriting its measurements.
