# Predictive refresh validation

The frozen `competitive-refresh/1` small-tier run accounts for all 45 planned
jobs: 37 successes, six housing exceptions and two superconductivity timeouts.
All rows retain the same loaded-package identity as the frozen source at
6c4c194. The reporter verifies input and partition digests against the original
720-job comparison artifact, whose SHA256 is unchanged.

Four of thirteen task means meet the 5% target against the strongest complete
recorded non-mboostrs method across small and tuned tiers: credit,
smooth_sparse, joint_boundary and correlated_proxies. The housing and
superconductivity tasks remain unresolved in this baseline. Complete-panel
excess losses include bike_future 49.85%, bike_iid 44.00%, diamonds 20.50%,
protein 41.45%, concrete 57.74% and cancer 127.41%. These are descriptive
development results on known tasks, not independent confirmation or a general
library ranking.

## Reporter checks

Regression checks exercise the strongest complete tier, simple-reference D11
grades, missing and duplicate rows, input/partition mismatches, non-finite
primary losses, and failed draws whose successful subset looks competitive.
The benchmark, navigation and earlier replay-artifact checks pass: 24 tests.
Affected-file Ruff and whitespace checks pass. Scorecard consistency and the
existing benchmark-provenance ratchet pass. The new study additionally pins
its frozen protocol, scorecard, comparator and loaded-package identities.

The book notebook executes in the repository environment, producing five HTML
tables and no cell errors. Its committed sources have empty outputs. The
opening table compares every recorded method and both historical search tiers
with the current fixed tier. Per-task grades and failure accounting remain
visible. No aggregate score is assigned to the incomplete twelve-dimension
profile, and task-level D11 grades are not averaged.

## Follow-up evidence

The separate repair for [issue #538](https://github.com/fcela/mboostrs/issues/538)
is [MR #540](https://github.com/fcela/mboostrs/pull/540). Its isolated six-job
rerun resolves the housing exceptions and reports remaining predictive gaps;
those corrected rows do not replace this baseline's failure records.

The joint-replacement diagnostic for
[issue #539](https://github.com/fcela/mboostrs/issues/539) uses the same three
outer splits of protein, bike_iid and diamonds. Validation chooses between
depths 3 and 6, each with nu=0.05 and a 1,200-update cap, before test scoring.
Its portable repository runner reproduces all nine diagnostic predictions
exactly. Mean competitor-relative gaps become 8.94%, 7.19% and −0.66%,
respectively. This is an exploratory capacity counterfactual, not an adopted
policy or confirmation on new task families. Some paths select the cap.

The cancer failure is tracked separately in
[issue #541](https://github.com/fcela/mboostrs/issues/541). A diagnostic
all-linear specification improves mean outer log loss but has worse selected
validation loss on every draw. Merely adding that candidate under the same
minimum-validation criterion is therefore not an established fix; its
selection, calibration and structural mechanisms remain to be isolated.

No library code changes in this refresh branch beyond parent MR #537. The
original protocol, comparator results and row identities remain intact. No
current tuned-tier result is inferred from this small-tier run. Causal,
distributional, interval and analyst-workflow targets remain outside this
predictive battery, so the broader goal is still active.
