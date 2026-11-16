# Pair-form stopping replay

Protocol frozen on 2026-10-14 before correcting issue #535. This is an
exploratory regression study on known development generators. It cannot
establish independent confirmation of the discovery policy.

The battery reuses all five frozen draws (seeds 0–4) of joint_boundary,
smooth_sparse and correlated_proxies from the competitive shootout, including
its disjoint training, validation, calibration and test partitions. Autospec
receives the training and validation rows, Gaussian response semantics,
a budget of 32 fits and one supplied validation fold, mstop=200, nu=0.1,
interactions="adopt", residual_tree="auto". It receives no formula or truth.
Test outcomes never select settings. Before and after phases differ only in
the library correction; the runner and inputs are unchanged. Jobs run serially
in fresh processes with one thread and a 240-second ceiling. Keep every
failure and require all five draws before reporting a task as complete.

The primary metric is test mean squared error (MSE), lower is better. Report
paired after/before relative changes, all five values and their range. For
context, compare task means with every complete non-mboostrs method in the
original small-tier shootout. Preserve its artifact hash and input/partition
digests; abort on mismatches. Those comparator measurements are historical
and their runtimes must not be used for a fresh implementation speed ratio.
The best comparator selected by test MSE is a retrospective oracle comparison.
The engineering target remains at most 5% excess loss on every task, not
merely an average. Report failures and incomplete tasks as unresolved.

Record formula, selected stopping iteration, full aggregate validation path,
pair-form decision, fit count, elapsed workflow seconds, predictions digest,
split-conformal 90% prediction coverage and width. Calibration rows determine
the conformal radius; they do not choose the formula or stopping iteration.
With five draws, coverage is descriptive and does not establish a calibration
guarantee. Independent draws support Monte Carlo spread; within each draw,
pair phases and comparators on exactly the same rows.

The scorecard is mboostrs-scorecard/2.1, with its content hash recorded by the
runner. This study measures predictive validity (D11) and auditability (D10).
Anchor D11 at 3 only when the complete outer draws improve over the
prespecified Ridge baseline; these three Gaussian generators and unequal
search budgets cannot support level 4. Anchor D10 at 5 when code, environment,
data, partition and output hashes accompany the reproducible run. Other
dimensions remain unmeasured: in particular, a pair's structural split use
does not establish relationship recovery, and predictive interval coverage
alone does not establish importance uncertainty (D4). Report all twelve
dimensions with evidence and no weighted aggregate.

Reproduction from the repository root:

```bash
.venv/bin/python -m benches.bench_pair_form_stopping --phase before
# Apply the library fix; preserve before.json.
.venv/bin/python -m benches.bench_pair_form_stopping --phase after
```

Each phase records its loaded Python and native extension hashes independently
of the source commit. The before artifact remains frozen when rerunning after.
