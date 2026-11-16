# Validation and review evidence

## Protocol and source

Primary protocol interaction-capacity/1 and its runner were committed before
outcomes in1b676eb. The main-effect recovery confirmation was separately frozen
inb6ceedf before its new draws. The all-named-reference audit is explicitly
posthoc and retains the original four-tree comparison and D11 rules.

Each raw phase records exact runner, protocol, guide and scorecard hashes,
package/native identity, dependency versions, thread settings, settings,
inputs and prediction digests. The release extensions were rebuilt for the
repository Python3.12 with maturin and native CPU instructions in isolated
worktrees. The scorer fix changes the native binding; no statistical numerical
routine was added to Python. Native/source identities are distinct in every
*_library.json. An initial system-Python provenance observer is marked excluded;
it ran no fit. Corrected observers match every actual job in each phase.

## Experiment checks

- Baseline primary:272/280 successful jobs; all8 known-null calls raise the
  recorded empty-x_data exception. Failures remain in the denominator.
- Tree/scorer repair:280/280 successful primary jobs.
- Main-recovery repair:280/280 successful primary jobs.
- Every phase uses the identical primary code, rows, seeds, competitors and
  settings. All200 competitor prediction digests and32 previously successful
  known-specification predictions replay exactly against baseline.
- Both32-job smooth-dictionary diagnostics complete and replay exactly.
- First independent policy gate:36 baseline/36 candidate fits complete; each
  family after/before mean MSE ratio<=1.01. Null interaction adoption stays0/6.
  Final tree/scorer integration exactly replays all36 candidate predictions.
- Second independent policy gate:36 new before/36 candidate fits complete;
  all family ratios<=1.01 and no individual MSE regression. Null interaction
  adoption stays0/6. Final integration exactly replays all36 predictions.
- No individual primary MSE increases from baseline to final, nor from the
  intermediate tree/scorer phase to the final recovery phase. Full per-draw
  changes and optional-stage errors remain in the raw records and audit files.
- All10 final process/track point gaps are<=1%, including the audit admitting
  Ridge. Winner-conditional paired SEs do not establish a population guarantee.

The primary outcomes were used to diagnose mechanisms; repaired primary phases
are development reruns, not independent competitive confirmation. The two
newly frozen finite control batches support the specific proposed rules, not
broader structure recovery, false-discovery control or interval calibration.
Elapsed times share machine load with other suites; D5 and memory are unmeasured.

## Software checks

- Tree-form regression: fails on parent; its five public constraint controls
  pass there. All349 discovery tests pass with the first repair.
- Main-effect recovery regression: fails on parent. All350 discovery tests
  pass with recovery; the new confirmation gate has27 exact and9 improved
  predictions across36 draws.
- Offset-only response scorer: all3 weighted Gaussian/Binomial/Poisson tests
  fail on the old native extension. All45 scorer/tuning/continuation tests
  pass after rebuilding. Rust formatting and CI all-targets Clippy pass.
- Guide repair: all115 guide/Poisson tests pass; size and sigma guard tests are
  unchanged. The complete CI Ruff command and scorecard consistency check pass.
- Final integrated full Python suite:7164 passed,15 skipped,409 R-parity
  deselected,1 expected failure; five pre-existing documentation checks fail.
  Those same five reproduce on main plus the guide-only repair (29 pass/5
  fail), and are tracked in#619: removed output-strip API/stale orphan-output
  expectations, and an external scorecard artifact declaration rejected by
  notebook ZIP containment. No statistical regression is attributed to these.
- The final guide also documents both new workflow rules and the tree-refit
  limitation in8969 characters; all112 guide tests pass after that prose update.
- Notebook:16 code cells executed without errors,17 HTML GT tables and1 SVG
  figure. Longest code cell21 lines. The explicit CatBoost refit exactly
  reproduces the measured job's prediction digest.

## Review dependencies

No GitHub MR was merged by this work. Review sequence:

1. #614 / #613: pair-tree form comparison.
2. #618 / #616: offset-only response scoring, based on#614.
3. #617 / #615: guide and lint gates, based on#618.
4. #621 / #620: conditional main-effect recovery, based on#617.
5. Benchmark notebook MR, based on#621.

Review branches contain identical statistical source to the measured final
integration; merge-only commits organize dependencies. Raw measured source
revisions remain reachable. Each fix has its own MR. Rebase dependent MRs to
main when their bases merge; do not merge without user authorization.

## Reproduction

Use the source revision in the desired phase's library identity, build the
matching release extension, and run bench_interaction_capacity with phase
baseline, after or recovered, suite primary. The primary runner also accepts
diagnostic and controls for their frozen v1 draws. Run bench_mean_recovery_controls
with before, candidate or final for the separate new controls. Use separate
phase/cache directories; never relabel a result from a different source.

Run report_interaction_capacity for each primary phase and
`python -m benches.audit_interaction_capacity --final-phase recovered` for the
complete paired audit. The optional all-named reference audit is specified in
REFERENCE_AUDIT.md. ARCHIVES.json records lossless gzip content hashes; reporters
and the notebook read either plain JSON or its .json.gz archive. Execute the
registered notebook with the repository Jupyter interpreter and retained
outputs. Scripts run from repository root with PYTHONPATH=python:.
