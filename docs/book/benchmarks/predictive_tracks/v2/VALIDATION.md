# Predictive-track validation

## Frozen outcomes and sources

The corrected instrument is `predictive-tracks/2`. Its full protocol and
score-anchor rules were frozen before collecting the valid baseline.
Both phases contain every planned method/draw: 3 processes × 8 independent
seeds × 7 methods = **168 successful jobs per phase**. No failed draw is
omitted. Both information tracks use the same development/test rows; every
method's input digest agrees within a draw and across phases.

The benchmark runner remains unchanged across phases. The baseline Python
source is 5cbd29a (library content from main 056d475), loaded package/native
hash `ea4c640bf3913fd08894646ea879c55d070571134fad6ccd981b0175df444b2a`.
The measured after source is f290194, hash
`9616241c3fea249974330b3af19b55798cfe70f6aa96e8c780f5469a3886fd8c`.
The separate `baseline_library.json` and `after_library.json` distinguish the
imported package from the runner checkout. Later report/runner-only commits
leave the measured library unchanged. The release extension was built from
056d475 for Python 3.12; the native wheel hash and build flags are retained
in `../native_build.json`. No Rust routine changed in the repairs.

The first version's supplied factor formula omitted its reference-level
curve. Its complete protocol, source commit and all 168 raw outcomes remain
retained outside this v2 directory. That known-specification result is invalid
instrument evidence, not a library estimation failure. A separate noiseless
representation regression fails on it and passes on the corrected formula.
No version-1 outcome is pooled into the corrected baseline or its paired rerun.

## Paired findings and replay

The strongest complete named competitor is CatBoost on all three processes,
chosen retrospectively by mean test MSE. Its source settings and predictions
are unchanged across tracks/phases. Discovery's factor gap changes from
+75.14% (SE 4.14 points) to −8.23% (SE 2.25). Mean MSE falls from 2.01692 to
1.05660; paired change −0.96032, SE 0.02317 squared response units, or
−47.58%, SE 0.73 points. All eight factor draws improve. All six after
track/process cells meet the 1% point-gap target in this study.

All 120 competitor/Ridge prediction digests and all 40 unaffected mboostrs
digests replay exactly. `paired_improvement.json` retains every track/process
change with its paired SE. The fresh control gate is separate: seven families,
six previously unseen draws each, all 84 phase/draw jobs successful, every
mean-loss ratio ≤1.01, numeric predictions exact, no increased null adoptions,
and nonconstant reference curves in both opposed-label families. Its initial
and version-isolated captures agree exactly on all 84 inputs/predictions/MSEs.
They are repeated captures, not additional independent confirmation draws.

The original controls also retain the three reporting failures fixed by the
separate schema repair (#564 / MR #567). That original confirmation gate did
not pass. The schema-only replay restores all three failures and leaves all
39 previously successful predictions unchanged. Dictionary completion is
issue #560 / MR #568; each product fix has a separate branch and MR.

## Independent fold audit and checks

`capture_predictive_folds.py` independently repeats all 48 mboostrs method/draw
fits in each phase and observes actual Python resampling returns. Both 48-job
audits complete successfully and match every frozen input digest, prediction
digest and MSE. Their nonempty per-stage fold fingerprints, source/dependency
identities and runner hashes are retained. Rust-internal screening half-splits
remain specified by frozen source and seeds, rather than claimed as observed
Python returns. The audit selects no setting and adds no scorecard grade.

- 31 study-evidence, known-specification representation, navigation and
  benchmark-contract checks pass, including missing/non-finite results,
  incomplete competitors, changed data/protocol and changed peer predictions.
- The schema repair's full non-R-parity suite passes 7,037 tests; the
  dictionary stack's measured-library full suite passes **7,048**, with
  15 skipped, 407 deselected and one expected failure. Separate focused
  discovery, schema, replay and confirmation checks also pass. Their detailed
  counts live in the respective factor-reference validation records.
- Ruff passes on affected runner/report/audit/test files. The canonical
  scorecard consistency tool verifies all twelve dimension definitions,
  template hashes and links. No aggregate is computed; only D11 is measured.
- The notebook is executed with the repository Python environment and stores
  its HTML great_tables outputs as required by BENCHMARK_GUIDANCE.md. Its
  source cells, local links, complete outcome counts, scorecard profiles,
  provenance hashes and rendered-table outputs are checked independently.

No R parity or full PDF build was run for these Python orchestration/metadata
repairs. The global book source check has separate existing defects outside
this study; it must not be reported as a clean full-book build. Issue #569
aligns the checker with the guide's retained outputs only for registered
benchmark pages, keeping ordinary and orphan output checks intact.

## Interpretation limits

All means and SEs refer to independent synthetic draws at the stated sample
size; the selected-competitor SE treats that winner as fixed. Competitor grids
are finite and search designs differ, so D11 levels 4/5 are unavailable. The
frozen rule assigns D11=3 separately in each cell for improvement over Ridge;
that is distinct from meeting the 1% competitor target. Eleven other scorecard
dimensions remain unmeasured. Timings are descriptive under shared load.
These focused results do not complete the broader benchmark goal.

Final scoped rendering/source audit: six stored HTML tables render with no cell
errors, and execution leaves every code/markdown source unchanged. All local
links resolve. All 96 fold-audit fits exactly replay their original outcomes,
and the currently loaded library matches the recorded after package hash.
`validation_checks.json` retains these checks and counts. The new checker policy
has 51 passing documentation/example tests. On the same dependency book input,
it removes only 64 registered-benchmark output errors (1,064 → 1,000) and adds
none; adding this study introduces no further global source error. The global
cell checker has zero syntax errors but still flags one existing printed-frame
cell in `case_studies/kdd_scorecard_expansion.ipynb`. Those broader source
checks are not reported as a clean full-book pass. The provenance ratchet
passes with no artifact regression, and the canonical scorecard check passes.
