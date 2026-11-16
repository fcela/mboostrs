# Validation record

The numerical rerun uses the committed harness at `ebe1e45` and the
repository-local CPython 3.12 environment. The release extension was rebuilt
against main before running; results record its build/source identity separately
from the benchmark source revision. The final merge revision also restores
Python adoption-reporting metadata from an existing comparison; it changes no
Rust engine, boosting numerics or adoption decision. The numerical ablation does
not call autospec, and its frozen evidence is retained unchanged.

## Focused checks

The stopping, fold-scheme, metric-tuning, resume and stopping-report tests passed:
110 passed, one artifact test skipped before the new run existed. The final
run after recording and compression passed all 111 tests, including
the completed-artifact check; no focused test was skipped. The checks
cover matched nonextended risk curves, iteration-zero handling and null-response
ties, complete fold-local curves, whole-time-unit forecast origins, preserved
judge masks, grouped folds, and artifact-derived summaries with harmful seeds.

Rust workspace checks exclude `mboostrs-r` as CI does on macOS: 783 tests and
doctests passed, with no failures. `cargo fmt --check --all`,
`cargo clippy --workspace -- -D warnings`, `ruff check python/` and Ruff checks
on the changed benchmark scripts/tests passed.

## Initial broader Python suite

`pytest -m "not r_parity"` completed with 6,993 passed, 15 skipped,
407 deselected, one expected failure and four failures. The four failures
also reproduce together in isolation (four failed in 1.39 seconds):

- `tests/test_adoption_bar.py::test_adopted_pairs_carry_adoption_evidence`
  finds an adopted tensor with no adoption evidence.
- `tests/test_policy_ladder.py::test_pair_screen_reuses_the_actual_short_incumbent_path`
  expects one lead but receives a tensor and a joint tree.
- `tests/test_policy_ladder.py::test_pair_prescan_does_not_reintroduce_columns_absent_from_incumbent`
  expects one pair record but receives both forms of that pair.
- `tests/test_residual_tree_reporting.py::test_adopted_pair_is_listed_with_its_structure_in_the_report`
  does not find the expected tensor-structure text in the report.

At the numerical-run revision, `git diff origin/main -- python/mboostrs tests/test_adoption_bar.py
 tests/test_policy_ladder.py tests/test_residual_tree_reporting.py` is empty.
The package code and these tests matched main (`3451c0a`); the failures were
pre-existing validation limitations. They were subsequently addressed for the
authorized merge, as described below. No R parity run is claimed.

## Recorded evidence

All 25 jobs succeeded, covering 200 configurations and 1,600 rule/configuration
records. The artifact test verifies unchanged source, development/assessment
frames and partition fingerprints against v1, matched nonextended curves,
forecasting origins, exact generated summaries/report and the SHA-256 of the
uncompressed JSON. Gzip publication changes storage only. The original v1 result
file and the archived v1 report remain byte-identical to their original versions.
The annotated GT headline table renders HTML with units, median/range cells,
spanners, bold best values and explanatory notes.

Memory measurements are per-job `peak_rss_bytes` (whole-process peak resident
set size in bytes). The numerical file retains the generic evidence helper's
top-level `memory.status = "not measured"` placeholder; the measured values live
in the job records. Parent-runner and individual-rule peak memory were not
measured, and no per-rule memory comparison is claimed. The numerical source was
committed; the stored dirty-tree flag also reflects pre-existing untracked local
files, which were preserved throughout this work.

## Adoption-reporting fixes

The competing tensor/joint-tree form picker already compared held-out risk
against the additive incumbent but discarded the gain and paired-SE statistic
when recording the adopted term. The merge revision retains those results, the
fold count and the existing 2-SE threshold in `adoption_evidence`; the report
therefore shows the evidence used for that adoption. No comparison is rerun and
no fitting or adoption decision changes. The planted regression exercises both
tensor and joint-tree winners.

The two stale policy-ladder assertions now check the two intended forms for the
same incumbent columns and verify that both reuse one residual screen. The
structure-report assertion checks the main effects that the form picker actually
retains. Earlier initial-suite failures remain above as the validation history.

## Final merge checks

After merging main at `11d58e3` and fixing the adoption-reporting failures, the
full non-R-parity suite passed: 7,006 passed, 14 skipped, 407 deselected and one
expected failure, with no failures (479.50 seconds). The 64 focused adoption,
policy-ladder, residual-tree reporting, joint-tree discovery and stopping-artifact
checks also passed, including the added joint-tree-winner parameterization.
The initial four failures are resolved.

`cargo clippy --workspace --exclude mboostrs-r --all-targets -- -D warnings`,
`cargo fmt --check --all`, Python Ruff and changed-file Ruff passed. The prior
783 passing Rust tests/doctests remain applicable: no Rust source changed.
The release extension and numerical evidence are unchanged; no R parity run
is claimed for this Python reporting correction.
