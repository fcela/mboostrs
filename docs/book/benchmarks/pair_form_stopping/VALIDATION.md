# Validation record

The correction is proposed in [MR #537](https://github.com/fcela/mboostrs/pull/537)
for [issue #535](https://github.com/fcela/mboostrs/issues/535). The separate
candidate-cap limitation is [issue #536](https://github.com/fcela/mboostrs/issues/536).

- Full Python suite: `pytest -m 'not r_parity' -q` — 7,016 passed,
  14 skipped, 407 deselected, one expected failure, 477.90 seconds.
- After merging main's paired-battery changes: replay, artifact-integrity,
  paired-battery and book-navigation checks — 36 passed. This includes the
  added structured rolling-fold judge test and the supplemental strongest-tier
  assessment. The merged changes affect the paired benchmark and its tests,
  with no additional library change.
- Interaction, picker-budget, adoption, spatial and joint-tree regression
  checks — 39 passed before the merge.
- Ruff passes for the two changed library modules, the benchmark runner and
  reporter, and their regression tests.
- `tools/check_scorecard.py` passes. `tools/check_benchmark_provenance.py`
  passes its existing ratchet (47 artifacts, 84 previously recorded missing
  fields, no increase). The new benchmark separately verifies its frozen
  protocol and every input/partition identity.
- The new notebook executes with the repository environment. All four tables
  produce HTML display outputs and no cell errors. Its committed sources have
  empty outputs and compile successfully.

The before and after artifacts have identical runner hashes; their loaded
package hashes differ as intended. The after package hash matches the final
library source and loaded extension. The assessment records reporter and
artifact hashes separately. The later main merge did not change either
library module or the native extension.

The global book validators still report pre-existing stored outputs throughout
the book and the existing printed dataframe in
`case_studies/kdd_scorecard_expansion.ipynb`. Those sources are unchanged;
the broader documentation refresh is tracked in
[issue #304](https://github.com/fcela/mboostrs/issues/304). The new notebook
has no stored outputs, syntax errors or printed dataframe cells.

No Rust numerical routine changed, so this correction did not require a
native rebuild. R parity was not run: the changed path is Python autospec's
coordination of existing Rust-backed fits.

All three regression task means meet the 5% target against both the frozen
small-tier reference and, in the explicitly supplemental comparison, the
strongest complete recorded competitor across small and tuned tiers. The
broader real-data, causal, distributional, interval and workflow targets remain
unverified by this study. These known generators are development regression
evidence, not independent confirmation on held-back task families.
