# Composed-schema repair validation

Issue #564 is a reporting/schema defect, separate from #560's factor dictionary.
The parent main source is 056d475; the measured schema-only library source is
9edd203 (repair bf6013b). The release Python 3.12 extension was built from
056d475; there are no Rust changes or numerical fitting/prediction changes.
The raw phase artifacts retain actual loaded package/native identities.

The frozen original controls contain 42 planned draws. The parent completed
39 and crashed on logged replicates 0,1,3 when sensitivity requested internal
`log__x`. The repaired package completes all 42. Every one of the 39 previously
successful input/prediction digests and MSE values replays exactly. Recovered
rows retain their missing before outcomes; no successful-only mean or numeric
before/after gain is manufactured. `schema_replay.json` records every pair,
all three restored failures, and source/runner hashes.

The original factor-reference non-regression gate did not pass because three
before jobs failed. This schema replay does not promote that separate policy.
Its independent version-2 confirmation uses fresh seeds and the schema-repaired
parent; those results live under `factor_reference/v2/` with the dictionary fix.

Validation:

- Four new regressions fail on main and pass after the repair: raw-input
  sensitivity for transformed factor-by terms, restored models, shared composed
  sources, and cyclic malformed recipe metadata.
- 53 focused explore/discovery/schema tests pass; 15 LSS sensitivity tests pass.
- Four replay-evidence tests pass, including omitted controls, changed data,
  changed predictions and separation of restored failures from numeric gains.
- Full non-R-parity Python suite: **7,037 passed, 15 skipped, 407 deselected,
  one expected failure**, 658.75 seconds. Warnings were retained in the local
  test log; no R parity or full PDF build was run for this metadata repair.
- Ruff and diff checks pass on affected source, tests and replay reporters.

Reproduce on the repository environment with the native build matched to the
source. From the control-runner checkout, import the parent library for
`python -m benches.bench_factor_reference --phase before` and the fixed library
for `--phase after` using distinct `PYTHONPATH` values. Then run
`python -m benches.report_factor_schema_replay`. The original runner remains
versioned and unchanged by the subsequent dictionary control runner.
