# Response-unit diagnostic validation

The protocol was committed at 2edc1f3 and the independent runner at e139635
before measurement. Baseline production source is main 64a13b1, including
the predictor-coding correction from MR #654. Both retained phases identify
their loaded native build separately from their Python source hashes.
The production implementation is committed at 024bf18; final docstrings
were clarified after the measured corrected phase. No numeric Rust or R
source changed. The native source digest matches the current Rust tree;
NATIVE_IDENTITY.json records the release binary identity and digest.

All 132 prespecified cases are retained: 112 fixed, 8 supported tuned, 12 named
refusals. All 120 supported corrected cases agree with the independently
constructed original-response reference within 1e-10; the maximum discrepancy
is 1.3322676295501878e-15. The 24 baseline exceptions and 76 completed cases
outside the original-response contract remain in the baseline archive.
Baseline discrepancies include a change in reporting units. A separate
affine endpoint conversion isolates the actual calibration-coordinate error,
with a maximum 0.15272434075736374 in original-response units.

Validation recomputes endpoint, point and correction discrepancies from all
arrays; verifies matching inputs, reference records and native identities;
checks every supported inner-CV risk curve within 1e-12 and verifies each
curve's exact positive-grid argmin. Twelve center/scale tuning calls preserve
the explicit fold-dependent-scale refusal. The broader #652 issue remains
open for that separate loss contract.

The archived generic helper's uncertainty metadata is unsuitable for these
deterministic diagnostics (#655). The protocol, notebook and validation.json
explicitly correct its interpretation: one shared fixture, no independent
sampling replicates, no Monte Carlo standard errors, no coverage experiment
and no controlled speed comparison. Frozen raw evidence is preserved.

## Checks

- 360 scoped Python tests passed (35 routine warnings), including 73 new
  response-unit tests and the existing conformal, predictor-coding, LSS
  response-scale, response-transform and refit suites. No full Python suite
  or R parity rerun was performed.
- Ruff passed on the changed implementation, runner, summarizer and tests.
- Notebook execution passed with the repository Python 3.12 interpreter and
  matching worktree release extension: 6 code cells and 5 rich HTML GT tables.
- Focused MyST site build passed with --execute --strict --ci. All 4 inline
  expressions resolved to measured values and all 5 table payloads rendered.
- Book cell and source checks passed: 184 ToC notebooks and 2295 Rust items.
  Runtime fit-contract validation passed. The notebook bundle includes all
  declared evidence, with byte agreement checked against the source files.

These checks establish numerical coordinates and supported API behavior.
They do not establish a gain in predictive coverage or competitor rankings.
