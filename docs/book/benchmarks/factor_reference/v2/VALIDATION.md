# Dictionary-repair validation

Issue #560 is fixed by completing the categorical interaction dictionary,
with the separate composed-schema repair (#564, MR #567) as its parent.
The measured Python package at f290194 has hash
`9616241c3fea249974330b3af19b55798cfe70f6aa96e8c780f5469a3886fd8c`.
Subsequent runner/report-only commits leave that loaded package unchanged.
The native release extension is the matching main-056d475 build; no Rust
routine or statistical arithmetic changed.

## Fresh controls

Version 1's three logged parent failures remain in the schema-repair evidence;
its all-success confirmation gate did not pass. Version 2 froze new seeds and
uses the schema-repaired parent, isolating dictionary completion. Both phases
complete all 42 planned draws. The gate passes every family:

- Opposed curves: mean test MSE 1.28349 → 0.32396; paired change
  −0.95953, SE 0.02252 squared response units.
- Opposite reference labels: 1.28621 → 0.33290; change −0.95331,
  SE 0.02081.
- Log-scale curves: 1.31424 → 0.30616; change −1.00808, SE 0.01630.
- Smooth, null, numeric-indicator and additive-factor predictions are
  byte-identical on every draw. Numeric predictions pass the exact gate;
  null interaction adoptions do not increase. Reference curves are nonconstant
  under both label orders. No calibrated FDR claim is made.

The initial complete version-2 capture and the canonical versioned-runner
capture agree exactly on every input digest, prediction digest, MSE and
reference-curve range in all 84 phase/draw rows. REPRODUCTION.md explains the
module separation, which preserves the original schema replay unchanged.
The repeated capture is not counted as new independent confirmation.

## Tests and full benchmark

Three direct regressions fail on the old discovery code; the heredity guard
already passes. With the repair, both reference-label curves are recovered,
log-scale inputs replay correctly, and strong heredity still applies.
31 focused discovery/schema checks pass; 11 gate-evidence tests pass. The
full non-R-parity suite on the measured library passes **7,048 tests**, with
15 skipped, 407 deselected and one expected failure (904.63 seconds).
The final parent-artifact integration is additionally checked by focused
source/replay tests. Ruff and diff checks pass.

The unchanged `predictive-tracks/2` competitor study completes all 168 jobs
before and after. Discovery's factor gap changes from +75.14% (SE 4.14 points)
to −8.23% (SE 2.25 points) versus retrospectively selected CatBoost. Mean MSE
falls 2.01692 → 1.05660; paired change −0.96032, SE 0.02317. The paired percent
change is −47.58%, SE 0.73 points. All eight factor draws improve. The 120
competitor/Ridge predictions and 40 unaffected mboostrs predictions replay
exactly. All six track/process cells meet the 1% point-gap target in that
study. It is a focused generator battery with unequal search designs and
leaves eleven scorecard dimensions unmeasured.

No R parity or full PDF book build was run for this orchestration change.
The executed benchmark notebook and phase-specific scorecards are in the
separate predictive-validity study MR; no cross-study aggregate is computed.
