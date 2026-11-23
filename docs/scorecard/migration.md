# Scorecard Migration and Historical Crosswalk

`mboostrs-scorecard/2.1` ([scorecard.md](scorecard.md)) is the sole instrument for new evaluations. Earlier studies retain their original rubrics and grades as historical evidence; this document explains how earlier dimensions map to version 2.1 and where numerical grades cannot be transferred.

## Version 2.0 to 2.1

Version 2.1 splits **D1 Relationship discovery** into two dimensions determined by the analyst's information contract:

- **D1a (Relationship recovery under a prescribed specification):** recovery of structure left open by a rough supplied formula; terms already named in the formula earn no discovery credit.
- **D1b (Relationship discovery from data alone):** recovery of structure when no formula is supplied, scored by true positive rate and false-discovery rate across structure types, including pure-noise tasks.

All other dimensions (D2–D11) retain their version 2.0 definitions and 1–5 anchors. Because the scorecard now defines twelve dimensions, a weighted aggregate requires all twelve dimensions to be measured on a single prespecified battery in one operating mode.

### Handling Historical D1 Evidence

Earlier D1 grades are not silently converted into D1a or D1b grades:

| Historical D1 study type | Mapping under 2.1 | Rule |
|---|---|---|
| Studies supplying a rough formula, family, or candidate specification (hostile workflows v4, causal battery, chooser studies) | D1a proxy evidence | Credit only structure left open by the supplied specification; no numerical conversion without re-scoring open terms alone |
| Studies supplying data and response semantics alone with no formula | D1b | Must evaluate TPR/FDR by structure type and verify zero false discoveries on pure-noise tasks |
| Mixed or unrecorded information contract | Unmeasured for both | Record `unmeasured`; do not split a legacy numeric D1 score |

### Design Separations Retained in Version 2.1

1. **D3 (Explanation quality) vs. D4 (Importance quantification):** D3 evaluates whether the reported narrative faithfully describes the fitted model without inventing or omitting structure; D4 evaluates effect magnitudes and interval coverage. A clear narrative with uncalibrated intervals passes D3 and fails D4, and vice versa.
2. **D1a/D1b (Structure recovery) vs. D11 (Predictive validity):** D1a/D1b evaluate recovery of true structure; D11 evaluates out-of-sample predictive loss under matched budgets. Paired excess-risk decompositions (`estimation excess + discovery gap = end-to-end excess`) document how the two relate without allowing cross-credit.

## Historical Instruments

| Instrument | Status | Treatment |
|---|---|---|
| Version 1 (ten-dimension scorecard, D1–D10) | Historical | Retained with its original studies; lacked a standalone predictive-validity dimension (later added as D11) |
| Claim-cost and causal discovery protocols | Frozen version-1 task batteries | Preserved with original generators, graders, and protocols as named task evidence, not as version 2.1 aggregates |
| Beijing PM2.5 competition | Historical partial assessment | Retained for forecasting loss, pinball loss, and temporal interval coverage |
| Hostile workflows v4 | Archived local rubric (different numbering) | Preserve grades inside the local namespace; use the crosswalk below only to locate qualitative evidence |
| KDD scorecard study and expansion notebooks | Historical instrument studies | Preserved under their original protocols; do not supersede version 2.1 |

## Hostile Workflows v4 Crosswalk

The hostile-workflows v4 study used a local six-dimension rubric whose labels do not match version 2.1 numbering:

| Local v4 label | Local construct | Related version 2.1 evidence |
|---|---|---|
| D1 (simulated cases) | Structure discovery and trap handling | D1a proxy; trap handling also informs D2 and D6 |
| D1 (real-data case) | Held-out RMSE and validation honesty | D11 predictive validity |
| D2 | Uncertainty honesty, caveats, and causal framing | D2 causal scope, D4 effect uncertainty, and D9 calibration |
| D3 | Method appropriateness | Qualitative context for D2, D6, and D11 (no standalone canonical ID) |
| D5 | Reproducibility | D10 auditability (**not** D5 iteration speed) |
| D8 | Tool fluency | D8 agent workflow evidence (different local anchors) |
| D10 | Communication quality | D3 explanation quality (**not** D10 auditability) |

Local v4 scores are never copied into version 2.1 tables. In particular, the v4 effort table records end-to-end agent session time rather than the standardized D5 50k × 20 follow-up benchmark.

## Recording New Evaluations

Every new evaluation records `scorecard_id`, `scorecard_sha256`, dimension IDs and names, protocol identity, package release/build identity, task and split specification, operating mode, evidence links, coverage status (`scored`, `unmeasured`, `not_applicable`), and written anchor justifications using [report-template.json](report-template.json). Any modification to dimension meanings, anchors, or mode weights requires incrementing the scorecard version.
