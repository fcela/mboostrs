# Scorecard migration

`mboostrs-scorecard/2.1` in [scorecard.md](scorecard.md) is the sole instrument
for new evaluations. Historical protocols remain evidence of what was actually
measured; they are not alternative active scorecards.

## 2.0 → 2.1

Version 2.1 splits the former **D1 Relationship discovery** into **D1a**
(recovery under a prescribed specification) and **D1b** (discovery from data
alone), and documents the Track A / partial-knowledge / Track B information
contracts (real-data prespecified ≠ oracle; paired excess/gap formulas). All other IDs keep their 2.0 meaning and anchors.
The card is now twelve dimensions, and the aggregate/renormalization rules read
"twelve" instead of "eleven".

### Old D1 evidence → D1a / D1b

Old D1 grades are not silently regraded. The split is informational: a study is
reassignable only if its protocol recorded how much specification was supplied.

| Old D1 evidence class | Likely new home | Rule |
|---|---|---|
| Studies that supplied a formula/family/candidate predictors (hostile workflows v4, causal reruns, chooser batteries) | D1a proxy | Credit only the structure the specification left open; supplied terms earn no discovery credit |
| Studies that genuinely supplied data alone, no formula | D1b | Must score TPR/FDR by structure type and test that silence cannot win |
| Mixed or unrecorded information contract | Unmeasured for both | Record `unmeasured`; do not split a numeric D1 grade |

Because no recorded battery documents a Track B (data-alone) run, old D1
evidence below is treated as D1a proxy evidence only; D1b remains unmeasured.

### Label impact

- **Keep `dim:D1` for now.** Existing labels, dashboards and issue references
  that use `dim:D1` are retained; 2.1 does not remove or rename the label.
- New or refreshed records should use `dim:D1a` and `dim:D1b` and state the
  information contract. A later version may retire `dim:D1` only after the
  label inventory is migrated; until then `dim:D1` is a legacy alias, not a
  thirteenth dimension.

### Resolved (2026-10-12)

1. **D3 vs D4 — split, no merge.** D3 (Explanation quality) owns *narrative
   faithfulness and completeness*: does the reported explanation describe what
   the model actually does — no invented structure, no omitted structure? D4
   (Importance quantification) owns *magnitudes and uncertainty*: are effect
   sizes correct, are intervals calibrated, is selection honesty preserved?
   The two fail independently: a faithful story with wrong magnitudes fails
   D4, not D3; correct magnitudes wrapped in a misleading story fails D3,
   not D4. No cross-credit either way.

2. **D1 vs D11 — split, with the paired decomposition as the documented
   relationship.** D1a/D1b score *recovery of open structure* (recall and
   false discovery proportion by structure type; terms supplied by the
   specification earn no discovery credit). D11 scores *prediction under
   matched budgets* (held-out loss, no credit for which structure was
   recovered). The two share held-out data but answer different questions:
   a good prediction from a wrong structure fails D1 and passes D11, and
   that is the correct record. The paired decomposition — estimation excess
   + discovery gap = end-to-end excess — is the documented account of how
   the numbers relate, not a license for cross-credit.

## Historical instruments

| Instrument | Status | Treatment |
|---|---|---|
| Original ten-dimension knowledge-discovery scorecard | Historical version 1 | D1–D10 retain their meanings; original grades and ten-dimension aggregates remain version-1 results |
| Claims and causal discovery protocols | Frozen version-1 task-specific thresholds | Preserve their generators, graders, protocols and hashes; use raw results as named evidence, not as an eleven-dimension aggregate |
| PM2.5 competition | Historical partial version-1 assessment | Iteration-cost proxies and interval metrics retain their limitations; other dimensions were unmeasured |
| Hostile workflows v4 | Archived local rubric with incompatible numbering | Preserve grades in their local namespace; use the crosswalk below to find evidence, not to copy numerical grades |
| KDD benchmark and expansion notebooks | Historical instrument explorations | Their rubrics, task coverage and reported scores do not supersede the canonical card |

The new D11 fills the original card's lack of an explicit predictive-validity
dimension. It does not change what D1 discovery (now D1a/D1b) or D9 calibration
mean.

## Hostile workflows v4 crosswalk

| Local v4 label | Local construct | Relevant canonical evidence |
|---|---|---|
| D1, simulated cases | Relationship discovery plus case-specific handling | D1 discovery; handling/trap evidence can also inform D2/D6 |
| D1, real-data case | Held-out RMSE and validation honesty | D11 predictive validity; no numerical D1 conversion |
| D2 | Uncertainty honesty, limitations and causal language | D4 effect uncertainty, D9 calibration, and D2 causal scope; no one-to-one numerical conversion |
| D3 | Method fit | Evidence for D2/D6/D11 depending on the claim; no standalone canonical dimension or automatic numerical conversion |
| D5 | Reproducibility | D10 auditability evidence; **not D5 speed** |
| D8 | Tool fluency | D8 agent workflow evidence; local anchors differ |
| D10 | Communication | D3 explanation evidence; **not D10 auditability** |

A local score of 5 does not become a canonical score of 5. The raw evidence
must satisfy the canonical anchor before a new assessment can assign a grade.
The v4 efficiency table is agent-session effort, not the standardized D5
latency experiment. Its D5 grade does not measure iteration speed.

The original ten-dimension instrument is available at the [pre-consolidation
revision](https://github.com/fcela/mboostrs/blob/ff92ead/docs/scorecard/scorecard.md).
It is historical material, not a second active standard.

## New evaluations

Record `scorecard_id`, `scorecard_sha256`, dimension IDs and names, benchmark
protocol identity, source/native revision, task and split identity, mode,
evidence references, coverage status and grading justification. The registry
[dimensions.json](dimensions.json) records the stable names; the Markdown
scorecard owns the anchors and weights.

Incomplete studies report unmeasured dimensions. Changed anchors, weights or
meaning require a new explicit scorecard version and a migration note; retain
frozen historical evidence and do not reuse an existing dimension ID for a
different construct.
