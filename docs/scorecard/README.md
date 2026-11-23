# Project Scorecard

The canonical evaluation instrument for `mboostrs` is the [Knowledge Discovery Scorecard 2.1](scorecard.md) (`mboostrs-scorecard/2.1`). Every new benchmark evaluation uses its twelve dimension definitions, 1–5 anchored rubrics, and evidence-reporting rules.

Start with [status.md](status.md) for the consolidated assessment of measured strengths, known weaknesses, and open measurement gaps across all twelve dimensions.

| Code | Dimension |
|---|---|
| D1a | Relationship recovery under a prescribed specification |
| D1b | Relationship discovery from data alone |
| D2 | Causal humility and support |
| D3 | Explanation quality |
| D4 | Importance quantification |
| D5 | Experimental iteration speed |
| D6 | Robustness across conditions |
| D7 | Human-friendliness |
| D8 | Agent-friendliness |
| D9 | Uncertainty honesty (calibration) |
| D10 | Reproducibility and auditability |
| D11 | Predictive validity |

D1a and D1b separate the same discovery construct by information contract: D1a evaluates recovery of structure left open by a prescribed specification, whereas D1b evaluates discovery from data alone. Predictive generalization is evaluated under D11, effect-interval calibration under D4, prediction-interval calibration under D9, and iteration latency under D5.

## Supporting Documents

- [scorecard.md](scorecard.md) — Normative definitions, 1–5 anchors, information contracts, and mode weights.
- [status.md](status.md) — Current evidence-backed profile of `mboostrs` across all twelve dimensions.
- [paper.md](paper.md) — Literature background and design rationale for the framework.
- [migration.md](migration.md) — Crosswalk for earlier benchmark rubrics and version history.
- [dimensions.json](dimensions.json) and [report-template.json](report-template.json) — Machine-readable dimension registry and blank report template for new studies.

Run `python tools/check_scorecard.py` to verify dimension names, template hashes, mode weights, and document links. Dimensions without supporting evidence remain explicitly `unmeasured`; historical aggregates are not carried forward onto version 2.1.
