# Project scorecard

The single authoritative instrument is [Knowledge Discovery Scorecard 2.1](scorecard.md),
identifier `mboostrs-scorecard/2.1`. Every new evaluation uses its twelve
dimension names, anchored grades and evidence rules.

**Start with [mboostrs status](status.md)** for the consolidated view of where
we are strong, where we lag, and what needs measuring. This is an evidence-led
assessment; it does not invent a composite score from incompatible studies.

| Dimension | Name |
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

D1a and D1b share one discovery construct under the [#511 information
contracts](https://github.com/fcela/mboostrs/issues/511): D1a measures recovery
of structure a prescribed specification left open, D1b measures discovery from
data alone. Predictive accuracy/generalization is D11. Effect-interval validity
is D4; prediction-interval calibration is D9. D5 always refers to iteration
speed.

The [migration crosswalk](migration.md) identifies historical numbering and
explains which evidence can be reused. The [working paper](paper.md) is
background; it delegates current grading definitions to the canonical card.
Frozen historical protocols and numerical grades remain unchanged.

Use [report-template.json](report-template.json) for new evidence records.
Record the scorecard content hash when freezing a protocol.

Run `python tools/check_scorecard.py` to check canonical dimension names,
metadata and document links. Scores without sufficient evidence remain
unmeasured. Historical score aggregates are not aggregates under version 2.1.
