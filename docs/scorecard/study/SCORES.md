# KDD scorecard study: scores

Protocol: `PROTOCOL.md` (pre-registered; hashes in `PREREG_HASHES.txt`).
Truth: `truth.json`, `*_test_truth.csv`. Generator: `gen.py`. Scorer: `score.py`.
Outputs: `runs/run_A` (mboostrs only), `runs/run_B` (free library choice).

Scores are for mboostrs from run A. Run B is a reference: the same battery run by an
analyst who used numpy, scipy and scikit-learn and did not use mboostrs. Its scores show
what the free analyst reached on the same data, and are not a score for any library.

## Score sheet

| Dim | mboostrs (run A) | Free analyst (run B) | Evidence |
|---|---|---|---|
| D1 Relationship discovery | 3.5 | 3.5 | TPR on main 1.0 for both. FDR on main: A 0.29 (strict), B 0.00. Small-n TPR: A 1.0, B 0.8. |
| D2 Causal humility | not scored | not scored | No causal workflow was in the battery. |
| D3 Explanation / story | 3.5 | 4 | Both reports rank effects with intervals and list caveats and next checks. A's report says x6 and x7 are interaction members only, but its `findings_main.json` marks them `selected: true` as main effects. |
| D4 Importance quantification | 3 | 3.5 | Planted-effect coverage: A 2 of 4, B 3 of 4. A's overall 0.89 includes 14 null inputs with zero-width intervals that cover 0 by convention. |
| D5 Iteration speed (proxy) | 4 | 4 | Logged main fits, n=2000. A: median 0.034 s, p90 0.72 s (21 fits). B: median 0.051 s, p90 0.48 s (16 fits). Not the 50k×20 session. |
| D6 Robustness | 3 | 3 | Small-n (n=200) and messy (15% missing, 1% outlying y). A: TPR 1.0 on both. B: TPR 0.8 on small, 1.0 on messy. |
| D7 Human-friendliness | not scored | not scored | No task-based study was run. |
| D8 Agent-friendliness | 4 | not applicable | The mboostrs-only analyst completed all deliverables with no human help, using the mboostrs agent skill. One battery and one run only. |
| D9 Uncertainty honesty | 3.5 | 3 | Predictive coverage on main: A 0.89, B 0.88 (target 0.90). Spearman(half-width, true sigma): A 0.98, B 0.58. A dropped its scale model at n=200 (widths reached 953), so small-n widths are not adaptive. |
| D10 Reproducibility | 4 | 4 | Fresh-directory rerun of `run_all.py` reproduced all six findings and prediction files byte for byte, for both runs. The final report text is not regenerated. |
| **Weighted (auto, §6.2)** | **3.6** | not computed | Over the eight scored dimensions, weights D1 2, D3 2, D4 1, D5 2, D6 1, D8 2, D9 2, D10 1. D2 and D7 are excluded. |

Weighted (semi-automated) is not computed: the study is fully automated.

## Evidence notes

**Planted-effect intervals, main (truth contrast in brackets).**

- mboostrs (A): x1 [1.31, 1.37] covers 1.354; x2 [0.58, 0.64] misses 0.655; x4 [3.77, 3.98] misses 3.984; x5 [1.30, 1.50] covers 1.5. Estimates are slightly low for x2 and x4.
- Free (B): x1, x2, x4 covered; x5 [1.54, 1.70] misses 1.5.

**Pair x6:x7.** Both runs select the pair. Run A's pair p-value is below 1e-100.

**x6 and x7 selection (A).** The protocol's `effects` contract says `selected` means the input has a mean effect. Run A's report says x6 and x7 have no main effect and are selected only as members of the interaction. Its `effects` block marks them `selected: true`. The strict scorer counts them as false positives, which gives FDR 0.29. If they are read as interaction members, FDR is 0.00. The protocol does not define this case. The protocol should have specified it before the run; the strict reading is reported above.

**Messy.** Run A keeps 38 rows with robust |z| > 4 in the fit, and it fills missing inputs with the median plus a missing-indicator term. Run B fills with an iterative imputer. Neither run added a clean-row sensitivity check.

**Small n.** Run A's predictive intervals at n=200 come from a conformal mean model, not the scale model, and were checked on only 50 held-out rows.

## Limits

- One planted battery with one seed per dataset. Scores describe this battery only.
- D2 and D7 are not scored. D5 is a proxy and does not follow the 50k×20 session.
- The truth file was kept outside the repo during the runs. The analysts were told not to read it, but that rule was not enforced by a sandbox.
- The scores were assigned by the orchestrating session from the outputs, the reports and the scorer. The run transcripts were not reviewed line by line.
- The scoring rule for x6 and x7 was not fixed in the protocol. This is a protocol gap, and the scores above use the strict reading.
