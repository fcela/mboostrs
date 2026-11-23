# mboostrs against five libraries on held-out data

Scorecard `mboostrs-scorecard/2.1` (`docs/scorecard/scorecard.md` SHA-256
`12d0ea33…3daec`), fully automated mode. The [protocol](competitive_heldout/PROTOCOL.md)
was written before any run; [deviations](competitive_heldout/DEVIATIONS.md)
are recorded. Six dimensions are measured and six are not, so no aggregate
score is reported.

## Score sheet

| Dimension | mboostrs | CatBoost | LightGBM | XGBoost | EBM | Ridge/logistic reference |
|---|---|---|---|---|---|---|
| D1a Relationship recovery (prescribed) | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured |
| D1b Relationship discovery (data alone) | **3** | 2 | 2 | 2 | 2 | 1 |
| D2 Causal humility | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured |
| D3 Explanation quality | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured |
| D4 Importance quantification | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured |
| D5 Iteration speed | 2 | 3 | 4 | 4 | 1.5 | 4 |
| D6 Robustness | 2 | 3 | 3 | 3 | 2.5 | 3 |
| D7 Human-friendliness | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured |
| D8 Agent-friendliness | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured |
| D9 Uncertainty honesty | **4** | 2 | 2 | 2 | 1 | 3 |
| D10 Reproducibility | **3** | 2 | 2 | 2 | 2 | 2 |
| D11 Predictive validity | 3 | **4** | 3 | 2.5 | 3.5 | n/a (the reference) |

Each score is anchored to the scorecard and justified, with its measurements,
in [scorecards/](competitive_heldout/scorecards/) (one report per library, in
the scorecard's report template).

In short, mboostrs leads on what the scorecard weights for automated
discovery. It reports the least false structure, its prediction intervals hold
their stated coverage, and its fits leave a decision trail. It trails on the
scorecard's operational dimensions. It predicts worse than CatBoost on most
real data, it misses pure interactions, it fails on 29% of real-data runs, and
it is the second-slowest library per follow-up fit.

## Libraries and tasks

Each library was called once per task with its default settings and no user
tuning, on one thread: `mb.fit` (working tree), CatBoost 1.2.10, LightGBM
4.7.0, XGBoost 3.4.1, EBM (interpret 0.7.8), and a prespecified scikit-learn
ridge/logistic pipeline with imputation and one-hot encoding as the simple
reference D11 requires.

Two task sets were used:

- **Real.** The 95 datasets of the [held-out study](2026-11-21-fit-heldout.md):
  45 regression and 50 binary, three new seeds each. mboostrs was developed
  against these tables in that study and the competitors were not, which
  favours mboostrs.
- **Planted.** Seven generators with known structure, five draws each: smooth
  additive (S1), pure interaction (S2), effect on the spread only (S3),
  no effect (S4), 200 training rows (S5), near-duplicate proxies (S6) and binary
  additive (S7).

## Results by dimension

**D11 predictive validity.** Real data, median gap to the best complete arm
and the number of tasks within 5% of it:

| | mboostrs | CatBoost | LightGBM | XGBoost | EBM | Reference |
|---|---|---|---|---|---|---|
| Regression: completed / 45 | 33 | 45 | 45 | 45 | 39 | 45 |
| Regression: median gap to best | 35.3% | 0.0% | 9.5% | 19.3% | 7.8% | 52.4% |
| Regression: within 5% of best | 8 | 34 | 15 | 6 | 18 | 7 |
| Binary: completed / 50 | 30 | 50 | 50 | 50 | 47 | 50 |
| Binary: median gap to best | 12.6% | 3.9% | 29.4% | 35.7% | 1.9% | 20.5% |
| Binary: within 5% of best | 13 | 29 | 6 | 4 | 33 | 14 |

mboostrs beats the reference on 27 of 33 regression tasks and 23 of 30 binary
tasks (Wilcoxon p = 3e-4 for each). Against CatBoost it is worse on 24 of 33
regression tasks, with a median loss ratio of 1.28 (p = 0.004). On the planted
additive tasks it has the lowest loss of all six arms (S1, S3, S4, S5, S6, S7).
On the pure interaction S2 its MSE is 9.32, the same as the linear reference,
against 0.43 for CatBoost: `fit` does not adopt interactions.

**D1b discovery from data alone.** A predictor counts as found when permuting
it raises held-out loss by more than twice its standard error; the same rule
applies to every library.

| | mboostrs | CatBoost | LightGBM | XGBoost | EBM | Reference | mboostrs's own formula |
|---|---|---|---|---|---|---|---|
| True predictors found | 70/80 | 80/80 | 80/80 | 80/80 | 80/80 | 57/80 | 77/80 |
| — of which interactions | 0/10 | 10/10 | 10/10 | 10/10 | 10/10 | 0/10 | 7/10 |
| False-discovery rate | 1.4% | 41.2% | 31.0% | 45.9% | 31.0% | 3.4% | 64.5% |
| Null tasks with no finding | 5/5 | 1/5 | 1/5 | 1/5 | 5/5 | 5/5 | 0/5 |

The fitted mboostrs model carries almost no false structure, but the formula it
reports keeps noise terms. Its deselection step does not remove them, so what
the library states is less reliable than what it fits. On the 38 published
effects of the well-studied real datasets, every library recovers all 75
(effect, seed) pairs that every library completed, so that test does not
separate the libraries.

**D9 uncertainty honesty.** Each library's own 90% prediction interval:

| | mboostrs (split conformal) | CatBoost (MultiQuantile) | LightGBM (quantile) | XGBoost (quantile) | EBM | Reference (OLS) |
|---|---|---|---|---|---|---|
| Median coverage | 0.900 | 0.752 | 0.790 | 0.733 | not offered | 0.914 |
| Runs with coverage 0.85–0.95 | 96% | 9% | 8% | 8% | — | 73% |
| S3 coverage by tercile of the spread variable | 0.89 / 0.91 / 0.91 | 0.66 / 0.73 / 0.72 | 0.81 / 0.76 / 0.74 | 0.65 / 0.69 / 0.67 | — | 1.00 / 0.92 / 0.79 |

**D5 iteration speed.** The scorecard session: ten follow-up fits on 50,000 ×
20 data, each with prediction and permutation importance. Median seconds per
follow-up: reference 0.06, XGBoost 0.58, LightGBM 0.76, CatBoost 6.7,
mboostrs 34.4, EBM 67.5.

**D6 robustness.** Completed real-data runs out of 285: CatBoost, LightGBM,
XGBoost and the reference 285; EBM 260 (25 timeouts on wide tables);
mboostrs 202. The mboostrs failures are those recorded in
fcela/mboostrs-dev#915: missing values and unseen factor levels at
prediction, and tables in which every row has a missing value. With 200
training rows (S5), mboostrs has the lowest loss (0.265 against 0.338 for EBM
and 0.405 for CatBoost).

**D10 reproducibility.** Every library reproduced its predictions exactly on
10 of 10 reruns. Only mboostrs records a decision trail.

## What this means for the roadmap

1. **D6: make `fit` → `predict` complete on real data** (#915). This is the
   largest gap, and it also removes 12 regression and 20 binary tasks from
   mboostrs's D11 comparison.
2. **D11 and D1b: interaction capacity** (#911, #608). The S2 result (MSE 9.32
   against 0.43) and the real-data gap to CatBoost point the same way.
3. **D1b: make the reported formula match the fitted structure.** Native FDR
   is 64.5%, against 1.4% for the fitted model under the permutation rule.
4. **D5: follow-up fits take 34 s at 50,000 rows** (#503, #88, #296).

## Limits

- Default settings for every library; no library was tuned. Tuning changes
  each library's results by different amounts, and this study does not
  measure it. The earlier study with tuning grids is
  [competitive_shootout_report.md](competitive_shootout_report.md).
- The real datasets were used to develop `mb.fit`; the competitors had no such
  exposure.
- D1b on real data has no oracle; the published-effects check is coarse (sign
  of a 10th-to-90th percentile change) and did not separate the libraries.
- One machine (Apple M4), eight concurrent single-thread runs.
- Unmeasured: D1a, D2, D3, D4, D7, D8. D8 needs the agent battery
  (fcela/mboostrs-dev#910).

## Artifacts

[competitive_heldout/](competitive_heldout/): protocol and hashes, deviations,
planted-task manifest, every run (`results.jsonl`), the D5 sessions, the D10
reruns, the analysis and the six score reports. Runner:
[benches/competitive_heldout](../../benches/competitive_heldout/README.md).
