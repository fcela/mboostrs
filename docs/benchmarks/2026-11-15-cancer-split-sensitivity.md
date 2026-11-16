# Cancer split sensitivity

The supplied single-holdout autospec procedure has mean raw test log loss
0.10171 on twenty additional random splits, compared with 0.07984 for tuned
logistic regression: 27.39% excess loss. All twenty jobs completed. The
comparison is paired within each split; the splits reuse the same 569
observations, so their standard deviations describe partition sensitivity
and are not standard errors for a population difference.

[Comparison table: all methods and four scores](../book/benchmarks/competitive_refresh/cancer_splits_541_headline.html)

## Evaluation design

The evaluation design was committed as `2180d123` before any of its results
were computed. It uses the Wisconsin diagnostic breast cancer panel and the
shootout's unchanged stratified fit/validation/calibration/test partitions
for seeds 201 through 220. Each split has 273 fitting rows, 91 validation
rows, 91 calibration rows and 114 test rows. The response is malignancy,
with malignant coded 1. The procedure receives thirty numeric measurements
and no supplied formula.

The autospec call uses budget 32, initial stopping iteration 200, explicit
step length 0.1, `interactions="adopt"` and `residual_tree="auto"`, with the
original single holdout supplied as fold weights. The step length is fixed
by this call; the new optional step-length search is not invoked. The
recommended specification is refitted on all 364 development rows before
calibration and test scoring.

The fixed logistic reference uses the first prespecified penalty in the
shootout's grid. The tuned reference selects among all four penalties using
the same validation rows, then refits on all development rows. The study
changes no selection or stopping rule. Calibration fits a logistic
calibrator on the separate calibration rows; raw log loss is the primary
score. Each draw retains its four scores, test predictions, paired per-row
losses, partition identities, prevalence, chosen specification, step count,
fit accounting and selected validation loss. Missing or failed jobs cannot
be omitted from the planned denominator.

## Predictive scores

Autospec is 1.59% above the fixed reference in mean raw log loss, but 27.39%
above the tuned reference. It beats the fixed reference on 9 of 20 splits
and the tuned reference on 5 of 20. Its paired loss difference against the
tuned reference averages 0.02187 nats per observation, with descriptive
split standard deviation 0.05037 and a range from -0.04976 to 0.19740.
These additional splits show that the earlier three favourable splits
(seeds 101--103 in the stopping-reserve study) do not establish parity.

Mean Brier scores for autospec / fixed logistic / tuned logistic are
0.02739 / 0.02514 / 0.02180, and mean AUCs are 0.99365 / 0.99239 / 0.99537.
Mean calibrated log losses are 0.17465 / 0.21121 / 0.15903. Calibration
increases average loss for all three methods in this experiment; it does
not remove autospec's gap against the tuned reference.

## Loss concentration and selection

The five largest test-row losses account for a mean 72.1% of autospec's
loss within a split, compared with 65.5% for tuned logistic regression.
Across all splits, 561 distinct observations appear in the test partition,
with between one and nine test appearances per observed row. The summary
ranks observations by their mean paired excess loss and retains the number
of appearances beside each mean. These rankings describe the fitted
procedures on this panel and do not identify erroneous observations.

The selected stopping iteration ranges from 358 to 10,000, with median
2,325. Search fit counts range from 20 to 32 and stay within the declared
budget. Raw test loss exceeds the selected validation loss by 0.03716 nats
per observation on average. The validation rows selected the specification
and stopping iteration, so that difference combines selection optimism and
variation between partitions; it does not isolate either contribution.

## Original battery and issue status

The rerun of the original three-split battery on the rebuilt extension
reproduces mean log loss 0.11788, 30.82% above the historical tuned logistic
reference at 0.09010. Its three raw losses are unchanged from the preceding
diagnosis. The scorecard and four-metric per-draw records now retain the
current Python and native identities.

The additional splits complement the original three-split acceptance
battery. They do not replace its results, and their fixed and tuned
logistic comparisons do not remeasure other historical competitors. Issues
#541 and #810 require further work because the tuned-reference comparison
remains outside the 5% target. No estimator defaults change in this MR.

## Source and reproduction

The Python procedure is current main revision `be538363`, with the
additional benchmark runner committed as `2180d123`. The release native
extension was rebuilt from `be538363`; the evidence retains its clean build
identity, source digest and compiler information. Measurement and reporting
scripts are under `benches/`; the report is generated without refitting:

```bash
.venv/bin/python -m benches.bench_cancer_splits_541
.venv/bin/python -m benches.report_cancer_splits_541
.venv/bin/python -m pytest tests/test_cancer_split_evidence.py tests/test_competitive_refresh.py -q
```

The frozen plan, complete measurements, per-draw summary and annotated HTML
table are `cancer_splits_541.planned.json`, `cancer_splits_541.json`,
`cancer_splits_541_summary.json` and `cancer_splits_541_headline.html` under
`docs/book/benchmarks/competitive_refresh/`. Rerunning on another library
identity creates a new measurement. The original battery uses
`python -m benches.refresh_competitive --tasks cancer`, followed by
`python -m benches.report_competitive_refresh --rerun small_cancer.json`.
