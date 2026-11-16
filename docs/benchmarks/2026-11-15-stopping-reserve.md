# Cancer stopping-reserve experiment

A reserve used only to select the stopping iteration increased mean cancer
log loss from 0.06908 to 0.08773 on three additional random splits. The
candidate failed the rule fixed before fitting: improve the cancer comparison
without worsening any control mean, and stay within 5% of the matched tuned
linear reference. The nonlinear control also worsened, by 0.00250 log loss.
Estimator defaults are unchanged; issues #541 and #810 remain open.

[Comparison table: all methods and four scores](../book/benchmarks/competitive_refresh/stopping_reserve_headline.html)

## Data and fitting

The study uses the Wisconsin diagnostic breast cancer panel from the frozen
shootout and three generated binary-response controls. Each method uses the
same fit, validation, calibration and test rows within a draw. Seeds are
101, 102 and 103. The cancer splits overlap with each other and with earlier
splits of the same 569 observations; their spread is descriptive and does not
provide independent population replications.

The generated controls have ten correlated normal predictors, 273 fitting
rows, 91 validation rows, 182 calibration rows and 20,000 test rows. Their
log-odds are respectively `x0 + x1 - 0.5`,
`1.5*sin(x0) + x1*x2 - 0.5`, and `-0.5`. The last process has no predictor
effect. The retained protocol and runner define the predictor distribution.

The baseline is the supplied single-holdout autospec procedure, with budget
32, step length 0.1 and initial stopping iteration 200. The candidate reserves
20% of the original fitting rows, stratified by response. Specification search
uses the other fitting rows and the original validation rows. After search,
the candidate fits the chosen specification on those rows, scores the
reserve on iterations 1 and 25 through 10,000 in steps of 25, and selects the
smallest minimizing iteration. It then refits on all development rows.
The reserve's responses never enter specification search. Reserving rows also
reduces the data available for that search, so this experiment measures the
complete changed procedure rather than an isolated stopping-rule effect.

The reference is logistic regression selected from the shootout's four
prespecified penalties on the original validation rows. Calibration uses
the separate calibration rows. Every draw records raw log loss, Brier score,
AUC and calibrated log loss, plus paired per-row raw losses and response
prevalence. All twelve jobs completed.

## Results

Mean raw log loss for baseline / reserve / tuned linear is 0.06908 / 0.08773 /
0.07812 on cancer, 0.50749 / 0.50142 / 0.49473 on the correlated-linear
control, 0.59224 / 0.59473 / 0.64772 on the nonlinear control, and 0.67917 /
0.66657 / 0.66627 on the null control. Lower loss is better. The reserve is
12.30% above tuned linear on these cancer splits.

The unchanged baseline beats the linear reference on these additional cancer
splits, whereas it loses on the original three. This sensitivity prevents a
claim that the original 30.82% gap is a stable population difference. It also
does not establish that the original battery's 5% acceptance target is met.

## Source and reproduction

These measurements use Python source revision `0a1852f7`, before the
step-length search merged in #901. The loaded native release binary records
revision `4278a31e` and its source digest in the retained artifact. They describe
that recorded combination; they do not evaluate the latest procedure on main.
The runner is retained byte-for-byte as executed, with its SHA256 in the plan.

The complete JSON is stored as deterministic gzip under
`docs/book/benchmarks/competitive_refresh/stopping_reserve.json.gz`.
`stopping_reserve_archive.json` checks the archive and uncompressed hashes;
`stopping_reserve_summary.json` records every draw's four metrics and the
experimental decision. Rebuild the summary without fitting:

```bash
.venv/bin/python -m benches.report_stopping_reserve
.venv/bin/python -m pytest tests/test_stopping_reserve_evidence.py -q
```

To rerun the experimental design on the installed library, use
`.venv/bin/python -m benches.cancer_stopping_reserve`. That produces an
uncompressed JSON measurement with the new source and native identities.
It is a new measurement when either identity changes.
