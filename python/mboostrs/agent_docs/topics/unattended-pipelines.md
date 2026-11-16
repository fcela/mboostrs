# Unattended pipelines

A script that will be re-run on new data, by you or by a grader, must make
every decision from the data it is given and leave a record of each one.

- **Fix every seed.** Fold assignments (`seed=` in `tune_mstop`,
  `compare_specifications`, `term_report`, `screen_interactions`,
  `conformal_interval`, `stabsel` folds) are the only randomness; the
  boosting itself is deterministic, bit for bit, across thread counts.
- **Decide by rule, not by reading output.** Write the adoption rule into
  the code: a candidate is adopted when it lowers out-of-fold risk by more
  than two `change se` in `compare_specifications`; a term is kept when its
  `term_report` `earns` exceeds two SE; a spread driver is adopted when the
  mapping specification `{"mu": ..., "sigma": ...}` wins the same
  comparison. Thresholds tuned by eye on one dataset will not transfer.
- **Check the rule on data with no signal.** Run the pipeline once with the
  response permuted (`df.with_columns(pl.col("y").shuffle(seed=1))`). Every
  finding it reports there is a false discovery; if it reports any, tighten
  the rule before trusting the real run.
- **Handle an empty selection.** On data with no signal a rule should
  drop every term. Build the formula as `"y ~ 1"` then. `gamboost()` and
  `glmboost()` fit it as an offset-only model: every prediction is the
  family's offset and no term is selected. `gamboostLSS()` and
  `glmboostLSS()` fit the intercept as a parameter learner. Never build the
  formula as `"y ~ "` with nothing after the tilde. `gam_refit` refuses an
  intercept-only model and says why.
- **Check it on small data.** Run it on a few hundred rows. A pipeline whose
  cost or rules assume thousands of rows will fail or over-claim there.
- **Record structured output.** Results expose `to_dict()` (models,
  refits, comparisons, `ConformalInterval`) or `to_frame()`; write those to
  JSON instead of parsing printed text. Each of those records carries a
  `"provenance"` entry: the library build, the Python and NumPy versions, and
  a SHA-256 fingerprint of the training data. `mb.provenance(data, seed=s)`
  returns the same record for a pipeline's own output. Record the seeds,
  every candidate compared with its score, and the reason each was kept or
  dropped.
- **Keep the warnings.** Capture them (`warnings.catch_warnings(record=True)`)
  and write them out. Extrapolation warnings at prediction time name the
  term and how many rows fall outside its training range; a shifted test
  set is a finding to report.
- **Record covariates outside the training range.** Before scoring a test
  set, write `model.extrapolation_report(test)` to the run record: one row per
  term and covariate with rows outside its training range, with the rows
  below and above and the worst distance in training ranges. Every mean
  prediction (`predict`, `predict_parameters`, `params`, `mean`, `predict_all`,
  refit `predict`, conformal intervals) continues an out-of-range spline at
  its edge slope by default and warns, naming the covariate and the rows.
  That continuation can be far from the response scale when the training
  sample is small and the covariate is heavy-tailed, so when any row is
  listed, pass `extrapolation="clip"` to the prediction (and to
  `conformal_interval`), which holds the effect flat at the training edge.
  A `by=` covariate needs this most: an out-of-range `by=` value multiplies
  the whole term.
- **Validate the test set before predicting.** Call
  `mb.validate(model, test, output="records")` first and write its issues to
  the run record. Each issue has a `kind` (`missing_column`, `unseen_level`,
  `na`, `extra_column`, `row_count`, `offset`, `extrapolation`,
  `not_range_checked`, `check_skipped`), a `severity`, the `column` or `term`
  it concerns, a `count`, and a `suggested_policy`. An `"error"` means
  `predict` refuses those rows, so stop the run; a `"warning"` means predict
  runs and the result needs a note. An empty list means the checks found
  nothing. `validate` does not predict and does not change `test`. Terms
  built from `btree`, `bkernel`, `buser` or `brad` have no training-range
  check, and appear as `not_range_checked` warnings.
- **Budget the run.** `compare_specifications(..., fit_budget=N)` refuses
  before any work when a comparison would exceed `N` fits; see
  `agent_guide("costs")` for what each call costs.

```python
import json, warnings
import numpy as np, polars as pl
import mboostrs as mb

def pipeline(df, seed=1):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rows = mb.compare_specifications(
            ["bbs(x, df=4)", "bbs(x, df=4) + bbs(z, df=4)"], df, "y",
            folds=5, mstop=300, seed=seed)
    base, cand = rows[0], rows[1]
    keep_z = cand["change"] < -2 * cand["change se"]
    return {"keep_z": bool(keep_z), "rows": rows,
            "warnings": [str(w.message) for w in caught],
            "version": mb.__version__, "seed": seed}

rng = np.random.default_rng(0)
df = pl.DataFrame({"x": rng.uniform(0, 1, 400), "z": rng.uniform(0, 1, 400)})
df = df.with_columns(y=np.sin(3 * df["x"].to_numpy()) + rng.normal(0, 0.3, 400))
real = pipeline(df)
null = pipeline(df.with_columns(pl.col("y").shuffle(seed=1)))
print(real["keep_z"], null["keep_z"])        # z has no effect: False, False
print(json.dumps(real, default=str)[:80])
```

## Loops and repeated warnings

Wrap loops of mboostrs calls in `mb.collect_warnings()` to show each repeated warning once, with its count.
