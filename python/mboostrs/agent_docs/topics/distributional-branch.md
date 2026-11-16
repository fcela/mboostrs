# The distributional branch (step 8)

## Step 8 — Distributional diagnostics and family comparison

**`heteroscedasticity(model)`** regresses `log(residual²)` on the fitted
value. Under constant variance the slope is zero with a known standard error
(the variance of `log(chi²₁)` is `π²/2`, whatever the data's scale), so `t`
is comparable across datasets — an R² here is not, because the irreducible
variance of `log(chi²₁)` is so large that a real pattern still explains only
a few percent of it. `spread_ratio` is the residual spread at the 90th
percentile of the fitted values against the 10th — the number that says
whether the pattern matters. It refuses non-Gaussian families by name: a
Poisson or Gamma fit is *expected* to show spread that grows with the mean,
and reading that as heteroscedasticity is a category error.

The thresholds the library itself uses when it turns this test into a
suggestion: a hit on the fitted value is `|t| > 3` with `spread_ratio > 1.3`
(read both directions — a ratio of 0.7 is 1/1.4); a driver column (from
`by_column`, because spread can track a predictor the fitted value does not
move with) is `|t| > 3.5` with ratio `> 1.3`. Below those, treat it as noise
and say so. Above them: the mean fit is still the best guess at the mean,
but every interval around it is too narrow where the spread is large and
too wide where it is small — that is reportable on its own. The remedy is a
distributional model: `gamboostLSS` with `GaussianLSS()`, giving the scale
(sigma) its own predictor — the driver column names the first sigma term.
Then *measure* it: `compare_specifications` with the mean model as reference
and the mapping specification `{"mu": ..., "sigma": ...}` as candidate,
under the joint proper score. A spread diagnosis nominates; only the
comparison adopts.

**`scan_lss_families(spec, data, support=..., metric="cvrisk", folds=...)`**
chooses a response distribution by fitting the alternatives — the direct
way to make the least-informed decision in a distributional model. Prefer
`metric="cvrisk"` (out-of-bag risk on folds you pass): the legacy
`"GAIC"` penalty counts boosting updates plus one, so it largely ranks by
parameter count and promotes one-parameter families over better-fitting
two-parameter ones (measured on zero-inflated Poisson data: GAIC promotes
`GeometricLSS` over the correctly-identifying `ZeroAdjustedPoissonLSS`).
`support=` picks the candidate set — `"realline"`, `"realplus"`,
`"real0to1"`, `"counts"`, `"zeroplus"` — from the response's support.
`"realplus"` includes `LogNormalLSS`, `WeibullLSS` and `LogLogLSS`, which take
a plain positive response as well as `Surv(time, status)`. Cross-family comparison rests on every family's risk being the exact
negative log-likelihood of the same data, which the test suite pins. A
mapping specification is matched to each family's parameter count by
position. Failed fits are recorded with `status="error"`, not silently
missing — say why a candidate could not fit when the winner beat it by
default. Candidates fitted to different rows, responses or effective weights
raise `ValueError` in `scan_lss_families` and `gaic_table`; candidates that
share their rows but differ in response transform (`log(y)` against `y`) or
weight policy come back unsorted with a `comparison` column giving the reason. With a distributional family adopted, tune it with
`tune_mstop(..., score="family")` (a mapping spec needs one rhs per
parameter) and read its terms with `term_report` on the mapping form.

Once a distributional fit exists, `scale_absorption(fit, folds=5, seed=0)` refits its mean more flexibly on the same folds and returns a verdict, `"scale effect robust"` or `"scale effect absorbs mean misspecification"` (or `"inconclusive"`): when the scale effect is absorbing an underfit mean, the mean needs fixing before the scale is interpreted.

Enter this branch when the Gaussian spread test crosses its threshold, the
response has unusual support (for example, proportions or zero-inflated
counts), or `autospec` nominates a distributional family. Otherwise report the
mean model and its residual-spread limitation.
