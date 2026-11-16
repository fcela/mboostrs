# Release limitations

The fitting engine, automated specification search and causal inference
functions have different validation coverage. The capability tables in the
book identify the model and operation combinations checked against R.

## Automated specification search

`autospec` selects a formula and stopping iteration using the supplied
validation design. Repeatedly selecting on the same holdout can produce
optimistic validation loss. Evaluate the complete selected procedure on
separate observations and compare it with a simple prespecified reference.

On the Wisconsin diagnostic breast cancer panel, the single-holdout
procedure with budget 32 and explicit step length 0.1 has 30.82% greater
mean raw test log loss than tuned logistic regression on the original
three splits. Twenty additional overlapping splits show a 27.39% gap.
These results describe that procedure on this panel; they do not establish
a general ranking of the fitting engine. The additional splits reuse the
same observations and do not provide independent population replications.
The procedure remains outside the study's 5% target. The full experiment
is described in [Cancer split sensitivity](benchmarks/2026-11-15-cancer-split-sensitivity.md).

## Causal intervals

`aipw` influence-function intervals condition on the fitted nuisance
predictions. Where treatment probabilities approach zero or one, nuisance
model errors and extrapolation can cause intervals to cover less often
than their stated level. Repeated cross-fitting and resampling the whole
procedure have not established a general correction for this failure.

Inspect overlap and outcome-model support before interpreting an interval.
Trimming observations or requesting an overlap-weighted effect changes the
target population and therefore the estimand, the quantity being estimated.
Neither operation establishes nominal coverage for the original all-row
ATE. See the [`aipw` reference](book/api/functions/aipw.md) for interval
assumptions and the diagnostics returned by the function.

## R confidence intervals

R's `confint()` for a `gamboost` fit returns intervals for basis
coefficients. Python's partial-effect intervals describe fitted term
contributions. Those outputs have different interpretations and should
not be treated as interchangeable. R bootstrap intervals retain zero
contributions from unselected learners and report failed refits through
warnings and result attributes; check the number of successful replicates.

## Artifact verification

The release workflow builds and verifies artifacts before publication. A
manual workflow run builds wheels and a source distribution without
publishing them. A passing local build does not establish that the hosted
platform matrix or publication credentials work. Record a successful hosted
verification of the intended release revision before publishing a tag.
