# Cross-validation stopping ties

The default stopping rule remains the first exact computed minimum on the
positive iteration grid. This decision is specified for #653 before new
numerical-scale checks. It retains the current R-compatible finite-curve
behavior and introduces no approximate-tie tolerance.

For finite aggregated risks, compare their stored floating-point values.
Exclude iteration 0 when positive iterations exist. Select the minimum value;
among exactly equal minima, take the first grid position. A genuinely distinct
smaller value wins even when separated from another risk by one representable
floating-point increment. Risk units, magnitude and the observed 76-versus-1556
case do not define an approximate-equality threshold.

Equivalent fitting pipelines may accumulate risks differently at roundoff
scale. The rule therefore does not promise identical stopping labels for
slightly different curves. Retain each computed curve and exact minimum,
attained risk and resulting predictions. In the archived #648 example, the
prediction difference is 6.51e-15 and the risk difference is 5.55e-17; its original
reported stops remain unchanged. Those observed numbers are diagnostic
comparison errors, not new parameters of the stopping rule.

The existing one-standard-error rule is an optional statistical simplification,
not a numerical tie rule. It must be chosen before comparing outcomes and
reported separately. Its across-fold error scale is not a confidence interval
for the location of the minimum because folds share training observations.

Contract checks use three independently specified scales (2^-600,1,2^600),
exact repeated minima, and distinct adjacent floating-point minima. They
exercise the native foldwise Quantile(.5) scorer, the Python selection helper,
and R's finite-curve selection semantics. Median quantile loss is used so the
requested risk sequence can be represented exactly by zero observed responses
and predictions equal to minus twice the target risk. This isolates selection
from model-fitting differences.

This decision addresses the missing specification and regression checks. It
does not claim to eliminate the stopping-label sensitivity reported in #653.
Nonfinite-only curves and offset-only models retain their existing documented
fallbacks; they are outside these finite-curve numerical comparisons. Any new
approximate or statistically stabilized policy requires its own specification,
compatibility decision and fresh evaluation before replacing this default.
