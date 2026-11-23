(record-stopping-rules)=
# Stopping rules, tie-breaking, and holdout sensitivity

This record consolidates the empirical studies on boosting stopping rules ([](../../benchmarks/stopping_rule_ablation.ipynb)), cross-validation tie-breaking semantics, and stopping instability on small single-holdout splits.

## Stopping-rule ablation

**Design.** Eight stopping rules were evaluated across seven shootout panels using matched fold-local preprocessing and rolling-origin inner validation for time-ordered forecasting. Rules compared the standard fixed-grid `cvrisk` argmin against boundary-doubling growth (`continue_to` when the optimum falls near the cap), patience stopping, and repeated cross-validation growth. An earlier prototype confounded stopping rules with differences between refitted and fixed-basis preprocessing; its superseded records are archived separately and excluded from conclusions.

**Outcomes.** On fold-local preprocessing, expanding the iteration cap when the validation curve is still falling prevents underfitting without systematically degrading well-stopped panels, whereas naive patience rules can stop prematurely on flat plateaus or non-monotone curves. On small binary panels (notably breast cancer), single-split validation curves exhibit high variance across splits, making repeated cross-validation more reliable than any single-split stopping heuristic.

## Cross-validation tie-breaking semantics

For finite aggregated cross-validation risks, `cvrisk` excludes iteration 0 whenever positive iterations are evaluated, ignores non-finite values, and selects the **first** grid position achieving the exact minimum floating-point risk. No approximate-equality tolerance is applied: a strictly smaller finite risk wins regardless of magnitude, preserving compatibility with R `mboost`. Because equivalent pipelines can differ by floating-point roundoff on flat curves, callers who want parsimony on flat plateaus should use the explicit one-standard-error rule rather than expecting exact-argmin ties to coincide across implementations.

## Small-sample single-holdout sensitivity (breast cancer panel)

Three targeted studies investigated why automatic search on the 569-row Wisconsin breast cancer panel (273 training, 91 validation, 91 calibration, 114 test rows) exhibited excess log loss relative to logistic regression:

1. **Twenty-split partition sensitivity.** Across 20 additional stratified splits, single-holdout `autospec` attained mean test log loss 0.10171 versus 0.07984 for tuned logistic regression (+27.39% excess; 20/20 completed). Because all 20 splits partition the same 569 observations, split-to-split dispersion measures partition sensitivity rather than population standard error.
2. **Single-holdout stopping heuristics.** On the three-split battery (mean test log loss 0.1179), one draw's 91-row validation curve fell monotonically to step 9,989 (validation risk 0.0276) while test loss bottomed near step 995 (0.1246) and rose to 0.1604 at the cap. Candidate early-stopping rules applied to the single-holdout curve—paired 1-SE, 0.5% plateau, and 1% plateau—improved the cancer battery mean (to 0.0994, 0.1143, and 0.1125, respectively) but regressed on synthetic correlated-linear controls (up to +0.0320 log loss), failing the no-harm gate.
3. **Dedicated stopping reserve.** Splitting off an additional reserve solely for stopping selection worsened mean cancer log loss from 0.06908 to 0.08773 across three fresh splits and degraded a nonlinear control (+0.00250), because reducing the training sample hurt estimation more than decoupling stopping helped.

Taken together, these studies show that a single 91-row holdout cannot reliably identify the test-loss minimum of a long boosting path; multi-fold cross-validation remains the recommended stopping design on small samples.

**Audit artifacts.** Full tables, per-seed logs, and archived v1/v2 ablation runs live in `docs/benchmarks/stopping_ablation_509/`, `docs/benchmarks/cv_stopping_tie_decision.md`, `docs/benchmarks/2026-11-15-cancer-split-sensitivity.md`, `docs/benchmarks/2026-11-15-stopping-reserve.md`, and `docs/book/benchmarks/single_holdout_stopping.md`.
