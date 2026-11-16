# Production candidate selection

This decision follows development and precedes any reserved fresh observation.
Select the actual-incumbent guard plus native line-search steps for standard
NBinomial mean fits ONLY. Preserve fixed steps for Poisson, Gaussian and every
other automatic family, existing nu values,200screening and400mean cap, all
score/tie/scale/partition policies. The optimizer minimizes fitting loss along
the chosen direction before shrinking by nu=.02; its effective update changes,
even though the numeric shrinkage parameter is preserved.

Guard-only reduced nonlinearZI MSE22.99949 to13.91765 and all six degrading
interaction substitutions disappeared, but its count-size upper excess3.25825%
missed1%. Screening400 reduced MSE only to12.30600, so do not select that
extension. Correctly instrumented line search reduced nonlinearZI MSE to
3.37098, count-size upper excess-0.28747% and width upper-3.27928%, coverage.901.
The genuine NBinomial-family diagnostic retains9/10 interactions, with paired
mean-error upper excess2.47481% vs its own baseline. Aggregate mean MSE there
is6.85283 versus6.45203; the paired-ratio criterion is not a ratio of aggregates.

Applying line search also to Poisson worsened the nonlinearPoisson control
(MSE.61322 vs.42366,3/10 interaction selections vs0). Therefore keep Poisson's
existing automatic rule. Keep ALL these outcomes, and the failed version2
instrument, with their original labels. Do not rename the all-count prototype
as the selected NBinomial-only algorithm. Validate an actual production run
against the selected per-family development components before fresh draws.

Production exposes an ordinary-mean step choice (auto/fixed/line_search), with
auto's family lookup in Rust and thin Python/R bindings. The same resolved step
must reach mean-form comparison, elimination, residual screening, interaction
and recovery CV, mean stopping, final/dropped-term fits and cost admission.
Family inference keeps its existing rule before this choice. LSS fitting and
Gaussian scale selection retain their existing update procedure. Compare
specifications must explicitly forward its step option to the existing
core-backed evaluator; no global research monkeypatches enter the product.

The original acceptance targets and reserved6799100–6799149/count and
6799300–6799349/Gaussian cohorts remain unchanged and unread. Commit the final
validated production source, recorder and complete analysis before generating
these observations. Reuse old pilot datasets only as development. A failed fresh
gate leaves#679open, and failed Gaussian sharp-mean attribution remains#629.
