(record-engine-and-estimation)=
# Engine optimizations, double-penalty REML, and estimation–discovery decomposition

This record summarizes the engine and estimation methodology studies supporting [](../../benchmarks/joint_regularization_pilot.ipynb) and [](../../benchmarks/paired_estimation_discovery.ipynb).

## Duplicate-design Gram-block sharing

In the Gaussian Gram boosting route (`try_fit_gaussian_gram`), every pair $(j, w)$ of base learners previously paid a full $O(n p^2)$ weighted cross-Gram pass even when two learners shared an identical design matrix and row-weight vector. Identifying earlier learners with bit-identical designs (via pointer-equality fast path and exact bit comparison) allows duplicate learners to reuse both diagonal and cross-Gram blocks directly, eliminating redundant matrix multiplications without altering floating-point arithmetic.

## Double-penalty REML pilot

**Design.** To test whether separating term suppression from smoothness selection improves Gaussian additive models, four estimators were compared on 20 independent seeds ($n_{\text{train}} = 600$, $n_{\text{test}} = 2{,}000$, $p = 8$, Gaussian noise $\sigma = 1$) across four scenarios—sparse smooth, correlated substitute, dense smooth, and pure null: (1) component-wise boosting tuned by 3-fold CV ($\nu = 0.1$, max 3,000 steps), (2) joint REML refit of boosting-selected terms, (3) joint REML on all 8 terms, and (4) double-penalty REML adding an independently estimated shrinkage penalty on each smooth's unpenalized null space.

**Outcomes.** Mean test signal MSE (error against the noiseless conditional mean; lower is better):

| Scenario | Tuned boosting | Selected-term REML | All-term REML | Double-penalty REML |
|---|---:|---:|---:|---:|
| Sparse smooth | 0.02975 | 0.02870 | 0.02945 | **0.02188** |
| Correlated substitute | 0.09063 | 0.04733 | 0.04730 | **0.03878** |
| Dense smooth | 0.04617 | **0.04238** | **0.04238** | 0.05991 |
| Pure null | **0.00310** | 0.01162 | 0.01808 | 0.00670 |

Paired differences (double-penalty REML minus boosting) were $-0.00787$ (SE 0.00175) on sparse smooth, $-0.05186$ (SE 0.00326) with a correlated substitute, $+0.01374$ (SE 0.00367) on dense smooth, and $+0.00361$ (SE 0.00062) on the null scenario. Because double-penalty REML degraded dense smooth signals by 30%, it failed the prespecified across-scenario gate and is not the default; choosing among boosting, joint REML, and double-penalty REML on shared inner CV folds is recommended instead.

## Estimation–discovery decomposition

**Design.** A paired three-arm study evaluated four Gaussian generators (`linear_factors`, `smooth_additive`, `kinked`, and `interaction`) across 8 replicates (32/32 completed) on identical train/validation/test partitions: `A_supplied` (true structure supplied), `A_partial` (all smooth main effects, no interactions), and `B_discovery` (automatic `autospec` at budget 32 with `interactions="adopt"`). Excess risk is expressed as a fraction of $\operatorname{Var}(y)$ above Bayes risk.

**Outcomes.** Median excess risk across 8 replicates (units of $\operatorname{Var}(y)$; lower is better):

| Panel | Estimation excess | Discovery gap | Partial gap | End-to-end excess |
|---|---:|---:|---:|---:|
| `linear_factors` | 0.0003 | 0.0003 | 0.0010 | 0.0006 |
| `smooth_additive` | 0.0014 | 0.0083 | 0.0009 | 0.0087 |
| `kinked` | 0.0005 | 0.0008 | 0.0027 | 0.0009 |
| `interaction` | 0.0282 | 0.0577 | 0.0581 | 0.0841 |

On additive and kinked generators, both estimation excess and discovery gap sit below 1% of $\operatorname{Var}(y)$ (structural recall 0.75–1.00, FDP $\le 0.25$). On the centered product-interaction generator, end-to-end excess is 8.4% of $\operatorname{Var}(y)$, roughly two-thirds from missed interaction discovery (5.8%, missed in 7 of 8 replicates at budget 32) and one-third from tensor estimation error (2.8%).

## Scope of deterministic diagnostic metadata

Three deterministic numerical-verification audits (`conformal-predictor-coding/1`, `conformal-response-units/1`, and `binomial-prediction-scale/1`) inherited a generic simulation-uncertainty metadata block from a shared evidence helper. Those audits evaluate a single shared deterministic fixture (or three fixed splits of one dataset) to verify exact coordinate and reference agreement; they do not provide independent Monte Carlo standard errors, coverage estimates, or controlled speed comparisons.

**Audit artifacts.** Original protocols and records live in `docs/benchmarks/gram_duplicate_designs.md`, `docs/benchmarks/evidence_metadata_scope.md`, `docs/benchmarks/joint_regularization/`, `docs/benchmarks/paired_estimation_discovery_protocol.md`, and `docs/benchmarks/paired_ab/`.
