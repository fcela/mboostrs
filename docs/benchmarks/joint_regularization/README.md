# Joint regularization pilot

This study investigates whether explicit penalties can separate term suppression from smoothness in Gaussian additive models. It uses the existing Rust REML solver and introduces no product behavior changes. The experiment and decision rules were frozen in [PROTOCOL.md](PROTOCOL.md) before the substantive run.

## Results

The additional penalty helps on sparse signals and correlated substitutes, but fails the prespecified across-scenario accuracy gate. It should not replace the current default. All 320 fits completed, all joint fits reported convergence, and no tuned boosting fit selected the 3,000-step cap.

Mean test signal MSE over 20 seeds; lower is better. The noise variance is 1. These errors compare predictions with the noiseless conditional mean, so they exclude irreducible observation noise.

| Scenario | Tuned boosting | Selected-term REML | All-term REML | Double-penalty REML |
|---|---:|---:|---:|---:|
| Sparse smooth | 0.02975 | 0.02870 | 0.02945 | **0.02188** |
| Correlated substitute | 0.09063 | 0.04733 | 0.04730 | **0.03878** |
| Dense smooth | 0.04617 | **0.04238** | **0.04238** | 0.05991 |
| Null | **0.00310** | 0.01162 | 0.01808 | 0.00670 |

The paired double-penalty-minus-boosting MSE differences are -0.00787 (SE 0.00175), -0.05186 (0.00326), +0.01374 (0.00367), and +0.00361 (0.00062), respectively. Each SE is the standard deviation of the 20 seed-level paired differences divided by sqrt(20); it describes Monte Carlo precision within this scenario, not transfer to other problems.

Median training seconds, including tuning where applicable. The selected-term refit reuses the same tuned fit but is charged its complete tuning cost plus refit time. Prediction and effect reporting are excluded from these timings.

| Scenario | Tuned boosting | Selected-term REML | All-term REML | Double-penalty REML |
|---|---:|---:|---:|---:|
| Sparse smooth | 0.01460 | 0.02324 | 0.01324 | 0.02158 |
| Correlated substitute | 0.05747 | 0.07121 | 0.01433 | 0.02490 |
| Dense smooth | 0.02604 | 0.04187 | 0.01176 | 0.01812 |
| Null | 0.00949 | 0.01102 | 0.01509 | 0.02025 |

The correlated-signal gain is largely available from ordinary joint refitting: its MSE is already 0.04733 versus boosting's 0.09063. Adding the null-space penalty improves it further to 0.03878. On the dense scenario the same extra penalty increases MSE by 30% relative to boosting and 41% relative to ordinary joint REML. This isolates a consequential regularization choice even when the basis and optimizer machinery are shared.

The null result also matters: REML refitting after selection increases error relative to the original boosted estimate. The selected support and the original stopping rule are not a guarantee that removing the path's shrinkage improves prediction.

Double-penalty fits had, on average, 3.20, 3.40, 0.10 and 5.80 terms with EDF below 0.1 in the four scenarios. At cutoffs 0.01 / 0.5 those counts were 3.20 / 3.40, 3.40 / 3.85, 0.10 / 0.10 and 5.65 / 6.40. These counts are descriptive, not exact support recovery; the raw artifact also records each term's effect RMS.

## Recommendation

Keep boosting as the default and defer a new group-norm solver. The promising next experiment is training-only selection between ordinary joint REML, double-penalty REML and boosting, scored on common inner folds and evaluated on new outer samples and generator families. Include an intercept-only candidate. Do not choose the winner using the four published test scenarios or add a hand-coded sparse/dense rule from these results.

The direct `gam()` entry point landed while this study was running. The smallest possible product extension is therefore an explicit null-space-shrinkage option on that entry point, using the existing core. This pilot establishes feasibility and a conditional benefit, not a justified automatic selection policy. Any automatic policy must recover the sparse gains without the observed dense and null losses before it can pass #462's gate.

The experiment covers Gaussian responses, eight smooth predictors and 600 training rows. It does not establish high-dimensional scaling, external-library superiority, distributional fitting performance or interval coverage. The 20 seeds reuse each generator family; confirmation needs different signal shapes and correlation structures.

## Statistical comparison

An ordinary spline penalty penalizes curvature but leaves a linear function unpenalized after removing the intercept. Increasing its smoothing parameter therefore approaches a straight line, which can still fit noise. A second penalty on that remaining null space allows the entire term to become negligible. This construction is described in [mgcv's term-selection documentation](https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/gam.selection.html) and [Marra and Wood (2011)](https://doi.org/10.1016/j.csda.2011.02.004).

Boosting regularizes differently. For one fixed Gaussian smoother S, repeated updates give the fitted-value operator H_m = I - (I - nu*S)^m. Its shrinkage in an eigendirection with eigenvalue s is 1 - (1 - nu*s)^m. Under stable step sizes this approaches one as m grows. The stopping iteration controls smoothing and, with competing learners, term inclusion together. The pilot compares this statistical procedure with joint smoothing-parameter estimation; it does not compare optimizers of an identical objective.

A new group-norm sparse additive solver would provide exact zero groups, but needs another optimization and tuning system. The algorithm of [Haris, Simon and Shojaie (2022)](https://www.jmlr.org/papers/v23/20-108.html) uses structural seminorms; its smooth-then-scale step cannot simply be applied to the current quadratic spline penalty and assumed to solve the same objective. The existing multiple-penalty REML engine provides a smaller first experiment.

## Implementation

The research runner constructs the second penalty in the constrained coefficient space and lifts it back to the raw basis. All coefficient fitting and smoothing-parameter optimization run in `mboostrs-core`. The NumPy eigendecomposition is confined to the benchmark. Tests check the penalty geometry, agreement of direct native REML with the public refit, and independence of the all-term refit from the initial boosting budget.

Finite smoothing parameters produce small effects rather than exact exclusions. Effective degrees of freedom (EDF) and effect root-mean-square magnitude describe suppression; an EDF threshold is not a hypothesis test or a false-discovery guarantee. The study estimates prediction error and computational cost, not confidence-interval coverage.

## Reproduction

From the repository root, with its release extension and Python environment available:

```sh
RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=python:. .venv/bin/python -m benches.probe_joint_regularization \
  --out docs/benchmarks/joint_regularization/results.json
```

The artifact records the runner hash, data hashes, source revision, native build identity, seeds, versions and thread settings. The native build used for the recorded pilot predates the research branch but has identical Rust source files; its distinct build revision remains visible in the artifact. The runner verifies the complete native source digest before fitting; the baseline amendment in the protocol records the discarded mismatched run. Timings are descriptive measurements on a shared machine.
