# Joint regularization pilot

This pilot tests whether separating term inclusion from smoothness is promising enough to develop further under issue #462. It compares four Gaussian estimators using the existing Rust fitting engines. No product API or default changes are proposed.

## Estimators

1. Component-wise boosting, with three-fold CV choosing the stopping iteration; learning rate 0.1, initial path 100, maximum 3,000 steps.
2. Joint REML refitting of the terms selected by estimator 1. Its total time includes selection and tuning.
3. Joint REML fitting of all eight prespecified terms.
4. The same joint fit with a second, independently estimated penalty on each smooth's remaining unpenalized directions.

Every term is an uncentered cubic `bbs(x, df=4, knots=12)`. The boosting `df` sets its base-learner penalty; the joint fits estimate smoothing parameters. These are different statistical estimators, not different optimizers of one objective. All four use the same raw basis span, response and training rows. Test responses and noiseless means are unavailable to fitting or tuning.

For the extra penalty, let B and K be a term's raw design and roughness penalty. Let C have orthonormal columns spanning the complement of B'1, the fitted sum-to-zero constraint. If U0 spans the null space of C'KC, supply P0 = C U0 U0' C' as the second raw penalty. This penalizes only the constrained null space and leaves the global intercept free. Verify that it annihilates the constrained roughness range. A raw-basis null projector need not have that property after imposing a data-dependent constraint.

## Data and replication

There are 600 training rows, 2,000 independent test rows, eight predictors and independent Gaussian noise with standard deviation 1. Twenty seeds, 1000 through 1019, are used in each scenario:

- Sparse smooth: independent uniform(-1,1) predictors; mean sin(3*x0) + x1² + 0.5*x2.
- Correlated substitute: normal predictors, x3 = 0.9*x0 + sqrt(0.19)*z with independent standard-normal z; the same mean as above. x3 has no direct term but predicts an active covariate.
- Dense smooth: independent uniform(-1,1) predictors; mean sum_j sin((1+j mod 3)*xj)/sqrt(8).
- Null: independent uniform(-1,1) predictors; zero mean.

A separate tiny smoke run may check interfaces and numerical construction. Its rows are not included in the pilot. The simulation settings and estimator rules remain frozen once substantive runs start. This is an exploratory comparison of four prespecified scenarios, not confirmation on unseen generator families.

## Measurements

Record signal MSE against the noiseless test mean, its paired seed-level differences, training time including CV, core convergence status, warnings, failures, stopping-boundary hits, per-term EDF and effect RMS. Signal MSE measures estimation error rather than irreducible observation noise. Report timing medians; a shared machine limits timing conclusions.

EDF cutoffs of 0.01, 0.1 and 0.5 describe negligible terms, not exact support selection or false-discovery control. Report effect magnitudes alongside EDF. Never compare REML criteria across models with different fixed-effect or penalty null spaces.

The primary reference is tuned boosting; selected-term refitting is also retained as an existing workflow. An estimator is a candidate for follow-up if mean signal MSE is no more than 5% worse on each non-null scenario, its median total training time is at most twice the reference, and at least 95% of fits converge. These are engineering screening thresholds, not hypothesis tests. Report the null scenario separately because relative differences around zero are unstable. Failures and boundary optima qualify any decision; they are not removed from the record.

## Scope and reproduction

All coefficient fitting and REML estimation run in the current Rust core. NumPy construction of the experimental penalty stays in the benchmark runner. Correctness checks compare ordinary native REML with the public refit and verify independence from the seed model's boosting iteration count.

The artifact records runner and data hashes, source revision, native build identity, versions and thread settings. The runner must record a native build from another revision explicitly; source equivalence of the Rust crates must be checked before using it. The initial environment has a release extension built at dabd8f6; its Rust sources match baseline ede8121.

This pilot cannot establish confidence-interval coverage, high-dimensional scaling, distributional-family performance, or superiority over external libraries. It does not implement a group-norm sparse solver. Those extensions depend on the pilot's result.

## Baseline amendment

Before retaining results, a native-build check found that the copied extension was built at `82dbd576`, not the initially inspected `dabd8f6`. A first run against mismatched Python/native sources was discarded. The research branch was advanced to merged revision `82dbd576`, and the runner now refuses execution unless the complete native source digest matches the extension. The four estimators, data-generating processes, seeds, metrics and screening thresholds were unchanged. Only the consistent rerun is published. Direct `gam()` support had also landed by this baseline; the pilot still calls its underlying native engine to vary the second penalty explicitly.

The build digest recursively includes generated Rust files under the ignored R-crate `target/` directory. Those files were copied from the native build workspace into this isolated worktree to reproduce the digest; they are not part of the MR. The tracked Rust sources also match `82dbd576` exactly.
