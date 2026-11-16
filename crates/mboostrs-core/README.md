# mboostrs-core

Pure-Rust component-wise functional gradient boosting engine, re-implementing
the semantics of R's [`mboost`](https://cran.r-project.org/package=mboost)
(2.9-11) and [`gamboostLSS`](https://cran.r-project.org/package=gamboostLSS)
(2.2-0). The crate is the numerical core of the `mboostrs` package; the Python
bindings live in [`mboostrs-py`](../mboostrs-py) and the R bindings in
[`mboostrs-r`](../mboostrs-r).

## Overview

`mboostrs-core` provides:

- The `fit`, `fit_with`, `fit_profiled`, and `fit_selected` boosting entry points
- Cross-validation via `cv` / `cvrisk`
- Component-wise model selection via `stabsel`
- The 22 standard families (`Gaussian`, `Binomial`, `Poisson`, `PropOdds`,
  `CoxPH`, `Weibull`, `Loglog`, `Lognormal`, `Multinomial`, `NBinomial`,
  `Hurdle`, `GaussReg`, `GammaReg`, `QuantReg`, `ExpectReg`, `Huber`, `AdaExp`,
  `HingeLoss`, `Laplace`, and the three ranking losses `Auc`, `Cindex` and
  `Gehan`, which score orderings rather than a density and so refuse
  `logLik`) plus 25 distributional (`*LSS`) families
- Base learners: `bols`, `bbs`, `bmono`, `btree`, `bmrf`, `bns`, `bspatial`,
  `bkernel`, `buser`, and Kronecker tensor products. `brandom` and `brad` are
  built on these rather than being separate `prepare_*` entry points --
  `brandom` on block `bols`, `brad` on `buser` with a Matérn design
- Path coefficients and prediction (`predict_glmboost`, `predict_glmboost_path`)
- Versioned route/fallback reports plus on-demand `PathRow` iteration evidence

The hot path uses a covariance / Gram-update inner loop for Gaussian smooth
learners (the glmnet covariance trick), avoiding a residual re-projection each
iteration. Non-Gaussian families fall back to the per-iteration general path.

## Quick example

```rust
use mboostrs_core::{
    control::BoostControl, families::Gaussian, learners::prepare_scalar_bols, try_fit,
};
use ndarray::Array1;

fn main() -> Result<(), mboostrs_core::MboostrsError> {
    let n = 400usize;
    let mut rng = 42u64;
    let mut next = || -> f64 {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        (rng as f64) / (u64::MAX as f64)
    };

    let x: Array1<f64> = (0..n).map(|_| next() * 2.0 - 1.0).collect();
    let y: Array1<f64> = (0..n).map(|i| 1.5 * x[i] + (next() - 0.5) * 0.6).collect();
    let w = Array1::ones(n);

    let l1 = prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None);
    let mut learners = vec![l1];

    let ctrl = BoostControl::new(100, 0.1).unwrap();
    let mut fam = Gaussian;
    let path = try_fit(&mut learners, y.view(), w.view(), &mut fam, &ctrl, None)?;

    println!("selected  : {:?}", path.selected);
    println!("final risk: {:.6}", path.risk.last().unwrap());
    println!("fitted[0] : {:.6}", path.fitted[0]);
    println!("route      : {}", path.report.route.as_str());
    println!("first row  : {:?}", path.path_rows().first());
    Ok(())
}
```

Run the checked-in version with:

```bash
cargo run -p mboostrs-core --example quickstart
```

## Features

- `panic = "unwind"`: any panic in the engine is caught by the PyO3 layer and
  converted to a Python exception. Never change this to `abort` — it SIGABRTs
  the interpreter.
- Fit, fold construction, cross-validation, cyclic/non-cyclic LSS, and
  stability-bound calculations have `try_*` entry points that return typed
  errors. The Rust API also retains panicking wrappers for callers using the
  earlier interface; Python and R use the checked entry points.
- `FitReport` records prepared learner/solver counts, retained factor-cache
  bytes, Gaussian cross-Gram bytes, exact winning-score ties, active nuisance-
  calibration calls, and tracked iterative calibration stop outcomes.
  These values are derived from existing state and therefore remain available
  with timing instrumentation disabled. `try_fit_profiled` clones the supplied
  control and enables summary timing without mutating caller state. Its
  nanosecond fields separate core setup, Gaussian cross-Gram construction,
  boosting iterations, final reconstruction, and total core-fit time.
  Frontends can use `PreparationTimer` to separate design/basis/penalty and
  weighted-Gram work from cached solver factorization. The `basis_ns`,
  `factorization_ns`, and `binding_ns` fields are intentionally empty for pure
  Rust fits; Python and R populate them as exclusive frontend intervals.
- `GramSolver::try_new` accepts positive-definite systems through Cholesky,
  verifies the residual of indefinite LU systems, explicitly tags singular
  systems solved with diagonal jitter, and rejects nonfinite/nonsquare input.
  Regularization count and maximum jitter flow into `FitReport`. Cached LU
  slice solves use packed factors and caller-owned output memory, with a
  counting-allocator test enforcing zero steady-state heap allocations.
- Validation rejects empty data/learners, non-finite responses and offsets,
  negative weights, incompatible shapes, and folds without both in-bag and
  positive-weight out-of-bag observations. An all-zero weight vector remains
  defined for a direct fit (finite no-op path), but is rejected for bootstrap
  fold generation because its sampling distribution is undefined.
- R type-7 quantiles, type-1 (randomized PIT), and weighted-quantile types
  are reproduced bit-for-bit.

## Documentation

- The full book (HTML/PDF), methodology chapters, and per-chapter API notes
  live in the workspace `docs/book/`.
- Per-module rustdoc lives in each `src/*.rs` file.

## Testing

```bash
cargo test -p mboostrs-core           # 268+ unit tests
cargo test --doc -p mboostrs-core     # doctests (gated by CI)
cargo bench -p mboostrs-core          # Criterion micro-benchmarks
```

The instrumentation acceptance benchmark includes a feature-gated
report-free baseline that is unavailable to ordinary callers:

```bash
cargo bench -p mboostrs-core --bench bench_core \
  --features instrumentation-baseline -- instrumentation_overhead
```

## License

MIT, same as the workspace.
