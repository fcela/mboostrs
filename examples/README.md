# mboostrs examples

Examples demonstrating the mboostrs API. Organized by language:

## Python (`python/`)

| # | File | Topics covered |
|---|------|----------------|
| 01 | `01_package_tour.ipynb` | End-to-end: glmboost, gamboost, cvrisk, families, predict, Altair plots |
| 02 | `02_r_parity.ipynb` | Numerical parity with R's mboost (requires rpy2) |
| 03 | `03_performance.ipynb` | Performance methodology, legacy visualization, and local scaling |
| 04 | `04_families.ipynb` | All 18 families: Gaussian, Binomial, Poisson, NBinomial, GammaReg, GaussReg, Laplace, Huber, Quantile, Expectile, AdaExp, PropOdds, CoxPH, Weibull, Loglog, Lognormal, Multinomial, HingeLoss |
| 05 | `05_base_learners.ipynb` | All 10+ base learners: bols, bbs, bmono, bns, bspatial, bmrf, brandom, btree, bkernel, bconstrained |
| 06 | `06_model_selection.ipynb` | CV (kfold/bootstrap/subsampling), AIC (corrected/classical/gMDL), stability selection, model comparison, grid search |
| 07 | `07_sklearn_api.ipynb` | scikit-learn wrappers: GLMBoostRegressor, GLMBoostClassifier, GAMBoostRegressor, GAMBoostClassifier; cross_val_score, GridSearchCV, Pipeline |
| 08 | `08_gamboostlss.ipynb` | Distributional (GAMLSS) boosting: GaussianLSS, GammaLSS, StudentTLSS, BetaLSS; per-parameter formulas, per-parameter mstop, cyclic vs non-cyclic, LSS CV, LSS stabsel |

### Running Python examples

```bash
# Notebooks (requires jupyter):
uv run jupyter notebook examples/python/

# Or execute top-to-bottom non-interactively:
uv run jupyter nbconvert --to notebook --execute examples/python/04_families.ipynb
```

## R (`r/`)

| # | File | Topics covered |
|---|------|----------------|
| 01 | `01_glmboost.R` | glmboost, CV, AIC, Poisson/Binomial families, variable importance |
| 02 | `02_gamboost.R` | gamboost with bbs, mixed terms, bmono, brandom, prediction |
| 03 | `03_gamboostLSS.R` | gamboostLSS with GaussianLSS, per-parameter formulas, GammaLSS, prediction |

### Running R examples

```bash
Rscript examples/r/01_glmboost.R
Rscript examples/r/02_gamboost.R
Rscript examples/r/03_gamboostLSS.R
```

Requires R packages `mboost` and `gamboostLSS`.

## Rust (`../crates/mboostrs-core/examples/`)

| Example | Topics covered |
|---------|----------------|
| `bench_gamboost.rs` | Benchmarks gamboost with 2x bbs; general vs Gram fast path |
| `bench_glmboost.rs` | Benchmarks glmboost at scale |
| `proto_gram_update.rs` | Prototype: naive vs Gram-update boosting loop |

### Running Rust examples

```bash
cargo run --example bench_gamboost
cargo run --example bench_glmboost
cargo run --example proto_gram_update
```
