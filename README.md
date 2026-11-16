# mboostrs

mboostrs is a statistical modeling library with a Rust engine and Python and
R interfaces. It provides additive and distributional regression, model
selection and refitting, effect estimation, prediction intervals, and causal
analysis. Models can be specified directly with formulas or developed through
specification-search tools.

Its methods and interfaces build on the work of the authors of
[`mboost`](https://cran.r-project.org/package=mboost),
[`gamboostLSS`](https://cran.r-project.org/package=gamboostLSS), and
[`gamlss`](https://cran.r-project.org/package=gamlss). Familiar R formula
conventions are retained where practical to simplify migration. mboostrs is
an independent implementation; comparisons against these packages validate
specific methods and operations.

The Python API includes:

- **Regression:** component-wise boosting and penalized additive models,
  with linear, smooth, constrained, grouped, spatial and interaction terms.
- **Distributional models:** predictors for multiple response-distribution
  parameters, with quantiles, centiles and distributional diagnostics.
- **Model development:** specification search, cross-validation, stability
  selection, and joint refitting with smoothing parameters estimated by
  restricted maximum likelihood (REML).
- **Interpretation and prediction:** partial effects, contrasts, marginal
  effects, residual diagnostics, and conformal prediction intervals.
- **Causal analysis:** cross-fitted average treatment effects, treatment-effect
  models, instrumental-variable estimation, and overlap and sensitivity checks.
- **Integration:** Polars data, scikit-learn estimators, saved model artifacts,
  and structured results for interactive or agent-assisted analysis.

Support varies by model, operation and language interface. The
[capability contract](docs/book/appendix/capabilities.ipynb) records that
coverage; the [release limitations](docs/release-limitations.md) describe
current limits of automated search, causal intervals and cross-language
inference.

```python
import numpy as np, polars as pl
import mboostrs as mb

rng = np.random.default_rng(0)
n = 400
x, z = rng.normal(size=n), rng.normal(size=n)
y = 1.5 * x + np.sin(2 * z) + rng.normal(scale=0.3, size=n)
data = pl.DataFrame({"x": x, "z": z, "y": y})

fit = mb.gamboost(
    "y ~ bols(x) + bbs(z, df=4)",          # linear term + P-spline
    data=data,
    family=mb.Gaussian(),
    control=mb.boost_control(mstop=200, nu=0.1),
)

# early-stopping iteration by cross-validated risk
cvr = fit.cvrisk(type="kfold", B=5, seed=42)
m_opt = cvr.optimal_mstop

yhat = fit.predict(newdata=data, mstop=m_opt, type="response")
```

The example fits an additive model with a linear term in `x` and a P-spline in
`z` for 200 iterations with step length `nu=0.1`, chooses the iteration with
the lowest risk over five cross-validation folds, and predicts at that
iteration on the scale of `y`.

<!-- capability-summary:start -->
The generated capability contract records **53 family/helper rows**, **15 learner rows**, and **150 explicit validation records**. For fitting, 29 Python rows and 13 R rows are externally verified; 4 Python rows are explicitly unsupported rather than inferred from missing code. See the [generated capability and validation matrix](docs/book/appendix/capabilities.ipynb).
<!-- capability-summary:end -->

## Install

Requires Python ≥ 3.11. Wheels are built for Linux, macOS and Windows.

```bash
pip install mboostrs
```

This installs NumPy and Polars and nothing else, which is enough for fitting,
prediction, cross-validation, stability selection and every table. Charts need
Altair: `mboostrs.plot` and the `model.plot*()` methods import it. Altair is the
`plot` extra rather than a base dependency because importing it took 509 ms of
a 559 ms `import mboostrs`, a cost every user paid whether or not they drew a
chart:

```bash
pip install "mboostrs[plot]"
```

Without the extra, importing `mboostrs.plot` raises an `ImportError` that names
the extra.

Released wheels use the portable instruction-set baseline of each platform. For
local benchmarks on known hardware you can enable CPU-specific instructions
with `RUSTFLAGS="-C target-cpu=native"`; release and CI builds do not.

### Python from source

Building from source needs a Rust toolchain and Python ≥ 3.11:

```bash
pip install maturin
maturin develop --release
```

Other optional extras: `mboostrs[sklearn]` (scikit-learn estimators; needs
scikit-learn ≥ 1.3) and `mboostrs[r-parity]` (rpy2, for the R-parity test
suite).

### Rust crate

The core engine is published on crates.io as a separate crate:

```bash
cargo add mboostrs-core
```

### R package

The R package calls the Rust engine through extendr. Building it needs a Rust
toolchain and R ≥ 4.0:

```r
# install.packages("devtools")
devtools::install_github("fcela/mboostrs", subdir = "mboostrs")
```

## Models

`glmboost` fits component-wise **linear** models, with one `bols` term per
predictor. `gamboost` fits **additive** models with arbitrary base learners,
chosen per term in the formula.

In component-wise boosting, each iteration fits the candidate terms to the
negative gradient of the loss and adds a fraction `nu` of the best-fitting
term to the model. The stopping iteration, `mstop`, controls complexity and
variable selection. For squared-error loss, the negative gradient is the
current residual.

```python
# linear / GLM-style boosting
glm = mb.glmboost("y ~ x + z", data=data, family=mb.Gaussian(),
                  control=mb.boost_control(mstop=300, nu=0.1))

# additive boosting with mixed terms
gam = mb.gamboost("y ~ bols(x) + bbs(z, df=4)", data=data,
                  family=mb.Gaussian(), control=mb.boost_control(200, 0.1))
```

> **Bare names are smooths in `gamboost`.** In a `gamboost` formula a bare
> predictor name becomes `bbs(name)`, a P-spline smooth, so `"y ~ x"` fits a
> curve. Write `bols(x)` for a linear term. `glmboost` fits only linear terms,
> so bare names are linear there.

The formula accepts `y ~ .` (every column), term removal (`. - z`), inline
transforms (`log(x)`, `scale(x)`, `I(x**2)`), categorical predictors for
`brandom`/`bmrf`, and the `%X%` interaction operator
(`bbs(x) %X% brandom(g)`). A transform that estimates parameters from the data,
such as the training mean and standard deviation in `scale(x)`, stores them and applies the same
values at prediction time. Missing values raise an error by default;
`na_action="omit"` drops the affected rows. `fit.continue_to(N)` continues
boosting a fitted model up to a larger `mstop`, `N`. When a model has many
candidate terms, `subsample=f` evaluates a random fraction `f` of the base
learners at each iteration. The
[**Working with real data**](docs/book/usage/real_data.ipynb) chapter covers
these options.

A fitted model exposes:

| Attribute / method | Meaning |
|--------------------|---------|
| `fit.fitted_` | fitted values at the final iteration |
| `fit.risk_` | empirical risk path (length `mstop+1`) |
| `fit.selected` | base learner selected at each iteration |
| `fit.predict(newdata=, mstop=, type=)` | predict (`type="link"` or `"response"`) |
| `fit[:k]` | the model truncated to `k` iterations |
| `fit.coef()`, `fit.varimp()`, `fit.confint()`, `fit.residuals()`, `fit.extract(...)` | inference (below) |
| `fit.aic(method=...)` | AIC / corrected-AIC path and its optimum |
| `fit.execution_report()` | numerical route, fast-path decisions, dimensions, threads, fallbacks, and optional timings |
| `fit.path_frame(output=...)` | one auditable row per learner update, as records, pandas, or polars |
| `fit.plot_risk()`, `fit.plot_varimp()`, `fit.plot_selected()`, `fit.plot_residuals()` | quick diagnostic plots |

`type="link"` returns predictions on the scale of the additive predictor, and
`type="response"` returns them on the scale of `y`.

### Base learners

Base learners are specified inside the formula. Common choices are listed
below; the [capability contract](docs/book/appendix/capabilities.ipynb)
records the supported learners and their operations.

| Learner | Role |
|---------|------|
| `bols(x)` | linear / ordinary least-squares term |
| `bbs(x, df=4)` | penalized B-spline (P-spline), smooth term |
| `bmono(x, constraint="increasing")` | monotone / shape-constrained P-spline |
| `btree(x1, x2, ...)` | regression tree term |
| `brandom(g)` | random-effect / ridge-penalized factor |
| `bmrf(region, bnd=...)` | Markov random field (areal / spatial) |
| `bspatial(x1, x2, df=6)` | tensor-product spline (2-D surface) |

### Families

A family sets the loss that boosting minimizes. The families include
`Gaussian`, `Binomial`, `Poisson`, `GammaReg`, `Laplace`, `Quantile(tau=...)`,
`Expectile(tau=...)` and `Huber`; the capability contract lists the rest. A
classification example:

```python
# Binomial needs a 0/1 (or -1/+1) response, not the continuous `y` above.
binary = data.with_columns((data["y"] > 0).cast(int).alias("hit"))
clf = mb.glmboost("hit ~ x + z", data=binary, family=mb.Binomial(),
                  control=mb.boost_control(100, 0.1))
proba = clf.predict(newdata=binary, type="response")  # P(hit = 1)
```

## Specification search and refitting

[`autospec`](docs/book/api/functions/autospec.md) proposes a family, formula
and stopping iteration from the data and the supplied validation design.
The recommendation can be inspected and fitted with `recommendation.fit(data)`.
[`open_search`](docs/book/api/functions/open_search.md) supports an interactive
search with retained comparisons and fit accounting. Separate evaluation
rows are needed to assess the selected procedure's predictive performance.

[`gam`](docs/book/api/functions/gam.md) fits a penalized additive model
with REML smoothing directly. After boosting,
[`gam_refit`](docs/book/api/functions/gam_refit.md) jointly refits selected
terms; [`gamlss_refit`](docs/book/api/functions/gamlss_refit.md) provides
refitting for supported distributional models. Refit intervals are
conditional on the selected specification; they do not account for the
whole preceding search.

## Distributional models

`glmboostLSS` and `gamboostLSS` fit an additive predictor for each parameter
of a response distribution, such as its location and scale. Formulas can
differ between parameters. The fitted model provides distribution-based
prediction intervals and centiles; quantile residuals, probability integral
transforms and worm-plot data support checking the fitted distribution.
See the [distributional modeling guide](docs/book/usage/gamboostlss.ipynb).

## Model selection: `cvrisk`

`cvrisk` refits the model on resampled data (bootstrap, k-fold or subsampling),
computes the out-of-sample risk at every iteration, and reports the iteration
with the lowest mean risk as `optimal_mstop`. Truncating the model at that
iteration gives the early-stopped fit.

```python
cvr = fit.cvrisk(type="bootstrap", B=25, seed=1)   # also "kfold", "subsampling"
m_opt = cvr.optimal_mstop                               # risk-minimizing iteration
final = fit[:m_opt]                                 # truncated model
```

## Inference

Each call returns one summary of the fitted model:

```python
fit.coef()                       # coefficients per selected base learner
fit.varimp(percent=True)         # variable importance (risk reduction share)
fit.confint(parm="x", level=0.95, B=1000)   # bootstrap confidence bands
fit.residuals()                  # response residuals y - mu; type="working" gives the negative gradient
fit.extract("design")            # design matrices / penalties / etc.
```

## Effects and causal analysis

[`prediction_contrast`](docs/book/api/functions/prediction_contrast.md)
compares model predictions under specified changes in predictors.
[`average_marginal_effect`](docs/book/api/functions/average_marginal_effect.md)
summarizes prediction derivatives over observations. These describe the
fitted model; a causal interpretation requires additional assumptions.

[`aipw`](docs/book/api/functions/aipw.md) estimates average treatment effects
using cross-fitted outcome and propensity models.
[`effect_model`](docs/book/api/functions/effect_model.md) estimates how a
treatment effect varies with covariates, and
[`iv_effect`](docs/book/api/functions/iv_effect.md) provides
instrumental-variable estimation. These functions require an explicit causal
specification. Overlap diagnostics and sensitivity analyses help assess the
conditions on which their estimates depend.

## Prediction intervals

`confint` describes uncertainty about the fitted effects. For an interval
around a new observation, `conformal_interval` gives distribution-free,
finite-sample marginal coverage (coverage averaged over new observations, not
at each value of the predictors), without assuming that the early-stopping
iteration was chosen well:

```python
ci = fit.conformal_interval(newdata=data, level=0.9, method="cv+")
ci.empirical_coverage(data["y"])          # ~0.9
```

Methods: `split` (constant width, one extra fit), `cv+` (every observation is
used for calibration), and `cqr` (conformalized quantile regression, width that
adapts to the predictors). For a distributional model, `glmboostLSS` and
`gamboostLSS` give `prediction_interval` and `centiles` directly from the
fitted distribution.

## scikit-learn interface

`mboostrs.sklearn` adapts the models to the scikit-learn estimator protocol, so
they work in `Pipeline`, `cross_val_score` and `GridSearchCV`:

```python
from mboostrs.sklearn import GAMBoostRegressor
from sklearn.model_selection import GridSearchCV

X = data.select(["x", "z"])                        # a frame, so names are kept
est = GAMBoostRegressor(df=4, nu=0.1)              # one bbs term per column
gs = GridSearchCV(est, {"mstop": [50, 100, 200]}, cv=5).fit(X, y)
```

Estimators: `GLMBoostRegressor`, `GAMBoostRegressor`, `TreeBoostRegressor`,
`GLMBoostClassifier`, `GAMBoostClassifier`, `TreeBoostClassifier`,
`LSSRegressor`, `SurvivalBoost`. The classifiers choose
the family from the number of classes: Binomial for two, Multinomial for more.
The regressors take a `family`, including `quantile` and `expectile` with a
`tau` level. The `GAM*` estimators also accept a full `formula=` in the
constructor for per-term control, and `clone` and `GridSearchCV` carry it.
`LSSRegressor.predict_interval` returns an interval whose width adapts to the
predictors. `SurvivalBoost` takes the scikit-survival structured `y`.
[`docs/sklearn.md`](docs/sklearn.md) explains how the formula and matrix
interfaces are reconciled.

## Serialization

Models can be pickled. `save_model` writes a JSON file with a schema version;
the file does not depend on the Python version or the compiled extension:

```python
mb.save_model(fit, "model.json")     # or fit.save("model.json")
fit2 = mb.load_model("model.json")   # predicts identically
```

## Execution reports: `execution_report` and `path_frame`

Every standard and LSS fit carries a versioned execution report. It records
which numerical route the fit used (the Gaussian Gram fast path, the packed
scalar kernel, the hybrid engine or the general engine) and, when a fast path
that looked eligible was not used, the reason:

```python
report = fit.execution_report()
print(report.route, report.fallback_reasons)
print(report.cholesky_solver_count, report.lu_solver_count)
print(report.regularized_solver_count, report.max_solver_jitter)
print(report.factor_cache_bytes, report.gram_cache_bytes)
print(report.calibration_count)
print(report.tie_selection_count)
print(report.calibration_converged_count, report.calibration_limit_count)
print(report.to_dict())                 # JSON-serializable evidence

path = fit.path_frame(output="records")
print(path[0])  # learner, risk reduction, coefficient norm, nuisance state

# Trace mode records phase timings and prints this same structured summary.
profiled = mb.gamboost(
    "y ~ bols(x) + bbs(z, df=4)", data=data,
    control=mb.boost_control(200, 0.1, trace=True),
)
```

`path_frame()` builds its rows on demand from the retained fit, so ordinary
fitting does not store a second iteration log. Its stable columns include the
one-based iteration, the parameter and cycle for LSS models, the learner
identity, the risk and risk reduction, the raw and learning-rate-scaled
coefficient norms, and the scalar or vector nuisance state (family parameters
estimated during boosting, such as a negative binomial dispersion). Use
`output="polars"` for a Polars frame, or `output="records"` for a serializable
result with no frame library.

Timing fields are excluded from report equality, so two reports with the same
route decisions compare equal in regression tests and after serialization. Slicing a model keeps its
route and changes only `completed_mstop`. In R, use `execution_report(fit)`,
`path_frame(fit)` and `boost_control(..., trace = TRUE)`.

Timing is opt-in: ordinary fits record the route and the counters but do not
read the clock. The nanosecond fields measure, separately, frontend design,
basis and penalty preparation; cached solver factorization; core setup;
Gaussian cross-Gram construction; boosting iterations; and the final
reconstruction of fitted values. `total_ns` covers the Rust fit itself. Python
and R also fill `binding_ns`, which holds only the input and output conversion
outside preparation and the core fit. The frontend intervals do not overlap:
`basis_ns + factorization_ns` is the measured learner-preparation time, and it
does not overlap `binding_ns`. A phase that a route does not run is `None` in
Python and `NA` in R; the compact text renderer shows it as zero.

The solver counts come from the prepared learners: scalar closed-form
learners, cached Cholesky factors and LU fallbacks are counted separately. A
singular finite system is solved with a regularized Cholesky factor, which the
report records; `max_solver_jitter` is the largest diagonal adjustment. The
solver constructor returns an error for nonfinite and nonsquare matrices.
`factor_cache_bytes` is the storage held by retained factorizations;
`gram_cache_bytes` is the full cross-Gram matrix built by the Gaussian fast
path (zero on other routes). `calibration_count` counts only active
family-specific nuisance calibration: adaptive Huber counts each call, and
Huber with a user-supplied `delta` counts none. `tie_selection_count` counts
exact ties for the winning score, which are resolved by choosing the learner
with the lowest index; no approximate tolerance is used and selection is not
changed. Iterative calibrators report how they stopped: PropOdds records
convergence, exhaustion of its five coordinate-descent sweeps, or a non-finite
calibration failure.

An LSS fit is cyclic when every distribution parameter is updated in turn at
each iteration, and non-cyclic when each iteration updates only the parameter
whose best learner reduces the risk most. For LSS models, `requested_mstop` and
`completed_mstop` count parameter updates: a cyclic two-parameter model with 10
updates for each parameter reports 20, and a 10-step non-cyclic model reports
10. This count also represents unequal per-parameter stopping values without
losing information. On a cyclic LSS path, the risk is recorded on the row that
completes a cycle, and the intermediate parameter-update rows have a missing
risk. A non-cyclic LSS path records the joint risk after every update.

## Documentation

The [Jupyter Book](docs/book/) has the full documentation, tutorials and case
studies:

- **Usage guides**: GLMBoost, GAMBoost, GAMLSS, Multinomial, base learners, families,
  model selection, stability selection, plotting, scikit-learn interface
- **Methodology**: algorithm derivations, loss functions, model comparison theory
- **Implementation**: architecture, engine internals, linear algebra kernels, parity notes
- **API reference**: compact function-by-function reference

Build the [HTML and PDF documentation](docs/book/README.md) with
`cd docs/book && bash build-html.sh` or `bash build-pdf.sh`. The rendered HTML,
the PDF and the notebook download bundle are build outputs and are not tracked
in git. The documentation workflow builds them on every push to `main`,
publishes the HTML to GitHub Pages, and attaches the PDF and the bundle to the
workflow run. The book is built with [MyST](https://mystmd.org/), not with the
obsolete `jupyter-book` build command. For a local preview that executes the
code, run `cd docs/book && bash start.sh`; it activates the matching Python
environment before MyST starts Jupyter.

Example notebooks are in [`examples/python/`](examples/python/):

| Notebook | Description |
|----------|-------------|
| [`01_package_tour.ipynb`](examples/python/01_package_tour.ipynb) | Walk through glmboost, cvrisk, gamboost, families, predict |
| [`02_r_parity.ipynb`](examples/python/02_r_parity.ipynb) | R parity test results across families and learners |
| [`03_performance.ipynb`](examples/python/03_performance.ipynb) | Benchmarks comparing mboostrs to R mboost |
| [`04_families.ipynb`](examples/python/04_families.ipynb) | 18 of the 22 families |
| [`05_base_learners.ipynb`](examples/python/05_base_learners.ipynb) | All 10+ base learners |
| [`06_model_selection.ipynb`](examples/python/06_model_selection.ipynb) | CV, AIC, stability selection, model comparison, grid search |
| [`07_sklearn_api.ipynb`](examples/python/07_sklearn_api.ipynb) | scikit-learn wrappers, `cross_val_score`, `GridSearchCV`, `Pipeline` |
| [`08_gamboostlss.ipynb`](examples/python/08_gamboostlss.ipynb) | Distributional (GAMLSS) boosting: `GaussianLSS`, `GammaLSS`, etc. |

### For AI agents

- `mboostrs.agent_guide()` returns the analysis workflow written for coding
  agents. It is the same text as
  [`docs/skills/analyzing-data-with-mboostrs/SKILL.md`](docs/skills/analyzing-data-with-mboostrs/SKILL.md);
  installed as an agent skill, that file is picked up automatically.
- `mboostrs.insights(df, response)` proposes ranked candidate findings, each
  marked as either measured on held-out folds or only detected, for the agent
  to verify before reporting.

## Performance

Each benchmark is a scenario in the catalog
[`benches/scenarios.json`](benches/scenarios.json), which declares the timed
boundary (what the timer includes), the seed, the dimensions and the thread
mode. [`benches/bench_all.py`](benches/bench_all.py) writes the raw
repetitions, robust dispersion and confidence intervals, digests of the inputs
and outputs, dependency, compiler and machine metadata, and whether memory and
allocations were measured. A cross-language speed ratio is reported only when
the fitted outputs of both implementations share a correctness digest at the
recorded tolerance.

The earlier 2026-06-24 headline table is no longer presented as current
evidence. Its CV row timed an already-fitted R model against a Python closure
that also ran the initial fit, and it compared serial R with Rust running in
parallel by default. The current runner excludes the initial fit on both sides
of the CV comparison and reports serial-against-serial and
default-against-default comparisons separately. A quick local check of the
benchmark contract:

```bash
pip install -e ".[benchmark]"  # Memray + R bridge used by allocation evidence
python benches/bench_all.py --quick --output benchmark.json
python benches/bench_python.py --quick --output benchmark-python.json
python benches/bench_thread_scaling.py --quick --output benchmark-threads.json
```

Each cross-language timing row has a separate allocation pass, run after
warm-up. Memray counts system and CPython allocator events across Python, Rust
and Rayon threads. Rprofmem counts R allocator events and, for default-parallel
CV, sums the parent and fork-worker profiles. The two profilers count different
events, so the artifact records each scope, and allocation counts are compared
within one implementation and never as a ratio between profilers.

[`benches/profile_memory.py`](benches/profile_memory.py) measures engine stages
and memory separately. Each point runs in a fresh process and records exclusive
stage shares (each stage's time as a fraction of the total, without overlap),
normalized process peak RSS (resident memory), and the exact bytes held by the
retained factor and cross-Gram caches. The core
[`allocation_probe`](crates/mboostrs-core/examples/allocation_probe.rs) example
uses a counting allocator to report the allocation events of a complete fit
after warm-up.

Parallel block selection keeps candidate coefficients in scratch buffers owned
by each learner and combines only the candidate scores; after selection, the
engine copies the winning coefficient vector once. On the development machine,
a parallel scan of BBS candidates becomes faster than a sequential scan at
about 10 learners, so automatic execution scans fewer learners sequentially to
avoid Rayon's thread overhead. In full Poisson BBS fits this reduced the median
times for `j=2` and `j=3` learners by 27.6% and 22.0%; `j=10` and `j=50` were
unchanged. The allocation regression test is `parallel_selection_allocation.rs`,
and the crossover benchmarks are the `block_selection_crossover` and
`non_gaussian_bbs_crossover` Criterion groups. These are development
measurements, not a speed claim across machines.

The checked catalog (`benches/scenarios.json`) lists 32 scenarios, all with status `implemented`. It covers packed,
Gram, general-block, monotone, tree, survival, multinomial, cyclic and
non-cyclic LSS, CV at serial and default settings, model operations,
instrumentation, both cached solver routes, stability selection at serial and
default settings, and an explicit 1/2/4/8-thread scaling sweep. The scaling
runner runs each configuration in a fresh process and rejects the artifact
unless all result digests match exactly. The performance chapter shows the
checked [raw development scaling artifact](docs/book/data/thread-scaling-2026-07-15.json)
but does not treat its timings, taken on a working tree with uncommitted
changes, as a release claim.

Gaussian fits use the covariance update that glmnet uses: the products of every
base learner's design with every other's are computed once, and each iteration
updates the learners' projections of the residual from them instead of
recomputing the projections from the residual vector. `cvrisk` builds the same
fast path in each fold for in-bag fits. Non-Gaussian families use the general
path, which solves each learner's normal equations at every iteration, so
their speedups are smaller but still material. Selection-only Gaussian smooth
refits also use the Gram route, so stability selection does not build the
coefficient, risk, increment, nuisance and fitted-value paths it would discard.

## Parity with R

The R packages provide reference implementations for the methods they
established. The R-parity test suite (`pytest --run-r-parity`, needs R and
rpy2) checks boosting fits against `mboost` 2.9.11, with additional
comparisons for distributional methods recorded in the capability contract.
On verified paths, fitted values and risk paths match the reference to about
1e-6 or better; tolerances and deviations are specified per operation.
`bss`, `blackboost`, `FP` and `RCG` are not implemented. Known deviations, such as the `Laplace` offset and
the scale of weighted `risk_`, are documented in
[`docs/book/implementation/parity_and_bugs.ipynb`](docs/book/implementation/parity_and_bugs.ipynb).

The [R parity evidence workflow](https://github.com/fcela/mboostrs/actions/workflows/parity.yml)
runs all 232 marked tests weekly and on demand; pull requests that change
statistical code run a focused subset of 12 test nodes. Every run uploads a
JSON ledger, validated against a schema, with the commit, platform, dependency
versions, status of each test node, tolerance and measured numeric error.
Older assertions that do not report a numeric comparison are recorded as
`not_emitted` and are never counted as zero error.

## Testing

```bash
cargo test --workspace                 # Rust unit tests
pytest -m "not r_parity"               # Python tests (no R needed)
pytest --run-r-parity                  # full parity suite (needs R + rpy2)
python tools/run_parity.py --scope focused --output parity-results.json
```

## Evaluation scorecard

[The consolidated mboostrs status](docs/scorecard/status.md) summarizes the
project's recorded strengths, gaps and priorities. All new evaluations use the
single [canonical scorecard](docs/scorecard/scorecard.md),
`mboostrs-scorecard/2.1`. Its twelve dimensions score predictive validity
(D11), iteration speed (D5), effect uncertainty (D4) and prediction-interval
calibration (D9) separately. The [migration crosswalk](docs/scorecard/migration.md)
identifies historical rubrics and grades.

## License

MIT
