(benchmarks-blind-agent-five-libraries)=
# Blind-agent study on an unseen heteroscedastic dataset

## Study objective

Benchmark suites repeated across development iterations risk implicit
overfitting to familiar tasks. This case study evaluates `mboostrs` against four
established tabular modeling libraries — CatBoost, LightGBM, InterpretML
Explainable Boosting Machines (EBM), and NGBoost — under a Track B (no
formula supplied) blind-agent protocol on a newly generated dataset with mixed
numeric and categorical predictors, nonlinear and interaction mean structure,
nonmonotone conditional variance, missing covariates, a correlated decoy, pure
noise columns, and an unseen test category.

The study tests two questions:

1. **Statistical accuracy and uncertainty quality:** Given identical data and
   task instructions, how closely does each library recover the true conditional
   mean, conditional standard deviation, Gaussian negative log-likelihood, and
   calibrated 90% prediction intervals?
2. **End-to-end autonomy and API ergonomics:** Which parts of the `mboostrs`
   workflow succeed cleanly for an analyst working strictly from the library
   documentation, and where do automatic entry points still fail on messy data?

## Experimental design

### Data-generating process

A fresh synthetic dataset was generated with $n_{\text{train}} = 3{,}000$
training rows, $n_{\text{test}} = 1{,}500$ test rows, and 11 candidate
predictors. The conditional mean combines a smooth nonlinearity, a linear term,
a hinge threshold, a factor-by-continuous interaction, and a bilinear
interaction:

$$
\mu(x) = 10 + 2.4\sin(\pi \cdot \text{temp}) + 1.8\,\text{load} + 3.0\max(0, \text{wind} - 0.65) + 1.6\,\text{load}\cdot s(\text{region}) + 1.8\,\text{temp}\cdot\text{humidity},
$$

where $s(\text{region}) = +1$ for `north` and `west`, $-1$ for `south` and
`east`, and $0$ for an unseen test-only category. Conditional noise is Gaussian
with symmetric nonmonotone spread:

$$
Y \mid X = x \sim \mathcal{N}\!\left(\mu(x),\, \sigma(x)^2\right), \qquad \sigma(x) = 0.45 + 0.85\,|\text{load}|.
$$

Four realistic data complications were planted alongside the true signal:

* **Correlated decoy:** `decoy_load = 0.85 * load + N(0, 0.35)` ($r \approx 0.903$ with `load`, zero conditional effect once `load` is included).
* **Pure noise predictors:** Four continuous noise columns (`n1`–`n4`) and one three-level string factor (`cat_noise`).
* **Missing covariates:** Approximately 7% of entries in `wind` and `humidity` are missing (`NaN`) in both training and test splits (193 test rows have at least one missing predictor). Because `wind` enters $\mu(x)$ through a nonlinear hinge, rows with missing `wind` also exhibit elevated conditional residual variance.
* **Unseen test category:** 15 test rows (1%) belong to a fifth `region` label absent from training.

### Blind protocol and software environment

Two isolated analyst agents received only `train.csv`, `test_X.csv`, and an
identical written brief asking for test point predictions, 90% prediction
intervals, conditional scale estimates where supported, and a written structural
report. Neither agent had access to the sealed test truth (`test_truth.csv`),
benchmark harnesses, or prior notes.

* **Arm 1 (`mboostrs` alone):** `mboostrs 0.2.2` with `numpy` and `polars` for
  data I/O. The analyst could inspect installed APIs and built-in guides only.
* **Arm 2 (Competitors):** `CatBoost 1.2.10` (`RMSEWithUncertainty` plus
  conformal calibration), `LightGBM 4.7.0` (`RMSE` plus quantile CQR),
  `InterpretML EBM 0.7.8` (locally weighted conformal intervals), and
  `NGBoost 0.6.1` (`Normal` distribution, scored both raw and conformally
  recalibrated), together with `scikit-learn 1.9.1`.

All predictions were scored once against the sealed truth on test RMSE for $y$,
latent-mean RMSE against $\mu(x)$ overall and on three subsets (clean rows,
unseen-region rows, and missing-covariate rows), empirical 90% interval coverage
overall and across low/medium/high thirds of true $\sigma(x)$, mean 90%
interval width, and — for distributional models — Gaussian negative
log-likelihood (NLL) and Pearson correlation between estimated $\hat{\sigma}(x)$
and true $\sigma(x)$.

## Quantitative results

| Model | Fit time | Test RMSE ($y$) | Latent $\mu$ RMSE (all / clean) | Unseen region ($n=15$) | Missing rows ($n=193$) | 90% PI cov (L / M / H $\sigma$) | 90% PI width | Gaussian NLL | $\text{corr}(\hat{\sigma}, \sigma)$ |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| **Oracle (truth)** | — | 1.1715 | 0.000 / 0.000 | 0.000 | 0.000 | 0.900 | 3.573 | 1.4544 | 1.000 |
| **`mboostrs` (`gamboostLSS`, conformal)** | **0.07 s** | **1.2074** | **0.294 / 0.087** | 0.205 | **0.787** | 0.878 (`0.88 / 0.91 / 0.85`) | **3.598** | **1.5119** | **0.943** |
| **`mboostrs` (`gamboostLSS`, raw)** | **0.07 s** | **1.2074** | **0.294 / 0.087** | 0.205 | **0.787** | 0.878 (`0.88 / 0.91 / 0.85`) | **3.594** | **1.5119** | **0.943** |
| **InterpretML EBM** (locally weighted conf.) | 71.9 s | 1.2375 | 0.395 / 0.293 | 0.265 | 0.796 | 0.893 (`0.91 / 0.91 / 0.86`) | 3.757 | — | — |
| **CatBoost** (`RMSEWithUncertainty` + conf.) | 1.9 s | 1.2420 | 0.395 / 0.277 | 0.222 | 0.834 | **0.897** (`0.92 / 0.92 / 0.85`) | 3.986 | — | — |
| **NGBoost** (conformalized) | 11.4 s | 1.2564 | 0.435 / 0.349 | **0.198** | 0.806 | 0.888 (`0.90 / 0.91 / 0.85`) | 3.961 | 1.6475 | 0.861 |
| **NGBoost** (raw parametric) | 11.4 s | 1.2564 | 0.435 / 0.349 | **0.198** | 0.806 | 0.819 (`0.83 / 0.85 / 0.77`) | 3.130 | 1.6475 | 0.861 |
| **LightGBM** (`RMSE` + quantile CQR) | 7.6 s | 1.2592 | 0.445 / 0.345 | 1.144 | 0.806 | 0.892 (`0.91 / 0.91 / 0.86`) | 4.243 | — | — |

### Point accuracy and structural recovery

* **Mean function recovery:** The `mboostrs` analyst discovered every true mean
  component — smooth `temp`, linear `load`, thresholded `wind`,
  `brandom(region, by=load)`, and the `temp`–`humidity` interaction — while
  excluding `decoy_load` and all five noise columns. On clean test rows
  ($n = 1{,}292$), latent-mean RMSE was `0.087`, roughly 3.2×–4.0× lower than
  CatBoost (`0.277`), EBM (`0.293`), LightGBM (`0.345`), and NGBoost (`0.349`).
  Overall test RMSE (`1.2074`) sat 3.1% above the irreducible oracle floor
  (`1.1715`).
* **Missing-data variance signal:** With `na_action="impute"`, `mboostrs`
  automatically generated a missingness indicator for `wind` in both the $\mu$
  and $\sigma$ predictors, capturing the extra residual variance on rows where
  `wind` was unobserved (`0.787` latent RMSE vs. `0.796`–`0.834` for
  competitors).
* **Unseen categorical level:** Using `unseen_levels="average"`, `mboostrs`
  scored the 15 unseen-region rows at `0.205` latent RMSE, matching NGBoost
  (`0.198`) and CatBoost (`0.222`), whereas LightGBM's default categorical
  handling degraded to `1.144`.

### Distributional accuracy and interval calibration

* **Head-to-head distributional comparison (`mboostrs` vs. NGBoost):** Both
  libraries fit parametric Gaussian location-scale models. `mboostrs` achieved
  substantially lower test NLL (`1.5119` vs. `1.6475`, against the `1.4544`
  oracle), higher correlation with the true conditional standard deviation
  (`0.943` vs. `0.861`), and tighter 90% intervals (`3.598` vs. `3.961`
  conformalized, oracle `3.573`).
* **Raw parametric vs. conformalized intervals:** Raw `gamboostLSS` intervals
  (`0.878` coverage, `3.594` width) nearly matched their conformally calibrated
  counterpart (`0.878` coverage, `3.598` width, conformal multiplier `1.6468`
  vs. nominal normal `1.6449`). By contrast, raw NGBoost parametric intervals
  substantially under-covered (`0.819` overall; `0.77` in the high-variance
  third) due to in-sample residual shrinkage, requiring conformal recalibration
  to reach `0.888` coverage at width `3.961`.
* **Conditional coverage across spread thirds:** Every calibrated method tracked
  the U-shaped heteroscedasticity well in the low and middle thirds of $\sigma$
  (`0.88`–`0.92`) and slightly under-covered in the highest-variance third
  (`0.85`–`0.86`).
* **Runtime:** The final `gamboostLSS` fit on 3,000 rows completed in `0.07 s`
  (`0.53 s` for 5-fold cross-validation), compared with `1.9 s` for CatBoost,
  `7.6 s` for LightGBM, `11.4 s` for NGBoost, and `71.9 s` for EBM.

## Observed workflow friction and follow-up fixes

Running an analyst strictly through the public API on messy data exposed
several gaps between the core engine and the automated entry points:

1. **Automated entry points lagged the core engine on messy inputs:**
   * `mb.fit(train, "y")` defaulted to dropping missing rows, failed when
     predicting test rows containing `NaN`, and rejected `na_action="impute"`;
     its hygiene report also misclassified the three-level string column
     `cat_noise` as `"discrete-under-float"`.
   * `mb.discover(train, "y", newdata=test_X)` recovered both hidden
     interactions and dropped the decoy and all five noise columns in `3.65 s`,
     but rejected the `sigma: bbs(load)` spread term because the mean model hit
     its default `400`-step cap during screening, and returned
     `predictions = None` because `discover()` did not expose `unseen_levels=`.
2. **Inconsistent `unseen_levels=` coverage across post-fit helpers:** While
   `model.predict(..., unseen_levels="average")` worked, `conformal_interval()`,
   `LSSModel.predict_parameters()`, and `LSSModel.cvrisk()` did not accept
   `unseen_levels=`, requiring the analyst to write manual CV conformal code.
3. **`LSSModel.summary()` crash on non-cyclic and per-parameter `mstop`:**
   Calling `.summary()` on a `gamboostLSS` model fitted with dict `mstop` or
   `method="noncyclic"` raised a `TypeError`.

Several of these gaps — `LSSModel.summary()` for dict/noncyclic `mstop`,
`unseen_levels=` plumbing across `conformal_interval()`,
`LSSModel.predict_parameters()`, `LSSModel.cvrisk()`, and `discover()`,
`na_action=` plumbing in `fit()` and `autospec()`, and the string-column
hygiene bug — were fixed immediately after the study with regression tests.
Addressing `discover()` spread screening when the mean model hits its step cap
remains open work.

## Scope and limitations

This is a single-dataset, single-run blind case study rather than a multi-panel
average. Moreover, the planted DGP consists of smooth, hinge, bilinear, and
smooth log-scale components — a regime well aligned with penalized B-spline
distributional boosting (`gamboostLSS`) once the model specification is
identified. Tree-based ensembles (`CatBoost`, `LightGBM`, `NGBoost`) would hold
a structural advantage on high-order axis-aligned step interactions or very
high-dimensional sparse tables where additive smooth bases lose capacity.
Read these numbers alongside the multi-panel studies in
[the competitive shootout](competitive_shootout.ipynb) and
[the predictive tracks comparison](predictive_tracks.ipynb).
