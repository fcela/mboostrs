(record-count-models)=
# Count-model specification, interaction guards, and conformal calibration

This record consolidates the empirical studies on count-family specification search, interaction adoption guards, stopping budgets, and discrete conformal calibration supporting [](../../benchmarks/poisson_tuning_consistency.ipynb), [](../../benchmarks/count_rank_calibration.ipynb), and [](../../benchmarks/count_mean_backbone.ipynb).

## Learning-rate consistency in count tuning

Automatic discovery uses a reduced step length ($\nu = 0.02$) for non-Gaussian count families to prevent gradient overshoot under log-link curvature. Aligning both stopping selection and cost preflight with this fitted rate was tested on 250 fresh draws across five count regimes (50 draws each; $n = 1{,}000$ analysis split 500/500 for fitting and calibration, $n_{\text{test}} = 1{,}000$; prespecified LightGBM CQR reference; target $\le 1\%$ paired real-width excess with coverage in $[0.88, 0.92]$):

- **High-rate Poisson:** Default discovery real-width excess dropped from +23.47% to −6.34% (MC SE 0.50 points, one-sided 95% upper −5.51%), coverage 0.901, latent-mean MSE 14.36 → 0.272, and one-update degenerate stops fell from 24/50 to 0/50.
- **Original-rate Poisson and zero-inflated counts:** Original-rate default excess remained +3.35% (upper +6.07%), and zero-inflated counts retained a width/cardinality gap, motivating the randomized count-rank calibration and incumbent-guard studies below. Six score rows failed on one overdispersed draw where discovery selected a distributional LSS model; these remain in the denominator.

## Randomized count-rank conformal calibration

For discrete count responses, randomized count-rank conformal sets construct the shortest probability window around the fitted count distribution and widen it with a single conformal quantile $q$. On fresh confirmation draws across low/original/high Poisson, overdispersed negative binomial, and zero-inflated Poisson generators, randomized count-rank calibration passed both real-width and nonnegative-integer cardinality promotion gates (zero-inflated discovery count-size one-sided 95% upper excess +0.028%, real-width upper excess −4.86%, coverage 0.900).

## Actual-incumbent interaction guard and step-rule evaluations

When screening interactions for count models, comparing a candidate tensor product only against a rewritten matched-margin baseline could adopt an interaction whose cross-validated risk was worse than the actual incumbent mean model. Adding an **actual-incumbent guard**—requiring any adopted interaction to strictly lower paired fold CV risk against the actual current mean model (exact ties retain the incumbent)—eliminated all degrading interaction substitutions (258–282 adoptions across confirmation cohorts, 0 violations).

Three candidate extensions to the default count mean-fitting procedure were tested on fresh 50-draw confirmation and development cohorts and **not** promoted as defaults:

| Candidate change | Effect on curved zero-inflated panel | Reason not promoted |
|---|---|---|
| Native line search for `NBinomial` means | Lowered development MSE 22.99 → 3.37; passed nonlinear ZI cardinality gate (upper −0.16%) | Failed other-panel mean-error gate on 50 fresh draws (`low` +131%, `noise_zero_inflated` +89%, `zero_inflated` +14%, `interaction_supplied_family` +319%) |
| Guard + fixed step (`mean_step="fixed"`, shipped default) | Confirmed representation (100% smooth $x_1$, 0–6% false interactions, 0 false null inclusions) and genuine-interaction recovery (38/50 ordinary, 40/50 curved) | Remaining narrow efficiency gap on curved zero-inflated panel: count-size excess +1.67% (one-sided 95% upper +2.73% vs. +1% bound), while interval width passed (upper −2.70%) and coverage was 0.901 |
| Rate-scaled budget (`mean_budget="rate_scaled"`), starved-fit travel cap, and per-term linear mean forms | Reduced path-travel starvation ($400 \times 0.02 = 8$ vs. Gaussian $400 \times 0.1 = 40$) on curved zero-inflated draws | Failed do-no-harm gates on other count panels in development; kept opt-in rather than default |

**Mechanism of the remaining curved zero-inflated gap.** Offline replay of the 50 confirmation draws isolated the cause to mean shrinkage rather than the conformal construction: under the default 400-step cap at $\nu = 0.02$, 47 of 50 negative-binomial fits reached the cap, underpredicting the top quartile by 1.93 counts (conformal quantile $q = 0.0391$ vs. $0.0141$ for a tuned supplied fit). Substituting the true mean into the identical conformal construction dropped count-size excess to −3.02%.

**Audit artifacts.** Protocols, development diagnoses, confirmation summaries, and validation records are archived in `docs/book/benchmarks/count_incumbent_guard/`, `docs/book/benchmarks/negative_binomial_mean_rate/`, `docs/book/benchmarks/count_mean_backbone/`, `docs/book/benchmarks/count_rank_calibration/`, and `docs/book/benchmarks/poisson_tuning_consistency/`.
