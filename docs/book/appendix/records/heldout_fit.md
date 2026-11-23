(record-heldout-fit)=
# Held-out evaluation of automatic fitting

This record summarizes the protocols, design parameters, and aggregate outcomes behind [](../../benchmarks/heldout_competition.ipynb). All evaluations measure `mb.fit(data, "y")` — the single-call specification search and fitting workflow — without user-supplied formulas or hyperparameters.

## Fit-budget calibration

`mb.fit` delegates specification search to `autospec`, whose `budget` parameter bounds the number of model evaluations spent probing candidate families, column learners, interactions, and stopping depths. A calibration sweep evaluated budgets of 0, 8, 16, 32, 128, and 512 model fits across a separate tuning set of tabular tasks, establishing the named presets used by `mb.fit`:

| Preset | Fit budget | Wall-clock limit | Fallback on timeout |
|---|---:|---:|---|
| `"quick"` | 16 fits | 60 s | Default additive specification (budget 0) |
| `"standard"` (default) | 32 fits | 120 s | Default additive specification (budget 0) |
| `"thorough"` | 512 fits | 1,200 s | Default additive specification (budget 0) |

## Held-out evaluation of budgeted search

**Design.** The evaluation compares budgeted search (`budget="standard"`, 32 fits, 120 s cap) against unbudgeted default specification (budget 0: `bbs` on continuous predictors, dtype-assigned defaults otherwise) across 95 public tabular datasets: 36 from OpenML, 56 from UCI, and 3 bundled datasets (Munich rent, India stunting, bodyfat). Eighteen datasets carry 38 published effect directions compiled from the original papers and verified against marginal Spearman signs. Every dataset was evaluated on three random 80/20 splits (stratified for binary targets), capped at 4,000 training and 2,000 test rows, on a single thread with a 600 s hard timeout. Loss is test MSE (regression) or test log loss (binary classification) on the response scale.

**Outcomes.** Across the 62 datasets where both arms completed all three splits, budgeted search lowered held-out loss on 38 datasets and raised it on 20 (two-sided Wilcoxon $p = 8.3 \times 10^{-5}$). The median loss ratio (budgeted / unbudgeted) was 0.990, while the geometric mean ratio was 0.154 due to large improvements on datasets with near-deterministic affine structure. On the 13 completed well-studied datasets, gains were not statistically significant (8 better, 5 worse; median ratio 0.998; $p = 0.24$). Published effect signs were recovered in 76 of 77 (effect, seed) pairs (98.7%) versus 73 of 77 (94.8%) for the unbudgeted baseline. Median wall-clock time was 5.9 s (95th percentile 82 s) against 0.11 s for the unbudgeted default.

**Failures and scope.** Both arms failed on roughly 28% of real-data runs (79 of 285 for budgeted search; 88 of 285 for unbudgeted), driven primarily by unseen categorical levels at prediction time, missing values, and tables where every row contains at least one missing entry. Because the wall-clock limit and line-search fallback were motivated by earlier iterations on the same 95 tables, the real-data splits are not independent of development at the dataset level.

## Six-library automated comparison

**Design.** Under `mboostrs-scorecard/2.1` (fully automated mode, one thread, default hyperparameters for every library), `mb.fit` was compared against CatBoost 1.2.10, LightGBM 4.7.0, XGBoost 3.4.1, InterpretML EBM 0.7.8, and a prespecified scikit-learn ridge/logistic reference pipeline with imputation and one-hot encoding. Tasks comprised the 95 real datasets (45 regression, 50 binary; three new splits each) and 35 planted-truth simulations (7 generators × 5 draws: smooth additive, pure interaction, spread-only effect, pure null, small-sample $n=200$, near-duplicate proxies, and binary additive). Six scorecard dimensions were measured; D1a, D2, D3, D4, D7, and D8 were not exercised, so no aggregate score is computed.

| Dimension | mboostrs | CatBoost | LightGBM | XGBoost | EBM | Reference |
|---|---:|---:|---:|---:|---:|---:|
| D1b Discovery (data alone) | **3** | 2 | 2 | 2 | 2 | 1 |
| D5 Iteration speed | 2 | 3 | 4 | 4 | 1.5 | 4 |
| D6 Robustness | 2 | 3 | 3 | 3 | 2.5 | 3 |
| D9 Uncertainty calibration | **4** | 2 | 2 | 2 | 1 | 3 |
| D10 Reproducibility | **3** | 2 | 2 | 2 | 2 | 2 |
| D11 Predictive validity | 3 | **4** | 3 | 2.5 | 3.5 | n/a |

**Key quantitative findings:**

- **Predictive accuracy (D11):** On completed real tasks, `mb.fit` beat the ridge/logistic reference on 27/33 regression and 23/30 binary tasks (Wilcoxon $p = 3 \times 10^{-4}$ each), but trailed CatBoost on 24/33 regression tasks (median gap to best complete library: 35.3% regression, 12.6% binary; CatBoost: 0.0% and 3.9%). On planted additive generators, `mb.fit` achieved the lowest test loss; on the pure-interaction generator, it failed to adopt an interaction term (MSE 9.32 vs. 0.43 for CatBoost).
- **Discovery (D1b):** Under a common held-out permutation test (raising loss by $> 2$ SE), `mb.fit` found 70/80 planted predictors (0/10 interactions) with a 1.4% false-discovery rate and 5/5 silent null tasks; tree ensembles found 80/80 with 31–46% FDR. However, the formula string reported by `mb.fit` retained unpruned noise terms (64.5% FDR), indicating that deselection lags behind actual fitted shrinkage.
- **Calibration (D9):** Split-conformal nominal 90% intervals achieved 0.900 median coverage (96% of runs within 0.85–0.95) and remained flat across heteroscedastic terciles (0.89 / 0.91 / 0.91). Default tree quantile intervals covered 0.73–0.79.
- **Speed and completion (D5, D6):** On the 50,000 × 20 follow-up session (10 refits with prediction and permutation importance), median latency was 34.4 s for `mb.fit` versus 0.58 s (XGBoost), 0.76 s (LightGBM), 6.7 s (CatBoost), and 67.5 s (EBM). Completion across 285 real runs was 202/285 (71%) for `mb.fit`, 260/285 for EBM, and 285/285 for the remaining four libraries.

**Audit artifacts.** Pre-registrations, SHA-256 digests, dataset manifests, deviation logs, raw JSONL results, and per-library scorecards are archived in the repository under `docs/benchmarks/fit_heldout/`, `docs/benchmarks/competitive_heldout/`, `docs/benchmarks/2026-11-20-fit-budget.md`, `docs/benchmarks/2026-11-21-fit-heldout.md`, and `docs/benchmarks/2026-11-21-competitive-heldout.md`.
