(record-heldout-fit)=
# Held-out evaluation of automatic fitting

This record summarizes the protocols, design parameters, and aggregate outcomes behind [](../../benchmarks/heldout_competition.ipynb). All evaluations measure `mb.fit(data, "y")` — the single-call specification search and fitting workflow — without user-supplied formulas or hyperparameters.

## Fit-budget calibration

`mb.fit` delegates specification search to `autospec`, whose `budget` parameter bounds the number of model evaluations spent probing candidate families, column learners, interactions, and stopping depths. A calibration sweep evaluated budgets of 0, 8, 16, 32, 128, and 512 model fits across a separate tuning set of tabular tasks, establishing the named presets used by `mb.fit`:

| Preset | Fit budget | Wall-clock limit | Fallback on timeout |
|---|---:|---:|---|
| `"quick"` | 16 fits | 60 s | Default additive specification (budget 0) |
| `"standard"` (default) | 64 fits (originally 32) | 120 s | Default additive specification (budget 0) |
| `"thorough"` | 512 fits | 1,200 s | Default additive specification (budget 0) |

## Held-out evaluation of budgeted search

**Design.** The evaluation compares budgeted search against unbudgeted default specification (budget 0: `bbs` on continuous predictors, dtype-assigned defaults otherwise) across 95 public tabular datasets: 36 from OpenML, 56 from UCI, and 3 bundled datasets (Munich rent, India stunting, bodyfat). Eighteen datasets carry 38 published effect directions compiled from the original papers and verified against marginal Spearman signs. Every dataset was evaluated on three random 80/20 splits (stratified for binary targets), capped at 4,000 training and 2,000 test rows, on a single thread with a 600 s hard timeout. Loss is test MSE (regression) or test log loss (binary classification) on the response scale.

**Outcomes.** On the initial move to budgeted search (seeds 4000–4002), across the 62 datasets where both arms completed all three splits, budgeted search lowered held-out loss on 38 datasets and raised it on 20 (two-sided Wilcoxon $p = 8.3 \times 10^{-5}$; median loss ratio 0.990; geometric mean ratio 0.154). On the subsequent remediation check (seeds 6000–6002, comparing current `mb.fit` with default imputation, unseen-level handling, guarded interaction adoption, and `residual_tree="auto"` against the prior budgeted release), `fit` + `predict` failures dropped from **87 to 2** across 285 runs and paired held-out loss improved further across 62 common datasets (43 better, 17 worse; geometric-mean ratio **0.787**, 95% CI 0.611–0.978; Wilcoxon $p = 1.4 \times 10^{-3}$).

## Six-library automated comparison

**Design.** Under `mboostrs-scorecard/2.1` (fully automated mode, one thread, default hyperparameters for every library), `mb.fit` was compared against CatBoost 1.2.10, LightGBM 4.7.0, XGBoost 3.4.1, InterpretML EBM 0.7.8, and a prespecified scikit-learn ridge/logistic reference pipeline with imputation and one-hot encoding. Tasks comprised the 95 real datasets (45 regression, 50 binary; seeds 5000–5002) and 35 planted-truth simulations (7 generators × 5 draws: smooth additive, pure interaction, spread-only effect, pure null, small-sample $n=200$, near-duplicate proxies, and binary additive). Six scorecard dimensions were measured; D1a, D2, D3, D4, D7, and D8 were not exercised, so no aggregate score is computed. The `mboostrs` arm was refreshed on the same Apple M4 harness and seeds against the unchanged competitor runs (`DEVIATIONS.md` #4).

| Dimension | mboostrs | CatBoost | LightGBM | XGBoost | EBM | Reference |
|---|---:|---:|---:|---:|---:|---:|
| D1b Discovery (data alone) | **4** | 2 | 2 | 2 | 2 | 1 |
| D5 Iteration speed | 3 | 3 | **4** | **4** | 1.5 | **4** |
| D6 Robustness | 2.5 | **3** | **3** | **3** | 2.5 | **3** |
| D9 Uncertainty calibration | **4** | 2 | 2 | 2 | 1 | 3 |
| D10 Reproducibility | **3** | 2 | 2 | 2 | 2 | 2 |
| D11 Predictive validity | 3.5 | **4** | 3 | 2.5 | 3.5 | n/a |

**Key quantitative findings:**

- **Predictive accuracy (D11):** On completed real tasks, `mb.fit` beat the ridge/logistic reference on 33/38 regression ($p = 7 \times 10^{-6}$) and 31/46 binary ($p = 0.034$) tasks. Median gap to the best complete library is **9.0%** on regression and **7.5%** on binary (ahead of LightGBM 9.9% / 30.7% and XGBoost 19.3% / 35.7%; within 3.6% median of CatBoost on regression, $p = 0.16$). On planted tasks, `mb.fit` achieved the lowest test loss on 6 of 7 generators and 0.842 MSE on pure-interaction S2 (ahead of LightGBM 0.918 and XGBoost 1.712, behind CatBoost 0.428 and EBM 0.739).
- **Discovery (D1b):** Under the shared held-out permutation test (raising loss by $> 2$ SE and $> 0.5\%$), `mb.fit` recovered **80/80** planted predictors (**10/10** interactions) at **16.7% FDR** (2.8% outside S2 residual-tree draws) and **5/5** silent null tasks; tree ensembles and EBM had 31–46% FDR. The active-pruned formula reported by `mb.fit` achieved **3.9% FDR** (73/80 TPR, 8/10 interactions, 5/5 silent null tasks).
- **Calibration (D9):** Split-conformal nominal 90% intervals achieved **0.901** median coverage (94% of 156 runs within 0.85–0.95) at 1.52 width/SD and remained flat across heteroscedastic terciles (0.894 / 0.906 / 0.908). Default tree quantile intervals covered 0.73–0.79.
- **Speed and completion (D5, D6):** On the 50,000 × 20 follow-up session (10 refits with prediction and permutation importance), median latency was **3.04 s** (p90 7.19 s, total 39.75 s) for `mb.fit` — faster than CatBoost (6.69 s) and EBM (67.5 s), behind XGBoost (0.58 s) and LightGBM (0.76 s). Real-data completion reached **269/285 (94.4%)**, ahead of EBM (260/285); 8 of the 16 remaining failures occur inside `conformal_interval` on LSS models with adopted interactions.

**Audit artifacts.** Pre-registrations, SHA-256 digests, dataset manifests, deviation logs, raw JSONL results, and per-library scorecards are archived in the repository under `docs/benchmarks/fit_heldout/`, `docs/benchmarks/competitive_heldout/`, `docs/benchmarks/2026-11-20-fit-budget.md`, `docs/benchmarks/2026-11-21-fit-heldout.md`, `docs/benchmarks/2026-11-21-competitive-heldout.md`, and `docs/benchmarks/2026-11-25-heldout-eyesores-remediation.md`.
