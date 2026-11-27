(record-heldout-fit)=
# Held-out evaluation of automatic fitting

This record summarizes the protocols, design parameters, and aggregate outcomes behind [](../../benchmarks/heldout_competition.ipynb). All evaluations measure `mb.fit(data, "y")` — the single-call specification search and fitting workflow — without user-supplied formulas or hyperparameters.

## Fit-budget calibration

`mb.fit` delegates specification search to `autospec`, whose `budget` parameter bounds the number of model evaluations spent probing candidate families, column learners, interactions, and stopping depths. A calibration sweep evaluated budgets of 0, 8, 16, 32, 128, and 512 model fits across a separate tuning set of tabular tasks, establishing the named presets used by `mb.fit`:

| Preset | Fit budget | Wall-clock limit | Search strategy | Fallback on timeout |
|---|---:|---:|---|---|
| `"quick"` | 16 fits | 60 s | Fast radar search (`search="fast"`) | Default additive specification (budget 0) |
| `"standard"` (default) | 64 fits (originally 32) | 120 s | Fast radar search (`search="fast"`) | Default additive specification (budget 0) |
| `"thorough"` | 512 fits | 1,200 s | Exhaustive 12-stage search | Default additive specification (budget 0) |

## Held-out evaluation of budgeted search

**Design.** The evaluation compares budgeted search against unbudgeted default specification (budget 0: `bbs` on continuous predictors, dtype-assigned defaults otherwise) across 95 public tabular datasets: 36 from OpenML, 56 from UCI, and 3 bundled datasets (Munich rent, India stunting, bodyfat). Eighteen datasets carry 38 published effect directions compiled from the original papers and verified against marginal Spearman signs. Every dataset was evaluated on three random 80/20 splits (stratified for binary targets), capped at 4,000 training and 2,000 test rows, on a single thread with a 600 s hard timeout. Loss is test MSE (regression) or test log loss (binary classification) on the response scale.

**Outcomes.** On the initial move to budgeted search (seeds 4000–4002), across the 62 datasets where both arms completed all three splits, budgeted search lowered held-out loss on 38 datasets and raised it on 20 (two-sided Wilcoxon $p = 8.3 \times 10^{-5}$; median loss ratio 0.990; geometric mean ratio 0.154). On the subsequent remediation check (seeds 6000–6002, comparing updated `mb.fit` with default imputation, unseen-level handling, guarded interaction adoption, and `residual_tree="auto"` against the prior budgeted release), `fit` + `predict` failures dropped from **87 to 2** across 285 runs and paired held-out loss improved further across 62 common datasets (43 better, 17 worse; geometric-mean ratio **0.787**, 95% CI 0.611–0.978; Wilcoxon $p = 1.4 \times 10^{-3}$).

## Six-library automated comparison

**Design.** Under `mboostrs-scorecard/2.1` (fully automated mode, one thread, default hyperparameters for every library), `mb.fit` was compared against CatBoost 1.2.10, LightGBM 4.7.0, XGBoost 3.4.1, InterpretML EBM 0.7.8, and a prespecified scikit-learn ridge/logistic reference pipeline with imputation and one-hot encoding. Tasks comprised the 95 real datasets (45 regression, 50 binary; seeds 5000–5002) and 35 planted-truth simulations (7 generators × 5 draws: smooth additive, pure interaction, spread-only effect, pure null, small-sample $n=200$, near-duplicate proxies, and binary additive). Six scorecard dimensions were measured; D1a, D2, D3, D4, D7, and D8 were not exercised, so no aggregate score is computed. The `mboostrs` arm was refreshed on the same Apple M4 harness and seeds against the unchanged competitor runs (`DEVIATIONS.md` #5).

| Dimension | mboostrs | CatBoost | LightGBM | XGBoost | EBM | Reference |
|---|---:|---:|---:|---:|---:|---:|
| D1b Discovery (data alone) | **5** | 2 | 2 | 2 | 2 | 1 |
| D5 Iteration speed | 3.5 | 3 | **4** | **4** | 1.5 | **4** |
| D6 Robustness | **3** | **3** | **3** | **3** | 2.5 | **3** |
| D9 Uncertainty calibration | **4.5** | 2 | 2 | 2 | 1 | 3 |
| D10 Reproducibility | **4** | 2 | 2 | 2 | 2 | 2 |
| D11 Predictive validity | **4** | **4** | 3 | 2.5 | 3.5 | n/a |

**Key quantitative findings:**

- **Predictive accuracy (D11):** `mb.fit` completed all 45/45 regression and 50/50 binary tasks, beating the ridge/logistic reference on **37/45** regression ($p = 1.2 \times 10^{-6}$, median ratio 0.805) and **37/50** binary ($p = 5.7 \times 10^{-6}$, median ratio 0.947) tasks. Median gap to the best complete library is **8.0%** on regression and **6.0%** on binary (ahead of LightGBM 9.9% / 30.4% and XGBoost 19.5% / 35.7%). Paired head-to-head, `mb.fit` beat XGBoost on both tracks (33–12 regression $p = 0.0038$; 37–13 binary $p = 0.0039$), beat LightGBM on both tracks (28–17 regression; 36–14 binary $p = 0.038$), beat EBM on regression (21–18, median ratio 0.999), and was statistically indistinguishable from CatBoost on both regression (median ratio 1.030, $p = 0.067$) and binary (23–27, median ratio 1.015, $p = 0.33$). On planted tasks, `mb.fit` achieved the lowest test loss on 6 of 7 generators and **0.533** MSE on pure-interaction S2 (ahead of EBM 0.739, LightGBM 0.918, and XGBoost 1.712, behind CatBoost 0.428).
- **Discovery (D1b):** Under the shared held-out permutation test (raising loss by $> 2$ SE and $> 0.5\%$), `mb.fit` recovered **80/80** planted predictors (**10/10** interactions) at **0.0% FDR** and **5/5** silent null tasks; tree ensembles and EBM had 31–46% FDR. The active-pruned formula reported by `mb.fit` also achieved **80/80 (100%) TPR** (**10/10** interactions, **30/30** linear, **40/40** nonlinear), **8.0% FDR**, **5/5** silent null tasks, and **5/5** scale-only detections on S3, and `mb.fit` recovered **108/114 (94.7%)** real-data published effect directions.
- **Calibration (D9):** Split-conformal nominal 90% intervals across all 165 regression runs achieved **0.902** median coverage (**95.8%** within 0.85–0.95, 1.2% below 0.85) at **1.50** width/SD and remained flat across heteroscedastic terciles (0.891 / 0.905 / 0.908). Default tree quantile intervals covered 0.73–0.79. Binary median ECE dropped to **0.0221** (tied best overall with EBM, ahead of CatBoost 0.0269).
- **Speed, completion, and reproducibility (D5, D6, D10):** On the 50,000 × 20 follow-up session (10 refits with prediction and permutation importance), median latency dropped to **1.52 s** (p90 2.84 s, total 17.83 s) — 4.4× faster than CatBoost (6.69 s) and 45× faster than EBM (67.5 s); 320-run median fit-plus-predict dropped 6.5× to **2.64 s** (ahead of CatBoost 3.01 s). Real-data completion reached **285/285 (100.0%, zero failures)**, ahead of EBM (260/285). All 10/10 reproducibility runs matched bit-identically, with one-command verification via `mb.reproduce(trail, data)` and `python -m mboostrs reproduce`.

**Audit artifacts.** Pre-registrations, SHA-256 digests, dataset manifests, deviation logs, raw JSONL results, and per-library scorecards are archived in the repository under `docs/benchmarks/fit_heldout/`, `docs/benchmarks/competitive_heldout/`, `docs/benchmarks/2026-11-20-fit-budget.md`, `docs/benchmarks/2026-11-21-fit-heldout.md`, `docs/benchmarks/2026-11-21-competitive-heldout.md`, and `docs/benchmarks/2026-11-25-heldout-eyesores-remediation.md`.
