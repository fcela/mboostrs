# Remediation of held-out benchmark gaps (D6, D1b, D11, D5)

Follow-up engineering record to [2026-11-21-competitive-heldout.md](2026-11-21-competitive-heldout.md) and [status.md](../scorecard/status.md). Every change was measured against the same task generators, real-data tables, and single-thread session harness before recording results here. Canonical scorecard grades remain frozen at version 2.1 until a full fresh-seed rerun completes.

## Root causes diagnosed

1. **Real-data completion (D6 — 83 of 285 runs failed):** All 83 failures fell into three defects:
   - *Unseen categorical levels (45 runs):* `mb.fit` left `unseen_levels="error"` on predictions and `conformal_interval`, `_replay_transforms` refused unseen levels on lowered factor `bols` (`contrasts`) and dummy encodings, and `_encode_factor` sorted mixed `str`/`None` unseen values without a key.
   - *Missing values (27 runs) and transform-domain crashes (11 runs):* `mb.fit` defaulted to `na_action="omit"` instead of `"impute"`, `autospec` dropped rows with missing predictors even under `na_action="impute"`, complete training categorical columns recorded no `impute_level` fallback for test `None`, and test `NaN`s entered `sqrt`/`log` transforms.
   - *Polars `Object` columns:* `impute_level` wrote `dtype=object` arrays into Polars frames, turning string columns into `pl.Object` which `_is_categorical` did not recognize.
2. **Reported-formula false discoveries (D1b — 64.5% native FDR vs. 1.4% fitted permutation FDR; 0/5 clean null tasks):** `mb.fit` recorded the unpruned candidate `spec.formula` as `trail.final["formula"]`, skipping deselection on short paths (`mstop < 10`), null curves, budget exhaustion, and multi-parameter `LSSModel` fits.
3. **Missed interactions and real-data regression gap (D11 / D1b — S2 MSE 9.32, 0/10 interactions, 35.3% regression median gap):** `mb.fit` called `autospec` with `interactions="report"`, `residual_tree=False`, and a 32-fit budget that exhausted before pair adoption or residual-tree validation could run (`unvalidated_interactions` on `kin8nm`, `grid_stability`; unmeasured `mstop=100` on `airfoil`, `california_housing`). `_residual_tree_step` also re-ran two redundant 5-fold CV sweeps instead of reusing `MstopEvidence.fold_scores`.
4. **Follow-up latency at 50,000 rows (D5 — 34.4 s median, 128.7 s binary, 431.5 s total):** `autospec` ran GBDT pre-exploration (`preexplore_json`, 41.1 s), shape probing across all noise columns (26.0 s), and repeated CV sweeps (62.2 s) on all 50,000 rows.

## Changes implemented

- **`python/mboostrs/_procedure/_fit.py`:**
  - Default `na_action="impute"`, `unseen_levels="average"`, `interactions="adopt"`, `residual_tree="auto"`, and `_BUDGET_FITS["standard"] = 64`.
  - Compute active-term `trail.final["formula"]` from fitted `varimp` risk-reduction shares (for `GAMBoostModel`) or per-parameter partial-effect variance shares (for `LSSModel`), keeping terms with share >= 1% (plus caller-declared terms) and reporting intercept-only `y ~ 1` when the CV curve fails to beat step 0 by 1%.
  - Above 3,000 rows (with no explicit `folds` or `groups`), run `autospec` search on a deterministic 3,000-row subsample (pinning `argmin`/`argmax` response rows to preserve support, with default `nu=0.3`) and fit the selected specification on all rows.
- **`python/mboostrs/autospec.py` & `python/mboostrs/_autospec.py`:**
  - Preserve all observed-response rows during `autospec` search under `na_action="impute*"` (pre-imputing the search copy while keeping `incomplete_rows` reporting exact and replaying full `na_action` in `spec.fit`), drop post-omission constant columns, and clamp default `kfold` `B <= n_rows`.
  - Carry `MstopEvidence.fold_scores` from `tune_mstop`, cutting `_residual_tree_step` CV cost from `4 * n_folds` to `n_folds`. Add held-out 80/20 residual-tree detection (`val_tree_r2 >= 0.04`) and non-additive residual triggering (`leftover >= 5%` with `max pair H^2 >= 0.15` or unvalidated interactions), plus sparse active-column scoping under `residual_tree="auto"`.
  - Fall back to the measured initial `depth_row` when budget reservations skip the post-shape `mstop` sweep rather than reverting to unmeasured `mstop=100`.
  - Cap `preexplore_json` at 10,000 rows (`rounds <= 150` when subsampled) and skip 1D shape probing on pure-noise columns (< 0.2% share on tables with > 8 predictors).
  - Recover cleanly from wall-clock `BudgetExhausted(resource="seconds")` inside `probe_shapes`, `probe_mstop`, `interactions`, `residual_tree`, and `compare_finished_routes`.
- **`python/mboostrs/transforms.py`, `glmboost.py`, `gamboost.py`, `gamlss.py`, `conformal.py`:**
  - Record training-mode `impute_level` steps on complete training categorical columns, preserve Polars `Utf8`/`String` dtypes on imputation, recognize Polars `Object` string columns in `_is_categorical`, and fix `None`-safe sorting in `_encode_factor`.
  - Plumb `_model_unseen_levels` defaults across `GAMBoostModel.predict`, `LSSModel.predict`, `LSSModel.predict_parameters`, `conformal_interval`, lowered factor `bols` (`contrasts`), and dummy encodings.

## Measured results

### Planted tasks S1–S7 (35 runs, exact `PROTOCOL.md` scoring)

| Metric | Baseline (v2.1) | Remediated |
| --- | ---: | ---: |
| S2 pure-interaction mean MSE | 9.32 | **0.375** (CatBoost: 0.43) |
| Permutation D1b: true predictors found | 70 / 80 | **80 / 80** |
| Permutation D1b: interactions found | 0 / 10 | **10 / 10** |
| Permutation D1b: false-discovery rate | 1.4% | **1.2%** |
| Permutation D1b: clean null tasks (S4) | 5 / 5 | **5 / 5** |
| Reported formula: true predictors | 77 / 80 (7/10 inter) | **80 / 80 (10/10 inter)** |
| Reported formula: false-discovery rate | 64.5% | **2.4%** |
| Reported formula: clean null tasks (S4) | 0 / 5 | **5 / 5** |

### Real-data regression gap (worst-12 datasets vs. CatBoost, seed 5000)

| Dataset | Baseline `mb / cb` | Remediated `mb / cb` | Interaction adopted |
| --- | ---: | ---: | :---: |
| `energy_efficiency` (44960) | 12.19× | **1.90×** | Yes |
| `airfoil_self_noise` (44957) | 10.93× | **2.78×** | Yes |
| `grid_stability` (44973) | 4.85× | **2.92×** | Yes |
| `pumadyn32nh` (44981) | 2.29× | **0.98×** | Yes |
| `california_housing` (44977) | 1.81× | **1.11×** | Yes |
| `miami_housing` (44983) | 2.66× | **2.44×** | Yes |
| `sarcos` (44976) | 2.34× | **2.04×** | No |
| `kin8nm` (44980) | 3.26× | **3.18×** | Yes |

*(Note: on `naval_propulsion_plant`, the higher budget enabled `GaussianLSS` adoption, which requires the separate `response_scale="auto"` fix being landed concurrently; at 32 fits without unscaled LSS it scored 1.09× CatBoost vs. 1.75× baseline.)*

### D5 50,000 × 20 ten-step analyst session (single thread)

| Metric | Baseline (v2.1) | Remediated (32 fits) | Remediated (64 fits, contended) | CatBoost |
| --- | ---: | ---: | ---: | ---: |
| Median follow-up (s) | 34.40 | **5.11** | **9.40** | 6.69 |
| Binary follow-up (s) | 128.74 | **9.37** | **40.32** | — |
| Total 10-step session (s) | 431.45 | **54.62** | **125.16** | 70.60 |

### Real-data completion (full 285-run held-out check with distributional v2)

- Full 285-run held-out `mb.fit` suite (`benches/fit_heldout`, 95 datasets × 3 seeds) on the combined branch: failed runs dropped from **87 → 2** (32 datasets fixed), and paired geometric-mean test loss ratio improved to **0.806** (95% bootstrap interval **0.647–0.977**; 34 better, 22 tied, 13 worse across 69 common datasets).

## Verification

- `cargo fmt --check --all`, `cargo clippy --workspace --exclude mboostrs-r --all-targets -- -D warnings`, and `cargo test --workspace --exclude mboostrs-r`: passed.
- Full `pytest` suite (`8,256` tests including `test_heldout_remediation.py` and `test_benchmark_registry.py`): passed.
- `R CMD check --no-manual` (`mboostrs_0.2.3.tar.gz`, 108 R test files): `Status: OK`.
- `ruff check` and `docs/book/check_register.py`: 0 findings.
