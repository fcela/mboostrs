# Distributional prediction shootout, version 2

Run 2026-11-25 under [PROTOCOL_v2.md](PROTOCOL_v2.md). Full tables live in
[v2/tables.md](v2/tables.md) and [v2/v2_tables.md](v2/v2_tables.md); every
number below is read from [v2/analysis.json](v2/analysis.json) and
[v2/v2_comparison.json](v2/v2_comparison.json). Runner:
`benches/distributional_shootout` (also registered as `distributional_shootout`
in `benches/registry.json`).

Twenty-one procedures each predicted a full conditional distribution on raw
(unstandardized) continuous responses for every test row. Every procedure was
scored identically by CRPS computed from its 99 predicted quantiles (0.01, …,
0.99), alongside quantile calibration MAE, central interval coverage/width, and
(for parametric arms) negative log likelihood. Two suites were evaluated: 16
planted-truth settings (8 conditional distributions at 500 and 5,000 development
rows, 3 seeds each = 48 runs/arm) and all 35 regression datasets of
OpenML-CTR23 (3 splits each = 105 runs/arm).

## Results

**Planted truth: mboostrs holds the top five ranks overall.** Across the 16
planted settings, `mboostrs.LSS_select_noncyclic` ranks first (mean rank 5.25,
median gap to best 0.2%), followed by `mboostrs.fit_lss_trees` (5.34, 5
settings won), `mboostrs.fit` (5.41, highest mean skill over climatology at
0.381), `mboostrs.LSS_select` (6.25), and `mboostrs.quantiles` (6.44). Tuned
hessboost ranks sixth (7.75), followed by `drforest.mmd` (8.75) and tuned
DisTreebution CRPS forest (9.09).

| Truth | Best arms (excess CRPS over the true distribution, 5,000 rows) |
|---|---|
| Additive parametric LSS families (`S1`, `S2`, `S3`, `S4`, `S8`) | `mboostrs.LSS_select_noncyclic`, `mboostrs.LSS_select`, and `mb.fit`: 0.0–2.5%; tuned hessboost 0.6–1.7%; forests 1.3–5.6% |
| Bimodal mixture (`S5`) | `mboostrs.quantiles` and `drforest`: 1.1%; CRPS forest 1.4%; parametric arms 10–14% |
| Shape-varying with constant mean (`S6`) | `mboostrs.quantiles` 0.4%; `drforest` and CRPS forest 1.0% |
| Pure interaction `x0·x1` in mean and spread (`S7`) | `mb.fit` with `lss_trees=True` 1.5%, CatBoost 3.5%, default `mb.fit` 4.2%, tuned hessboost 7.7%, CRPS forest 7.9%, LightGBM quantiles 8.0%; additive-only `LSS_select` 57.2% |

**Real data (OpenML-CTR23): tree ensembles still lead on CRPS, while mboostrs
leads on calibration.** Over the 31 datasets completed by every arm above the
90% completion threshold, mean CRPS ranks are: LightGBM quantiles 3.48, QRF
5.16, tuned DisTreebution CRPS forest 5.77, tuned hessboost 6.13, CatBoost
uncertainty 7.26, tuned `drforest` 9.10, NGBoost 9.16, `mboostrs.quantiles`
10.19, `mb.fit` 11.39 (105/105 complete, 2 datasets won), `mboostrs.LSS_select_tree`
11.45 (median skill 0.574), `mboostrs.fit_lss_trees` 11.74,
`mboostrs.LSS_select_noncyclic` 11.77, and additive `mboostrs.LSS_select`
12.85.

In paired real-data comparisons across all 35 datasets:

- `mb.fit` beats additive `mboostrs.LSS_select` by 13.1% in geometric-mean CRPS
  (ratio 0.869; 15 better, 2 tied, 18 worse) and sits close to `drforest.tuned`
  (ratio 1.066; 13 better, 2 tied, 19 worse).
- `mb.fit` trails tuned CRPS forest (ratio 1.213), CatBoost (1.370), QRF
  (1.375), tuned hessboost (1.382), and LightGBM quantiles (1.561).
- Adding a joint depth-6 tree to every LSS parameter (`mboostrs.LSS_select_tree`)
  cuts geometric-mean CRPS over additive `mboostrs.LSS_select` by 13.2% (20
  better, 5 tied, 10 worse; median skill 0.574 vs. 0.477).

**Calibration.** On real OpenML-CTR23 data, `mboostrs.quantiles` achieves the
lowest mean absolute quantile calibration error of all 21 arms (0.024; 90%
coverage 0.898, 98% coverage 0.978), followed by `mboostrs.LSS_select_noncyclic`
(0.028; 90% coverage 0.907). LightGBM quantiles under-covers (90% coverage
0.846, MAE 0.049), while `drforest` (90% coverage 0.960–0.961, MAE 0.068–0.077)
and tuned CRPS forests (90% coverage 0.931, MAE 0.067) over-cover.

**Log score (parametric arms only).** On planted truth (15 tasks), mboostrs
holds the top four ranks: `LSS_select_noncyclic` 2.40, `LSS_select` 3.53,
`mb.fit` with `lss_trees=True` 3.93, and default `mb.fit` 4.13 (ahead of tuned
hessboost at 5.20). On real data (29 tasks), tuned hessboost ranks first
(2.66), followed by CatBoost (3.69), `mboostrs.LSS_select_tree` (4.79),
`mboostrs.LSS_select_noncyclic` (4.86), and NGBoost (4.86).

**Cost (median seconds per real-data run, one thread).** QRF 5.9, CatBoost 7.8,
LightGBM quantiles 12.7, tuned hessboost 13.5, `mboostrs.LSS_select` 16.1,
`mb.fit` 20.6, `mboostrs.LSS_select_tree` 22.0, `mb.fit` with `lss_trees=True`
30.7, NGBoost 49.9, tuned CRPS forest 68.4, `mboostrs.quantiles` 89.7,
`mboostrs.LSS_select_noncyclic` 111.6, and tuned `drforest` 119.7.

## Failures

`mb.fit`, `mb.fit` with `lss_trees=True`, `mboostrs.LSS_select`,
`mboostrs.LSS_select_tree`, and `mboostrs.GaussianLSS` completed 48/48 planted
runs and 105/105 real runs (zero failures).

| Arm | Failed real runs | Cause |
|---|---|---|
| `distreebution.pmqrf` | 95 / 105 | internal unwrap panic in `distreebu_rs` |
| `mboostrs.LSS_select_noncyclic` | 10 / 105 | 3,600 s timeouts on wide tables; `StudentTLSS` finite-mean refusal when `df <= 1` |
| `mboostrs.quantiles` | 3 / 105 | 3,600 s timeout on superconductivity |
| `drforest.mmd`, `drforest.tuned` | 3 / 105 each | degenerate median pairwise distance on `solar_flare` |

## Prespecified default checks

1. **`lss_trees` default (`PROTOCOL_v2.md` rule):** `mb.fit(..., lss_trees=True)`
   against default `mb.fit` has geometric-mean CRPS ratio 1.004 (95% CI
   0.995–1.018; 1 better, 14 tied, 1 worse) on planted tasks and 0.998 (95% CI
   0.958–1.036; 1 better, 31 tied, 3 worse) on real tasks at 1.22× median time.
   The rule requires real ratio < 0.99 with CI strictly below 1, so
   `lss_trees="auto"` remains the default.
2. **Held-out mean accuracy of `mb.fit` (95 datasets × 3 seeds = 285 runs):**
   Failed runs fell from **87 to 2** (32 datasets fixed; 1 newly failing), and
   paired geometric-mean test loss ratio (after / before) improved to **0.806**
   (95% CI 0.647–0.977; 34 better, 22 tied, 13 worse across 69 common
   datasets; median seconds 9.34 → 16.29). The mean-accuracy check passes.

## Exposure and limits

Library changes were developed with the version-1 results in view on seed-0
data of a subset of tasks (`S1`, `S3`, `S7`, superconductivity, California
housing, kin8nm, concrete, CPU activity, abalone, Miami housing, and video
transcoding). Because version 2 uses the same seeds, these numbers are not an
independent fresh-seed test. Eight workers shared a ten-core machine during the
run, so timings are descriptive.
