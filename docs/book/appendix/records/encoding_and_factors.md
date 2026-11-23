(record-encoding-and-factors)=
# Factor and binary encoding dictionaries and stopping-grid contracts

This record summarizes the dictionary-completion and stopping-grid validation studies supporting [](../../benchmarks/encoding_tracks.ipynb) and [](../../benchmarks/predictive_tracks.ipynb).

## Categorical and binary reference-level smooths

When a continuous predictor $x$ has opposite curves across levels of a two-level factor or binary indicator $g$ (for example, $2\sin(\pi x)(1 - 2g) + 0.5z$), marginal screening can eliminate the main smooth `bbs(x)` because its marginal average cancels across $g$. Under treatment coding, adding only a `by=g` interaction term without a baseline `bbs(x)` leaves the reference level ($g = 0$) constant.

Completing the interaction dictionary—supplying the baseline smooth alongside a treatment-coded factor-by or binary-by smooth whenever marginal elimination removed the continuous partner—was validated on fresh synthetic controls ($n_{\text{train}} = 500$, $n_{\text{test}} = 2{,}000$, $\sigma = 0.5$, 6 draws per family) and on the multi-library encoding benchmarks:

- **Categorical factor controls (42/42 draws completed):** Test MSE dropped from 1.283 to 0.324 on opposed curves (paired change −0.960, SE 0.023), from 1.286 to 0.333 under reversed reference labels (−0.953, SE 0.021), and from 1.314 to 0.306 on log-transformed inputs (−1.008, SE 0.016). Smooth, null, numeric-indicator, and additive-factor predictions were byte-identical. On the 168-job `predictive-tracks/2` study, discovery's factor-curve gap versus CatBoost moved from +75.14% (SE 4.14) to −8.23% (SE 2.25; mean MSE 2.017 → 1.057).
- **Numeric and boolean binary controls (54/54 draws completed):** Extending the same dictionary completion to 0/1 numeric and boolean indicators cut test MSE by 73–74% on opposed curves and closed the `encoding-tracks/1` integer/boolean discovery gap versus CatBoost from +70.84% (SE 2.87) to −10.68% (SE 1.82; MSE 2.074 → 1.084), matching categorical predictions identically on all 8 draws.
- **Remaining encoding gap:** Under shifted coding (non-zero-centered indicator values), both known-specification (+10.99%, SE 1.33) and discovery (+14.12%, SE 2.01) fits hit the fixed 1,600-update budget ceiling because collinearity slows component-wise convergence; extending the stopping ceiling in a diagnostic run reduced shifted MSE from 1.349 to 1.032.

## Stopping-grid maximum propagation contract

An API audit found that calling `discover(..., max_mstop=25600)` previously forwarded the default 10,000 ceiling to the internal `tune_mstop` call while labeling the reported grid as 12,800 or 25,600. Propagating the effective maximum to both `tune_mstop` and preflight checks was verified across 8 shifted-coding draws at `max_mstop=25600` (all reported grids matched actual searched grids; 5 of 8 optima exceeded 10,000, ranging from 8,769 to 11,800) alongside exact bit-for-bit replay of all 80 original 1,600-cap encoding fits.

**Audit artifacts.** Protocols, schema-replay checks, and validation logs are archived under `docs/book/benchmarks/factor_reference/`, `docs/book/benchmarks/binary_reference/`, and `docs/book/benchmarks/encoding_tracks/cap_contract/`.
