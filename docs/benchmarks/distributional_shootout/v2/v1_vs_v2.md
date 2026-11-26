## Version 1 to version 2, per arm

Geometric-mean CRPS ratio, version 2 over version 1, on tasks complete in both (below 1 is better in version 2).

| Arm | Planted: ratio (95% interval) | better / tied / worse | Real: ratio (95% interval) | better / tied / worse |
|---|---|---|---|---|
| catboost.uncertainty | 0.995 (0.991–0.999) | 3 / 13 / 0 | 0.940 (0.902–0.978) | 21 / 6 / 8 |
| distreebution.crps_rf | 1.000 (1.000–1.000) | 0 / 16 / 0 | 1.000 (1.000–1.000) | 0 / 35 / 0 |
| distreebution.crps_rf_tuned | 1.000 (1.000–1.001) | 0 / 16 / 0 | 1.000 (1.000–1.000) | 0 / 35 / 0 |
| distreebution.pmqrf | 1.000 (1.000–1.000) | 0 / 16 / 0 | 1.000 (1.000–1.000) | 0 / 2 / 0 |
| drforest.cart | 1.000 (1.000–1.000) | 0 / 16 / 0 | 1.000 (0.999–1.000) | 1 / 34 / 0 |
| drforest.mmd | 1.000 (1.000–1.000) | 0 / 16 / 0 | 1.000 (1.000–1.000) | 0 / 34 / 0 |
| drforest.tuned | 1.000 (1.000–1.000) | 0 / 16 / 0 | 1.000 (1.000–1.001) | 0 / 34 / 0 |
| hessboost.normal | 0.950 (0.861–0.999) | 3 / 13 / 0 | 0.724 (0.320–1.416) | 7 / 10 / 18 |
| hessboost.tuned | 1.000 (0.999–1.002) | 0 / 16 / 0 | 1.064 (1.013–1.140) | 1 / 23 / 11 |
| lightgbm.quantiles | 1.000 (1.000–1.000) | 0 / 16 / 0 | 1.000 (0.999–1.002) | 0 / 33 / 2 |
| mboostrs.GaussianLSS | 1.000 (1.000–1.000) | 0 / 16 / 0 | 0.998 (0.996–1.000) | 2 / 33 / 0 |
| mboostrs.LSS_select | 1.000 (1.000–1.000) | 0 / 16 / 0 | 0.999 (0.998–1.001) | 1 / 34 / 0 |
| mboostrs.LSS_select_noncyclic | – | – | – | – |
| mboostrs.LSS_select_tree | – | – | – | – |
| mboostrs.fit | 0.957 (0.891–1.000) | 2 / 14 / 0 | 0.984 (0.806–1.313) | 18 / 10 / 6 |
| mboostrs.fit_lss_trees | – | – | – | – |
| mboostrs.quantiles | 1.000 (1.000–1.001) | 0 / 16 / 0 | 1.025 (0.977–1.083) | 3 / 22 / 9 |
| ngboost.normal | 1.000 (1.000–1.000) | 0 / 16 / 0 | 1.000 (0.999–1.001) | 1 / 34 / 0 |
| qrf | 1.000 (1.000–1.000) | 0 / 16 / 0 | 0.999 (0.998–1.000) | 1 / 34 / 0 |
| reference.climatology | 1.000 (1.000–1.000) | 0 / 16 / 0 | 1.000 (1.000–1.000) | 0 / 35 / 0 |
| reference.gaussian_linear | 1.000 (1.000–1.000) | 0 / 16 / 0 | 1.000 (1.000–1.000) | 0 / 35 / 0 |
