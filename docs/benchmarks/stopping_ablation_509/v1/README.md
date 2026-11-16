# Stopping-rule ablation (issue #509)

Produced 2026-10-12T16:43:05+00:00 from commit `fb3233282d38411339b7646fb4a13d35e24e5a7d` by `python -m benches.stopping_ablation_509 --run`.

Four prespecified mstop rules compared as ablations on the shootout's frozen panels: `fixed_argmin` (the status quo: single-split `cvrisk` argmin, budget 1200), `boundary_growth` (doubling rule, cap 2400), `patience_60`, and `repeated_cv_growth` (2x repeated 5-fold plus the doubling rule). Every rule sees the same development rows, the same first grid and the same family score; calibration and test rows enter only after all rules have chosen. Differences are paired within a seed on identical test frames; negative favors the rule, positive the status quo. `None` marks a non-finite recorded value.

## cancer

| method | tier | rule | per-seed paired test-loss difference | median | seeds with an unresolved boundary at the tier's pick |
|---|---|---|---|---|---|
| mboost_hybrid | small | boundary_growth | +0.0002459, +0, +0.01542 | +0.0002459 | 1 |
| mboost_hybrid | small | patience_60 | +0.0002459, +0, +0.005808 | +0.0002459 | 0 |
| mboost_hybrid | small | repeated_cv_growth | -0.000488, -0.006162, -0.006783 | -0.006162 | 0 |
| mboost_hybrid | tuned | boundary_growth | -0.001025, +0, -0.1658 | -0.001025 | 1 |
| mboost_hybrid | tuned | patience_60 | -0.001025, +0, -0.1755 | -0.001025 | 0 |
| mboost_hybrid | tuned | repeated_cv_growth | +0.04329, +0.006166, -0.188 | +0.006166 | 0 |
| mboost_additive | small | boundary_growth | +0.001551, +0.0002534, -0.004127 | +0.0002534 | 2 |
| mboost_additive | small | patience_60 | +0.001551, +0.0002534, -0.004127 | +0.0002534 | 2 |
| mboost_additive | small | repeated_cv_growth | +0.000787, +0.002699, +0.0007453 | +0.000787 | 0 |
| mboost_additive | tuned | boundary_growth | +0.0002411, -0.004535, -0.004127 | -0.004127 | 3 |
| mboost_additive | tuned | patience_60 | +0.0002411, -0.003142, -0.004127 | -0.003142 | 2 |
| mboost_additive | tuned | repeated_cv_growth | +0.004394, -0.001119, +0.0007453 | +0.0007453 | 0 |

Chosen iterations and exhaustion, per configuration (`fixed_argmin` -> `boundary_growth` -> `patience_60` -> `repeated_cv_growth`):

| method | config | rule | mstop | largest grid | at boundary | tuned-tier pick differs from fixed rule |
|---|---|---|---|---|---|---|
| mboost_hybrid (seed 0) | 0 | fixed_argmin | 130 | 1200 | False | False |
| mboost_hybrid (seed 0) | 0 | boundary_growth | 126 | 1200 | False | False |
| mboost_hybrid (seed 0) | 0 | patience_60 | 126 | 1200 | False | False |
| mboost_hybrid (seed 0) | 0 | repeated_cv_growth | 156 | 1200 | False | True |
| mboost_hybrid (seed 0) | 1 | fixed_argmin | 913 | 1200 | False | False |
| mboost_hybrid (seed 0) | 1 | boundary_growth | 747 | 1200 | False | False |
| mboost_hybrid (seed 0) | 1 | patience_60 | 747 | 1200 | False | False |
| mboost_hybrid (seed 0) | 1 | repeated_cv_growth | 119 | 1200 | False | True |
| mboost_hybrid (seed 0) | 2 | fixed_argmin | 130 | 1200 | False | False |
| mboost_hybrid (seed 0) | 2 | boundary_growth | 126 | 1200 | False | False |
| mboost_hybrid (seed 0) | 2 | patience_60 | 126 | 1200 | False | False |
| mboost_hybrid (seed 0) | 2 | repeated_cv_growth | 156 | 1200 | False | True |
| mboost_hybrid (seed 0) | 3 | fixed_argmin | 913 | 1200 | False | False |
| mboost_hybrid (seed 0) | 3 | boundary_growth | 747 | 1200 | False | False |
| mboost_hybrid (seed 0) | 3 | patience_60 | 747 | 1200 | False | False |
| mboost_hybrid (seed 0) | 3 | repeated_cv_growth | 119 | 1200 | False | True |
| mboost_additive (seed 0) | 0 | fixed_argmin | 1200 | 1200 | True | False |
| mboost_additive (seed 0) | 0 | boundary_growth | 2397 | 2400 | True | True |
| mboost_additive (seed 0) | 0 | patience_60 | 2397 | 2400 | True | True |
| mboost_additive (seed 0) | 0 | repeated_cv_growth | 2073 | 2400 | False | True |
| mboost_additive (seed 0) | 1 | fixed_argmin | 1200 | 1200 | True | False |
| mboost_additive (seed 0) | 1 | boundary_growth | 1711 | 2400 | False | True |
| mboost_additive (seed 0) | 1 | patience_60 | 1561 | 1621 | False | True |
| mboost_additive (seed 0) | 1 | repeated_cv_growth | 772 | 1200 | False | True |
| mboost_additive (seed 0) | 2 | fixed_argmin | 1197 | 1200 | True | False |
| mboost_additive (seed 0) | 2 | boundary_growth | 2372 | 2400 | True | True |
| mboost_additive (seed 0) | 2 | patience_60 | 2372 | 2400 | True | True |
| mboost_additive (seed 0) | 2 | repeated_cv_growth | 2400 | 2400 | True | True |
| mboost_additive (seed 0) | 3 | fixed_argmin | 1196 | 1200 | True | False |
| mboost_additive (seed 0) | 3 | boundary_growth | 2390 | 2400 | True | True |
| mboost_additive (seed 0) | 3 | patience_60 | 2390 | 2400 | True | True |
| mboost_additive (seed 0) | 3 | repeated_cv_growth | 1543 | 2400 | False | True |
| mboost_hybrid (seed 1) | 0 | fixed_argmin | 136 | 1200 | False | False |
| mboost_hybrid (seed 1) | 0 | boundary_growth | 136 | 1200 | False | False |
| mboost_hybrid (seed 1) | 0 | patience_60 | 136 | 1200 | False | False |
| mboost_hybrid (seed 1) | 0 | repeated_cv_growth | 185 | 1200 | False | True |
| mboost_hybrid (seed 1) | 1 | fixed_argmin | 204 | 1200 | False | False |
| mboost_hybrid (seed 1) | 1 | boundary_growth | 204 | 1200 | False | False |
| mboost_hybrid (seed 1) | 1 | patience_60 | 204 | 1200 | False | False |
| mboost_hybrid (seed 1) | 1 | repeated_cv_growth | 140 | 1200 | False | True |
| mboost_hybrid (seed 1) | 2 | fixed_argmin | 136 | 1200 | False | False |
| mboost_hybrid (seed 1) | 2 | boundary_growth | 136 | 1200 | False | False |
| mboost_hybrid (seed 1) | 2 | patience_60 | 136 | 1200 | False | False |
| mboost_hybrid (seed 1) | 2 | repeated_cv_growth | 185 | 1200 | False | True |
| mboost_hybrid (seed 1) | 3 | fixed_argmin | 204 | 1200 | False | False |
| mboost_hybrid (seed 1) | 3 | boundary_growth | 204 | 1200 | False | False |
| mboost_hybrid (seed 1) | 3 | patience_60 | 204 | 1200 | False | False |
| mboost_hybrid (seed 1) | 3 | repeated_cv_growth | 140 | 1200 | False | True |
| mboost_additive (seed 1) | 0 | fixed_argmin | 1074 | 1200 | False | False |
| mboost_additive (seed 1) | 0 | boundary_growth | 1057 | 1200 | False | False |
| mboost_additive (seed 1) | 0 | patience_60 | 1057 | 1200 | False | False |
| mboost_additive (seed 1) | 0 | repeated_cv_growth | 857 | 1200 | False | True |
| mboost_additive (seed 1) | 1 | fixed_argmin | 1199 | 1200 | True | False |
| mboost_additive (seed 1) | 1 | boundary_growth | 2341 | 2400 | True | False |
| mboost_additive (seed 1) | 1 | patience_60 | 1772 | 1832 | False | False |
| mboost_additive (seed 1) | 1 | repeated_cv_growth | 1155 | 2400 | False | True |
| mboost_additive (seed 1) | 2 | fixed_argmin | 1191 | 1200 | True | False |
| mboost_additive (seed 1) | 2 | boundary_growth | 2231 | 2400 | False | False |
| mboost_additive (seed 1) | 2 | patience_60 | 2231 | 2291 | False | False |
| mboost_additive (seed 1) | 2 | repeated_cv_growth | 1816 | 2400 | False | True |
| mboost_additive (seed 1) | 3 | fixed_argmin | 1200 | 1200 | True | False |
| mboost_additive (seed 1) | 3 | boundary_growth | 2400 | 2400 | True | False |
| mboost_additive (seed 1) | 3 | patience_60 | 2400 | 2400 | True | False |
| mboost_additive (seed 1) | 3 | repeated_cv_growth | 2363 | 2400 | True | True |
| mboost_hybrid (seed 2) | 0 | fixed_argmin | 1075 | 1200 | False | False |
| mboost_hybrid (seed 2) | 0 | boundary_growth | 2400 | 2400 | True | True |
| mboost_hybrid (seed 2) | 0 | patience_60 | 1513 | 1573 | False | True |
| mboost_hybrid (seed 2) | 0 | repeated_cv_growth | 262 | 1200 | False | True |
| mboost_hybrid (seed 2) | 1 | fixed_argmin | 1195 | 1200 | True | False |
| mboost_hybrid (seed 2) | 1 | boundary_growth | 2400 | 2400 | True | True |
| mboost_hybrid (seed 2) | 1 | patience_60 | 1415 | 1475 | False | True |
| mboost_hybrid (seed 2) | 1 | repeated_cv_growth | 177 | 1200 | False | True |
| mboost_hybrid (seed 2) | 2 | fixed_argmin | 1075 | 1200 | False | False |
| mboost_hybrid (seed 2) | 2 | boundary_growth | 2400 | 2400 | True | True |
| mboost_hybrid (seed 2) | 2 | patience_60 | 1513 | 1573 | False | True |
| mboost_hybrid (seed 2) | 2 | repeated_cv_growth | 262 | 1200 | False | True |
| mboost_hybrid (seed 2) | 3 | fixed_argmin | 1195 | 1200 | True | False |
| mboost_hybrid (seed 2) | 3 | boundary_growth | 2400 | 2400 | True | True |
| mboost_hybrid (seed 2) | 3 | patience_60 | 1415 | 1475 | False | True |
| mboost_hybrid (seed 2) | 3 | repeated_cv_growth | 177 | 1200 | False | True |
| mboost_additive (seed 2) | 0 | fixed_argmin | 1200 | 1200 | True | False |
| mboost_additive (seed 2) | 0 | boundary_growth | 2398 | 2400 | True | False |
| mboost_additive (seed 2) | 0 | patience_60 | 2398 | 2400 | True | False |
| mboost_additive (seed 2) | 0 | repeated_cv_growth | 889 | 1200 | False | False |
| mboost_additive (seed 2) | 1 | fixed_argmin | 1096 | 1200 | False | False |
| mboost_additive (seed 2) | 1 | boundary_growth | 1100 | 1200 | False | False |
| mboost_additive (seed 2) | 1 | patience_60 | 1100 | 1200 | False | False |
| mboost_additive (seed 2) | 1 | repeated_cv_growth | 469 | 1200 | False | False |
| mboost_additive (seed 2) | 2 | fixed_argmin | 1200 | 1200 | True | False |
| mboost_additive (seed 2) | 2 | boundary_growth | 2394 | 2400 | True | False |
| mboost_additive (seed 2) | 2 | patience_60 | 2394 | 2400 | True | False |
| mboost_additive (seed 2) | 2 | repeated_cv_growth | 1722 | 2400 | False | False |
| mboost_additive (seed 2) | 3 | fixed_argmin | 1200 | 1200 | True | False |
| mboost_additive (seed 2) | 3 | boundary_growth | 2151 | 2400 | False | False |
| mboost_additive (seed 2) | 3 | patience_60 | 2063 | 2123 | False | False |
| mboost_additive (seed 2) | 3 | repeated_cv_growth | 971 | 1200 | False | False |

## bike_iid

| method | tier | rule | per-seed paired test-loss difference | median | seeds with an unresolved boundary at the tier's pick |
|---|---|---|---|---|---|
| mboost_hybrid | small | boundary_growth | +0, -0.0008912, -0.001085 | -0.0008912 | 0 |
| mboost_hybrid | small | patience_60 | +0, +2.663e-06, +0 | +0 | 0 |
| mboost_hybrid | small | repeated_cv_growth | -0.003253, -0.00166, -0.001187 | -0.00166 | 0 |
| mboost_hybrid | tuned | boundary_growth | +0, -0.0008912, -0.001085 | -0.0008912 | 0 |
| mboost_hybrid | tuned | patience_60 | +0, +2.663e-06, +0 | +0 | 0 |
| mboost_hybrid | tuned | repeated_cv_growth | -0.01297, -0.00166, -0.001187 | -0.00166 | 0 |
| mboost_additive | small | boundary_growth | -0.0112, -0.01135, -0.01025 | -0.0112 | 3 |
| mboost_additive | small | patience_60 | -0.0112, -0.01135, -0.01025 | -0.0112 | 3 |
| mboost_additive | small | repeated_cv_growth | -0.0112, -0.01135, -0.01025 | -0.0112 | 3 |
| mboost_additive | tuned | boundary_growth | +0, +0, +0 | +0 | 0 |
| mboost_additive | tuned | patience_60 | +0, +0, +0 | +0 | 0 |
| mboost_additive | tuned | repeated_cv_growth | -0.0004859, -0.0003923, -0.001006 | -0.0004859 | 2 |

Chosen iterations and exhaustion, per configuration (`fixed_argmin` -> `boundary_growth` -> `patience_60` -> `repeated_cv_growth`):

| method | config | rule | mstop | largest grid | at boundary | tuned-tier pick differs from fixed rule |
|---|---|---|---|---|---|---|
| mboost_hybrid (seed 0) | 0 | fixed_argmin | 1139 | 1200 | False | False |
| mboost_hybrid (seed 0) | 0 | boundary_growth | 1139 | 1200 | False | False |
| mboost_hybrid (seed 0) | 0 | patience_60 | 1139 | 1200 | False | False |
| mboost_hybrid (seed 0) | 0 | repeated_cv_growth | 2136 | 2400 | False | True |
| mboost_hybrid (seed 0) | 1 | fixed_argmin | 232 | 1200 | False | False |
| mboost_hybrid (seed 0) | 1 | boundary_growth | 232 | 1200 | False | False |
| mboost_hybrid (seed 0) | 1 | patience_60 | 232 | 1200 | False | False |
| mboost_hybrid (seed 0) | 1 | repeated_cv_growth | 283 | 1200 | False | True |
| mboost_hybrid (seed 0) | 2 | fixed_argmin | 1078 | 1200 | False | False |
| mboost_hybrid (seed 0) | 2 | boundary_growth | 1078 | 1200 | False | False |
| mboost_hybrid (seed 0) | 2 | patience_60 | 1078 | 1200 | False | False |
| mboost_hybrid (seed 0) | 2 | repeated_cv_growth | 2322 | 2400 | True | True |
| mboost_hybrid (seed 0) | 3 | fixed_argmin | 232 | 1200 | False | False |
| mboost_hybrid (seed 0) | 3 | boundary_growth | 232 | 1200 | False | False |
| mboost_hybrid (seed 0) | 3 | patience_60 | 232 | 1200 | False | False |
| mboost_hybrid (seed 0) | 3 | repeated_cv_growth | 283 | 1200 | False | True |
| mboost_additive (seed 0) | 0 | fixed_argmin | 1200 | 1200 | True | False |
| mboost_additive (seed 0) | 0 | boundary_growth | 2400 | 2400 | True | False |
| mboost_additive (seed 0) | 0 | patience_60 | 2400 | 2400 | True | False |
| mboost_additive (seed 0) | 0 | repeated_cv_growth | 2400 | 2400 | True | False |
| mboost_additive (seed 0) | 1 | fixed_argmin | 538 | 1200 | False | False |
| mboost_additive (seed 0) | 1 | boundary_growth | 538 | 1200 | False | False |
| mboost_additive (seed 0) | 1 | patience_60 | 538 | 1200 | False | False |
| mboost_additive (seed 0) | 1 | repeated_cv_growth | 2397 | 2400 | True | False |
| mboost_additive (seed 0) | 2 | fixed_argmin | 1200 | 1200 | True | False |
| mboost_additive (seed 0) | 2 | boundary_growth | 2400 | 2400 | True | False |
| mboost_additive (seed 0) | 2 | patience_60 | 2400 | 2400 | True | False |
| mboost_additive (seed 0) | 2 | repeated_cv_growth | 2400 | 2400 | True | False |
| mboost_additive (seed 0) | 3 | fixed_argmin | 1065 | 1200 | False | False |
| mboost_additive (seed 0) | 3 | boundary_growth | 1065 | 1200 | False | False |
| mboost_additive (seed 0) | 3 | patience_60 | 1065 | 1200 | False | False |
| mboost_additive (seed 0) | 3 | repeated_cv_growth | 2397 | 2400 | True | False |
| mboost_hybrid (seed 1) | 0 | fixed_argmin | 1198 | 1200 | True | False |
| mboost_hybrid (seed 1) | 0 | boundary_growth | 1510 | 2400 | False | False |
| mboost_hybrid (seed 1) | 0 | patience_60 | 1215 | 1275 | False | False |
| mboost_hybrid (seed 1) | 0 | repeated_cv_growth | 1866 | 2400 | False | False |
| mboost_hybrid (seed 1) | 1 | fixed_argmin | 469 | 1200 | False | False |
| mboost_hybrid (seed 1) | 1 | boundary_growth | 498 | 1200 | False | False |
| mboost_hybrid (seed 1) | 1 | patience_60 | 498 | 1200 | False | False |
| mboost_hybrid (seed 1) | 1 | repeated_cv_growth | 386 | 1200 | False | False |
| mboost_hybrid (seed 1) | 2 | fixed_argmin | 1199 | 1200 | True | False |
| mboost_hybrid (seed 1) | 2 | boundary_growth | 1686 | 2400 | False | False |
| mboost_hybrid (seed 1) | 2 | patience_60 | 1199 | 1259 | False | False |
| mboost_hybrid (seed 1) | 2 | repeated_cv_growth | 2392 | 2400 | True | False |
| mboost_hybrid (seed 1) | 3 | fixed_argmin | 469 | 1200 | False | False |
| mboost_hybrid (seed 1) | 3 | boundary_growth | 498 | 1200 | False | False |
| mboost_hybrid (seed 1) | 3 | patience_60 | 498 | 1200 | False | False |
| mboost_hybrid (seed 1) | 3 | repeated_cv_growth | 386 | 1200 | False | False |
| mboost_additive (seed 1) | 0 | fixed_argmin | 1200 | 1200 | True | False |
| mboost_additive (seed 1) | 0 | boundary_growth | 2399 | 2400 | True | False |
| mboost_additive (seed 1) | 0 | patience_60 | 2399 | 2400 | True | False |
| mboost_additive (seed 1) | 0 | repeated_cv_growth | 2400 | 2400 | True | False |
| mboost_additive (seed 1) | 1 | fixed_argmin | 654 | 1200 | False | False |
| mboost_additive (seed 1) | 1 | boundary_growth | 654 | 1200 | False | False |
| mboost_additive (seed 1) | 1 | patience_60 | 654 | 1200 | False | False |
| mboost_additive (seed 1) | 1 | repeated_cv_growth | 2390 | 2400 | True | False |
| mboost_additive (seed 1) | 2 | fixed_argmin | 1200 | 1200 | True | False |
| mboost_additive (seed 1) | 2 | boundary_growth | 2400 | 2400 | True | False |
| mboost_additive (seed 1) | 2 | patience_60 | 2400 | 2400 | True | False |
| mboost_additive (seed 1) | 2 | repeated_cv_growth | 2400 | 2400 | True | False |
| mboost_additive (seed 1) | 3 | fixed_argmin | 1193 | 1200 | True | False |
| mboost_additive (seed 1) | 3 | boundary_growth | 1406 | 2400 | False | False |
| mboost_additive (seed 1) | 3 | patience_60 | 1406 | 1466 | False | False |
| mboost_additive (seed 1) | 3 | repeated_cv_growth | 2361 | 2400 | True | False |
| mboost_hybrid (seed 2) | 0 | fixed_argmin | 1191 | 1200 | True | False |
| mboost_hybrid (seed 2) | 0 | boundary_growth | 1690 | 2400 | False | False |
| mboost_hybrid (seed 2) | 0 | patience_60 | 1191 | 1251 | False | False |
| mboost_hybrid (seed 2) | 0 | repeated_cv_growth | 1715 | 2400 | False | False |
| mboost_hybrid (seed 2) | 1 | fixed_argmin | 381 | 1200 | False | False |
| mboost_hybrid (seed 2) | 1 | boundary_growth | 381 | 1200 | False | False |
| mboost_hybrid (seed 2) | 1 | patience_60 | 381 | 1200 | False | False |
| mboost_hybrid (seed 2) | 1 | repeated_cv_growth | 321 | 1200 | False | False |
| mboost_hybrid (seed 2) | 2 | fixed_argmin | 1130 | 1200 | False | False |
| mboost_hybrid (seed 2) | 2 | boundary_growth | 1130 | 1200 | False | False |
| mboost_hybrid (seed 2) | 2 | patience_60 | 1130 | 1200 | False | False |
| mboost_hybrid (seed 2) | 2 | repeated_cv_growth | 2084 | 2400 | False | False |
| mboost_hybrid (seed 2) | 3 | fixed_argmin | 381 | 1200 | False | False |
| mboost_hybrid (seed 2) | 3 | boundary_growth | 381 | 1200 | False | False |
| mboost_hybrid (seed 2) | 3 | patience_60 | 381 | 1200 | False | False |
| mboost_hybrid (seed 2) | 3 | repeated_cv_growth | 321 | 1200 | False | False |
| mboost_additive (seed 2) | 0 | fixed_argmin | 1200 | 1200 | True | False |
| mboost_additive (seed 2) | 0 | boundary_growth | 2400 | 2400 | True | False |
| mboost_additive (seed 2) | 0 | patience_60 | 2400 | 2400 | True | False |
| mboost_additive (seed 2) | 0 | repeated_cv_growth | 2400 | 2400 | True | False |
| mboost_additive (seed 2) | 1 | fixed_argmin | 449 | 1200 | False | False |
| mboost_additive (seed 2) | 1 | boundary_growth | 449 | 1200 | False | False |
| mboost_additive (seed 2) | 1 | patience_60 | 449 | 1200 | False | False |
| mboost_additive (seed 2) | 1 | repeated_cv_growth | 1047 | 1200 | False | False |
| mboost_additive (seed 2) | 2 | fixed_argmin | 1200 | 1200 | True | False |
| mboost_additive (seed 2) | 2 | boundary_growth | 2400 | 2400 | True | False |
| mboost_additive (seed 2) | 2 | patience_60 | 2400 | 2400 | True | False |
| mboost_additive (seed 2) | 2 | repeated_cv_growth | 2400 | 2400 | True | False |
| mboost_additive (seed 2) | 3 | fixed_argmin | 848 | 1200 | False | False |
| mboost_additive (seed 2) | 3 | boundary_growth | 848 | 1200 | False | False |
| mboost_additive (seed 2) | 3 | patience_60 | 848 | 1200 | False | False |
| mboost_additive (seed 2) | 3 | repeated_cv_growth | 2140 | 2400 | False | False |

## bike_future

| method | tier | rule | per-seed paired test-loss difference | median | seeds with an unresolved boundary at the tier's pick |
|---|---|---|---|---|---|
| mboost_hybrid | small | boundary_growth | +0, +0, +0.000122 | +0 | 0 |
| mboost_hybrid | small | patience_60 | +0, +0, -8.071e-05 | +0 | 0 |
| mboost_hybrid | small | repeated_cv_growth | +0.01053, -0.003925, +0.0003959 | +0.0003959 | 2 |
| mboost_hybrid | tuned | boundary_growth | -0.0008877, -0.002175, -0.004709 | -0.002175 | 1 |
| mboost_hybrid | tuned | patience_60 | -0.0008877, -0.002175, -8.071e-05 | -0.0008877 | 0 |
| mboost_hybrid | tuned | repeated_cv_growth | +0.005845, -0.003691, +0.002971 | +0.002971 | 0 |
| mboost_additive | small | boundary_growth | +3.231e-05, +0.000699, -0.01746 | +3.231e-05 | 1 |
| mboost_additive | small | patience_60 | +3.231e-05, +0.000699, -0.01746 | +3.231e-05 | 1 |
| mboost_additive | small | repeated_cv_growth | -0.02467, -0.01826, -0.01746 | -0.01826 | 3 |
| mboost_additive | tuned | boundary_growth | +0, +0.002302, -0.0003749 | +0 | 1 |
| mboost_additive | tuned | patience_60 | +0, +0.002302, -0.0003749 | +0 | 1 |
| mboost_additive | tuned | repeated_cv_growth | -0.01046, -0.04649, -0.0003752 | -0.01046 | 2 |

Chosen iterations and exhaustion, per configuration (`fixed_argmin` -> `boundary_growth` -> `patience_60` -> `repeated_cv_growth`):

| method | config | rule | mstop | largest grid | at boundary | tuned-tier pick differs from fixed rule |
|---|---|---|---|---|---|---|
| mboost_hybrid (seed 0) | 0 | fixed_argmin | 313 | 1200 | False | False |
| mboost_hybrid (seed 0) | 0 | boundary_growth | 313 | 1200 | False | False |
| mboost_hybrid (seed 0) | 0 | patience_60 | 313 | 1200 | False | False |
| mboost_hybrid (seed 0) | 0 | repeated_cv_growth | 2190 | 2400 | False | False |
| mboost_hybrid (seed 0) | 1 | fixed_argmin | 111 | 1200 | False | False |
| mboost_hybrid (seed 0) | 1 | boundary_growth | 135 | 1200 | False | False |
| mboost_hybrid (seed 0) | 1 | patience_60 | 135 | 1200 | False | False |
| mboost_hybrid (seed 0) | 1 | repeated_cv_growth | 245 | 1200 | False | False |
| mboost_hybrid (seed 0) | 2 | fixed_argmin | 561 | 1200 | False | False |
| mboost_hybrid (seed 0) | 2 | boundary_growth | 561 | 1200 | False | False |
| mboost_hybrid (seed 0) | 2 | patience_60 | 561 | 1200 | False | False |
| mboost_hybrid (seed 0) | 2 | repeated_cv_growth | 2390 | 2400 | True | False |
| mboost_hybrid (seed 0) | 3 | fixed_argmin | 111 | 1200 | False | False |
| mboost_hybrid (seed 0) | 3 | boundary_growth | 135 | 1200 | False | False |
| mboost_hybrid (seed 0) | 3 | patience_60 | 135 | 1200 | False | False |
| mboost_hybrid (seed 0) | 3 | repeated_cv_growth | 245 | 1200 | False | False |
| mboost_additive (seed 0) | 0 | fixed_argmin | 1048 | 1200 | False | False |
| mboost_additive (seed 0) | 0 | boundary_growth | 1047 | 1200 | False | False |
| mboost_additive (seed 0) | 0 | patience_60 | 1047 | 1200 | False | False |
| mboost_additive (seed 0) | 0 | repeated_cv_growth | 2400 | 2400 | True | False |
| mboost_additive (seed 0) | 1 | fixed_argmin | 175 | 1200 | False | False |
| mboost_additive (seed 0) | 1 | boundary_growth | 175 | 1200 | False | False |
| mboost_additive (seed 0) | 1 | patience_60 | 175 | 1200 | False | False |
| mboost_additive (seed 0) | 1 | repeated_cv_growth | 491 | 1200 | False | False |
| mboost_additive (seed 0) | 2 | fixed_argmin | 1200 | 1200 | True | False |
| mboost_additive (seed 0) | 2 | boundary_growth | 2177 | 2400 | False | False |
| mboost_additive (seed 0) | 2 | patience_60 | 1821 | 1881 | False | False |
| mboost_additive (seed 0) | 2 | repeated_cv_growth | 2400 | 2400 | True | False |
| mboost_additive (seed 0) | 3 | fixed_argmin | 365 | 1200 | False | False |
| mboost_additive (seed 0) | 3 | boundary_growth | 361 | 1200 | False | False |
| mboost_additive (seed 0) | 3 | patience_60 | 361 | 1200 | False | False |
| mboost_additive (seed 0) | 3 | repeated_cv_growth | 1017 | 1200 | False | False |
| mboost_hybrid (seed 1) | 0 | fixed_argmin | 790 | 1200 | False | False |
| mboost_hybrid (seed 1) | 0 | boundary_growth | 790 | 1200 | False | False |
| mboost_hybrid (seed 1) | 0 | patience_60 | 790 | 1200 | False | False |
| mboost_hybrid (seed 1) | 0 | repeated_cv_growth | 2361 | 2400 | True | False |
| mboost_hybrid (seed 1) | 1 | fixed_argmin | 1156 | 1200 | True | False |
| mboost_hybrid (seed 1) | 1 | boundary_growth | 639 | 1200 | False | False |
| mboost_hybrid (seed 1) | 1 | patience_60 | 639 | 1200 | False | False |
| mboost_hybrid (seed 1) | 1 | repeated_cv_growth | 319 | 1200 | False | False |
| mboost_hybrid (seed 1) | 2 | fixed_argmin | 819 | 1200 | False | False |
| mboost_hybrid (seed 1) | 2 | boundary_growth | 819 | 1200 | False | False |
| mboost_hybrid (seed 1) | 2 | patience_60 | 819 | 1200 | False | False |
| mboost_hybrid (seed 1) | 2 | repeated_cv_growth | 2399 | 2400 | True | False |
| mboost_hybrid (seed 1) | 3 | fixed_argmin | 1156 | 1200 | True | False |
| mboost_hybrid (seed 1) | 3 | boundary_growth | 639 | 1200 | False | False |
| mboost_hybrid (seed 1) | 3 | patience_60 | 639 | 1200 | False | False |
| mboost_hybrid (seed 1) | 3 | repeated_cv_growth | 319 | 1200 | False | False |
| mboost_additive (seed 1) | 0 | fixed_argmin | 943 | 1200 | False | False |
| mboost_additive (seed 1) | 0 | boundary_growth | 930 | 1200 | False | False |
| mboost_additive (seed 1) | 0 | patience_60 | 930 | 1200 | False | False |
| mboost_additive (seed 1) | 0 | repeated_cv_growth | 2400 | 2400 | True | True |
| mboost_additive (seed 1) | 1 | fixed_argmin | 94 | 1200 | False | False |
| mboost_additive (seed 1) | 1 | boundary_growth | 94 | 1200 | False | False |
| mboost_additive (seed 1) | 1 | patience_60 | 94 | 1200 | False | False |
| mboost_additive (seed 1) | 1 | repeated_cv_growth | 2397 | 2400 | True | True |
| mboost_additive (seed 1) | 2 | fixed_argmin | 1200 | 1200 | True | False |
| mboost_additive (seed 1) | 2 | boundary_growth | 1834 | 2400 | False | False |
| mboost_additive (seed 1) | 2 | patience_60 | 1834 | 1894 | False | False |
| mboost_additive (seed 1) | 2 | repeated_cv_growth | 2400 | 2400 | True | True |
| mboost_additive (seed 1) | 3 | fixed_argmin | 204 | 1200 | False | False |
| mboost_additive (seed 1) | 3 | boundary_growth | 194 | 1200 | False | False |
| mboost_additive (seed 1) | 3 | patience_60 | 194 | 1200 | False | False |
| mboost_additive (seed 1) | 3 | repeated_cv_growth | 2400 | 2400 | True | True |
| mboost_hybrid (seed 2) | 0 | fixed_argmin | 873 | 1200 | False | False |
| mboost_hybrid (seed 2) | 0 | boundary_growth | 1956 | 2400 | False | True |
| mboost_hybrid (seed 2) | 0 | patience_60 | 1746 | 1806 | False | False |
| mboost_hybrid (seed 2) | 0 | repeated_cv_growth | 2367 | 2400 | True | True |
| mboost_hybrid (seed 2) | 1 | fixed_argmin | 261 | 1200 | False | False |
| mboost_hybrid (seed 2) | 1 | boundary_growth | 890 | 1200 | False | True |
| mboost_hybrid (seed 2) | 1 | patience_60 | 890 | 1200 | False | False |
| mboost_hybrid (seed 2) | 1 | repeated_cv_growth | 507 | 1200 | False | True |
| mboost_hybrid (seed 2) | 2 | fixed_argmin | 1055 | 1200 | False | False |
| mboost_hybrid (seed 2) | 2 | boundary_growth | 2400 | 2400 | True | True |
| mboost_hybrid (seed 2) | 2 | patience_60 | 1301 | 1361 | False | False |
| mboost_hybrid (seed 2) | 2 | repeated_cv_growth | 2400 | 2400 | True | True |
| mboost_hybrid (seed 2) | 3 | fixed_argmin | 261 | 1200 | False | False |
| mboost_hybrid (seed 2) | 3 | boundary_growth | 890 | 1200 | False | True |
| mboost_hybrid (seed 2) | 3 | patience_60 | 890 | 1200 | False | False |
| mboost_hybrid (seed 2) | 3 | repeated_cv_growth | 507 | 1200 | False | True |
| mboost_additive (seed 2) | 0 | fixed_argmin | 1200 | 1200 | True | False |
| mboost_additive (seed 2) | 0 | boundary_growth | 2400 | 2400 | True | False |
| mboost_additive (seed 2) | 0 | patience_60 | 2400 | 2400 | True | False |
| mboost_additive (seed 2) | 0 | repeated_cv_growth | 2400 | 2400 | True | False |
| mboost_additive (seed 2) | 1 | fixed_argmin | 1193 | 1200 | True | False |
| mboost_additive (seed 2) | 1 | boundary_growth | 2398 | 2400 | True | False |
| mboost_additive (seed 2) | 1 | patience_60 | 2398 | 2400 | True | False |
| mboost_additive (seed 2) | 1 | repeated_cv_growth | 2400 | 2400 | True | False |
| mboost_additive (seed 2) | 2 | fixed_argmin | 1200 | 1200 | True | False |
| mboost_additive (seed 2) | 2 | boundary_growth | 2400 | 2400 | True | False |
| mboost_additive (seed 2) | 2 | patience_60 | 2400 | 2400 | True | False |
| mboost_additive (seed 2) | 2 | repeated_cv_growth | 2400 | 2400 | True | False |
| mboost_additive (seed 2) | 3 | fixed_argmin | 1199 | 1200 | True | False |
| mboost_additive (seed 2) | 3 | boundary_growth | 2391 | 2400 | True | False |
| mboost_additive (seed 2) | 3 | patience_60 | 1748 | 1808 | False | False |
| mboost_additive (seed 2) | 3 | repeated_cv_growth | 2391 | 2400 | True | False |

## concrete

| method | tier | rule | per-seed paired test-loss difference | median | seeds with an unresolved boundary at the tier's pick |
|---|---|---|---|---|---|
| mboost_hybrid | small | boundary_growth | -1.17, +0.04199, -1.213 | -1.17 | 2 |
| mboost_hybrid | small | patience_60 | -0.6473, +0.01397, -1.213 | -0.6473 | 0 |
| mboost_hybrid | small | repeated_cv_growth | -1.17, +0.1307, -1.788 | -1.17 | 2 |
| mboost_hybrid | tuned | boundary_growth | -1.605, +0, -1.401 | -1.401 | 2 |
| mboost_hybrid | tuned | patience_60 | -1.605, +0, -1.213 | -1.213 | 1 |
| mboost_hybrid | tuned | repeated_cv_growth | -1.6, -0.607, -1.788 | -1.6 | 3 |
| mboost_additive | small | boundary_growth | -2.772, +0, -0.3171 | -0.3171 | 2 |
| mboost_additive | small | patience_60 | -2.772, +0, -0.3171 | -0.3171 | 2 |
| mboost_additive | small | repeated_cv_growth | -2.769, -8.301, -0.3171 | -2.769 | 3 |
| mboost_additive | tuned | boundary_growth | -1.027, +0, -0.9199 | -0.9199 | 2 |
| mboost_additive | tuned | patience_60 | -1.027, +0, -0.9199 | -0.9199 | 2 |
| mboost_additive | tuned | repeated_cv_growth | -1.027, -3.23, -0.6086 | -1.027 | 2 |

Chosen iterations and exhaustion, per configuration (`fixed_argmin` -> `boundary_growth` -> `patience_60` -> `repeated_cv_growth`):

| method | config | rule | mstop | largest grid | at boundary | tuned-tier pick differs from fixed rule |
|---|---|---|---|---|---|---|
| mboost_hybrid (seed 0) | 0 | fixed_argmin | 1198 | 1200 | True | False |
| mboost_hybrid (seed 0) | 0 | boundary_growth | 2400 | 2400 | True | True |
| mboost_hybrid (seed 0) | 0 | patience_60 | 1810 | 1870 | False | True |
| mboost_hybrid (seed 0) | 0 | repeated_cv_growth | 2400 | 2400 | True | True |
| mboost_hybrid (seed 0) | 1 | fixed_argmin | 1199 | 1200 | True | False |
| mboost_hybrid (seed 0) | 1 | boundary_growth | 908 | 1200 | False | True |
| mboost_hybrid (seed 0) | 1 | patience_60 | 908 | 1200 | False | True |
| mboost_hybrid (seed 0) | 1 | repeated_cv_growth | 2270 | 2400 | False | True |
| mboost_hybrid (seed 0) | 2 | fixed_argmin | 1200 | 1200 | True | False |
| mboost_hybrid (seed 0) | 2 | boundary_growth | 2369 | 2400 | True | True |
| mboost_hybrid (seed 0) | 2 | patience_60 | 2369 | 2400 | True | True |
| mboost_hybrid (seed 0) | 2 | repeated_cv_growth | 2382 | 2400 | True | True |
| mboost_hybrid (seed 0) | 3 | fixed_argmin | 1199 | 1200 | True | False |
| mboost_hybrid (seed 0) | 3 | boundary_growth | 908 | 1200 | False | True |
| mboost_hybrid (seed 0) | 3 | patience_60 | 908 | 1200 | False | True |
| mboost_hybrid (seed 0) | 3 | repeated_cv_growth | 2336 | 2400 | True | True |
| mboost_additive (seed 0) | 0 | fixed_argmin | 1199 | 1200 | True | False |
| mboost_additive (seed 0) | 0 | boundary_growth | 2400 | 2400 | True | False |
| mboost_additive (seed 0) | 0 | patience_60 | 2400 | 2400 | True | False |
| mboost_additive (seed 0) | 0 | repeated_cv_growth | 2399 | 2400 | True | False |
| mboost_additive (seed 0) | 1 | fixed_argmin | 1200 | 1200 | True | False |
| mboost_additive (seed 0) | 1 | boundary_growth | 2400 | 2400 | True | False |
| mboost_additive (seed 0) | 1 | patience_60 | 2400 | 2400 | True | False |
| mboost_additive (seed 0) | 1 | repeated_cv_growth | 2400 | 2400 | True | False |
| mboost_additive (seed 0) | 2 | fixed_argmin | 1200 | 1200 | True | False |
| mboost_additive (seed 0) | 2 | boundary_growth | 2400 | 2400 | True | False |
| mboost_additive (seed 0) | 2 | patience_60 | 2400 | 2400 | True | False |
| mboost_additive (seed 0) | 2 | repeated_cv_growth | 2400 | 2400 | True | False |
| mboost_additive (seed 0) | 3 | fixed_argmin | 1193 | 1200 | True | False |
| mboost_additive (seed 0) | 3 | boundary_growth | 2400 | 2400 | True | False |
| mboost_additive (seed 0) | 3 | patience_60 | 2400 | 2400 | True | False |
| mboost_additive (seed 0) | 3 | repeated_cv_growth | 2400 | 2400 | True | False |
| mboost_hybrid (seed 1) | 0 | fixed_argmin | 1190 | 1200 | True | False |
| mboost_hybrid (seed 1) | 0 | boundary_growth | 2357 | 2400 | True | False |
| mboost_hybrid (seed 1) | 0 | patience_60 | 1200 | 1260 | False | False |
| mboost_hybrid (seed 1) | 0 | repeated_cv_growth | 1975 | 2400 | False | True |
| mboost_hybrid (seed 1) | 1 | fixed_argmin | 734 | 1200 | False | False |
| mboost_hybrid (seed 1) | 1 | boundary_growth | 734 | 1200 | False | False |
| mboost_hybrid (seed 1) | 1 | patience_60 | 734 | 1200 | False | False |
| mboost_hybrid (seed 1) | 1 | repeated_cv_growth | 839 | 1200 | False | True |
| mboost_hybrid (seed 1) | 2 | fixed_argmin | 1200 | 1200 | True | False |
| mboost_hybrid (seed 1) | 2 | boundary_growth | 2361 | 2400 | True | False |
| mboost_hybrid (seed 1) | 2 | patience_60 | 2237 | 2297 | False | False |
| mboost_hybrid (seed 1) | 2 | repeated_cv_growth | 2399 | 2400 | True | True |
| mboost_hybrid (seed 1) | 3 | fixed_argmin | 734 | 1200 | False | False |
| mboost_hybrid (seed 1) | 3 | boundary_growth | 734 | 1200 | False | False |
| mboost_hybrid (seed 1) | 3 | patience_60 | 734 | 1200 | False | False |
| mboost_hybrid (seed 1) | 3 | repeated_cv_growth | 727 | 1200 | False | True |
| mboost_additive (seed 1) | 0 | fixed_argmin | 323 | 1200 | False | False |
| mboost_additive (seed 1) | 0 | boundary_growth | 323 | 1200 | False | False |
| mboost_additive (seed 1) | 0 | patience_60 | 323 | 1200 | False | False |
| mboost_additive (seed 1) | 0 | repeated_cv_growth | 2399 | 2400 | True | True |
| mboost_additive (seed 1) | 1 | fixed_argmin | 347 | 1200 | False | False |
| mboost_additive (seed 1) | 1 | boundary_growth | 347 | 1200 | False | False |
| mboost_additive (seed 1) | 1 | patience_60 | 347 | 1200 | False | False |
| mboost_additive (seed 1) | 1 | repeated_cv_growth | 2398 | 2400 | True | True |
| mboost_additive (seed 1) | 2 | fixed_argmin | 614 | 1200 | False | False |
| mboost_additive (seed 1) | 2 | boundary_growth | 614 | 1200 | False | False |
| mboost_additive (seed 1) | 2 | patience_60 | 614 | 1200 | False | False |
| mboost_additive (seed 1) | 2 | repeated_cv_growth | 2400 | 2400 | True | True |
| mboost_additive (seed 1) | 3 | fixed_argmin | 725 | 1200 | False | False |
| mboost_additive (seed 1) | 3 | boundary_growth | 725 | 1200 | False | False |
| mboost_additive (seed 1) | 3 | patience_60 | 725 | 1200 | False | False |
| mboost_additive (seed 1) | 3 | repeated_cv_growth | 2399 | 2400 | True | True |
| mboost_hybrid (seed 2) | 0 | fixed_argmin | 1179 | 1200 | True | False |
| mboost_hybrid (seed 2) | 0 | boundary_growth | 1919 | 2400 | False | True |
| mboost_hybrid (seed 2) | 0 | patience_60 | 1919 | 1979 | False | False |
| mboost_hybrid (seed 2) | 0 | repeated_cv_growth | 2399 | 2400 | True | False |
| mboost_hybrid (seed 2) | 1 | fixed_argmin | 1199 | 1200 | True | False |
| mboost_hybrid (seed 2) | 1 | boundary_growth | 720 | 1200 | False | True |
| mboost_hybrid (seed 2) | 1 | patience_60 | 720 | 1200 | False | False |
| mboost_hybrid (seed 2) | 1 | repeated_cv_growth | 1341 | 2400 | False | False |
| mboost_hybrid (seed 2) | 2 | fixed_argmin | 1197 | 1200 | True | False |
| mboost_hybrid (seed 2) | 2 | boundary_growth | 2357 | 2400 | True | True |
| mboost_hybrid (seed 2) | 2 | patience_60 | 1685 | 1745 | False | False |
| mboost_hybrid (seed 2) | 2 | repeated_cv_growth | 1676 | 2400 | False | False |
| mboost_hybrid (seed 2) | 3 | fixed_argmin | 1199 | 1200 | True | False |
| mboost_hybrid (seed 2) | 3 | boundary_growth | 720 | 1200 | False | True |
| mboost_hybrid (seed 2) | 3 | patience_60 | 720 | 1200 | False | False |
| mboost_hybrid (seed 2) | 3 | repeated_cv_growth | 1341 | 2400 | False | False |
| mboost_additive (seed 2) | 0 | fixed_argmin | 1200 | 1200 | True | False |
| mboost_additive (seed 2) | 0 | boundary_growth | 2400 | 2400 | True | False |
| mboost_additive (seed 2) | 0 | patience_60 | 2400 | 2400 | True | False |
| mboost_additive (seed 2) | 0 | repeated_cv_growth | 2400 | 2400 | True | False |
| mboost_additive (seed 2) | 1 | fixed_argmin | 1198 | 1200 | True | False |
| mboost_additive (seed 2) | 1 | boundary_growth | 2389 | 2400 | True | False |
| mboost_additive (seed 2) | 1 | patience_60 | 2389 | 2400 | True | False |
| mboost_additive (seed 2) | 1 | repeated_cv_growth | 1952 | 2400 | False | False |
| mboost_additive (seed 2) | 2 | fixed_argmin | 1200 | 1200 | True | False |
| mboost_additive (seed 2) | 2 | boundary_growth | 2399 | 2400 | True | False |
| mboost_additive (seed 2) | 2 | patience_60 | 2399 | 2400 | True | False |
| mboost_additive (seed 2) | 2 | repeated_cv_growth | 2400 | 2400 | True | False |
| mboost_additive (seed 2) | 3 | fixed_argmin | 1200 | 1200 | True | False |
| mboost_additive (seed 2) | 3 | boundary_growth | 2400 | 2400 | True | False |
| mboost_additive (seed 2) | 3 | patience_60 | 2400 | 2400 | True | False |
| mboost_additive (seed 2) | 3 | repeated_cv_growth | 2400 | 2400 | True | False |

## joint_boundary

| method | tier | rule | per-seed paired test-loss difference | median | seeds with an unresolved boundary at the tier's pick |
|---|---|---|---|---|---|
| mboost_hybrid | small | boundary_growth | -0.04787, -0.09789, -0.1261, -0.03044, -0.004047 | -0.04787 | 3 |
| mboost_hybrid | small | patience_60 | -0.03246, -0.012, +0, -0.02303, +0.001323 | -0.012 | 0 |
| mboost_hybrid | small | repeated_cv_growth | -0.04744, -0.09787, -0.127, -0.03055, -0.004057 | -0.04744 | 5 |
| mboost_hybrid | tuned | boundary_growth | -0.06353, -0.08898, -0.001184, -0.009734, -0.007202 | -0.009734 | 0 |
| mboost_hybrid | tuned | patience_60 | -0.04812, -0.012, -0.0002944, -0.00352, -0.00332 | -0.00352 | 0 |
| mboost_hybrid | tuned | repeated_cv_growth | -0.05839, -0.09787, -0.1263, -0.3815, -0.004057 | -0.09787 | 5 |
| mboost_additive | small | boundary_growth | +0, +0, +0, +0, +0 | +0 | 0 |
| mboost_additive | small | patience_60 | +0, +0, +0, +0, +0 | +0 | 0 |
| mboost_additive | small | repeated_cv_growth | -0.002873, +0.002983, +0.0001529, -0.003031, -0.002372 | -0.002372 | 0 |
| mboost_additive | tuned | boundary_growth | +0, +0, +0, +0, +0 | +0 | 0 |
| mboost_additive | tuned | patience_60 | +0, +0, +0, +0, +0 | +0 | 0 |
| mboost_additive | tuned | repeated_cv_growth | -0.002873, +0.002983, +5.03e-05, -0.003031, -0.002372 | -0.002372 | 0 |

Chosen iterations and exhaustion, per configuration (`fixed_argmin` -> `boundary_growth` -> `patience_60` -> `repeated_cv_growth`):

| method | config | rule | mstop | largest grid | at boundary | tuned-tier pick differs from fixed rule |
|---|---|---|---|---|---|---|
| mboost_hybrid (seed 0) | 0 | fixed_argmin | 1188 | 1200 | True | False |
| mboost_hybrid (seed 0) | 0 | boundary_growth | 2261 | 2400 | False | True |
| mboost_hybrid (seed 0) | 0 | patience_60 | 1529 | 1589 | False | True |
| mboost_hybrid (seed 0) | 0 | repeated_cv_growth | 2398 | 2400 | True | False |
| mboost_hybrid (seed 0) | 1 | fixed_argmin | 1179 | 1200 | True | False |
| mboost_hybrid (seed 0) | 1 | boundary_growth | 2380 | 2400 | True | True |
| mboost_hybrid (seed 0) | 1 | patience_60 | 1252 | 1312 | False | True |
| mboost_hybrid (seed 0) | 1 | repeated_cv_growth | 2273 | 2400 | False | False |
| mboost_hybrid (seed 0) | 2 | fixed_argmin | 1173 | 1200 | True | False |
| mboost_hybrid (seed 0) | 2 | boundary_growth | 2332 | 2400 | True | True |
| mboost_hybrid (seed 0) | 2 | patience_60 | 1154 | 1214 | False | True |
| mboost_hybrid (seed 0) | 2 | repeated_cv_growth | 2317 | 2400 | True | False |
| mboost_hybrid (seed 0) | 3 | fixed_argmin | 1179 | 1200 | True | False |
| mboost_hybrid (seed 0) | 3 | boundary_growth | 2380 | 2400 | True | True |
| mboost_hybrid (seed 0) | 3 | patience_60 | 1252 | 1312 | False | True |
| mboost_hybrid (seed 0) | 3 | repeated_cv_growth | 2273 | 2400 | False | False |
| mboost_additive (seed 0) | 0 | fixed_argmin | 15 | 1200 | False | False |
| mboost_additive (seed 0) | 0 | boundary_growth | 15 | 1200 | False | False |
| mboost_additive (seed 0) | 0 | patience_60 | 15 | 1200 | False | False |
| mboost_additive (seed 0) | 0 | repeated_cv_growth | 16 | 1200 | False | False |
| mboost_additive (seed 0) | 1 | fixed_argmin | 13 | 1200 | False | False |
| mboost_additive (seed 0) | 1 | boundary_growth | 13 | 1200 | False | False |
| mboost_additive (seed 0) | 1 | patience_60 | 13 | 1200 | False | False |
| mboost_additive (seed 0) | 1 | repeated_cv_growth | 15 | 1200 | False | False |
| mboost_additive (seed 0) | 2 | fixed_argmin | 26 | 1200 | False | False |
| mboost_additive (seed 0) | 2 | boundary_growth | 26 | 1200 | False | False |
| mboost_additive (seed 0) | 2 | patience_60 | 26 | 1200 | False | False |
| mboost_additive (seed 0) | 2 | repeated_cv_growth | 36 | 1200 | False | False |
| mboost_additive (seed 0) | 3 | fixed_argmin | 26 | 1200 | False | False |
| mboost_additive (seed 0) | 3 | boundary_growth | 26 | 1200 | False | False |
| mboost_additive (seed 0) | 3 | patience_60 | 26 | 1200 | False | False |
| mboost_additive (seed 0) | 3 | repeated_cv_growth | 34 | 1200 | False | False |
| mboost_hybrid (seed 1) | 0 | fixed_argmin | 1200 | 1200 | True | False |
| mboost_hybrid (seed 1) | 0 | boundary_growth | 2369 | 2400 | True | True |
| mboost_hybrid (seed 1) | 0 | patience_60 | 1323 | 1383 | False | False |
| mboost_hybrid (seed 1) | 0 | repeated_cv_growth | 2383 | 2400 | True | False |
| mboost_hybrid (seed 1) | 1 | fixed_argmin | 1018 | 1200 | False | False |
| mboost_hybrid (seed 1) | 1 | boundary_growth | 1113 | 1200 | False | True |
| mboost_hybrid (seed 1) | 1 | patience_60 | 1113 | 1200 | False | False |
| mboost_hybrid (seed 1) | 1 | repeated_cv_growth | 2187 | 2400 | False | False |
| mboost_hybrid (seed 1) | 2 | fixed_argmin | 1158 | 1200 | True | False |
| mboost_hybrid (seed 1) | 2 | boundary_growth | 2132 | 2400 | False | True |
| mboost_hybrid (seed 1) | 2 | patience_60 | 1158 | 1218 | False | False |
| mboost_hybrid (seed 1) | 2 | repeated_cv_growth | 2400 | 2400 | True | False |
| mboost_hybrid (seed 1) | 3 | fixed_argmin | 1018 | 1200 | False | False |
| mboost_hybrid (seed 1) | 3 | boundary_growth | 1113 | 1200 | False | True |
| mboost_hybrid (seed 1) | 3 | patience_60 | 1113 | 1200 | False | False |
| mboost_hybrid (seed 1) | 3 | repeated_cv_growth | 2187 | 2400 | False | False |
| mboost_additive (seed 1) | 0 | fixed_argmin | 32 | 1200 | False | False |
| mboost_additive (seed 1) | 0 | boundary_growth | 32 | 1200 | False | False |
| mboost_additive (seed 1) | 0 | patience_60 | 32 | 1200 | False | False |
| mboost_additive (seed 1) | 0 | repeated_cv_growth | 24 | 1200 | False | False |
| mboost_additive (seed 1) | 1 | fixed_argmin | 25 | 1200 | False | False |
| mboost_additive (seed 1) | 1 | boundary_growth | 25 | 1200 | False | False |
| mboost_additive (seed 1) | 1 | patience_60 | 25 | 1200 | False | False |
| mboost_additive (seed 1) | 1 | repeated_cv_growth | 27 | 1200 | False | False |
| mboost_additive (seed 1) | 2 | fixed_argmin | 62 | 1200 | False | False |
| mboost_additive (seed 1) | 2 | boundary_growth | 62 | 1200 | False | False |
| mboost_additive (seed 1) | 2 | patience_60 | 62 | 1200 | False | False |
| mboost_additive (seed 1) | 2 | repeated_cv_growth | 59 | 1200 | False | False |
| mboost_additive (seed 1) | 3 | fixed_argmin | 48 | 1200 | False | False |
| mboost_additive (seed 1) | 3 | boundary_growth | 48 | 1200 | False | False |
| mboost_additive (seed 1) | 3 | patience_60 | 48 | 1200 | False | False |
| mboost_additive (seed 1) | 3 | repeated_cv_growth | 51 | 1200 | False | False |
| mboost_hybrid (seed 2) | 0 | fixed_argmin | 1196 | 1200 | True | False |
| mboost_hybrid (seed 2) | 0 | boundary_growth | 2326 | 2400 | True | False |
| mboost_hybrid (seed 2) | 0 | patience_60 | 1196 | 1256 | False | False |
| mboost_hybrid (seed 2) | 0 | repeated_cv_growth | 2400 | 2400 | True | True |
| mboost_hybrid (seed 2) | 1 | fixed_argmin | 1115 | 1200 | False | False |
| mboost_hybrid (seed 2) | 1 | boundary_growth | 1734 | 2400 | False | False |
| mboost_hybrid (seed 2) | 1 | patience_60 | 1283 | 1343 | False | False |
| mboost_hybrid (seed 2) | 1 | repeated_cv_growth | 2266 | 2400 | False | True |
| mboost_hybrid (seed 2) | 2 | fixed_argmin | 1200 | 1200 | True | False |
| mboost_hybrid (seed 2) | 2 | boundary_growth | 2396 | 2400 | True | False |
| mboost_hybrid (seed 2) | 2 | patience_60 | 1482 | 1542 | False | False |
| mboost_hybrid (seed 2) | 2 | repeated_cv_growth | 2397 | 2400 | True | True |
| mboost_hybrid (seed 2) | 3 | fixed_argmin | 1115 | 1200 | False | False |
| mboost_hybrid (seed 2) | 3 | boundary_growth | 1734 | 2400 | False | False |
| mboost_hybrid (seed 2) | 3 | patience_60 | 1283 | 1343 | False | False |
| mboost_hybrid (seed 2) | 3 | repeated_cv_growth | 2266 | 2400 | False | True |
| mboost_additive (seed 2) | 0 | fixed_argmin | 17 | 1200 | False | False |
| mboost_additive (seed 2) | 0 | boundary_growth | 17 | 1200 | False | False |
| mboost_additive (seed 2) | 0 | patience_60 | 17 | 1200 | False | False |
| mboost_additive (seed 2) | 0 | repeated_cv_growth | 27 | 1200 | False | True |
| mboost_additive (seed 2) | 1 | fixed_argmin | 21 | 1200 | False | False |
| mboost_additive (seed 2) | 1 | boundary_growth | 21 | 1200 | False | False |
| mboost_additive (seed 2) | 1 | patience_60 | 21 | 1200 | False | False |
| mboost_additive (seed 2) | 1 | repeated_cv_growth | 23 | 1200 | False | True |
| mboost_additive (seed 2) | 2 | fixed_argmin | 33 | 1200 | False | False |
| mboost_additive (seed 2) | 2 | boundary_growth | 33 | 1200 | False | False |
| mboost_additive (seed 2) | 2 | patience_60 | 33 | 1200 | False | False |
| mboost_additive (seed 2) | 2 | repeated_cv_growth | 53 | 1200 | False | True |
| mboost_additive (seed 2) | 3 | fixed_argmin | 40 | 1200 | False | False |
| mboost_additive (seed 2) | 3 | boundary_growth | 40 | 1200 | False | False |
| mboost_additive (seed 2) | 3 | patience_60 | 40 | 1200 | False | False |
| mboost_additive (seed 2) | 3 | repeated_cv_growth | 40 | 1200 | False | True |
| mboost_hybrid (seed 3) | 0 | fixed_argmin | 1181 | 1200 | True | False |
| mboost_hybrid (seed 3) | 0 | boundary_growth | 2400 | 2400 | True | False |
| mboost_hybrid (seed 3) | 0 | patience_60 | 1675 | 1735 | False | False |
| mboost_hybrid (seed 3) | 0 | repeated_cv_growth | 2396 | 2400 | True | True |
| mboost_hybrid (seed 3) | 1 | fixed_argmin | 1198 | 1200 | True | False |
| mboost_hybrid (seed 3) | 1 | boundary_growth | 1932 | 2400 | False | False |
| mboost_hybrid (seed 3) | 1 | patience_60 | 1330 | 1390 | False | False |
| mboost_hybrid (seed 3) | 1 | repeated_cv_growth | 1809 | 2400 | False | True |
| mboost_hybrid (seed 3) | 2 | fixed_argmin | 1181 | 1200 | True | False |
| mboost_hybrid (seed 3) | 2 | boundary_growth | 2379 | 2400 | True | False |
| mboost_hybrid (seed 3) | 2 | patience_60 | 1675 | 1735 | False | False |
| mboost_hybrid (seed 3) | 2 | repeated_cv_growth | 2399 | 2400 | True | True |
| mboost_hybrid (seed 3) | 3 | fixed_argmin | 1198 | 1200 | True | False |
| mboost_hybrid (seed 3) | 3 | boundary_growth | 1932 | 2400 | False | False |
| mboost_hybrid (seed 3) | 3 | patience_60 | 1330 | 1390 | False | False |
| mboost_hybrid (seed 3) | 3 | repeated_cv_growth | 1809 | 2400 | False | True |
| mboost_additive (seed 3) | 0 | fixed_argmin | 34 | 1200 | False | False |
| mboost_additive (seed 3) | 0 | boundary_growth | 34 | 1200 | False | False |
| mboost_additive (seed 3) | 0 | patience_60 | 34 | 1200 | False | False |
| mboost_additive (seed 3) | 0 | repeated_cv_growth | 28 | 1200 | False | False |
| mboost_additive (seed 3) | 1 | fixed_argmin | 15 | 1200 | False | False |
| mboost_additive (seed 3) | 1 | boundary_growth | 15 | 1200 | False | False |
| mboost_additive (seed 3) | 1 | patience_60 | 15 | 1200 | False | False |
| mboost_additive (seed 3) | 1 | repeated_cv_growth | 26 | 1200 | False | False |
| mboost_additive (seed 3) | 2 | fixed_argmin | 74 | 1200 | False | False |
| mboost_additive (seed 3) | 2 | boundary_growth | 74 | 1200 | False | False |
| mboost_additive (seed 3) | 2 | patience_60 | 74 | 1200 | False | False |
| mboost_additive (seed 3) | 2 | repeated_cv_growth | 73 | 1200 | False | False |
| mboost_additive (seed 3) | 3 | fixed_argmin | 31 | 1200 | False | False |
| mboost_additive (seed 3) | 3 | boundary_growth | 31 | 1200 | False | False |
| mboost_additive (seed 3) | 3 | patience_60 | 31 | 1200 | False | False |
| mboost_additive (seed 3) | 3 | repeated_cv_growth | 53 | 1200 | False | False |
| mboost_hybrid (seed 4) | 0 | fixed_argmin | 1189 | 1200 | True | False |
| mboost_hybrid (seed 4) | 0 | boundary_growth | 1717 | 2400 | False | True |
| mboost_hybrid (seed 4) | 0 | patience_60 | 1183 | 1243 | False | True |
| mboost_hybrid (seed 4) | 0 | repeated_cv_growth | 2397 | 2400 | True | False |
| mboost_hybrid (seed 4) | 1 | fixed_argmin | 884 | 1200 | False | False |
| mboost_hybrid (seed 4) | 1 | boundary_growth | 884 | 1200 | False | True |
| mboost_hybrid (seed 4) | 1 | patience_60 | 884 | 1200 | False | True |
| mboost_hybrid (seed 4) | 1 | repeated_cv_growth | 2389 | 2400 | True | False |
| mboost_hybrid (seed 4) | 2 | fixed_argmin | 1193 | 1200 | True | False |
| mboost_hybrid (seed 4) | 2 | boundary_growth | 2213 | 2400 | False | True |
| mboost_hybrid (seed 4) | 2 | patience_60 | 1447 | 1507 | False | True |
| mboost_hybrid (seed 4) | 2 | repeated_cv_growth | 2395 | 2400 | True | False |
| mboost_hybrid (seed 4) | 3 | fixed_argmin | 884 | 1200 | False | False |
| mboost_hybrid (seed 4) | 3 | boundary_growth | 884 | 1200 | False | True |
| mboost_hybrid (seed 4) | 3 | patience_60 | 884 | 1200 | False | True |
| mboost_hybrid (seed 4) | 3 | repeated_cv_growth | 2389 | 2400 | True | False |
| mboost_additive (seed 4) | 0 | fixed_argmin | 23 | 1200 | False | False |
| mboost_additive (seed 4) | 0 | boundary_growth | 23 | 1200 | False | False |
| mboost_additive (seed 4) | 0 | patience_60 | 23 | 1200 | False | False |
| mboost_additive (seed 4) | 0 | repeated_cv_growth | 56 | 1200 | False | False |
| mboost_additive (seed 4) | 1 | fixed_argmin | 40 | 1200 | False | False |
| mboost_additive (seed 4) | 1 | boundary_growth | 40 | 1200 | False | False |
| mboost_additive (seed 4) | 1 | patience_60 | 40 | 1200 | False | False |
| mboost_additive (seed 4) | 1 | repeated_cv_growth | 41 | 1200 | False | False |
| mboost_additive (seed 4) | 2 | fixed_argmin | 44 | 1200 | False | False |
| mboost_additive (seed 4) | 2 | boundary_growth | 44 | 1200 | False | False |
| mboost_additive (seed 4) | 2 | patience_60 | 44 | 1200 | False | False |
| mboost_additive (seed 4) | 2 | repeated_cv_growth | 113 | 1200 | False | False |
| mboost_additive (seed 4) | 3 | fixed_argmin | 91 | 1200 | False | False |
| mboost_additive (seed 4) | 3 | boundary_growth | 91 | 1200 | False | False |
| mboost_additive (seed 4) | 3 | patience_60 | 91 | 1200 | False | False |
| mboost_additive (seed 4) | 3 | repeated_cv_growth | 94 | 1200 | False | False |

## smooth_sparse

| method | tier | rule | per-seed paired test-loss difference | median | seeds with an unresolved boundary at the tier's pick |
|---|---|---|---|---|---|
| mboost_hybrid | small | boundary_growth | +0, +0.0001838, +3.379e-05, +0, -0.00139 | +0 | 0 |
| mboost_hybrid | small | patience_60 | +0, +0.0001838, +3.379e-05, +0, -0.00139 | +0 | 0 |
| mboost_hybrid | small | repeated_cv_growth | +0.0002314, +0.0001968, +0.0007828, +4.081e-06, -0.0004495 | +0.0001968 | 0 |
| mboost_hybrid | tuned | boundary_growth | +0, +0, +0, +0.0001307, -0.001105 | +0 | 0 |
| mboost_hybrid | tuned | patience_60 | +0, +0, +0, +0.0001307, -0.001105 | +0 | 0 |
| mboost_hybrid | tuned | repeated_cv_growth | -0.001312, +0.0001362, +0.0005716, +0.0003313, -0.0004008 | +0.0001362 | 0 |
| mboost_additive | small | boundary_growth | +0, +0, +0, +0, +0 | +0 | 0 |
| mboost_additive | small | patience_60 | +0, +0, +0, +0, +0 | +0 | 0 |
| mboost_additive | small | repeated_cv_growth | -0.002278, -0.0008173, -0.000493, -0.002484, -5.31e-05 | -0.0008173 | 0 |
| mboost_additive | tuned | boundary_growth | +0, +0, +0, +0, +0 | +0 | 0 |
| mboost_additive | tuned | patience_60 | +0, +0, +0, +0, +0 | +0 | 0 |
| mboost_additive | tuned | repeated_cv_growth | -0.002402, +0.002527, -0.0005375, -0.0003689, -6.385e-05 | -0.0003689 | 0 |

Chosen iterations and exhaustion, per configuration (`fixed_argmin` -> `boundary_growth` -> `patience_60` -> `repeated_cv_growth`):

| method | config | rule | mstop | largest grid | at boundary | tuned-tier pick differs from fixed rule |
|---|---|---|---|---|---|---|
| mboost_hybrid (seed 0) | 0 | fixed_argmin | 175 | 1200 | False | False |
| mboost_hybrid (seed 0) | 0 | boundary_growth | 175 | 1200 | False | False |
| mboost_hybrid (seed 0) | 0 | patience_60 | 175 | 1200 | False | False |
| mboost_hybrid (seed 0) | 0 | repeated_cv_growth | 169 | 1200 | False | False |
| mboost_hybrid (seed 0) | 1 | fixed_argmin | 121 | 1200 | False | False |
| mboost_hybrid (seed 0) | 1 | boundary_growth | 115 | 1200 | False | False |
| mboost_hybrid (seed 0) | 1 | patience_60 | 115 | 1200 | False | False |
| mboost_hybrid (seed 0) | 1 | repeated_cv_growth | 105 | 1200 | False | False |
| mboost_hybrid (seed 0) | 2 | fixed_argmin | 149 | 1200 | False | False |
| mboost_hybrid (seed 0) | 2 | boundary_growth | 149 | 1200 | False | False |
| mboost_hybrid (seed 0) | 2 | patience_60 | 149 | 1200 | False | False |
| mboost_hybrid (seed 0) | 2 | repeated_cv_growth | 163 | 1200 | False | False |
| mboost_hybrid (seed 0) | 3 | fixed_argmin | 121 | 1200 | False | False |
| mboost_hybrid (seed 0) | 3 | boundary_growth | 115 | 1200 | False | False |
| mboost_hybrid (seed 0) | 3 | patience_60 | 115 | 1200 | False | False |
| mboost_hybrid (seed 0) | 3 | repeated_cv_growth | 105 | 1200 | False | False |
| mboost_additive (seed 0) | 0 | fixed_argmin | 136 | 1200 | False | False |
| mboost_additive (seed 0) | 0 | boundary_growth | 136 | 1200 | False | False |
| mboost_additive (seed 0) | 0 | patience_60 | 136 | 1200 | False | False |
| mboost_additive (seed 0) | 0 | repeated_cv_growth | 173 | 1200 | False | True |
| mboost_additive (seed 0) | 1 | fixed_argmin | 104 | 1200 | False | False |
| mboost_additive (seed 0) | 1 | boundary_growth | 104 | 1200 | False | False |
| mboost_additive (seed 0) | 1 | patience_60 | 104 | 1200 | False | False |
| mboost_additive (seed 0) | 1 | repeated_cv_growth | 116 | 1200 | False | True |
| mboost_additive (seed 0) | 2 | fixed_argmin | 282 | 1200 | False | False |
| mboost_additive (seed 0) | 2 | boundary_growth | 282 | 1200 | False | False |
| mboost_additive (seed 0) | 2 | patience_60 | 282 | 1200 | False | False |
| mboost_additive (seed 0) | 2 | repeated_cv_growth | 360 | 1200 | False | True |
| mboost_additive (seed 0) | 3 | fixed_argmin | 206 | 1200 | False | False |
| mboost_additive (seed 0) | 3 | boundary_growth | 206 | 1200 | False | False |
| mboost_additive (seed 0) | 3 | patience_60 | 206 | 1200 | False | False |
| mboost_additive (seed 0) | 3 | repeated_cv_growth | 237 | 1200 | False | True |
| mboost_hybrid (seed 1) | 0 | fixed_argmin | 296 | 1200 | False | False |
| mboost_hybrid (seed 1) | 0 | boundary_growth | 321 | 1200 | False | False |
| mboost_hybrid (seed 1) | 0 | patience_60 | 321 | 1200 | False | False |
| mboost_hybrid (seed 1) | 0 | repeated_cv_growth | 172 | 1200 | False | False |
| mboost_hybrid (seed 1) | 1 | fixed_argmin | 105 | 1200 | False | False |
| mboost_hybrid (seed 1) | 1 | boundary_growth | 143 | 1200 | False | False |
| mboost_hybrid (seed 1) | 1 | patience_60 | 143 | 1200 | False | False |
| mboost_hybrid (seed 1) | 1 | repeated_cv_growth | 97 | 1200 | False | False |
| mboost_hybrid (seed 1) | 2 | fixed_argmin | 202 | 1200 | False | False |
| mboost_hybrid (seed 1) | 2 | boundary_growth | 202 | 1200 | False | False |
| mboost_hybrid (seed 1) | 2 | patience_60 | 202 | 1200 | False | False |
| mboost_hybrid (seed 1) | 2 | repeated_cv_growth | 185 | 1200 | False | False |
| mboost_hybrid (seed 1) | 3 | fixed_argmin | 105 | 1200 | False | False |
| mboost_hybrid (seed 1) | 3 | boundary_growth | 143 | 1200 | False | False |
| mboost_hybrid (seed 1) | 3 | patience_60 | 143 | 1200 | False | False |
| mboost_hybrid (seed 1) | 3 | repeated_cv_growth | 97 | 1200 | False | False |
| mboost_additive (seed 1) | 0 | fixed_argmin | 160 | 1200 | False | False |
| mboost_additive (seed 1) | 0 | boundary_growth | 160 | 1200 | False | False |
| mboost_additive (seed 1) | 0 | patience_60 | 160 | 1200 | False | False |
| mboost_additive (seed 1) | 0 | repeated_cv_growth | 182 | 1200 | False | True |
| mboost_additive (seed 1) | 1 | fixed_argmin | 104 | 1200 | False | False |
| mboost_additive (seed 1) | 1 | boundary_growth | 104 | 1200 | False | False |
| mboost_additive (seed 1) | 1 | patience_60 | 104 | 1200 | False | False |
| mboost_additive (seed 1) | 1 | repeated_cv_growth | 105 | 1200 | False | True |
| mboost_additive (seed 1) | 2 | fixed_argmin | 334 | 1200 | False | False |
| mboost_additive (seed 1) | 2 | boundary_growth | 334 | 1200 | False | False |
| mboost_additive (seed 1) | 2 | patience_60 | 334 | 1200 | False | False |
| mboost_additive (seed 1) | 2 | repeated_cv_growth | 371 | 1200 | False | True |
| mboost_additive (seed 1) | 3 | fixed_argmin | 221 | 1200 | False | False |
| mboost_additive (seed 1) | 3 | boundary_growth | 221 | 1200 | False | False |
| mboost_additive (seed 1) | 3 | patience_60 | 221 | 1200 | False | False |
| mboost_additive (seed 1) | 3 | repeated_cv_growth | 218 | 1200 | False | True |
| mboost_hybrid (seed 2) | 0 | fixed_argmin | 182 | 1200 | False | False |
| mboost_hybrid (seed 2) | 0 | boundary_growth | 183 | 1200 | False | False |
| mboost_hybrid (seed 2) | 0 | patience_60 | 183 | 1200 | False | False |
| mboost_hybrid (seed 2) | 0 | repeated_cv_growth | 207 | 1200 | False | False |
| mboost_hybrid (seed 2) | 1 | fixed_argmin | 145 | 1200 | False | False |
| mboost_hybrid (seed 2) | 1 | boundary_growth | 141 | 1200 | False | False |
| mboost_hybrid (seed 2) | 1 | patience_60 | 141 | 1200 | False | False |
| mboost_hybrid (seed 2) | 1 | repeated_cv_growth | 100 | 1200 | False | False |
| mboost_hybrid (seed 2) | 2 | fixed_argmin | 167 | 1200 | False | False |
| mboost_hybrid (seed 2) | 2 | boundary_growth | 167 | 1200 | False | False |
| mboost_hybrid (seed 2) | 2 | patience_60 | 167 | 1200 | False | False |
| mboost_hybrid (seed 2) | 2 | repeated_cv_growth | 198 | 1200 | False | False |
| mboost_hybrid (seed 2) | 3 | fixed_argmin | 145 | 1200 | False | False |
| mboost_hybrid (seed 2) | 3 | boundary_growth | 141 | 1200 | False | False |
| mboost_hybrid (seed 2) | 3 | patience_60 | 141 | 1200 | False | False |
| mboost_hybrid (seed 2) | 3 | repeated_cv_growth | 100 | 1200 | False | False |
| mboost_additive (seed 2) | 0 | fixed_argmin | 195 | 1200 | False | False |
| mboost_additive (seed 2) | 0 | boundary_growth | 195 | 1200 | False | False |
| mboost_additive (seed 2) | 0 | patience_60 | 195 | 1200 | False | False |
| mboost_additive (seed 2) | 0 | repeated_cv_growth | 335 | 1200 | False | True |
| mboost_additive (seed 2) | 1 | fixed_argmin | 97 | 1200 | False | False |
| mboost_additive (seed 2) | 1 | boundary_growth | 97 | 1200 | False | False |
| mboost_additive (seed 2) | 1 | patience_60 | 97 | 1200 | False | False |
| mboost_additive (seed 2) | 1 | repeated_cv_growth | 154 | 1200 | False | True |
| mboost_additive (seed 2) | 2 | fixed_argmin | 398 | 1200 | False | False |
| mboost_additive (seed 2) | 2 | boundary_growth | 398 | 1200 | False | False |
| mboost_additive (seed 2) | 2 | patience_60 | 398 | 1200 | False | False |
| mboost_additive (seed 2) | 2 | repeated_cv_growth | 716 | 1200 | False | True |
| mboost_additive (seed 2) | 3 | fixed_argmin | 198 | 1200 | False | False |
| mboost_additive (seed 2) | 3 | boundary_growth | 198 | 1200 | False | False |
| mboost_additive (seed 2) | 3 | patience_60 | 198 | 1200 | False | False |
| mboost_additive (seed 2) | 3 | repeated_cv_growth | 281 | 1200 | False | True |
| mboost_hybrid (seed 3) | 0 | fixed_argmin | 199 | 1200 | False | False |
| mboost_hybrid (seed 3) | 0 | boundary_growth | 199 | 1200 | False | False |
| mboost_hybrid (seed 3) | 0 | patience_60 | 199 | 1200 | False | False |
| mboost_hybrid (seed 3) | 0 | repeated_cv_growth | 203 | 1200 | False | False |
| mboost_hybrid (seed 3) | 1 | fixed_argmin | 104 | 1200 | False | False |
| mboost_hybrid (seed 3) | 1 | boundary_growth | 104 | 1200 | False | False |
| mboost_hybrid (seed 3) | 1 | patience_60 | 104 | 1200 | False | False |
| mboost_hybrid (seed 3) | 1 | repeated_cv_growth | 99 | 1200 | False | False |
| mboost_hybrid (seed 3) | 2 | fixed_argmin | 205 | 1200 | False | False |
| mboost_hybrid (seed 3) | 2 | boundary_growth | 206 | 1200 | False | False |
| mboost_hybrid (seed 3) | 2 | patience_60 | 206 | 1200 | False | False |
| mboost_hybrid (seed 3) | 2 | repeated_cv_growth | 153 | 1200 | False | False |
| mboost_hybrid (seed 3) | 3 | fixed_argmin | 104 | 1200 | False | False |
| mboost_hybrid (seed 3) | 3 | boundary_growth | 104 | 1200 | False | False |
| mboost_hybrid (seed 3) | 3 | patience_60 | 104 | 1200 | False | False |
| mboost_hybrid (seed 3) | 3 | repeated_cv_growth | 99 | 1200 | False | False |
| mboost_additive (seed 3) | 0 | fixed_argmin | 136 | 1200 | False | False |
| mboost_additive (seed 3) | 0 | boundary_growth | 136 | 1200 | False | False |
| mboost_additive (seed 3) | 0 | patience_60 | 136 | 1200 | False | False |
| mboost_additive (seed 3) | 0 | repeated_cv_growth | 254 | 1200 | False | True |
| mboost_additive (seed 3) | 1 | fixed_argmin | 92 | 1200 | False | False |
| mboost_additive (seed 3) | 1 | boundary_growth | 92 | 1200 | False | False |
| mboost_additive (seed 3) | 1 | patience_60 | 92 | 1200 | False | False |
| mboost_additive (seed 3) | 1 | repeated_cv_growth | 112 | 1200 | False | True |
| mboost_additive (seed 3) | 2 | fixed_argmin | 278 | 1200 | False | False |
| mboost_additive (seed 3) | 2 | boundary_growth | 278 | 1200 | False | False |
| mboost_additive (seed 3) | 2 | patience_60 | 278 | 1200 | False | False |
| mboost_additive (seed 3) | 2 | repeated_cv_growth | 520 | 1200 | False | True |
| mboost_additive (seed 3) | 3 | fixed_argmin | 189 | 1200 | False | False |
| mboost_additive (seed 3) | 3 | boundary_growth | 189 | 1200 | False | False |
| mboost_additive (seed 3) | 3 | patience_60 | 189 | 1200 | False | False |
| mboost_additive (seed 3) | 3 | repeated_cv_growth | 233 | 1200 | False | True |
| mboost_hybrid (seed 4) | 0 | fixed_argmin | 224 | 1200 | False | False |
| mboost_hybrid (seed 4) | 0 | boundary_growth | 165 | 1200 | False | False |
| mboost_hybrid (seed 4) | 0 | patience_60 | 165 | 1200 | False | False |
| mboost_hybrid (seed 4) | 0 | repeated_cv_growth | 215 | 1200 | False | False |
| mboost_hybrid (seed 4) | 1 | fixed_argmin | 94 | 1200 | False | False |
| mboost_hybrid (seed 4) | 1 | boundary_growth | 94 | 1200 | False | False |
| mboost_hybrid (seed 4) | 1 | patience_60 | 94 | 1200 | False | False |
| mboost_hybrid (seed 4) | 1 | repeated_cv_growth | 99 | 1200 | False | False |
| mboost_hybrid (seed 4) | 2 | fixed_argmin | 211 | 1200 | False | False |
| mboost_hybrid (seed 4) | 2 | boundary_growth | 188 | 1200 | False | False |
| mboost_hybrid (seed 4) | 2 | patience_60 | 188 | 1200 | False | False |
| mboost_hybrid (seed 4) | 2 | repeated_cv_growth | 206 | 1200 | False | False |
| mboost_hybrid (seed 4) | 3 | fixed_argmin | 94 | 1200 | False | False |
| mboost_hybrid (seed 4) | 3 | boundary_growth | 94 | 1200 | False | False |
| mboost_hybrid (seed 4) | 3 | patience_60 | 94 | 1200 | False | False |
| mboost_hybrid (seed 4) | 3 | repeated_cv_growth | 99 | 1200 | False | False |
| mboost_additive (seed 4) | 0 | fixed_argmin | 370 | 1200 | False | False |
| mboost_additive (seed 4) | 0 | boundary_growth | 370 | 1200 | False | False |
| mboost_additive (seed 4) | 0 | patience_60 | 370 | 1200 | False | False |
| mboost_additive (seed 4) | 0 | repeated_cv_growth | 309 | 1200 | False | True |
| mboost_additive (seed 4) | 1 | fixed_argmin | 187 | 1200 | False | False |
| mboost_additive (seed 4) | 1 | boundary_growth | 187 | 1200 | False | False |
| mboost_additive (seed 4) | 1 | patience_60 | 187 | 1200 | False | False |
| mboost_additive (seed 4) | 1 | repeated_cv_growth | 155 | 1200 | False | True |
| mboost_additive (seed 4) | 2 | fixed_argmin | 802 | 1200 | False | False |
| mboost_additive (seed 4) | 2 | boundary_growth | 802 | 1200 | False | False |
| mboost_additive (seed 4) | 2 | patience_60 | 802 | 1200 | False | False |
| mboost_additive (seed 4) | 2 | repeated_cv_growth | 615 | 1200 | False | True |
| mboost_additive (seed 4) | 3 | fixed_argmin | 374 | 1200 | False | False |
| mboost_additive (seed 4) | 3 | boundary_growth | 374 | 1200 | False | False |
| mboost_additive (seed 4) | 3 | patience_60 | 374 | 1200 | False | False |
| mboost_additive (seed 4) | 3 | repeated_cv_growth | 312 | 1200 | False | True |

## housing_iid

| method | tier | rule | per-seed paired test-loss difference | median | seeds with an unresolved boundary at the tier's pick |
|---|---|---|---|---|---|
| mboost_hybrid | small | boundary_growth | -0.001003, +0, +0 | +0 | 0 |
| mboost_hybrid | small | patience_60 | +0, +0, +0 | +0 | 0 |
| mboost_hybrid | small | repeated_cv_growth | +0.0005142, -0.0005679, -0.0008216 | -0.0005679 | 0 |
| mboost_hybrid | tuned | boundary_growth | +8.825e-06, -0.0005942, +0 | +0 | 0 |
| mboost_hybrid | tuned | patience_60 | +8.825e-06, -0.0001598, +0 | +0 | 0 |
| mboost_hybrid | tuned | repeated_cv_growth | +0.0004406, -0.0006186, -0.001828 | -0.0006186 | 0 |
| mboost_additive | small | boundary_growth | -0.003052, -0.002553, -0.002503 | -0.002553 | 3 |
| mboost_additive | small | patience_60 | -0.003052, -0.002553, -0.002503 | -0.002553 | 3 |
| mboost_additive | small | repeated_cv_growth | +0.002464, -0.002548, +0.104 | +0.002464 | 1 |
| mboost_additive | tuned | boundary_growth | +0, -0.001958, -0.0005365 | -0.0005365 | 2 |
| mboost_additive | tuned | patience_60 | +0, -0.001958, -0.000539 | -0.000539 | 1 |
| mboost_additive | tuned | repeated_cv_growth | +0.007051, -0.001568, +0.08069 | +0.007051 | 0 |

Chosen iterations and exhaustion, per configuration (`fixed_argmin` -> `boundary_growth` -> `patience_60` -> `repeated_cv_growth`):

| method | config | rule | mstop | largest grid | at boundary | tuned-tier pick differs from fixed rule |
|---|---|---|---|---|---|---|
| mboost_hybrid (seed 0) | 0 | fixed_argmin | 1200 | 1200 | True | False |
| mboost_hybrid (seed 0) | 0 | boundary_growth | 1945 | 2400 | False | False |
| mboost_hybrid (seed 0) | 0 | patience_60 | 1200 | 1260 | False | False |
| mboost_hybrid (seed 0) | 0 | repeated_cv_growth | 1066 | 1200 | False | False |
| mboost_hybrid (seed 0) | 1 | fixed_argmin | 1083 | 1200 | False | False |
| mboost_hybrid (seed 0) | 1 | boundary_growth | 1111 | 1200 | False | False |
| mboost_hybrid (seed 0) | 1 | patience_60 | 1111 | 1200 | False | False |
| mboost_hybrid (seed 0) | 1 | repeated_cv_growth | 567 | 1200 | False | False |
| mboost_hybrid (seed 0) | 2 | fixed_argmin | 1190 | 1200 | True | False |
| mboost_hybrid (seed 0) | 2 | boundary_growth | 1576 | 2400 | False | False |
| mboost_hybrid (seed 0) | 2 | patience_60 | 1157 | 1217 | False | False |
| mboost_hybrid (seed 0) | 2 | repeated_cv_growth | 614 | 1200 | False | False |
| mboost_hybrid (seed 0) | 3 | fixed_argmin | 1083 | 1200 | False | False |
| mboost_hybrid (seed 0) | 3 | boundary_growth | 1111 | 1200 | False | False |
| mboost_hybrid (seed 0) | 3 | patience_60 | 1111 | 1200 | False | False |
| mboost_hybrid (seed 0) | 3 | repeated_cv_growth | 567 | 1200 | False | False |
| mboost_additive (seed 0) | 0 | fixed_argmin | 1199 | 1200 | True | False |
| mboost_additive (seed 0) | 0 | boundary_growth | 2395 | 2400 | True | False |
| mboost_additive (seed 0) | 0 | patience_60 | 2395 | 2400 | True | False |
| mboost_additive (seed 0) | 0 | repeated_cv_growth | 862 | 1200 | False | True |
| mboost_additive (seed 0) | 1 | fixed_argmin | 760 | 1200 | False | False |
| mboost_additive (seed 0) | 1 | boundary_growth | 760 | 1200 | False | False |
| mboost_additive (seed 0) | 1 | patience_60 | 760 | 1200 | False | False |
| mboost_additive (seed 0) | 1 | repeated_cv_growth | 77 | 1200 | False | True |
| mboost_additive (seed 0) | 2 | fixed_argmin | 1199 | 1200 | True | False |
| mboost_additive (seed 0) | 2 | boundary_growth | 2399 | 2400 | True | False |
| mboost_additive (seed 0) | 2 | patience_60 | 2399 | 2400 | True | False |
| mboost_additive (seed 0) | 2 | repeated_cv_growth | 1797 | 2400 | False | True |
| mboost_additive (seed 0) | 3 | fixed_argmin | 1200 | 1200 | True | False |
| mboost_additive (seed 0) | 3 | boundary_growth | 1611 | 2400 | False | False |
| mboost_additive (seed 0) | 3 | patience_60 | 1611 | 1671 | False | False |
| mboost_additive (seed 0) | 3 | repeated_cv_growth | 160 | 1200 | False | True |
| mboost_hybrid (seed 1) | 0 | fixed_argmin | 1092 | 1200 | False | False |
| mboost_hybrid (seed 1) | 0 | boundary_growth | 1092 | 1200 | False | False |
| mboost_hybrid (seed 1) | 0 | patience_60 | 1092 | 1200 | False | False |
| mboost_hybrid (seed 1) | 0 | repeated_cv_growth | 1322 | 2400 | False | True |
| mboost_hybrid (seed 1) | 1 | fixed_argmin | 625 | 1200 | False | False |
| mboost_hybrid (seed 1) | 1 | boundary_growth | 622 | 1200 | False | False |
| mboost_hybrid (seed 1) | 1 | patience_60 | 622 | 1200 | False | False |
| mboost_hybrid (seed 1) | 1 | repeated_cv_growth | 686 | 1200 | False | True |
| mboost_hybrid (seed 1) | 2 | fixed_argmin | 1200 | 1200 | True | False |
| mboost_hybrid (seed 1) | 2 | boundary_growth | 2228 | 2400 | False | False |
| mboost_hybrid (seed 1) | 2 | patience_60 | 1347 | 1407 | False | False |
| mboost_hybrid (seed 1) | 2 | repeated_cv_growth | 1454 | 2400 | False | True |
| mboost_hybrid (seed 1) | 3 | fixed_argmin | 625 | 1200 | False | False |
| mboost_hybrid (seed 1) | 3 | boundary_growth | 622 | 1200 | False | False |
| mboost_hybrid (seed 1) | 3 | patience_60 | 622 | 1200 | False | False |
| mboost_hybrid (seed 1) | 3 | repeated_cv_growth | 686 | 1200 | False | True |
| mboost_additive (seed 1) | 0 | fixed_argmin | 1199 | 1200 | True | False |
| mboost_additive (seed 1) | 0 | boundary_growth | 2400 | 2400 | True | False |
| mboost_additive (seed 1) | 0 | patience_60 | 2400 | 2400 | True | False |
| mboost_additive (seed 1) | 0 | repeated_cv_growth | 2399 | 2400 | True | False |
| mboost_additive (seed 1) | 1 | fixed_argmin | 1198 | 1200 | True | False |
| mboost_additive (seed 1) | 1 | boundary_growth | 2383 | 2400 | True | False |
| mboost_additive (seed 1) | 1 | patience_60 | 2383 | 2400 | True | False |
| mboost_additive (seed 1) | 1 | repeated_cv_growth | 1866 | 2400 | False | False |
| mboost_additive (seed 1) | 2 | fixed_argmin | 1200 | 1200 | True | False |
| mboost_additive (seed 1) | 2 | boundary_growth | 2400 | 2400 | True | False |
| mboost_additive (seed 1) | 2 | patience_60 | 2400 | 2400 | True | False |
| mboost_additive (seed 1) | 2 | repeated_cv_growth | 2400 | 2400 | True | False |
| mboost_additive (seed 1) | 3 | fixed_argmin | 1200 | 1200 | True | False |
| mboost_additive (seed 1) | 3 | boundary_growth | 2400 | 2400 | True | False |
| mboost_additive (seed 1) | 3 | patience_60 | 2400 | 2400 | True | False |
| mboost_additive (seed 1) | 3 | repeated_cv_growth | 2399 | 2400 | True | False |
| mboost_hybrid (seed 2) | 0 | fixed_argmin | 758 | 1200 | False | False |
| mboost_hybrid (seed 2) | 0 | boundary_growth | 758 | 1200 | False | False |
| mboost_hybrid (seed 2) | 0 | patience_60 | 758 | 1200 | False | False |
| mboost_hybrid (seed 2) | 0 | repeated_cv_growth | 1113 | 1200 | False | True |
| mboost_hybrid (seed 2) | 1 | fixed_argmin | 276 | 1200 | False | False |
| mboost_hybrid (seed 2) | 1 | boundary_growth | 300 | 1200 | False | False |
| mboost_hybrid (seed 2) | 1 | patience_60 | 300 | 1200 | False | False |
| mboost_hybrid (seed 2) | 1 | repeated_cv_growth | 985 | 1200 | False | True |
| mboost_hybrid (seed 2) | 2 | fixed_argmin | 970 | 1200 | False | False |
| mboost_hybrid (seed 2) | 2 | boundary_growth | 953 | 1200 | False | False |
| mboost_hybrid (seed 2) | 2 | patience_60 | 953 | 1200 | False | False |
| mboost_hybrid (seed 2) | 2 | repeated_cv_growth | 652 | 1200 | False | True |
| mboost_hybrid (seed 2) | 3 | fixed_argmin | 276 | 1200 | False | False |
| mboost_hybrid (seed 2) | 3 | boundary_growth | 300 | 1200 | False | False |
| mboost_hybrid (seed 2) | 3 | patience_60 | 300 | 1200 | False | False |
| mboost_hybrid (seed 2) | 3 | repeated_cv_growth | 985 | 1200 | False | True |
| mboost_additive (seed 2) | 0 | fixed_argmin | 1200 | 1200 | True | False |
| mboost_additive (seed 2) | 0 | boundary_growth | 2400 | 2400 | True | False |
| mboost_additive (seed 2) | 0 | patience_60 | 2400 | 2400 | True | False |
| mboost_additive (seed 2) | 0 | repeated_cv_growth | 11 | 1200 | False | False |
| mboost_additive (seed 2) | 1 | fixed_argmin | 1200 | 1200 | True | False |
| mboost_additive (seed 2) | 1 | boundary_growth | 2398 | 2400 | True | False |
| mboost_additive (seed 2) | 1 | patience_60 | 2246 | 2306 | False | False |
| mboost_additive (seed 2) | 1 | repeated_cv_growth | 16 | 1200 | False | False |
| mboost_additive (seed 2) | 2 | fixed_argmin | 1200 | 1200 | True | False |
| mboost_additive (seed 2) | 2 | boundary_growth | 2398 | 2400 | True | False |
| mboost_additive (seed 2) | 2 | patience_60 | 2398 | 2400 | True | False |
| mboost_additive (seed 2) | 2 | repeated_cv_growth | 22 | 1200 | False | False |
| mboost_additive (seed 2) | 3 | fixed_argmin | 1199 | 1200 | True | False |
| mboost_additive (seed 2) | 3 | boundary_growth | 2397 | 2400 | True | False |
| mboost_additive (seed 2) | 3 | patience_60 | 2397 | 2400 | True | False |
| mboost_additive (seed 2) | 3 | repeated_cv_growth | 31 | 1200 | False | False |

## Protocol notes

- The original #509 diagnostics (`benches/verify_selection/`, `docs/benchmarks/path_diagnostics_509.json`, `~/workspace/d5-stopping-treeform/`) are unrecoverable: no commit, branch or local file ever contained them. They are not recreated under their old names; this directory is a new, differently named experiment.
- `fixed_argmin` reproduces the shootout path exactly (`preprocessing="refit"`); the growing rules use `tune_mstop`'s native route, whose per-fold preprocessing is fixed, because grid extension with refitted preprocessing is not supported (issue #189). The panel digests record which route produced each curve.
- All recorded panels have zero missing cells, so 'preprocessing inside training folds' holds trivially; the count is stored per job, not assumed.
- Fold-level risk curves are stored in full for `fixed_argmin` (one split) and as the aggregate curve plus per-fold scores at the chosen iteration for the growing rules, which is what `tune_mstop` exposes publicly.
- No public library defaults changed; the 1-SE option is untouched.

