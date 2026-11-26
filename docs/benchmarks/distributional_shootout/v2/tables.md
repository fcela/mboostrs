## Planted-truth tasks

16 tasks. Mean rank, gap to best and best-on count use the 16 tasks completed by every arm that finished at least 90% of its runs; other arms show a dash.

| Arm | Completed | Mean rank | Skill, median | Skill, mean | Median gap to best | Best on | Calibration MAE | Cov 50 | Cov 90 | Cov 98 | Median s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mboostrs.LSS_select_noncyclic | 48/48 | 5.25 | 0.325 | 0.360 | 0.002 | 2 | 0.016 | 0.497 | 0.905 | 0.977 | 8.1 |
| mboostrs.fit_lss_trees | 48/48 | 5.34 | 0.410 | 0.378 | 0.006 | 5 | 0.025 | 0.512 | 0.879 | 0.953 | 5.3 |
| mboostrs.fit | 48/48 | 5.41 | 0.410 | 0.381 | 0.010 | 3 | 0.025 | 0.513 | 0.881 | 0.956 | 4.6 |
| mboostrs.LSS_select | 48/48 | 6.25 | 0.326 | 0.360 | 0.005 | 3 | 0.016 | 0.492 | 0.903 | 0.977 | 6.2 |
| mboostrs.quantiles | 48/48 | 6.44 | 0.345 | 0.363 | 0.012 | 4 | 0.011 | 0.498 | 0.887 | 0.975 | 12.5 |
| hessboost.tuned | 48/48 | 7.75 | 0.381 | 0.366 | 0.033 | 0 | 0.015 | 0.492 | 0.889 | 0.969 | 2.8 |
| drforest.mmd | 48/48 | 8.75 | 0.361 | 0.363 | 0.038 | 1 | 0.022 | 0.553 | 0.928 | 0.985 | 11.3 |
| distreebution.crps_rf_tuned | 48/48 | 9.09 | 0.373 | 0.367 | 0.043 | 0 | 0.012 | 0.504 | 0.901 | 0.975 | 16.9 |
| drforest.tuned | 48/48 | 9.19 | 0.360 | 0.363 | 0.037 | 0 | 0.022 | 0.554 | 0.928 | 0.985 | 34.4 |
| mboostrs.GaussianLSS | 48/48 | 10.06 | 0.302 | 0.349 | 0.031 | 0 | 0.029 | 0.526 | 0.915 | 0.972 | 0.9 |
| lightgbm.quantiles | 48/48 | 10.31 | 0.367 | 0.370 | 0.038 | 1 | 0.025 | 0.450 | 0.840 | 0.958 | 2.5 |
| drforest.cart | 48/48 | 10.50 | 0.357 | 0.357 | 0.037 | 0 | 0.022 | 0.551 | 0.926 | 0.985 | 6.3 |
| distreebution.pmqrf | 48/48 | 10.56 | 0.367 | 0.361 | 0.043 | 0 | 0.012 | 0.498 | 0.897 | 0.973 | 91.0 |
| distreebution.crps_rf | 48/48 | 10.97 | 0.369 | 0.360 | 0.043 | 0 | 0.012 | 0.501 | 0.899 | 0.973 | 6.6 |
| catboost.uncertainty | 48/48 | 13.44 | 0.327 | 0.350 | 0.049 | 0 | 0.031 | 0.484 | 0.893 | 0.962 | 1.9 |
| ngboost.normal | 48/48 | 13.88 | 0.316 | 0.337 | 0.078 | 0 | 0.030 | 0.516 | 0.903 | 0.967 | 15.7 |
| qrf | 48/48 | 14.62 | 0.353 | 0.348 | 0.075 | 0 | 0.013 | 0.493 | 0.880 | 0.959 | 3.2 |
| reference.gaussian_linear | 48/48 | 16.06 | 0.219 | 0.303 | 0.132 | 0 | 0.031 | 0.569 | 0.913 | 0.973 | 0.0 |
| mboostrs.LSS_select_tree | 48/48 | 17.69 | 0.321 | 0.319 | 0.118 | 0 | 0.020 | 0.492 | 0.893 | 0.972 | 11.7 |
| hessboost.normal | 48/48 | 18.75 | 0.170 | < −10 | 0.204 | 0 | 0.029 | 0.551 | 0.903 | 0.967 | 0.2 |
| reference.climatology | 48/48 | 20.69 | 0.000 | 0.000 | 0.704 | 0 | 0.011 | 0.499 | 0.900 | 0.978 | 0.0 |

### mboostrs.LSS_select against each arm

| Other arm | Tasks | CRPS ratio (geometric mean) | 95% interval | Wins | Ties | Losses | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| hessboost.normal | 16 | 0.201 | 0.015–0.897 | 14 | 1 | 1 | 0.0027 |
| reference.climatology | 16 | 0.608 | 0.511–0.710 | 16 | 0 | 0 | 0.0000 |
| reference.gaussian_linear | 16 | 0.909 | 0.871–0.946 | 14 | 2 | 0 | 0.0002 |
| mboostrs.LSS_select_tree | 16 | 0.929 | 0.890–0.976 | 14 | 1 | 1 | 0.0042 |
| ngboost.normal | 16 | 0.964 | 0.938–0.994 | 15 | 0 | 1 | 0.0042 |
| qrf | 16 | 0.973 | 0.933–1.030 | 12 | 0 | 4 | 0.0443 |
| catboost.uncertainty | 16 | 0.978 | 0.923–1.053 | 14 | 0 | 2 | 0.0443 |
| drforest.cart | 16 | 0.984 | 0.947–1.026 | 11 | 2 | 3 | 0.1591 |
| mboostrs.GaussianLSS | 16 | 0.985 | 0.973–0.994 | 7 | 9 | 0 | 0.0122 |
| distreebution.crps_rf | 16 | 0.992 | 0.956–1.042 | 11 | 1 | 4 | 0.1928 |
| distreebution.pmqrf | 16 | 0.993 | 0.957–1.042 | 11 | 1 | 4 | 0.2114 |
| drforest.tuned | 16 | 0.994 | 0.950–1.053 | 10 | 2 | 4 | 0.2312 |
| drforest.mmd | 16 | 0.994 | 0.950–1.053 | 10 | 2 | 4 | 0.2522 |
| mboostrs.LSS_select_noncyclic | 16 | 1.001 | 1.000–1.002 | 0 | 16 | 0 | 0.0250 |
| distreebution.crps_rf_tuned | 16 | 1.002 | 0.959–1.066 | 11 | 0 | 5 | 0.3484 |
| mboostrs.quantiles | 16 | 1.003 | 0.988–1.022 | 6 | 7 | 3 | 0.5966 |
| hessboost.tuned | 16 | 1.006 | 0.973–1.062 | 10 | 4 | 2 | 0.0290 |
| lightgbm.quantiles | 16 | 1.008 | 0.959–1.073 | 11 | 1 | 4 | 0.4037 |
| mboostrs.fit_lss_trees | 16 | 1.028 | 0.992–1.092 | 5 | 9 | 2 | 0.4637 |
| mboostrs.fit | 16 | 1.032 | 0.992–1.100 | 5 | 9 | 2 | 0.4332 |

### mboostrs.fit against each arm

| Other arm | Tasks | CRPS ratio (geometric mean) | 95% interval | Wins | Ties | Losses | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| hessboost.normal | 16 | 0.195 | 0.014–0.867 | 16 | 0 | 0 | 0.0000 |
| reference.climatology | 16 | 0.589 | 0.497–0.690 | 16 | 0 | 0 | 0.0000 |
| reference.gaussian_linear | 16 | 0.880 | 0.824–0.933 | 11 | 5 | 0 | 0.0000 |
| mboostrs.LSS_select_tree | 16 | 0.900 | 0.867–0.933 | 15 | 1 | 0 | 0.0000 |
| ngboost.normal | 16 | 0.933 | 0.896–0.966 | 15 | 0 | 1 | 0.0002 |
| qrf | 16 | 0.943 | 0.920–0.969 | 14 | 0 | 2 | 0.0017 |
| catboost.uncertainty | 16 | 0.947 | 0.920–0.974 | 13 | 2 | 1 | 0.0003 |
| drforest.cart | 16 | 0.953 | 0.913–0.995 | 12 | 1 | 3 | 0.0443 |
| mboostrs.GaussianLSS | 16 | 0.954 | 0.899–0.994 | 6 | 8 | 2 | 0.0290 |
| distreebution.crps_rf | 16 | 0.961 | 0.927–0.996 | 13 | 0 | 3 | 0.0443 |
| distreebution.pmqrf | 16 | 0.962 | 0.930–0.996 | 13 | 0 | 3 | 0.0386 |
| drforest.tuned | 16 | 0.963 | 0.926–1.001 | 11 | 2 | 3 | 0.0654 |
| drforest.mmd | 16 | 0.963 | 0.927–1.001 | 11 | 2 | 3 | 0.0654 |
| mboostrs.LSS_select | 16 | 0.969 | 0.909–1.009 | 2 | 9 | 5 | 0.4332 |
| mboostrs.LSS_select_noncyclic | 16 | 0.970 | 0.910–1.009 | 2 | 9 | 5 | 0.2522 |
| distreebution.crps_rf_tuned | 16 | 0.970 | 0.939–1.004 | 12 | 1 | 3 | 0.0654 |
| mboostrs.quantiles | 16 | 0.972 | 0.913–1.021 | 8 | 3 | 5 | 0.4332 |
| hessboost.tuned | 16 | 0.975 | 0.951–0.993 | 10 | 4 | 2 | 0.0182 |
| lightgbm.quantiles | 16 | 0.976 | 0.953–1.002 | 12 | 0 | 4 | 0.0654 |
| mboostrs.fit_lss_trees | 16 | 0.996 | 0.983–1.005 | 1 | 14 | 1 | 1.0000 |

### Excess CRPS over the true distribution, by task

| Task | mboostrs.LSS_select_noncyclic | mboostrs.fit_lss_trees | mboostrs.fit | mboostrs.LSS_select | mboostrs.quantiles | hessboost.tuned | drforest.mmd | distreebution.crps_rf_tuned | drforest.tuned | mboostrs.GaussianLSS | lightgbm.quantiles | drforest.cart | distreebution.pmqrf | distreebution.crps_rf | catboost.uncertainty | ngboost.normal | qrf | reference.gaussian_linear | mboostrs.LSS_select_tree | hessboost.normal | reference.climatology |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1_hetero_gauss@500 | 0.017 | 0.015 | 0.015 | 0.023 | 0.034 | 0.082 | 0.207 | 0.160 | 0.206 | 0.022 | 0.140 | 0.203 | 0.146 | 0.150 | 0.221 | 0.110 | 0.167 | 0.225 | 0.311 | 0.235 | 2.226 |
| S1_hetero_gauss@5000 | 0.003 | 0.004 | 0.004 | 0.005 | 0.005 | 0.016 | 0.052 | 0.056 | 0.052 | 0.005 | 0.042 | 0.057 | 0.056 | 0.056 | 0.025 | 0.020 | 0.081 | 0.202 | 0.109 | 0.068 | 2.219 |
| S2_student_t3@500 | 0.026 | 0.041 | 0.041 | 0.027 | 0.033 | 0.063 | 0.131 | 0.099 | 0.132 | 0.038 | 0.107 | 0.134 | 0.103 | 0.106 | 0.179 | 0.109 | 0.137 | 0.151 | 0.243 | 0.214 | 1.484 |
| S2_student_t3@5000 | 0.007 | 0.025 | 0.025 | 0.007 | 0.005 | 0.017 | 0.035 | 0.039 | 0.035 | 0.027 | 0.036 | 0.037 | 0.039 | 0.039 | 0.036 | 0.035 | 0.079 | 0.145 | 0.086 | 15.374 | 1.456 |
| S3_gamma@500 | 0.023 | 0.026 | 0.026 | 0.021 | 0.046 | 0.054 | 0.064 | 0.078 | 0.064 | 0.052 | 0.083 | 0.064 | 0.080 | 0.078 | 0.098 | 0.106 | 0.105 | 0.133 | 0.102 | 0.357 | 0.304 |
| S3_gamma@5000 | 0.002 | 0.001 | 0.001 | 0.002 | 0.020 | 0.006 | 0.013 | 0.016 | 0.014 | 0.034 | 0.025 | 0.015 | 0.016 | 0.016 | 0.030 | 0.032 | 0.052 | 0.129 | 0.031 | 0.195 | 0.292 |
| S4_lognormal@500 | 0.023 | 0.030 | 0.030 | 0.023 | 0.066 | 0.057 | 0.087 | 0.086 | 0.090 | 0.105 | 0.089 | 0.086 | 0.102 | 0.098 | 0.194 | 0.194 | 0.135 | 0.338 | 0.155 | 3.506 | 0.727 |
| S4_lognormal@5000 | 0.001 | 0.004 | 0.004 | 0.001 | 0.035 | 0.008 | 0.020 | 0.025 | 0.020 | 0.062 | 0.031 | 0.022 | 0.025 | 0.024 | 0.039 | 0.052 | 0.068 | 0.295 | 0.049 | 578184539.167 | 0.722 |
| S5_bimodal@500 | 0.136 | 0.156 | 0.155 | 0.136 | 0.047 | 0.148 | 0.051 | 0.059 | 0.051 | 0.137 | 0.090 | 0.054 | 0.076 | 0.073 | 0.204 | 0.190 | 0.104 | 0.160 | 0.285 | 0.257 | 0.453 |
| S5_bimodal@5000 | 0.104 | 0.136 | 0.136 | 0.104 | 0.011 | 0.108 | 0.011 | 0.014 | 0.012 | 0.104 | 0.029 | 0.011 | 0.017 | 0.016 | 0.117 | 0.117 | 0.058 | 0.148 | 0.150 | 0.153 | 0.460 |
| S6_skew_flip@500 | 0.029 | 0.038 | 0.038 | 0.030 | 0.021 | 0.049 | 0.032 | 0.043 | 0.034 | 0.040 | 0.059 | 0.036 | 0.051 | 0.049 | 0.070 | 0.067 | 0.083 | 0.045 | 0.089 | 0.112 | 0.156 |
| S6_skew_flip@5000 | 0.023 | 0.035 | 0.035 | 0.024 | 0.004 | 0.024 | 0.010 | 0.010 | 0.010 | 0.036 | 0.019 | 0.020 | 0.014 | 0.013 | 0.042 | 0.045 | 0.050 | 0.037 | 0.043 | 0.238 | 0.149 |
| S7_interaction@500 | 0.593 | 0.451 | 0.320 | 0.595 | 0.604 | 0.555 | 0.588 | 0.561 | 0.588 | 0.595 | 0.303 | 0.622 | 0.570 | 0.606 | 0.368 | 0.649 | 0.493 | 0.670 | 0.599 | 0.593 | 1.004 |
| S7_interaction@5000 | 0.571 | 0.015 | 0.042 | 0.572 | 0.562 | 0.077 | 0.132 | 0.079 | 0.132 | 0.572 | 0.080 | 0.275 | 0.191 | 0.183 | 0.035 | 0.352 | 0.143 | 0.649 | 0.298 | 0.225 | 0.998 |
| S8_linear@500 | 0.013 | 0.007 | 0.009 | 0.017 | 0.027 | 0.091 | 0.129 | 0.088 | 0.129 | 0.016 | 0.107 | 0.128 | 0.092 | 0.092 | 0.156 | 0.083 | 0.111 | 0.014 | 0.200 | 0.161 | 0.742 |
| S8_linear@5000 | 0.001 | 0.000 | 0.000 | 0.001 | 0.002 | 0.015 | 0.033 | 0.033 | 0.033 | 0.001 | 0.033 | 0.032 | 0.032 | 0.033 | 0.018 | 0.019 | 0.058 | 0.000 | 0.071 | 0.047 | 0.741 |

### Log score, parametric arms (mean rank over 15 tasks)

- mboostrs.LSS_select_noncyclic: 2.40 (runs with a zero-density test row: 0)
- mboostrs.LSS_select: 3.53 (runs with a zero-density test row: 0)
- mboostrs.fit_lss_trees: 3.93 (runs with a zero-density test row: 1)
- mboostrs.fit: 4.13 (runs with a zero-density test row: 1)
- hessboost.tuned: 5.20 (runs with a zero-density test row: 0)
- mboostrs.GaussianLSS: 5.67 (runs with a zero-density test row: 1)
- catboost.uncertainty: 7.53 (runs with a zero-density test row: 0)
- ngboost.normal: 7.73 (runs with a zero-density test row: 0)
- mboostrs.LSS_select_tree: 7.80 (runs with a zero-density test row: 0)
- reference.gaussian_linear: 8.47 (runs with a zero-density test row: 0)
- hessboost.normal: 9.60 (runs with a zero-density test row: 0)

## OpenML-CTR23

35 tasks. Mean rank, gap to best and best-on count use the 31 tasks completed by every arm that finished at least 90% of its runs; other arms show a dash.

| Arm | Completed | Mean rank | Skill, median | Skill, mean | Median gap to best | Best on | Calibration MAE | Cov 50 | Cov 90 | Cov 98 | Median s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lightgbm.quantiles | 105/105 | 3.48 | 0.686 | 0.592 | 0.011 | 14 | 0.049 | 0.484 | 0.846 | 0.968 | 12.7 |
| qrf | 105/105 | 5.16 | 0.621 | 0.562 | 0.106 | 5 | 0.076 | 0.586 | 0.885 | 0.951 | 5.9 |
| distreebution.crps_rf_tuned | 105/105 | 5.77 | 0.592 | 0.548 | 0.161 | 1 | 0.067 | 0.620 | 0.931 | 0.980 | 68.4 |
| hessboost.tuned | 105/105 | 6.13 | 0.592 | 0.540 | 0.109 | 2 | 0.038 | 0.488 | 0.867 | 0.951 | 13.5 |
| catboost.uncertainty | 105/105 | 7.26 | 0.659 | 0.521 | 0.181 | 1 | 0.045 | 0.488 | 0.879 | 0.958 | 7.8 |
| drforest.tuned | 102/105 | 9.10 | 0.538 | 0.524 | 0.294 | 1 | 0.068 | 0.670 | 0.960 | 0.992 | 119.7 |
| ngboost.normal | 105/105 | 9.16 | 0.649 | 0.494 | 0.310 | 0 | 0.045 | 0.498 | 0.892 | 0.964 | 49.9 |
| drforest.cart | 105/105 | 9.50 | 0.530 | 0.509 | 0.291 | 1 | 0.077 | 0.675 | 0.961 | 0.992 | 17.9 |
| drforest.mmd | 102/105 | 9.66 | 0.536 | 0.522 | 0.300 | 0 | 0.068 | 0.669 | 0.960 | 0.992 | 38.2 |
| distreebution.crps_rf | 105/105 | 10.00 | 0.480 | 0.485 | 0.295 | 2 | 0.061 | 0.612 | 0.937 | 0.986 | 16.8 |
| mboostrs.quantiles | 102/105 | 10.19 | 0.477 | 0.467 | 0.375 | 1 | 0.024 | 0.525 | 0.898 | 0.978 | 89.7 |
| mboostrs.fit | 105/105 | 11.39 | 0.459 | 0.424 | 0.344 | 2 | 0.067 | 0.529 | 0.840 | 0.911 | 20.6 |
| mboostrs.LSS_select_tree | 105/105 | 11.45 | 0.574 | 0.463 | 0.402 | 0 | 0.043 | 0.567 | 0.914 | 0.978 | 22.0 |
| mboostrs.fit_lss_trees | 105/105 | 11.74 | 0.461 | 0.428 | 0.344 | 1 | 0.065 | 0.524 | 0.839 | 0.910 | 30.7 |
| mboostrs.LSS_select_noncyclic | 95/105 | 11.77 | 0.484 | 0.437 | 0.497 | 0 | 0.028 | 0.463 | 0.907 | 0.978 | 111.6 |
| mboostrs.LSS_select | 105/105 | 12.85 | 0.477 | 0.404 | 0.501 | 0 | 0.042 | 0.539 | 0.917 | 0.980 | 16.1 |
| hessboost.normal | 105/105 | 14.90 | 0.114 | < −10 | 1.261 | 0 | 0.066 | 0.429 | 0.912 | 0.972 | 0.3 |
| mboostrs.GaussianLSS | 105/105 | 15.34 | 0.380 | < −10 | 0.739 | 0 | 0.053 | 0.582 | 0.930 | 0.980 | 2.5 |
| reference.gaussian_linear | 105/105 | 16.06 | 0.348 | 0.314 | 0.730 | 1 | 0.052 | 0.617 | 0.919 | 0.971 | 0.0 |
| reference.climatology | 105/105 | 19.06 | 0.000 | 0.000 | 2.488 | 1 | 0.042 | 0.555 | 0.913 | 0.985 | 0.0 |
| distreebution.pmqrf | 10/105 | – | 0.413 | 0.413 | 876.654 | 0 | 0.028 | 0.578 | 0.942 | 0.991 | 662.0 |

### mboostrs.LSS_select against each arm

| Other arm | Tasks | CRPS ratio (geometric mean) | 95% interval | Wins | Ties | Losses | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| hessboost.normal | 35 | 0.080 | 0.012–0.378 | 21 | 0 | 14 | 0.0107 |
| distreebution.pmqrf | 2 | 0.482 | 0.270–0.861 | 2 | 0 | 0 | – |
| reference.climatology | 35 | 0.522 | 0.438–0.619 | 32 | 0 | 3 | 0.0000 |
| mboostrs.GaussianLSS | 35 | 0.736 | 0.490–0.934 | 15 | 20 | 0 | 0.0000 |
| reference.gaussian_linear | 35 | 1.035 | 0.798–1.579 | 28 | 0 | 7 | 0.0107 |
| mboostrs.LSS_select_noncyclic | 31 | 1.044 | 1.002–1.101 | 3 | 18 | 10 | 0.0054 |
| mboostrs.fit | 35 | 1.151 | 0.992–1.363 | 18 | 2 | 15 | 0.4128 |
| mboostrs.fit_lss_trees | 35 | 1.152 | 1.005–1.358 | 18 | 2 | 15 | 0.3945 |
| mboostrs.LSS_select_tree | 35 | 1.152 | 1.040–1.293 | 10 | 5 | 20 | 0.0161 |
| distreebution.crps_rf | 35 | 1.167 | 0.988–1.380 | 11 | 3 | 21 | 0.0300 |
| mboostrs.quantiles | 34 | 1.204 | 1.045–1.441 | 6 | 6 | 22 | 0.0026 |
| drforest.mmd | 34 | 1.221 | 1.051–1.434 | 8 | 1 | 25 | 0.0012 |
| drforest.cart | 35 | 1.222 | 1.050–1.435 | 9 | 1 | 25 | 0.0015 |
| drforest.tuned | 34 | 1.231 | 1.056–1.453 | 8 | 1 | 25 | 0.0010 |
| ngboost.normal | 35 | 1.359 | 1.172–1.609 | 8 | 2 | 25 | 0.0005 |
| distreebution.crps_rf_tuned | 35 | 1.396 | 1.176–1.677 | 5 | 2 | 28 | 0.0000 |
| catboost.uncertainty | 35 | 1.576 | 1.293–1.972 | 7 | 1 | 27 | 0.0000 |
| qrf | 35 | 1.582 | 1.290–1.992 | 6 | 0 | 29 | 0.0000 |
| hessboost.tuned | 35 | 1.590 | 1.282–2.042 | 4 | 2 | 29 | 0.0000 |
| lightgbm.quantiles | 35 | 1.796 | 1.438–2.319 | 2 | 4 | 29 | 0.0000 |

### mboostrs.fit against each arm

| Other arm | Tasks | CRPS ratio (geometric mean) | 95% interval | Wins | Ties | Losses | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| hessboost.normal | 35 | 0.070 | 0.011–0.319 | 25 | 0 | 10 | 0.0003 |
| distreebution.pmqrf | 2 | 0.343 | 0.163–0.724 | 2 | 0 | 0 | – |
| reference.climatology | 35 | 0.454 | 0.351–0.578 | 33 | 0 | 2 | 0.0000 |
| mboostrs.GaussianLSS | 35 | 0.640 | 0.404–0.865 | 22 | 6 | 7 | 0.0004 |
| mboostrs.LSS_select | 35 | 0.869 | 0.737–1.006 | 15 | 2 | 18 | 0.4128 |
| reference.gaussian_linear | 35 | 0.900 | 0.669–1.358 | 28 | 1 | 6 | 0.0004 |
| mboostrs.LSS_select_noncyclic | 31 | 0.910 | 0.757–1.069 | 13 | 0 | 18 | 0.7063 |
| mboostrs.fit_lss_trees | 35 | 1.002 | 0.966–1.045 | 3 | 31 | 1 | 0.6250 |
| mboostrs.LSS_select_tree | 35 | 1.002 | 0.851–1.182 | 15 | 3 | 17 | 0.9935 |
| distreebution.crps_rf | 35 | 1.014 | 0.841–1.210 | 15 | 1 | 19 | 0.6799 |
| mboostrs.quantiles | 34 | 1.033 | 0.871–1.238 | 11 | 0 | 23 | 0.4671 |
| drforest.mmd | 34 | 1.058 | 0.859–1.324 | 13 | 2 | 19 | 0.4265 |
| drforest.cart | 35 | 1.062 | 0.860–1.316 | 14 | 1 | 20 | 0.3592 |
| drforest.tuned | 34 | 1.066 | 0.860–1.316 | 13 | 2 | 19 | 0.3516 |
| ngboost.normal | 35 | 1.181 | 1.020–1.411 | 12 | 2 | 21 | 0.0934 |
| distreebution.crps_rf_tuned | 35 | 1.213 | 1.001–1.488 | 7 | 5 | 23 | 0.0082 |
| catboost.uncertainty | 35 | 1.370 | 1.124–1.738 | 9 | 1 | 25 | 0.0074 |
| qrf | 35 | 1.375 | 1.101–1.766 | 9 | 1 | 25 | 0.0011 |
| hessboost.tuned | 35 | 1.382 | 1.078–1.825 | 10 | 0 | 25 | 0.0070 |
| lightgbm.quantiles | 35 | 1.561 | 1.242–2.050 | 5 | 1 | 29 | 0.0000 |

### Log score, parametric arms (mean rank over 29 tasks)

- hessboost.tuned: 2.66 (runs with a zero-density test row: 0)
- catboost.uncertainty: 3.69 (runs with a zero-density test row: 0)
- mboostrs.LSS_select_tree: 4.79 (runs with a zero-density test row: 0)
- mboostrs.LSS_select_noncyclic: 4.86 (runs with a zero-density test row: 1)
- ngboost.normal: 4.86 (runs with a zero-density test row: 0)
- mboostrs.LSS_select: 5.95 (runs with a zero-density test row: 0)
- hessboost.normal: 7.17 (runs with a zero-density test row: 0)
- mboostrs.GaussianLSS: 7.29 (runs with a zero-density test row: 4)
- mboostrs.fit: 7.72 (runs with a zero-density test row: 1)
- mboostrs.fit_lss_trees: 7.97 (runs with a zero-density test row: 1)
- reference.gaussian_linear: 9.03 (runs with a zero-density test row: 0)
