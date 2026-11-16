# Superseded stopping-ablation findings

The unchanged [v1 artifact](results.json) was recorded 2026-10-12T16:43:05+00:00 from `fb3233282d38411339b7646fb4a13d35e24e5a7d`. Its dependency/build identities remain in that file. This audit supersedes the numerical conclusions in PR #521 and the October 12 issue progress report; it does not modify the experiment record.

The historical refitted baseline and native growing routes have different preprocessing. Their loss differences combine preprocessing, stopping and configuration changes. They cannot identify a budget-growth effect. Random inner folds on `bike_future` also target random-row prediction rather than forecasting. Use [version 2](v2/README.md) for matched-route and rolling-origin comparisons.

## Historical comparisons

The blanket claims that growth never materially worsens loss and that patience never improves on growth are withdrawn. The following tables use paired per-seed relative changes, with the named reference loss as denominator. A median improvement can coexist with a harmful seed.

| panel | method | tier | rule / reference | finite seeds / all | median change | range | worst seed | reference loss → rule loss at worst seed |
|---|---|---|---|---|---|---|---|---|
| cancer | mboost_hybrid | small | boundary_growth / fixed_argmin | 3/3 | +0.2% | +0.0% to +9.1% | 2 | 0.168826 → 0.184241 |
| cancer | mboost_hybrid | small | patience_60 / fixed_argmin | 3/3 | +0.2% | +0.0% to +3.4% | 2 | 0.168826 → 0.174633 |
| cancer | mboost_hybrid | small | repeated_cv_growth / fixed_argmin | 3/3 | -4.0% | -4.8% to -0.3% | 0 | 0.141338 → 0.14085 |
| cancer | mboost_hybrid | tuned | boundary_growth / fixed_argmin | 3/3 | -1.1% | -47.4% to +0.0% | 1 | 0.116359 → 0.116359 |
| cancer | mboost_hybrid | tuned | patience_60 / fixed_argmin | 3/3 | -1.1% | -50.1% to +0.0% | 1 | 0.116359 → 0.116359 |
| cancer | mboost_hybrid | tuned | repeated_cv_growth / fixed_argmin | 3/3 | +5.3% | -53.7% to +44.4% | 0 | 0.0975601 → 0.14085 |
| cancer | mboost_additive | small | boundary_growth / fixed_argmin | 3/3 | +0.4% | -3.3% to +1.6% | 0 | 0.0998846 → 0.101435 |
| cancer | mboost_additive | small | patience_60 / fixed_argmin | 3/3 | +0.4% | -3.3% to +1.6% | 0 | 0.0998846 → 0.101435 |
| cancer | mboost_additive | small | repeated_cv_growth / fixed_argmin | 3/3 | +0.8% | +0.6% to +4.2% | 1 | 0.0647909 → 0.0674904 |
| cancer | mboost_additive | tuned | boundary_growth / fixed_argmin | 3/3 | -3.3% | -6.6% to +0.3% | 0 | 0.096278 → 0.0965191 |
| cancer | mboost_additive | tuned | patience_60 / fixed_argmin | 3/3 | -3.3% | -4.6% to +0.3% | 0 | 0.096278 → 0.0965191 |
| cancer | mboost_additive | tuned | repeated_cv_growth / fixed_argmin | 3/3 | +0.6% | -1.6% to +4.6% | 0 | 0.096278 → 0.100672 |
| bike_iid | mboost_hybrid | small | boundary_growth / fixed_argmin | 3/3 | -0.9% | -1.1% to +0.0% | 0 | 0.109603 → 0.109603 |
| bike_iid | mboost_hybrid | small | patience_60 / fixed_argmin | 3/3 | +0.0% | +0.0% to +0.0% | 1 | 0.098649 → 0.0986517 |
| bike_iid | mboost_hybrid | small | repeated_cv_growth / fixed_argmin | 3/3 | -1.7% | -3.0% to -1.3% | 2 | 0.0945415 → 0.0933546 |
| bike_iid | mboost_hybrid | tuned | boundary_growth / fixed_argmin | 3/3 | -0.9% | -1.1% to +0.0% | 0 | 0.119161 → 0.119161 |
| bike_iid | mboost_hybrid | tuned | patience_60 / fixed_argmin | 3/3 | +0.0% | +0.0% to +0.0% | 1 | 0.098649 → 0.0986517 |
| bike_iid | mboost_hybrid | tuned | repeated_cv_growth / fixed_argmin | 3/3 | -1.7% | -10.9% to -1.3% | 2 | 0.0945415 → 0.0933546 |
| bike_iid | mboost_additive | small | boundary_growth / fixed_argmin | 3/3 | -2.9% | -3.2% to -2.8% | 2 | 0.366459 → 0.356214 |
| bike_iid | mboost_additive | small | patience_60 / fixed_argmin | 3/3 | -2.9% | -3.2% to -2.8% | 2 | 0.366459 → 0.356214 |
| bike_iid | mboost_additive | small | repeated_cv_growth / fixed_argmin | 3/3 | -2.9% | -3.2% to -2.8% | 2 | 0.366459 → 0.356214 |
| bike_iid | mboost_additive | tuned | boundary_growth / fixed_argmin | 3/3 | +0.0% | +0.0% to +0.0% | 0 | 0.355147 → 0.355147 |
| bike_iid | mboost_additive | tuned | patience_60 / fixed_argmin | 3/3 | +0.0% | +0.0% to +0.0% | 0 | 0.355147 → 0.355147 |
| bike_iid | mboost_additive | tuned | repeated_cv_growth / fixed_argmin | 3/3 | -0.1% | -0.3% to -0.1% | 1 | 0.333845 → 0.333453 |
| bike_future | mboost_hybrid | small | boundary_growth / fixed_argmin | 3/3 | +0.0% | +0.0% to +0.0% | 2 | 0.264642 → 0.264764 |
| bike_future | mboost_hybrid | small | patience_60 / fixed_argmin | 3/3 | +0.0% | -0.0% to +0.0% | 0 | 0.198785 → 0.198785 |
| bike_future | mboost_hybrid | small | repeated_cv_growth / fixed_argmin | 3/3 | +0.1% | -2.1% to +5.3% | 0 | 0.198785 → 0.209315 |
| bike_future | mboost_hybrid | tuned | boundary_growth / fixed_argmin | 3/3 | -1.1% | -1.8% to -0.5% | 0 | 0.191839 → 0.190952 |
| bike_future | mboost_hybrid | tuned | patience_60 / fixed_argmin | 3/3 | -0.5% | -1.1% to -0.0% | 2 | 0.264642 → 0.264562 |
| bike_future | mboost_hybrid | tuned | repeated_cv_growth / fixed_argmin | 3/3 | +1.1% | -1.9% to +3.0% | 0 | 0.191839 → 0.197685 |
| bike_future | mboost_additive | small | boundary_growth / fixed_argmin | 3/3 | +0.0% | -3.4% to +0.2% | 1 | 0.443962 → 0.444661 |
| bike_future | mboost_additive | small | patience_60 / fixed_argmin | 3/3 | +0.0% | -3.4% to +0.2% | 1 | 0.443962 → 0.444661 |
| bike_future | mboost_additive | small | repeated_cv_growth / fixed_argmin | 3/3 | -4.1% | -5.5% to -3.4% | 2 | 0.520537 → 0.503081 |
| bike_future | mboost_additive | tuned | boundary_growth / fixed_argmin | 3/3 | +0.0% | -0.1% to +0.5% | 1 | 0.450221 → 0.452524 |
| bike_future | mboost_additive | tuned | patience_60 / fixed_argmin | 3/3 | +0.0% | -0.1% to +0.5% | 1 | 0.450221 → 0.452524 |
| bike_future | mboost_additive | tuned | repeated_cv_growth / fixed_argmin | 3/3 | -2.5% | -10.3% to -0.1% | 2 | 0.484696 → 0.484321 |
| concrete | mboost_hybrid | small | boundary_growth / fixed_argmin | 3/3 | -5.0% | -5.7% to +0.2% | 1 | 27.1108 → 27.1528 |
| concrete | mboost_hybrid | small | patience_60 / fixed_argmin | 3/3 | -3.2% | -5.0% to +0.1% | 1 | 27.1108 → 27.1248 |
| concrete | mboost_hybrid | small | repeated_cv_growth / fixed_argmin | 3/3 | -5.7% | -7.4% to +0.5% | 1 | 27.1108 → 27.2415 |
| concrete | mboost_hybrid | tuned | boundary_growth / fixed_argmin | 3/3 | -5.8% | -7.8% to +0.0% | 1 | 28.33 → 28.33 |
| concrete | mboost_hybrid | tuned | patience_60 / fixed_argmin | 3/3 | -5.0% | -7.8% to +0.0% | 1 | 28.33 → 28.33 |
| concrete | mboost_hybrid | tuned | repeated_cv_growth / fixed_argmin | 3/3 | -7.4% | -7.8% to -2.1% | 1 | 28.33 → 27.7229 |
| concrete | mboost_additive | small | boundary_growth / fixed_argmin | 3/3 | -0.8% | -7.1% to +0.0% | 1 | 49.85 → 49.85 |
| concrete | mboost_additive | small | patience_60 / fixed_argmin | 3/3 | -0.8% | -7.1% to +0.0% | 1 | 49.85 → 49.85 |
| concrete | mboost_additive | small | repeated_cv_growth / fixed_argmin | 3/3 | -7.1% | -16.7% to -0.8% | 2 | 41.6645 → 41.3475 |
| concrete | mboost_additive | tuned | boundary_growth / fixed_argmin | 3/3 | -2.4% | -3.3% to +0.0% | 1 | 38.9102 → 38.9102 |
| concrete | mboost_additive | tuned | patience_60 / fixed_argmin | 3/3 | -2.4% | -3.3% to +0.0% | 1 | 38.9102 → 38.9102 |
| concrete | mboost_additive | tuned | repeated_cv_growth / fixed_argmin | 3/3 | -3.3% | -8.3% to -1.6% | 2 | 38.8382 → 38.2296 |
| joint_boundary | mboost_hybrid | small | boundary_growth / fixed_argmin | 5/5 | -9.6% | -17.0% to -0.9% | 4 | 0.456952 → 0.452904 |
| joint_boundary | mboost_hybrid | small | patience_60 / fixed_argmin | 5/5 | -1.7% | -6.5% to +0.3% | 4 | 0.456952 → 0.458275 |
| joint_boundary | mboost_hybrid | small | repeated_cv_growth / fixed_argmin | 5/5 | -9.5% | -17.1% to -0.9% | 4 | 0.456952 → 0.452895 |
| joint_boundary | mboost_hybrid | tuned | boundary_growth / fixed_argmin | 5/5 | -1.6% | -12.4% to -0.2% | 2 | 0.737382 → 0.736198 |
| joint_boundary | mboost_hybrid | tuned | patience_60 / fixed_argmin | 5/5 | -0.7% | -9.4% to -0.0% | 2 | 0.737382 → 0.737088 |
| joint_boundary | mboost_hybrid | tuned | repeated_cv_growth / fixed_argmin | 5/5 | -13.7% | -39.9% to -0.9% | 4 | 0.456952 → 0.452895 |
| joint_boundary | mboost_additive | small | boundary_growth / fixed_argmin | 5/5 | +0.0% | +0.0% to +0.0% | 0 | 9.18768 → 9.18768 |
| joint_boundary | mboost_additive | small | patience_60 / fixed_argmin | 5/5 | +0.0% | +0.0% to +0.0% | 0 | 9.18768 → 9.18768 |
| joint_boundary | mboost_additive | small | repeated_cv_growth / fixed_argmin | 5/5 | -0.0% | -0.0% to +0.0% | 1 | 9.13517 → 9.13816 |
| joint_boundary | mboost_additive | tuned | boundary_growth / fixed_argmin | 5/5 | +0.0% | +0.0% to +0.0% | 0 | 9.18768 → 9.18768 |
| joint_boundary | mboost_additive | tuned | patience_60 / fixed_argmin | 5/5 | +0.0% | +0.0% to +0.0% | 0 | 9.18768 → 9.18768 |
| joint_boundary | mboost_additive | tuned | repeated_cv_growth / fixed_argmin | 5/5 | -0.0% | -0.0% to +0.0% | 1 | 9.13517 → 9.13816 |
| smooth_sparse | mboost_hybrid | small | boundary_growth / fixed_argmin | 5/5 | +0.0% | -0.5% to +0.1% | 1 | 0.259227 → 0.259411 |
| smooth_sparse | mboost_hybrid | small | patience_60 / fixed_argmin | 5/5 | +0.0% | -0.5% to +0.1% | 1 | 0.259227 → 0.259411 |
| smooth_sparse | mboost_hybrid | small | repeated_cv_growth / fixed_argmin | 5/5 | +0.1% | -0.2% to +0.3% | 2 | 0.275331 → 0.276114 |
| smooth_sparse | mboost_hybrid | tuned | boundary_growth / fixed_argmin | 5/5 | +0.0% | -0.4% to +0.0% | 3 | 0.28279 → 0.28292 |
| smooth_sparse | mboost_hybrid | tuned | patience_60 / fixed_argmin | 5/5 | +0.0% | -0.4% to +0.0% | 3 | 0.28279 → 0.28292 |
| smooth_sparse | mboost_hybrid | tuned | repeated_cv_growth / fixed_argmin | 5/5 | +0.1% | -0.5% to +0.2% | 2 | 0.275858 → 0.27643 |
| smooth_sparse | mboost_additive | small | boundary_growth / fixed_argmin | 5/5 | +0.0% | +0.0% to +0.0% | 0 | 0.258837 → 0.258837 |
| smooth_sparse | mboost_additive | small | patience_60 / fixed_argmin | 5/5 | +0.0% | +0.0% to +0.0% | 0 | 0.258837 → 0.258837 |
| smooth_sparse | mboost_additive | small | repeated_cv_growth / fixed_argmin | 5/5 | -0.3% | -0.9% to -0.0% | 4 | 0.242359 → 0.242306 |
| smooth_sparse | mboost_additive | tuned | boundary_growth / fixed_argmin | 5/5 | +0.0% | +0.0% to +0.0% | 0 | 0.258837 → 0.258837 |
| smooth_sparse | mboost_additive | tuned | patience_60 / fixed_argmin | 5/5 | +0.0% | +0.0% to +0.0% | 0 | 0.258837 → 0.258837 |
| smooth_sparse | mboost_additive | tuned | repeated_cv_growth / fixed_argmin | 5/5 | -0.1% | -0.9% to +1.0% | 1 | 0.243889 → 0.246416 |
| housing_iid | mboost_hybrid | small | boundary_growth / fixed_argmin | 3/3 | +0.0% | -1.7% to +0.0% | 1 | 0.0615226 → 0.0615226 |
| housing_iid | mboost_hybrid | small | patience_60 / fixed_argmin | 3/3 | +0.0% | +0.0% to +0.0% | 0 | 0.058684 → 0.058684 |
| housing_iid | mboost_hybrid | small | repeated_cv_growth / fixed_argmin | 3/3 | -0.9% | -1.4% to +0.9% | 0 | 0.058684 → 0.0591981 |
| housing_iid | mboost_hybrid | tuned | boundary_growth / fixed_argmin | 3/3 | +0.0% | -1.0% to +0.0% | 0 | 0.0555417 → 0.0555505 |
| housing_iid | mboost_hybrid | tuned | patience_60 / fixed_argmin | 3/3 | +0.0% | -0.3% to +0.0% | 0 | 0.0555417 → 0.0555505 |
| housing_iid | mboost_hybrid | tuned | repeated_cv_growth / fixed_argmin | 3/3 | -1.0% | -3.0% to +0.8% | 0 | 0.0555417 → 0.0559823 |
| housing_iid | mboost_additive | small | boundary_growth / fixed_argmin | 3/3 | -2.9% | -3.5% to -2.9% | 1 | 0.0881342 → 0.0855814 |
| housing_iid | mboost_additive | small | patience_60 / fixed_argmin | 3/3 | -2.9% | -3.5% to -2.9% | 1 | 0.0881342 → 0.0855814 |
| housing_iid | mboost_additive | small | repeated_cv_growth / fixed_argmin | 3/3 | +2.8% | -2.9% to +122.4% | 2 | 0.0850357 → 0.189078 |
| housing_iid | mboost_additive | tuned | boundary_growth / fixed_argmin | 3/3 | -0.7% | -2.3% to +0.0% | 0 | 0.0822268 → 0.0822268 |
| housing_iid | mboost_additive | tuned | patience_60 / fixed_argmin | 3/3 | -0.7% | -2.3% to +0.0% | 0 | 0.0822268 → 0.0822268 |
| housing_iid | mboost_additive | tuned | repeated_cv_growth / fixed_argmin | 3/3 | +8.6% | -1.9% to +101.7% | 2 | 0.0793402 → 0.160031 |

## Patience comparisons

| panel | method | tier | rule / reference | finite seeds / all | median change | range | worst seed | reference loss → rule loss at worst seed |
|---|---|---|---|---|---|---|---|---|
| cancer | mboost_hybrid | small | patience_60 / boundary_growth | 3/3 | +0.0% | -5.2% to +0.0% | 0 | 0.141584 → 0.141584 |
| cancer | mboost_hybrid | tuned | patience_60 / boundary_growth | 3/3 | +0.0% | -5.2% to +0.0% | 0 | 0.0965354 → 0.0965354 |
| cancer | mboost_additive | small | patience_60 / boundary_growth | 3/3 | +0.0% | +0.0% to +0.0% | 0 | 0.101435 → 0.101435 |
| cancer | mboost_additive | tuned | patience_60 / boundary_growth | 3/3 | +0.0% | +0.0% to +2.2% | 1 | 0.0640752 → 0.0654674 |
| bike_iid | mboost_hybrid | small | patience_60 / boundary_growth | 3/3 | +0.9% | +0.0% to +1.2% | 2 | 0.0934566 → 0.0945415 |
| bike_iid | mboost_hybrid | tuned | patience_60 / boundary_growth | 3/3 | +0.9% | +0.0% to +1.2% | 2 | 0.0934566 → 0.0945415 |
| bike_iid | mboost_additive | small | patience_60 / boundary_growth | 3/3 | +0.0% | +0.0% to +0.0% | 0 | 0.369034 → 0.369034 |
| bike_iid | mboost_additive | tuned | patience_60 / boundary_growth | 3/3 | +0.0% | +0.0% to +0.0% | 0 | 0.355147 → 0.355147 |
| bike_future | mboost_hybrid | small | patience_60 / boundary_growth | 3/3 | +0.0% | -0.1% to +0.0% | 0 | 0.198785 → 0.198785 |
| bike_future | mboost_hybrid | tuned | patience_60 / boundary_growth | 3/3 | +0.0% | +0.0% to +1.8% | 2 | 0.259934 → 0.264562 |
| bike_future | mboost_additive | small | patience_60 / boundary_growth | 3/3 | +0.0% | +0.0% to +0.0% | 0 | 0.445544 → 0.445544 |
| bike_future | mboost_additive | tuned | patience_60 / boundary_growth | 3/3 | +0.0% | +0.0% to +0.0% | 0 | 0.422323 → 0.422323 |
| concrete | mboost_hybrid | small | patience_60 / boundary_growth | 3/3 | +0.0% | -0.1% to +2.7% | 0 | 19.358 → 19.8803 |
| concrete | mboost_hybrid | tuned | patience_60 / boundary_growth | 3/3 | +0.0% | +0.0% to +0.8% | 2 | 22.8787 → 23.0669 |
| concrete | mboost_additive | small | patience_60 / boundary_growth | 3/3 | +0.0% | +0.0% to +0.0% | 0 | 36.2338 → 36.2338 |
| concrete | mboost_additive | tuned | patience_60 / boundary_growth | 3/3 | +0.0% | +0.0% to +0.0% | 0 | 30.1931 → 30.1931 |
| joint_boundary | mboost_hybrid | small | patience_60 / boundary_growth | 5/5 | +3.4% | +1.2% to +20.5% | 2 | 0.615354 → 0.741462 |
| joint_boundary | mboost_hybrid | tuned | patience_60 / boundary_growth | 5/5 | +0.9% | +0.1% to +12.3% | 1 | 0.627351 → 0.704331 |
| joint_boundary | mboost_additive | small | patience_60 / boundary_growth | 5/5 | +0.0% | +0.0% to +0.0% | 0 | 9.18768 → 9.18768 |
| joint_boundary | mboost_additive | tuned | patience_60 / boundary_growth | 5/5 | +0.0% | +0.0% to +0.0% | 0 | 9.18768 → 9.18768 |
| smooth_sparse | mboost_hybrid | small | patience_60 / boundary_growth | 5/5 | +0.0% | +0.0% to +0.0% | 0 | 0.276494 → 0.276494 |
| smooth_sparse | mboost_hybrid | tuned | patience_60 / boundary_growth | 5/5 | +0.0% | +0.0% to +0.0% | 0 | 0.277935 → 0.277935 |
| smooth_sparse | mboost_additive | small | patience_60 / boundary_growth | 5/5 | +0.0% | +0.0% to +0.0% | 0 | 0.258837 → 0.258837 |
| smooth_sparse | mboost_additive | tuned | patience_60 / boundary_growth | 5/5 | +0.0% | +0.0% to +0.0% | 0 | 0.258837 → 0.258837 |
| housing_iid | mboost_hybrid | small | patience_60 / boundary_growth | 3/3 | +0.0% | +0.0% to +1.7% | 0 | 0.0576806 → 0.058684 |
| housing_iid | mboost_hybrid | tuned | patience_60 / boundary_growth | 3/3 | +0.0% | +0.0% to +0.7% | 1 | 0.0613065 → 0.0617409 |
| housing_iid | mboost_additive | small | patience_60 / boundary_growth | 3/3 | +0.0% | +0.0% to +0.0% | 0 | 0.0837616 → 0.0837616 |
| housing_iid | mboost_additive | tuned | patience_60 / boundary_growth | 3/3 | +0.0% | -0.0% to +0.0% | 0 | 0.0822268 → 0.0822268 |

## Configuration selection

The claim that every mboost configuration choice was benign is withdrawn. Tuned-minus-small changes include harmful seeds inside this experiment's own mboost grids.

| panel | method | tier | rule / reference | finite seeds / all | median change | range | worst seed | reference loss → rule loss at worst seed |
|---|---|---|---|---|---|---|---|---|
| cancer | mboost_hybrid | tuned vs small | fixed_argmin / small | 3/3 | -9.6% | -31.0% to +107.4% | 2 | 0.168826 → 0.350084 |
| cancer | mboost_hybrid | tuned vs small | boundary_growth / small | 3/3 | -9.6% | -31.8% to +0.0% | 2 | 0.184241 → 0.184241 |
| cancer | mboost_hybrid | tuned vs small | patience_60 / small | 3/3 | -9.6% | -31.8% to +0.0% | 2 | 0.174633 → 0.174633 |
| cancer | mboost_hybrid | tuned vs small | repeated_cv_growth / small | 3/3 | +0.0% | +0.0% to +0.0% | 0 | 0.14085 → 0.14085 |
| cancer | mboost_additive | tuned vs small | fixed_argmin / small | 3/3 | +0.0% | -3.6% to +5.9% | 1 | 0.0647909 → 0.0686097 |
| cancer | mboost_additive | tuned vs small | boundary_growth / small | 3/3 | -1.5% | -4.8% to +0.0% | 2 | 0.120564 → 0.120564 |
| cancer | mboost_additive | tuned vs small | patience_60 / small | 3/3 | +0.0% | -4.8% to +0.7% | 1 | 0.0650443 → 0.0654674 |
| cancer | mboost_additive | tuned vs small | repeated_cv_growth / small | 3/3 | +0.0% | +0.0% to +0.0% | 0 | 0.100672 → 0.100672 |
| bike_iid | mboost_hybrid | tuned vs small | fixed_argmin / small | 3/3 | +0.0% | +0.0% to +8.7% | 0 | 0.109603 → 0.119161 |
| bike_iid | mboost_hybrid | tuned vs small | boundary_growth / small | 3/3 | +0.0% | +0.0% to +8.7% | 0 | 0.109603 → 0.119161 |
| bike_iid | mboost_hybrid | tuned vs small | patience_60 / small | 3/3 | +0.0% | +0.0% to +8.7% | 0 | 0.109603 → 0.119161 |
| bike_iid | mboost_hybrid | tuned vs small | repeated_cv_growth / small | 3/3 | +0.0% | -0.2% to +0.0% | 1 | 0.0969888 → 0.0969888 |
| bike_iid | mboost_additive | tuned vs small | fixed_argmin / small | 3/3 | -6.4% | -6.6% to -5.8% | 2 | 0.366459 → 0.345058 |
| bike_iid | mboost_additive | tuned vs small | boundary_growth / small | 3/3 | -3.3% | -3.8% to -3.1% | 2 | 0.356214 → 0.345058 |
| bike_iid | mboost_additive | tuned vs small | patience_60 / small | 3/3 | -3.3% | -3.8% to -3.1% | 2 | 0.356214 → 0.345058 |
| bike_iid | mboost_additive | tuned vs small | repeated_cv_growth / small | 3/3 | -3.4% | -3.9% to -3.4% | 1 | 0.345193 → 0.333453 |
| bike_future | mboost_hybrid | tuned vs small | fixed_argmin / small | 3/3 | +0.0% | -3.5% to +4.9% | 1 | 0.187355 → 0.196483 |
| bike_future | mboost_hybrid | tuned vs small | boundary_growth / small | 3/3 | -1.8% | -3.9% to +3.7% | 1 | 0.187355 → 0.194308 |
| bike_future | mboost_hybrid | tuned vs small | patience_60 / small | 3/3 | +0.0% | -3.9% to +3.7% | 1 | 0.187355 → 0.194308 |
| bike_future | mboost_hybrid | tuned vs small | repeated_cv_growth / small | 3/3 | +1.0% | -5.6% to +5.1% | 1 | 0.183429 → 0.192792 |
| bike_future | mboost_additive | tuned vs small | fixed_argmin / small | 3/3 | -5.2% | -6.9% to +1.4% | 1 | 0.443962 → 0.450221 |
| bike_future | mboost_additive | tuned vs small | boundary_growth / small | 3/3 | -3.7% | -5.2% to +1.8% | 1 | 0.444661 → 0.452524 |
| bike_future | mboost_additive | tuned vs small | patience_60 / small | 3/3 | -3.7% | -5.2% to +1.8% | 1 | 0.444661 → 0.452524 |
| bike_future | mboost_additive | tuned vs small | repeated_cv_growth / small | 3/3 | -3.7% | -5.2% to -2.1% | 0 | 0.420837 → 0.411862 |
| concrete | mboost_hybrid | tuned vs small | fixed_argmin / small | 3/3 | +0.0% | +0.0% to +4.5% | 1 | 27.1108 → 28.33 |
| concrete | mboost_hybrid | tuned vs small | boundary_growth / small | 3/3 | -0.8% | -2.2% to +4.3% | 1 | 27.1528 → 28.33 |
| concrete | mboost_hybrid | tuned vs small | patience_60 / small | 3/3 | +0.0% | -4.8% to +4.4% | 1 | 27.1248 → 28.33 |
| concrete | mboost_hybrid | tuned vs small | repeated_cv_growth / small | 3/3 | +0.0% | -2.2% to +1.8% | 1 | 27.2415 → 27.7229 |
| concrete | mboost_additive | tuned vs small | fixed_argmin / small | 3/3 | -20.0% | -21.9% to -6.8% | 2 | 41.6645 → 38.8382 |
| concrete | mboost_additive | tuned vs small | boundary_growth / small | 3/3 | -16.7% | -21.9% to -8.3% | 2 | 41.3475 → 37.9184 |
| concrete | mboost_additive | tuned vs small | patience_60 / small | 3/3 | -16.7% | -21.9% to -8.3% | 2 | 41.3475 → 37.9184 |
| concrete | mboost_additive | tuned vs small | repeated_cv_growth / small | 3/3 | -14.1% | -16.7% to -7.5% | 2 | 41.3475 → 38.2296 |
| joint_boundary | mboost_hybrid | tuned vs small | fixed_argmin / small | 5/5 | +0.0% | -0.6% to +58.1% | 3 | 0.604245 → 0.955144 |
| joint_boundary | mboost_hybrid | tuned vs small | boundary_growth / small | 5/5 | +1.4% | -0.7% to +64.8% | 3 | 0.573802 → 0.94541 |
| joint_boundary | mboost_hybrid | tuned vs small | patience_60 / small | 5/5 | +0.0% | -1.0% to +63.7% | 3 | 0.581214 → 0.951624 |
| joint_boundary | mboost_hybrid | tuned vs small | repeated_cv_growth / small | 5/5 | +0.0% | -0.6% to +1.0% | 0 | 0.449959 → 0.454674 |
| joint_boundary | mboost_additive | tuned vs small | fixed_argmin / small | 5/5 | +0.0% | +0.0% to +0.0% | 0 | 9.18768 → 9.18768 |
| joint_boundary | mboost_additive | tuned vs small | boundary_growth / small | 5/5 | +0.0% | +0.0% to +0.0% | 0 | 9.18768 → 9.18768 |
| joint_boundary | mboost_additive | tuned vs small | patience_60 / small | 5/5 | +0.0% | +0.0% to +0.0% | 0 | 9.18768 → 9.18768 |
| joint_boundary | mboost_additive | tuned vs small | repeated_cv_growth / small | 5/5 | +0.0% | -0.0% to +0.0% | 0 | 9.18481 → 9.18481 |
| smooth_sparse | mboost_hybrid | tuned vs small | fixed_argmin / small | 5/5 | -0.2% | -0.6% to +0.5% | 0 | 0.276494 → 0.277935 |
| smooth_sparse | mboost_hybrid | tuned vs small | boundary_growth / small | 5/5 | -0.2% | -0.4% to +0.5% | 0 | 0.276494 → 0.277935 |
| smooth_sparse | mboost_hybrid | tuned vs small | patience_60 / small | 5/5 | -0.2% | -0.4% to +0.5% | 0 | 0.276494 → 0.277935 |
| smooth_sparse | mboost_hybrid | tuned vs small | repeated_cv_growth / small | 5/5 | -0.1% | -0.5% to +0.1% | 2 | 0.276114 → 0.27643 |
| smooth_sparse | mboost_additive | tuned vs small | fixed_argmin / small | 5/5 | +0.0% | -1.4% to +0.0% | 2 | 0.259631 → 0.259683 |
| smooth_sparse | mboost_additive | tuned vs small | boundary_growth / small | 5/5 | +0.0% | -1.4% to +0.0% | 2 | 0.259631 → 0.259683 |
| smooth_sparse | mboost_additive | tuned vs small | patience_60 / small | 5/5 | +0.0% | -1.4% to +0.0% | 2 | 0.259631 → 0.259683 |
| smooth_sparse | mboost_additive | tuned vs small | repeated_cv_growth / small | 5/5 | -0.0% | -0.0% to +0.0% | 2 | 0.259138 → 0.259146 |
| housing_iid | mboost_hybrid | tuned vs small | fixed_argmin / small | 3/3 | +0.0% | -5.4% to +0.6% | 1 | 0.0615226 → 0.0619007 |
| housing_iid | mboost_hybrid | tuned vs small | boundary_growth / small | 3/3 | -0.4% | -3.7% to +0.0% | 2 | 0.0601142 → 0.0601142 |
| housing_iid | mboost_hybrid | tuned vs small | patience_60 / small | 3/3 | +0.0% | -5.3% to +0.4% | 1 | 0.0615226 → 0.0617409 |
| housing_iid | mboost_hybrid | tuned vs small | repeated_cv_growth / small | 3/3 | -1.7% | -5.4% to +0.5% | 1 | 0.0609547 → 0.0612821 |
| housing_iid | mboost_additive | tuned vs small | fixed_argmin / small | 3/3 | -5.3% | -6.7% to -4.2% | 1 | 0.0881342 → 0.0844405 |
| housing_iid | mboost_additive | tuned vs small | boundary_growth / small | 3/3 | -3.6% | -4.5% to -1.8% | 0 | 0.0837616 → 0.0822268 |
| housing_iid | mboost_additive | tuned vs small | patience_60 / small | 3/3 | -3.6% | -4.5% to -1.8% | 0 | 0.0837616 → 0.0822268 |
| housing_iid | mboost_additive | tuned vs small | repeated_cv_growth / small | 3/3 | -3.2% | -15.4% to +0.0% | 0 | 0.0892776 → 0.0892776 |

## Diagnostic artifacts

The claim that the original diagnostics were unrecoverable from every branch is withdrawn. Commit `da27cd9` contains `benches/verify_selection/`, `benches/path_diagnostics_509.py` and `docs/benchmarks/path_diagnostics_509.json`, now on main. Their presence does not validate an oracle stopping rule: a minimum selected by scanning an assessment path remains exploratory.
