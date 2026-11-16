# Stopping-rule ablation (issue #509)

Version `stopping-ablation/2`. Produced 2026-10-14T09:57:09+00:00 from commit `ebe1e45cddf9f52065a268dde69370d88c6f0d30`. [Full results](results.json.gz), [machine-readable summaries](summary.json).

This run replaces the stopping conclusions of the October 12 v1 run. The original results are retained unchanged in [the archive](../results.json); its claims are audited in [the correction report](../audit_v1.md).

Eight prespecified rules use the same development rows and first budget. The historical `fixed_argmin` uses refitted preprocessing. `fixed_preprocessing` supplies the matched native baseline for `boundary_growth` and `patience_60`. Deployment-facing comparisons use `fold_local_fixed`, `fold_local_growth`, and `fold_local_cv_growth`, whose recipes are fitted on each fold's training rows.

Negative change favors the rule. Every percentage divides by the named reference's assessment loss on the same panel, method, tier and seed. Each table reports the median of those paired percentages, their full range and the seed with the largest increase. These are descriptive summaries, with no confidence intervals. Real-data assessment splits overlap and temporal origins share training history; they are not independent experiments. Synthetic seeds are independent draws. Calibration/test scores never choose the stopping rule or configuration.

## Preprocessing routes

Native fixed-preprocessing comparisons are diagnostic. Their held-out rows participate in basis construction; deployment conclusions use the fold-local tables below.

| panel | method | tier | rule / reference | finite seeds / all | median change | range | worst seed | reference loss → rule loss at worst seed |
|---|---|---|---|---|---|---|---|---|
| cancer | mboost_hybrid | small | fixed_preprocessing / fixed_argmin | 3/3 | +0.2% | +0.0% to +0.9% | 2 | 0.168826 → 0.170339 |
| cancer | mboost_hybrid | tuned | fixed_preprocessing / fixed_argmin | 3/3 | -1.1% | -51.3% to +0.0% | 1 | 0.116359 → 0.116359 |
| cancer | mboost_additive | small | fixed_preprocessing / fixed_argmin | 3/3 | +0.0% | +0.0% to +0.4% | 1 | 0.0647909 → 0.0650443 |
| cancer | mboost_additive | tuned | fixed_preprocessing / fixed_argmin | 3/3 | +0.1% | +0.0% to +0.2% | 1 | 0.0686097 → 0.0687536 |
| bike_iid | mboost_hybrid | small | fixed_preprocessing / fixed_argmin | 3/3 | +0.0% | +0.0% to +0.0% | 0 | 0.109603 → 0.109603 |
| bike_iid | mboost_hybrid | tuned | fixed_preprocessing / fixed_argmin | 3/3 | +0.0% | +0.0% to +0.0% | 0 | 0.119161 → 0.119161 |
| bike_iid | mboost_additive | small | fixed_preprocessing / fixed_argmin | 3/3 | +0.0% | +0.0% to +0.0% | 0 | 0.380234 → 0.380234 |
| bike_iid | mboost_additive | tuned | fixed_preprocessing / fixed_argmin | 3/3 | +0.0% | +0.0% to +0.0% | 0 | 0.355147 → 0.355147 |
| bike_future | mboost_hybrid | small | fixed_preprocessing / fixed_argmin | 3/3 | +0.0% | -0.5% to +0.0% | 0 | 0.198785 → 0.198785 |
| bike_future | mboost_hybrid | tuned | fixed_preprocessing / fixed_argmin | 3/3 | -0.7% | -1.1% to -0.5% | 0 | 0.191839 → 0.190952 |
| bike_future | mboost_additive | small | fixed_preprocessing / fixed_argmin | 3/3 | +0.0% | +0.0% to +0.2% | 1 | 0.443962 → 0.444661 |
| bike_future | mboost_additive | tuned | fixed_preprocessing / fixed_argmin | 3/3 | +0.0% | +0.0% to +0.5% | 1 | 0.450221 → 0.452524 |
| concrete | mboost_hybrid | small | fixed_preprocessing / fixed_argmin | 3/3 | +0.0% | +0.0% to +0.1% | 1 | 27.1108 → 27.1248 |
| concrete | mboost_hybrid | tuned | fixed_preprocessing / fixed_argmin | 3/3 | +0.0% | +0.0% to +0.0% | 2 | 24.2797 → 24.2851 |
| concrete | mboost_additive | small | fixed_preprocessing / fixed_argmin | 3/3 | +0.0% | +0.0% to +0.0% | 0 | 39.0056 → 39.0056 |
| concrete | mboost_additive | tuned | fixed_preprocessing / fixed_argmin | 3/3 | +0.0% | +0.0% to +0.0% | 2 | 38.8382 → 38.8439 |
| joint_boundary | mboost_hybrid | small | fixed_preprocessing / fixed_argmin | 5/5 | +0.0% | +0.0% to +0.3% | 4 | 0.456952 → 0.458275 |
| joint_boundary | mboost_hybrid | tuned | fixed_preprocessing / fixed_argmin | 5/5 | +0.1% | +0.0% to +2.0% | 1 | 0.716329 → 0.730766 |
| joint_boundary | mboost_additive | small | fixed_preprocessing / fixed_argmin | 5/5 | +0.0% | +0.0% to +0.0% | 0 | 9.18768 → 9.18768 |
| joint_boundary | mboost_additive | tuned | fixed_preprocessing / fixed_argmin | 5/5 | +0.0% | +0.0% to +0.0% | 0 | 9.18768 → 9.18768 |
| smooth_sparse | mboost_hybrid | small | fixed_preprocessing / fixed_argmin | 5/5 | +0.0% | -0.5% to +0.1% | 1 | 0.259227 → 0.259411 |
| smooth_sparse | mboost_hybrid | tuned | fixed_preprocessing / fixed_argmin | 5/5 | +0.0% | -0.4% to +0.0% | 3 | 0.28279 → 0.28292 |
| smooth_sparse | mboost_additive | small | fixed_preprocessing / fixed_argmin | 5/5 | +0.0% | +0.0% to +0.0% | 0 | 0.258837 → 0.258837 |
| smooth_sparse | mboost_additive | tuned | fixed_preprocessing / fixed_argmin | 5/5 | +0.0% | +0.0% to +0.0% | 0 | 0.258837 → 0.258837 |
| housing_iid | mboost_hybrid | small | fixed_preprocessing / fixed_argmin | 3/3 | +0.0% | +0.0% to +0.0% | 0 | 0.058684 → 0.058684 |
| housing_iid | mboost_hybrid | tuned | fixed_preprocessing / fixed_argmin | 3/3 | +0.0% | +0.0% to +0.0% | 0 | 0.0555417 → 0.0555505 |
| housing_iid | mboost_additive | small | fixed_preprocessing / fixed_argmin | 3/3 | +0.0% | -0.0% to +0.0% | 1 | 0.0881342 → 0.0881342 |
| housing_iid | mboost_additive | tuned | fixed_preprocessing / fixed_argmin | 3/3 | +0.0% | +0.0% to +0.0% | 0 | 0.0822268 → 0.0822268 |

## Native budget growth

Native fixed-preprocessing comparisons are diagnostic. Their held-out rows participate in basis construction; deployment conclusions use the fold-local tables below.

| panel | method | tier | rule / reference | finite seeds / all | median change | range | worst seed | reference loss → rule loss at worst seed |
|---|---|---|---|---|---|---|---|---|
| cancer | mboost_hybrid | small | boundary_growth / fixed_preprocessing | 3/3 | +0.0% | +0.0% to +8.2% | 2 | 0.170339 → 0.184241 |
| cancer | mboost_hybrid | tuned | boundary_growth / fixed_preprocessing | 3/3 | +0.0% | +0.0% to +8.2% | 2 | 0.170339 → 0.184241 |
| cancer | mboost_additive | small | boundary_growth / fixed_preprocessing | 3/3 | +0.0% | -3.3% to +1.5% | 0 | 0.0999297 → 0.101435 |
| cancer | mboost_additive | tuned | boundary_growth / fixed_preprocessing | 3/3 | -3.3% | -6.8% to +0.2% | 0 | 0.0963621 → 0.0965191 |
| bike_iid | mboost_hybrid | small | boundary_growth / fixed_preprocessing | 3/3 | -0.9% | -1.1% to +0.0% | 0 | 0.109603 → 0.109603 |
| bike_iid | mboost_hybrid | tuned | boundary_growth / fixed_preprocessing | 3/3 | -0.9% | -1.1% to +0.0% | 0 | 0.119161 → 0.119161 |
| bike_iid | mboost_additive | small | boundary_growth / fixed_preprocessing | 3/3 | -2.9% | -3.2% to -2.8% | 2 | 0.366459 → 0.356214 |
| bike_iid | mboost_additive | tuned | boundary_growth / fixed_preprocessing | 3/3 | +0.0% | +0.0% to +0.0% | 0 | 0.355147 → 0.355147 |
| bike_future | mboost_hybrid | small | boundary_growth / fixed_preprocessing | 3/3 | +0.0% | +0.0% to +0.5% | 2 | 0.263341 → 0.264764 |
| bike_future | mboost_hybrid | tuned | boundary_growth / fixed_preprocessing | 3/3 | +0.0% | -1.1% to +0.0% | 0 | 0.190952 → 0.190952 |
| bike_future | mboost_additive | small | boundary_growth / fixed_preprocessing | 3/3 | +0.0% | -3.4% to +0.0% | 0 | 0.445544 → 0.445544 |
| bike_future | mboost_additive | tuned | boundary_growth / fixed_preprocessing | 3/3 | +0.0% | -0.1% to +0.0% | 0 | 0.422323 → 0.422323 |
| concrete | mboost_hybrid | small | boundary_growth / fixed_preprocessing | 3/3 | -5.0% | -5.7% to +0.1% | 1 | 27.1248 → 27.1528 |
| concrete | mboost_hybrid | tuned | boundary_growth / fixed_preprocessing | 3/3 | -5.8% | -7.8% to +0.0% | 1 | 28.33 → 28.33 |
| concrete | mboost_additive | small | boundary_growth / fixed_preprocessing | 3/3 | -0.8% | -7.1% to +0.0% | 1 | 49.85 → 49.85 |
| concrete | mboost_additive | tuned | boundary_growth / fixed_preprocessing | 3/3 | -2.4% | -3.3% to +0.0% | 1 | 38.9102 → 38.9102 |
| joint_boundary | mboost_hybrid | small | boundary_growth / fixed_preprocessing | 5/5 | -9.7% | -17.0% to -1.2% | 4 | 0.458275 → 0.452904 |
| joint_boundary | mboost_hybrid | tuned | boundary_growth / fixed_preprocessing | 5/5 | -1.9% | -14.2% to -0.2% | 2 | 0.737609 → 0.736198 |
| joint_boundary | mboost_additive | small | boundary_growth / fixed_preprocessing | 5/5 | +0.0% | +0.0% to +0.0% | 0 | 9.18768 → 9.18768 |
| joint_boundary | mboost_additive | tuned | boundary_growth / fixed_preprocessing | 5/5 | +0.0% | +0.0% to +0.0% | 0 | 9.18768 → 9.18768 |
| smooth_sparse | mboost_hybrid | small | boundary_growth / fixed_preprocessing | 5/5 | +0.0% | +0.0% to +0.0% | 0 | 0.276494 → 0.276494 |
| smooth_sparse | mboost_hybrid | tuned | boundary_growth / fixed_preprocessing | 5/5 | +0.0% | +0.0% to +0.0% | 0 | 0.277935 → 0.277935 |
| smooth_sparse | mboost_additive | small | boundary_growth / fixed_preprocessing | 5/5 | +0.0% | +0.0% to +0.0% | 0 | 0.258837 → 0.258837 |
| smooth_sparse | mboost_additive | tuned | boundary_growth / fixed_preprocessing | 5/5 | +0.0% | +0.0% to +0.0% | 0 | 0.258837 → 0.258837 |
| housing_iid | mboost_hybrid | small | boundary_growth / fixed_preprocessing | 3/3 | +0.0% | -1.7% to +0.0% | 1 | 0.0615226 → 0.0615226 |
| housing_iid | mboost_hybrid | tuned | boundary_growth / fixed_preprocessing | 3/3 | +0.0% | -1.0% to +0.0% | 0 | 0.0555505 → 0.0555505 |
| housing_iid | mboost_additive | small | boundary_growth / fixed_preprocessing | 3/3 | -2.9% | -3.5% to -2.9% | 1 | 0.0881342 → 0.0855814 |
| housing_iid | mboost_additive | tuned | boundary_growth / fixed_preprocessing | 3/3 | -0.7% | -2.3% to +0.0% | 0 | 0.0822268 → 0.0822268 |

## Native patience

Native fixed-preprocessing comparisons are diagnostic. Their held-out rows participate in basis construction; deployment conclusions use the fold-local tables below.

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

## Native validation schemes

Native fixed-preprocessing comparisons are diagnostic. Their held-out rows participate in basis construction; deployment conclusions use the fold-local tables below.

| panel | method | tier | rule / reference | finite seeds / all | median change | range | worst seed | reference loss → rule loss at worst seed |
|---|---|---|---|---|---|---|---|---|
| cancer | mboost_hybrid | small | repeated_cv_growth / boundary_growth | 3/3 | -4.8% | -12.0% to -0.5% | 0 | 0.141584 → 0.14085 |
| cancer | mboost_hybrid | tuned | repeated_cv_growth / boundary_growth | 3/3 | +5.3% | -12.0% to +45.9% | 0 | 0.0965354 → 0.14085 |
| cancer | mboost_additive | small | repeated_cv_growth / boundary_growth | 3/3 | +3.8% | -0.8% to +4.0% | 2 | 0.120564 → 0.125436 |
| cancer | mboost_additive | tuned | repeated_cv_growth / boundary_growth | 3/3 | +4.3% | +4.0% to +5.3% | 1 | 0.0640752 → 0.0674904 |
| bike_iid | mboost_hybrid | small | repeated_cv_growth / boundary_growth | 3/3 | -0.8% | -3.0% to -0.1% | 2 | 0.0934566 → 0.0933546 |
| bike_iid | mboost_hybrid | tuned | repeated_cv_growth / boundary_growth | 3/3 | -0.8% | -10.9% to -0.1% | 2 | 0.0934566 → 0.0933546 |
| bike_iid | mboost_additive | small | repeated_cv_growth / boundary_growth | 3/3 | +0.0% | -0.0% to +0.0% | 0 | 0.369034 → 0.369034 |
| bike_iid | mboost_additive | tuned | repeated_cv_growth / boundary_growth | 3/3 | -0.1% | -0.3% to -0.1% | 1 | 0.333845 → 0.333453 |
| bike_future | mboost_hybrid | small | repeated_cv_growth / boundary_growth | 3/3 | +0.9% | -2.6% to +2.4% | 1 | 0.187355 → 0.19185 |
| bike_future | mboost_hybrid | tuned | repeated_cv_growth / boundary_growth | 3/3 | +2.7% | +1.9% to +3.5% | 0 | 0.190952 → 0.197716 |
| bike_future | mboost_additive | small | repeated_cv_growth / boundary_growth | 3/3 | -4.2% | -5.5% to +0.0% | 2 | 0.503081 → 0.5031 |
| bike_future | mboost_additive | tuned | repeated_cv_growth / boundary_growth | 3/3 | -2.3% | -5.2% to +5.0% | 2 | 0.484321 → 0.5084 |
| concrete | mboost_hybrid | small | repeated_cv_growth / boundary_growth | 3/3 | +0.0% | -2.5% to +0.3% | 1 | 27.1528 → 27.2415 |
| concrete | mboost_hybrid | tuned | repeated_cv_growth / boundary_growth | 3/3 | -1.7% | -2.1% to +0.0% | 0 | 18.923 → 18.9276 |
| concrete | mboost_additive | small | repeated_cv_growth / boundary_growth | 3/3 | +0.0% | -16.7% to +0.0% | 0 | 36.2338 → 36.237 |
| concrete | mboost_additive | tuned | repeated_cv_growth / boundary_growth | 3/3 | +0.0% | -8.3% to +0.8% | 2 | 37.9184 → 38.2296 |
| joint_boundary | mboost_hybrid | small | repeated_cv_growth / boundary_growth | 5/5 | -0.0% | -0.1% to +0.1% | 0 | 0.449536 → 0.449959 |
| joint_boundary | mboost_hybrid | tuned | repeated_cv_growth / boundary_growth | 5/5 | -1.4% | -39.3% to +1.1% | 0 | 0.449536 → 0.454674 |
| joint_boundary | mboost_additive | small | repeated_cv_growth / boundary_growth | 5/5 | -0.0% | -0.0% to +0.0% | 1 | 9.13517 → 9.13816 |
| joint_boundary | mboost_additive | tuned | repeated_cv_growth / boundary_growth | 5/5 | -0.0% | -0.0% to +0.0% | 1 | 9.13517 → 9.13816 |
| smooth_sparse | mboost_hybrid | small | repeated_cv_growth / boundary_growth | 5/5 | +0.1% | +0.0% to +0.4% | 4 | 0.261644 → 0.262585 |
| smooth_sparse | mboost_hybrid | tuned | repeated_cv_growth / boundary_growth | 5/5 | +0.1% | -0.5% to +0.3% | 4 | 0.260467 → 0.261172 |
| smooth_sparse | mboost_additive | small | repeated_cv_growth / boundary_growth | 5/5 | -0.3% | -0.9% to -0.0% | 4 | 0.242359 → 0.242306 |
| smooth_sparse | mboost_additive | tuned | repeated_cv_growth / boundary_growth | 5/5 | -0.1% | -0.9% to +1.0% | 1 | 0.243889 → 0.246416 |
| housing_iid | mboost_hybrid | small | repeated_cv_growth / boundary_growth | 3/3 | -0.9% | -1.4% to +2.6% | 0 | 0.0576806 → 0.0591981 |
| housing_iid | mboost_hybrid | tuned | repeated_cv_growth / boundary_growth | 3/3 | -0.0% | -3.0% to +0.8% | 0 | 0.0555505 → 0.0559823 |
| housing_iid | mboost_additive | small | repeated_cv_growth / boundary_growth | 3/3 | +6.6% | +0.0% to +129.1% | 2 | 0.0825325 → 0.189078 |
| housing_iid | mboost_additive | tuned | repeated_cv_growth / boundary_growth | 3/3 | +8.6% | +0.5% to +103.1% | 2 | 0.0788036 → 0.160031 |

## Fold-local budget growth

| panel | method | tier | rule / reference | finite seeds / all | median change | range | worst seed | reference loss → rule loss at worst seed |
|---|---|---|---|---|---|---|---|---|
| cancer | mboost_hybrid | small | fold_local_growth / fold_local_fixed | 3/3 | +0.0% | +0.0% to +0.0% | 0 | 0.141338 → 0.141338 |
| cancer | mboost_hybrid | tuned | fold_local_growth / fold_local_fixed | 3/3 | +0.0% | +0.0% to +11.3% | 2 | 0.350084 → 0.389818 |
| cancer | mboost_additive | small | fold_local_growth / fold_local_fixed | 3/3 | +0.0% | -3.3% to +1.6% | 0 | 0.0998846 → 0.101452 |
| cancer | mboost_additive | tuned | fold_local_growth / fold_local_fixed | 3/3 | -3.3% | -6.4% to +0.2% | 0 | 0.096278 → 0.0964751 |
| bike_iid | mboost_hybrid | small | fold_local_growth / fold_local_fixed | 3/3 | -0.9% | -1.1% to +0.0% | 0 | 0.109603 → 0.109603 |
| bike_iid | mboost_hybrid | tuned | fold_local_growth / fold_local_fixed | 3/3 | -0.9% | -1.1% to +0.0% | 0 | 0.119161 → 0.119161 |
| bike_iid | mboost_additive | small | fold_local_growth / fold_local_fixed | 3/3 | -2.9% | -3.2% to -2.8% | 2 | 0.366459 → 0.356214 |
| bike_iid | mboost_additive | tuned | fold_local_growth / fold_local_fixed | 3/3 | +0.0% | +0.0% to +0.0% | 0 | 0.355147 → 0.355147 |
| bike_future | mboost_hybrid | small | fold_local_growth / fold_local_fixed | 3/3 | +0.0% | +0.0% to +0.0% | 0 | 0.198785 → 0.198785 |
| bike_future | mboost_hybrid | tuned | fold_local_growth / fold_local_fixed | 3/3 | +0.0% | +0.0% to +0.0% | 0 | 0.191839 → 0.191839 |
| bike_future | mboost_additive | small | fold_local_growth / fold_local_fixed | 3/3 | +0.0% | -3.4% to +0.0% | 0 | 0.445512 → 0.445512 |
| bike_future | mboost_additive | tuned | fold_local_growth / fold_local_fixed | 3/3 | +0.0% | -0.1% to +0.0% | 0 | 0.422323 → 0.422323 |
| concrete | mboost_hybrid | small | fold_local_growth / fold_local_fixed | 3/3 | -4.9% | -5.7% to +0.2% | 1 | 27.1108 → 27.1528 |
| concrete | mboost_hybrid | tuned | fold_local_growth / fold_local_fixed | 3/3 | -4.9% | -7.8% to +0.0% | 1 | 28.33 → 28.33 |
| concrete | mboost_additive | small | fold_local_growth / fold_local_fixed | 3/3 | -0.8% | -7.1% to +0.0% | 1 | 49.85 → 49.85 |
| concrete | mboost_additive | tuned | fold_local_growth / fold_local_fixed | 3/3 | -2.4% | -3.3% to +0.0% | 1 | 38.9102 → 38.9102 |
| joint_boundary | mboost_hybrid | small | fold_local_growth / fold_local_fixed | 5/5 | -9.6% | -17.1% to -0.9% | 4 | 0.456952 → 0.452904 |
| joint_boundary | mboost_hybrid | tuned | fold_local_growth / fold_local_fixed | 5/5 | -1.2% | -12.4% to +0.0% | 2 | 0.737382 → 0.737382 |
| joint_boundary | mboost_additive | small | fold_local_growth / fold_local_fixed | 5/5 | +0.0% | +0.0% to +0.0% | 0 | 9.18768 → 9.18768 |
| joint_boundary | mboost_additive | tuned | fold_local_growth / fold_local_fixed | 5/5 | +0.0% | +0.0% to +0.0% | 0 | 9.18768 → 9.18768 |
| smooth_sparse | mboost_hybrid | small | fold_local_growth / fold_local_fixed | 5/5 | +0.0% | +0.0% to +0.0% | 0 | 0.276494 → 0.276494 |
| smooth_sparse | mboost_hybrid | tuned | fold_local_growth / fold_local_fixed | 5/5 | +0.0% | +0.0% to +0.0% | 0 | 0.277935 → 0.277935 |
| smooth_sparse | mboost_additive | small | fold_local_growth / fold_local_fixed | 5/5 | +0.0% | +0.0% to +0.0% | 0 | 0.258837 → 0.258837 |
| smooth_sparse | mboost_additive | tuned | fold_local_growth / fold_local_fixed | 5/5 | +0.0% | +0.0% to +0.0% | 0 | 0.258837 → 0.258837 |
| housing_iid | mboost_hybrid | small | fold_local_growth / fold_local_fixed | 3/3 | +0.0% | -1.7% to +0.0% | 1 | 0.0615226 → 0.0615226 |
| housing_iid | mboost_hybrid | tuned | fold_local_growth / fold_local_fixed | 3/3 | +0.0% | -1.0% to +0.0% | 0 | 0.0555417 → 0.0555417 |
| housing_iid | mboost_additive | small | fold_local_growth / fold_local_fixed | 3/3 | -2.9% | -3.5% to -2.9% | 1 | 0.0881342 → 0.0855814 |
| housing_iid | mboost_additive | tuned | fold_local_growth / fold_local_fixed | 3/3 | -0.7% | -2.3% to +0.0% | 0 | 0.0822268 → 0.0822268 |

## Fold-local validation schemes

| panel | method | tier | rule / reference | finite seeds / all | median change | range | worst seed | reference loss → rule loss at worst seed |
|---|---|---|---|---|---|---|---|---|
| cancer | mboost_hybrid | small | fold_local_cv_growth / fold_local_growth | 3/3 | -4.0% | -4.8% to -0.3% | 0 | 0.141338 → 0.14085 |
| cancer | mboost_hybrid | tuned | fold_local_cv_growth / fold_local_growth | 3/3 | +5.3% | -58.4% to +44.4% | 0 | 0.0975601 → 0.14085 |
| cancer | mboost_additive | small | fold_local_cv_growth / fold_local_growth | 3/3 | +3.9% | -0.6% to +4.0% | 2 | 0.120578 → 0.125442 |
| cancer | mboost_additive | tuned | fold_local_cv_growth / fold_local_growth | 3/3 | +4.5% | +4.0% to +4.8% | 1 | 0.0642067 → 0.067308 |
| bike_iid | mboost_hybrid | small | fold_local_cv_growth / fold_local_growth | 3/3 | -0.8% | -3.0% to -0.1% | 2 | 0.0934566 → 0.0933546 |
| bike_iid | mboost_hybrid | tuned | fold_local_cv_growth / fold_local_growth | 3/3 | -0.8% | -10.9% to -0.1% | 2 | 0.0934566 → 0.0933546 |
| bike_iid | mboost_additive | small | fold_local_cv_growth / fold_local_growth | 3/3 | +0.0% | -0.0% to +0.0% | 0 | 0.369034 → 0.369034 |
| bike_iid | mboost_additive | tuned | fold_local_cv_growth / fold_local_growth | 3/3 | -0.1% | -0.3% to -0.1% | 1 | 0.333845 → 0.333452 |
| bike_future | mboost_hybrid | small | fold_local_cv_growth / fold_local_growth | 3/3 | +1.0% | -2.6% to +2.4% | 1 | 0.187355 → 0.19185 |
| bike_future | mboost_hybrid | tuned | fold_local_cv_growth / fold_local_growth | 3/3 | +0.9% | +0.8% to +3.0% | 0 | 0.191839 → 0.19763 |
| bike_future | mboost_additive | small | fold_local_cv_growth / fold_local_growth | 3/3 | -4.1% | -5.5% to +0.0% | 2 | 0.503088 → 0.5031 |
| bike_future | mboost_additive | tuned | fold_local_cv_growth / fold_local_growth | 3/3 | -2.3% | -4.6% to +3.9% | 2 | 0.484321 → 0.5031 |
| concrete | mboost_hybrid | small | fold_local_cv_growth / fold_local_growth | 3/3 | +0.0% | -2.6% to +0.3% | 1 | 27.1528 → 27.2415 |
| concrete | mboost_hybrid | tuned | fold_local_cv_growth / fold_local_growth | 3/3 | -2.1% | -2.6% to -0.0% | 0 | 18.9276 → 18.923 |
| concrete | mboost_additive | small | fold_local_cv_growth / fold_local_growth | 3/3 | +0.0% | -16.7% to +0.0% | 0 | 36.2338 → 36.2338 |
| concrete | mboost_additive | tuned | fold_local_cv_growth / fold_local_growth | 3/3 | -0.0% | -8.3% to +0.7% | 2 | 37.9176 → 38.1673 |
| joint_boundary | mboost_hybrid | small | fold_local_cv_growth / fold_local_growth | 5/5 | -0.0% | -0.1% to +0.0% | 1 | 0.618436 → 0.618461 |
| joint_boundary | mboost_hybrid | tuned | fold_local_cv_growth / fold_local_growth | 5/5 | -1.4% | -39.3% to +0.8% | 0 | 0.449536 → 0.453202 |
| joint_boundary | mboost_additive | small | fold_local_cv_growth / fold_local_growth | 5/5 | -0.0% | -0.0% to +0.0% | 1 | 9.13517 → 9.13816 |
| joint_boundary | mboost_additive | tuned | fold_local_cv_growth / fold_local_growth | 5/5 | -0.0% | -0.0% to +0.0% | 1 | 9.13517 → 9.13816 |
| smooth_sparse | mboost_hybrid | small | fold_local_cv_growth / fold_local_growth | 5/5 | +0.1% | -0.2% to +0.3% | 2 | 0.275331 → 0.276177 |
| smooth_sparse | mboost_hybrid | tuned | fold_local_cv_growth / fold_local_growth | 5/5 | +0.1% | -0.5% to +0.2% | 2 | 0.275858 → 0.27643 |
| smooth_sparse | mboost_additive | small | fold_local_cv_growth / fold_local_growth | 5/5 | -0.3% | -0.9% to -0.0% | 4 | 0.242359 → 0.242306 |
| smooth_sparse | mboost_additive | tuned | fold_local_cv_growth / fold_local_growth | 5/5 | -0.1% | -0.9% to +1.0% | 1 | 0.243889 → 0.246416 |
| housing_iid | mboost_hybrid | small | fold_local_cv_growth / fold_local_growth | 3/3 | -1.2% | -1.5% to +2.2% | 0 | 0.0576806 → 0.0589755 |
| housing_iid | mboost_hybrid | tuned | fold_local_cv_growth / fold_local_growth | 3/3 | -0.0% | -2.8% to +0.8% | 0 | 0.0555417 → 0.0559823 |
| housing_iid | mboost_additive | small | fold_local_cv_growth / fold_local_growth | 3/3 | +5.0% | +0.0% to +76.6% | 2 | 0.0825366 → 0.145742 |
| housing_iid | mboost_additive | tuned | fold_local_cv_growth / fold_local_growth | 3/3 | +6.9% | +0.7% to +107.4% | 2 | 0.0788061 → 0.163482 |

## Configuration selection

The tuned configuration minimizes development-side risk among four candidates. The small tier uses the first candidate. Positive assessment changes show where configuration selection was harmful.

| panel | method | tier | rule / reference | finite seeds / all | median change | range | worst seed | reference loss → rule loss at worst seed |
|---|---|---|---|---|---|---|---|---|
| cancer | mboost_hybrid | tuned vs small | fixed_argmin / small | 3/3 | -9.6% | -31.0% to +107.4% | 2 | 0.168826 → 0.350084 |
| cancer | mboost_hybrid | tuned vs small | fixed_preprocessing / small | 3/3 | -9.6% | -31.8% to +0.0% | 2 | 0.170339 → 0.170339 |
| cancer | mboost_hybrid | tuned vs small | boundary_growth / small | 3/3 | -9.6% | -31.8% to +0.0% | 2 | 0.184241 → 0.184241 |
| cancer | mboost_hybrid | tuned vs small | patience_60 / small | 3/3 | -9.6% | -31.8% to +0.0% | 2 | 0.174633 → 0.174633 |
| cancer | mboost_hybrid | tuned vs small | repeated_cv_growth / small | 3/3 | +0.0% | +0.0% to +0.0% | 0 | 0.14085 → 0.14085 |
| cancer | mboost_hybrid | tuned vs small | fold_local_fixed / small | 3/3 | -9.6% | -31.0% to +107.4% | 2 | 0.168826 → 0.350084 |
| cancer | mboost_hybrid | tuned vs small | fold_local_growth / small | 3/3 | -9.6% | -31.0% to +130.9% | 2 | 0.168826 → 0.389818 |
| cancer | mboost_hybrid | tuned vs small | fold_local_cv_growth / small | 3/3 | +0.0% | +0.0% to +0.0% | 0 | 0.14085 → 0.14085 |
| cancer | mboost_additive | tuned vs small | fixed_argmin / small | 3/3 | +0.0% | -3.6% to +5.9% | 1 | 0.0647909 → 0.0686097 |
| cancer | mboost_additive | tuned vs small | fixed_preprocessing / small | 3/3 | +0.0% | -3.6% to +5.7% | 1 | 0.0650443 → 0.0687536 |
| cancer | mboost_additive | tuned vs small | boundary_growth / small | 3/3 | -1.5% | -4.8% to +0.0% | 2 | 0.120564 → 0.120564 |
| cancer | mboost_additive | tuned vs small | patience_60 / small | 3/3 | +0.0% | -4.8% to +0.7% | 1 | 0.0650443 → 0.0654674 |
| cancer | mboost_additive | tuned vs small | repeated_cv_growth / small | 3/3 | +0.0% | +0.0% to +0.0% | 0 | 0.100672 → 0.100672 |
| cancer | mboost_additive | tuned vs small | fold_local_fixed / small | 3/3 | +0.0% | -3.6% to +5.9% | 1 | 0.0647909 → 0.0686097 |
| cancer | mboost_additive | tuned vs small | fold_local_growth / small | 3/3 | -0.9% | -4.9% to +0.0% | 2 | 0.120578 → 0.120578 |
| cancer | mboost_additive | tuned vs small | fold_local_cv_growth / small | 3/3 | +0.0% | +0.0% to +0.0% | 0 | 0.1008 → 0.1008 |
| bike_iid | mboost_hybrid | tuned vs small | fixed_argmin / small | 3/3 | +0.0% | +0.0% to +8.7% | 0 | 0.109603 → 0.119161 |
| bike_iid | mboost_hybrid | tuned vs small | fixed_preprocessing / small | 3/3 | +0.0% | +0.0% to +8.7% | 0 | 0.109603 → 0.119161 |
| bike_iid | mboost_hybrid | tuned vs small | boundary_growth / small | 3/3 | +0.0% | +0.0% to +8.7% | 0 | 0.109603 → 0.119161 |
| bike_iid | mboost_hybrid | tuned vs small | patience_60 / small | 3/3 | +0.0% | +0.0% to +8.7% | 0 | 0.109603 → 0.119161 |
| bike_iid | mboost_hybrid | tuned vs small | repeated_cv_growth / small | 3/3 | +0.0% | -0.2% to +0.0% | 1 | 0.0969888 → 0.0969888 |
| bike_iid | mboost_hybrid | tuned vs small | fold_local_fixed / small | 3/3 | +0.0% | +0.0% to +8.7% | 0 | 0.109603 → 0.119161 |
| bike_iid | mboost_hybrid | tuned vs small | fold_local_growth / small | 3/3 | +0.0% | +0.0% to +8.7% | 0 | 0.109603 → 0.119161 |
| bike_iid | mboost_hybrid | tuned vs small | fold_local_cv_growth / small | 3/3 | +0.0% | -0.2% to +0.0% | 1 | 0.0969888 → 0.0969888 |
| bike_iid | mboost_additive | tuned vs small | fixed_argmin / small | 3/3 | -6.4% | -6.6% to -5.8% | 2 | 0.366459 → 0.345058 |
| bike_iid | mboost_additive | tuned vs small | fixed_preprocessing / small | 3/3 | -6.4% | -6.6% to -5.8% | 2 | 0.366459 → 0.345058 |
| bike_iid | mboost_additive | tuned vs small | boundary_growth / small | 3/3 | -3.3% | -3.8% to -3.1% | 2 | 0.356214 → 0.345058 |
| bike_iid | mboost_additive | tuned vs small | patience_60 / small | 3/3 | -3.3% | -3.8% to -3.1% | 2 | 0.356214 → 0.345058 |
| bike_iid | mboost_additive | tuned vs small | repeated_cv_growth / small | 3/3 | -3.4% | -3.9% to -3.4% | 1 | 0.345193 → 0.333453 |
| bike_iid | mboost_additive | tuned vs small | fold_local_fixed / small | 3/3 | -6.4% | -6.6% to -5.8% | 2 | 0.366459 → 0.345058 |
| bike_iid | mboost_additive | tuned vs small | fold_local_growth / small | 3/3 | -3.3% | -3.8% to -3.1% | 2 | 0.356214 → 0.345058 |
| bike_iid | mboost_additive | tuned vs small | fold_local_cv_growth / small | 3/3 | -3.4% | -3.9% to -3.4% | 1 | 0.345193 → 0.333452 |
| bike_future | mboost_hybrid | tuned vs small | fixed_argmin / small | 3/3 | +0.0% | -3.5% to +4.9% | 1 | 0.187355 → 0.196483 |
| bike_future | mboost_hybrid | tuned vs small | fixed_preprocessing / small | 3/3 | -0.2% | -3.9% to +3.7% | 1 | 0.187355 → 0.194308 |
| bike_future | mboost_hybrid | tuned vs small | boundary_growth / small | 3/3 | -1.8% | -3.9% to +3.7% | 1 | 0.187355 → 0.194308 |
| bike_future | mboost_hybrid | tuned vs small | patience_60 / small | 3/3 | +0.0% | -3.9% to +3.7% | 1 | 0.187355 → 0.194308 |
| bike_future | mboost_hybrid | tuned vs small | repeated_cv_growth / small | 3/3 | +2.1% | -0.1% to +3.2% | 1 | 0.19185 → 0.198045 |
| bike_future | mboost_hybrid | tuned vs small | fold_local_fixed / small | 3/3 | +0.0% | -3.5% to +4.9% | 1 | 0.187355 → 0.196483 |
| bike_future | mboost_hybrid | tuned vs small | fold_local_growth / small | 3/3 | +0.0% | -3.5% to +4.9% | 1 | 0.187355 → 0.196483 |
| bike_future | mboost_hybrid | tuned vs small | fold_local_cv_growth / small | 3/3 | +2.1% | -0.1% to +3.2% | 1 | 0.19185 → 0.198045 |
| bike_future | mboost_additive | tuned vs small | fixed_argmin / small | 3/3 | -5.2% | -6.9% to +1.4% | 1 | 0.443962 → 0.450221 |
| bike_future | mboost_additive | tuned vs small | fixed_preprocessing / small | 3/3 | -5.2% | -6.9% to +1.8% | 1 | 0.444661 → 0.452524 |
| bike_future | mboost_additive | tuned vs small | boundary_growth / small | 3/3 | -3.7% | -5.2% to +1.8% | 1 | 0.444661 → 0.452524 |
| bike_future | mboost_additive | tuned vs small | patience_60 / small | 3/3 | -3.7% | -5.2% to +1.8% | 1 | 0.444661 → 0.452524 |
| bike_future | mboost_additive | tuned vs small | repeated_cv_growth / small | 3/3 | +0.8% | -2.0% to +1.1% | 2 | 0.5031 → 0.5084 |
| bike_future | mboost_additive | tuned vs small | fold_local_fixed / small | 3/3 | -5.2% | -6.9% to +1.4% | 1 | 0.443962 → 0.450221 |
| bike_future | mboost_additive | tuned vs small | fold_local_growth / small | 3/3 | -3.7% | -5.2% to +1.4% | 1 | 0.443962 → 0.450221 |
| bike_future | mboost_additive | tuned vs small | fold_local_cv_growth / small | 3/3 | +0.0% | -2.0% to +0.9% | 1 | 0.425713 → 0.429458 |
| concrete | mboost_hybrid | tuned vs small | fixed_argmin / small | 3/3 | +0.0% | +0.0% to +4.5% | 1 | 27.1108 → 28.33 |
| concrete | mboost_hybrid | tuned vs small | fixed_preprocessing / small | 3/3 | +0.0% | +0.0% to +4.4% | 1 | 27.1248 → 28.33 |
| concrete | mboost_hybrid | tuned vs small | boundary_growth / small | 3/3 | -0.8% | -2.2% to +4.3% | 1 | 27.1528 → 28.33 |
| concrete | mboost_hybrid | tuned vs small | patience_60 / small | 3/3 | +0.0% | -4.8% to +4.4% | 1 | 27.1248 → 28.33 |
| concrete | mboost_hybrid | tuned vs small | repeated_cv_growth / small | 3/3 | +0.0% | -2.2% to +1.8% | 1 | 27.2415 → 27.7229 |
| concrete | mboost_hybrid | tuned vs small | fold_local_fixed / small | 3/3 | +0.0% | +0.0% to +4.5% | 1 | 27.1108 → 28.33 |
| concrete | mboost_hybrid | tuned vs small | fold_local_growth / small | 3/3 | +0.0% | -2.2% to +4.3% | 1 | 27.1528 → 28.33 |
| concrete | mboost_hybrid | tuned vs small | fold_local_cv_growth / small | 3/3 | +0.0% | -2.2% to +1.8% | 1 | 27.2415 → 27.7236 |
| concrete | mboost_additive | tuned vs small | fixed_argmin / small | 3/3 | -20.0% | -21.9% to -6.8% | 2 | 41.6645 → 38.8382 |
| concrete | mboost_additive | tuned vs small | fixed_preprocessing / small | 3/3 | -20.0% | -21.9% to -6.8% | 2 | 41.6645 → 38.8439 |
| concrete | mboost_additive | tuned vs small | boundary_growth / small | 3/3 | -16.7% | -21.9% to -8.3% | 2 | 41.3475 → 37.9184 |
| concrete | mboost_additive | tuned vs small | patience_60 / small | 3/3 | -16.7% | -21.9% to -8.3% | 2 | 41.3475 → 37.9184 |
| concrete | mboost_additive | tuned vs small | repeated_cv_growth / small | 3/3 | -14.1% | -16.7% to -7.5% | 2 | 41.3475 → 38.2296 |
| concrete | mboost_additive | tuned vs small | fold_local_fixed / small | 3/3 | -20.0% | -21.9% to -6.8% | 2 | 41.6645 → 38.8382 |
| concrete | mboost_additive | tuned vs small | fold_local_growth / small | 3/3 | -16.7% | -21.9% to -8.3% | 2 | 41.3475 → 37.9176 |
| concrete | mboost_additive | tuned vs small | fold_local_cv_growth / small | 3/3 | -14.1% | -16.7% to -7.7% | 2 | 41.3475 → 38.1673 |
| joint_boundary | mboost_hybrid | tuned vs small | fixed_argmin / small | 5/5 | +0.0% | -0.6% to +58.1% | 3 | 0.604245 → 0.955144 |
| joint_boundary | mboost_hybrid | tuned vs small | fixed_preprocessing / small | 5/5 | +2.0% | -0.5% to +58.1% | 3 | 0.604245 → 0.955144 |
| joint_boundary | mboost_hybrid | tuned vs small | boundary_growth / small | 5/5 | +1.4% | -0.7% to +64.8% | 3 | 0.573802 → 0.94541 |
| joint_boundary | mboost_hybrid | tuned vs small | patience_60 / small | 5/5 | +0.0% | -1.0% to +63.7% | 3 | 0.581214 → 0.951624 |
| joint_boundary | mboost_hybrid | tuned vs small | repeated_cv_growth / small | 5/5 | +0.0% | -0.6% to +1.0% | 0 | 0.449959 → 0.454674 |
| joint_boundary | mboost_hybrid | tuned vs small | fold_local_fixed / small | 5/5 | +0.0% | -0.6% to +58.1% | 3 | 0.604245 → 0.955144 |
| joint_boundary | mboost_hybrid | tuned vs small | fold_local_growth / small | 5/5 | +1.4% | -0.4% to +64.8% | 3 | 0.573802 → 0.94541 |
| joint_boundary | mboost_hybrid | tuned vs small | fold_local_cv_growth / small | 5/5 | +0.0% | -0.4% to +0.9% | 0 | 0.449288 → 0.453202 |
| joint_boundary | mboost_additive | tuned vs small | fixed_argmin / small | 5/5 | +0.0% | +0.0% to +0.0% | 0 | 9.18768 → 9.18768 |
| joint_boundary | mboost_additive | tuned vs small | fixed_preprocessing / small | 5/5 | +0.0% | +0.0% to +0.0% | 0 | 9.18768 → 9.18768 |
| joint_boundary | mboost_additive | tuned vs small | boundary_growth / small | 5/5 | +0.0% | +0.0% to +0.0% | 0 | 9.18768 → 9.18768 |
| joint_boundary | mboost_additive | tuned vs small | patience_60 / small | 5/5 | +0.0% | +0.0% to +0.0% | 0 | 9.18768 → 9.18768 |
| joint_boundary | mboost_additive | tuned vs small | repeated_cv_growth / small | 5/5 | +0.0% | -0.0% to +0.0% | 0 | 9.18481 → 9.18481 |
| joint_boundary | mboost_additive | tuned vs small | fold_local_fixed / small | 5/5 | +0.0% | +0.0% to +0.0% | 0 | 9.18768 → 9.18768 |
| joint_boundary | mboost_additive | tuned vs small | fold_local_growth / small | 5/5 | +0.0% | +0.0% to +0.0% | 0 | 9.18768 → 9.18768 |
| joint_boundary | mboost_additive | tuned vs small | fold_local_cv_growth / small | 5/5 | +0.0% | -0.0% to +0.0% | 0 | 9.18481 → 9.18481 |
| smooth_sparse | mboost_hybrid | tuned vs small | fixed_argmin / small | 5/5 | -0.2% | -0.6% to +0.5% | 0 | 0.276494 → 0.277935 |
| smooth_sparse | mboost_hybrid | tuned vs small | fixed_preprocessing / small | 5/5 | -0.2% | -0.4% to +0.5% | 0 | 0.276494 → 0.277935 |
| smooth_sparse | mboost_hybrid | tuned vs small | boundary_growth / small | 5/5 | -0.2% | -0.4% to +0.5% | 0 | 0.276494 → 0.277935 |
| smooth_sparse | mboost_hybrid | tuned vs small | patience_60 / small | 5/5 | -0.2% | -0.4% to +0.5% | 0 | 0.276494 → 0.277935 |
| smooth_sparse | mboost_hybrid | tuned vs small | repeated_cv_growth / small | 5/5 | -0.1% | -0.5% to +0.1% | 2 | 0.276114 → 0.27643 |
| smooth_sparse | mboost_hybrid | tuned vs small | fold_local_fixed / small | 5/5 | -0.2% | -0.6% to +0.5% | 0 | 0.276494 → 0.277935 |
| smooth_sparse | mboost_hybrid | tuned vs small | fold_local_growth / small | 5/5 | -0.2% | -0.6% to +0.5% | 0 | 0.276494 → 0.277935 |
| smooth_sparse | mboost_hybrid | tuned vs small | fold_local_cv_growth / small | 5/5 | -0.1% | -0.5% to +0.1% | 2 | 0.276177 → 0.27643 |
| smooth_sparse | mboost_additive | tuned vs small | fixed_argmin / small | 5/5 | +0.0% | -1.4% to +0.0% | 2 | 0.259631 → 0.259683 |
| smooth_sparse | mboost_additive | tuned vs small | fixed_preprocessing / small | 5/5 | +0.0% | -1.4% to +0.0% | 2 | 0.259631 → 0.259683 |
| smooth_sparse | mboost_additive | tuned vs small | boundary_growth / small | 5/5 | +0.0% | -1.4% to +0.0% | 2 | 0.259631 → 0.259683 |
| smooth_sparse | mboost_additive | tuned vs small | patience_60 / small | 5/5 | +0.0% | -1.4% to +0.0% | 2 | 0.259631 → 0.259683 |
| smooth_sparse | mboost_additive | tuned vs small | repeated_cv_growth / small | 5/5 | -0.0% | -0.0% to +0.0% | 2 | 0.259138 → 0.259146 |
| smooth_sparse | mboost_additive | tuned vs small | fold_local_fixed / small | 5/5 | +0.0% | -1.4% to +0.0% | 2 | 0.259631 → 0.259683 |
| smooth_sparse | mboost_additive | tuned vs small | fold_local_growth / small | 5/5 | +0.0% | -1.4% to +0.0% | 2 | 0.259631 → 0.259683 |
| smooth_sparse | mboost_additive | tuned vs small | fold_local_cv_growth / small | 5/5 | -0.0% | -0.0% to +0.0% | 2 | 0.259138 → 0.259146 |
| housing_iid | mboost_hybrid | tuned vs small | fixed_argmin / small | 3/3 | +0.0% | -5.4% to +0.6% | 1 | 0.0615226 → 0.0619007 |
| housing_iid | mboost_hybrid | tuned vs small | fixed_preprocessing / small | 3/3 | +0.0% | -5.3% to +0.6% | 1 | 0.0615226 → 0.0619007 |
| housing_iid | mboost_hybrid | tuned vs small | boundary_growth / small | 3/3 | -0.4% | -3.7% to +0.0% | 2 | 0.0601142 → 0.0601142 |
| housing_iid | mboost_hybrid | tuned vs small | patience_60 / small | 3/3 | +0.0% | -5.3% to +0.4% | 1 | 0.0615226 → 0.0617409 |
| housing_iid | mboost_hybrid | tuned vs small | repeated_cv_growth / small | 3/3 | -1.7% | -5.4% to +0.5% | 1 | 0.0609547 → 0.0612821 |
| housing_iid | mboost_hybrid | tuned vs small | fold_local_fixed / small | 3/3 | +0.0% | -5.4% to +0.6% | 1 | 0.0615226 → 0.0619007 |
| housing_iid | mboost_hybrid | tuned vs small | fold_local_growth / small | 3/3 | -0.4% | -3.7% to +0.0% | 2 | 0.0601142 → 0.0601142 |
| housing_iid | mboost_hybrid | tuned vs small | fold_local_cv_growth / small | 3/3 | -1.4% | -5.1% to +0.8% | 1 | 0.060796 → 0.0612821 |
| housing_iid | mboost_additive | tuned vs small | fixed_argmin / small | 3/3 | -5.3% | -6.7% to -4.2% | 1 | 0.0881342 → 0.0844405 |
| housing_iid | mboost_additive | tuned vs small | fixed_preprocessing / small | 3/3 | -5.3% | -6.7% to -4.2% | 1 | 0.0881342 → 0.0844405 |
| housing_iid | mboost_additive | tuned vs small | boundary_growth / small | 3/3 | -3.6% | -4.5% to -1.8% | 0 | 0.0837616 → 0.0822268 |
| housing_iid | mboost_additive | tuned vs small | patience_60 / small | 3/3 | -3.6% | -4.5% to -1.8% | 0 | 0.0837616 → 0.0822268 |
| housing_iid | mboost_additive | tuned vs small | repeated_cv_growth / small | 3/3 | -3.2% | -15.4% to +0.0% | 0 | 0.0892776 → 0.0892776 |
| housing_iid | mboost_additive | tuned vs small | fold_local_fixed / small | 3/3 | -5.3% | -6.7% to -4.2% | 1 | 0.0881342 → 0.0844405 |
| housing_iid | mboost_additive | tuned vs small | fold_local_growth / small | 3/3 | -3.6% | -4.5% to -1.8% | 0 | 0.0837531 → 0.0822268 |
| housing_iid | mboost_additive | tuned vs small | fold_local_cv_growth / small | 3/3 | +0.0% | -2.9% to +12.2% | 2 | 0.145742 → 0.163482 |

## Boundaries and validation

A boundary flag means the minimum remains in the library's boundary band; reaching the cap leaves it unresolved. Counts refer to the configuration selected for that tier, over the displayed seeds.

| panel | method | tier | rule | selected boundaries / seeds | extensions / seeds |
|---|---|---|---|---|---|
| cancer | mboost_hybrid | small | fixed_argmin | 0/3 | 0/3 |
| cancer | mboost_hybrid | tuned | fixed_argmin | 1/3 | 0/3 |
| cancer | mboost_hybrid | small | fixed_preprocessing | 1/3 | 0/3 |
| cancer | mboost_hybrid | tuned | fixed_preprocessing | 1/3 | 0/3 |
| cancer | mboost_hybrid | small | boundary_growth | 1/3 | 1/3 |
| cancer | mboost_hybrid | tuned | boundary_growth | 1/3 | 1/3 |
| cancer | mboost_hybrid | small | patience_60 | 0/3 | 1/3 |
| cancer | mboost_hybrid | tuned | patience_60 | 0/3 | 1/3 |
| cancer | mboost_hybrid | small | repeated_cv_growth | 0/3 | 0/3 |
| cancer | mboost_hybrid | tuned | repeated_cv_growth | 0/3 | 0/3 |
| cancer | mboost_hybrid | small | fold_local_fixed | 0/3 | 0/3 |
| cancer | mboost_hybrid | tuned | fold_local_fixed | 1/3 | 0/3 |
| cancer | mboost_hybrid | small | fold_local_growth | 0/3 | 0/3 |
| cancer | mboost_hybrid | tuned | fold_local_growth | 1/3 | 1/3 |
| cancer | mboost_hybrid | small | fold_local_cv_growth | 0/3 | 0/3 |
| cancer | mboost_hybrid | tuned | fold_local_cv_growth | 0/3 | 0/3 |
| cancer | mboost_additive | small | fixed_argmin | 2/3 | 0/3 |
| cancer | mboost_additive | tuned | fixed_argmin | 3/3 | 0/3 |
| cancer | mboost_additive | small | fixed_preprocessing | 2/3 | 0/3 |
| cancer | mboost_additive | tuned | fixed_preprocessing | 3/3 | 0/3 |
| cancer | mboost_additive | small | boundary_growth | 2/3 | 2/3 |
| cancer | mboost_additive | tuned | boundary_growth | 3/3 | 3/3 |
| cancer | mboost_additive | small | patience_60 | 2/3 | 2/3 |
| cancer | mboost_additive | tuned | patience_60 | 2/3 | 3/3 |
| cancer | mboost_additive | small | repeated_cv_growth | 0/3 | 1/3 |
| cancer | mboost_additive | tuned | repeated_cv_growth | 0/3 | 1/3 |
| cancer | mboost_additive | small | fold_local_fixed | 2/3 | 0/3 |
| cancer | mboost_additive | tuned | fold_local_fixed | 3/3 | 0/3 |
| cancer | mboost_additive | small | fold_local_growth | 2/3 | 2/3 |
| cancer | mboost_additive | tuned | fold_local_growth | 2/3 | 3/3 |
| cancer | mboost_additive | small | fold_local_cv_growth | 0/3 | 1/3 |
| cancer | mboost_additive | tuned | fold_local_cv_growth | 0/3 | 1/3 |
| bike_iid | mboost_hybrid | small | fixed_argmin | 2/3 | 0/3 |
| bike_iid | mboost_hybrid | tuned | fixed_argmin | 2/3 | 0/3 |
| bike_iid | mboost_hybrid | small | fixed_preprocessing | 2/3 | 0/3 |
| bike_iid | mboost_hybrid | tuned | fixed_preprocessing | 2/3 | 0/3 |
| bike_iid | mboost_hybrid | small | boundary_growth | 0/3 | 2/3 |
| bike_iid | mboost_hybrid | tuned | boundary_growth | 0/3 | 2/3 |
| bike_iid | mboost_hybrid | small | patience_60 | 0/3 | 2/3 |
| bike_iid | mboost_hybrid | tuned | patience_60 | 0/3 | 2/3 |
| bike_iid | mboost_hybrid | small | repeated_cv_growth | 0/3 | 3/3 |
| bike_iid | mboost_hybrid | tuned | repeated_cv_growth | 0/3 | 2/3 |
| bike_iid | mboost_hybrid | small | fold_local_fixed | 2/3 | 0/3 |
| bike_iid | mboost_hybrid | tuned | fold_local_fixed | 2/3 | 0/3 |
| bike_iid | mboost_hybrid | small | fold_local_growth | 0/3 | 2/3 |
| bike_iid | mboost_hybrid | tuned | fold_local_growth | 0/3 | 2/3 |
| bike_iid | mboost_hybrid | small | fold_local_cv_growth | 0/3 | 3/3 |
| bike_iid | mboost_hybrid | tuned | fold_local_cv_growth | 0/3 | 2/3 |
| bike_iid | mboost_additive | small | fixed_argmin | 3/3 | 0/3 |
| bike_iid | mboost_additive | tuned | fixed_argmin | 0/3 | 0/3 |
| bike_iid | mboost_additive | small | fixed_preprocessing | 3/3 | 0/3 |
| bike_iid | mboost_additive | tuned | fixed_preprocessing | 0/3 | 0/3 |
| bike_iid | mboost_additive | small | boundary_growth | 3/3 | 3/3 |
| bike_iid | mboost_additive | tuned | boundary_growth | 0/3 | 0/3 |
| bike_iid | mboost_additive | small | patience_60 | 3/3 | 3/3 |
| bike_iid | mboost_additive | tuned | patience_60 | 0/3 | 0/3 |
| bike_iid | mboost_additive | small | repeated_cv_growth | 3/3 | 3/3 |
| bike_iid | mboost_additive | tuned | repeated_cv_growth | 2/3 | 2/3 |
| bike_iid | mboost_additive | small | fold_local_fixed | 3/3 | 0/3 |
| bike_iid | mboost_additive | tuned | fold_local_fixed | 0/3 | 0/3 |
| bike_iid | mboost_additive | small | fold_local_growth | 3/3 | 3/3 |
| bike_iid | mboost_additive | tuned | fold_local_growth | 0/3 | 0/3 |
| bike_iid | mboost_additive | small | fold_local_cv_growth | 3/3 | 3/3 |
| bike_iid | mboost_additive | tuned | fold_local_cv_growth | 2/3 | 2/3 |
| bike_future | mboost_hybrid | small | fixed_argmin | 0/3 | 0/3 |
| bike_future | mboost_hybrid | tuned | fixed_argmin | 1/3 | 0/3 |
| bike_future | mboost_hybrid | small | fixed_preprocessing | 1/3 | 0/3 |
| bike_future | mboost_hybrid | tuned | fixed_preprocessing | 1/3 | 0/3 |
| bike_future | mboost_hybrid | small | boundary_growth | 0/3 | 1/3 |
| bike_future | mboost_hybrid | tuned | boundary_growth | 1/3 | 1/3 |
| bike_future | mboost_hybrid | small | patience_60 | 0/3 | 1/3 |
| bike_future | mboost_hybrid | tuned | patience_60 | 0/3 | 1/3 |
| bike_future | mboost_hybrid | small | repeated_cv_growth | 0/3 | 0/3 |
| bike_future | mboost_hybrid | tuned | repeated_cv_growth | 0/3 | 0/3 |
| bike_future | mboost_hybrid | small | fold_local_fixed | 0/3 | 0/3 |
| bike_future | mboost_hybrid | tuned | fold_local_fixed | 1/3 | 0/3 |
| bike_future | mboost_hybrid | small | fold_local_growth | 0/3 | 0/3 |
| bike_future | mboost_hybrid | tuned | fold_local_growth | 0/3 | 1/3 |
| bike_future | mboost_hybrid | small | fold_local_cv_growth | 0/3 | 0/3 |
| bike_future | mboost_hybrid | tuned | fold_local_cv_growth | 0/3 | 0/3 |
| bike_future | mboost_additive | small | fixed_argmin | 1/3 | 0/3 |
| bike_future | mboost_additive | tuned | fixed_argmin | 1/3 | 0/3 |
| bike_future | mboost_additive | small | fixed_preprocessing | 1/3 | 0/3 |
| bike_future | mboost_additive | tuned | fixed_preprocessing | 1/3 | 0/3 |
| bike_future | mboost_additive | small | boundary_growth | 1/3 | 1/3 |
| bike_future | mboost_additive | tuned | boundary_growth | 1/3 | 1/3 |
| bike_future | mboost_additive | small | patience_60 | 1/3 | 1/3 |
| bike_future | mboost_additive | tuned | patience_60 | 1/3 | 1/3 |
| bike_future | mboost_additive | small | repeated_cv_growth | 3/3 | 3/3 |
| bike_future | mboost_additive | tuned | repeated_cv_growth | 0/3 | 0/3 |
| bike_future | mboost_additive | small | fold_local_fixed | 1/3 | 0/3 |
| bike_future | mboost_additive | tuned | fold_local_fixed | 1/3 | 0/3 |
| bike_future | mboost_additive | small | fold_local_growth | 1/3 | 1/3 |
| bike_future | mboost_additive | tuned | fold_local_growth | 1/3 | 1/3 |
| bike_future | mboost_additive | small | fold_local_cv_growth | 3/3 | 3/3 |
| bike_future | mboost_additive | tuned | fold_local_cv_growth | 1/3 | 1/3 |
| concrete | mboost_hybrid | small | fixed_argmin | 3/3 | 0/3 |
| concrete | mboost_hybrid | tuned | fixed_argmin | 2/3 | 0/3 |
| concrete | mboost_hybrid | small | fixed_preprocessing | 3/3 | 0/3 |
| concrete | mboost_hybrid | tuned | fixed_preprocessing | 2/3 | 0/3 |
| concrete | mboost_hybrid | small | boundary_growth | 2/3 | 3/3 |
| concrete | mboost_hybrid | tuned | boundary_growth | 2/3 | 2/3 |
| concrete | mboost_hybrid | small | patience_60 | 0/3 | 3/3 |
| concrete | mboost_hybrid | tuned | patience_60 | 1/3 | 2/3 |
| concrete | mboost_hybrid | small | repeated_cv_growth | 2/3 | 3/3 |
| concrete | mboost_hybrid | tuned | repeated_cv_growth | 3/3 | 3/3 |
| concrete | mboost_hybrid | small | fold_local_fixed | 3/3 | 0/3 |
| concrete | mboost_hybrid | tuned | fold_local_fixed | 2/3 | 0/3 |
| concrete | mboost_hybrid | small | fold_local_growth | 2/3 | 3/3 |
| concrete | mboost_hybrid | tuned | fold_local_growth | 1/3 | 2/3 |
| concrete | mboost_hybrid | small | fold_local_cv_growth | 2/3 | 3/3 |
| concrete | mboost_hybrid | tuned | fold_local_cv_growth | 3/3 | 3/3 |
| concrete | mboost_additive | small | fixed_argmin | 2/3 | 0/3 |
| concrete | mboost_additive | tuned | fixed_argmin | 2/3 | 0/3 |
| concrete | mboost_additive | small | fixed_preprocessing | 2/3 | 0/3 |
| concrete | mboost_additive | tuned | fixed_preprocessing | 2/3 | 0/3 |
| concrete | mboost_additive | small | boundary_growth | 2/3 | 2/3 |
| concrete | mboost_additive | tuned | boundary_growth | 2/3 | 2/3 |
| concrete | mboost_additive | small | patience_60 | 2/3 | 2/3 |
| concrete | mboost_additive | tuned | patience_60 | 2/3 | 2/3 |
| concrete | mboost_additive | small | repeated_cv_growth | 3/3 | 3/3 |
| concrete | mboost_additive | tuned | repeated_cv_growth | 2/3 | 3/3 |
| concrete | mboost_additive | small | fold_local_fixed | 2/3 | 0/3 |
| concrete | mboost_additive | tuned | fold_local_fixed | 2/3 | 0/3 |
| concrete | mboost_additive | small | fold_local_growth | 2/3 | 2/3 |
| concrete | mboost_additive | tuned | fold_local_growth | 2/3 | 2/3 |
| concrete | mboost_additive | small | fold_local_cv_growth | 3/3 | 3/3 |
| concrete | mboost_additive | tuned | fold_local_cv_growth | 2/3 | 3/3 |
| joint_boundary | mboost_hybrid | small | fixed_argmin | 5/5 | 0/5 |
| joint_boundary | mboost_hybrid | tuned | fixed_argmin | 4/5 | 0/5 |
| joint_boundary | mboost_hybrid | small | fixed_preprocessing | 5/5 | 0/5 |
| joint_boundary | mboost_hybrid | tuned | fixed_preprocessing | 5/5 | 0/5 |
| joint_boundary | mboost_hybrid | small | boundary_growth | 3/5 | 5/5 |
| joint_boundary | mboost_hybrid | tuned | boundary_growth | 0/5 | 5/5 |
| joint_boundary | mboost_hybrid | small | patience_60 | 0/5 | 5/5 |
| joint_boundary | mboost_hybrid | tuned | patience_60 | 0/5 | 5/5 |
| joint_boundary | mboost_hybrid | small | repeated_cv_growth | 5/5 | 5/5 |
| joint_boundary | mboost_hybrid | tuned | repeated_cv_growth | 5/5 | 5/5 |
| joint_boundary | mboost_hybrid | small | fold_local_fixed | 5/5 | 0/5 |
| joint_boundary | mboost_hybrid | tuned | fold_local_fixed | 4/5 | 0/5 |
| joint_boundary | mboost_hybrid | small | fold_local_growth | 3/5 | 5/5 |
| joint_boundary | mboost_hybrid | tuned | fold_local_growth | 0/5 | 4/5 |
| joint_boundary | mboost_hybrid | small | fold_local_cv_growth | 5/5 | 5/5 |
| joint_boundary | mboost_hybrid | tuned | fold_local_cv_growth | 5/5 | 5/5 |
| joint_boundary | mboost_additive | small | fixed_argmin | 0/5 | 0/5 |
| joint_boundary | mboost_additive | tuned | fixed_argmin | 0/5 | 0/5 |
| joint_boundary | mboost_additive | small | fixed_preprocessing | 0/5 | 0/5 |
| joint_boundary | mboost_additive | tuned | fixed_preprocessing | 0/5 | 0/5 |
| joint_boundary | mboost_additive | small | boundary_growth | 0/5 | 0/5 |
| joint_boundary | mboost_additive | tuned | boundary_growth | 0/5 | 0/5 |
| joint_boundary | mboost_additive | small | patience_60 | 0/5 | 0/5 |
| joint_boundary | mboost_additive | tuned | patience_60 | 0/5 | 0/5 |
| joint_boundary | mboost_additive | small | repeated_cv_growth | 0/5 | 0/5 |
| joint_boundary | mboost_additive | tuned | repeated_cv_growth | 0/5 | 0/5 |
| joint_boundary | mboost_additive | small | fold_local_fixed | 0/5 | 0/5 |
| joint_boundary | mboost_additive | tuned | fold_local_fixed | 0/5 | 0/5 |
| joint_boundary | mboost_additive | small | fold_local_growth | 0/5 | 0/5 |
| joint_boundary | mboost_additive | tuned | fold_local_growth | 0/5 | 0/5 |
| joint_boundary | mboost_additive | small | fold_local_cv_growth | 0/5 | 0/5 |
| joint_boundary | mboost_additive | tuned | fold_local_cv_growth | 0/5 | 0/5 |
| smooth_sparse | mboost_hybrid | small | fixed_argmin | 0/5 | 0/5 |
| smooth_sparse | mboost_hybrid | tuned | fixed_argmin | 0/5 | 0/5 |
| smooth_sparse | mboost_hybrid | small | fixed_preprocessing | 0/5 | 0/5 |
| smooth_sparse | mboost_hybrid | tuned | fixed_preprocessing | 0/5 | 0/5 |
| smooth_sparse | mboost_hybrid | small | boundary_growth | 0/5 | 0/5 |
| smooth_sparse | mboost_hybrid | tuned | boundary_growth | 0/5 | 0/5 |
| smooth_sparse | mboost_hybrid | small | patience_60 | 0/5 | 0/5 |
| smooth_sparse | mboost_hybrid | tuned | patience_60 | 0/5 | 0/5 |
| smooth_sparse | mboost_hybrid | small | repeated_cv_growth | 0/5 | 0/5 |
| smooth_sparse | mboost_hybrid | tuned | repeated_cv_growth | 0/5 | 0/5 |
| smooth_sparse | mboost_hybrid | small | fold_local_fixed | 0/5 | 0/5 |
| smooth_sparse | mboost_hybrid | tuned | fold_local_fixed | 0/5 | 0/5 |
| smooth_sparse | mboost_hybrid | small | fold_local_growth | 0/5 | 0/5 |
| smooth_sparse | mboost_hybrid | tuned | fold_local_growth | 0/5 | 0/5 |
| smooth_sparse | mboost_hybrid | small | fold_local_cv_growth | 0/5 | 0/5 |
| smooth_sparse | mboost_hybrid | tuned | fold_local_cv_growth | 0/5 | 0/5 |
| smooth_sparse | mboost_additive | small | fixed_argmin | 0/5 | 0/5 |
| smooth_sparse | mboost_additive | tuned | fixed_argmin | 0/5 | 0/5 |
| smooth_sparse | mboost_additive | small | fixed_preprocessing | 0/5 | 0/5 |
| smooth_sparse | mboost_additive | tuned | fixed_preprocessing | 0/5 | 0/5 |
| smooth_sparse | mboost_additive | small | boundary_growth | 0/5 | 0/5 |
| smooth_sparse | mboost_additive | tuned | boundary_growth | 0/5 | 0/5 |
| smooth_sparse | mboost_additive | small | patience_60 | 0/5 | 0/5 |
| smooth_sparse | mboost_additive | tuned | patience_60 | 0/5 | 0/5 |
| smooth_sparse | mboost_additive | small | repeated_cv_growth | 0/5 | 0/5 |
| smooth_sparse | mboost_additive | tuned | repeated_cv_growth | 0/5 | 0/5 |
| smooth_sparse | mboost_additive | small | fold_local_fixed | 0/5 | 0/5 |
| smooth_sparse | mboost_additive | tuned | fold_local_fixed | 0/5 | 0/5 |
| smooth_sparse | mboost_additive | small | fold_local_growth | 0/5 | 0/5 |
| smooth_sparse | mboost_additive | tuned | fold_local_growth | 0/5 | 0/5 |
| smooth_sparse | mboost_additive | small | fold_local_cv_growth | 0/5 | 0/5 |
| smooth_sparse | mboost_additive | tuned | fold_local_cv_growth | 0/5 | 0/5 |
| housing_iid | mboost_hybrid | small | fixed_argmin | 1/3 | 0/3 |
| housing_iid | mboost_hybrid | tuned | fixed_argmin | 1/3 | 0/3 |
| housing_iid | mboost_hybrid | small | fixed_preprocessing | 1/3 | 0/3 |
| housing_iid | mboost_hybrid | tuned | fixed_preprocessing | 1/3 | 0/3 |
| housing_iid | mboost_hybrid | small | boundary_growth | 0/3 | 1/3 |
| housing_iid | mboost_hybrid | tuned | boundary_growth | 0/3 | 1/3 |
| housing_iid | mboost_hybrid | small | patience_60 | 0/3 | 1/3 |
| housing_iid | mboost_hybrid | tuned | patience_60 | 0/3 | 1/3 |
| housing_iid | mboost_hybrid | small | repeated_cv_growth | 0/3 | 1/3 |
| housing_iid | mboost_hybrid | tuned | repeated_cv_growth | 0/3 | 0/3 |
| housing_iid | mboost_hybrid | small | fold_local_fixed | 1/3 | 0/3 |
| housing_iid | mboost_hybrid | tuned | fold_local_fixed | 1/3 | 0/3 |
| housing_iid | mboost_hybrid | small | fold_local_growth | 0/3 | 1/3 |
| housing_iid | mboost_hybrid | tuned | fold_local_growth | 0/3 | 1/3 |
| housing_iid | mboost_hybrid | small | fold_local_cv_growth | 0/3 | 2/3 |
| housing_iid | mboost_hybrid | tuned | fold_local_cv_growth | 0/3 | 0/3 |
| housing_iid | mboost_additive | small | fixed_argmin | 3/3 | 0/3 |
| housing_iid | mboost_additive | tuned | fixed_argmin | 2/3 | 0/3 |
| housing_iid | mboost_additive | small | fixed_preprocessing | 3/3 | 0/3 |
| housing_iid | mboost_additive | tuned | fixed_preprocessing | 2/3 | 0/3 |
| housing_iid | mboost_additive | small | boundary_growth | 3/3 | 3/3 |
| housing_iid | mboost_additive | tuned | boundary_growth | 2/3 | 2/3 |
| housing_iid | mboost_additive | small | patience_60 | 3/3 | 3/3 |
| housing_iid | mboost_additive | tuned | patience_60 | 1/3 | 2/3 |
| housing_iid | mboost_additive | small | repeated_cv_growth | 1/3 | 1/3 |
| housing_iid | mboost_additive | tuned | repeated_cv_growth | 0/3 | 1/3 |
| housing_iid | mboost_additive | small | fold_local_fixed | 3/3 | 0/3 |
| housing_iid | mboost_additive | tuned | fold_local_fixed | 2/3 | 0/3 |
| housing_iid | mboost_additive | small | fold_local_growth | 3/3 | 3/3 |
| housing_iid | mboost_additive | tuned | fold_local_growth | 2/3 | 2/3 |
| housing_iid | mboost_additive | small | fold_local_cv_growth | 1/3 | 1/3 |
| housing_iid | mboost_additive | tuned | fold_local_cv_growth | 0/3 | 1/3 |

| panel | seed | dependency design | inner validation |
|---|---|---|---|
| cancer | 0 | iid | 2x repeated 5-fold |
| cancer | 1 | iid | 2x repeated 5-fold |
| cancer | 2 | iid | 2x repeated 5-fold |
| bike_iid | 0 | iid | 2x repeated 5-fold |
| bike_iid | 1 | iid | 2x repeated 5-fold |
| bike_iid | 2 | iid | 2x repeated 5-fold |
| bike_future | 0 | temporal | rolling-origin 10-fold (expanding window); whole time units |
| bike_future | 1 | temporal | rolling-origin 10-fold (expanding window); whole time units |
| bike_future | 2 | temporal | rolling-origin 10-fold (expanding window); whole time units |
| concrete | 0 | iid | 2x repeated 5-fold |
| concrete | 1 | iid | 2x repeated 5-fold |
| concrete | 2 | iid | 2x repeated 5-fold |
| joint_boundary | 0 | iid | 2x repeated 5-fold |
| joint_boundary | 1 | iid | 2x repeated 5-fold |
| joint_boundary | 2 | iid | 2x repeated 5-fold |
| joint_boundary | 3 | iid | 2x repeated 5-fold |
| joint_boundary | 4 | iid | 2x repeated 5-fold |
| smooth_sparse | 0 | iid | 2x repeated 5-fold |
| smooth_sparse | 1 | iid | 2x repeated 5-fold |
| smooth_sparse | 2 | iid | 2x repeated 5-fold |
| smooth_sparse | 3 | iid | 2x repeated 5-fold |
| smooth_sparse | 4 | iid | 2x repeated 5-fold |
| housing_iid | 0 | iid | 2x repeated 5-fold |
| housing_iid | 1 | iid | 2x repeated 5-fold |
| housing_iid | 2 | iid | 2x repeated 5-fold |

## Protocol notes

- All routes score MSE for Gaussian or log loss for Binomial. Fold-local routes use `tune_mstop`'s Rust-backed response path. Contrasts isolate stopping within a preprocessing route.
- Patience is unsupported with fold-local preprocessing by the existing public API. Its native result remains diagnostic; no fold-local patience result or new stopping subsystem is claimed.
- IID panels use 2 × 5 random folds. `bike_future` uses 10 expanding rolling origins over whole dates, with explicit held-out/judge masks. Later unused rows do not enter a fold's score. These origins are not repeated random K-fold partitions. Grouped/spatial designs use whole-group folds when included.
- Every record includes aggregate risk, grid ends, selected-step fold scores and warnings. Fold-local and fixed-budget records also include every fold's full risk curve; native growing `tune_mstop` exposes only the aggregate and selected-step scores.
- Absence of missing cells does not establish fold-local preprocessing. Spline boundaries and factor vocabularies can still depend on held-out rows on the native route.
- The original `benches/verify_selection/`, `benches/path_diagnostics_509.py` and `docs/benchmarks/path_diagnostics_509.json` are present on main. The v1 claim that they were absent from every branch is withdrawn. Their oracle path minima remain exploratory diagnostics.
- No public defaults, tree topology or tree-AIC support changed. Tree-form and adaptive-degrees-of-freedom research remain separate. This ablation does not rerun the competitor shootout or establish the original 4–5% excess-loss target.

