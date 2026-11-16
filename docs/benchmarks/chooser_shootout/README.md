# Chooser on the shootout battery

The frozen competitive-shootout panels rerun with `select_capacity` (PR
#479) choosing the capacity, scored against the recorded tuned-tier
competitor losses. The full method, deviations and per-job rows are in
`results.json`; the probe is `benches/probe_chooser_shootout.py`, and the
sibling probe `benches/probe_pm25_chooser.py` runs the same question on the
PM2.5 forecast competition. The findings are catalogued with their issue
numbers in the book's benchmark-findings appendix.

## Result (mean excess loss vs the best recorded competitor, tuned tier)

| panel | tuned hybrid | chooser | chooser's rung |
|---|---:|---:|---|
| cancer | +104.0% | **+7.6%** | base |
| smooth_sparse | +5.6% | **−0.9%** | base |
| correlated_proxies | +2.7% | **−3.5%** | base |
| concrete | +19.6% | **+12.7%** | +tree |
| credit | +1.3% | +4.2% | base |
| housing_spatial | +5.8% | +7.9% | +tree |
| superconductivity | +7.7% | +10.6% | +tree |
| protein | +8.9% | +17.1% | +tree |
| housing_iid | +7.9% | +17.2% | +tensors/+tree |
| diamonds | +3.1% | +39.8% | +tensors/+tree |
| joint_boundary | +65.5% | +249.1% | +tree/+tensors |
| bike_iid | +11.3% | +294.7% | base |
| bike_future | +70.0% | +241.6% | +tree (one seed crashed) |
| **median** | **+7.9%** | **+12.7%** | |

The chooser reduces the cancer excess loss from +104% to +7.6%, and wins or
ties every smooth-dominated panel. The remaining gaps trace to three causes
filed as #493: the tree
rung reads only numeric columns (the bike panels' hour-by-working-day
interaction is categorical), a screened pair on a two-valued column crashes
the tensor construction (bike_future seed 1), and one shared step budget
with a fixed tree depth undertrains richer rungs.

## PM2.5

The chooser kept the analyst's base formula (all three rungs within noise
on rolling-origin folds: 0.2511 / 0.2507 / 0.2518) and finished at 2014
RMSE(log1p) 0.6604 against the 0.6615 submission, the 0.6360 manual
reflective loop, LightGBM's 0.6213 and CatBoost's 0.6181. The manual loop's
gains came from lags, which are new features. The chooser did not add capacity
when it did not improve risk on time-ordered folds. The tolerance's scale on
few rolling folds is its own
finding: #500.

Run on one shared machine (Apple M4, release extension); losses are
accuracy measurements and unaffected by load, timings in the artifacts are
descriptive.
