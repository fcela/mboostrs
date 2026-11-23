(record-competitive-shootout)=
# Competitive shootout, capacity selection, and predictive refresh

This record summarizes the protocols and main outcomes of the multi-library shootout ([](../../benchmarks/competitive_shootout.ipynb)), the capacity-selection follow-up ([](../../benchmarks/chooser_shootout.ipynb)), and the fixed-tier predictive refresh ([](../../benchmarks/competitive_refresh.ipynb)).

## Multi-library shootout

**Design.** Eight model families were evaluated on a single protocol spanning 720 predictive runs (10 real-data panels and 3 synthetic panels with known response surfaces, 3 outer splits each) and 140 causal-effect runs (4 designs × 7 draws). Real panels covered California housing (random and spatially blocked splits), hourly bike rentals (random and future-time splits), diamonds, protein structure, superconductivity, concrete strength, German credit, and Wisconsin breast cancer. Two engineering thresholds framed the results: *competitive* ($\le 5\%$ excess test loss over the best complete non-mboostrs competitor) and *material gap* ($> 10\%$ excess loss). Because the best competitor is identified retrospectively on test loss, these ratios represent an oracle-competitor gap rather than a prospective baseline.

**Outcomes.** Across the ten real-data panels in the tuned tier, median excess test loss over the retrospective best competitor was 0.1% for CatBoost, 8.4% for `mboostrs` with an added tree term, 38.3% for automatic specification search, and 46.5% for a purely additive model. Adding an unconditional joint tree reduced error on high-interaction panels but severely degraded sparse or linear-dominant panels (e.g., turning the breast-cancer panel from ahead of competitors to +109% excess log loss).

## Nested capacity selection (`select_capacity`)

**Design.** To avoid the penalty of always-on joint trees, `select_capacity` evaluates three nested rungs on shared out-of-fold splits—base additive formula, base plus screened tensor products, and base plus a joint tree—and retains the simplest rung within two paired standard errors of the best. Each rung selects its own stopping iteration out of fold, and tensor pairs require both columns to have more than two distinct values.

**Outcomes.** Evaluated on the same shootout panels against the recorded tuned-tier competitors, `select_capacity` reduced median excess loss to +12.7% (from +46.5% additive) and cut breast-cancer excess loss from +104.0% to +7.6% by selecting the base rung. It preserved smooth specifications where optimal (`smooth_sparse` −0.9%, `correlated_proxies` −3.5%) and improved `concrete` from +19.6% to +12.7%. Large gaps remained on future-time bike rentals (+241.6%) and sharp joint boundaries (+249.1%).

## Fixed-tier predictive refresh

**Design.** A refresh benchmark re-evaluated all 13 shootout predictive panels under the small-tier search budget (45 planned jobs across 3 splits per panel) using identical partition digests against the archived 720-job competitor results.

**Outcomes.** Thirty-seven of 45 jobs completed (6 housing runs raised exceptions on single-holdout splits; 2 superconductivity jobs hit timeouts). All failures remain in the denominator. Four of 13 panels met the 5% target against the strongest recorded competitor (`credit`, `smooth_sparse`, `joint_boundary`, `correlated_proxies`). Complete-panel excess losses remained high on `bike_future` (+49.85%), `bike_iid` (+44.00%), `diamonds` (+20.50%), `protein` (+41.45%), `concrete` (+57.74%), and `cancer` (+127.41%).

Exploratory counterfactuals on three high-gap panels (`protein`, `bike_iid`, `diamonds`) replacing the additive model with a depth-selected joint tree ($\nu = 0.05$, cap 1,200) reduced competitor-relative gaps to +8.94%, +7.19%, and −0.66%, respectively, confirming that representation capacity—not boosting optimization—drives most of the gap on non-additive tables. On breast cancer, a prespecified all-linear specification beat the searched model on outer test loss while losing on the reused validation split, motivating the separate stopping and holdout studies in [](stopping_rules.md) and [](adoption_rules.md).

**Audit artifacts.** Protocols, full per-job JSON artifacts, and detailed tables are archived in `docs/benchmarks/competitive_shootout_protocol.md`, `docs/benchmarks/competitive_shootout_report.md`, `docs/benchmarks/chooser_shootout/`, and `docs/book/benchmarks/competitive_refresh/`.
