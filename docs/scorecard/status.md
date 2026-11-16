# mboostrs status against the knowledge-discovery scorecard

**Evidence snapshot: 2026-10-12; D11 update: 2026-10-15.** This is a consolidated evidence-led status view for `mboostrs-scorecard/2.1`. D2–D10 retain their existing definitions (including D5 iteration speed, D9 calibration and D10 auditability); D11 adds predictive validity; the former D1 is split into D1a (recovery under a prescribed specification) and D1b (discovery from data alone), per the #511 information contracts. The labels below are qualitative summaries, not canonical numeric scores. Results from older rubrics and named studies remain historical evidence and have not been silently regraded. No cross-study average is meaningful because tasks, versions, outcomes and protocols differ.

| Dimension | Evidence-supported status | Concrete metric or example (source) | Main gap and existing issue |
|---|---|---|---|
| **D1a Relationship recovery (prescribed specification)** | **Unmeasured; mixed proxy** | Hostile-workflow and causal batteries prescribe a specification, so their verdicts are Track A proxies: all arms identified the core verdict in 18/18 hostile runs and causal rerun 6 moved adjustment-set validity to every tested set ([report](../benchmarks/agentic_hostile_workflows_report.md), [comparison](../book/benchmarks/causal_discovery/rerun6/COMPARISON.md)). | No run isolates recovery of the *open* structure from supplied terms, so no D1a score is assigned. Causal rerun 6 had valid adjustment sets in 8/8 tasks (up from 6/8 in rerun 5); the chooser also misses categorical interactions. See [#478](https://github.com/fcela/mboostrs/issues/478), [#493](https://github.com/fcela/mboostrs/issues/493) and [#511](https://github.com/fcela/mboostrs/issues/511). **Rerun limit:** hostile v4 ran before PR #497; its sharp-smooth case was mild (logistic k=6), so it does not verify the steep-transition fix. |
| **D1b Relationship discovery (data alone)** | **Unmeasured** | No recorded battery supplies data, semantics, metadata, design and budget with no formula and scores TPR/FDR by structure type. Causal rerun 6 moved adjustment-set validity to every tested set but still handed the analyst a specification ([comparison](../book/benchmarks/causal_discovery/rerun6/COMPARISON.md)). | Data-alone discovery is not yet measured; "silence" (reporting nothing) has not been tested to ensure it cannot win. Requires a no-formula planted-truth battery under the #511 Track B contract. See [#511](https://github.com/fcela/mboostrs/issues/511) and [#493](https://github.com/fcela/mboostrs/issues/493). |
| **D2 Causal humility and support** | **Mixed** | Hostile v4 reported associations-only language for confounding, explained the marginal-to-conditional reversal (−1.35 to about +1.6), and caught leakage in all three arms ([report](../benchmarks/agentic_hostile_workflows_report.md)). | Rerun 6 had valid adjustment sets in 8/8 tasks and zero false claims on no-effect draws, but its small-sample ATE construction remains unresolved and there is no built-in doubly robust second stage. See [#484](https://github.com/fcela/mboostrs/issues/484) and [#487](https://github.com/fcela/mboostrs/issues/487). |
| **D3 Explanation quality** | **Mixed** | Hostile v4 agents gave decision-useful explanations for missingness, leakage, heteroscedasticity and confounding; the report records correct planted values and explicit limitations ([report](../benchmarks/agentic_hostile_workflows_report.md)). | Existing benchmark evidence is task-specific and does not establish consistent report quality across general user analyses. The hostile battery also showed no correctness gain from the skills arm despite much higher effort. Efficiency friction is tracked in [#492](https://github.com/fcela/mboostrs/issues/492). |
| **D4 Importance quantification** | **Needs attention** | Causal rerun 6 reports ATE RMSE $2,747 and per-applicant 90% interval coverage 0.61; its comparison reports second-stage intervals omit first-stage uncertainty ([comparison](../book/benchmarks/causal_discovery/rerun6/COMPARISON.md)). | Effect magnitudes and intervals are not yet reliably decision-calibrated across studies; coverage is sensitive to the analyst's estimator construction. Causal friction and missing doubly robust inference are tracked in [#443](https://github.com/fcela/mboostrs/issues/443) and [#487](https://github.com/fcela/mboostrs/issues/487). |
| **D5 Iteration speed** | **Unmeasured; cost proxies concerning** | Joint-regularization pilot: all 320 fits completed; median tuned-boosting time ranged 0.009–0.057 s for its four 600-row scenarios ([study](../benchmarks/joint_regularization/README.md)). Hostile v4 skills arm used ~686 fits/~127 min versus ~378/~87 min without skills ([report](../benchmarks/agentic_hostile_workflows_report.md)). | Individual fits can be fast, while end-to-end discovery can consume many fits and minutes with no accuracy gain. The historical [KDD notebook](../book/case_studies/kdd_scorecard_benchmark.ipynb) has a 50k single-fit and 5k ten-refit proxy; the required 50k×20 follow-up session with diagnostics remains unmeasured. Reduce workflow fit count and measure comparable session-level latency ([#503](https://github.com/fcela/mboostrs/issues/503)). See [#492](https://github.com/fcela/mboostrs/issues/492). |
| **D6 Robustness across regimes** | **Needs attention** | Hostile v4 handled missingness, leakage, heteroscedasticity and confounding, but its real-data test RMSEs were 58.51–59.74 and the battery warns of small-sample/test uncertainty ([report](../benchmarks/agentic_hostile_workflows_report.md)). Joint regularization improved sparse signals but degraded dense and null scenarios ([study](../benchmarks/joint_regularization/README.md)). | Performance varies materially by regime; the joint-regularization gate failed, so boosting remains the default. Chooser bike-future had a seed crash and other unresolved panel gaps. See [#462](https://github.com/fcela/mboostrs/issues/462) and [#493](https://github.com/fcela/mboostrs/issues/493). |
| **D7 Human-friendliness** | **Unmeasured; workflow friction observed** | The hostile report documents actionable warnings and workflows, while the chooser and PM2.5 materials preserve tuning logs and friction findings ([report](../benchmarks/agentic_hostile_workflows_report.md), [PM2.5 report](../book/benchmarks/pm25_forecast/runs/mboostrs/REPORT.md)). | PM2.5 required a hand-written rolling validation loop; extrapolation warnings and non-crossing quantile sorting needed extra handling. A grouped-folds demonstration is stale ([#499](https://github.com/fcela/mboostrs/issues/499)); quantile calibration and workflow gaps remain open. |
| **D8 Agent-friendliness** | **Mixed** | Under its local tool-fluency rubric, hostile v4 scored the skills arm 4 and the no-skills mboostrs arm 5; the no-skills arm reached the same planted truths with fewer reported fits ([report](../benchmarks/agentic_hostile_workflows_report.md)). | Agent workflows can be effective but are not reliably economical; time-to-answer is a material weakness. The hostile battery predates PR #497 and cannot validate that fix; efficiency is tracked in [#492](https://github.com/fcela/mboostrs/issues/492). |
| **D9 Uncertainty calibration** | **Needs attention** | In the PM2.5 forecasting study, nominal 80% prediction bands covered 0.74 overall and 0.68 in winter ([findings](../book/benchmarks/pm25_forecast_competition.ipynb)). The first claims study recorded approximately 0.90 prediction coverage on completed full-size runs, but failures lowered its all-run assessment ([study](../book/benchmarks/claims_discovery.ipynb)). | Predictive calibration varies with task, subgroup and deployment shift. Effect-interval undercoverage is listed under D4. Evaluate prediction coverage together with width, failures and dependence-aware uncertainty; see [#136](https://github.com/fcela/mboostrs/issues/136). |
| **D10 Reproducibility and auditability** | **Strong** | Rerun 6 includes a comparison, friction log, run records, findings and predictions; the chooser and joint pilot preserve protocol/results, versions, hashes and seeds ([rerun 6](../book/benchmarks/causal_discovery/rerun6/COMPARISON.md), [chooser results](../benchmarks/chooser_shootout/README.md), [joint protocol](../benchmarks/joint_regularization/PROTOCOL.md)). The hostile v4 reproducibility evidence was graded under its local D5, not canonical D10. | Audit evidence is strongest in these named research batteries; independent one-command reproduction of a complete end-user discovery report has not been demonstrated. Preserve run provenance when refreshing results. |
| **D11 Predictive validity** | **Mixed; focused two-track repair verified** | The paired encoding study retains two shifted-budget losses (+10.99%, SE 1.33 points, known; +14.12%, SE 2.01, discovery) while eight of ten cells meet 1%; the longer-path diagnosis leaves the original budget unchanged ([encoding study](../book/benchmarks/encoding_tracks.ipynb), [remaining gap #573](https://github.com/fcela/mboostrs/issues/573)).  A new frozen two-track study has all six smooth/interaction/factor cells within 1% of the strongest complete named competitor on independent draws. Factor discovery changes from +75.14% (SE 4.14 points) to −8.23% (SE 2.25) after proposed repairs; every cell scores D11=3 under its frozen Ridge-reference rule. Other dimensions stay unmeasured in that study, with no aggregate ([notebook](../book/benchmarks/predictive_tracks.ipynb), [profiles](../book/benchmarks/predictive_tracks/v2/after/scorecard.json)). These are measured proposals under review, not claims about an installed release.  Chooser rerun (artifact predates PR #502 in-flight factor/degenerate fixes; no post-fix metric is claimed; the original probe also evaluated only boosting) had median excess loss +12.7% versus the realized best test-scored competitor (a post-hoc oracle comparison), improving cancer from +104.0% to +7.6% and winning/tieing smooth panels ([chooser study](../benchmarks/chooser_shootout/README.md)). On PM2.5 it achieved 2014 RMSE(log1p) 0.6604 versus the prior submission's 0.6615, but LightGBM/CatBoost scored 0.6213/0.6181 ([PM2.5 report](../book/benchmarks/pm25_forecast/runs/mboostrs/REPORT.md)). | Prediction is competitive on some structures but has substantial gaps on difficult panels (bike-future +241.6%, joint-boundary +249.1%; chooser study) and time-series baselines. Numeric-only tree rung, tensor crash and shared budget/depth are recorded in [#493](https://github.com/fcela/mboostrs/issues/493); a validation tolerance issue is [#500](https://github.com/fcela/mboostrs/issues/500). |

## Current priorities

1. **Predictive validity:** resolve chooser gaps and evaluate on new outer samples and generator families; do not turn the joint-regularization pilot into an automatic policy before it passes its across-regime gate ([#493](https://github.com/fcela/mboostrs/issues/493), [#462](https://github.com/fcela/mboostrs/issues/462)).
2. **Effects and calibration:** improve average-effect inference and interval coverage, including time-ordered quantile calibration; retain the observed 0.75 ATE coverage and 0.61 per-applicant coverage as evidence until a relevant rerun measures a change ([#443](https://github.com/fcela/mboostrs/issues/443)).
3. **Workflow efficiency:** shorten agent sessions and reduce redundant fits without losing checks; hostile v4 found equal correctness at materially higher cost for the skills arm ([#492](https://github.com/fcela/mboostrs/issues/492)).

## Predictive refresh (2026-10-14)

The [complete predictive refresh](../book/benchmarks/competitive_refresh.ipynb)
measures all thirteen frozen predictive panels after the pair-form replay
correction in MR #537. The small tier completes 37 of 45 jobs and meets the
5% competitor-loss target on four panels. Six housing exceptions and two
superconductivity timeouts remain in the denominator. Historical comparators
retain their original identities; task-level D11 assessments compare with the
prespecified same-tier Ridge/logistic reference and are not averaged into a
canonical grade. This focused update does not regrade the older evidence above.

The separate [single-holdout repair in MR #540](https://github.com/fcela/mboostrs/pull/540)
resolves all six housing
exceptions in its frozen rerun. Spatial-housing MSE is 2.79% below the best
complete recorded comparator; IID-housing MSE remains 6.42% above it. The
pre-repair refresh preserves its original failed rows. Available joint-tree
capacity closes much of the measured non-coordinate gap on three completed
panels; its search restriction and confirmation requirements are recorded in
[#539](https://github.com/fcela/mboostrs/issues/539). The full causal,
distributional, calibration and workflow targets remain unverified by these
predictive panels.

The complete cancer panel fails the prespecified simple-reference comparison
and receives task-level D11=1. Its +127.41% outer log-loss gap persists despite
low reported validation minima; the separate instability investigation is
[#541](https://github.com/fcela/mboostrs/issues/541). A linear counterfactual
improves outer loss while losing on the reused validation criterion, so a
larger candidate catalogue alone is not an established remedy.

## Evidence limits

- Each result describes its named study, version, task battery, split and machine. These studies are not pooled into a numeric score or weighted aggregate.
- Historical rubric numbers are retained only with their original study and version; this page does not map them onto scorecard 2.1.
- A merged fix is not evidence of a corrected outcome until a relevant rerun measures it. In particular, hostile v4 predates PR #497; the benchmark's mild transition does not test the steep regime that motivated that fix. Causal rerun 6 measures mboostrs 0.1.3 on main 26fed4d; its own comparison describes the interval construction and 540 unranked applicants as workflow limitations. The chooser results also predate PR #502 in-flight factor/degenerate fixes; no post-fix performance change is established here.
- The recorded status is qualitative and evidence-supported, not a pre-registered independent scorecard evaluation. D11 is a distinct predictive-validity dimension; D1a and D1b split the former D1 discovery construct by information contract (#511) without changing what discovery means. Hostile and causal evidence predates this split and is treated as Track A proxy evidence only.


## Sharp interaction diagnosis (2026-10-17)

The [interaction-capacity study](../book/benchmarks/interaction_capacity.ipynb)
has all ten final process/track point gaps within 1% of the retrospective
strongest complete named competitor, including the post hoc audit that admits
Ridge. Axis-boundary discovery changes from +329.33% (paired SE 22.93 percentage
points) to −5.72% (SE 9.87); shifted-boundary discovery changes from +425.39% (SE
37.50) to −17.77% (SE 2.80). The former remains unresolved at two SE. Known-specification
gaps on those processes are −1.95% (SE 11.43) and −20.01% (SE 2.20). The phases retain
272/280 completed baseline jobs, all 280 tree-only jobs and all 280 final jobs;
the eight null-model errors remain in the baseline denominator.

The diagnosis separates missing tree-form comparisons, weaker main effects
masked before interactions, and an intercept-only response-scoring defect.
Two separately frozen 36-draw control batches support the proposed workflow
changes; numerical fits and paired risk calculations use existing Rust APIs.
Proposed fixes are in [#614](https://github.com/fcela/mboostrs/pull/614),
[#618](https://github.com/fcela/mboostrs/pull/618) and
[#621](https://github.com/fcela/mboostrs/pull/621); a baseline guide repair is
[#617](https://github.com/fcela/mboostrs/pull/617). These are review proposals,
not release claims. Only D11 is graded under frozen per-cell Ridge rules;
eleven other dimensions remain unmeasured, with no aggregate. The post-repair
primary results reuse the diagnosed processes and are not independent
best-in-class confirmation. Tree-effect inference, higher-order structures,
forecasting and classification gaps remain outside this study.
