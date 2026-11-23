# Knowledge Discovery Scorecard

**Canonical identifier: `mboostrs-scorecard/2.1`. Effective: 2026-10-12.**
This is the sole scorecard for new project evaluations. It defines twelve
stable dimensions: D1a, D1b, and D2–D11. D2–D10 retain their original
meanings; D11 explicitly measures predictive validity. D5 means iteration
speed, D9 means uncertainty calibration, and D10 means reproducibility. Those
IDs must not be repurposed. The former D1 is split into D1a and D1b because the
amount of information handed to the analyst changes what counts as discovery;
see the information contracts below and [migration.md](migration.md).

The [mboostrs status page](status.md) is the consolidated view of our recorded
strengths, gaps and priorities. The [migration crosswalk](migration.md) explains
historical rubrics. The [working paper](paper.md) supplies background, not a
second set of grading rules.

## Evidence and reporting contract

- New protocols cite this identifier, the scorecard content SHA256, and the
  exact dimension IDs and names. Freeze the task battery, scoring thresholds,
  comparator settings and budget before running it.
- A study may measure a subset of dimensions. It may define task-specific
  measurements and preregistered thresholds within these definitions; it
  cannot rename a dimension or substitute another construct under its ID.
- Distinguish `scored`, `unmeasured` and `not_applicable`. Missing evidence is
  not a score of 1 or 0. Numerical scores require traceable supporting evidence
  and a written anchor justification; qualitative status is a separate field.
- Preserve the original rubric and grades for historical studies. A crosswalk
  locates relevant evidence; it does not convert old numerical grades into
  current grades. Regrading requires an explicit new assessment.
- Report the full profile and its evidence coverage. A weighted aggregate is
  permitted only for all twelve dimensions on a common, prespecified battery
  with the same version and operating mode. Do not average mixed studies or
  renormalize an incomplete profile into a seemingly complete score.
- Separate library capabilities, the executed pipeline, agent effort and
  numerical outcomes. A merged fix does not improve the measured status until
  an appropriate rerun supplies evidence.
- Report failed runs and selection/search scope. CV adoption heuristics are
  predictive decisions, not calibrated hypothesis tests. A realized best
  competitor selected by test loss is an oracle comparison, not a prospectively
  chosen baseline.

## Information contracts for discovery

D1a and D1b evaluate the same construct — finding true structure and avoiding
false structure — under different information contracts so that a library is
neither credited for structure it was handed nor penalized for structure it was
never asked to find.

- **Track A — prescribed specification (D1a).** The analyst receives partial
  DGP knowledge: a rough formula naming family/link, candidate predictors and
  term types, but not the true structure. Only structure the formula left open
  is scored; supplied terms earn no discovery credit.
- **Partial-knowledge.** A middle track where some components are fixed and
  others are open. Credit the open components only, and report which components
  were prescribed.
- **Track B — data alone (D1b).** The analyst receives the data, response
  semantics, metadata, design and budget, but no formula. All recovered
  structure is scored.

Accounting rules:

- **Real-data prespecified ≠ oracle.** On planted-truth benchmarks the
  generator's structure is an oracle. Real data has no oracle, so findings are
  compared against a *prespecified* reference specification, never a post-hoc
  best fit; a realized best competitor selected by test loss is an oracle
  comparison, not a baseline.
- **Paired excess/gap formulas.** Report paired per-task differences rather
  than pooled means. For losses `L`, a prespecified reference `r` and the best
  oracle `L*`: `excess = L_lib − L_ref`, `gap = L_lib − L*`, and the normalized
  `gap_ratio = (L_lib − L_ref) / (L* − L_ref)` when `L* ≠ L_ref`. State the
  reference before running and report uncertainty across tasks/seeds.
- **Reporting.** A discovery score is reported as TPR and FDR by structure
  type, labelled with the track and the prescribed terms; recovery of open
  structure and prediction of held-out observations are separate claims (D1a/D1b
  vs D11).

**Purpose.** Grade data modeling libraries on *knowledge discovery effectiveness* — for fully automated (AI agent) and semi-automated (human analyst) data modeling. Companion to `paper.md`, which justifies every dimension from the literature (or marks it as new).

**How to use.**
1. Define the task battery first (planted-truth DGPs, benchmark datasets, report tasks) — see paper §6.3.
2. Score each dimension 1–5 using the anchored rubrics below. Half-points allowed with written justification.
3. Record evidence per dimension. A score without evidence is a guess.
4. Report the full profile, not a single number. Apply mode weights from the table at the end if an aggregate is needed.

**Legend.** [L] = grounded in cited literature. [N] = new dimension; absent from pre-AI literature, motivated by the agentic discovery problem.

---

## D1a. Relationship recovery under a prescribed specification [L]

*Recovering structure a rough formula leaves open; supplied terms earn no discovery credit.*

| Score | Anchor |
|---|---|
| 1 | Fits the prescribed formula literally; no search over the open structure |
| 2 | Ad-hoc residual scanning (variable importance) with no false-discovery control |
| 3 | Structured search over the open terms with explicit evidence and decision rules; error-control claims state their assumptions |
| 4 | Validated on planted-truth benchmarks under a frozen prescribed formula; false-discovery behavior on the open structure characterized |
| 5 | All of 4, plus higher-order open structure (interactions, distributional effects) with published recovery rates |

**Measure.** Freeze the prescribed formula (family/link, candidate predictors, term types). Planted-truth suite adds structure the formula omits (interactions, nonlinearities, heteroscedasticity, pure noise). Report TPR and FDR per structure type for the *open* terms only. List the prescribed terms and award no discovery credit for recovering them.
**Evidence:**

## D1b. Relationship discovery from data alone [L]

*Finding true structure from data alone and controlling false discoveries on pure-noise tasks.*

| Score | Anchor |
|---|---|
| 1 | No systematic discovery; relationships found only by manual specification |
| 2 | Ad-hoc aids (variable importance) with no false-discovery control |
| 3 | Structured screening with explicit evidence and decision rules; any error-control claims state their assumptions |
| 4 | Screening + adoption validated on planted-truth benchmarks; false-discovery behavior characterized |
| 5 | All of 4, plus higher-order structure (interactions, distributional effects) with published recovery rates |

**Measure.** Supply no formula: data, response semantics, metadata, design and budget only. Score TPR and FDR by structure type against planted truth. Silence must not win: a library that reports nothing cannot outscore one that reports true structure, and must also show zero false structure on pure-noise tasks. Report the discovery budget and the search scope.
**Evidence:**

## D2. Causal humility and support [L]

*Restricting causal claims to designs and assumptions that support them.*

| Score | Anchor |
|---|---|
| 1 | Encourages causal reading of predictive features ("top drivers," no caveats) |
| 2 | Neutral — neither encourages nor discourages causal interpretation |
| 3 | Docs scope findings as associational; supports confounder-adjustment workflows |
| 4 | All of 3, plus built-in sensitivity/robustness checks for key findings |
| 5 | All of 4, plus experimental-design support integrated with modeling |

**Measure.** Documentation audit + workflow checklist.
**Evidence:**

## D3. Explanation quality [L/N]

*Composing faithful, complete, decision-useful narratives from findings.*

| Score | Anchor |
|---|---|
| 1 | No explanation outputs; analyst invents narrative from raw coefficients |
| 2 | Per-prediction explanations only (SHAP-style); no dataset-level story |
| 3 | Dataset-level summaries (effect plots, importance rankings) faithful to the fitted model |
| 4 | Ranked, evidence-graded findings with uncertainty — reportable nearly verbatim |
| 5 | Findings compose into a coherent report: what we know / don't know / check next |

**Measure.** Human-grounded task: analysts/agents write reports from library outputs on a planted-truth dataset; grade faithfulness and completeness against the key. Record whether the run was Track A (prescribed specification) or Track B (data alone) and report explanation credit for discovered versus prescribed terms separately.
**Evidence:**

## D4. Importance quantification [L]

*Effect magnitudes with calibrated uncertainty.*

| Score | Anchor |
|---|---|
| 1 | Point estimates only; no uncertainty |
| 2 | Heuristic scores (gain, permutation) with no sampling interpretation |
| 3 | Confidence/credible intervals under stated assumptions |
| 4 | Coverage validated empirically (90% interval covers ~90%) |
| 5 | Decision-relevant magnitudes ("changes prediction by X ± Y on the outcome scale") |

**Measure.** Coverage calibration on planted-truth DGPs; interval width at matched coverage. Under the Track A contract, report intervals for discovered (open) effects separately from prescribed effects.
**Evidence:**

## D5. Experimental iteration speed [N]

*Wall-clock time from "what if we tried X?" to an answer.*

| Score | Anchor |
|---|---|
| 1 | Single fits take minutes+ on medium data; iteration impractical |
| 2 | Tens of seconds per fit; a handful of iterations per session |
| 3 | Seconds per fit; dozens of iterations; cheap diagnostics |
| 4 | Sub-second refits; batched counterfactuals; interactive loop |
| 5 | Incremental/online updating — follow-ups don't require full refits |

**Measure.** Standardized "analyst session": 10 specified follow-up fits on 50k×20 data, incl. diagnostics. Report median and p90 wall-clock.
**Evidence:**

## D6. Robustness across conditions [L]

*Graceful behavior on small-n, large-n, messy, and shifted data.*

| Score | Anchor |
|---|---|
| 1 | Requires clean complete numeric matrices; fails/silently misbehaves otherwise |
| 2 | Basic missingness/mixed-type handling, documented |
| 3 | Discovery quality measured across small-n and messy benchmarks; graceful, documented degradation |
| 4 | All of 3, plus large-n scalability demonstrated |
| 5 | All of 4, plus shift/stress diagnostics that warn when findings may not transfer |

**Measure.** Fixed battery: small-n (n=200), messy (15% MCAR + outliers + mixed types), large-n (5M rows). Report D1a and D1b recovery + runtime per condition, under both information contracts where feasible.
**Evidence:**

## D7. Human-friendliness [L]

*Learnability, diagnosability, and trust for a competent analyst.*

| Score | Anchor |
|---|---|
| 1 | Cryptic API, undocumented failure modes, misleading defaults |
| 2 | Usable with effort; discovery workflows unguided |
| 3 | Coherent API, guided workflows, informative errors |
| 4 | Executable case studies for main discovery patterns; diagnostics point at fixes |
| 5 | Evidence the docs work (user studies or issue-tracker analysis) |

**Measure.** Task-based study: time-to-first-discovery, error-recovery rate (or issue-tracker mining as proxy). Run the task under both the Track A prescribed-specification and Track B data-alone contracts and report separately, since guidance needs differ.
**Evidence:**

## D8. Agent-friendliness [N]

*Autonomous completion of the discovery loop through the API.*

| Score | Anchor |
|---|---|
| 1 | Every step needs human judgment ("look at the plot and decide") |
| 2 | Programmatic API, but unstructured outputs or uninformative errors |
| 3 | Structured outputs (typed findings, machine-readable diagnostics); actionable error codes |
| 4 | Documented agent contract (assumptions, checks, preflight/validation hooks) |
| 5 | Demonstrated: agent completes discovery benchmarks end-to-end, zero human assists |

**Measure.** Frozen-prompt LLM agent performs discovery on planted-truth datasets using only the library; grade D1a/D1b recovery and D3 report quality. Report Track A and Track B runs separately, with the prescribed terms identified for Track A.
**Evidence:**

## D9. Uncertainty honesty (calibration) [L]

*Stated uncertainty matches reality; widens when data warrants.*

| Score | Anchor |
|---|---|
| 1 | No uncertainty quantification |
| 2 | Uncertainty reported but uncalibrated |
| 3 | Coverage validated on standard DGPs; approximately nominal |
| 4 | Adaptive: widens under heteroscedasticity, small-n, extrapolation — demonstrated |
| 5 | Diagnostics warn when the uncertainty itself is unreliable |

**Measure.** Coverage experiments on planted-truth DGPs incl. heteroscedastic and small-n variants; empirical vs. nominal coverage. Attribute coverage to the track that produced the estimate (D1a prescribed vs D1b discovered) so calibration is not credited to supplied structure.
**Evidence:**

## D10. Reproducibility and auditability [L]

*Third parties can reproduce every number and audit every decision.*

| Score | Anchor |
|---|---|
| 1 | Undocumented randomness or hidden state |
| 2 | Seeds fixable, but decision trail not recorded |
| 3 | Full decision log (candidates, adoptions/rejections with reasons, seeds) |
| 4 | One-command reproduction of the entire discovery report |
| 5 | Artifact versioning (data, code, environment hashes alongside findings) |

**Measure.** Independent reproduction from the report alone; grade numerical agreement and trail completeness.
**Evidence:**

---

## D11. Predictive validity [L]

*Useful predictions on observations not used to construct or select the model.*

| Score | Anchor |
|---|---|
| 1 | Measured leakage, invalid prediction timing, or failure to beat a prespecified simple baseline |
| 2 | Some predictive utility, but evaluation or comparison is too limited to support generalization claims |
| 3 | Honest outer validation shows useful improvement over a prespecified baseline, with uncertainty and failures reported |
| 4 | Competitive predictive performance demonstrated across prespecified task families under matched tuning and compute budgets |
| 5 | Level 4 confirmed on held-back task families and relevant deployment splits, with material losses and transfer limits characterized |

**Measure.** Freeze the estimand, prediction-time information, loss and reference
population. Use untouched outer test data or nested validation; put all learned
preprocessing, screening and tuning inside the permitted training data. Use
splits appropriate to dependence, timing and deployment. Report per-task loss,
paired differences with uncertainty, comparator selection and budgets, sample
sizes, failures and test-use history. On real data the prespecified reference is
not an oracle; use the paired excess/gap accounting defined above. Keep
classification, mean prediction, distribution prediction and causal-effect
prediction targets explicit; their scores cannot be pooled indiscriminately. Use
D9 for prediction-interval coverage and D4 for effect-interval coverage in
consolidated reporting; record any broader historical use explicitly.

**Evidence:**

## Mode weights

| Dimension | Fully automated | Semi-automated |
|---|---|---|
| D1a Relationship recovery (prescribed) | 2× | 2× |
| D1b Relationship discovery (data alone) | 2× | 2× |
| D2 Causal humility | 1× | 1× |
| D3 Explanation / story | 2× | 1× |
| D4 Importance quantification | 1× | 2× |
| D5 Iteration speed | 2× | 1× |
| D6 Robustness | 1× | 1× |
| D7 Human-friendliness | 0.5× | 2× |
| D8 Agent-friendliness | 2× | 0.5× |
| D9 Uncertainty honesty | 2× | 1× |
| D10 Reproducibility | 1× | 1× |
| D11 Predictive validity | 2× | 2× |

## Fairness rules

1. Same tasks, same budgets, same agent prompts for every library.
2. Use an idiomatic workflow for each library, with tuning effort logged.
3. Pre-register the task battery before scoring.
4. Always publish per-dimension scores; never a lone aggregate.

## Score sheet

| Dim | Library A | Library B | Library C | Evidence refs |
|---|---|---|---|---|
| D1a | | | | |
| D1b | | | | |
| D2 | | | | |
| D3 | | | | |
| D4 | | | | |
| D5 | | | | |
| D6 | | | | |
| D7 | | | | |
| D8 | | | | |
| D9 | | | | |
| D10 | | | | |
| D11 | | | | |
| **Weighted (auto)** | | | | |
| **Weighted (semi)** | | | | |
