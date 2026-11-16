# Grading Knowledge Discovery: A Literature-Grounded Scorecard for Evaluating Data Modeling Libraries

## Working paper — DRAFT (research in progress)

**Purpose.** This paper develops a literature-grounded framework for evaluating how well data-modeling libraries support knowledge discovery by human analysts and autonomous agents. It surveys relevant research and proposes a grading scorecard applicable to mboostrs and alternative libraries, including scikit-learn, XGBoost, LightGBM, CatBoost and Explainable Boosting Machines.

**Scope.** Much of the formal literature on automated modeling predates LLM agents. Research on causality and interpretability remains relevant, while agent usability, follow-up iteration speed and multi-finding explanations receive less direct coverage. The paper identifies which dimensions are grounded in prior literature and which extend beyond it. Gaps in earlier research are treated as subjects for evaluation.

---

## 1. Knowledge discovery

The term comes from Fayyad, Piatetsky-Shapiro & Smyth (1996), "From Data Mining to Knowledge Discovery in Databases" (*AI Magazine* 17(3):37–54), which defined KDD as "the overall process of discovering useful knowledge from data," with data mining as one step inside it — alongside selection, preprocessing, transformation, and *interpretation/evaluation* of results. Their central warning is that blind application of mining algorithms ("data dredging") can produce meaningless patterns; evaluation and interpretation are part of the discovery process.

For this scorecard, **knowledge discovery** means starting from a dataset and a question, or from a dataset alone, and producing *true, communicable, decision-relevant* statements about how the world works, with quantified uncertainty. Predictive accuracy is one part of this assessment. A model may predict well without identifying the underlying relationships, or report spurious patterns from overfitting. The scorecard grades discovery as well as prediction.

Two operating modes matter, and the scorecard is designed to serve both:

- **Fully automated discovery**: an AI agent drives the analysis loop, selecting a model class, specifying the formula, diagnosing problems and reporting findings without human input.
- **Semi-automated discovery**: a human analyst works with the library to test hypotheses, visualize relationships and interpret findings.

These modes share most dimensions but weight them differently; the current dimensions and weights are defined in the [canonical scorecard, version 2.1](scorecard.md).

---

## 2. Literature review

### 2.1 The KDD process and interestingness

Fayyad et al. (1996) established that discovery is a *process*, not an algorithm. The CRISP-DM methodology (Chapman et al., 2000) later operationalized this for industry. For the scorecard, the key takeaway is that a library's value lies in how well it supports the *whole loop* — not just the fitting step.

The pattern-mining literature developed "interestingness measures" (Geng & Hamilton, 2006, survey *Interestingness Measures for Data Mining: A Survey*, ACM Computing Surveys) to rank discovered patterns by criteria beyond raw statistical strength: novelty, utility, actionability, and simplicity among them. This directly informs the scorecard's treatment of *finding ranking* — a discovery library must not just find candidate relationships but order them so the true ones surface.

### 2.2 Evaluating interpretability

Doshi-Velez & Kim (2017, arXiv:1702.08608) proposed the foundational taxonomy for evaluating interpretability rigorously, with three levels:

1. **Application-grounded**: real humans, real tasks — does the explanation help a domain expert do their job?
2. **Human-grounded**: real humans, simplified tasks — can people predict the model's behavior, or choose the better of two explanations?
3. **Functionally-grounded**: no humans, proxy tasks — e.g., sparsity, fidelity of a surrogate.

Murdoch et al. (2019, *PNAS* 116(44):22071–22080) added the PDR framework — **predictive accuracy, descriptive accuracy, relevancy** — and emphasized that relevancy is always relative to an audience. For agentic discovery, the "audience" is partly the agent itself, which reframes relevancy in a way the literature didn't anticipate (§4).

Rudin (2019, *Nature Machine Intelligence* 1:206–215) argued that for high-stakes decisions, post-hoc explanations of black boxes should be replaced by inherently interpretable models — because explanations can be unfaithful. This is directly relevant: a discovery scorecard should reward models whose *stated* relationships are the *actual* relationships (faithfulness), not plausible-sounding rationalizations.

### 2.3 Causal discovery vs. pattern finding

The causal discovery literature (Spirtes, Glymour & Scheines, *Causation, Prediction, and Search*; Peters, Janzing & Schölkopf, 2017, *Elements of Causal Inference*) distinguishes between finding *associations* and finding *causal structure*. Constraint-based methods (PC, FCI) and score-based methods (GES) recover Markov equivalence classes under assumptions (causal sufficiency, faithfulness) that are routinely violated in practice.

The scorecard does not assume that general-purpose modeling libraries can reliably discover causal structure from observational data. It evaluates whether a library avoids presenting associations as causes, supports experimental and quasi-experimental designs, and quantifies uncertainty. D2 therefore grades causal scope and support rather than causal discovery itself.

### 2.4 AutoML evaluation — and its pre-LLM limits

The AutoML literature (benchmarks like OpenML, AMLB) evaluates primarily on predictive performance under time budgets. Those measures address AutoML's stated goal, while knowledge-discovery evaluation also needs to address:

- whether a library's API is navigable by an autonomous agent,
- whether its outputs are structured for machine consumption,
- how quickly an agent can iterate on follow-up questions,
- or whether its explanations can be combined into a report with supported findings.

These questions are central to agent-assisted discovery. The scorecard includes them as dimensions marked "beyond literature" (§4).

---

## 3. Scorecard design principles

1. **Discovery, not prediction.** Predictive accuracy appears (a useless model discovers nothing trustworthy), but it is one dimension among many, and not the most heavily weighted.
2. **Literature basis.** Each dimension is tagged [L] when grounded in the cited literature or [N] when proposed for aspects of agent-assisted discovery not covered by that review.
3. **Measurable.** Every dimension has a concrete measurement protocol. Where measurement requires human judgment, the rubric anchors it.
4. **Comparative.** The scorecard is designed to be applied to multiple libraries on the same tasks, producing comparable profiles rather than single numbers.
5. **No mboostrs bias.** Dimensions were selected from literature + the agentic problem structure, not from mboostrs's feature list. Where mboostrs is strong (e.g., distributional modeling), the dimension must still be one the literature independently motivates.

---

## 4. The scorecard dimensions

The current dimension names, definitions, anchored grades, and measurement
protocols are maintained in the [canonical scorecard, version 2.1](scorecard.md)
(identifier `mboostrs-scorecard/2.1`). It splits the original D1 into D1a and
D1b, retains D2–D10 from the original instrument, and adds D11, predictive
validity. This paper does not maintain a
second set of normative anchors; the literature discussion below explains the
reasons for the dimensions and their historical development.

## 5. Dimensions beyond prior literature

The original ten-dimension instrument marked D5, D8, and the narrative-composition half of D3 as having no direct literature precedent. These dimensions address aspects of agent-assisted discovery that were not covered by the literature reviewed here:

- **Agent-friendliness (D8)** presupposes autonomous agents as API consumers. The API-usability studies reviewed here primarily assume a human reader. The assessment includes structured findings, machine-readable diagnostics and documented tool interfaces.
- **Iteration speed (D5)** was previously framed as compute efficiency (a cost concern). Reframed as *discovery-loop* speed — how many hypotheses an analyst can test per session — it becomes a capability dimension, not a cost dimension.
- **Narrative composition (D3, levels 4–5)** goes beyond per-model explanations (well covered) to *multi-finding reports with evidence grading and next actions*. The closest literature is in science communication, not ML.

The original proposal identified these dimensions before the literature survey. The original instrument marked them [N]. This paper retains that history; current labels and anchors are defined in the canonical scorecard.

---

## 6. Using the scorecard

### 6.1 Scoring protocol

Use the dimension definitions, evidence requirements, and scoring procedure in
the [canonical scorecard, version 2.1](scorecard.md). Fix the task battery
before scoring, record evidence for every grade, and report the full dimension
profile. Historical results scored under earlier local rubrics remain tied to
those rubrics; the [migration crosswalk](migration.md) explains which results
can and cannot be interpreted under version 2.1.

### 6.2 Mode-dependent weighting

The current mode weights are defined in the [canonical scorecard](scorecard.md).
The original paper proposed different weights for fully automated and
semi-automated use because the audience and workflow differ. Version 2.1 is the
normative source for the weights used in new evaluations.

### 6.3 Fairness rules

- **Same tasks, same budgets.** Every library gets the same DGPs, datasets, time budgets, and agent prompts.
- **Use recommended workflows.** Each library may use its idiomatic best practices, including tuned hyperparameters and recommended workflows. Keep the effort comparable and log any manual tuning as analyst effort.
- **Pre-register the battery.** Publish the evaluation tasks before scoring to prevent benchmark-shopping.
- **Report the profile.** Never collapse to a single number without showing the per-dimension scores.

---

## 7. Limitations and future work

1. **The [N] dimensions need validation.** D5, D8, and D3-levels-4–5 are proposed from problem structure, not from validated literature. They should be treated as hypotheses until agent-benchmark results accumulate.
2. **Causal discovery is graded as humility, not capability.** If the causal discovery literature produces genuinely reliable observational methods, D2 should be revised to grade capability.
3. **Cost is out of scope.** The scorecard doesn't price compute, licenses, or engineering time. A separate cost-effectiveness analysis should accompany any adoption decision.
4. **Historical evaluations have limited coverage.** Existing batteries were designed under earlier local rubrics. Their preserved grades do not establish coverage of every version 2.1 dimension; see the [migration crosswalk](migration.md) before comparing them with new evaluations.

---

## References

- Chapman, P. et al. (2000). *CRISP-DM 1.0: Step-by-step data mining guide.* SPSS.
- Doshi-Velez, F. & Kim, B. (2017). Towards a rigorous science of interpretable machine learning. *arXiv:1702.08608*.
- Fayyad, U., Piatetsky-Shapiro, G. & Smyth, P. (1996). From data mining to knowledge discovery in databases. *AI Magazine*, 17(3), 37–54.
- Geng, L. & Hamilton, H. J. (2006). Interestingness measures for data mining: A survey. *ACM Computing Surveys*, 38(3).
- Murdoch, W. J., Singh, C., Kumbier, K., Abbasi-Asl, R. & Yu, B. (2019). Definitions, methods, and applications in interpretable machine learning. *PNAS*, 116(44), 22071–22080.
- Peters, J., Janzing, D. & Schölkopf, B. (2017). *Elements of Causal Inference.* MIT Press.
- Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*, 1, 206–215.
- Spirtes, P., Glymour, C. & Scheines, R. (2000). *Causation, Prediction, and Search* (2nd ed.). MIT Press.
