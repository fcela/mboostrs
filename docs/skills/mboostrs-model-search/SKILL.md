---
name: mboostrs-model-search
description: Choose and audit the next public mboostrs specification-search action when an analyst wants an agent-directed investigation rather than one automatic formula.
---

# Agent-directed model search with mboostrs

Use this workflow when the analyst wants to inspect competing specifications
or decide which diagnostic to run next. The public entry point is
`mboostrs.open_search(training, response, ...)`. Its `SearchSession` lists
legal actions and fit costs in `state()["available_actions"]`; `run(action,
...)` returns a `SearchEvidence` with a result, status, evidence ID and actual
fit charges. The [usage page](../../book/usage/specification_search.ipynb)
defines the arguments, and the
[CZ case](../../book/cases/cz_agent_search.ipynb) shows a short investigation.

## Search path: expand, measure and simplify

Work an existing or provisional black box — a boosted tree, a neural probe,
a fat additive baseline — as a teacher to unfold, not a result to freeze:

1. **Teacher.** Run `pre_scan` (native, neural or dual) and, when a
   `tree_candidate` soak clears the two-SE rule at `compare`, adopt it
   knowing it is temporary. Do not freeze here.
2. **Unfold.** `structure_candidates` on the soak's residual returns typed
   terms — hinges, cutpoints, `btree(z, by=log1p(x), maxdepth=1)` — with the
   soak's own split points (`model.btree_splits()`) seeding the cutpoints.
   `transform_candidates` adds `log1p` slopes and `response_probe` the
   response scale; for counts, its `dispersion_hint` separates mean from
   sigma covariates.
3. **Measure.** `compare` the typed candidates against the soak incumbent
   under the session's two-SE rule.
4. **Prefer interpretable.** Read `two_se_candidates` and `also_cleared`;
   among winners that clear, keep the typed form over the catch-all.
5. **Soft-remove.** Drop the soak or redundant terms within two paired SE
   via `adopt(..., simplification_se=2)`; `soft_removal_candidates` names
   the candidates.
6. **Tune and rescreen.** Tune the adopted model before freezing — a
   catch-all at the starting `mstop` can overfit it away (an untuned tree
   scored 0.266 on this case's holdout where the tuned path scored 0.251) —
   then re-run `interaction_screen` on the tuned residuals; freeze only
   when no screen-positive lead is left uncompared or the budget is spent.

When a `btree(...)` term clears, compare typed candidates that describe the
structure it may have absorbed before freezing the search. Otherwise the
selected tree can retain structure that a more explicit model would represent
directly.

The routes below name each step's detail; the spine above is the order.

Before opening a session, define the prediction task and reserve evaluation
units that reflect deployment: rows, whole groups, or later time periods.
Pass only training rows to `open_search`. Do not inspect held-out responses
through a preliminary `autospec`, marginal screen or plot. Record variable
roles from subject-matter context: measurement timing, possible pathway,
proxy, or leakage risk. A predictive association does not identify an
intervention effect; ask for a causal design separately when that is the
analyst's target. With missing values, `na_action='omit'` fits and every
residual diagnostic read complete-case rows; `drop_nulls()` upfront gives
the same fit when that is clearer.

At each branch, state the uncertainty and choose **one** public action whose
result could change the decision. Useful routes include:

- Unclear coding or scale: `classify` or `column_screen`. Treat marginal
  associations and a native/neural `pre_scan` as screening proposals.
  `transform_candidates` charges a native pre-scan and nominates `log1p`
  predictor terms or one-split varying effects; measure its `candidate_rhs`
  with `compare` before using it. `structure_candidates` charges the same
  probe and screens a typed basis — hinges (`bols(pmax(z, c))`), cutpoint
  indicators (`btree(z, maxdepth=1)`) and centered basis products
  (`btree(z, by=log1p(x), maxdepth=1)`) — against the incumbent's residual,
  residualized on its design span so a term collinear with the fit is not
  punished for it. Raw-column pair screens cannot see structure that lives
  in these coordinates; measure the returned `candidate_rhs` with `compare`.
  When the incumbent carries a `btree` soak, its own split points seed the
  cutpoint grid: read them with `model.btree_splits()` (per variable, the
  cutpoints each tree used and how many boosting steps chose them), and read
  `model.coef()` for tree learners as per-step decoded node records — a
  tree's coefficients encode its structure, so they are never summed.
- Unclear family or shape: request bounded `autospec` evidence, then set a
  family/specification explicitly and compare complete alternatives on the
  session's fixed folds.
- A selected effect may be redundant: `term_report`; a residual pair may
  remain: fit or tune first, then `interaction_screen`, then `compare` a
  complete model. `term_report(proxy_hygiene=True)` marks a weak correlated
  sibling of a locked predictor and provides a nested drop candidate.
  `interaction_screen` may flag a thresholded sibling; its residual
  `removed` score and any pair lead alone are not adoption evidence.
- Positive skew or changing residual spread: use `response_probe` for a
  fixed-iteration, original-scale MSE comparison of identity, log or log1p
  Gaussian responses. A spread hint nominates explicit LSS modeling. Neither
  action switches the accepted family or response silently. For count
  families (`Poisson`, `NBinomial`) the same probe returns a
  `dispersion_hint` instead: a covariate that lines up with the signed
  Pearson residual is a missing mean term, one that moves only `log` of its
  square drives the dispersion — a sigma candidate, not another mean term.
  Measure the distributional alternative (`gamboostLSS`, the mean terms on
  mu and the dispersion columns on sigma) with `compare` before adopting.
- A count session's working order: thin baseline on measured earns, then
  `structure_candidates` and `interaction_screen` on the current residual,
  the dispersion attribution above before adding any spread-looking
  covariate to the mean, and a post-adopt screen after a catch-all term.
  Freeze when no screen-positive lead is left uncompared or the budget is
  spent; `drive(judgment)` sequences these if an unattended pass is wanted.
- A curve suggests an omitted structural variable: check training support
  and compare drop, replacement and adjustment formulas. The extra variable
  may explain shared predictive information without identifying a latent
  cause.
- A group is unseen in deployment: use grouped folds and declare an
  unseen-level policy before scoring. Do not let row-random folds decide a
  new-group claim.
- A causal question has a declared treatment and time zero: open with a
  `CausalSpec`, run `causal_traps`, and keep baseline covariates locked while
  treatment and post-treatment columns remain outside predictive screens.
  A clean design check still does not prove exchangeability or identify an
  effect from predictive scores.

Use `run("hypothesis", statement=..., role=..., would_count_against=...)`
before inspecting the specific result when possible. Candidate adoption
requires a measured `compare` evidence ID and the session's rule. Preserve
refusals, failures, tied alternatives and the cost of tests that did not help.
Read `two_se_candidates`, `also_cleared`, `winner_gaps` and
`soft_removal_candidates` before choosing among several winners; the first by
risk need not be the most interpretable. `winner_gaps` pairs each other
clearing candidate with the best one: how far it trails, in paired SE on the
shared folds, and whether that is within two — the measured input to keeping
the typed form over the catch-all. A nested drop within two paired standard errors needs explicit
`simplification_se=2` on `adopt`.
Adopting a catch-all term (`btree(...)`, a fat tensor) is a reason to look
again, not a reason to stop: refit and re-run `interaction_screen` on the new
residuals before `freeze`. A tree that only partially captures threshold
structure leaves it in the residuals, and the same screen that nominated it
can say how much is left. Freeze with a soak term adopted and no post-adopt
screen on record leaves that question open.
`checkpoint()` preserves the evidence/fit ledger without serializing the
native model; resume with the same training data and family, then fit or
tune explicitly before further model diagnostics. After freeze, use
`export_transcript()` for a JSON-only record of the decisions and final score;
that record cannot resume fitting.

## Action map

One line per public session action, in the order a search usually meets
them; `state()["available_actions"]` lists the ones legal right now with
their fit costs.

- Setup and screening: `classify`, `column_screen`, `pre_scan`,
  `transform_candidates`, `structure_candidates`, `autospec`,
  `hypothesis`, `set_family`, `set_spec`, `lock`, `causal_traps`
- Nominations from a fit: `interaction_screen`, `pair_candidates`
  (`surrogate=` a `pre_scan` report or `True`), `tree_candidate`,
  `partial_effects`, `term_report` (`proxy_hygiene=True`)
- Measurement: `compare`, `adopt` (`simplification_se=`), `tune`, `fit`,
  `response_probe`
- Boundaries: `freeze`, `final_evaluate`, `checkpoint` /
  `SearchSession.resume`, `export_transcript`

The measured adoption rule is the two-SE `compare` evidence ID; nothing
else adopts. Distributional work continues outside the session with
`gamboostLSS` / `glmboostLSS` and `cvrisk`, which `compare` reaches
through family objects and the dispersion recipe above.

For an optional unattended action-selection loop, pass a bounded callable
to `session.drive(judgment, max_actions=...)`. `LLMJudge` can use this route
with a configured client; it sees only the bounded session state and cannot
request final test evaluation. Its provider call hashes and usage enter the
session action record. An external agent can instead inspect `state()` and
call `run` directly, without sending any evidence to a provider.

Stop when the budget or the analyst's question is resolved. Freeze the
selected model before one `final_evaluate` call on the untouched sample.
With `score='family'` the holdout reports the weighted mean family loss —
the same quantity `cvrisk` reports for a fold, so a Poisson or NBinomial
search keeps its deviance scale on the test frame; distributional (LSS)
and multinomial families need a response-scale score there.
Report what the measurement shows, its units and fold target, what remains
uncertain, and the next question if one is warranted. Do not describe a
scripted `modeler_loop` callback or canned LLM reply as an autonomous agent
choosing its own tools. The built-in loop remains a separate, fixed action
sequence; this playbook uses the public session directly.
