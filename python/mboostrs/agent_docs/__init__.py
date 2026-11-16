"""Agent-facing documentation shipped with the package.

``agent_guide()`` returns the entry point of the "analyzing data with
mboostrs" skill: what the library is, the first five steps of the
measured-decision workflow, and the topic index. ``agent_guide("<topic>")``
returns one topic (the workflow steps 6 to 10, the pitfalls, the uncertainty
and causal guides, and so on), and ``agent_guide("index")`` lists them, so an
agent reads only the part it needs. Each file stays under about 8,000
characters, which keeps it within what an agent tool shows in one read. ``install_skill()`` copies the skill and its
topics into a project's ``.claude/skills/`` directory, where coding agents
that read skills find it without being told.

The same files live in the repository under
``docs/skills/analyzing-data-with-mboostrs/``; a test keeps the two
byte-identical.
"""

from __future__ import annotations

import shutil
from importlib import resources
from pathlib import Path

_SKILL = "SKILL.md"
_SKILL_NAME = "analyzing-data-with-mboostrs"

#: Topic name -> (file, one-line description). The order is the index order.
TOPICS = {
    "golden-path": ("golden-path.md",
                    "start here: data -> fit() -> report() and three follow-ups, "
                    "each with the topic to open only when it is needed"),
    "quick-reference": ("quick-reference.md",
                        "one runnable block that runs the workflow on simulated data"),
    "insights": ("insights.md",
                 "insights(): an optional screen, fit, diagnose and propose pass whose "
                 "findings are hypotheses to confirm"),
    "before-the-first-fit": ("before-the-first-fit.md",
                             "prediction target, evaluation rows, variable roles and the "
                             "missing-data policy, settled before any fit"),
    "screen-and-draft": ("screen-and-draft.md",
                         "steps 1 and 2: column_screen, classify_columns, suggest_terms "
                         "and autospec, with the df rule for comparing terms"),
    "fit-and-tune": ("fit-and-tune.md",
                     "steps 3 and 4: gamboost, "
                     "tune_mstop(formula, data, *, start=100, max_mstop=10000) and "
                     "its boundary flag, grouped folds, scores"),
    "interactions-and-terms": ("interactions-and-terms.md",
                               "step 5: screen_interactions, term_report, what each term "
                               "earns, the binary-response caveat"),
    "modeling-tactics": ("modeling-tactics.md",
                         "step 5: the gap to a flexible reference (reference_gap) and "
                         "the changes that closed gaps in benchmarks: interactions, lags, "
                         "log scales, imputation, basis size, null and small-data checks"),
    "specification-search": ("specification-search.md",
                             "steps 6 and 7: suggest_specifications, "
                             "compare_specifications(specifications, data, "
                             "response, *, folds=5, mstop=500) and the two-SE adoption rule"),
    "distributional-branch": ("distributional-branch.md",
                              "step 8: heteroscedasticity, "
                              "scan_lss_families(formula, data, *, metric='GAIC', "
                              "fit_func='glmboostLSS'), scale_absorption and the "
                              "distributional model"),
    "sensitivity-and-contrasts": ("sensitivity-and-contrasts.md",
                                  "step 9: sensitivity, prediction_contrast and "
                                  "distributional_contrast, with their support limits"),
    "explain": ("explain.md",
                "step 10: llm_explain_prompt, what to give it and how to check its reply"),
    "pitfalls": ("pitfalls.md",
                 "the tensor df floor, screening on the wrong residual scale, group "
                 "folds, unseen levels, unequal df"),
    "interpretation": ("interpretation.md",
                       "thresholds that make a turn, a spread ratio or an exceedance "
                       "shift worth reporting"),
    "reading-a-fit": ("reading-a-fit.md",
                      "how to read the effects of an additive fit, what it cannot "
                      "tell, and what a first reading misses "
                      "(predict(newdata=None, *, type='link'))"),
    "closing-the-analysis": ("closing-the-analysis.md",
                             "what the final report must state, with refusals, failures "
                             "and ties"),
    "missing-data": ("missing-data.md",
                     "na_action='omit' or 'impute' for every fitting function; "
                     "predicting rows with missing values"),
    "uncertainty-selection": ("uncertainty-selection.md",
                              "intervals and tests after the same rows chose the terms: "
                              "split refits (mb.split_refit), deselection and stability "
                              "selection (stabsel(model, *, q=, PFER=))"),
    "uncertainty": ("uncertainty.md",
                    "effect intervals and term tests (REML refits: "
                    "gamlss_refit(model) -> GamlssRefit), contrast "
                    "intervals from draws, prediction intervals "
                    "(conformal_interval(model, *, level=0.9)), stability selection"),
    "direct-reml-gam": ("direct-reml-gam.md",
                        "penalized GAM by REML without boosting: mb.gam and mb.gamlss "
                        "fitted from a formula, with contrasts and intervals"),
    "categoricals": ("categoricals.md",
                     "bols vs brandom for labels, unseen levels, passing labels "
                     "to predict and refits"),
    "costs": ("costs.md",
              "what each call costs in fits and seconds, and how to budget a run"),
    "unattended-pipelines": ("unattended-pipelines.md",
                             "scripts that decide by rule on new data: seeds, "
                             "null checks, structured records, warnings"),
    "discover-limits": ("discover-limits.md",
                        "discover(): constraints, contrasts, null and small-data behaviour, "
                        "budgets and limits"),
    "discover": ("discover.md",
                 "mb.discover(): the whole rule-driven analysis in one call, its "
                 "decision log, null and small-data behaviour, budgets"),
    "task-map": ("task-map.md",
                 "which function performs each task (fit, tune, evaluate, diagnose, "
                 "compare, interpret, predict with uncertainty, export) in Python, R "
                 "and the Rust core, and where a frontend has no such function"),
    "task-map-interpret-export": ("task-map-interpret-export.md",
                                  "the functions that interpret a fit, predict with uncertainty "
                                  "and export it (save_model(model, path) / load_model(path)), "
                                  "with a worked example"),
    "diagnostics": ("diagnostics.md",
                    "residual checks that name a revision: clusters (brandom), time "
                    "(time smooth), influential rows (refit without them), with "
                    "false-alarm rates and a pattern-to-revision guide"),
    "causal": ("causal.md",
               "treatment effects from observational data: the workflow in order, "
               "with one example and the causal-* sub-topics that hold the details"),
    "causal-roles": ("causal-roles.md",
                     "the causal question and its effect scale, causal_trap_report, "
                     "the adjustment set and post-treatment variables, ci_test"),
    "causal-average": ("causal-average.md",
                       "aipw: the average effect with an influence interval, repeats, "
                       "missing outcomes, smoothing, custom nuisance formulas"),
    "causal-outcome-models": ("causal-outcome-models.md",
                              "aipw nuisance steps and warnings, the outcome model "
                              "(separate, pooled, two-part), the sparse-arm check"),
    "causal-heterogeneity": ("causal-heterogeneity.md",
                             "effect_model: the effect at each covariate pattern with "
                             "per-row intervals and interval methods; rank_by_effect"),
    "causal-effect-model-checks": ("causal-effect-model-checks.md",
                                   "effect_model's candidate choice, the comparison with "
                                   "a constant effect, pseudo-outcome clipping, warnings"),
    "causal-effect-modifiers": ("causal-effect-modifiers.md",
                                "effect_modifiers: BLP and GATES tests of which covariates "
                                "change the effect, Holm adjustment, projections"),
    "causal-overlap-sensitivity": ("causal-overlap-sensitivity.md",
                                   "overlap_report, the overlap-weighted target, "
                                   "confounding_sensitivity and confounding_benchmark"),
    "causal-instruments": ("causal-instruments.md",
                           "iv_effect: the instrumental-variable estimate, its four "
                           "assumptions, the first-stage F check"),
}

#: Worked analyses, read with ``agent_guide("example:<name>")``. Each one runs a
#: whole analysis on a described data set, including the wrong turns and how
#: the evidence corrected them.
EXAMPLES = {
    "rent-analysis": ("rent-analysis.md",
                      "apartment rents: the whole workflow, with a wrong turn the "
                      "screen corrected"),
    "claims-distributional": ("claims-distributional.md",
                              "insurance claims: when the spread depends on covariates "
                              "and the distributional model pays"),
    "readmissions-pitfalls": ("readmissions-pitfalls.md",
                              "hospital readmissions: unseen levels and the tensor df "
                              "floor"),
    "pm25-reference-gap": ("pm25-reference-gap.md",
                           "hourly PM2.5: a tuned additive model trails a tree ensemble; "
                           "lags and interactions close the gap"),
    "disc-screening": ("disc-screening.md",
                       "a diagonal regime jump: the screen names the boundary, "
                       "and the step indicator beats the joint tree"),
}


def guide_hint(topic: str) -> str:
    """The sentence that closes a warning by naming the guide topic behind it."""
    if topic not in TOPICS:
        raise ValueError(f"unknown agent_guide topic {topic!r}")
    return f" Guide: python -m mboostrs guide {topic}."


def _root():
    return resources.files(__name__)


def _index() -> str:
    width = max(len(name) for name in TOPICS)
    lines = ["mboostrs agent guide topics. Read one with mboostrs.agent_guide(\"<topic>\");",
             "the full workflow is mboostrs.agent_guide().", ""]
    lines += [f"  {name:<{width}}  {description}" for name, (_, description) in TOPICS.items()]
    lines += ["", "Worked analyses. Read one with mboostrs.agent_guide(\"example:<name>\").", ""]
    width = max(len(name) for name in EXAMPLES)
    lines += [f"  example:{name:<{width}}  {description}"
              for name, (_, description) in EXAMPLES.items()]
    return "\n".join(lines) + "\n"


def agent_guide(topic: str | None = None, path_only: bool = False) -> str:
    """Return the agent skill text, a topic section, or a file path.

    ``topic=None`` returns the full workflow (``SKILL.md``); ``"index"``
    lists the topics and the worked examples; a topic name returns that
    section; ``"example:<name>"`` returns a worked analysis. ``path_only=True``
    returns the file's path instead of its text (for ``"index"``, the topics
    directory).

    Point an agent at it before its first fit::

        import mboostrs
        print(mboostrs.agent_guide())
        print(mboostrs.agent_guide("index"))
    """
    root = _root()
    if topic is None:
        ref = root.joinpath(_SKILL)
    elif topic == "index":
        if path_only:
            return str(root.joinpath("topics"))
        return _index()
    elif topic in TOPICS:
        ref = root.joinpath("topics", TOPICS[topic][0])
    elif topic.startswith("example:") and topic[len("example:"):] in EXAMPLES:
        ref = root.joinpath("examples", EXAMPLES[topic[len("example:"):]][0])
    else:
        raise ValueError(
            f"unknown agent_guide topic {topic!r}; the topics are "
            f"{', '.join(TOPICS)}; the worked examples are "
            f"{', '.join('example:' + name for name in EXAMPLES)} "
            "(or 'index', or None for the full workflow)")
    if path_only:
        return str(ref)
    return ref.read_text(encoding="utf-8")


def install_skill(dest: str | Path = ".claude/skills", *, overwrite: bool = False) -> Path:
    """Copy the skill, its topics and examples to ``dest/analyzing-data-with-mboostrs/``.

    ``dest`` defaults to ``.claude/skills`` under the current directory, the
    project skills directory that Claude Code reads; pass
    ``~/.claude/skills`` for every project of the current user. An existing
    copy is refused unless ``overwrite=True``, so a hand-edited skill is not
    replaced silently. Returns the directory written.

    The same is available from a shell::

        python -m mboostrs skill install [--dest DIR] [--overwrite]
    """
    target = Path(dest).expanduser() / _SKILL_NAME
    if target.exists():
        if not overwrite:
            raise FileExistsError(
                f"{target} already exists; pass overwrite=True (or --overwrite) to replace it")
        shutil.rmtree(target)
    (target / "topics").mkdir(parents=True)
    root = _root()
    (target / _SKILL).write_text(root.joinpath(_SKILL).read_text(encoding="utf-8"),
                                 encoding="utf-8")
    for filename, _ in TOPICS.values():
        text = root.joinpath("topics", filename).read_text(encoding="utf-8")
        (target / "topics" / filename).write_text(text, encoding="utf-8")
    (target / "examples").mkdir()
    for filename, _ in EXAMPLES.values():
        text = root.joinpath("examples", filename).read_text(encoding="utf-8")
        (target / "examples" / filename).write_text(text, encoding="utf-8")
    return target


__all__ = ["EXAMPLES", "TOPICS", "agent_guide", "guide_hint", "install_skill"]
