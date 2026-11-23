#!/usr/bin/env python3
"""Generate the book's API reference from the objects themselves.

The page it replaces was one hand-written 8 KB markdown cell. It documented a
few dozen of 146 public names, and what it did document had drifted: it showed
``glmboost(formula, *, data, ...)`` after `data` became positional, listed
``control()`` and ``offset()`` after both became properties, and gave
``risk(parameter=None)`` for a method whose arguments were removed. A reader
comparing it with the package would have found it wrong in every one of those
places, and nothing would have said so.

Everything here is read from the live objects: `inspect.signature` for the
arguments and their defaults, `__doc__` for the summary line,
`inspect.getattr_static` for whether a member is a property or a method, and
the capability manifest for the row each family and learner is verified
against. ``--check`` compares the generated page with the committed one and
fails when they differ, so a changed signature fails CI rather than waiting
for someone to notice.
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
from pathlib import Path
from typing import Any, get_type_hints

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/book/appendix/api_reference.ipynb"
MANIFEST = ROOT / "python/mboostrs/capability_manifest.json"

sys.path.insert(0, str(ROOT / "python"))

import mboostrs as mb  # noqa: E402

#: How the public names are grouped on the page, in the order a reader meets
#: them: fit something, choose what it optimises, choose the shapes, read the
#: result, resample, diagnose. Anything not matched falls into "Other", which
#: is a signal that a new export needs a home rather than a place to leave it.
SECTIONS: list[tuple[str, str, Any]] = [
    ("Fitting", "`fit(data, response, ...)` selects and fits a specification "
                "and returns a `Fit`: the fitted model plus its decision trail, a "
                "JSON-serialisable record of the data, the folds, each candidate "
                "change with its out-of-fold score, and the final specification. "
                "`glmboost` and `gamboost` fit a supplied formula; `glmboostLSS` "
                "and `gamboostLSS` fit distributional models (LSS: location, scale "
                "and shape), in which each distribution parameter has its own "
                "additive predictor. Arguments after `*` in a signature are "
                "keyword-only.",
     lambda name, obj: name in ("fit", "glmboost", "gamboost", "glmboostLSS",
                                "gamboostLSS", "gam", "gamlss")),
    ("Control", "Set the maximum iterations and update step length.",
     lambda name, obj: name in ("boost_control", "BoostControl")),
    ("Fitted models", "Fitted-model classes and stopping budgets.",
     lambda name, obj: name in ("Fit", "GLMBoostModel", "GAMBoostModel",
                                "LSSModel", "StoppingBudget")),
    ("Families", "Each family defines the loss and response encoding used by "
                 "the fit. The *Python fit* and *R fit* columns give the evidence "
                 "the capability manifest (`mb.family_capabilities()`) records for "
                 "fitting in each frontend: `verified` means the fit is compared "
                 "with a reference implementation or reference value within a "
                 "stated tolerance; `smoke` means tests check that the fit runs "
                 "and satisfies stated invariants, with no reference to compare "
                 "against; `intentional-deviation` means the tested behaviour "
                 "differs deliberately from the R `mboost` or `gamboostLSS` "
                 "counterpart; `unverified` means the manifest makes no "
                 "correctness claim. The same columns appear in the "
                 "distributional-family and base-learner tables. The "
                 "[capabilities](capabilities.ipynb) page lists every status "
                 "value, and the [R parity matrix](r_parity_matrix.ipynb) lists "
                 "the comparisons with R.",
     lambda name, obj: _is_family(name, obj) and not name.endswith("LSS")),
    ("Distributional families", "Each distribution parameter has its own additive predictor.",
     lambda name, obj: _is_family(name, obj) and name.endswith("LSS")
                       and name != "LSSModel"),
    ("Base learners", "Specify linear, smooth, spatial, and other effects and their penalties.",
     lambda name, obj: _is_learner(name, obj)
                       or name in ("learner_design", "LearnerDesign")),
    ("Cross-validation and folds", "Create resampling partitions and select `mstop` from held-out loss.",
     lambda name, obj: name in (
         "cv_weights", "make_grid", "make_grid_lss", "diagonal_grid",
         "FoldScheme", "stratified_folds", "grouped_folds", "known_group_folds",
         "blocked_folds",
         "rolling_folds", "repeated_kfold_folds", "loo_folds", "CvRiskResult",
         "CvReport",
         "LssCvRiskResult", "EvaluationResult", "evaluate_specification",
         "cvrisk_glmboostLSS", "cvrisk_gamboostLSS", "tune_mstop",
         "MstopTuning")),
    ("Stability selection", "Select terms by resampling, compare candidate effects, and refit selected specifications.",
     lambda name, obj: name in ("stabsel", "StabSelResult", "LssStabSelResult",
                                "stabsel_glmboostLSS", "stabsel_gamboostLSS",
                                "deselect", "Deselection", "probe", "Probing",
                                "gam_refit", "GamRefit", "gamlss_refit", "GamlssRefit",
                                "EffectContrast",
                                "split_refit", "SplitRefitResult",
                                "equivalence_sets", "EquivalenceSets",
                                "substitution_sets", "SubstitutionSets",
                                "selection_report", "SelectionReport")),
    ("Diagnostics", "Inspect residuals, calibration, prediction intervals, and model diagnostics.",
     lambda name, obj: name in (
         "acf_residuals", "calibration_frame", "centile_fan_data", "coverage", "coef_all",
         "density_overlay_data", "deviance_increment", "deviance_profile",
         "interval_metrics", "ipc_weights", "outlier_frame", "pit",
         "predict_all", "q_stats", "quantile_residuals", "residual_frame",
         "same_family_gap", "SameFamilyGap",
         "worm_data", "conformal_interval", "ConformalInterval",
         "rearrange_quantiles", "RearrangedQuantiles",
         "heteroscedasticity", "interval_metrics", "evaluate_distribution",
         "cluster_residual_check", "time_residual_check", "influence_check",
         "imputation_diagnostics",
         "revision_checks", "RevisionFinding", "InfluenceFinding", "RevisionReport",
         "DistributionEvaluationResult", "validate",
         "evaluate_survival", "SurvivalEvaluationResult",
         "scale_absorption", "ScaleAbsorption")),
    ("Choosing a specification", "Compare candidate families and terms using the specified folds and scoring rules.",
     lambda name, obj: name in (
         "autospec", "OrderedCategoryWarning", "check_declarations", "DeclarationCheck", "classify_columns", "column_screen", "insights", "Finding", "discover", "DiscoveryResult", "agent_guide", "api_card", "install_skill", "suggest_terms", "term_report", "cv_risk",
         "compare_specifications", "suggest_specifications", "sensitivity",
         "estimate_cost",
         "holdout",
         "TermReport", "Suggestion", "ColumnKind", "ScreenedPair",
         "DiscontinuityHit",
         "expand_pairs", "screen_interactions", "screen_discontinuities",
         "expand_interactions",
         "expand_formula", "parse_formula", "parse_gamboost_formula",
         "label_evidence", "LabelEvidence", "pair_term_for", "select_capacity", "CapacitySelection", "reference_gap", "ReferenceGapResult",
         "auto_policy", "CostBudget", "Spent", "FitCache", "PolicyResult", "PairedTest",
         "modeler_loop", "ModelerResult", "SpecificationJudgment", "LLMJudge", "Transcript",
         "open_search", "SearchSession", "SearchEvidence",
         "specification_curve", "SpecificationCurve")),
    ("Model selection and comparison", "Compare fitted losses, information criteria, and model summaries.",
     lambda name, obj: name in (
         "compare_models", "gaic_table", "profile_mstop", "r_squared",
         "scan_lss_families", "FAMILY_SETS", "family_set", "compare_summaries",
         "Summary", "SummaryTable", "AicResult", "compare_to_r", "CompareReport",
         "CompareSpec")),
    ("Effects and contrasts", "Estimate prediction changes, derivatives, and contrasts over specified rows.",
     lambda name, obj: name in (
         "ALE1DResult", "ContrastResult", "ElasticityResult",
         "MarginalEffectResult", "PredictionProfileResult",
         "accumulated_local_effect",
         "average_elasticity", "average_marginal_effect",
         "distributional_contrast", "prediction_contrast", "prediction_profile")),
    ("Causal estimands", "Specify causal targets and adjustment assumptions, then fit treatment-effect or instrumental-variable models.",
     lambda name, obj: name in ("CausalSpec", "AIPWResult", "aipw",
                                "EffectModel", "effect_model", "CITest", "ci_test",
                                "NegativeControlResult", "confounding_sensitivity",
                                "confounding_benchmark", "negative_control",
                                "overlap_report", "OverlapReport",
                                "rank_by_effect", "OverlapRanking",
                                "effect_modifiers", "EffectModifiers", "BLPProjection",
                                "iv_effect", "IVResult",
                                "Assumption", "AssumptionLedger", "CausalDraft",
                                "TrapFinding", "TrapReport", "causal_trap_report",
                                "elicit_causal_spec")),
    ("Persistence and reporting", "Save fitted models, inspect recorded decisions, and report predictions.",
     lambda name, obj: name in ("report", "DecisionTrail", "save_model", "load_model", "dump_result_json", "load_result_json", "ExecutionReport",
                                "PathRow", "build_info", "provenance", "llm_describe",
                                "llm_explain_prompt")),
    ("Capabilities", "List supported operations and the tests used to validate them.",
     lambda name, obj: name.endswith("capabilities")),
    ("Errors", "What a failed call raises. Each class subclasses the built-in "
               "exception it replaces and carries `code`, `subject`, "
               "`details` and `rust_variant` attributes.",
     lambda name, obj: name in ("MboostrsError", "MboostrsValueError",
                                "MboostrsRuntimeError", "MboostrsKeyError")),
    ("Utilities", "",
     lambda name, obj: name in ("weighted_median", "datasets", "plot",
                                "collect_warnings")),
]

#: One minimal copyable snippet per public name, where a hand-written one
#: earns its place over the mechanical default. Snippets assume the book's
#: running context -- `import mboostrs as mb` and a frame named `df` where
#: one is built inline otherwise. `tools` tests compile every snippet and
#: resolve every `mb.` attribute against the package, so an example naming a
#: renamed export fails CI rather than misleading a reader.
EXAMPLES: dict[str, str] = {
    "validate": (
        'issues = mb.validate(model, test, output="records")\n'
        'refused = [issue for issue in issues if issue["severity"] == "error"]'
    ),
    "CostBudget": 'budget = mb.CostBudget(fits=100, seconds=60)',
    "open_search": (
        'df = mb.datasets.bodyfat()\n'
        'session = mb.open_search(df, "DEXfat", family=mb.Gaussian(),\n'
        '                         folds=3, score="mse", budget=mb.CostBudget(fits=60))\n'
        'session.state()'
    ),
    "SearchSession": 'session.state()["available_actions"]',
    "specification_curve": (
        'curve = mb.specification_curve(session, df, "DEXfat", budget=200)\n'
        'curve.rows  # one row per specification: gap to the selected one, with its paired SE'
    ),
    "select_capacity": (
        'df = mb.datasets.bodyfat()\n'
        'pick = mb.select_capacity("DEXfat ~ .", df, mstop=500)\n'
        'pick.selected  # "base", "+tensors" or "+tree": the smallest rung within\n'
        '                # tolerance standard errors of the best held-out score'
    ),
    "CapacitySelection": (
        'pick.rungs  # one record per rung: label, formula, held-out score,\n'
        '             # fold SE, whether it was selected, and why it was built'
    ),
    "SpecificationCurve": 'curve.joint_test["p_value"], curve.label',
    "SearchEvidence": 'evidence = session.run("classify")\nevidence.action, evidence.value',
    "auto_policy": (
        'df = mb.datasets.bodyfat()\n'
        'result = mb.auto_policy(df, "DEXfat", family=mb.Gaussian(),\n'
        '                        folds=3, budget=mb.CostBudget(fits=100))'
    ),
    "modeler_loop": (
        'df = mb.datasets.bodyfat()\n'
        'result = mb.modeler_loop(df, "DEXfat", family=mb.Gaussian(),\n'
        '                         max_rounds=1, workflow_budget=mb.CostBudget(fits=100))'
    ),
    "CausalSpec": 'design = mb.CausalSpec(\n'
                  '    treatment="treated", outcome="outcome",\n'
                  '    covariates=("age", "baseline"), estimand="ate",\n'
                  '    time_zero="treatment assignment", follow_up="one year",\n'
                  '    adjustment_rationale="age and baseline precede treatment and are common causes",\n'
                  ')',
    "aipw": 'result = mb.aipw(data, design, folds=5, mstop=200)',
    "AIPWResult": 'rows = result.to_frame(output="polars")\nresult.describe()',
    "effect_model": 'fx = mb.effect_model(data, design, nuisance=result)',
    "ci_test": 'mb.ci_test(data, "proxy", "y", given=["driver"]).describe()',
    "collect_warnings": 'with mb.collect_warnings():\n    for seed in range(20):\n        mb.ci_test(data, "x", "y", given=["z"], seed=seed)',
    "column_screen": 'mb.column_screen(data, "y")',
    "insights": 'for f in mb.insights(data, "y"):\n    print(f.finding, f.basis, f.next_action)',
    "Finding": 'f.finding, f.effect_size, f.uncertainty, f.basis, f.evidence, f.next_action',
    "discover": 'result = mb.discover(data, "y", newdata=new, seed=0)\nprint(result.formula, result.mean_effects)',
    "DiscoveryResult": 'result.contrast("x", -1.0, 1.0).estimate\nresult.log[:3]  # stage, candidate, change, se, adopted, reason',
    "reference_gap": 'res = mb.reference_gap("y ~ bbs(x, df=4) + bols(g)", data, folds=5, seed=0)\nprint(res.describe())\nres.candidates  # ranked terms, with measured held-out gains',
    "ReferenceGapResult": 'res.gap, res.gap_se, res.material, res.closed_share\nres.to_dict()',
    "agent_guide": 'print(mb.agent_guide())\nprint(mb.agent_guide("index"))\nprint(mb.agent_guide("missing-data"))',
    "install_skill": 'mb.install_skill(".claude/skills")  # or: python -m mboostrs skill install',
    "api_card": 'print(mb.api_card(mb.tune_mstop))\n# or: python -m mboostrs sig tune_mstop',
    "CITest": 'test.p_value, test.residual_correlation, test.describe()',
    "EffectModel": 'fx.predict(new_rows)\nfx.model.partial_effects()\nfx.describe()',
    "confounding_sensitivity": (
        'grid = mb.confounding_sensitivity(result, outcome_effects=[0, 1],\n'
        '                                   imbalances=[0, 0.5])'
    ),
    "confounding_benchmark": (
        'benchmarks = mb.confounding_benchmark(data, result,\n'
        '                                      variables=["age", "baseline"])'
    ),
    "negative_control": (
        'check = mb.negative_control(data, control_design, kind="outcome",\n'
        '                            rationale="measured before treatment",\n'
        '                            folds=5, mstop=200)'
    ),
    "NegativeControlResult": 'check.interval_excludes_expected\ncheck.describe()',
    "overlap_report": (
        'report = mb.overlap_report(result, data=data, bounds=(0.02, 0.98))\n'
        'report.share_outside, report.estimate, report.estimate_inside'
    ),
    "OverlapReport": 'report.describe()\nreport.regions',
    "rank_by_effect": (
        'ranking = mb.rank_by_effect(fx, new_rows, share=0.2)\n'
        'ranking.chosen, ranking.outside, ranking.gain_interval'
    ),
    "OverlapRanking": 'ranking.describe()\nranking.table(output="polars")',
    "effect_modifiers": (
        'tests = mb.effect_modifiers(result, data, method="blp", level=0.9)\n'
        'tests.table(output="polars"), tests.selected()'
    ),
    "EffectModifiers": 'tests.describe()\ntests.projection("passing")',
    "BLPProjection": 'tau_hat = proj.predict(new_rows)\n'
        'ranking = mb.rank_by_effect(proj, new_rows, share=0.2, propensity=e_new)',
    "iv_effect": (
        'iv = mb.iv_effect(data, design, "offer", folds=5, mstop=200)\n'
        'iv.estimate, iv.interval, iv.first_stage_f'
    ),
    "IVResult": 'iv.describe()\niv.to_frame(output="polars")',
    "prediction_contrast": (
        'change = mb.prediction_contrast(fit, df, "waistcirc", 80, 100,\n'
        '                                type="response")'
    ),
    "ContrastResult": 'rows = change.to_frame(output="polars")\nchange.describe()',
    "accumulated_local_effect": (
        'ale = mb.accumulated_local_effect(fit, df, "waistcirc", bins=10)'
    ),
    "ALE1DResult": 'curve = ale.to_frame(output="polars")\nale.describe()',
    "average_elasticity": (
        'elasticity = mb.average_elasticity(count_fit, df, "exposure",\n'
        '                                   kind="elasticity")'
    ),
    "ElasticityResult": (
        'rows = elasticity.to_frame(output="polars")\nelasticity.describe()'
    ),
    "prediction_profile": (
        'profile = mb.prediction_profile(fit, df, "waistcirc", points=15)'
    ),
    "PredictionProfileResult": (
        'curve = profile.to_frame(output="polars")\nprofile.describe()'
    ),
    "average_marginal_effect": (
        'effect = mb.average_marginal_effect(fit, df, "waistcirc",\n'
        '                                    type="response")'
    ),
    "MarginalEffectResult": 'rows = effect.to_frame(output="polars")\neffect.describe()',
    "distributional_contrast": (
        'upper_tail = mb.distributional_contrast(\n'
        '    lss_fit, df, "x", 0, 1, quantity="quantile", value=0.9,\n'
        ')'
    ),
    "glmboost": 'df = mb.datasets.bodyfat()\n'
                'fit = mb.glmboost("DEXfat ~ waistcirc + hipcirc", data=df)',
    "gamboost": 'df = mb.datasets.bodyfat()\n'
                'fit = mb.gamboost("DEXfat ~ bbs(hipcirc, df=4) + bols(waistcirc)",\n'
                '                   data=df)',
    "check_declarations": (
        'df = mb.datasets.bodyfat()\n'
        'rec = mb.autospec(df, "DEXfat", monotone={"waistcirc": "increasing"})\n'
        'mb.check_declarations(rec, df, folds=3)'
    ),
    "DeclarationCheck": (
        'df = mb.datasets.bodyfat()\n'
        'rec = mb.autospec(df, "DEXfat", monotone={"waistcirc": "increasing"})\n'
        'mb.check_declarations(rec, df, folds=3)[0].verdict'
    ),
    "autospec": 'df = mb.datasets.bodyfat()\n'
                'formula, learners, family, mstop = mb.autospec(df, "DEXfat")\n'
                'fit = mb.gamboost(formula, data=df, learners=learners,\n'
                '                   family=family, control=mb.boost_control(mstop))',
    "glmboostLSS": 'df = mb.datasets.heteroscedastic_lss()\n'
                   'fit = mb.glmboostLSS("y ~ bols(x1) + bols(x2)", data=df,\n'
                   '                     families=mb.GaussianLSS())',
    "gamboostLSS": 'df = mb.datasets.heteroscedastic_lss()\n'
                   'fit = mb.gamboostLSS("y ~ bbs(x1, df=3) + bols(x2)", data=df,\n'
                   '                     families=mb.GaussianLSS())',
    "boost_control": 'ctrl = mb.boost_control(mstop=200, nu=0.1)\n'
                     'fit = mb.glmboost("DEXfat ~ waistcirc",\n'
                     '                   data=mb.datasets.bodyfat(), control=ctrl)',
    "MboostrsError": 'try:\n'
                     '    fit = mb.gamboost("y ~ bbs(x)", data=data_faulty)\n'
                     'except mb.MboostrsError as error:\n'
                     '    print(error.code, error.details)',
    "MboostrsValueError": 'try:\n'
                          '    fit = mb.glmboost("y ~ x", data=data_faulty,\n'
                          '                       control=mb.boost_control(10, nu=3))\n'
                          'except mb.MboostrsValueError as error:\n'
                          '    print(error.code, error.subject)',
"MboostrsRuntimeError": 'try:\n'
                        '    fit = mb.gamboost("y ~ bbs(x)", data=data_faulty)\n'
                        'except mb.MboostrsRuntimeError as error:\n'
                        '    print(error.rust_variant)',
    "MboostrsKeyError": 'try:\n'
                        '    fit = mb.glmboost("y ~ zzz", data=data_faulty)\n'
                        'except mb.MboostrsKeyError as error:\n'
                        '    print(error.code, error.subject)',
    "cv_weights": 'folds = mb.cv_weights(n=71, type="kfold", B=5, seed=42)\n'
                  'curve = fit.cvrisk(folds=folds)\n'
                  'curve.optimal_mstop',
    "FoldScheme": 'scheme = mb.rolling_folds(120, B=4, initial=60, horizon=12)\n'
                  'fit.cvrisk(folds=scheme).held_out_fraction',
    "stratified_folds": 'folds = mb.stratified_folds(y_binary, B=5)\n'
                        'fit.cvrisk(folds=folds)',
    "grouped_folds": 'folds = mb.grouped_folds(groups, B=4)\n'
                     'fit.cvrisk(folds=folds)',
    "known_group_folds": 'folds = mb.known_group_folds(groups, B=4)\n'
                         'mb.evaluate_specification("bols(x) + brandom(g)",\n'
                         '                          data, "y", folds=folds)',
    "blocked_folds": 'folds = mb.blocked_folds(n=120, B=5)\n'
                     'fit.cvrisk(folds=folds)',
    "rolling_folds": 'folds = mb.rolling_folds(n=120, B=4, initial=60,\n'
                     '                 horizon=12)\n'
                     'fit.cvrisk(folds=folds)',
    "repeated_kfold_folds": 'folds = mb.repeated_kfold_folds(n=120, k=5,\n'
                            '                        repeats=10, seed=1)\n'
                            'fit.cvrisk(folds=folds)',
    "loo_folds": 'folds = mb.loo_folds(n=60)\n'
                 'fit.cvrisk(folds=folds)',
    "CvReport": ('report = fit.cvrisk(folds=folds).report()\n'
                 'report.optimum, report.risk_at_optimum, report.at_boundary'),
    "StoppingBudget": ('budget = fit.stopping_budget()\n'
                       'budget.scheme, budget.total, budget.per_parameter'),
    "evaluate_specification": (
        'evaluation = mb.evaluate_specification(\n'
        '    "bols(x1) + bbs(x2, df=3)",\n'
        '    mb.datasets.heteroscedastic_lss(), "y", folds=5, score="mse",\n'
        ')'
    ),
    "holdout": (
        'search, test = mb.holdout(data, fraction=0.2)\n'
        'search_g, test_g = mb.holdout(data, fraction=0.2, groups="district")'
    ),
    "EvaluationResult": (
        'evaluation.fold_frame(output="polars")\n'
        'evaluation.describe()'
    ),
    "stabsel": 'sel = fit.stabsel(cutoff=0.8, q=2, PFER=1.5, B=50)\n'
               'sel.selected',
    "dump_result_json": 'text = mb.dump_result_json({"width": float("inf"), "missing": None})',
    "load_result_json": 'text = mb.dump_result_json({"width": float("inf")})\nrecord = mb.load_result_json(text)',
    "save_model": 'mb.save_model(fit, "fit.json")',
    "load_model": 'fit = mb.load_model("fit.json")\n'
                  'fit.unavailable_operations()',
    "conformal_interval": 'interval = mb.conformal_interval(fit, level=0.9)\n'
                          'interval.lower[:3]',
    "rearrange_quantiles": 'r = mb.rearrange_quantiles([fit_q10, fit_q50, fit_q90], rows)\n'
                           'r.crossing_share, r.predictions',
    "RearrangedQuantiles": 'r = mb.rearrange_quantiles([fit_q10, fit_q50, fit_q90], rows)\n'
                           'r.pair_crossings, r.max_crossing',
    "interval_metrics": (
        'metrics = mb.interval_metrics(y, interval.lower, interval.upper,\n'
        '                              level=0.9)'
    ),
    "evaluate_distribution": (
        'evaluation = mb.evaluate_distribution(\n'
        '    lss_fit, test_data, response="y", data_role="external",\n'
        '    levels=(0.8, 0.95), thresholds=(0,),\n'
        ')'
    ),
    "DistributionEvaluationResult": (
        'scores = evaluation.to_frame(output="polars")\n'
        'evaluation.describe()'
    ),
    "evaluate_survival": (
        'evaluation = mb.evaluate_survival(\n'
        '    cox_fit, test_data, horizons=[1.0, 2.0], data_role="external",\n'
        ')\n'
        'evaluation.horizon_frame(output="polars")'
    ),
    "SurvivalEvaluationResult": (
        'evaluation.survival_frame(output="polars")\n'
        'evaluation.describe()'
    ),
    "heteroscedasticity": 'report = mb.heteroscedasticity(fit)\n'
                          'report["spread_ratio"]',
    "cluster_residual_check": 'finding = mb.cluster_residual_check(fit, "site")\n'
                              'finding.flagged, finding.revision',
    "time_residual_check": 'finding = mb.time_residual_check(fit, "year")\n'
                           'finding.statistic, finding.p_value',
    "imputation_diagnostics": 'fit = mb.gamboost("y ~ bbs(x) + bbs(z)", data,\n'
                              '                  na_action="impute_model")\n'
                              'mb.imputation_diagnostics(fit)  # {"x": R-squared}\n',
    "influence_check": 'finding = mb.influence_check(fit)\n'
                       'finding.rows, finding.refits',
    "revision_checks": 'report = mb.revision_checks(fit, cluster="site", time="year")\n'
                       'report.revisions',
    "RevisionFinding": 'finding = mb.cluster_residual_check(fit, "site")\n'
                       'print(finding.summary)',
    "InfluenceFinding": 'finding = mb.influence_check(fit)\n'
                        'finding.cooks, finding.rows',
    "RevisionReport": 'report = mb.revision_checks(fit, cluster="site")\n'
                      'report.flagged',
    "scale_absorption": 'result = mb.scale_absorption(fit, folds=5, seed=0)\n'
                        'result.verdict',
    "ScaleAbsorption": 'result = mb.scale_absorption(fit, folds=5, seed=0)\n'
                       'result.to_dict()["absorbed"]',
    "term_report": 'report = mb.term_report("DEXfat ~ bols(waistcirc) "\n'
                   '                    "+ bbs(hipcirc, df=4)",\n'
                   '                    data=mb.datasets.bodyfat(),\n'
                   '                    folds=5)\n'
                   'report.baseline',
    "label_evidence": 'india = mb.datasets.india(output="polars")\n'
                      'evidence = mb.label_evidence("mcdist", india["mcdist"],\n'
                      '                             india["stunting"])\n'
                      'evidence.labels',
    "pair_term_for": 'mb.pair_term_for("dose", "treated",\n'
                     '                 levels={"dose": 200, "treated": 2})',
    "tune_mstop": 'tuning = mb.tune_mstop("DEXfat ~ bbs(hipcirc, df=4)",\n'
                  '                       data=mb.datasets.bodyfat(), folds=5)\n'
                  'tuning.mstop',
    "deselect": 'tuned = mb.tune_mstop("DEXfat ~ .", data=mb.datasets.bodyfat()).model\n'
                'mb.deselect(tuned).kept',
    "probe": 'mb.probe("DEXfat ~ .", data=mb.datasets.bodyfat()).selected',
    "Deselection": 'tuned = mb.tune_mstop("DEXfat ~ .", data=mb.datasets.bodyfat()).model\n'
                   'mb.deselect(tuned).dropped',
    "equivalence_sets": 'tuned = mb.tune_mstop("DEXfat ~ .", data=mb.datasets.bodyfat()).model\n'
                        'mb.equivalence_sets(tuned, threshold=0.9).groups',
    "EquivalenceSets": 'tuned = mb.tune_mstop("DEXfat ~ .", data=mb.datasets.bodyfat()).model\n'
                       'mb.equivalence_sets(tuned).overlap',
    "substitution_sets": 'tuned = mb.tune_mstop("DEXfat ~ .", data=mb.datasets.bodyfat()).model\n'
                         'stab = mb.stabsel(tuned, cutoff=0.6, PFER=2.0, B=50, verbose=False)\n'
                         'mb.substitution_sets(stab, equivalence=mb.equivalence_sets(tuned)).groups',
    "SubstitutionSets": 'tuned = mb.tune_mstop("DEXfat ~ .", data=mb.datasets.bodyfat()).model\n'
                        'stab = mb.stabsel(tuned, cutoff=0.6, PFER=2.0, B=50, verbose=False)\n'
                        'mb.substitution_sets(stab, equivalence=mb.equivalence_sets(tuned)).pairs',
    "selection_report": 'tuned = mb.tune_mstop("DEXfat ~ .", data=mb.datasets.bodyfat()).model\n'
                        'mb.selection_report(tuned, deselection=mb.deselect(tuned)).dropped_alone',
    "SelectionReport": 'tuned = mb.tune_mstop("DEXfat ~ .", data=mb.datasets.bodyfat()).model\n'
                       'mb.selection_report(tuned, deselection=mb.deselect(tuned)).equivalence.groups',
    "Probing": 'probing = mb.probe("DEXfat ~ .", data=mb.datasets.bodyfat())\n'
               '(probing.stop, probing.first_shadow)',
    "gam": 'body = mb.datasets.bodyfat()\n'
           'fit = mb.gam("DEXfat ~ bbs(hipcirc, knots=6) + bbs(waistcirc, knots=6)", data=body)\n'
           'fit.contrast("hipcirc", 90.0, 110.0, data=body).estimate',
    "EffectContrast": 'body = mb.datasets.bodyfat()\n'
                      'fit = mb.gam("DEXfat ~ bbs(hipcirc, knots=6)", data=body)\n'
                      'change = fit.contrast("hipcirc", 90.0, 110.0, data=body)\n'
                      '(change.estimate, change.lower, change.upper)',
    "gamlss": 'data = mb.datasets.heteroscedastic_lss()\n'
              'fit = mb.gamlss({"mu": "y ~ bbs(x1, knots=6)", "sigma": "y ~ bbs(x2, knots=6)"},\n'
              '                data=data, family=mb.GaussianLSS())\n'
              'fit.predict(data.head(3), parameter="sigma", interval=True)',
    "gam_refit": 'tuned = mb.tune_mstop("DEXfat ~ bbs(hipcirc) + bbs(waistcirc)",\n'
                 '                      data=mb.datasets.bodyfat()).model\n'
                 'mb.gam_refit(tuned).edf',
    "GamRefit": 'body = mb.datasets.bodyfat()\n'
                'tuned = mb.tune_mstop("DEXfat ~ bbs(hipcirc) + bbs(waistcirc)", data=body).model\n'
                'estimate, lower, upper = mb.gam_refit(tuned).effects(body)["bbs(hipcirc)"]\n'
                '(lower[:3], upper[:3])',
    "gamlss_refit": 'fit = mb.gamboostLSS({"mu": "y ~ bbs(x1)", "sigma": "y ~ bbs(x2)"},\n'
                    '                     data=mb.datasets.heteroscedastic_lss(),\n'
                    '                     families=mb.GaussianLSS())\n'
                    'mb.gamlss_refit(fit).edf',
    "GamlssRefit": 'data = mb.datasets.heteroscedastic_lss()\n'
                   'fit = mb.gamboostLSS({"mu": "y ~ bbs(x1)", "sigma": "y ~ bbs(x2)"},\n'
                   '                     data=data, families=mb.GaussianLSS())\n'
                   'mb.gamlss_refit(fit).predict(data, parameter="sigma", type="response")[:3]',
    "split_refit": 'data = mb.datasets.bodyfat()\n'
                   'result = mb.split_refit("DEXfat ~ bbs(hipcirc) + bbs(waistcirc)", data,\n'
                   '                        repeats=5, seed=0,\n'
                   '                        contrasts=[("hipcirc", 90.0, 110.0)])\n'
                   'result.summary()',
    "SplitRefitResult": 'data = mb.datasets.bodyfat()\n'
                        'result = mb.split_refit("DEXfat ~ bbs(hipcirc) + bbs(waistcirc)", data,\n'
                        '                        seed=0, contrasts=[("hipcirc", 90.0, 110.0)])\n'
                        'result.to_dict()["contrasts"]',
    "cv_risk": 'score = mb.cv_risk("bols(x1) + bbs(x2, df=3)",\n'
               '            data=mb.datasets.heteroscedastic_lss(),\n'
               '            response="y", folds=5)\n'
               'score',
    "estimate_cost": 'cost = mb.estimate_cost("compare_specifications",\n'
                      '                        ["bols(x1)", "bbs(x1, df=4)"],\n'
                      '                        mb.datasets.heteroscedastic_lss(), "y")\n'
                      'cost.fits',
    "compare_specifications": 'table = mb.compare_specifications(\n'
                              '    ["bols(x1)", "bbs(x1, df=4)"],\n'
                              '    data=mb.datasets.heteroscedastic_lss(),\n'
                              '    response="y")',
    "sensitivity": 'rows = mb.sensitivity(fit)\n'
                   'rows[0]["range of the effect"]',
    "profile_mstop": 'table = mb.profile_mstop([fit_a, fit_b])',
    "gaic_table": 'table = mb.gaic_table([fit_a, fit_b])',
    "compare_models": 'report = mb.compare_models(fit_a, fit_b)',
    "r_squared": 'mb.r_squared(fit)',
    "FAMILY_SETS": 'mb.FAMILY_SETS["realplus"]',
    "family_set": 'mb.family_set("realplus")',
    "coverage": 'mb.coverage(observed, lower, upper)',
    "pit": 'mb.pit(fit)',
    "same_family_gap": (
        'fit = mb.glmboost("y ~ x",\n'
        '    data={"y": [1.0, 2.0, 1.4, 2.2], "x": [0.0, 1.0, 0.2, 0.9]},\n'
        '    control=mb.boost_control(30, 0.2))\n'
        'gap = mb.same_family_gap(fit)\n'
        'gap.mstop, gap.gap, gap.material'
    ),
    "quantile_residuals": 'mb.quantile_residuals(fit)',
    "q_stats": 'mb.q_stats(fit)',
    "centile_fan_data": 'table = mb.centile_fan_data(lss_fit, xvar="x")',
    "density_overlay_data": 'table = mb.density_overlay_data(lss_fit)',
    "deviance_profile": 'table = mb.deviance_profile(lss_fit, xvar="x")',
    "llm_explain_prompt": 'print(fit.llm_explain_prompt(include=("importance",)))',
    "llm_describe": 'reply = mb.llm_describe(fit)',
    "weighted_median": 'mb.weighted_median(values, weights)',
    "data_card": 'mb.datasets.data_card("bodyfat")',
    "family_capabilities": 'caps = mb.family_capabilities(output="polars")',
    "learner_capabilities": 'caps = mb.learner_capabilities(output="polars")',
    "model_capabilities": 'caps = mb.model_capabilities(output="polars")',
    "method_capabilities": 'caps = mb.method_capabilities(output="polars")',
    "validation_capabilities": 'vals = mb.validation_capabilities(output="polars")',
    "dataset_capabilities": 'cards = mb.dataset_capabilities(output="polars")',
    "capabilities": 'caps = mb.capabilities(output="polars")',
    "build_info": 'identity = dict(mb.build_info())\nidentity["build_id"]',
    "provenance": (
        'record = mb.provenance({"y": [1.0, 2.0, 3.0]}, seed=1)\n'
        'record["data"]["sha256"]'
    ),
    "fit": (
        'df = mb.datasets.bodyfat()\n'
        'm = mb.fit(df, "DEXfat", seed=0)\n'
        'm.trail.validate()  # schema-backed evidence of how m.model was chosen'
    ),
    "Fit": (
        'df = mb.datasets.bodyfat()\n'
        'm = mb.fit(df, "DEXfat", seed=0)\n'
        'm.model, m.trail, m.stability  # the fit, its trail, and the uncertainty step'
    ),
    "report": (
        'm = mb.fit(mb.datasets.bodyfat(), "DEXfat", seed=0)\n'
        'print(mb.report(m))              # text\n'
        'mb.report(m, format="json")["final"]  # the trail dict itself'
    ),
    "DecisionTrail": (
        'm = mb.fit(mb.datasets.bodyfat(), "DEXfat", seed=0)\n'
        'restored = mb.DecisionTrail.from_dict(m.trail.to_json())\n'
        'restored.to_dict() == m.trail.to_dict()'
    ),
    "OrderedCategoryWarning": (
        'import warnings\n'
        'warnings.simplefilter("error", mb.OrderedCategoryWarning)'
    ),
    "suggest_specifications": 'report = mb.term_report(\n'
                              '    "y ~ bols(x1)",\n'
                              '    data=mb.datasets.heteroscedastic_lss(),\n'
                              '    folds=5)\n'
                              'for suggestion in mb.suggest_specifications(report):\n'
                              '    print(suggestion.action, suggestion.term)',
}

#: Families whose construction is the whole example: the snippet names the
#: response contract the family requires, because that is the spelling a
#: reader gets wrong. `None` values carry a hand-written comment.
_FAMILY_RESPONSE: dict[str, str] = {
    "Gaussian": "y: any real value",
    "GaussReg": "y: any real value; the same model as Gaussian",
    "Binomial": "y in {0, 1} (or {-1, +1}, or [0, 1] proportions)",
    "Poisson": "y: counts, non-negative integers",
    "NBinomial": "y: counts; overdispersed relative to Poisson",
    "Hurdle": "y: positive counts only; the zero part is a separate binary model",
    "GammaReg": "y: strictly positive",
    "Laplace": "y: any real value; median regression",
    "Quantile": "y: any real value; quantile set at construction",
    "Expectile": "y: any real value; expectile set at construction",
    "Huber": "y: any real value; robust to outliers",
    "AdaExp": "y in {0, 1} or {-1, +1}; exponential-loss classification",
    "AUC": "y: two-level factor or {-1, +1}; ranking objective",
    "Cindex": "Surv(time, status); concordance",
    "Gehan": "Surv(time, status); AFT rank loss",
    "CoxPH": "Surv(time, status); Cox partial likelihood",
    "Weibull": "Surv(time, status); parametric survival",
    "Lognormal": "Surv(time, status); parametric survival",
    "Loglog": "Surv(time, status); parametric survival",
    "PropOdds": "y: ordered factor; proportional-odds",
    "HingeLoss": "y in {-1, +1}; margin classification",
    "Multinomial": "y: integer class codes or factor; softmax over classes",
}

#: LSS families: the construction is the same shape for all of them; the
#: comment names the response support.
_LSS_RESPONSE: dict[str, str] = {
    "GaussianLSS": "y: any real value",
    "GaussianKnownVarLSS": "cbind(y, s2): outcome and its known sampling variance",
    "GammaLSS": "y: strictly positive",
    "StudentTLSS": "y: any real value",
    "BetaLSS": "y: in (0, 1), proportions",
    "NBinomialLSS": "y: counts",
    "ZIPoLSS": "y: counts with zeros",
    "ZINBLSS": "y: counts with zeros, overdispersed",
    "PoissonLSS": "y: counts",
    "GeometricLSS": "y: counts",
    "BinomialLSS": "y in {0, 1} or [0, 1]",
    "NegativeBinomialLSS": "y: counts",
    "ZeroAdjustedGammaLSS": "y: non-negative, with zeros",
    "InverseGaussianLSS": "y: strictly positive",
    "LogisticLSS": "y: any real value",
    "GumbelLSS": "y: any real value",
    "Pareto2LSS": "y: strictly positive",
    "ZeroAdjustedPoissonLSS": "y: non-negative integers with zeros",
    "PIGLSS": "y: counts",
    "BCCGLSS": "y: strictly positive",
    "BCPELSS": "y: strictly positive",
    "BCTLSS": "y: strictly positive",
    "JSULSS": "y: any real value",
    "DirichletLSS": "cbind(y1, ..., yK): composition rows summing to 1",
    "LogNormalLSS": "Surv(time, status), log-normal baseline",
    "LogLogLSS": "Surv(time, status), log-logistic baseline",
    "WeibullLSS": "Surv(time, status), Weibull baseline",
}

#: Base learners: the formula spelling is the example; a few need programmatic
#: construction and say so.
_LEARNER_EXAMPLES: dict[str, str] = {
    "Bbs": 'mb.gamboost("DEXfat ~ bbs(hipcirc, df=4)", data=df)',
    "Bols": 'mb.gamboost("DEXfat ~ bols(waistcirc)", data=df)',
    "Bns": 'mb.gamboost("DEXfat ~ bns(hipcirc, df=4)", data=df)',
    "Bspatial": 'mb.gamboost("y ~ bspatial(x, z, df=6)", data=df)',
    "Brandom": 'mb.gamboost("rent ~ brandom(district)", data=df)',
    "Btree": 'mb.gamboost("DEXfat ~ btree(hipcirc)", data=df)',
    "Bmono": 'mb.gamboost("DEXfat ~ bmono(hipcirc)", data=df)',
    "Bkernel": 'mb.gamboost("DEXfat ~ bkernel(hipcirc)", data=df)',
    "Bconstrained": 'C = ...  # (constraints, p) matrix, p = spline columns\n'
                    'mb.gamboost("DEXfat ~ bconstrained(hipcirc, C=C)",\n'
                    '            data=df, env={"C": C})',
    "Buser": 'X = ...  # (n, p) design; K optional penalty\n'
             'mb.gamboost("y ~ buser(X, K=K)", data=df,\n'
             '            env={"X": X, "K": K})',
    "Bmrf": 'K = ...  # neighbourhood matrix, programmatic\n'
            'mb.gamboost("rate ~ bmrf(district, K=K)", data=df,\n'
            '            env={"K": K})',
    "Brad": 'mb.gamboost("DEXfat ~ brad(hipcirc)", data=df)',
    "Hinge": 'mb.gamboost("DEXfat ~ hinge(hipcirc)", data=df)',
}


def example_for(name: str) -> str | None:
    """The minimal copyable snippet for a public name, or None.

    Hand-written first (the workflow entries where a reader's first ten
    minutes go), then the mechanical defaults: a family constructs with a
    comment naming its response contract, an LSS family likewise, a learner
    shows its formula spelling.
    """
    if name in EXAMPLES:
        return EXAMPLES[name]
    if name in _FAMILY_RESPONSE:
        return f"family = mb.{name}()  # {_FAMILY_RESPONSE[name]}"
    if name in _LSS_RESPONSE:
        contract = _LSS_RESPONSE[name]
        args = "K=3" if name == "DirichletLSS" else ""
        return f"families = mb.{name}({args})  # {contract}"
    if name in _LEARNER_EXAMPLES:
        return _LEARNER_EXAMPLES[name]
    return None


def _is_family(name: str, obj: Any) -> bool:
    return (inspect.isclass(obj)
            and name in capability_rows()
            and obj.__module__ in ("mboostrs.families", "mboostrs.gamlss"))


def _is_learner(name: str, obj: Any) -> bool:
    return inspect.isclass(obj) and obj.__module__ == "mboostrs.learners"


def summary_line(obj: Any) -> str:
    """The first sentence of the docstring, or an empty string.

    One line, because a reference table that carries three paragraphs per
    entry is a chapter with a table of contents, not a reference.
    """
    # A constant such as a dict inherits its type's docstring from `inspect`,
    # which would describe `dict`, not the object: only classes, functions and
    # modules carry a docstring of their own here.
    if not (inspect.isclass(obj) or inspect.isroutine(obj) or inspect.ismodule(obj)):
        return ""
    doc = inspect.getdoc(obj) or ""
    if not doc:
        return ""
    first = doc.split("\n\n", 1)[0].replace("\n", " ").strip()
    # reST roles read as noise in a markdown table: `:meth:`x`` becomes `x`.
    first = re.sub(r":[a-z]+:`~?([^`]+)`", r"`\1`", first)
    first = first.replace("``", "`")
    # A reST literal-block marker (`Usage::`, `Example::`) introduces code
    # that the summary does not carry, so the marker is dropped; any other
    # trailing `::` becomes a full stop.
    first = re.sub(r"\s*\b(Usage|Examples?)::$", "", first)
    if first.endswith("::"):
        first = first[:-2].rstrip() + "."
    return first.replace("|", "\\|")


def render_signature(obj: Any) -> str:
    """``name(args)`` with defaults and the keyword-only marker kept.

    The modules use `from __future__ import annotations`, so `inspect` reports
    every annotation as a quoted string -- `data: 'str'`, `bins: 'int | None'`.
    The quotes are noise in a reference table, so they come off.
    """
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):
        return ""
    try:
        hints = get_type_hints(obj)
    except (NameError, TypeError):
        hints = {}
    if hints:
        parameters = [
            parameter.replace(annotation=hints.get(name, parameter.annotation))
            for name, parameter in signature.parameters.items()
        ]
        signature = signature.replace(
            parameters=parameters,
            return_annotation=hints.get("return", signature.return_annotation),
        )
    text = f"{getattr(obj, '__name__', '?')}{signature}"
    # Only in annotation positions -- after `:` or `->`. A *default* that is a
    # string keeps its quotes, because `na_action: str = 'error'` says the
    # default is the string `error` and `= error` says something else.
    text = re.sub(r"(:\s*)'([^']*)'", r"\1\2", text)
    text = re.sub(r"(->\s*)'([^']*)'", r"\1\2", text)
    # Python 3.13 moved the pathlib classes into the private module
    # `pathlib._local`, so the same annotation renders as `pathlib.Path` on
    # 3.12 and `pathlib._local.Path` on 3.13. The page must name what a reader
    # can type, and must not change with the interpreter that generated it.
    text = text.replace("pathlib._local.", "pathlib.")
    return text.replace("|", "\\|")


def capability_rows() -> dict[str, dict[str, Any]]:
    """Family or learner name to its full manifest row."""
    manifest = json.loads(MANIFEST.read_text())
    rows: dict[str, dict[str, Any]] = {}
    for table in ("families", "learners"):
        for row in manifest["tables"].get(table, []):
            rows[row["name"]] = row
    return rows


def _member_cell(cls: type, name: str) -> str:
    """Form and class-specific meaning of one fitted-model member."""
    attribute = inspect.getattr_static(cls, name, None)
    if attribute is None:
        return "--"
    if isinstance(attribute, property):
        form = "property"
        summary = summary_line(attribute.fget)
    elif callable(attribute):
        form = "method"
        summary = summary_line(attribute)
    else:
        form = "attribute"
        summary = ""
    return form + (f" — {summary}" if summary else "")


def model_surface() -> list[tuple[str, str, str, str]]:
    """Every public member of the three model classes, and its form.

    `(member, glmboost, gamboost, LSSModel)`. Each cell carries the form and
    that class's own summary, so a scalar standard-model property is not
    described by an unrelated LSS mapping docstring. The *form* is the point:
    a member that is a property on one class and a method on another cannot be
    used by a helper written against either, and the table shows it at a
    glance.
    """
    from mboostrs.gamboost import GAMBoostModel
    from mboostrs.gamlss import LSSModel
    from mboostrs.glmboost import GLMBoostModel

    classes = [("glmboost", GLMBoostModel), ("gamboost", GAMBoostModel),
               ("LSS", LSSModel)]
    names: set[str] = set()
    for _label, cls in classes:
        names |= {n for n in dir(cls) if not n.startswith("_")}

    out = []
    for name in sorted(names):
        out.append((f"`{name}`", *[_member_cell(cls, name)
                                    for _label, cls in classes]))
    return out


def markdown_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def render_markdown() -> str:
    capabilities = capability_rows()
    # `__version__` is exported but is a string, not an API entry.
    exported = {name: getattr(mb, name) for name in sorted(mb.__all__)
                if not name.startswith("_")}
    placed: set[str] = set()
    parts = [
        "# API reference",
        "",
        "Every public name in the `mboostrs` Python package, grouped by task, "
        "with its signature and defaults, a one-line description, and for "
        "most entries a short copyable example.",
        "",
        "For argument descriptions, return values and raised errors, open the "
        "symbol's page in the [Python API manual](../api/index.md), for example "
        "[`glmboost`](../api/functions/glmboost.md). In the snippets below, "
        "`mb` refers to `mboostrs` and `fit` refers to a fitted model.",
        "",
    ]

    for title, blurb, matches in SECTIONS:
        rows = []
        section_names: list[str] = []
        for name, obj in exported.items():
            if name in placed or not matches(name, obj):
                continue
            placed.add(name)
            section_names.append(name)
            signature = (name if title == "Fitted models"
                         else render_signature(obj))
            entry = f"`{signature}`" if signature else f"`{name}`"
            capability = capabilities.get(name)
            if capability is None:
                rows.append([entry, summary_line(obj) or ""])
            else:
                support = capability.get("support", {})
                py_fit = support.get("python", {}).get("operations", {}).get("fit", "")
                r_fit = support.get("r", {}).get("operations", {}).get("fit", "")
                rows.append([entry, summary_line(obj) or "", py_fit, r_fit])
        if not rows:
            continue
        parts += [f"## {title}", ""]
        if blurb:
            parts += [blurb, ""]
        header = (["Name", "What it is", "Python fit", "R fit"]
                  if any(len(row) == 4 for row in rows)
                  else ["Name", "What it is"])
        parts += [markdown_table(header, rows), ""]
        examples = [(name, example_for(name)) for name in section_names]
        examples = [(name, snippet) for name, snippet in examples if snippet]
        if examples:
            for name, snippet in examples:
                parts += [f"**`{name}`**", "", "```python", snippet, "```", ""]
            parts.append("")

    remaining = [name for name in exported if name not in placed]
    if remaining:
        parts += [
            "## Other",
            "",
            "Exported and not yet placed in a section above. An entry here "
            "means `tools/generate_api_reference.py` needs a home for it, not "
            "that it is less public.",
            "",
            markdown_table(
                ["Name", "What it is"],
                [[f"`{render_signature(exported[n]) or n}`",
                  summary_line(exported[n]) or ""] for n in remaining],
            ),
            "",
        ]

    parts += [
        "(the-model-surface)=",
        "",
        "## Model methods and properties",
        "",
        "The table identifies each public member as a property or a method "
        "for each fitted-model class. Access a property without parentheses; "
        "call a method with parentheses. `--` marks an unavailable member. "
        "Check the model class before writing code that handles several "
        "kinds of fit.",
        "",
        markdown_table(
            ["Member", "glmboost", "gamboost", "LSS"],
            [list(row) for row in model_surface()],
        ),
        "",
    ]

    parts += rust_surface_section()
    return "\n".join(parts) + "\n"


def rust_surface_section() -> list[str]:
    """The `mboostrs._rust` extension boundary, rendered from its stub.

    `_rust.pyi` is generated from the compiled module's own signatures
    (`tools/generate_rust_stub.py`), so this section cannot drift from the
    extension any more than the stub can. It is rendered because the stub
    lived only in the package directory -- a reader of the book had no map of
    the boundary at all. These are not the public API. Production GAM fitting
    and resampling use compact learner-spec calls; deprecated parallel-array
    compatibility symbols remain visible until their scheduled 0.2 removal.
    """
    stub = (ROOT / "python/mboostrs/_rust.pyi").read_text()
    # The class grouping: a `class X:` line followed by its methods.
    entries: list[str] = []
    current_class = ""
    for line in stub.splitlines():
        stripped = line.strip()
        if stripped.startswith("class ") and stripped.endswith(":"):
            current_class = stripped[6:-1].split("(")[0].strip('"')
            entries.append(f"`{current_class}`")
            continue
        if line.startswith("def "):
            # A top-level function ends the preceding class block.
            current_class = ""
        if stripped.startswith("def ") and stripped.endswith("..."):
            definition = stripped[4:-3].rstrip(":").strip()
            name = definition.split("(", 1)[0]
            owner = (f"`{current_class}.{name}`" if current_class
                     else f"`{name}`")
            entries.append(f"{owner} — `{definition}`")
    return [
        "(the-rust-extension-surface)=",
        "",
        "## Compiled extension functions",
        "",
        "The `mboostrs._rust` module is the compiled engine boundary, "
        "rendered here from `python/mboostrs/_rust.pyi` (itself generated "
        "from the compiled module's signatures). **These are not the public "
        "API**. Production GAM fitting and resampling use compact learner-spec "
        "calls; the wide `fit_gamboost` and `cvrisk_gamboost` entries are "
        "deprecated compatibility symbols scheduled for removal in 0.2. "
        "They remain visible so a reader debugging across versions can "
        "identify the boundary in use. Argument types are `Any` because PyO3 does not "
        "surface them. For the crate's Rust-side public items, see the crate "
        "README and `cargo doc -p mboostrs-core`; for R, `help(package = "
        "\"mboostrs\")`.",
        "",
        markdown_table(["Symbol", "Signature"], [[r.split(" — ")[0], r]
                                                 for r in entries]),
        "",
    ]


def render_notebook() -> str:
    markdown = render_markdown()
    notebook = {
        "cells": [{"cell_type": "markdown", "metadata": {},
                   "source": markdown.splitlines(keepends=True)}],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, indent=1) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="fail when the committed page is out of date")
    arguments = parser.parse_args()

    generated = render_notebook()
    if arguments.check:
        current = OUTPUT.read_text() if OUTPUT.exists() else ""
        if current != generated:
            print(
                "docs/book/appendix/api_reference.ipynb is out of date: a "
                "signature, default, docstring summary or property/method "
                "form changed. Run `python tools/generate_api_reference.py` "
                "and commit the result.",
                file=sys.stderr,
            )
            return 1
        print("API reference is up to date")
        return 0

    OUTPUT.write_text(generated)
    print(f"API reference generated: {len(mb.__all__)} public names")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
