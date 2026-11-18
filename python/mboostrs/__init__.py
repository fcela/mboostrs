"""mboostrs — Component-wise functional gradient boosting (R mboost semantics in Rust).

AI agents and new users: call ``mboostrs.agent_guide()`` for the analysis
workflow and ``mboostrs.agent_guide("index")`` for its topic sections
(missing values, uncertainty, categoricals, costs, unattended pipelines);
``python -m mboostrs skill install`` installs it as a coding-agent skill
(also at ``docs/skills/analyzing-data-with-mboostrs/SKILL.md``);
``mboostrs.insights(df, response)`` proposes candidate findings to verify.
"""

from mboostrs import datasets
from mboostrs._control import BoostControl, boost_control
from mboostrs._errors import (
    MboostrsError,
    MboostrsKeyError,
    MboostrsRuntimeError,
    MboostrsValueError,
)
from mboostrs._model_api import StoppingBudget
from mboostrs._preflight import validate
from mboostrs._provenance import provenance
from mboostrs._results import AicResult
from mboostrs._warning_collect import collect_warnings
from mboostrs.absorption import ScaleAbsorption, scale_absorption
from mboostrs.agent_docs import agent_guide, install_skill
from mboostrs.api_card import api_card
from mboostrs.autospec import OrderedCategoryWarning, autospec
from mboostrs.build import build_info
from mboostrs.capabilities import (
    capabilities,
    family_capabilities,
    learner_capabilities,
    method_capabilities,
    validation_capabilities,
)
from mboostrs.capacity import CapacitySelection, select_capacity
from mboostrs.causal import (
    AIPWResult,
    CausalSpec,
    CITest,
    EffectModel,
    IVResult,
    NegativeControlResult,
    OverlapRanking,
    OverlapReport,
    TrapFinding,
    TrapReport,
    aipw,
    causal_trap_report,
    ci_test,
    confounding_benchmark,
    confounding_sensitivity,
    effect_model,
    iv_effect,
    negative_control,
    overlap_report,
    rank_by_effect,
)
from mboostrs.compare import CompareReport, CompareSpec, compare_to_r
from mboostrs.conformal import ConformalInterval, conformal_interval
from mboostrs.cost import estimate_cost
from mboostrs.cvrisk import CvReport, CvRiskResult, cv_weights, make_grid, weighted_median
from mboostrs.declarations import DeclarationCheck, check_declarations
from mboostrs.diagnostics import (
    SameFamilyGap,
    acf_residuals,
    calibration_frame,
    centile_fan_data,
    coef_all,
    coverage,
    density_overlay_data,
    deviance_increment,
    deviance_profile,
    interval_metrics,
    ipc_weights,
    outlier_frame,
    pit,
    predict_all,
    q_stats,
    quantile_residuals,
    residual_frame,
    same_family_gap,
    worm_data,
)
from mboostrs.direct_gam import gam
from mboostrs.discover import DiscoveryResult, discover
from mboostrs.distribution_evaluation import (
    DistributionEvaluationResult,
    evaluate_distribution,
)
from mboostrs.effect_modifiers import BLPProjection, EffectModifiers, effect_modifiers
from mboostrs.effects import (
    ALE1DResult,
    ContrastResult,
    ElasticityResult,
    MarginalEffectResult,
    PredictionProfileResult,
    accumulated_local_effect,
    average_elasticity,
    average_marginal_effect,
    distributional_contrast,
    prediction_contrast,
    prediction_profile,
)
from mboostrs.equivalence import (
    EquivalenceSets,
    SelectionReport,
    SubstitutionSets,
    equivalence_sets,
    selection_report,
    substitution_sets,
)
from mboostrs.evaluation import EvaluationResult
from mboostrs.explain import llm_describe, llm_explain_prompt
from mboostrs.explore import (
    MstopTuning,
    Suggestion,
    TermReport,
    compare_specifications,
    cv_risk,
    evaluate_specification,
    heteroscedasticity,
    holdout,
    pair_term_for,
    sensitivity,
    suggest_specifications,
    term_report,
    tune_mstop,
)
from mboostrs.families import (
    AUC,
    AdaExp,
    Binomial,
    Cindex,
    CoxPH,
    Expectile,
    GammaReg,
    Gaussian,
    GaussReg,
    Gehan,
    HingeLoss,
    Huber,
    Hurdle,
    Laplace,
    Loglog,
    Lognormal,
    Multinomial,
    NBinomial,
    Poisson,
    PropOdds,
    Quantile,
    Weibull,
)
from mboostrs.folds import (
    FoldScheme,
    blocked_folds,
    grouped_folds,
    known_group_folds,
    loo_folds,
    repeated_kfold_folds,
    rolling_folds,
    stratified_folds,
)
from mboostrs.formula import expand_formula, parse_formula, parse_gamboost_formula
from mboostrs.gamboost import GAMBoostModel, gamboost
from mboostrs.gamlss import (
    BCCGLSS,
    BCPELSS,
    BCTLSS,
    JSULSS,
    PIGLSS,
    ZINBLSS,
    BetaLSS,
    BinomialLSS,
    DirichletLSS,
    GammaLSS,
    GaussianKnownVarLSS,
    GaussianLSS,
    GeometricLSS,
    GumbelLSS,
    InverseGaussianLSS,
    LogisticLSS,
    LogLogLSS,
    LogNormalLSS,
    LssCvRiskResult,
    LSSModel,
    LssStabSelResult,
    NBinomialLSS,
    NegativeBinomialLSS,
    Pareto2LSS,
    PoissonLSS,
    StudentTLSS,
    WeibullLSS,
    ZeroAdjustedGammaLSS,
    ZeroAdjustedPoissonLSS,
    ZIPoLSS,
    cvrisk_gamboostLSS,
    cvrisk_glmboostLSS,
    diagonal_grid,
    gamboostLSS,
    glmboostLSS,
    make_grid_lss,
    stabsel_gamboostLSS,
    stabsel_glmboostLSS,
)
from mboostrs.glmboost import GLMBoostModel, glmboost

# `insights` is bound to the function, not the submodule of the same name: the
# from-import runs after the submodule is bound on the package and replaces it.
from mboostrs.insights import Finding, insights
from mboostrs.interactions import LearnerDesign, expand_interactions, learner_design
from mboostrs.learners import (
    Bbs,
    Bconstrained,
    Bkernel,
    Bmono,
    Bmrf,
    Bns,
    Bols,
    Brad,
    Brandom,
    Bspatial,
    Btree,
    Buser,
    Hinge,
)
from mboostrs.model_selection import (
    FAMILY_SETS,
    compare_models,
    family_set,
    gaic_table,
    profile_mstop,
    r_squared,
    scan_lss_families,
)
from mboostrs.modeler import (
    Assumption,
    AssumptionLedger,
    CausalDraft,
    LLMJudge,
    ModelerResult,
    SpecificationJudgment,
    Transcript,
    elicit_causal_spec,
    modeler_loop,
)
from mboostrs.policy import (
    CostBudget,
    FitCache,
    PairedTest,
    PolicyResult,
    Spent,
    auto_policy,
)
from mboostrs.rearrange import RearrangedQuantiles, rearrange_quantiles
from mboostrs.reference_gap import ReferenceGapResult, reference_gap
from mboostrs.report import ExecutionReport, PathRow
from mboostrs.residual_checks import (
    InfluenceFinding,
    RevisionFinding,
    RevisionReport,
    cluster_residual_check,
    influence_check,
    revision_checks,
    time_residual_check,
)
from mboostrs.result_json import dump_result_json, load_result_json
from mboostrs.search_session import SearchEvidence, SearchSession, open_search
from mboostrs.selection import (
    Deselection,
    EffectContrast,
    GamlssRefit,
    GamRefit,
    Probing,
    deselect,
    gam_refit,
    gamlss_refit,
    probe,
)
from mboostrs.serialize import load_model, save_model
from mboostrs.specification_curve import SpecificationCurve, specification_curve
from mboostrs.split_refit import SplitRefitResult, split_refit
from mboostrs.stabsel import StabSelResult, stabsel
from mboostrs.summary import Summary, SummaryTable, compare_summaries
from mboostrs.survival_evaluation import SurvivalEvaluationResult, evaluate_survival
from mboostrs.terms import (
    ColumnKind,
    DiscontinuityHit,
    LabelEvidence,
    ScreenedPair,
    classify_columns,
    column_screen,
    expand_pairs,
    label_evidence,
    screen_discontinuities,
    screen_interactions,
    suggest_terms,
)
from mboostrs.transforms import imputation_diagnostics

# isort: split
# `mboostrs.report` is deliberately two things at once. The stable
# execution-report submodule is imported by the lines above, and binding the
# `report(fit)` entry point (#474) here -- after that import -- replaces the
# package attribute with the function, exactly as `mboostrs.insights` does
# above. `from mboostrs.report import ExecutionReport` still works: the
# submodule is in `sys.modules`, so the import system finds it without
# consulting the shadowed package attribute, and `importlib.import_module`
# (which some tests use) goes through `sys.modules` too. The one visible
# consequence is that `import mboostrs.report; mboostrs.report` resolves to the
# function; use `from mboostrs.report import ...` for the submodule's names.
from mboostrs._procedure import DecisionTrail, Fit, fit, report

# Importing public names from these submodules causes Python to bind the
# submodule itself on the package.  They are implementation modules, not part
# of the flat public API (unlike ``datasets`` and ``plot`` above).
for _submodule in (
    "build", "causal", "compare", "conformal", "cvrisk", "diagnostics",
    "explain", "serialize",
):
    globals().pop(_submodule, None)
del _submodule

__version__ = "0.2.1"


def __getattr__(name):
    """Load `mboostrs.plot` the first time something asks for it.

    Importing it eagerly cost **509 ms of a 559 ms** `import mboostrs` -- 453
    of them inside `altair`, which pulls `jsonschema` and, through it,
    `rfc3987_syntax`, whose module body alone took 349 ms (measured with
    `python -X importtime`). Every caller paid that to fit a model, and a
    caller who never draws a chart paid it for nothing.

    `mboostrs.plot` still resolves on first use, so `mb.plot.plot_risk(fit)`
    and `from mboostrs import plot` are unchanged; only the moment of the
    import moves. PEP 562 module `__getattr__` is what makes that possible
    without a stub object standing in for the module.
    """
    if name == "plot":
        import importlib

        try:
            module = importlib.import_module("mboostrs.plot")
        except ImportError as exc:
            # An AttributeError, not the ImportError itself: hasattr() and
            # inspect.getmembers() -- the first calls an agent makes to
            # discover the API -- only skip a name whose lookup raises
            # AttributeError (#308). `from mboostrs import plot` and
            # `import mboostrs.plot` go through the import system and still
            # raise plot.py's ImportError with the install hint.
            raise AttributeError(str(exc)) from exc
        globals()["plot"] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted([*globals(), "plot"])

__all__ = [
    "dump_result_json",
    "load_result_json",
    "agent_guide",
    "api_card",
    "install_skill",
    "fit",
    "report",
    "Fit",
    "DecisionTrail",
    "Finding",
    "insights",
    "discover",
    "DiscoveryResult",
    "reference_gap",
    "ReferenceGapResult",
    "aipw",
    "AIPWResult",
    "CausalSpec",
    "NegativeControlResult",
    "causal_trap_report",
    "TrapFinding",
    "TrapReport",
    "confounding_benchmark",
    "confounding_sensitivity",
    "effect_model",
    "EffectModel",
    "ci_test",
    "CITest",
    "collect_warnings",
    "negative_control",
    "overlap_report",
    "OverlapReport",
    "rank_by_effect",
    "OverlapRanking",
    "effect_modifiers",
    "EffectModifiers",
    "BLPProjection",
    "iv_effect",
    "IVResult",
    "autospec",
    "check_declarations",
    "DeclarationCheck",
    "auto_policy",
    "CostBudget",
    "FitCache",
    "PairedTest",
    "PolicyResult",
    "Spent",
    "OrderedCategoryWarning",
    "CapacitySelection",
    "select_capacity",
    "compare_specifications",

    "Suggestion",
    "TermReport",
    "cv_risk",
    "DistributionEvaluationResult",
    "ContrastResult",
    "ElasticityResult",
    "ALE1DResult",
    "MarginalEffectResult",
    "PredictionProfileResult",
    "average_marginal_effect",
    "accumulated_local_effect",
    "average_elasticity",
    "distributional_contrast",
    "prediction_profile",
    "EvaluationResult",
    "evaluate_specification",
    "evaluate_distribution",
    "holdout",
    "prediction_contrast",
    "sensitivity",
    "suggest_specifications",
    "pair_term_for",
    "tune_mstop",
    "modeler_loop",
    "SearchEvidence",
    "SearchSession",
    "open_search",
    "SpecificationCurve",
    "specification_curve",
    "SurvivalEvaluationResult",
    "evaluate_survival",
    "ModelerResult",
    "SpecificationJudgment",
    "LLMJudge",
    "AssumptionLedger",
    "Assumption",
    "Transcript",
    "CausalDraft",
    "elicit_causal_spec",
    "MstopTuning",
    "deselect",
    "Deselection",
    "equivalence_sets",
    "EquivalenceSets",
    "substitution_sets",
    "SubstitutionSets",
    "selection_report",
    "SelectionReport",
    "probe",
    "Probing",
    "gam",
    "gam_refit",
    "GamRefit",
    "EffectContrast",
    "gamlss_refit",
    "GamlssRefit",
    "split_refit",
    "SplitRefitResult",
    "term_report",

    "ColumnKind",
    "ScreenedPair",
    "DiscontinuityHit",
    "classify_columns",
    "label_evidence",
    "LabelEvidence",
    "expand_pairs",
    "screen_interactions",
    "screen_discontinuities",
    "column_screen",
    "suggest_terms",

    "__version__",
    "boost_control",
    "build_info",
    "provenance",
    "BoostControl",
    "compare_summaries",
    "ExecutionReport",
    "PathRow",
    "capabilities",
    "family_capabilities",
    "learner_capabilities",
    "method_capabilities",
    "Summary",
    "SummaryTable",
    "validation_capabilities",
    "ConformalInterval",
    "load_model",
    "save_model",
    "conformal_interval",
    "rearrange_quantiles",
    "RearrangedQuantiles",
    "calibration_frame",
    "centile_fan_data",
    "coef_all",
    "coverage",
    "density_overlay_data",
    "deviance_increment",
    "deviance_profile",
    "ipc_weights",
    "interval_metrics",
    "outlier_frame",
    "pit",
    "predict_all",
    "q_stats",
    "quantile_residuals",
    "residual_frame",
    "same_family_gap",
    "SameFamilyGap",
    "worm_data",
    "glmboost",
    "GLMBoostModel",
    "estimate_cost",
    "gamboost",
    "GAMBoostModel",
    "Gaussian",
    "Binomial",
    "NBinomial",
    "Hurdle",
    "PropOdds",
    "Poisson",
    "Laplace",
    "GammaReg",
    "GaussReg",
    "AdaExp",
    "Quantile",
    "Expectile",
    "Huber",
    "CoxPH",
    "Weibull",
    "Loglog",
    "Lognormal",
    "HingeLoss",
    "Hinge",
    "Multinomial",
    "AUC",
    "Gehan",
    "Cindex",
    "BCCGLSS",
    "BCPELSS",
    "BCTLSS",
    "BetaLSS",
    "BinomialLSS",
    "DirichletLSS",
    "GammaLSS",
    "GaussianLSS",
    "GaussianKnownVarLSS",
    "GeometricLSS",
    "GumbelLSS",
    "ZeroAdjustedGammaLSS",
    "LogisticLSS",
    "InverseGaussianLSS",
    "JSULSS",
    "LogLogLSS",
    "LogNormalLSS",
    "LssCvRiskResult",
    "LssStabSelResult",
    "NBinomialLSS",
    "NegativeBinomialLSS",
    "Pareto2LSS",
    "PIGLSS",
    "PoissonLSS",
    "StudentTLSS",
    "WeibullLSS",
    "ZeroAdjustedPoissonLSS",
    "ZINBLSS",
    "ZIPoLSS",
    "LSSModel",
    "glmboostLSS",
    "gamboostLSS",
    "cvrisk_glmboostLSS",
    "cvrisk_gamboostLSS",
    "stabsel_glmboostLSS",
    "stabsel_gamboostLSS",
    "make_grid_lss",
    "Bols",
    "Bbs",
    "Bmrf",
    "Bmono",
    "Bns",
    "Bspatial",
    "Btree",
    "Brandom",
    "Bkernel",
    "Bconstrained",
    "Buser",
    "Brad",
    "Hinge",
    "expand_formula",
    "expand_interactions",
    "learner_design",
    "LearnerDesign",
    "parse_formula",
    "parse_gamboost_formula",
    "acf_residuals",
    "cv_weights",
    "heteroscedasticity",
    "cluster_residual_check",
    "imputation_diagnostics",
    "influence_check",
    "time_residual_check",
    "revision_checks",
    "RevisionFinding",
    "InfluenceFinding",
    "RevisionReport",
    "scale_absorption",
    "ScaleAbsorption",
    "diagonal_grid",
    "FoldScheme",
    "blocked_folds",
    "grouped_folds",
    "known_group_folds",
    "loo_folds",
    "StoppingBudget",
    "CvReport",
    "repeated_kfold_folds",
    "rolling_folds",
    "stratified_folds",
    "r_squared",
    "weighted_median",
    "make_grid",
    "CvRiskResult",
    "AicResult",
    "stabsel",
    "StabSelResult",
    "validate",
    "compare_models",
    "CompareReport",
    "CompareSpec",
    "compare_to_r",
    "gaic_table",
    "profile_mstop",
    "scan_lss_families",
    "FAMILY_SETS",
    "family_set",
    "llm_explain_prompt",
    "llm_describe",
    "datasets",
    "plot",
    "MboostrsError",
    "MboostrsValueError",
    "MboostrsRuntimeError",
    "MboostrsKeyError",
]
