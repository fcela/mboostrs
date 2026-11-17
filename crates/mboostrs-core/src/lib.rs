//! # mboostrs-core
//!
//! Pure-Rust component-wise functional gradient boosting engine,
//! re-implementing R's `mboost` (2.9.11) and `gamboostLSS` (2.2.0) semantics.

// Lints we deliberately allow crate-wide:
// - needless_range_loop: indexed `for i in 0..n` loops are clearer in the
//   numeric kernels (GEMV, B-spline, tree split) where the index addresses
//   several parallel buffers.
// - too_many_arguments: the `prepare_*` learner constructors take many
//   parameters by design (the R base-learner option surface).
// - manual_is_multiple_of / manual_div_ceil / manual_range_contains: clippy
//   suggests std methods stabilised only very recently; the explicit forms
//   keep the MSRV low.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::manual_range_contains)]
// Algorithm doc comments use intentional indentation for math derivations
// (KKT steps, Demmler–Reinsch); clippy's list-formatting opinions conflict.
#![allow(clippy::doc_overindented_list_items)]
#![allow(clippy::doc_lazy_continuation)]
#![warn(missing_docs)]

pub mod artifact;
pub mod build_info;
pub mod causal;
pub mod conformal;
pub mod contrast;
pub mod contribution;
pub mod control;
pub mod count_conformal;
pub mod cv;
pub mod declarations;
pub mod disc;
pub mod effect_modifiers;
pub mod engine;
#[cfg(test)]
pub mod engine_test_parity_perf;
/// Boosting families: loss, risk, and negative-gradient definitions.
pub mod families;
pub mod gam;
pub mod grid_search;
pub mod impute;
pub mod inference;
pub mod learners;
/// Linear-algebra primitives: packed Cholesky and the cached Gram solver.
pub mod linalg;
pub mod lss;
pub mod lss_distribution;
pub mod lss_quantile;
pub mod path;
pub mod preexplore;
pub mod radial;
pub mod rearrange;
pub mod report;
pub mod residual_structure;
/// Maximum-likelihood linear fit in the same family as a boost.
pub mod same_family;
pub mod scoring;
pub mod screen;
pub mod selection;
pub mod selection_stats;
pub mod special;
pub mod stabsel;
pub mod support;
pub mod survival;
pub mod transform;

pub use artifact::{Artifact, ArtifactError, FamilySpec, NewData, Scale};
pub use build_info::{build_info, BuildInfo, NATIVE_ABI_VERSION};
pub use control::{BoostControl, RiskKind, StepRule};
// The deprecated panicking wrappers stay re-exported until their 0.2.0 removal
// (P0.8 deprecation schedule); new code should use the try_* forms.
#[allow(deprecated)]
pub use cv::{
    cv, cvrisk, cvrisk_selected, cvrisk_selected_with, cvrisk_with, try_cv, try_cvrisk,
    try_cvrisk_extending, try_cvrisk_selected, try_cvrisk_selected_with,
    try_cvrisk_selected_with_offset_vec, try_cvrisk_with, CvRiskResult, CvSelectedResult,
    ExtendRule, ExtensionReport, FoldSpec, FoldType, LearnerPreparer,
};
#[cfg(feature = "instrumentation-baseline")]
#[doc(hidden)]
pub use engine::try_fit_without_report;
#[allow(deprecated)]
pub use engine::{
    compute_aic, compute_aic_trace, compute_aic_trace_fast, compute_aic_trace_from_hats,
    compute_hat_diag, fit, fit_profiled, fit_selected, fit_with, penalty_drift, try_compute_aic,
    try_compute_aic_trace, try_compute_aic_trace_fast, try_compute_aic_trace_fast_steps,
    try_compute_aic_trace_from_hats, try_compute_aic_trace_steps, try_compute_hat_diag,
    try_compute_hat_diag_steps, try_compute_hat_weighted_steps, try_fit, try_fit_continue,
    try_fit_profiled, try_fit_selected, try_fit_with, LearnerDrift, WarmStart,
};
pub use families::{
    held_out_row_losses, propodds_response, type7_quantile, weighted_quantile, AdaExp, Auc,
    Binomial, Cindex, CoxPH, ExpectReg, Family, GammaReg, GaussReg, Gaussian, Gehan, HingeLoss,
    Huber, Hurdle, Laplace, Loglog, Lognormal, Multinomial, NBinomial, Poisson, PropOdds, QuantReg,
    Weibull,
};
pub use learners::{
    bspline_basis, bspline_basis_extrap, bspline_basis_with_deriv, bspline_knots,
    build_mono_constraint, center_spectral, compute_centering_transform, cyclic_bspline_basis,
    cyclic_bspline_basis_from_knots, cyclic_bspline_knots, cyclic_diff_penalty, df_to_lambda,
    df_to_lambda_scalar, diff_penalty, kernel_epanechnikov, kernel_gaussian, kernel_uniform, kron,
    prepare_bbs, prepare_bbs_with_lambda, prepare_bkernel, prepare_block_bols,
    prepare_block_bols_shared_gram, prepare_bmono, prepare_bmrf, prepare_bns, prepare_bspatial,
    prepare_btree, prepare_buser, prepare_kronecker_bbs, prepare_kronecker_bbs_centered,
    prepare_kronecker_bols, prepare_scalar_bols, prepare_scalar_bols_fixed_center, BasisInfo,
    BmonoSolver, FittedTree, HatFactor, KroneckerRebuild, LearnerBlueprint, MonoConstraint,
    PreparedLearner, TreeNode, UnseenLevels,
};
pub use linalg::{CachedCholesky, CsrMatrix};
#[allow(deprecated)]
pub use path::{
    coef_gamboost, coef_glmboost, predict_glmboost, predict_glmboost_path, risk_rises,
    try_coef_gamboost, try_coef_glmboost, try_predict_glmboost, try_predict_glmboost_path,
    try_predict_glmboost_path_with_offset_vec, try_predict_glmboost_with_offset_vec, BoostingPath,
    CoefResult, GAMCoefResult,
};
pub use report::{
    EngineRoute, FallbackReason, FitReport, InstrumentationLevel, PathRow, PreparationTimer,
    PreparationTiming,
};
pub use stabsel::{
    d_rc, max_q, min_d, stabsel_parameters, try_d_rc, try_max_q, try_min_d, try_um_const, um_const,
    StabSelParams,
};

/// Top-level error type for all mboostrs operations.
#[derive(thiserror::Error, Debug)]
pub enum MboostrsError {
    #[error("shape mismatch: {0}")]
    /// Vector or matrix dimensions did not match what an operation requires.
    Shape(String),

    #[error("learner '{0}' rejected: {1}")]
    /// A learner was set up with invalid parameters or inconsistent lengths.
    LearnerSetup(String, String),

    #[error("family '{0}' rejected response: {1}")]
    /// The response values are invalid for the family's domain.
    BadResponse(String, String),

    #[error("solver failed: {0}")]
    /// The linear solver could not produce a finite solution.
    Solver(String),

    #[error("invalid control: {0}")]
    /// A boosting control parameter was out of range.
    Control(String),

    /// A family constructor was given a value outside its domain. Separate
    /// from [`MboostrsError::BadResponse`], which is about the data: this one
    /// is about the argument, and it is reported before any response is seen.
    #[error("family '{0}' rejected parameter: {1}")]
    BadFamilyParameter(String, String),

    /// A stability-selection parameter combination has no solution, or is
    /// outside the range the error bound is derived for. These used to be
    /// `Result<_, String>`, so a caller could only match on the text.
    #[error("stability selection rejected: {0}")]
    StabSel(String),

    /// A level-coded learner met a code the fit never saw, under a replay
    /// policy that refuses such codes. Separate from
    /// [`MboostrsError::Shape`] so that each frontend can name the argument
    /// that changes the policy.
    #[error(
        "unknown {learner} level {code} at observation {observation}; the code was not observed during fitting"
    )]
    UnseenLevel {
        /// Learner kind, for example `"Brandom"`.
        learner: &'static str,
        /// The level code as supplied.
        code: f64,
        /// Row of the first observation carrying the code.
        observation: usize,
    },
}

/// TEMPORARY experiment: faer parallelism from MBOOSTRS_FAER_TASKS ("none" or a count).
pub fn experiment_faer_par() -> faer::Parallelism<'static> {
    match std::env::var("MBOOSTRS_FAER_TASKS").ok().as_deref() {
        Some("none") => faer::Parallelism::None,
        Some(n) => faer::Parallelism::Rayon(n.parse().unwrap()),
        None => faer::Parallelism::Rayon(0),
    }
}
pub fn experiment_init() {
    faer::set_global_parallelism(experiment_faer_par());
}
