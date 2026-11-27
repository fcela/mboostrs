//! R bindings for mboostrs-core via extendr.
//!
//! Exposes the core boosting engine functions to R, providing
//! a fast Rust backend for gradient boosting that matches R mboost semantics.

// extendr-generated wrappers and the wide #[extendr] signatures trip these:
// - too_many_arguments: the fit entry mirrors the R-facing arg list
// - needless_range_loop: clearer in the numeric predict kernel
// - not_unsafe_ptr_arg_deref: extendr's &[f64] params are a safe wrapper pattern
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use extendr_api::prelude::*;

mod gam_refit;
mod selection_stats;
mod transform_bindings;
use mboostrs_core::control::{BoostControl, RiskKind};
use mboostrs_core::cv::{self, FoldSpec, FoldType};
use mboostrs_core::engine;
use mboostrs_core::families::{
    AdaExp, Auc, Binomial, Cindex, CoxPH, ExpectReg, Family, GammaReg, GaussReg, Gaussian, Gehan,
    HingeLoss, Huber, Laplace, Loglog, Lognormal, Multinomial, NBinomial, Poisson, PropOdds,
    QuantReg, Weibull,
};
use mboostrs_core::learners::{
    bspline_basis_with_deriv, build_bbs_template, df_to_lambda, prepare_bbs_from_template,
    prepare_bbs_with_lambda, prepare_block_bols, prepare_bns, prepare_bspatial,
    prepare_kronecker_bols, prepare_scalar_bols, prepare_scalar_bols_fixed_center, BasisInfo,
    BmonoSolver, KroneckerRebuild, LearnerBlueprint, MonoConstraint, PreparedLearner,
};
use std::collections::HashMap;
use std::time::Instant;

use mboostrs_core::{PreparationTimer, PreparationTiming};

/// Identity embedded in the loaded native R library.
#[extendr]
fn build_info_r() -> List {
    let info = mboostrs_core::build_info();
    list!(
        package_version = info.package_version,
        source_digest = info.source_digest,
        git_revision = info.git_revision,
        git_dirty = info.git_dirty,
        rustc = info.rustc,
        target = info.target,
        profile = info.profile,
        target_features = info.target_features,
        crate_features = info.crate_features,
        abi_version = info.abi_version as i32,
        build_id = info.build_id
    )
}

fn attach_frontend_timings(
    report: &mut mboostrs_core::FitReport,
    preparation: Option<PreparationTiming>,
    input_elapsed_ns: Option<u64>,
    output_started: Option<Instant>,
) {
    if let Some(preparation) = preparation {
        report.basis_ns = Some(preparation.basis_ns);
        report.factorization_ns = Some(preparation.factorization_ns);
    }
    report.binding_ns = input_elapsed_ns.map(|input_elapsed_ns| {
        input_elapsed_ns.saturating_sub(preparation.map_or(0, |value| value.total_ns))
            + output_started.map_or(0, |started| started.elapsed().as_nanos() as u64)
    });
}

fn fit_report_to_r(report: &mboostrs_core::FitReport) -> List {
    let fallbacks: Vec<&str> = report
        .fallback_reasons
        .iter()
        .map(|reason| reason.as_str())
        .collect();
    list!(
        schema_version = report.schema_version as i32,
        instrumentation = report.instrumentation.as_str(),
        route = report.route.as_str(),
        n_observations = report.n_observations as i32,
        n_learners = report.n_learners as i32,
        total_basis_columns = report.total_basis_columns as i32,
        requested_mstop = report.requested_mstop as i32,
        completed_mstop = report.completed_mstop as i32,
        effective_threads = report.effective_threads as i32,
        gram_eligible = report.gram_eligible,
        fallback_reasons = fallbacks,
        scalar_learner_count = report.scalar_learner_count as i32,
        block_learner_count = report.block_learner_count as i32,
        tree_learner_count = report.tree_learner_count as i32,
        cholesky_solver_count = report.cholesky_solver_count as i32,
        lu_solver_count = report.lu_solver_count as i32,
        regularized_solver_count = report.regularized_solver_count as i32,
        max_solver_jitter = report.max_solver_jitter.unwrap_or(f64::NAN),
        factor_cache_bytes = report.factor_cache_bytes as f64,
        gram_cache_bytes = report.gram_cache_bytes as f64,
        calibration_count = report.calibration_count as i32,
        tie_selection_count = report.tie_selection_count as i32,
        component_subsample_iterations = report.component_subsample_iterations as i32,
        calibration_converged_count = report.calibration_converged_count as i32,
        calibration_limit_count = report.calibration_limit_count as i32,
        calibration_failure_count = report.calibration_failure_count as i32,
        nonfinite_candidate_fallbacks = report.nonfinite_candidate_fallbacks as i32,
        setup_ns = report.setup_ns.map(|value| value as f64),
        basis_ns = report.basis_ns.map(|value| value as f64),
        factorization_ns = report.factorization_ns.map(|value| value as f64),
        cross_gram_ns = report.cross_gram_ns.map(|value| value as f64),
        iteration_ns = report.iteration_ns.map(|value| value as f64),
        reconstruction_ns = report.reconstruction_ns.map(|value| value as f64),
        binding_ns = report.binding_ns.map(|value| value as f64),
        total_ns = report.total_ns.map(|value| value as f64),
    )
}

/// Named checks for family parameters crossing the R binding.
///
/// The R constructors validate the same values, but a family also arrives
/// here from a serialized model and from cross-validation and
/// stability-selection dispatch, which do not go back through a constructor.
/// A value rejected when typed must be rejected when loaded.
fn r_param_unit_interval(family: &str, name: &str, value: f64) -> f64 {
    if !value.is_finite() || value <= 0.0 || value >= 1.0 {
        throw_r_error(format!("{family}: {name} must be in (0, 1), got {value}"));
    }
    value
}

fn r_param_positive(family: &str, name: &str, value: f64) -> f64 {
    if !value.is_finite() || value <= 0.0 {
        throw_r_error(format!("{family}: {name} must be positive, got {value}"));
    }
    value
}

/// A category count: a whole number at least `minimum`.
///
/// This used to be `value as usize` followed by `.max(2)`, which turned an
/// invalid count into a valid-looking one and fitted a different model than
/// the caller asked for.
fn r_param_category_count(family: &str, name: &str, value: f64, minimum: usize) -> usize {
    if !value.is_finite() || value.fract() != 0.0 || value < 0.0 {
        throw_r_error(format!(
            "{family}: {name} must be a whole number >= {minimum}, got {value}"
        ));
    }
    let count = value as usize;
    if count < minimum {
        throw_r_error(format!(
            "{family}: {name} must be at least {minimum}, got {count}"
        ));
    }
    count
}

/// A family parameter map's `response_scale`: absent fits the raw response
/// (R mboost), `0` asks for the automatic scale, a positive value fixes it.
fn family_response_scale(params: &HashMap<String, f64>) -> mboostrs_core::control::ResponseScale {
    use mboostrs_core::control::ResponseScale;
    match params.get("response_scale").copied() {
        None => ResponseScale::Raw,
        Some(0.0) => ResponseScale::Auto,
        Some(v) if v.is_finite() && v > 0.0 => ResponseScale::Fixed(v),
        Some(_) => ResponseScale::Raw,
    }
}

macro_rules! with_concrete_family {
    ($name:expr, $params:expr, |$fam:ident| $body:expr) => {{
        #[allow(unused_mut)]
        match $name {
            "gaussian" => {
                let mut $fam = Gaussian;
                $body
            }
            "gaussreg" => {
                let mut $fam = GaussReg;
                $body
            }
            "binomial" => {
                let mut $fam = Binomial;
                $body
            }
            "poisson" => {
                let mut $fam = Poisson;
                $body
            }
            "laplace" => {
                let mut $fam = Laplace::with_response_scale(family_response_scale(&$params));
                $body
            }
            "quantreg" => {
                let tau = r_param_unit_interval(
                    "quantreg",
                    "tau",
                    $params.get("tau").copied().unwrap_or(0.5),
                );
                let qoffset = r_param_unit_interval(
                    "quantreg",
                    "qoffset",
                    $params.get("qoffset").copied().unwrap_or(0.5),
                );
                let mut $fam = QuantReg::with_qoffset(tau, qoffset)
                    .with_response_scale(family_response_scale(&$params));
                $body
            }
            "expectreg" => {
                let tau = r_param_unit_interval(
                    "expectreg",
                    "tau",
                    $params.get("tau").copied().unwrap_or(0.5),
                );
                let mut $fam = ExpectReg::new(tau);
                $body
            }
            "huber" => {
                let delta = $params
                    .get("delta")
                    .copied()
                    .map(|d| r_param_positive("huber", "delta", d));
                let mut $fam = Huber::new(delta);
                $body
            }
            "gammasreg" => {
                let mut $fam = GammaReg::new();
                $body
            }
            "nbinomial" => {
                let mut $fam = NBinomial::new();
                $body
            }
            "hurdle" => {
                let mut $fam = mboostrs_core::Hurdle::new();
                $body
            }
            "propodds" => {
                // A supplied count names ordered categories, so it needs at
                // least three: two is a binary response, which Binomial fits.
                let n_levels = match $params.get("n_levels").copied() {
                    Some(raw) => r_param_category_count("propodds", "n_levels", raw, 3),
                    None => 2,
                };
                let mut $fam = PropOdds::new(n_levels);
                $body
            }
            "adaexp" => {
                let mut $fam = AdaExp;
                $body
            }
            "coxph" => {
                let mut $fam = CoxPH::new();
                $body
            }
            "weibull" => {
                let mut $fam = Weibull::new();
                $body
            }
            "loglog" => {
                let mut $fam = Loglog::new();
                $body
            }
            "lognormal" => {
                let mut $fam = Lognormal::new();
                $body
            }
            "hingeloss" => {
                let alpha_cost = r_param_unit_interval(
                    "hingeloss",
                    "alpha_cost",
                    $params.get("alpha_cost").copied().unwrap_or(0.5),
                );
                let mut $fam = HingeLoss::new(alpha_cost);
                $body
            }
            "auc" => {
                let mut $fam = Auc;
                $body
            }
            "gehan" => {
                let mut $fam = Gehan;
                $body
            }
            "cindex" => {
                let sigma = r_param_positive(
                    "cindex",
                    "sigma",
                    $params.get("sigma").copied().unwrap_or(0.1),
                );
                let mut $fam = Cindex::new(sigma);
                $body
            }
            "multinomial" => {
                let n_classes = match $params.get("n_classes").copied() {
                    Some(raw) => r_param_category_count("multinomial", "n_classes", raw, 3),
                    None => 3,
                };
                let mut $fam = Multinomial::new(n_classes);
                $body
            }
            other => throw_r_error(format!("Unknown family: {other}")),
        }
    }};
}

fn parse_risk_kind(risk: &str) -> RiskKind {
    match risk {
        "inbag" => RiskKind::Inbag,
        "oobag" => RiskKind::Oobag,
        "none" => RiskKind::None,
        other => throw_r_error(format!(
            "Unknown risk kind: {other}. Use 'inbag', 'oobag', or 'none'."
        )),
    }
}

/// Apply stochastic component subsampling from R-passed arguments.
///
/// `component_fraction` is off unless it is finite and strictly inside (0, 1);
/// R passes `NA_real_` (mapped to NaN) or a non-positive value for "off".
/// `subsample_seed` is a non-negative integer carried as `f64`.
fn apply_r_subsampling(
    ctrl: BoostControl,
    component_fraction: f64,
    subsample_seed: f64,
) -> BoostControl {
    if !component_fraction.is_finite() || component_fraction <= 0.0 || component_fraction >= 1.0 {
        return ctrl;
    }
    let seed = if subsample_seed.is_finite() && subsample_seed >= 0.0 {
        subsample_seed as u64
    } else {
        0
    };
    ctrl.with_component_subsampling(component_fraction, seed)
        .unwrap_or_else(|e| throw_r_error(format!("invalid subsampling: {e}")))
}

/// Convert an R `mstop` (passed as `i32`) to `u32`, rejecting non-positive
/// values. A bare `mstop as u32` would wrap a negative `i32` to a huge `u32`
/// (e.g. -1 -> 4_294_967_295), producing an effectively infinite fit.
fn checked_mstop(mstop: i32) -> u32 {
    if mstop <= 0 {
        throw_r_error("mstop must be a positive integer");
    }
    mstop as u32
}

fn apply_r_step(control: BoostControl, step: &str, iterations: i32) -> BoostControl {
    if iterations < 1 {
        throw_r_error("step_iterations must be a positive integer");
    }
    let rule = match step {
        "fixed" => mboostrs_core::control::StepRule::Fixed,
        "line_search" => mboostrs_core::control::StepRule::LineSearch {
            max_iter: iterations as usize,
        },
        _ => throw_r_error("step must be 'fixed' or 'line_search'"),
    };
    control.with_step(rule)
}

/// Box a concrete family by name (for `LearnerPreparer::family`).
fn make_boxed_family(name: &str, params: &HashMap<String, f64>) -> Box<dyn Family> {
    with_concrete_family!(name, params, |fam| fam.clone_family())
}

/// Binary class indices, one-based for indexing the R factor levels.
#[extendr]
fn family_class_r(family_name: &str, family_params: List, f: &[f64]) -> Vec<i32> {
    let family = make_boxed_family(family_name, &extract_family_params(&family_params));
    let indices = family
        .class(ndarray::ArrayView1::from(f))
        .unwrap_or_else(|| {
            throw_r_error(format!(
            "type = 'class' requires a binary family (Binomial, AdaExp, HingeLoss or AUC), got {}",
            family.name()
        ))
        });
    indices.into_iter().map(|i| i as i32 + 1).collect()
}

/// Centered glmboost design, returned in row-major order.
#[extendr]
fn glmboost_design_r(
    x: &[f64],
    n: i32,
    p: i32,
    centering_means: &[f64],
    intercept: bool,
) -> Vec<f64> {
    if n < 0 || p < 0 {
        throw_r_error("design dimensions must be non-negative");
    }
    let view = ndarray::ArrayView2::from_shape((n as usize, p as usize), x)
        .unwrap_or_else(|e| throw_r_error(format!("invalid design dimensions: {e}")));
    let design = mboostrs_core::path::try_glmboost_design(view, centering_means, intercept)
        .unwrap_or_else(|e| throw_r_error(e.to_string()));
    design.into_raw_vec_and_offset().0
}

/// `LearnerPreparer` for glmboost CV: rebuilds the intercept + per-column
/// `bols` learners against each fold's weights, mirroring the Python binding's
/// `GlmBoostPreparer` so the R `cvrisk` uses the same parity-tested core path.
struct GlmBoostPreparerR {
    x_cols: Vec<Vec<f64>>,
    names: Vec<String>,
    family_name: String,
    family_params: HashMap<String, f64>,
    column_means: Vec<f64>,
    /// Whether the constant learner is offered (false for `y ~ x - 1`).
    intercept: bool,
}

impl cv::LearnerPreparer for GlmBoostPreparerR {
    fn prepare(&self, weights: ndarray::ArrayView1<f64>) -> Vec<PreparedLearner> {
        let n = weights.len();
        let ones = ndarray::Array1::ones(n);
        let mut learners: Vec<PreparedLearner> = Vec::with_capacity(self.x_cols.len() + 1);
        if self.intercept {
            learners.push(prepare_scalar_bols(
                ones.view(),
                weights,
                0.0,
                false,
                "(Intercept)",
                None,
            ));
        }
        for (j, col) in self.x_cols.iter().enumerate() {
            let view = ndarray::ArrayView1::from(col.as_slice());
            learners.push(prepare_scalar_bols_fixed_center(
                view,
                weights,
                0.0,
                self.column_means[j],
                self.names.get(j).map(|s| s.as_str()).unwrap_or("?"),
            ));
        }
        learners
    }

    fn family(&self) -> Box<dyn Family> {
        make_boxed_family(&self.family_name, &self.family_params)
    }
}

/// An R `offset` argument: `NaN` (none), one number, or one entry per
/// fitted coordinate (`log(exposure)`), which boosting never re-estimates.
enum ROffset {
    None,
    Scalar(f64),
    Vector(ndarray::Array1<f64>),
}

impl ROffset {
    fn from_r(values: &[f64]) -> Self {
        match values {
            [] => ROffset::None,
            [v] if v.is_nan() => ROffset::None,
            [v] => ROffset::Scalar(*v),
            many => {
                check_finite(many, "offset");
                ROffset::Vector(ndarray::Array1::from_vec(many.to_vec()))
            }
        }
    }

    fn scalar(&self) -> Option<f64> {
        match self {
            ROffset::Scalar(v) => Some(*v),
            _ => None,
        }
    }

    fn vector(&self) -> Option<ndarray::ArrayView1<'_, f64>> {
        match self {
            ROffset::Vector(v) => Some(v.view()),
            _ => None,
        }
    }
}

/// The starting linear predictor of a prediction over `n` rows.
fn eta_from_offset(offset: &[f64], n: usize) -> Vec<f64> {
    match offset {
        [v] => vec![*v; n],
        many if many.len() == n => many.to_vec(),
        many => throw_r_error(format!(
            "offset has {} entries but {n} rows are predicted; supply one per row",
            many.len()
        )),
    }
}

fn check_finite(slice: &[f64], label: &str) {
    if let Some(pos) = slice.iter().position(|x| !x.is_finite()) {
        throw_r_error(format!(
            "{label} contains non-finite value at index {pos}: {}",
            slice[pos]
        ));
    }
}

fn extract_family_params(family_params: &List) -> HashMap<String, f64> {
    let mut map = HashMap::new();
    for (key, val) in family_params.iter() {
        // `Robj::as_real()` returns `None` for R's NA_REAL (it is a non-finite
        // sentinel, not a valid double), which would otherwise fall through
        // silently and let the family substitute its default. Reject it
        // explicitly instead.
        if val.is_na() {
            throw_r_error(format!("family parameter '{key}' must be finite, got NA"));
        }
        if let Some(v) = val.as_real() {
            if !v.is_finite() {
                throw_r_error(format!("family parameter '{key}' must be finite, got {v}"));
            }
            map.insert(key.to_string(), v);
        }
    }
    map
}

fn parse_mono_constraint(s: &str) -> MonoConstraint {
    match s {
        "increasing" => MonoConstraint::Increasing,
        "decreasing" => MonoConstraint::Decreasing,
        "convex" => MonoConstraint::Convex,
        "concave" => MonoConstraint::Concave,
        "positive" => MonoConstraint::Positive,
        "negative" => MonoConstraint::Negative,
        "constrained" => MonoConstraint::Constrained,
        other => throw_r_error(format!("Unknown mono constraint: {other}")),
    }
}

fn parse_bmono_solver(s: &str) -> BmonoSolver {
    match s {
        "iterative" => BmonoSolver::Iterative,
        "quad.prog" | "quadprog" => BmonoSolver::QuadProg,
        other => throw_r_error(format!("Unknown bmono solver: {other}")),
    }
}

/// Get a named element from an R list by name.
/// Returns the Robj value or throws an R error.
fn list_named_elt(list: &List, name: &str) -> Robj {
    if let Some(names) = list.names() {
        let names_vec: Vec<&str> = names.collect();
        if let Some(pos) = names_vec.iter().position(|&n| n == name) {
            return list.elt(pos).unwrap_or_else(|_| {
                throw_r_error(format!("list element '{}' not accessible", name))
            });
        }
    }
    throw_r_error(format!("list element '{}' not found", name));
}

fn list_optional_elt(list: &List, name: &str) -> Option<Robj> {
    let names = list.names()?;
    for (position, candidate) in names.enumerate() {
        if candidate == name {
            return list.elt(position).ok();
        }
    }
    None
}

// ─── glmboost ─────────────────────────────────────────────────────────

/// Builds the prepared learners of an R `glmboost` fit: an optional constant
/// and one centered linear term per column of `x_arr`. `fit_glmboost` and
/// `glmboost_hat_r` share it.
fn prepare_glmboost_learners_r(
    x_arr: &ndarray::Array2<f64>,
    w_owned: &ndarray::Array1<f64>,
    ones: &ndarray::Array1<f64>,
    centering_means: &[f64],
    intercept: bool,
) -> Vec<PreparedLearner> {
    let p = x_arr.ncols();
    let mut learners: Vec<PreparedLearner> = Vec::with_capacity(p + 1);
    // `y ~ x - 1`: no constant learner, as R's model.matrix drops the column.
    if intercept {
        learners.push(prepare_scalar_bols(
            ones.view(),
            w_owned.view(),
            0.0,
            false,
            "(Intercept)",
            None,
        ));
    }
    for j in 0..p {
        let col = x_arr.column(j).to_owned();
        let mean = centering_means.get(j).copied().unwrap_or(0.0);
        learners.push(prepare_scalar_bols_fixed_center(
            col.view(),
            w_owned.view(),
            0.0,
            mean,
            &format!("V{}", j),
        ));
    }
    learners
}

/// The boosting hat matrix of a fit, as mboost's `hatvalues` forms it: the
/// trace after every step and the final diagonal. `selected` is 1-based and
/// `coefs` holds each step's coefficients. For Binomial and Poisson each step
/// is weighted by the IRLS weights at the fit after that step, rebuilt here
/// from `offset` (length 1 or n) and the coefficient path; other families use
/// unit weights. A `btree` step's matrix is built from the partition encoded
/// in its coefficients. `NULL` when a selected learner has no hat matrix.
fn hat_path_r(
    learners: &[PreparedLearner],
    selected: &[i32],
    coefs: &List,
    offset: &[f64],
    nu: f64,
    family: &str,
) -> Robj {
    let n = learners.first().map(|l| l.n()).unwrap_or(0);
    let selected: Vec<usize> = selected
        .iter()
        .map(|&s| {
            if s < 1 || s as usize > learners.len() {
                throw_r_error(format!(
                    "selected index {s} is outside 1..{}",
                    learners.len()
                ));
            }
            (s - 1) as usize
        })
        .collect();
    if coefs.len() != selected.len() {
        throw_r_error(format!(
            "{} coefficient vectors for {} steps",
            coefs.len(),
            selected.len()
        ));
    }
    // The fitted path itself, which the weighted recursion needs for its
    // step fits and the hat resolution needs for a tree step's partition.
    let step_coefs: Vec<Vec<f64>> = (0..selected.len())
        .map(|m| {
            coefs
                .elt(m)
                .ok()
                .and_then(|c| c.as_real_vector())
                .unwrap_or_else(|| throw_r_error(format!("coefs[[{}]] is not numeric", m + 1)))
        })
        .collect();
    let weighted = mboostrs_core::engine::glm_hat_weight(family, 0.0).is_some();
    let weights: Vec<Vec<f64>> = if weighted {
        let mut f: Vec<f64> = match offset.len() {
            1 => vec![offset[0]; n],
            m if m == n => offset.to_vec(),
            m => throw_r_error(format!("offset has {m} entries for {n} rows")),
        };
        let mut step = vec![0.0; n];
        selected
            .iter()
            .enumerate()
            .map(|(m, &sel)| {
                let beta = &step_coefs[m];
                learners[sel].predict_into(beta, &mut step);
                f.iter_mut().zip(&step).for_each(|(fi, si)| *fi += nu * si);
                f.iter()
                    .map(|&fi| mboostrs_core::engine::glm_hat_weight(family, fi).unwrap_or(1.0))
                    .collect()
            })
            .collect()
    } else {
        vec![vec![1.0; n]; selected.len()]
    };
    match mboostrs_core::engine::try_compute_hat_weighted_steps(
        learners,
        &selected,
        Some(&step_coefs),
        nu,
        &weights,
    ) {
        Ok(Some((trace, diag))) => list!(trace = trace, diag = diag).into(),
        Ok(None) => Robj::from(()),
        Err(e) => throw_r_error(format!("{e}")),
    }
}

/// A `glmboost` fit's coefficients, as mboost's `coef.glmboost` reports them:
/// the intercept corrected for the centered predictors, the offset added to
/// it when `off2int`, and only nonzero entries unless `which` names them.
/// `selected` and `which` are 1-based learner positions with the intercept
/// first; returns 1-based `indices` and their `values`.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn coef_glmboost_r(
    selected: &[i32],
    coefs: &[f64],
    nu: f64,
    n_learners: i32,
    centering_means: &[f64],
    off2int: bool,
    offset: f64,
    which: &[i32],
) -> List {
    let to_zero_based = |values: &[i32]| -> Vec<usize> {
        values
            .iter()
            .map(|&v| {
                if v < 1 {
                    throw_r_error(format!("learner position {v} must be at least 1"));
                }
                (v - 1) as usize
            })
            .collect()
    };
    let selected = to_zero_based(selected);
    let which = (!which.is_empty()).then(|| to_zero_based(which));
    let result = mboostrs_core::try_coef_glmboost(
        &selected,
        coefs,
        nu,
        n_learners.max(0) as usize,
        centering_means,
        off2int,
        offset,
        which.as_deref(),
    )
    .unwrap_or_else(|e| throw_r_error(format!("{e}")));
    let indices: Vec<i32> = result.indices.iter().map(|&i| i as i32 + 1).collect();
    list!(indices = indices, values = result.values)
}

/// Each requested learner of an R `gamboost` fit as the core prepared it:
/// `design` (n x p), `penalty` (lambda * K, p x p), `lambda`, `df` (the
/// `tr(2H - HᵀH)` mboost reports) and `centering` (the value a single-column
/// linear learner subtracts; `NA` otherwise). `which` is 1-based.
#[extendr]
fn gamboost_learner_info_r(
    learner_specs: List,
    weights: &[f64],
    multinomial_n_classes: i32,
    which: &[i32],
) -> List {
    let w_owned = ndarray::Array1::from_vec(weights.to_vec());
    let learners = prepare_gamboost_learners_r(&learner_specs, w_owned, multinomial_n_classes);
    let as_matrix = |m: Option<ndarray::Array2<f64>>| -> Robj {
        match m {
            Some(m) => {
                let (rows, cols) = m.dim();
                let mut out = RMatrix::<f64>::new_matrix(rows, cols, |r, c| m[[r, c]]);
                out.set_attrib("dim", [rows as i32, cols as i32])
                    .expect("dim attribute");
                out.into()
            }
            None => Robj::from(()),
        }
    };
    let items: Vec<Robj> = which
        .iter()
        .map(|&w| {
            let learner = learners
                .get((w - 1).max(0) as usize)
                .filter(|_| w >= 1)
                .unwrap_or_else(|| throw_r_error(format!("learner position {w} is out of range")));
            let centering = match learner {
                PreparedLearner::ScalarBols { centering, .. }
                | PreparedLearner::BinnedScalarBols { centering, .. } => *centering,
                _ => f64::NAN,
            };
            list!(
                design = as_matrix(learner.design_matrix()),
                penalty = as_matrix(learner.penalty_matrix()),
                lambda = learner.lambda().unwrap_or(f64::NAN),
                df = learner.df_mboost().unwrap_or(f64::NAN),
                centering = centering
            )
            .into()
        })
        .collect();
    List::from_values(items)
}

/// Hat matrix of an R `gamboost` fit; see `hat_path_r`.
#[extendr]
fn gamboost_hat_r(
    learner_specs: List,
    weights: &[f64],
    multinomial_n_classes: i32,
    selected: &[i32],
    coefs: List,
    offset: &[f64],
    nu: f64,
    family: &str,
) -> Robj {
    let w_owned = ndarray::Array1::from_vec(weights.to_vec());
    let learners = prepare_gamboost_learners_r(&learner_specs, w_owned, multinomial_n_classes);
    hat_path_r(&learners, selected, &coefs, offset, nu, family)
}

/// First-order approximation of an R `gamboost` fit's hat diagonal; see
/// `mboostrs_core::engine::try_approximate_hat_diag`. `selected` is 1-based.
/// `NULL` when a selected learner has no hat matrix.
#[extendr]
fn gamboost_hat_approx_r(
    learner_specs: List,
    weights: &[f64],
    multinomial_n_classes: i32,
    selected: &[i32],
    nu: f64,
) -> Robj {
    let w_owned = ndarray::Array1::from_vec(weights.to_vec());
    let learners = prepare_gamboost_learners_r(&learner_specs, w_owned, multinomial_n_classes);
    let selected: Vec<usize> = selected
        .iter()
        .map(|&s| {
            if s < 1 {
                throw_r_error(format!("selected index {s} is below 1"));
            }
            (s - 1) as usize
        })
        .collect();
    match mboostrs_core::engine::try_approximate_hat_diag(&learners, &selected, nu) {
        Ok(Some(h)) => Robj::from(h),
        Ok(None) => Robj::from(()),
        Err(e) => throw_r_error(format!("{e}")),
    }
}

/// One learner's partial effect at the rows of `design` (row-major, `p`
/// columns); see `mboostrs_core::contribution::try_learner_effect`.
/// `selected` and `learner` are 1-based. `NULL` when a selected step's
/// coefficients do not have `p` entries.
#[extendr]
fn r_learner_effect(
    design: &[f64],
    p: i32,
    centering: &[f64],
    selected: &[i32],
    coefs: List,
    learner: i32,
    nu: f64,
) -> Robj {
    let step_coefs: Vec<Vec<f64>> = (0..coefs.len())
        .map(|m| {
            coefs
                .elt(m)
                .ok()
                .and_then(|c| c.as_real_vector())
                .unwrap_or_else(|| throw_r_error(format!("coefs[[{}]] is not numeric", m + 1)))
        })
        .collect();
    // An unselected index can never match `learner`, so 0 stands in for it.
    let selected: Vec<usize> = selected.iter().map(|&s| s.max(0) as usize).collect();
    match mboostrs_core::contribution::try_learner_effect(
        design,
        p.max(0) as usize,
        centering,
        &selected,
        &step_coefs,
        learner.max(1) as usize,
        nu,
    ) {
        Ok(Some(effect)) => Robj::from(effect),
        Ok(None) => Robj::from(()),
        Err(e) => throw_r_error(format!("{e}")),
    }
}

/// Per-group partial effect of a grouped learner (`brandom`, `bmrf`) of an
/// R `gamboost` fit: [`gamboost_grouped_contribution_r`] at the identity
/// indicator rows. For `brandom` that is `b_g - sum_j c_j b_j`, with `c` the
/// column centering the core applied (nonzero for `center = TRUE`); for
/// `bmrf(center = TRUE)` it is row `g` of `Z b`. `learner`, `selected` are
/// 1-based. `NULL` when the coefficients do not match the learner's design.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn gamboost_group_effect_r(
    learner_specs: List,
    weights: &[f64],
    multinomial_n_classes: i32,
    learner: i32,
    n_groups: i32,
    selected: &[i32],
    coefs: List,
    nu: f64,
) -> Robj {
    let p = n_groups.max(0) as usize;
    let identity: Vec<f64> = (0..p * p)
        .map(|k| if k / p == k % p { 1.0 } else { 0.0 })
        .collect();
    grouped_effect_r(
        &learner_specs,
        weights,
        multinomial_n_classes,
        learner,
        &identity,
        n_groups,
        &[],
        selected,
        &coefs,
        nu,
    )
}

/// Contribution of a grouped learner (`brandom`, `bmrf`) of an R `gamboost`
/// fit at rows given as group indicators (row-major, `n_groups` columns),
/// each row scaled by `by_values` (empty for none); see
/// `mboostrs_core::contribution::try_grouped_learner_effect`. The column
/// centering is the one the core applies when it prepares the learner from
/// the fit's specification and weights, and a `bmrf(center = TRUE)` term
/// projects the indicators onto the penalized directions of its `K`.
/// `learner` and `selected` are 1-based. `NULL` when the coefficients do not
/// match the learner's design.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn gamboost_grouped_contribution_r(
    learner_specs: List,
    weights: &[f64],
    multinomial_n_classes: i32,
    learner: i32,
    indicators: &[f64],
    n_groups: i32,
    by_values: &[f64],
    selected: &[i32],
    coefs: List,
    nu: f64,
) -> Robj {
    grouped_effect_r(
        &learner_specs,
        weights,
        multinomial_n_classes,
        learner,
        indicators,
        n_groups,
        by_values,
        selected,
        &coefs,
        nu,
    )
}

#[allow(clippy::too_many_arguments)]
fn grouped_effect_r(
    learner_specs: &List,
    weights: &[f64],
    multinomial_n_classes: i32,
    learner: i32,
    indicators: &[f64],
    n_groups: i32,
    by_values: &[f64],
    selected: &[i32],
    coefs: &List,
    nu: f64,
) -> Robj {
    if learner < 1 || learner as usize > learner_specs.len() {
        throw_r_error(format!("learner position {learner} is out of range"));
    }
    let spec_robj = learner_specs.elt((learner - 1) as usize).expect("elt");
    let spec = spec_robj
        .as_list()
        .unwrap_or_else(|| throw_r_error(format!("learner_specs[[{learner}]] is not a list")));
    // Only this learner is prepared: its centering depends on its own
    // specification and the weights.
    let one = List::from_values([spec_robj.clone()]);
    let w_owned = ndarray::Array1::from_vec(weights.to_vec());
    let prepared = prepare_gamboost_learners_r(&one, w_owned, multinomial_n_classes);
    let p = n_groups.max(0) as usize;
    let centering: Vec<f64> = match prepared.first() {
        Some(PreparedLearner::BlockBols { centering, .. }) if centering.len() == p => {
            centering.to_vec()
        }
        _ => Vec::new(),
    };
    let is_bmrf = list_named_elt(&spec, "type").as_str() == Some("bmrf");
    let centered = list_optional_elt(&spec, "center")
        .and_then(|c| c.as_bool())
        .unwrap_or(false);
    let penalty = if is_bmrf && centered {
        let k_flat = list_named_elt(&spec, "K")
            .as_real_vector()
            .unwrap_or_else(|| throw_r_error("bmrf $K not numeric"));
        Some(
            ndarray::Array2::from_shape_vec((p, p), k_flat)
                .unwrap_or_else(|e| throw_r_error(format!("bmrf K shape error: {e}"))),
        )
    } else {
        None
    };
    let step_coefs: Vec<Vec<f64>> = (0..coefs.len())
        .map(|m| {
            coefs
                .elt(m)
                .ok()
                .and_then(|c| c.as_real_vector())
                .unwrap_or_else(|| throw_r_error(format!("coefs[[{}]] is not numeric", m + 1)))
        })
        .collect();
    // An unselected index can never match `learner`, so 0 stands in for it.
    let selected: Vec<usize> = selected.iter().map(|&s| s.max(0) as usize).collect();
    match mboostrs_core::contribution::try_grouped_learner_effect(
        indicators,
        p,
        by_values,
        &centering,
        penalty.as_ref(),
        &selected,
        &step_coefs,
        learner as usize,
        nu,
    ) {
        Ok(Some(effect)) => Robj::from(effect),
        Ok(None) => Robj::from(()),
        Err(e) => throw_r_error(format!("{e}")),
    }
}

/// Hat matrix of an R `glmboost` fit; see `hat_path_r`.
#[extendr]
fn glmboost_hat_r(
    x_matrix: &[f64],
    n: i32,
    p: i32,
    weights: &[f64],
    centering_means: &[f64],
    intercept: bool,
    selected: &[i32],
    coefs: List,
    offset: &[f64],
    nu: f64,
    family: &str,
) -> Robj {
    let (n, p) = (n as usize, p as usize);
    let x_arr = ndarray::Array2::from_shape_vec((n, p), x_matrix.to_vec())
        .unwrap_or_else(|e| throw_r_error(format!("invalid x_matrix dimensions: {e}")));
    let w_owned = ndarray::Array1::from_vec(weights.to_vec());
    let ones = ndarray::Array1::ones(n);
    let learners = prepare_glmboost_learners_r(&x_arr, &w_owned, &ones, centering_means, intercept);
    hat_path_r(&learners, selected, &coefs, offset, nu, family)
}

/// The fit's nuisance value for R: the vector nuisance (PropOdds thresholds)
/// when the family has one, otherwise the scalar one (NBinomial and Hurdle
/// dispersion, GammaReg shape, survival scale) after the last step, as
/// mboost's `nuisance()` reports it.
fn nuisance_for_r(path: &mboostrs_core::BoostingPath) -> Vec<f64> {
    if !path.nuisance_vec.is_empty() {
        return path.nuisance_vec.clone();
    }
    path.nuisance_path
        .last()
        .copied()
        .flatten()
        .map(|value| vec![value])
        .unwrap_or_default()
}

#[extendr]
fn fit_glmboost(
    x_matrix: &[f64],
    n: i32,
    p: i32,
    y: &[f64],
    weights: &[f64],
    mstop: i32,
    nu: f64,
    family_name: &str,
    family_params: List,
    centering_means: &[f64],
    offset_val: &[f64],
    risk: &str,
    trace: bool,
    component_fraction: f64,
    subsample_seed: f64,
    intercept: bool,
    step: &str,
    step_iterations: i32,
) -> List {
    let binding_started = trace.then(Instant::now);
    let n = n as usize;
    let p = p as usize;
    let mstop = checked_mstop(mstop);
    if nu <= 0.0 || nu > 1.0 {
        throw_r_error(format!("nu must be in (0, 1], got {nu}"));
    }
    if x_matrix.len() != n * p {
        throw_r_error(format!(
            "x_matrix length {} does not match n={} * p={}",
            x_matrix.len(),
            n,
            p
        ));
    }
    if centering_means.len() != p {
        throw_r_error(format!(
            "centering_means length {} does not match p={}",
            centering_means.len(),
            p
        ));
    }

    check_finite(y, "y");
    check_finite(weights, "weights");
    check_finite(x_matrix, "x_matrix");

    let is_survival = matches!(
        family_name,
        "coxph" | "weibull" | "loglog" | "lognormal" | "gehan" | "cindex"
    );
    let n_y = if is_survival { y.len() / 2 } else { y.len() };

    if weights.len() != n_y {
        throw_r_error(format!(
            "weights length {} does not match n={}",
            weights.len(),
            n_y
        ));
    }

    let preparation_timer = trace.then(PreparationTimer::start);
    let x_arr = ndarray::Array2::from_shape_vec((n, p), x_matrix.to_vec())
        .unwrap_or_else(|e| throw_r_error(format!("invalid x_matrix dimensions: {e}")));

    let w_owned = ndarray::Array1::from_vec(weights.to_vec());
    let ones = ndarray::Array1::ones(if is_survival { n_y } else { n });

    let mut learners =
        prepare_glmboost_learners_r(&x_arr, &w_owned, &ones, centering_means, intercept);
    let preparation_timing = preparation_timer.map(PreparationTimer::finish);

    // The family owns any internal response encoding so direct Rust, Python,
    // R, fit, and resampling routes execute one contract.
    let y_arr = ndarray::Array1::from_vec(y.to_vec());
    let risk_kind = parse_risk_kind(risk);
    let ctrl = BoostControl::new(mstop, nu)
        .unwrap_or_else(|e| throw_r_error(format!("invalid BoostControl: {e}")))
        .with_risk(risk_kind)
        .with_trace(trace);
    let ctrl = apply_r_subsampling(ctrl, component_fraction, subsample_seed);
    let ctrl = apply_r_step(ctrl, step, step_iterations);

    let offset = ROffset::from_r(offset_val);

    let params = extract_family_params(&family_params);
    // Validate the response up front (check_y was otherwise never called), so
    // invalid y errors clearly instead of producing NaN deep in the fit.
    make_boxed_family(family_name, &params)
        .check_y(y_arr.view())
        .unwrap_or_else(|e| throw_r_error(format!("invalid response: {e}")));
    let binding_input_ns = binding_started.map(|started| started.elapsed().as_nanos() as u64);
    let path = with_concrete_family!(family_name, &params, |fam| {
        match offset.vector() {
            Some(vector) => engine::try_fit_with_vector_offset(
                &mut learners,
                y_arr.view(),
                w_owned.view(),
                &mut fam,
                &ctrl,
                vector,
                false,
            ),
            None => engine::try_fit_with(
                &mut learners,
                y_arr.view(),
                w_owned.view(),
                &mut fam,
                &ctrl,
                offset.scalar(),
                false,
            ),
        }
    });
    let mut path = path.unwrap_or_else(|e| throw_r_error(format!("{e}")));

    let binding_output_started = trace.then(Instant::now);
    if path.risk.iter().any(|v| !v.is_finite()) || path.fitted.iter().any(|v| !v.is_finite()) {
        throw_r_error(
            "boosting fit diverged (non-finite risk/fitted values); \
             decrease the step-size nu in boost_control",
        );
    }

    let selected: Vec<i32> = path.selected.iter().map(|&s| (s + 1) as i32).collect();
    let coefs_flat: Vec<f64> = path.coefs.iter().map(|c| c[0]).collect();
    let nuisance_rows = path.nuisance_rows();
    let nuisance_vec_path_robj = extendr_api::List::from_values(
        nuisance_rows
            .iter()
            .map(|v| extendr_api::Robj::from(v.as_slice())),
    );
    attach_frontend_timings(
        &mut path.report,
        preparation_timing,
        binding_input_ns,
        binding_output_started,
    );

    let nuisance = nuisance_for_r(&path);
    list!(
        offset = path.offset,
        selected = selected,
        coefs = coefs_flat,
        risk = path.risk,
        fitted = path.fitted.to_vec(),
        nuisance = nuisance,
        nuisance_vec_path = nuisance_vec_path_robj,
        execution_report = fit_report_to_r(&path.report),
    )
}

#[extendr]
fn predict_glmboost(
    offset: &[f64],
    selected: &[i32],
    coefs: &[f64],
    nu: f64,
    x_matrix: &[f64],
    centering_means: &[f64],
    mstop: i32,
) -> Vec<f64> {
    if selected.len() != coefs.len() {
        throw_r_error(format!(
            "selected length {} does not match coefs length {}",
            selected.len(),
            coefs.len()
        ));
    }
    let k = if mstop <= 0 {
        selected.len()
    } else {
        (mstop as usize).min(selected.len())
    };

    let p = centering_means.len().max(1);
    let n = x_matrix.len() / p;

    if n * p != x_matrix.len() {
        throw_r_error(format!(
            "x_matrix length {} is not a multiple of p={}",
            x_matrix.len(),
            p
        ));
    }

    let mut eta = eta_from_offset(offset, n);

    for step in 0..k {
        let col_idx = (selected[step] - 1) as usize;
        if col_idx == 0 {
            for i in 0..n {
                eta[i] += nu * coefs[step];
            }
        } else if col_idx - 1 < p {
            let j = col_idx - 1;
            let centering = centering_means[j];
            for i in 0..n {
                let x_ij = x_matrix[i * p + j];
                eta[i] += nu * coefs[step] * (x_ij - centering);
            }
        }
    }

    eta
}

// ─── cv / cvrisk for glmboost ─────────────────────────────────────────

#[extendr]
fn cv_weights(n: i32, fold_type: &str, b: i32, prob: f64, seed: i32) -> List {
    if n <= 0 {
        throw_r_error("n must be a positive integer");
    }
    if b <= 0 {
        throw_r_error("b must be a positive integer");
    }
    let n = n as usize;
    let b = b as usize;
    let ft = match fold_type {
        "kfold" => FoldType::KFold,
        "bootstrap" => FoldType::Bootstrap,
        "subsampling" => FoldType::Subsampling,
        other => throw_r_error(format!("unknown fold type: {other}")),
    };
    let spec = FoldSpec::try_new(ft, b, prob, seed as u64)
        .unwrap_or_else(|e| throw_r_error(format!("{e}")));
    let w = ndarray::Array1::ones(n);
    let folds = cv::try_cv(&spec, n, w.as_slice().unwrap())
        .unwrap_or_else(|e| throw_r_error(format!("{e}")));
    // Return as B-length list of n-vectors (each is the oobag weight vector)
    let mut result = Vec::with_capacity(b);
    for col in 0..b {
        let mut fold_col = Vec::with_capacity(n);
        for row in 0..n {
            fold_col.push(folds[[row, col]]);
        }
        result.push(fold_col);
    }
    {
        let folds_robj = extendr_api::List::from_values(
            result.iter().map(|v| extendr_api::Robj::from(v.as_slice())),
        );
        list!(folds = folds_robj)
    }
}

#[extendr]
fn cvrisk_glmboost(
    x_matrix: &[f64],
    n: i32,
    p: i32,
    y: &[f64],
    weights: &[f64],
    mstop: i32,
    nu: f64,
    family_name: &str,
    family_params: List,
    learner_names: Vec<String>,
    folds_list: List,
    centering_means: &[f64],
    offset_val: &[f64],
    risk: &str,
    intercept: bool,
    step: &str,
    step_iterations: i32,
) -> List {
    let n_val = n as usize;
    let p_val = p as usize;
    let mstop_u = checked_mstop(mstop);

    check_finite(y, "y");
    check_finite(weights, "weights");
    check_finite(x_matrix, "x_matrix");

    let x_arr = ndarray::Array2::from_shape_vec((n_val, p_val), x_matrix.to_vec())
        .unwrap_or_else(|e| throw_r_error(format!("invalid x_matrix: {e}")));

    // Compute the per-fold OOB risk path via the core `cvrisk_with` — the same
    // implementation the Python binding uses and that is parity-tested against
    // R. The previous version recorded the per-fold *inbag* risk path (it
    // computed an OOB value then overwrote it with `path.risk`), which decreases
    // monotonically and selects the wrong mstop.
    let x_cols: Vec<Vec<f64>> = (0..p_val).map(|j| x_arr.column(j).to_vec()).collect();
    // Unweighted full-sample column means as fixed centers (matches Python's
    // cvrisk path and R's cvrisk).
    // The fit's own centering: column means, or zeros for center = FALSE.
    // Recomputing the means here refitted every fold of an uncentered model
    // as a centered one (0.30 from mboost's cvrisk).
    let column_means: Vec<f64> = if centering_means.len() == x_cols.len() {
        centering_means.to_vec()
    } else {
        x_cols
            .iter()
            .map(|c| c.iter().sum::<f64>() / c.len() as f64)
            .collect()
    };

    let params = extract_family_params(&family_params);

    let preparer = GlmBoostPreparerR {
        x_cols,
        names: learner_names,
        family_name: family_name.to_string(),
        family_params: params.clone(),
        column_means,
        intercept,
    };

    // Folds as an (n x b) matrix.
    let b = folds_list.len();
    let mut folds_arr = ndarray::Array2::<f64>::zeros((n_val, b));
    for i in 0..b {
        let col = folds_list
            .elt(i)
            .expect("elt")
            .as_real_vector()
            .unwrap_or_else(|| throw_r_error(format!("fold {} is not numeric", i + 1)));
        if col.len() != n_val {
            throw_r_error(format!(
                "fold {} has {} entries, expected {}",
                i + 1,
                col.len(),
                n_val
            ));
        }
        for row in 0..n_val {
            folds_arr[[row, i]] = col[row];
        }
    }

    let risk_kind = parse_risk_kind(risk);
    let ctrl = BoostControl::new(mstop_u, nu)
        .unwrap_or_else(|e| throw_r_error(format!("invalid BoostControl: {e}")))
        .with_risk(risk_kind);
    let ctrl = apply_r_step(ctrl, step, step_iterations);

    // A fitted scalar offset is recomputed on each fold (matching R's cvrisk
    // and the Python binding), so the R side sends NaN for it. A per-row
    // offset (log exposure) is known, not fitted, and starts every fold.
    let offset = ROffset::from_r(offset_val);
    let offset_vec = offset.vector();

    let y_owned = ndarray::Array1::from_vec(y.to_vec());
    let w_view = ndarray::ArrayView1::from(weights);

    let result = with_concrete_family!(family_name, &params, |fam| {
        cv::try_cvrisk_with(
            &preparer,
            y_owned.view(),
            w_view,
            &ctrl,
            &folds_arr,
            offset.scalar(),
            offset_vec,
            // No judge mask: the R cvrisk surface passes fold weight
            // matrices only. Rolling-origin judge sets are a Python
            // explore-layer concept; if the R side grows them, this is the
            // slot they thread through.
            None,
            false,
            fam,
            "bootstrap",
        )
    });
    let result = result.unwrap_or_else(|e| throw_r_error(format!("{e}")));

    let n_folds = result.risk_matrix.nrows();
    let n_cols = result.risk_matrix.ncols();
    let mut risk_flat = Vec::with_capacity(n_folds * n_cols);
    for i in 0..n_folds {
        for j in 0..n_cols {
            risk_flat.push(result.risk_matrix[[i, j]]);
        }
    }
    let aggregate: Vec<f64> = result.aggregate.to_vec();
    let optimal_mstop = result.optimal_mstop() as i32;

    list!(
        risk_matrix = risk_flat,
        n_folds = n_folds as i32,
        n_cols = n_cols as i32,
        optimal_mstop = optimal_mstop,
        aggregate = aggregate
    )
}

// ─── gamboost ─────────────────────────────────────────────────────────

/// Builds the prepared learners of an R `gamboost` learner specification
/// list with the given fit weights. `fit_gamboost` and `gamboost_hat_r`
/// share it, so the hat matrix is formed from the learners the fit used.
fn prepare_gamboost_learners_r(
    learner_specs: &List,
    w_owned: ndarray::Array1<f64>,
    multinomial_n_classes: i32,
) -> Vec<PreparedLearner> {
    let sw: f64 = w_owned.sum();
    let n_learners = learner_specs.len();
    let mut learners: Vec<PreparedLearner> = Vec::with_capacity(n_learners);

    for i in 0..n_learners {
        let spec = learner_specs
            .elt(i)
            .expect("elt")
            .as_list()
            .unwrap_or_else(|| throw_r_error(format!("learner_specs[[{}]] is not a list", i + 1)));

        let ltype = list_named_elt(&spec, "type")
            .as_str()
            .unwrap_or_else(|| throw_r_error(format!("learner_specs[[{}]]$type missing", i + 1)));
        let lname = list_named_elt(&spec, "name")
            .as_str()
            .unwrap_or_else(|| throw_r_error(format!("learner_specs[[{}]]$name missing", i + 1)));

        match ltype {
            "scalar" => {
                let design_vec = list_named_elt(&spec, "design")
                    .as_real_vector()
                    .unwrap_or_else(|| throw_r_error("scalar $design not numeric"));
                let lambda = list_named_elt(&spec, "lambda")
                    .as_real()
                    .unwrap_or_else(|| throw_r_error("scalar $lambda not numeric"));
                let center = list_named_elt(&spec, "center").as_bool().unwrap_or(false);

                let design = ndarray::Array1::from_vec(design_vec);
                // A "scalar" learner is a linear (bols) term: a single covariate
                // column fit by (optionally ridge-penalized) least squares.
                // lambda < 0 means unpenalized.
                let lam = if lambda >= 0.0 { lambda } else { 0.0 };
                learners.push(prepare_scalar_bols(
                    design.view(),
                    w_owned.view(),
                    lam,
                    center,
                    lname,
                    None,
                ));
            }
            "block" => {
                let design_flat = list_named_elt(&spec, "design")
                    .as_real_vector()
                    .unwrap_or_else(|| throw_r_error("block $design not numeric"));
                let penalty_flat = list_named_elt(&spec, "penalty")
                    .as_real_vector()
                    .unwrap_or_else(|| throw_r_error("block $penalty not numeric"));
                let dims = list_named_elt(&spec, "np")
                    .as_integer_vector()
                    .unwrap_or_else(|| throw_r_error("block $np not integer"));
                let center = list_named_elt(&spec, "center").as_bool().unwrap_or(false);

                let nr = dims[0] as usize;
                let nc = dims[1] as usize;
                if design_flat.len() != nr * nc {
                    throw_r_error(format!(
                        "block $design length {} != n*p = {}*{} = {}",
                        design_flat.len(),
                        nr,
                        nc,
                        nr * nc
                    ));
                }
                if penalty_flat.len() != nc * nc {
                    throw_r_error(format!(
                        "block $penalty length {} != p*p = {}*{} = {}",
                        penalty_flat.len(),
                        nc,
                        nc,
                        nc * nc
                    ));
                }

                let design = ndarray::Array2::from_shape_vec((nr, nc), design_flat)
                    .unwrap_or_else(|e| throw_r_error(format!("block design shape error: {e}")));
                let mut penalty = ndarray::Array2::from_shape_vec((nc, nc), penalty_flat)
                    .unwrap_or_else(|e| throw_r_error(format!("block penalty shape error: {e}")));
                // An optional `$df` asks for the penalty's lambda to be
                // calibrated here, against the fit weights, as `bbs(df=)` is;
                // without it the penalty is taken as given.
                if let Some(df) = list_named_elt(&spec, "df").as_real() {
                    if df > 0.0 {
                        let lambda = mboostrs_core::df_to_lambda(&design, &penalty, &w_owned, df);
                        penalty.mapv_inplace(|v| v * lambda);
                    }
                }

                learners.push(prepare_block_bols(
                    design,
                    w_owned.view(),
                    penalty,
                    center,
                    lname,
                ));
            }
            "bbs" => {
                if multinomial_n_classes > 1 {
                    // Smooth multinomial: build the raw B-spline basis with the
                    // (I + K1) per-class penalty and per-class df via
                    // prepare_kronecker_bbs (matches R's bbs(x) %O% bols(class)).
                    let x = list_named_elt(&spec, "x")
                        .as_real_vector()
                        .unwrap_or_else(|| throw_r_error("bbs $x not numeric"));
                    let knots = list_named_elt(&spec, "knots").as_real().unwrap_or(20.0) as usize;
                    let degree = list_named_elt(&spec, "degree").as_real().unwrap_or(3.0) as usize;
                    let differences = list_named_elt(&spec, "differences")
                        .as_real()
                        .unwrap_or(2.0) as usize;
                    let df = list_named_elt(&spec, "df").as_real().unwrap_or(4.0);
                    let n_resp = multinomial_n_classes as usize - 1;
                    let deriv = list_named_elt(&spec, "deriv").as_real().unwrap_or(0.0) as usize;
                    let center = list_named_elt(&spec, "center").as_bool().unwrap_or(false);
                    let xa = ndarray::Array1::from_vec(x);
                    learners.push(mboostrs_core::prepare_kronecker_bbs_centered(
                        xa.view(),
                        w_owned.view(),
                        knots,
                        degree,
                        differences,
                        df,
                        n_resp,
                        lname,
                        deriv,
                        center,
                    ));
                } else {
                    // Build the design with the CORE B-spline machinery (same as
                    // mboost / the Python binding) from the raw spline params,
                    // rather than a pre-built splines::bs design — so the fit
                    // matches R mboost.
                    let x = list_named_elt(&spec, "x")
                        .as_real_vector()
                        .unwrap_or_else(|| throw_r_error("bbs $x not numeric"));
                    let knots = list_named_elt(&spec, "knots").as_real().unwrap_or(20.0) as usize;
                    let degree = list_named_elt(&spec, "degree").as_real().unwrap_or(3.0) as usize;
                    let differences = list_named_elt(&spec, "differences")
                        .as_real()
                        .unwrap_or(2.0) as usize;
                    let df = list_named_elt(&spec, "df").as_real().unwrap_or(4.0);
                    let lambda = list_named_elt(&spec, "lambda").as_real().unwrap_or(-1.0);
                    let center = list_named_elt(&spec, "center").as_bool().unwrap_or(false);
                    let deriv = list_named_elt(&spec, "deriv").as_real().unwrap_or(0.0) as usize;
                    let lambda_opt = if lambda >= 0.0 { Some(lambda) } else { None };
                    let xa = ndarray::Array1::from_vec(x);
                    // `bbs(x, by = z)`: the core multiplies the design by z and
                    // calibrates df on the result, as mboost does.
                    let by = list_optional_elt(&spec, "by_values")
                        .and_then(|value| value.as_real_vector())
                        .map(ndarray::Array1::from_vec);
                    if let Some(values) = by.as_ref() {
                        if values.len() != xa.len() {
                            throw_r_error(format!(
                                "bbs {lname}: by has {} values for {} rows",
                                values.len(),
                                xa.len()
                            ));
                        }
                    }
                    learners.push(prepare_bbs_with_lambda(
                        xa.view(),
                        w_owned.view(),
                        knots,
                        degree,
                        differences,
                        df,
                        lambda_opt,
                        center,
                        lname,
                        by.as_ref().map(|values| values.view()),
                        false,
                        None,
                        deriv,
                    ));
                }
            }
            "tree" => {
                let design_flat = list_named_elt(&spec, "design")
                    .as_real_vector()
                    .unwrap_or_else(|| throw_r_error("tree $design not numeric"));
                let dims = list_named_elt(&spec, "nq")
                    .as_integer_vector()
                    .unwrap_or_else(|| throw_r_error("tree $nq not integer"));
                let minbucket =
                    list_named_elt(&spec, "minbucket").as_integer().unwrap_or(4) as usize;
                let minsplit =
                    list_named_elt(&spec, "minsplit").as_integer().unwrap_or(10) as usize;
                let maxdepth = list_named_elt(&spec, "maxdepth").as_integer().unwrap_or(1) as usize;

                let nr = dims[0] as usize;
                let nc = dims[1] as usize;
                let design = ndarray::Array2::from_shape_vec((nr, nc), design_flat)
                    .unwrap_or_else(|e| throw_r_error(format!("tree design shape error: {e}")));

                let categorical: Vec<usize> = list_named_elt(&spec, "categorical")
                    .as_integer_vector()
                    .unwrap_or_default()
                    .into_iter()
                    .map(|v| v.max(0) as usize)
                    .collect();
                use mboostrs_core::learners::SplitSelection;
                let selection = match list_named_elt(&spec, "selection")
                    .as_str()
                    .unwrap_or("ctree")
                {
                    "sse" => SplitSelection::Sse,
                    "ctree" => SplitSelection::CtreeStatistic,
                    "ctree_pvalue" => SplitSelection::CtreePvalue,
                    other => throw_r_error(format!(
                        "btree selection must be 'sse', 'ctree' or 'ctree_pvalue', got {other:?}"
                    )),
                };
                let by = r_by_values(&spec, nr, lname);
                learners.push(mboostrs_core::learners::prepare_btree_selected(
                    design.view(),
                    w_owned.view(),
                    minbucket,
                    minsplit,
                    maxdepth,
                    by,
                    lname,
                    categorical,
                    selection,
                ));
            }
            "bmono" => {
                // Build the monotone P-spline with the CORE machinery from the
                // raw spline params, so the constraint is actually enforced (the
                // old path fell through to prepare_block_bols — an unconstrained
                // smooth — silently ignoring the monotonicity).
                let x = list_named_elt(&spec, "x")
                    .as_real_vector()
                    .unwrap_or_else(|| throw_r_error("bmono $x not numeric"));
                let knots = list_named_elt(&spec, "knots").as_real().unwrap_or(20.0) as usize;
                let degree = list_named_elt(&spec, "degree").as_real().unwrap_or(3.0) as usize;
                let differences = list_named_elt(&spec, "differences")
                    .as_real()
                    .unwrap_or(2.0) as usize;
                let df = list_named_elt(&spec, "df").as_real().unwrap_or(4.0);
                let solver_str = list_named_elt(&spec, "solver")
                    .as_str()
                    .unwrap_or("quad.prog");
                let lambda2 = list_named_elt(&spec, "lambda2").as_real().unwrap_or(1e6);
                let niter = list_named_elt(&spec, "niter").as_integer().unwrap_or(10) as usize;
                let constraint_str = list_named_elt(&spec, "constraint")
                    .as_str()
                    .unwrap_or("increasing");
                let constraint = parse_mono_constraint(constraint_str);
                let solver = parse_bmono_solver(solver_str);
                let xa = ndarray::Array1::from_vec(x);
                learners.push(mboostrs_core::learners::prepare_bmono(
                    xa.view(),
                    w_owned.view(),
                    knots,
                    degree,
                    differences,
                    df,
                    constraint,
                    solver,
                    lambda2,
                    niter,
                    lname,
                    r_by_values(&spec, xa.len(), lname)
                        .as_ref()
                        .map(|z| z.view()),
                    None,
                    false,
                ));
            }
            "bconstrained" => {
                let x = list_named_elt(&spec, "x")
                    .as_real_vector()
                    .unwrap_or_else(|| throw_r_error("bconstrained $x not numeric"));
                let dims = list_named_elt(&spec, "C_dims")
                    .as_integer_vector()
                    .unwrap_or_else(|| throw_r_error("bconstrained $C_dims not integer"));
                if dims.len() != 2 {
                    throw_r_error("bconstrained C_dims must contain two dimensions");
                }
                let rows = dims[0] as usize;
                let columns = dims[1] as usize;
                let c_values = list_named_elt(&spec, "C")
                    .as_real_vector()
                    .unwrap_or_else(|| throw_r_error("bconstrained $C not numeric"));
                let c = ndarray::Array2::from_shape_fn((rows, columns), |(row, column)| {
                    c_values[column * rows + row]
                });
                let lambda_value = list_named_elt(&spec, "lambda").as_real().unwrap_or(-1.0);
                let xa = ndarray::Array1::from_vec(x);
                learners.push(mboostrs_core::learners::prepare_bmono_with_lambda(
                    xa.view(),
                    w_owned.view(),
                    list_named_elt(&spec, "knots").as_real().unwrap_or(20.0) as usize,
                    list_named_elt(&spec, "degree").as_real().unwrap_or(3.0) as usize,
                    list_named_elt(&spec, "differences")
                        .as_real()
                        .unwrap_or(2.0) as usize,
                    list_named_elt(&spec, "df").as_real().unwrap_or(4.0),
                    (lambda_value >= 0.0).then_some(lambda_value),
                    MonoConstraint::Constrained,
                    // The exact QP by default, as for bmono: the iterative
                    // violation penalty leaves a same-signed violation each
                    // step that boosting accumulates.
                    parse_bmono_solver(
                        list_optional_elt(&spec, "solver")
                            .and_then(|value| value.as_str().map(str::to_string))
                            .as_deref()
                            .unwrap_or("quad.prog"),
                    ),
                    list_named_elt(&spec, "lambda2").as_real().unwrap_or(1e6),
                    list_named_elt(&spec, "niter").as_integer().unwrap_or(10) as usize,
                    lname,
                    None,
                    Some(c.view()),
                    false,
                ));
            }
            "bspatial" => {
                let x1 = list_named_elt(&spec, "x1")
                    .as_real_vector()
                    .unwrap_or_else(|| throw_r_error("bspatial $x1 not numeric"));
                let x2 = list_named_elt(&spec, "x2")
                    .as_real_vector()
                    .unwrap_or_else(|| throw_r_error("bspatial $x2 not numeric"));
                let knots = list_named_elt(&spec, "knots").as_real().unwrap_or(10.0) as usize;
                let degree = list_named_elt(&spec, "degree").as_real().unwrap_or(3.0) as usize;
                let differences = list_named_elt(&spec, "differences")
                    .as_real()
                    .unwrap_or(2.0) as usize;
                let df = list_named_elt(&spec, "df").as_real().unwrap_or(6.0);
                let lambda = list_named_elt(&spec, "lambda").as_real().unwrap_or(-1.0);
                let center = list_named_elt(&spec, "center").as_bool().unwrap_or(true);
                let lambda_opt = if lambda >= 0.0 { Some(lambda) } else { None };
                let x1a = ndarray::Array1::from_vec(x1);
                let x2a = ndarray::Array1::from_vec(x2);
                learners.push(prepare_bspatial(
                    x1a.view(),
                    x2a.view(),
                    w_owned.view(),
                    knots,
                    degree,
                    differences,
                    df,
                    lambda_opt,
                    center,
                    lname,
                    r_by_values(&spec, x1a.len(), lname)
                        .as_ref()
                        .map(|z| z.view()),
                ));
            }
            "brandom" => {
                let design_flat = list_named_elt(&spec, "design")
                    .as_real_vector()
                    .unwrap_or_else(|| throw_r_error("brandom $design not numeric"));
                let dims = list_named_elt(&spec, "np")
                    .as_integer_vector()
                    .unwrap_or_else(|| throw_r_error("brandom $np not integer"));
                let n_levels = list_named_elt(&spec, "n_levels").as_integer().unwrap_or(0) as usize;
                let df_val = list_named_elt(&spec, "df").as_real().unwrap_or(4.0);
                let lambda_val = list_named_elt(&spec, "lambda").as_real().unwrap_or(-1.0);
                let center = list_named_elt(&spec, "center").as_bool().unwrap_or(false);

                let nr = dims[0] as usize;
                let nc = n_levels;
                let design = ndarray::Array2::from_shape_vec((nr, nc), design_flat)
                    .unwrap_or_else(|e| throw_r_error(format!("brandom design shape error: {e}")));

                let lambda = if lambda_val >= 0.0 {
                    lambda_val
                } else {
                    let identity = ndarray::Array2::eye(nc);
                    if center {
                        let mut centered = design.clone();
                        for j in 0..centered.ncols() {
                            let cj = centered.column(j).dot(&w_owned) / sw;
                            centered.column_mut(j).mapv_inplace(|v| v - cj);
                        }
                        df_to_lambda(&centered, &identity, &w_owned.to_owned(), df_val)
                    } else {
                        df_to_lambda(&design, &identity, &w_owned.to_owned(), df_val)
                    }
                };

                let mut penalty_diag = ndarray::Array2::zeros((nc, nc));
                for d in 0..nc {
                    penalty_diag[[d, d]] = lambda;
                }
                learners.push(prepare_block_bols(
                    design,
                    w_owned.view(),
                    penalty_diag,
                    center,
                    lname,
                ));
            }
            "bmrf" => {
                let levels_vec = list_named_elt(&spec, "levels")
                    .as_integer_vector()
                    .unwrap_or_else(|| throw_r_error("bmrf $levels not integer"));
                let df_val = list_named_elt(&spec, "df").as_real().unwrap_or(4.0);
                let lambda_val = list_named_elt(&spec, "lambda").as_real().unwrap_or(-1.0);
                let center = list_named_elt(&spec, "center").as_bool().unwrap_or(false);

                let k_flat = list_named_elt(&spec, "K")
                    .as_real_vector()
                    .unwrap_or_else(|| {
                        throw_r_error("bmrf $K not numeric (supply K before fitting)")
                    });
                let n_levels = list_named_elt(&spec, "n_levels").as_integer().unwrap_or(0) as usize;
                if n_levels == 0 || k_flat.is_empty() {
                    throw_r_error(
                        "bmrf requires K (neighbourhood matrix); supply it before fitting",
                    );
                }
                let k = ndarray::Array2::from_shape_vec((n_levels, n_levels), k_flat)
                    .unwrap_or_else(|e| throw_r_error(format!("bmrf K shape error: {e}")));

                let levels =
                    ndarray::Array1::from_vec(levels_vec.iter().map(|&l| l as usize).collect());
                let lambda_opt = if lambda_val >= 0.0 {
                    Some(lambda_val)
                } else {
                    None
                };
                let by = r_by_values(&spec, levels.len(), lname);
                learners.push(mboostrs_core::learners::prepare_bmrf(
                    levels.view(),
                    w_owned.view(),
                    &k,
                    df_val,
                    lambda_opt,
                    center,
                    lname,
                    by.as_ref().map(|values| values.view()),
                ));
            }
            "bns" => {
                let x = list_named_elt(&spec, "x")
                    .as_real_vector()
                    .unwrap_or_else(|| throw_r_error("bns $x not numeric"));
                let knots = list_named_elt(&spec, "knots").as_real().unwrap_or(20.0) as usize;
                let degree = list_named_elt(&spec, "degree").as_real().unwrap_or(3.0) as usize;
                let df = list_named_elt(&spec, "df").as_real().unwrap_or(-1.0);
                let center = list_named_elt(&spec, "center").as_bool().unwrap_or(false);

                let xa = ndarray::Array1::from_vec(x);
                learners.push(prepare_bns(
                    xa.view(),
                    w_owned.view(),
                    knots,
                    degree,
                    df,
                    center,
                    lname,
                    r_by_values(&spec, xa.len(), lname)
                        .as_ref()
                        .map(|z| z.view()),
                ));
            }
            other => throw_r_error(format!("Unknown learner type: {other}")),
        }
    }
    learners
}

/// Fit a gamboost model from pre-processed learner specifications.
///
/// The R side does the B-spline basis construction, penalty matrix setup,
/// and centering. This function receives flat arrays per learner and
/// reconstructs the PreparedLearner objects.
///
/// Each learner is specified by a "type" string:
/// - "scalar": 1-column linear effect (n design values + lambda + center flag)
/// - "block": multi-column linear/smooth effect (n*p design + p*p penalty + center flag)
/// - "tree": tree base-learner (n*q design matrix + minbucket + minsplit + maxdepth)
/// - "bmono": monotone bbs (n*p design + p*p penalty + p*q constraint D + solver + lambda2 + niter)
///
/// learner_specs is a list of lists, each with:
///   type: string ("scalar", "block", "tree", "bmono")
///   name: string
///   For "scalar": design (n vector), lambda (scalar), center (logical)
///   For "block": design (n*p vector), penalty (p*p vector), np (integer c(n,p)), center (logical)
///   For "tree": design (n*q vector), nq (integer c(n,q)), minbucket, minsplit, maxdepth (integers)
///   For "bmono": design (n*p), penalty (p*p), d_mat (q*p), np (c(n,p)), solver (string), lambda2, niter
fn prepare_gamboost_store_r(
    learner_specs: &List,
    w_owned: ndarray::Array1<f64>,
    multinomial_n_classes: i32,
) -> (Vec<PreparedLearner>, ndarray::Array1<f64>) {
    let mut learners =
        prepare_gamboost_learners_r(learner_specs, w_owned.clone(), multinomial_n_classes);

    // Multinomial: wrap each (linear) learner as a KroneckerBols with x2 = I_{K-1}
    // and expand the per-observation weights to length n_obs*(K-1). Mirrors the
    // Python binding; only bols-style learners (scalar / factor block) are
    // supported, matching R's `bols(x) %O% bols(class)`.
    let w_expanded: ndarray::Array1<f64> = if multinomial_n_classes > 1 {
        let n_resp = multinomial_n_classes as usize - 1;
        let x2 = ndarray::Array2::eye(n_resp);
        let n_obs = w_owned.len();
        let mut kron_learners = Vec::with_capacity(learners.len());
        for learner in learners.into_iter() {
            match learner {
                PreparedLearner::ScalarBols {
                    design,
                    name,
                    centering,
                    ..
                } => {
                    let (x1, pen) = if centering != 0.0 {
                        // bols(x, intercept=TRUE): reconstruct [1, x].
                        let mut x1 = ndarray::Array2::ones((n_obs, 2));
                        for i in 0..n_obs {
                            x1[[i, 1]] = design[i] + centering;
                        }
                        (x1, ndarray::Array2::zeros((2, 2)))
                    } else {
                        let x1 = design.into_shape_with_order((n_obs, 1)).unwrap();
                        (x1, ndarray::Array2::zeros((1, 1)))
                    };
                    let mut k =
                        prepare_kronecker_bols(x1, w_owned.view(), pen, x2.clone(), false, &name);
                    let rebuild = if centering != 0.0 {
                        KroneckerRebuild::BolsIntercept
                    } else {
                        KroneckerRebuild::BolsNoIntercept
                    };
                    k.set_x1_rebuild(rebuild);
                    kron_learners.push(k);
                }
                PreparedLearner::BlockBols {
                    design,
                    penalty,
                    name,
                    ..
                } => {
                    kron_learners.push(prepare_kronecker_bols(
                        design,
                        w_owned.view(),
                        penalty,
                        x2.clone(),
                        false,
                        &name,
                    ));
                }
                // bbs learners are already built as KroneckerBols above.
                k @ PreparedLearner::KroneckerBols { .. } => {
                    kron_learners.push(k);
                }
                other => {
                    throw_r_error(format!(
                        "Learner '{}' is not supported for the Multinomial family \
                         (only bols, bbs and brandom learners are)",
                        other.name()
                    ));
                }
            }
        }
        learners = kron_learners;

        let mut w_exp = ndarray::Array1::zeros(n_obs * n_resp);
        for i in 0..n_obs {
            for k in 0..n_resp {
                w_exp[i * n_resp + k] = w_owned[i];
            }
        }
        w_exp
    } else {
        w_owned.clone()
    };
    (learners, w_expanded)
}

#[extendr]
fn fit_gamboost(
    learner_specs: List,
    y: &[f64],
    weights: &[f64],
    mstop: i32,
    nu: f64,
    family_name: &str,
    family_params: List,
    offset_val: &[f64],
    risk: &str,
    trace: bool,
    multinomial_n_classes: i32,
    component_fraction: f64,
    subsample_seed: f64,
    oob_weights: &[f64],
    step: &str,
    step_iterations: i32,
) -> List {
    let binding_started = trace.then(Instant::now);
    let mstop_u32 = checked_mstop(mstop);
    if nu <= 0.0 || nu > 1.0 {
        throw_r_error(format!("nu must be in (0, 1], got {nu}"));
    }

    check_finite(y, "y");
    check_finite(weights, "weights");

    let w_owned = ndarray::Array1::from_vec(weights.to_vec());

    let preparation_timer = trace.then(PreparationTimer::start);
    let n_obs_weights = weights.len();
    let (mut learners, w_expanded) =
        prepare_gamboost_store_r(&learner_specs, w_owned, multinomial_n_classes);
    // Held-out risk weights for a cross-validation fold (empty: the engine's
    // unit weights), expanded per class like the fit weights.
    let n_resp = (multinomial_n_classes.max(2) - 1) as usize;
    let oob_expanded: Option<ndarray::Array1<f64>> = if oob_weights.is_empty() {
        None
    } else {
        if oob_weights.len() != n_obs_weights {
            throw_r_error(format!(
                "oob_weights has length {} but weights has {}",
                oob_weights.len(),
                n_obs_weights
            ));
        }
        Some(if multinomial_n_classes > 1 {
            oob_weights
                .iter()
                .flat_map(|&v| std::iter::repeat_n(v, n_resp))
                .collect()
        } else {
            ndarray::Array1::from_vec(oob_weights.to_vec())
        })
    };
    let preparation_timing = preparation_timer.map(PreparationTimer::finish);

    let y_arr = ndarray::Array1::from_vec(y.to_vec());

    let risk_kind = parse_risk_kind(risk);
    let ctrl = BoostControl::new(mstop_u32, nu)
        .unwrap_or_else(|e| throw_r_error(format!("invalid BoostControl: {e}")))
        .with_risk(risk_kind)
        .with_trace(trace);
    let ctrl = apply_r_subsampling(ctrl, component_fraction, subsample_seed);
    let ctrl = apply_r_step(ctrl, step, step_iterations);

    let offset = ROffset::from_r(offset_val);

    let params = extract_family_params(&family_params);
    // Validate the response up front (check_y was otherwise never called), so
    // invalid y errors clearly instead of producing NaN deep in the fit.
    make_boxed_family(family_name, &params)
        .check_y(y_arr.view())
        .unwrap_or_else(|e| throw_r_error(format!("invalid response: {e}")));
    let binding_input_ns = binding_started.map(|started| started.elapsed().as_nanos() as u64);
    let path = with_concrete_family!(family_name, &params, |fam| {
        match (oob_expanded.as_ref(), offset.vector()) {
            (Some(held_out), vector) => engine::try_fit_with_oob_weights(
                &mut learners,
                y_arr.view(),
                w_expanded.view(),
                &mut fam,
                &ctrl,
                offset.scalar(),
                vector,
                held_out.view(),
            ),
            (None, vector) => match vector {
                Some(vector) => engine::try_fit_with_vector_offset(
                    &mut learners,
                    y_arr.view(),
                    w_expanded.view(),
                    &mut fam,
                    &ctrl,
                    vector,
                    false,
                ),
                None => engine::try_fit_with(
                    &mut learners,
                    y_arr.view(),
                    w_expanded.view(),
                    &mut fam,
                    &ctrl,
                    offset.scalar(),
                    false,
                ),
            },
        }
    });
    let mut path = path.unwrap_or_else(|e| throw_r_error(format!("{e}")));

    let binding_output_started = trace.then(Instant::now);
    if path.risk.iter().any(|v| !v.is_finite()) || path.fitted.iter().any(|v| !v.is_finite()) {
        throw_r_error(
            "boosting fit diverged (non-finite risk/fitted values); \
             decrease the step-size nu in boost_control",
        );
    }

    // Return 1-based selected indices, coefs as list of vectors,
    // plus the fitted path info.
    let selected: Vec<i32> = path.selected.iter().map(|&s| (s + 1) as i32).collect();
    let coefs_list: Vec<Vec<f64>> = path.coefs.clone();
    let increments_list: Vec<Vec<f64>> = if path.increments.is_empty() {
        vec![]
    } else {
        path.increments.clone()
    };
    let nuisance_vec = nuisance_for_r(&path);
    // Penalization drift, computed only when some learner was selected often
    // enough to have drifted (nu * selections >= 50): one Gram and
    // eigendecomposition per learner. NA where a learner has no closed-form
    // smoother. R's fit object keeps no prepared learners, so this is the one
    // place it can be measured.
    let drift_robj = {
        let mut counts = vec![0usize; learners.len()];
        for &j in &path.selected {
            if let Some(c) = counts.get_mut(j) {
                *c += 1;
            }
        }
        let busiest = counts.iter().copied().max().unwrap_or(0) as f64;
        if multinomial_n_classes <= 1 && nu * busiest >= 50.0 {
            match mboostrs_core::penalty_drift(&learners, &path.selected, nu) {
                Ok(rows) => {
                    let pick = |f: fn(&mboostrs_core::LearnerDrift) -> f64| -> Vec<f64> {
                        rows.iter()
                            .map(|r| r.as_ref().map_or(f64::NAN, f))
                            .collect()
                    };
                    list!(
                        selections = counts.iter().map(|&c| c as f64).collect::<Vec<f64>>(),
                        single_step_df = pick(|r| r.single_step),
                        boosted_df = pick(|r| r.boosted),
                        unpenalized_df = pick(|r| r.unpenalized)
                    )
                    .into_robj()
                }
                Err(_) => extendr_api::NULL.into_robj(),
            }
        } else {
            extendr_api::NULL.into_robj()
        }
    };

    {
        let coefs_robj = extendr_api::List::from_values(
            coefs_list
                .iter()
                .map(|v| extendr_api::Robj::from(v.as_slice())),
        );
        let increments_robj = if increments_list.is_empty() {
            extendr_api::List::new(0)
        } else {
            extendr_api::List::from_values(
                increments_list
                    .iter()
                    .map(|v| extendr_api::Robj::from(v.as_slice())),
            )
        };
        let nuisance_rows = path.nuisance_rows();
        let nuisance_vec_path_robj = extendr_api::List::from_values(
            nuisance_rows
                .iter()
                .map(|v| extendr_api::Robj::from(v.as_slice())),
        );
        attach_frontend_timings(
            &mut path.report,
            preparation_timing,
            binding_input_ns,
            binding_output_started,
        );
        list!(
            offset = path.offset,
            selected = selected,
            coefs = coefs_robj,
            risk = path.risk,
            fitted = path.fitted.to_vec(),
            increments = increments_robj,
            nuisance = nuisance_vec,
            nuisance_vec_path = nuisance_vec_path_robj,
            execution_report = fit_report_to_r(&path.report),
            penalty_drift = drift_robj,
            learner_centering = List::from_values(learners.iter().map(|learner| match learner {
                PreparedLearner::BlockBols { centering, .. } => Robj::from(centering.to_vec()),
                // The core centres a scalar bols at the weighted mean of its
                // column; returning it lets predict(newdata =) subtract the
                // same constant.
                PreparedLearner::ScalarBols { centering, .. } => Robj::from(vec![*centering]),
                _ => Robj::from(Vec::<f64>::new()),
            })),
        )
    }
}

/// mgcv's three concurvity measures of a fitted gamboost model's terms,
/// from the learner specifications the fit was given and the boosting path
/// it returned (`mboostrs_core::gam::boosted_concurvity`). Rebuilding the
/// store from the same specifications and weights gives the same learner
/// order the fit's `selected` indices refer to. Multinomial fits wrap every
/// learner as a Kronecker learner, which carries no function-space basis
/// for this measure, so they return an empty result.
///
/// `selected` is the fit's per-step 1-based indices and `coefs` its list of
/// per-step unshrunken updates. Returns `list(included, worst, observed,
/// estimate, pair_i, pair_j, pair_worst, pair_observed, pair_estimate)`:
/// `included` is 1-based into the learner order, the summary vectors are one
/// entry per included learner, and the pair vectors are one entry per
/// ordered pair of included learners (diagonal excluded).
#[extendr]
fn gamboost_concurvity_r(
    learner_specs: List,
    weights: &[f64],
    nu: f64,
    selected: &[i32],
    coefs: List,
    multinomial_n_classes: i32,
) -> List {
    let w_owned = ndarray::Array1::from_vec(weights.to_vec());
    let (learners, _) = prepare_gamboost_store_r(&learner_specs, w_owned, multinomial_n_classes);
    let selected0: Vec<usize> = selected.iter().map(|&s| (s - 1) as usize).collect();
    let coefs_vecs: Vec<Vec<f64>> = (0..coefs.len())
        .map(|s| {
            coefs
                .elt(s)
                .expect("elt")
                .as_real_vector()
                .unwrap_or_default()
        })
        .collect();
    let (included, mc) =
        mboostrs_core::gam::boosted_concurvity(&learners, nu, &selected0, &coefs_vecs)
            .unwrap_or_else(|e| throw_r_error(format!("concurvity: {e}")));
    concurvity_result_r(&included, &mc)
}

/// The same measures for a fitted glmboost model, from its design matrix
/// (row-major, as `t(x_matrix)` passes it), the centering means the fit
/// used, and the boosting path it returned. Every glmboost learner is one
/// column, so each measure is the centered column's squared-multiple-R²
/// against the rest; the intercept learner, constant after centering,
/// reports NaN. `coefs` is the fit's flat per-step coefficient vector.
#[extendr]
fn glmboost_concurvity_r(
    x_matrix: &[f64],
    n: i32,
    p: i32,
    weights: &[f64],
    nu: f64,
    selected: &[i32],
    coefs: &[f64],
    centering_means: &[f64],
    intercept: bool,
) -> List {
    let n = n as usize;
    let p = p as usize;
    if x_matrix.len() != n * p {
        throw_r_error(format!(
            "x_matrix length {} does not match n={} * p={}",
            x_matrix.len(),
            n,
            p
        ));
    }
    if selected.len() != coefs.len() {
        throw_r_error(format!(
            "selected length {} does not match coefs length {}",
            selected.len(),
            coefs.len()
        ));
    }
    let x_arr = ndarray::Array2::from_shape_vec((n, p), x_matrix.to_vec())
        .unwrap_or_else(|e| throw_r_error(format!("invalid x_matrix dimensions: {e}")));
    let w_owned = ndarray::Array1::from_vec(weights.to_vec());
    let ones = ndarray::Array1::ones(n);
    let learners = prepare_glmboost_learners_r(&x_arr, &w_owned, &ones, centering_means, intercept);
    let selected0: Vec<usize> = selected.iter().map(|&s| (s - 1) as usize).collect();
    let coefs_vecs: Vec<Vec<f64>> = coefs.iter().map(|&c| vec![c]).collect();
    let (included, mc) =
        mboostrs_core::gam::boosted_concurvity(&learners, nu, &selected0, &coefs_vecs)
            .unwrap_or_else(|e| throw_r_error(format!("concurvity: {e}")));
    concurvity_result_r(&included, &mc)
}

/// The core's measures as flat R vectors: one entry per included learner for
/// the summary, one entry per ordered pair of included learners for the
/// pairwise measures, with 1-based `pair_i`/`pair_j` into `included`.
fn concurvity_result_r(included: &[usize], mc: &mboostrs_core::gam::ModelConcurvity) -> List {
    let l = included.len();
    let mut worst = Vec::with_capacity(l);
    let mut observed = Vec::with_capacity(l);
    let mut estimate = Vec::with_capacity(l);
    for row in &mc.summary {
        worst.push(row.worst);
        observed.push(row.observed);
        estimate.push(row.estimate);
    }
    let mut pair_i = Vec::with_capacity(l * (l.saturating_sub(1)));
    let mut pair_j = Vec::with_capacity(pair_i.capacity());
    let mut pair_worst = Vec::with_capacity(pair_i.capacity());
    let mut pair_observed = Vec::with_capacity(pair_i.capacity());
    let mut pair_estimate = Vec::with_capacity(pair_i.capacity());
    for (i, row) in mc.pairwise.iter().enumerate() {
        for (j, c) in row.iter().enumerate() {
            if i == j {
                continue;
            }
            pair_i.push((i + 1) as i32);
            pair_j.push((j + 1) as i32);
            pair_worst.push(c.worst);
            pair_observed.push(c.observed);
            pair_estimate.push(c.estimate);
        }
    }
    list!(
        included = included
            .iter()
            .map(|&i| (i + 1) as i32)
            .collect::<Vec<i32>>(),
        worst = worst,
        observed = observed,
        estimate = estimate,
        pair_i = pair_i,
        pair_j = pair_j,
        pair_worst = pair_worst,
        pair_observed = pair_observed,
        pair_estimate = pair_estimate,
    )
}

/// Predict from a fitted gamboost model.
///
/// learner_designs: list of design matrices (one per learner, flat n*p vectors).
///   Scalar learners have a length-n vector.
///   Block learners have a length-(n*ncol) vector.
///   Tree learners have a length-(n*ncol) vector.
/// learner_ncol: integer vector, number of columns per learner design.
///   (1 for scalar, p for block/tree)
/// learner_centering: list of centering vectors (one per learner).
///   Scalar: single value; Block: p-vector; Tree: empty.
#[extendr]
fn predict_gamboost(
    offset: &[f64],
    selected: &[i32],
    coefs_list: List,
    nu: f64,
    learner_designs: List,
    learner_ncol: &[i32],
    learner_centering: List,
    mstop: i32,
    by_values: List,
) -> Vec<f64> {
    let k = if mstop <= 0 {
        selected.len()
    } else {
        (mstop as usize).min(selected.len())
    };

    let n_obs = if learner_ncol.is_empty() {
        0
    } else {
        let first_ncol = learner_ncol[0] as usize;
        if first_ncol == 0 {
            0
        } else {
            learner_designs
                .elt(0)
                .expect("elt")
                .as_real_vector()
                .map(|v| v.len() / first_ncol.max(1))
                .unwrap_or(0)
        }
    };
    if n_obs == 0 {
        return vec![];
    }

    let mut eta = eta_from_offset(offset, n_obs);

    for step in 0..k {
        let learner_idx = (selected[step] - 1) as usize;
        let coefs_vec = coefs_list
            .elt(step)
            .expect("elt")
            .as_real_vector()
            .unwrap_or_default();

        let design_raw = learner_designs
            .elt(learner_idx)
            .expect("elt")
            .as_real_vector()
            .unwrap_or_default();
        let nc = learner_ncol.get(learner_idx).copied().unwrap_or(1) as usize;
        let centering_raw = learner_centering
            .elt(learner_idx)
            .expect("elt")
            .as_real_vector()
            .unwrap_or_default();

        // by variable (optional)
        let has_by = by_values.len() > learner_idx;
        let by_vec: Vec<f64> = if has_by {
            by_values
                .elt(learner_idx)
                .expect("elt")
                .as_real_vector()
                .unwrap_or_default()
        } else {
            vec![]
        };

        if nc == 1 && coefs_vec.len() == 1 {
            // Scalar learner
            let c = coefs_vec[0];
            let centering = centering_raw.first().copied().unwrap_or(0.0);
            for i in 0..n_obs {
                let x_i = design_raw.get(i).copied().unwrap_or(0.0) - centering;
                let by_i = by_vec.get(i).copied().unwrap_or(1.0);
                eta[i] += nu * c * x_i * by_i;
            }
        } else if nc > 1 && coefs_vec.len() == nc {
            // Block learner: eta += nu * design %*% coefs
            // Design is n*nc row-major, centering is nc-vector
            for i in 0..n_obs {
                let mut pred_i = 0.0f64;
                for j in 0..nc {
                    let x_ij = design_raw.get(i * nc + j).copied().unwrap_or(0.0);
                    let cj = centering_raw.get(j).copied().unwrap_or(0.0);
                    pred_i += coefs_vec[j] * (x_ij - cj);
                }
                let by_i = by_vec.get(i).copied().unwrap_or(1.0);
                eta[i] += nu * pred_i * by_i;
            }
        }
        // Tree learners: coefs encode the stump, prediction is more complex.
        // For simplicity, the R side should handle tree prediction natively.
    }

    eta
}

/// The full Multinomial negative log-likelihood on the training rows after
/// each boosting step 0..=mstop (`Family::held_out_risk`, all classes). The
/// in-bag risk path is mboost's loss, which leaves the reference class out
/// and can rise on a step that lowers this one; variable importance
/// attributes each step's change in this path. `y` is the K-1 indicator
/// layout of the fit, `w` one weight per observation.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn multinomial_staged_nll(
    offset: f64,
    selected: &[i32],
    coefs_list: List,
    nu: f64,
    x1_designs: List,
    x1_ncol: &[i32],
    n_resp: i32,
    y: &[f64],
    w: &[f64],
) -> Vec<f64> {
    use mboostrs_core::families::{Family, Multinomial};

    let n_resp_u = n_resp as usize;
    let n_obs = w.len();
    if y.len() != n_obs * n_resp_u {
        throw_r_error(format!(
            "y has {} entries, expected {} * {}",
            y.len(),
            n_obs,
            n_resp_u
        ));
    }
    let designs: Vec<Vec<f64>> = (0..x1_designs.len())
        .map(|i| {
            x1_designs
                .elt(i)
                .ok()
                .and_then(|v| v.as_real_vector())
                .unwrap_or_default()
        })
        .collect();
    let family = Multinomial::new(n_resp_u + 1);
    let y_arr = ndarray::Array1::from_vec(y.to_vec());
    let w_arr: ndarray::Array1<f64> = w
        .iter()
        .flat_map(|&v| std::iter::repeat_n(v, n_resp_u))
        .collect();
    let mut eta = ndarray::Array1::from_elem(n_obs * n_resp_u, offset);
    let mut out = Vec::with_capacity(selected.len() + 1);
    out.push(family.held_out_risk(y_arr.view(), eta.view(), w_arr.view()));
    for (step, &sel) in selected.iter().enumerate() {
        let learner_idx = (sel - 1) as usize;
        let coef = coefs_list
            .elt(step)
            .expect("elt")
            .as_real_vector()
            .unwrap_or_default();
        let p1 = x1_ncol.get(learner_idx).copied().unwrap_or(1) as usize;
        let x1 = &designs[learner_idx];
        if coef.len() == p1 * n_resp_u && x1.len() == n_obs * p1 {
            for i in 0..n_obs {
                for k in 0..n_resp_u {
                    let mut s = 0.0f64;
                    for j in 0..p1 {
                        s += x1[i * p1 + j] * coef[k * p1 + j];
                    }
                    eta[i * n_resp_u + k] += nu * s;
                }
            }
        }
        out.push(family.held_out_risk(y_arr.view(), eta.view(), w_arr.view()));
    }
    out
}

/// Predictions of one stored tree (its coefficient encoding) at the rows of a
/// row-major `(n, q)` design; factor columns hold training-level codes, -1
/// for a level the fit never saw.
#[extendr]
fn r_tree_predict(
    coef: &[f64],
    design: &[f64],
    q: i32,
    categorical: &[i32],
    average_unseen: bool,
) -> Vec<f64> {
    let q = q.max(1) as usize;
    if !design.len().is_multiple_of(q) {
        throw_r_error(format!(
            "tree design length {} is not a multiple of {q}",
            design.len()
        ));
    }
    let view = ndarray::ArrayView2::from_shape((design.len() / q, q), design)
        .unwrap_or_else(|e| throw_r_error(format!("tree design shape error: {e}")));
    let tree = mboostrs_core::learners::FittedTree::from_coef_vec(ndarray::ArrayView1::from(coef));
    if average_unseen && !categorical.is_empty() {
        // A level the fit never saw: the mean of the predictions over the
        // column's levels (`unseen_levels = "average"`).
        let levels: Vec<usize> = categorical.iter().map(|&l| l.max(0) as usize).collect();
        return (0..view.nrows())
            .map(|i| tree.predict_row_level_average(view.row(i), &levels))
            .collect();
    }
    tree.predict(view).to_vec()
}

/// Accumulate Kronecker-structured predictions for Multinomial models.
///
/// Each selected learner `m` stores `p1 * n_resp` coefficients.  For each
/// boosting step, `x1_designs[[learner_index]]` is the pre-built x1 matrix
/// (flat, row-major, `n_obs * p1`).  The result is an `n_obs * n_resp`
/// observation-major flat vector of predictions.
#[extendr]
fn predict_gamboost_kronecker(
    offset: f64,
    selected: &[i32],
    coefs_list: List,
    nu: f64,
    x1_designs: List,
    x1_ncol: &[i32],
    n_resp: i32,
    mstop: i32,
) -> Vec<f64> {
    let n_resp_u = n_resp as usize;
    let k = if mstop <= 0 {
        selected.len()
    } else {
        (mstop as usize).min(selected.len())
    };

    // Determine n_obs from the first non-empty x1 design
    let n_obs = (0..x1_designs.len())
        .filter_map(|i| {
            let v = x1_designs.elt(i).ok()?.as_real_vector()?;
            let nc = x1_ncol.get(i).copied().unwrap_or(1) as usize;
            v.len().checked_div(nc)
        })
        .next()
        .unwrap_or(0);
    if n_obs == 0 || n_resp_u == 0 {
        return vec![];
    }

    let mut eta = vec![offset; n_obs * n_resp_u];

    // The predictor is linear in the coefficients and, for a fixed learner,
    // its x1 design is identical at every step that selects it. Aggregate
    // each learner's coefficients over its selected steps first (one
    // O(mstop) pass over the small per-step coefficient vectors, which stay
    // R-list lookups), then convert each distinct learner's design out of
    // the R list and apply once — instead of converting the *entire*
    // flattened design (an extendr copy) and re-applying it at every
    // boosting step that selected it. Same aggregate-then-apply pattern as
    // the Python binding's `predict_newdata_kronecker`.
    let n_learners = x1_ncol.len();
    let mut agg: Vec<Option<Vec<f64>>> = vec![None; n_learners];
    for step in 0..k {
        let learner_idx = (selected[step] - 1) as usize;
        let coef_vec = coefs_list
            .elt(step)
            .expect("elt")
            .as_real_vector()
            .unwrap_or_default();
        match &mut agg[learner_idx] {
            Some(entry) => {
                for (a, c) in entry.iter_mut().zip(coef_vec.iter()) {
                    *a += c;
                }
            }
            slot @ None => *slot = Some(coef_vec),
        }
    }

    for (learner_idx, coef_vec) in agg
        .iter()
        .enumerate()
        .filter_map(|(i, c)| c.as_ref().map(|c| (i, c)))
    {
        let x1_flat = x1_designs
            .elt(learner_idx)
            .expect("elt")
            .as_real_vector()
            .unwrap_or_default();
        let p1 = x1_ncol.get(learner_idx).copied().unwrap_or(1) as usize;

        if coef_vec.len() != p1 * n_resp_u {
            // Skip learners with mismatched dimensions
            continue;
        }

        for i in 0..n_obs {
            let row_offset = i * n_resp_u;
            let x1_row_start = i * p1;
            for k in 0..n_resp_u {
                let coef_offset = k * p1;
                let mut s = 0.0f64;
                for j in 0..p1 {
                    s += x1_flat[x1_row_start + j] * coef_vec[coef_offset + j];
                }
                eta[row_offset + k] += nu * s;
            }
        }
    }

    eta
}

// ─── stabsel helpers ──────────────────────────────────────────────────

#[extendr]
fn stabsel_params_rust(
    p: i32,
    b: i32,
    cutoff: f64,
    q: i32,
    pfer: f64,
    assumption: &str,
    sampling_type: &str,
) -> List {
    use mboostrs_core::stabsel::stabsel_parameters;
    let p_usize = p as usize;
    let b_usize = b as usize;
    let cutoff_opt = if cutoff < 0.0 { None } else { Some(cutoff) };
    let q_usize = if q <= 0 { None } else { Some(q as usize) };
    let pfer_opt = if pfer < 0.0 { None } else { Some(pfer) };

    let result = stabsel_parameters(
        p_usize,
        cutoff_opt,
        q_usize,
        pfer_opt,
        b_usize,
        assumption,
        sampling_type,
    )
    .unwrap_or_else(|e| throw_r_error(format!("stabsel_parameters error: {e}")));

    list!(
        cutoff = result.cutoff,
        q = result.q as i32,
        pfer = result.pfer,
        p = result.p as i32,
        b = result.b as i32,
        sampling_type = result.sampling_type,
        assumption = result.assumption
    )
}

/// An `n_rows x n_cols` matrix as R's column-major vector.
fn column_major(m: &ndarray::Array2<f64>) -> Vec<f64> {
    m.t().iter().copied().collect()
}

/// Selection frequencies from R selection sequences (`seqs`, a list of
/// 1-based learner indices, one vector per subsample; an empty vector is a
/// failed fit). Returns the `p x m` matrix as a column-major vector; see
/// `mboostrs_core::stabsel::try_selection_frequencies`.
#[extendr]
fn stabsel_frequencies_r(seqs: List, p: i32, m: i32, q: i32) -> Vec<f64> {
    let paths: Vec<Vec<usize>> = (0..seqs.len())
        .map(|b| {
            let sel = seqs.elt(b).expect("elt");
            let values: Vec<i32> = match sel.as_integer_vector() {
                Some(v) => v,
                None => sel
                    .as_real_vector()
                    .map(|v| v.iter().map(|&x| x as i32).collect())
                    .unwrap_or_else(|| throw_r_error(format!("seqs[[{}]] is not numeric", b + 1))),
            };
            values
                .iter()
                .map(|&s| {
                    if s < 1 {
                        throw_r_error(format!("seqs[[{}]] holds index {s} below 1", b + 1));
                    }
                    (s - 1) as usize
                })
                .collect()
        })
        .collect();
    let phat = mboostrs_core::stabsel::try_selection_frequencies(
        &paths,
        p.max(0) as usize,
        m.max(0) as usize,
        q.max(0) as usize,
    )
    .unwrap_or_else(|e| throw_r_error(format!("{e}")));
    column_major(&phat)
}

/// Stability selection for an R `glmboost` model: one fit per column of
/// `folds` (column-major `n x n_fits` in-bag weights), with the centering,
/// intercept and offset of the full-data fit. Returns `phat` (column-major
/// `n_learners x mstop`), `max` and `failed`, the number of fits that failed
/// and selected nothing.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn stabsel_glmboost_r(
    x_matrix: &[f64],
    n: i32,
    p: i32,
    y: &[f64],
    weights: &[f64],
    mstop: i32,
    nu: f64,
    family_name: &str,
    family_params: List,
    learner_names: Vec<String>,
    folds: &[f64],
    n_fits: i32,
    centering_means: &[f64],
    offset_val: &[f64],
    intercept: bool,
    q: i32,
    component_fraction: f64,
    subsample_seed: f64,
    step: &str,
    step_iterations: i32,
) -> List {
    let (n_val, p_val) = (n as usize, p as usize);
    let n_fits = n_fits.max(0) as usize;
    let mstop_u = checked_mstop(mstop);
    check_finite(y, "y");
    check_finite(weights, "weights");
    check_finite(x_matrix, "x_matrix");
    if centering_means.len() != p_val {
        throw_r_error(format!(
            "centering_means length {} does not match p={p_val}",
            centering_means.len()
        ));
    }
    let x_arr = ndarray::Array2::from_shape_vec((n_val, p_val), x_matrix.to_vec())
        .unwrap_or_else(|e| throw_r_error(format!("invalid x_matrix: {e}")));
    if folds.len() != n_val * n_fits {
        throw_r_error(format!(
            "folds has {} entries, expected n={n_val} x {n_fits}",
            folds.len()
        ));
    }
    let folds_arr = ndarray::Array2::from_shape_fn((n_val, n_fits), |(i, b)| folds[b * n_val + i]);
    let params = extract_family_params(&family_params);
    let preparer = GlmBoostPreparerR {
        x_cols: (0..p_val).map(|j| x_arr.column(j).to_vec()).collect(),
        names: learner_names,
        family_name: family_name.to_string(),
        family_params: params.clone(),
        column_means: centering_means.to_vec(),
        intercept,
    };
    let ctrl = BoostControl::new(mstop_u, nu)
        .unwrap_or_else(|e| throw_r_error(format!("invalid BoostControl: {e}")));
    let ctrl = apply_r_subsampling(ctrl, component_fraction, subsample_seed);
    let ctrl = apply_r_step(ctrl, step, step_iterations);
    let offset = ROffset::from_r(offset_val);
    let y_owned = ndarray::Array1::from_vec(y.to_vec());
    let w_view = ndarray::ArrayView1::from(weights);
    let n_learners = p_val + usize::from(intercept);
    let result = with_concrete_family!(family_name, &params, |fam| {
        mboostrs_core::stabsel::try_stability_selection_with(
            &preparer,
            y_owned.view(),
            w_view,
            &ctrl,
            &folds_arr,
            offset.scalar(),
            offset.vector(),
            n_learners,
            q.max(0) as usize,
            fam,
        )
    })
    .unwrap_or_else(|e| throw_r_error(format!("{e}")));
    list!(
        phat = column_major(&result.phat),
        max = result.max,
        failed = result.failed as i32
    )
}

/// glmboostLSS: cyclic component-wise boosting for a GAMLSS family with linear
/// (scalar bols) learners. Mirrors the Python `fit_glmboostlss` binding.
/// `x_matrix` is row-major (n × sum(param_ncols)). Returns 1-based selected
/// indices (R convention).
#[extendr]
#[allow(clippy::too_many_arguments)]
fn fit_glmboostlss(
    x_matrix: &[f64],
    n: i32,
    param_ncols: &[i32],
    y: &[f64],
    weights: &[f64],
    mstop: i32,
    nu: f64,
    family_name: &str,
    offsets: &[f64],
    mstop_per_param: &[i32],
    trace: bool,
    stabilization: &str,
    nu_per_param: &[f64],
    step: &str,
    step_iterations: i32,
    response_scale: &str,
) -> List {
    use mboostrs_core::lss::{
        try_fit_lss, BetaLSS, BinomialLSS, DirichletLSS, GammaLSS, GaussianKnownVarLSS,
        GaussianLSS, GeometricLSS, GumbelLSS, InverseGaussianLSS, LogisticLSS, LssFamily,
        NBinomialLSS, NegativeBinomialLSS, Pareto2LSS, PoissonLSS, StudentTLSS, SurvivalLSS,
        ZIPoLSS, ZeroAdjustedGammaLSS, ZeroAdjustedPoissonLSS, BCCGLSS, BCPELSS, BCTLSS, JSULSS,
        PIGLSS, ZINBLSS,
    };
    let binding_started = trace.then(Instant::now);
    if n <= 0 {
        throw_r_error("n must be a positive integer");
    }
    let n = n as usize;
    let mstop = checked_mstop(mstop);
    let param_ncols: Vec<usize> = param_ncols.iter().map(|&c| c as usize).collect();
    let total: usize = param_ncols.iter().sum();
    if x_matrix.len() != n * total {
        throw_r_error(format!(
            "x_matrix length {} does not match n={} * total_cols={}",
            x_matrix.len(),
            n,
            total
        ));
    }
    check_finite(y, "y");
    check_finite(weights, "weights");
    check_finite(x_matrix, "x_matrix");

    let preparation_timer = trace.then(PreparationTimer::start);
    let x_arr = ndarray::Array2::from_shape_vec((n, total), x_matrix.to_vec())
        .unwrap_or_else(|e| throw_r_error(format!("invalid x_matrix dimensions: {e}")));
    let w_owned = ndarray::Array1::from_vec(weights.to_vec());
    let ones = ndarray::Array1::ones(n);

    let family: Box<dyn LssFamily> = match family_name {
        "gaussianlss" => Box::new(GaussianLSS),
        "gammalss" => Box::new(GammaLSS),
        "studenttlss" => Box::new(StudentTLSS),
        "betalss" => Box::new(BetaLSS),
        "nbinomiallss" => Box::new(NBinomialLSS),
        "poissonlss" => Box::new(PoissonLSS),
        "bccglss" => Box::new(BCCGLSS),
        "binomiallss" => Box::new(BinomialLSS),
        "geometriclss" => Box::new(GeometricLSS),
        "gumbellss" => Box::new(GumbelLSS),
        "inversegaussianlss" => Box::new(InverseGaussianLSS),
        "jsulss" => Box::new(JSULSS),
        "bctlss" => Box::new(BCTLSS),
        "bcpelss" => Box::new(BCPELSS),
        "logisticlss" => Box::new(LogisticLSS),
        "zeroadjustedgammalss" => Box::new(ZeroAdjustedGammaLSS),
        "negativebinomiallss" => Box::new(NegativeBinomialLSS),
        "pareto2lss" => Box::new(Pareto2LSS),
        "piglss" => Box::new(PIGLSS),
        "zeroadjustedpoissonlss" => Box::new(ZeroAdjustedPoissonLSS),
        "lognormallss" => Box::new(SurvivalLSS::log_normal()),
        "logloglss" => Box::new(SurvivalLSS::log_log()),
        "weibulllss" => Box::new(SurvivalLSS::weibull()),
        "zipolss" => Box::new(ZIPoLSS),
        "zinblss" => Box::new(ZINBLSS),
        "gaussianknownvarlss" => Box::new(GaussianKnownVarLSS),
        "dirichletlss" => Box::new(
            DirichletLSS::try_new(param_ncols.len())
                .unwrap_or_else(|error| throw_r_error(error.to_string())),
        ),
        other => throw_r_error(format!("Unknown LSS family: {other}")),
    };
    if param_ncols.len() != family.n_params() {
        throw_r_error(format!(
            "expected {} parameter blocks, got {}",
            family.n_params(),
            param_ncols.len()
        ));
    }
    let ctrl = BoostControl::new(mstop, nu)
        .unwrap_or_else(|e| throw_r_error(format!("invalid BoostControl: {e}")))
        .with_trace(trace);
    let ctrl = lss_settings(
        ctrl,
        stabilization,
        nu_per_param,
        step,
        step_iterations,
        response_scale,
    );

    let mut per_param: Vec<Vec<PreparedLearner>> = Vec::with_capacity(param_ncols.len());
    let mut col0 = 0usize;
    for &pc in &param_ncols {
        let mut learners: Vec<PreparedLearner> = Vec::with_capacity(pc + 1);
        learners.push(prepare_scalar_bols(
            ones.view(),
            w_owned.view(),
            0.0,
            false,
            "(Intercept)",
            None,
        ));
        for j in 0..pc {
            let col = x_arr.column(col0 + j);
            // gamboostLSS centers every covariate by its simple column mean
            // (weights enter the normal equations and the offset, not the
            // centering). Weighted means here would break predict/fitted
            // agreement and upstream parity under weights.
            let mean = col.mean().unwrap_or(0.0);
            learners.push(prepare_scalar_bols_fixed_center(
                col,
                w_owned.view(),
                0.0,
                mean,
                "x",
            ));
        }
        per_param.push(learners);
        col0 += pc;
    }
    let preparation_timing = preparation_timer.map(PreparationTimer::finish);

    let offsets_ref = if offsets.is_empty() {
        None
    } else {
        Some(offsets.to_vec())
    };
    let mstop_pp: Option<Vec<usize>> = if mstop_per_param.is_empty() {
        None
    } else {
        Some(mstop_per_param.iter().map(|&m| m as usize).collect())
    };
    let binding_input_ns = binding_started.map(|started| started.elapsed().as_nanos() as u64);
    let path = try_fit_lss(
        &mut per_param,
        ndarray::ArrayView1::from(y),
        w_owned.view(),
        family.as_ref(),
        &ctrl,
        mstop_pp.as_deref(),
        offsets_ref.as_deref(),
    )
    .unwrap_or_else(|e| throw_r_error(format!("{e}")));

    let binding_output_started = trace.then(Instant::now);
    let results: Vec<Robj> = path
        .per_param
        .iter()
        .map(|bp| {
            let coefs_flat: Vec<f64> = bp.coefs.iter().map(|c| c[0]).collect();
            let selected: Vec<i32> = bp.selected.iter().map(|&s| (s + 1) as i32).collect();
            list!(
                offset = bp.offset,
                selected = selected,
                coefs = coefs_flat,
                fitted = bp.fitted.to_vec()
            )
            .into()
        })
        .collect();
    let mut report = path.execution_report();
    attach_frontend_timings(
        &mut report,
        preparation_timing,
        binding_input_ns,
        binding_output_started,
    );
    let execution_report = fit_report_to_r(&report);
    list!(
        results = List::from_values(results),
        param_risk = List::from_values(path.per_param.iter().map(|p| p.risk.clone())),
        risk = path.risk,
        selection_order = path
            .selection_order
            .iter()
            .map(|&p| (p + 1) as i32)
            .collect::<Vec<_>>(),
        execution_report = execution_report
    )
}

fn parse_lss_learner_blueprints_r(learner_specs: &List) -> Vec<Vec<LearnerBlueprint>> {
    let mut per_param = Vec::with_capacity(learner_specs.len());
    for parameter in 0..learner_specs.len() {
        let specs = learner_specs
            .elt(parameter)
            .unwrap_or_else(|_| {
                throw_r_error(format!(
                    "learner_specs[[{}]] is not accessible",
                    parameter + 1
                ))
            })
            .as_list()
            .unwrap_or_else(|| {
                throw_r_error(format!("learner_specs[[{}]] must be a list", parameter + 1))
            });
        if specs.is_empty() {
            throw_r_error(format!(
                "LSS parameter {} requires at least one base learner",
                parameter + 1
            ));
        }
        let mut parameter_blueprints = Vec::with_capacity(specs.len());
        for learner_index in 0..specs.len() {
            let spec = specs
                .elt(learner_index)
                .unwrap_or_else(|_| {
                    throw_r_error(format!(
                        "learner_specs[[{}]][[{}]] is not accessible",
                        parameter + 1,
                        learner_index + 1
                    ))
                })
                .as_list()
                .unwrap_or_else(|| {
                    throw_r_error(format!(
                        "learner_specs[[{}]][[{}]] must be a list",
                        parameter + 1,
                        learner_index + 1
                    ))
                });
            let kind = list_named_elt(&spec, "type")
                .as_str()
                .unwrap_or_else(|| throw_r_error("LSS learner $type is missing"));
            let name = list_named_elt(&spec, "name")
                .as_str()
                .unwrap_or_else(|| throw_r_error("LSS learner $name is missing"));
            let lambda_value = list_optional_elt(&spec, "lambda")
                .and_then(|value| value.as_real())
                .unwrap_or(-1.0);
            let lambda = (lambda_value >= 0.0).then_some(lambda_value);
            let blueprint = match kind {
                "scalar" | "bols" => {
                    let x = list_optional_elt(&spec, "design")
                        .and_then(|value| value.as_real_vector())
                        .or_else(|| {
                            list_optional_elt(&spec, "x").and_then(|value| value.as_real_vector())
                        })
                        .unwrap_or_else(|| throw_r_error("bols learner requires numeric $design"));
                    let df_value = list_optional_elt(&spec, "df")
                        .and_then(|value| value.as_real())
                        .unwrap_or(-1.0);
                    LearnerBlueprint::scalar_bols(
                        ndarray::Array1::from_vec(x),
                        name,
                        lambda,
                        (df_value > 0.0).then_some(df_value),
                        list_optional_elt(&spec, "center")
                            .and_then(|value| value.as_bool())
                            .unwrap_or(true),
                        list_optional_elt(&spec, "intercept")
                            .and_then(|value| value.as_bool())
                            .or_else(|| {
                                list_optional_elt(&spec, "center").and_then(|value| value.as_bool())
                            })
                            .unwrap_or(true),
                        None,
                    )
                }
                "bbs" => {
                    let x = list_named_elt(&spec, "x")
                        .as_real_vector()
                        .unwrap_or_else(|| throw_r_error("bbs learner requires numeric $x"));
                    let boundary_lo =
                        list_optional_elt(&spec, "bk_lo").and_then(|value| value.as_real());
                    let boundary_hi =
                        list_optional_elt(&spec, "bk_hi").and_then(|value| value.as_real());
                    let boundary_knots = boundary_lo.zip(boundary_hi);
                    LearnerBlueprint::bbs(
                        ndarray::Array1::from_vec(x),
                        name,
                        list_optional_elt(&spec, "knots")
                            .and_then(|value| value.as_real())
                            .unwrap_or(20.0) as usize,
                        list_optional_elt(&spec, "degree")
                            .and_then(|value| value.as_real())
                            .unwrap_or(3.0) as usize,
                        list_optional_elt(&spec, "differences")
                            .and_then(|value| value.as_real())
                            .unwrap_or(2.0) as usize,
                        list_optional_elt(&spec, "df")
                            .and_then(|value| value.as_real())
                            .unwrap_or(4.0),
                        lambda,
                        list_optional_elt(&spec, "center")
                            .and_then(|value| value.as_bool())
                            .unwrap_or(false),
                        // `bbs(x, by = z)` with a numeric z: the modifier the
                        // design is multiplied by, row by row.
                        list_optional_elt(&spec, "by_values")
                            .and_then(|value| value.as_real_vector())
                            .filter(|values| !values.is_empty())
                            .map(ndarray::Array1::from_vec),
                        list_optional_elt(&spec, "cyclic")
                            .and_then(|value| value.as_bool())
                            .unwrap_or(false),
                        boundary_knots,
                        list_optional_elt(&spec, "deriv")
                            .and_then(|value| value.as_real())
                            .unwrap_or(0.0) as usize,
                    )
                }
                "bns" => {
                    let x = list_named_elt(&spec, "x")
                        .as_real_vector()
                        .unwrap_or_else(|| throw_r_error("bns learner requires numeric $x"));
                    let boundary_lo =
                        list_optional_elt(&spec, "bk_lo").and_then(|value| value.as_real());
                    let boundary_hi =
                        list_optional_elt(&spec, "bk_hi").and_then(|value| value.as_real());
                    LearnerBlueprint::bns(
                        ndarray::Array1::from_vec(x),
                        name,
                        list_optional_elt(&spec, "knots")
                            .and_then(|value| value.as_real())
                            .unwrap_or(20.0) as usize,
                        list_optional_elt(&spec, "degree")
                            .and_then(|value| value.as_real())
                            .unwrap_or(3.0) as usize,
                        list_optional_elt(&spec, "df").and_then(|value| value.as_real()),
                        list_optional_elt(&spec, "center")
                            .and_then(|value| value.as_bool())
                            .unwrap_or(false),
                        list_optional_elt(&spec, "by_values")
                            .and_then(|value| value.as_real_vector())
                            .filter(|values| !values.is_empty())
                            .map(ndarray::Array1::from_vec),
                        boundary_lo.zip(boundary_hi),
                    )
                }
                "brandom" => {
                    let levels = list_named_elt(&spec, "x")
                        .as_real_vector()
                        .unwrap_or_else(|| throw_r_error("brandom learner requires numeric $x"));
                    let by = list_optional_elt(&spec, "by_values")
                        .and_then(|value| value.as_real_vector())
                        .map(ndarray::Array1::from_vec);
                    LearnerBlueprint::brandom(
                        ndarray::Array1::from_vec(levels),
                        name,
                        list_optional_elt(&spec, "df")
                            .and_then(|value| value.as_real())
                            .unwrap_or(4.0),
                        lambda,
                        list_optional_elt(&spec, "center")
                            .and_then(|value| value.as_bool())
                            .unwrap_or(false),
                        by,
                        None,
                    )
                }
                "bmrf" => {
                    let levels = list_named_elt(&spec, "levels")
                        .as_integer_vector()
                        .unwrap_or_else(|| throw_r_error("bmrf learner requires integer $levels"));
                    let n_levels = list_named_elt(&spec, "n_levels")
                        .as_integer()
                        .unwrap_or(0) as usize;
                    let k_values = list_named_elt(&spec, "K")
                        .as_real_vector()
                        .unwrap_or_else(|| throw_r_error("bmrf learner requires numeric $K"));
                    let k = ndarray::Array2::from_shape_vec((n_levels, n_levels), k_values)
                        .unwrap_or_else(|error| {
                            throw_r_error(format!("bmrf K shape error: {error}"))
                        });
                    LearnerBlueprint::bmrf(
                        ndarray::Array1::from_vec(
                            levels.into_iter().map(|value| value as f64).collect(),
                        ),
                        name,
                        k,
                        list_optional_elt(&spec, "df")
                            .and_then(|value| value.as_real())
                            .unwrap_or(4.0),
                        lambda,
                        list_optional_elt(&spec, "center")
                            .and_then(|value| value.as_bool())
                            .unwrap_or(false),
                        list_optional_elt(&spec, "by_values")
                            .and_then(|value| value.as_real_vector())
                            .filter(|values| !values.is_empty())
                            .map(ndarray::Array1::from_vec),
                    )
                }
                "bspatial" => {
                    let x1 = list_named_elt(&spec, "x1")
                        .as_real_vector()
                        .unwrap_or_else(|| throw_r_error("bspatial learner requires numeric $x1"));
                    let x2 = list_named_elt(&spec, "x2")
                        .as_real_vector()
                        .unwrap_or_else(|| throw_r_error("bspatial learner requires numeric $x2"));
                    let boundary_knots1 = list_optional_elt(&spec, "bk1_lo")
                        .and_then(|value| value.as_real())
                        .zip(
                            list_optional_elt(&spec, "bk1_hi")
                                .and_then(|value| value.as_real()),
                        );
                    let boundary_knots2 = list_optional_elt(&spec, "bk2_lo")
                        .and_then(|value| value.as_real())
                        .zip(
                            list_optional_elt(&spec, "bk2_hi")
                                .and_then(|value| value.as_real()),
                        );
                    LearnerBlueprint::bspatial(
                        ndarray::Array1::from_vec(x1),
                        ndarray::Array1::from_vec(x2),
                        name,
                        list_optional_elt(&spec, "knots")
                            .and_then(|value| value.as_real())
                            .unwrap_or(10.0) as usize,
                        list_optional_elt(&spec, "degree")
                            .and_then(|value| value.as_real())
                            .unwrap_or(3.0) as usize,
                        list_optional_elt(&spec, "differences")
                            .and_then(|value| value.as_real())
                            .unwrap_or(2.0) as usize,
                        list_optional_elt(&spec, "df")
                            .and_then(|value| value.as_real())
                            .unwrap_or(6.0),
                        lambda,
                        list_optional_elt(&spec, "center")
                            .and_then(|value| value.as_bool())
                            .unwrap_or(true),
                        list_optional_elt(&spec, "by_values")
                            .and_then(|value| value.as_real_vector())
                            .filter(|values| !values.is_empty())
                            .map(ndarray::Array1::from_vec),
                        boundary_knots1,
                        boundary_knots2,
                    )
                }
                "bmono" => {
                    let x = list_named_elt(&spec, "x")
                        .as_real_vector()
                        .unwrap_or_else(|| throw_r_error("bmono learner requires numeric $x"));
                    let constraint_name = list_optional_elt(&spec, "constraint")
                        .and_then(|value| value.as_str().map(str::to_string))
                        .unwrap_or_else(|| "increasing".to_string());
                    let solver_name = list_optional_elt(&spec, "solver")
                        .and_then(|value| value.as_str().map(str::to_string))
                        .unwrap_or_else(|| "quad.prog".to_string());
                    LearnerBlueprint::bmono(
                        ndarray::Array1::from_vec(x),
                        name,
                        list_optional_elt(&spec, "knots")
                            .and_then(|value| value.as_real())
                            .unwrap_or(20.0) as usize,
                        list_optional_elt(&spec, "degree")
                            .and_then(|value| value.as_real())
                            .unwrap_or(3.0) as usize,
                        list_optional_elt(&spec, "differences")
                            .and_then(|value| value.as_real())
                            .unwrap_or(2.0) as usize,
                        list_optional_elt(&spec, "df")
                            .and_then(|value| value.as_real())
                            .unwrap_or(4.0),
                        parse_mono_constraint(&constraint_name),
                        parse_bmono_solver(&solver_name),
                        list_optional_elt(&spec, "lambda2")
                            .and_then(|value| value.as_real())
                            .unwrap_or(1e6),
                        list_optional_elt(&spec, "niter")
                            .and_then(|value| value.as_integer())
                            .unwrap_or(10) as usize,
                        // `bmono(x, by = z)`: the numeric modifier.
                        list_optional_elt(&spec, "by_values")
                            .and_then(|value| value.as_real_vector())
                            .filter(|values| !values.is_empty())
                            .map(ndarray::Array1::from_vec),
                        false,
                    )
                }
                "bconstrained" => {
                    let x = list_named_elt(&spec, "x")
                        .as_real_vector()
                        .unwrap_or_else(|| {
                            throw_r_error("bconstrained learner requires numeric $x")
                        });
                    let dims = list_named_elt(&spec, "C_dims")
                        .as_integer_vector()
                        .unwrap_or_else(|| {
                            throw_r_error("bconstrained learner requires integer $C_dims")
                        });
                    if dims.len() != 2 {
                        throw_r_error("bconstrained C_dims must contain two dimensions");
                    }
                    let rows = dims[0] as usize;
                    let columns = dims[1] as usize;
                    let c_values = list_named_elt(&spec, "C")
                        .as_real_vector()
                        .unwrap_or_else(|| {
                            throw_r_error("bconstrained learner requires numeric $C")
                        });
                    let c = ndarray::Array2::from_shape_fn(
                        (rows, columns),
                        |(row, column)| c_values[column * rows + row],
                    );
                    LearnerBlueprint::bconstrained(
                        ndarray::Array1::from_vec(x),
                        name,
                        list_optional_elt(&spec, "knots")
                            .and_then(|value| value.as_real())
                            .unwrap_or(20.0) as usize,
                        list_optional_elt(&spec, "degree")
                            .and_then(|value| value.as_real())
                            .unwrap_or(3.0) as usize,
                        list_optional_elt(&spec, "differences")
                            .and_then(|value| value.as_real())
                            .unwrap_or(2.0) as usize,
                        list_optional_elt(&spec, "df")
                            .and_then(|value| value.as_real())
                            .unwrap_or(4.0),
                        lambda,
                        c,
                        list_optional_elt(&spec, "lambda2")
                            .and_then(|value| value.as_real())
                            .unwrap_or(1e6),
                        list_optional_elt(&spec, "niter")
                            .and_then(|value| value.as_integer())
                            .unwrap_or(10) as usize,
                        None,
                    )
                    .map(|blueprint| {
                        blueprint.with_bmono_solver(parse_bmono_solver(
                            list_optional_elt(&spec, "solver")
                                .and_then(|value| value.as_str().map(str::to_string))
                                .as_deref()
                                .unwrap_or("quad.prog"),
                        ))
                    })
                }
                "tree" | "btree" => {
                    let dims = list_named_elt(&spec, "dims")
                        .as_integer_vector()
                        .unwrap_or_else(|| throw_r_error("btree learner requires integer $dims"));
                    if dims.len() != 2 {
                        throw_r_error("btree dims must contain two dimensions");
                    }
                    let rows = dims[0] as usize;
                    let columns = dims[1] as usize;
                    let values = list_named_elt(&spec, "design")
                        .as_real_vector()
                        .unwrap_or_else(|| throw_r_error("btree learner requires numeric $design"));
                    let design = ndarray::Array2::from_shape_vec((rows, columns), values)
                        .unwrap_or_else(|error| {
                            throw_r_error(format!("btree design shape error: {error}"))
                        });
                    LearnerBlueprint::btree(
                        design,
                        name,
                        list_optional_elt(&spec, "minbucket")
                            .and_then(|value| value.as_integer())
                            .unwrap_or(4) as usize,
                        list_optional_elt(&spec, "minsplit")
                            .and_then(|value| value.as_integer())
                            .unwrap_or(10) as usize,
                        list_optional_elt(&spec, "maxdepth")
                            .and_then(|value| value.as_integer())
                            .unwrap_or(1) as usize,
                        list_optional_elt(&spec, "by_values")
                            .and_then(|value| value.as_real_vector())
                            .filter(|values| !values.is_empty())
                            .map(ndarray::Array1::from_vec),
                    )
                    .map(|blueprint| {
                        // The factor columns' level counts and the split
                        // rule, as the gamboost tree arm reads them: without
                        // them a gamboostLSS tree split factor codes as
                        // ordered numbers and chose columns by SSE.
                        let categorical: Vec<usize> = list_optional_elt(&spec, "categorical")
                            .and_then(|value| value.as_integer_vector())
                            .unwrap_or_default()
                            .into_iter()
                            .map(|v| v.max(0) as usize)
                            .collect();
                        let selection = match list_optional_elt(&spec, "selection")
                            .and_then(|value| value.as_str().map(str::to_string))
                            .as_deref()
                            .unwrap_or("ctree")
                        {
                            "sse" => mboostrs_core::learners::SplitSelection::Sse,
                            "ctree_pvalue" => mboostrs_core::learners::SplitSelection::CtreePvalue,
                            _ => mboostrs_core::learners::SplitSelection::CtreeStatistic,
                        };
                        blueprint.with_categorical(categorical).with_selection(selection)
                    })
                }
                "bkernel" => {
                    let x = list_named_elt(&spec, "x")
                        .as_real_vector()
                        .unwrap_or_else(|| throw_r_error("bkernel learner requires numeric $x"));
                    let landmarks = list_optional_elt(&spec, "landmarks")
                        .and_then(|value| value.as_integer())
                        .map(|value| value as usize);
                    match landmarks {
                        None => LearnerBlueprint::bkernel(
                            ndarray::Array1::from_vec(x),
                            name,
                            list_optional_elt(&spec, "bandwidth")
                                .and_then(|value| value.as_real())
                                .unwrap_or(1.0),
                            list_optional_elt(&spec, "df")
                                .and_then(|value| value.as_real())
                                .unwrap_or(4.0),
                            list_optional_elt(&spec, "kernel")
                                .and_then(|value| value.as_str().map(str::to_string))
                                .unwrap_or_else(|| "gaussian".to_string()),
                            list_optional_elt(&spec, "center")
                                .and_then(|value| value.as_bool())
                                .unwrap_or(false),
                            None,
                        ),
                        Some(landmarks) => LearnerBlueprint::bkernel_nystrom(
                            ndarray::Array1::from_vec(x),
                            name,
                            list_optional_elt(&spec, "bandwidth")
                                .and_then(|value| value.as_real())
                                .unwrap_or(1.0),
                            list_optional_elt(&spec, "df")
                                .and_then(|value| value.as_real())
                                .unwrap_or(4.0),
                            list_optional_elt(&spec, "kernel")
                                .and_then(|value| value.as_str().map(str::to_string))
                                .unwrap_or_else(|| "gaussian".to_string()),
                            landmarks,
                            list_optional_elt(&spec, "center")
                                .and_then(|value| value.as_bool())
                                .unwrap_or(false),
                            None,
                        ),
                    }
                }
                "buser" => {
                    let dims = list_named_elt(&spec, "dims")
                        .as_integer_vector()
                        .unwrap_or_else(|| throw_r_error("buser learner requires integer $dims"));
                    if dims.len() != 2 {
                        throw_r_error("buser dims must contain two dimensions");
                    }
                    let rows = dims[0] as usize;
                    let columns = dims[1] as usize;
                    let x_values = list_named_elt(&spec, "X")
                        .as_real_vector()
                        .unwrap_or_else(|| throw_r_error("buser learner requires numeric $X"));
                    if x_values.len() != rows * columns {
                        throw_r_error("buser X length does not match dims");
                    }
                    let design = ndarray::Array2::from_shape_fn(
                        (rows, columns),
                        |(row, column)| x_values[column * rows + row],
                    );
                    let k_dims = list_named_elt(&spec, "K_dims")
                        .as_integer_vector()
                        .unwrap_or_else(|| {
                            throw_r_error("buser learner requires integer $K_dims")
                        });
                    if k_dims.len() != 2 {
                        throw_r_error("buser K_dims must contain two dimensions");
                    }
                    let k_rows = k_dims[0] as usize;
                    let k_columns = k_dims[1] as usize;
                    let k_values = list_named_elt(&spec, "K")
                        .as_real_vector()
                        .unwrap_or_else(|| throw_r_error("buser learner requires numeric $K"));
                    if k_values.len() != k_rows * k_columns {
                        throw_r_error("buser K length does not match K_dims");
                    }
                    let penalty = ndarray::Array2::from_shape_fn(
                        (k_rows, k_columns),
                        |(row, column)| k_values[column * k_rows + row],
                    );
                    LearnerBlueprint::buser(
                        design,
                        penalty,
                        name,
                        list_optional_elt(&spec, "df")
                            .and_then(|value| value.as_real())
                            .unwrap_or(4.0),
                        lambda,
                        list_optional_elt(&spec, "center")
                            .and_then(|value| value.as_bool())
                            .unwrap_or(false),
                        None,
                    )
                }
                other => throw_r_error(format!(
                    "gamboostLSS currently supports bols, bbs, bns, brandom, bmrf, bspatial, bmono, bconstrained, btree, bkernel, and buser learners; got {other}"
                )),
            }
            .unwrap_or_else(|error| throw_r_error(error.to_string()));
            parameter_blueprints.push(blueprint);
        }
        per_param.push(parameter_blueprints);
    }
    per_param
}

fn prepare_lss_blueprints_r(
    blueprints: &[Vec<LearnerBlueprint>],
    weights: ndarray::ArrayView1<f64>,
) -> Result<Vec<Vec<PreparedLearner>>, mboostrs_core::MboostrsError> {
    blueprints
        .iter()
        .map(|parameter| {
            parameter
                .iter()
                .map(|blueprint| blueprint.prepare(weights).map(|(learner, _)| learner))
                .collect()
        })
        .collect()
}

fn make_lss_family_r(family_name: &str, n_params: usize) -> Box<dyn mboostrs_core::lss::LssFamily> {
    use mboostrs_core::lss::{
        BetaLSS, BinomialLSS, DirichletLSS, GammaLSS, GaussianKnownVarLSS, GaussianLSS,
        GeometricLSS, GumbelLSS, InverseGaussianLSS, LogisticLSS, NBinomialLSS,
        NegativeBinomialLSS, Pareto2LSS, PoissonLSS, StudentTLSS, SurvivalLSS, ZIPoLSS,
        ZeroAdjustedGammaLSS, ZeroAdjustedPoissonLSS, BCCGLSS, BCPELSS, BCTLSS, JSULSS, PIGLSS,
        ZINBLSS,
    };

    match family_name {
        "gaussianlss" => Box::new(GaussianLSS),
        "gammalss" => Box::new(GammaLSS),
        "studenttlss" => Box::new(StudentTLSS),
        "betalss" => Box::new(BetaLSS),
        "nbinomiallss" => Box::new(NBinomialLSS),
        "poissonlss" => Box::new(PoissonLSS),
        "bccglss" => Box::new(BCCGLSS),
        "binomiallss" => Box::new(BinomialLSS),
        "geometriclss" => Box::new(GeometricLSS),
        "gumbellss" => Box::new(GumbelLSS),
        "inversegaussianlss" => Box::new(InverseGaussianLSS),
        "jsulss" => Box::new(JSULSS),
        "bctlss" => Box::new(BCTLSS),
        "bcpelss" => Box::new(BCPELSS),
        "logisticlss" => Box::new(LogisticLSS),
        "zeroadjustedgammalss" => Box::new(ZeroAdjustedGammaLSS),
        "negativebinomiallss" => Box::new(NegativeBinomialLSS),
        "pareto2lss" => Box::new(Pareto2LSS),
        "piglss" => Box::new(PIGLSS),
        "zeroadjustedpoissonlss" => Box::new(ZeroAdjustedPoissonLSS),
        "lognormallss" => Box::new(SurvivalLSS::log_normal()),
        "logloglss" => Box::new(SurvivalLSS::log_log()),
        "weibulllss" => Box::new(SurvivalLSS::weibull()),
        "zipolss" => Box::new(ZIPoLSS),
        "zinblss" => Box::new(ZINBLSS),
        "gaussianknownvarlss" => Box::new(GaussianKnownVarLSS),
        "dirichletlss" => Box::new(
            DirichletLSS::try_new(n_params)
                .unwrap_or_else(|error| throw_r_error(error.to_string())),
        ),
        other => throw_r_error(format!("Unknown LSS family: {other}")),
    }
}

/// Replay one distribution parameter's heterogeneous gamboostLSS path using
/// the same basis recipes created by the fit. This covers both truncated
/// in-sample prediction and new data without reconstructing centered splines
/// in R.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn predict_gamboostlss_mixed(
    learner_specs: List,
    weights: &[f64],
    offset: f64,
    selected: &[i32],
    coefs: List,
    nu: f64,
    x_data: List,
    mstop: i32,
    by_data: List,
    unseen_levels: &str,
) -> Vec<f64> {
    if mstop < 0 {
        throw_r_error("mstop must be non-negative for prediction");
    }
    let unseen = match unseen_levels {
        "zero_effect" => mboostrs_core::learners::UnseenLevels::ZeroEffect,
        "average" => mboostrs_core::learners::UnseenLevels::Average,
        _ => mboostrs_core::learners::UnseenLevels::Error,
    };
    if !offset.is_finite() || !nu.is_finite() {
        throw_r_error("prediction offset and nu must be finite");
    }
    check_finite(weights, "weights");
    let blueprints = parse_lss_learner_blueprints_r(&learner_specs);
    if blueprints.len() != 1 {
        throw_r_error("prediction requires exactly one parameter learner set");
    }
    let parameter_blueprints = &blueprints[0];
    if x_data.len() != parameter_blueprints.len() {
        throw_r_error(format!(
            "x_data contains {} learners but the fitted parameter has {}",
            x_data.len(),
            parameter_blueprints.len()
        ));
    }
    if selected.len() != coefs.len() {
        throw_r_error(format!(
            "selected has length {} but coefs has length {}",
            selected.len(),
            coefs.len()
        ));
    }

    let weights = ndarray::Array1::from_vec(weights.to_vec());
    let basis_info = parameter_blueprints
        .iter()
        .map(|blueprint| blueprint.prepare(weights.view()).map(|(_, basis)| basis))
        .collect::<Result<Vec<_>, _>>()
        .unwrap_or_else(|error| throw_r_error(error.to_string()));
    let raw_x = (0..x_data.len())
        .map(|index| {
            x_data
                .elt(index)
                .unwrap_or_else(|_| {
                    throw_r_error(format!("x_data[[{}]] is not accessible", index + 1))
                })
                .as_real_vector()
                .unwrap_or_else(|| {
                    throw_r_error(format!("x_data[[{}]] must be numeric", index + 1))
                })
        })
        .collect::<Vec<_>>();
    let n = raw_x
        .first()
        .zip(basis_info.first())
        .map_or(0, |(values, basis)| match basis {
            BasisInfo::Bspatial { .. } => values.len() / 2,
            BasisInfo::Btree { n_features, .. } => values.len() / *n_features,
            BasisInfo::Buser { n_features, .. } => values.len() / *n_features,
            _ => values.len(),
        });
    if n == 0
        || raw_x.iter().zip(&basis_info).any(|(values, basis)| {
            values.len()
                != match basis {
                    BasisInfo::Bspatial { .. } => 2 * n,
                    BasisInfo::Btree { n_features, .. } => n * *n_features,
                    BasisInfo::Buser { n_features, .. } => n * *n_features,
                    _ => n,
                }
        })
    {
        throw_r_error(
            "prediction learner vectors have inconsistent observation counts or packed design dimensions",
        );
    }
    if raw_x.iter().flatten().any(|value| !value.is_finite()) {
        throw_r_error("prediction data contains NA, NaN, or Inf values");
    }

    // The new rows' `by` modifier per learner (NULL where a learner has
    // none). A numeric `bbs(x, by = z)` multiplies its design by z, and the
    // blueprint holds the training rows' z.
    let by_rows: Vec<Option<Vec<f64>>> = (0..basis_info.len())
        .map(|index| {
            by_data
                .elt(index)
                .ok()
                .and_then(|value| value.as_real_vector())
                .filter(|values| !values.is_empty())
        })
        .collect();
    if by_rows.iter().flatten().any(|values| values.len() != n) {
        throw_r_error("by_data vectors must have one value per prediction row");
    }
    let steps = (mstop as usize).min(selected.len());
    let mut fitted = vec![offset; n];
    for step in 0..steps {
        let learner = selected[step];
        if learner <= 0 || learner as usize > basis_info.len() {
            throw_r_error(format!(
                "selected learner {} at step {} is out of range",
                learner,
                step + 1
            ));
        }
        let learner = learner as usize - 1;
        let coef = coefs
            .elt(step)
            .unwrap_or_else(|_| throw_r_error(format!("coefs[[{}]] is not accessible", step + 1)))
            .as_real_vector()
            .unwrap_or_else(|| throw_r_error(format!("coefs[[{}]] must be numeric", step + 1)));
        basis_info[learner]
            .predict_into_with(
                &coef,
                &raw_x[learner],
                &mut fitted,
                nu,
                by_rows[learner].as_deref(),
                unseen,
            )
            .unwrap_or_else(|error| throw_r_error(error.to_string()));
    }
    fitted
}

/// gamboostLSS: cyclic component-wise boosting for a GAMLSS family with smooth
/// (bbs) learners. One bbs learner per smooth predictor per parameter, no
/// separate intercept (the per-parameter offset plays that role, matching R).
/// `bbs_x` is row-major (n × total_learners): one column per bbs learner,
/// grouped by parameter. Returns full (vector) coefs per step.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn fit_gamboostlss(
    bbs_x: &[f64],
    n: i32,
    param_nlearners: &[i32],
    bbs_knots: &[i32],
    bbs_degree: &[i32],
    bbs_differences: &[i32],
    bbs_df: &[f64],
    bbs_center: &[i32],
    y: &[f64],
    weights: &[f64],
    mstop: i32,
    nu: f64,
    family_name: &str,
    offsets: &[f64],
    mstop_per_param: &[i32],
    trace: bool,
) -> List {
    use mboostrs_core::lss::{
        try_fit_lss, BetaLSS, BinomialLSS, DirichletLSS, GammaLSS, GaussianKnownVarLSS,
        GaussianLSS, GeometricLSS, GumbelLSS, InverseGaussianLSS, LogisticLSS, LssFamily,
        NBinomialLSS, NegativeBinomialLSS, Pareto2LSS, PoissonLSS, StudentTLSS, SurvivalLSS,
        ZIPoLSS, ZeroAdjustedGammaLSS, ZeroAdjustedPoissonLSS, BCCGLSS, BCPELSS, BCTLSS, JSULSS,
        PIGLSS, ZINBLSS,
    };
    let binding_started = trace.then(Instant::now);
    if n <= 0 {
        throw_r_error("n must be a positive integer");
    }
    let n = n as usize;
    let mstop = checked_mstop(mstop);
    let param_nlearners: Vec<usize> = param_nlearners.iter().map(|&c| c as usize).collect();
    let total: usize = param_nlearners.iter().sum();
    if bbs_x.len() != n * total {
        throw_r_error(format!(
            "bbs_x length {} does not match n={} * total_learners={}",
            bbs_x.len(),
            n,
            total
        ));
    }
    if bbs_knots.len() != total
        || bbs_degree.len() != total
        || bbs_differences.len() != total
        || bbs_df.len() != total
        || bbs_center.len() != total
    {
        throw_r_error("all bbs spec vectors must have length sum(param_nlearners)");
    }
    check_finite(y, "y");
    check_finite(weights, "weights");
    check_finite(bbs_x, "bbs_x");

    let preparation_timer = trace.then(PreparationTimer::start);
    let x_arr = ndarray::Array2::from_shape_vec((n, total), bbs_x.to_vec())
        .unwrap_or_else(|e| throw_r_error(format!("invalid bbs_x dimensions: {e}")));
    let w_owned = ndarray::Array1::from_vec(weights.to_vec());

    let family: Box<dyn LssFamily> = match family_name {
        "gaussianlss" => Box::new(GaussianLSS),
        "gammalss" => Box::new(GammaLSS),
        "studenttlss" => Box::new(StudentTLSS),
        "betalss" => Box::new(BetaLSS),
        "nbinomiallss" => Box::new(NBinomialLSS),
        "poissonlss" => Box::new(PoissonLSS),
        "bccglss" => Box::new(BCCGLSS),
        "binomiallss" => Box::new(BinomialLSS),
        "geometriclss" => Box::new(GeometricLSS),
        "gumbellss" => Box::new(GumbelLSS),
        "inversegaussianlss" => Box::new(InverseGaussianLSS),
        "jsulss" => Box::new(JSULSS),
        "bctlss" => Box::new(BCTLSS),
        "bcpelss" => Box::new(BCPELSS),
        "logisticlss" => Box::new(LogisticLSS),
        "zeroadjustedgammalss" => Box::new(ZeroAdjustedGammaLSS),
        "negativebinomiallss" => Box::new(NegativeBinomialLSS),
        "pareto2lss" => Box::new(Pareto2LSS),
        "piglss" => Box::new(PIGLSS),
        "zeroadjustedpoissonlss" => Box::new(ZeroAdjustedPoissonLSS),
        "lognormallss" => Box::new(SurvivalLSS::log_normal()),
        "logloglss" => Box::new(SurvivalLSS::log_log()),
        "weibulllss" => Box::new(SurvivalLSS::weibull()),
        "zipolss" => Box::new(ZIPoLSS),
        "zinblss" => Box::new(ZINBLSS),
        "gaussianknownvarlss" => Box::new(GaussianKnownVarLSS),
        "dirichletlss" => Box::new(
            DirichletLSS::try_new(param_nlearners.len())
                .unwrap_or_else(|error| throw_r_error(error.to_string())),
        ),
        other => throw_r_error(format!("Unknown LSS family: {other}")),
    };
    if param_nlearners.len() != family.n_params() {
        throw_r_error(format!(
            "expected {} parameter blocks, got {}",
            family.n_params(),
            param_nlearners.len()
        ));
    }
    let ctrl = BoostControl::new(mstop, nu)
        .unwrap_or_else(|e| throw_r_error(format!("invalid BoostControl: {e}")))
        .with_trace(trace);

    let mut per_param: Vec<Vec<PreparedLearner>> = Vec::with_capacity(param_nlearners.len());
    let mut col0 = 0usize;
    for &nl in &param_nlearners {
        let mut learners: Vec<PreparedLearner> = Vec::with_capacity(nl);
        for _ in 0..nl {
            let col = x_arr.column(col0).to_owned();
            learners.push(prepare_bbs_with_lambda(
                col.view(),
                w_owned.view(),
                bbs_knots[col0] as usize,
                bbs_degree[col0] as usize,
                bbs_differences[col0] as usize,
                bbs_df[col0],
                None,
                bbs_center[col0] != 0,
                "bbs",
                None,
                false,
                None,
                0,
            ));
            col0 += 1;
        }
        per_param.push(learners);
    }
    let preparation_timing = preparation_timer.map(PreparationTimer::finish);

    let offsets_ref = if offsets.is_empty() {
        None
    } else {
        Some(offsets.to_vec())
    };
    let mstop_pp: Option<Vec<usize>> = if mstop_per_param.is_empty() {
        None
    } else {
        Some(mstop_per_param.iter().map(|&m| m as usize).collect())
    };
    let binding_input_ns = binding_started.map(|started| started.elapsed().as_nanos() as u64);
    let path = try_fit_lss(
        &mut per_param,
        ndarray::ArrayView1::from(y),
        w_owned.view(),
        family.as_ref(),
        &ctrl,
        mstop_pp.as_deref(),
        offsets_ref.as_deref(),
    )
    .unwrap_or_else(|e| throw_r_error(format!("{e}")));

    let binding_output_started = trace.then(Instant::now);
    let results: Vec<Robj> = path
        .per_param
        .iter()
        .map(|bp| {
            let coefs: Vec<Robj> = bp.coefs.iter().map(|c| c.clone().into()).collect();
            let selected: Vec<i32> = bp.selected.iter().map(|&s| (s + 1) as i32).collect();
            list!(
                offset = bp.offset,
                selected = selected,
                coefs = List::from_values(coefs),
                fitted = bp.fitted.to_vec()
            )
            .into()
        })
        .collect();
    let mut report = path.execution_report();
    attach_frontend_timings(
        &mut report,
        preparation_timing,
        binding_input_ns,
        binding_output_started,
    );
    let execution_report = fit_report_to_r(&report);
    list!(
        results = List::from_values(results),
        param_risk = List::from_values(path.per_param.iter().map(|p| p.risk.clone())),
        risk = path.risk,
        selection_order = path
            .selection_order
            .iter()
            .map(|&p| (p + 1) as i32)
            .collect::<Vec<_>>(),
        execution_report = execution_report
    )
}

/// Heterogeneous bols/bbs/bns gamboostLSS fit. Learner specifications are nested by
/// distribution parameter and converted to the same core blueprints used for
/// every weight-dependent preparation.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn fit_gamboostlss_mixed(
    learner_specs: List,
    y: &[f64],
    weights: &[f64],
    mstop: i32,
    nu: f64,
    family_name: &str,
    offsets: &[f64],
    mstop_per_param: &[i32],
    method: &str,
    trace: bool,
    stabilization: &str,
    nu_per_param: &[f64],
    step: &str,
    step_iterations: i32,
    response_scale: &str,
) -> List {
    use mboostrs_core::lss::{
        try_fit_lss, try_fit_lss_noncyclic, BetaLSS, BinomialLSS, DirichletLSS, GammaLSS,
        GaussianKnownVarLSS, GaussianLSS, GeometricLSS, GumbelLSS, InverseGaussianLSS, LogisticLSS,
        LssFamily, NBinomialLSS, NegativeBinomialLSS, Pareto2LSS, PoissonLSS, StudentTLSS,
        SurvivalLSS, ZIPoLSS, ZeroAdjustedGammaLSS, ZeroAdjustedPoissonLSS, BCCGLSS, BCPELSS,
        BCTLSS, JSULSS, PIGLSS, ZINBLSS,
    };

    if method != "cyclic" && method != "noncyclic" {
        throw_r_error("method must be 'cyclic' or 'noncyclic'");
    }
    if method == "noncyclic" && !mstop_per_param.is_empty() {
        throw_r_error("noncyclic gamboostLSS requires one global mstop");
    }
    check_finite(y, "y");
    check_finite(weights, "weights");
    let binding_started = trace.then(Instant::now);
    let mstop = checked_mstop(mstop);
    let blueprints = parse_lss_learner_blueprints_r(&learner_specs);
    let n_params = blueprints.len();
    let family: Box<dyn LssFamily> = match family_name {
        "gaussianlss" => Box::new(GaussianLSS),
        "gammalss" => Box::new(GammaLSS),
        "studenttlss" => Box::new(StudentTLSS),
        "betalss" => Box::new(BetaLSS),
        "nbinomiallss" => Box::new(NBinomialLSS),
        "poissonlss" => Box::new(PoissonLSS),
        "bccglss" => Box::new(BCCGLSS),
        "binomiallss" => Box::new(BinomialLSS),
        "geometriclss" => Box::new(GeometricLSS),
        "gumbellss" => Box::new(GumbelLSS),
        "inversegaussianlss" => Box::new(InverseGaussianLSS),
        "jsulss" => Box::new(JSULSS),
        "bctlss" => Box::new(BCTLSS),
        "bcpelss" => Box::new(BCPELSS),
        "logisticlss" => Box::new(LogisticLSS),
        "zeroadjustedgammalss" => Box::new(ZeroAdjustedGammaLSS),
        "negativebinomiallss" => Box::new(NegativeBinomialLSS),
        "pareto2lss" => Box::new(Pareto2LSS),
        "piglss" => Box::new(PIGLSS),
        "zeroadjustedpoissonlss" => Box::new(ZeroAdjustedPoissonLSS),
        "lognormallss" => Box::new(SurvivalLSS::log_normal()),
        "logloglss" => Box::new(SurvivalLSS::log_log()),
        "weibulllss" => Box::new(SurvivalLSS::weibull()),
        "zipolss" => Box::new(ZIPoLSS),
        "zinblss" => Box::new(ZINBLSS),
        "gaussianknownvarlss" => Box::new(GaussianKnownVarLSS),
        "dirichletlss" => Box::new(
            DirichletLSS::try_new(n_params)
                .unwrap_or_else(|error| throw_r_error(error.to_string())),
        ),
        other => throw_r_error(format!("Unknown LSS family: {other}")),
    };
    if family.n_params() != n_params {
        throw_r_error(format!(
            "expected {} parameter learner sets, got {n_params}",
            family.n_params()
        ));
    }
    let weights = ndarray::Array1::from_vec(weights.to_vec());
    let control = BoostControl::new(mstop, nu)
        .unwrap_or_else(|error| throw_r_error(format!("invalid BoostControl: {error}")))
        .with_trace(trace);
    let control = lss_settings(
        control,
        stabilization,
        nu_per_param,
        step,
        step_iterations,
        response_scale,
    );
    let preparation_timer = trace.then(PreparationTimer::start);
    let mut per_param = prepare_lss_blueprints_r(&blueprints, weights.view())
        .unwrap_or_else(|error| throw_r_error(error.to_string()));
    let preparation_timing = preparation_timer.map(PreparationTimer::finish);
    let mstop_per_param: Option<Vec<usize>> = if mstop_per_param.is_empty() {
        None
    } else {
        Some(
            mstop_per_param
                .iter()
                .map(|value| *value as usize)
                .collect(),
        )
    };
    let offset_values = (!offsets.is_empty()).then(|| offsets.to_vec());
    let binding_input_ns = binding_started.map(|started| started.elapsed().as_nanos() as u64);
    let path = if method == "noncyclic" {
        try_fit_lss_noncyclic(
            &mut per_param,
            ndarray::ArrayView1::from(y),
            weights.view(),
            family.as_ref(),
            &control,
            offset_values.as_deref(),
        )
    } else {
        try_fit_lss(
            &mut per_param,
            ndarray::ArrayView1::from(y),
            weights.view(),
            family.as_ref(),
            &control,
            mstop_per_param.as_deref(),
            offset_values.as_deref(),
        )
    }
    .unwrap_or_else(|error| throw_r_error(error.to_string()));

    let binding_output_started = trace.then(Instant::now);
    let results = path
        .per_param
        .iter()
        .map(|parameter_path| {
            let coefficients = parameter_path
                .coefs
                .iter()
                .cloned()
                .map(Robj::from)
                .collect::<Vec<_>>();
            list!(
                offset = parameter_path.offset,
                selected = parameter_path
                    .selected
                    .iter()
                    .map(|index| (*index + 1) as i32)
                    .collect::<Vec<_>>(),
                coefs = List::from_values(coefficients),
                fitted = parameter_path.fitted.to_vec()
            )
            .into()
        })
        .collect::<Vec<Robj>>();
    let mut report = path.execution_report();
    attach_frontend_timings(
        &mut report,
        preparation_timing,
        binding_input_ns,
        binding_output_started,
    );
    list!(
        results = List::from_values(results),
        param_risk = List::from_values(path.per_param.iter().map(|p| p.risk.clone())),
        risk = path.risk,
        selection_order = path
            .selection_order
            .iter()
            .map(|parameter| (*parameter + 1) as i32)
            .collect::<Vec<_>>(),
        execution_report = fit_report_to_r(&report)
    )
}

/// Cross-validated risk for heterogeneous bols/bbs/bns gamboostLSS learner sets.
/// The blueprint cache owns fold-invariant spline bases and penalties; every
/// fold still recomputes its weight-dependent centering, df-to-lambda mapping,
/// and factorization.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn cvrisk_gamboostlss_mixed(
    learner_specs: List,
    y: &[f64],
    weights: &[f64],
    mstop: i32,
    nu: f64,
    family_name: &str,
    folds: &[f64],
    b: i32,
    grid: &[i32],
    n_grid: i32,
    stabilization: &str,
    nu_per_param: &[f64],
    step: &str,
    step_iterations: i32,
    response_scale: &str,
) -> Vec<f64> {
    use mboostrs_core::lss::try_cvrisk_lss_with;

    if b <= 0 || n_grid <= 0 {
        throw_r_error("cross-validation requires positive fold and grid counts");
    }
    let mstop = checked_mstop(mstop);
    check_finite(y, "y");
    check_finite(weights, "weights");
    check_finite(folds, "folds");
    let blueprints = parse_lss_learner_blueprints_r(&learner_specs);
    let n_params = blueprints.len();
    let n = weights.len();
    let b = b as usize;
    let n_grid = n_grid as usize;
    if folds.len() != n * b {
        throw_r_error(format!(
            "fold data has length {} but expected n={} * B={}",
            folds.len(),
            n,
            b
        ));
    }
    if grid.len() != n_grid * n_params || grid.iter().any(|value| *value < 0) {
        throw_r_error(format!(
            "grid must contain {n_grid} non-negative rows of length {n_params}"
        ));
    }
    let family = make_lss_family_r(family_name, n_params);
    if family.n_params() != n_params {
        throw_r_error(format!(
            "expected {} parameter learner sets, got {n_params}",
            family.n_params()
        ));
    }
    let control = BoostControl::new(mstop, nu)
        .unwrap_or_else(|error| throw_r_error(format!("invalid BoostControl: {error}")));
    let control = lss_settings(
        control,
        stabilization,
        nu_per_param,
        step,
        step_iterations,
        response_scale,
    );
    let weights = ndarray::Array1::from_vec(weights.to_vec());
    let y = ndarray::Array1::from_vec(y.to_vec());
    let folds = ndarray::Array2::from_shape_vec((n, b), folds.to_vec())
        .unwrap_or_else(|error| throw_r_error(format!("invalid fold dimensions: {error}")));
    let grid = (0..n_grid)
        .map(|row| {
            (0..n_params)
                .map(|parameter| grid[row * n_params + parameter] as usize)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let risk = try_cvrisk_lss_with(
        |fold_weights| prepare_lss_blueprints_r(&blueprints, fold_weights),
        y.view(),
        weights.view(),
        family.as_ref(),
        &control,
        folds.view(),
        None,
        &grid,
    )
    .unwrap_or_else(|error| throw_r_error(error.to_string()));
    risk.iter().copied().collect()
}

/// Per-subsample parameter-qualified learner sequences for heterogeneous
/// non-cyclic gamboostLSS, used by stability selection.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn stabsel_selected_gamboostlss_mixed(
    learner_specs: List,
    y: &[f64],
    weights: &[f64],
    mstop: i32,
    nu: f64,
    family_name: &str,
    folds: &[f64],
    b: i32,
    stabilization: &str,
    nu_per_param: &[f64],
    step: &str,
    step_iterations: i32,
    response_scale: &str,
) -> List {
    use mboostrs_core::lss::try_fit_lss_noncyclic;

    if b <= 0 {
        throw_r_error("stability selection requires at least one subsample");
    }
    let mstop = checked_mstop(mstop);
    check_finite(y, "y");
    check_finite(weights, "weights");
    check_finite(folds, "folds");
    let blueprints = parse_lss_learner_blueprints_r(&learner_specs);
    let n_params = blueprints.len();
    let family = make_lss_family_r(family_name, n_params);
    if family.n_params() != n_params {
        throw_r_error(format!(
            "expected {} parameter learner sets, got {n_params}",
            family.n_params()
        ));
    }
    let n = weights.len();
    let b = b as usize;
    if folds.len() != n * b {
        throw_r_error(format!(
            "fold data has length {} but expected n={} * B={}",
            folds.len(),
            n,
            b
        ));
    }
    let folds = ndarray::Array2::from_shape_vec((n, b), folds.to_vec())
        .unwrap_or_else(|error| throw_r_error(format!("invalid fold dimensions: {error}")));
    let weights = ndarray::Array1::from_vec(weights.to_vec());
    let y = ndarray::Array1::from_vec(y.to_vec());
    let control = BoostControl::new(mstop, nu)
        .unwrap_or_else(|error| throw_r_error(format!("invalid BoostControl: {error}")));
    let control = lss_settings(
        control,
        stabilization,
        nu_per_param,
        step,
        step_iterations,
        response_scale,
    );
    let mut parameter_offsets = Vec::with_capacity(n_params);
    let mut total = 0usize;
    for parameter in &blueprints {
        parameter_offsets.push(total);
        total += parameter.len();
    }

    let mut sequences: Vec<Robj> = Vec::with_capacity(b);
    for fold in 0..b {
        let fold_weights = ndarray::Array1::from_iter(
            weights
                .iter()
                .enumerate()
                .map(|(row, weight)| weight * folds[[row, fold]]),
        );
        let mut per_param = prepare_lss_blueprints_r(&blueprints, fold_weights.view())
            .unwrap_or_else(|error| throw_r_error(error.to_string()));
        let path = try_fit_lss_noncyclic(
            &mut per_param,
            y.view(),
            fold_weights.view(),
            family.as_ref(),
            &control,
            None,
        )
        .unwrap_or_else(|error| throw_r_error(error.to_string()));
        let mut cursors = vec![0usize; n_params];
        let sequence = path
            .selection_order
            .iter()
            .map(|parameter| {
                let learner = path.per_param[*parameter].selected[cursors[*parameter]];
                cursors[*parameter] += 1;
                (parameter_offsets[*parameter] + learner + 1) as i32
            })
            .collect::<Vec<_>>();
        sequences.push(sequence.into());
    }
    List::from_values(sequences)
}

/// Cross-validated risk for glmboostLSS over a per-parameter mstop grid (R's
/// cvrisk.mboostLSS). `x_matrix` is row-major (n x sum(param_ncols)); `folds` is
/// row-major (n x b) in-bag weights; `grid` is row-major (n_grid x k) of
/// per-parameter mstop values. Returns a row-major (b x n_grid) flat vector of
/// per-fold out-of-bag risks.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn cvrisk_glmboostlss(
    x_matrix: &[f64],
    n: i32,
    param_ncols: &[i32],
    y: &[f64],
    weights: &[f64],
    mstop: i32,
    nu: f64,
    family_name: &str,
    folds: &[f64],
    b: i32,
    grid: &[i32],
    n_grid: i32,
    stabilization: &str,
    nu_per_param: &[f64],
    step: &str,
    step_iterations: i32,
    response_scale: &str,
) -> Vec<f64> {
    use mboostrs_core::lss::{
        BetaLSS, BinomialLSS, DirichletLSS, GammaLSS, GaussianKnownVarLSS, GaussianLSS,
        GeometricLSS, GumbelLSS, InverseGaussianLSS, LogisticLSS, LssFamily, NBinomialLSS,
        NegativeBinomialLSS, Pareto2LSS, PoissonLSS, StudentTLSS, SurvivalLSS, ZIPoLSS,
        ZeroAdjustedGammaLSS, ZeroAdjustedPoissonLSS, BCCGLSS, BCPELSS, BCTLSS, JSULSS, PIGLSS,
        ZINBLSS,
    };
    let n = n as usize;
    let b = b as usize;
    let n_grid = n_grid as usize;
    let mstop = checked_mstop(mstop);
    let param_ncols: Vec<usize> = param_ncols.iter().map(|&c| c as usize).collect();
    let k = param_ncols.len();
    let total: usize = param_ncols.iter().sum();
    check_finite(y, "y");
    check_finite(weights, "weights");
    check_finite(x_matrix, "x_matrix");

    let x_arr = ndarray::Array2::from_shape_vec((n, total), x_matrix.to_vec())
        .unwrap_or_else(|e| throw_r_error(format!("invalid x_matrix dimensions: {e}")));
    let w_owned = ndarray::Array1::from_vec(weights.to_vec());
    let ones = ndarray::Array1::ones(n);

    let family: Box<dyn LssFamily> = match family_name {
        "gaussianlss" => Box::new(GaussianLSS),
        "gammalss" => Box::new(GammaLSS),
        "studenttlss" => Box::new(StudentTLSS),
        "betalss" => Box::new(BetaLSS),
        "nbinomiallss" => Box::new(NBinomialLSS),
        "poissonlss" => Box::new(PoissonLSS),
        "bccglss" => Box::new(BCCGLSS),
        "binomiallss" => Box::new(BinomialLSS),
        "geometriclss" => Box::new(GeometricLSS),
        "gumbellss" => Box::new(GumbelLSS),
        "inversegaussianlss" => Box::new(InverseGaussianLSS),
        "jsulss" => Box::new(JSULSS),
        "bctlss" => Box::new(BCTLSS),
        "bcpelss" => Box::new(BCPELSS),
        "logisticlss" => Box::new(LogisticLSS),
        "zeroadjustedgammalss" => Box::new(ZeroAdjustedGammaLSS),
        "negativebinomiallss" => Box::new(NegativeBinomialLSS),
        "pareto2lss" => Box::new(Pareto2LSS),
        "piglss" => Box::new(PIGLSS),
        "zeroadjustedpoissonlss" => Box::new(ZeroAdjustedPoissonLSS),
        "lognormallss" => Box::new(SurvivalLSS::log_normal()),
        "logloglss" => Box::new(SurvivalLSS::log_log()),
        "weibulllss" => Box::new(SurvivalLSS::weibull()),
        "zipolss" => Box::new(ZIPoLSS),
        "zinblss" => Box::new(ZINBLSS),
        "gaussianknownvarlss" => Box::new(GaussianKnownVarLSS),
        "dirichletlss" => Box::new(
            DirichletLSS::try_new(param_ncols.len())
                .unwrap_or_else(|error| throw_r_error(error.to_string())),
        ),
        other => throw_r_error(format!("Unknown LSS family: {other}")),
    };
    let ctrl = BoostControl::new(mstop, nu)
        .unwrap_or_else(|e| throw_r_error(format!("invalid BoostControl: {e}")));
    let ctrl = lss_settings(
        ctrl,
        stabilization,
        nu_per_param,
        step,
        step_iterations,
        response_scale,
    );

    let mut per_param: Vec<Vec<PreparedLearner>> = Vec::with_capacity(k);
    let mut col0 = 0usize;
    for &pc in &param_ncols {
        let mut learners: Vec<PreparedLearner> = Vec::with_capacity(pc + 1);
        learners.push(prepare_scalar_bols(
            ones.view(),
            w_owned.view(),
            0.0,
            false,
            "(Intercept)",
            None,
        ));
        for j in 0..pc {
            let col = x_arr.column(col0 + j);
            // gamboostLSS centers every covariate by its simple column mean
            // (weights enter the normal equations and the offset, not the
            // centering). Weighted means here would break predict/fitted
            // agreement and upstream parity under weights.
            let mean = col.mean().unwrap_or(0.0);
            learners.push(prepare_scalar_bols_fixed_center(
                col,
                w_owned.view(),
                0.0,
                mean,
                "x",
            ));
        }
        per_param.push(learners);
        col0 += pc;
    }

    // R passes `t(folds)` so the column-major R vector is observation-major,
    // matching ndarray's default row-major `(n, b)` layout.
    let folds_arr = ndarray::Array2::from_shape_vec((n, b), folds.to_vec())
        .unwrap_or_else(|e| throw_r_error(format!("invalid folds dimensions: {e}")));
    let grid: Vec<Vec<usize>> = (0..n_grid)
        .map(|g| (0..k).map(|p| grid[g * k + p] as usize).collect())
        .collect();

    let risk = mboostrs_core::lss::try_cvrisk_lss(
        &per_param,
        ndarray::ArrayView1::from(y),
        w_owned.view(),
        family.as_ref(),
        &ctrl,
        folds_arr.view(),
        &grid,
    )
    .unwrap_or_else(|e| throw_r_error(e.to_string()));
    // row-major (b x n_grid)
    risk.iter().copied().collect()
}

/// Non-cyclic glmboostLSS (R's method="noncyclic"). Same intercept + scalar-bols
/// learners as fit_glmboostlss; runs the non-cyclic driver (one parameter per
/// step, chosen by risk reduction). Returns 1-based selected indices + scalar
/// coefs per parameter.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn fit_glmboostlss_noncyclic(
    x_matrix: &[f64],
    n: i32,
    param_ncols: &[i32],
    y: &[f64],
    weights: &[f64],
    mstop: i32,
    nu: f64,
    family_name: &str,
    offsets: &[f64],
    trace: bool,
    stabilization: &str,
    nu_per_param: &[f64],
    step: &str,
    step_iterations: i32,
    response_scale: &str,
) -> List {
    use mboostrs_core::lss::{
        try_fit_lss_noncyclic, BetaLSS, BinomialLSS, DirichletLSS, GammaLSS, GaussianKnownVarLSS,
        GaussianLSS, GeometricLSS, GumbelLSS, InverseGaussianLSS, LogisticLSS, LssFamily,
        NBinomialLSS, NegativeBinomialLSS, Pareto2LSS, PoissonLSS, StudentTLSS, SurvivalLSS,
        ZIPoLSS, ZeroAdjustedGammaLSS, ZeroAdjustedPoissonLSS, BCCGLSS, BCPELSS, BCTLSS, JSULSS,
        PIGLSS, ZINBLSS,
    };
    let binding_started = trace.then(Instant::now);
    let n = n as usize;
    let mstop = checked_mstop(mstop);
    let param_ncols: Vec<usize> = param_ncols.iter().map(|&c| c as usize).collect();
    let total: usize = param_ncols.iter().sum();
    check_finite(y, "y");
    check_finite(weights, "weights");
    check_finite(x_matrix, "x_matrix");
    let preparation_timer = trace.then(PreparationTimer::start);
    let x_arr = ndarray::Array2::from_shape_vec((n, total), x_matrix.to_vec())
        .unwrap_or_else(|e| throw_r_error(format!("invalid x_matrix dimensions: {e}")));
    let w_owned = ndarray::Array1::from_vec(weights.to_vec());
    let ones = ndarray::Array1::ones(n);
    let family: Box<dyn LssFamily> = match family_name {
        "gaussianlss" => Box::new(GaussianLSS),
        "gammalss" => Box::new(GammaLSS),
        "studenttlss" => Box::new(StudentTLSS),
        "betalss" => Box::new(BetaLSS),
        "nbinomiallss" => Box::new(NBinomialLSS),
        "poissonlss" => Box::new(PoissonLSS),
        "bccglss" => Box::new(BCCGLSS),
        "binomiallss" => Box::new(BinomialLSS),
        "geometriclss" => Box::new(GeometricLSS),
        "gumbellss" => Box::new(GumbelLSS),
        "inversegaussianlss" => Box::new(InverseGaussianLSS),
        "jsulss" => Box::new(JSULSS),
        "bctlss" => Box::new(BCTLSS),
        "bcpelss" => Box::new(BCPELSS),
        "logisticlss" => Box::new(LogisticLSS),
        "zeroadjustedgammalss" => Box::new(ZeroAdjustedGammaLSS),
        "negativebinomiallss" => Box::new(NegativeBinomialLSS),
        "pareto2lss" => Box::new(Pareto2LSS),
        "piglss" => Box::new(PIGLSS),
        "zeroadjustedpoissonlss" => Box::new(ZeroAdjustedPoissonLSS),
        "lognormallss" => Box::new(SurvivalLSS::log_normal()),
        "logloglss" => Box::new(SurvivalLSS::log_log()),
        "weibulllss" => Box::new(SurvivalLSS::weibull()),
        "zipolss" => Box::new(ZIPoLSS),
        "zinblss" => Box::new(ZINBLSS),
        "gaussianknownvarlss" => Box::new(GaussianKnownVarLSS),
        "dirichletlss" => Box::new(
            DirichletLSS::try_new(param_ncols.len())
                .unwrap_or_else(|error| throw_r_error(error.to_string())),
        ),
        other => throw_r_error(format!("Unknown LSS family: {other}")),
    };
    let ctrl = BoostControl::new(mstop, nu)
        .unwrap_or_else(|e| throw_r_error(format!("invalid BoostControl: {e}")))
        .with_trace(trace);
    let ctrl = lss_settings(
        ctrl,
        stabilization,
        nu_per_param,
        step,
        step_iterations,
        response_scale,
    );
    let mut per_param: Vec<Vec<PreparedLearner>> = Vec::with_capacity(param_ncols.len());
    let mut col0 = 0usize;
    for &pc in &param_ncols {
        let mut learners: Vec<PreparedLearner> = Vec::with_capacity(pc + 1);
        learners.push(prepare_scalar_bols(
            ones.view(),
            w_owned.view(),
            0.0,
            false,
            "(Intercept)",
            None,
        ));
        for j in 0..pc {
            let col = x_arr.column(col0 + j);
            // gamboostLSS centers every covariate by its simple column mean
            // (weights enter the normal equations and the offset, not the
            // centering). Weighted means here would break predict/fitted
            // agreement and upstream parity under weights.
            let mean = col.mean().unwrap_or(0.0);
            learners.push(prepare_scalar_bols_fixed_center(
                col,
                w_owned.view(),
                0.0,
                mean,
                "x",
            ));
        }
        per_param.push(learners);
        col0 += pc;
    }
    let preparation_timing = preparation_timer.map(PreparationTimer::finish);
    let offsets_ref = if offsets.is_empty() {
        None
    } else {
        Some(offsets.to_vec())
    };
    let binding_input_ns = binding_started.map(|started| started.elapsed().as_nanos() as u64);
    let path = try_fit_lss_noncyclic(
        &mut per_param,
        ndarray::ArrayView1::from(y),
        w_owned.view(),
        family.as_ref(),
        &ctrl,
        offsets_ref.as_deref(),
    )
    .unwrap_or_else(|e| throw_r_error(e.to_string()));
    let binding_output_started = trace.then(Instant::now);
    let results: Vec<Robj> = path
        .per_param
        .iter()
        .map(|bp| {
            let coefs_flat: Vec<f64> = bp.coefs.iter().map(|c| c[0]).collect();
            let selected: Vec<i32> = bp.selected.iter().map(|&s| (s + 1) as i32).collect();
            list!(
                offset = bp.offset,
                selected = selected,
                coefs = coefs_flat,
                fitted = bp.fitted.to_vec()
            )
            .into()
        })
        .collect();
    let mut report = path.execution_report();
    attach_frontend_timings(
        &mut report,
        preparation_timing,
        binding_input_ns,
        binding_output_started,
    );
    let execution_report = fit_report_to_r(&report);
    list!(
        results = List::from_values(results),
        param_risk = List::from_values(path.per_param.iter().map(|p| p.risk.clone())),
        risk = path.risk,
        selection_order = path
            .selection_order
            .iter()
            .map(|&p| (p + 1) as i32)
            .collect::<Vec<_>>(),
        execution_report = execution_report
    )
}

/// Per-subsample combined base-learner selection sequences for non-cyclic
/// glmboostLSS (stability selection). `x_matrix` is row-major (n x total);
/// `folds` is row-major (n x b) in-bag weights. Returns a list of `b` integer
/// vectors, each the ordered 1-based combined-learner index sequence (parameter
/// p's learners occupy a contiguous block, intercept first).
#[extendr]
#[allow(clippy::too_many_arguments)]
fn stabsel_selected_glmboostlss(
    x_matrix: &[f64],
    n: i32,
    param_ncols: &[i32],
    y: &[f64],
    _weights: &[f64],
    mstop: i32,
    nu: f64,
    family_name: &str,
    folds: &[f64],
    b: i32,
    stabilization: &str,
    nu_per_param: &[f64],
    step: &str,
    step_iterations: i32,
    response_scale: &str,
) -> List {
    use mboostrs_core::lss::{
        try_fit_lss_noncyclic, BetaLSS, BinomialLSS, DirichletLSS, GammaLSS, GaussianKnownVarLSS,
        GaussianLSS, GeometricLSS, GumbelLSS, InverseGaussianLSS, LogisticLSS, LssFamily,
        NBinomialLSS, NegativeBinomialLSS, Pareto2LSS, PoissonLSS, StudentTLSS, SurvivalLSS,
        ZIPoLSS, ZeroAdjustedGammaLSS, ZeroAdjustedPoissonLSS, BCCGLSS, BCPELSS, BCTLSS, JSULSS,
        PIGLSS, ZINBLSS,
    };
    let n = n as usize;
    let b = b as usize;
    let mstop = checked_mstop(mstop);
    let param_ncols: Vec<usize> = param_ncols.iter().map(|&c| c as usize).collect();
    let total: usize = param_ncols.iter().sum();
    let x_arr = ndarray::Array2::from_shape_vec((n, total), x_matrix.to_vec())
        .unwrap_or_else(|e| throw_r_error(format!("invalid x_matrix dimensions: {e}")));
    let folds_arr = ndarray::Array2::from_shape_vec((n, b), folds.to_vec())
        .unwrap_or_else(|e| throw_r_error(format!("invalid folds dimensions: {e}")));
    let family: Box<dyn LssFamily> = match family_name {
        "gaussianlss" => Box::new(GaussianLSS),
        "gammalss" => Box::new(GammaLSS),
        "studenttlss" => Box::new(StudentTLSS),
        "betalss" => Box::new(BetaLSS),
        "nbinomiallss" => Box::new(NBinomialLSS),
        "poissonlss" => Box::new(PoissonLSS),
        "bccglss" => Box::new(BCCGLSS),
        "binomiallss" => Box::new(BinomialLSS),
        "geometriclss" => Box::new(GeometricLSS),
        "gumbellss" => Box::new(GumbelLSS),
        "inversegaussianlss" => Box::new(InverseGaussianLSS),
        "jsulss" => Box::new(JSULSS),
        "bctlss" => Box::new(BCTLSS),
        "bcpelss" => Box::new(BCPELSS),
        "logisticlss" => Box::new(LogisticLSS),
        "zeroadjustedgammalss" => Box::new(ZeroAdjustedGammaLSS),
        "negativebinomiallss" => Box::new(NegativeBinomialLSS),
        "pareto2lss" => Box::new(Pareto2LSS),
        "piglss" => Box::new(PIGLSS),
        "zeroadjustedpoissonlss" => Box::new(ZeroAdjustedPoissonLSS),
        "lognormallss" => Box::new(SurvivalLSS::log_normal()),
        "logloglss" => Box::new(SurvivalLSS::log_log()),
        "weibulllss" => Box::new(SurvivalLSS::weibull()),
        "zipolss" => Box::new(ZIPoLSS),
        "zinblss" => Box::new(ZINBLSS),
        "gaussianknownvarlss" => Box::new(GaussianKnownVarLSS),
        "dirichletlss" => Box::new(
            DirichletLSS::try_new(param_ncols.len())
                .unwrap_or_else(|error| throw_r_error(error.to_string())),
        ),
        other => throw_r_error(format!("Unknown LSS family: {other}")),
    };
    let ctrl = BoostControl::new(mstop, nu)
        .unwrap_or_else(|e| throw_r_error(format!("invalid BoostControl: {e}")));
    let ctrl = lss_settings(
        ctrl,
        stabilization,
        nu_per_param,
        step,
        step_iterations,
        response_scale,
    );
    let mut offset = vec![0usize; param_ncols.len()];
    let mut acc = 0usize;
    for (p, &pc) in param_ncols.iter().enumerate() {
        offset[p] = acc;
        acc += pc + 1;
    }
    let mut result: Vec<Robj> = Vec::with_capacity(b);
    for fold in 0..b {
        let fw = ndarray::Array1::from_shape_fn(n, |i| folds_arr[[i, fold]]);
        let sw: f64 = fw.sum();
        let ones = ndarray::Array1::ones(n);
        let mut per_param: Vec<Vec<PreparedLearner>> = Vec::with_capacity(param_ncols.len());
        let mut col0 = 0usize;
        for &pc in &param_ncols {
            let mut learners: Vec<PreparedLearner> = Vec::with_capacity(pc + 1);
            learners.push(prepare_scalar_bols(
                ones.view(),
                fw.view(),
                0.0,
                false,
                "(Intercept)",
                None,
            ));
            for j in 0..pc {
                let col = x_arr.column(col0 + j);
                let mean = if sw > 0.0 {
                    col.iter().zip(fw.iter()).map(|(c, wi)| c * wi).sum::<f64>() / sw
                } else {
                    0.0
                };
                learners.push(prepare_scalar_bols_fixed_center(
                    col,
                    fw.view(),
                    0.0,
                    mean,
                    "x",
                ));
            }
            per_param.push(learners);
            col0 += pc;
        }
        let path = try_fit_lss_noncyclic(
            &mut per_param,
            ndarray::ArrayView1::from(y),
            fw.view(),
            family.as_ref(),
            &ctrl,
            None,
        )
        .unwrap_or_else(|e| throw_r_error(e.to_string()));
        let mut cursor = vec![0usize; param_ncols.len()];
        let seq: Vec<i32> = path
            .selection_order
            .iter()
            .map(|&p| {
                let l = path.per_param[p].selected[cursor[p]];
                cursor[p] += 1;
                (offset[p] + l + 1) as i32 // 1-based for R
            })
            .collect();
        result.push(seq.into());
    }
    List::from_values(result)
}

/// Raw (un-centered) B-spline basis at `x` using the given boundary knots,
/// matching the core's `prepare_bbs_with_lambda` design (center = FALSE). Returned
/// observation-major (row 0's `p` values, then row 1's, ...) as a flat vector
/// plus the number of basis columns `p`. The R front-end uses this so the bbs
/// design seen by both fitting and prediction is the same one `mboost` uses,
/// rather than an `splines::bs` approximation.
#[extendr]
fn r_bbs_design(x: &[f64], bk_lo: f64, bk_hi: f64, n_knots: i32, degree: i32, deriv: i32) -> List {
    let knots =
        mboostrs_core::learners::bspline_knots((bk_lo, bk_hi), n_knots as usize, degree as usize);
    let basis = if deriv > 0 {
        bspline_basis_with_deriv(x, &knots, degree as usize, deriv as usize)
    } else {
        mboostrs_core::learners::bspline_basis_extrap(x, &knots, degree as usize, (bk_lo, bk_hi))
    };
    let p = basis.ncols();
    let flat: Vec<f64> = basis.iter().copied().collect(); // ndarray iter is row-major (obs-major)
    list!(design = flat, p = p as i32)
}

/// Joint support of new rows (`m x p`, column-major `new`) relative to
/// training rows (`n x p`, column-major `train`): the squared Mahalanobis
/// distance from the training mean and the share of training rows at least
/// as far (`mboostrs_core::support::joint_support`).
#[extendr]
fn r_joint_support(train: &[f64], n: i32, new: &[f64], m: i32, p: i32) -> List {
    let (n, m, p) = (n as usize, m as usize, p as usize);
    if train.len() != n * p || new.len() != m * p {
        throw_r_error("joint support: matrix lengths do not match their dimensions");
    }
    let train = ndarray::Array2::from_shape_fn((n, p), |(i, j)| train[j * n + i]);
    let new = ndarray::Array2::from_shape_fn((m, p), |(i, j)| new[j * m + i]);
    match mboostrs_core::support::joint_support(train.view(), new.view()) {
        Some(rows) => list!(
            distance_sq = rows.iter().map(|r| r.distance_sq).collect::<Vec<f64>>(),
            training_share_beyond = rows
                .iter()
                .map(|r| r.training_share_beyond)
                .collect::<Vec<f64>>()
        ),
        None => throw_r_error("joint support needs at least two training rows"),
    }
}

/// Difference penalty `D'D` of order `differences` over `p` coefficients,
/// column-major, the penalty of a `bbs()` basis with `p` columns.
#[extendr]
fn r_diff_penalty(p: i32, differences: i32) -> Vec<f64> {
    let k = mboostrs_core::diff_penalty(p as usize, differences as usize);
    k.t().iter().copied().collect()
}

/// Build the *centered* B-spline design a `bbs(center = TRUE)` fit uses.
///
/// `bbs(center = TRUE)` removes the penalty's null space by a spectral
/// decomposition, so the fitted design has `p - d` columns where the raw basis
/// has `p`, and the fit's coefficients are in that reduced space. The R
/// prediction bridge rebuilt the raw basis, found `p` columns against `p - d`
/// coefficients, and refused -- correctly, because multiplying them would have
/// been silently wrong, but it left `predict(newdata=)` unavailable for a
/// default `bbs(center = TRUE)` and for every `bspatial`, whose `center`
/// defaults to TRUE.
///
/// The transform depends only on the penalty, not on the data, so applying it
/// to a basis built at new `x` values gives coordinates in the same reduced
/// space as the fit. That is what makes new-data prediction possible at all.
#[extendr]
fn r_bbs_design_centered(
    x: &[f64],
    bk_lo: f64,
    bk_hi: f64,
    n_knots: i32,
    degree: i32,
    differences: i32,
    deriv: i32,
) -> List {
    let knots =
        mboostrs_core::learners::bspline_knots((bk_lo, bk_hi), n_knots as usize, degree as usize);
    let basis = if deriv > 0 {
        bspline_basis_with_deriv(x, &knots, degree as usize, deriv as usize)
    } else {
        mboostrs_core::learners::bspline_basis_extrap(x, &knots, degree as usize, (bk_lo, bk_hi))
    };
    // The fit centers by the difference matrix -- `X · D'(DD')⁻¹`, R's
    // `"differenceMatrix"` method -- and falls back to the spectral transform
    // only when `D·D'` is not factorable. Rebuilding with the spectral one
    // instead lands in a *different* reduced space: it runs, and it is wrong
    // by an amount that looks like a poor fit rather than an error (measured:
    // an in-sample replay off by 0.80 where the uncentered path replays to
    // 9e-16). So the order here has to match `prepare_bbs`'s exactly.
    let penalty = mboostrs_core::learners::diff_penalty(basis.ncols(), differences as usize);
    let centered = mboostrs_core::learners::raw_diff_matrix(basis.ncols(), differences as usize)
        .and_then(|d_raw| mboostrs_core::learners::center_difference_matrix(&basis, &d_raw))
        .unwrap_or_else(|| {
            mboostrs_core::learners::center_spectral(&basis, &penalty, differences as usize)
        });
    let p = centered.ncols();
    let flat: Vec<f64> = centered.iter().copied().collect();
    list!(design = flat, p = p as i32)
}

/// Build the bspatial tensor-product B-spline design for a 2-d spline surface.
/// Returns observation-major flat design vector + number of basis columns.
#[extendr]
fn r_bspatial_design(
    x1: &[f64],
    x2: &[f64],
    bk1_lo: f64,
    bk1_hi: f64,
    bk2_lo: f64,
    bk2_hi: f64,
    n_knots: i32,
    degree: i32,
    differences: i32,
    center: bool,
) -> List {
    let n = x1.len();
    let k = n_knots as usize;
    let d = degree as usize;
    let knots1 = mboostrs_core::learners::bspline_knots((bk1_lo, bk1_hi), k, d);
    let basis1 = mboostrs_core::learners::bspline_basis_extrap(x1, &knots1, d, (bk1_lo, bk1_hi));
    let knots2 = mboostrs_core::learners::bspline_knots((bk2_lo, bk2_hi), k, d);
    let basis2 = mboostrs_core::learners::bspline_basis_extrap(x2, &knots2, d, (bk2_lo, bk2_hi));
    let p1 = basis1.ncols();
    let p2 = basis2.ncols();
    let p = p1 * p2;
    let mut design = vec![0.0_f64; n * p];
    for i in 0..n {
        for j1 in 0..p1 {
            let b1 = basis1[(i, j1)];
            for j2 in 0..p2 {
                design[i * p + j1 * p2 + j2] = b1 * basis2[(i, j2)];
            }
        }
    }

    if !center {
        return list!(
            design = design,
            p = p as i32,
            p1 = p1 as i32,
            p2 = p2 as i32
        );
    }

    // `bspatial` centers by the *spectral* transform of the tensor penalty
    // (`prepare_bspatial`), not by the difference matrix a one-dimensional
    // `bbs` uses. This argument was accepted and ignored, so a centered
    // surface came back with its raw `p1 * p2` columns against a fit whose
    // coefficients had fewer, and the R prediction bridge refused. `center`
    // defaults to TRUE for `bspatial`, so that was every surface.
    let differences = differences as usize;
    let penalty1 = mboostrs_core::learners::diff_penalty(p1, differences);
    let penalty2 = mboostrs_core::learners::diff_penalty(p2, differences);
    let tensor_penalty = mboostrs_core::learners::kron(&penalty1, &ndarray::Array2::eye(p2))
        + mboostrs_core::learners::kron(&ndarray::Array2::eye(p1), &penalty2);
    let raw = match ndarray::Array2::from_shape_vec((n, p), design) {
        Ok(matrix) => matrix,
        Err(error) => throw_r_error(format!("bspatial design shape error: {error}")),
    };
    let centered = mboostrs_core::learners::center_spectral(&raw, &tensor_penalty, differences);
    let reduced = centered.ncols();
    let flat: Vec<f64> = centered.iter().copied().collect();
    list!(
        design = flat,
        p = reduced as i32,
        p1 = p1 as i32,
        p2 = p2 as i32
    )
}

/// Build the bns (unpenalised B-spline) design matrix.
/// Returns observation-major flat design vector + number of basis columns.
#[extendr]
fn r_bns_design(x: &[f64], bk_lo: f64, bk_hi: f64, n_knots: i32, degree: i32) -> List {
    let knots =
        mboostrs_core::learners::bspline_knots((bk_lo, bk_hi), n_knots as usize, degree as usize);
    let basis =
        mboostrs_core::learners::bspline_basis_extrap(x, &knots, degree as usize, (bk_lo, bk_hi));
    let p = basis.ncols();
    let flat: Vec<f64> = basis.iter().copied().collect();
    list!(design = flat, p = p as i32)
}

/// Cross-validated risk for smooth GAMLSS (gamboostLSS with bbs learners).
/// Builds bbs templates once, rebuilds df->lambda per fold, evaluates OOB
/// risk on the actual held-out rows for every grid point.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn cvrisk_gamboostlss(
    bbs_x: &[f64],
    n: i32,
    param_nlearners: &[i32],
    bbs_knots: &[i32],
    bbs_degree: &[i32],
    bbs_differences: &[i32],
    bbs_df: &[f64],
    bbs_center: &[i32],
    y: &[f64],
    weights: &[f64],
    mstop: i32,
    nu: f64,
    family_name: &str,
    folds: &[f64],
    b: i32,
    grid: &[i32],
    n_grid: i32,
) -> Vec<f64> {
    use mboostrs_core::lss::{
        try_cvrisk_lss_with, BetaLSS, BinomialLSS, DirichletLSS, GammaLSS, GaussianKnownVarLSS,
        GaussianLSS, GeometricLSS, GumbelLSS, InverseGaussianLSS, LogisticLSS, LssFamily,
        NBinomialLSS, NegativeBinomialLSS, Pareto2LSS, PoissonLSS, StudentTLSS, SurvivalLSS,
        ZIPoLSS, ZeroAdjustedGammaLSS, ZeroAdjustedPoissonLSS, BCCGLSS, BCPELSS, BCTLSS, JSULSS,
        PIGLSS, ZINBLSS,
    };
    let n = n as usize;
    let b = b as usize;
    let n_grid = n_grid as usize;
    let mstop = checked_mstop(mstop);
    let param_nlearners: Vec<usize> = param_nlearners.iter().map(|&c| c as usize).collect();
    let k = param_nlearners.len();
    let total: usize = param_nlearners.iter().sum();
    check_finite(y, "y");
    check_finite(weights, "weights");
    check_finite(bbs_x, "bbs_x");

    if bbs_x.len() != n * total {
        throw_r_error(format!(
            "bbs_x length {} does not match n={} * total_learners={}",
            bbs_x.len(),
            n,
            total
        ));
    }
    if bbs_knots.len() != total
        || bbs_degree.len() != total
        || bbs_differences.len() != total
        || bbs_df.len() != total
        || bbs_center.len() != total
    {
        throw_r_error("all bbs spec vectors must have length sum(param_nlearners)");
    }

    let family: Box<dyn LssFamily> = match family_name {
        "gaussianlss" => Box::new(GaussianLSS),
        "gammalss" => Box::new(GammaLSS),
        "studenttlss" => Box::new(StudentTLSS),
        "betalss" => Box::new(BetaLSS),
        "nbinomiallss" => Box::new(NBinomialLSS),
        "poissonlss" => Box::new(PoissonLSS),
        "bccglss" => Box::new(BCCGLSS),
        "binomiallss" => Box::new(BinomialLSS),
        "geometriclss" => Box::new(GeometricLSS),
        "gumbellss" => Box::new(GumbelLSS),
        "inversegaussianlss" => Box::new(InverseGaussianLSS),
        "jsulss" => Box::new(JSULSS),
        "bctlss" => Box::new(BCTLSS),
        "bcpelss" => Box::new(BCPELSS),
        "logisticlss" => Box::new(LogisticLSS),
        "zeroadjustedgammalss" => Box::new(ZeroAdjustedGammaLSS),
        "negativebinomiallss" => Box::new(NegativeBinomialLSS),
        "pareto2lss" => Box::new(Pareto2LSS),
        "piglss" => Box::new(PIGLSS),
        "zeroadjustedpoissonlss" => Box::new(ZeroAdjustedPoissonLSS),
        "lognormallss" => Box::new(SurvivalLSS::log_normal()),
        "logloglss" => Box::new(SurvivalLSS::log_log()),
        "weibulllss" => Box::new(SurvivalLSS::weibull()),
        "zipolss" => Box::new(ZIPoLSS),
        "zinblss" => Box::new(ZINBLSS),
        "gaussianknownvarlss" => Box::new(GaussianKnownVarLSS),
        "dirichletlss" => Box::new(
            DirichletLSS::try_new(k).unwrap_or_else(|error| throw_r_error(error.to_string())),
        ),
        other => throw_r_error(format!("Unknown LSS family: {other}")),
    };
    if k != family.n_params() {
        throw_r_error(format!(
            "expected {} parameter blocks, got {}",
            family.n_params(),
            k
        ));
    }
    let ctrl = BoostControl::new(mstop, nu)
        .unwrap_or_else(|e| throw_r_error(format!("invalid BoostControl: {e}")));

    let x_arr = ndarray::Array2::from_shape_vec((n, total), bbs_x.to_vec())
        .unwrap_or_else(|e| throw_r_error(format!("invalid bbs_x dimensions: {e}")));
    let w_owned = ndarray::Array1::from_vec(weights.to_vec());
    let y_arr = ndarray::Array1::from_vec(y.to_vec());
    let folds_arr = ndarray::Array2::from_shape_vec((n, b), folds.to_vec())
        .unwrap_or_else(|e| throw_r_error(format!("invalid folds dimensions: {e}")));
    let grid: Vec<Vec<usize>> = (0..n_grid)
        .map(|g| (0..k).map(|p| grid[g * k + p] as usize).collect())
        .collect();

    // Build fold-invariant templates (basis + penalty) once.
    let mut templates: Vec<(ndarray::Array2<f64>, ndarray::Array2<f64>)> =
        Vec::with_capacity(total);
    let mut idx = 0usize;
    for &nl in &param_nlearners {
        for _ in 0..nl {
            // ndarray columns are strided when `total > 1`; the spline
            // template builder uses contiguous slice kernels.
            let col = x_arr.column(idx).to_owned();
            let (design, penalty) = build_bbs_template(
                col.view(),
                bbs_knots[idx] as usize,
                bbs_degree[idx] as usize,
                bbs_differences[idx] as usize,
                bbs_center[idx] != 0,
                None,
                false,
                None,
                0,
            );
            templates.push((design, penalty));
            idx += 1;
        }
    }

    let risk = try_cvrisk_lss_with(
        |fold_w| {
            let mut per_param: Vec<Vec<PreparedLearner>> =
                Vec::with_capacity(param_nlearners.len());
            let mut idx = 0usize;
            for &nl in &param_nlearners {
                let mut learners: Vec<PreparedLearner> = Vec::with_capacity(nl);
                for _ in 0..nl {
                    let (design, penalty) = templates[idx].clone();
                    learners.push(prepare_bbs_from_template(
                        design,
                        penalty,
                        fold_w,
                        bbs_df[idx],
                        None,
                        &format!("bbs_{idx}"),
                    ));
                    idx += 1;
                }
                per_param.push(learners);
            }
            Ok(per_param)
        },
        y_arr.view(),
        w_owned.view(),
        family.as_ref(),
        &ctrl,
        folds_arr.view(),
        None,
        &grid,
    )
    .unwrap_or_else(|e| throw_r_error(e.to_string()));
    // row-major (b x n_grid)
    risk.iter().copied().collect()
}

/// Compute PropOdds cumulative-logit response probabilities from link-scale
/// eta and K-1 threshold vectors (sigma), one per row.
/// Returns a flat Vec<f64> of length n*K (row-major).
#[extendr]
fn propodds_response_r(f: &[f64], sigma: &[f64]) -> Vec<f64> {
    mboostrs_core::propodds_response(f, sigma)
}

/// Compute the AIC path with the core's criterion, for the R frontend.
///
/// The R frontend used to reimplement AIC in `mboostrs_extra.R`, which made it
/// a third implementation beside the core's and Python's. It disagreed with
/// both: its `method = "corrected"` computed a textbook AICc,
/// `-2*loglik + k*df + correction`, where the core computes `mboost`'s
/// log-scale Hurvich-Clifford form, so the two differed by ~500 units on the
/// same model; and its `method = "gmdl"` was the corrected branch again with
/// `pmax` in place of `ifelse`, returning bit-identical values under a
/// different criterion's name.
///
/// `risk` has length `mstop + 1` (index 0 is the offset-only state) and `df`
/// has length `mstop`, matching the core's expectation. Returns the AIC path
/// and the 1-based optimal iteration.
#[extendr]
fn compute_aic_r(method: &str, risk: &[f64], df: &[f64], sumw: f64, k: f64, y_ss: f64) -> List {
    if risk.len() != df.len() + 1 {
        throw_r_error(format!(
            "risk must have one more entry than df (got {} and {})",
            risk.len(),
            df.len()
        ));
    }
    let (aic, mstop) = match mboostrs_core::try_compute_aic(method, risk, df, sumw, k, y_ss) {
        Ok(result) => result,
        Err(e) => throw_r_error(format!("compute_aic: {e}")),
    };
    list!(aic = aic, mstop = mstop as i32)
}

/// Risk reduction and selection frequency per group of updates.
///
/// `groups` holds the 0-based group of each update; `n_groups` is the number
/// of groups, and a group no update selected reports zero. See
/// `mboostrs_core::inference::varimp_attribution`.
#[extendr]
fn varimp_attribution_r(changes: &[f64], groups: &[i32], n_groups: i32, percent: bool) -> List {
    if groups.iter().any(|g| *g < 0) || n_groups < 0 {
        throw_r_error("varimp: group indices and the group count must be non-negative");
    }
    let groups: Vec<usize> = groups.iter().map(|&g| g as usize).collect();
    let out = match mboostrs_core::inference::varimp_attribution(
        changes,
        &groups,
        n_groups as usize,
        percent,
    ) {
        Ok(out) => out,
        Err(e) => throw_r_error(format!("varimp: {e}")),
    };
    list!(reduction = out.reduction, selfreq = out.selfreq)
}

/// Preliminary exploration over a row-major design, with the same report as Python.
#[extendr]
fn r_preexplore(
    design: &[f64],
    y: &[f64],
    categorical: &[i32],
    transforms: &[i32],
    engine: &str,
    seed: i32,
    rounds: i32,
    pairs: bool,
) -> String {
    let p = categorical.len();
    if p == 0
        || design.len() != y.len() * p
        || seed < 0
        || rounds <= 0
        || categorical.iter().any(|v| *v < 0)
        || transforms.iter().any(|v| *v < 0 || *v > 2)
    {
        throw_r_error("invalid pre-exploration dimensions or controls");
    }
    let x = ndarray::ArrayView2::from_shape((y.len(), p), design)
        .unwrap_or_else(|e| throw_r_error(e.to_string()));
    mboostrs_core::preexplore::preexplore(
        x,
        y,
        &categorical.iter().map(|v| *v as usize).collect::<Vec<_>>(),
        &transforms.iter().map(|v| *v as u8).collect::<Vec<_>>(),
        engine,
        seed as u64,
        rounds as u32,
        pairs,
    )
    .unwrap_or_else(|e| throw_r_error(e.to_string()))
    .to_string()
}

#[extendr]
fn r_response_score(
    y: &[f64],
    predicted: &[f64],
    weights: &[f64],
    classes: i32,
    metric: &str,
    parameter: Robj,
) -> List {
    if classes <= 0 {
        throw_r_error("classes must be positive");
    }
    let p = ndarray::ArrayView2::from_shape((y.len(), classes as usize), predicted)
        .unwrap_or_else(|e| throw_r_error(e.to_string()));
    let parameter = if parameter.is_null() {
        None
    } else {
        Some(
            parameter
                .as_real()
                .unwrap_or_else(|| throw_r_error("parameter must be numeric or NULL")),
        )
    };
    let (mean, losses) = mboostrs_core::scoring::response_score(y, p, weights, metric, parameter)
        .unwrap_or_else(|e| throw_r_error(e.to_string()));
    list!(mean = mean, losses = losses)
}

#[extendr]
fn r_inverse_response_transform(values: &[f64], kind: &str) -> Vec<f64> {
    mboostrs_core::scoring::inverse_response_transform(values, kind)
        .unwrap_or_else(|e| throw_r_error(e.to_string()))
}

// Conformal interval numerics; see mboostrs_core::conformal. Counts and ranks
// cross the boundary as doubles, and a missing CV+ guarantee as NaN.

#[extendr]
fn r_conformal_min_calibration_size(level: f64) -> f64 {
    mboostrs_core::conformal::min_calibration_size(level)
        .unwrap_or_else(|e| throw_r_error(e.to_string())) as f64
}

#[extendr]
fn r_conformal_calibration_count(n: i32, fraction: f64) -> f64 {
    if n < 0 {
        throw_r_error("n must be non-negative");
    }
    mboostrs_core::conformal::calibration_count(n as usize, fraction)
        .unwrap_or_else(|e| throw_r_error(e.to_string())) as f64
}

#[extendr]
fn r_discovery_learning_rate(family: &str) -> f64 {
    mboostrs_core::control::discovery_learning_rate(family)
}

#[extendr]
fn r_discovery_mean_step(family: &str) -> String {
    mboostrs_core::control::discovery_mean_step(family).to_string()
}

#[extendr]
fn r_prefer_simpler_cv(simple: &[f64], rich: &[f64], multiple: f64) -> List {
    let (prefer_simple, difference, se) =
        mboostrs_core::contribution::prefer_simpler_cv(simple, rich, multiple)
            .unwrap_or_else(|e| throw_r_error(e.to_string()));
    list!(
        prefer_simple = prefer_simple,
        difference = difference,
        se = se
    )
}

#[extendr]
fn r_conformal_quantile(scores: &[f64], level: f64) -> f64 {
    mboostrs_core::conformal::conformal_quantile(scores, level)
        .unwrap_or_else(|e| throw_r_error(e.to_string()))
}

#[extendr]
fn r_conformal_absolute_residuals(y: &[f64], predicted: &[f64]) -> Vec<f64> {
    mboostrs_core::conformal::absolute_residuals(y, predicted)
        .unwrap_or_else(|e| throw_r_error(e.to_string()))
}

#[extendr]
fn r_conformal_location_scale_scores(y: &[f64], mu: &[f64], sigma: &[f64]) -> Vec<f64> {
    mboostrs_core::conformal::location_scale_scores(y, mu, sigma)
        .unwrap_or_else(|e| throw_r_error(e.to_string()))
}

#[extendr]
fn r_conformal_location_scale_band(mu: &[f64], sigma: &[f64], q: f64) -> List {
    let (lower, upper) = mboostrs_core::conformal::location_scale_band(mu, sigma, q)
        .unwrap_or_else(|e| throw_r_error(e.to_string()));
    list!(lower = lower, upper = upper)
}

#[extendr]
fn r_conformal_symmetric_band(predicted: &[f64], q: f64) -> List {
    let (lower, upper) = mboostrs_core::conformal::symmetric_band(predicted, q);
    list!(lower = lower, upper = upper)
}

#[extendr]
fn r_conformal_poisson_scores(y: &[f64], predicted: &[f64], level: f64, score: &str) -> Vec<f64> {
    mboostrs_core::conformal::poisson_scores(y, predicted, level, score)
        .unwrap_or_else(|e| throw_r_error(e.to_string()))
}

#[extendr]
fn r_conformal_poisson_band(predicted: &[f64], level: f64, q: f64, score: &str) -> List {
    let (lower, upper) = mboostrs_core::conformal::poisson_band(predicted, level, q, score)
        .unwrap_or_else(|e| throw_r_error(e.to_string()));
    list!(lower = lower, upper = upper)
}

#[extendr]
fn r_conformal_ordered_pair(a: &[f64], b: &[f64]) -> List {
    let (lower, upper) = mboostrs_core::conformal::ordered_pair(a, b)
        .unwrap_or_else(|e| throw_r_error(e.to_string()));
    list!(lower = lower, upper = upper)
}

#[extendr]
fn r_rearrange_levels(predictions: &[f64], n_rows: i32, n_levels: i32) -> List {
    use ndarray::ShapeBuilder;
    let view =
        ndarray::ArrayView2::from_shape((n_rows as usize, n_levels as usize).f(), predictions)
            .unwrap_or_else(|e| throw_r_error(e.to_string()));
    let r = mboostrs_core::rearrange::rearrange_levels(view)
        .unwrap_or_else(|e| throw_r_error(e.to_string()));
    let sorted: Vec<f64> = r.sorted.t().iter().copied().collect();
    list!(
        sorted = sorted,
        crossing_rows = r.crossing_rows as i32,
        crossing_share = r.crossing_share,
        pair_crossings = r
            .pair_crossings
            .iter()
            .map(|v| *v as i32)
            .collect::<Vec<i32>>(),
        max_crossing = r.max_crossing
    )
}

#[extendr]
fn r_conformal_count_default_score(family: &str, y: &[f64]) -> String {
    mboostrs_core::count_conformal::default_score(family, y).to_owned()
}

#[extendr]
fn r_conformal_count_reference(
    y: &[f64],
    predicted: &[f64],
    weights: &[f64],
    zero_inflated: bool,
) -> List {
    let (pi, ratio) =
        mboostrs_core::count_conformal::fit_reference(y, predicted, weights, zero_inflated)
            .unwrap_or_else(|e| throw_r_error(e.to_string()));
    list!(
        zero_probability = pi,
        dispersion_ratio = ratio,
        automatic_score = mboostrs_core::count_conformal::automatic_score(ratio)
    )
}

#[extendr]
fn r_conformal_count_uniforms(n: i32, seed: f64, stream: i32, offset: i32) -> List {
    if n < 0
        || stream < 0
        || offset < 0
        || (!seed.is_nan()
            && (!seed.is_finite()
                || !(0.0..=9_007_199_254_740_991.0).contains(&seed)
                || seed.fract() != 0.0))
    {
        throw_r_error(
            "count randomization sizes must be nonnegative and seed an integer in [0, 2^53-1]",
        );
    }
    let supplied = if seed.is_nan() {
        None
    } else {
        Some(seed as u64)
    };
    let (actual, values) = mboostrs_core::count_conformal::uniforms(
        n as usize,
        supplied,
        stream as u64,
        offset as usize,
    )
    .unwrap_or_else(|e| throw_r_error(e.to_string()));
    list!(seed = actual as f64, values = values)
}

#[extendr]
fn r_conformal_count_rank_scores(
    y: &[f64],
    predicted: &[f64],
    zero_probability: f64,
    uniforms: &[f64],
    level: f64,
) -> Vec<f64> {
    mboostrs_core::count_conformal::scores(y, predicted, zero_probability, uniforms, level)
        .unwrap_or_else(|e| throw_r_error(e.to_string()))
}

#[extendr]
fn r_conformal_count_rank_band(
    predicted: &[f64],
    zero_probability: f64,
    q: f64,
    uniforms: &[f64],
    level: f64,
) -> List {
    let (lower, upper) =
        mboostrs_core::count_conformal::band(predicted, zero_probability, q, uniforms, level)
            .unwrap_or_else(|e| throw_r_error(e.to_string()));
    list!(lower = lower, upper = upper)
}

#[extendr]
fn r_conformal_count_cardinalities(lower: &[f64], upper: &[f64]) -> Vec<f64> {
    mboostrs_core::count_conformal::cardinalities(lower, upper)
        .unwrap_or_else(|e| throw_r_error(e.to_string()))
}

#[extendr]
fn r_conformal_quantile_scores(lower: &[f64], upper: &[f64], y: &[f64]) -> Vec<f64> {
    mboostrs_core::conformal::quantile_scores(lower, upper, y)
        .unwrap_or_else(|e| throw_r_error(e.to_string()))
}

#[extendr]
fn r_conformal_widen_band(lower: &[f64], upper: &[f64], q: f64) -> List {
    let (lo, hi) = mboostrs_core::conformal::widen_band(lower, upper, q)
        .unwrap_or_else(|e| throw_r_error(e.to_string()));
    list!(lower = lo, upper = hi)
}

#[extendr]
fn r_conformal_band_midpoint(lower: &[f64], upper: &[f64]) -> Vec<f64> {
    mboostrs_core::conformal::band_midpoint(lower, upper)
        .unwrap_or_else(|e| throw_r_error(e.to_string()))
}

#[extendr]
fn r_conformal_clip_to_band(values: &[f64], lower: &[f64], upper: &[f64]) -> Vec<f64> {
    mboostrs_core::conformal::clip_to_band(values, lower, upper)
        .unwrap_or_else(|e| throw_r_error(e.to_string()))
}

#[extendr]
fn r_conformal_cv_plus_guaranteed_coverage(n: i32, k: i32, level: f64) -> f64 {
    if n < 0 || k < 0 {
        throw_r_error("n and n_folds must be non-negative");
    }
    mboostrs_core::conformal::cv_plus_guaranteed_coverage(n as usize, k as usize, level)
        .unwrap_or_else(|e| throw_r_error(e.to_string()))
        .unwrap_or(f64::NAN)
}

/// CV+ endpoints. `fold_predictions` is the K x m prediction matrix in R's
/// column-major layout, so element (k, j) sits at `k + j * K`.
#[extendr]
fn r_conformal_cv_plus_bounds(
    residuals: &[f64],
    fold_of: &[i32],
    fold_predictions: &[f64],
    n_folds: i32,
    level: f64,
) -> List {
    if n_folds <= 0 || !fold_predictions.len().is_multiple_of(n_folds as usize) {
        throw_r_error("fold_predictions must hold one column per target for each fold");
    }
    let k = n_folds as usize;
    let m = fold_predictions.len() / k;
    let folds: Vec<usize> = fold_of
        .iter()
        .map(|&f| {
            usize::try_from(f)
                .unwrap_or_else(|_| throw_r_error("fold indices must be non-negative"))
        })
        .collect();
    let preds = ndarray::Array2::from_shape_fn((k, m), |(i, j)| fold_predictions[i + j * k]);
    let (lower, upper) =
        mboostrs_core::conformal::cv_plus_bounds(residuals, &folds, preds.view(), level)
            .unwrap_or_else(|e| throw_r_error(e.to_string()));
    list!(lower = lower, upper = upper)
}

#[extendr]
fn r_conformal_tuned_mstop_start(model_mstop: i32) -> f64 {
    mboostrs_core::conformal::tuned_mstop_start(model_mstop.max(0) as usize) as f64
}

#[extendr]
fn r_conformal_tuned_mstop_next(largest: f64, cap: f64) -> f64 {
    mboostrs_core::conformal::tuned_mstop_next(largest as usize, cap as usize) as f64
}

#[extendr]
fn r_conformal_tuned_mstop_choice(optimal: f64) -> f64 {
    mboostrs_core::conformal::tuned_mstop_choice(optimal.max(0.0) as usize) as f64
}

#[extendr]
fn r_conformal_worst_covariate_bin(
    x: &[f64],
    lower: &[f64],
    upper: &[f64],
    y: &[f64],
    level: f64,
    bins: i32,
) -> Vec<f64> {
    if bins < 0 {
        throw_r_error("bins must be non-negative");
    }
    let worst =
        mboostrs_core::conformal::worst_covariate_bin(x, lower, upper, y, level, bins as usize)
            .unwrap_or_else(|e| throw_r_error(e.to_string()));
    vec![
        worst.z,
        worst.bin as f64,
        worst.coverage,
        worst.count as f64,
    ]
}

#[extendr]
fn r_response_score_path(
    y: &[f64],
    predicted: &[f64],
    weights: &[f64],
    positions: i32,
    metric: &str,
    parameter: Robj,
) -> Vec<f64> {
    if positions <= 0 {
        throw_r_error("positions must be positive");
    }
    let p = ndarray::ArrayView2::from_shape((positions as usize, y.len()), predicted)
        .unwrap_or_else(|e| throw_r_error(e.to_string()));
    let parameter = if parameter.is_null() {
        None
    } else {
        Some(
            parameter
                .as_real()
                .unwrap_or_else(|| throw_r_error("parameter must be numeric or NULL")),
        )
    };
    mboostrs_core::scoring::response_score_path(y, p, weights, metric, parameter)
        .unwrap_or_else(|e| throw_r_error(e.to_string()))
}

#[extendr]
fn r_select_score_path(scores: &[f64], folds: i32) -> List {
    if folds <= 0 || !scores.len().is_multiple_of(folds as usize) {
        throw_r_error("invalid fold score dimensions");
    }
    let scores =
        ndarray::ArrayView2::from_shape((folds as usize, scores.len() / folds as usize), scores)
            .unwrap_or_else(|e| throw_r_error(e.to_string()));
    let (curve, best) = mboostrs_core::scoring::select_score_path(scores)
        .unwrap_or_else(|e| throw_r_error(e.to_string()));
    list!(curve = curve, mstop = best as i32)
}

/// CV with a frontend recipe rebuilt inside every training fold.
#[extendr]
fn cvrisk_foldwise_r(
    family_name: &str,
    family_params: List,
    n_folds: i32,
    mstop: i32,
    provider: Robj,
) -> List {
    if n_folds <= 0 || mstop <= 0 || !provider.is_function() {
        throw_r_error("fold-local CV needs folds, a positive mstop and a fold provider");
    }
    let family = make_boxed_family(family_name, &extract_family_params(&family_params));
    let result = cv::try_cvrisk_foldwise(family.as_ref(), n_folds as usize, mstop as usize, |b| {
        let error = |e: extendr_api::Error| {
            mboostrs_core::MboostrsError::Control(format!(
                "CV fold {} could not rebuild its preprocessing: {e}",
                b + 1
            ))
        };
        let data = provider.call(pairlist!(b as i32 + 1)).map_err(error)?;
        let data = data.as_list().ok_or_else(|| {
            mboostrs_core::MboostrsError::Shape(
                "fold provider must return response, weights, predictions and nuisance".into(),
            )
        })?;
        let numeric = |i| -> Result<Vec<f64>, mboostrs_core::MboostrsError> {
            data.elt(i).map_err(error)?.as_real_vector().ok_or_else(|| {
                mboostrs_core::MboostrsError::Shape("fold data must be numeric".into())
            })
        };
        let vectors = |i| -> Result<Vec<Vec<f64>>, mboostrs_core::MboostrsError> {
            let value = data.elt(i).map_err(error)?;
            let list = value.as_list().ok_or_else(|| {
                mboostrs_core::MboostrsError::Shape("fold paths must be lists".into())
            })?;
            list.values()
                .map(|row| {
                    row.as_real_vector().ok_or_else(|| {
                        mboostrs_core::MboostrsError::Shape("fold paths must be numeric".into())
                    })
                })
                .collect()
        };
        let paths = vectors(2)?;
        let columns = paths.first().map_or(0, Vec::len);
        if paths.iter().any(|row| row.len() != columns) {
            return Err(mboostrs_core::MboostrsError::Shape(
                "fold path rows must have equal length".into(),
            ));
        }
        let predictions = ndarray::Array2::from_shape_vec(
            (paths.len(), columns),
            paths.into_iter().flatten().collect(),
        )
        .map_err(|e| mboostrs_core::MboostrsError::Shape(e.to_string()))?;
        Ok(cv::CvPredictionFold {
            response: ndarray::Array1::from_vec(numeric(0)?),
            weights: ndarray::Array1::from_vec(numeric(1)?),
            predictions,
            nuisance: vectors(3)?,
        })
    })
    .unwrap_or_else(|e| throw_r_error(e.to_string()));
    list!(
        risk_matrix = result.risk_matrix.iter().copied().collect::<Vec<_>>(),
        n_folds = n_folds,
        n_cols = mstop + 1,
        optimal_mstop = result.optimal_mstop() as i32,
        aggregate = result.aggregate.to_vec(),
    )
}

#[extendr]
fn r_group_components(codes: &[f64], columns: i32) -> Vec<i32> {
    if columns <= 0 || !codes.len().is_multiple_of(columns as usize) {
        throw_r_error("invalid group code dimensions");
    }
    let codes =
        ndarray::ArrayView2::from_shape((codes.len() / columns as usize, columns as usize), codes)
            .unwrap_or_else(|e| throw_r_error(e.to_string()));
    mboostrs_core::cv::group_components(codes)
        .unwrap_or_else(|e| throw_r_error(e.to_string()))
        .into_iter()
        .map(|i| i32::try_from(i).unwrap_or_else(|_| throw_r_error("too many groups")))
        .collect()
}

/// Steps at which a fit's in-bag risk path rose: `c(count, largest relative
/// rise, final rise above the minimum, best step)`, or `NULL` when there
/// is no material rise. The core handles comparable profiled likelihoods
/// and omits changing adaptive losses.
#[extendr]
fn r_risk_path_rises(family_name: &str, family_params: List, risk: &[f64]) -> Robj {
    let params = extract_family_params(&family_params);
    let family = make_boxed_family(family_name, &params);
    match mboostrs_core::path::risk_rise_warning(family.as_ref(), risk) {
        Some(warning) => Robj::from(vec![
            warning.count as f64,
            warning.largest_step_rise,
            warning.final_above_best,
            warning.best_step as f64,
        ]),
        None => Robj::from(()),
    }
}

/// Construct a radial basis and retain its centres and whitening transform.
#[extendr]
fn r_brad_basis(x: &[f64], n: i32, p: i32, knots: i32, theta: f64, lambda: f64) -> List {
    if n < 1 || p < 1 || knots < 1 {
        throw_r_error("brad requires rows, columns and positive knots");
    }
    let x = ndarray::Array2::from_shape_vec((n as usize, p as usize), x.to_vec())
        .unwrap_or_else(|e| throw_r_error(format!("brad coordinate shape: {e}")));
    let recipe = mboostrs_core::radial::RadialBasis::new(
        x.view(),
        knots as usize,
        if theta < 0.0 { None } else { Some(theta) },
    )
    .unwrap_or_else(|e| throw_r_error(e.to_string()));
    let design = recipe
        .design(x.view())
        .unwrap_or_else(|e| throw_r_error(e.to_string()));
    let k = recipe.centres.nrows();
    let scale = if lambda < 0.0 { 1.0 } else { lambda };
    if !scale.is_finite() {
        throw_r_error("brad lambda must be finite");
    }
    let penalty = ndarray::Array2::<f64>::eye(k).mapv(|v| v * scale);
    list!(
        design = design.iter().copied().collect::<Vec<f64>>(),
        penalty = penalty.iter().copied().collect::<Vec<f64>>(),
        centres = recipe.centres.iter().copied().collect::<Vec<f64>>(),
        whitening = recipe.whitening.iter().copied().collect::<Vec<f64>>(),
        theta = recipe.theta,
        k = k as i32,
        p = p
    )
}

/// Evaluate a fitted radial recipe at new locations.
#[extendr]
fn r_brad_design(
    x: &[f64],
    n: i32,
    p: i32,
    centres: &[f64],
    whitening: &[f64],
    k: i32,
    theta: f64,
) -> Vec<f64> {
    if n < 0 || p < 1 || k < 1 {
        throw_r_error("invalid brad prediction dimensions");
    }
    let matrix = |nr, nc, values: &[f64]| {
        ndarray::Array2::from_shape_vec((nr, nc), values.to_vec())
            .unwrap_or_else(|e| throw_r_error(format!("brad prediction shape: {e}")))
    };
    let x = matrix(n as usize, p as usize, x);
    let recipe = mboostrs_core::radial::RadialBasis {
        centres: matrix(k as usize, p as usize, centres),
        whitening: matrix(k as usize, k as usize, whitening),
        theta,
    };
    recipe
        .design(x.view())
        .unwrap_or_else(|e| throw_r_error(e.to_string()))
        .iter()
        .copied()
        .collect()
}

/// Apply the LSS-only settings an R frontend passes to a fit, cross-validation
/// or stability-selection binding: gradient `stabilization` (`"none"`,
/// `"MAD"`, `"L2"`; upstream's `GaussianLSS(stabilization = ...)`) and
/// per-parameter step lengths, in family parameter order (empty: the scalar
/// `nu` for every parameter).
fn lss_settings(
    mut control: BoostControl,
    stabilization: &str,
    nu_per_param: &[f64],
    step: &str,
    step_iterations: i32,
    response_scale: &str,
) -> BoostControl {
    use mboostrs_core::control::ResponseScale;
    control.stabilization = mboostrs_core::lss::Stabilization::parse(stabilization)
        .unwrap_or_else(|error| throw_r_error(format!("invalid stabilization: {error}")));
    if !nu_per_param.is_empty() {
        control.nu_per_param = Some(nu_per_param.to_vec());
    }
    // The R frontend passes "auto", "none" or the scale as text.
    let scale = match response_scale {
        "auto" => ResponseScale::Auto,
        "none" => ResponseScale::Raw,
        text => ResponseScale::Fixed(text.parse::<f64>().unwrap_or_else(|_| {
            throw_r_error(format!(
                "response_scale must be \"auto\", \"none\" or a positive number, got {text:?}"
            ))
        })),
    };
    let control = control
        .with_response_scale(scale)
        .unwrap_or_else(|error| throw_r_error(format!("{error}")));
    apply_r_step(control, step, step_iterations)
}

/// The automatic response scale of a distributional fit (the weighted SD of
/// the scaled part of `y`), or `NA` for a family with no scale or a response
/// with no spread.
#[extendr]
fn lss_auto_response_scale_r(family_name: &str, n_params: i32, y: &[f64], weights: &[f64]) -> f64 {
    let family = mboostrs_core::lss::make_family(family_name, n_params as usize)
        .unwrap_or_else(|error| throw_r_error(format!("{error}")));
    mboostrs_core::lss::auto_response_scale(family.as_ref(), y, weights).unwrap_or(f64::NAN)
}

/// A spline spec's numeric `by = z` modifier (`$by_values`), checked to have
/// one value per row; `None` when the learner has none.
fn r_by_values(spec: &List, n: usize, name: &str) -> Option<ndarray::Array1<f64>> {
    let values = list_optional_elt(spec, "by_values")
        .and_then(|value| value.as_real_vector())
        .filter(|values| !values.is_empty())?;
    if values.len() != n {
        throw_r_error(format!(
            "{name}: by has {} values for {n} rows",
            values.len()
        ));
    }
    Some(ndarray::Array1::from_vec(values))
}

/// LSS distribution helpers of the core, for R's `lss_pdf`, `lss_cdf`,
/// `lss_quantile`, `lss_mean` and `lss_deviance_increment`. `kind` is `"pdf"`
/// (`mboostrs_core::lss_distribution::lss_pdf`), `"cdf"` (`lss_cdf`),
/// `"deviance"` (`lss_deviance_increment`), `"mean"` (`lss_mean`, which
/// takes the parameter columns alone) or `"quantile"`
/// (`mboostrs_core::lss_quantile::lss_quantile`, whose first column holds the
/// probabilities).
/// `values` holds `n_cols` columns of `n_rows` values each, column after
/// column, so R passes `unlist(columns)`. Returns the values and the number of
/// rows whose special-function evaluation did not converge.
#[extendr]
fn r_lss_distribution(kind: &str, family: &str, values: &[f64], n_rows: i32, n_cols: i32) -> List {
    let n_rows = n_rows.max(0) as usize;
    let n_cols = n_cols.max(0) as usize;
    if values.len() != n_rows * n_cols {
        throw_r_error(format!(
            "LSS {kind}: {} values for {n_cols} columns of {n_rows} rows",
            values.len()
        ));
    }
    let columns: Vec<Vec<f64>> = (0..n_cols)
        .map(|j| values[j * n_rows..(j + 1) * n_rows].to_vec())
        .collect();
    use mboostrs_core::lss_distribution as dist;
    let out = match kind {
        "pdf" => dist::lss_pdf(family, &columns),
        "cdf" => dist::lss_cdf(family, &columns),
        "mean" => dist::lss_mean(family, &columns).map(|values| dist::LssValues {
            values,
            unconverged: 0,
        }),
        "quantile" => {
            mboostrs_core::lss_quantile::lss_quantile(family, &columns).map(|q| dist::LssValues {
                values: q.values,
                unconverged: q.unconverged,
            })
        }
        "deviance" => dist::lss_deviance_increment(family, &columns),
        other => throw_r_error(format!("unknown LSS distribution helper '{other}'")),
    }
    .unwrap_or_else(|e| throw_r_error(e.to_string()));
    list!(values = out.values, unconverged = out.unconverged as i32)
}

/// Percentile interval of bootstrap draws (`mboostrs_core::inference::
/// percentile_interval`): one lower and one upper bound per column, by the
/// type-7 quantile R's `quantile()` uses. `values` holds the `n_rows` draws of
/// `n_cols` coefficients row-major, so R passes `as.double(t(draws))`. A
/// column with no finite value comes back as `NaN` in both bounds.
#[extendr]
fn r_percentile_interval(values: &[f64], n_rows: i32, n_cols: i32, level: f64) -> List {
    let n_rows = n_rows.max(0) as usize;
    let n_cols = n_cols.max(0) as usize;
    if values.len() != n_rows * n_cols {
        throw_r_error(format!(
            "percentile interval: {} values for {n_rows} draws x {n_cols} coefficients",
            values.len()
        ));
    }
    let draws = ndarray::Array2::from_shape_vec((n_rows, n_cols), values.to_vec())
        .unwrap_or_else(|e| throw_r_error(format!("percentile interval shape: {e}")));
    let (lower, upper) = mboostrs_core::inference::percentile_interval(draws.view(), level, false)
        .unwrap_or_else(|e| throw_r_error(e.to_string()));
    list!(lower = lower, upper = upper)
}

#[extendr]
fn r_centered_group_effects(coef: &[f64], centering: &[f64]) -> Vec<f64> {
    mboostrs_core::path::centered_group_effects(coef, centering)
        .unwrap_or_else(|e| throw_r_error(e.to_string()))
}

extendr_module! {
    mod mboostrs;
    fn lss_auto_response_scale_r;
    fn r_centered_group_effects;
    fn r_lss_distribution;
    fn r_percentile_interval;
    use gam_refit;
    use selection_stats;
    use transform_bindings;
    fn build_info_r;
    fn r_risk_path_rises;
    fn r_brad_basis;
    fn r_brad_design;
    fn r_bbs_design;
    fn r_diff_penalty;
    fn r_joint_support;
    fn r_bbs_design_centered;
    fn r_bspatial_design;
    fn r_bns_design;
    fn fit_glmboost;
    fn cvrisk_glmboostlss;
    fn fit_glmboostlss_noncyclic;
    fn stabsel_selected_glmboostlss;
    fn fit_glmboostlss;
    fn fit_gamboostlss;
    fn fit_gamboostlss_mixed;
    fn predict_gamboostlss_mixed;
    fn cvrisk_gamboostlss_mixed;
    fn stabsel_selected_gamboostlss_mixed;
    fn predict_glmboost;
    fn cv_weights;
    fn cvrisk_glmboost;
    fn fit_gamboost;
    fn gamboost_concurvity_r;
    fn glmboost_concurvity_r;
    fn gamboost_hat_r;
    fn coef_glmboost_r;
    fn gamboost_learner_info_r;
    fn family_class_r;
    fn glmboost_design_r;
    fn glmboost_hat_r;
    fn predict_gamboost;
    fn predict_gamboost_kronecker;
    fn multinomial_staged_nll;
    fn r_tree_predict;
    fn r_preexplore;
    fn r_response_score;
    fn r_inverse_response_transform;
    fn r_conformal_min_calibration_size;
    fn r_conformal_calibration_count;
    fn r_discovery_learning_rate;
    fn r_discovery_mean_step;
    fn r_prefer_simpler_cv;
    fn r_conformal_quantile;
    fn r_conformal_absolute_residuals;
    fn r_conformal_symmetric_band;
    fn r_conformal_location_scale_scores;
    fn r_conformal_location_scale_band;
    fn r_conformal_poisson_scores;
    fn r_conformal_poisson_band;
    fn r_conformal_count_default_score;
    fn r_conformal_count_reference;
    fn r_conformal_count_uniforms;
    fn r_conformal_count_rank_scores;
    fn r_conformal_count_rank_band;
    fn r_conformal_count_cardinalities;
    fn r_conformal_ordered_pair;
    fn r_rearrange_levels;
    fn r_conformal_quantile_scores;
    fn r_conformal_widen_band;
    fn r_conformal_band_midpoint;
    fn r_conformal_clip_to_band;
    fn r_conformal_cv_plus_guaranteed_coverage;
    fn r_conformal_cv_plus_bounds;
    fn r_conformal_tuned_mstop_start;
    fn r_conformal_tuned_mstop_next;
    fn r_conformal_tuned_mstop_choice;
    fn r_conformal_worst_covariate_bin;
    fn r_response_score_path;
    fn r_select_score_path;
    fn cvrisk_foldwise_r;
    fn r_group_components;
    fn stabsel_params_rust;
    fn stabsel_frequencies_r;
    fn stabsel_glmboost_r;
    fn gamboost_hat_approx_r;
    fn r_learner_effect;
    fn gamboost_group_effect_r;
    fn gamboost_grouped_contribution_r;
    fn cvrisk_gamboostlss;
    fn propodds_response_r;
    fn compute_aic_r;
    fn varimp_attribution_r;
}

#[no_mangle]
#[allow(non_snake_case)]
pub extern "C" fn R_init_mboostrs(info: *mut extendr_api::DllInfo) {
    mboostrs_core::linalg::configure_faer();
    unsafe {
        extendr_api::register_call_methods(info, get_mboostrs_metadata());
    }
}
