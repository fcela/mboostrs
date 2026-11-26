// PyO3 binding layer: large tuple return types and many-argument #[pyfunction]
// signatures mirror the Python API surface; indexed loops walk parallel numpy
// columns. These pedantic lints are noise here.
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

use std::collections::HashMap;
use std::time::Instant;

use ndarray::ArrayView1;
use numpy::{
    AllowTypeChange, IntoPyArray, PyArray1, PyArray2, PyArrayLike1, PyArrayLike2, PyReadonlyArray1,
    PyReadonlyArray2,
};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::types::PyDict;

use mboostrs_core::{
    bspline_basis, bspline_basis_extrap, bspline_basis_with_deriv, bspline_knots, center_spectral,
    compute_centering_transform, cyclic_bspline_basis_from_knots, cyclic_bspline_knots,
    cyclic_diff_penalty, df_to_lambda, df_to_lambda_scalar, diff_penalty, kron, prepare_block_bols,
    prepare_block_bols_shared_gram, prepare_bmrf, prepare_kronecker_bols,
    try_coef_glmboost as core_coef_glmboost, try_compute_aic, try_cv, try_cvrisk_selected_with,
    try_cvrisk_selected_with_offset_vec, try_cvrisk_with, try_fit_continue, try_fit_with,
    try_predict_glmboost_path_with_offset_vec as core_predict_glmboost_path_vec,
    try_predict_glmboost_with_offset_vec as core_predict_glmboost_vec, AdaExp, Auc, BasisInfo,
    Binomial, BmonoSolver, BoostControl, Cindex, CoxPH, ExpectReg, Family, FittedTree, FoldSpec,
    FoldType, GammaReg, GaussReg, Gaussian, Gehan, HingeLoss, Huber, KroneckerRebuild, Laplace,
    LearnerBlueprint, LearnerPreparer, Loglog, Lognormal, MonoConstraint, NBinomial, Poisson,
    PreparedLearner, PropOdds, QuantReg, RiskKind, StepRule, UnseenLevels, WarmStart, Weibull,
};
use mboostrs_core::{
    stabsel_parameters as rust_stabsel_parameters, try_d_rc, try_max_q, try_min_d, try_um_const,
};
use mboostrs_core::{FitReport, MboostrsError, PreparationTimer, PreparationTiming};

type ByVecs = Option<Vec<Option<Vec<f64>>>>;

fn attach_frontend_timings(
    report: &mut FitReport,
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

/// Convert a `MboostrsError` into a Python exception.
///
/// Constructs the matching `mboostrs._errors` class so the error carries the
/// `code`/`subject`/`details`/`rust_variant` attributes, falling back to the
/// plain builtin exception with the same message if the class cannot be built
/// (for example when the extension runs without the importable package).
mod gam_refit;
mod selection_bindings;
mod selection_stats;
mod transform_bindings;

fn mboostrs_error_to_pyerr(e: MboostrsError) -> pyo3::PyErr {
    use pyo3::exceptions::{PyRuntimeError, PyValueError};
    let (class_name, message, code, subject, details, rust_variant) = match &e {
        MboostrsError::Shape(msg) => (
            "MboostrsValueError",
            msg.clone(),
            "shape".to_string(),
            None,
            msg.clone(),
            "Shape".to_string(),
        ),
        MboostrsError::LearnerSetup(name, msg) => (
            "MboostrsValueError",
            format!("learner '{name}': {msg}"),
            "learner_setup".to_string(),
            Some(name.clone()),
            msg.clone(),
            "LearnerSetup".to_string(),
        ),
        MboostrsError::BadResponse(name, msg) => (
            "MboostrsValueError",
            format!("family '{name}': {msg}"),
            "bad_response".to_string(),
            Some(name.clone()),
            msg.clone(),
            "BadResponse".to_string(),
        ),
        MboostrsError::Solver(msg) => (
            "MboostrsRuntimeError",
            msg.clone(),
            "solver".to_string(),
            None,
            msg.clone(),
            "Solver".to_string(),
        ),
        MboostrsError::Control(msg) => (
            "MboostrsValueError",
            msg.clone(),
            "control".to_string(),
            None,
            msg.clone(),
            "Control".to_string(),
        ),
        MboostrsError::BadFamilyParameter(name, msg) => (
            "MboostrsValueError",
            format!("family '{name}': {msg}"),
            "bad_family_parameter".to_string(),
            Some(name.clone()),
            msg.clone(),
            "BadFamilyParameter".to_string(),
        ),
        MboostrsError::StabSel(msg) => (
            "MboostrsValueError",
            msg.clone(),
            "stabsel".to_string(),
            None,
            msg.clone(),
            "StabSel".to_string(),
        ),
        MboostrsError::UnseenLevel {
            learner,
            code,
            observation,
        } => (
            "MboostrsValueError",
            format!(
                "{e}. To predict a level the fit never saw as an average group (the mean \
                 of the term's level coefficients), pass unseen_levels=\"average\"; for a \
                 zero coefficient, the value the ridge penalty shrinks every level towards, \
                 pass unseen_levels=\"zero_effect\""
            ),
            "unseen_level".to_string(),
            Some(learner.to_string()),
            format!("unknown level {code} at observation {observation}"),
            "UnseenLevel".to_string(),
        ),
    };
    pyo3::Python::with_gil(|py| {
        let built: PyResult<PyErr> = (|| {
            let module = py.import("mboostrs._errors")?;
            let class: Bound<PyAny> = module.getattr(class_name)?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("code", &code)?;
            if let Some(subject) = &subject {
                kwargs.set_item("subject", subject)?;
            }
            kwargs.set_item("details", &details)?;
            kwargs.set_item("rust_variant", rust_variant)?;
            let instance = class.call((message.as_str(),), Some(&kwargs))?;
            Ok(PyErr::from_value(instance))
        })();
        match built {
            Ok(err) => err,
            Err(_) => {
                if class_name == "MboostrsRuntimeError" {
                    PyRuntimeError::new_err(message)
                } else {
                    PyValueError::new_err(message)
                }
            }
        }
    })
}

/// Parse the `unseen_levels=` replay policy from Python ("error" / "zero_effect").
fn parse_unseen_levels(value: Option<&str>) -> PyResult<UnseenLevels> {
    match value.unwrap_or("error") {
        "error" => Ok(UnseenLevels::Error),
        "zero_effect" => Ok(UnseenLevels::ZeroEffect),
        "average" => Ok(UnseenLevels::Average),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unseen_levels must be \"error\", \"zero_effect\" or \"average\", got {other:?}"
        ))),
    }
}

/// Identity embedded in the loaded native extension, not the source checkout.
#[pyfunction(name = "build_info")]
fn native_build_info(py: Python<'_>) -> PyResult<PyObject> {
    let info = mboostrs_core::build_info();
    let dict = PyDict::new(py);
    dict.set_item("package_version", info.package_version)?;
    dict.set_item("source_digest", info.source_digest)?;
    dict.set_item("git_revision", info.git_revision)?;
    dict.set_item("git_dirty", info.git_dirty)?;
    dict.set_item("rustc", info.rustc)?;
    dict.set_item("target", info.target)?;
    dict.set_item("profile", info.profile)?;
    dict.set_item("target_features", info.target_features)?;
    dict.set_item("crate_features", info.crate_features)?;
    dict.set_item("abi_version", info.abi_version)?;
    dict.set_item("build_id", info.build_id)?;
    Ok(dict.into_pyobject(py)?.into_any().unbind())
}

/// Parse a bmono solver-type string from Python ("iterative" / "quad.prog")
/// to the `BmonoSolver` enum. Returns a PyValueError on unknown strings.
fn parse_bmono_solver(s: &str) -> PyResult<BmonoSolver> {
    match s {
        "iterative" => Ok(BmonoSolver::Iterative),
        "quad.prog" | "quadprog" => Ok(BmonoSolver::QuadProg),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown bmono solver type: {other:?} (expected 'iterative' or 'quad.prog')"
        ))),
    }
}

fn fold_matrix_from_py(
    folds: PyArrayLike2<'_, f64, AllowTypeChange>,
    n: usize,
) -> PyResult<ndarray::Array2<f64>> {
    let fold_view = folds.as_array();
    if fold_view.nrows() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "fold matrix has {} rows but n = {}",
            fold_view.nrows(),
            n
        )));
    }
    Ok(fold_view.to_owned())
}

/// Convert an explicit (n, B) judge mask, or None, into the engine's mask.
fn judge_mask_from_py(
    judge: Option<PyArrayLike2<'_, bool, AllowTypeChange>>,
    n: usize,
) -> PyResult<Option<ndarray::Array2<bool>>> {
    match judge {
        None => Ok(None),
        Some(value) => {
            let view = value.as_array();
            if view.nrows() != n {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "judge mask has {} rows but n = {}",
                    view.nrows(),
                    n
                )));
            }
            Ok(Some(view.to_owned()))
        }
    }
}

/// Convert an explicit (n, B) judge mask from a read-only array, or None.
fn judge_mask_from_readonly(
    judge: Option<PyReadonlyArray2<'_, bool>>,
    n: usize,
) -> PyResult<Option<ndarray::Array2<bool>>> {
    match judge {
        None => Ok(None),
        Some(value) => {
            let view = value.as_array();
            if view.nrows() != n {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "judge mask has {} rows but n = {}",
                    view.nrows(),
                    n
                )));
            }
            Ok(Some(view.to_owned()))
        }
    }
}

// (offset, selected, coefs_flat, risk, fitted, nuisance paths, learner store, execution report)
type FitResult = (
    f64,
    Vec<usize>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<Vec<f64>>,
    Py<LearnerStore>,
    PyObject,
);
type GamFitResult = (
    f64,
    Vec<usize>,
    Vec<Vec<f64>>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<Vec<f64>>,
    Py<LearnerStore>,
    PyObject,
    Vec<f64>,
);

fn fit_report_to_py(py: Python<'_>, report: &FitReport) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("schema_version", report.schema_version)?;
    dict.set_item("instrumentation", report.instrumentation.as_str())?;
    dict.set_item("route", report.route.as_str())?;
    dict.set_item("n_observations", report.n_observations)?;
    dict.set_item("n_learners", report.n_learners)?;
    dict.set_item("total_basis_columns", report.total_basis_columns)?;
    dict.set_item("requested_mstop", report.requested_mstop)?;
    dict.set_item("completed_mstop", report.completed_mstop)?;
    dict.set_item("effective_threads", report.effective_threads)?;
    dict.set_item("gram_eligible", report.gram_eligible)?;
    dict.set_item(
        "fallback_reasons",
        report
            .fallback_reasons
            .iter()
            .map(|reason| reason.as_str())
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("scalar_learner_count", report.scalar_learner_count)?;
    dict.set_item("block_learner_count", report.block_learner_count)?;
    dict.set_item("tree_learner_count", report.tree_learner_count)?;
    dict.set_item("cholesky_solver_count", report.cholesky_solver_count)?;
    dict.set_item("lu_solver_count", report.lu_solver_count)?;
    dict.set_item("regularized_solver_count", report.regularized_solver_count)?;
    dict.set_item("max_solver_jitter", report.max_solver_jitter)?;
    dict.set_item("factor_cache_bytes", report.factor_cache_bytes)?;
    dict.set_item("retained_learner_bytes", report.retained_learner_bytes)?;
    dict.set_item("gram_cache_bytes", report.gram_cache_bytes)?;
    dict.set_item("calibration_count", report.calibration_count)?;
    dict.set_item("tie_selection_count", report.tie_selection_count)?;
    dict.set_item(
        "component_subsample_iterations",
        report.component_subsample_iterations,
    )?;
    dict.set_item(
        "calibration_converged_count",
        report.calibration_converged_count,
    )?;
    dict.set_item("calibration_limit_count", report.calibration_limit_count)?;
    dict.set_item(
        "calibration_failure_count",
        report.calibration_failure_count,
    )?;
    dict.set_item(
        "nonfinite_candidate_fallbacks",
        report.nonfinite_candidate_fallbacks,
    )?;
    dict.set_item("setup_ns", report.setup_ns)?;
    dict.set_item("basis_ns", report.basis_ns)?;
    dict.set_item("factorization_ns", report.factorization_ns)?;
    dict.set_item("cross_gram_ns", report.cross_gram_ns)?;
    dict.set_item("iteration_ns", report.iteration_ns)?;
    dict.set_item("reconstruction_ns", report.reconstruction_ns)?;
    dict.set_item("binding_ns", report.binding_ns)?;
    dict.set_item("total_ns", report.total_ns)?;
    Ok(dict.into_pyobject(py)?.into_any().unbind())
}

/// Named checks for family parameters crossing the binding.
///
/// The Python constructors validate the same values, but a family also
/// arrives here from a deserialized artifact and from cross-validation and
/// stability-selection dispatch, which do not go back through a constructor.
/// A value rejected when typed must be rejected when loaded, so the check
/// lives on both sides rather than being trusted from one.
/// Lift a parameter-check failure into a Python `ValueError`.
fn check_param<T>(checked: Result<T, String>) -> PyResult<T> {
    checked.map_err(pyo3::exceptions::PyValueError::new_err)
}

fn param_unit_interval(family: &str, name: &str, value: f64) -> Result<f64, String> {
    if !value.is_finite() || value <= 0.0 || value >= 1.0 {
        return Err(format!("{family}: {name} must be in (0, 1), got {value}"));
    }
    Ok(value)
}

fn param_positive(family: &str, name: &str, value: f64) -> Result<f64, String> {
    if !value.is_finite() || value <= 0.0 {
        return Err(format!("{family}: {name} must be positive, got {value}"));
    }
    Ok(value)
}

/// A category count: a whole number at least `minimum`.
///
/// This used to be `value as usize` followed by `.max(2)` / `.max(3)`, which
/// turned an invalid count into a valid-looking one and fitted a different
/// model than the caller asked for.
fn param_category_count(
    family: &str,
    name: &str,
    value: f64,
    minimum: usize,
) -> Result<usize, String> {
    if !value.is_finite() || value.fract() != 0.0 || value < 0.0 {
        return Err(format!(
            "{family}: {name} must be a whole number >= {minimum}, got {value}"
        ));
    }
    let count = value as usize;
    if count < minimum {
        return Err(format!(
            "{family}: {name} must be at least {minimum}, got {count}"
        ));
    }
    Ok(count)
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

fn make_family(name: &str, params: &HashMap<String, f64>) -> Result<Box<dyn Family>, String> {
    match name {
        "gaussian" => Ok(Box::new(Gaussian)),
        "gaussreg" => Ok(Box::new(GaussReg)),
        "binomial" => Ok(Box::new(Binomial)),
        "poisson" => Ok(Box::new(Poisson)),
        "laplace" => Ok(Box::new(Laplace::with_response_scale(family_response_scale(params)))),
        "quantreg" => {
            let tau = param_unit_interval("quantreg", "tau",
                params.get("tau").copied().unwrap_or(0.5))?;
            // Honor qoffset (the starting offset quantile) so CV folds match a
            // direct fit, which reads it via with_concrete_family!.
            let qoffset = param_unit_interval("quantreg", "qoffset",
                params.get("qoffset").copied().unwrap_or(0.5))?;
            Ok(Box::new(
                QuantReg::with_qoffset(tau, qoffset)
                    .with_response_scale(family_response_scale(params)),
            ))
        }
        "expectreg" => {
            let tau = param_unit_interval("expectreg", "tau",
                params.get("tau").copied().unwrap_or(0.5))?;
            Ok(Box::new(ExpectReg::new(tau)))
        }
        "huber" => {
            let delta = params
                .get("delta")
                .copied()
                .map(|d| param_positive("huber", "delta", d))
                .transpose()?;
            Ok(Box::new(Huber::new(delta)))
        }
        "gammasreg" => Ok(Box::new(GammaReg::new())),
        "nbinomial" => Ok(Box::new(NBinomial::new())),
        "hurdle" => Ok(Box::new(mboostrs_core::Hurdle::new())),
        "propodds" => {
            // A supplied count names ordered categories, so it needs at least
            // three: two is a binary response, which Binomial fits. The key is
            // absent when the caller did not supply one, so there is no
            // sentinel value to exempt from the check.
            let n_levels = match params.get("n_levels").copied() {
                Some(raw) => param_category_count("propodds", "n_levels", raw, 3)?,
                None => 2,
            };
            Ok(Box::new(PropOdds::new(n_levels)))
        }
        "adaexp" => Ok(Box::new(AdaExp)),
        "hingeloss" => {
            let alpha_cost = param_unit_interval("hingeloss", "alpha_cost",
                params.get("alpha_cost").copied().unwrap_or(0.5))?;
            Ok(Box::new(HingeLoss::new(alpha_cost)))
        }
        "auc" => Ok(Box::new(Auc)),
        "gehan" => Ok(Box::new(Gehan)),
        "cindex" => {
            let sigma = param_positive("cindex", "sigma",
                params.get("sigma").copied().unwrap_or(0.1))?;
            Ok(Box::new(Cindex::new(sigma)))
        }
        "coxph" => Ok(Box::new(CoxPH::new())),
        "weibull" => Ok(Box::new(Weibull::new())),
        "loglog" => Ok(Box::new(Loglog::new())),
        "lognormal" => Ok(Box::new(Lognormal::new())),
        "multinomial" => {
            let n_classes = match params.get("n_classes").copied() {
                Some(raw) => param_category_count("multinomial", "n_classes", raw, 3)?,
                None => 3,
            };
            Ok(Box::new(mboostrs_core::Multinomial::new(n_classes)))
        }
        other => Err(format!(
            "unknown family: {other}. Families are compiled implementations selected by name, so a class the package does not define cannot be used: there is no run-time equivalent of R's mboost::Family(), which builds a family from closures. See the R migration chapter for the list of what is available and the reasoning behind the trade."
        )),
    }
}

/// Binary class indices from the core family's decision rule.
#[pyfunction]
fn family_class(
    family_name: &str,
    family_params: HashMap<String, f64>,
    f: PyReadonlyArray1<f64>,
) -> PyResult<Vec<usize>> {
    let family = make_family(family_name, &family_params)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    family.class(f.as_array()).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "type='class' requires a binary family (Binomial, AdaExp, HingeLoss or AUC), got {}",
            family.name()
        ))
    })
}

/// The centered linear design retained by glmboost, with its intercept.
#[pyfunction]
fn glmboost_design<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    centering_means: Vec<f64>,
    intercept: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let design =
        mboostrs_core::path::try_glmboost_design(x.as_array(), &centering_means, intercept)
            .map_err(mboostrs_error_to_pyerr)?;
    Ok(design.into_pyarray(py))
}

/// Maximum-likelihood linear fit in the same family as a boost
/// (`mboostrs_core::same_family::fit_same_family`). `designs`, `offsets` and
/// `starts` hold one entry per linear predictor (two for `BetaLSS`). Returns
/// `loglik`, `start_loglik`, `converged`, `iterations`, `coefficients` and
/// `nuisance`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn same_family_glm_fit(
    py: Python<'_>,
    family: &str,
    y: PyReadonlyArray1<f64>,
    weights: PyReadonlyArray1<f64>,
    designs: Vec<PyReadonlyArray2<f64>>,
    offsets: Vec<PyReadonlyArray1<f64>>,
    starts: Vec<PyReadonlyArray1<f64>>,
    nuisance: Vec<f64>,
) -> PyResult<PyObject> {
    use mboostrs_core::same_family::{fit_same_family, LinearPredictor, SameFamily};
    let kind = SameFamily::from_name(family).map_err(mboostrs_error_to_pyerr)?;
    if designs.len() != offsets.len() || designs.len() != starts.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "designs, offsets and starts have lengths {}, {} and {}; they must match",
            designs.len(),
            offsets.len(),
            starts.len()
        )));
    }
    let predictors: Vec<LinearPredictor<'_>> = designs
        .iter()
        .zip(&offsets)
        .zip(&starts)
        .map(|((design, offset), start)| LinearPredictor {
            design: design.as_array(),
            offset: offset.as_array(),
            start: start.as_array(),
        })
        .collect();
    let fit = fit_same_family(
        kind,
        y.as_array(),
        weights.as_array(),
        &predictors,
        &nuisance,
    )
    .map_err(mboostrs_error_to_pyerr)?;
    let dict = PyDict::new(py);
    dict.set_item("loglik", fit.loglik)?;
    dict.set_item("start_loglik", fit.start_loglik)?;
    dict.set_item("converged", fit.converged)?;
    dict.set_item("iterations", fit.iterations)?;
    let coefficients: Vec<_> = fit
        .coefficients
        .into_iter()
        .map(|c| c.into_pyarray(py))
        .collect();
    dict.set_item("coefficients", coefficients)?;
    dict.set_item("nuisance", fit.nuisance)?;
    Ok(dict.into_pyobject(py)?.into_any().unbind())
}

/// Evaluate a standard family's offset, negative gradient, and risk at given
/// points. Exposed for the docs/book correctness audit
/// (methodology/functional_gradient_boosting.ipynb), which finite-differences
/// the risk to check it against the analytic gradient returned here, for
/// every family in `family_capabilities()`. Not on the fit/predict hot path
/// (which stays monomorphised via `with_concrete_family!`); this is a
/// `dyn Family` call, fine for one-off diagnostic use. Survival families
/// expect `y` as `[time..., status...]` (length `2*f.len()`), matching
/// `fit_glmboost`. `w` has one entry per observation for every family; a
/// Multinomial `y` and `f` hold K-1 entries per observation, row-major.
///
/// `nuisance` / `nuisance_vec` restore a fitted model's nuisance parameters
/// (NBinomial dispersion, GammaReg shape, PropOdds thresholds, survival
/// scale) after the offset is computed and before the gradient and risk, so
/// both describe that fit rather than the constructor default.
#[pyfunction]
#[pyo3(signature = (name, params, y, f, w, nuisance=None, nuisance_vec=None))]
fn family_diagnostics<'py>(
    py: Python<'py>,
    name: &str,
    params: HashMap<String, f64>,
    y: PyReadonlyArray1<'_, f64>,
    f: PyReadonlyArray1<'_, f64>,
    w: PyReadonlyArray1<'_, f64>,
    nuisance: Option<f64>,
    nuisance_vec: Option<Vec<f64>>,
) -> PyResult<(f64, Py<PyArray1<f64>>, f64)> {
    let mut fam = make_family(name, &params).map_err(pyo3::exceptions::PyValueError::new_err)?;
    // Diagnostics are one-off calls, so own the inputs to accept ordinary
    // strided NumPy views without reaching a family's contiguous-slice
    // assertions.
    let y_owned = y.as_array().to_owned();
    let f_owned = f.as_array().to_owned();
    let w_owned = w.as_array().to_owned();
    let y_raw = y_owned.view();
    let f = f_owned.view();
    let w = w_owned.view();
    if y_raw.is_empty() || f.is_empty() || w.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "family_diagnostics requires at least one observation",
        ));
    }
    if y_raw.iter().any(|value| !value.is_finite())
        || f.iter().any(|value| !value.is_finite())
        || w.iter().any(|value| !value.is_finite())
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "family_diagnostics inputs must contain only finite values",
        ));
    }
    if w.iter().any(|&value| value < 0.0) || w.iter().all(|&value| value == 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "family_diagnostics weights must be non-negative with a positive sum",
        ));
    }
    fam.check_y(y_raw).map_err(mboostrs_error_to_pyerr)?;
    // Apply the family's own response encoding, as `fit` and `cvrisk` do.
    // Without this, a Binomial response given as {0, 1} reaches a loss written
    // for {-1, +1}: the offset, gradient and risk are then all self-consistent
    // but describe a different function from the one the fit minimises, so a
    // finite-difference audit of them passes while checking the wrong thing.
    let encoded = fam.encode_response(y_raw);
    let y = match encoded.as_ref() {
        Some(recoded) => recoded.view(),
        None => y_raw,
    };
    // `f` has one entry per predictor coordinate (`n_from_y`); `w` has one
    // entry per observation, as every fitting entry point takes it.
    // Multinomial holds K-1 coordinates per observation, so its weights are
    // repeated over each observation's block before reaching the family,
    // exactly as `expand_multinomial_weights` does for a fit.
    let n = fam.n_from_y(y);
    let span = fam.flat_response_span();
    if n != f.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "family_diagnostics response implies {n} fitted values, got {}",
            f.len()
        )));
    }
    if n % span != 0 || w.len() != n / span {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "family_diagnostics expects one weight per observation ({} observations), got {} weights",
            n / span,
            w.len()
        )));
    }
    let w_flat = (span > 1).then(|| {
        let mut expanded = ndarray::Array1::zeros(n);
        for (i, &value) in w.iter().enumerate() {
            expanded
                .slice_mut(ndarray::s![i * span..(i + 1) * span])
                .fill(value);
        }
        expanded
    });
    let w = w_flat.as_ref().map_or(w, |expanded| expanded.view());
    let offset = fam.offset(y, w);
    if let Some(value) = nuisance {
        fam.set_nuisance(value);
    }
    if let Some(values) = nuisance_vec {
        if !values.is_empty() {
            fam.set_nuisance_vec(values);
        }
    }
    let mut grad = ndarray::Array1::zeros(f.len());
    fam.negative_gradient(y, f, w, grad.view_mut());
    let risk = fam.risk(y, f, w);
    Ok((offset, grad.into_pyarray(py).unbind(), risk))
}

/// Dispatch on `family_name` to a stack-allocated, concrete `Family` so the
/// inner boosting loop monomorphises and inlines `ngradient_and_risk`,
/// `calibrate`, etc. Returns a `PyResult` so the unknown-family case can
/// produce a Python error.
macro_rules! with_concrete_family {
    ($name:expr, $params:expr, |$fam:ident| $body:expr) => {{
        #[allow(unused_mut)]
        match $name {
            "gaussian" => {
                let mut $fam = mboostrs_core::Gaussian;
                Ok($body)
            }
            "gaussreg" => {
                let mut $fam = mboostrs_core::GaussReg;
                Ok($body)
            }
            "binomial" => {
                let mut $fam = mboostrs_core::Binomial;
                Ok($body)
            }
            "poisson" => {
                let mut $fam = mboostrs_core::Poisson;
                Ok($body)
            }
            "laplace" => {
                let mut $fam =
                    mboostrs_core::Laplace::with_response_scale(family_response_scale(&$params));
                Ok($body)
            }
            "quantreg" => {
                let tau = check_param(param_unit_interval("quantreg", "tau",
                    $params.get("tau").copied().unwrap_or(0.5)))?;
                let qoffset = check_param(param_unit_interval("quantreg", "qoffset",
                    $params.get("qoffset").copied().unwrap_or(0.5)))?;
                let mut $fam = mboostrs_core::QuantReg::with_qoffset(tau, qoffset)
                    .with_response_scale(family_response_scale(&$params));
                Ok($body)
            }
            "expectreg" => {
                let tau = check_param(param_unit_interval("expectreg", "tau",
                    $params.get("tau").copied().unwrap_or(0.5)))?;
                let mut $fam = mboostrs_core::ExpectReg::new(tau);
                Ok($body)
            }
            "huber" => {
                let delta = match $params.get("delta").copied() {
                    Some(d) => Some(check_param(param_positive("huber", "delta", d))?),
                    None => None,
                };
                let mut $fam = mboostrs_core::Huber::new(delta);
                Ok($body)
            }
            "gammasreg" => {
                let mut $fam = mboostrs_core::GammaReg::new();
                Ok($body)
            }
            "nbinomial" => {
                let mut $fam = mboostrs_core::NBinomial::new();
                Ok($body)
            }
            "hurdle" => {
                let mut $fam = mboostrs_core::Hurdle::new();
                Ok($body)
            }
            "propodds" => {
                let n_levels = match $params.get("n_levels").copied() {
                    Some(raw) => check_param(
                        param_category_count("propodds", "n_levels", raw, 3))?,
                    None => 2,
                };
                let mut $fam = mboostrs_core::PropOdds::new(n_levels);
                Ok($body)
            }
            "adaexp" => {
                let mut $fam = mboostrs_core::AdaExp;
                Ok($body)
            }
            "hingeloss" => {
                let alpha_cost = check_param(param_unit_interval("hingeloss", "alpha_cost",
                    $params.get("alpha_cost").copied().unwrap_or(0.5)))?;
                let mut $fam = mboostrs_core::HingeLoss::new(alpha_cost);
                Ok($body)
            }
            "auc" => {
                let mut $fam = mboostrs_core::Auc;
                Ok($body)
            }
            "gehan" => {
                let mut $fam = mboostrs_core::Gehan;
                Ok($body)
            }
            "cindex" => {
                let sigma = check_param(param_positive("cindex", "sigma",
                    $params.get("sigma").copied().unwrap_or(0.1)))?;
                let mut $fam = mboostrs_core::Cindex::new(sigma);
                Ok($body)
            }
            "coxph" => {
                let mut $fam = mboostrs_core::CoxPH::new();
                Ok($body)
            }
            "weibull" => {
                let mut $fam = mboostrs_core::Weibull::new();
                Ok($body)
            }
            "loglog" => {
                let mut $fam = mboostrs_core::Loglog::new();
                Ok($body)
            }
            "lognormal" => {
                let mut $fam = mboostrs_core::Lognormal::new();
                Ok($body)
            }
            "multinomial" => {
                let n_classes = match $params.get("n_classes").copied() {
                    Some(raw) => check_param(
                        param_category_count("multinomial", "n_classes", raw, 3))?,
                    None => 3,
                };
                let mut $fam = mboostrs_core::Multinomial::new(n_classes);
                Ok($body)
            }
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown family: {other}. Families are compiled implementations selected by name, so a class the package does not define cannot be used: there is no run-time equivalent of R's mboost::Family(), which builds a family from closures. See the R migration chapter for the list of what is available and the reasoning behind the trade."
            ))),
        }
    }};
}

#[pyfunction(signature = (mstop, nu, risk="inbag", trace=false))]
fn boost_control(
    mstop: u32,
    nu: f64,
    risk: &str,
    trace: bool,
) -> PyResult<(u32, f64, String, bool)> {
    let mut ctrl = BoostControl::new(mstop, nu).map_err(mboostrs_error_to_pyerr)?;
    let risk_kind = match risk {
        "inbag" => RiskKind::Inbag,
        "oobag" => RiskKind::Oobag,
        "none" => RiskKind::None,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "risk must be 'inbag', 'oobag', or 'none', got {risk:?}"
            )))
        }
    };
    ctrl = ctrl.with_risk(risk_kind).with_trace(trace);
    let risk_str = match ctrl.risk {
        RiskKind::Inbag => "inbag",
        RiskKind::Oobag => "oobag",
        RiskKind::None => "none",
    };
    Ok((ctrl.mstop, ctrl.nu, risk_str.to_string(), ctrl.trace))
}

/// Apply stochastic component subsampling to a control if a fraction was given.
fn apply_subsampling(
    ctrl: BoostControl,
    component_fraction: Option<f64>,
    subsample_seed: u64,
) -> PyResult<BoostControl> {
    match component_fraction {
        None => Ok(ctrl),
        Some(fraction) => ctrl
            .with_component_subsampling(fraction, subsample_seed)
            .map_err(mboostrs_error_to_pyerr),
    }
}

/// Apply the step rule a caller asked for to a control.
///
/// `step="fixed"` is the classic constant step; `"line_search"` finds the
/// loss-optimal multiple of the winning learner's direction each boosting
/// step and shrinks that by `nu` instead. `step_iterations` bounds the
/// golden-section refinement after the grid bracket.
fn apply_step(ctrl: BoostControl, step: &str, step_iterations: u32) -> PyResult<BoostControl> {
    let rule = match step {
        "fixed" => StepRule::Fixed,
        "line_search" => {
            if step_iterations < 1 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "step_iterations must be at least 1, got {step_iterations}"
                )));
            }
            StepRule::LineSearch {
                max_iter: step_iterations as usize,
            }
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "step must be 'fixed' or 'line_search', got {step:?}"
            )))
        }
    };
    Ok(ctrl.with_step(rule))
}

/// Reject a diverged fit with a catchable error instead of returning inf/NaN.
///
/// For log-link families (Poisson, etc.) too large a step-size `nu` lets the
/// linear predictor grow until `exp(f)` overflows, turning risk/fitted
/// non-finite. R's mboost stops with "Infinite residuals: please decrease
/// step-size nu"; we mirror that with a `ValueError` (catchable by callers such
/// as confint's bootstrap loop, unlike a Rust panic).
fn reject_diverged(risk: &[f64], fitted: &ndarray::Array1<f64>) -> PyResult<()> {
    if risk.iter().any(|v| !v.is_finite()) || fitted.iter().any(|v| !v.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "boosting fit diverged (non-finite risk/fitted values); \
             decrease the step-size nu in boost_control",
        ));
    }
    Ok(())
}

/// An offset as Python sends it: absent, one number for every observation,
/// or one entry per observation (R's `offset=` semantics). A vector replaces
/// the scalar family-calibrated start of the fit and is never re-estimated.
enum OffsetArg {
    None,
    Scalar(f64),
    Vector(Vec<f64>),
}

fn extract_offset_arg(offset: Option<&Bound<'_, PyAny>>) -> PyResult<OffsetArg> {
    match offset {
        None => Ok(OffsetArg::None),
        Some(value) => {
            if let Ok(scalar) = value.extract::<f64>() {
                return Ok(OffsetArg::Scalar(scalar));
            }
            Ok(OffsetArg::Vector(value.extract::<Vec<f64>>()?))
        }
    }
}

/// The starting linear predictor of a prediction: the offset as sent, one
/// number or one entry per row. A vector offset must describe every row.
fn eta_start(offset: &OffsetArg, n: usize) -> PyResult<Vec<f64>> {
    match offset {
        OffsetArg::None => Ok(vec![0.0; n]),
        OffsetArg::Scalar(value) => Ok(vec![*value; n]),
        OffsetArg::Vector(values) => {
            if values.len() != n {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "offset has {} entries but {n} rows are predicted",
                    values.len()
                )));
            }
            Ok(values.clone())
        }
    }
}

/// Copy one Python float vector into a Rust `Vec<f64>`.
///
/// The fast path is a 1-D NumPy `float64` array: it is borrowed and copied
/// without boxing each element. Any other value falls back to `Vec<f64>`
/// extraction, so callers that still hand in Python lists keep working.
/// The calling Python boundary guarantees `np.ascontiguousarray(...,
/// dtype=np.float64)` for the array path; a strided or non-float array still
/// converts correctly through `as_array` or the sequence fallback.
fn extract_f64_vec(value: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(array) = value.extract::<PyReadonlyArray1<'_, f64>>() {
        return Ok(match array.as_slice() {
            Ok(slice) => slice.to_vec(),
            Err(_) => array.as_array().to_vec(),
        });
    }
    value.extract::<Vec<f64>>()
}

/// `extract_f64_vec` over a Python sequence, one entry per learner or step.
fn extract_f64_vecs(values: Vec<Bound<'_, PyAny>>) -> PyResult<Vec<Vec<f64>>> {
    values.iter().map(extract_f64_vec).collect()
}

/// `extract_f64_vec` over an optional per-learner list, preserving `None`.
fn extract_opt_f64_vecs(
    values: Option<Vec<Option<Bound<'_, PyAny>>>>,
) -> PyResult<Vec<Option<Vec<f64>>>> {
    match values {
        None => Ok(Vec::new()),
        Some(values) => values
            .iter()
            .map(|value| match value {
                Some(inner) => Ok(Some(extract_f64_vec(inner)?)),
                None => Ok(None),
            })
            .collect(),
    }
}

impl OffsetArg {
    fn scalar(&self) -> Option<f64> {
        match self {
            Self::None => None,
            Self::Scalar(value) => Some(*value),
            Self::Vector(_) => None,
        }
    }

    fn vector(&self) -> Option<&[f64]> {
        match self {
            Self::Vector(values) => Some(values),
            _ => None,
        }
    }
}

#[pyfunction(signature = (x_matrix, y, weights, mstop, nu, family_name, family_params, learner_names, offset=None, risk="inbag", trace=false, component_fraction=None, subsample_seed=0, center=true, step="fixed", step_iterations=4, intercept=true))]
#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
fn fit_glmboost(
    py: Python<'_>,
    x_matrix: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    mstop: u32,
    nu: f64,
    family_name: &str,
    family_params: HashMap<String, f64>,
    learner_names: Vec<String>,
    offset: Option<Bound<'_, PyAny>>,
    risk: &str,
    trace: bool,
    component_fraction: Option<f64>,
    subsample_seed: u64,
    center: bool,
    step: &str,
    step_iterations: u32,
    intercept: bool,
) -> PyResult<FitResult> {
    let binding_started = trace.then(Instant::now);
    let offset = extract_offset_arg(offset.as_ref())?;
    let x = x_matrix.as_array();
    let w = weights.as_array();

    let y = y.as_array();

    // Validate the response up front (previously every family's check_y was dead
    // code), so invalid y gives a clear error instead of silently producing NaN
    // gradients/risk deep in the fit.
    make_family(family_name, &family_params)
        .map_err(pyo3::exceptions::PyValueError::new_err)?
        .check_y(y)
        .map_err(mboostrs_error_to_pyerr)?;

    let is_survival = matches!(
        family_name,
        "coxph" | "weibull" | "loglog" | "lognormal" | "gehan" | "cindex"
    );
    let n = if is_survival { y.len() / 2 } else { y.len() };

    if x.nrows() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "x_matrix has {} rows but y has {} elements",
            x.nrows(),
            n
        )));
    }

    let risk_kind = match risk {
        "inbag" => RiskKind::Inbag,
        "oobag" => RiskKind::Oobag,
        "none" => RiskKind::None,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "risk must be 'inbag', 'oobag', or 'none', got {risk:?}"
            )))
        }
    };
    let ctrl = BoostControl::new(mstop, nu)
        .map_err(mboostrs_error_to_pyerr)?
        .with_risk(risk_kind)
        .with_trace(trace);
    let ctrl = apply_step(ctrl, step, step_iterations)?;
    let ctrl = apply_subsampling(ctrl, component_fraction, subsample_seed)?;

    let preparation_timer = trace.then(PreparationTimer::start);
    let ones = ndarray::Array1::ones(n);
    // Unweighted column means, matching R's glmboost(center=TRUE) which uses
    // colMeans(x) (unweighted). R always uses unweighted centering regardless
    // of case weights — weights only enter the OLS coefficient computation.
    let n_f64 = n as f64;
    // `center = false` reproduces R's glmboost(center = FALSE): the raw
    // columns are used, which converges more slowly because each predictor
    // stays correlated with the intercept. The same means must reach the
    // Python model for prediction replay, or a replayed prediction disagrees
    // with the fit it came from.
    let column_means: Vec<f64> = (0..x.ncols())
        .map(|j| {
            if !center || n == 0 {
                return 0.0;
            }
            x.column(j).iter().sum::<f64>() / n_f64
        })
        .collect();
    let w_owned = w.to_owned();
    let mut learners: Vec<PreparedLearner> = Vec::with_capacity(x.ncols() + 1);
    // `y ~ x - 1`: R's model.matrix drops the intercept column, so mboost
    // offers no constant learner; the offset is the only level.
    if intercept {
        learners.push(mboostrs_core::learners::prepare_scalar_bols(
            ones.view(),
            w_owned.view(),
            0.0,
            false,
            "(Intercept)",
            None,
        ));
    }
    for j in 0..x.ncols() {
        learners.push(mboostrs_core::learners::prepare_scalar_bols_fixed_center(
            x.column(j),
            w_owned.view(),
            0.0,
            column_means[j],
            learner_names.get(j).unwrap_or(&"?".to_string()),
        ));
    }
    let preparation_timing = preparation_timer.map(PreparationTimer::finish);

    let family_name_owned = family_name.to_string();
    let binding_input_ns = binding_started.map(|started| started.elapsed().as_nanos() as u64);
    let mut path = py
        .allow_threads(|| {
            with_concrete_family!(family_name_owned.as_str(), &family_params, |fam| {
                match offset.vector() {
                    Some(vector) => {
                        let offset_owned = ndarray::Array1::from(vector.to_vec());
                        mboostrs_core::engine::try_fit_with_vector_offset(
                            &mut learners,
                            y,
                            w_owned.view(),
                            &mut fam,
                            &ctrl,
                            offset_owned.view(),
                            false,
                        )
                    }
                    None => try_fit_with(
                        &mut learners,
                        y,
                        w_owned.view(),
                        &mut fam,
                        &ctrl,
                        offset.scalar(),
                        false,
                    ),
                }
            })
        })?
        .map_err(mboostrs_error_to_pyerr)?;

    let binding_output_started = trace.then(Instant::now);
    reject_diverged(&path.risk, &path.fitted)?;
    let nuisance_vec_path = path.nuisance_rows();
    let offset_out = path.offset;
    let selected = path.selected;
    let coefs_flat: Vec<f64> = path.coefs.iter().map(|c| c[0]).collect();
    let risk = path.risk;
    let nuisance_vec = path.nuisance_vec;
    let fitted = path.fitted.into_raw_vec_and_offset().0;

    let store = Py::new(
        py,
        LearnerStore {
            learners,
            nu,
            basis_info: Vec::new(),
        },
    )?;
    attach_frontend_timings(
        &mut path.report,
        preparation_timing,
        binding_input_ns,
        binding_output_started,
    );
    let execution_report = fit_report_to_py(py, &path.report)?;
    Ok((
        offset_out,
        selected,
        coefs_flat,
        risk,
        fitted,
        nuisance_vec,
        nuisance_vec_path,
        store,
        execution_report,
    ))
}

/// Per-parameter result for an LSS fit: (offset, selected, coefs_flat, fitted).
type LssParamResult = (f64, Vec<usize>, Vec<f64>, Vec<f64>);
/// (per-parameter results, joint risk, global parameter order, report).
/// The last element is each parameter's risk after its own updates (entry 0
/// is the offset-only model), as gamboostLSS's `risk(merge = FALSE)`.
type LssFitResult = (
    Vec<LssParamResult>,
    Vec<f64>,
    Vec<usize>,
    PyObject,
    Vec<Vec<f64>>,
);

/// Construct an LSS family by name and validate the response via `check_y`.
/// Shared by all LSS fit, CV, and stabsel entry points so that validation
/// cannot be bypassed.
fn make_lss_family_checked(
    family_name: &str,
    n_params: usize,
    y: ndarray::ArrayView1<'_, f64>,
) -> PyResult<Box<dyn mboostrs_core::lss::LssFamily>> {
    let family = mboostrs_core::lss::make_family(family_name, n_params).map_err(|e| match e {
        mboostrs_core::MboostrsError::Control(msg) | mboostrs_core::MboostrsError::Shape(msg) => {
            pyo3::exceptions::PyValueError::new_err(msg)
        }
        other => mboostrs_error_to_pyerr(other),
    })?;
    family.check_y(y).map_err(mboostrs_error_to_pyerr)?;
    Ok(family)
}

fn required_spec_item<'py>(spec: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    spec.get_item(key)?.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "LSS learner specification is missing required field {key:?}"
        ))
    })
}

fn optional_spec_value<'py, T>(spec: &Bound<'py, PyDict>, key: &str) -> PyResult<Option<T>>
where
    T: FromPyObject<'py>,
{
    match spec.get_item(key)? {
        Some(value) if !value.is_none() => value.extract().map(Some),
        _ => Ok(None),
    }
}

/// Convert the compact language-level learner dictionaries into core-owned,
/// weight-independent blueprints. Fitting, CV, and stability selection all use
/// this function and [`LearnerBlueprint::prepare`], preventing their learner
/// construction rules from drifting apart.
/// Parse one flat list of learner dictionaries into core blueprints.
///
/// This is the whole learner boundary for the Python binding. A learner
/// crosses it as one self-describing, type-tagged record — the shape the R
/// binding has always used — rather than as a slot in each of ~80 parallel
/// arrays that three call sites had to keep index-aligned by hand. An error
/// names the learner it came from, so a malformed field is attributable.
fn parse_learner_blueprint_list(
    py: Python<'_>,
    specs: Vec<PyObject>,
) -> PyResult<Vec<LearnerBlueprint>> {
    let mut parameter_blueprints = Vec::with_capacity(specs.len());
    {
        {
            for raw_spec in specs {
                let spec = raw_spec.bind(py).downcast::<PyDict>().map_err(|_| {
                    pyo3::exceptions::PyTypeError::new_err(
                        "each learner specification must be a dict",
                    )
                })?;
                let kind: String = required_spec_item(spec, "kind")?.extract()?;
                let name: String = required_spec_item(spec, "name")?.extract()?;
                let x_obj = required_spec_item(spec, "x")?;
                let x: PyReadonlyArray1<'_, f64> = x_obj.extract().map_err(|_| {
                    pyo3::exceptions::PyTypeError::new_err(format!(
                        "LSS learner {name:?} field 'x' must be a one-dimensional float64 array"
                    ))
                })?;
                let x = x.as_array().to_owned();
                let by = match spec.get_item("by")? {
                    Some(value) if !value.is_none() => {
                        let values: PyReadonlyArray1<'_, f64> = value.extract().map_err(|_| {
                        pyo3::exceptions::PyTypeError::new_err(format!(
                            "LSS learner {name:?} field 'by' must be a one-dimensional float64 array"
                        ))
                    })?;
                        Some(values.as_array().to_owned())
                    }
                    _ => None,
                };
                let lambda = optional_spec_value::<f64>(spec, "lambda")?;
                let blueprint = match kind.as_str() {
                "bols" => {
                    let df = optional_spec_value::<f64>(spec, "df")?;
                    let center = optional_spec_value::<bool>(spec, "center")?.unwrap_or(true);
                    // `intercept` distinguishes the two forms of
                    // `bols(x, intercept = TRUE)`: LSS fits an explicit
                    // `[1, x]` block, `gamboost` fits a centered single column
                    // and leaves the constant to the offset. Defaulting it to
                    // `center` keeps every caller that predates the field on
                    // the block form they were getting.
                    let intercept = optional_spec_value::<bool>(spec, "intercept")?
                        .or(optional_spec_value::<bool>(spec, "center")?)
                        .unwrap_or(true);
                    if let Some(bins) = optional_spec_value::<usize>(spec, "bins")? {
                        LearnerBlueprint::scalar_bols_binned(
                            x, name, lambda, df, center, intercept, by, bins,
                        )
                    } else {
                        LearnerBlueprint::scalar_bols(
                            x, name, lambda, df, center, intercept, by,
                        )
                    }
                },
                "bbs" => {
                    let knots = optional_spec_value::<usize>(spec, "knots")?.unwrap_or(20);
                    let degree = optional_spec_value::<usize>(spec, "degree")?.unwrap_or(3);
                    let differences =
                        optional_spec_value::<usize>(spec, "differences")?.unwrap_or(2);
                    let df = optional_spec_value::<f64>(spec, "df")?.unwrap_or(4.0);
                    let center = optional_spec_value::<bool>(spec, "center")?.unwrap_or(false);
                    let cyclic = optional_spec_value::<bool>(spec, "cyclic")?.unwrap_or(false);
                    let boundary =
                        optional_spec_value::<(f64, f64)>(spec, "boundary_knots")?;
                    let deriv = optional_spec_value::<usize>(spec, "deriv")?.unwrap_or(0);
                    if let Some(bins) = optional_spec_value::<usize>(spec, "bins")? {
                        LearnerBlueprint::bbs_binned(
                            x, name, knots, degree, differences, df, lambda, center, by,
                            cyclic, boundary, deriv, bins,
                        )
                    } else {
                        LearnerBlueprint::bbs(
                            x, name, knots, degree, differences, df, lambda, center, by,
                            cyclic, boundary, deriv,
                        )
                    }
                },
                "kronecker_bbs" => {
                    let knots = optional_spec_value::<usize>(spec, "knots")?.unwrap_or(20);
                    let degree = optional_spec_value::<usize>(spec, "degree")?.unwrap_or(3);
                    let differences =
                        optional_spec_value::<usize>(spec, "differences")?.unwrap_or(2);
                    let df = optional_spec_value::<f64>(spec, "df")?.unwrap_or(4.0);
                    let deriv = optional_spec_value::<usize>(spec, "deriv")?.unwrap_or(0);
                    let n_resp = optional_spec_value::<usize>(spec, "n_resp")?
                        .ok_or_else(|| {
                            pyo3::exceptions::PyValueError::new_err(
                                "kronecker_bbs requires n_resp",
                            )
                        })?;
                    let center = optional_spec_value::<bool>(spec, "center")?.unwrap_or(false);
                    LearnerBlueprint::kronecker_bbs(
                        x, name, knots, degree, differences, df, n_resp, deriv,
                    )
                    .map(|blueprint| blueprint.with_center(center))
                },
                "bns" => LearnerBlueprint::bns(
                    x,
                    name,
                    optional_spec_value::<usize>(spec, "knots")?.unwrap_or(20),
                    optional_spec_value::<usize>(spec, "degree")?.unwrap_or(3),
                    optional_spec_value::<f64>(spec, "df")?,
                    optional_spec_value::<bool>(spec, "center")?.unwrap_or(false),
                    by,
                    optional_spec_value::<(f64, f64)>(spec, "boundary_knots")?,
                ),
                "brandom" => LearnerBlueprint::brandom(
                    x,
                    name,
                    optional_spec_value::<f64>(spec, "df")?.unwrap_or(4.0),
                    lambda,
                    optional_spec_value::<bool>(spec, "center")?.unwrap_or(false),
                    by,
                    optional_spec_value::<usize>(spec, "bins")?,
                ),
                "bmrf" => {
                    let k_obj = required_spec_item(spec, "K")?;
                    let k: PyReadonlyArray2<'_, f64> = k_obj.extract().map_err(|_| {
                        pyo3::exceptions::PyTypeError::new_err(format!(
                            "LSS learner {name:?} field 'K' must be a two-dimensional float64 array"
                        ))
                    })?;
                    LearnerBlueprint::bmrf(
                        x,
                        name,
                        k.as_array().to_owned(),
                        optional_spec_value::<f64>(spec, "df")?.unwrap_or(4.0),
                        lambda,
                        optional_spec_value::<bool>(spec, "center")?.unwrap_or(false),
                        by,
                    )
                }
                "bspatial" => {
                    let x2_obj = required_spec_item(spec, "x2")?;
                    let x2: PyReadonlyArray1<'_, f64> = x2_obj.extract().map_err(|_| {
                        pyo3::exceptions::PyTypeError::new_err(format!(
                            "LSS learner {name:?} field 'x2' must be a one-dimensional float64 array"
                        ))
                    })?;
                    let x2 = x2.as_array().to_owned();
                    let knots = optional_spec_value::<usize>(spec, "knots")?.unwrap_or(20);
                    let degree = optional_spec_value::<usize>(spec, "degree")?.unwrap_or(3);
                    let differences =
                        optional_spec_value::<usize>(spec, "differences")?.unwrap_or(2);
                    let df = optional_spec_value::<f64>(spec, "df")?.unwrap_or(6.0);
                    let center = optional_spec_value::<bool>(spec, "center")?.unwrap_or(false);
                    let boundary1 =
                        optional_spec_value::<(f64, f64)>(spec, "boundary_knots1")?;
                    let boundary2 =
                        optional_spec_value::<(f64, f64)>(spec, "boundary_knots2")?;
                    if let Some(bins) = optional_spec_value::<usize>(spec, "bins")? {
                        LearnerBlueprint::bspatial_binned(
                            x, x2, name, knots, degree, differences, df, lambda, center, by,
                            boundary1, boundary2, bins,
                        )
                    } else {
                        LearnerBlueprint::bspatial(
                            x, x2, name, knots, degree, differences, df, lambda, center, by,
                            boundary1, boundary2,
                        )
                    }
                }
                "bmono" => {
                    let constraint_name = optional_spec_value::<String>(spec, "constraint")?
                        .unwrap_or_else(|| "increasing".to_string());
                    let constraint = match constraint_name.as_str() {
                        "increasing" => MonoConstraint::Increasing,
                        "decreasing" => MonoConstraint::Decreasing,
                        "convex" => MonoConstraint::Convex,
                        "concave" => MonoConstraint::Concave,
                        "positive" => MonoConstraint::Positive,
                        "negative" => MonoConstraint::Negative,
                        other => {
                            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                                "unknown bmono constraint {other:?}"
                            )))
                        }
                    };
                    let solver_name = optional_spec_value::<String>(spec, "type")?
                        .unwrap_or_else(|| "quad.prog".to_string());
                    LearnerBlueprint::bmono(
                        x,
                        name,
                        optional_spec_value::<usize>(spec, "knots")?.unwrap_or(20),
                        optional_spec_value::<usize>(spec, "degree")?.unwrap_or(3),
                        optional_spec_value::<usize>(spec, "differences")?.unwrap_or(2),
                        optional_spec_value::<f64>(spec, "df")?.unwrap_or(4.0),
                        constraint,
                        parse_bmono_solver(&solver_name)?,
                        optional_spec_value::<f64>(spec, "lambda2")?.unwrap_or(1e6),
                        optional_spec_value::<usize>(spec, "niter")?.unwrap_or(10),
                        by,
                        optional_spec_value::<bool>(spec, "center")?.unwrap_or(false),
                    )
                }
                "bconstrained" => {
                    let c_obj = required_spec_item(spec, "C")?;
                    let c: PyReadonlyArray2<'_, f64> = c_obj.extract().map_err(|_| {
                        pyo3::exceptions::PyTypeError::new_err(format!(
                            "LSS learner {name:?} field 'C' must be a two-dimensional float64 array"
                        ))
                    })?;
                    // The exact QP by default, as for bmono: the iterative
                    // violation penalty leaves a same-signed violation each
                    // step that boosting accumulates.
                    let solver = parse_bmono_solver(
                        &optional_spec_value::<String>(spec, "type")?
                            .unwrap_or_else(|| "quad.prog".to_string()),
                    )?;
                    LearnerBlueprint::bconstrained(
                        x,
                        name,
                        optional_spec_value::<usize>(spec, "knots")?.unwrap_or(20),
                        optional_spec_value::<usize>(spec, "degree")?.unwrap_or(3),
                        optional_spec_value::<usize>(spec, "differences")?.unwrap_or(2),
                        optional_spec_value::<f64>(spec, "df")?.unwrap_or(4.0),
                        lambda,
                        c.as_array().to_owned(),
                        optional_spec_value::<f64>(spec, "lambda2")?.unwrap_or(1e6),
                        optional_spec_value::<usize>(spec, "niter")?.unwrap_or(10),
                        by,
                    )
                    .map(|blueprint| blueprint.with_bmono_solver(solver))
                }
                "btree" => {
                    let n_features = optional_spec_value::<usize>(spec, "n_features")?
                        .ok_or_else(|| {
                            pyo3::exceptions::PyValueError::new_err(format!(
                                "LSS learner {name:?} requires n_features"
                            ))
                        })?;
                    if n_features == 0 || !x.len().is_multiple_of(n_features) {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "LSS learner {name:?} packed tree design length {} is not divisible by n_features={n_features}",
                            x.len()
                        )));
                    }
                    let design = ndarray::Array2::from_shape_vec(
                        (x.len() / n_features, n_features),
                        x.to_vec(),
                    )
                    .map_err(|error| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "LSS learner {name:?} tree design shape error: {error}"
                        ))
                    })?;
                    let categorical =
                        optional_spec_value::<Vec<usize>>(spec, "categorical")?.unwrap_or_default();
                    let selection = parse_split_selection(
                        &optional_spec_value::<String>(spec, "selection")?
                            .unwrap_or_else(|| "ctree".to_string()),
                    )?;
                    let split_method = parse_split_method(
                        &optional_spec_value::<String>(spec, "split_method")?
                            .unwrap_or_else(|| "exact".to_string()),
                    )?;
                    LearnerBlueprint::btree(
                        design,
                        name,
                        optional_spec_value::<usize>(spec, "minbucket")?.unwrap_or(4),
                        optional_spec_value::<usize>(spec, "minsplit")?.unwrap_or(10),
                        optional_spec_value::<usize>(spec, "maxdepth")?.unwrap_or(1),
                        by,
                    )
                    .map(|blueprint| {
                        blueprint
                            .with_categorical(categorical)
                            .with_selection(selection)
                            .with_split_method(split_method)
                    })
                }
                "bhinge" => LearnerBlueprint::bhinge(
                    x,
                    name,
                    optional_spec_value::<usize>(spec, "n_candidates")?.unwrap_or(20),
                    optional_spec_value::<f64>(spec, "df")?.unwrap_or(4.0),
                    lambda,
                    optional_spec_value::<bool>(spec, "center")?.unwrap_or(true),
                    by,
                ),
                "bkernel" => LearnerBlueprint::bkernel(
                    x,
                    name,
                    optional_spec_value::<f64>(spec, "bandwidth")?.unwrap_or(1.0),
                    optional_spec_value::<f64>(spec, "df")?.unwrap_or(4.0),
                    optional_spec_value::<String>(spec, "kernel")?
                        .unwrap_or_else(|| "gaussian".to_string()),
                    optional_spec_value::<bool>(spec, "center")?.unwrap_or(false),
                    by,
                ),
                "bkernel_nystrom" => {
                    let landmarks = optional_spec_value::<usize>(spec, "landmarks")?
                        .ok_or_else(|| {
                            pyo3::exceptions::PyValueError::new_err(format!(
                                "LSS learner {name:?} kind 'bkernel_nystrom' requires landmarks"
                            ))
                        })?;
                    LearnerBlueprint::bkernel_nystrom(
                        x,
                        name,
                        optional_spec_value::<f64>(spec, "bandwidth")?.unwrap_or(1.0),
                        optional_spec_value::<f64>(spec, "df")?.unwrap_or(4.0),
                        optional_spec_value::<String>(spec, "kernel")?
                            .unwrap_or_else(|| "gaussian".to_string()),
                        landmarks,
                        optional_spec_value::<bool>(spec, "center")?.unwrap_or(false),
                        by,
                    )
                }
                "buser" => {
                    let n_features = optional_spec_value::<usize>(spec, "n_features")?
                        .ok_or_else(|| {
                            pyo3::exceptions::PyValueError::new_err(format!(
                                "LSS learner {name:?} requires n_features"
                            ))
                        })?;
                    if n_features == 0 || !x.len().is_multiple_of(n_features) {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "LSS learner {name:?} packed Buser design length {} is not divisible by n_features={n_features}",
                            x.len()
                        )));
                    }
                    let design = ndarray::Array2::from_shape_vec(
                        (x.len() / n_features, n_features),
                        x.to_vec(),
                    )
                    .map_err(|error| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "LSS learner {name:?} Buser design shape error: {error}"
                        ))
                    })?;
                    let k_obj = required_spec_item(spec, "K")?;
                    let k: PyReadonlyArray2<'_, f64> = k_obj.extract().map_err(|_| {
                        pyo3::exceptions::PyTypeError::new_err(format!(
                            "LSS learner {name:?} field 'K' must be a two-dimensional float64 array"
                        ))
                    })?;
                    LearnerBlueprint::buser(
                        design,
                        k.as_array().to_owned(),
                        name,
                        optional_spec_value::<f64>(spec, "df")?.unwrap_or(4.0),
                        lambda,
                        optional_spec_value::<bool>(spec, "center")?.unwrap_or(false),
                        by,
                    )
                }
                other => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "unsupported LSS learner kind {other:?}; expected 'bols', 'bbs', 'bns', 'brandom', 'bmrf', 'bspatial', 'bmono', 'bconstrained', 'btree', 'bhinge', 'bkernel', or 'buser'"
                    )))
                }
            }
            .map_err(mboostrs_error_to_pyerr)?;
                parameter_blueprints.push(blueprint);
            }
        }
    }
    Ok(parameter_blueprints)
}

/// Per-parameter wrapper for the LSS paths, which need one blueprint list per
/// distribution parameter and a parameter-indexed "requires at least one base
/// learner" check.
fn parse_lss_learner_blueprints(
    py: Python<'_>,
    per_param_specs: Vec<Vec<PyObject>>,
) -> PyResult<Vec<Vec<LearnerBlueprint>>> {
    let mut result = Vec::with_capacity(per_param_specs.len());
    for (parameter, specs) in per_param_specs.into_iter().enumerate() {
        if specs.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "LSS parameter {parameter} requires at least one base learner"
            )));
        }
        result.push(parse_learner_blueprint_list(py, specs)?);
    }
    Ok(result)
}

fn prepare_lss_blueprints(
    blueprints: &[Vec<LearnerBlueprint>],
    weights: ArrayView1<'_, f64>,
) -> Result<(Vec<Vec<PreparedLearner>>, Vec<Vec<BasisInfo>>), MboostrsError> {
    let mut per_param = Vec::with_capacity(blueprints.len());
    let mut basis_per_param = Vec::with_capacity(blueprints.len());
    for parameter_blueprints in blueprints {
        let mut learners = Vec::with_capacity(parameter_blueprints.len());
        let mut basis_info = Vec::with_capacity(parameter_blueprints.len());
        for blueprint in parameter_blueprints {
            let (learner, basis) = blueprint.prepare(weights)?;
            learners.push(learner);
            basis_info.push(basis);
        }
        per_param.push(learners);
        basis_per_param.push(basis_info);
    }
    Ok((per_param, basis_per_param))
}

/// Evaluate an LSS family's per-parameter offsets, negative gradients, and
/// joint risk at given points. Exposed for the docs/book correctness audit
/// (methodology/gamlss_framework.ipynb), analogous to `family_diagnostics`
/// but for `LssFamily`. `etas` is one length-`n` array per parameter (in
/// `param_names()` order); offsets are computed sequentially, matching
/// `fit_with`'s own `family.offset(k, y, w, &offs[..k])` usage, so a family
/// whose later-parameter offset formula reads earlier offsets (e.g. GaussianLSS's
/// sigma reading mu) gets the same `prior` it would during a real fit.
#[pyfunction]
fn lss_family_diagnostics<'py>(
    py: Python<'py>,
    name: &str,
    n_params: usize,
    y: PyReadonlyArray1<'_, f64>,
    etas: Vec<PyReadonlyArray1<'_, f64>>,
    w: PyReadonlyArray1<'_, f64>,
) -> PyResult<(Vec<f64>, Vec<Py<PyArray1<f64>>>, f64)> {
    let y_owned = y.as_array().to_owned();
    let family = make_lss_family_checked(name, n_params, y_owned.view())?;
    let w_owned = w.as_array().to_owned();
    let ys = y_owned.as_slice().expect("owned y is contiguous");
    let ws = w_owned.as_slice().expect("owned weights are contiguous");
    let n = ws.len();
    if n == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "lss_family_diagnostics requires at least one observation",
        ));
    }
    if ys.len() != n * family.response_span() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "lss_family_diagnostics expected {} response values for {n} observations, got {}",
            n * family.response_span(),
            ys.len()
        )));
    }
    if etas.len() != n_params {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "lss_family_diagnostics expected {n_params} eta arrays, got {}",
            etas.len()
        )));
    }
    if ys.iter().any(|value| !value.is_finite())
        || ws.iter().any(|value| !value.is_finite())
        || ws.iter().any(|&value| value < 0.0)
        || ws.iter().all(|&value| value == 0.0)
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "lss_family_diagnostics requires finite responses and non-negative finite weights with a positive sum",
        ));
    }

    let mut offsets = Vec::with_capacity(n_params);
    for k in 0..n_params {
        let off = family.offset(k, ys, ws, &offsets);
        offsets.push(off);
    }

    let etas_owned: Vec<ndarray::Array1<f64>> =
        etas.iter().map(|e| e.as_array().to_owned()).collect();
    for (parameter, eta) in etas_owned.iter().enumerate() {
        if eta.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "lss_family_diagnostics eta {parameter} has {} rows, expected {n}",
                eta.len()
            )));
        }
        if eta.iter().any(|value| !value.is_finite()) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "lss_family_diagnostics eta {parameter} contains a non-finite value"
            )));
        }
    }
    let mut grads = Vec::with_capacity(n_params);
    for k in 0..n_params {
        let mut out = vec![0.0f64; n];
        family.ngradient(k, ys, &etas_owned, ws, &mut out);
        grads.push(ndarray::Array1::from_vec(out).into_pyarray(py).unbind());
    }
    let risk = family.risk(ys, &etas_owned, ws);
    Ok((offsets, grads, risk))
}

/// Each observation's negative log-likelihood under an LSS family at the
/// link-scale predictors `etas` (one length-`n` array per parameter, in
/// `param_names()` order), and the natural-scale parameter values; see
/// `mboostrs_core::lss::row_nll_and_parameters`. The response is checked by
/// the family's `check_y` first.
#[pyfunction]
fn lss_row_nll<'py>(
    py: Python<'py>,
    name: &str,
    n_params: usize,
    y: PyReadonlyArray1<'_, f64>,
    etas: Vec<PyReadonlyArray1<'_, f64>>,
) -> PyResult<(Py<PyArray1<f64>>, Vec<Py<PyArray1<f64>>>)> {
    let y_owned = y.as_array().to_owned();
    let family = make_lss_family_checked(name, n_params, y_owned.view())?;
    let etas_owned: Vec<ndarray::Array1<f64>> =
        etas.iter().map(|e| e.as_array().to_owned()).collect();
    let ys = y_owned.as_slice().expect("owned y is contiguous");
    let (nll, parameters) = py
        .allow_threads(|| {
            mboostrs_core::lss::row_nll_and_parameters(family.as_ref(), ys, &etas_owned)
        })
        .map_err(mboostrs_error_to_pyerr)?;
    Ok((
        ndarray::Array1::from_vec(nll).into_pyarray(py).unbind(),
        parameters
            .into_iter()
            .map(|p| ndarray::Array1::from_vec(p).into_pyarray(py).unbind())
            .collect(),
    ))
}

/// glmboostLSS: cyclic component-wise boosting for a GAMLSS family.
///
/// `x_all` is the per-parameter linear designs concatenated column-wise, grouped
/// in family-parameter order; `param_ncols[k]` is the number of predictor columns
/// for parameter `k` (so the blocks are `x_all[:, sum(ncols[..k]) .. +ncols[k]]`).
/// Each parameter is fitted with an intercept learner + one scalar-bols learner
/// per column (centered), matching `fit_glmboost`, so the Python side can reuse
/// `predict_glmboost` per parameter.
#[pyfunction(signature = (x_all, param_ncols, learner_names, y, weights, mstop, nu, family_name, offsets=None, mstop_per_param=None, trace=false, stabilization="none", nu_per_param=None, step="fixed", step_iterations=4, response_scale=None))]
#[allow(clippy::too_many_arguments)]
fn fit_glmboostlss(
    py: Python<'_>,
    x_all: PyReadonlyArray2<'_, f64>,
    param_ncols: Vec<usize>,
    learner_names: Vec<String>,
    y: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    mstop: u32,
    nu: f64,
    family_name: &str,
    offsets: Option<Vec<f64>>,
    mstop_per_param: Option<Vec<usize>>,
    trace: bool,
    stabilization: &str,
    nu_per_param: Option<Vec<f64>>,
    step: &str,
    step_iterations: u32,
    response_scale: Option<Bound<'_, PyAny>>,
) -> PyResult<LssFitResult> {
    use mboostrs_core::lss::try_fit_lss;

    let binding_started = trace.then(Instant::now);
    let x = x_all.as_array();
    let yv = y.as_array();
    let wv = weights.as_array();
    // Observation count comes from weights (survival families carry a 2-column
    // response, so y may be longer than the number of observations).
    let n = wv.len();

    if x.nrows() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "x_all has {} rows but weights has {} elements",
            x.nrows(),
            n
        )));
    }
    let total_cols: usize = param_ncols.iter().sum();
    if x.ncols() != total_cols || learner_names.len() != total_cols {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "x_all columns and learner_names must equal sum(param_ncols)",
        ));
    }

    let family = make_lss_family_checked(family_name, param_ncols.len(), yv)?;

    let stabilization =
        mboostrs_core::lss::Stabilization::parse(stabilization).map_err(mboostrs_error_to_pyerr)?;
    let ctrl = BoostControl::new(mstop, nu)
        .map_err(mboostrs_error_to_pyerr)?
        .with_trace(trace);
    let mut ctrl = apply_step(ctrl, step, step_iterations)?;
    ctrl.stabilization = stabilization;
    ctrl.nu_per_param = nu_per_param;
    ctrl = ctrl
        .with_response_scale(parse_response_scale(response_scale.as_ref())?)
        .map_err(mboostrs_error_to_pyerr)?;

    let preparation_timer = trace.then(PreparationTimer::start);
    let ones = ndarray::Array1::ones(n);
    let w_owned = wv.to_owned();

    // One learner set per parameter: intercept + a centered scalar bols/column.
    let mut per_param: Vec<Vec<PreparedLearner>> = Vec::with_capacity(param_ncols.len());
    let mut col0 = 0usize;
    for &pc in &param_ncols {
        let mut learners: Vec<PreparedLearner> = Vec::with_capacity(pc + 1);
        learners.push(mboostrs_core::learners::prepare_scalar_bols(
            ones.view(),
            w_owned.view(),
            0.0,
            false,
            "(Intercept)",
            None,
        ));
        for j in 0..pc {
            let col = x.column(col0 + j);
            // gamboostLSS centers every covariate by its simple column mean
            // (weights enter the normal equations and the offset, not the
            // centering). Weighted means here break upstream parity under
            // weights (R gamboostLSS reproduces colMeans centering exactly).
            let mean = col.mean().unwrap_or(0.0);
            learners.push(mboostrs_core::learners::prepare_scalar_bols_fixed_center(
                col,
                w_owned.view(),
                0.0,
                mean,
                learner_names
                    .get(col0 + j)
                    .map(|s| s.as_str())
                    .unwrap_or("?"),
            ));
        }
        per_param.push(learners);
        col0 += pc;
    }
    let preparation_timing = preparation_timer.map(PreparationTimer::finish);

    if let Some(ref m) = mstop_per_param {
        if m.len() != param_ncols.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "mstop_per_param must have one entry per parameter",
            ));
        }
    }
    let offsets_ref = offsets.as_deref();
    let mstop_ref = mstop_per_param.as_deref();
    let binding_input_ns = binding_started.map(|started| started.elapsed().as_nanos() as u64);
    let path = py
        .allow_threads(|| {
            try_fit_lss(
                &mut per_param,
                yv,
                w_owned.view(),
                family.as_ref(),
                &ctrl,
                mstop_ref,
                offsets_ref,
            )
        })
        .map_err(mboostrs_error_to_pyerr)?;

    let binding_output_started = trace.then(Instant::now);
    let mut results: Vec<LssParamResult> = Vec::with_capacity(path.per_param.len());
    for bp in &path.per_param {
        let coefs_flat: Vec<f64> = bp.coefs.iter().map(|c| c[0]).collect();
        results.push((
            bp.offset,
            bp.selected.clone(),
            coefs_flat,
            bp.fitted.to_vec(),
        ));
    }
    let mut report = path.execution_report();
    attach_frontend_timings(
        &mut report,
        preparation_timing,
        binding_input_ns,
        binding_output_started,
    );
    let execution_report = fit_report_to_py(py, &report)?;
    let param_risk = path.per_param.iter().map(|p| p.risk.clone()).collect();
    Ok((
        results,
        path.risk,
        path.selection_order,
        execution_report,
        param_risk,
    ))
}

/// Non-cyclic glmboostLSS (R's `method = "noncyclic"`): builds the same
/// intercept + scalar-bols learners as `fit_glmboostlss`, then runs the
/// non-cyclic driver (one parameter updated per step, chosen by risk reduction).
#[pyfunction(signature = (x_all, param_ncols, learner_names, y, weights, mstop, nu, family_name, offsets=None, trace=false, stabilization="none", nu_per_param=None, step="fixed", step_iterations=4, response_scale=None))]
#[allow(clippy::too_many_arguments)]
fn fit_glmboostlss_noncyclic(
    py: Python<'_>,
    x_all: PyReadonlyArray2<'_, f64>,
    param_ncols: Vec<usize>,
    learner_names: Vec<String>,
    y: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    mstop: u32,
    nu: f64,
    family_name: &str,
    offsets: Option<Vec<f64>>,
    trace: bool,
    stabilization: &str,
    nu_per_param: Option<Vec<f64>>,
    step: &str,
    step_iterations: u32,
    response_scale: Option<Bound<'_, PyAny>>,
) -> PyResult<LssFitResult> {
    use mboostrs_core::lss::try_fit_lss_noncyclic;
    let binding_started = trace.then(Instant::now);
    let x = x_all.as_array();
    let yv = y.as_array();
    let wv = weights.as_array();
    let n = wv.len();
    let w_owned = wv.to_owned();
    let family = make_lss_family_checked(family_name, param_ncols.len(), yv)?;
    let stabilization =
        mboostrs_core::lss::Stabilization::parse(stabilization).map_err(mboostrs_error_to_pyerr)?;
    let ctrl = BoostControl::new(mstop, nu)
        .map_err(mboostrs_error_to_pyerr)?
        .with_trace(trace);
    let mut ctrl = apply_step(ctrl, step, step_iterations)?;
    ctrl.stabilization = stabilization;
    ctrl.nu_per_param = nu_per_param;
    ctrl = ctrl
        .with_response_scale(parse_response_scale(response_scale.as_ref())?)
        .map_err(mboostrs_error_to_pyerr)?;
    let preparation_timer = trace.then(PreparationTimer::start);
    let ones = ndarray::Array1::ones(n);
    let mut per_param: Vec<Vec<PreparedLearner>> = Vec::with_capacity(param_ncols.len());
    let mut col0 = 0usize;
    for &pc in &param_ncols {
        let mut learners: Vec<PreparedLearner> = Vec::with_capacity(pc + 1);
        learners.push(mboostrs_core::learners::prepare_scalar_bols(
            ones.view(),
            w_owned.view(),
            0.0,
            false,
            "(Intercept)",
            None,
        ));
        for j in 0..pc {
            let col = x.column(col0 + j);
            // gamboostLSS centers every covariate by its simple column mean
            // (weights enter the normal equations and the offset, not the
            // centering). Weighted means here break upstream parity under
            // weights (R gamboostLSS reproduces colMeans centering exactly).
            let mean = col.mean().unwrap_or(0.0);
            learners.push(mboostrs_core::learners::prepare_scalar_bols_fixed_center(
                col,
                w_owned.view(),
                0.0,
                mean,
                learner_names
                    .get(col0 + j)
                    .map(|s| s.as_str())
                    .unwrap_or("?"),
            ));
        }
        per_param.push(learners);
        col0 += pc;
    }
    let preparation_timing = preparation_timer.map(PreparationTimer::finish);
    let offsets_ref = offsets.as_deref();
    let binding_input_ns = binding_started.map(|started| started.elapsed().as_nanos() as u64);
    let path = py
        .allow_threads(|| {
            try_fit_lss_noncyclic(
                &mut per_param,
                yv,
                w_owned.view(),
                family.as_ref(),
                &ctrl,
                offsets_ref,
            )
        })
        .map_err(mboostrs_error_to_pyerr)?;
    let binding_output_started = trace.then(Instant::now);
    let mut results: Vec<LssParamResult> = Vec::with_capacity(path.per_param.len());
    for bp in &path.per_param {
        let coefs_flat: Vec<f64> = bp.coefs.iter().map(|c| c[0]).collect();
        results.push((
            bp.offset,
            bp.selected.clone(),
            coefs_flat,
            bp.fitted.to_vec(),
        ));
    }
    let mut report = path.execution_report();
    attach_frontend_timings(
        &mut report,
        preparation_timing,
        binding_input_ns,
        binding_output_started,
    );
    let execution_report = fit_report_to_py(py, &report)?;
    let param_risk = path.per_param.iter().map(|p| p.risk.clone()).collect();
    Ok((
        results,
        path.risk,
        path.selection_order,
        execution_report,
        param_risk,
    ))
}

/// Per-subsample combined base-learner selection sequences for non-cyclic
/// glmboostLSS, used by stability selection. For each fold (column of `folds`,
/// the in-bag weights), runs the non-cyclic fit and returns the ordered sequence
/// of selected *combined* learner indices, where parameter `p`'s learners occupy
/// the block `[offset_p .. offset_p + param_ncols[p] + 1)` (intercept first),
/// matching R's `name.parameter` learner ordering.
///
/// `q` is stability selection's cap on distinct components: a fold stops as
/// soon as it has selected that many, because nothing it selects afterwards
/// reaches the reported frequencies. `None` runs every fold to `mstop`.
#[pyfunction(signature = (x_all, param_ncols, y, weights, mstop, nu, family_name, folds, q=None, stabilization="none", nu_per_param=None, step="fixed", step_iterations=4, response_scale=None))]
#[allow(clippy::too_many_arguments)]
fn stabsel_selected_glmboostlss(
    py: Python<'_>,
    x_all: PyReadonlyArray2<'_, f64>,
    param_ncols: Vec<usize>,
    y: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    mstop: u32,
    nu: f64,
    family_name: &str,
    folds: PyReadonlyArray2<'_, f64>,
    q: Option<usize>,
    stabilization: &str,
    nu_per_param: Option<Vec<f64>>,
    step: &str,
    step_iterations: u32,
    response_scale: Option<Bound<'_, PyAny>>,
) -> PyResult<Vec<Vec<usize>>> {
    use mboostrs_core::lss::try_fit_lss_noncyclic_capped;
    let x = x_all.as_array();
    let yv = y.as_array();
    let wv = weights.as_array();
    let n = wv.len();
    let family = make_lss_family_checked(family_name, param_ncols.len(), yv)?;
    let ctrl = BoostControl::new(mstop, nu).map_err(mboostrs_error_to_pyerr)?;
    let mut ctrl = apply_step(ctrl, step, step_iterations)?;
    ctrl.stabilization =
        mboostrs_core::lss::Stabilization::parse(stabilization).map_err(mboostrs_error_to_pyerr)?;
    ctrl.nu_per_param = nu_per_param;
    ctrl = ctrl
        .with_response_scale(parse_response_scale(response_scale.as_ref())?)
        .map_err(mboostrs_error_to_pyerr)?;
    let mut offset = vec![0usize; param_ncols.len()];
    let mut acc = 0usize;
    for (p, &pc) in param_ncols.iter().enumerate() {
        offset[p] = acc;
        acc += pc + 1;
    }
    let folds_arr = folds.as_array().to_owned();
    let b = folds_arr.ncols();
    let result = py
        .allow_threads(|| {
            use rayon::prelude::*;
            // Subsamples are independent; run them in parallel. The indexed
            // collect keeps fold order, so output is identical to sequential.
            (0..b)
                .into_par_iter()
                .map(|fold| {
                    let fold_w: Vec<f64> = (0..n).map(|i| folds_arr[[i, fold]]).collect();
                    let fw = ndarray::Array1::from_vec(fold_w);
                    let sw: f64 = fw.sum();
                    let ones = ndarray::Array1::ones(n);
                    let mut per_param: Vec<Vec<PreparedLearner>> =
                        Vec::with_capacity(param_ncols.len());
                    let mut col0 = 0usize;
                    for &pc in &param_ncols {
                        let mut learners: Vec<PreparedLearner> = Vec::with_capacity(pc + 1);
                        learners.push(mboostrs_core::learners::prepare_scalar_bols(
                            ones.view(),
                            fw.view(),
                            0.0,
                            false,
                            "(Intercept)",
                            None,
                        ));
                        for j in 0..pc {
                            let col = x.column(col0 + j);
                            let mean = if sw > 0.0 {
                                col.iter().zip(fw.iter()).map(|(c, wi)| c * wi).sum::<f64>() / sw
                            } else {
                                0.0
                            };
                            learners.push(
                                mboostrs_core::learners::prepare_scalar_bols_fixed_center(
                                    col,
                                    fw.view(),
                                    0.0,
                                    mean,
                                    "x",
                                ),
                            );
                        }
                        per_param.push(learners);
                        col0 += pc;
                    }
                    let path = try_fit_lss_noncyclic_capped(
                        &mut per_param,
                        yv,
                        fw.view(),
                        family.as_ref(),
                        &ctrl,
                        None,
                        q,
                    )?;
                    let mut cursor = vec![0usize; param_ncols.len()];
                    let mut seq = Vec::with_capacity(path.selection_order.len());
                    for &p in &path.selection_order {
                        let l = path.per_param[p].selected[cursor[p]];
                        cursor[p] += 1;
                        seq.push(offset[p] + l);
                    }
                    Ok::<_, MboostrsError>(seq)
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .map_err(mboostrs_error_to_pyerr)?;
    Ok(result)
}

/// Per-subsample selection order for **smooth** gamboostLSS (bbs), for stability
/// selection. Mirrors `stabsel_selected_glmboostlss` but with bbs learners and
/// no per-parameter intercept (parameter `p`'s learners occupy
/// `[offset_p .. offset_p + param_nlearners[p])`). bbs learners are rebuilt per
/// subsample so `df -> lambda` is recomputed from the subsample weights.
#[pyfunction(signature = (bbs_x, param_nlearners, bbs_knots, bbs_degree, bbs_differences, bbs_df, bbs_center, learner_names, y, weights, mstop, nu, family_name, folds, stabilization="none", nu_per_param=None, step="fixed", step_iterations=4, response_scale=None))]
#[allow(clippy::too_many_arguments)]
fn stabsel_selected_gamboostlss(
    py: Python<'_>,
    bbs_x: Vec<PyReadonlyArray1<'_, f64>>,
    param_nlearners: Vec<usize>,
    bbs_knots: Vec<usize>,
    bbs_degree: Vec<usize>,
    bbs_differences: Vec<usize>,
    bbs_df: Vec<f64>,
    bbs_center: Vec<bool>,
    learner_names: Vec<String>,
    y: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    mstop: u32,
    nu: f64,
    family_name: &str,
    folds: PyReadonlyArray2<'_, f64>,
    stabilization: &str,
    nu_per_param: Option<Vec<f64>>,
    step: &str,
    step_iterations: u32,
    response_scale: Option<Bound<'_, PyAny>>,
) -> PyResult<Vec<Vec<usize>>> {
    let _ = weights;
    let y_owned = y.as_array().to_owned();
    let n = y_owned.len();
    let total: usize = param_nlearners.iter().sum();
    if bbs_x.len() != total
        || bbs_knots.len() != total
        || bbs_degree.len() != total
        || bbs_differences.len() != total
        || bbs_df.len() != total
        || bbs_center.len() != total
        || learner_names.len() != total
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "all bbs spec lists must have length sum(param_nlearners)",
        ));
    }
    let family = make_lss_family_checked(family_name, param_nlearners.len(), y_owned.view())?;
    let ctrl = BoostControl::new(mstop, nu).map_err(mboostrs_error_to_pyerr)?;
    let mut ctrl = apply_step(ctrl, step, step_iterations)?;
    ctrl.stabilization =
        mboostrs_core::lss::Stabilization::parse(stabilization).map_err(mboostrs_error_to_pyerr)?;
    ctrl.nu_per_param = nu_per_param;
    ctrl = ctrl
        .with_response_scale(parse_response_scale(response_scale.as_ref())?)
        .map_err(mboostrs_error_to_pyerr)?;
    // Global learner index offsets (no per-parameter intercept for bbs).
    let mut offset = vec![0usize; param_nlearners.len()];
    let mut acc = 0usize;
    for (p, &nl) in param_nlearners.iter().enumerate() {
        offset[p] = acc;
        acc += nl;
    }
    let bbs_cols: Vec<Vec<f64>> = bbs_x.iter().map(|a| a.as_array().to_vec()).collect();
    let folds_arr = folds.as_array().to_owned();
    let b = folds_arr.ncols();
    let result = py
        .allow_threads(|| {
            use rayon::prelude::*;
            // Fold-invariant templates (basis + penalty), built once before the
            // per-subsample loop — every subsample previously rebuilt every
            // bbs learner's basis/centering from the raw x columns from
            // scratch. Only the weight-dependent df -> lambda solve below runs
            // per subsample.
            let templates: Vec<(ndarray::Array2<f64>, ndarray::Array2<f64>)> = (0..total)
                .map(|idx| {
                    let col = ndarray::ArrayView1::from(bbs_cols[idx].as_slice());
                    mboostrs_core::learners::build_bbs_template(
                        col,
                        bbs_knots[idx],
                        bbs_degree[idx],
                        bbs_differences[idx],
                        bbs_center[idx],
                        None,
                        false,
                        None,
                        0,
                    )
                })
                .collect();

            // Subsamples are independent; run them in parallel. The indexed
            // collect keeps fold order, so output is identical to sequential.
            (0..b)
                .into_par_iter()
                .map(|fold| {
                    let fold_w: Vec<f64> = (0..n).map(|i| folds_arr[[i, fold]]).collect();
                    let fw = ndarray::Array1::from_vec(fold_w);
                    let mut per_param: Vec<Vec<PreparedLearner>> =
                        Vec::with_capacity(param_nlearners.len());
                    let mut idx = 0usize;
                    for &nl in &param_nlearners {
                        let mut learners: Vec<PreparedLearner> = Vec::with_capacity(nl);
                        for _ in 0..nl {
                            let (design, penalty) = templates[idx].clone();
                            learners.push(mboostrs_core::learners::prepare_bbs_from_template(
                                design,
                                penalty,
                                fw.view(),
                                bbs_df[idx],
                                None,
                                &learner_names[idx],
                            ));
                            idx += 1;
                        }
                        per_param.push(learners);
                    }
                    let path = mboostrs_core::lss::try_fit_lss_noncyclic(
                        &mut per_param,
                        y_owned.view(),
                        fw.view(),
                        family.as_ref(),
                        &ctrl,
                        None,
                    )?;
                    let mut cursor = vec![0usize; param_nlearners.len()];
                    let mut seq = Vec::with_capacity(path.selection_order.len());
                    for &p in &path.selection_order {
                        let l = path.per_param[p].selected[cursor[p]];
                        cursor[p] += 1;
                        seq.push(offset[p] + l);
                    }
                    Ok::<_, MboostrsError>(seq)
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .map_err(mboostrs_error_to_pyerr)?;
    Ok(result)
}

/// Cross-validated risk for glmboostLSS over a grid of per-parameter mstop
/// combinations (R's `cvrisk.mboostLSS`). Builds the same intercept + scalar-bols
/// learners as `fit_glmboostlss`, then evaluates per-fold out-of-bag risk for
/// each grid point. `folds` is n x B in-bag weights; `grid[g]` is a per-parameter
/// mstop vector. Returns a B x grid.len() matrix of per-fold OOB risks.
#[pyfunction(signature = (x_all, param_ncols, learner_names, y, weights, mstop, nu, family_name, folds, grid, stabilization="none", nu_per_param=None, step="fixed", step_iterations=4, response_scale=None))]
#[allow(clippy::too_many_arguments)]
fn cvrisk_glmboostlss(
    py: Python<'_>,
    x_all: PyReadonlyArray2<'_, f64>,
    param_ncols: Vec<usize>,
    learner_names: Vec<String>,
    y: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    mstop: u32,
    nu: f64,
    family_name: &str,
    folds: PyReadonlyArray2<'_, f64>,
    grid: Vec<Vec<usize>>,
    stabilization: &str,
    nu_per_param: Option<Vec<f64>>,
    step: &str,
    step_iterations: u32,
    response_scale: Option<Bound<'_, PyAny>>,
) -> PyResult<Vec<Vec<f64>>> {
    use mboostrs_core::lss::try_cvrisk_lss;

    let x = x_all.as_array();
    let yv = y.as_array();
    let wv = weights.as_array();
    let n = wv.len();
    let w_owned = wv.to_owned();

    let family = make_lss_family_checked(family_name, param_ncols.len(), yv)?;
    let ctrl = BoostControl::new(mstop, nu).map_err(mboostrs_error_to_pyerr)?;
    let mut ctrl = apply_step(ctrl, step, step_iterations)?;
    ctrl.stabilization =
        mboostrs_core::lss::Stabilization::parse(stabilization).map_err(mboostrs_error_to_pyerr)?;
    ctrl.nu_per_param = nu_per_param;
    ctrl = ctrl
        .with_response_scale(parse_response_scale(response_scale.as_ref())?)
        .map_err(mboostrs_error_to_pyerr)?;

    // Build intercept + scalar-bols learners per parameter (as in fit_glmboostlss).
    let ones = ndarray::Array1::ones(n);
    let mut per_param: Vec<Vec<PreparedLearner>> = Vec::with_capacity(param_ncols.len());
    let mut col0 = 0usize;
    for &pc in &param_ncols {
        let mut learners: Vec<PreparedLearner> = Vec::with_capacity(pc + 1);
        learners.push(mboostrs_core::learners::prepare_scalar_bols(
            ones.view(),
            w_owned.view(),
            0.0,
            false,
            "(Intercept)",
            None,
        ));
        for j in 0..pc {
            let col = x.column(col0 + j);
            // gamboostLSS centers every covariate by its simple column mean
            // (weights enter the normal equations and the offset, not the
            // centering). Weighted means here break upstream parity under
            // weights (R gamboostLSS reproduces colMeans centering exactly).
            let mean = col.mean().unwrap_or(0.0);
            learners.push(mboostrs_core::learners::prepare_scalar_bols_fixed_center(
                col,
                w_owned.view(),
                0.0,
                mean,
                learner_names
                    .get(col0 + j)
                    .map(|s| s.as_str())
                    .unwrap_or("?"),
            ));
        }
        per_param.push(learners);
        col0 += pc;
    }

    let folds_arr = folds.as_array().to_owned();
    let risk = py
        .allow_threads(|| {
            try_cvrisk_lss(
                &per_param,
                yv,
                w_owned.view(),
                family.as_ref(),
                &ctrl,
                folds_arr.view(),
                &grid,
            )
        })
        .map_err(mboostrs_error_to_pyerr)?;
    Ok(risk.outer_iter().map(|row| row.to_vec()).collect())
}

/// Cross-validated risk for **smooth** gamboostLSS (bbs base-learners) over a
/// grid of per-parameter mstop combinations. Mirrors `cvrisk_glmboostlss` but
/// **rebuilds the bbs learners per fold** (via `cvrisk_lss_with`) so the
/// `df -> lambda` penalty is recomputed from each fold's weights, matching R's
/// per-fold `dpp`. `folds` is n x B in-bag weights; returns a B x grid.len()
/// matrix of per-fold OOB risks.
#[pyfunction(signature = (bbs_x, param_nlearners, bbs_knots, bbs_degree, bbs_differences, bbs_df, bbs_center, learner_names, y, weights, mstop, nu, family_name, folds, grid, judge=None, stabilization="none", nu_per_param=None, step="fixed", step_iterations=4, response_scale=None))]
#[allow(clippy::too_many_arguments)]
fn cvrisk_gamboostlss(
    py: Python<'_>,
    bbs_x: Vec<PyReadonlyArray1<'_, f64>>,
    param_nlearners: Vec<usize>,
    bbs_knots: Vec<usize>,
    bbs_degree: Vec<usize>,
    bbs_differences: Vec<usize>,
    bbs_df: Vec<f64>,
    bbs_center: Vec<bool>,
    learner_names: Vec<String>,
    y: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    mstop: u32,
    nu: f64,
    family_name: &str,
    folds: PyReadonlyArray2<'_, f64>,
    grid: Vec<Vec<usize>>,
    judge: Option<PyReadonlyArray2<'_, bool>>,
    stabilization: &str,
    nu_per_param: Option<Vec<f64>>,
    step: &str,
    step_iterations: u32,
    response_scale: Option<Bound<'_, PyAny>>,
) -> PyResult<Vec<Vec<f64>>> {
    use mboostrs_core::lss::try_cvrisk_lss_with;

    let y_owned = y.as_array().to_owned();
    let w_owned = weights.as_array().to_owned();
    let folds_arr = folds.as_array().to_owned();
    let judge_arr = judge_mask_from_readonly(judge, y_owned.len())?;
    let total: usize = param_nlearners.iter().sum();
    if bbs_x.len() != total
        || bbs_knots.len() != total
        || bbs_degree.len() != total
        || bbs_differences.len() != total
        || bbs_df.len() != total
        || bbs_center.len() != total
        || learner_names.len() != total
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "all bbs spec lists must have length sum(param_nlearners)",
        ));
    }

    let family = make_lss_family_checked(family_name, param_nlearners.len(), y_owned.view())?;
    let ctrl = BoostControl::new(mstop, nu).map_err(mboostrs_error_to_pyerr)?;
    let mut ctrl = apply_step(ctrl, step, step_iterations)?;
    ctrl.stabilization =
        mboostrs_core::lss::Stabilization::parse(stabilization).map_err(mboostrs_error_to_pyerr)?;
    ctrl.nu_per_param = nu_per_param;
    ctrl = ctrl
        .with_response_scale(parse_response_scale(response_scale.as_ref())?)
        .map_err(mboostrs_error_to_pyerr)?;

    // Owned bbs columns so the per-fold preparer can run without the GIL.
    let bbs_cols: Vec<Vec<f64>> = bbs_x.iter().map(|a| a.as_array().to_vec()).collect();

    let risk = py
        .allow_threads(|| {
            // Fold-invariant templates (basis + penalty), built once — not
            // rebuilt from the raw x columns on every fold. Only the
            // weight-dependent df -> lambda solve below runs per fold (R
            // re-runs dpp per fold too, so this matches R's semantics while
            // skipping the redundant basis/centering recomputation).
            let templates: Vec<(ndarray::Array2<f64>, ndarray::Array2<f64>)> = (0..total)
                .map(|idx| {
                    let col = ndarray::ArrayView1::from(bbs_cols[idx].as_slice());
                    mboostrs_core::learners::build_bbs_template(
                        col,
                        bbs_knots[idx],
                        bbs_degree[idx],
                        bbs_differences[idx],
                        bbs_center[idx],
                        None,
                        false,
                        None,
                        0,
                    )
                })
                .collect();

            let prepare = |fold_w: ndarray::ArrayView1<f64>| -> Vec<Vec<PreparedLearner>> {
                let mut per_param: Vec<Vec<PreparedLearner>> =
                    Vec::with_capacity(param_nlearners.len());
                let mut idx = 0usize;
                for &nl in &param_nlearners {
                    let mut learners: Vec<PreparedLearner> = Vec::with_capacity(nl);
                    for _ in 0..nl {
                        let (design, penalty) = templates[idx].clone();
                        learners.push(mboostrs_core::learners::prepare_bbs_from_template(
                            design,
                            penalty,
                            fold_w,
                            bbs_df[idx],
                            None,
                            &learner_names[idx],
                        ));
                        idx += 1;
                    }
                    per_param.push(learners);
                }
                per_param
            };
            try_cvrisk_lss_with(
                |fold_weights| Ok(prepare(fold_weights)),
                y_owned.view(),
                w_owned.view(),
                family.as_ref(),
                &ctrl,
                folds_arr.view(),
                judge_arr.as_ref(),
                &grid,
            )
        })
        .map_err(mboostrs_error_to_pyerr)?;
    Ok(risk.outer_iter().map(|row| row.to_vec()).collect())
}

/// Per-parameter result for a gamboostLSS fit, with full (vector) coefs per
/// boosting step: (offset, selected, coefs_per_step, fitted).
type GamLssParamResult = (f64, Vec<usize>, Vec<Vec<f64>>, Vec<f64>);
/// (per-parameter results, joint risk, global parameter order, report).
type GamLssFitResult = (Vec<GamLssParamResult>, Vec<f64>, Vec<usize>, PyObject);
/// Generic mixed-learner result. One `LearnerStore` is returned per
/// distribution parameter so training and new-data prediction replay the same
/// prepared design used for fitting.
type MixedGamLssFitResult = (
    Vec<GamLssParamResult>,
    Vec<f64>,
    Vec<usize>,
    Vec<Py<LearnerStore>>,
    PyObject,
    Vec<Vec<f64>>,
);

/// gamboostLSS: cyclic component-wise boosting for a GAMLSS family with smooth
/// (bbs) base-learners. Each parameter gets `param_nlearners[k]` bbs learners
/// built from the flat `bbs_x` columns; there is no separate intercept learner
/// (the per-parameter offset plays that role, matching R's gamboostLSS).
#[pyfunction(signature = (bbs_x, param_nlearners, bbs_knots, bbs_degree, bbs_differences, bbs_df, bbs_center, learner_names, y, weights, mstop, nu, family_name, offsets=None, mstop_per_param=None, trace=false, stabilization="none", nu_per_param=None, step="fixed", step_iterations=4, response_scale=None))]
#[allow(clippy::too_many_arguments)]
fn fit_gamboostlss(
    py: Python<'_>,
    bbs_x: Vec<PyReadonlyArray1<'_, f64>>,
    param_nlearners: Vec<usize>,
    bbs_knots: Vec<usize>,
    bbs_degree: Vec<usize>,
    bbs_differences: Vec<usize>,
    bbs_df: Vec<f64>,
    bbs_center: Vec<bool>,
    learner_names: Vec<String>,
    y: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    mstop: u32,
    nu: f64,
    family_name: &str,
    offsets: Option<Vec<f64>>,
    mstop_per_param: Option<Vec<usize>>,
    trace: bool,
    stabilization: &str,
    nu_per_param: Option<Vec<f64>>,
    step: &str,
    step_iterations: u32,
    response_scale: Option<Bound<'_, PyAny>>,
) -> PyResult<GamLssFitResult> {
    use mboostrs_core::lss::try_fit_lss;

    let binding_started = trace.then(Instant::now);
    let yv = y.as_array();
    let wv = weights.as_array();
    let w_owned = wv.to_owned();
    let total: usize = param_nlearners.iter().sum();
    if bbs_x.len() != total
        || bbs_knots.len() != total
        || bbs_degree.len() != total
        || bbs_differences.len() != total
        || bbs_df.len() != total
        || bbs_center.len() != total
        || learner_names.len() != total
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "all bbs spec lists must have length sum(param_nlearners)",
        ));
    }

    let family = make_lss_family_checked(family_name, param_nlearners.len(), yv)?;

    let stabilization =
        mboostrs_core::lss::Stabilization::parse(stabilization).map_err(mboostrs_error_to_pyerr)?;
    let ctrl = BoostControl::new(mstop, nu)
        .map_err(mboostrs_error_to_pyerr)?
        .with_trace(trace);
    let mut ctrl = apply_step(ctrl, step, step_iterations)?;
    ctrl.stabilization = stabilization;
    ctrl.nu_per_param = nu_per_param;
    ctrl = ctrl
        .with_response_scale(parse_response_scale(response_scale.as_ref())?)
        .map_err(mboostrs_error_to_pyerr)?;

    let preparation_timer = trace.then(PreparationTimer::start);
    let mut per_param: Vec<Vec<PreparedLearner>> = Vec::with_capacity(param_nlearners.len());
    let mut idx = 0usize;
    for &nl in &param_nlearners {
        let mut learners: Vec<PreparedLearner> = Vec::with_capacity(nl);
        for _ in 0..nl {
            let view = bbs_x[idx].as_array();
            let lam_opt = None; // df-based lambda, matching R gamboostLSS bbs default
            learners.push(mboostrs_core::learners::prepare_bbs_with_lambda(
                view,
                w_owned.view(),
                bbs_knots[idx],
                bbs_degree[idx],
                bbs_differences[idx],
                bbs_df[idx],
                lam_opt,
                bbs_center[idx],
                &learner_names[idx],
                None,
                false,
                None,
                0,
            ));
            idx += 1;
        }
        per_param.push(learners);
    }
    let preparation_timing = preparation_timer.map(PreparationTimer::finish);

    if let Some(ref m) = mstop_per_param {
        if m.len() != param_nlearners.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "mstop_per_param must have one entry per parameter",
            ));
        }
    }
    let offsets_ref = offsets.as_deref();
    let mstop_ref = mstop_per_param.as_deref();
    let binding_input_ns = binding_started.map(|started| started.elapsed().as_nanos() as u64);
    let path = py
        .allow_threads(|| {
            try_fit_lss(
                &mut per_param,
                yv,
                w_owned.view(),
                family.as_ref(),
                &ctrl,
                mstop_ref,
                offsets_ref,
            )
        })
        .map_err(mboostrs_error_to_pyerr)?;

    let binding_output_started = trace.then(Instant::now);
    let mut results: Vec<GamLssParamResult> = Vec::with_capacity(path.per_param.len());
    for bp in &path.per_param {
        results.push((
            bp.offset,
            bp.selected.clone(),
            bp.coefs.clone(),
            bp.fitted.to_vec(),
        ));
    }
    let mut report = path.execution_report();
    attach_frontend_timings(
        &mut report,
        preparation_timing,
        binding_input_ns,
        binding_output_started,
    );
    let execution_report = fit_report_to_py(py, &report)?;
    Ok((results, path.risk, path.selection_order, execution_report))
}

/// Generic `gamboostLSS` entry point for heterogeneous parameter-scoped
/// learner sets. The language front end supplies a list of learner dictionaries
/// per distribution parameter; the dictionaries are converted immediately to
/// core-owned [`LearnerBlueprint`] values before releasing the GIL.
/// A front end's `response_scale`: `None` fits the raw response, `"auto"`
/// the SD of the response each fit sees, a number a fixed scale.
fn parse_response_scale(
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<mboostrs_core::control::ResponseScale> {
    use mboostrs_core::control::ResponseScale;
    let Some(value) = value else {
        return Ok(ResponseScale::Raw);
    };
    if value.is_none() {
        return Ok(ResponseScale::Raw);
    }
    if let Ok(text) = value.extract::<String>() {
        if text == "auto" {
            return Ok(ResponseScale::Auto);
        }
        if text == "none" {
            return Ok(ResponseScale::Raw);
        }
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "response_scale must be \"auto\", \"none\" (or None) or a positive number, got {text:?}"
        )));
    }
    let scale: f64 = value.extract().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "response_scale must be \"auto\", None or a positive number",
        )
    })?;
    Ok(ResponseScale::Fixed(scale))
}

/// The default response scale of a distributional fit: the SD of the part of
/// `y` the family's scale applies to, over rows with positive weight. `None`
/// for families with no scale (counts, proportions, compositions) or a
/// response with no spread.
#[pyfunction]
fn lss_auto_response_scale(
    family_name: &str,
    n_params: usize,
    y: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
) -> PyResult<Option<f64>> {
    let family =
        mboostrs_core::lss::make_family(family_name, n_params).map_err(mboostrs_error_to_pyerr)?;
    let ys = y.as_slice()?;
    let ws = weights.as_slice()?;
    Ok(mboostrs_core::lss::auto_response_scale(
        family.as_ref(),
        ys,
        ws,
    ))
}

#[pyfunction(signature = (learner_specs, y, weights, mstop, nu, family_name, offsets=None, mstop_per_param=None, method="cyclic", trace=false, stabilization="none", nu_per_param=None, step="fixed", step_iterations=4, response_scale=None))]
#[allow(clippy::too_many_arguments)]
fn fit_gamboostlss_mixed(
    py: Python<'_>,
    learner_specs: Vec<Vec<PyObject>>,
    y: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    mstop: u32,
    nu: f64,
    family_name: &str,
    offsets: Option<Vec<f64>>,
    mstop_per_param: Option<Vec<usize>>,
    method: &str,
    trace: bool,
    stabilization: &str,
    nu_per_param: Option<Vec<f64>>,
    step: &str,
    step_iterations: u32,
    response_scale: Option<Bound<'_, PyAny>>,
) -> PyResult<MixedGamLssFitResult> {
    use mboostrs_core::lss::{try_fit_lss, try_fit_lss_noncyclic};

    if method != "cyclic" && method != "noncyclic" {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "method must be 'cyclic' or 'noncyclic'",
        ));
    }
    if method == "noncyclic" && mstop_per_param.is_some() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "noncyclic gamboostLSS requires one global mstop",
        ));
    }

    let binding_started = trace.then(Instant::now);
    let blueprints = parse_lss_learner_blueprints(py, learner_specs)?;
    let y_owned = y.as_array().to_owned();
    let weights_owned = weights.as_array().to_owned();
    let family = make_lss_family_checked(family_name, blueprints.len(), y_owned.view())?;
    let stabilization =
        mboostrs_core::lss::Stabilization::parse(stabilization).map_err(mboostrs_error_to_pyerr)?;
    let ctrl = BoostControl::new(mstop, nu)
        .map_err(mboostrs_error_to_pyerr)?
        .with_trace(trace);
    let mut ctrl = apply_step(ctrl, step, step_iterations)?;
    ctrl.stabilization = stabilization;
    ctrl.nu_per_param = nu_per_param;
    ctrl = ctrl
        .with_response_scale(parse_response_scale(response_scale.as_ref())?)
        .map_err(mboostrs_error_to_pyerr)?;
    if let Some(ref values) = mstop_per_param {
        if values.len() != blueprints.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "mstop_per_param must have one entry per parameter",
            ));
        }
    }

    let preparation_timer = trace.then(PreparationTimer::start);
    let (mut per_param, basis_per_param) =
        prepare_lss_blueprints(&blueprints, weights_owned.view())
            .map_err(mboostrs_error_to_pyerr)?;
    let preparation_timing = preparation_timer.map(PreparationTimer::finish);
    let binding_input_ns = binding_started.map(|started| started.elapsed().as_nanos() as u64);
    let path = py
        .allow_threads(|| {
            if method == "noncyclic" {
                try_fit_lss_noncyclic(
                    &mut per_param,
                    y_owned.view(),
                    weights_owned.view(),
                    family.as_ref(),
                    &ctrl,
                    offsets.as_deref(),
                )
            } else {
                try_fit_lss(
                    &mut per_param,
                    y_owned.view(),
                    weights_owned.view(),
                    family.as_ref(),
                    &ctrl,
                    mstop_per_param.as_deref(),
                    offsets.as_deref(),
                )
            }
        })
        .map_err(mboostrs_error_to_pyerr)?;

    let binding_output_started = trace.then(Instant::now);
    let results = path
        .per_param
        .iter()
        .map(|parameter_path| {
            (
                parameter_path.offset,
                parameter_path.selected.clone(),
                parameter_path.coefs.clone(),
                parameter_path.fitted.to_vec(),
            )
        })
        .collect();
    let nus = mboostrs_core::lss::resolve_nu_per_param(&ctrl, per_param.len())
        .map_err(mboostrs_error_to_pyerr)?;
    let stores = per_param
        .into_iter()
        .zip(basis_per_param)
        .zip(nus)
        .map(|((learners, basis_info), nu)| {
            Py::new(
                py,
                LearnerStore {
                    learners,
                    nu,
                    basis_info,
                },
            )
        })
        .collect::<PyResult<Vec<_>>>()?;
    let mut report = path.execution_report();
    attach_frontend_timings(
        &mut report,
        preparation_timing,
        binding_input_ns,
        binding_output_started,
    );
    let execution_report = fit_report_to_py(py, &report)?;
    let param_risk = path.per_param.iter().map(|p| p.risk.clone()).collect();
    Ok((
        results,
        path.risk,
        path.selection_order,
        stores,
        execution_report,
        param_risk,
    ))
}

/// Rebuild the prediction stores for a serialized mixed gamboostLSS model
/// without rerunning the boosting path.
#[pyfunction(signature = (learner_specs, weights, nu, nu_per_param=None))]
fn prepare_gamboostlss_stores(
    py: Python<'_>,
    learner_specs: Vec<Vec<PyObject>>,
    weights: PyReadonlyArray1<'_, f64>,
    nu: f64,
    nu_per_param: Option<Vec<f64>>,
) -> PyResult<Vec<Py<LearnerStore>>> {
    if !nu.is_finite() || nu <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "nu must be finite and positive",
        ));
    }
    let blueprints = parse_lss_learner_blueprints(py, learner_specs)?;
    let mut ctrl = BoostControl::new(1, nu).map_err(mboostrs_error_to_pyerr)?;
    ctrl.nu_per_param = nu_per_param;
    let nus = mboostrs_core::lss::resolve_nu_per_param(&ctrl, blueprints.len())
        .map_err(mboostrs_error_to_pyerr)?;
    let weights = weights.as_array().to_owned();
    let (per_param, basis_per_param) =
        prepare_lss_blueprints(&blueprints, weights.view()).map_err(mboostrs_error_to_pyerr)?;
    per_param
        .into_iter()
        .zip(basis_per_param)
        .zip(nus)
        .map(|((learners, basis_info), nu)| {
            Py::new(
                py,
                LearnerStore {
                    learners,
                    nu,
                    basis_info,
                },
            )
        })
        .collect()
}

/// Cross-validated risk for heterogeneous `gamboostLSS` learner sets. Every
/// fold prepares the same blueprints with its effective in-bag weights, while
/// spline designs and penalties remain cached inside the blueprint.
#[pyfunction(signature = (learner_specs, y, weights, mstop, nu, family_name, folds, grid, judge=None, stabilization="none", nu_per_param=None, step="fixed", step_iterations=4, response_scale=None))]
#[allow(clippy::too_many_arguments)]
fn cvrisk_gamboostlss_mixed(
    py: Python<'_>,
    learner_specs: Vec<Vec<PyObject>>,
    y: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    mstop: u32,
    nu: f64,
    family_name: &str,
    folds: PyReadonlyArray2<'_, f64>,
    grid: Vec<Vec<usize>>,
    judge: Option<PyReadonlyArray2<'_, bool>>,
    stabilization: &str,
    nu_per_param: Option<Vec<f64>>,
    step: &str,
    step_iterations: u32,
    response_scale: Option<Bound<'_, PyAny>>,
) -> PyResult<Vec<Vec<f64>>> {
    use mboostrs_core::lss::try_cvrisk_lss_with;

    let blueprints = parse_lss_learner_blueprints(py, learner_specs)?;
    let y_owned = y.as_array().to_owned();
    let weights_owned = weights.as_array().to_owned();
    let folds_owned = folds.as_array().to_owned();
    let judge_arr = judge_mask_from_readonly(judge, y_owned.len())?;
    let family = make_lss_family_checked(family_name, blueprints.len(), y_owned.view())?;
    let ctrl = BoostControl::new(mstop, nu).map_err(mboostrs_error_to_pyerr)?;
    let mut ctrl = apply_step(ctrl, step, step_iterations)?;
    ctrl.stabilization =
        mboostrs_core::lss::Stabilization::parse(stabilization).map_err(mboostrs_error_to_pyerr)?;
    ctrl.nu_per_param = nu_per_param;
    ctrl = ctrl
        .with_response_scale(parse_response_scale(response_scale.as_ref())?)
        .map_err(mboostrs_error_to_pyerr)?;

    let risk = py
        .allow_threads(|| {
            try_cvrisk_lss_with(
                |fold_weights| {
                    prepare_lss_blueprints(&blueprints, fold_weights).map(|(learners, _)| learners)
                },
                y_owned.view(),
                weights_owned.view(),
                family.as_ref(),
                &ctrl,
                folds_owned.view(),
                judge_arr.as_ref(),
                &grid,
            )
        })
        .map_err(mboostrs_error_to_pyerr)?;
    Ok(risk.outer_iter().map(|row| row.to_vec()).collect())
}

/// The doubling search of `tune_mstop` for a distributional specification.
///
/// `ends[k]` and `grids[k]` describe grid `k`: its largest iteration and the
/// diagonal step counts whose aggregated risk it compares. Every fold is
/// fitted once and its path is resumed from one grid to the next. Returns
/// `(risk, curve, grid, best, at_boundary, ends)`: the per-fold risk on the
/// final grid, the aggregated curve on it, the final grid's step counts, the
/// optimum, whether it lies in the boundary band, and the grid ends evaluated.
#[pyfunction(signature = (learner_specs, y, weights, nu, family_name, folds, ends, grids, boundary_share, judge=None, stabilization="none", nu_per_param=None, step="fixed", step_iterations=4, response_scale=None, method="cyclic"))]
#[allow(clippy::too_many_arguments)]
fn search_gamboostlss_grid(
    py: Python<'_>,
    learner_specs: Vec<Vec<PyObject>>,
    y: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    nu: f64,
    family_name: &str,
    folds: PyReadonlyArray2<'_, f64>,
    ends: Vec<usize>,
    grids: Vec<Vec<usize>>,
    boundary_share: f64,
    judge: Option<PyReadonlyArray2<'_, bool>>,
    stabilization: &str,
    nu_per_param: Option<Vec<f64>>,
    step: &str,
    step_iterations: u32,
    response_scale: Option<Bound<'_, PyAny>>,
    method: &str,
) -> PyResult<(Vec<Vec<f64>>, Vec<f64>, Vec<usize>, usize, bool, Vec<usize>)> {
    use mboostrs_core::grid_search::GridStep;
    use mboostrs_core::lss::try_cvrisk_lss_search;

    if method != "cyclic" && method != "noncyclic" {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "method must be 'cyclic' or 'noncyclic'",
        ));
    }

    if ends.is_empty() || ends.len() != grids.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "the search needs one grid per grid end; got {} ends and {} grids",
            ends.len(),
            grids.len()
        )));
    }
    let max_end = *ends.iter().max().expect("ends is non-empty");
    let blueprints = parse_lss_learner_blueprints(py, learner_specs)?;
    let y_owned = y.as_array().to_owned();
    let weights_owned = weights.as_array().to_owned();
    let folds_owned = folds.as_array().to_owned();
    let judge_arr = judge_mask_from_readonly(judge, y_owned.len())?;
    let family = make_lss_family_checked(family_name, blueprints.len(), y_owned.view())?;
    let ctrl = BoostControl::new(max_end as u32, nu).map_err(mboostrs_error_to_pyerr)?;
    let mut ctrl = apply_step(ctrl, step, step_iterations)?;
    ctrl.stabilization =
        mboostrs_core::lss::Stabilization::parse(stabilization).map_err(mboostrs_error_to_pyerr)?;
    ctrl.nu_per_param = nu_per_param;
    ctrl = ctrl
        .with_response_scale(parse_response_scale(response_scale.as_ref())?)
        .map_err(mboostrs_error_to_pyerr)?;
    let schedule: Vec<GridStep> = ends
        .iter()
        .zip(grids)
        .map(|(&end, grid)| GridStep { end, grid })
        .collect();

    let search = py
        .allow_threads(|| {
            try_cvrisk_lss_search(
                |fold_weights| {
                    prepare_lss_blueprints(&blueprints, fold_weights).map(|(learners, _)| learners)
                },
                y_owned.view(),
                weights_owned.view(),
                family.as_ref(),
                &ctrl,
                method == "noncyclic",
                folds_owned.view(),
                judge_arr.as_ref(),
                &schedule,
                boundary_share,
            )
        })
        .map_err(mboostrs_error_to_pyerr)?;
    Ok((
        search.risk,
        search.curve,
        search.grid,
        search.best,
        search.at_boundary,
        search.ends,
    ))
}

/// The doubling search of `tune_mstop` where Python scores each fold.
///
/// The search logic runs in the core (`mboostrs_core::grid_search::search_grid`).
/// The folds' models live in Python, so the core asks for them through
/// `advance(end)`: a callable that grows every fold's model to `end` steps and
/// returns, per fold, the scores at the steps after the previous call's `end`
/// (step 0 on the first call). `ends` and `grids` describe the schedule as in
/// `search_gamboostlss_grid`. Returns the same tuple as that function.
#[pyfunction(signature = (advance, ends, grids, boundary_share))]
fn search_fold_scores(
    py: Python<'_>,
    advance: PyObject,
    ends: Vec<usize>,
    grids: Vec<Vec<usize>>,
    boundary_share: f64,
) -> PyResult<(Vec<Vec<f64>>, Vec<f64>, Vec<usize>, usize, bool, Vec<usize>)> {
    use mboostrs_core::grid_search::{search_grid, FoldRisks, GridStep};

    struct PyFolds<'py> {
        py: Python<'py>,
        callback: PyObject,
        error: Option<PyErr>,
    }

    impl FoldRisks for PyFolds<'_> {
        fn advance(&mut self, end: usize) -> Result<Vec<Vec<f64>>, mboostrs_core::MboostrsError> {
            let outcome = self
                .callback
                .call1(self.py, (end,))
                .and_then(|rows| rows.extract::<Vec<Vec<f64>>>(self.py));
            match outcome {
                Ok(rows) => Ok(rows),
                Err(err) => {
                    self.error = Some(err);
                    Err(mboostrs_core::MboostrsError::Solver(
                        "the fold scorer raised an error".to_string(),
                    ))
                }
            }
        }
    }

    if ends.is_empty() || ends.len() != grids.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "the search needs one grid per grid end; got {} ends and {} grids",
            ends.len(),
            grids.len()
        )));
    }
    let schedule: Vec<GridStep> = ends
        .iter()
        .zip(grids)
        .map(|(&end, grid)| GridStep { end, grid })
        .collect();
    let mut folds = PyFolds {
        py,
        callback: advance,
        error: None,
    };
    match search_grid(&mut folds, &schedule, boundary_share) {
        Ok(search) => Ok((
            search.risk,
            search.curve,
            search.grid,
            search.best,
            search.at_boundary,
            search.ends,
        )),
        Err(err) => Err(folds
            .error
            .take()
            .unwrap_or_else(|| mboostrs_error_to_pyerr(err))),
    }
}

/// Per-subsample combined learner sequences for heterogeneous non-cyclic
/// `gamboostLSS`, used by stability selection.
///
/// `q` is stability selection's cap on distinct components: a fold stops as
/// soon as it has selected that many, because nothing it selects afterwards
/// reaches the reported frequencies. `None` runs every fold to `mstop`.
#[pyfunction(signature = (learner_specs, y, weights, mstop, nu, family_name, folds, q=None, stabilization="none", nu_per_param=None, step="fixed", step_iterations=4, response_scale=None))]
#[allow(clippy::too_many_arguments)]
fn stabsel_selected_gamboostlss_mixed(
    py: Python<'_>,
    learner_specs: Vec<Vec<PyObject>>,
    y: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    mstop: u32,
    nu: f64,
    family_name: &str,
    folds: PyReadonlyArray2<'_, f64>,
    q: Option<usize>,
    stabilization: &str,
    nu_per_param: Option<Vec<f64>>,
    step: &str,
    step_iterations: u32,
    response_scale: Option<Bound<'_, PyAny>>,
) -> PyResult<Vec<Vec<usize>>> {
    use mboostrs_core::lss::try_fit_lss_noncyclic_capped;
    use rayon::prelude::*;

    let blueprints = parse_lss_learner_blueprints(py, learner_specs)?;
    let y_owned = y.as_array().to_owned();
    let weights_owned = weights.as_array().to_owned();
    let folds_owned = folds.as_array().to_owned();
    if folds_owned.nrows() != weights_owned.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "folds have {} rows but weights have length {}",
            folds_owned.nrows(),
            weights_owned.len()
        )));
    }
    let family = make_lss_family_checked(family_name, blueprints.len(), y_owned.view())?;
    let ctrl = BoostControl::new(mstop, nu).map_err(mboostrs_error_to_pyerr)?;
    let mut ctrl = apply_step(ctrl, step, step_iterations)?;
    ctrl.stabilization =
        mboostrs_core::lss::Stabilization::parse(stabilization).map_err(mboostrs_error_to_pyerr)?;
    ctrl.nu_per_param = nu_per_param;
    ctrl = ctrl
        .with_response_scale(parse_response_scale(response_scale.as_ref())?)
        .map_err(mboostrs_error_to_pyerr)?;
    let mut parameter_offsets = Vec::with_capacity(blueprints.len());
    let mut total = 0usize;
    for parameter_blueprints in &blueprints {
        parameter_offsets.push(total);
        total += parameter_blueprints.len();
    }

    py.allow_threads(|| {
        (0..folds_owned.ncols())
            .into_par_iter()
            .map(|fold| {
                let fold_weights = ndarray::Array1::from_iter(
                    weights_owned
                        .iter()
                        .enumerate()
                        .map(|(row, weight)| weight * folds_owned[[row, fold]]),
                );
                let (mut per_param, _) = prepare_lss_blueprints(&blueprints, fold_weights.view())?;
                let path = try_fit_lss_noncyclic_capped(
                    &mut per_param,
                    y_owned.view(),
                    fold_weights.view(),
                    family.as_ref(),
                    &ctrl,
                    None,
                    q,
                )?;
                let mut cursors = vec![0usize; blueprints.len()];
                let mut sequence = Vec::with_capacity(path.selection_order.len());
                for &parameter in &path.selection_order {
                    let learner = path.per_param[parameter].selected[cursors[parameter]];
                    cursors[parameter] += 1;
                    sequence.push(parameter_offsets[parameter] + learner);
                }
                Ok::<_, MboostrsError>(sequence)
            })
            .collect::<Result<Vec<_>, _>>()
    })
    .map_err(mboostrs_error_to_pyerr)
}

/// Wrap each learner as a Kronecker product with `I_{K-1}` for Multinomial.
///
/// Reshapes `ScalarBols` and `BlockBols` into `KroneckerBols` with
/// `x2 = I_{K-1}`; `KroneckerBols` (already-wrapped bbs) pass through. Only
/// `bols` and `bbs` are valid Multinomial base-learners, so anything else is
/// refused. `w_obs` is the observation-level weight vector used to build the
/// per-learner gram; its first `n_obs` entries must hold the observation
/// weights (the CV preparer passes the flattened fold weights, whose leading
/// `n_obs` entries are the per-observation weights because the fold matrix
/// repeats each row `K-1` times).
fn wrap_kronecker_learners(
    learners: Vec<PreparedLearner>,
    basis_info: Option<&[BasisInfo]>,
    w_obs: &ndarray::Array1<f64>,
    n_classes: usize,
) -> PyResult<Vec<PreparedLearner>> {
    let n_resp = n_classes - 1;
    let x2 = ndarray::Array2::eye(n_resp);
    let mut kron_learners = Vec::with_capacity(learners.len());
    for (index, learner) in learners.into_iter().enumerate() {
        // A `brandom` (or a categorical `bols`, lowered to one) is a block
        // learner over level indicators; its replay recipe comes from the
        // blueprint's basis information.
        let levels_rebuild = match basis_info.and_then(|infos| infos.get(index)) {
            Some(BasisInfo::Brandom {
                level_codes,
                centering,
                by,
            }) => {
                if by.is_some() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Learner '{}': a Multinomial model does not support brandom with by=",
                        learner.name()
                    )));
                }
                Some(mboostrs_core::KroneckerRebuild::Levels {
                    level_codes: level_codes.clone(),
                    centering: centering.clone(),
                })
            }
            _ => None,
        };
        match learner {
            mboostrs_core::PreparedLearner::ScalarBols {
                design,
                penalty: _,
                name,
                centering,
                ..
            } => {
                let n_obs = design.len();
                let (x1, pen, rebuild) = if centering != 0.0 {
                    // bols(x, intercept=True): reconstruct original x and build
                    // 2-column design [1, x] matching R's bols(x, df=2).
                    let mut x1 = ndarray::Array2::ones((n_obs, 2));
                    for i in 0..n_obs {
                        x1[[i, 1]] = design[i] + centering;
                    }
                    (
                        x1,
                        ndarray::Array2::zeros((2, 2)),
                        mboostrs_core::KroneckerRebuild::BolsIntercept,
                    )
                } else {
                    let x1 = design.into_shape_with_order((n_obs, 1)).unwrap();
                    (
                        x1,
                        ndarray::Array2::zeros((1, 1)),
                        mboostrs_core::KroneckerRebuild::BolsNoIntercept,
                    )
                };
                let mut learner =
                    prepare_kronecker_bols(x1, w_obs.view(), pen, x2.clone(), false, &name);
                learner.set_x1_rebuild(rebuild);
                kron_learners.push(learner);
            }
            mboostrs_core::PreparedLearner::BlockBols {
                design,
                penalty,
                name,
                ..
            } => {
                let mut learner =
                    prepare_kronecker_bols(design, w_obs.view(), penalty, x2.clone(), false, &name);
                if let Some(rebuild) = levels_rebuild {
                    learner.set_x1_rebuild(rebuild);
                }
                kron_learners.push(learner);
            }
            // bbs learners are already built as KroneckerBols above.
            k @ mboostrs_core::PreparedLearner::KroneckerBols { .. } => {
                kron_learners.push(k);
            }
            // Materialize the compact factored state (left marginal + groups +
            // block-diagonal penalty) back into the full row-wise tensor so the
            // Multinomial Kronecker layout can consume it.
            l @ mboostrs_core::PreparedLearner::FactoredBlockBols { .. } => {
                let name = l.name().to_string();
                let design = l
                    .design_matrix()
                    .expect("FactoredBlockBols design reconstruction");
                let penalty = l
                    .penalty_matrix()
                    .expect("FactoredBlockBols penalty reconstruction");
                kron_learners.push(prepare_kronecker_bols(
                    design,
                    w_obs.view(),
                    penalty,
                    x2.clone(),
                    false,
                    &name,
                ));
            }
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Learner '{}' not supported for Multinomial family",
                    other.name()
                )));
            }
        }
    }
    Ok(kron_learners)
}

/// Expand per-observation weights to the flattened Multinomial layout
/// `w_exp[i * n_resp + k] = w[i]` for `k in 0..n_resp`. Used by the fit path
/// and, on the CV side, by the fold-matrix flattening.
fn expand_multinomial_weights(
    w_obs: &ndarray::Array1<f64>,
    n_classes: usize,
) -> ndarray::Array1<f64> {
    let n_resp = n_classes - 1;
    let n_obs = w_obs.len();
    let mut w_exp = ndarray::Array1::zeros(n_obs * n_resp);
    for i in 0..n_obs {
        for k in 0..n_resp {
            w_exp[i * n_resp + k] = w_obs[i];
        }
    }
    w_exp
}

/// Expand an observation-row-major matrix (folds or judge mask) to the
/// flattened Multinomial layout: each observation row is repeated `n_resp`
/// consecutive rows, preserving column index. Mirrors
/// `expand_multinomial_weights` in the row dimension.
fn expand_fold_matrix(m: &ndarray::Array2<f64>, n_resp: usize) -> ndarray::Array2<f64> {
    let n_obs = m.nrows();
    let n_cols = m.ncols();
    let mut out = ndarray::Array2::zeros((n_obs * n_resp, n_cols));
    for i in 0..n_obs {
        for k in 0..n_resp {
            for col in 0..n_cols {
                out[[i * n_resp + k, col]] = m[[i, col]];
            }
        }
    }
    out
}

/// Bool-matrix form of [`expand_fold_matrix`] for the judge mask.
fn expand_bool_matrix(m: &ndarray::Array2<bool>, n_resp: usize) -> ndarray::Array2<bool> {
    let n_obs = m.nrows();
    let n_cols = m.ncols();
    let mut out = ndarray::Array2::from_elem((n_obs * n_resp, n_cols), false);
    for i in 0..n_obs {
        for k in 0..n_resp {
            for col in 0..n_cols {
                out[[i * n_resp + k, col]] = m[[i, col]];
            }
        }
    }
    out
}

/// Wrap each learner as a Kronecker product with `I_{K-1}` for Multinomial,
/// and expand the observation weights to match.
///
/// Shared by both `gamboost` entry points: the response for a K-class
/// multinomial is stacked `(n_obs * (K-1))`, so both the learners and the
/// weights have to be lifted before fitting. Returns the weights unchanged for
/// every other family.
fn apply_multinomial_kronecker(
    learners: Vec<PreparedLearner>,
    basis_info: Option<&[BasisInfo]>,
    w_owned: &ndarray::Array1<f64>,
    multinomial_n_classes: Option<usize>,
) -> PyResult<(Vec<PreparedLearner>, ndarray::Array1<f64>)> {
    match multinomial_n_classes {
        Some(n_classes) => {
            let learners = wrap_kronecker_learners(learners, basis_info, w_owned, n_classes)?;
            let w_expanded = expand_multinomial_weights(w_owned, n_classes);
            Ok((learners, w_expanded))
        }
        None => Ok((learners, w_owned.clone())),
    }
}

/// Fit prepared learners and marshal the result, shared by the compact
/// production entry point and the legacy compatibility entry point.
///
/// `fit_gamboost` reaches here after unpacking ~75 parallel-array arguments;
/// `fit_gamboost_specs` reaches here after parsing one list of self-describing
/// learner records. Everything from the fit call onward is identical, so
/// The Python production layer calls only `fit_gamboost_specs`; the wide
/// `fit_gamboost` symbol remains until the scheduled 0.2 compatibility break.
#[allow(clippy::too_many_arguments)]
fn run_gamboost_fit(
    py: Python<'_>,
    mut learners: Vec<PreparedLearner>,
    y: ndarray::ArrayView1<'_, f64>,
    w_expanded: ndarray::Array1<f64>,
    ctrl: BoostControl,
    family_name: &str,
    family_params: &HashMap<String, f64>,
    offset: OffsetArg,
    nu: f64,
    preparation_timing: Option<PreparationTiming>,
    binding_started: Option<Instant>,
    trace: bool,
    basis_info: Vec<BasisInfo>,
) -> PyResult<GamFitResult> {
    let family_name_owned = family_name.to_string();
    let binding_input_ns = binding_started.map(|started| started.elapsed().as_nanos() as u64);
    let mut path = py
        .allow_threads(|| {
            with_concrete_family!(family_name_owned.as_str(), family_params, |fam| {
                match offset.vector() {
                    Some(vector) => {
                        let offset_owned = ndarray::Array1::from(vector.to_vec());
                        mboostrs_core::engine::try_fit_with_vector_offset(
                            &mut learners,
                            y,
                            w_expanded.view(),
                            &mut fam,
                            &ctrl,
                            offset_owned.view(),
                            false,
                        )
                    }
                    None => try_fit_with(
                        &mut learners,
                        y,
                        w_expanded.view(),
                        &mut fam,
                        &ctrl,
                        offset.scalar(),
                        false,
                    ),
                }
            })
        })?
        .map_err(mboostrs_error_to_pyerr)?;

    let binding_output_started = trace.then(Instant::now);
    reject_diverged(&path.risk, &path.fitted)?;
    let nuisance_vec_path = path.nuisance_rows();
    let coefs_vecs: Vec<Vec<f64>> = path.coefs;
    let offset_out = path.offset;
    let selected = path.selected;
    let risk = path.risk;
    let fitted = path.fitted.into_raw_vec_and_offset().0;
    let nuisance_vec = path.nuisance_vec;
    let residual = path.residual.into_raw_vec_and_offset().0;

    let store = Py::new(
        py,
        LearnerStore {
            learners,
            nu,
            basis_info,
        },
    )?;
    attach_frontend_timings(
        &mut path.report,
        preparation_timing,
        binding_input_ns,
        binding_output_started,
    );
    let execution_report = fit_report_to_py(py, &path.report)?;

    Ok((
        offset_out,
        selected,
        coefs_vecs,
        risk,
        fitted,
        nuisance_vec,
        nuisance_vec_path,
        store,
        execution_report,
        residual,
    ))
}

/// Rebuild a `LearnerStore` from learner specifications, without fitting.
///
/// A fitted model's store cannot be pickled — it owns Gram factorizations and
/// work buffers — so a deserialized model used to lose the ability to predict
/// through the core, and had to rebuild designs Python-side from thirty cached
/// per-kind arrays. Given the specifications and the training weights, this
/// reconstructs the same store the fit produced, so a restored model predicts,
/// and computes AIC and hat values, exactly as the live one does.
#[pyfunction]
fn prepare_learner_store(
    py: Python<'_>,
    learner_specs: Vec<PyObject>,
    weights: PyReadonlyArray1<'_, f64>,
    nu: f64,
) -> PyResult<Py<LearnerStore>> {
    let w = weights.as_array().to_owned();
    let blueprints = parse_learner_blueprint_list(py, learner_specs)?;
    let mut learners = Vec::with_capacity(blueprints.len());
    let mut basis_info = Vec::with_capacity(blueprints.len());
    for blueprint in &blueprints {
        let (learner, basis) = blueprint
            .prepare(w.view())
            .map_err(mboostrs_error_to_pyerr)?;
        learners.push(learner);
        basis_info.push(basis);
    }
    Py::new(
        py,
        LearnerStore {
            learners,
            nu,
            basis_info,
        },
    )
}

#[pyfunction(signature = (learner_specs, y, weights, mstop, nu, family_name, family_params, offset=None, multinomial_n_classes=None, risk="inbag", trace=false, component_fraction=None, subsample_seed=0, step="fixed", step_iterations=4))]
#[allow(clippy::too_many_arguments)]
fn fit_gamboost_specs(
    py: Python<'_>,
    learner_specs: Vec<PyObject>,
    y: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    mstop: u32,
    nu: f64,
    family_name: &str,
    family_params: HashMap<String, f64>,
    offset: Option<Bound<'_, PyAny>>,
    multinomial_n_classes: Option<usize>,
    risk: &str,
    trace: bool,
    component_fraction: Option<f64>,
    subsample_seed: u64,
    step: &str,
    step_iterations: u32,
) -> PyResult<GamFitResult> {
    let offset = extract_offset_arg(offset.as_ref())?;
    let binding_started = trace.then(Instant::now);
    let risk_kind = match risk {
        "inbag" => RiskKind::Inbag,
        "oobag" => RiskKind::Oobag,
        "none" => RiskKind::None,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "risk must be 'inbag', 'oobag', or 'none', got {risk:?}"
            )))
        }
    };

    let y = y.as_array();

    make_family(family_name, &family_params)
        .map_err(pyo3::exceptions::PyValueError::new_err)?
        .check_y(y)
        .map_err(mboostrs_error_to_pyerr)?;

    let w_owned = weights.as_array().to_owned();
    let ctrl = BoostControl::new(mstop, nu)
        .map_err(mboostrs_error_to_pyerr)?
        .with_risk(risk_kind)
        .with_trace(trace);
    let ctrl = apply_step(ctrl, step, step_iterations)?;
    let ctrl = apply_subsampling(ctrl, component_fraction, subsample_seed)?;

    // An empty `learner_specs` is the offset-only model: the engine runs no
    // boosting step and the path is the family offset.
    let preparation_timer = trace.then(PreparationTimer::start);
    let blueprints = parse_learner_blueprint_list(py, learner_specs)?;
    // `prepare` returns the new-data replay recipe alongside the prepared
    // learner. Keeping it here is what lets prediction go back through the same
    // boundary the fit came in on: one `BasisInfo` per learner, in formula
    // order, for every learner kind. The array path throws this away and has
    // Python rebuild a partial copy through `set_basis_info` instead.
    //
    // Preparing a learner (design matrix, penalty factorization, df -> lambda
    // solve) is independent per learner and touches no Python object, so it
    // runs on the rayon pool with the GIL released. On a wide model this is
    // most of a short fit's wall time -- 80 learners on 437 rows spend ~0.15 s
    // here against ~0.02 s for 40 boosting iterations -- and it is what a
    // per-fold caller (stabsel's subsample refits) pays on every fold.
    // `par_iter().collect()` preserves formula order.
    let prepared = py
        .allow_threads(|| {
            use rayon::prelude::*;
            // Consumed: a one-shot fit moves each template design into its
            // learner instead of holding the template and a clone of it.
            blueprints
                .into_par_iter()
                .map(|blueprint| blueprint.into_prepared(w_owned.view()))
                .collect::<Result<Vec<(PreparedLearner, BasisInfo)>, _>>()
        })
        .map_err(mboostrs_error_to_pyerr)?;
    let (learners, mut basis_info): (Vec<PreparedLearner>, Vec<BasisInfo>) =
        prepared.into_iter().unzip();
    let (learners, w_expanded) =
        apply_multinomial_kronecker(learners, Some(&basis_info), &w_owned, multinomial_n_classes)?;
    // The Kronecker lift replaces the learner list wholesale, so the per-learner
    // replay recipes no longer line up with it.
    if multinomial_n_classes.is_some() {
        basis_info.clear();
    }
    let preparation_timing = preparation_timer.map(PreparationTimer::finish);

    run_gamboost_fit(
        py,
        learners,
        y,
        w_expanded,
        ctrl,
        family_name,
        &family_params,
        offset,
        nu,
        preparation_timing,
        binding_started,
        trace,
        basis_info,
    )
}

/// Warm-continue a spec-path gamboost fit from the state a shorter fit left.
///
/// Mirrors `fit_gamboost_specs` in learner preparation (same blueprints,
/// same Kronecker lift), then resumes from the retained fitted/offset/risk/
/// selected/coefs/`residual` via `try_fit_continue` instead of refitting from
/// zero. The retained nuisance rows keep every path position aligned with
/// risk, including slices before the resume point. `increments` are not
/// supplied; path-dependent families are refused by the caller.
#[pyfunction(signature = (learner_specs, y, weights, mstop, nu, family_name, family_params, warm_fitted, warm_offset, warm_risk, warm_selected, warm_coefs, warm_residual, step="fixed", step_iterations=4, warm_nuisance=None))]
#[allow(clippy::too_many_arguments)]
fn fit_gamboost_continue_specs(
    py: Python<'_>,
    learner_specs: Vec<PyObject>,
    y: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    mstop: u32,
    nu: f64,
    family_name: &str,
    family_params: HashMap<String, f64>,
    warm_fitted: PyReadonlyArray1<'_, f64>,
    warm_offset: f64,
    warm_risk: Vec<f64>,
    warm_selected: Vec<usize>,
    warm_coefs: Vec<Vec<f64>>,
    warm_residual: PyReadonlyArray1<'_, f64>,
    step: &str,
    step_iterations: u32,
    warm_nuisance: Option<Vec<Vec<f64>>>,
) -> PyResult<GamFitResult> {
    let risk_kind = RiskKind::Inbag;
    let y_owned: ndarray::Array1<f64> = y.as_array().to_owned();
    let y = y_owned.view();
    make_family(family_name, &family_params)
        .map_err(pyo3::exceptions::PyValueError::new_err)?
        .check_y(y)
        .map_err(mboostrs_error_to_pyerr)?;

    let w_owned = weights.as_array().to_owned();
    let ctrl = BoostControl::new(mstop, nu)
        .map_err(mboostrs_error_to_pyerr)?
        .with_risk(risk_kind);
    let ctrl = apply_step(ctrl, step, step_iterations)?;

    let blueprints = parse_learner_blueprint_list(py, learner_specs)?;
    let prepared: Result<Vec<(PreparedLearner, BasisInfo)>, _> = blueprints
        .iter()
        .map(|blueprint| blueprint.prepare(w_owned.view()))
        .collect();
    let prepared = prepared.map_err(mboostrs_error_to_pyerr)?;
    let (mut learners, basis_info): (Vec<PreparedLearner>, Vec<BasisInfo>) =
        prepared.into_iter().unzip();
    let (learners_owned, w_expanded) =
        apply_multinomial_kronecker(learners, Some(&basis_info), &w_owned, None)?;
    learners = learners_owned;

    let fitted = warm_fitted.as_array().to_owned();
    let residual = warm_residual.as_array().to_owned();
    let warm_nuisance = warm_nuisance.unwrap_or_default();
    if !warm_nuisance.is_empty() && warm_nuisance.len() != warm_risk.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "warm nuisance rows must align with the retained risk path",
        ));
    }
    let scalar_nuisance: Vec<Option<f64>> = warm_nuisance
        .iter()
        .map(|row| (row.len() == 1).then(|| row[0]))
        .collect();
    let warm = WarmStart {
        fitted: fitted.view(),
        offset: warm_offset,
        risk: &warm_risk,
        selected: &warm_selected,
        coefs: &warm_coefs,
        increments: &[],
        nuisance_path: &scalar_nuisance,
        nuisance_vec_path: &warm_nuisance,
        nuisance_at_resume: scalar_nuisance.last().copied().flatten(),
        nuisance_vec_at_resume: warm_nuisance.last().map(Vec::as_slice).unwrap_or(&[]),
        residual: residual.view(),
    };

    let family_name_owned = family_name.to_string();
    let mut path = py
        .allow_threads(|| {
            with_concrete_family!(family_name_owned.as_str(), &family_params, |fam| {
                try_fit_continue(&mut learners, y, w_expanded.view(), &mut fam, &ctrl, &warm)
            })
        })?
        .map_err(mboostrs_error_to_pyerr)?;

    // Marshal the same 10-tuple `run_gamboost_fit` produces. `learners` is
    // still owned here, so the grown model carries a real prediction store.
    reject_diverged(&path.risk, &path.fitted)?;
    let nuisance_vec_path = path.nuisance_rows();
    let coefs_vecs = std::mem::take(&mut path.coefs);
    let offset_out = path.offset;
    let selected = path.selected.clone();
    let risk = path.risk.clone();
    let fitted_out = path.fitted.into_raw_vec_and_offset().0;
    let nuisance_vec = path.nuisance_vec.clone();
    let residual_out = path.residual.into_raw_vec_and_offset().0;

    let store = Py::new(
        py,
        LearnerStore {
            learners,
            nu,
            basis_info,
        },
    )?;
    let execution_report = fit_report_to_py(py, &path.report)?;
    Ok((
        offset_out,
        selected,
        coefs_vecs,
        risk,
        fitted_out,
        nuisance_vec,
        nuisance_vec_path,
        store,
        execution_report,
        residual_out,
    ))
}

#[pyfunction(signature = (scalar_x, scalar_names, scalar_lambda, scalar_center, scalar_df, bbs_x, bbs_names, bbs_knots, bbs_degree, bbs_differences, bbs_df, bbs_lambda, bbs_center, bbs_cyclic, bbs_boundary_knots_lo, bbs_boundary_knots_hi, bbs_deriv, bmono_x, bmono_names, bmono_knots, bmono_degree, bmono_differences, bmono_df, bmono_constraint, bmono_type, bmono_lambda2, bmono_niter, bmono_center, btree_x, btree_names, btree_minbucket, btree_minsplit, btree_maxdepth, bmrf_levels, bmrf_names, bmrf_k, bmrf_df, bmrf_lambda, bmrf_center, bspatial_x1, bspatial_x2, bspatial_names, bspatial_knots, bspatial_degree, bspatial_differences, bspatial_df, bspatial_lambda, bspatial_center, y, weights, mstop, nu, family_name, family_params, offset=None, scalar_by=None, bbs_by=None, bmono_by=None, btree_by=None, bmrf_by=None, bspatial_by=None, bkernel_x=None, bkernel_names=None, bkernel_bandwidth=None, bkernel_df=None, bkernel_kernel=None, bkernel_center=None, bkernel_by=None, brandom_x=None, brandom_names=None, brandom_n_levels=None, brandom_df=None, brandom_lambda=None, brandom_center=None, brandom_by=None, bmono_C=None, buser_X=None, buser_K=None, buser_names=None, buser_df=None, buser_lambda=None, buser_center=None, buser_by=None, multinomial_n_classes=None, risk="inbag", trace=false, component_fraction=None, subsample_seed=0, step="fixed", step_iterations=4))]
#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
#[allow(non_snake_case)]
fn fit_gamboost(
    py: Python<'_>,
    scalar_x: Vec<PyReadonlyArray1<'_, f64>>,
    scalar_names: Vec<String>,
    scalar_lambda: Vec<f64>,
    scalar_center: Vec<bool>,
    scalar_df: Vec<f64>,
    bbs_x: Vec<PyReadonlyArray1<'_, f64>>,
    bbs_names: Vec<String>,
    bbs_knots: Vec<usize>,
    bbs_degree: Vec<usize>,
    bbs_differences: Vec<usize>,
    bbs_df: Vec<f64>,
    bbs_lambda: Vec<f64>,
    bbs_center: Vec<bool>,
    bbs_cyclic: Vec<bool>,
    bbs_boundary_knots_lo: Vec<f64>,
    bbs_boundary_knots_hi: Vec<f64>,
    bbs_deriv: Vec<usize>,
    bmono_x: Vec<PyReadonlyArray1<'_, f64>>,
    bmono_names: Vec<String>,
    bmono_knots: Vec<usize>,
    bmono_degree: Vec<usize>,
    bmono_differences: Vec<usize>,
    bmono_df: Vec<f64>,
    bmono_constraint: Vec<String>,
    bmono_type: Vec<String>,
    bmono_lambda2: Vec<f64>,
    bmono_niter: Vec<usize>,
    bmono_center: Vec<bool>,
    btree_x: Vec<PyReadonlyArray2<'_, f64>>,
    btree_names: Vec<String>,
    btree_minbucket: Vec<usize>,
    btree_minsplit: Vec<usize>,
    btree_maxdepth: Vec<usize>,
    bmrf_levels: Vec<Vec<usize>>,
    bmrf_names: Vec<String>,
    bmrf_k: Vec<PyReadonlyArray2<'_, f64>>,
    bmrf_df: Vec<f64>,
    bmrf_lambda: Vec<f64>,
    bmrf_center: Vec<bool>,
    bspatial_x1: Vec<PyReadonlyArray1<'_, f64>>,
    bspatial_x2: Vec<PyReadonlyArray1<'_, f64>>,
    bspatial_names: Vec<String>,
    bspatial_knots: Vec<usize>,
    bspatial_degree: Vec<usize>,
    bspatial_differences: Vec<usize>,
    bspatial_df: Vec<f64>,
    bspatial_lambda: Vec<f64>,
    bspatial_center: Vec<bool>,
    y: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    mstop: u32,
    nu: f64,
    family_name: &str,
    family_params: HashMap<String, f64>,
    offset: Option<Bound<'_, PyAny>>,
    scalar_by: ByVecs,
    bbs_by: ByVecs,
    bmono_by: ByVecs,
    btree_by: ByVecs,
    bmrf_by: ByVecs,
    bspatial_by: ByVecs,
    bkernel_x: Option<Vec<PyReadonlyArray1<'_, f64>>>,
    bkernel_names: Option<Vec<String>>,
    bkernel_bandwidth: Option<Vec<f64>>,
    bkernel_df: Option<Vec<f64>>,
    bkernel_kernel: Option<Vec<String>>,
    bkernel_center: Option<Vec<bool>>,
    bkernel_by: ByVecs,
    brandom_x: Option<Vec<PyReadonlyArray2<'_, f64>>>,
    brandom_names: Option<Vec<String>>,
    brandom_n_levels: Option<Vec<usize>>,
    brandom_df: Option<Vec<f64>>,
    brandom_lambda: Option<Vec<f64>>,
    brandom_center: Option<Vec<bool>>,
    brandom_by: ByVecs,
    bmono_C: Option<Vec<Option<PyReadonlyArray2<'_, f64>>>>,
    buser_X: Option<Vec<PyReadonlyArray2<'_, f64>>>,
    buser_K: Option<Vec<PyReadonlyArray2<'_, f64>>>,
    buser_names: Option<Vec<String>>,
    buser_df: Option<Vec<f64>>,
    buser_lambda: Option<Vec<f64>>,
    buser_center: Option<Vec<bool>>,
    buser_by: ByVecs,
    multinomial_n_classes: Option<usize>,
    risk: &str,
    trace: bool,
    component_fraction: Option<f64>,
    subsample_seed: u64,
    step: &str,
    step_iterations: u32,
) -> PyResult<GamFitResult> {
    let binding_started = trace.then(Instant::now);
    let offset = extract_offset_arg(offset.as_ref())?;
    let risk_kind = match risk {
        "inbag" => RiskKind::Inbag,
        "oobag" => RiskKind::Oobag,
        "none" => RiskKind::None,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "risk must be 'inbag', 'oobag', or 'none', got {risk:?}"
            )))
        }
    };

    let n_scalar = scalar_x.len();
    if scalar_names.len() != n_scalar
        || scalar_lambda.len() != n_scalar
        || scalar_center.len() != n_scalar
        || scalar_df.len() != n_scalar
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "scalar_x/names/lambda/center/df must have the same length",
        ));
    }
    let n_bbs = bbs_x.len();
    if bbs_names.len() != n_bbs
        || bbs_knots.len() != n_bbs
        || bbs_degree.len() != n_bbs
        || bbs_differences.len() != n_bbs
        || bbs_df.len() != n_bbs
        || bbs_lambda.len() != n_bbs
        || bbs_center.len() != n_bbs
        || bbs_cyclic.len() != n_bbs
        || bbs_boundary_knots_lo.len() != n_bbs
        || bbs_boundary_knots_hi.len() != n_bbs
        || bbs_deriv.len() != n_bbs
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bbs arrays must all have the same length",
        ));
    }
    let n_bmono = bmono_x.len();
    if bmono_names.len() != n_bmono
        || bmono_knots.len() != n_bmono
        || bmono_degree.len() != n_bmono
        || bmono_differences.len() != n_bmono
        || bmono_df.len() != n_bmono
        || bmono_constraint.len() != n_bmono
        || bmono_type.len() != n_bmono
        || bmono_lambda2.len() != n_bmono
        || bmono_niter.len() != n_bmono
        || bmono_center.len() != n_bmono
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bmono arrays must all have the same length",
        ));
    }
    let n_btree = btree_x.len();
    if btree_names.len() != n_btree
        || btree_minbucket.len() != n_btree
        || btree_minsplit.len() != n_btree
        || btree_maxdepth.len() != n_btree
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "btree arrays must all have the same length",
        ));
    }
    let n_bmrf = bmrf_levels.len();
    if bmrf_names.len() != n_bmrf
        || bmrf_k.len() != n_bmrf
        || bmrf_df.len() != n_bmrf
        || bmrf_lambda.len() != n_bmrf
        || bmrf_center.len() != n_bmrf
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bmrf arrays must all have the same length",
        ));
    }
    let n_bspatial = bspatial_x1.len();
    if bspatial_names.len() != n_bspatial
        || bspatial_x2.len() != n_bspatial
        || bspatial_knots.len() != n_bspatial
        || bspatial_degree.len() != n_bspatial
        || bspatial_differences.len() != n_bspatial
        || bspatial_df.len() != n_bspatial
        || bspatial_lambda.len() != n_bspatial
        || bspatial_center.len() != n_bspatial
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bspatial arrays must all have the same length",
        ));
    }
    let n_bkernel = bkernel_x.as_ref().map_or(0, |v| v.len());
    if n_bkernel > 0 {
        let bn = bkernel_names.as_ref().map_or(0, |v| v.len());
        let bb = bkernel_bandwidth.as_ref().map_or(0, |v| v.len());
        let bd = bkernel_df.as_ref().map_or(0, |v| v.len());
        let bk = bkernel_kernel.as_ref().map_or(0, |v| v.len());
        let bc = bkernel_center.as_ref().map_or(0, |v| v.len());
        if bn != n_bkernel
            || bb != n_bkernel
            || bd != n_bkernel
            || bk != n_bkernel
            || bc != n_bkernel
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "bkernel arrays must all have the same length",
            ));
        }
    }
    let n_brandom = brandom_x.as_ref().map_or(0, |v| v.len());
    if n_brandom > 0 {
        let bn = brandom_names.as_ref().map_or(0, |v| v.len());
        let bnl = brandom_n_levels.as_ref().map_or(0, |v| v.len());
        let bd = brandom_df.as_ref().map_or(0, |v| v.len());
        let bl = brandom_lambda.as_ref().map_or(0, |v| v.len());
        let bc = brandom_center.as_ref().map_or(0, |v| v.len());
        if bn != n_brandom
            || bnl != n_brandom
            || bd != n_brandom
            || bl != n_brandom
            || bc != n_brandom
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "brandom arrays must all have the same length",
            ));
        }
    }

    let y = y.as_array();

    // Validate the response up front (every family's check_y was otherwise dead
    // code), so invalid y errors clearly instead of producing NaN deep in the fit.
    make_family(family_name, &family_params)
        .map_err(pyo3::exceptions::PyValueError::new_err)?
        .check_y(y)
        .map_err(mboostrs_error_to_pyerr)?;

    let w_owned = weights.as_array().to_owned();

    let ctrl = BoostControl::new(mstop, nu)
        .map_err(mboostrs_error_to_pyerr)?
        .with_risk(risk_kind)
        .with_trace(trace);
    let ctrl = apply_step(ctrl, step, step_iterations)?;
    let ctrl = apply_subsampling(ctrl, component_fraction, subsample_seed)?;

    let scalar_by_vecs: Vec<Option<Vec<f64>>> = scalar_by.unwrap_or_else(|| vec![None; n_scalar]);
    if scalar_by_vecs.len() != n_scalar {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "scalar_by must have the same length as scalar_x",
        ));
    }
    let bbs_by_vecs: Vec<Option<Vec<f64>>> = bbs_by.unwrap_or_else(|| vec![None; n_bbs]);
    if bbs_by_vecs.len() != n_bbs {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bbs_by must have the same length as bbs_x",
        ));
    }
    let bmono_by_vecs: Vec<Option<Vec<f64>>> = bmono_by.unwrap_or_else(|| vec![None; n_bmono]);
    if bmono_by_vecs.len() != n_bmono {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bmono_by must have the same length as bmono_x",
        ));
    }
    let btree_by_vecs: Vec<Option<Vec<f64>>> = btree_by.unwrap_or_else(|| vec![None; n_btree]);
    if btree_by_vecs.len() != n_btree {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "btree_by must have the same length as btree_x",
        ));
    }
    let bmrf_by_vecs: Vec<Option<Vec<f64>>> = bmrf_by.unwrap_or_else(|| vec![None; n_bmrf]);
    if bmrf_by_vecs.len() != n_bmrf {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bmrf_by must have the same length as bmrf_levels",
        ));
    }
    let bspatial_by_vecs: Vec<Option<Vec<f64>>> =
        bspatial_by.unwrap_or_else(|| vec![None; n_bspatial]);
    if bspatial_by_vecs.len() != n_bspatial {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bspatial_by must have the same length as bspatial_x1",
        ));
    }
    let bkernel_by_vecs: Vec<Option<Vec<f64>>> =
        bkernel_by.unwrap_or_else(|| vec![None; n_bkernel]);
    let brandom_by_vecs: Vec<Option<Vec<f64>>> =
        brandom_by.unwrap_or_else(|| vec![None; n_brandom]);
    if brandom_by_vecs.len() != n_brandom {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "brandom_by must have the same length as brandom_x",
        ));
    }
    let n_buser = buser_X.as_ref().map_or(0, |v| v.len());
    if n_buser > 0 {
        let nk = buser_K.as_ref().map_or(0, |v| v.len());
        let nn = buser_names.as_ref().map_or(0, |v| v.len());
        let nd = buser_df.as_ref().map_or(0, |v| v.len());
        let nl = buser_lambda.as_ref().map_or(0, |v| v.len());
        let nc = buser_center.as_ref().map_or(0, |v| v.len());
        if nk != n_buser || nn != n_buser || nd != n_buser || nl != n_buser || nc != n_buser {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "buser arrays must all have the same length",
            ));
        }
    }
    let buser_by_vecs: Vec<Option<Vec<f64>>> = buser_by.unwrap_or_else(|| vec![None; n_buser]);

    let preparation_timer = trace.then(PreparationTimer::start);
    let mut learners: Vec<PreparedLearner> = Vec::with_capacity(
        n_scalar + n_bbs + n_bmono + n_btree + n_bmrf + n_bspatial + n_bkernel + n_brandom,
    );

    for i in 0..n_scalar {
        let view = scalar_x[i].as_array();
        let lambda = if scalar_df[i] >= 0.0 {
            df_to_lambda_scalar(view, w_owned.view(), scalar_df[i], scalar_center[i])
        } else {
            scalar_lambda[i]
        };
        let by_view = scalar_by_vecs[i]
            .as_ref()
            .map(|b| ArrayView1::from(b.as_slice()));
        learners.push(mboostrs_core::learners::prepare_scalar_bols(
            view,
            w_owned.view(),
            lambda,
            scalar_center[i],
            &scalar_names[i],
            by_view,
        ));
    }

    for i in 0..n_bbs {
        let view = bbs_x[i].as_array();
        let lam_opt = if bbs_lambda[i] >= 0.0 {
            Some(bbs_lambda[i])
        } else {
            None
        };
        let by_view = bbs_by_vecs[i]
            .as_ref()
            .map(|b| ArrayView1::from(b.as_slice()));
        if let Some(n_classes) = multinomial_n_classes {
            // Multinomial: build the bbs Kronecker learner directly (raw basis,
            // penalty I + K1, per-class df), matching R's bbs %O% bols(class).
            learners.push(mboostrs_core::learners::prepare_kronecker_bbs(
                view,
                w_owned.view(),
                bbs_knots[i],
                bbs_degree[i],
                bbs_differences[i],
                bbs_df[i],
                n_classes - 1,
                &bbs_names[i],
                bbs_deriv[i],
            ));
            continue;
        }
        learners.push(mboostrs_core::learners::prepare_bbs_with_lambda(
            view,
            w_owned.view(),
            bbs_knots[i],
            bbs_degree[i],
            bbs_differences[i],
            bbs_df[i],
            lam_opt,
            bbs_center[i],
            &bbs_names[i],
            by_view,
            bbs_cyclic[i],
            if bbs_boundary_knots_lo[i].is_nan() || bbs_boundary_knots_hi[i].is_nan() {
                None
            } else {
                Some((bbs_boundary_knots_lo[i], bbs_boundary_knots_hi[i]))
            },
            bbs_deriv[i],
        ));
    }

    for i in 0..n_bmono {
        let view = bmono_x[i].as_array();
        let constraint = match bmono_constraint[i].as_str() {
            "increasing" => mboostrs_core::MonoConstraint::Increasing,
            "decreasing" => mboostrs_core::MonoConstraint::Decreasing,
            "convex" => mboostrs_core::MonoConstraint::Convex,
            "concave" => mboostrs_core::MonoConstraint::Concave,
            "positive" => mboostrs_core::MonoConstraint::Positive,
            "negative" => mboostrs_core::MonoConstraint::Negative,
            "constrained" => mboostrs_core::MonoConstraint::Constrained,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown bmono constraint: {other}"
                )));
            }
        };
        let solver = parse_bmono_solver(&bmono_type[i])?;
        let by_view = bmono_by_vecs[i]
            .as_ref()
            .map(|b| ArrayView1::from(b.as_slice()));
        let constraint_mat = bmono_C
            .as_ref()
            .and_then(|cs| cs[i].as_ref().map(|c| c.as_array()));
        learners.push(mboostrs_core::learners::prepare_bmono(
            view,
            w_owned.view(),
            bmono_knots[i],
            bmono_degree[i],
            bmono_differences[i],
            bmono_df[i],
            constraint,
            solver,
            bmono_lambda2[i],
            bmono_niter[i],
            &bmono_names[i],
            by_view,
            constraint_mat,
            bmono_center[i],
        ));
    }

    for i in 0..n_btree {
        let design = btree_x[i].as_array().to_owned();
        let by_owned = btree_by_vecs[i]
            .as_ref()
            .map(|b| ndarray::Array1::from_vec(b.clone()));
        learners.push(mboostrs_core::learners::prepare_btree(
            design.view(),
            w_owned.view(),
            btree_minbucket[i],
            btree_minsplit[i],
            btree_maxdepth[i],
            by_owned,
            &btree_names[i],
        ));
    }

    for i in 0..n_bmrf {
        let levels = ndarray::Array1::from_vec(bmrf_levels[i].clone());
        let k_arr = bmrf_k[i].as_array().to_owned();
        let lam_opt = if bmrf_lambda[i] >= 0.0 {
            Some(bmrf_lambda[i])
        } else {
            None
        };
        let by_view = bmrf_by_vecs[i]
            .as_ref()
            .map(|b| ndarray::ArrayView1::from(b.as_slice()));
        learners.push(mboostrs_core::learners::prepare_bmrf(
            levels.view(),
            w_owned.view(),
            &k_arr,
            bmrf_df[i],
            lam_opt,
            bmrf_center[i],
            &bmrf_names[i],
            by_view,
        ));
    }

    for i in 0..n_bspatial {
        let x1_view = bspatial_x1[i].as_array();
        let x2_view = bspatial_x2[i].as_array();
        let lam_opt = if bspatial_lambda[i] >= 0.0 {
            Some(bspatial_lambda[i])
        } else {
            None
        };
        let by_view = bspatial_by_vecs[i]
            .as_ref()
            .map(|b| ndarray::ArrayView1::from(b.as_slice()));
        learners.push(mboostrs_core::learners::prepare_bspatial(
            x1_view,
            x2_view,
            w_owned.view(),
            bspatial_knots[i],
            bspatial_degree[i],
            bspatial_differences[i],
            bspatial_df[i],
            lam_opt,
            bspatial_center[i],
            &bspatial_names[i],
            by_view,
        ));
    }

    let bkernel_x_owned: Vec<Vec<f64>> = bkernel_x
        .unwrap_or_default()
        .iter()
        .map(|x| x.as_array().to_vec())
        .collect();
    let bkernel_names_owned: Vec<String> = bkernel_names.unwrap_or_default();
    let bkernel_bandwidth_owned: Vec<f64> = bkernel_bandwidth.unwrap_or_default();
    let bkernel_df_owned: Vec<f64> = bkernel_df.unwrap_or_default();
    let bkernel_kernel_owned: Vec<String> = bkernel_kernel.unwrap_or_default();
    let bkernel_center_owned: Vec<bool> = bkernel_center.unwrap_or_default();
    for i in 0..n_bkernel {
        let view = ArrayView1::from(bkernel_x_owned[i].as_slice());
        let by_view = bkernel_by_vecs[i]
            .as_ref()
            .map(|b| ArrayView1::from(b.as_slice()));
        learners.push(mboostrs_core::learners::prepare_bkernel(
            view,
            w_owned.view(),
            bkernel_bandwidth_owned[i],
            bkernel_df_owned[i],
            &bkernel_kernel_owned[i],
            bkernel_center_owned[i],
            &bkernel_names_owned[i],
            by_view,
        ));
    }

    let brandom_x_owned: Vec<ndarray::Array2<f64>> = brandom_x
        .unwrap_or_default()
        .iter()
        .map(|x| x.as_array().to_owned())
        .collect();
    let brandom_names_owned: Vec<String> = brandom_names.unwrap_or_default();
    let brandom_n_levels_owned: Vec<usize> = brandom_n_levels.unwrap_or_default();
    let brandom_df_owned: Vec<f64> = brandom_df.unwrap_or_default();
    let brandom_lambda_owned: Vec<f64> = brandom_lambda.unwrap_or_default();
    let brandom_center_owned: Vec<bool> = brandom_center.unwrap_or_default();
    for i in 0..n_brandom {
        let design = &brandom_x_owned[i];
        let l = brandom_n_levels_owned[i];
        let lambda_val = if brandom_df_owned[i] >= 0.0 {
            let identity = ndarray::Array2::eye(l);
            // When center=True, df_to_lambda must see the centered design
            // so it finds the lambda that yields the target df on the
            // *centered* gram matrix.  Otherwise the same lambda produces
            // a different effective df after prepare_block_bols centers.
            if i < brandom_center_owned.len() && brandom_center_owned[i] {
                let sw: f64 = w_owned.sum();
                let mut centered = design.clone();
                for j in 0..centered.ncols() {
                    let cj = centered.column(j).dot(&w_owned) / sw;
                    centered.column_mut(j).mapv_inplace(|v| v - cj);
                }
                df_to_lambda(
                    &centered,
                    &identity,
                    &w_owned.to_owned(),
                    brandom_df_owned[i],
                )
            } else {
                df_to_lambda(design, &identity, &w_owned.to_owned(), brandom_df_owned[i])
            }
        } else {
            brandom_lambda_owned[i]
        };
        let mut design_owned = design.clone();
        if let Some(ref by) = brandom_by_vecs[i] {
            for r in 0..design_owned.nrows() {
                let b = by[r];
                for c in 0..design_owned.ncols() {
                    design_owned[[r, c]] *= b;
                }
            }
        }
        let mut penalty_diag = ndarray::Array2::zeros((l, l));
        for d in 0..l {
            penalty_diag[[d, d]] = lambda_val;
        }
        learners.push(prepare_block_bols(
            design_owned,
            w_owned.view(),
            penalty_diag,
            brandom_center_owned[i],
            &brandom_names_owned[i],
        ));
    }

    let buser_X_owned: Vec<ndarray::Array2<f64>> = buser_X
        .unwrap_or_default()
        .iter()
        .map(|x| x.as_array().to_owned())
        .collect();
    let buser_K_owned: Vec<ndarray::Array2<f64>> = buser_K
        .unwrap_or_default()
        .iter()
        .map(|k| k.as_array().to_owned())
        .collect();
    let buser_names_owned: Vec<String> = buser_names.unwrap_or_default();
    let buser_df_owned: Vec<f64> = buser_df.unwrap_or_default();
    let buser_lambda_owned: Vec<f64> = buser_lambda.unwrap_or_default();
    let buser_center_owned: Vec<bool> = buser_center.unwrap_or_default();
    for i in 0..n_buser {
        let design = buser_X_owned[i].clone();
        let penalty = buser_K_owned[i].clone();
        let lam_opt = if buser_lambda_owned[i] >= 0.0 {
            Some(buser_lambda_owned[i])
        } else {
            None
        };
        let by_view = buser_by_vecs[i]
            .as_ref()
            .map(|b| ndarray::ArrayView1::from(b.as_slice()));
        learners.push(mboostrs_core::learners::prepare_buser(
            design,
            penalty,
            w_owned.view(),
            buser_df_owned[i],
            lam_opt,
            buser_center_owned[i],
            &buser_names_owned[i],
            by_view,
        ));
    }

    let (learners, w_expanded) =
        apply_multinomial_kronecker(learners, None, &w_owned, multinomial_n_classes)?;
    let preparation_timing = preparation_timer.map(PreparationTimer::finish);
    run_gamboost_fit(
        py,
        learners,
        y,
        w_expanded,
        ctrl,
        family_name,
        &family_params,
        offset,
        nu,
        preparation_timing,
        binding_started,
        trace,
        // The array path has Python call `set_basis_info` afterwards, so the
        // store starts without replay recipes.
        Vec::new(),
    )
}

#[pyfunction(signature = (offset, selected, coefs_flat, nu, x_matrix, mstop=None, centering_means=None))]
#[allow(clippy::too_many_arguments)]
fn predict_glmboost(
    py: Python<'_>,
    offset: Bound<'_, PyAny>,
    selected: Vec<usize>,
    coefs_flat: Vec<f64>,
    nu: f64,
    x_matrix: PyReadonlyArray2<'_, f64>,
    mstop: Option<usize>,
    centering_means: Option<Vec<f64>>,
) -> PyResult<Vec<f64>> {
    let offset = extract_offset_arg(Some(&offset))?;
    let x_owned = x_matrix.as_array().to_owned();
    let n = x_owned.nrows();
    let p = x_owned.ncols();
    let x_flat: Vec<f64> = x_owned.into_raw_vec_and_offset().0;
    let cm = centering_means.unwrap_or_default();

    py.allow_threads(|| {
        core_predict_glmboost_vec(
            offset.scalar().unwrap_or(0.0),
            offset.vector(),
            &selected,
            &coefs_flat,
            nu,
            &x_flat,
            &cm,
            mstop,
            n,
            p,
        )
    })
    .map_err(mboostrs_error_to_pyerr)
}

/// Predict at multiple `mstop` values in one incremental pass — for sweeps
/// (path plots, manual early-stopping inspection) that would otherwise call
/// `predict_glmboost` once per grid point and redo the full design-application
/// loop each time. See `predict_glmboost_path` in `mboostrs-core::path` for
/// the algorithm. Returns a 2-D array `(len(grid), n)`.
#[pyfunction(signature = (offset, selected, coefs_flat, nu, x_matrix, grid, centering_means=None))]
#[allow(clippy::too_many_arguments)]
fn predict_glmboost_path<'py>(
    py: Python<'py>,
    offset: Bound<'_, PyAny>,
    selected: Vec<usize>,
    coefs_flat: Vec<f64>,
    nu: f64,
    x_matrix: PyReadonlyArray2<'_, f64>,
    grid: Vec<usize>,
    centering_means: Option<Vec<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let offset = extract_offset_arg(Some(&offset))?;
    let x_owned = x_matrix.as_array().to_owned();
    let n = x_owned.nrows();
    let p = x_owned.ncols();
    let x_flat: Vec<f64> = x_owned.into_raw_vec_and_offset().0;
    let cm = centering_means.unwrap_or_default();

    let rows = py
        .allow_threads(|| {
            core_predict_glmboost_path_vec(
                offset.scalar().unwrap_or(0.0),
                offset.vector(),
                &selected,
                &coefs_flat,
                nu,
                &x_flat,
                &cm,
                &grid,
                n,
                p,
            )
        })
        .map_err(mboostrs_error_to_pyerr)?;
    let n_grid = rows.len();
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    let arr = ndarray::Array2::from_shape_vec((n_grid, n), flat)
        .expect("predict_glmboost_path: row length mismatch");
    Ok(arr.into_pyarray(py).unbind())
}

#[pyfunction(signature = (selected, coefs_flat, nu, n_learners, centering_means, off2int, offset, which=None))]
#[allow(clippy::too_many_arguments)]
fn coef_glmboost_py(
    selected: Vec<usize>,
    coefs_flat: Vec<f64>,
    nu: f64,
    n_learners: usize,
    centering_means: Vec<f64>,
    off2int: bool,
    offset: f64,
    which: Option<Vec<usize>>,
) -> PyResult<(Vec<usize>, Vec<f64>)> {
    let result = core_coef_glmboost(
        &selected,
        &coefs_flat,
        nu,
        n_learners,
        &centering_means,
        off2int,
        offset,
        which.as_deref(),
    )
    .map_err(mboostrs_error_to_pyerr)?;
    Ok((result.indices, result.values))
}

/// Raw (un-centered) B-spline basis at `x` using the training boundary knots,
/// matching the design `prepare_kronecker_bbs` builds. Returned row-major as a
/// flat `(n * p)` vector with `p = n_knots + degree + 1`. Used by the Python
/// Multinomial newdata predictor to evaluate bbs learners on new x.
#[pyfunction]
fn bbs_raw_design(
    x: PyReadonlyArray1<'_, f64>,
    boundary_knots: (f64, f64),
    n_knots: usize,
    degree: usize,
) -> PyResult<(Vec<f64>, usize)> {
    if n_knots == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_knots must be a positive integer",
        ));
    }
    if !boundary_knots.0.is_finite()
        || !boundary_knots.1.is_finite()
        || boundary_knots.0 >= boundary_knots.1
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "boundary_knots must contain two finite increasing values",
        ));
    }
    let xv = x.as_array().to_owned();
    if xv.is_empty() || xv.iter().any(|value| !value.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "x must contain at least one finite value and no NaN or infinity",
        ));
    }
    let knots = bspline_knots(boundary_knots, n_knots, degree);
    let basis = bspline_basis_extrap(
        xv.as_slice().expect("owned x is contiguous"),
        &knots,
        degree,
        boundary_knots,
    );
    let p = basis.ncols();
    Ok((basis.iter().copied().collect(), p))
}

#[pyfunction(signature = (offset, selected, coefs_vecs, nu, scalar_x, bbs_x, bbs_knots, bbs_degree, bbs_differences, _bbs_df, bbs_center, scalar_center, _scalar_lambda, bbs_boundary_knots, bbs_deriv, scalar_centering_means, bmono_x, bmono_knots, bmono_degree, bmono_differences, bmono_boundary_knots, btree_x, mstop=None, scalar_by=None, bbs_by=None, bmono_by=None, btree_by=None, bmrf_levels=None, bmrf_k=None, bmrf_center=None, bmrf_by=None, bspatial_x1=None, bspatial_x2=None, bspatial_knots=None, bspatial_degree=None, bspatial_differences=None, bspatial_center=None, bspatial_by=None, bspatial_boundary_knots=None, brandom_x=None, brandom_n_levels=None, brandom_center=None, brandom_by=None, brandom_centering_means=None))]
#[allow(clippy::too_many_arguments)]
fn predict_gamboost(
    py: Python<'_>,
    offset: Bound<'_, PyAny>,
    selected: Vec<usize>,
    coefs_vecs: Vec<Vec<f64>>,
    nu: f64,
    scalar_x: Vec<PyReadonlyArray1<'_, f64>>,
    bbs_x: Vec<PyReadonlyArray1<'_, f64>>,
    bbs_knots: Vec<usize>,
    bbs_degree: Vec<usize>,
    bbs_differences: Vec<usize>,
    _bbs_df: Vec<f64>,
    bbs_center: Vec<bool>,
    scalar_center: Vec<bool>,
    _scalar_lambda: Vec<f64>,
    bbs_boundary_knots: Vec<(f64, f64)>,
    bbs_deriv: Vec<usize>,
    scalar_centering_means: Vec<f64>,
    bmono_x: Vec<PyReadonlyArray1<'_, f64>>,
    bmono_knots: Vec<usize>,
    bmono_degree: Vec<usize>,
    bmono_differences: Vec<usize>,
    bmono_boundary_knots: Vec<(f64, f64)>,
    btree_x: Vec<PyReadonlyArray2<'_, f64>>,
    mstop: Option<usize>,
    scalar_by: ByVecs,
    bbs_by: ByVecs,
    bmono_by: ByVecs,
    btree_by: ByVecs,
    bmrf_levels: Option<Vec<Vec<usize>>>,
    bmrf_k: Option<Vec<PyReadonlyArray2<'_, f64>>>,
    bmrf_center: Option<Vec<bool>>,
    bmrf_by: ByVecs,
    bspatial_x1: Option<Vec<PyReadonlyArray1<'_, f64>>>,
    bspatial_x2: Option<Vec<PyReadonlyArray1<'_, f64>>>,
    bspatial_knots: Option<Vec<usize>>,
    bspatial_degree: Option<Vec<usize>>,
    bspatial_differences: Option<Vec<usize>>,
    bspatial_center: Option<Vec<bool>>,
    bspatial_by: ByVecs,
    bspatial_boundary_knots: Option<Vec<((f64, f64), (f64, f64))>>,
    brandom_x: Option<Vec<PyReadonlyArray2<'_, f64>>>,
    brandom_n_levels: Option<Vec<usize>>,
    brandom_center: Option<Vec<bool>>,
    brandom_by: ByVecs,
    brandom_centering_means: Option<Vec<f64>>,
) -> PyResult<Vec<f64>> {
    let n_scalar = scalar_x.len();
    let n_bbs = bbs_x.len();
    let n_bmono = bmono_x.len();
    let n_btree = btree_x.len();
    let n_bmrf = bmrf_levels.as_ref().map_or(0, |v| v.len());
    let n_bspatial = bspatial_x1.as_ref().map_or(0, |v| v.len());
    let n_brandom = brandom_x.as_ref().map_or(0, |v| v.len());
    let _ = bmono_differences;
    let n = if n_scalar > 0 {
        scalar_x[0].as_array().len()
    } else if n_bbs > 0 {
        bbs_x[0].as_array().len()
    } else if n_bmono > 0 {
        bmono_x[0].as_array().len()
    } else if n_btree > 0 {
        btree_x[0].as_array().nrows()
    } else if n_bmrf > 0 {
        bmrf_levels.as_ref().unwrap()[0].len()
    } else if n_bspatial > 0 {
        bspatial_x1.as_ref().unwrap()[0].as_array().len()
    } else if n_brandom > 0 {
        brandom_x.as_ref().unwrap()[0].as_array().nrows()
    } else {
        return Ok(vec![]);
    };

    let scalar_by_vecs: Vec<Option<Vec<f64>>> = scalar_by.unwrap_or_else(|| vec![None; n_scalar]);
    let bbs_by_vecs: Vec<Option<Vec<f64>>> = bbs_by.unwrap_or_else(|| vec![None; n_bbs]);
    let bmono_by_vecs: Vec<Option<Vec<f64>>> = bmono_by.unwrap_or_else(|| vec![None; n_bmono]);
    let btree_by_vecs: Vec<Option<Vec<f64>>> = btree_by.unwrap_or_else(|| vec![None; n_btree]);
    let bmrf_by_vecs: Vec<Option<Vec<f64>>> = bmrf_by.unwrap_or_else(|| vec![None; n_bmrf]);
    let bspatial_by_vecs: Vec<Option<Vec<f64>>> =
        bspatial_by.unwrap_or_else(|| vec![None; n_bspatial]);
    let brandom_by_vecs: Vec<Option<Vec<f64>>> =
        brandom_by.unwrap_or_else(|| vec![None; n_brandom]);

    // Extract all PyO3 borrows into owned data BEFORE releasing the GIL
    let scalar_designs: Vec<ndarray::Array1<f64>> = scalar_x
        .iter()
        .enumerate()
        .map(|(i, sx)| {
            let x_arr = sx.as_array();
            let mut design = x_arr.to_owned();
            if let Some(ref by) = scalar_by_vecs[i] {
                for j in 0..design.len() {
                    design[j] *= by[j];
                }
            }
            if scalar_center[i] {
                design.mapv_inplace(|v| v - scalar_centering_means[i]);
            }
            design
        })
        .collect();

    let bbs_designs: Vec<ndarray::Array2<f64>> = bbs_x
        .iter()
        .enumerate()
        .map(|(i, bx)| {
            let x_arr = bx.as_array();
            let x_slice = x_arr.as_slice().expect("x contiguous");
            let bk = bbs_boundary_knots[i];
            let knots = bspline_knots(bk, bbs_knots[i], bbs_degree[i]);
            let mut basis = if bbs_deriv[i] > 0 {
                bspline_basis_with_deriv(x_slice, &knots, bbs_degree[i], bbs_deriv[i])
            } else {
                bspline_basis_extrap(x_slice, &knots, bbs_degree[i], bk)
            };
            let k = diff_penalty(basis.ncols(), bbs_differences[i]);

            if let Some(ref by) = bbs_by_vecs[i] {
                for r in 0..basis.nrows() {
                    for c in 0..basis.ncols() {
                        basis[[r, c]] *= by[r];
                    }
                }
            }

            if bbs_center[i] {
                center_spectral(&basis, &k, bbs_differences[i])
            } else {
                basis
            }
        })
        .collect();

    // bmono designs: same B-spline basis as bbs, no centering (R bmono
    // hard-codes center = FALSE, and our prepare_bmono enforces it).
    let bmono_designs: Vec<ndarray::Array2<f64>> = bmono_x
        .iter()
        .enumerate()
        .map(|(i, bx)| {
            let x_arr = bx.as_array();
            let x_slice = x_arr.as_slice().expect("x contiguous");
            let bk = bmono_boundary_knots[i];
            let knots = bspline_knots(bk, bmono_knots[i], bmono_degree[i]);
            let basis = bspline_basis_extrap(x_slice, &knots, bmono_degree[i], bk);
            if let Some(ref by) = bmono_by_vecs[i] {
                let mut result = basis;
                for r in 0..result.nrows() {
                    for c in 0..result.ncols() {
                        result[[r, c]] *= by[r];
                    }
                }
                result
            } else {
                basis
            }
        })
        .collect();

    let btree_designs_owned: Vec<ndarray::Array2<f64>> =
        btree_x.iter().map(|bx| bx.as_array().to_owned()).collect();

    // bmrf designs: indicator matrix from levels, optionally centered
    let bmrf_levels_owned: Vec<Vec<usize>> = bmrf_levels.unwrap_or_default();
    let bmrf_k_owned: Vec<ndarray::Array2<f64>> = bmrf_k
        .unwrap_or_default()
        .iter()
        .map(|k| k.as_array().to_owned())
        .collect();
    let bmrf_center_owned: Vec<bool> = bmrf_center.unwrap_or_default();
    let bmrf_designs: Vec<ndarray::Array2<f64>> = (0..n_bmrf)
        .map(|i| {
            let levels = &bmrf_levels_owned[i];
            let l = bmrf_k_owned[i].nrows();
            let mut design = ndarray::Array2::zeros((n, l));
            for (i_obs, &lev) in levels.iter().enumerate() {
                design[[i_obs, lev]] = 1.0;
            }
            if let Some(ref by) = bmrf_by_vecs[i] {
                for i_obs in 0..n {
                    let b = by[i_obs];
                    for j in 0..l {
                        design[[i_obs, j]] *= b;
                    }
                }
            }
            if bmrf_center_owned[i] {
                center_spectral(&design, &bmrf_k_owned[i], 1)
            } else {
                design
            }
        })
        .collect();

    // bspatial designs: tensor-product B-spline, optionally centered
    let bspatial_x1_owned: Vec<Vec<f64>> = bspatial_x1
        .unwrap_or_default()
        .iter()
        .map(|x| x.as_array().to_vec())
        .collect();
    let bspatial_x2_owned: Vec<Vec<f64>> = bspatial_x2
        .unwrap_or_default()
        .iter()
        .map(|x| x.as_array().to_vec())
        .collect();
    let bspatial_knots_owned: Vec<usize> = bspatial_knots.unwrap_or_default();
    let bspatial_degree_owned: Vec<usize> = bspatial_degree.unwrap_or_default();
    let bspatial_differences_owned: Vec<usize> = bspatial_differences.unwrap_or_default();
    let bspatial_center_owned: Vec<bool> = bspatial_center.unwrap_or_default();
    // Use training boundary knots if provided; otherwise fall back to computing from current x1/x2.
    let bspatial_bk_owned: Vec<((f64, f64), (f64, f64))> =
        bspatial_boundary_knots.unwrap_or_else(|| {
            bspatial_x1_owned
                .iter()
                .zip(bspatial_x2_owned.iter())
                .map(|(x1, x2)| {
                    let bk1 = (
                        x1.iter().cloned().fold(f64::INFINITY, f64::min),
                        x1.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                    );
                    let bk2 = (
                        x2.iter().cloned().fold(f64::INFINITY, f64::min),
                        x2.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                    );
                    (bk1, bk2)
                })
                .collect()
        });
    let bspatial_designs: Vec<ndarray::Array2<f64>> = (0..n_bspatial)
        .map(|i| {
            let x1_vec = &bspatial_x1_owned[i];
            let x2_vec = &bspatial_x2_owned[i];
            let n_knots = bspatial_knots_owned[i];
            let degree = bspatial_degree_owned[i];
            let diffs = bspatial_differences_owned[i];
            let (bk1, bk2) = bspatial_bk_owned[i];
            let knots1 = bspline_knots(bk1, n_knots, degree);
            let knots2 = bspline_knots(bk2, n_knots, degree);
            let b1 = bspline_basis_extrap(x1_vec, &knots1, degree, bk1);
            let b2 = bspline_basis_extrap(x2_vec, &knots2, degree, bk2);
            let p1 = b1.ncols();
            let p2 = b2.ncols();
            let p = p1 * p2;
            let mut design = ndarray::Array2::zeros((n, p));
            for i_obs in 0..n {
                for j1 in 0..p1 {
                    let v1 = b1[[i_obs, j1]];
                    if v1 == 0.0 {
                        continue;
                    }
                    for j2 in 0..p2 {
                        design[[i_obs, j1 * p2 + j2]] = v1 * b2[[i_obs, j2]];
                    }
                }
            }
            if let Some(ref by) = bspatial_by_vecs[i] {
                for i_obs in 0..n {
                    let b = by[i_obs];
                    for j in 0..p {
                        design[[i_obs, j]] *= b;
                    }
                }
            }
            if bspatial_center_owned[i] {
                let k1 = diff_penalty(p1, diffs);
                let k2 = diff_penalty(p2, diffs);
                let i1 = ndarray::Array2::eye(p1);
                let i2 = ndarray::Array2::eye(p2);
                let penalty = kron(&k1, &i2) + kron(&i1, &k2);
                center_spectral(&design, &penalty, 1)
            } else {
                design
            }
        })
        .collect();
    let brandom_n_levels_owned: Vec<usize> = brandom_n_levels.unwrap_or_default();
    let brandom_center_owned: Vec<bool> = brandom_center.unwrap_or_default();
    let brandom_centering_means_owned: Vec<f64> = brandom_centering_means.unwrap_or_default();
    let brandom_cum_levels: Vec<usize> = {
        let mut cum = Vec::with_capacity(brandom_n_levels_owned.len());
        let mut s = 0usize;
        for &lvl in &brandom_n_levels_owned {
            cum.push(s);
            s += lvl;
        }
        cum
    };
    let brandom_designs: Vec<ndarray::Array2<f64>> = brandom_x
        .unwrap_or_default()
        .iter()
        .enumerate()
        .map(|(i, bx)| {
            let x_arr = bx.as_array();
            let l = if i < brandom_n_levels_owned.len() {
                brandom_n_levels_owned[i]
            } else {
                x_arr.ncols()
            };
            let mut design = ndarray::Array2::zeros((n, l));
            for r in 0..x_arr.nrows() {
                for c in 0..l.min(x_arr.ncols()) {
                    design[[r, c]] = x_arr[[r, c]];
                }
            }
            if let Some(ref by) = brandom_by_vecs[i] {
                for r in 0..n {
                    let b = by[r];
                    for c in 0..l {
                        design[[r, c]] *= b;
                    }
                }
            }
            if i < brandom_center_owned.len() && brandom_center_owned[i] {
                let start = if i < brandom_cum_levels.len() {
                    brandom_cum_levels[i]
                } else {
                    i * l
                };
                for c in 0..l {
                    let cm = if start + c < brandom_centering_means_owned.len() {
                        brandom_centering_means_owned[start + c]
                    } else {
                        0.0
                    };
                    for r in 0..n {
                        design[[r, c]] -= cm;
                    }
                }
                design
            } else {
                design
            }
        })
        .collect();

    let max_m = mstop.unwrap_or(selected.len()).min(selected.len());
    let offset = extract_offset_arg(Some(&offset))?;
    let start = eta_start(&offset, n)?;

    let eta = py.allow_threads(|| {
        let mut eta = ndarray::Array1::from(start);

        for m in 0..max_m {
            let rust_idx = selected[m];
            let coef = &coefs_vecs[m];

            if rust_idx < n_scalar {
                let design_col = &scalar_designs[rust_idx];
                for i in 0..n {
                    eta[i] += nu * design_col[i] * coef[0];
                }
            } else if rust_idx < n_scalar + n_bbs {
                let bbs_idx = rust_idx - n_scalar;
                let design = &bbs_designs[bbs_idx];
                let coef_arr = ndarray::Array1::from_vec(coef.to_vec());
                let pred = design.dot(&coef_arr);
                for i in 0..n {
                    eta[i] += nu * pred[i];
                }
            } else if rust_idx < n_scalar + n_bbs + n_bmono {
                let bmono_idx = rust_idx - n_scalar - n_bbs;
                let design = &bmono_designs[bmono_idx];
                let coef_arr = ndarray::Array1::from_vec(coef.to_vec());
                let pred = design.dot(&coef_arr);
                for i in 0..n {
                    eta[i] += nu * pred[i];
                }
            } else if rust_idx < n_scalar + n_bbs + n_bmono + n_btree {
                let btree_idx = rust_idx - n_scalar - n_bbs - n_bmono;
                let design = &btree_designs_owned[btree_idx];
                let coef_arr = ndarray::Array1::from_vec(coef.to_vec());
                let tree = FittedTree::from_coef_vec(coef_arr.view());
                let pred = tree.predict(design.view());
                let by_opt = btree_by_vecs[btree_idx].as_ref();
                for i in 0..n {
                    let by_val = by_opt.map_or(1.0, |b| b[i]);
                    eta[i] += nu * pred[i] * by_val;
                }
            } else if rust_idx < n_scalar + n_bbs + n_bmono + n_btree + n_bmrf {
                let bmrf_idx = rust_idx - n_scalar - n_bbs - n_bmono - n_btree;
                let design = &bmrf_designs[bmrf_idx];
                let coef_arr = ndarray::Array1::from_vec(coef.to_vec());
                let pred = design.dot(&coef_arr);
                for i in 0..n {
                    eta[i] += nu * pred[i];
                }
            } else if rust_idx < n_scalar + n_bbs + n_bmono + n_btree + n_bmrf + n_bspatial {
                let bspatial_idx = rust_idx - n_scalar - n_bbs - n_bmono - n_btree - n_bmrf;
                let design = &bspatial_designs[bspatial_idx];
                let coef_arr = ndarray::Array1::from_vec(coef.to_vec());
                let pred = design.dot(&coef_arr);
                for i in 0..n {
                    eta[i] += nu * pred[i];
                }
            } else {
                let brandom_idx =
                    rust_idx - n_scalar - n_bbs - n_bmono - n_btree - n_bmrf - n_bspatial;
                if brandom_idx < n_brandom {
                    let design = &brandom_designs[brandom_idx];
                    let coef_arr = ndarray::Array1::from_vec(coef.to_vec());
                    let pred = design.dot(&coef_arr);
                    for i in 0..n {
                        eta[i] += nu * pred[i];
                    }
                }
            }
        }

        eta.into_raw_vec_and_offset().0
    });

    Ok(eta)
}

struct GlmBoostPreparer {
    x_cols: Vec<Vec<f64>>,
    names: Vec<String>,
    family_name: String,
    family_params: HashMap<String, f64>,
    column_means: Vec<f64>,
    /// Whether the constant learner is offered (false for `y ~ x - 1`).
    intercept: bool,
}

impl LearnerPreparer for GlmBoostPreparer {
    fn prepare(&self, weights: ArrayView1<f64>) -> Vec<PreparedLearner> {
        let n = weights.len();
        let ones = ndarray::Array1::ones(n);
        let mut learners: Vec<PreparedLearner> = Vec::with_capacity(self.x_cols.len() + 1);
        if self.intercept {
            learners.push(mboostrs_core::learners::prepare_scalar_bols(
                ones.view(),
                weights,
                0.0,
                false,
                "(Intercept)",
                None,
            ));
        }
        for (j, col) in self.x_cols.iter().enumerate() {
            let view = ArrayView1::from(col.as_slice());
            learners.push(mboostrs_core::learners::prepare_scalar_bols_fixed_center(
                view,
                weights,
                0.0,
                self.column_means[j],
                self.names.get(j).unwrap_or(&"?".to_string()),
            ));
        }
        learners
    }
    fn family(&self) -> Box<dyn Family> {
        make_family(&self.family_name, &self.family_params).unwrap()
    }
}

#[allow(non_snake_case)]
struct CachedBlockTemplate {
    design: ndarray::Array2<f64>,
    penalty: ndarray::Array2<f64>,
}

#[allow(clippy::too_many_arguments)]
fn build_bbs_templates(
    bbs_cols: &[Vec<f64>],
    bbs_knots: &[usize],
    bbs_degree: &[usize],
    bbs_differences: &[usize],
    bbs_center: &[bool],
    bbs_cyclic: &[bool],
    bbs_boundary_knots_lo: &[f64],
    bbs_boundary_knots_hi: &[f64],
    bbs_deriv: &[usize],
    bbs_by: &[Option<Vec<f64>>],
) -> Vec<CachedBlockTemplate> {
    let mut templates = Vec::with_capacity(bbs_cols.len());
    for i in 0..bbs_cols.len() {
        let x = &bbs_cols[i];
        let bk = if bbs_boundary_knots_lo[i].is_nan() || bbs_boundary_knots_hi[i].is_nan() {
            (
                x.iter().copied().fold(f64::INFINITY, f64::min),
                x.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            )
        } else {
            (bbs_boundary_knots_lo[i], bbs_boundary_knots_hi[i])
        };

        let (mut design, mut penalty) = if bbs_cyclic[i] {
            let knots = cyclic_bspline_knots(bk, bbs_knots[i], bbs_degree[i]);
            let basis = cyclic_bspline_basis_from_knots(x, &knots, bbs_degree[i], bk);
            let k = cyclic_diff_penalty(basis.ncols(), bbs_differences[i]);
            (basis, k)
        } else {
            let knots = bspline_knots(bk, bbs_knots[i], bbs_degree[i]);
            let basis = if bbs_deriv[i] > 0 {
                bspline_basis_with_deriv(x, &knots, bbs_degree[i], bbs_deriv[i])
            } else {
                bspline_basis(x, &knots, bbs_degree[i])
            };
            let k = diff_penalty(basis.ncols(), bbs_differences[i]);
            (basis, k)
        };

        if let Some(by_vec) = &bbs_by[i] {
            for r in 0..design.nrows() {
                let by_val = by_vec[r];
                for c in 0..design.ncols() {
                    design[[r, c]] *= by_val;
                }
            }
        }

        if bbs_center[i] {
            design = center_spectral(&design, &penalty, bbs_differences[i]);
            penalty = ndarray::Array2::eye(design.ncols());
        }

        templates.push(CachedBlockTemplate { design, penalty });
    }
    templates
}

fn build_bspatial_templates(
    bspatial_x1: &[Vec<f64>],
    bspatial_x2: &[Vec<f64>],
    bspatial_knots: &[usize],
    bspatial_degree: &[usize],
    bspatial_differences: &[usize],
    bspatial_center: &[bool],
    bspatial_by: &[Option<Vec<f64>>],
) -> Vec<CachedBlockTemplate> {
    let mut templates = Vec::with_capacity(bspatial_x1.len());
    for i in 0..bspatial_x1.len() {
        let x1 = &bspatial_x1[i];
        let x2 = &bspatial_x2[i];
        let bk1 = (
            x1.iter().copied().fold(f64::INFINITY, f64::min),
            x1.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        );
        let bk2 = (
            x2.iter().copied().fold(f64::INFINITY, f64::min),
            x2.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        );
        let knots1 = bspline_knots(bk1, bspatial_knots[i], bspatial_degree[i]);
        let knots2 = bspline_knots(bk2, bspatial_knots[i], bspatial_degree[i]);
        let b1 = bspline_basis(x1, &knots1, bspatial_degree[i]);
        let b2 = bspline_basis(x2, &knots2, bspatial_degree[i]);

        let p1 = b1.ncols();
        let p2 = b2.ncols();
        let p = p1 * p2;
        let mut design = ndarray::Array2::zeros((x1.len(), p));
        for r in 0..x1.len() {
            for j1 in 0..p1 {
                let v1 = b1[[r, j1]];
                if v1 == 0.0 {
                    continue;
                }
                for j2 in 0..p2 {
                    design[[r, j1 * p2 + j2]] = v1 * b2[[r, j2]];
                }
            }
        }

        if let Some(by_vec) = &bspatial_by[i] {
            for r in 0..design.nrows() {
                let by_val = by_vec[r];
                for c in 0..design.ncols() {
                    design[[r, c]] *= by_val;
                }
            }
        }

        let k1 = diff_penalty(p1, bspatial_differences[i]);
        let k2 = diff_penalty(p2, bspatial_differences[i]);
        let i1 = ndarray::Array2::eye(p1);
        let i2 = ndarray::Array2::eye(p2);
        let mut penalty = kron(&k1, &i2) + kron(&i1, &k2);

        if bspatial_center[i] {
            design = center_spectral(&design, &penalty, 1);
            penalty = ndarray::Array2::eye(design.ncols());
        }

        templates.push(CachedBlockTemplate { design, penalty });
    }
    templates
}

/// Compact-spec preparer shared by binned fits and their resampling paths.
/// `LearnerBlueprint::BinnedScalarBols` owns an `Arc` bin index, so every fold
/// reuses one discretization while rebuilding only weight-dependent moments.
struct BlueprintPreparer {
    blueprints: Vec<LearnerBlueprint>,
    family_name: String,
    family_params: HashMap<String, f64>,
    multinomial_n_classes: Option<usize>,
}

impl LearnerPreparer for BlueprintPreparer {
    fn prepare(&self, weights: ArrayView1<f64>) -> Vec<PreparedLearner> {
        // For Multinomial the fold weights arrive at the flattened length
        // n_obs*(K-1): `expand_fold_matrix` repeats each observation's row in
        // place, so observation i owns entries i*(K-1)..(i+1)*(K-1). The
        // latent blueprints build their per-observation design with
        // observation-length weights, so take the first entry of each block.
        // Slicing the leading n_obs entries instead gives observation i the
        // weight of observation i/(K-1).
        let (obs_weights, n_classes) = match self.multinomial_n_classes {
            Some(n_classes) => {
                let n_resp = n_classes - 1;
                debug_assert_eq!(weights.len() % n_resp, 0);
                (
                    ndarray::Array1::from_iter(weights.iter().step_by(n_resp).copied()),
                    Some(n_classes),
                )
            }
            None => (weights.to_owned(), None),
        };
        let (learners, basis_info): (Vec<PreparedLearner>, Vec<BasisInfo>) = self
            .blueprints
            .iter()
            .map(|blueprint| {
                blueprint
                    .prepare(obs_weights.view())
                    .expect("validated learner blueprint failed during fold preparation")
            })
            .unzip();
        match n_classes {
            // The Kronecker lift replaces the learner list wholesale.
            Some(n_classes) => {
                wrap_kronecker_learners(learners, Some(&basis_info), &obs_weights, n_classes)
                    .expect("validated Multinomial learner failed the Kronecker lift")
            }
            None => learners,
        }
    }

    fn family(&self) -> Box<dyn Family> {
        make_family(&self.family_name, &self.family_params)
            .expect("validated family failed during fold preparation")
    }
}

#[allow(non_snake_case)]
struct GamBoostPreparer {
    scalar_cols: Vec<Vec<f64>>,
    scalar_names: Vec<String>,
    scalar_lambda: Vec<f64>,
    scalar_center: Vec<bool>,
    scalar_df: Vec<f64>,
    scalar_by: Vec<Option<Vec<f64>>>,
    bbs_names: Vec<String>,
    bbs_df: Vec<f64>,
    bbs_lambda: Vec<f64>,
    bbs_templates: Vec<CachedBlockTemplate>,
    bmono_cols: Vec<Vec<f64>>,
    bmono_names: Vec<String>,
    bmono_knots: Vec<usize>,
    bmono_degree: Vec<usize>,
    bmono_differences: Vec<usize>,
    bmono_df: Vec<f64>,
    bmono_constraint: Vec<MonoConstraint>,
    bmono_solver: Vec<BmonoSolver>,
    bmono_lambda2: Vec<f64>,
    bmono_niter: Vec<usize>,
    bmono_center: Vec<bool>,
    bmono_C: Vec<Option<ndarray::Array2<f64>>>,
    bmono_by: Vec<Option<Vec<f64>>>,
    btree_designs: Vec<ndarray::Array2<f64>>,
    btree_names: Vec<String>,
    btree_minbucket: Vec<usize>,
    btree_minsplit: Vec<usize>,
    btree_maxdepth: Vec<usize>,
    btree_by: Vec<Option<Vec<f64>>>,
    bmrf_levels: Vec<Vec<usize>>,
    bmrf_names: Vec<String>,
    bmrf_k: Vec<ndarray::Array2<f64>>,
    bmrf_df: Vec<f64>,
    bmrf_lambda: Vec<f64>,
    bmrf_center: Vec<bool>,
    bmrf_by: Vec<Option<Vec<f64>>>,
    bspatial_names: Vec<String>,
    bspatial_df: Vec<f64>,
    bspatial_lambda: Vec<f64>,
    bspatial_templates: Vec<CachedBlockTemplate>,
    bkernel_cols: Vec<Vec<f64>>,
    bkernel_names: Vec<String>,
    bkernel_bandwidth: Vec<f64>,
    bkernel_df: Vec<f64>,
    bkernel_kernel: Vec<String>,
    bkernel_center: Vec<bool>,
    bkernel_by: Vec<Option<Vec<f64>>>,
    brandom_cols: Vec<ndarray::Array2<f64>>,
    brandom_names: Vec<String>,
    brandom_n_levels: Vec<usize>,
    brandom_df: Vec<f64>,
    brandom_lambda: Vec<f64>,
    brandom_center: Vec<bool>,
    brandom_by: Vec<Option<Vec<f64>>>,
    buser_X: Vec<ndarray::Array2<f64>>,
    buser_K: Vec<ndarray::Array2<f64>>,
    buser_names: Vec<String>,
    buser_df: Vec<f64>,
    buser_lambda: Vec<f64>,
    buser_center: Vec<bool>,
    buser_by: Vec<Option<Vec<f64>>>,
    family_name: String,
    family_params: HashMap<String, f64>,
}

impl LearnerPreparer for GamBoostPreparer {
    fn prepare(&self, weights: ArrayView1<f64>) -> Vec<PreparedLearner> {
        let mut learners: Vec<PreparedLearner> = Vec::with_capacity(
            self.scalar_cols.len()
                + self.bbs_templates.len()
                + self.bmono_cols.len()
                + self.btree_designs.len()
                + self.bmrf_levels.len()
                + self.bspatial_templates.len()
                + self.brandom_cols.len()
                + self.buser_X.len(),
        );
        for i in 0..self.scalar_cols.len() {
            let view = ArrayView1::from(self.scalar_cols[i].as_slice());
            let lambda = if self.scalar_df[i] >= 0.0 {
                df_to_lambda_scalar(view, weights, self.scalar_df[i], self.scalar_center[i])
            } else {
                self.scalar_lambda[i]
            };
            let by_view = self.scalar_by[i]
                .as_ref()
                .map(|b| ArrayView1::from(b.as_slice()));
            learners.push(mboostrs_core::learners::prepare_scalar_bols(
                view,
                weights,
                lambda,
                self.scalar_center[i],
                &self.scalar_names[i],
                by_view,
            ));
        }
        for i in 0..self.bbs_templates.len() {
            let template = &self.bbs_templates[i];
            let lambda = if self.bbs_lambda[i] >= 0.0 {
                self.bbs_lambda[i]
            } else {
                df_to_lambda(
                    &template.design,
                    &template.penalty,
                    &weights.to_owned(),
                    self.bbs_df[i],
                )
            };
            learners.push(prepare_block_bols(
                template.design.clone(),
                weights,
                template.penalty.mapv(|v| lambda * v),
                false,
                &self.bbs_names[i],
            ));
        }
        for i in 0..self.bmono_cols.len() {
            let view = ArrayView1::from(self.bmono_cols[i].as_slice());
            let by_view = self.bmono_by[i]
                .as_ref()
                .map(|b| ArrayView1::from(b.as_slice()));
            let constraint_mat = self.bmono_C[i].as_ref().map(|c| c.view());
            learners.push(mboostrs_core::learners::prepare_bmono(
                view,
                weights,
                self.bmono_knots[i],
                self.bmono_degree[i],
                self.bmono_differences[i],
                self.bmono_df[i],
                self.bmono_constraint[i],
                self.bmono_solver[i],
                self.bmono_lambda2[i],
                self.bmono_niter[i],
                &self.bmono_names[i],
                by_view,
                constraint_mat,
                self.bmono_center[i],
            ));
        }
        for i in 0..self.btree_designs.len() {
            let by_owned = self.btree_by[i]
                .as_ref()
                .map(|b| ndarray::Array1::from_vec(b.clone()));
            learners.push(mboostrs_core::learners::prepare_btree(
                self.btree_designs[i].view(),
                weights,
                self.btree_minbucket[i],
                self.btree_minsplit[i],
                self.btree_maxdepth[i],
                by_owned,
                &self.btree_names[i],
            ));
        }
        for i in 0..self.bmrf_levels.len() {
            let levels = ndarray::Array1::from_vec(self.bmrf_levels[i].clone());
            let lam_opt = if self.bmrf_lambda[i] >= 0.0 {
                Some(self.bmrf_lambda[i])
            } else {
                None
            };
            let by_view = self.bmrf_by[i]
                .as_ref()
                .map(|b| ArrayView1::from(b.as_slice()));
            learners.push(prepare_bmrf(
                levels.view(),
                weights,
                &self.bmrf_k[i],
                self.bmrf_df[i],
                lam_opt,
                self.bmrf_center[i],
                &self.bmrf_names[i],
                by_view,
            ));
        }
        for i in 0..self.bspatial_templates.len() {
            let template = &self.bspatial_templates[i];
            let lambda_override = if self.bspatial_lambda[i] >= 0.0 {
                Some(self.bspatial_lambda[i])
            } else {
                None
            };
            // One `X'WX` serves both the df→λ calibration and the solver Gram
            // (`X'WX + λK`) instead of the `df_to_lambda` + `prepare_block_bols`
            // pair's second O(n·p²) pass.
            learners.push(prepare_block_bols_shared_gram(
                template.design.clone(),
                weights,
                template.penalty.clone(),
                self.bspatial_df[i],
                lambda_override,
                &self.bspatial_names[i],
            ));
        }
        for i in 0..self.bkernel_cols.len() {
            let view = ArrayView1::from(self.bkernel_cols[i].as_slice());
            let by_view = self.bkernel_by[i]
                .as_ref()
                .map(|b| ArrayView1::from(b.as_slice()));
            learners.push(mboostrs_core::learners::prepare_bkernel(
                view,
                weights,
                self.bkernel_bandwidth[i],
                self.bkernel_df[i],
                &self.bkernel_kernel[i],
                self.bkernel_center[i],
                &self.bkernel_names[i],
                by_view,
            ));
        }
        for i in 0..self.brandom_cols.len() {
            let design = &self.brandom_cols[i];
            let l = self.brandom_n_levels[i];
            let lambda_val = if self.brandom_df[i] >= 0.0 {
                let identity = ndarray::Array2::eye(l);
                if self.brandom_center[i] {
                    let sw: f64 = weights.sum();
                    let mut centered = design.clone();
                    for j in 0..centered.ncols() {
                        let cj = centered.column(j).dot(&weights) / sw;
                        centered.column_mut(j).mapv_inplace(|v| v - cj);
                    }
                    df_to_lambda(
                        &centered,
                        &identity,
                        &weights.to_owned(),
                        self.brandom_df[i],
                    )
                } else {
                    df_to_lambda(design, &identity, &weights.to_owned(), self.brandom_df[i])
                }
            } else {
                self.brandom_lambda[i]
            };
            let mut design_owned = design.clone();
            if let Some(ref by) = self.brandom_by[i] {
                for r in 0..design_owned.nrows() {
                    let b = by[r];
                    for c in 0..design_owned.ncols() {
                        design_owned[[r, c]] *= b;
                    }
                }
            }
            let mut penalty_diag = ndarray::Array2::zeros((l, l));
            for d in 0..l {
                penalty_diag[[d, d]] = lambda_val;
            }
            learners.push(prepare_block_bols(
                design_owned,
                weights,
                penalty_diag,
                self.brandom_center[i],
                &self.brandom_names[i],
            ));
        }
        for i in 0..self.buser_X.len() {
            let design = self.buser_X[i].clone();
            let penalty = self.buser_K[i].clone();
            let lam_opt = if self.buser_lambda[i] >= 0.0 {
                Some(self.buser_lambda[i])
            } else {
                None
            };
            let by_view = self.buser_by[i]
                .as_ref()
                .map(|b| ArrayView1::from(b.as_slice()));
            learners.push(mboostrs_core::learners::prepare_buser(
                design,
                penalty,
                weights,
                self.buser_df[i],
                lam_opt,
                self.buser_center[i],
                &self.buser_names[i],
                by_view,
            ));
        }
        learners
    }
    fn family(&self) -> Box<dyn Family> {
        make_family(&self.family_name, &self.family_params).unwrap()
    }
}

#[pyfunction]
#[pyo3(signature = (y, predicted, weights, metric, parameter=None))]
fn response_score(
    y: PyReadonlyArray1<'_, f64>,
    predicted: PyReadonlyArray2<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    metric: &str,
    parameter: Option<f64>,
) -> PyResult<(f64, Vec<f64>)> {
    mboostrs_core::scoring::response_score(
        &y.as_array().to_vec(),
        predicted.as_array(),
        &weights.as_array().to_vec(),
        metric,
        parameter,
    )
    .map_err(mboostrs_error_to_pyerr)
}

/// Breslow baseline survival curves (`mboostrs_core::survival::breslow`).
/// Returns `(surv, event_times, n_event)`, with `surv` of shape
/// `(n_event_times, n_curves)`.
#[pyfunction]
#[pyo3(signature = (time, event, linpred_train, linpred_new=None))]
fn breslow_survfit<'py>(
    py: Python<'py>,
    time: PyReadonlyArray1<'_, f64>,
    event: PyReadonlyArray1<'_, f64>,
    linpred_train: PyReadonlyArray1<'_, f64>,
    linpred_new: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Vec<f64>, Vec<f64>)> {
    let fit = mboostrs_core::survival::breslow(
        &time.as_array().to_vec(),
        &event.as_array().to_vec(),
        &linpred_train.as_array().to_vec(),
    )
    .map_err(mboostrs_error_to_pyerr)?;
    let new = linpred_new.map(|values| values.as_array().to_vec());
    let curves = fit
        .curves(new.as_deref())
        .map_err(mboostrs_error_to_pyerr)?;
    Ok((curves.into_pyarray(py), fit.event_times, fit.n_event))
}

/// Survival curves at the event times of a stored Breslow baseline
/// (`mboostrs_core::survival::Breslow::curves`). Shape
/// `(n_event_times, n_curves)`; `linpred_new=None` gives the baseline curve.
#[pyfunction]
#[pyo3(signature = (event_times, n_event, cumhaz, center, linpred_new=None))]
fn breslow_curves<'py>(
    py: Python<'py>,
    event_times: Vec<f64>,
    n_event: Vec<f64>,
    cumhaz: Vec<f64>,
    center: f64,
    linpred_new: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let fit = mboostrs_core::survival::Breslow {
        event_times,
        n_event,
        cumhaz,
        center,
    };
    let new = linpred_new.map(|values| values.as_array().to_vec());
    let curves = fit
        .curves(new.as_deref())
        .map_err(mboostrs_error_to_pyerr)?;
    Ok(curves.into_pyarray(py))
}

/// Breslow baseline for storage on a fitted model
/// (`mboostrs_core::survival::breslow`). Returns
/// `(event_times, n_event, cumhaz, center)`.
#[pyfunction]
fn breslow_baseline(
    time: PyReadonlyArray1<'_, f64>,
    event: PyReadonlyArray1<'_, f64>,
    linpred_train: PyReadonlyArray1<'_, f64>,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>, f64)> {
    let fit = mboostrs_core::survival::breslow(
        &time.as_array().to_vec(),
        &event.as_array().to_vec(),
        &linpred_train.as_array().to_vec(),
    )
    .map_err(mboostrs_error_to_pyerr)?;
    Ok((fit.event_times, fit.n_event, fit.cumhaz, fit.center))
}

/// Fixed-horizon survival `S(t | x)` from a stored Breslow baseline
/// (`mboostrs_core::survival::Breslow::survival_at`). Shape
/// `(n_subjects, n_horizons)`.
#[pyfunction]
fn breslow_horizon_survival<'py>(
    py: Python<'py>,
    event_times: Vec<f64>,
    n_event: Vec<f64>,
    cumhaz: Vec<f64>,
    center: f64,
    horizons: Vec<f64>,
    linpred_new: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let fit = mboostrs_core::survival::Breslow {
        event_times,
        n_event,
        cumhaz,
        center,
    };
    let surv = fit
        .survival_at(&horizons, &linpred_new.as_array().to_vec())
        .map_err(mboostrs_error_to_pyerr)?;
    Ok(surv.into_pyarray(py))
}

/// IPCW Brier score at one horizon (`mboostrs_core::survival::ipcw_brier`).
/// Returns `(score, censoring_survival_at_horizon, below_floor)`.
#[pyfunction]
fn ipcw_brier_at(
    time: PyReadonlyArray1<'_, f64>,
    event: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    survival: PyReadonlyArray1<'_, f64>,
    horizon: f64,
    censoring_floor: f64,
) -> PyResult<(f64, f64, bool)> {
    let score = mboostrs_core::survival::ipcw_brier(
        &time.as_array().to_vec(),
        &event.as_array().to_vec(),
        &weights.as_array().to_vec(),
        &survival.as_array().to_vec(),
        horizon,
        censoring_floor,
    )
    .map_err(mboostrs_error_to_pyerr)?;
    Ok((score.value, score.censoring_at_horizon, score.below_floor))
}

/// Cumulative/dynamic IPCW AUC at one horizon
/// (`mboostrs_core::survival::ipcw_auc`). Returns
/// `(auc, censoring_survival_at_horizon, below_floor, case_weight, control_weight)`.
#[pyfunction]
fn ipcw_auc_at(
    time: PyReadonlyArray1<'_, f64>,
    event: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    score: PyReadonlyArray1<'_, f64>,
    horizon: f64,
    censoring_floor: f64,
) -> PyResult<(f64, f64, bool, f64, f64)> {
    let result = mboostrs_core::survival::ipcw_auc(
        &time.as_array().to_vec(),
        &event.as_array().to_vec(),
        &weights.as_array().to_vec(),
        &score.as_array().to_vec(),
        horizon,
        censoring_floor,
    )
    .map_err(mboostrs_error_to_pyerr)?;
    Ok((
        result.auc,
        result.censoring_at_horizon,
        result.below_floor,
        result.case_weight,
        result.control_weight,
    ))
}

/// Type-7 quantiles of `values` at each probability in `probs`
/// (`mboostrs_core::declarations::quantiles`).
#[pyfunction]
fn declaration_quantiles(values: PyReadonlyArray1<'_, f64>, probs: Vec<f64>) -> PyResult<Vec<f64>> {
    mboostrs_core::declarations::quantiles(values.as_array(), &probs)
        .map_err(mboostrs_error_to_pyerr)
}

/// Sign verdict for a declared positive (`sign=1`) or negative (`sign=-1`)
/// effect (`mboostrs_core::declarations::check_sign`). `boot` holds the
/// full-data contrast recomputed on each bootstrap refit. Returns the full-data
/// contrast, the fold contrasts, the lower and upper ends of the bootstrap
/// interval, the number of folds whose sign matches the full-data contrast, and
/// the verdict label.
#[pyfunction]
#[pyo3(signature = (sign, hi, lo, fold_hi, fold_lo, boot))]
fn declaration_sign_check(
    sign: f64,
    hi: PyReadonlyArray1<'_, f64>,
    lo: PyReadonlyArray1<'_, f64>,
    fold_hi: Vec<PyReadonlyArray1<'_, f64>>,
    fold_lo: Vec<PyReadonlyArray1<'_, f64>>,
    boot: PyReadonlyArray1<'_, f64>,
) -> PyResult<(f64, Vec<f64>, f64, f64, usize, &'static str)> {
    if fold_hi.len() != fold_lo.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "fold_hi and fold_lo must hold one array per fold",
        ));
    }
    let folds: Vec<_> = fold_hi
        .iter()
        .zip(fold_lo.iter())
        .map(|(h, l)| (h.as_array(), l.as_array()))
        .collect();
    let check = mboostrs_core::declarations::check_sign(
        sign,
        hi.as_array(),
        lo.as_array(),
        &folds,
        boot.as_array(),
    )
    .map_err(mboostrs_error_to_pyerr)?;
    Ok((
        check.contrast,
        check.fold_contrasts,
        check.lower,
        check.upper,
        check.folds_agreeing,
        check.verdict.as_str(),
    ))
}

/// Monotone or curvature verdict for a declared shape
/// (`mboostrs_core::declarations::check_shape`). `grid` holds one row per
/// grid value and one column per data row. Returns the average curve, the
/// number of moving steps, the share of steps in the declared direction (None
/// for a flat curve), the mean constrained and unconstrained held-out risk,
/// their relative change, and the verdict label.
#[pyfunction]
#[pyo3(signature = (sign, second_difference, grid, risk_constrained, risk_unconstrained))]
fn declaration_shape_check(
    sign: f64,
    second_difference: bool,
    grid: PyReadonlyArray2<'_, f64>,
    risk_constrained: Vec<f64>,
    risk_unconstrained: Vec<f64>,
) -> PyResult<(Vec<f64>, usize, Option<f64>, f64, f64, f64, &'static str)> {
    let check = mboostrs_core::declarations::check_shape(
        sign,
        second_difference,
        grid.as_array(),
        &risk_constrained,
        &risk_unconstrained,
    )
    .map_err(mboostrs_error_to_pyerr)?;
    Ok((
        check.curve,
        check.steps,
        check.share,
        check.risk_constrained,
        check.risk_unconstrained,
        check.risk_change,
        check.verdict.as_str(),
    ))
}

/// Joint support of new rows relative to training rows
/// (`mboostrs_core::support::joint_support`): per new row, the squared
/// Mahalanobis distance from the training mean and the share of training
/// rows at least as far. `None` if there are fewer than two training rows.
#[pyfunction]
fn joint_support(
    py: Python<'_>,
    train: PyReadonlyArray2<'_, f64>,
    new: PyReadonlyArray2<'_, f64>,
) -> PyResult<Option<(Vec<f64>, Vec<f64>)>> {
    let train = train.as_array().to_owned();
    let new = new.as_array().to_owned();
    if train.ncols() != new.ncols() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "train has {} columns but new has {}",
            train.ncols(),
            new.ncols()
        )));
    }
    Ok(py.allow_threads(|| {
        mboostrs_core::support::joint_support(train.view(), new.view()).map(|rows| {
            (
                rows.iter().map(|r| r.distance_sq).collect(),
                rows.iter().map(|r| r.training_share_beyond).collect(),
            )
        })
    }))
}

/// Same-cell nearest-neighbour distance of new rows to training rows
/// (`mboostrs_core::support::cell_nearest_distance`), in training standard
/// deviations. Categorical predictors are integer codes (negative for an
/// unseen level); a new row with no same-cell training row gets `inf`.
#[pyfunction]
fn cell_nearest_distance(
    py: Python<'_>,
    train_cells: PyReadonlyArray2<'_, f64>,
    train_numeric: PyReadonlyArray2<'_, f64>,
    new_cells: PyReadonlyArray2<'_, f64>,
    new_numeric: PyReadonlyArray2<'_, f64>,
) -> PyResult<Vec<f64>> {
    let train_cells = train_cells.as_array().to_owned();
    let train_numeric = train_numeric.as_array().to_owned();
    let new_cells = new_cells.as_array().to_owned();
    let new_numeric = new_numeric.as_array().to_owned();
    py.allow_threads(|| {
        mboostrs_core::support::cell_nearest_distance(
            train_cells.view(),
            train_numeric.view(),
            new_cells.view(),
            new_numeric.view(),
        )
    })
    .map_err(mboostrs_error_to_pyerr)
}

/// Weighted mean of per-row values with cluster-level uncertainty
/// (`mboostrs_core::contrast::cluster_average`): the cluster-robust sandwich
/// standard error, a cluster-bootstrap standard error and percentile
/// interval. Returns `(estimate, se_sandwich, se_bootstrap, lower, upper,
/// clusters, draws)`.
#[pyfunction(signature = (values, weights, clusters, draws=1000, seed=0, level=0.95))]
fn cluster_average(
    py: Python<'_>,
    values: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    clusters: PyReadonlyArray1<'_, i64>,
    draws: usize,
    seed: u64,
    level: f64,
) -> PyResult<(f64, f64, f64, f64, f64, usize, usize)> {
    let values = values.as_array().to_vec();
    let weights = weights.as_array().to_vec();
    let codes = clusters
        .as_array()
        .iter()
        .map(|&c| {
            usize::try_from(c).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(
                    "cluster codes must be non-negative integers",
                )
            })
        })
        .collect::<PyResult<Vec<usize>>>()?;
    let out = py
        .allow_threads(|| {
            mboostrs_core::contrast::cluster_average(&values, &weights, &codes, draws, seed, level)
        })
        .map_err(mboostrs_error_to_pyerr)?;
    Ok((
        out.estimate,
        out.se_sandwich,
        out.se_bootstrap,
        out.lower,
        out.upper,
        out.clusters,
        out.draws,
    ))
}

/// Convert cluster or time codes from a non-negative integer array.
fn residual_codes(clusters: &PyReadonlyArray1<'_, i64>) -> PyResult<Vec<usize>> {
    clusters
        .as_array()
        .iter()
        .map(|&c| {
            usize::try_from(c).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(
                    "cluster codes must be non-negative integers",
                )
            })
        })
        .collect()
}

/// Intra-cluster correlation of residuals with its F and permutation references
/// (`mboostrs_core::residual_structure::cluster_icc`). Returns `(icc,
/// f_statistic, df_between, df_within, f_p_value, permutation_p_value,
/// permutations, clusters, cluster_sizes, cluster_means)`.
#[pyfunction(signature = (residuals, clusters, permutations=999, seed=0))]
#[allow(clippy::type_complexity)]
fn residual_cluster_icc(
    py: Python<'_>,
    residuals: PyReadonlyArray1<'_, f64>,
    clusters: PyReadonlyArray1<'_, i64>,
    permutations: usize,
    seed: u64,
) -> PyResult<(
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    usize,
    usize,
    Vec<usize>,
    Vec<f64>,
)> {
    let residuals = residuals.as_array().to_vec();
    let codes = residual_codes(&clusters)?;
    let out = py
        .allow_threads(|| {
            mboostrs_core::residual_structure::cluster_icc(&residuals, &codes, permutations, seed)
        })
        .map_err(mboostrs_error_to_pyerr)?;
    Ok((
        out.icc,
        out.f_statistic,
        out.df_between,
        out.df_within,
        out.f_p_value,
        out.permutation_p_value,
        out.permutations,
        out.clusters,
        out.cluster_sizes,
        out.cluster_means,
    ))
}

/// Autocorrelation of residuals ordered by time, with the Ljung-Box statistic
/// (`mboostrs_core::residual_structure::serial_dependence`). Returns
/// `(autocorrelation, ljung_box, lags, chisq_p_value, permutation_p_value,
/// permutations, series_length, bound, times)`.
#[pyfunction(signature = (residuals, times, lags, permutations=999, seed=0))]
#[allow(clippy::type_complexity)]
fn residual_serial_dependence(
    py: Python<'_>,
    residuals: PyReadonlyArray1<'_, f64>,
    times: PyReadonlyArray1<'_, f64>,
    lags: usize,
    permutations: usize,
    seed: u64,
) -> PyResult<(Vec<f64>, f64, usize, f64, f64, usize, usize, f64, Vec<f64>)> {
    let residuals = residuals.as_array().to_vec();
    let times = times.as_array().to_vec();
    let out = py
        .allow_threads(|| {
            mboostrs_core::residual_structure::serial_dependence(
                &residuals,
                &times,
                lags,
                permutations,
                seed,
            )
        })
        .map_err(mboostrs_error_to_pyerr)?;
    Ok((
        out.autocorrelation,
        out.ljung_box,
        out.lags,
        out.chisq_p_value,
        out.permutation_p_value,
        out.permutations,
        out.series_length,
        out.bound,
        out.times,
    ))
}

/// Leave-one-out and influence measures of a Gaussian linear smoother
/// (`mboostrs_core::residual_structure::gaussian_influence`). Returns `(df,
/// sigma, loo_residual, cooks, studentized, p_value, flagged)`, with
/// `flagged` as zero-based row indices.
#[pyfunction(signature = (residuals, hat, alpha=0.05))]
#[allow(clippy::type_complexity)]
fn residual_influence(
    py: Python<'_>,
    residuals: PyReadonlyArray1<'_, f64>,
    hat: PyReadonlyArray1<'_, f64>,
    alpha: f64,
) -> PyResult<(f64, f64, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>)> {
    let residuals = residuals.as_array().to_vec();
    let hat = hat.as_array().to_vec();
    let out = py
        .allow_threads(|| {
            mboostrs_core::residual_structure::gaussian_influence(&residuals, &hat, alpha)
        })
        .map_err(mboostrs_error_to_pyerr)?;
    Ok((
        out.df,
        out.sigma,
        out.loo_residual,
        out.cooks,
        out.studentized,
        out.p_value,
        out.flagged,
    ))
}

/// Empirical false-alarm rate of the cluster rule on independent residuals
/// with the given cluster layout (`residual_structure::cluster_false_alarm_rate`).
#[pyfunction(signature = (clusters, draws=200, permutations=199, seed=0, alpha=0.05))]
fn residual_cluster_false_alarm_rate(
    py: Python<'_>,
    clusters: PyReadonlyArray1<'_, i64>,
    draws: usize,
    permutations: usize,
    seed: u64,
    alpha: f64,
) -> PyResult<f64> {
    let codes = residual_codes(&clusters)?;
    py.allow_threads(|| {
        mboostrs_core::residual_structure::cluster_false_alarm_rate(
            &codes,
            draws,
            permutations,
            seed,
            alpha,
        )
    })
    .map_err(mboostrs_error_to_pyerr)
}

/// Empirical false-alarm rate of the serial rule on independent residuals at the
/// given times (`residual_structure::serial_false_alarm_rate`).
#[pyfunction(signature = (times, lags, draws=200, permutations=199, seed=0, alpha=0.05))]
fn residual_serial_false_alarm_rate(
    py: Python<'_>,
    times: PyReadonlyArray1<'_, f64>,
    lags: usize,
    draws: usize,
    permutations: usize,
    seed: u64,
    alpha: f64,
) -> PyResult<f64> {
    let times = times.as_array().to_vec();
    py.allow_threads(|| {
        mboostrs_core::residual_structure::serial_false_alarm_rate(
            &times,
            lags,
            draws,
            permutations,
            seed,
            alpha,
        )
    })
    .map_err(mboostrs_error_to_pyerr)
}

/// Empirical false-alarm rate of the influence rule for a fit with the given
/// hat-matrix diagonal and residual scale (`residual_structure::influence_false_alarm_rate`).
#[pyfunction(signature = (hat, sigma, draws=200, seed=0, alpha=0.05))]
fn residual_influence_false_alarm_rate(
    py: Python<'_>,
    hat: PyReadonlyArray1<'_, f64>,
    sigma: f64,
    draws: usize,
    seed: u64,
    alpha: f64,
) -> PyResult<f64> {
    let hat = hat.as_array().to_vec();
    py.allow_threads(|| {
        mboostrs_core::residual_structure::influence_false_alarm_rate(
            &hat, sigma, draws, seed, alpha,
        )
    })
    .map_err(mboostrs_error_to_pyerr)
}

/// Evaluate one special function of `mboostrs_core::special` elementwise.
///
/// `args` are equal-length arrays (the caller broadcasts): `gamma_p(a, x)`,
/// `beta_i(a, b, x)`, `t_cdf(t, df)`, `t_quantile(p, df)`,
/// `gamma_quantile(p, a)`, `beta_quantile(p, a, b)`, `normal_cdf(z)`,
/// `normal_log_cdf(z)`, `normal_quantile(p)`, `gamma_q(a, x)`,
/// `poisson_cdf(q, mu)`, `nbinom_cdf(q, mu, size)`, `binom_cdf(q, n, prob)`,
/// `binom_quantile(p, n, prob)`, `poisson_quantile_affine(p, mu, a, b)` and
/// `nbinom_quantile_affine(p, mu, size, a, b)` (smallest `k` with
/// `a + b * F(k) >= p`), `inverse_gaussian_cdf(y, mu, sigma)` and
/// `inverse_gaussian_quantile(p, mu, sigma)`. Returns the values and how many
/// elements reached an iteration limit before their tolerance.
#[pyfunction]
fn special_function(
    py: Python<'_>,
    name: &str,
    args: Vec<Vec<f64>>,
) -> PyResult<(Vec<f64>, usize)> {
    use mboostrs_core::special as sp;
    let arity = match name {
        "normal_cdf" | "normal_log_cdf" | "normal_quantile" => 1,
        "gamma_p" | "gamma_q" | "t_cdf" | "t_quantile" | "gamma_quantile" | "poisson_cdf" => 2,
        "beta_i"
        | "beta_quantile"
        | "nbinom_cdf"
        | "binom_cdf"
        | "binom_quantile"
        | "inverse_gaussian_cdf"
        | "inverse_gaussian_quantile" => 3,
        "poisson_quantile_affine" => 4,
        "nbinom_quantile_affine" => 5,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown special function {other:?}"
            )))
        }
    };
    if args.len() != arity {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} takes {arity} argument array(s), got {}",
            args.len()
        )));
    }
    let n = args[0].len();
    if args.iter().any(|a| a.len() != n) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name}: argument arrays must have equal length"
        )));
    }
    let exact = |value: f64| sp::Special {
        value,
        converged: true,
    };
    Ok(py.allow_threads(|| {
        let mut unconverged = 0usize;
        let values = (0..n)
            .map(|i| {
                let s = match name {
                    "normal_cdf" => exact(sp::normal_cdf(args[0][i])),
                    "normal_log_cdf" => exact(sp::normal_log_cdf(args[0][i])),
                    "normal_quantile" => exact(sp::normal_quantile(args[0][i])),
                    "gamma_p" => sp::gamma_p(args[0][i], args[1][i]),
                    "t_cdf" => sp::student_t_cdf(args[0][i], args[1][i]),
                    "t_quantile" => sp::student_t_quantile(args[0][i], args[1][i]),
                    "gamma_quantile" => sp::gamma_quantile(args[0][i], args[1][i]),
                    "gamma_q" => sp::gamma_q(args[0][i], args[1][i]),
                    "inverse_gaussian_cdf" => {
                        exact(sp::inverse_gaussian_cdf(args[0][i], args[1][i], args[2][i]))
                    }
                    "inverse_gaussian_quantile" => {
                        sp::inverse_gaussian_quantile(args[0][i], args[1][i], args[2][i])
                    }
                    "poisson_cdf" => sp::poisson_cdf(args[0][i], args[1][i]),
                    "beta_i" => sp::beta_i(args[0][i], args[1][i], args[2][i]),
                    "nbinom_cdf" => sp::nbinom_cdf(args[0][i], args[1][i], args[2][i]),
                    "binom_cdf" => sp::binom_cdf(args[0][i], args[1][i], args[2][i]),
                    "binom_quantile" => sp::binom_quantile(args[0][i], args[1][i], args[2][i]),
                    "poisson_quantile_affine" => {
                        sp::poisson_quantile_affine(args[0][i], args[1][i], args[2][i], args[3][i])
                    }
                    "nbinom_quantile_affine" => sp::nbinom_quantile_affine(
                        args[0][i], args[1][i], args[2][i], args[3][i], args[4][i],
                    ),
                    _ => sp::beta_quantile(args[0][i], args[1][i], args[2][i]),
                };
                if !s.converged {
                    unconverged += 1;
                }
                s.value
            })
            .collect();
        (values, unconverged)
    }))
}

/// Risk reduction and selection frequency per group of updates; see
/// `mboostrs_core::inference::varimp_attribution`.
#[pyfunction(signature = (changes, groups, n_groups, percent))]
fn varimp_attribution(
    py: Python<'_>,
    changes: Vec<f64>,
    groups: Vec<usize>,
    n_groups: usize,
    percent: bool,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let out = py
        .allow_threads(|| {
            mboostrs_core::inference::varimp_attribution(&changes, &groups, n_groups, percent)
        })
        .map_err(mboostrs_error_to_pyerr)?;
    Ok((out.reduction, out.selfreq))
}

/// Whole-group bootstrap multiplicities, one column per sample; see
/// `mboostrs_core::inference::group_bootstrap_counts`.
#[pyfunction(signature = (codes, eligible, b, seed))]
fn group_bootstrap_counts(
    py: Python<'_>,
    codes: Vec<usize>,
    eligible: Vec<bool>,
    b: usize,
    seed: u64,
) -> PyResult<Py<PyArray2<f64>>> {
    let counts = py
        .allow_threads(|| {
            mboostrs_core::inference::group_bootstrap_counts(&codes, &eligible, b, seed)
        })
        .map_err(mboostrs_error_to_pyerr)?;
    Ok(counts.into_pyarray(py).unbind())
}

/// Failed, untuned and at-cap replicate counts from per-replicate flags.
#[pyfunction(signature = (flags))]
fn bootstrap_counts(flags: Vec<u8>) -> (usize, usize, usize) {
    let c = mboostrs_core::inference::bootstrap_counts(&flags);
    (c.failed, c.untuned, c.at_cap)
}

/// Percentile interval per column of a replicate-by-column matrix; see
/// `mboostrs_core::inference::percentile_interval`.
#[pyfunction(signature = (values, level, centre_rows))]
fn percentile_interval(
    py: Python<'_>,
    values: PyReadonlyArray2<'_, f64>,
    level: f64,
    centre_rows: bool,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let values = values.as_array().to_owned();
    py.allow_threads(|| {
        mboostrs_core::inference::percentile_interval(values.view(), level, centre_rows)
    })
    .map_err(mboostrs_error_to_pyerr)
}

/// Quantile of each row of an LSS family; see `mboostrs_core::lss_quantile`.
/// Returns the values and the number of rows that did not converge.
#[pyfunction(signature = (family, columns))]
fn lss_quantile(
    py: Python<'_>,
    family: &str,
    columns: Vec<Vec<f64>>,
) -> PyResult<(Vec<f64>, usize)> {
    let out = py
        .allow_threads(|| mboostrs_core::lss_quantile::lss_quantile(family, &columns))
        .map_err(mboostrs_error_to_pyerr)?;
    Ok((out.values, out.unconverged))
}

/// Build a Matérn radial design and its retained whitening recipe.
#[pyfunction(signature = (x, knots=100, theta=None))]
fn brad_basis(
    py: Python<'_>,
    x: PyReadonlyArray2<'_, f64>,
    knots: usize,
    theta: Option<f64>,
) -> PyResult<(
    Py<PyArray2<f64>>,
    Py<PyArray2<f64>>,
    Py<PyArray2<f64>>,
    Py<PyArray2<f64>>,
    f64,
)> {
    let x = x.as_array().to_owned();
    let (recipe, design) = py
        .allow_threads(|| {
            let recipe = mboostrs_core::radial::RadialBasis::new(x.view(), knots, theta)?;
            let design = recipe.design(x.view())?;
            Ok::<_, mboostrs_core::MboostrsError>((recipe, design))
        })
        .map_err(mboostrs_error_to_pyerr)?;
    let penalty = ndarray::Array2::eye(recipe.centres.nrows());
    Ok((
        design.into_pyarray(py).unbind(),
        penalty.into_pyarray(py).unbind(),
        recipe.centres.into_pyarray(py).unbind(),
        recipe.whitening.into_pyarray(py).unbind(),
        recipe.theta,
    ))
}

/// Evaluate a retained radial recipe without selecting new centres.
#[pyfunction]
fn brad_design(
    py: Python<'_>,
    x: PyReadonlyArray2<'_, f64>,
    centres: PyReadonlyArray2<'_, f64>,
    whitening: PyReadonlyArray2<'_, f64>,
    theta: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let x = x.as_array().to_owned();
    let recipe = mboostrs_core::radial::RadialBasis {
        centres: centres.as_array().to_owned(),
        whitening: whitening.as_array().to_owned(),
        theta,
    };
    let design = py
        .allow_threads(|| recipe.design(x.view()))
        .map_err(mboostrs_error_to_pyerr)?;
    Ok(design.into_pyarray(py).unbind())
}

/// Matérn 3/2 correlations for a vector of distances.
#[pyfunction]
fn brad_covariance(distance: Vec<f64>, theta: f64) -> PyResult<Vec<f64>> {
    if !theta.is_finite() || theta <= 0.0 || distance.iter().any(|d| !d.is_finite() || *d < 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "brad distances must be finite and nonnegative; theta must be positive",
        ));
    }
    Ok(distance
        .into_iter()
        .map(|d| mboostrs_core::radial::matern32(d, theta))
        .collect())
}

/// Effective Matérn range, with pair distances streamed in the core.
#[pyfunction(signature = (x, eps=1e-3))]
fn brad_effective_range(py: Python<'_>, x: PyReadonlyArray2<'_, f64>, eps: f64) -> PyResult<f64> {
    let x = x.as_array().to_owned();
    py.allow_threads(|| mboostrs_core::radial::effective_range(x.view(), eps))
        .map_err(mboostrs_error_to_pyerr)
}

/// Material in-bag risk increases: `(count, largest step rise, final rise
/// above the minimum, best step)`. The core distinguishes profiled likelihoods
/// from changing adaptive losses. `None` means no material rise to report.
#[pyfunction]
fn risk_path_rises(
    family: &str,
    params: HashMap<String, f64>,
    risk: Vec<f64>,
) -> PyResult<Option<(usize, f64, f64, usize)>> {
    let fam = make_family(family, &params).map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(
        mboostrs_core::path::risk_rise_warning(fam.as_ref(), &risk).map(|warning| {
            (
                warning.count,
                warning.largest_step_rise,
                warning.final_above_best,
                warning.best_step,
            )
        }),
    )
}

/// mgcv's concurvity measures (`worst`, `observed`, `estimate`) of each
/// term `spans[i] = (start, end)` of a model matrix `x` at coefficients
/// `beta`; see `mboostrs_core::gam::concurvity`. Exposed so the computation
/// can be checked against mgcv on mgcv's own model matrix.
#[pyfunction]
fn concurvity_measures(
    x: numpy::PyReadonlyArray2<'_, f64>,
    spans: Vec<(usize, usize)>,
    beta: PyReadonlyArray1<'_, f64>,
) -> PyResult<Vec<(f64, f64, f64)>> {
    let x = x.as_array().to_owned();
    let beta = beta.as_array().to_owned();
    if beta.len() != x.ncols() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "beta must have one entry per column of x",
        ));
    }
    if spans.iter().any(|&(a, b)| a >= b || b > x.ncols()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "each span must satisfy start < end <= ncol(x)",
        ));
    }
    Ok(
        mboostrs_core::gam::concurvity(x.view(), &spans, beta.view())
            .into_iter()
            .map(|c| (c.worst, c.observed, c.estimate))
            .collect(),
    )
}

/// Weighted mean family loss of link-scale predictions on held-out rows.
///
/// This is the number `cvrisk` reports for a fold, computed once on a given
/// frame instead of over a refit path: the family's `held_out_risk` on the
/// encoded response and link-scale predictor, divided by the weight sum, with
/// the fitted model's nuisance parameters restored first (scalar then vector,
/// the same order the CV engine restores them in).
#[pyfunction]
#[pyo3(signature = (family, params, y, eta, weights, nuisance=None, nuisance_vec=None))]
fn family_held_out_score(
    family: &str,
    params: HashMap<String, f64>,
    y: PyReadonlyArray1<'_, f64>,
    eta: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    nuisance: Option<f64>,
    nuisance_vec: Option<Vec<f64>>,
) -> PyResult<f64> {
    use ndarray::Array1;

    let y_vec = y.as_array().to_vec();
    let eta_vec = eta.as_array().to_vec();
    let w_vec = weights.as_array().to_vec();
    let n = eta_vec.len();
    if w_vec.len() != n {
        return Err(mboostrs_error_to_pyerr(MboostrsError::Shape(
            "eta and weights must have the same length".into(),
        )));
    }
    let w_sum: f64 = w_vec.iter().sum();
    if !w_sum.is_finite() || w_sum <= 0.0 {
        return Err(mboostrs_error_to_pyerr(MboostrsError::Shape(
            "weights must sum to a positive finite value".into(),
        )));
    }
    let mut fam = make_family(family, &params)
        .map_err(|e| mboostrs_error_to_pyerr(MboostrsError::Control(e)))?;
    let y_arr = Array1::from(y_vec);
    // A survival response is [time, event] (length 2n) against n predictor
    // entries; any other family takes one response entry per predictor.
    let survival_layout = y_arr.len() == 2 * n && fam.n_from_y(y_arr.view()) == n;
    if y_arr.len() != n && !survival_layout {
        return Err(mboostrs_error_to_pyerr(MboostrsError::Shape(
            "y must have one entry per predictor, or [time, event] (2n entries) for a survival family".into(),
        )));
    }
    fam.check_y(y_arr.view()).map_err(mboostrs_error_to_pyerr)?;
    let encoded = fam
        .encode_response(y_arr.view())
        .unwrap_or_else(|| y_arr.clone());
    if let Some(value) = nuisance {
        fam.set_nuisance(value);
    }
    if let Some(values) = nuisance_vec {
        if !values.is_empty() {
            fam.set_nuisance_vec(values);
        }
    }
    let raw = fam.held_out_risk(
        encoded.view(),
        Array1::from(eta_vec).view(),
        Array1::from(w_vec).view(),
    );
    Ok(raw / w_sum)
}

/// Each held-out row's family loss at the fit's nuisance value, unweighted;
/// `None` for a family whose loss couples rows (CoxPH, AUC, Cindex, Gehan).
/// The weighted mean of these is `family_held_out_score`.
#[pyfunction]
#[pyo3(signature = (family, params, y, eta, nuisance=None, nuisance_vec=None))]
fn family_held_out_rows(
    family: &str,
    params: HashMap<String, f64>,
    y: PyReadonlyArray1<'_, f64>,
    eta: PyReadonlyArray1<'_, f64>,
    nuisance: Option<f64>,
    nuisance_vec: Option<Vec<f64>>,
) -> PyResult<Option<Vec<f64>>> {
    let mut fam = make_family(family, &params)
        .map_err(|e| mboostrs_error_to_pyerr(MboostrsError::Control(e)))?;
    let y_arr = y.as_array().to_owned();
    fam.check_y(y_arr.view()).map_err(mboostrs_error_to_pyerr)?;
    let encoded = fam
        .encode_response(y_arr.view())
        .unwrap_or_else(|| y_arr.clone());
    if let Some(value) = nuisance {
        fam.set_nuisance(value);
    }
    if let Some(values) = nuisance_vec {
        if !values.is_empty() {
            fam.set_nuisance_vec(values);
        }
    }
    Ok(mboostrs_core::held_out_row_losses(
        fam.as_ref(),
        encoded.view(),
        eta.as_array(),
    ))
}

#[pyfunction]
#[pyo3(signature = (y, predictions, weights, metric, parameter=None))]
fn response_score_path(
    y: PyReadonlyArray1<'_, f64>,
    predictions: PyReadonlyArray2<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    metric: &str,
    parameter: Option<f64>,
) -> PyResult<Vec<f64>> {
    mboostrs_core::scoring::response_score_path(
        &y.as_array().to_vec(),
        predictions.as_array(),
        &weights.as_array().to_vec(),
        metric,
        parameter,
    )
    .map_err(mboostrs_error_to_pyerr)
}

#[pyfunction]
fn inverse_response_transform(values: PyReadonlyArray1<'_, f64>, kind: &str) -> PyResult<Vec<f64>> {
    mboostrs_core::scoring::inverse_response_transform(&values.as_array().to_vec(), kind)
        .map_err(mboostrs_error_to_pyerr)
}

// Conformal interval numerics; see mboostrs_core::conformal.

#[pyfunction]
fn conformal_min_calibration_size(level: f64) -> PyResult<usize> {
    mboostrs_core::conformal::min_calibration_size(level).map_err(mboostrs_error_to_pyerr)
}

#[pyfunction]
fn conformal_calibration_count(n: usize, fraction: f64) -> PyResult<usize> {
    mboostrs_core::conformal::calibration_count(n, fraction).map_err(mboostrs_error_to_pyerr)
}

/// Fit the model-based imputation of one column. Returns the parameter vector
/// the recipe stores and the R-squared of the conditional mean.
#[pyfunction]
fn impute_model_fit(
    target: PyReadonlyArray1<'_, f64>,
    x: PyReadonlyArray2<'_, f64>,
    levels: Vec<usize>,
) -> PyResult<(Vec<f64>, f64)> {
    let model =
        mboostrs_core::impute::ImputationModel::fit(target.as_array(), x.as_array(), &levels)
            .map_err(mboostrs_error_to_pyerr)?;
    Ok((model.to_params(), model.r2))
}

/// Predicted values of an imputed column from its stored parameter vector.
#[pyfunction]
fn impute_model_predict(params: Vec<f64>, x: PyReadonlyArray2<'_, f64>) -> PyResult<Vec<f64>> {
    let model = mboostrs_core::impute::ImputationModel::from_params(&params)
        .map_err(mboostrs_error_to_pyerr)?;
    model.predict(x.as_array()).map_err(mboostrs_error_to_pyerr)
}

#[pyfunction]
fn conformal_quantile(scores: PyReadonlyArray1<'_, f64>, level: f64) -> PyResult<f64> {
    mboostrs_core::conformal::conformal_quantile(&scores.as_array().to_vec(), level)
        .map_err(mboostrs_error_to_pyerr)
}

#[pyfunction]
fn conformal_worst_covariate_bin(
    x: PyReadonlyArray1<'_, f64>,
    lower: PyReadonlyArray1<'_, f64>,
    upper: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    level: f64,
    bins: usize,
) -> PyResult<(f64, usize, f64, usize)> {
    let worst = mboostrs_core::conformal::worst_covariate_bin(
        &x.as_array().to_vec(),
        &lower.as_array().to_vec(),
        &upper.as_array().to_vec(),
        &y.as_array().to_vec(),
        level,
        bins,
    )
    .map_err(mboostrs_error_to_pyerr)?;
    Ok((worst.z, worst.bin, worst.coverage, worst.count))
}

#[pyfunction]
fn conformal_absolute_residuals(
    y: PyReadonlyArray1<'_, f64>,
    predicted: PyReadonlyArray1<'_, f64>,
) -> PyResult<Vec<f64>> {
    mboostrs_core::conformal::absolute_residuals(
        &y.as_array().to_vec(),
        &predicted.as_array().to_vec(),
    )
    .map_err(mboostrs_error_to_pyerr)
}

#[pyfunction]
fn conformal_location_scale_scores(
    y: PyReadonlyArray1<'_, f64>,
    mu: PyReadonlyArray1<'_, f64>,
    sigma: PyReadonlyArray1<'_, f64>,
) -> PyResult<Vec<f64>> {
    mboostrs_core::conformal::location_scale_scores(
        &y.as_array().to_vec(),
        &mu.as_array().to_vec(),
        &sigma.as_array().to_vec(),
    )
    .map_err(mboostrs_error_to_pyerr)
}

#[pyfunction]
fn conformal_location_scale_band(
    mu: PyReadonlyArray1<'_, f64>,
    sigma: PyReadonlyArray1<'_, f64>,
    q: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    mboostrs_core::conformal::location_scale_band(
        &mu.as_array().to_vec(),
        &sigma.as_array().to_vec(),
        q,
    )
    .map_err(mboostrs_error_to_pyerr)
}

#[pyfunction]
fn conformal_symmetric_band(predicted: PyReadonlyArray1<'_, f64>, q: f64) -> (Vec<f64>, Vec<f64>) {
    mboostrs_core::conformal::symmetric_band(&predicted.as_array().to_vec(), q)
}

#[pyfunction]
fn conformal_poisson_scores(
    y: PyReadonlyArray1<'_, f64>,
    predicted: PyReadonlyArray1<'_, f64>,
    level: f64,
    score: &str,
) -> PyResult<Vec<f64>> {
    mboostrs_core::conformal::poisson_scores(
        &y.as_array().to_vec(),
        &predicted.as_array().to_vec(),
        level,
        score,
    )
    .map_err(mboostrs_error_to_pyerr)
}

#[pyfunction]
fn conformal_poisson_band(
    predicted: PyReadonlyArray1<'_, f64>,
    level: f64,
    q: f64,
    score: &str,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    mboostrs_core::conformal::poisson_band(&predicted.as_array().to_vec(), level, q, score)
        .map_err(mboostrs_error_to_pyerr)
}

#[pyfunction]
fn conformal_ordered_pair(
    a: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    mboostrs_core::conformal::ordered_pair(&a.as_array().to_vec(), &b.as_array().to_vec())
        .map_err(mboostrs_error_to_pyerr)
}

#[pyfunction]
fn rearrange_levels<'py>(
    py: Python<'py>,
    predictions: PyReadonlyArray2<'_, f64>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, usize, f64, Vec<usize>, f64)> {
    let r = mboostrs_core::rearrange::rearrange_levels(predictions.as_array())
        .map_err(mboostrs_error_to_pyerr)?;
    Ok((
        r.sorted.into_pyarray(py),
        r.crossing_rows,
        r.crossing_share,
        r.pair_crossings,
        r.max_crossing,
    ))
}

#[pyfunction]
fn conformal_count_default_score(family: &str, y: PyReadonlyArray1<'_, f64>) -> String {
    mboostrs_core::count_conformal::default_score(family, &y.as_array().to_vec()).to_owned()
}

#[pyfunction]
fn conformal_count_reference(
    y: PyReadonlyArray1<'_, f64>,
    predicted: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    zero_inflated: bool,
) -> PyResult<(f64, f64, String)> {
    let (pi, ratio) = mboostrs_core::count_conformal::fit_reference(
        &y.as_array().to_vec(),
        &predicted.as_array().to_vec(),
        &weights.as_array().to_vec(),
        zero_inflated,
    )
    .map_err(mboostrs_error_to_pyerr)?;
    Ok((
        pi,
        ratio,
        mboostrs_core::count_conformal::automatic_score(ratio).to_owned(),
    ))
}

#[pyfunction]
#[pyo3(signature=(n, seed=None, stream=0, offset=0))]
fn conformal_count_uniforms(
    n: usize,
    seed: Option<u64>,
    stream: u64,
    offset: usize,
) -> PyResult<(u64, Vec<f64>)> {
    mboostrs_core::count_conformal::uniforms(n, seed, stream, offset)
        .map_err(mboostrs_error_to_pyerr)
}

#[pyfunction]
fn conformal_count_rank_scores(
    y: PyReadonlyArray1<'_, f64>,
    predicted: PyReadonlyArray1<'_, f64>,
    zero_probability: f64,
    uniforms: PyReadonlyArray1<'_, f64>,
    level: f64,
) -> PyResult<Vec<f64>> {
    mboostrs_core::count_conformal::scores(
        &y.as_array().to_vec(),
        &predicted.as_array().to_vec(),
        zero_probability,
        &uniforms.as_array().to_vec(),
        level,
    )
    .map_err(mboostrs_error_to_pyerr)
}

#[pyfunction]
fn conformal_count_rank_band(
    predicted: PyReadonlyArray1<'_, f64>,
    zero_probability: f64,
    q: f64,
    uniforms: PyReadonlyArray1<'_, f64>,
    level: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    mboostrs_core::count_conformal::band(
        &predicted.as_array().to_vec(),
        zero_probability,
        q,
        &uniforms.as_array().to_vec(),
        level,
    )
    .map_err(mboostrs_error_to_pyerr)
}

#[pyfunction]
fn conformal_count_cardinalities(
    lower: PyReadonlyArray1<'_, f64>,
    upper: PyReadonlyArray1<'_, f64>,
) -> PyResult<Vec<f64>> {
    mboostrs_core::count_conformal::cardinalities(
        &lower.as_array().to_vec(),
        &upper.as_array().to_vec(),
    )
    .map_err(mboostrs_error_to_pyerr)
}

#[pyfunction]
fn conformal_quantile_scores(
    lower: PyReadonlyArray1<'_, f64>,
    upper: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
) -> PyResult<Vec<f64>> {
    mboostrs_core::conformal::quantile_scores(
        &lower.as_array().to_vec(),
        &upper.as_array().to_vec(),
        &y.as_array().to_vec(),
    )
    .map_err(mboostrs_error_to_pyerr)
}

#[pyfunction]
fn conformal_widen_band(
    lower: PyReadonlyArray1<'_, f64>,
    upper: PyReadonlyArray1<'_, f64>,
    q: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    mboostrs_core::conformal::widen_band(&lower.as_array().to_vec(), &upper.as_array().to_vec(), q)
        .map_err(mboostrs_error_to_pyerr)
}

#[pyfunction]
fn conformal_band_midpoint(
    lower: PyReadonlyArray1<'_, f64>,
    upper: PyReadonlyArray1<'_, f64>,
) -> PyResult<Vec<f64>> {
    mboostrs_core::conformal::band_midpoint(&lower.as_array().to_vec(), &upper.as_array().to_vec())
        .map_err(mboostrs_error_to_pyerr)
}

#[pyfunction]
fn conformal_clip_to_band(
    values: PyReadonlyArray1<'_, f64>,
    lower: PyReadonlyArray1<'_, f64>,
    upper: PyReadonlyArray1<'_, f64>,
) -> PyResult<Vec<f64>> {
    mboostrs_core::conformal::clip_to_band(
        &values.as_array().to_vec(),
        &lower.as_array().to_vec(),
        &upper.as_array().to_vec(),
    )
    .map_err(mboostrs_error_to_pyerr)
}

#[pyfunction]
fn conformal_cv_plus_ranks(n: usize, level: f64) -> PyResult<(usize, usize)> {
    mboostrs_core::conformal::cv_plus_ranks(n, level).map_err(mboostrs_error_to_pyerr)
}

#[pyfunction]
fn conformal_cv_plus_guaranteed_coverage(n: usize, k: usize, level: f64) -> PyResult<Option<f64>> {
    mboostrs_core::conformal::cv_plus_guaranteed_coverage(n, k, level)
        .map_err(mboostrs_error_to_pyerr)
}

/// CV+ endpoints. `fold_of` holds one fold index per calibrated row (int64)
/// and `fold_predictions` is the (folds x targets) matrix of out-of-fold
/// predictions.
#[pyfunction]
fn conformal_cv_plus_bounds(
    residuals: PyReadonlyArray1<'_, f64>,
    fold_of: PyReadonlyArray1<'_, i64>,
    fold_predictions: PyReadonlyArray2<'_, f64>,
    level: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let folds: Vec<usize> = fold_of
        .as_array()
        .iter()
        .map(|&f| {
            usize::try_from(f).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err("fold indices must be non-negative")
            })
        })
        .collect::<PyResult<_>>()?;
    mboostrs_core::conformal::cv_plus_bounds(
        &residuals.as_array().to_vec(),
        &folds,
        fold_predictions.as_array(),
        level,
    )
    .map_err(mboostrs_error_to_pyerr)
}

#[pyfunction]
fn conformal_tuned_mstop_start(model_mstop: usize) -> usize {
    mboostrs_core::conformal::tuned_mstop_start(model_mstop)
}

#[pyfunction]
fn conformal_tuned_mstop_choice(optimal: usize) -> usize {
    mboostrs_core::conformal::tuned_mstop_choice(optimal)
}

#[pyfunction]
fn conformal_tuned_mstop_next(largest: usize, cap: usize) -> usize {
    mboostrs_core::conformal::tuned_mstop_next(largest, cap)
}

#[pyfunction]
fn select_score_path(scores: PyReadonlyArray2<'_, f64>) -> PyResult<(Vec<f64>, usize)> {
    mboostrs_core::scoring::select_score_path(scores.as_array()).map_err(mboostrs_error_to_pyerr)
}

#[pyfunction]
fn crps_normal(
    y: PyReadonlyArray1<'_, f64>,
    mu: PyReadonlyArray1<'_, f64>,
    sigma: PyReadonlyArray1<'_, f64>,
) -> PyResult<Vec<f64>> {
    mboostrs_core::scoring::crps_normal(
        &y.as_array().to_vec(),
        &mu.as_array().to_vec(),
        &sigma.as_array().to_vec(),
    )
    .map_err(mboostrs_error_to_pyerr)
}

#[pyfunction]
fn crps_discrete_sum(
    y: PyReadonlyArray1<'_, f64>,
    cdf: PyReadonlyArray2<'_, f64>,
) -> PyResult<Vec<f64>> {
    mboostrs_core::scoring::crps_discrete_sum(&y.as_array().to_vec(), cdf.as_array())
        .map_err(mboostrs_error_to_pyerr)
}

#[pyfunction]
fn crps_cdf_quadrature(
    cdf_below: PyReadonlyArray2<'_, f64>,
    cdf_above: PyReadonlyArray2<'_, f64>,
    step_below: PyReadonlyArray1<'_, f64>,
    step_above: PyReadonlyArray1<'_, f64>,
) -> PyResult<Vec<f64>> {
    mboostrs_core::scoring::crps_cdf_quadrature(
        cdf_below.as_array(),
        cdf_above.as_array(),
        &step_below.as_array().to_vec(),
        &step_above.as_array().to_vec(),
    )
    .map_err(mboostrs_error_to_pyerr)
}

/// Weighted AUC per score column and its DeLong covariance matrix.
#[pyfunction]
fn auc_delong(
    observed: PyReadonlyArray1<'_, f64>,
    scores: PyReadonlyArray2<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
) -> PyResult<(Vec<f64>, Vec<Vec<f64>>)> {
    let (auc, covariance) = mboostrs_core::scoring::auc_delong(
        &observed.as_array().to_vec(),
        scores.as_array(),
        &weights.as_array().to_vec(),
    )
    .map_err(mboostrs_error_to_pyerr)?;
    let rows = covariance.outer_iter().map(|row| row.to_vec()).collect();
    Ok((auc, rows))
}

/// CV over frontend recipes rebuilt inside each training fold.
#[pyfunction]
fn cvrisk_foldwise(
    py: Python<'_>,
    family_name: &str,
    family_params: HashMap<String, f64>,
    n_folds: usize,
    mstop: usize,
    provider: Py<PyAny>,
) -> PyResult<(Vec<Vec<f64>>, Vec<f64>, Vec<usize>, usize)> {
    use mboostrs_core::cv;
    let family = make_family(family_name, &family_params)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let mut callback_error = None;
    let result = cv::try_cvrisk_foldwise(family.as_ref(), n_folds, mstop, |b| {
        let supplied = (|| -> PyResult<cv::CvPredictionFold> {
            let data = provider.call1(py, (b,))?;
            let (y, w, eta, nuisance): (
                Vec<f64>,
                Vec<f64>,
                PyReadonlyArray2<'_, f64>,
                Vec<Vec<f64>>,
            ) = data.extract(py)?;
            Ok(cv::CvPredictionFold {
                response: ndarray::Array1::from_vec(y),
                weights: ndarray::Array1::from_vec(w),
                predictions: eta.as_array().to_owned(),
                nuisance,
            })
        })();
        supplied.map_err(|error| {
            callback_error = Some(error);
            MboostrsError::Control(format!("CV fold {b} could not rebuild its preprocessing"))
        })
    });
    if let Some(error) = callback_error {
        return Err(error);
    }
    let result = result.map_err(mboostrs_error_to_pyerr)?;
    let optimum = result.optimal_mstop();
    Ok((
        result
            .risk_matrix
            .outer_iter()
            .map(|row| row.to_vec())
            .collect(),
        result.aggregate.to_vec(),
        result.grid,
        optimum,
    ))
}

#[pyfunction]
fn group_components(codes: PyReadonlyArray2<'_, f64>) -> PyResult<Vec<usize>> {
    mboostrs_core::cv::group_components(codes.as_array()).map_err(mboostrs_error_to_pyerr)
}

fn parse_split_selection(value: &str) -> PyResult<mboostrs_core::learners::SplitSelection> {
    use mboostrs_core::learners::SplitSelection;
    match value {
        "sse" => Ok(SplitSelection::Sse),
        "ctree" => Ok(SplitSelection::CtreeStatistic),
        "ctree_pvalue" => Ok(SplitSelection::CtreePvalue),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "btree selection must be 'sse', 'ctree' or 'ctree_pvalue', got {other:?}"
        ))),
    }
}

fn parse_split_method(value: &str) -> PyResult<mboostrs_core::learners::SplitMethod> {
    use mboostrs_core::learners::SplitMethod;
    match value {
        "exact" => Ok(SplitMethod::Exact),
        "histogram" => Ok(SplitMethod::Histogram),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "btree split_method must be 'exact' or 'histogram', got {other:?}"
        ))),
    }
}

/// Preliminary tree exploration; the complete numerical pipeline lives in Rust.
#[pyfunction]
#[pyo3(signature = (x, y, categorical, transforms, engine="native", seed=0, rounds=300, pairs=true))]
fn preexplore_json(
    py: Python<'_>,
    x: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    categorical: Vec<usize>,
    transforms: Vec<u8>,
    engine: &str,
    seed: u64,
    rounds: u32,
    pairs: bool,
) -> PyResult<String> {
    let x = x.as_array().to_owned();
    let y = y.as_array().to_vec();
    py.allow_threads(|| {
        mboostrs_core::preexplore::preexplore(
            x.view(),
            &y,
            &categorical,
            &transforms,
            engine,
            seed,
            rounds,
            pairs,
        )
    })
    .map(|v| v.to_string())
    .map_err(mboostrs_error_to_pyerr)
}

/// Screening diagnostics of `y` against one column
/// (`mboostrs_core::screen::column_diagnostics`).
#[pyfunction]
fn column_diagnostics(
    py: Python<'_>,
    x: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let d =
        mboostrs_core::screen::column_diagnostics(&x.as_array().to_vec(), &y.as_array().to_vec())
            .map_err(mboostrs_error_to_pyerr)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("distinct", d.distinct)?;
    out.set_item("integer_like", d.integer_like)?;
    out.set_item("skewness", d.skewness)?;
    out.set_item("pearson", d.pearson)?;
    out.set_item("spearman", d.spearman)?;
    out.set_item("linear_r2", d.linear_r2)?;
    out.set_item("curve_r2", d.curve_r2)?;
    out.set_item("step_r2", d.step_r2)?;
    out.set_item("curvature", d.curvature())?;
    out.set_item("step_share", d.step_share())?;
    Ok(out.unbind())
}

/// Covariance and absolute shares of one learner's contribution in a
/// prediction (`mboostrs_core::contribution::contribution_shares`, #124).
#[pyfunction]
fn contribution_shares(
    part: PyReadonlyArray1<'_, f64>,
    rest: PyReadonlyArray1<'_, f64>,
) -> PyResult<(f64, f64)> {
    let s = mboostrs_core::contribution::contribution_shares(
        &part.as_array().to_vec(),
        &rest.as_array().to_vec(),
    )
    .map_err(mboostrs_error_to_pyerr)?;
    Ok((s.covariance_share, s.absolute_share))
}

/// Mean and standard error of paired fold-score differences
/// (`mboostrs_core::contribution::paired_mean_se`, #124).
#[pyfunction]
fn discovery_learning_rate(family: &str) -> f64 {
    mboostrs_core::control::discovery_learning_rate(family)
}

#[pyfunction]
fn discovery_mean_step(family: &str) -> &'static str {
    mboostrs_core::control::discovery_mean_step(family)
}

#[pyfunction]
fn prefer_simpler_cv(
    simple: PyReadonlyArray1<'_, f64>,
    rich: PyReadonlyArray1<'_, f64>,
    multiple: f64,
) -> PyResult<(bool, f64, f64)> {
    mboostrs_core::contribution::prefer_simpler_cv(
        &simple.as_array().to_vec(),
        &rich.as_array().to_vec(),
        multiple,
    )
    .map_err(mboostrs_error_to_pyerr)
}

#[pyfunction]
fn paired_mean_se(
    first: PyReadonlyArray1<'_, f64>,
    second: PyReadonlyArray1<'_, f64>,
) -> PyResult<(f64, f64)> {
    mboostrs_core::contribution::paired_mean_se(
        &first.as_array().to_vec(),
        &second.as_array().to_vec(),
    )
    .map_err(mboostrs_error_to_pyerr)
}

/// Share of `y`'s variance left unexplained by `fitted`
/// (`mboostrs_core::contribution::leftover_share`, #124).
#[pyfunction]
fn leftover_share(
    y: PyReadonlyArray1<'_, f64>,
    fitted: PyReadonlyArray1<'_, f64>,
) -> PyResult<f64> {
    mboostrs_core::contribution::leftover_share(&y.as_array().to_vec(), &fitted.as_array().to_vec())
        .map_err(mboostrs_error_to_pyerr)
}

/// Friedman's H^2 and the interaction variance of a pair from its partial
/// dependences (`mboostrs_core::screen::h_statistic_with_variance`).
#[pyfunction]
fn h_statistic(
    pd_jk: PyReadonlyArray1<'_, f64>,
    pd_j: PyReadonlyArray1<'_, f64>,
    pd_k: PyReadonlyArray1<'_, f64>,
) -> PyResult<(f64, f64)> {
    mboostrs_core::screen::h_statistic_with_variance(
        &pd_jk.as_array().to_vec(),
        &pd_j.as_array().to_vec(),
        &pd_k.as_array().to_vec(),
    )
    .map_err(mboostrs_error_to_pyerr)
}

/// Rank candidate interaction pairs by the held-out residual MSE their
/// binned cell means remove (`mboostrs_core::screen::screen_pairs`,
/// added for #298 phase 1, 2026-10-08).
///
/// Both columns of a pair are binned into `bins` quantile cells with the
/// edges learned on the fit half only; a column with
/// `factor_levels[j] > 0` uses its values as that many level codes
/// instead. Returns `(left, right, removed)` column-index triples sorted
/// by `removed` descending, ties by `(left, right)` ascending. An empty
/// `pairs` list means all column pairs; `top_k=None` keeps every pair.
#[pyfunction]
#[pyo3(signature = (x, residual, bins, seed, factor_levels, pairs, top_k=None))]
fn screen_pairs(
    py: Python<'_>,
    x: PyReadonlyArray2<'_, f64>,
    residual: PyReadonlyArray1<'_, f64>,
    bins: usize,
    seed: u64,
    factor_levels: Vec<usize>,
    pairs: Vec<(usize, usize)>,
    top_k: Option<usize>,
) -> PyResult<Vec<(usize, usize, f64)>> {
    let x = x.as_array().to_owned();
    let residual = residual.as_array().to_vec();
    py.allow_threads(|| {
        mboostrs_core::screen::screen_pairs(
            x.view(),
            &residual,
            bins,
            seed,
            &factor_levels,
            &pairs,
            top_k,
        )
    })
    .map(|scored| {
        scored
            .into_iter()
            .map(|pair| (pair.left, pair.right, pair.removed))
            .collect()
    })
    .map_err(mboostrs_error_to_pyerr)
}

/// Rank columns by the held-out residual MSE their 1-D bin means remove
/// (`mboostrs_core::screen::screen_main_effects`, added for #298 phase 1,
/// 2026-10-08) — the residual-aware counterpart of `column_diagnostics`.
///
/// Returns `(column, removed)` index pairs sorted by `removed` descending,
/// ties by column ascending; `top_k=None` keeps every column.
#[pyfunction]
#[pyo3(signature = (x, residual, bins, seed, factor_levels, top_k=None))]
fn screen_main_effects(
    py: Python<'_>,
    x: PyReadonlyArray2<'_, f64>,
    residual: PyReadonlyArray1<'_, f64>,
    bins: usize,
    seed: u64,
    factor_levels: Vec<usize>,
    top_k: Option<usize>,
) -> PyResult<Vec<(usize, f64)>> {
    let x = x.as_array().to_owned();
    let residual = residual.as_array().to_vec();
    py.allow_threads(|| {
        mboostrs_core::screen::screen_main_effects(
            x.view(),
            &residual,
            bins,
            seed,
            &factor_levels,
            top_k,
        )
    })
    .map(|scored| {
        scored
            .into_iter()
            .map(|main| (main.column, main.removed))
            .collect()
    })
    .map_err(mboostrs_error_to_pyerr)
}

/// Evaluate every `u > c` step candidate against a smooth background
/// (`mboostrs_core::disc::disc_scan`, the kernel of
/// `mboostrs.terms.screen_discontinuities`).
///
/// `u` is the projected score, `background` the smooth part of the design
/// (n × q), `target` the response or response-scale residual, `fit` the
/// 0/1 fit-half membership, `cuts` the candidate thresholds on `u` and
/// `min_side` the fewest rows a side of either half may hold. For each cut
/// the step model `[background, 1{u > c}]` is fitted on the fit half
/// (`jump_fit`, `t_fit`) and evaluated on the other half (`jump`, `se`,
/// `z`), with the held-out `gain` of the step model over the fit-half base
/// model. Returns `(threshold, n_above, jump_fit, t_fit, jump, se, z,
/// gain)` rows in cut order; candidates with too few rows on a side are
/// skipped, and a singular fit-half background raises.
#[pyfunction]
#[pyo3(signature = (u, background, target, fit, cuts, min_side))]
fn disc_scan(
    py: Python<'_>,
    u: PyReadonlyArray1<'_, f64>,
    background: PyReadonlyArray2<'_, f64>,
    target: PyReadonlyArray1<'_, f64>,
    fit: PyReadonlyArray1<'_, u8>,
    cuts: PyReadonlyArray1<'_, f64>,
    min_side: usize,
) -> PyResult<Vec<(f64, usize, f64, f64, f64, f64, f64, f64)>> {
    let u = u.as_array().to_vec();
    let background = background.as_array().to_owned();
    let target = target.as_array().to_vec();
    let fit = fit.as_array().to_vec();
    let cuts = cuts.as_array().to_vec();
    py.allow_threads(|| {
        mboostrs_core::disc::disc_scan(&u, background.view(), &target, &fit, &cuts, min_side)
    })
    .map(|rows| {
        rows.into_iter()
            .map(|r| {
                (
                    r.threshold,
                    r.n_above,
                    r.jump_fit,
                    r.t_fit,
                    r.jump,
                    r.se,
                    r.z,
                    r.gain,
                )
            })
            .collect()
    })
    .map_err(mboostrs_error_to_pyerr)
}

/// The discontinuity typing statistic (`mboostrs_core::disc::disc_step_z`):
/// the t-statistic of the step indicator in `[background, indicator]`
/// fitted on the eval-half rows, which stays exactly calibrated because
/// candidate selection used fit-half statistics only.
///
/// `eval` is the 0/1 eval-half membership. Returns `None` when the small
/// system is not positive definite, the residual variance is zero, or there
/// are fewer eval rows than columns; the caller records a missing
/// `type_z` for such a candidate.
#[pyfunction]
#[pyo3(signature = (background, indicator, target, eval))]
fn disc_step_z(
    py: Python<'_>,
    background: PyReadonlyArray2<'_, f64>,
    indicator: PyReadonlyArray1<'_, f64>,
    target: PyReadonlyArray1<'_, f64>,
    eval: PyReadonlyArray1<'_, u8>,
) -> PyResult<Option<f64>> {
    let background = background.as_array().to_owned();
    let indicator = indicator.as_array().to_vec();
    let target = target.as_array().to_vec();
    let eval = eval.as_array().to_vec();
    py.allow_threads(|| {
        mboostrs_core::disc::disc_step_z(background.view(), &indicator, &target, &eval)
    })
    .map_err(mboostrs_error_to_pyerr)
}

/// R-loss pseudo-outcomes and weights (`mboostrs_core::causal`).
#[pyfunction]
fn rloss_transform<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'_, f64>,
    t: PyReadonlyArray1<'_, f64>,
    m_hat: PyReadonlyArray1<'_, f64>,
    e_hat: PyReadonlyArray1<'_, f64>,
    clip: (f64, f64),
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, usize)> {
    let own = |a: &PyReadonlyArray1<'_, f64>| a.as_array().to_vec();
    let out = mboostrs_core::causal::rloss_transform(
        &own(&y),
        &own(&t),
        &own(&m_hat),
        &own(&e_hat),
        clip,
    )
    .map_err(mboostrs_error_to_pyerr)?;
    Ok((
        out.pseudo_outcome.into_pyarray(py).unbind(),
        out.weights.into_pyarray(py).unbind(),
        out.n_clipped,
    ))
}

/// Generalised covariance measure test from out-of-fold residuals
/// (`mboostrs_core::causal::gcm_test`): `(statistic, p_value,
/// residual_correlation, n)`.
#[pyfunction]
fn gcm_test(
    r_x: PyReadonlyArray1<'_, f64>,
    r_y: PyReadonlyArray1<'_, f64>,
) -> PyResult<(f64, f64, f64, usize)> {
    let out = mboostrs_core::causal::gcm_test(&r_x.as_array().to_vec(), &r_y.as_array().to_vec())
        .map_err(mboostrs_error_to_pyerr)?;
    Ok((out.statistic, out.p_value, out.residual_correlation, out.n))
}

/// Multivariate GCM test (`mboostrs_core::causal::gcm_test_multi`) for a
/// row-major `(n, k)` matrix of residuals of `k` functions of `X`.
#[pyfunction]
fn gcm_test_multi(
    r_x: PyReadonlyArray1<'_, f64>,
    k: usize,
    r_y: PyReadonlyArray1<'_, f64>,
) -> PyResult<(f64, f64, f64, usize)> {
    let out = mboostrs_core::causal::gcm_test_multi(
        &r_x.as_array().to_vec(),
        k,
        &r_y.as_array().to_vec(),
    )
    .map_err(mboostrs_error_to_pyerr)?;
    Ok((out.statistic, out.p_value, out.residual_correlation, out.n))
}

/// Partially linear IV ratio from cross-fitted residuals
/// (`mboostrs_core::causal::iv_ratio`): `(estimate, variance, influence,
/// first_stage_r2, first_stage_f, n)`.
#[pyfunction]
fn iv_ratio<'py>(
    py: Python<'py>,
    y_res: PyReadonlyArray1<'_, f64>,
    d_res: PyReadonlyArray1<'_, f64>,
    z_res: PyReadonlyArray1<'_, f64>,
) -> PyResult<(f64, f64, Py<PyArray1<f64>>, f64, f64, usize)> {
    let out = mboostrs_core::causal::iv_ratio(
        &y_res.as_array().to_vec(),
        &d_res.as_array().to_vec(),
        &z_res.as_array().to_vec(),
    )
    .map_err(mboostrs_error_to_pyerr)?;
    Ok((
        out.estimate,
        out.variance,
        out.influence.into_pyarray(py).unbind(),
        out.first_stage_r2,
        out.first_stage_f,
        out.n,
    ))
}

/// Partially linear IV ratio with inverse response weights
/// (`mboostrs_core::causal::iv_ratio_weighted`): `(estimate, variance,
/// influence, first_stage_r2, first_stage_f, n)`.
#[pyfunction]
fn iv_ratio_weighted<'py>(
    py: Python<'py>,
    y_res: PyReadonlyArray1<'_, f64>,
    d_res: PyReadonlyArray1<'_, f64>,
    z_res: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
) -> PyResult<(f64, f64, Py<PyArray1<f64>>, f64, f64, usize)> {
    let out = mboostrs_core::causal::iv_ratio_weighted(
        &y_res.as_array().to_vec(),
        &d_res.as_array().to_vec(),
        &z_res.as_array().to_vec(),
        &weights.as_array().to_vec(),
    )
    .map_err(mboostrs_error_to_pyerr)?;
    Ok((
        out.estimate,
        out.variance,
        out.influence.into_pyarray(py).unbind(),
        out.first_stage_r2,
        out.first_stage_f,
        out.n,
    ))
}

/// Weighted least squares with HC1 covariance
/// (`mboostrs_core::effect_modifiers::robust_wls`): `(coefficients, se,
/// covariance, n)`, with `covariance` flattened row-major `(p, p)`.
#[pyfunction]
fn effect_robust_wls<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'_, f64>,
    x: PyReadonlyArray1<'_, f64>,
    p: usize,
    weights: PyReadonlyArray1<'_, f64>,
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    usize,
)> {
    let out = mboostrs_core::effect_modifiers::robust_wls(
        &y.as_array().to_vec(),
        &x.as_array().to_vec(),
        p,
        &weights.as_array().to_vec(),
    )
    .map_err(mboostrs_error_to_pyerr)?;
    Ok((
        out.coefficients.into_pyarray(py).unbind(),
        out.se.into_pyarray(py).unbind(),
        out.covariance.into_pyarray(py).unbind(),
        out.n,
    ))
}

/// Joint Wald test of the coefficients at `indices`
/// (`mboostrs_core::effect_modifiers::wald_test`): `(chi_squared, p_value)`.
#[pyfunction]
fn effect_wald_test(
    coefficients: PyReadonlyArray1<'_, f64>,
    covariance: PyReadonlyArray1<'_, f64>,
    p: usize,
    indices: Vec<usize>,
) -> PyResult<(f64, f64)> {
    mboostrs_core::effect_modifiers::wald_test(
        &coefficients.as_array().to_vec(),
        &covariance.as_array().to_vec(),
        p,
        &indices,
    )
    .map_err(mboostrs_error_to_pyerr)
}

/// Linear predictor for a row-major `(n, p)` design
/// (`mboostrs_core::effect_modifiers::linear_predict`).
#[pyfunction]
fn effect_linear_predict<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'_, f64>,
    p: usize,
    beta: PyReadonlyArray1<'_, f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let out = mboostrs_core::effect_modifiers::linear_predict(
        &x.as_array().to_vec(),
        p,
        &beta.as_array().to_vec(),
    )
    .map_err(mboostrs_error_to_pyerr)?;
    Ok(out.into_pyarray(py).unbind())
}

/// GATES group means and contrasts
/// (`mboostrs_core::effect_modifiers::gates`). Returns `(means, se, counts,
/// ordered, extreme, omnibus)` where `ordered` and `extreme` are
/// `(contrast, se, p, top, bottom)` and `omnibus` is `(chi_squared, df, p)`.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn effect_gates<'py>(
    py: Python<'py>,
    score: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    groups: Vec<usize>,
    n_groups: usize,
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Vec<usize>,
    (f64, f64, f64, usize, usize),
    (f64, f64, f64, usize, usize),
    (f64, usize, f64),
)> {
    let out = mboostrs_core::effect_modifiers::gates(
        &score.as_array().to_vec(),
        &weights.as_array().to_vec(),
        &groups,
        n_groups,
    )
    .map_err(mboostrs_error_to_pyerr)?;
    Ok((
        out.means.into_pyarray(py).unbind(),
        out.se.into_pyarray(py).unbind(),
        out.counts,
        (
            out.ordered_contrast,
            out.ordered_se,
            out.ordered_p,
            out.ordered_top,
            out.ordered_bottom,
        ),
        (
            out.extreme_contrast,
            out.extreme_se,
            out.extreme_p,
            out.extreme_top,
            out.extreme_bottom,
        ),
        (out.omnibus_chi2, out.omnibus_df, out.omnibus_p),
    ))
}

/// Holm step-down adjusted p-values
/// (`mboostrs_core::effect_modifiers::holm_adjust`).
#[pyfunction]
fn effect_holm_adjust<'py>(py: Python<'py>, p: PyReadonlyArray1<'_, f64>) -> Py<PyArray1<f64>> {
    mboostrs_core::effect_modifiers::holm_adjust(&p.as_array().to_vec())
        .into_pyarray(py)
        .unbind()
}

/// Columns of a column-major `(n, p)` design that lie in the span of the
/// columns before them (`mboostrs_core::causal::linear_dependence`). Returns
/// `(position, [(kept position, coefficient), ...])` for each such column.
#[pyfunction]
fn linear_dependence(
    columns: PyReadonlyArray1<'_, f64>,
    n: usize,
    p: usize,
    tol: f64,
) -> PyResult<Vec<(usize, Vec<(usize, f64)>)>> {
    let out = mboostrs_core::causal::linear_dependence(&columns.as_array().to_vec(), n, p, tol)
        .map_err(mboostrs_error_to_pyerr)?;
    Ok(out.into_iter().map(|d| (d.column, d.depends_on)).collect())
}

/// Mean R-loss of an effect prediction (`mboostrs_core::causal`).
#[pyfunction]
fn rloss(
    y: PyReadonlyArray1<'_, f64>,
    t: PyReadonlyArray1<'_, f64>,
    m_hat: PyReadonlyArray1<'_, f64>,
    e_hat: PyReadonlyArray1<'_, f64>,
    tau_hat: PyReadonlyArray1<'_, f64>,
    clip: (f64, f64),
) -> PyResult<f64> {
    let own = |a: &PyReadonlyArray1<'_, f64>| a.as_array().to_vec();
    mboostrs_core::causal::rloss(
        &own(&y),
        &own(&t),
        &own(&m_hat),
        &own(&e_hat),
        &own(&tau_hat),
        clip,
    )
    .map_err(mboostrs_error_to_pyerr)
}

/// R-loss transform with observation weights, an outcome-observed indicator
/// and a response propensity (`mboostrs_core::causal::rloss_transform_weighted`):
/// `(pseudo_outcome, weights, n_clipped, n_clipped_response)`.
#[pyfunction(signature = (y, t, m_hat, e_hat, clip, sample_weight, observed, response, response_clip=None))]
#[allow(clippy::too_many_arguments)]
fn rloss_transform_weighted<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'_, f64>,
    t: PyReadonlyArray1<'_, f64>,
    m_hat: PyReadonlyArray1<'_, f64>,
    e_hat: PyReadonlyArray1<'_, f64>,
    clip: (f64, f64),
    sample_weight: PyReadonlyArray1<'_, f64>,
    observed: PyReadonlyArray1<'_, f64>,
    response: PyReadonlyArray1<'_, f64>,
    response_clip: Option<(f64, f64)>,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, usize, usize)> {
    let own = |a: &PyReadonlyArray1<'_, f64>| a.as_array().to_vec();
    let out = mboostrs_core::causal::rloss_transform_weighted(
        &own(&y),
        &own(&t),
        &own(&m_hat),
        &own(&e_hat),
        clip,
        &own(&sample_weight),
        &own(&observed),
        &own(&response),
        response_clip,
    )
    .map_err(mboostrs_error_to_pyerr)?;
    Ok((
        out.pseudo_outcome.into_pyarray(py).unbind(),
        out.weights.into_pyarray(py).unbind(),
        out.n_clipped,
        out.n_clipped_response,
    ))
}

/// Scale and central range of a second-stage modifier
/// (`mboostrs_core::causal::modifier_scale`): `(log_scale, lower, upper)`.
#[pyfunction(signature = (values, tail, skew_ratio))]
fn modifier_scale(
    values: PyReadonlyArray1<'_, f64>,
    tail: f64,
    skew_ratio: f64,
) -> PyResult<(bool, f64, f64)> {
    let out = mboostrs_core::causal::modifier_scale(&values.as_array().to_vec(), tail, skew_ratio)
        .map_err(mboostrs_error_to_pyerr)?;
    Ok((out.log_scale, out.lower, out.upper))
}

/// Pseudo-outcomes clipped to the `tail` and `1 - tail` type-7 quantiles of the
/// positive-weight rows (`mboostrs_core::causal::winsorize_pseudo_outcome`):
/// `(values, lower, upper, n_clipped)`.
#[pyfunction(signature = (pseudo, weights, tail))]
fn winsorize_pseudo_outcome<'py>(
    py: Python<'py>,
    pseudo: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    tail: f64,
) -> PyResult<(Py<PyArray1<f64>>, f64, f64, usize)> {
    let out = mboostrs_core::causal::winsorize_pseudo_outcome(
        &pseudo.as_array().to_vec(),
        &weights.as_array().to_vec(),
        tail,
    )
    .map_err(mboostrs_error_to_pyerr)?;
    Ok((
        out.values.into_pyarray(py).unbind(),
        out.lower,
        out.upper,
        out.n_clipped,
    ))
}

/// Weighted mean binomial deviance of predicted probabilities
/// (`mboostrs_core::causal::binomial_deviance`); lower is better.
#[pyfunction(signature = (treatment, probability, weights))]
fn binomial_deviance(
    treatment: PyReadonlyArray1<'_, f64>,
    probability: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
) -> PyResult<f64> {
    mboostrs_core::causal::binomial_deviance(
        &treatment.as_array().to_vec(),
        &probability.as_array().to_vec(),
        &weights.as_array().to_vec(),
    )
    .map_err(mboostrs_error_to_pyerr)
}

/// Augmented inverse-probability scores (`mboostrs_core::causal::augmented_scores`).
///
/// `estimand` is `"ate"`, `"att"` or `"overlap"`. Returns `(scores, estimate, influence,
/// treated_share, propensity_used, response_used, n_clipped_propensity,
/// n_clipped_response, response_ess)`.
#[pyfunction(signature = (y, a, observed, weights, e_hat, pi_hat, mu0, mu1, estimand, clip=None, response_clip=None))]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn augmented_scores<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'_, f64>,
    a: PyReadonlyArray1<'_, f64>,
    observed: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    e_hat: PyReadonlyArray1<'_, f64>,
    pi_hat: PyReadonlyArray1<'_, f64>,
    mu0: PyReadonlyArray1<'_, f64>,
    mu1: PyReadonlyArray1<'_, f64>,
    estimand: &str,
    clip: Option<(f64, f64)>,
    response_clip: Option<(f64, f64)>,
) -> PyResult<(
    Py<PyArray1<f64>>,
    f64,
    Py<PyArray1<f64>>,
    f64,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    usize,
    usize,
    f64,
)> {
    let which = match estimand {
        "ate" => mboostrs_core::causal::Estimand::Ate,
        "att" => mboostrs_core::causal::Estimand::Att,
        "overlap" => mboostrs_core::causal::Estimand::Overlap,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "estimand must be 'ate', 'att' or 'overlap', got {other:?}"
            )))
        }
    };
    let own = |v: &PyReadonlyArray1<'_, f64>| v.as_array().to_vec();
    let out = mboostrs_core::causal::augmented_scores(
        &own(&y),
        &own(&a),
        &own(&observed),
        &own(&weights),
        &own(&e_hat),
        &own(&pi_hat),
        &own(&mu0),
        &own(&mu1),
        which,
        clip,
        response_clip,
    )
    .map_err(mboostrs_error_to_pyerr)?;
    Ok((
        out.scores.into_pyarray(py).unbind(),
        out.estimate,
        out.influence.into_pyarray(py).unbind(),
        out.treated_share,
        out.propensity_used.into_pyarray(py).unbind(),
        out.response_used.into_pyarray(py).unbind(),
        out.n_clipped_propensity,
        out.n_clipped_response,
        out.response_ess,
    ))
}

/// Variance of a weighted mean of influence values, with frequency weights
/// or, given dense group codes, the cluster-robust form
/// (`mboostrs_core::causal::influence_variance`).
#[pyfunction(signature = (influence, weights, groups=None))]
fn influence_variance(
    influence: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    groups: Option<PyReadonlyArray1<'_, usize>>,
) -> PyResult<f64> {
    let g = groups.as_ref().map(|g| g.as_array().to_vec());
    mboostrs_core::causal::influence_variance(
        &influence.as_array().to_vec(),
        &weights.as_array().to_vec(),
        g.as_deref(),
    )
    .map_err(mboostrs_error_to_pyerr)
}

/// Mean and variance of augmented scores within subgroups
/// (`mboostrs_core::causal::subgroup_scores`): one `(n, weight, mean,
/// variance, n_clusters)` tuple per level; `variance` and `n_clusters` are
/// None when unavailable.
#[pyfunction(signature = (scores, weights, codes, n_levels, groups=None))]
fn subgroup_scores(
    scores: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    codes: PyReadonlyArray1<'_, i64>,
    n_levels: usize,
    groups: Option<PyReadonlyArray1<'_, usize>>,
) -> PyResult<Vec<(usize, f64, f64, Option<f64>, Option<usize>)>> {
    let g = groups.as_ref().map(|g| g.as_array().to_vec());
    let out = mboostrs_core::causal::subgroup_scores(
        &scores.as_array().to_vec(),
        &weights.as_array().to_vec(),
        &codes.as_array().to_vec(),
        n_levels,
        g.as_deref(),
    )
    .map_err(mboostrs_error_to_pyerr)?;
    Ok(out
        .into_iter()
        .map(|r| (r.n, r.weight, r.mean, r.variance, r.n_clusters))
        .collect())
}

/// Sample mean and IID standard error of scores (`mboostrs_core::causal::mean_and_iid_se`).
#[pyfunction]
fn mean_and_iid_se(values: PyReadonlyArray1<'_, f64>) -> PyResult<(f64, f64)> {
    mboostrs_core::causal::mean_and_iid_se(&values.as_array().to_vec())
        .map_err(mboostrs_error_to_pyerr)
}

/// Effective sample size `(sum w)^2 / sum w^2` (`mboostrs_core::causal::effective_sample_size`).
#[pyfunction]
fn effective_sample_size(weights: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    mboostrs_core::causal::effective_sample_size(&weights.as_array().to_vec())
        .map_err(mboostrs_error_to_pyerr)
}

/// Median-rule combination of repeated fits (`mboostrs_core::causal::median_repeat_combine`).
///
/// `estimates` and `ses` have shape `(repeats, m)`. Returns `(estimate, se)`, each of
/// length `m`.
#[pyfunction]
fn median_repeat_combine<'py>(
    py: Python<'py>,
    estimates: PyReadonlyArray2<'_, f64>,
    ses: PyReadonlyArray2<'_, f64>,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let rows = |a: &PyReadonlyArray2<'_, f64>| -> Vec<Vec<f64>> {
        a.as_array().outer_iter().map(|r| r.to_vec()).collect()
    };
    let out = mboostrs_core::causal::median_repeat_combine(&rows(&estimates), &rows(&ses))
        .map_err(mboostrs_error_to_pyerr)?;
    Ok((
        out.estimate.into_pyarray(py).unbind(),
        out.se.into_pyarray(py).unbind(),
    ))
}

/// Percentile interval from bootstrap draws (`mboostrs_core::causal::bootstrap_percentile_interval`).
///
/// `draws` has shape `(B, m)`. Returns `(lower, upper, se)`, each of length `m`.
#[pyfunction]
fn bootstrap_percentile_interval<'py>(
    py: Python<'py>,
    draws: PyReadonlyArray2<'_, f64>,
    level: f64,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let rows: Vec<Vec<f64>> = draws.as_array().outer_iter().map(|r| r.to_vec()).collect();
    let out = mboostrs_core::causal::bootstrap_percentile_interval(&rows, level)
        .map_err(mboostrs_error_to_pyerr)?;
    Ok((
        out.lower.into_pyarray(py).unbind(),
        out.upper.into_pyarray(py).unbind(),
        out.se.into_pyarray(py).unbind(),
    ))
}

/// Weighted mean square of log-scale residuals (`mboostrs_core::causal::log_residual_variance`).
#[pyfunction]
fn log_residual_variance(
    y: PyReadonlyArray1<'_, f64>,
    fitted: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
) -> PyResult<f64> {
    mboostrs_core::causal::log_residual_variance(
        &y.as_array().to_vec(),
        &fitted.as_array().to_vec(),
        &weights.as_array().to_vec(),
    )
    .map_err(mboostrs_error_to_pyerr)
}

/// Two-part outcome mean `p * exp(m + s2 / 2)` (`mboostrs_core::causal::two_part_mean`).
#[pyfunction]
fn two_part_mean<'py>(
    py: Python<'py>,
    employed: PyReadonlyArray1<'_, f64>,
    log_mean: PyReadonlyArray1<'_, f64>,
    log_variance: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let mu = mboostrs_core::causal::two_part_mean(
        &employed.as_array().to_vec(),
        &log_mean.as_array().to_vec(),
        log_variance,
    )
    .map_err(mboostrs_error_to_pyerr)?;
    Ok(mu.into_pyarray(py).unbind())
}

#[pyfunction(signature = (before, after, partitions=1, threshold_se=2.0))]
fn paired_gain(
    before: PyReadonlyArray1<'_, f64>,
    after: PyReadonlyArray1<'_, f64>,
    partitions: usize,
    threshold_se: f64,
) -> PyResult<Py<PyDict>> {
    let gain = mboostrs_core::selection::try_paired_gain(
        before.as_array().as_slice().unwrap_or(&[]),
        after.as_array().as_slice().unwrap_or(&[]),
        partitions,
        threshold_se,
    )
    .map_err(mboostrs_error_to_pyerr)?;
    let py = before.py();
    let record = PyDict::new(py);
    record.set_item("baseline", gain.baseline)?;
    record.set_item("candidate", gain.candidate)?;
    record.set_item("gain", gain.gain)?;
    record.set_item("se", gain.se)?;
    record.set_item("t", gain.t)?;
    record.set_item("passed", gain.passed)?;
    record.set_item("worse_folds", gain.worse_folds)?;
    record.set_item("n_folds", gain.n_folds)?;
    Ok(record.unbind())
}

#[pyfunction(signature = (scores, selected, partitions=1, max_draws=10000, seed=0))]
fn specification_curve_sign_test(
    py: Python<'_>,
    scores: PyReadonlyArray2<'_, f64>,
    selected: usize,
    partitions: usize,
    max_draws: usize,
    seed: u64,
) -> PyResult<Py<PyDict>> {
    let rows: Vec<Vec<f64>> = scores
        .as_array()
        .outer_iter()
        .map(|row| row.to_vec())
        .collect();
    let test =
        mboostrs_core::selection::try_curve_sign_test(&rows, selected, partitions, max_draws, seed)
            .map_err(mboostrs_error_to_pyerr)?;
    let record = PyDict::new(py);
    record.set_item("statistic", test.statistic)?;
    record.set_item("p_value", test.p_value)?;
    record.set_item("units", test.units)?;
    record.set_item("exact", test.exact)?;
    record.set_item("draws", test.draws)?;
    Ok(record.unbind())
}

#[pyfunction(signature = (n, fold_type, b, prob, seed, weights=None))]
fn cv_weights(
    py: Python<'_>,
    n: usize,
    fold_type: &str,
    b: usize,
    prob: f64,
    seed: u64,
    weights: Option<PyArrayLike1<'_, f64, AllowTypeChange>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let w = weights
        .map(|weights| weights.as_array().to_vec())
        .unwrap_or_else(|| vec![1.0; n]);
    if w.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "weights length {} != n {}",
            w.len(),
            n
        )));
    }
    let ft = match fold_type {
        "kfold" => FoldType::KFold,
        "bootstrap" => FoldType::Bootstrap,
        "subsampling" => FoldType::Subsampling,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown fold type: {other}"
            )))
        }
    };
    let spec = FoldSpec::try_new(ft, b, prob, seed).map_err(mboostrs_error_to_pyerr)?;
    let folds = try_cv(&spec, n, &w).map_err(mboostrs_error_to_pyerr)?;
    Ok(folds.into_pyarray(py).unbind())
}

#[pyfunction(signature = (x_matrix, y, weights, mstop, nu, family_name, family_params, learner_names, folds, judge=None, offset=None, risk="inbag", trace=false, parallel=true, fold_label="bootstrap", step="fixed", step_iterations=4, intercept=true, center=true))]
#[allow(clippy::too_many_arguments)]
fn cvrisk_glmboost(
    py: Python<'_>,
    x_matrix: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    mstop: u32,
    nu: f64,
    family_name: &str,
    family_params: HashMap<String, f64>,
    learner_names: Vec<String>,
    folds: PyArrayLike2<'_, f64, AllowTypeChange>,
    judge: Option<PyArrayLike2<'_, bool, AllowTypeChange>>,
    offset: Option<Bound<'_, PyAny>>,
    risk: &str,
    trace: bool,
    parallel: bool,
    fold_label: &str,
    step: &str,
    step_iterations: u32,
    intercept: bool,
    center: bool,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray1<f64>>, Vec<usize>, usize)> {
    let offset = extract_offset_arg(offset.as_ref())?;
    let offset_vec_owned = offset
        .vector()
        .map(|values| ndarray::Array1::from(values.to_vec()));
    let offset_vec_view = offset_vec_owned.as_ref().map(|vector| vector.view());
    let x = x_matrix.as_array();
    let y_arr = y.as_array();
    let w_arr = weights.as_array();
    let is_survival = matches!(
        family_name,
        "coxph" | "weibull" | "loglog" | "lognormal" | "gehan" | "cindex"
    );
    let n = if is_survival {
        y_arr.len() / 2
    } else {
        y_arr.len()
    };

    if x.nrows() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "x_matrix has {} rows but y has {} elements",
            x.nrows(),
            n
        )));
    }

    let x_cols: Vec<Vec<f64>> = (0..x.ncols()).map(|j| x.column(j).to_vec()).collect();

    // center = false (glmboost(center=False)) refits every fold on the raw
    // columns, as the model was fitted.
    let column_means: Vec<f64> = x_cols
        .iter()
        .map(|col| {
            if center {
                col.iter().sum::<f64>() / col.len() as f64
            } else {
                0.0
            }
        })
        .collect();

    let preparer = GlmBoostPreparer {
        x_cols,
        names: learner_names,
        family_name: family_name.to_string(),
        family_params: family_params.clone(),
        column_means,
        intercept,
    };

    let folds_arr = fold_matrix_from_py(folds, n)?;
    let judge_arr = judge_mask_from_py(judge, n)?;

    let risk_kind = match risk {
        "inbag" => RiskKind::Inbag,
        "oobag" => RiskKind::Oobag,
        "none" => RiskKind::None,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "risk must be 'inbag', 'oobag', or 'none', got {risk:?}"
            )))
        }
    };
    let ctrl = BoostControl::new(mstop, nu)
        .map_err(mboostrs_error_to_pyerr)?
        .with_risk(risk_kind)
        .with_trace(trace);
    let ctrl = apply_step(ctrl, step, step_iterations)?;

    let y_owned = y_arr.to_owned();
    let w_owned = w_arr.to_owned();
    let family_name_owned = family_name.to_string();
    let family_params_owned = family_params.clone();
    let result = py
        .allow_threads(|| {
            with_concrete_family!(family_name_owned.as_str(), &family_params_owned, |fam| {
                try_cvrisk_with(
                    &preparer,
                    y_owned.view(),
                    w_owned.view(),
                    &ctrl,
                    &folds_arr,
                    offset.scalar(),
                    offset_vec_view,
                    judge_arr.as_ref(),
                    parallel,
                    fam,
                    fold_label,
                )
            })
        })?
        .map_err(mboostrs_error_to_pyerr)?;

    let opt = result.optimal_mstop();
    let grid = result.grid;
    let risk_matrix = result.risk_matrix.into_pyarray(py).unbind();
    let aggregate = result.aggregate.into_pyarray(py).unbind();

    Ok((risk_matrix, aggregate, grid, opt))
}

#[pyfunction(signature = (x_matrix, y, weights, mstop, nu, family_name, family_params, learner_names, folds, offset=None, step="fixed", step_iterations=4, intercept=true, center=true))]
#[allow(clippy::too_many_arguments)]
fn cvrisk_selected_glmboost(
    py: Python<'_>,
    x_matrix: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    mstop: u32,
    nu: f64,
    family_name: &str,
    family_params: HashMap<String, f64>,
    learner_names: Vec<String>,
    folds: PyArrayLike2<'_, f64, AllowTypeChange>,
    offset: Option<Bound<'_, PyAny>>,
    step: &str,
    step_iterations: u32,
    intercept: bool,
    center: bool,
) -> PyResult<Py<PyArray2<u64>>> {
    let offset = extract_offset_arg(offset.as_ref())?;
    let x = x_matrix.as_array();
    let y_arr = y.as_array();
    let w_arr = weights.as_array();
    let is_survival = matches!(
        family_name,
        "coxph" | "weibull" | "loglog" | "lognormal" | "gehan" | "cindex"
    );
    let n = if is_survival {
        y_arr.len() / 2
    } else {
        y_arr.len()
    };

    let x_cols: Vec<Vec<f64>> = (0..x.ncols()).map(|j| x.column(j).to_vec()).collect();
    // center = false (glmboost(center=False)) refits every fold on the raw
    // columns, as the model was fitted.
    let column_means: Vec<f64> = x_cols
        .iter()
        .map(|col| {
            if center {
                col.iter().sum::<f64>() / col.len() as f64
            } else {
                0.0
            }
        })
        .collect();

    let preparer = GlmBoostPreparer {
        x_cols,
        names: learner_names,
        family_name: family_name.to_string(),
        family_params: family_params.clone(),
        column_means,
        intercept,
    };

    let folds_arr = fold_matrix_from_py(folds, n)?;

    let ctrl = BoostControl::new(mstop, nu).map_err(mboostrs_error_to_pyerr)?;
    let ctrl = apply_step(ctrl, step, step_iterations)?;

    let y_owned = y_arr.to_owned();
    let w_owned = w_arr.to_owned();
    let family_name_owned = family_name.to_string();
    let family_params_owned = family_params.clone();
    let result = py
        .allow_threads(|| {
            with_concrete_family!(family_name_owned.as_str(), &family_params_owned, |fam| {
                match offset.vector() {
                    Some(vector) => try_cvrisk_selected_with_offset_vec(
                        &preparer,
                        y_owned.view(),
                        w_owned.view(),
                        &ctrl,
                        &folds_arr,
                        ndarray::ArrayView1::from(vector),
                        fam,
                    ),
                    None => try_cvrisk_selected_with(
                        &preparer,
                        y_owned.view(),
                        w_owned.view(),
                        &ctrl,
                        &folds_arr,
                        offset.scalar(),
                        fam,
                    ),
                }
            })
        })?
        .map_err(mboostrs_error_to_pyerr)?;

    Ok(result
        .selected
        .mapv(|idx| idx as u64)
        .into_pyarray(py)
        .unbind())
}

#[allow(clippy::too_many_arguments)]
fn cvrisk_gamboost_specs_impl(
    py: Python<'_>,
    learner_specs: Vec<PyObject>,
    y: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    mstop: u32,
    nu: f64,
    family_name: &str,
    family_params: HashMap<String, f64>,
    folds: PyArrayLike2<'_, f64, AllowTypeChange>,
    judge: Option<PyArrayLike2<'_, bool, AllowTypeChange>>,
    offset: Option<Bound<'_, PyAny>>,
    risk: &str,
    trace: bool,
    parallel: bool,
    fold_label: &str,
    multinomial_n_classes: Option<usize>,
    step: &str,
    step_iterations: u32,
    rule: Option<mboostrs_core::ExtendRule>,
) -> PyResult<(
    (Py<PyArray2<f64>>, Py<PyArray1<f64>>, Vec<usize>, usize),
    Option<mboostrs_core::ExtensionReport>,
)> {
    let risk_kind = match risk {
        "inbag" => RiskKind::Inbag,
        "oobag" => RiskKind::Oobag,
        "none" => RiskKind::None,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "risk must be 'inbag', 'oobag', or 'none', got {risk:?}"
            )))
        }
    };
    // No blueprints is the offset-only model (see the fit binding above).
    let blueprints = parse_learner_blueprint_list(py, learner_specs)?;
    let y_owned = y.as_array().to_owned();
    let w_owned = weights.as_array().to_owned();
    let family_probe = make_family(family_name, &family_params)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    // For Multinomial the response arrives already flattened to n_obs*(K-1),
    // while weights and folds arrive at the observation granularity. Lift the
    // latter to the flattened layout here so `try_cvrisk_with` sees one
    // uniform contract, exactly as `apply_multinomial_kronecker` expands the
    // fit-time weights.
    let (_n, w_owned, folds_arr, judge_arr) = match multinomial_n_classes {
        Some(n_classes) => {
            let n_resp = n_classes - 1;
            let n_obs = w_owned.len();
            // The flattened response length must equal n_obs * (K-1).
            let n_flat = n_obs * n_resp;
            if y_owned.len() != n_flat {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Multinomial response has length {}, but weights describe {n_obs} observations and n_classes={n_classes}, so {n_flat} entries were expected",
                    y_owned.len()
                )));
            }
            let w_flat = expand_multinomial_weights(&w_owned, n_classes);
            let folds_obs = fold_matrix_from_py(folds, n_obs)?;
            let judge_obs = judge_mask_from_py(judge, n_obs)?;
            let folds_flat = expand_fold_matrix(&folds_obs, n_resp);
            let judge_flat = judge_obs.map(|j| expand_bool_matrix(&j, n_resp));
            (n_flat, w_flat, folds_flat, judge_flat)
        }
        None => {
            let n = family_probe.n_from_y(y_owned.view());
            if n != w_owned.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "response describes {n} observations but weights has length {}",
                    w_owned.len()
                )));
            }
            let folds_arr = fold_matrix_from_py(folds, n)?;
            let judge_arr = judge_mask_from_py(judge, n)?;
            (n, w_owned, folds_arr, judge_arr.map(|j| j.to_owned()))
        }
    };
    let control = BoostControl::new(mstop, nu)
        .map_err(mboostrs_error_to_pyerr)?
        .with_risk(risk_kind)
        .with_trace(trace);
    let control = apply_step(control, step, step_iterations)?;
    let offset = extract_offset_arg(offset.as_ref())?;
    let offset_vec_owned = offset
        .vector()
        .map(|values| ndarray::Array1::from(values.to_vec()));
    let offset_vec_view = offset_vec_owned.as_ref().map(|vector| vector.view());
    let preparer = BlueprintPreparer {
        blueprints,
        family_name: family_name.to_string(),
        family_params: family_params.clone(),
        multinomial_n_classes,
    };
    let family_name_owned = family_name.to_string();
    let family_params_owned = family_params.clone();
    let result = py
        .allow_threads(|| {
            with_concrete_family!(family_name_owned.as_str(), &family_params_owned, |fam| {
                match rule.as_ref() {
                    Some(rule) => mboostrs_core::try_cvrisk_extending(
                        &preparer,
                        y_owned.view(),
                        w_owned.view(),
                        &control,
                        &folds_arr,
                        offset.scalar(),
                        offset_vec_view,
                        judge_arr.as_ref(),
                        parallel,
                        fam,
                        fold_label,
                        rule,
                    )
                    .map(|(result, report)| (result, Some(report))),
                    None => try_cvrisk_with(
                        &preparer,
                        y_owned.view(),
                        w_owned.view(),
                        &control,
                        &folds_arr,
                        offset.scalar(),
                        offset_vec_view,
                        judge_arr.as_ref(),
                        parallel,
                        fam,
                        fold_label,
                    )
                    .map(|result| (result, None)),
                }
            })
        })?
        .map_err(mboostrs_error_to_pyerr)?;
    let (result, report) = result;
    let optimal = result.optimal_mstop();
    Ok((
        (
            result.risk_matrix.into_pyarray(py).unbind(),
            result.aggregate.into_pyarray(py).unbind(),
            result.grid,
            optimal,
        ),
        report,
    ))
}

#[pyfunction(signature = (learner_specs, y, weights, mstop, nu, family_name, family_params, folds, judge=None, offset=None, risk="inbag", trace=false, parallel=true, fold_label="bootstrap", multinomial_n_classes=None, step="fixed", step_iterations=4))]
#[allow(clippy::too_many_arguments)]
fn cvrisk_gamboost_specs(
    py: Python<'_>,
    learner_specs: Vec<PyObject>,
    y: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    mstop: u32,
    nu: f64,
    family_name: &str,
    family_params: HashMap<String, f64>,
    folds: PyArrayLike2<'_, f64, AllowTypeChange>,
    judge: Option<PyArrayLike2<'_, bool, AllowTypeChange>>,
    offset: Option<Bound<'_, PyAny>>,
    risk: &str,
    trace: bool,
    parallel: bool,
    fold_label: &str,
    multinomial_n_classes: Option<usize>,
    step: &str,
    step_iterations: u32,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray1<f64>>, Vec<usize>, usize)> {
    cvrisk_gamboost_specs_impl(
        py,
        learner_specs,
        y,
        weights,
        mstop,
        nu,
        family_name,
        family_params,
        folds,
        judge,
        offset,
        risk,
        trace,
        parallel,
        fold_label,
        multinomial_n_classes,
        step,
        step_iterations,
        None,
    )
    .map(|(out, _)| out)
}

/// `cvrisk_gamboost_specs` on a grid that grows until the held-out risk
/// turns (`mboostrs_core::try_cvrisk_extending`): each fold's path is resumed
/// rather than refitted. Returns the usual four values plus the grid end
/// after each extension and whether the search stopped at the boundary.
#[pyfunction(signature = (learner_specs, y, weights, mstop, nu, family_name, family_params, folds, judge=None, offset=None, risk="inbag", trace=false, parallel=true, fold_label="bootstrap", multinomial_n_classes=None, step="fixed", step_iterations=4, max_mstop=10000, patience=None, boundary_share=0.02, max_extensions=None))]
#[allow(clippy::too_many_arguments)]
fn cvrisk_gamboost_specs_extending(
    py: Python<'_>,
    learner_specs: Vec<PyObject>,
    y: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    mstop: u32,
    nu: f64,
    family_name: &str,
    family_params: HashMap<String, f64>,
    folds: PyArrayLike2<'_, f64, AllowTypeChange>,
    judge: Option<PyArrayLike2<'_, bool, AllowTypeChange>>,
    offset: Option<Bound<'_, PyAny>>,
    risk: &str,
    trace: bool,
    parallel: bool,
    fold_label: &str,
    multinomial_n_classes: Option<usize>,
    step: &str,
    step_iterations: u32,
    max_mstop: u32,
    patience: Option<u32>,
    boundary_share: f64,
    max_extensions: Option<u32>,
) -> PyResult<(
    (Py<PyArray2<f64>>, Py<PyArray1<f64>>, Vec<usize>, usize),
    Vec<usize>,
    bool,
)> {
    let rule = mboostrs_core::ExtendRule {
        start: mstop,
        max_mstop,
        patience,
        boundary_share,
        max_extensions,
    };
    let (out, report) = cvrisk_gamboost_specs_impl(
        py,
        learner_specs,
        y,
        weights,
        mstop,
        nu,
        family_name,
        family_params,
        folds,
        judge,
        offset,
        risk,
        trace,
        parallel,
        fold_label,
        multinomial_n_classes,
        step,
        step_iterations,
        Some(rule),
    )?;
    let report = report.expect("an extension rule yields a report");
    Ok((out, report.grid_ends, report.at_boundary))
}

#[pyfunction(signature = (learner_specs, y, weights, mstop, nu, family_name, family_params, folds, offset=None, multinomial_n_classes=None, step="fixed", step_iterations=4))]
#[allow(clippy::too_many_arguments)]
fn cvrisk_selected_gamboost_specs(
    py: Python<'_>,
    learner_specs: Vec<PyObject>,
    y: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    mstop: u32,
    nu: f64,
    family_name: &str,
    family_params: HashMap<String, f64>,
    folds: PyArrayLike2<'_, f64, AllowTypeChange>,
    offset: Option<Bound<'_, PyAny>>,
    multinomial_n_classes: Option<usize>,
    step: &str,
    step_iterations: u32,
) -> PyResult<Py<PyArray2<u64>>> {
    let offset = extract_offset_arg(offset.as_ref())?;
    let blueprints = parse_learner_blueprint_list(py, learner_specs)?;
    let y_owned = y.as_array().to_owned();
    let w_owned = weights.as_array().to_owned();
    // Multinomial: lift weights and folds to the flattened layout, exactly as
    // `cvrisk_gamboost_specs` does. The response already arrives flattened.
    let (w_owned, folds_flat) = match multinomial_n_classes {
        Some(n_classes) => {
            let n_resp = n_classes - 1;
            let n_obs = w_owned.len();
            if y_owned.len() != n_obs * n_resp {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Multinomial response has length {}, but weights describe {n_obs} observations and n_classes={n_classes}",
                    y_owned.len()
                )));
            }
            let folds_obs = fold_matrix_from_py(folds, n_obs)?;
            (
                expand_multinomial_weights(&w_owned, n_classes),
                expand_fold_matrix(&folds_obs, n_resp),
            )
        }
        None => {
            let family_probe = make_family(family_name, &family_params)
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            let n = family_probe.n_from_y(y_owned.view());
            (w_owned, fold_matrix_from_py(folds, n)?)
        }
    };
    let control = BoostControl::new(mstop, nu).map_err(mboostrs_error_to_pyerr)?;
    let control = apply_step(control, step, step_iterations)?;
    let preparer = BlueprintPreparer {
        blueprints,
        family_name: family_name.to_string(),
        family_params: family_params.clone(),
        multinomial_n_classes,
    };
    let family_name_owned = family_name.to_string();
    let family_params_owned = family_params.clone();
    let result = py
        .allow_threads(|| {
            with_concrete_family!(family_name_owned.as_str(), &family_params_owned, |fam| {
                match offset.vector() {
                    Some(vector) => try_cvrisk_selected_with_offset_vec(
                        &preparer,
                        y_owned.view(),
                        w_owned.view(),
                        &control,
                        &folds_flat,
                        ndarray::ArrayView1::from(vector),
                        fam,
                    ),
                    None => try_cvrisk_selected_with(
                        &preparer,
                        y_owned.view(),
                        w_owned.view(),
                        &control,
                        &folds_flat,
                        offset.scalar(),
                        fam,
                    ),
                }
            })
        })?
        .map_err(mboostrs_error_to_pyerr)?;
    Ok(result
        .selected
        .mapv(|index| index as u64)
        .into_pyarray(py)
        .unbind())
}

#[pyfunction(signature = (scalar_x, scalar_names, scalar_lambda, scalar_center, scalar_df, bbs_x, bbs_names, bbs_knots, bbs_degree, bbs_differences, bbs_df, bbs_lambda, bbs_center, bbs_cyclic, bbs_boundary_knots_lo, bbs_boundary_knots_hi, bbs_deriv, bmono_x, bmono_names, bmono_knots, bmono_degree, bmono_differences, bmono_df, bmono_constraint, bmono_type, bmono_lambda2, bmono_niter, bmono_center, btree_x, btree_names, btree_minbucket, btree_minsplit, btree_maxdepth, y, weights, mstop, nu, family_name, family_params, folds, judge=None, offset=None, scalar_by=None, bbs_by=None, bmono_by=None, btree_by=None, bmrf_levels=None, bmrf_names=None, bmrf_k=None, bmrf_df=None, bmrf_lambda=None, bmrf_center=None, bmrf_by=None, bspatial_x1=None, bspatial_x2=None, bspatial_names=None, bspatial_knots=None, bspatial_degree=None, bspatial_differences=None, bspatial_df=None, bspatial_lambda=None, bspatial_center=None, bspatial_by=None, bkernel_x=None, bkernel_names=None, bkernel_bandwidth=None, bkernel_df=None, bkernel_kernel=None, bkernel_center=None, bkernel_by=None, brandom_x=None, brandom_names=None, brandom_n_levels=None, brandom_df=None, brandom_lambda=None, brandom_center=None, brandom_by=None, bmono_C=None, buser_X=None, buser_K=None, buser_names=None, buser_df=None, buser_lambda=None, buser_center=None, buser_by=None, risk="inbag", trace=false, parallel=true, fold_label="bootstrap", step="fixed", step_iterations=4))]
#[allow(clippy::type_complexity)]
#[allow(non_snake_case)]
fn cvrisk_gamboost(
    py: Python<'_>,
    scalar_x: Vec<PyReadonlyArray1<'_, f64>>,
    scalar_names: Vec<String>,
    scalar_lambda: Vec<f64>,
    scalar_center: Vec<bool>,
    scalar_df: Vec<f64>,
    bbs_x: Vec<PyReadonlyArray1<'_, f64>>,
    bbs_names: Vec<String>,
    bbs_knots: Vec<usize>,
    bbs_degree: Vec<usize>,
    bbs_differences: Vec<usize>,
    bbs_df: Vec<f64>,
    bbs_lambda: Vec<f64>,
    bbs_center: Vec<bool>,
    bbs_cyclic: Vec<bool>,
    bbs_boundary_knots_lo: Vec<f64>,
    bbs_boundary_knots_hi: Vec<f64>,
    bbs_deriv: Vec<usize>,
    bmono_x: Vec<PyReadonlyArray1<'_, f64>>,
    bmono_names: Vec<String>,
    bmono_knots: Vec<usize>,
    bmono_degree: Vec<usize>,
    bmono_differences: Vec<usize>,
    bmono_df: Vec<f64>,
    bmono_constraint: Vec<String>,
    bmono_type: Vec<String>,
    bmono_lambda2: Vec<f64>,
    bmono_niter: Vec<usize>,
    bmono_center: Vec<bool>,
    btree_x: Vec<PyReadonlyArray2<'_, f64>>,
    btree_names: Vec<String>,
    btree_minbucket: Vec<usize>,
    btree_minsplit: Vec<usize>,
    btree_maxdepth: Vec<usize>,
    y: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    mstop: u32,
    nu: f64,
    family_name: &str,
    family_params: HashMap<String, f64>,
    folds: PyArrayLike2<'_, f64, AllowTypeChange>,
    judge: Option<PyArrayLike2<'_, bool, AllowTypeChange>>,
    offset: Option<Bound<'_, PyAny>>,
    scalar_by: ByVecs,
    bbs_by: ByVecs,
    bmono_by: ByVecs,
    btree_by: ByVecs,
    bmrf_levels: Option<Vec<Vec<usize>>>,
    bmrf_names: Option<Vec<String>>,
    bmrf_k: Option<Vec<PyReadonlyArray2<'_, f64>>>,
    bmrf_df: Option<Vec<f64>>,
    bmrf_lambda: Option<Vec<f64>>,
    bmrf_center: Option<Vec<bool>>,
    bmrf_by: ByVecs,
    bspatial_x1: Option<Vec<PyReadonlyArray1<'_, f64>>>,
    bspatial_x2: Option<Vec<PyReadonlyArray1<'_, f64>>>,
    bspatial_names: Option<Vec<String>>,
    bspatial_knots: Option<Vec<usize>>,
    bspatial_degree: Option<Vec<usize>>,
    bspatial_differences: Option<Vec<usize>>,
    bspatial_df: Option<Vec<f64>>,
    bspatial_lambda: Option<Vec<f64>>,
    bspatial_center: Option<Vec<bool>>,
    bspatial_by: ByVecs,
    bkernel_x: Option<Vec<PyReadonlyArray1<'_, f64>>>,
    bkernel_names: Option<Vec<String>>,
    bkernel_bandwidth: Option<Vec<f64>>,
    bkernel_df: Option<Vec<f64>>,
    bkernel_kernel: Option<Vec<String>>,
    bkernel_center: Option<Vec<bool>>,
    bkernel_by: ByVecs,
    brandom_x: Option<Vec<PyReadonlyArray2<'_, f64>>>,
    brandom_names: Option<Vec<String>>,
    brandom_n_levels: Option<Vec<usize>>,
    brandom_df: Option<Vec<f64>>,
    brandom_lambda: Option<Vec<f64>>,
    brandom_center: Option<Vec<bool>>,
    brandom_by: ByVecs,
    bmono_C: Option<Vec<Option<PyReadonlyArray2<'_, f64>>>>,
    buser_X: Option<Vec<PyReadonlyArray2<'_, f64>>>,
    buser_K: Option<Vec<PyReadonlyArray2<'_, f64>>>,
    buser_names: Option<Vec<String>>,
    buser_df: Option<Vec<f64>>,
    buser_lambda: Option<Vec<f64>>,
    buser_center: Option<Vec<bool>>,
    buser_by: ByVecs,
    risk: &str,
    trace: bool,
    parallel: bool,
    fold_label: &str,
    step: &str,
    step_iterations: u32,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray1<f64>>, Vec<usize>, usize)> {
    let risk_kind = match risk {
        "inbag" => RiskKind::Inbag,
        "oobag" => RiskKind::Oobag,
        "none" => RiskKind::None,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "risk must be 'inbag', 'oobag', or 'none', got {risk:?}"
            )))
        }
    };

    let n_scalar = scalar_x.len();
    if scalar_names.len() != n_scalar
        || scalar_lambda.len() != n_scalar
        || scalar_center.len() != n_scalar
        || scalar_df.len() != n_scalar
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "scalar_x/names/lambda/center/df must have the same length",
        ));
    }
    let n_bbs = bbs_x.len();
    if bbs_names.len() != n_bbs
        || bbs_knots.len() != n_bbs
        || bbs_degree.len() != n_bbs
        || bbs_differences.len() != n_bbs
        || bbs_df.len() != n_bbs
        || bbs_lambda.len() != n_bbs
        || bbs_center.len() != n_bbs
        || bbs_cyclic.len() != n_bbs
        || bbs_boundary_knots_lo.len() != n_bbs
        || bbs_boundary_knots_hi.len() != n_bbs
        || bbs_deriv.len() != n_bbs
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bbs arrays must all have the same length",
        ));
    }
    let n_bmono = bmono_x.len();
    if bmono_names.len() != n_bmono
        || bmono_knots.len() != n_bmono
        || bmono_degree.len() != n_bmono
        || bmono_differences.len() != n_bmono
        || bmono_df.len() != n_bmono
        || bmono_constraint.len() != n_bmono
        || bmono_type.len() != n_bmono
        || bmono_lambda2.len() != n_bmono
        || bmono_niter.len() != n_bmono
        || bmono_center.len() != n_bmono
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bmono arrays must all have the same length",
        ));
    }
    let n_btree = btree_x.len();
    if btree_names.len() != n_btree
        || btree_minbucket.len() != n_btree
        || btree_minsplit.len() != n_btree
        || btree_maxdepth.len() != n_btree
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "btree arrays must all have the same length",
        ));
    }
    let n_bmrf = bmrf_levels.as_ref().map_or(0, |v| v.len());
    if n_bmrf > 0 {
        let bn = bmrf_names.as_ref().map_or(0, |v| v.len());
        let bk = bmrf_k.as_ref().map_or(0, |v| v.len());
        let bd = bmrf_df.as_ref().map_or(0, |v| v.len());
        let bl = bmrf_lambda.as_ref().map_or(0, |v| v.len());
        let bc = bmrf_center.as_ref().map_or(0, |v| v.len());
        if bn != n_bmrf || bk != n_bmrf || bd != n_bmrf || bl != n_bmrf || bc != n_bmrf {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "bmrf arrays must all have the same length",
            ));
        }
    }
    let n_bspatial = bspatial_x1.as_ref().map_or(0, |v| v.len());
    if n_bspatial > 0 {
        let bx2 = bspatial_x2.as_ref().map_or(0, |v| v.len());
        let bn = bspatial_names.as_ref().map_or(0, |v| v.len());
        let bk = bspatial_knots.as_ref().map_or(0, |v| v.len());
        let bdeg = bspatial_degree.as_ref().map_or(0, |v| v.len());
        let bdiff = bspatial_differences.as_ref().map_or(0, |v| v.len());
        let bdf = bspatial_df.as_ref().map_or(0, |v| v.len());
        let blam = bspatial_lambda.as_ref().map_or(0, |v| v.len());
        let bc = bspatial_center.as_ref().map_or(0, |v| v.len());
        if bx2 != n_bspatial
            || bn != n_bspatial
            || bk != n_bspatial
            || bdeg != n_bspatial
            || bdiff != n_bspatial
            || bdf != n_bspatial
            || blam != n_bspatial
            || bc != n_bspatial
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "bspatial arrays must all have the same length",
            ));
        }
    }
    let n_bkernel = bkernel_x.as_ref().map_or(0, |v| v.len());
    if n_bkernel > 0 {
        let bn = bkernel_names.as_ref().map_or(0, |v| v.len());
        let bb = bkernel_bandwidth.as_ref().map_or(0, |v| v.len());
        let bd = bkernel_df.as_ref().map_or(0, |v| v.len());
        let bk = bkernel_kernel.as_ref().map_or(0, |v| v.len());
        let bc = bkernel_center.as_ref().map_or(0, |v| v.len());
        if bn != n_bkernel
            || bb != n_bkernel
            || bd != n_bkernel
            || bk != n_bkernel
            || bc != n_bkernel
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "bkernel arrays must all have the same length",
            ));
        }
    }
    let n_brandom = brandom_x.as_ref().map_or(0, |v| v.len());
    if n_brandom > 0 {
        let bn = brandom_names.as_ref().map_or(0, |v| v.len());
        let bnl = brandom_n_levels.as_ref().map_or(0, |v| v.len());
        let bd = brandom_df.as_ref().map_or(0, |v| v.len());
        let bl = brandom_lambda.as_ref().map_or(0, |v| v.len());
        let bc = brandom_center.as_ref().map_or(0, |v| v.len());
        if bn != n_brandom
            || bnl != n_brandom
            || bd != n_brandom
            || bl != n_brandom
            || bc != n_brandom
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "brandom arrays must all have the same length",
            ));
        }
    }
    let n_buser = buser_X.as_ref().map_or(0, |v| v.len());
    if n_buser > 0 {
        let nk = buser_K.as_ref().map_or(0, |v| v.len());
        let nn = buser_names.as_ref().map_or(0, |v| v.len());
        let nd = buser_df.as_ref().map_or(0, |v| v.len());
        let nl = buser_lambda.as_ref().map_or(0, |v| v.len());
        let nc = buser_center.as_ref().map_or(0, |v| v.len());
        if nk != n_buser || nn != n_buser || nd != n_buser || nl != n_buser || nc != n_buser {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "buser arrays must all have the same length",
            ));
        }
    }
    let bmono_solvers: Vec<BmonoSolver> = bmono_type
        .iter()
        .map(|s| parse_bmono_solver(s))
        .collect::<PyResult<Vec<_>>>()?;

    let y_arr = y.as_array();
    let w_arr = weights.as_array();
    let n = if matches!(family_name, "coxph" | "weibull" | "loglog") {
        y_arr.len() / 2
    } else {
        y_arr.len()
    };

    let scalar_cols: Vec<Vec<f64>> = scalar_x.iter().map(|sx| sx.as_array().to_vec()).collect();
    let bbs_cols: Vec<Vec<f64>> = bbs_x.iter().map(|bx| bx.as_array().to_vec()).collect();
    let bmono_cols: Vec<Vec<f64>> = bmono_x.iter().map(|bx| bx.as_array().to_vec()).collect();
    let n_bmono = bmono_cols.len();
    let bmono_constraints: Vec<MonoConstraint> = bmono_constraint
        .iter()
        .map(|s| match s.as_str() {
            "increasing" => MonoConstraint::Increasing,
            "decreasing" => MonoConstraint::Decreasing,
            "convex" => MonoConstraint::Convex,
            "concave" => MonoConstraint::Concave,
            "positive" => MonoConstraint::Positive,
            "negative" => MonoConstraint::Negative,
            _ => MonoConstraint::Increasing,
        })
        .collect();
    let btree_designs: Vec<ndarray::Array2<f64>> =
        btree_x.iter().map(|bx| bx.as_array().to_owned()).collect();

    let scalar_by_resolved: Vec<Option<Vec<f64>>> =
        scalar_by.unwrap_or_else(|| vec![None; n_scalar]);
    let bbs_by_resolved: Vec<Option<Vec<f64>>> = bbs_by.unwrap_or_else(|| vec![None; n_bbs]);
    let bmono_by_resolved: Vec<Option<Vec<f64>>> = bmono_by.unwrap_or_else(|| vec![None; n_bmono]);
    let btree_by_resolved: Vec<Option<Vec<f64>>> = btree_by.unwrap_or_else(|| vec![None; n_btree]);
    let bmrf_by_resolved: Vec<Option<Vec<f64>>> = bmrf_by.unwrap_or_else(|| vec![None; n_bmrf]);
    let bspatial_by_resolved: Vec<Option<Vec<f64>>> =
        bspatial_by.unwrap_or_else(|| vec![None; n_bspatial]);
    let bkernel_by_resolved: Vec<Option<Vec<f64>>> =
        bkernel_by.unwrap_or_else(|| vec![None; n_bkernel]);
    let brandom_by_resolved: Vec<Option<Vec<f64>>> =
        brandom_by.unwrap_or_else(|| vec![None; n_brandom]);
    let buser_by_resolved: Vec<Option<Vec<f64>>> = buser_by.unwrap_or_else(|| vec![None; n_buser]);

    let bmrf_levels_owned: Vec<Vec<usize>> = bmrf_levels.unwrap_or_default();
    let bmrf_names_owned: Vec<String> = bmrf_names.unwrap_or_default();
    let bmrf_k_owned: Vec<ndarray::Array2<f64>> = bmrf_k
        .unwrap_or_default()
        .iter()
        .map(|k| k.as_array().to_owned())
        .collect();
    let bmrf_df_owned: Vec<f64> = bmrf_df.unwrap_or_default();
    let bmrf_lambda_owned: Vec<f64> = bmrf_lambda.unwrap_or_default();
    let bmrf_center_owned: Vec<bool> = bmrf_center.unwrap_or_default();

    let bspatial_x1_owned: Vec<Vec<f64>> = bspatial_x1
        .unwrap_or_default()
        .iter()
        .map(|x| x.as_array().to_vec())
        .collect();
    let bspatial_x2_owned: Vec<Vec<f64>> = bspatial_x2
        .unwrap_or_default()
        .iter()
        .map(|x| x.as_array().to_vec())
        .collect();
    let bspatial_names_owned: Vec<String> = bspatial_names.unwrap_or_default();
    let bspatial_knots_owned: Vec<usize> = bspatial_knots.unwrap_or_default();
    let bspatial_degree_owned: Vec<usize> = bspatial_degree.unwrap_or_default();
    let bspatial_differences_owned: Vec<usize> = bspatial_differences.unwrap_or_default();
    let bspatial_df_owned: Vec<f64> = bspatial_df.unwrap_or_default();
    let bspatial_lambda_owned: Vec<f64> = bspatial_lambda.unwrap_or_default();
    let bspatial_center_owned: Vec<bool> = bspatial_center.unwrap_or_default();

    let brandom_cols_owned: Vec<ndarray::Array2<f64>> = brandom_x
        .unwrap_or_default()
        .iter()
        .map(|x| x.as_array().to_owned())
        .collect();
    let brandom_names_owned: Vec<String> = brandom_names.unwrap_or_default();
    let brandom_n_levels_owned: Vec<usize> = brandom_n_levels.unwrap_or_default();
    let brandom_df_owned: Vec<f64> = brandom_df.unwrap_or_default();
    let brandom_lambda_owned: Vec<f64> = brandom_lambda.unwrap_or_default();
    let brandom_center_owned: Vec<bool> = brandom_center.unwrap_or_default();

    let buser_X_owned: Vec<ndarray::Array2<f64>> = buser_X
        .unwrap_or_default()
        .iter()
        .map(|x| x.as_array().to_owned())
        .collect();
    let buser_K_owned: Vec<ndarray::Array2<f64>> = buser_K
        .unwrap_or_default()
        .iter()
        .map(|k| k.as_array().to_owned())
        .collect();
    let buser_names_owned: Vec<String> = buser_names.unwrap_or_default();
    let buser_df_owned: Vec<f64> = buser_df.unwrap_or_default();
    let buser_lambda_owned: Vec<f64> = buser_lambda.unwrap_or_default();
    let buser_center_owned: Vec<bool> = buser_center.unwrap_or_default();

    let bbs_templates = build_bbs_templates(
        &bbs_cols,
        &bbs_knots,
        &bbs_degree,
        &bbs_differences,
        &bbs_center,
        &bbs_cyclic,
        &bbs_boundary_knots_lo,
        &bbs_boundary_knots_hi,
        &bbs_deriv,
        &bbs_by_resolved,
    );
    let bspatial_templates = build_bspatial_templates(
        &bspatial_x1_owned,
        &bspatial_x2_owned,
        &bspatial_knots_owned,
        &bspatial_degree_owned,
        &bspatial_differences_owned,
        &bspatial_center_owned,
        &bspatial_by_resolved,
    );

    let preparer = GamBoostPreparer {
        scalar_cols,
        scalar_names,
        scalar_lambda,
        scalar_center,
        scalar_df,
        scalar_by: scalar_by_resolved,
        bbs_names,
        bbs_df,
        bbs_lambda,
        bbs_templates,
        bmono_cols,
        bmono_names,
        bmono_knots,
        bmono_degree,
        bmono_differences,
        bmono_df,
        bmono_constraint: bmono_constraints,
        bmono_solver: bmono_solvers,
        bmono_lambda2,
        bmono_niter,
        bmono_center,
        bmono_by: bmono_by_resolved,
        // `bmono_C` is `None` whenever no bmono learner uses a custom
        // constraint matrix (the common case: a named constraint like
        // "increasing"/"decreasing" builds its own D internally). Default
        // to one `None` per bmono column, not an empty Vec — the latter
        // panics on the first per-fold rebuild's `self.bmono_C[i]` index.
        bmono_C: bmono_C
            .unwrap_or_else(|| (0..n_bmono).map(|_| None).collect())
            .into_iter()
            .map(|c| c.map(|a| a.as_array().to_owned()))
            .collect(),
        btree_designs,
        btree_names,
        btree_minbucket,
        btree_minsplit,
        btree_maxdepth,
        btree_by: btree_by_resolved,
        bmrf_levels: bmrf_levels_owned,
        bmrf_names: bmrf_names_owned,
        bmrf_k: bmrf_k_owned,
        bmrf_df: bmrf_df_owned,
        bmrf_lambda: bmrf_lambda_owned,
        bmrf_center: bmrf_center_owned,
        bmrf_by: bmrf_by_resolved,
        bspatial_names: bspatial_names_owned,
        bspatial_df: bspatial_df_owned,
        bspatial_lambda: bspatial_lambda_owned,
        bspatial_templates,
        bkernel_cols: bkernel_x
            .unwrap_or_default()
            .iter()
            .map(|x| x.as_array().to_vec())
            .collect(),
        bkernel_names: bkernel_names.unwrap_or_default(),
        bkernel_bandwidth: bkernel_bandwidth.unwrap_or_default(),
        bkernel_df: bkernel_df.unwrap_or_default(),
        bkernel_kernel: bkernel_kernel.unwrap_or_default(),
        bkernel_center: bkernel_center.unwrap_or_default(),
        bkernel_by: bkernel_by_resolved,
        brandom_cols: brandom_cols_owned,
        brandom_names: brandom_names_owned,
        brandom_n_levels: brandom_n_levels_owned,
        brandom_df: brandom_df_owned,
        brandom_lambda: brandom_lambda_owned,
        brandom_center: brandom_center_owned,
        brandom_by: brandom_by_resolved,
        buser_X: buser_X_owned,
        buser_K: buser_K_owned,
        buser_names: buser_names_owned,
        buser_df: buser_df_owned,
        buser_lambda: buser_lambda_owned,
        buser_center: buser_center_owned,
        buser_by: buser_by_resolved,
        family_name: family_name.to_string(),
        family_params: family_params.clone(),
    };

    let folds_arr = fold_matrix_from_py(folds, n)?;
    let judge_arr = judge_mask_from_py(judge, n)?;

    let ctrl = BoostControl::new(mstop, nu)
        .map_err(mboostrs_error_to_pyerr)?
        .with_risk(risk_kind)
        .with_trace(trace);
    let ctrl = apply_step(ctrl, step, step_iterations)?;

    let y_owned = y_arr.to_owned();
    let w_owned = w_arr.to_owned();
    let offset = extract_offset_arg(offset.as_ref())?;
    let offset_vec_owned = offset
        .vector()
        .map(|values| ndarray::Array1::from(values.to_vec()));
    let offset_vec_view = offset_vec_owned.as_ref().map(|vector| vector.view());
    let family_name_owned = family_name.to_string();
    let family_params_owned = family_params.clone();
    let result = py
        .allow_threads(|| {
            with_concrete_family!(family_name_owned.as_str(), &family_params_owned, |fam| {
                try_cvrisk_with(
                    &preparer,
                    y_owned.view(),
                    w_owned.view(),
                    &ctrl,
                    &folds_arr,
                    offset.scalar(),
                    offset_vec_view,
                    judge_arr.as_ref(),
                    parallel,
                    fam,
                    fold_label,
                )
            })
        })?
        .map_err(mboostrs_error_to_pyerr)?;

    let opt = result.optimal_mstop();
    let grid = result.grid;
    let risk_matrix = result.risk_matrix.into_pyarray(py).unbind();
    let aggregate = result.aggregate.into_pyarray(py).unbind();

    Ok((risk_matrix, aggregate, grid, opt))
}

/// Holds prepared learners after fitting, for lazy AIC trace computation.
/// Avoids O(n²) hat matrix construction during fit — only computed on demand.
#[pyclass]
struct LearnerStore {
    learners: Vec<PreparedLearner>,
    nu: f64,
    /// Per-learner metadata for rebuilding design matrices from new x data.
    /// Set via `set_basis_info` after fitting.
    basis_info: Vec<BasisInfo>,
}

impl LearnerStore {
    /// Add step `idx`'s increment `nu * X_idx coef` to `eta` (length
    /// `n`, or `n * (K-1)` for a Kronecker learner).
    fn add_step(&self, eta: &mut [f64], idx: usize, coef: &[f64], n: usize) {
        let learner = &self.learners[idx];
        match learner {
            PreparedLearner::ScalarBols { design, .. } => {
                let c = coef[0];
                for i in 0..n {
                    eta[i] += self.nu * design[i] * c;
                }
            }
            PreparedLearner::BinnedScalarBols {
                bin_index,
                bin_values,
                ..
            } => {
                let coefficient = coef[0];
                for i in 0..n {
                    eta[i] += self.nu * bin_values[bin_index[i] as usize] * coefficient;
                }
            }
            PreparedLearner::BinnedBlockBols {
                bin_index, basis, ..
            } => {
                let mut bin_prediction = vec![0.0f64; basis.nrows()];
                for bin in 0..basis.nrows() {
                    for column in 0..basis.ncols() {
                        bin_prediction[bin] += basis[[bin, column]] * coef[column];
                    }
                }
                for i in 0..n {
                    eta[i] += self.nu * bin_prediction[bin_index[i] as usize];
                }
            }
            PreparedLearner::BlockBols { design, .. }
            | PreparedLearner::BmonoBbs { design, .. } => {
                let coef_arr = ndarray::Array1::from_vec(coef.to_vec());
                let pred = design.dot(&coef_arr);
                for i in 0..n {
                    eta[i] += self.nu * pred[i];
                }
            }
            PreparedLearner::Btree { design, .. } => {
                let coef_arr = ndarray::Array1::from_vec(coef.to_vec());
                let tree = FittedTree::from_coef_vec(coef_arr.view());
                let pred = tree.predict(design.view());
                for i in 0..n {
                    eta[i] += self.nu * pred[i];
                }
            }
            PreparedLearner::Bhinge { x, by, .. } => {
                let (knot, direction, coefficient, centering) =
                    (coef[0], coef[1], coef[2], coef[3]);
                let scale = by.as_ref();
                for i in 0..n {
                    let hinge = if direction > 0.0 {
                        (x[i] - knot).max(0.0)
                    } else {
                        (knot - x[i]).max(0.0)
                    };
                    let column = match scale {
                        Some(values) => hinge * values[i],
                        None => hinge,
                    };
                    eta[i] += self.nu * coefficient * (column - centering);
                }
            }
            PreparedLearner::FactoredBlockBols {
                x_l,
                groups,
                p_l,
                n_groups,
                ..
            } => {
                for i in 0..n {
                    let gr = groups[i] as usize;
                    let mut s = 0.0f64;
                    for l in 0..*p_l {
                        s += x_l[[i, l]] * coef[l * *n_groups + gr];
                    }
                    eta[i] += self.nu * s;
                }
            }
            PreparedLearner::BspatialStreamed { .. } => {
                let mut pred = vec![0.0f64; n];
                learner.predict_into(coef, &mut pred);
                for i in 0..n {
                    eta[i] += self.nu * pred[i];
                }
            }
            PreparedLearner::KroneckerBols {
                x1,
                x2,
                x2_is_identity,
                n_resp,
                ..
            } => {
                // pred[i*n_resp+k] = sum_j sum_l x1[i,j]*coef[l*p1+j]*x2[k,l]
                let p1 = x1.ncols();
                let p2 = x2.ncols();
                let n_obs = x1.nrows();
                if *x2_is_identity {
                    debug_assert_eq!(p2, *n_resp);
                    for i in 0..n_obs {
                        let row_offset = i * *n_resp;
                        for k in 0..*n_resp {
                            let coef_k = &coef[k * p1..(k + 1) * p1];
                            let mut s = 0.0f64;
                            for j in 0..p1 {
                                s += x1[[i, j]] * coef_k[j];
                            }
                            eta[row_offset + k] += self.nu * s;
                        }
                    }
                } else {
                    for i in 0..n_obs {
                        for kk in 0..*n_resp {
                            let mut s = 0.0f64;
                            for j in 0..p1 {
                                for l in 0..p2 {
                                    s += x1[[i, j]] * coef[l * p1 + j] * x2[[kk, l]];
                                }
                            }
                            eta[i * *n_resp + kk] += self.nu * s;
                        }
                    }
                }
            }
        }
    }

    /// Number of observations a new-data replay slice describes.
    ///
    /// Most learners take one value per observation; `Bspatial` packs
    /// `[x1, x2]`, and `Btree` / `Buser` take a row-major `(n, n_features)`
    /// design, so the row count comes from the learner's own recipe rather
    /// than from the slice length alone.
    fn replay_rows(&self, x_data: &[Vec<f64>]) -> PyResult<usize> {
        x_data
            .first()
            .zip(self.basis_info.first())
            .map(|(values, basis)| match basis {
                BasisInfo::Bspatial { .. } | BasisInfo::BinnedSpatial { .. } => values.len() / 2,
                BasisInfo::Btree { n_features, .. } => values.len() / *n_features,
                BasisInfo::Buser { n_features, .. } => values.len() / *n_features,
                _ => values.len(),
            })
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("empty x_data"))
    }

    fn check_replay_shapes(&self, x_data: &[Vec<f64>], n: usize) -> PyResult<()> {
        for (index, (values, basis)) in x_data.iter().zip(&self.basis_info).enumerate() {
            let expected = match basis {
                BasisInfo::Bspatial { .. } | BasisInfo::BinnedSpatial { .. } => 2 * n,
                BasisInfo::Btree { n_features, .. } => n * *n_features,
                BasisInfo::Buser { n_features, .. } => n * *n_features,
                _ => n,
            };
            if values.len() != expected {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "x_data[{index}] has length {}; expected {expected}",
                    values.len()
                )));
            }
        }
        Ok(())
    }

    fn check_by_shapes(by_data: &[Option<Vec<f64>>], n: usize) -> PyResult<()> {
        for (index, values) in by_data.iter().enumerate() {
            if let Some(values) = values {
                if values.len() != n {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "by_data[{index}] has length {}; expected {n}",
                        values.len()
                    )));
                }
            }
        }
        Ok(())
    }
}

#[pymethods]
impl LearnerStore {
    /// Compute AIC trace (degrees of freedom) on demand.
    /// Returns None if any selected learner lacks a hat matrix (bmono; a
    /// btree step resolves from its recorded coefficients when `coefs`
    /// supplies the fitted path).
    #[pyo3(signature = (selected, risk, sumw, y_ss, method="corrected", k=2.0, df=None, coefs=None))]
    #[allow(clippy::too_many_arguments)]
    fn compute_aic(
        &self,
        py: Python<'_>,
        selected: Vec<usize>,
        risk: Vec<f64>,
        sumw: f64,
        y_ss: f64,
        method: &str,
        k: f64,
        df: Option<Vec<f64>>,
        coefs: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Option<(Vec<f64>, usize, Vec<f64>)>> {
        let method_lower: String = method.to_lowercase();
        let method_normalized = method_lower.as_str();
        // `df`: hat-matrix traces computed by the caller (the IRLS-weighted
        // ones of a GLM family); otherwise the unweighted boosting traces.
        let trace = match df {
            Some(df) => Some(df),
            None => py
                .allow_threads(|| {
                    mboostrs_core::engine::try_compute_aic_trace_fast_steps(
                        &self.learners,
                        &selected,
                        coefs.as_deref(),
                        self.nu,
                    )
                })
                .map_err(mboostrs_error_to_pyerr)?,
        };
        match trace {
            Some(df) => {
                let (aic, mstop) = try_compute_aic(method_normalized, &risk, &df, sumw, k, y_ss)
                    .map_err(mboostrs_error_to_pyerr)?;
                Ok(Some((aic, mstop, df)))
            }
            None => Ok(None),
        }
    }
    fn hat_traces(&self, py: Python<'_>) -> Option<Vec<f64>> {
        py.allow_threads(|| {
            let n = self.learners.first()?.hat_matrix()?.nrows();
            let mut traces = Vec::with_capacity(self.learners.len());
            for learner in &self.learners {
                match learner.hat_matrix() {
                    Some(h) => {
                        let tr: f64 = (0..n).map(|i| h[[i, i]]).sum();
                        traces.push(tr * self.nu);
                    }
                    None => return None,
                }
            }
            Some(traces)
        })
    }

    /// Compute per-observation hat diagonal (leverage) at the final iteration.
    /// Returns None if any selected learner lacks a hat matrix (bmono; a
    /// btree step resolves from its recorded coefficients when `coefs`
    /// supplies the fitted path); raises on malformed inputs (out-of-bounds
    /// `selected`, non-finite nu).
    #[pyo3(signature = (selected, coefs=None))]
    fn compute_hat_diag_py(
        &self,
        py: Python<'_>,
        selected: Vec<usize>,
        coefs: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Option<Vec<f64>>> {
        py.allow_threads(|| {
            mboostrs_core::engine::try_compute_hat_diag_steps(
                &self.learners,
                &selected,
                coefs.as_deref(),
                self.nu,
            )
        })
        .map_err(mboostrs_error_to_pyerr)
    }

    /// mboost's weighted boosting hat matrix for a GLM family (`bhatmat`):
    /// `step_fits[m]` is the linear predictor after step `m`, offset
    /// included. Returns `(traces, final diagonal)`, or None when the family
    /// is L2 in mboost or a selected learner has no hat matrix.
    #[pyo3(signature = (selected, step_fits, family, coefs=None))]
    fn compute_hat_weighted(
        &self,
        py: Python<'_>,
        selected: Vec<usize>,
        step_fits: Vec<Vec<f64>>,
        family: &str,
        coefs: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Option<(Vec<f64>, Vec<f64>)>> {
        if mboostrs_core::engine::glm_hat_weight(family, 0.0).is_none() {
            return Ok(None);
        }
        let weights: Vec<Vec<f64>> = step_fits
            .iter()
            .map(|fits| {
                fits.iter()
                    .map(|&f| mboostrs_core::engine::glm_hat_weight(family, f).unwrap_or(1.0))
                    .collect()
            })
            .collect();
        py.allow_threads(|| {
            mboostrs_core::engine::try_compute_hat_weighted_steps(
                &self.learners,
                &selected,
                coefs.as_deref(),
                self.nu,
                &weights,
            )
        })
        .map_err(mboostrs_error_to_pyerr)
    }

    #[pyo3(signature = (which))]
    fn extract_design(&self, py: Python<'_>, which: usize) -> Option<Vec<Vec<f64>>> {
        py.allow_threads(|| {
            let learner = self.learners.get(which)?;
            let design = learner.design_matrix()?;
            let (n, p) = (design.nrows(), design.ncols());
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let mut row = Vec::with_capacity(p);
                for j in 0..p {
                    row.push(design[[i, j]]);
                }
                out.push(row);
            }
            Some(out)
        })
    }

    #[pyo3(signature = (which))]
    fn extract_penalty(&self, py: Python<'_>, which: usize) -> Option<Vec<Vec<f64>>> {
        py.allow_threads(|| {
            let learner = self.learners.get(which)?;
            let pen = learner.penalty_matrix()?;
            let (n, p) = (pen.nrows(), pen.ncols());
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let mut row = Vec::with_capacity(p);
                for j in 0..p {
                    row.push(pen[[i, j]]);
                }
                out.push(row);
            }
            Some(out)
        })
    }

    /// The value a single-column linear learner subtracts from its column
    /// (the weighted mean for a centered `bols`, 0 otherwise); `None` for
    /// any other learner.
    #[pyo3(signature = (which))]
    fn scalar_centering(&self, which: usize) -> Option<f64> {
        match self.learners.get(which)? {
            PreparedLearner::ScalarBols { centering, .. }
            | PreparedLearner::BinnedScalarBols { centering, .. } => Some(*centering),
            _ => None,
        }
    }

    #[pyo3(signature = (which))]
    fn extract_lambda(&self, which: usize) -> Option<f64> {
        self.learners.get(which)?.lambda()
    }

    #[pyo3(signature = (which))]
    fn extract_df(&self, py: Python<'_>, which: usize) -> Option<f64> {
        py.allow_threads(|| self.learners.get(which)?.df())
    }

    /// `tr(2H - HᵀH)`, which is the quantity `mboost`'s `df` names and the
    /// one its `df2lambda` inverts. See `PreparedLearner::df_mboost`.
    #[pyo3(signature = (which))]
    fn extract_df_mboost(&self, py: Python<'_>, which: usize) -> Option<f64> {
        py.allow_threads(|| self.learners.get(which)?.df_mboost())
    }

    /// Per learner `(selections, single_step, boosted, unpenalized)` df, or
    /// `None` for a learner without a closed-form smoother. See
    /// `mboostrs_core::LearnerDrift`.
    #[pyo3(signature = (selected))]
    fn penalty_drift(
        &self,
        py: Python<'_>,
        selected: Vec<usize>,
    ) -> PyResult<Vec<Option<(usize, f64, f64, f64)>>> {
        let drift = py
            .allow_threads(|| mboostrs_core::penalty_drift(&self.learners, &selected, self.nu))
            .map_err(mboostrs_error_to_pyerr)?;
        Ok(drift
            .into_iter()
            .map(|d| d.map(|d| (d.selections, d.single_step, d.boosted, d.unpenalized)))
            .collect())
    }

    fn learner_names(&self) -> Vec<String> {
        self.learners.iter().map(|l| l.name().to_string()).collect()
    }

    /// mgcv's concurvity measures of the boosted model's terms, computed in
    /// the core from the prepared designs (`mboostrs_core::boosted_concurvity`):
    /// `summary[i]` is store learner `included[i]` against the intercept
    /// offset and every other term (mgcv's `concurvity(full = TRUE)`), and
    /// `pairwise[i][j]` is term `included[j]` measured against term
    /// `included[i]` (mgcv's `full = FALSE`, diagonal 1, no intercept).
    /// `selected[s]`/`coefs_vecs[s]` are the boosting path the fit returned;
    /// the fitted coefficients of a learner are `nu` times the sum of its
    /// steps' updates, and `observed` is NaN for a learner that was never
    /// selected. Learners without a function-space basis (trees, `%O%`
    /// Kronecker learners) are excluded from all three outputs, and every
    /// measured term is first reduced by the constant direction, as mgcv
    /// measures its constrained smooths; a term that is constant after the
    /// reduction has no function space left and reports NaN.
    #[pyo3(signature = (selected, coefs_vecs))]
    fn concurvity(
        &self,
        py: Python<'_>,
        selected: Vec<usize>,
        coefs_vecs: Vec<Vec<f64>>,
    ) -> PyResult<(Vec<usize>, Vec<(f64, f64, f64)>, Vec<Vec<(f64, f64, f64)>>)> {
        let (included, mc) = py
            .allow_threads(|| {
                mboostrs_core::gam::boosted_concurvity(
                    &self.learners,
                    self.nu,
                    &selected,
                    &coefs_vecs,
                )
            })
            .map_err(mboostrs_error_to_pyerr)?;
        let flat = |c: mboostrs_core::gam::Concurvity| (c.worst, c.observed, c.estimate);
        Ok((
            included,
            mc.summary.iter().map(|&c| flat(c)).collect(),
            mc.pairwise
                .iter()
                .map(|row| row.iter().map(|&c| flat(c)).collect())
                .collect(),
        ))
    }

    /// Fast predict using cached learner designs. Skips the expensive
    /// B-spline basis / centering rebuild that `predict_gamboost` does.
    /// The family's full held-out loss (`Family::held_out_risk`) on the
    /// training rows after each step 0..=mstop of the path. For Multinomial
    /// this is the negative log-likelihood over all classes, which the in-bag
    /// risk path (mboost's loss, reference class left out) is not; variable
    /// importance attributes each step's change in it. `y` and `w` are in the
    /// fit's layout (K-1 entries per row for Multinomial).
    #[allow(clippy::too_many_arguments)]
    fn staged_held_out_risk(
        &self,
        py: Python<'_>,
        offset: Bound<'_, PyAny>,
        selected: Vec<usize>,
        coefs_vecs: Vec<Bound<'_, PyAny>>,
        family_name: &str,
        family_params: HashMap<String, f64>,
        y: Vec<f64>,
        w: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        let coefs_vecs = extract_f64_vecs(coefs_vecs)?;
        let family = make_family(family_name, &family_params)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let n = self
            .learners
            .first()
            .map(|l| l.n())
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("empty learner store"))?;
        let offset = extract_offset_arg(Some(&offset))?;
        let mut eta = eta_start(&offset, n)?;
        if y.len() != eta.len() || w.len() != eta.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "y and w must have {} entries, got {} and {}",
                eta.len(),
                y.len(),
                w.len()
            )));
        }
        if coefs_vecs.len() < selected.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "fewer coefficient vectors than selected steps",
            ));
        }
        Ok(py.allow_threads(|| {
            let y = ndarray::Array1::from_vec(y);
            let w = ndarray::Array1::from_vec(w);
            let mut out = Vec::with_capacity(selected.len() + 1);
            let eval = |eta: &Vec<f64>| {
                let f = ndarray::Array1::from_vec(eta.clone());
                family.held_out_risk(y.view(), f.view(), w.view())
            };
            out.push(eval(&eta));
            for (m, &idx) in selected.iter().enumerate() {
                self.add_step(&mut eta, idx, &coefs_vecs[m], n);
                out.push(eval(&eta));
            }
            out
        }))
    }

    #[pyo3(signature = (offset, selected, coefs_vecs, mstop=None))]
    fn predict(
        &self,
        py: Python<'_>,
        offset: Bound<'_, PyAny>,
        selected: Vec<usize>,
        coefs_vecs: Vec<Bound<'_, PyAny>>,
        mstop: Option<usize>,
    ) -> PyResult<Vec<f64>> {
        let coefs_vecs = extract_f64_vecs(coefs_vecs)?;
        let max_m = mstop.unwrap_or(selected.len()).min(selected.len());
        let n = self
            .learners
            .first()
            .map(|l| l.n())
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("empty learner store"))?;
        let offset = extract_offset_arg(Some(&offset))?;
        let start = eta_start(&offset, n)?;

        if coefs_vecs.len() < max_m {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "fewer coefficient vectors than selected steps",
            ));
        }
        let eta = py.allow_threads(|| {
            let mut eta = start;
            // Every learner except a tree or an adaptive hinge is linear in
            // its coefficients, so the steps that chose it collapse into one
            // summed update: one design product per learner instead of one
            // per step. Slicing a long path of a wide learner (bspatial) is
            // otherwise a full-design product for every step it replays. A
            // tree's vector encodes its structure and a hinge's encodes
            // `[knot, direction, coefficient, centering]` — two selections
            // do not add, so both replay step by step.
            let mut summed: Vec<Option<Vec<f64>>> = vec![None; self.learners.len()];
            for m in 0..max_m {
                let idx = selected[m];
                let coef = &coefs_vecs[m];
                if matches!(
                    self.learners.get(idx),
                    Some(PreparedLearner::Btree { .. } | PreparedLearner::Bhinge { .. })
                ) {
                    self.add_step(&mut eta, idx, coef, n);
                    continue;
                }
                match summed.get_mut(idx) {
                    Some(Some(total)) if total.len() == coef.len() => {
                        for (slot, value) in total.iter_mut().zip(coef) {
                            *slot += value;
                        }
                    }
                    Some(slot @ None) => *slot = Some(coef.clone()),
                    // A length change along the path is not expected; replay
                    // that step on its own rather than guess.
                    _ => self.add_step(&mut eta, idx, coef, n),
                }
            }
            for (idx, coef) in summed.iter().enumerate() {
                if let Some(coef) = coef {
                    self.add_step(&mut eta, idx, coef, n);
                }
            }
            eta
        });

        Ok(eta)
    }

    /// Store per-learner basis reconstruction metadata for new-data prediction.
    /// Called once after fitting. The metadata enables `predict_newdata` to
    /// rebuild B-spline bases and apply cached centering transforms without
    /// recomputing eigendecompositions or marshalling 40+ arguments through PyO3.
    #[pyo3(signature = (
        scalar_center, scalar_centering,
        bbs_center, bbs_differences, bbs_n_knots,
        bbs_degree, bbs_boundary_knots, bbs_cyclic, bbs_deriv,
    ))]
    fn set_basis_info(
        &mut self,
        py: Python<'_>,
        scalar_center: Vec<bool>,
        scalar_centering: Vec<f64>,
        bbs_center: Vec<bool>,
        bbs_differences: Vec<usize>,
        bbs_n_knots: Vec<usize>,
        bbs_degree: Vec<usize>,
        bbs_boundary_knots: Vec<(f64, f64)>,
        bbs_cyclic: Vec<bool>,
        bbs_deriv: Vec<usize>,
    ) -> PyResult<()> {
        py.allow_threads(|| {
            let total = self.learners.len();
            self.basis_info = Vec::with_capacity(total);
            let n_scalar = scalar_center.len();
            let n_bbs = bbs_center.len();
            for i in 0..n_scalar {
                self.basis_info.push(BasisInfo::ScalarBols {
                    centering: scalar_centering[i],
                    by: None,
                });
            }
            for i in 0..n_bbs {
                let bk = bbs_boundary_knots[i];
                let info = if bbs_center[i] {
                    let knots = bspline_knots(bk, bbs_n_knots[i], bbs_degree[i]);
                    let p = knots.len() - bbs_degree[i] - 1;
                    let k = diff_penalty(p, bbs_differences[i]);
                    let (cmat, r) = compute_centering_transform(&k, p);
                    BasisInfo::BbsCentered {
                        knots,
                        degree: bbs_degree[i],
                        boundary_knots: bk,
                        centering_mat: cmat,
                        centering_rows: p,
                        centering_cols: r,
                        by: None,
                        deriv: bbs_deriv[i],
                    }
                } else if bbs_cyclic[i] {
                    let knots = cyclic_bspline_knots(bk, bbs_n_knots[i], bbs_degree[i]);
                    BasisInfo::BbsUncentered {
                        knots,
                        degree: bbs_degree[i],
                        boundary_knots: bk,
                        by: None,
                        cyclic: true,
                        deriv: bbs_deriv[i],
                    }
                } else {
                    let knots = bspline_knots(bk, bbs_n_knots[i], bbs_degree[i]);
                    BasisInfo::BbsUncentered {
                        knots,
                        degree: bbs_degree[i],
                        boundary_knots: bk,
                        by: None,
                        cyclic: false,
                        deriv: bbs_deriv[i],
                    }
                };
                self.basis_info.push(info);
            }
            for learner in self.learners.iter().skip(n_scalar + n_bbs) {
                self.basis_info.push(BasisInfo::Unsupported {
                    kind: learner.name().to_string(),
                    reason: "new-data metadata is not available for this learner in LearnerStore"
                        .to_string(),
                });
            }
            Ok(())
        })
    }

    /// Returns the number of learners with basis info populated.
    /// Used by Python to check whether the fast new-data predict path is available.
    fn basis_info_len(&self) -> usize {
        self.basis_info.len()
    }

    /// Fast new-data prediction using cached basis info.
    /// Accepts raw x values (one `Vec<f64>` per learner, in learner-index order)
    /// and runs the prediction loop without Python re-entry.
    ///
    /// `by_data` supplies fresh newdata `by=` (varying-coefficient) values, one
    /// entry per learner in the same indexing as `x_data` / `self.basis_info`.
    /// `Some(v)` overrides the training-length `by` values cached on the
    /// learner's `BasisInfo`; `None` (or a missing entry) falls back to those
    /// cached values, matching the pre-existing (in-sample replay) behavior.
    #[pyo3(signature = (offset, selected, coefs_vecs, x_data, mstop=None, by_data=None, unseen_levels=None))]
    fn predict_newdata(
        &self,
        py: Python<'_>,
        offset: Bound<'_, PyAny>,
        selected: Vec<usize>,
        coefs_vecs: Vec<Bound<'_, PyAny>>,
        x_data: Vec<Bound<'_, PyAny>>,
        mstop: Option<usize>,
        by_data: Option<Vec<Option<Bound<'_, PyAny>>>>,
        unseen_levels: Option<&str>,
    ) -> PyResult<Vec<f64>> {
        let coefs_vecs = extract_f64_vecs(coefs_vecs)?;
        let x_data = extract_f64_vecs(x_data)?;
        let by_data = extract_opt_f64_vecs(by_data)?;
        let max_m = mstop.unwrap_or(selected.len()).min(selected.len());
        let n = self.replay_rows(&x_data)?;
        self.check_replay_shapes(&x_data, n)?;
        Self::check_by_shapes(&by_data, n)?;
        let unseen = parse_unseen_levels(unseen_levels)?;
        let offset = extract_offset_arg(Some(&offset))?;
        let start = eta_start(&offset, n)?;

        let eta = py.allow_threads(|| {
            let mut eta = start;
            for m in 0..max_m {
                let idx = selected[m];
                let coef = &coefs_vecs[m];
                let x_slice = &x_data[idx];
                if let Some(bi) = self.basis_info.get(idx) {
                    let by_override = by_data.get(idx).and_then(|o| o.as_deref());
                    bi.predict_into_with(coef, x_slice, &mut eta, self.nu, by_override, unseen)?;
                }
            }
            Ok::<Vec<f64>, MboostrsError>(eta)
        });
        eta.map_err(mboostrs_error_to_pyerr)
    }

    /// New-data prediction that evaluates each learner's basis once.
    ///
    /// `predict_newdata` walks the boosting path and rebuilds a learner's basis
    /// on every step that selected it, which costs O(mstop) B-spline
    /// evaluations. Here the coefficients of the steps that selected a learner
    /// are summed first, so each basis is built once — O(n_learners) — which is
    /// what makes this usable as the ordinary `predict(newdata=)` path rather
    /// than only as an in-sample replay.
    ///
    /// Summing is valid because every learner is linear in its coefficients.
    /// `Btree` and `Bhinge` are the exceptions: a tree's coefficient vector
    /// encodes a fitted tree's structure and a hinge's encodes
    /// `[knot, direction, coefficient, centering]`, so two selections do not
    /// add and those steps stay per-step. They cost nothing extra — neither
    /// replay builds a basis.
    #[pyo3(signature = (offset, selected, coefs_vecs, x_data, mstop=None, by_data=None, unseen_levels=None, n_rows=None))]
    fn predict_newdata_grouped(
        &self,
        py: Python<'_>,
        offset: Bound<'_, PyAny>,
        selected: Vec<usize>,
        coefs_vecs: Vec<Bound<'_, PyAny>>,
        x_data: Vec<Bound<'_, PyAny>>,
        mstop: Option<usize>,
        by_data: Option<Vec<Option<Bound<'_, PyAny>>>>,
        unseen_levels: Option<&str>,
        n_rows: Option<usize>,
    ) -> PyResult<Vec<f64>> {
        let coefs_vecs = extract_f64_vecs(coefs_vecs)?;
        let x_data = extract_f64_vecs(x_data)?;
        let by_data = extract_opt_f64_vecs(by_data)?;
        let max_m = mstop.unwrap_or(selected.len()).min(selected.len());
        // An offset-only model has no learner data to read the row count from,
        // so the caller supplies it; the prediction is then the offset.
        let n = match (x_data.is_empty(), n_rows) {
            (true, Some(rows)) => rows,
            _ => self.replay_rows(&x_data)?,
        };
        self.check_replay_shapes(&x_data, n)?;
        Self::check_by_shapes(&by_data, n)?;
        let unseen = parse_unseen_levels(unseen_levels)?;
        let offset = extract_offset_arg(Some(&offset))?;
        let start = eta_start(&offset, n)?;

        let eta = py.allow_threads(|| {
            let mut eta = start;
            // Learner index -> summed coefficients across the steps that chose it.
            let mut summed: Vec<Option<Vec<f64>>> = vec![None; self.basis_info.len()];
            for m in 0..max_m {
                let idx = selected[m];
                let coef = &coefs_vecs[m];
                let Some(basis) = self.basis_info.get(idx) else {
                    continue;
                };
                if matches!(basis, BasisInfo::Btree { .. } | BasisInfo::Bhinge { .. }) {
                    let by_override = by_data.get(idx).and_then(|o| o.as_deref());
                    basis.predict_into_with(
                        coef,
                        &x_data[idx],
                        &mut eta,
                        self.nu,
                        by_override,
                        unseen,
                    )?;
                    continue;
                }
                match &mut summed[idx] {
                    Some(total) => {
                        if total.len() != coef.len() {
                            return Err(MboostrsError::Shape(format!(
                                "learner {idx} produced coefficient vectors of differing \
                                 lengths ({} then {}) along the path",
                                total.len(),
                                coef.len()
                            )));
                        }
                        for (slot, value) in total.iter_mut().zip(coef) {
                            *slot += value;
                        }
                    }
                    None => summed[idx] = Some(coef.clone()),
                }
            }
            for (idx, coef) in summed.iter().enumerate() {
                let Some(coef) = coef else { continue };
                let by_override = by_data.get(idx).and_then(|o| o.as_deref());
                self.basis_info[idx].predict_into_with(
                    coef,
                    &x_data[idx],
                    &mut eta,
                    self.nu,
                    by_override,
                    unseen,
                )?;
            }
            Ok::<Vec<f64>, MboostrsError>(eta)
        });
        eta.map_err(mboostrs_error_to_pyerr)
    }

    #[pyo3(signature = (offset, selected, coefs_vecs, x_data, y, weights, family_name, family_params, metric, parameter=None, by_data=None, unseen_levels=None))]
    fn score_newdata_path(
        &self,
        py: Python<'_>,
        offset: Bound<'_, PyAny>,
        selected: Vec<usize>,
        coefs_vecs: Vec<Bound<'_, PyAny>>,
        x_data: Vec<Bound<'_, PyAny>>,
        y: Bound<'_, PyAny>,
        weights: Bound<'_, PyAny>,
        family_name: &str,
        family_params: HashMap<String, f64>,
        metric: &str,
        parameter: Option<f64>,
        by_data: Option<Vec<Option<Bound<'_, PyAny>>>>,
        unseen_levels: Option<&str>,
    ) -> PyResult<Vec<f64>> {
        let coefs_vecs = extract_f64_vecs(coefs_vecs)?;
        let x_data = extract_f64_vecs(x_data)?;
        let y = extract_f64_vec(&y)?;
        let weights = extract_f64_vec(&weights)?;
        let by = extract_opt_f64_vecs(by_data)?;
        // The observed response supplies the row count even when an
        // intercept-only fit has no learner slices to inspect. The core
        // validates response, weights, offsets and replay dimensions.
        let n = y.len();
        self.check_replay_shapes(&x_data, n)?;
        Self::check_by_shapes(&by, n)?;
        let offset = eta_start(&extract_offset_arg(Some(&offset))?, n)?;
        let unseen = parse_unseen_levels(unseen_levels)?;
        let family = make_family(family_name, &family_params)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        py.allow_threads(|| {
            mboostrs_core::scoring::response_prediction_path(
                &self.basis_info,
                &selected,
                &coefs_vecs,
                &x_data,
                &by,
                &offset,
                self.nu,
                &y,
                &weights,
                family.as_ref(),
                metric,
                parameter,
                unseen,
            )
        })
        .map_err(mboostrs_error_to_pyerr)
    }

    /// New-data predictions at several path positions, building each basis
    /// once for the whole grid.
    ///
    /// `predict_newdata_grouped` builds every learner's basis once *per
    /// call*, so a caller comparing six stopping points paid for six sets of
    /// B-spline evaluations to change nothing but how many path steps were
    /// summed. Measured on 20,000 rows and eight `bbs` learners, a grid of
    /// six cost 52.1 ms against 8.7 for one.
    ///
    /// The grid must be sorted ascending, which lets the coefficient sums
    /// carry forward: position `k`'s sums are position `k-1`'s plus the steps
    /// between them. `Btree` and `Bhinge` steps are replayed per step as they
    /// are in the grouped path, for the same reason — neither selection's
    /// coefficient vector adds to the next.
    #[pyo3(signature = (offset, selected, coefs_vecs, x_data, mstops, by_data=None, unseen_levels=None))]
    fn predict_newdata_grid(
        &self,
        py: Python<'_>,
        offset: Bound<'_, PyAny>,
        selected: Vec<usize>,
        coefs_vecs: Vec<Bound<'_, PyAny>>,
        x_data: Vec<Bound<'_, PyAny>>,
        mstops: Vec<usize>,
        by_data: Option<Vec<Option<Bound<'_, PyAny>>>>,
        unseen_levels: Option<&str>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let coefs_vecs = extract_f64_vecs(coefs_vecs)?;
        let x_data = extract_f64_vecs(x_data)?;
        let by_data = extract_opt_f64_vecs(by_data)?;
        if mstops.windows(2).any(|pair| pair[1] < pair[0]) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "mstops must be sorted ascending: the grid is walked once and \
                 each position continues the previous one's sums",
            ));
        }
        let n = self.replay_rows(&x_data)?;
        self.check_replay_shapes(&x_data, n)?;
        Self::check_by_shapes(&by_data, n)?;
        let unseen = parse_unseen_levels(unseen_levels)?;
        let offset = extract_offset_arg(Some(&offset))?;
        let start = eta_start(&offset, n)?;

        let out = py.allow_threads(|| {
            let path_len = selected.len();
            let mut summed: Vec<Option<Vec<f64>>> = vec![None; self.basis_info.len()];
            // Tree and hinge contributions accumulate into their own vector,
            // because they are replayed step by step and must not be
            // recomputed for every grid position.
            let mut tree_eta = vec![0.0f64; n];
            let mut walked = 0usize;
            let mut results = Vec::with_capacity(mstops.len());

            for &stop in &mstops {
                let target = stop.min(path_len);
                while walked < target {
                    let idx = selected[walked];
                    let coef = &coefs_vecs[walked];
                    walked += 1;
                    let Some(basis) = self.basis_info.get(idx) else {
                        continue;
                    };
                    if matches!(basis, BasisInfo::Btree { .. } | BasisInfo::Bhinge { .. }) {
                        let by_override = by_data.get(idx).and_then(|o| o.as_deref());
                        basis.predict_into_with(
                            coef,
                            &x_data[idx],
                            &mut tree_eta,
                            self.nu,
                            by_override,
                            unseen,
                        )?;
                        continue;
                    }
                    match &mut summed[idx] {
                        Some(total) => {
                            if total.len() != coef.len() {
                                return Err(MboostrsError::Shape(format!(
                                    "learner {idx} produced coefficient vectors of \
                                     differing lengths ({} then {}) along the path",
                                    total.len(),
                                    coef.len()
                                )));
                            }
                            for (slot, value) in total.iter_mut().zip(coef) {
                                *slot += value;
                            }
                        }
                        None => summed[idx] = Some(coef.clone()),
                    }
                }

                let mut eta = start.clone();
                for (row, value) in eta.iter_mut().zip(&tree_eta) {
                    *row += value;
                }
                for (idx, coef) in summed.iter().enumerate() {
                    let Some(coef) = coef else { continue };
                    let by_override = by_data.get(idx).and_then(|o| o.as_deref());
                    self.basis_info[idx].predict_into_with(
                        coef,
                        &x_data[idx],
                        &mut eta,
                        self.nu,
                        by_override,
                        unseen,
                    )?;
                }
                results.push(eta);
            }
            Ok::<Vec<Vec<f64>>, MboostrsError>(results)
        });
        out.map_err(mboostrs_error_to_pyerr)
    }

    /// Predictions at each value of one covariate, evaluating every learner
    /// that reads that covariate once per grid value and every other learner
    /// once in total. The numerics are `mboostrs_core::scoring::predict_over_grid`.
    ///
    /// `grid_x` and `grid_by` map a learner index to the values its covariate
    /// or `by=` takes at each of the `grid_len` positions. Returns one
    /// linear predictor per position, each of length `n`.
    #[pyo3(signature = (offset, selected, coefs_vecs, x_data, by_data, grid_len, grid_x, grid_by, mstop=None, unseen_levels=None, n_rows=None))]
    fn predict_over_grid(
        &self,
        py: Python<'_>,
        offset: Bound<'_, PyAny>,
        selected: Vec<usize>,
        coefs_vecs: Vec<Bound<'_, PyAny>>,
        x_data: Vec<Bound<'_, PyAny>>,
        by_data: Option<Vec<Option<Bound<'_, PyAny>>>>,
        grid_len: usize,
        grid_x: Vec<(usize, Bound<'_, PyAny>)>,
        grid_by: Vec<(usize, Bound<'_, PyAny>)>,
        mstop: Option<usize>,
        unseen_levels: Option<&str>,
        n_rows: Option<usize>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let coefs_vecs = extract_f64_vecs(coefs_vecs)?;
        let x_data = extract_f64_vecs(x_data)?;
        let by_data = extract_opt_f64_vecs(by_data)?;
        let mut grid_x_values = Vec::with_capacity(grid_x.len());
        for (index, values) in &grid_x {
            grid_x_values.push((*index, extract_f64_vec(values)?));
        }
        let mut grid_by_values = Vec::with_capacity(grid_by.len());
        for (index, values) in &grid_by {
            grid_by_values.push((*index, extract_f64_vec(values)?));
        }
        let max_m = mstop.unwrap_or(selected.len()).min(selected.len());
        // An offset-only model has no learner data to read the row count from.
        let n = match (x_data.is_empty(), n_rows) {
            (true, Some(rows)) => rows,
            _ => self.replay_rows(&x_data)?,
        };
        self.check_replay_shapes(&x_data, n)?;
        Self::check_by_shapes(&by_data, n)?;
        let unseen = parse_unseen_levels(unseen_levels)?;
        let offset = extract_offset_arg(Some(&offset))?;
        let start = eta_start(&offset, n)?;
        py.allow_threads(|| {
            mboostrs_core::scoring::predict_over_grid(
                &self.basis_info,
                &selected,
                &coefs_vecs,
                &start,
                &x_data,
                &by_data,
                self.nu,
                max_m,
                &grid_x_values,
                &grid_by_values,
                grid_len,
                unseen,
            )
        })
        .map_err(mboostrs_error_to_pyerr)
    }

    /// New-data prediction for Kronecker models (Multinomial family).
    ///
    /// Like `predict_newdata`, but each learner is a `KroneckerBols` whose
    /// coefficients are packed as `(n_resp * p1)` per selection step.  Returns
    /// `vec![f64; n * n_resp]` in row-major order (observation-major,
    /// class-minor).
    #[pyo3(signature = (offset, selected, coefs_vecs, x_data, n_resp, mstop=None))]
    fn predict_newdata_kronecker(
        &self,
        py: Python<'_>,
        offset: f64,
        selected: Vec<usize>,
        coefs_vecs: Vec<Bound<'_, PyAny>>,
        x_data: Vec<PyReadonlyArray1<'_, f64>>,
        n_resp: usize,
        mstop: Option<usize>,
    ) -> PyResult<Vec<f64>> {
        let coefs_vecs = extract_f64_vecs(coefs_vecs)?;
        let max_m = mstop.unwrap_or(selected.len()).min(selected.len());
        let n = x_data
            .first()
            .map(|v| v.as_array().len())
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("empty x_data"))?;
        let x_data: Vec<Vec<f64>> = x_data.iter().map(|a| a.as_array().to_vec()).collect();
        let eta = py.allow_threads(|| {
            let mut eta = vec![offset; n * n_resp];

            // The predictor is linear in the coefficients and, for a fixed
            // learner, the rebuilt x1 design is identical at every step that
            // selected it. Aggregate each learner's coefficients over its
            // selected steps first (one O(mstop) pass), then rebuild x1 and
            // apply once per *distinct* learner — instead of once per
            // boosting step. Same aggregate-then-apply pattern as
            // `predict_glmboost` / `_multinomial_newdata_eta`. Indexed by
            // learner index (not a HashMap) so iteration order — and hence
            // the floating-point summation order into `eta` — is
            // deterministic across runs.
            let mut agg: Vec<Option<Vec<f64>>> = vec![None; self.learners.len()];
            for m in 0..max_m {
                let idx = selected[m];
                let coef = &coefs_vecs[m];
                match &mut agg[idx] {
                    Some(entry) => {
                        for (a, c) in entry.iter_mut().zip(coef.iter()) {
                            *a += c;
                        }
                    }
                    slot @ None => *slot = Some(coef.clone()),
                }
            }

            for (idx, coef) in agg
                .iter()
                .enumerate()
                .filter_map(|(i, c)| c.as_ref().map(|c| (i, c)))
            {
                let x_slice = &x_data[idx];
                let learner = &self.learners[idx];
                let (x1_rebuild, p1, x2_is_identity) = match learner {
                    PreparedLearner::KroneckerBols {
                        x1,
                        x2_is_identity,
                        n_resp: nr,
                        x1_rebuild,
                        ..
                    } => {
                        if *nr != n_resp {
                            return Err(format!(
                                "KroneckerBols n_resp mismatch: learner has {nr}, expected {n_resp}"
                            ));
                        }
                        (x1_rebuild, x1.ncols(), x2_is_identity)
                    }
                    other => {
                        return Err(format!("learner '{}' is not a KroneckerBols", other.name()));
                    }
                };
                let x1 = rebuild_kronecker_x1(x_slice, x1_rebuild, n, p1)?;
                let n_obs = x1.nrows();
                if *x2_is_identity {
                    for i in 0..n_obs {
                        let row_offset = i * n_resp;
                        for k in 0..n_resp {
                            let coef_k = &coef[k * p1..(k + 1) * p1];
                            let mut s = 0.0f64;
                            for j in 0..p1 {
                                s += x1[[i, j]] * coef_k[j];
                            }
                            eta[row_offset + k] += self.nu * s;
                        }
                    }
                } else {
                    let x2 = match learner {
                        PreparedLearner::KroneckerBols { x2, .. } => x2,
                        _ => unreachable!(),
                    };
                    for i in 0..n_obs {
                        for k in 0..n_resp {
                            let mut s = 0.0f64;
                            for j in 0..p1 {
                                for l in 0..x2.ncols() {
                                    s += x1[[i, j]] * coef[l * p1 + j] * x2[[k, l]];
                                }
                            }
                            eta[i * n_resp + k] += self.nu * s;
                        }
                    }
                }
            }
            Ok::<Vec<f64>, String>(eta)
        });
        eta.map_err(pyo3::exceptions::PyValueError::new_err)
    }
}

/// Rebuild the x1 design matrix for a Kronecker learner from raw x values.
fn rebuild_kronecker_x1(
    x_slice: &[f64],
    x1_rebuild: &Option<KroneckerRebuild>,
    n: usize,
    p1: usize,
) -> Result<ndarray::Array2<f64>, String> {
    match x1_rebuild {
        Some(KroneckerRebuild::BolsIntercept) => {
            let mut x1 = ndarray::Array2::ones((n, 2));
            for i in 0..n {
                x1[[i, 1]] = x_slice[i];
            }
            Ok(x1)
        }
        Some(KroneckerRebuild::BolsNoIntercept) => {
            let mut x1 = ndarray::Array2::zeros((n, 1));
            for i in 0..n {
                x1[[i, 0]] = x_slice[i];
            }
            Ok(x1)
        }
        Some(KroneckerRebuild::Bbs {
            knots,
            degree,
            boundary_knots,
            deriv,
            transform,
        }) => {
            let raw = if *deriv > 0 {
                bspline_basis_with_deriv(x_slice, knots, *degree, *deriv)
            } else {
                bspline_basis_extrap(x_slice, knots, *degree, *boundary_knots)
            };
            // A spectrally centred term replays the same reparametrization.
            let basis = match transform {
                Some((flat, q)) => {
                    let t = ndarray::Array2::from_shape_vec((raw.ncols(), *q), flat.clone())
                        .map_err(|e| format!("centering transform shape: {e}"))?;
                    raw.dot(&t)
                }
                None => raw,
            };
            if basis.ncols() != p1 {
                return Err(format!(
                    "rebuilt x1 has {} columns but expected {}",
                    basis.ncols(),
                    p1
                ));
            }
            Ok(basis)
        }
        Some(KroneckerRebuild::Levels {
            level_codes,
            centering,
        }) => {
            if level_codes.len() != p1 || centering.len() != p1 {
                return Err(format!(
                    "level replay has {} levels but the learner has {} columns",
                    level_codes.len(),
                    p1
                ));
            }
            let mut x1 = ndarray::Array2::zeros((n, p1));
            for i in 0..n {
                let code = x_slice[i];
                if !code.is_finite() || code.fract() != 0.0 {
                    return Err(format!(
                        "level at row {i} must be a whole number; got {code}"
                    ));
                }
                // A level the fit never saw keeps a zero row: zero effect.
                if let Ok(column) = level_codes.binary_search(&(code as i64)) {
                    for j in 0..p1 {
                        x1[[i, j]] = -centering[j];
                    }
                    x1[[i, column]] += 1.0;
                }
            }
            Ok(x1)
        }
        None => Err("KroneckerBols learner has no x1_rebuild metadata; \
             the model was fitted without new-data prediction support"
            .to_string()),
    }
}

#[pyfunction]
#[pyo3(signature = (theta, which, b, r))]
fn stabsel_d_rc(theta: f64, which: usize, b: usize, r: f64) -> PyResult<f64> {
    try_d_rc(theta, which, b, r).map_err(mboostrs_error_to_pyerr)
}

#[pyfunction]
#[pyo3(signature = (q, p, cutoff, b))]
fn stabsel_min_d(q: usize, p: usize, cutoff: f64, b: usize) -> PyResult<f64> {
    try_min_d(q, p, cutoff, b).map_err(mboostrs_error_to_pyerr)
}

#[pyfunction]
#[pyo3(signature = (p, b))]
fn stabsel_max_q(p: usize, b: usize) -> PyResult<usize> {
    try_max_q(p, b).map_err(mboostrs_error_to_pyerr)
}

#[pyfunction]
#[pyo3(signature = (cutoff, b, theta))]
fn stabsel_um_const(cutoff: f64, b: usize, theta: f64) -> PyResult<f64> {
    // R's stabs::um_const stop()s with "cutoff out of bounds" for cutoffs below
    // the unimodal validity bound (or > 1). The core mirrors that as a panic;
    // surface it as a catchable ValueError (matching R's recoverable error)
    // instead of an uncatchable PanicException. The condition is identical to
    // the core's guard in `um_const`.
    let bf = b as f64;
    let out_of_bounds = if cutoff <= 0.75 {
        cutoff < 0.5 + theta.powi(2).min(1.0 / (2.0 * bf) + 0.75 * theta.powi(2))
    } else {
        cutoff > 1.0
    };
    if out_of_bounds {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "cutoff out of bounds",
        ));
    }
    try_um_const(cutoff, b, theta).map_err(mboostrs_error_to_pyerr)
}

/// PropOdds per-category response probabilities. Given the linear predictor `f`
/// (length n) and the K-1 fitted thresholds `sigma`, returns the row-major
/// (n x K) probability matrix flattened to a Vec (caller reshapes to (n, K)).
#[pyfunction]
#[pyo3(signature = (f, sigma))]
fn propodds_response(f: Vec<f64>, sigma: Vec<f64>) -> PyResult<Vec<f64>> {
    Ok(mboostrs_core::propodds_response(&f, &sigma))
}

#[pyfunction]
#[pyo3(signature = (p, cutoff=None, q=None, pfer=None, b=50, assumption="unimodal", sampling_type="SS"))]
fn stabsel_params(
    py: Python<'_>,
    p: usize,
    cutoff: Option<f64>,
    q: Option<usize>,
    pfer: Option<f64>,
    b: usize,
    assumption: &str,
    sampling_type: &str,
) -> PyResult<PyObject> {
    use pyo3::types::PyDict;
    let result = py
        .allow_threads(|| rust_stabsel_parameters(p, cutoff, q, pfer, b, assumption, sampling_type))
        .map_err(mboostrs_error_to_pyerr)?;
    let dict = PyDict::new(py);
    dict.set_item("cutoff", result.cutoff)?;
    dict.set_item("q", result.q)?;
    dict.set_item("PFER", result.pfer)?;
    dict.set_item("p", result.p)?;
    dict.set_item("B", result.b)?;
    dict.set_item("sampling_type", result.sampling_type)?;
    dict.set_item("assumption", result.assumption)?;
    Ok(dict.into_pyobject(py)?.into_any().unbind())
}

#[pyfunction]
fn student_t_pdf(y: Vec<f64>, mu: Vec<f64>, sigma: Vec<f64>, df: Vec<f64>) -> PyResult<Vec<f64>> {
    mboostrs_core::lss::student_t_pdf(&y, &mu, &sigma, &df).map_err(mboostrs_error_to_pyerr)
}

#[pyfunction]
fn centered_group_effects(coef: Vec<f64>, centering: Vec<f64>) -> PyResult<Vec<f64>> {
    mboostrs_core::path::centered_group_effects(&coef, &centering).map_err(mboostrs_error_to_pyerr)
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    mboostrs_core::linalg::configure_faer();
    m.add_function(wrap_pyfunction!(student_t_pdf, m)?)?;
    m.add_function(wrap_pyfunction!(centered_group_effects, m)?)?;
    selection_bindings::register(m)?;
    selection_stats::register(m)?;
    m.add_function(wrap_pyfunction!(gam_refit::gam_refit_core, m)?)?;
    m.add_class::<gam_refit::GamRefitCore>()?;
    m.add_function(wrap_pyfunction!(gam_refit::gam_refit_bootstrap_core, m)?)?;
    m.add_class::<gam_refit::GamRefitBootstrapCore>()?;
    m.add_function(wrap_pyfunction!(gam_refit::gamlss_refit_core, m)?)?;
    m.add_class::<gam_refit::GamlssRefitCore>()?;
    m.add_function(wrap_pyfunction!(gam_refit::term_basis_design, m)?)?;
    m.add_function(wrap_pyfunction!(gam_refit::row_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(gam_refit::kron, m)?)?;
    m.add_function(wrap_pyfunction!(gam_refit::gam_refit_core_from_json, m)?)?;
    m.add_function(wrap_pyfunction!(gam_refit::gamlss_refit_core_from_json, m)?)?;
    transform_bindings::register(m)?;
    m.add_function(wrap_pyfunction!(native_build_info, m)?)?;
    m.add_function(wrap_pyfunction!(boost_control, m)?)?;
    m.add_function(wrap_pyfunction!(fit_glmboost, m)?)?;
    m.add_function(wrap_pyfunction!(fit_gamboost, m)?)?;
    m.add_function(wrap_pyfunction!(fit_gamboost_specs, m)?)?;
    m.add_function(wrap_pyfunction!(fit_gamboost_continue_specs, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_learner_store, m)?)?;
    m.add_function(wrap_pyfunction!(predict_glmboost, m)?)?;
    m.add_function(wrap_pyfunction!(predict_glmboost_path, m)?)?;
    m.add_function(wrap_pyfunction!(coef_glmboost_py, m)?)?;
    m.add_function(wrap_pyfunction!(predict_gamboost, m)?)?;
    m.add_function(wrap_pyfunction!(bbs_raw_design, m)?)?;
    m.add_function(wrap_pyfunction!(family_diagnostics, m)?)?;
    m.add_function(wrap_pyfunction!(family_class, m)?)?;
    m.add_function(wrap_pyfunction!(glmboost_design, m)?)?;
    m.add_function(wrap_pyfunction!(same_family_glm_fit, m)?)?;
    m.add_function(wrap_pyfunction!(lss_family_diagnostics, m)?)?;
    m.add_function(wrap_pyfunction!(lss_row_nll, m)?)?;
    m.add_function(wrap_pyfunction!(cv_weights, m)?)?;
    m.add_function(wrap_pyfunction!(paired_gain, m)?)?;
    m.add_function(wrap_pyfunction!(specification_curve_sign_test, m)?)?;
    m.add_function(wrap_pyfunction!(rloss_transform, m)?)?;
    m.add_function(wrap_pyfunction!(rloss, m)?)?;
    m.add_function(wrap_pyfunction!(rloss_transform_weighted, m)?)?;
    m.add_function(wrap_pyfunction!(winsorize_pseudo_outcome, m)?)?;
    m.add_function(wrap_pyfunction!(binomial_deviance, m)?)?;
    m.add_function(wrap_pyfunction!(modifier_scale, m)?)?;
    m.add_function(wrap_pyfunction!(augmented_scores, m)?)?;
    m.add_function(wrap_pyfunction!(influence_variance, m)?)?;
    m.add_function(wrap_pyfunction!(mean_and_iid_se, m)?)?;
    m.add_function(wrap_pyfunction!(subgroup_scores, m)?)?;
    m.add_function(wrap_pyfunction!(effective_sample_size, m)?)?;
    m.add_function(wrap_pyfunction!(median_repeat_combine, m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap_percentile_interval, m)?)?;
    m.add_function(wrap_pyfunction!(log_residual_variance, m)?)?;
    m.add_function(wrap_pyfunction!(two_part_mean, m)?)?;
    m.add_function(wrap_pyfunction!(gcm_test, m)?)?;
    m.add_function(wrap_pyfunction!(gcm_test_multi, m)?)?;
    m.add_function(wrap_pyfunction!(iv_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(iv_ratio_weighted, m)?)?;
    m.add_function(wrap_pyfunction!(effect_robust_wls, m)?)?;
    m.add_function(wrap_pyfunction!(effect_wald_test, m)?)?;
    m.add_function(wrap_pyfunction!(effect_linear_predict, m)?)?;
    m.add_function(wrap_pyfunction!(effect_gates, m)?)?;
    m.add_function(wrap_pyfunction!(effect_holm_adjust, m)?)?;
    m.add_function(wrap_pyfunction!(linear_dependence, m)?)?;
    m.add_function(wrap_pyfunction!(column_diagnostics, m)?)?;
    m.add_function(wrap_pyfunction!(contribution_shares, m)?)?;
    m.add_function(wrap_pyfunction!(paired_mean_se, m)?)?;
    m.add_function(wrap_pyfunction!(discovery_learning_rate, m)?)?;
    m.add_function(wrap_pyfunction!(discovery_mean_step, m)?)?;
    m.add_function(wrap_pyfunction!(prefer_simpler_cv, m)?)?;
    m.add_function(wrap_pyfunction!(leftover_share, m)?)?;
    m.add_function(wrap_pyfunction!(preexplore_json, m)?)?;
    m.add_function(wrap_pyfunction!(response_score, m)?)?;
    m.add_function(wrap_pyfunction!(breslow_survfit, m)?)?;
    m.add_function(wrap_pyfunction!(breslow_baseline, m)?)?;
    m.add_function(wrap_pyfunction!(breslow_curves, m)?)?;
    m.add_function(wrap_pyfunction!(breslow_horizon_survival, m)?)?;
    m.add_function(wrap_pyfunction!(ipcw_brier_at, m)?)?;
    m.add_function(wrap_pyfunction!(ipcw_auc_at, m)?)?;
    m.add_function(wrap_pyfunction!(declaration_quantiles, m)?)?;
    m.add_function(wrap_pyfunction!(declaration_sign_check, m)?)?;
    m.add_function(wrap_pyfunction!(declaration_shape_check, m)?)?;
    m.add_function(wrap_pyfunction!(family_held_out_score, m)?)?;
    m.add_function(wrap_pyfunction!(family_held_out_rows, m)?)?;
    m.add_function(wrap_pyfunction!(concurvity_measures, m)?)?;
    m.add_function(wrap_pyfunction!(brad_basis, m)?)?;
    m.add_function(wrap_pyfunction!(brad_design, m)?)?;
    m.add_function(wrap_pyfunction!(brad_covariance, m)?)?;
    m.add_function(wrap_pyfunction!(brad_effective_range, m)?)?;
    m.add_function(wrap_pyfunction!(risk_path_rises, m)?)?;
    m.add_function(wrap_pyfunction!(special_function, m)?)?;
    m.add_function(wrap_pyfunction!(varimp_attribution, m)?)?;
    m.add_function(wrap_pyfunction!(group_bootstrap_counts, m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap_counts, m)?)?;
    m.add_function(wrap_pyfunction!(percentile_interval, m)?)?;
    m.add_function(wrap_pyfunction!(lss_quantile, m)?)?;
    m.add_function(wrap_pyfunction!(joint_support, m)?)?;
    m.add_function(wrap_pyfunction!(cell_nearest_distance, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_average, m)?)?;
    m.add_function(wrap_pyfunction!(residual_cluster_icc, m)?)?;
    m.add_function(wrap_pyfunction!(residual_serial_dependence, m)?)?;
    m.add_function(wrap_pyfunction!(residual_influence, m)?)?;
    m.add_function(wrap_pyfunction!(residual_cluster_false_alarm_rate, m)?)?;
    m.add_function(wrap_pyfunction!(residual_serial_false_alarm_rate, m)?)?;
    m.add_function(wrap_pyfunction!(residual_influence_false_alarm_rate, m)?)?;
    m.add_function(wrap_pyfunction!(response_score_path, m)?)?;
    m.add_function(wrap_pyfunction!(search_gamboostlss_grid, m)?)?;
    m.add_function(wrap_pyfunction!(search_fold_scores, m)?)?;
    m.add_function(wrap_pyfunction!(inverse_response_transform, m)?)?;
    m.add_function(wrap_pyfunction!(conformal_min_calibration_size, m)?)?;
    m.add_function(wrap_pyfunction!(conformal_calibration_count, m)?)?;
    m.add_function(wrap_pyfunction!(conformal_quantile, m)?)?;
    m.add_function(wrap_pyfunction!(conformal_worst_covariate_bin, m)?)?;
    m.add_function(wrap_pyfunction!(impute_model_fit, m)?)?;
    m.add_function(wrap_pyfunction!(impute_model_predict, m)?)?;
    m.add_function(wrap_pyfunction!(conformal_absolute_residuals, m)?)?;
    m.add_function(wrap_pyfunction!(conformal_symmetric_band, m)?)?;
    m.add_function(wrap_pyfunction!(conformal_location_scale_scores, m)?)?;
    m.add_function(wrap_pyfunction!(conformal_location_scale_band, m)?)?;
    m.add_function(wrap_pyfunction!(conformal_poisson_scores, m)?)?;
    m.add_function(wrap_pyfunction!(conformal_poisson_band, m)?)?;
    m.add_function(wrap_pyfunction!(conformal_count_default_score, m)?)?;
    m.add_function(wrap_pyfunction!(conformal_count_reference, m)?)?;
    m.add_function(wrap_pyfunction!(conformal_count_uniforms, m)?)?;
    m.add_function(wrap_pyfunction!(conformal_count_rank_scores, m)?)?;
    m.add_function(wrap_pyfunction!(conformal_count_rank_band, m)?)?;
    m.add_function(wrap_pyfunction!(conformal_count_cardinalities, m)?)?;
    m.add_function(wrap_pyfunction!(conformal_ordered_pair, m)?)?;
    m.add_function(wrap_pyfunction!(rearrange_levels, m)?)?;
    m.add_function(wrap_pyfunction!(conformal_quantile_scores, m)?)?;
    m.add_function(wrap_pyfunction!(conformal_widen_band, m)?)?;
    m.add_function(wrap_pyfunction!(conformal_band_midpoint, m)?)?;
    m.add_function(wrap_pyfunction!(conformal_clip_to_band, m)?)?;
    m.add_function(wrap_pyfunction!(conformal_cv_plus_ranks, m)?)?;
    m.add_function(wrap_pyfunction!(conformal_cv_plus_guaranteed_coverage, m)?)?;
    m.add_function(wrap_pyfunction!(conformal_cv_plus_bounds, m)?)?;
    m.add_function(wrap_pyfunction!(conformal_tuned_mstop_start, m)?)?;
    m.add_function(wrap_pyfunction!(conformal_tuned_mstop_choice, m)?)?;
    m.add_function(wrap_pyfunction!(conformal_tuned_mstop_next, m)?)?;
    m.add_function(wrap_pyfunction!(select_score_path, m)?)?;
    m.add_function(wrap_pyfunction!(crps_normal, m)?)?;
    m.add_function(wrap_pyfunction!(crps_discrete_sum, m)?)?;
    m.add_function(wrap_pyfunction!(crps_cdf_quadrature, m)?)?;
    m.add_function(wrap_pyfunction!(auc_delong, m)?)?;
    m.add_function(wrap_pyfunction!(cvrisk_foldwise, m)?)?;
    m.add_function(wrap_pyfunction!(group_components, m)?)?;
    m.add_function(wrap_pyfunction!(h_statistic, m)?)?;
    m.add_function(wrap_pyfunction!(screen_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(disc_scan, m)?)?;
    m.add_function(wrap_pyfunction!(disc_step_z, m)?)?;
    m.add_function(wrap_pyfunction!(screen_main_effects, m)?)?;
    m.add_function(wrap_pyfunction!(cvrisk_glmboost, m)?)?;
    m.add_function(wrap_pyfunction!(cvrisk_gamboost, m)?)?;
    m.add_function(wrap_pyfunction!(cvrisk_gamboost_specs, m)?)?;
    m.add_function(wrap_pyfunction!(cvrisk_gamboost_specs_extending, m)?)?;
    m.add_function(wrap_pyfunction!(cvrisk_selected_gamboost_specs, m)?)?;
    m.add_function(wrap_pyfunction!(cvrisk_selected_glmboost, m)?)?;
    m.add_function(wrap_pyfunction!(stabsel_d_rc, m)?)?;
    m.add_function(wrap_pyfunction!(stabsel_min_d, m)?)?;
    m.add_function(wrap_pyfunction!(stabsel_max_q, m)?)?;
    m.add_function(wrap_pyfunction!(stabsel_um_const, m)?)?;
    m.add_function(wrap_pyfunction!(stabsel_params, m)?)?;
    m.add_function(wrap_pyfunction!(propodds_response, m)?)?;
    m.add_function(wrap_pyfunction!(fit_glmboostlss, m)?)?;
    m.add_function(wrap_pyfunction!(fit_gamboostlss, m)?)?;
    m.add_function(wrap_pyfunction!(lss_auto_response_scale, m)?)?;
    m.add_function(wrap_pyfunction!(fit_gamboostlss_mixed, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_gamboostlss_stores, m)?)?;
    m.add_function(wrap_pyfunction!(cvrisk_glmboostlss, m)?)?;
    m.add_function(wrap_pyfunction!(cvrisk_gamboostlss, m)?)?;
    m.add_function(wrap_pyfunction!(cvrisk_gamboostlss_mixed, m)?)?;
    m.add_function(wrap_pyfunction!(stabsel_selected_gamboostlss, m)?)?;
    m.add_function(wrap_pyfunction!(stabsel_selected_gamboostlss_mixed, m)?)?;
    m.add_function(wrap_pyfunction!(fit_glmboostlss_noncyclic, m)?)?;
    m.add_function(wrap_pyfunction!(stabsel_selected_glmboostlss, m)?)?;
    m.add_class::<LearnerStore>()?;
    Ok(())
}
