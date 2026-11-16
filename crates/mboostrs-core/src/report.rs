//! Versioned, structured execution metadata for fitted boosting paths.

use std::cell::Cell;
use std::time::Instant;

thread_local! {
    static PREPARATION_FACTORIZATION_NS: Cell<Option<u64>> = const { Cell::new(None) };
}

/// Exclusive learner-preparation phase timings collected by a frontend.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PreparationTiming {
    /// Design/basis/penalty and weighted-Gram preparation, excluding solver
    /// factorization.
    pub basis_ns: u64,
    /// Time inside cached Gram-solver construction.
    pub factorization_ns: u64,
    /// Entire measured learner-preparation interval.
    pub total_ns: u64,
}

/// Opt-in preparation timer used by language bindings.
///
/// While active on the current thread, `GramSolver` construction contributes
/// to `factorization_ns`. No clock is read when no timer is active.
pub struct PreparationTimer {
    started: Instant,
    previous_factorization_ns: Option<u64>,
    finished: bool,
}

impl PreparationTimer {
    /// Begin measuring preparation time, nesting under any ongoing
    /// factorization clock.
    pub fn start() -> Self {
        let previous_factorization_ns =
            PREPARATION_FACTORIZATION_NS.with(|slot| slot.replace(Some(0)));
        Self {
            started: Instant::now(),
            previous_factorization_ns,
            finished: false,
        }
    }

    /// Stop the timer and return the split of basis-setting vs. factorization
    /// time.
    pub fn finish(mut self) -> PreparationTiming {
        let total_ns = self.started.elapsed().as_nanos() as u64;
        let factorization_ns = PREPARATION_FACTORIZATION_NS.with(|slot| {
            let measured = slot.get().unwrap_or(0);
            slot.set(self.previous_factorization_ns);
            measured
        });
        self.finished = true;
        PreparationTiming {
            basis_ns: total_ns.saturating_sub(factorization_ns),
            factorization_ns,
            total_ns,
        }
    }
}

impl Drop for PreparationTimer {
    fn drop(&mut self) {
        if !self.finished {
            PREPARATION_FACTORIZATION_NS.with(|slot| {
                slot.set(self.previous_factorization_ns);
            });
        }
    }
}

pub(crate) fn measure_factorization<T>(operation: impl FnOnce() -> T) -> T {
    let enabled = PREPARATION_FACTORIZATION_NS.with(|slot| slot.get().is_some());
    if !enabled {
        return operation();
    }
    let started = Instant::now();
    let result = operation();
    let elapsed = started.elapsed().as_nanos() as u64;
    PREPARATION_FACTORIZATION_NS.with(|slot| {
        if let Some(current) = slot.get() {
            slot.set(Some(current.saturating_add(elapsed)));
        }
    });
    result
}

/// Amount of execution instrumentation requested for a fit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum InstrumentationLevel {
    /// Record deterministic route metadata only; do not read the clock.
    #[default]
    Off,
    /// Add phase timings and aggregate counters.
    Summary,
    /// Reserved for per-iteration path rows.
    Iterations,
}

impl InstrumentationLevel {
    /// Stable string tag for this level (round-trips through `FromStr`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Summary => "summary",
            Self::Iterations => "iterations",
        }
    }
}

impl std::str::FromStr for InstrumentationLevel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "off" => Ok(Self::Off),
            "summary" => Ok(Self::Summary),
            "iterations" => Ok(Self::Iterations),
            other => Err(format!("unknown instrumentation level {other:?}")),
        }
    }
}

/// Numerical engine route that produced a fitted path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EngineRoute {
    #[default]
    /// No route was recorded (e.g. fit was never run).
    Unknown,
    /// Single packed scalar learners.
    PackedScalar,
    /// Mixed packed-scalar and dense block learners.
    Hybrid,
    /// All block learners, selected in parallel.
    ParallelBlock,
    /// General per-iteration projection route.
    General,
    /// Covariance/Gram fast path for Gaussian single fits.
    GaussianGram,
    /// Cyclic LSS fit (one parameter per cycle step).
    LssCyclic,
    /// Non-cyclic LSS fit (all parameters updated per step).
    LssNoncyclic,
}

impl EngineRoute {
    /// Stable string tag for this route.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::PackedScalar => "packed_scalar",
            Self::Hybrid => "hybrid",
            Self::ParallelBlock => "parallel_block",
            Self::General => "general",
            Self::GaussianGram => "gaussian_gram",
            Self::LssCyclic => "lss_cyclic",
            Self::LssNoncyclic => "lss_noncyclic",
        }
    }
}

impl std::str::FromStr for EngineRoute {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "unknown" => Ok(Self::Unknown),
            "packed_scalar" => Ok(Self::PackedScalar),
            "hybrid" => Ok(Self::Hybrid),
            "parallel_block" => Ok(Self::ParallelBlock),
            "general" => Ok(Self::General),
            "gaussian_gram" => Ok(Self::GaussianGram),
            "lss_cyclic" => Ok(Self::LssCyclic),
            "lss_noncyclic" => Ok(Self::LssNoncyclic),
            other => Err(format!("unknown engine route {other:?}")),
        }
    }
}

/// Why the Gaussian Gram fast path was not selected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FallbackReason {
    /// The caller asked to store per-iteration increments.
    IncrementStorageRequested,
    /// The caller asked to store per-iteration coefficients.
    CoefficientsNotStored,
    /// The caller asked to store the risk path.
    RiskNotStored,
    /// The family does not support the Gram projection shortcut.
    FamilyDoesNotSupportGram,
    /// The requested risk kind is not in-bag.
    RiskIsNotInbag,
    /// The model is pure-scalar (handled by the packed path instead).
    PureScalarModel,
    /// At least one learner type is unsupported by the Gram route.
    UnsupportedLearnerForGram,
}

impl FallbackReason {
    /// Stable string tag for this reason.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::IncrementStorageRequested => "increment_storage_requested",
            Self::CoefficientsNotStored => "coefficients_not_stored",
            Self::RiskNotStored => "risk_not_stored",
            Self::FamilyDoesNotSupportGram => "family_does_not_support_gram",
            Self::RiskIsNotInbag => "risk_is_not_inbag",
            Self::PureScalarModel => "pure_scalar_model",
            Self::UnsupportedLearnerForGram => "unsupported_learner_for_gram",
        }
    }
}

impl std::str::FromStr for FallbackReason {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "increment_storage_requested" => Ok(Self::IncrementStorageRequested),
            "coefficients_not_stored" => Ok(Self::CoefficientsNotStored),
            "risk_not_stored" => Ok(Self::RiskNotStored),
            "family_does_not_support_gram" => Ok(Self::FamilyDoesNotSupportGram),
            "risk_is_not_inbag" => Ok(Self::RiskIsNotInbag),
            "pure_scalar_model" => Ok(Self::PureScalarModel),
            "unsupported_learner_for_gram" => Ok(Self::UnsupportedLearnerForGram),
            other => Err(format!("unknown fallback reason {other:?}")),
        }
    }
}

/// Stable summary of how a fit was executed.
#[derive(Clone, Debug, PartialEq)]
pub struct FitReport {
    /// Artifact schema version of this report.
    pub schema_version: u32,
    /// Instrumentation level the fit ran under.
    pub instrumentation: InstrumentationLevel,
    /// Numerical route that produced the path.
    pub route: EngineRoute,
    /// Number of observations fitted.
    pub n_observations: usize,
    /// Number of candidate base learners.
    pub n_learners: usize,
    /// Total design columns across all learners.
    pub total_basis_columns: usize,
    /// Iterations the caller asked the engine to run.
    pub requested_mstop: usize,
    /// Iterations actually completed (less than requested only on early break).
    pub completed_mstop: usize,
    /// Threads used for learner selection.
    pub effective_threads: usize,
    /// Whether the Gaussian Gram fast path was eligible (before learner gates).
    pub gram_eligible: bool,
    /// Reasons the Gram path was not taken, empty when it was.
    pub fallback_reasons: Vec<FallbackReason>,
    /// Number of scalar closed-form learners retained by the fit.
    pub scalar_learner_count: usize,
    /// Number of vector/block learners retained by the fit.
    pub block_learner_count: usize,
    /// Number of tree learners retained by the fit.
    pub tree_learner_count: usize,
    /// Cached Cholesky factorizations used by prepared block learners.
    pub cholesky_solver_count: usize,
    /// Cached LU factorizations used after a Cholesky rejection.
    pub lu_solver_count: usize,
    /// Singular systems solved after explicit diagonal regularization.
    pub regularized_solver_count: usize,
    /// Largest diagonal jitter applied to a regularized solver.
    pub max_solver_jitter: Option<f64>,
    /// Bytes retained by cached learner factorizations.
    pub factor_cache_bytes: usize,
    /// Dynamic learner designs, bin indices, weights, and work buffers retained
    /// during the boosting loop (solver factors reported separately).
    pub retained_learner_bytes: usize,
    /// Bytes materialized for the Gaussian route's full cross-Gram cache
    /// and exact sparse product caches (cross-Grams and penalties).
    pub gram_cache_bytes: usize,
    /// Family-specific nuisance calibration calls that performed work.
    pub calibration_count: usize,
    /// Learner/parameter selections where multiple candidates had the exact
    /// winning score and the deterministic lowest-index rule resolved it.
    pub tie_selection_count: usize,
    /// Iterations that evaluated only a random subset of candidate learners
    /// (stochastic component subsampling). Zero when subsampling is off.
    pub component_subsample_iterations: usize,
    /// Iterative family calibrations that met their declared convergence rule.
    pub calibration_converged_count: usize,
    /// Iterative family calibrations that exhausted their iteration limit.
    pub calibration_limit_count: usize,
    /// Iterative family calibrations that produced a non-finite candidate.
    pub calibration_failure_count: usize,
    /// Candidate updates that fell back to a larger `nu` after a non-finite try.
    pub nonfinite_candidate_fallbacks: usize,
    /// Whole-run preparation wall time up to fit dispatch.
    pub setup_ns: Option<u64>,
    /// Frontend learner design/basis/penalty and weighted-Gram preparation,
    /// excluding cached solver factorization.
    pub basis_ns: Option<u64>,
    /// Cached Gram-solver construction during frontend learner preparation.
    pub factorization_ns: Option<u64>,
    /// Gaussian fast-path cross-Gram materialization time.
    pub cross_gram_ns: Option<u64>,
    /// Wall time per iteration (or per cycle for LSS fits).
    pub iteration_ns: Option<u64>,
    /// Final fitted-value reconstruction time for routes that replay totals.
    pub reconstruction_ns: Option<u64>,
    /// Frontend input/output conversion and learner-preparation time, excluding
    /// the core fit. Populated by language bindings, not pure Rust callers.
    pub binding_ns: Option<u64>,
    /// Whole-run wall time including conversion and learner preparation.
    pub total_ns: Option<u64>,
}

/// One derived row of a fitted boosting path.
///
/// Rows are created on demand from retained path state, so ordinary fitting
/// pays no allocation or clock overhead. `risk` and `risk_change` are absent
/// when the engine did not record risk at that individual update (notably
/// intermediate parameter updates inside a cyclic LSS cycle).
///
/// This is the schema the Python and R `path_frame()` outputs mirror
/// field-for-field. Both bindings add two resolved-name columns on top of it —
/// `parameter` (this row's `parameter_index` looked up in the LSS parameter
/// names) and `learner` (its `learner_index` looked up in the learner names) —
/// plus `scaled_coefficient_norm`, which is `nu * coefficient_norm` and needs
/// the control object the core row does not carry. Renaming a field here
/// without renaming it in `python/mboostrs/report.py` and
/// `mboostrs/R/mboostrs_methods.R` breaks that correspondence.
#[derive(Clone, Debug, PartialEq)]
pub struct PathRow {
    /// Global boosting step (1-based, matching R's iteration numbering).
    pub iteration: usize,
    /// Cycle number for LSS cyclic fits (`None` for single-parameter fits).
    pub cycle: Option<usize>,
    /// Index of the distribution parameter this update belongs to, or `None`
    /// for a single-parameter fit. Named `parameter_index` rather than
    /// `parameter` because the bindings' `parameter` column holds the
    /// parameter's *name*.
    pub parameter_index: Option<usize>,
    /// Index of the base learner updated at this step.
    pub learner_index: usize,
    /// Risk after this update, when the engine recorded it.
    pub risk: Option<f64>,
    /// Signed change in risk from the previous recorded value.
    pub risk_change: Option<f64>,
    /// Euclidean norm of `nu` times the updated coefficient block.
    pub coefficient_norm: f64,
    /// Family nuisance values at this update (empty for most families).
    pub nuisance: Vec<f64>,
}

impl Default for FitReport {
    fn default() -> Self {
        Self {
            schema_version: 1,
            instrumentation: InstrumentationLevel::Off,
            route: EngineRoute::Unknown,
            n_observations: 0,
            n_learners: 0,
            total_basis_columns: 0,
            requested_mstop: 0,
            completed_mstop: 0,
            effective_threads: 1,
            gram_eligible: false,
            fallback_reasons: Vec::new(),
            scalar_learner_count: 0,
            block_learner_count: 0,
            tree_learner_count: 0,
            cholesky_solver_count: 0,
            lu_solver_count: 0,
            regularized_solver_count: 0,
            max_solver_jitter: None,
            factor_cache_bytes: 0,
            retained_learner_bytes: 0,
            gram_cache_bytes: 0,
            calibration_count: 0,
            tie_selection_count: 0,
            component_subsample_iterations: 0,
            calibration_converged_count: 0,
            calibration_limit_count: 0,
            calibration_failure_count: 0,
            nonfinite_candidate_fallbacks: 0,
            setup_ns: None,
            basis_ns: None,
            factorization_ns: None,
            cross_gram_ns: None,
            iteration_ns: None,
            reconstruction_ns: None,
            binding_ns: None,
            total_ns: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::GramSolver;
    use std::str::FromStr;

    #[test]
    fn instrumentation_level_round_trips_through_as_str() {
        for level in [
            InstrumentationLevel::Off,
            InstrumentationLevel::Summary,
            InstrumentationLevel::Iterations,
        ] {
            assert_eq!(
                InstrumentationLevel::from_str(level.as_str()).unwrap(),
                level
            );
        }
        assert!(InstrumentationLevel::from_str("bogus").is_err());
    }

    #[test]
    fn engine_route_round_trips_through_as_str() {
        for route in [
            EngineRoute::Unknown,
            EngineRoute::PackedScalar,
            EngineRoute::Hybrid,
            EngineRoute::ParallelBlock,
            EngineRoute::General,
            EngineRoute::GaussianGram,
            EngineRoute::LssCyclic,
            EngineRoute::LssNoncyclic,
        ] {
            assert_eq!(EngineRoute::from_str(route.as_str()).unwrap(), route);
        }
        assert!(EngineRoute::from_str("bogus").is_err());
    }

    #[test]
    fn fallback_reason_round_trips_through_as_str() {
        for reason in [
            FallbackReason::IncrementStorageRequested,
            FallbackReason::CoefficientsNotStored,
            FallbackReason::RiskNotStored,
            FallbackReason::FamilyDoesNotSupportGram,
            FallbackReason::RiskIsNotInbag,
            FallbackReason::PureScalarModel,
            FallbackReason::UnsupportedLearnerForGram,
        ] {
            assert_eq!(FallbackReason::from_str(reason.as_str()).unwrap(), reason);
        }
        assert!(FallbackReason::from_str("bogus").is_err());
    }

    #[test]
    fn preparation_timer_separates_solver_factorization() {
        let timer = PreparationTimer::start();
        let gram = ndarray::Array2::<f64>::eye(32);
        let _solver = GramSolver::new(&gram);
        let timing = timer.finish();

        assert!(timing.factorization_ns > 0);
        assert!(timing.total_ns >= timing.factorization_ns);
        assert_eq!(timing.basis_ns, timing.total_ns - timing.factorization_ns);
    }
}
