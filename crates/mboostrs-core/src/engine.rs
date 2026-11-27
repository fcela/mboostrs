//! Core component-wise gradient boosting engine.
//!
//! Implements the component-wise gradient boosting loop.
//!
//! Performance: pre-converts packed matrices to raw `Vec<f64>` once before the
//! hot loop to eliminate ndarray indexing overhead. Uses ndarray `as_slice()`
//! for contiguous array access where possible. Block-learner fit uses
//! `fit_step_ref` (zero-alloc, returns slices into internal buffers) — only
//! the winning learner's coefficients are cloned into `coefs`.
//!
//! Three engine paths:
//! 1. ALL-SCALAR: all learners are ScalarBols → packed GEMV (fastest)
//! 2. HYBRID: mix of ScalarBols + block learners → packed GEMV for scalar
//!    learners + fit_step_ref for block learners, pick overall winner
//! 3. ALL-BLOCK: no scalar learners → fit_step_ref per learner

use faer::linalg::matmul::matmul;
use faer::Parallelism;
use ndarray::{Array1, Array2, ArrayView1};
use std::time::Instant;

/// Measured BBS candidate-sweep crossover on the benchmark matrix. Below ten
/// block learners Rayon scheduling costs more than the independent solves.
const PARALLEL_BLOCK_MIN_LEARNERS: usize = 10;

use crate::control::{BoostControl, RiskKind, StepRule};
use crate::families::{golden_grid_min, CalibrationStatus, Family};
use crate::learners::{
    pack_scalar_bols, pack_scalar_bols_partial, solve_bmono_iterative, solve_bmono_quadprog,
    BmonoSolver, HatFactor, PreparedLearner,
};
use crate::path::BoostingPath;
use crate::report::{EngineRoute, FallbackReason, FitReport, InstrumentationLevel};
use crate::MboostrsError;

/// Run the component-wise gradient boosting loop.
///
/// Algorithm (matches R `mboost::mboost_fit`):
/// 1. Compute offset from family
/// 2. Initialise fitted values to offset
/// 3. For each iteration 0..mstop:
///    a. Calibrate family, compute negative gradient (pseudo-response)
///    b. Fit each learner to pseudo-response, select lowest weighted SSE
///    c. Update fitted: f += nu * best_pred
///    d. Record selection, coefficient, risk
///
/// Ties are broken by lowest index (matching R). Risk is stored as sum (not mean).
pub fn fit<F: Family + ?Sized>(
    learners: &mut [PreparedLearner],
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    family: &mut F,
    control: &BoostControl,
    offset_override: Option<f64>,
) -> BoostingPath {
    fit_with(learners, y, weights, family, control, offset_override, true)
}

/// Fallible version of [`fit`] that returns a typed error instead
/// of panicking on invalid input.
pub fn try_fit<F: Family + ?Sized>(
    learners: &mut [PreparedLearner],
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    family: &mut F,
    control: &BoostControl,
    offset_override: Option<f64>,
) -> Result<BoostingPath, MboostrsError> {
    try_fit_with(learners, y, weights, family, control, offset_override, true)
}

/// State retained to continue a boosting path from its current iteration.
/// A resumed fit returns the full path and matches a cold fit of the same
/// length. The retained values needed for continuation are:
///
/// * `fitted` is the linear predictor after iteration `m`, which is the only
///   thing the candidate search and the family read.
/// * `offset` is the retained path's offset, used rather than recomputed, so
///   the resumed path reports the same constant.
/// * `risk`, `selected`, `coefs` and `increments` are the retained path's own,
///   copied into the result so the returned `BoostingPath` contains the full
///   fit history.
/// * `nuisance_at_resume` and `nuisance_vec_at_resume` restore the family for
///   the one family whose calibration is path-dependent. See
///   `Family::calibration_is_path_dependent`.
///
/// The Gaussian Gram fast path is not resumable because its cross-Gram
/// projections are built from the initial residual. A resumed fit uses the
/// general route; the continuation is checked against a cold fit using that
/// same route.
#[derive(Clone, Copy, Debug)]
pub struct WarmStart<'a> {
    /// Linear predictor after the retained iterations.
    pub fitted: ArrayView1<'a, f64>,
    /// The offset the retained path started from.
    pub offset: f64,
    /// Risk after each retained iteration, length `selected.len() + 1`.
    pub risk: &'a [f64],
    /// The learner selected at each retained iteration.
    pub selected: &'a [usize],
    /// The coefficient vector at each retained iteration.
    pub coefs: &'a [Vec<f64>],
    /// Per-iteration increments, when the retained path kept them.
    pub increments: &'a [Vec<f64>],
    /// Scalar nuisance after each retained iteration.
    pub nuisance_path: &'a [Option<f64>],
    /// Vector nuisance after each retained iteration.
    pub nuisance_vec_path: &'a [Vec<f64>],
    /// Scalar nuisance at the resume point, for a path-dependent family.
    pub nuisance_at_resume: Option<f64>,
    /// Vector nuisance at the resume point, for a path-dependent family.
    pub nuisance_vec_at_resume: &'a [f64],
    /// The negative gradient the retained path left. Empty when it was not
    /// retained, in which case it is recomputed and the continued path agrees
    /// with a cold one to machine precision rather than bit for bit.
    pub residual: ArrayView1<'a, f64>,
}

impl<'a> WarmStart<'a> {
    /// The state a completed path leaves, ready to continue from.
    pub fn from_path(path: &'a BoostingPath) -> Self {
        Self {
            fitted: path.fitted.view(),
            offset: path.offset,
            risk: &path.risk,
            selected: &path.selected,
            coefs: &path.coefs,
            increments: &path.increments,
            nuisance_path: &path.nuisance_path,
            nuisance_vec_path: &path.nuisance_vec_path,
            nuisance_at_resume: path.nuisance_path.last().copied().flatten(),
            nuisance_vec_at_resume: path
                .nuisance_vec_path
                .last()
                .map(Vec::as_slice)
                .unwrap_or(&[]),
            residual: path.residual.view(),
        }
    }

    fn validate(&self, n: usize, mstop: u32) -> Result<(), MboostrsError> {
        if self.fitted.len() != n {
            return Err(MboostrsError::Shape(format!(
                "warm start has {} fitted values but the fit has {n} observations",
                self.fitted.len()
            )));
        }
        let completed = self.selected.len();
        if completed as u32 >= mstop {
            return Err(MboostrsError::Control(format!(
                "warm start already ran {completed} iterations, which is not \
                 fewer than the requested mstop={mstop}; slice the retained \
                 path instead of continuing it"
            )));
        }
        if !self.risk.is_empty() && self.risk.len() != completed + 1 {
            return Err(MboostrsError::Shape(format!(
                "warm start has {} risk entries for {completed} iterations; \
                 expected {}",
                self.risk.len(),
                completed + 1
            )));
        }
        if self.fitted.iter().any(|value| !value.is_finite()) {
            return Err(MboostrsError::Solver(
                "warm start fitted values contain NaN or infinity".to_string(),
            ));
        }
        Ok(())
    }
}

/// Continue a boosting path from the state a shorter fit left.
///
/// `control.mstop` is the *total* length wanted, not the number of extra
/// iterations, so `try_fit_continue(.., mstop = 200, warm)` on a path of 100
/// runs 100 more. The result is the whole path.
///
/// The learners must be the ones the retained path was fitted with, prepared
/// against the same data and weights: this cannot check that, and
/// `WarmStart::validate` catches only the shapes. What it does check is the
/// risk -- recomputed at the resumed fitted values and compared against the
/// retained one -- which catches a state belonging to different data.
pub fn try_fit_continue<F: Family + ?Sized>(
    learners: &mut [PreparedLearner],
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    family: &mut F,
    control: &BoostControl,
    warm: &WarmStart<'_>,
) -> Result<BoostingPath, MboostrsError> {
    warm.validate(weights.len(), control.mstop)?;
    try_fit_with_options_warm(
        learners,
        y,
        weights,
        family,
        control,
        Some(warm.offset),
        FitOptions::public(!warm.increments.is_empty()),
        Some(warm),
    )
}

#[derive(Clone, Debug)]
struct FitOptions {
    store_increments: bool,
    store_coefs: bool,
    store_risk: bool,
    store_nuisance: bool,
    collect_report: bool,
    engine_policy: EnginePolicy,
    /// A per-observation starting offset, as R's `offset=` argument to
    /// `gamboost`/`glmboost`: a known contribution to the linear predictor
    /// (`log(exposure)` for a rate model) that boosting must not re-estimate.
    /// It replaces the scalar family-calibrated start outright; there is no
    /// added scalar part, because the vector already holds each row's
    /// constant. `None` keeps the scalar behaviour every existing caller
    /// has.
    offset_vec: Option<ndarray::Array1<f64>>,
    /// Weights of the held-out risk under `RiskKind::Oobag`. `None` scores
    /// every zero-weight row with weight one, as `mboost` does for unit
    /// weights; a caller whose model carries weights passes them for the
    /// held-out rows (and zero elsewhere), so the fold's risk is the
    /// model's weighted loss.
    oob_weights: Option<ndarray::Array1<f64>>,
}

/// Internal route selection used by differential tests. Production builds
/// retain only `Auto`, so the ordinary API cannot accidentally force a slower
/// or ineligible numerical route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EnginePolicy {
    Auto,
    #[cfg(test)]
    ReferenceGeneral,
    #[cfg(test)]
    ForcePackedScalar,
    #[cfg(test)]
    ForceParallelBlock,
    #[cfg(test)]
    ForceSequentialBlock,
    #[cfg(test)]
    ForceGaussianGram,
}

impl EnginePolicy {
    fn allows_gram(self) -> bool {
        match self {
            Self::Auto => true,
            #[cfg(test)]
            Self::ForceGaussianGram => true,
            #[cfg(test)]
            Self::ReferenceGeneral
            | Self::ForcePackedScalar
            | Self::ForceParallelBlock
            | Self::ForceSequentialBlock => false,
        }
    }

    fn allows_packed(self) -> bool {
        match self {
            Self::Auto => true,
            #[cfg(test)]
            Self::ForcePackedScalar => true,
            #[cfg(test)]
            Self::ReferenceGeneral
            | Self::ForceParallelBlock
            | Self::ForceSequentialBlock
            | Self::ForceGaussianGram => false,
        }
    }

    fn parallel_block(self, n_learners: usize) -> bool {
        match self {
            Self::Auto => n_learners >= PARALLEL_BLOCK_MIN_LEARNERS,
            #[cfg(test)]
            Self::ForceParallelBlock => true,
            #[cfg(test)]
            Self::ReferenceGeneral
            | Self::ForcePackedScalar
            | Self::ForceSequentialBlock
            | Self::ForceGaussianGram => false,
        }
    }
}

#[inline]
/// Deterministic per-iteration active-learner mask for component subsampling.
///
/// Returns `None` when subsampling is disabled (every learner active). When
/// enabled, returns a boolean mask of length `n_learners` with about
/// `fraction * n_learners` entries true, drawn from a ChaCha stream seeded by
/// `(subsample_seed, iteration)` so the path is reproducible and independent of
/// thread count. At least one learner is always active. Building the mask
/// serially before any parallel candidate scan keeps the result deterministic
/// regardless of the Rayon pool size.
/// The mask a single draw produces, allocated fresh.
///
/// The fit loop uses [`ComponentSubsampleScratch`] instead, which reuses its
/// buffers; this stays as the reference the differential test compares
/// against, so "reusing the buffers changed which learners are drawn" is a
/// failing test rather than a thing to reason about.
#[cfg(test)]
fn active_learner_mask(
    control: &BoostControl,
    n_learners: usize,
    iteration: usize,
) -> Option<Vec<bool>> {
    let fraction = control.component_fraction?;
    let k = ((fraction * n_learners as f64).round() as usize).clamp(1, n_learners);
    if k == n_learners {
        return None;
    }
    let mut scratch = ComponentSubsampleScratch::with_capacity(n_learners);
    scratch.draw(control, n_learners, iteration, k);
    Some(scratch.mask)
}

/// How many learners compete this iteration, or `None` for all of them.
///
/// Split out of [`active_learner_mask`] so the fit loop can decide whether to
/// draw before borrowing its scratch buffers.
fn active_learner_count(control: &BoostControl, n_learners: usize) -> Option<usize> {
    let fraction = control.component_fraction?;
    let k = ((fraction * n_learners as f64).round() as usize).clamp(1, n_learners);
    if k == n_learners {
        return None;
    }
    Some(k)
}

/// The two buffers a component-subsample draw needs, kept across iterations.
///
/// The draw shuffles `0..n_learners` and marks the first `k` of them, which
/// allocated a `Vec<usize>` and a `Vec<bool>` on every boosting step --
/// `2 * mstop` allocations for a fit that uses component subsampling at all,
/// and nothing about them changes between steps except their contents.
#[derive(Default)]
struct ComponentSubsampleScratch {
    indices: Vec<usize>,
    mask: Vec<bool>,
}

impl ComponentSubsampleScratch {
    fn with_capacity(n_learners: usize) -> Self {
        Self {
            indices: Vec::with_capacity(n_learners),
            mask: Vec::with_capacity(n_learners),
        }
    }

    /// Refill both buffers in place for `iteration`.
    ///
    /// The seed mixes the base seed with the iteration index, so each step
    /// samples independently and reproducibly -- reusing the buffers must not
    /// change which learners are drawn, and a differential test holds that.
    fn draw(&mut self, control: &BoostControl, n_learners: usize, iteration: usize, k: usize) {
        use rand::seq::SliceRandom;
        use rand::SeedableRng;

        self.indices.clear();
        self.indices.extend(0..n_learners);
        self.mask.clear();
        self.mask.resize(n_learners, false);

        let seed = control
            .subsample_seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(iteration as u64);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        self.indices.shuffle(&mut rng);
        for &index in self.indices.iter().take(k) {
            self.mask[index] = true;
        }
    }
}

/// Reject a boosting step whose risk is not finite.
///
/// Mirrors `try_fit_lss`'s check. A `NaN` or infinite risk means the fit cannot
/// proceed: every subsequent gradient is derived from the same state, so the
/// remaining iterations produce a path of meaningless numbers. Returning a
/// typed error here makes that diagnosable rather than leaving the caller to
/// notice that `risk_` stopped changing.
///
/// The message names the iteration (1-based), the family, the step rule and
/// the remedy, so the caller can act on it without reading the source.
fn check_step_risk(
    step_risk: f64,
    iteration: u32,
    family: &str,
    control: &BoostControl,
) -> Result<(), MboostrsError> {
    if step_risk.is_finite() {
        return Ok(());
    }
    let position = iteration + 1;
    let message = match control.step {
        StepRule::Fixed => format!(
            "{family} risk became non-finite after boosting iteration {position} \
             with the fixed step nu = {nu}: an update pushed the linear predictor \
             far enough that the inverse link overflowed. Remedy: fit with \
             boost_control(..., step=\"line_search\"), which scales each step to \
             the loss, or lower nu (for example nu = 0.1). The fixed step grows \
             with the fitted mean for families whose negative gradient is on the \
             count scale (Poisson), so a response whose values run to the hundreds \
             needs the line search.",
            nu = control.nu,
        ),
        StepRule::LineSearch { .. } => format!(
            "{family} risk became non-finite after boosting iteration {position} \
             with the line search (nu = {nu}): the chosen step has a non-finite \
             risk, which the search does not produce from finite inputs. Remedy: \
             check the response, offset and weights for values the {family} family \
             cannot take (infinite or NaN entries, or values outside its support). \
             A smaller nu does not help here.",
            nu = control.nu,
        ),
    };
    Err(MboostrsError::Solver(message))
}

/// The multiple `rho` of the winning learner's direction that minimizes the
/// training risk, for the `LineSearch` step rule.
///
/// `f` is the fitted vector before the step and `h` the winner's unshrunk
/// contribution — its least-squares fit to the negative gradient. The loss
/// searched is the family risk on the **training weights**, the loss the fit
/// minimizes, which is not necessarily what `control.risk` reports. The
/// minimum is bracketed on the grid 0 … 4 (dense near the classical unit
/// step: Poisson with large counts wants `rho` well below 1, low counts well
/// above), the upper end doubles while the risk still falls, and the bracket
/// is then narrowed by golden-section — together a handful of `O(n)` risk
/// evaluations per boosting step. Non-finite risks count as positive
/// infinity, so the search backs away from an overflow instead of chasing
/// it. Callers with a squared-error family do not call this at all: their
/// least-squares step is already optimal and `rho` is exactly 1.
fn line_search_rho<F: Family + ?Sized>(
    family: &F,
    y: ArrayView1<f64>,
    f: &[f64],
    h: &[f64],
    weights: &[f64],
    scratch: &mut [f64],
    max_iter: usize,
) -> f64 {
    let n = f.len();
    golden_grid_min(
        |rho| {
            if let Some(change) = family.risk_change(y, f, h, weights, rho) {
                return if change.is_finite() {
                    change
                } else {
                    f64::INFINITY
                };
            }
            for i in 0..n {
                scratch[i] = f[i] + rho * h[i];
            }
            let risk = family.risk(
                y,
                ArrayView1::from(&scratch[..n]),
                ArrayView1::from(weights),
            );
            if risk.is_finite() {
                risk
            } else {
                f64::INFINITY
            }
        },
        max_iter,
    )
}

/// Identity element for the parallel candidate scan: index 0, no score, no
/// winners counted.
pub(crate) const CANDIDATE_SCAN_IDENTITY: (usize, f64, usize) = (0, f64::INFINITY, 0);

/// Merge two partial results of the parallel candidate scan.
///
/// Each element is `(index, sse, winning_multiplicity)`. This is the reduction
/// operator Rayon applies over an arbitrary binary tree, so it must be
/// associative and must agree with the serial fold that `observe_candidate`
/// performs — otherwise the reported `tie_selection_count` would depend on how
/// the work happened to be partitioned, and the execution report would be
/// reproducible only for the serial paths.
///
/// The NaN case is the one that needs care and previously did not have it.
/// Every IEEE-754 comparison against NaN is false, so `b.1 < a.1` cannot
/// displace a NaN accumulator: a chunk whose first candidate scored NaN
/// absorbed every later chunk, and the whole reduction returned "no winner".
/// Whether that happened depended entirely on where Rayon split the work — on
/// `[NaN, 2.0, 2.0]` a left-leaning tree returned `(1, 2.0, 2)` and a
/// right-leaning one `(0, inf, 0)`. A non-finite score is now normalized to the
/// identity up front, which is exactly how the serial fold treats it: it never
/// lets a NaN or infinite score become `best_sse`.
///
/// The remaining rules are the serial fold's: a strictly better score replaces
/// the accumulator whole (so a losing chunk's multiplicity is correctly
/// discarded), and equal finite scores add their multiplicities and keep the
/// lower index.
///
/// Pinned by `candidate_scan_merge_is_partition_independent`, which enumerates
/// every binary tree shape over small score vectors and compares each against
/// the serial fold.
pub(crate) fn merge_candidate_scans(
    a: (usize, f64, usize),
    b: (usize, f64, usize),
) -> (usize, f64, usize) {
    let a = if a.1.is_finite() {
        a
    } else {
        CANDIDATE_SCAN_IDENTITY
    };
    let b = if b.1.is_finite() {
        b
    } else {
        CANDIDATE_SCAN_IDENTITY
    };
    if b.1 < a.1 {
        b
    } else if b.1 == a.1 && b.1.is_finite() {
        let multiplicity = a.2 + b.2;
        let mut winner = if b.0 < a.0 { b } else { a };
        winner.2 = multiplicity;
        winner
    } else {
        a
    }
}

pub(crate) fn observe_candidate(
    index: usize,
    sse: f64,
    best_index: &mut usize,
    best_sse: &mut f64,
    winning_multiplicity: &mut usize,
) -> bool {
    if sse < *best_sse {
        *best_index = index;
        *best_sse = sse;
        *winning_multiplicity = 1;
        true
    } else if sse == *best_sse && sse.is_finite() {
        *winning_multiplicity += 1;
        if index < *best_index {
            *best_index = index;
            return true;
        }
        false
    } else {
        false
    }
}

#[inline]
fn tally_calibration_status(
    status: CalibrationStatus,
    converged: &mut usize,
    limit: &mut usize,
    failed: &mut usize,
) {
    match status {
        CalibrationStatus::NotTracked => {}
        CalibrationStatus::Converged => *converged += 1,
        CalibrationStatus::LimitReached => *limit += 1,
        CalibrationStatus::Failed => *failed += 1,
    }
}

impl FitOptions {
    fn public(store_increments: bool) -> Self {
        Self {
            store_increments,
            store_coefs: true,
            store_risk: true,
            store_nuisance: true,
            collect_report: true,
            engine_policy: EnginePolicy::Auto,
            offset_vec: None,
            oob_weights: None,
        }
    }

    fn selected_only() -> Self {
        Self {
            store_increments: false,
            store_coefs: false,
            store_risk: false,
            store_nuisance: false,
            collect_report: false,
            engine_policy: EnginePolicy::Auto,
            offset_vec: None,
            oob_weights: None,
        }
    }
}

/// Benchmark-only full fit with deterministic report collection disabled.
/// This is not part of the ordinary API; it exists to measure the cost of the
/// `InstrumentationLevel::Off` metadata path against the same engine kernels.
#[cfg(feature = "instrumentation-baseline")]
#[doc(hidden)]
pub fn try_fit_without_report<F: Family + ?Sized>(
    learners: &mut [PreparedLearner],
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    family: &mut F,
    control: &BoostControl,
    offset_override: Option<f64>,
) -> Result<BoostingPath, MboostrsError> {
    let mut options = FitOptions::public(false);
    options.collect_report = false;
    try_fit_with_options(
        learners,
        y,
        weights,
        family,
        control,
        offset_override,
        options,
    )
}

#[cfg(test)]
fn try_fit_with_engine_policy<F: Family + ?Sized>(
    learners: &mut [PreparedLearner],
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    family: &mut F,
    control: &BoostControl,
    offset_override: Option<f64>,
    store_increments: bool,
    engine_policy: EnginePolicy,
) -> Result<BoostingPath, MboostrsError> {
    let mut options = FitOptions::public(store_increments);
    options.engine_policy = engine_policy;
    try_fit_with_options(
        learners,
        y,
        weights,
        family,
        control,
        offset_override,
        options,
    )
}

#[cfg(test)]
fn try_fit_selected_with_engine_policy<F: Family + ?Sized>(
    learners: &mut [PreparedLearner],
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    family: &mut F,
    control: &BoostControl,
    engine_policy: EnginePolicy,
) -> Result<BoostingPath, MboostrsError> {
    let mut options = FitOptions::selected_only();
    options.collect_report = true;
    options.engine_policy = engine_policy;
    try_fit_with_options(learners, y, weights, family, control, None, options)
}

/// Like `fit` but lets the caller skip per-step increment storage when not
/// needed. With `store_increments=false`, the hot loop avoids a per-step
/// 8*n-byte `Vec` allocation and the `BoostingPath::increments` field is
/// left empty. Used by the Python `fit_glmboost`/`fit_gamboost_specs` entry
/// points, which reconstruct sliced predictions on demand via `predict_*`.
/// CV still passes `true` because it walks the path forward to compute
/// out-of-bag risk at every iteration.
pub fn fit_with<F: Family + ?Sized>(
    learners: &mut [PreparedLearner],
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    family: &mut F,
    control: &BoostControl,
    offset_override: Option<f64>,
    store_increments: bool,
) -> BoostingPath {
    try_fit_with(
        learners,
        y,
        weights,
        family,
        control,
        offset_override,
        store_increments,
    )
    .expect("fit failed")
}

/// Fit with a per-observation starting offset, as R's `offset=` accepts for
/// `gamboost`/`glmboost`: one known contribution to the linear predictor per
/// fitted coordinate (`log(exposure)` for a rate model), which boosting
/// never re-estimates. The vector replaces the scalar family-calibrated
/// start; `offset_override` is unused when the vector is supplied, and the
/// returned path's `offset` field is the scalar the fit would otherwise have
/// started from — a caller holding the vector keeps supplying it, to
/// `predict` and to any refit, because the path does not retain it.
pub fn try_fit_with_vector_offset<F: Family + ?Sized>(
    learners: &mut [PreparedLearner],
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    family: &mut F,
    control: &BoostControl,
    offset_vec: ArrayView1<f64>,
    store_increments: bool,
) -> Result<BoostingPath, MboostrsError> {
    let mut options = FitOptions::public(store_increments);
    options.offset_vec = Some(offset_vec.to_owned());
    try_fit_with_options(learners, y, weights, family, control, None, options)
}

/// [`try_fit_with`] whose `RiskKind::Oobag` risk weights the held-out rows
/// by `oob_weights` rather than by one. A cross-validation fold of a weighted
/// model passes the model's weights on its held-out rows (zero on the rows it
/// trains on), so the recorded risk path is the model's weighted held-out
/// loss. `offset_vec`, when given, is a per-row offset as in
/// [`try_fit_with_vector_offset`].
#[allow(clippy::too_many_arguments)]
pub fn try_fit_with_oob_weights<F: Family + ?Sized>(
    learners: &mut [PreparedLearner],
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    family: &mut F,
    control: &BoostControl,
    offset_override: Option<f64>,
    offset_vec: Option<ArrayView1<f64>>,
    oob_weights: ArrayView1<f64>,
) -> Result<BoostingPath, MboostrsError> {
    if oob_weights.len() != weights.len() {
        return Err(MboostrsError::Shape(format!(
            "oob_weights has {} rows but weights has {}",
            oob_weights.len(),
            weights.len()
        )));
    }
    if oob_weights.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err(MboostrsError::Control(
            "oob_weights must be finite and non-negative".to_string(),
        ));
    }
    let mut options = FitOptions::public(false);
    options.offset_vec = offset_vec.map(|v| v.to_owned());
    options.oob_weights = Some(oob_weights.to_owned());
    let offset_override = if options.offset_vec.is_some() {
        None
    } else {
        offset_override
    };
    try_fit_with_options(
        learners,
        y,
        weights,
        family,
        control,
        offset_override,
        options,
    )
}

/// Fallible version of [`fit_with`] that returns a typed
/// error instead of panicking on invalid input.
pub fn try_fit_with<F: Family + ?Sized>(
    learners: &mut [PreparedLearner],
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    family: &mut F,
    control: &BoostControl,
    offset_override: Option<f64>,
    store_increments: bool,
) -> Result<BoostingPath, MboostrsError> {
    try_fit_with_options(
        learners,
        y,
        weights,
        family,
        control,
        offset_override,
        FitOptions::public(store_increments),
    )
}

/// Fit with summary instrumentation enabled on a cloned control value.
///
/// The caller chooses whether to retain per-observation increments, making the
/// numerical route identical to [`try_fit_with`] with the same storage choice.
/// The supplied control is not mutated, and textual tracing is unchanged.
pub fn try_fit_profiled<F: Family + ?Sized>(
    learners: &mut [PreparedLearner],
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    family: &mut F,
    control: &BoostControl,
    offset_override: Option<f64>,
    store_increments: bool,
) -> Result<BoostingPath, MboostrsError> {
    let profiled_control = control
        .clone()
        .with_instrumentation(InstrumentationLevel::Summary);
    try_fit_with(
        learners,
        y,
        weights,
        family,
        &profiled_control,
        offset_override,
        store_increments,
    )
}

/// Panicking compatibility wrapper around [`try_fit_profiled`].
pub fn fit_profiled<F: Family + ?Sized>(
    learners: &mut [PreparedLearner],
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    family: &mut F,
    control: &BoostControl,
    offset_override: Option<f64>,
    store_increments: bool,
) -> BoostingPath {
    try_fit_profiled(
        learners,
        y,
        weights,
        family,
        control,
        offset_override,
        store_increments,
    )
    .unwrap_or_else(|error| panic!("{error}"))
}

/// Fit only far enough to return the selected learner indices.
///
/// Used by stability-selection workloads where coefficients, risk paths,
/// fitted values, and increments are discarded. Public `fit()` semantics remain
/// unchanged, including `risk="none"` storing inbag risk for R parity.
pub fn fit_selected<F: Family + ?Sized>(
    learners: &mut [PreparedLearner],
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    family: &mut F,
    control: &BoostControl,
    offset_override: Option<f64>,
) -> Vec<usize> {
    fit_with_options(
        learners,
        y,
        weights,
        family,
        control,
        offset_override,
        FitOptions::selected_only(),
    )
    .selected
}

/// Fallible version of [`fit_selected`].
pub fn try_fit_selected<F: Family + ?Sized>(
    learners: &mut [PreparedLearner],
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    family: &mut F,
    control: &BoostControl,
    offset_override: Option<f64>,
) -> Result<Vec<usize>, MboostrsError> {
    let path = try_fit_with_options(
        learners,
        y,
        weights,
        family,
        control,
        offset_override,
        FitOptions::selected_only(),
    )?;
    Ok(path.selected)
}

/// [`try_fit_selected`] from a per-observation offset (`log(exposure)`):
/// the selection sequence of a fit whose linear predictor starts at
/// `offset_vec` rather than at a fitted constant.
pub fn try_fit_selected_with_vector_offset<F: Family + ?Sized>(
    learners: &mut [PreparedLearner],
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    family: &mut F,
    control: &BoostControl,
    offset_vec: ArrayView1<f64>,
) -> Result<Vec<usize>, MboostrsError> {
    let mut options = FitOptions::selected_only();
    options.offset_vec = Some(offset_vec.to_owned());
    let path = try_fit_with_options(learners, y, weights, family, control, None, options)?;
    Ok(path.selected)
}

fn fit_with_options<F: Family + ?Sized>(
    learners: &mut [PreparedLearner],
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    family: &mut F,
    control: &BoostControl,
    offset_override: Option<f64>,
    options: FitOptions,
) -> BoostingPath {
    try_fit_with_options(
        learners,
        y,
        weights,
        family,
        control,
        offset_override,
        options,
    )
    .unwrap_or_else(|e| panic!("{e}"))
}

fn try_fit_with_options<F: Family + ?Sized>(
    learners: &mut [PreparedLearner],
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    family: &mut F,
    control: &BoostControl,
    offset_override: Option<f64>,
    options: FitOptions,
) -> Result<BoostingPath, MboostrsError> {
    try_fit_with_options_warm(
        learners,
        y,
        weights,
        family,
        control,
        offset_override,
        options,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn try_fit_with_options_warm<F: Family + ?Sized>(
    learners: &mut [PreparedLearner],
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    family: &mut F,
    control: &BoostControl,
    offset_override: Option<f64>,
    options: FitOptions,
    warm: Option<&WarmStart<'_>>,
) -> Result<BoostingPath, MboostrsError> {
    let timing_enabled =
        options.collect_report && control.instrumentation != InstrumentationLevel::Off;
    let fit_started = timing_enabled.then(Instant::now);
    // Re-encode the response if the family requires it (Binomial {0,1}->{-1,+1});
    // a no-op for every other family. Done once here so all fit entry points and
    // every binding share one definition.
    //
    // An empty learner list is an offset-only model (`y ~ 1`): the setup below
    // computes the family offset and its risk, and the loop runs zero
    // iterations, so the path has no selections and its fitted values are the
    // offset.
    let boosting_steps = if learners.is_empty() {
        0
    } else {
        control.mstop
    };

    if y.iter().any(|value| !value.is_finite()) {
        return Err(MboostrsError::BadResponse(
            "fit".to_string(),
            "response contains NaN or infinite values".to_string(),
        ));
    }
    if weights.iter().any(|value| !value.is_finite()) {
        return Err(MboostrsError::Shape(
            "fit: weights contain NaN or infinite values".to_string(),
        ));
    }
    if weights.iter().any(|&value| value < 0.0) {
        return Err(MboostrsError::Shape(
            "fit: weights must be non-negative".to_string(),
        ));
    }
    if family.requires_unit_weights() && weights.iter().any(|&value| value != 1.0) {
        return Err(MboostrsError::BadResponse(
            family.name().to_string(),
            format!(
                "{} currently supports only unit weights (w \u{2261} 1); \
                 bootstrap/subsampling cvrisk is not yet supported for this family",
                family.name()
            ),
        ));
    }
    if family.requires_binary_weights() && weights.iter().any(|&value| value != 0.0 && value != 1.0)
    {
        return Err(MboostrsError::BadResponse(
            family.name().to_string(),
            format!("weights must be either 0 or 1 for family {}", family.name()),
        ));
    }
    if offset_override.is_some_and(|offset| !offset.is_finite()) {
        return Err(MboostrsError::Control(
            "offset override must be finite".to_string(),
        ));
    }
    if let Some(vector) = options.offset_vec.as_ref() {
        // A per-observation offset replaces the scalar start, so it must
        // describe exactly one known contribution per fitted coordinate.
        // `-inf` arrives as `log(0)` of a zero exposure; a row with no
        // exposure carries no information about the rate and is refused
        // here rather than poisoning the first risk evaluation.
        if vector.len() != family.n_from_y(y) {
            return Err(MboostrsError::Shape(format!(
                "offset has {} entries but the response implies {} fitted \
                 coordinates",
                vector.len(),
                family.n_from_y(y)
            )));
        }
        if vector.iter().any(|value| !value.is_finite()) {
            return Err(MboostrsError::Control(
                "offset must be finite in every entry; log(0) of a \
                 zero-exposure row is -inf, and such a row cannot enter the \
                 linear predictor"
                    .to_string(),
            ));
        }
    }

    // Validate the response values. PyO3 entry points already call check_y
    // before reaching here, but pure-Rust callers (fit, cv, benchmarks) skip
    // that step. Propagating the error beats silent NaN→0 in e.g. PropOdds.
    family.check_y(y)?;

    let y_encoded = family.encode_response(y);
    let y = match y_encoded.as_ref() {
        Some(a) => a.view(),
        None => y.view(),
    };
    let n = family.n_from_y(y);

    if n == 0 {
        return Err(MboostrsError::Shape(
            "fit requires at least one observation".to_string(),
        ));
    }

    // Validate y/weights length using the family's observation count (survival
    // families pack time+status into a 2*n-length y vector).
    if n != weights.len() {
        return Err(MboostrsError::Shape(format!(
            "fit: n observations ({n}) != weights len ({})",
            weights.len()
        )));
    }
    for learner in learners.iter() {
        let learner_n = learner.n_observations();
        if learner_n != n {
            return Err(MboostrsError::LearnerSetup(
                learner.name().to_string(),
                format!("design has {learner_n} rows but fit has {n} response entries"),
            ));
        }
    }

    let total_basis_columns = if options.collect_report {
        learners.iter().map(PreparedLearner::n_parameters).sum()
    } else {
        0
    };
    let learner_stats = if options.collect_report {
        crate::learners::aggregate_report_stats(learners)
    } else {
        crate::learners::LearnerReportStats::default()
    };
    let mut fallback_reasons = Vec::new();
    if options.collect_report && options.store_increments {
        fallback_reasons.push(FallbackReason::IncrementStorageRequested);
    }
    if options.collect_report && !family.supports_gram_update() {
        fallback_reasons.push(FallbackReason::FamilyDoesNotSupportGram);
    }
    if options.collect_report && !matches!(control.risk, RiskKind::Inbag) {
        fallback_reasons.push(FallbackReason::RiskIsNotInbag);
    }
    let mut has_block = false;
    let mut gram_learner_supported = true;
    for learner in learners.iter() {
        match learner {
            PreparedLearner::ScalarBols { .. } => {}
            PreparedLearner::BlockBols { .. } | PreparedLearner::BmonoBbs { .. } => {
                has_block = true;
            }
            // FactoredBlockBols is a block ridge learner but stores per-group
            // Gram state, not a single materialised linear system the Gram path
            // can consume directly.
            PreparedLearner::FactoredBlockBols { .. } => {
                has_block = true;
                gram_learner_supported = false;
            }
            // Trees run beside the projections on an explicit residual.
            PreparedLearner::Btree { .. } => {}
            _ => gram_learner_supported = false,
        }
    }
    if options.collect_report && !has_block {
        fallback_reasons.push(FallbackReason::PureScalarModel);
    }
    if options.collect_report && !gram_learner_supported {
        fallback_reasons.push(FallbackReason::UnsupportedLearnerForGram);
    }
    let gram_eligible = !options.store_increments
        && family.supports_gram_update()
        && matches!(control.risk, RiskKind::Inbag)
        && has_block
        && gram_learner_supported
        // Component subsampling changes which candidates compete each
        // iteration; the Gaussian Gram fast path scans a fixed candidate set,
        // so subsampling falls back to the general path.
        && control.component_fraction.is_none()
        // The Gram path starts from a scalar offset (its intercept algebra
        // has one); a per-observation offset vector runs on the general path.
        && options.offset_vec.is_none();
    let mut report = if options.collect_report {
        FitReport {
            instrumentation: control.instrumentation,
            n_observations: n,
            n_learners: learners.len(),
            total_basis_columns,
            requested_mstop: control.mstop as usize,
            gram_eligible,
            fallback_reasons,
            scalar_learner_count: learner_stats.scalar_learners,
            block_learner_count: learner_stats.block_learners,
            tree_learner_count: learner_stats.tree_learners,
            cholesky_solver_count: learner_stats.cholesky_solvers,
            lu_solver_count: learner_stats.lu_solvers,
            regularized_solver_count: learner_stats.regularized_solvers,
            max_solver_jitter: learner_stats.max_solver_jitter,
            factor_cache_bytes: learner_stats.factor_cache_bytes,
            retained_learner_bytes: learner_stats.retained_data_bytes,
            ..FitReport::default()
        }
    } else {
        FitReport::default()
    };

    #[cfg(test)]
    if options.engine_policy == EnginePolicy::ForceGaussianGram && !gram_eligible {
        return Err(MboostrsError::Control(
            "forced Gaussian Gram route has unmet preconditions".to_string(),
        ));
    }
    // Gaussian covariance/Gram fast path. It requires no increment storage,
    // an inbag squared-error family, all-linear learners, and at least one
    // block learner. Full fits and selection-only refits share this route;
    // the latter disable unused result-path construction below.
    if gram_eligible && options.engine_policy.allows_gram() && warm.is_none() {
        if let Some(mut path) = try_fit_gaussian_gram(
            learners,
            y,
            weights,
            control,
            offset_override,
            options.collect_report,
            options.store_coefs,
            options.store_risk,
        ) {
            if options.collect_report {
                report.route = EngineRoute::GaussianGram;
                report.completed_mstop = path.selected.len();
                report.effective_threads = 1;
                report.gram_cache_bytes = path.report.gram_cache_bytes;
                report.tie_selection_count = path.report.tie_selection_count;
                report.setup_ns = path.report.setup_ns;
                report.cross_gram_ns = path.report.cross_gram_ns;
                report.iteration_ns = path.report.iteration_ns;
                report.reconstruction_ns = path.report.reconstruction_ns;
                report.total_ns = fit_started.map(|started| started.elapsed().as_nanos() as u64);
                path.report = report;
            }
            return Ok(path);
        }
        #[cfg(test)]
        if options.engine_policy == EnginePolicy::ForceGaussianGram {
            return Err(MboostrsError::Control(
                "forced Gaussian Gram route rejected the prepared learners".to_string(),
            ));
        }
        if options.collect_report {
            report
                .fallback_reasons
                .push(FallbackReason::UnsupportedLearnerForGram);
            report.gram_eligible = false;
        }
    }

    let offset = match warm {
        Some(state) => state.offset,
        None => offset_override.unwrap_or_else(|| family.offset(y, weights)),
    };
    if !offset.is_finite() {
        return Err(MboostrsError::Solver(
            "family offset calibration produced a non-finite value".to_string(),
        ));
    }
    let mut f = match warm {
        Some(state) => state.fitted.to_owned(),
        None => match options.offset_vec.as_ref() {
            // A per-observation offset *is* the starting linear predictor;
            // there is no scalar part to add to it.
            Some(vector) => vector.clone(),
            None => Array1::from_elem(n, offset),
        },
    };
    let mut risk = if options.store_risk {
        Vec::with_capacity(control.mstop as usize + 1)
    } else {
        Vec::new()
    };
    let mut selected = Vec::with_capacity(control.mstop as usize);
    let mut coefs: Vec<Vec<f64>> = if options.store_coefs {
        Vec::with_capacity(control.mstop as usize)
    } else {
        Vec::new()
    };
    let mut increments: Vec<Vec<f64>> = if options.store_increments {
        Vec::with_capacity(control.mstop as usize)
    } else {
        Vec::new()
    };
    // Prime the retained path so the continued result is the whole path, not
    // only its tail: `BoostingPath` describes the fit, and a caller slicing it
    // back to the retained length must get what it had.
    if let Some(state) = warm {
        if options.store_risk {
            risk.extend_from_slice(state.risk);
        }
        selected.extend_from_slice(state.selected);
        if options.store_coefs {
            coefs.extend(state.coefs.iter().cloned());
        }
        if options.store_increments {
            increments.extend(state.increments.iter().cloned());
        }
    }
    let mut u = Array1::<f64>::zeros(n);

    // Try all-scalar packed path first (fastest when applicable)
    let packed_all = options
        .engine_policy
        .allows_packed()
        .then(|| pack_scalar_bols(learners))
        .flatten();
    // Try partial packed path (scalar learners packed, block learners via fit_step_ref)
    let packed_partial = if packed_all.is_none() {
        options
            .engine_policy
            .allows_packed()
            .then(|| pack_scalar_bols_partial(learners))
            .flatten()
    } else {
        None
    };

    #[cfg(test)]
    if options.engine_policy == EnginePolicy::ForcePackedScalar && packed_all.is_none() {
        return Err(MboostrsError::Control(
            "forced packed-scalar route requires only scalar bols learners".to_string(),
        ));
    }

    // Pre-convert packed matrices to raw Vecs once (avoids ndarray indexing overhead in hot loop)
    // Layout: row-major (J, n), so row j starts at j*n
    let packed_raw = packed_all.map(|(x_t, wxt_t, denoms)| {
        let j = x_t.nrows();
        debug_assert!(
            j.checked_mul(n).is_some(),
            "packed matrix row count × n overflows usize"
        );
        (
            x_t.into_raw_vec_and_offset().0,
            wxt_t.into_raw_vec_and_offset().0,
            denoms.into_raw_vec_and_offset().0,
            j,
        )
    });

    let partial_raw = packed_partial.map(|p| {
        let j = p.x_t.nrows();
        debug_assert!(
            j.checked_mul(n).is_some(),
            "packed matrix row count × n overflows usize"
        );
        (
            p.x_t.into_raw_vec_and_offset().0,
            p.wxt_t.into_raw_vec_and_offset().0,
            p.denoms.into_raw_vec_and_offset().0,
            j,
            p.scalar_indices.clone(),
            p.block_indices.clone(),
        )
    });

    let parallel_block = options.engine_policy.parallel_block(learners.len());
    let route = if packed_raw.is_some() {
        EngineRoute::PackedScalar
    } else if partial_raw.is_some() {
        EngineRoute::Hybrid
    } else if parallel_block {
        EngineRoute::ParallelBlock
    } else {
        EngineRoute::General
    };
    if options.collect_report {
        report.route = route;
        report.effective_threads = if matches!(route, EngineRoute::ParallelBlock)
            || (matches!(route, EngineRoute::Hybrid)
                && learners
                    .iter()
                    .filter(|learner| !matches!(learner, PreparedLearner::ScalarBols { .. }))
                    .count()
                    >= PARALLEL_BLOCK_MIN_LEARNERS)
        {
            rayon::current_num_threads()
        } else {
            1
        };
    }

    let mut numer_vec: Vec<f64> = if let Some((_, _, _, j, ..)) = partial_raw.as_ref() {
        vec![0.0f64; *j]
    } else if let Some((_, _, _, j)) = packed_raw.as_ref() {
        vec![0.0f64; *j]
    } else {
        Vec::new()
    };

    let w_raw: Vec<f64> = weights.to_vec();

    let risk_w: Array1<f64> = match (control.risk, options.oob_weights.as_ref()) {
        (RiskKind::Oobag, Some(held_out)) => held_out.clone(),
        (RiskKind::Oobag, None) => Array1::from_vec(
            weights
                .iter()
                .map(|&wi| if wi == 0.0 { 1.0 } else { 0.0 })
                .collect(),
        ),
        (RiskKind::Inbag | RiskKind::None, _) => Array1::from_vec(w_raw.clone()),
    };

    let mut best_pred_buf: Vec<f64> = vec![0.0f64; n];
    let mut best_beta_buf: Vec<f64> = Vec::new();
    // Scalar (all-scalar packed) path accumulates 1-element coefs here to keep
    // the hot loop free of per-iteration heap allocations; boxed into `coefs`
    // after the loop. Empty for the hybrid / all-block paths.
    let mut scalar_betas: Vec<f64> = if packed_raw.is_some() && options.store_coefs {
        Vec::with_capacity(control.mstop as usize)
    } else {
        Vec::new()
    };
    let mut trace_buf = String::new();
    // Probe the family once with a no-op increment: `None` means it has no
    // closed-form risk update and the engine recomputes each step's risk.
    let use_incr_risk = family.risk_from_increment(0.0, 0.0, 0.0, 0.0).is_some()
        && matches!(control.risk, RiskKind::Inbag | RiskKind::None);
    // Line search runs only where the incremental Gaussian route does not:
    // for a squared-error family the least-squares step is already the
    // loss-optimal one (`rho = 1` exactly), so `Fixed` and `LineSearch` agree
    // bit for bit and the Gram fast path is untouched.
    let (line_search, ls_max_iter) = match &control.step {
        StepRule::Fixed => (false, 0usize),
        StepRule::LineSearch { max_iter } => (!use_incr_risk, *max_iter),
    };
    let mut ls_h: Vec<f64> = if line_search {
        vec![0.0f64; n]
    } else {
        Vec::new()
    };
    let mut ls_scratch: Vec<f64> = if line_search {
        vec![0.0f64; n]
    } else {
        Vec::new()
    };
    let mut nuisance_path: Vec<Option<f64>> = if options.store_nuisance {
        Vec::with_capacity(control.mstop as usize + 1)
    } else {
        Vec::new()
    };
    let mut nuisance_vec_path: Vec<Vec<f64>> = if options.store_nuisance {
        Vec::with_capacity(control.mstop as usize + 1)
    } else {
        Vec::new()
    };
    if let Some(state) = warm {
        if options.store_nuisance {
            nuisance_path.extend_from_slice(state.nuisance_path);
            nuisance_vec_path.extend(state.nuisance_vec_path.iter().cloned());
        }
    }

    let mut calibration_converged_count = 0usize;
    let mut calibration_limit_count = 0usize;
    let mut calibration_failure_count = 0usize;
    // Put the family into the state the retained path left it in.
    //
    // Most families calibrate by a *pure* function of `(y, f, w)` -- Huber's
    // MAD, the Brent search Weibull/Loglog/Lognormal/GammaReg/NBinomial run
    // over a fixed interval -- so re-running the calibration at the resumed
    // fitted values reproduces the state exactly, and that is what the cold
    // path's end-of-iteration calibration computed. `PropOdds` is the
    // exception: its thresholds come from coordinate descent *seeded from the
    // current thresholds*, so the state is path-dependent and re-calibrating
    // would run five more sweeps the cold path never ran. Those are restored
    // from the retained nuisance instead.
    let resumed_path_dependent = warm.is_some() && family.calibration_is_path_dependent();
    if let Some(state) = warm.filter(|_| resumed_path_dependent) {
        if let Some(scalar) = state.nuisance_at_resume {
            family.set_nuisance(scalar);
        }
        if !state.nuisance_vec_at_resume.is_empty() {
            family.set_nuisance_vec(state.nuisance_vec_at_resume.to_vec());
        }
    } else {
        let initial_calibration = family.calibrate_with_status(y, f.view(), weights);
        if options.collect_report {
            tally_calibration_status(
                initial_calibration,
                &mut calibration_converged_count,
                &mut calibration_limit_count,
                &mut calibration_failure_count,
            );
        }
    }
    if options.store_risk {
        let initial_risk = { family.ngradient_and_risk(y, f.view(), risk_w.view(), u.view_mut()) };
        // `try_fit_lss` validates the initial risk and every step's risk, and
        // returns a typed error when either goes non-finite. The
        // single-parameter engine had no equivalent, so a fit that could not
        // progress returned "successfully" with a meaningless risk path
        // instead of saying so.
        if !initial_risk.is_finite() {
            return Err(MboostrsError::Solver(
                "risk of the offset-only model is not finite; check the \
                 response and weights for values the family cannot score"
                    .to_string(),
            ));
        }
        match warm {
            // The resumed risk is already in `risk`; recomputing it here is
            // the check that the warm state describes *this* data. A caller
            // that hands over a path fitted on a different response, or on
            // different weights, is told so instead of quietly continuing a
            // fit that means nothing.
            Some(state) => {
                let retained = state.risk.last().copied().unwrap_or(initial_risk);
                // The retained risk is maintained by the incremental update
                // `risk - 2 nu w'u + nu^2 w'p^2`, whose rounding is set by the
                // size of the terms subtracted, which is the risk the path
                // started from. The resume value can be far smaller than that
                // (an almost exact fit, or a near-collinear design, leaves a
                // risk of 1e-11 against a start of 60), so comparing against
                // the resume value alone refused a state that describes the
                // data. A different response or weights still differs by
                // O(start risk) and is refused.
                let start = state.risk.first().copied().unwrap_or(initial_risk);
                let scale = retained.abs().max(start.abs());
                if (initial_risk - retained).abs() > 1e-7 * scale {
                    return Err(MboostrsError::Solver(format!(
                        "the warm-start state does not describe this fit: its \
                         risk at the resume point is {retained}, but the same \
                         response, weights and fitted values give \
                         {initial_risk}"
                    )));
                }
            }
            None => risk.push(initial_risk),
        }
    } else {
        family.negative_gradient(y, f.view(), risk_w.view(), u.view_mut());
    }
    // Prefer the residual the retained path left over the one just recomputed.
    // For a squared-error family the loop maintains it incrementally, and the
    // rounding that accumulates in that subtraction is not what a fresh
    // evaluation gives; using the retained vector is what makes a continued
    // path bit-for-bit equal to a cold one rather than one ULP away at the
    // resume step.
    if let Some(state) = warm.filter(|state| state.residual.len() == n) {
        u.assign(&state.residual);
    }
    if warm.is_none() && options.store_nuisance {
        nuisance_path.push(family.nuisance());
        nuisance_vec_path.push(family.nuisance_vec().unwrap_or_default());
    }

    report.setup_ns = fit_started.map(|started| started.elapsed().as_nanos() as u64);
    let iteration_started = timing_enabled.then(Instant::now);
    let mut nonfinite_candidate_fallbacks = 0usize;
    let mut tie_selection_count = 0usize;
    let mut component_subsample_iterations = 0usize;
    // Drawn into once per iteration rather than allocated: the shuffle buffer
    // and the mask are the same size every step, and only their contents
    // change. This was `2 * mstop` allocations for any fit using component
    // subsampling.
    let mut subsample_scratch = ComponentSubsampleScratch::with_capacity(learners.len());

    let resume_at = warm.map_or(0, |state| state.selected.len() as u32);
    for iteration in resume_at..boosting_steps {
        let mut incr_wu_dot_pred = 0.0f64;
        let mut incr_w_pred_sq = 0.0f64;

        // Component subsampling: `None` means every candidate competes.
        let active_mask = match active_learner_count(control, learners.len()) {
            Some(k) => {
                subsample_scratch.draw(control, learners.len(), iteration as usize, k);
                Some(&subsample_scratch.mask)
            }
            None => None,
        };
        if options.collect_report && active_mask.is_some() {
            component_subsample_iterations += 1;
        }

        if let Some((ref x_raw, ref wxt_raw, ref denoms_raw, j)) = packed_raw {
            // ── ALL-SCALAR PATH: fast packed GEMV ──
            let u_slice = u.as_slice().unwrap();
            let f_slice = f.as_slice_mut().unwrap();

            // Precompute Σ w[i]*u[i]² — same for all columns (avoids 2nd pass per column)
            let wu2: f64 = w_raw
                .iter()
                .zip(u_slice.iter())
                .map(|(&w, &u)| w * u * u)
                .sum();

            let wxt_full = wxt_raw.as_slice();

            // GEMV: numer_vec = WXT · u (J-length result)
            if j >= 4 {
                let wxt_mat = faer::mat::from_row_major_slice::<f64, usize, usize>(wxt_full, j, n);
                let u_col = faer::mat::from_column_major_slice::<f64, usize, usize>(u_slice, n, 1);
                for x in numer_vec.iter_mut() {
                    *x = 0.0;
                }
                let mut numer_mat = faer::mat::from_column_major_slice_mut::<f64, usize, usize>(
                    numer_vec.as_mut_slice(),
                    j,
                    1,
                );
                matmul(
                    &mut numer_mat,
                    wxt_mat,
                    u_col,
                    Some(1.0),
                    1.0,
                    Parallelism::None,
                );
            } else {
                for col in 0..j {
                    let wxt_base = col * n;
                    let wxt_col = &wxt_full[wxt_base..wxt_base + n];
                    numer_vec[col] = wxt_col
                        .iter()
                        .zip(u_slice.iter())
                        .map(|(&w, &u)| w * u)
                        .sum();
                }
            }

            let mut best = 0usize;
            let mut best_sse = f64::INFINITY;
            let mut best_beta = 0.0f64;
            let mut best_multiplicity = 0usize;
            for col in 0..j {
                if let Some(mask) = active_mask {
                    if !mask[col] {
                        continue;
                    }
                }
                let numer = numer_vec[col];
                let denom = denoms_raw[col];
                let (sse, beta) = if denom > 0.0 {
                    (wu2 - numer * numer / denom, numer / denom)
                } else {
                    (wu2, 0.0)
                };
                if observe_candidate(col, sse, &mut best, &mut best_sse, &mut best_multiplicity) {
                    best_beta = beta;
                }
            }
            if !best_sse.is_finite() {
                return Err(MboostrsError::Solver(format!(
                    "no finite candidate SSE at boosting iteration {}; response or gradient values may have overflowed",
                    iteration + 1
                )));
            }
            if options.collect_report {
                tie_selection_count += usize::from(best_multiplicity > 1);
            }

            let best_base = best * n;
            let x_col = &x_raw[best_base..best_base + n];
            let nu = control.nu;
            let rho = if line_search {
                for i in 0..n {
                    ls_h[i] = x_col[i] * best_beta;
                }
                line_search_rho(
                    &*family,
                    y,
                    f_slice,
                    &ls_h,
                    &w_raw,
                    &mut ls_scratch,
                    ls_max_iter,
                )
            } else {
                1.0
            };
            let scale = nu * rho * best_beta;
            if options.store_increments {
                let mut inc_buf: Vec<f64> = vec![0.0f64; n];
                if use_incr_risk {
                    let u_mut = u.as_slice_mut().unwrap();
                    for i in 0..n {
                        let inc_i = x_col[i] * scale;
                        f_slice[i] += inc_i;
                        inc_buf[i] = inc_i;
                        u_mut[i] -= inc_i;
                    }
                } else {
                    for i in 0..n {
                        let inc_i = x_col[i] * scale;
                        f_slice[i] += inc_i;
                        inc_buf[i] = inc_i;
                    }
                }
                increments.push(inc_buf);
            } else if use_incr_risk {
                let u_mut = u.as_slice_mut().unwrap();
                for i in 0..n {
                    let inc_i = x_col[i] * scale;
                    f_slice[i] += inc_i;
                    u_mut[i] -= inc_i;
                }
            } else {
                for i in 0..n {
                    f_slice[i] += x_col[i] * scale;
                }
            }
            selected.push(best);
            if options.store_coefs {
                // The stored coefficient is the effective one: replay paths
                // multiply by `nu` only, so the line-search factor rides
                // along inside the coefficient, exactly as if the learner
                // had fitted a shrunk target.
                scalar_betas.push(rho * best_beta);
            }
            if use_incr_risk && options.store_risk {
                incr_wu_dot_pred = best_beta * numer_vec[best];
                // `denoms_raw` is X'WX + lambda. Risk updates need
                // sum(w * pred^2) = beta^2 X'WX, not the penalized solve
                // denominator. Including lambda made the reported Gaussian
                // risk disagree with the fitted values for penalized scalar
                // learners even though selection and coefficients were right.
                let unpenalized = denoms_raw[best] - learners[best].lambda().unwrap_or(0.0);
                incr_w_pred_sq = best_beta * best_beta * unpenalized;
            }
        } else if let Some((
            ref x_raw,
            ref wxt_raw,
            ref denoms_raw,
            j,
            ref scalar_indices,
            ref block_indices,
        )) = partial_raw
        {
            // ── HYBRID PATH: packed GEMV for scalar + fit_step_ref for block ──
            let u_slice = u.as_slice().unwrap();

            // --- Scalar learners via packed GEMV ---
            let wu2: f64 = w_raw
                .iter()
                .zip(u_slice.iter())
                .map(|(&w, &u)| w * u * u)
                .sum();

            let wxt_full = wxt_raw.as_slice();
            if j >= 4 {
                let wxt_mat = faer::mat::from_row_major_slice::<f64, usize, usize>(wxt_full, j, n);
                let u_col = faer::mat::from_column_major_slice::<f64, usize, usize>(u_slice, n, 1);
                for x in numer_vec.iter_mut() {
                    *x = 0.0;
                }
                let mut numer_mat = faer::mat::from_column_major_slice_mut::<f64, usize, usize>(
                    numer_vec.as_mut_slice(),
                    j,
                    1,
                );
                matmul(
                    &mut numer_mat,
                    wxt_mat,
                    u_col,
                    Some(1.0),
                    1.0,
                    Parallelism::None,
                );
            } else {
                for col in 0..j {
                    let wxt_base = col * n;
                    let wxt_col = &wxt_full[wxt_base..wxt_base + n];
                    numer_vec[col] = wxt_col
                        .iter()
                        .zip(u_slice.iter())
                        .map(|(&w, &u)| w * u)
                        .sum();
                }
            }

            // Find best scalar learner
            let mut best_scalar_col = 0usize;
            let mut best_scalar_sse = f64::INFINITY;
            let mut best_scalar_beta = 0.0f64;
            let mut best_scalar_multiplicity = 0usize;
            for col in 0..j {
                if let Some(mask) = active_mask {
                    // The scalar candidate `col` is global learner
                    // `scalar_indices[col]`; the mask is global-indexed.
                    if !mask[scalar_indices[col]] {
                        continue;
                    }
                }
                let numer = numer_vec[col];
                let denom = denoms_raw[col];
                let (sse, beta) = if denom > 0.0 {
                    (wu2 - numer * numer / denom, numer / denom)
                } else {
                    (wu2, 0.0)
                };
                if observe_candidate(
                    col,
                    sse,
                    &mut best_scalar_col,
                    &mut best_scalar_sse,
                    &mut best_scalar_multiplicity,
                ) {
                    best_scalar_beta = beta;
                }
            }

            // --- Block learners via fit_step_select (SSE by identity, no pred) ---
            let mut best_block_idx: Option<usize> = None;
            let mut best_block_sse = f64::INFINITY;
            let mut best_block_multiplicity = 0usize;
            if block_indices.len() >= PARALLEL_BLOCK_MIN_LEARNERS {
                use rayon::prelude::*;
                if let Some((bi, sse, multiplicity)) = learners
                    .par_iter_mut()
                    .enumerate()
                    .filter_map(|(idx, learner)| {
                        let inactive = active_mask.as_ref().is_some_and(|mask| !mask[idx]);
                        if inactive || matches!(learner, PreparedLearner::ScalarBols { .. }) {
                            None
                        } else {
                            let sse = learner.fit_step_select_sse(u.view(), weights, wu2);
                            Some((idx, sse, usize::from(sse.is_finite())))
                        }
                    })
                    .reduce_with(|a, b| {
                        if b.1 < a.1 {
                            b
                        } else if b.1 == a.1 && b.1.is_finite() {
                            let multiplicity = a.2 + b.2;
                            let mut winner = if b.0 < a.0 { b } else { a };
                            winner.2 = multiplicity;
                            winner
                        } else {
                            a
                        }
                    })
                {
                    best_block_idx = Some(bi);
                    best_block_sse = sse;
                    best_block_multiplicity = multiplicity;
                    best_beta_buf.clear();
                    best_beta_buf.extend_from_slice(learners[bi].selection_beta());
                }
            } else {
                for &bi in block_indices {
                    if let Some(mask) = active_mask {
                        if !mask[bi] {
                            continue;
                        }
                    }
                    let (beta_slice, sse) = learners[bi].fit_step_select(u.view(), weights, wu2);
                    let mut current_best = best_block_idx.unwrap_or(usize::MAX);
                    if observe_candidate(
                        bi,
                        sse,
                        &mut current_best,
                        &mut best_block_sse,
                        &mut best_block_multiplicity,
                    ) {
                        best_block_idx = Some(bi);
                        best_beta_buf.clear();
                        best_beta_buf.extend_from_slice(beta_slice);
                    }
                }
            }

            // Compute prediction for the winning block learner only
            if let Some(bi) = best_block_idx {
                learners[bi].predict_into(&best_beta_buf, &mut best_pred_buf[..n]);
                // Btree fallback: copy from its internal pred_buf
                let pb = learners[bi].pred_buf();
                if !pb.is_empty() && pb.len() == n {
                    best_pred_buf[..n].copy_from_slice(pb);
                }
            }

            // Pick overall winner (scalar vs block), break ties by lowest original index
            let best_scalar_orig_idx = scalar_indices[best_scalar_col];
            let winning_multiplicity = match best_block_idx {
                Some(_) if best_block_sse < best_scalar_sse => best_block_multiplicity,
                Some(_) if best_block_sse == best_scalar_sse && best_block_sse.is_finite() => {
                    best_block_multiplicity + best_scalar_multiplicity
                }
                _ => best_scalar_multiplicity,
            };
            if options.collect_report {
                tie_selection_count += usize::from(winning_multiplicity > 1);
            }
            let (winner_idx, winner_is_scalar) = match best_block_idx {
                Some(bi) if best_block_sse < best_scalar_sse => (bi, false),
                Some(bi) if best_block_sse == best_scalar_sse && bi < best_scalar_orig_idx => {
                    (bi, false)
                }
                _ => (best_scalar_orig_idx, true),
            };

            let nu = control.nu;
            let rho = if line_search {
                if winner_is_scalar {
                    let base = best_scalar_col * n;
                    for i in 0..n {
                        ls_h[i] = x_raw[base + i] * best_scalar_beta;
                    }
                } else {
                    ls_h[..n].copy_from_slice(&best_pred_buf[..n]);
                }
                line_search_rho(
                    &*family,
                    y,
                    f.as_slice().unwrap(),
                    &ls_h,
                    &w_raw,
                    &mut ls_scratch,
                    ls_max_iter,
                )
            } else {
                1.0
            };
            let step = nu * rho;
            let f_slice = f.as_slice_mut().unwrap();
            if winner_is_scalar {
                let best_base = best_scalar_col * n;
                let x_col = &x_raw[best_base..best_base + n];
                let scale = step * best_scalar_beta;
                if options.store_increments {
                    let mut inc_buf: Vec<f64> = vec![0.0f64; n];
                    if use_incr_risk {
                        let u_mut = u.as_slice_mut().unwrap();
                        for i in 0..n {
                            let inc_i = x_col[i] * scale;
                            f_slice[i] += inc_i;
                            inc_buf[i] = inc_i;
                            u_mut[i] -= inc_i;
                        }
                    } else {
                        for i in 0..n {
                            let inc_i = x_col[i] * scale;
                            f_slice[i] += inc_i;
                            inc_buf[i] = inc_i;
                        }
                    }
                    increments.push(inc_buf);
                } else if use_incr_risk {
                    let u_mut = u.as_slice_mut().unwrap();
                    for i in 0..n {
                        let inc_i = x_col[i] * scale;
                        f_slice[i] += inc_i;
                        u_mut[i] -= inc_i;
                    }
                } else {
                    for i in 0..n {
                        f_slice[i] += x_col[i] * scale;
                    }
                }
                selected.push(winner_idx);
                if options.store_coefs {
                    coefs.push(vec![rho * best_scalar_beta]);
                }
                if use_incr_risk && options.store_risk {
                    incr_wu_dot_pred = best_scalar_beta * numer_vec[best_scalar_col];
                    let unpenalized = denoms_raw[best_scalar_col]
                        - learners[best_scalar_orig_idx].lambda().unwrap_or(0.0);
                    incr_w_pred_sq = best_scalar_beta * best_scalar_beta * unpenalized;
                }
            } else {
                if options.store_increments {
                    let mut inc_buf: Vec<f64> = vec![0.0f64; n];
                    if use_incr_risk {
                        let u_mut = u.as_slice_mut().unwrap();
                        incr_wu_dot_pred = 0.0;
                        incr_w_pred_sq = 0.0;
                        for i in 0..n {
                            let p = best_pred_buf[i];
                            let inc_i = step * p;
                            let u_old = u_mut[i];
                            f_slice[i] += inc_i;
                            inc_buf[i] = inc_i;
                            incr_wu_dot_pred += w_raw[i] * u_old * p;
                            incr_w_pred_sq += w_raw[i] * p * p;
                            u_mut[i] = u_old - inc_i;
                        }
                    } else {
                        for i in 0..n {
                            let inc_i = step * best_pred_buf[i];
                            f_slice[i] += inc_i;
                            inc_buf[i] = inc_i;
                        }
                    }
                    increments.push(inc_buf);
                } else if use_incr_risk {
                    let u_mut = u.as_slice_mut().unwrap();
                    if options.store_risk {
                        incr_wu_dot_pred = 0.0;
                        incr_w_pred_sq = 0.0;
                        for i in 0..n {
                            let p = best_pred_buf[i];
                            let inc_i = step * p;
                            let u_old = u_mut[i];
                            f_slice[i] += inc_i;
                            incr_wu_dot_pred += w_raw[i] * u_old * p;
                            incr_w_pred_sq += w_raw[i] * p * p;
                            u_mut[i] = u_old - inc_i;
                        }
                    } else {
                        for i in 0..n {
                            let p = best_pred_buf[i];
                            let inc_i = step * p;
                            f_slice[i] += inc_i;
                            u_mut[i] -= inc_i;
                        }
                    }
                } else {
                    for i in 0..n {
                        f_slice[i] += step * best_pred_buf[i];
                    }
                }
                selected.push(winner_idx);
                if options.store_coefs {
                    // A tree's coefficient encodes its structure: scale the
                    // leaf means only.
                    coefs.push(learners[winner_idx].scale_coef(&best_beta_buf, rho));
                }
            }
        } else {
            // ── ALL-BLOCK PATH: fit_step_select per learner (SSE by identity) ──
            // Fall back to fit_step_ref (explicit pred + SSE) when identity SSE
            // produces NaN (e.g. singular Gram from center_spectral zero-columns).
            let u_slice = u.as_slice().unwrap();
            let wu2: f64 = w_raw
                .iter()
                .zip(u_slice.iter())
                .map(|(&w, &u)| w * u * u)
                .sum();

            let (mut best_idx, best_sse, mut best_multiplicity) = if parallel_block {
                use rayon::prelude::*;
                learners
                    .par_iter_mut()
                    .enumerate()
                    .map(|(idx, learner)| {
                        // An inactive candidate this iteration scores +inf so it
                        // never wins; its stale beta is never read.
                        let inactive = active_mask.as_ref().is_some_and(|mask| !mask[idx]);
                        if inactive {
                            return (idx, f64::INFINITY, 0);
                        }
                        let sse = learner.fit_step_select_sse(u.view(), weights, wu2);
                        (idx, sse, usize::from(sse.is_finite()))
                    })
                    .reduce(|| CANDIDATE_SCAN_IDENTITY, merge_candidate_scans)
            } else {
                let mut best_idx = 0usize;
                let mut best_sse = f64::INFINITY;
                let mut best_multiplicity = 0usize;
                for (idx, learner) in learners.iter_mut().enumerate() {
                    if let Some(mask) = active_mask {
                        if !mask[idx] {
                            continue;
                        }
                    }
                    let sse = learner.fit_step_select_sse(u.view(), weights, wu2);
                    observe_candidate(
                        idx,
                        sse,
                        &mut best_idx,
                        &mut best_sse,
                        &mut best_multiplicity,
                    );
                }
                (best_idx, best_sse, best_multiplicity)
            };

            // No candidate produced a usable score. This was `best_sse.is_nan()`,
            // which the serial fold can never satisfy — `observe_candidate`
            // refuses to store a non-finite score as the best — so the fallback
            // and its `nonfinite_candidate_fallbacks` counter only ever fired on
            // the parallel path, and only for the tree shapes that let a NaN
            // leak through the reduction. Both paths now leave `best_sse` at
            // positive infinity in this case and both take the fallback.
            let no_finite_candidate = !best_sse.is_finite();

            best_beta_buf.clear();
            if no_finite_candidate {
                if options.collect_report {
                    nonfinite_candidate_fallbacks += 1;
                }
                // Fallback: use explicit fit_step_ref (computes pred + SSE directly)
                best_beta_buf.clear();
                let mut fallback_best_sse = f64::INFINITY;
                let mut fallback_best_idx = best_idx;
                let mut fallback_multiplicity = 0usize;
                for (idx, learner) in learners.iter_mut().enumerate() {
                    // Honour component subsampling here too: rescanning every
                    // learner would let an inactive one win and break the
                    // seeded reproducibility the mask exists to provide.
                    if active_mask.as_ref().is_some_and(|mask| !mask[idx]) {
                        continue;
                    }
                    let (_beta, _pred, sse) = learner.fit_step_ref(u.view(), weights);
                    observe_candidate(
                        idx,
                        sse,
                        &mut fallback_best_idx,
                        &mut fallback_best_sse,
                        &mut fallback_multiplicity,
                    );
                }
                best_idx = fallback_best_idx;
                best_multiplicity = fallback_multiplicity;
                // Re-fit winner to get its prediction
                let (beta, pred, _sse) = learners[best_idx].fit_step_ref(u.view(), weights);
                best_beta_buf.extend_from_slice(beta);
                best_pred_buf[..n].copy_from_slice(&pred[..n]);
            } else {
                // Compute prediction for the winner only
                best_beta_buf.extend_from_slice(learners[best_idx].selection_beta());
                learners[best_idx].predict_into(&best_beta_buf, &mut best_pred_buf[..n]);
                // Btree fallback: copy from its internal pred_buf
                let pb = learners[best_idx].pred_buf();
                if !pb.is_empty() && pb.len() == n {
                    best_pred_buf[..n].copy_from_slice(pb);
                }
            }
            if options.collect_report {
                tie_selection_count += usize::from(best_multiplicity > 1);
            }

            let nu = control.nu;
            let rho = if line_search {
                ls_h[..n].copy_from_slice(&best_pred_buf[..n]);
                line_search_rho(
                    &*family,
                    y,
                    f.as_slice().unwrap(),
                    &ls_h,
                    &w_raw,
                    &mut ls_scratch,
                    ls_max_iter,
                )
            } else {
                1.0
            };
            let step = nu * rho;
            let f_slice = f.as_slice_mut().expect("f must be contiguous");
            if options.store_increments {
                let mut inc_buf: Vec<f64> = vec![0.0f64; n];
                if use_incr_risk {
                    let u_mut = u.as_slice_mut().unwrap();
                    incr_wu_dot_pred = 0.0;
                    incr_w_pred_sq = 0.0;
                    for i in 0..n {
                        let p = best_pred_buf[i];
                        let inc_i = step * p;
                        let u_old = u_mut[i];
                        f_slice[i] += inc_i;
                        inc_buf[i] = inc_i;
                        incr_wu_dot_pred += w_raw[i] * u_old * p;
                        incr_w_pred_sq += w_raw[i] * p * p;
                        u_mut[i] = u_old - inc_i;
                    }
                } else {
                    for i in 0..n {
                        let inc_i = step * best_pred_buf[i];
                        f_slice[i] += inc_i;
                        inc_buf[i] = inc_i;
                    }
                }
                increments.push(inc_buf);
            } else if use_incr_risk {
                let u_mut = u.as_slice_mut().unwrap();
                if options.store_risk {
                    incr_wu_dot_pred = 0.0;
                    incr_w_pred_sq = 0.0;
                    for i in 0..n {
                        let p = best_pred_buf[i];
                        let inc_i = step * p;
                        let u_old = u_mut[i];
                        f_slice[i] += inc_i;
                        incr_wu_dot_pred += w_raw[i] * u_old * p;
                        incr_w_pred_sq += w_raw[i] * p * p;
                        u_mut[i] = u_old - inc_i;
                    }
                } else {
                    for i in 0..n {
                        let p = best_pred_buf[i];
                        let inc_i = step * p;
                        f_slice[i] += inc_i;
                        u_mut[i] -= inc_i;
                    }
                }
            } else {
                for i in 0..n {
                    f_slice[i] += step * best_pred_buf[i];
                }
            }
            selected.push(best_idx);
            if options.store_coefs {
                // A tree's coefficient encodes its structure: scale the leaf
                // means only.
                coefs.push(learners[best_idx].scale_coef(&best_beta_buf, rho));
            }
        }

        let calibration = family.calibrate_with_status(y, f.view(), weights);
        if options.collect_report {
            tally_calibration_status(
                calibration,
                &mut calibration_converged_count,
                &mut calibration_limit_count,
                &mut calibration_failure_count,
            );
        }
        if use_incr_risk {
            if options.store_risk {
                let step_risk = *risk.last().unwrap();
                let step_risk = family
                    .risk_from_increment(step_risk, incr_wu_dot_pred, incr_w_pred_sq, control.nu)
                    .expect("use_incr_risk is set only when this returns Some");
                check_step_risk(step_risk, iteration, family.name(), control)?;
                risk.push(step_risk);
            }
        } else if options.store_risk {
            let step_risk = family.ngradient_and_risk(y, f.view(), risk_w.view(), u.view_mut());
            check_step_risk(step_risk, iteration, family.name(), control)?;
            risk.push(step_risk);
        } else {
            family.negative_gradient(y, f.view(), risk_w.view(), u.view_mut());
        }
        if options.store_nuisance {
            nuisance_path.push(family.nuisance());
        }
        if options.store_nuisance {
            nuisance_vec_path.push(family.nuisance_vec().unwrap_or_default());
        }
        if control.trace && options.store_risk {
            let step_risk = *risk.last().unwrap();
            let m = risk.len() - 1;
            use std::fmt::Write;
            let _ = writeln!(trace_buf, "[{m}] risk: {step_risk:.6}");
            if m % 100 == 0 {
                eprint!("{trace_buf}");
                trace_buf.clear();
            }
        }
    }

    if control.trace && options.store_risk {
        let final_risk = risk.last().copied().unwrap_or(0.0);
        use std::fmt::Write;
        let _ = writeln!(trace_buf, "Final risk: {final_risk:.6}");
        eprint!("{trace_buf}");
    }

    // All-scalar path collected betas flat; box into 1-element Vecs. A warm
    // fit primed `coefs` with the retained path, so those go in front --
    // without this the continued path reported only the iterations this call
    // ran.
    let coefs: Vec<Vec<f64>> = if !scalar_betas.is_empty() {
        let retained = warm.map_or(0, |state| state.coefs.len());
        let mut boxed: Vec<Vec<f64>> = coefs.into_iter().take(retained).collect();
        boxed.extend(scalar_betas.into_iter().map(|b| vec![b]));
        boxed
    } else {
        coefs
    };

    if options.collect_report {
        report.completed_mstop = selected.len();
        report.calibration_count = if family.calibrates() {
            selected.len() + 1
        } else {
            0
        };
        report.tie_selection_count = tie_selection_count;
        report.component_subsample_iterations = component_subsample_iterations;
        report.calibration_converged_count = calibration_converged_count;
        report.calibration_limit_count = calibration_limit_count;
        report.calibration_failure_count = calibration_failure_count;
        report.nonfinite_candidate_fallbacks = nonfinite_candidate_fallbacks;
        report.iteration_ns = iteration_started.map(|started| started.elapsed().as_nanos() as u64);
        report.total_ns = fit_started.map(|started| started.elapsed().as_nanos() as u64);
    }

    Ok(BoostingPath {
        report,
        offset,
        selected,
        coefs,
        risk,
        // The gradient the next iteration would regress on, kept so a
        // continuation resumes it rather than re-deriving it -- see
        // `BoostingPath::residual`. Moved, not copied: `u` is allocated once
        // at the top of the fit and is dead after the loop, so retaining it
        // costs nothing. Cross-validation and stability selection go through
        // this same path and would otherwise pay an n-vector copy per fold.
        residual: if options.store_coefs {
            u
        } else {
            Array1::zeros(0)
        },
        fitted: if options.store_coefs {
            f
        } else {
            Array1::zeros(0)
        },
        increments,
        nuisance_path,
        nuisance_vec_path,
        // Final vector nuisance (e.g. PropOdds K-1 thresholds); empty otherwise.
        nuisance_vec: if options.store_nuisance {
            family.nuisance_vec().unwrap_or_default()
        } else {
            Vec::new()
        },
    })
}


mod gaussian_gram;
pub(crate) use self::gaussian_gram::*;


/// Compute the cumulative hat-matrix trace at each boosting iteration,
/// matching R's C routine `R_trace_gamboost`.
///
/// Algorithm (from mboost/src/mboost.c `R_trace_gamboost`):
/// B = 0 (n×n boosting hat matrix)
/// for each step b:
///   `H` = per-learner hat matrix for `selected[b]`
///   z = H %*% B
/// B = nu*H + B - z (i.e. B_new = nu*H + (I - nu*H)·B_old)
/// `trace[b] = sum(diag(B))`
///
/// `nu` is the learning rate (step length) that scales each hat matrix.
/// Returns `None` if any selected learner lacks a hat matrix (bmono, or a
/// btree step whose fitted partition the caller did not supply)
/// or if `selected` contains an out-of-range index (the latter panics).
///
/// Compatibility wrapper around [`try_compute_aic_trace`]; panics on
/// malformed inputs and is deprecated on the P0.8 schedule.
#[deprecated(
    since = "0.1.3",
    note = "use try_compute_aic_trace (returns MboostrsError instead of panicking); removed in 0.2.0"
)]
pub fn compute_aic_trace(
    learners: &[PreparedLearner],
    selected: &[usize],
    nu: f64,
) -> Option<Vec<f64>> {
    try_compute_aic_trace(learners, selected, nu)
        .expect("compute_aic_trace: valid learners/selection/nu")
}

/// The hat matrices of one boosting path, resolved per step.
///
/// `learner_mats` holds the per-learner matrix of every learner that
/// exposes one. A `btree` learner's entry stays `None`: a tree's matrix
/// depends on the partition fitted at the step, and `step_mats[b]` holds
/// it for tree steps, built from the step's recorded coefficients.
/// `step_mats[b]` is `None` for every non-tree step.
struct StepHats {
    learner_mats: Vec<Option<Array2<f64>>>,
    step_mats: Vec<Option<Array2<f64>>>,
}

impl StepHats {
    /// The matrix step `step` applies, for the learner `selected[step]`.
    fn at(&self, step: usize, sel: usize) -> &Array2<f64> {
        match &self.step_mats[step] {
            Some(h) => h,
            None => self.learner_mats[sel]
                .as_ref()
                .expect("resolution stored a matrix for every non-tree learner"),
        }
    }
}

/// Resolve the hat matrices of the path `selected` over `learners`, plus
/// the row count `n` they share.
///
/// Returns `Ok(None)` — the refusal the hat computations have always
/// produced — when a needed matrix is unavailable:
///
/// * a non-tree learner without a hat matrix refuses, selected or not;
/// * a tree step refuses when `step_coefs` is not supplied, so callers
///   that cannot provide the fitted path keep the historical refusal,
///   or when its coefficients do not decode.
///
/// An unselected tree learner does not refuse when `step_coefs` is
/// supplied: it contributes no step. `step_coefs[b]` must be the
/// coefficient vector recorded for step `b`; a length mismatch is an
/// `Err`, as for the weighted path's step weights.
fn resolve_step_hats(
    learners: &[PreparedLearner],
    selected: &[usize],
    step_coefs: Option<&[Vec<f64>]>,
    what: &str,
) -> Result<Option<(StepHats, usize)>, MboostrsError> {
    if let Some(coefs) = step_coefs {
        if coefs.len() != selected.len() {
            return Err(MboostrsError::Solver(format!(
                "{what}: {} coefficient vectors for {} steps",
                coefs.len(),
                selected.len()
            )));
        }
    }
    let mut learner_mats: Vec<Option<Array2<f64>>> = Vec::with_capacity(learners.len());
    for learner in learners {
        match learner.hat_matrix() {
            Some(h) => learner_mats.push(Some(h)),
            None => {
                // A tree resolves per step when the caller supplies the
                // fitted path; every other learner without a matrix, and
                // a tree when it is not supplied, refuses as before.
                if step_coefs.is_some() && matches!(learner, PreparedLearner::Btree { .. }) {
                    learner_mats.push(None);
                } else {
                    return Ok(None);
                }
            }
        }
    }
    let mut step_mats: Vec<Option<Array2<f64>>> = Vec::with_capacity(selected.len());
    for (b, &sel) in selected.iter().enumerate() {
        let learner = &learners[sel];
        if matches!(learner, PreparedLearner::Btree { .. }) {
            let Some(coefs) = step_coefs else {
                return Ok(None);
            };
            match learner.btree_step_hat_matrix(&coefs[b]) {
                Some(h) => step_mats.push(Some(h)),
                None => return Ok(None),
            }
        } else {
            step_mats.push(None);
        }
    }
    let hats = StepHats {
        learner_mats,
        step_mats,
    };
    // The row count every matrix shares. The first learner's matrix
    // supplies it, as it always has; a leading tree learner has none, so
    // the first step's matrix does instead (every learner is prepared on
    // the same rows). With no steps and a leading tree there is no matrix
    // to take it from — the historical refusal.
    let n = if let Some(h) = hats.learner_mats.first().and_then(|h| h.as_ref()) {
        h.nrows()
    } else {
        match selected.first() {
            Some(&sel) => hats.at(0, sel).nrows(),
            None => return Ok(None),
        }
    };
    Ok(Some((hats, n)))
}

/// Fallible [`compute_aic_trace`]: never panics.
///
/// Returns `Ok(None)` when any selected learner lacks a hat matrix (the
/// `bmono` data condition, or a `btree` step whose fitted partition the
/// caller did not supply — see [`try_compute_aic_trace_steps`]), and
/// `Err` when the inputs are malformed (empty learners, an out-of-bounds
/// `selected` index, or non-finite `nu`).
pub fn try_compute_aic_trace(
    learners: &[PreparedLearner],
    selected: &[usize],
    nu: f64,
) -> Result<Option<Vec<f64>>, MboostrsError> {
    try_compute_aic_trace_steps(learners, selected, None, nu)
}

/// [`try_compute_aic_trace`] with the path's per-step coefficient
/// vectors, which give every selected `btree` step the averaging matrix
/// of the partition it was fitted with (see
/// [`PreparedLearner::btree_step_hat_matrix`]). `step_coefs[b]` is the
/// coefficient vector recorded for step `b`.
pub fn try_compute_aic_trace_steps(
    learners: &[PreparedLearner],
    selected: &[usize],
    step_coefs: Option<&[Vec<f64>]>,
    nu: f64,
) -> Result<Option<Vec<f64>>, MboostrsError> {
    if learners.is_empty() {
        return Err(MboostrsError::Solver(
            "compute_aic_trace: no learners".into(),
        ));
    }
    if !nu.is_finite() {
        return Err(MboostrsError::Solver(
            "compute_aic_trace: nu must be finite".into(),
        ));
    }
    validate_selected(selected, learners.len(), "compute_aic_trace")?;
    let Some((hats, n)) = resolve_step_hats(learners, selected, step_coefs, "compute_aic_trace")?
    else {
        return Ok(None);
    };

    let mut b = Array2::<f64>::zeros((n, n));
    let mut z = Array2::<f64>::zeros((n, n));
    let mut trace = Vec::with_capacity(selected.len());

    for (step, &sel) in selected.iter().enumerate() {
        let h = hats.at(step, sel);
        // z = (nu*H) · B
        ndarray::linalg::general_mat_mul(nu, h, &b, 0.0, &mut z);
        b.zip_mut_with(&z, |bj, &zj| *bj -= zj);
        b.zip_mut_with(h, |bj, &hj| *bj += nu * hj);
        let tr: f64 = b.diag().sum();
        trace.push(tr);
    }
    Ok(Some(trace))
}

/// Compute the cumulative hat-matrix diagonal (leverage) at the final boosting iteration.
///
/// Returns the per-observation leverage vector `diag(B_m)` where B_m is the
/// boosting hat matrix after all `selected.len()` iterations.
/// Returns `None` if any selected learner lacks a hat matrix.
///
/// Compatibility wrapper around [`try_compute_hat_diag`]; panics on malformed
/// inputs and is deprecated on the P0.8 schedule.
#[deprecated(
    since = "0.1.3",
    note = "use try_compute_hat_diag (returns MboostrsError instead of panicking); removed in 0.2.0"
)]
pub fn compute_hat_diag(
    learners: &[PreparedLearner],
    selected: &[usize],
    nu: f64,
) -> Option<Vec<f64>> {
    try_compute_hat_diag(learners, selected, nu)
        .expect("compute_hat_diag: valid learners/selection/nu")
}

/// Fallible [`compute_hat_diag`]: never panics.
///
/// Returns `Ok(None)` when any selected learner lacks a hat matrix (the
/// `bmono` data condition, or a `btree` step whose fitted partition the
/// caller did not supply — see [`try_compute_hat_diag_steps`]), and `Err`
/// when the inputs are malformed (empty learners, an out-of-bounds
/// `selected` index, or non-finite `nu`).
pub fn try_compute_hat_diag(
    learners: &[PreparedLearner],
    selected: &[usize],
    nu: f64,
) -> Result<Option<Vec<f64>>, MboostrsError> {
    try_compute_hat_diag_steps(learners, selected, None, nu)
}

/// [`try_compute_hat_diag`] with the path's per-step coefficient
/// vectors, which give every selected `btree` step the averaging matrix
/// of the partition it was fitted with (see
/// [`PreparedLearner::btree_step_hat_matrix`]). `step_coefs[b]` is the
/// coefficient vector recorded for step `b`.
pub fn try_compute_hat_diag_steps(
    learners: &[PreparedLearner],
    selected: &[usize],
    step_coefs: Option<&[Vec<f64>]>,
    nu: f64,
) -> Result<Option<Vec<f64>>, MboostrsError> {
    if learners.is_empty() {
        return Err(MboostrsError::Solver(
            "compute_hat_diag: no learners".into(),
        ));
    }
    if !nu.is_finite() {
        return Err(MboostrsError::Solver(
            "compute_hat_diag: nu must be finite".into(),
        ));
    }
    validate_selected(selected, learners.len(), "compute_hat_diag")?;
    let Some((hats, n)) = resolve_step_hats(learners, selected, step_coefs, "compute_hat_diag")?
    else {
        return Ok(None);
    };

    let mut b = Array2::<f64>::zeros((n, n));
    let mut z = Array2::<f64>::zeros((n, n));

    for (step, &sel) in selected.iter().enumerate() {
        let h = hats.at(step, sel);
        ndarray::linalg::general_mat_mul(nu, h, &b, 0.0, &mut z);
        b.zip_mut_with(&z, |bj, &zj| *bj -= zj);
        b.zip_mut_with(h, |bj, &hj| *bj += nu * hj);
    }

    Ok(Some(b.diag().to_vec()))
}

/// First-order approximation of the boosting hat-matrix diagonal:
/// `h_i = 1/n + nu * sum_m L_{s_m, i}`, clamped to `[0, 1]`, where `s_m` is
/// the learner selected at step `m` and `L_l` is learner `l`'s leverage.
///
/// The exact hat matrix after `m` steps is `B_m = I - prod (I - nu H_{s_m})`;
/// expanding the product and keeping the terms linear in `nu` gives
/// `nu * sum_m H_{s_m}`, whose diagonal is the sum above. The `1/n` term is
/// the offset. The approximation costs one leverage vector per distinct
/// selected learner instead of `n x n` products per step, and it
/// overstates the exact diagonal when a learner is selected many times
/// (each repeat adds its full leverage again, where the exact matrix adds
/// less as the learner's fit saturates).
///
/// `L_l` is the diagonal of the learner's hat matrix
/// ([`PreparedLearner::hat_diag`]), with one exception: for a
/// single-column linear learner (`ScalarBols`) it is the slope part
/// `w_i d_i^2 / (sum_j w_j d_j^2 + lambda)` of the centered column `d`,
/// without the constant that a centered `bols(x)` also fits. That is the
/// leverage the R package's `hatvalues(approximate = TRUE)` has always
/// reported for such learners; leaving the constant out puts the
/// approximation below the exact diagonal by up to `nu / n` per such step.
/// Returns `Ok(None)` when a selected learner
/// has no hat matrix (`btree`, `bmono`, a Kronecker learner).
pub fn try_approximate_hat_diag(
    learners: &[PreparedLearner],
    selected: &[usize],
    nu: f64,
) -> Result<Option<Vec<f64>>, MboostrsError> {
    if learners.is_empty() {
        return Err(MboostrsError::Solver(
            "approximate_hat_diag: no learners".into(),
        ));
    }
    if !nu.is_finite() {
        return Err(MboostrsError::Solver(
            "approximate_hat_diag: nu must be finite".into(),
        ));
    }
    validate_selected(selected, learners.len(), "approximate_hat_diag")?;
    let n = learners[0].n();
    let mut counts = vec![0usize; learners.len()];
    for &sel in selected {
        counts[sel] += 1;
    }
    let mut h = vec![1.0 / n as f64; n];
    for (learner, &count) in learners.iter().zip(&counts) {
        if count == 0 {
            continue;
        }
        let leverage = match learner {
            PreparedLearner::ScalarBols {
                design,
                scalar_denom,
                weights,
                ..
            } => {
                if *scalar_denom <= 0.0 {
                    continue;
                }
                design
                    .iter()
                    .zip(weights.iter())
                    .map(|(&d, &w)| w * d * d / scalar_denom)
                    .collect()
            }
            other => match other.hat_diag() {
                Some(diag) => diag,
                None => return Ok(None),
            },
        };
        if leverage.len() != n {
            return Err(MboostrsError::Shape(format!(
                "approximate_hat_diag: a learner has {} rows, the first has {n}",
                leverage.len()
            )));
        }
        let scale = nu * count as f64;
        for (hi, li) in h.iter_mut().zip(&leverage) {
            *hi += scale * li;
        }
    }
    h.iter_mut().for_each(|v| *v = v.clamp(0.0, 1.0));
    Ok(Some(h))
}

/// The IRLS weight `fW(f)` of mboost's GLM families, the factor by which the
/// boosting hat matrix weights each step for a family whose working response
/// is not the response itself. `None` for a family mboost treats as L2
/// (Gaussian, GaussReg and every family without `fW`), whose hat matrix is
/// unweighted.
pub fn glm_hat_weight(family: &str, f: f64) -> Option<f64> {
    match family {
        // mboost's Binomial: f is half the log odds, p = e^f / (e^f + e^-f).
        "binomial" => {
            let f = f.abs().min(36.0) * f.signum();
            let p = f.exp() / (f.exp() + (-f).exp());
            Some(4.0 * p * (1.0 - p))
        }
        "poisson" => Some(f.exp()),
        _ => None,
    }
}

/// The trace after every boosting step and the final diagonal of a hat matrix.
pub type HatPath = (Vec<f64>, Vec<f64>);

/// The boosting hat matrix of a GLM-family fit, as mboost's `bhatmat`:
/// `B <- B + diag(w_m) (nu H_m) (I - B)` with `w_m = fW(f_m)` the IRLS weights
/// at the fit after step `m` (`step_fits[m]`, offset included). Returns the
/// trace after every step and the final diagonal, or `Ok(None)` when a
/// selected learner has no hat matrix (the `bmono` data condition, or a
/// `btree` step whose fitted partition the caller did not supply — see
/// [`try_compute_hat_weighted_steps`]).
pub fn try_compute_hat_weighted(
    learners: &[PreparedLearner],
    selected: &[usize],
    nu: f64,
    step_weights: &[Vec<f64>],
) -> Result<Option<HatPath>, MboostrsError> {
    try_compute_hat_weighted_steps(learners, selected, None, nu, step_weights)
}

/// [`try_compute_hat_weighted`] with the path's per-step coefficient
/// vectors, which give every selected `btree` step the averaging matrix
/// of the partition it was fitted with (see
/// [`PreparedLearner::btree_step_hat_matrix`]). `step_coefs[b]` is the
/// coefficient vector recorded for step `b`.
pub fn try_compute_hat_weighted_steps(
    learners: &[PreparedLearner],
    selected: &[usize],
    step_coefs: Option<&[Vec<f64>]>,
    nu: f64,
    step_weights: &[Vec<f64>],
) -> Result<Option<HatPath>, MboostrsError> {
    if learners.is_empty() {
        return Err(MboostrsError::Solver(
            "compute_hat_weighted: no learners".into(),
        ));
    }
    if !nu.is_finite() {
        return Err(MboostrsError::Solver(
            "compute_hat_weighted: nu must be finite".into(),
        ));
    }
    validate_selected(selected, learners.len(), "compute_hat_weighted")?;
    if step_weights.len() != selected.len() {
        return Err(MboostrsError::Solver(format!(
            "compute_hat_weighted: {} weight vectors for {} steps",
            step_weights.len(),
            selected.len()
        )));
    }
    let Some((hats, n)) =
        resolve_step_hats(learners, selected, step_coefs, "compute_hat_weighted")?
    else {
        return Ok(None);
    };
    let mut b = Array2::<f64>::zeros((n, n));
    let mut z = Array2::<f64>::zeros((n, n));
    let mut trace = Vec::with_capacity(selected.len());
    for ((step, &sel), w) in selected.iter().enumerate().zip(step_weights) {
        if w.len() != n {
            return Err(MboostrsError::Solver(format!(
                "compute_hat_weighted: a weight vector has {} entries for {n} rows",
                w.len()
            )));
        }
        let h = hats.at(step, sel);
        // z = (nu H)(I - B) = nu H - nu H B
        ndarray::linalg::general_mat_mul(-nu, h, &b, 0.0, &mut z);
        z.zip_mut_with(h, |zj, &hj| *zj += nu * hj);
        for (i, mut row) in z.rows_mut().into_iter().enumerate() {
            row.mapv_inplace(|v| v * w[i]);
        }
        b += &z;
        trace.push(b.diag().sum());
    }
    Ok(Some((trace, b.diag().to_vec())))
}

/// Validate that every `selected` index lies in `0..len`.
fn validate_selected(selected: &[usize], len: usize, what: &str) -> Result<(), MboostrsError> {
    for (i, &sel) in selected.iter().enumerate() {
        if sel >= len {
            return Err(MboostrsError::Solver(format!(
                "{what}: selected[{i}] = {sel} out of bounds (expected 0..{len})"
            )));
        }
    }
    Ok(())
}

/// Compute the per-iteration hat-matrix trace from explicit hat matrices and
/// the selection path, or `None` when no hat matrices are available.
///
/// Compatibility wrapper around [`try_compute_aic_trace_from_hats`]; panics on
/// malformed inputs and is deprecated on the P0.8 schedule.
#[deprecated(
    since = "0.1.3",
    note = "use try_compute_aic_trace_from_hats (returns MboostrsError instead of panicking); removed in 0.2.0"
)]
pub fn compute_aic_trace_from_hats(
    hat_mats: &[Array2<f64>],
    selected: &[usize],
    nu: f64,
) -> Option<Vec<f64>> {
    if hat_mats.is_empty() {
        return None;
    }
    Some(
        try_compute_aic_trace_from_hats(hat_mats, selected, nu)
            .expect("compute_aic_trace_from_hats: valid inputs"),
    )
}

/// Fallible [`compute_aic_trace_from_hats`]: never panics.
///
/// Returns `Err` when no hat matrices are supplied, a `selected` index is
/// out of bounds, or `nu` is non-finite.
pub fn try_compute_aic_trace_from_hats(
    hat_mats: &[Array2<f64>],
    selected: &[usize],
    nu: f64,
) -> Result<Vec<f64>, MboostrsError> {
    if hat_mats.is_empty() {
        return Err(MboostrsError::Solver(
            "compute_aic_trace_from_hats: no hat matrices".into(),
        ));
    }
    if !nu.is_finite() {
        return Err(MboostrsError::Solver(
            "compute_aic_trace_from_hats: nu must be finite".into(),
        ));
    }
    validate_selected(selected, hat_mats.len(), "compute_aic_trace_from_hats")?;
    let n = hat_mats[0].nrows();
    let mut b = Array2::<f64>::zeros((n, n));
    let mut z = Array2::<f64>::zeros((n, n));
    let mut trace = Vec::with_capacity(selected.len());

    for &sel in selected {
        let h = &hat_mats[sel];
        ndarray::linalg::general_mat_mul(nu, h, &b, 0.0, &mut z);
        b.zip_mut_with(&z, |bj, &zj| *bj -= zj);
        b.zip_mut_with(h, |bj, &hj| *bj += nu * hj);
        let tr: f64 = b.diag().sum();
        trace.push(tr);
    }
    Ok(trace)
}

/// Compute the AIC degrees-of-freedom trace without materialising n×n hat matrices.
///
/// Uses the low-rank factorisation H_j = X·M·(XᵀW) and maintains B in the
/// P-dimensional Gram space where P = Σ p_j. Each step costs O(P²·p̄) instead
/// of O(n²), which is asymptotically faster when P ≪ n.
///
/// Falls back to the dense `compute_aic_trace` if any learner lacks a hat
/// factor, or if P² ≥ 2·n² (where the n-space cost would be lower).
///
/// Compatibility wrapper around [`try_compute_aic_trace_fast`]; panics on
/// malformed inputs and is deprecated on the P0.8 schedule.
#[deprecated(
    since = "0.1.3",
    note = "use try_compute_aic_trace_fast (returns MboostrsError instead of panicking); removed in 0.2.0"
)]
pub fn compute_aic_trace_fast(
    learners: &[PreparedLearner],
    selected: &[usize],
    nu: f64,
) -> Option<Vec<f64>> {
    try_compute_aic_trace_fast(learners, selected, nu)
        .expect("compute_aic_trace_fast: valid learners/selection/nu")
}

/// One learner's degrees of freedom along a boosting path.
///
/// A penalized learner with smoother eigenvalues `s_k` that is selected `m`
/// times at step `nu` has, on its own column space, the accumulated operator
/// `I - (I - nu S)^m`, with eigenvalues `1 - (1 - nu s_k)^m`. Their sum,
/// `boosted`, starts at `nu tr(S)` and rises with every selection towards
/// `unpenalized`, the number of positive `s_k` (the rank of the design): a
/// direction the penalty shrinks hard is still fitted once it has been
/// boosted often enough. This is the penalization drift of Schulte and
/// Rügamer (2025). The measure treats each learner on its own; overlap
/// between learners' column spaces is ignored.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LearnerDrift {
    /// Times the learner was selected.
    pub selections: usize,
    /// `tr(S)`, the df of one full (`nu = 1`) step.
    pub single_step: f64,
    /// `sum_k 1 - (1 - nu s_k)^m`.
    pub boosted: f64,
    /// Number of eigenvalues above `1e-8`: the df of the unpenalized fit.
    pub unpenalized: f64,
}

impl LearnerDrift {
    /// Share of the gap between one full step's df and the unpenalized df
    /// that boosting has used, `(boosted - single_step) / (unpenalized -
    /// single_step)`, floored at 0; `0` when the penalty removes nothing.
    pub fn share(&self) -> f64 {
        let gap = self.unpenalized - self.single_step;
        if gap <= 1e-8 {
            return 0.0;
        }
        ((self.boosted - self.single_step) / gap).max(0.0)
    }
}

/// [`LearnerDrift`] for every learner of a path, `None` where the learner has
/// no closed-form smoother. Errors on an out-of-bounds `selected` index or a
/// `nu` outside `(0, 1]`.
pub fn penalty_drift(
    learners: &[PreparedLearner],
    selected: &[usize],
    nu: f64,
) -> Result<Vec<Option<LearnerDrift>>, MboostrsError> {
    if !(nu > 0.0 && nu <= 1.0) {
        return Err(MboostrsError::Control(format!(
            "nu must be in (0, 1], got {nu}"
        )));
    }
    let mut counts = vec![0usize; learners.len()];
    for &j in selected {
        *counts.get_mut(j).ok_or_else(|| {
            MboostrsError::Shape(format!(
                "selected index {j} is out of bounds for {} learners",
                learners.len()
            ))
        })? += 1;
    }
    Ok(learners
        .iter()
        .zip(&counts)
        .map(|(learner, &m)| {
            let spectrum = crate::learners::smoother_spectrum(learner)?;
            let boosted = spectrum
                .iter()
                .map(|&s| 1.0 - (1.0 - nu * s).powi(m as i32))
                .sum();
            Some(LearnerDrift {
                selections: m,
                single_step: spectrum.iter().sum(),
                boosted,
                unpenalized: spectrum.iter().filter(|&&s| s > 1e-8).count() as f64,
            })
        })
        .collect())
}

/// Fallible [`compute_aic_trace_fast`]: never panics.
///
/// Returns `Ok(None)` when a learner lacks a hat factor *and* the dense
/// fallback also finds no hat matrices (or a learner exposes no weights), and
/// `Err` when the inputs are malformed (no learners, an out-of-bounds
/// `selected` index, or non-finite `nu`).
pub fn try_compute_aic_trace_fast(
    learners: &[PreparedLearner],
    selected: &[usize],
    nu: f64,
) -> Result<Option<Vec<f64>>, MboostrsError> {
    aic_trace_fast_impl(learners, selected, None, nu)
}

/// [`try_compute_aic_trace_fast`] with the path's per-step coefficient
/// vectors. A tree has no hat factor, so a path containing a `btree` step
/// takes the dense fallback, where the coefficients give the step the
/// averaging matrix of its fitted partition (see
/// [`try_compute_aic_trace_steps`]).
pub fn try_compute_aic_trace_fast_steps(
    learners: &[PreparedLearner],
    selected: &[usize],
    step_coefs: Option<&[Vec<f64>]>,
    nu: f64,
) -> Result<Option<Vec<f64>>, MboostrsError> {
    aic_trace_fast_impl(learners, selected, step_coefs, nu)
}

fn aic_trace_fast_impl(
    learners: &[PreparedLearner],
    selected: &[usize],
    step_coefs: Option<&[Vec<f64>]>,
    nu: f64,
) -> Result<Option<Vec<f64>>, MboostrsError> {
    let n_learners = learners.len();
    if n_learners == 0 {
        return Err(MboostrsError::Solver(
            "compute_aic_trace_fast: no learners".into(),
        ));
    }
    if selected.is_empty() {
        return Err(MboostrsError::Solver(
            "compute_aic_trace_fast: empty selection".into(),
        ));
    }
    if !nu.is_finite() {
        return Err(MboostrsError::Solver(
            "compute_aic_trace_fast: nu must be finite".into(),
        ));
    }
    validate_selected(selected, n_learners, "compute_aic_trace_fast")?;

    // Extract factors; Ok(None) if any learner lacks a closed-form hat.
    let factors: Vec<HatFactor> = learners.iter().filter_map(|l| l.hat_factor()).collect();
    if factors.len() != n_learners {
        // fall back to dense path
        return try_compute_aic_trace_steps(learners, selected, step_coefs, nu);
    }

    let n = factors[0].design.nrows();
    let offsets: Vec<usize> = factors
        .iter()
        .scan(0usize, |off, f| {
            let o = *off;
            *off += f.design.ncols();
            Some(o)
        })
        .collect();
    let p_total = offsets
        .last()
        .map(|&o| o + factors.last().unwrap().design.ncols())
        .unwrap_or(0);

    // Fallback: if P² ≥ 2·n², the Gram-space cost exceeds the n-space cost.
    if p_total * p_total >= 2 * n * n {
        return try_compute_aic_trace_steps(learners, selected, step_coefs, nu);
    }

    // Weights (same for all learners in a model)
    let w = match learners.first().and_then(|l| l.weights()) {
        Some(w) => w,
        None => return Ok(None),
    };

    // ── Pre-compute cross-Gram G = Xᵀ·W·X (P×P) with one matmul ──
    // Pack all learner designs into one (n × P) matrix, pre-scale a copy by
    // the weights, and let the matmul kernel do the O(n·P²) work.
    let mut x_all = Array2::<f64>::zeros((n, p_total));
    for (a, factor) in factors.iter().enumerate() {
        let off_a = offsets[a];
        for i in 0..n {
            for k in 0..factor.design.ncols() {
                x_all[[i, off_a + k]] = factor.design.read(i, k);
            }
        }
    }
    let mut wx = x_all.clone();
    for (i, mut row) in wx.outer_iter_mut().enumerate() {
        row *= w[i];
    }
    let gram: Array2<f64> = x_all.t().dot(&wx);

    // ── Per-step trace loop ──
    let max_pj = factors.iter().map(|f| f.hat_inv_p).max().unwrap_or(0);
    let mut c = Array2::<f64>::zeros((p_total, p_total));
    // Hoisted scratch buffers (row-major flat vecs) for faer matmuls.
    let mut diff_buf = vec![0.0f64; max_pj * p_total]; // C[j,:] − G[j,:]
    let mut d_buf = vec![0.0f64; max_pj * p_total]; // Mⱼ·(C[j,:] − G[j,:])
    let g_flat = gram.as_slice().expect("gram contiguous");
    let mut trace_cur = 0.0f64;
    let mut trace_full = Vec::with_capacity(selected.len());

    for &sel in selected {
        let p_j = factors[sel].hat_inv_p;
        let off_j = offsets[sel];
        let m_j = &factors[sel].hat_inv;
        let tr_hj = factors[sel].tr_h;

        // Read c and gram through raw slices (avoids ndarray per-element
        // bounds checks).  The scope ensures the shared borrow on c ends
        // before the mutable matmul below.
        let (tr_mc, diff_len) = {
            let c_flat = c.as_slice().expect("c contiguous");

            // tr(M_j · C_{j,j}) over the p_j×p_j diagonal block of C
            let mut tr_mc = 0.0f64;
            for k in 0..p_j {
                for l in 0..p_j {
                    tr_mc += m_j[k * p_j + l] * c_flat[(off_j + l) * p_total + off_j + k];
                }
            }

            // diff = C[j,:] − G[j,:]  (p_j × P, row-major)
            for l in 0..p_j {
                let row_off = (off_j + l) * p_total;
                for b in 0..p_total {
                    diff_buf[l * p_total + b] = c_flat[row_off + b] - g_flat[row_off + b];
                }
            }

            (tr_mc, p_j * p_total)
        };

        trace_cur = trace_cur - nu * tr_mc + nu * tr_hj;
        trace_full.push(trace_cur);

        // D = Mⱼ · diff  (p_j × P, faer matmul)
        {
            let m_view = faer::mat::from_row_major_slice::<f64, usize, usize>(m_j, p_j, p_j);
            let diff_view = faer::mat::from_row_major_slice::<f64, usize, usize>(
                &diff_buf[..diff_len],
                p_j,
                p_total,
            );
            for v in d_buf[..diff_len].iter_mut() {
                *v = 0.0;
            }
            let mut d_view = faer::mat::from_row_major_slice_mut::<f64, usize, usize>(
                &mut d_buf[..diff_len],
                p_j,
                p_total,
            );
            faer::linalg::matmul::matmul(
                &mut d_view,
                m_view,
                diff_view,
                Some(1.0),
                1.0,
                faer::Parallelism::None,
            );
        }

        // C ← C − ν·G[:,j]·D  (ndarray matmul)
        let d_nd = ndarray::ArrayView2::from_shape((p_j, p_total), &d_buf[..diff_len]).unwrap();
        let g_j = gram.slice(ndarray::s![.., off_j..off_j + p_j]);
        ndarray::linalg::general_mat_mul(-nu, &g_j, &d_nd, 1.0, &mut c);
    }

    Ok(Some(trace_full))
}

/// Compute AIC for each boosting iteration, returning (aic_values, optimal_mstop, df_trace).
///
/// `method`: "corrected" (Hurvich–Clifford for Gaussian), "classical" (AIC = 2*risk + k*df), "gmdl".
/// `k`: penalty multiplier (default 2 for AIC).
/// `risk`: empirical risk at each iteration (length mstop+1; `risk[0]` is offset-only).
/// `sumw`: sum of inbag weights.
///
/// Compatibility wrapper around [`try_compute_aic`]; panics on malformed
/// inputs and is deprecated on the P0.8 schedule.
#[deprecated(
    since = "0.1.3",
    note = "use try_compute_aic (returns MboostrsError instead of panicking); removed in 0.2.0"
)]
pub fn compute_aic(
    method: &str,
    risk: &[f64],
    df: &[f64],
    sumw: f64,
    k: f64,
    y_ss: f64,
) -> (Vec<f64>, usize) {
    try_compute_aic(method, risk, df, sumw, k, y_ss).expect("compute_aic: valid inputs")
}

/// Fallible [`compute_aic`]: never panics.
///
/// Returns `Err` when `risk` is shorter than `df.len() + 1` (the code reads
/// `risk[i+1]` for each `df[i]`, i.e. the post-update risk after step `i+1`),
/// when `sumw` is negative or non-finite, or when `k` is non-finite.
pub fn try_compute_aic(
    method: &str,
    risk: &[f64],
    df: &[f64],
    sumw: f64,
    k: f64,
    y_ss: f64,
) -> Result<(Vec<f64>, usize), MboostrsError> {
    let m = df.len();

    if !matches!(method, "corrected" | "classical" | "gmdl") {
        return Err(MboostrsError::Control(format!(
            "aic method must be corrected, classical or gmdl, got {method:?}"
        )));
    }

    if risk.len() < m.saturating_add(1) {
        return Err(MboostrsError::Solver(format!(
            "compute_aic: risk has {} entries but df has {}; risk must hold at least df + 1 values (risk[m] is the risk after m boosting steps)",
            risk.len(),
            m
        )));
    }
    if !sumw.is_finite() || sumw < 0.0 {
        return Err(MboostrsError::Solver(format!(
            "compute_aic: sumw must be finite and non-negative, got {sumw}"
        )));
    }
    if !k.is_finite() {
        return Err(MboostrsError::Solver(format!(
            "compute_aic: k must be finite, got {k}"
        )));
    }

    // Guard 1: no weight → no information
    if sumw == 0.0 {
        return Ok((vec![f64::INFINITY; m], 1));
    }

    let mut aic = Vec::with_capacity(m);

    for i in 0..m {
        let r = risk[i + 1];
        let d = df[i];
        let val = match method {
            "corrected" => {
                let ratio = r / sumw;
                // Guard 2: clamp denominator to > 0 so the correction term stays finite
                let denom = 1.0 - (d + 2.0) / sumw;
                let clamped_denom = if denom <= 0.0 {
                    f64::MIN_POSITIVE
                } else {
                    denom
                };
                ratio.ln() + (1.0 + d / sumw) / clamped_denom
            }
            "classical" => 2.0 * r + k * d,
            "gmdl" => {
                if d == 0.0 {
                    // Guard 3: d=0 would make d*s = 0 in the log denominator below
                    (r / sumw).ln()
                } else {
                    let s = r / (sumw - d);
                    // Guard 4: s must be positive for ln()
                    let clamped_s = if s <= 0.0 { f64::MIN_POSITIVE } else { s };
                    // Guard 5: (y_ss - r) must be positive for ln()
                    let y_ss_diff = y_ss - r;
                    let clamped_y_ss_diff = if y_ss_diff <= 0.0 {
                        f64::MIN_POSITIVE
                    } else {
                        y_ss_diff
                    };
                    clamped_s.ln() + d / sumw * (clamped_y_ss_diff / (d * clamped_s)).ln()
                }
            }
            _ => unreachable!("criterion method validated before evaluation"),
        };
        aic.push(val);
    }

    let mstop = argmin_skip_nan(&aic).map(|i| i + 1).unwrap_or(1);

    Ok((aic, mstop))
}

/// Index of the smallest non-NaN value (first one on ties), like R's
/// `which.min`, which skips NaN. `None` when every value is NaN or the slice
/// is empty.
fn argmin_skip_nan(v: &[f64]) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (i, x) in v.iter().enumerate() {
        if x.is_nan() {
            continue;
        }
        if best.is_none_or(|b| *x < v[b]) {
            best = Some(i);
        }
    }
    best
}


#[cfg(test)]
mod test_suite;
