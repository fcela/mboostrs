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

/// Gaussian-only covariance/Gram-update fast path (glmnet-style).
///
/// For squared-error loss the residual updates linearly with the fitted
/// increment (`u ← u − ν·X_w·β_w`), so each learner's gradient projection
/// `rhsⱼ = Xⱼ'W u` can be maintained by a cheap `pⱼ×p_w` cross-Gram update
/// instead of re-projecting against the length-n residual every iteration.
/// Per-iteration cost drops from O(L·p·n) to O(L·p²); the length-n vectors
/// are touched only at setup (build cross-Grams + initial projections) and
/// once at the end to reconstruct the fitted values. Selection-only callers
/// skip coefficient/risk retention and final fitted-value reconstruction.
///
/// Returns `None` (caller falls back to the general path) when there is no
/// block learner, a learner is not one of scalar `bols`, block `bols`, or
/// `bmono`, or a candidate solve becomes non-finite. There is deliberately no
/// width-versus-`mstop` crossover guard: for realistic boosting paths the
/// O(n·P²) setup is amortized by avoiding O(n) projection work per candidate
/// and iteration.
///
/// Squared-error semantics are assumed (offset = weighted mean, risk = Σw·u²);
/// the caller must only invoke this when `family.supports_gram_update()`.
/// `Bⱼ'W B_w`, accumulated over blocks of rows so that only a block of the
/// weighted design is ever formed.
fn weighted_cross_gram(
    left: ndarray::ArrayView2<f64>,
    weights: ArrayView1<f64>,
    right: ndarray::ArrayView2<f64>,
) -> Array2<f64> {
    use ndarray::s;
    const ROWS: usize = 4096;
    let n = left.nrows();
    let mut acc = Array2::<f64>::zeros((left.ncols(), right.ncols()));
    let mut scaled = Array2::<f64>::zeros((ROWS.min(n.max(1)), left.ncols()));
    let mut start = 0;
    while start < n {
        let end = (start + ROWS).min(n);
        let mut block = scaled.slice_mut(s![..end - start, ..]);
        block.assign(&left.slice(s![start..end, ..]));
        block *= &weights.slice(s![start..end]).insert_axis(ndarray::Axis(1));
        ndarray::linalg::general_mat_mul(
            1.0,
            &block.t(),
            &right.slice(s![start..end, ..]),
            1.0,
            &mut acc,
        );
        start = end;
    }
    acc
}

/// A repeatedly used matrix-vector product. Large sparse matrices retain an
/// exact CSR cache; small or dense matrices keep ndarray's dense kernel. This
/// discovers support from exact zeros, including any fill introduced by
/// spectral centering, rather than assuming every spatial basis is banded.
struct GramMatVec<'a> {
    matrix: &'a Array2<f64>,
    sparse: Option<crate::linalg::CsrMatrix>,
}

impl<'a> GramMatVec<'a> {
    fn new(matrix: &'a Array2<f64>) -> Self {
        Self {
            matrix,
            sparse: crate::linalg::CsrMatrix::try_from_dense(matrix.view()),
        }
    }

    fn mul_into(&self, rhs: &Array1<f64>, mut out: ndarray::ArrayViewMut1<f64>) {
        debug_assert_eq!(rhs.len(), self.matrix.ncols());
        debug_assert_eq!(out.len(), self.matrix.nrows());
        if let Some(rows) = &self.sparse {
            for row in 0..out.len() {
                let mut sum = 0.0;
                for entry in rows.offsets[row]..rows.offsets[row + 1] {
                    sum += rows.values[entry] * rhs[rows.columns[entry]];
                }
                out[row] = sum;
            }
        } else {
            ndarray::linalg::general_mat_vec_mul(1.0, self.matrix, rhs, 0.0, &mut out);
        }
    }

    fn cache_bytes(&self) -> usize {
        self.sparse
            .as_ref()
            .map_or(0, crate::linalg::CsrMatrix::byte_size)
    }
}

pub(crate) fn try_fit_gaussian_gram(
    learners: &mut [PreparedLearner],
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    control: &BoostControl,
    offset_override: Option<f64>,
    collect_report: bool,
    store_coefs: bool,
    store_risk: bool,
) -> Option<BoostingPath> {
    use crate::learners::PreparedLearner as PL;
    let timing_enabled = collect_report && control.instrumentation != InstrumentationLevel::Off;
    let gram_started = timing_enabled.then(Instant::now);
    let n = y.len(); // Gaussian gram path only called for Gaussian family (n == y.len())

    enum Solv<'a> {
        Scalar(f64),
        Block(&'a crate::linalg::GramSolver),
        // BmonoBbs: constrained solve, handled specially in the selection
        // loop below (never through `solve_into`/`Pen::quad` — a constrained
        // beta does not satisfy the ridge stationarity those rely on).
        Mono {
            solver_a: &'a crate::linalg::GramSolver,
            penalised_gram: &'a Array2<f64>,
            d_mat: &'a Array2<f64>,
            kind: BmonoSolver,
            lambda2: f64,
            niter: usize,
        },
    }
    impl Solv<'_> {
        /// Solve into a caller-owned buffer (zero-alloc hot path). Not called
        /// for `Mono` — see the selection loop.
        fn solve_into(&self, rhs: &Array1<f64>, out: &mut Array1<f64>) {
            match self {
                Solv::Scalar(d) => out[0] = if *d > 0.0 { rhs[0] / d } else { 0.0 },
                Solv::Block(s) => s.solve_into_slice(
                    rhs.as_slice().expect("rhs contiguous"),
                    out.as_slice_mut().expect("beta buffer contiguous"),
                ),
                Solv::Mono { .. } => unreachable!("Mono handled specially in the selection loop"),
            }
        }
    }
    enum Pen<'a> {
        Scalar(f64),
        Block(GramMatVec<'a>),
        // BmonoBbs never goes through `quad` — the selection loop computes
        // its SSE via the full-form identity directly. Kept so `pens` stays
        // one Vec<Pen> indexed in parallel with `solvers`/`designs`.
        Mono,
    }
    impl Pen<'_> {
        /// β'Pβ — the penalty term of the residual-SSE identity.
        fn quad(&self, beta: &Array1<f64>, scratch: &mut Array1<f64>) -> f64 {
            match self {
                Pen::Scalar(lam) => lam * beta[0] * beta[0],
                Pen::Block(p) => {
                    let mut out = scratch.slice_mut(ndarray::s![..beta.len()]);
                    p.mul_into(beta, out.view_mut());
                    beta.dot(&out)
                }
                Pen::Mono => unreachable!("Mono handled specially in the selection loop"),
            }
        }
    }

    // Cheap pre-scan (no cloning): confirm all-linear, ≥1 block learner, and
    // the crossover guard, before paying for any extraction. This keeps the
    // pure-scalar glmboost path (which falls through to the packed path) free
    // of overhead.
    let mstop = control.mstop as usize;
    let mut total_p = 0usize;
    let mut has_block = false;
    for lr in learners.iter() {
        match lr {
            PL::ScalarBols { .. } => total_p += 1,
            PL::BlockBols { design, .. } => {
                has_block = true;
                total_p += design.ncols();
            }
            PL::BmonoBbs { design, .. } => {
                has_block = true;
                total_p += design.ncols();
            }
            // A tree is fitted to an explicit residual kept beside the
            // projections; see `Candidate::Tree`.
            PL::Btree { .. } => {}
            _ => return None, // KroneckerBols and the rest are not handled here
        }
    }
    // Pure-scalar models keep the packed-scalar path. Gate the cross-Gram
    // setup on whether the per-iteration savings amortise the O(L²·p²·n)
    // setup cost. The general path costs O(L·p·n) per iteration; the Gram
    // path costs O(L·p²) per iteration with an O(n·P²) setup (P = Σpⱼ).
    // Break-even: n·P² / (L·p·(n − p̄)) ≈ P² / (L·p̄·(1 − p̄/n)) iterations,
    // simplified to P / L iterations when p̄ ≪ n. Conservative: always
    // try Gram when has_block, bail only for pure-scalar or zero-p models.
    if !has_block || total_p == 0 {
        return None;
    }

    // Split the learners: the linear ones are borrowed for their designs, the
    // trees mutably, since fitting one updates its internal buffers. The
    // candidate order is the learners' order, so ties still go to the
    // lowest index.
    enum Candidate {
        Linear(usize),
        Tree(usize),
    }
    let mut candidates: Vec<Candidate> = Vec::with_capacity(learners.len());
    let mut linear_learners: Vec<&PreparedLearner> = Vec::new();
    let mut trees: Vec<&mut PreparedLearner> = Vec::new();
    for lr in learners.iter_mut() {
        if matches!(lr, PL::Btree { .. }) {
            candidates.push(Candidate::Tree(trees.len()));
            trees.push(lr);
        } else {
            candidates.push(Candidate::Linear(linear_learners.len()));
            linear_learners.push(lr);
        }
    }
    let l = linear_learners.len();
    let has_trees = !trees.is_empty();

    // Extract linear pieces. The designs are borrowed from the learners, and
    // each learner's weights stand in for a stored `Bⱼ'W`: the cross-Grams
    // are accumulated over row blocks (`weighted_cross_gram`), so this route
    // adds no n-row copy of any design.
    let fit_weights = weights.to_owned();
    let mut designs: Vec<ndarray::ArrayView2<f64>> = Vec::with_capacity(l); // (n, pⱼ)
    let mut row_weights: Vec<ArrayView1<f64>> = Vec::with_capacity(l);
    let mut solvers: Vec<Solv> = Vec::with_capacity(l);
    let mut pens: Vec<Pen> = Vec::with_capacity(l);
    // Preparation's unpenalized X'WX for block learners that computed one
    // (the tensor spatial learner). The route's self-Gram block is the
    // largest single setup product at tensor sizes; reusing the stash removes
    // it from the critical path (#276). The stash is an exact CSR copy (the
    // spatial Gram is banded, so a tenth of the dense bytes), cleared by
    // weight-refitting, so a present stash always matches the learner's
    // weights.
    let mut self_grams: Vec<Option<&crate::linalg::CsrMatrix>> = Vec::with_capacity(l);
    for lr in linear_learners.iter().copied() {
        match lr {
            PL::ScalarBols {
                design,
                weights: learner_weights,
                scalar_denom,
                penalty,
                ..
            } => {
                designs.push(design.view().insert_axis(ndarray::Axis(1))); // (n,1)
                row_weights.push(learner_weights.view());
                solvers.push(Solv::Scalar(*scalar_denom));
                pens.push(Pen::Scalar(*penalty));
                self_grams.push(None);
            }
            PL::BlockBols {
                design,
                weights: learner_weights,
                solver,
                penalty,
                weighted_gram,
                ..
            } => {
                designs.push(design.view());
                row_weights.push(learner_weights.view());
                solvers.push(Solv::Block(solver));
                pens.push(Pen::Block(GramMatVec::new(penalty)));
                self_grams.push(weighted_gram.as_ref());
            }
            PL::BmonoBbs {
                design,
                solver_a,
                penalised_gram,
                d_mat,
                solver,
                lambda2,
                niter,
                ..
            } => {
                designs.push(design.view());
                // BmonoBbs keeps no weight vector; it was prepared with the
                // fit's weights, which its `X'W` was built from.
                row_weights.push(fit_weights.view());
                solvers.push(Solv::Mono {
                    solver_a,
                    penalised_gram,
                    d_mat,
                    kind: *solver,
                    lambda2: *lambda2,
                    niter: *niter,
                });
                pens.push(Pen::Mono);
                self_grams.push(None);
            }
            _ => unreachable!("non-linear learner rejected in pre-scan"),
        }
    }

    // Offset (Gaussian = weighted mean) and initial residual u₀ = y − offset.
    let sw: f64 = weights.sum();
    let offset =
        offset_override.unwrap_or_else(|| if sw > 0.0 { y.dot(&weights) / sw } else { 0.0 });
    let u0: Array1<f64> = &y.to_owned() - offset;
    // The explicit residual, kept only when a tree competes: a tree is
    // fitted to it, and it cannot be recovered from the projections.
    let mut u: Array1<f64> = if has_trees {
        u0.clone()
    } else {
        Array1::zeros(0)
    };
    let mut wu2: f64 = weights
        .iter()
        .zip(u0.iter())
        .map(|(&w, &u)| w * u * u)
        .sum();

    // Cross-Grams G[j][w] = Bⱼ'W·B_w (pⱼ × p_w); initial projections rhsⱼ.
    // Compute only the upper triangle (j ≤ w) and fill the lower triangle by
    // transposition: G[j][w] = G[w][j]ᵀ for j > w. For J ≥ 4 learners the
    // upper-triangle matmuls are independent and CPU-bound, so dispatch them
    // via rayon.
    // Exact-duplicate designs (#301): when two learners' designs and row
    // weights are bit-identical, every Gram block in the duplicate's
    // column equals the matching block in the source's column, so the
    // pair's cross block and the duplicate's diagonal are copies of
    // those blocks rather than fresh O(n·p²) matmuls. The
    // copies are exact — the same values in the same standard layout the
    // matmul produces — so duplicate learners keep the bit-identical blocks
    // the R-parity tie contract requires. The check is O(n·p) per pair
    // (pointer equality first for designs sharing one buffer).
    fn designs_bit_identical(a: ndarray::ArrayView2<f64>, b: ndarray::ArrayView2<f64>) -> bool {
        if a.dim() != b.dim() {
            return false;
        }
        if std::ptr::eq(a.as_ptr(), b.as_ptr()) {
            return true;
        }
        match (a.as_slice(), b.as_slice()) {
            (Some(x), Some(y)) => {
                x.len() == y.len()
                    && x.iter()
                        .zip(y.iter())
                        .all(|(u, v)| u.to_bits() == v.to_bits())
            }
            _ => a
                .iter()
                .zip(b.iter())
                .all(|(u, v)| u.to_bits() == v.to_bits()),
        }
    }
    fn weights_bit_identical(a: ArrayView1<f64>, b: ArrayView1<f64>) -> bool {
        if a.len() != b.len() {
            return false;
        }
        if std::ptr::eq(a.as_ptr(), b.as_ptr()) {
            return true;
        }
        match (a.as_slice(), b.as_slice()) {
            (Some(x), Some(y)) => x
                .iter()
                .zip(y.iter())
                .all(|(u, v)| u.to_bits() == v.to_bits()),
            _ => a
                .iter()
                .zip(b.iter())
                .all(|(u, v)| u.to_bits() == v.to_bits()),
        }
    }
    // dup_src[j] = the first earlier learner whose design and row weights
    // are bit-identical to learner j's, if any.
    let mut dup_src: Vec<Option<usize>> = vec![None; l];
    for j in 1..l {
        for w in 0..j {
            if weights_bit_identical(row_weights[j], row_weights[w])
                && designs_bit_identical(designs[j], designs[w])
            {
                dup_src[j] = Some(w);
                break;
            }
        }
    }

    let cross_gram_started = timing_enabled.then(Instant::now);
    let mut g: Vec<Vec<Array2<f64>>> = (0..l)
        .map(|_| (0..l).map(|_| Array2::zeros((0, 0))).collect())
        .collect();
    let pairs: Vec<(usize, usize)> = (0..l)
        .flat_map(|j| (j..l).map(move |w| (j, w)))
        .filter(|&(j, w)| {
            // A pair whose later member duplicates an earlier learner is
            // filled from that learner's blocks below, not computed.
            dup_src[w].is_none() || j == w
        })
        .collect();
    let upper_count = pairs.len();
    // Diagonal blocks prefer the stashed X'WX; cross blocks are always
    // computed. The materialized stash carries the same standard layout the
    // cross-Gram accumulator produces, so the mat-vec kernels treat them
    // identically.
    let compute_pair = |&(j, w): &(usize, usize)| {
        if j == w {
            self_grams[j].map_or_else(
                || weighted_cross_gram(designs[j], row_weights[j], designs[j]),
                |csr| csr.to_dense(),
            )
        } else {
            weighted_cross_gram(designs[j], row_weights[j], designs[w])
        }
    };
    let results: Vec<Array2<f64>> = if upper_count >= 10 {
        use rayon::prelude::*;
        pairs.par_iter().map(compute_pair).collect()
    } else {
        pairs.iter().map(compute_pair).collect()
    };
    for ((j, w), mat) in pairs.into_iter().zip(results) {
        g[j][w] = mat;
    }
    // Fill the pairs skipped above from the source learner's blocks: for
    // a duplicate w of source s, the block g[j][w] equals the block
    // between learners j and s. For j ≤ s that is the upper-triangle
    // block g[j][s] itself (the shared diagonal when j = s), which was
    // computed. For s < j < w it is the transpose of the computed upper
    // block g[s][j] — g[j][s] is a lower-triangle block and still empty
    // at this point — materialised in standard layout exactly as the
    // lower-triangle fill below does. The diagonal g[w][w] is the
    // source's diagonal g[s][s]. Duplicates are processed in increasing
    // w, so when a block g[s][j] was itself filled (j a duplicate of s),
    // that fill has already happened.
    for w in 0..l {
        if let Some(s) = dup_src[w] {
            for j in 0..=w {
                g[j][w] = if j <= s {
                    g[j][s].clone()
                } else if j < w {
                    let (rows, cols) = g[s][j].dim(); // (p_s, p_j)
                    let mut block = Array2::<f64>::zeros((cols, rows)); // (p_j, p_s)
                    block.assign(&g[s][j].t());
                    block
                } else {
                    g[s][s].clone()
                };
            }
        }
    }
    // Lower triangle: transpose the upper. O(p²) memcpy per matrix — trivial
    // compared to the O(p²·n) matmul saved. The copy is materialised in
    // standard (row-major) layout, matching the upper triangle's matmul
    // results: `.t().to_owned()` would keep the transposed layout, and the
    // mat-vec kernels accumulate in a layout-dependent order, so equal-valued
    // blocks would update bit-identical learners differently — breaking the
    // exact selection ties R preserves for duplicate-column learners
    // (gaussian_gram_twin_learners_stay_tied).
    for j in 1..l {
        for w in 0..j {
            let (rows, cols) = g[w][j].dim(); // (p_w, p_j)
            let mut block = Array2::<f64>::zeros((cols, rows)); // (p_j, p_w)
            block.assign(&g[w][j].t());
            g[j][w] = block;
        }
    }
    let products: Vec<Vec<GramMatVec<'_>>> = g
        .iter()
        .map(|row| row.iter().map(GramMatVec::new).collect())
        .collect();
    let sparse_cache_bytes: usize = products
        .iter()
        .flatten()
        .map(GramMatVec::cache_bytes)
        .sum::<usize>()
        + pens
            .iter()
            .map(|pen| match pen {
                Pen::Block(matrix) => matrix.cache_bytes(),
                _ => 0,
            })
            .sum::<usize>();
    let mut rhs: Vec<Array1<f64>> = (0..l)
        .map(|j| designs[j].t().dot(&(&row_weights[j] * &u0)))
        .collect();
    let cross_gram_ns = cross_gram_started.map(|started| started.elapsed().as_nanos() as u64);

    let nu = control.nu;
    let mut selected = Vec::with_capacity(mstop);
    let mut coefs: Vec<Vec<f64>> = if store_coefs {
        Vec::with_capacity(mstop)
    } else {
        Vec::new()
    };
    let mut risk = if store_risk {
        Vec::with_capacity(mstop + 1)
    } else {
        Vec::new()
    };
    if store_risk {
        risk.push(wu2);
    }

    // Per-learner beta buffers and a shared matvec scratch — the inner loop
    // below is then allocation-free (GRAM2).
    let p_js: Vec<usize> = designs.iter().map(|d| d.ncols()).collect();
    let max_p = p_js.iter().copied().max().unwrap_or(0);
    let mut betas: Vec<Array1<f64>> = p_js.iter().map(|&p| Array1::zeros(p)).collect();
    let mut scratch = Array1::<f64>::zeros(max_p);
    let mut tie_selection_count = 0usize;
    let setup_ns = gram_started.map(|started| started.elapsed().as_nanos() as u64);
    let iteration_started = timing_enabled.then(Instant::now);
    for _ in 0..mstop {
        let mut best = 0usize;
        let mut best_sse = f64::INFINITY;
        let mut best_multiplicity = 0usize;
        for (global, candidate) in candidates.iter().enumerate() {
            let j = match candidate {
                Candidate::Linear(j) => *j,
                Candidate::Tree(t) => {
                    let sse = trees[*t].fit_step_select(u.view(), weights, wu2).1;
                    observe_candidate(
                        global,
                        sse,
                        &mut best,
                        &mut best_sse,
                        &mut best_multiplicity,
                    );
                    continue;
                }
            };
            let sse = match &solvers[j] {
                Solv::Mono {
                    solver_a,
                    penalised_gram,
                    d_mat,
                    kind,
                    lambda2,
                    niter,
                } => {
                    let beta = match kind {
                        BmonoSolver::Iterative => solve_bmono_iterative(
                            solver_a,
                            penalised_gram.view(),
                            d_mat.view(),
                            rhs[j].view(),
                            *lambda2,
                            *niter,
                        ),
                        BmonoSolver::QuadProg => solve_bmono_quadprog(
                            solver_a,
                            penalised_gram.view(),
                            d_mat.view(),
                            rhs[j].view(),
                            1e-10,
                            4 * d_mat.nrows().max(8),
                        ),
                    };
                    betas[j].assign(&beta);
                    // Full-form SSE with the unpenalized cross-Gram
                    // g[j][j] = Xⱼ'W Xⱼ (a constrained beta does not satisfy
                    // the ridge stationarity the shortcut above relies on).
                    let mut t = scratch.slice_mut(ndarray::s![..p_js[j]]);
                    products[j][j].mul_into(&betas[j], t.view_mut());
                    let t_view = scratch.slice(ndarray::s![..p_js[j]]);
                    wu2 - 2.0 * betas[j].dot(&rhs[j]) + betas[j].dot(&t_view)
                }
                _ => {
                    solvers[j].solve_into(&rhs[j], &mut betas[j]);
                    // residual SSE identity: Σw(u−Xβ)² = wu2 − β'rhs − β'Pβ
                    wu2 - betas[j].dot(&rhs[j]) - pens[j].quad(&betas[j], &mut scratch)
                }
            };
            observe_candidate(
                global,
                sse,
                &mut best,
                &mut best_sse,
                &mut best_multiplicity,
            );
        }
        // No learner produced a finite SSE — the design is singular/degenerate
        // (e.g. a rank-deficient centered learner with λ=0, where the solver
        // returns NaN). Bail to the general path, which tolerates this (a
        // NaN SSE fails its `<` selection test, yielding a harmless no-op
        // step) rather than propagating NaN through the incremental updates.
        if !best_sse.is_finite() {
            return None;
        }
        if collect_report {
            tie_selection_count += usize::from(best_multiplicity > 1);
        }
        let w = match candidates[best] {
            Candidate::Tree(t) => {
                // A tree step: move the explicit residual, and each linear
                // learner's projection by the tree's weighted fit.
                let tree_pred = trees[t].pred_buf();
                for i in 0..n {
                    u[i] -= nu * tree_pred[i];
                }
                let w_tree_pred: Array1<f64> = Array1::from_shape_fn(n, |i| tree_pred[i]);
                for j in 0..l {
                    let projected = designs[j].t().dot(&(&row_weights[j] * &w_tree_pred));
                    rhs[j].scaled_add(-nu, &projected);
                }
                wu2 = weights
                    .iter()
                    .zip(u.iter())
                    .map(|(&wt, &ui)| wt * ui * ui)
                    .sum();
                selected.push(best);
                if store_coefs {
                    coefs.push(trees[t].selection_beta().to_vec());
                }
                None
            }
            Candidate::Linear(w) => Some(w),
        };
        if let Some(w) = w {
            // wu2 ← wu2 − 2ν·β_w'rhs_w + ν²·β_w'(B_w'W B_w)β_w (M_w = G[w][w])
            {
                let mut t = scratch.slice_mut(ndarray::s![..p_js[w]]);
                products[w][w].mul_into(&betas[w], t.view_mut());
            }
            let mw_beta = scratch.slice(ndarray::s![..p_js[w]]);
            wu2 = wu2 - 2.0 * nu * betas[w].dot(&rhs[w]) + nu * nu * betas[w].dot(&mw_beta);
            // Reuse the self-product for the winner's RHS. Other learners
            // use the same product-then-update arithmetic so duplicate designs
            // stay tied even when one of them is the winner.
            rhs[w].scaled_add(-nu, &mw_beta);
            for j in 0..l {
                if j == w {
                    continue;
                }
                let mut out = scratch.slice_mut(ndarray::s![..p_js[j]]);
                products[j][w].mul_into(&betas[w], out.view_mut());
                rhs[j].scaled_add(-nu, &out);
            }
            if has_trees {
                let step = designs[w].dot(&betas[w]);
                u.scaled_add(-nu, &step);
                wu2 = weights
                    .iter()
                    .zip(u.iter())
                    .map(|(&wt, &ui)| wt * ui * ui)
                    .sum();
            }
            selected.push(best);
            if store_coefs {
                coefs.push(betas[w].to_vec());
            }
        }
        if store_risk {
            if !wu2.is_finite() {
                // The Gram path returns `Option` — `None` means "decline, use
                // the general path". Declining here routes this fit to the
                // engine's own non-finite-risk check, which reports the
                // iteration in a typed error instead of publishing a path of
                // meaningless risks.
                return None;
            }
            risk.push(wu2);
        }
    }
    let iteration_ns = iteration_started.map(|started| started.elapsed().as_nanos() as u64);

    // Reconstruct fitted: f = offset + Σⱼ Bⱼ·(ν·Σ_{m:sel=j} coefs[m]).
    let reconstruction_started = (timing_enabled && store_coefs).then(Instant::now);
    let fitted = if store_coefs && has_trees {
        // `u` is y minus the fit, kept exactly alongside the projections.
        &y.to_owned() - &u
    } else if store_coefs {
        // Without trees the selected indices are the linear learners' own.
        let mut totals: Vec<Vec<f64>> = designs.iter().map(|d| vec![0.0; d.ncols()]).collect();
        for m in 0..mstop {
            let t = &mut totals[selected[m]];
            let c = &coefs[m];
            for k in 0..c.len().min(t.len()) {
                t[k] += nu * c[k];
            }
        }
        let mut fitted = Array1::from_elem(n, offset);
        for j in 0..l {
            let t_arr = Array1::from_vec(totals[j].clone());
            fitted += &designs[j].dot(&t_arr);
        }
        fitted
    } else {
        Array1::zeros(0)
    };
    let reconstruction_ns =
        reconstruction_started.map(|started| started.elapsed().as_nanos() as u64);

    Some(BoostingPath {
        report: if collect_report {
            FitReport {
                tie_selection_count,
                setup_ns,
                cross_gram_ns,
                iteration_ns,
                reconstruction_ns,
                gram_cache_bytes: total_p
                    .saturating_mul(total_p)
                    .saturating_mul(std::mem::size_of::<f64>())
                    .saturating_add(sparse_cache_bytes),
                ..FitReport::default()
            }
        } else {
            FitReport::default()
        },
        offset,
        selected,
        coefs,
        risk,
        // The Gram route maintains per-learner projections instead of a
        // length-n residual, so there is none to retain; a continuation of
        // such a fit takes the general route.
        residual: Array1::zeros(0),
        fitted,
        increments: Vec::new(),
        // Gaussian fast path: no scalar nuisance.
        nuisance_path: Vec::new(),
        nuisance_vec_path: Vec::new(),
        nuisance_vec: Vec::new(),
    })
}

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
mod tests {
    use super::*;

    #[test]
    fn argmin_skip_nan_ignores_nan_after_minimum() {
        // Previously [1, NaN, 5] selected index 2 (value 5) because a NaN
        // following the minimum replaced it.
        assert_eq!(argmin_skip_nan(&[1.0, f64::NAN, 5.0]), Some(0));
        assert_eq!(argmin_skip_nan(&[f64::NAN, 3.0, 2.0]), Some(2));
        // All NaN: no minimum; the caller falls back to mstop = 1.
        assert_eq!(argmin_skip_nan(&[f64::NAN, f64::NAN]), None);
    }

    // ─── per-observation offsets (R `offset=` semantics) ──────────────────

    #[test]
    fn vector_offset_is_not_estimated_and_recovers_the_known_contribution() {
        // y = o + 0.5*x with a per-observation o: fitting x with the offset
        // supplied must recover 0.5*x exactly, proving the offset entered the
        // start and was never re-estimated.
        let x = Array1::linspace(0.0, 1.0, 60);
        let o = Array1::from_iter((0..60).map(|i| (i as f64) * 0.2 - 4.0));
        let y = &o + &(0.5 * &x);
        let w = Array1::ones(60);
        let control = BoostControl::new(400, 1.0).unwrap();
        // The block form `[1, x]` is what a vector-offset fit needs: the
        // centered single column leans on the model offset for its constant
        // (`gamboost`'s equivalence, see `learner_spec`), and a per-observation
        // offset is not a free constant, so the Python layer routes vector
        // offset fits to the block form. This test uses it for the same
        // reason.
        let design =
            ndarray::Array2::from_shape_fn((60, 2), |(i, j)| if j == 0 { 1.0 } else { x[i] });
        let penalty = ndarray::Array2::zeros((2, 2));
        let mut learners = vec![prepare_block_bols(design, w.view(), penalty, false, "x")];
        let mut family = Gaussian;
        let path = try_fit_with_vector_offset(
            &mut learners,
            y.view(),
            w.view(),
            &mut family,
            &control,
            o.view(),
            false,
        )
        .unwrap();
        let max = path
            .fitted
            .iter()
            .zip(y.iter())
            .map(|(fitted, observed)| (fitted - observed).abs())
            .fold(0.0, f64::max);
        assert!(max < 1e-10, "fitted differs from the offset model by {max}");
        // The scalar field reports the scalar start this fit never used; it
        // is not part of the model, which is why the caller keeps the vector.
        let _ = path.offset;
    }

    #[test]
    fn vector_offset_poisson_rates_track_exposure() {
        // Counts over unequal exposure: mu_i = exposure_i * exp(b0 + b1 x).
        // The offset log(exposure) makes the model a rate model, and the fit
        // above the offset tracks x rather than absorbing the exposure.
        let mut rng: u64 = 7;
        let mut uniform = move || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let n = 200;
        let x = Array1::from_iter((0..n).map(|_| 2.0 * uniform() - 1.0));
        let exposure = Array1::from_iter((0..n).map(|_| 0.5 + 3.5 * uniform()));
        let log_rate = 0.4 + &(0.6 * &x);
        let y = Array1::from_iter((0..n).map(|i| {
            // Knuth's Poisson sampler around lambda = exposure * rate.
            let lambda = exposure[i] * log_rate[i].exp();
            let threshold = (-lambda).exp();
            let mut count = 0.0_f64;
            let mut product = uniform();
            while product > threshold {
                count += 1.0;
                product *= uniform();
            }
            count
        }));
        let w = Array1::ones(n);
        let control = BoostControl::new(300, 0.1).unwrap();
        let mut learners = vec![prepare_scalar_bols(
            x.view(),
            w.view(),
            0.0,
            true,
            "x",
            None,
        )];
        let mut family = crate::families::Poisson;
        let offset = exposure.map(|e| e.ln());
        let path = try_fit_with_vector_offset(
            &mut learners,
            y.view(),
            w.view(),
            &mut family,
            &control,
            offset.view(),
            false,
        )
        .unwrap();
        // eta_i = log(exposure_i) + b0 + b1 x_i: the fit above the offset is
        // a smooth function of x with real spread, not a constant that would
        // mean the exposure was absorbed into the estimate.
        let rate_fit = &path.fitted - &offset;
        let mean_rate = rate_fit.sum() / n as f64;
        let spread = rate_fit
            .iter()
            .map(|v| (v - mean_rate).abs())
            .fold(0.0, f64::max);
        assert!(
            spread > 0.1,
            "the fit above the offset tracks x (spread {spread})"
        );
    }

    #[test]
    fn vector_offset_length_and_finiteness_are_refused() {
        let x = Array1::linspace(0.0, 1.0, 10);
        let y = x.clone();
        let w = Array1::ones(10);
        let control = BoostControl::new(5, 0.1).unwrap();
        let mut learners = vec![prepare_scalar_bols(
            x.view(),
            w.view(),
            0.0,
            true,
            "x",
            None,
        )];
        let mut family = Gaussian;
        let wrong_length = Array1::zeros(9);
        let error = try_fit_with_vector_offset(
            &mut learners,
            y.view(),
            w.view(),
            &mut family,
            &control,
            wrong_length.view(),
            false,
        )
        .unwrap_err();
        assert!(error.to_string().contains("9 entries"));

        let zero_exposure =
            Array1::from_iter((0..10).map(|i| if i == 3 { f64::NEG_INFINITY } else { 0.0 }));
        let error = try_fit_with_vector_offset(
            &mut learners,
            y.view(),
            w.view(),
            &mut family,
            &control,
            zero_exposure.view(),
            false,
        )
        .unwrap_err();
        assert!(error.to_string().contains("log(0)"));
    }

    #[test]
    fn unknown_information_criterion_is_rejected() {
        let error = try_compute_aic("typo", &[2.0, 1.0], &[0.1], 10.0, 2.0, 2.0)
            .expect_err("an unknown method must not return a NaN path");
        assert!(error.to_string().contains("aic method"));
    }
    use crate::families::{AdaExp, Gaussian, HingeLoss, Huber, Poisson};
    use crate::learners::{
        prepare_bbs, prepare_binned_scalar_bols, prepare_block_bols, prepare_scalar_bols,
        quantile_bin_column,
    };
    // `prepare_block_bols` is also used directly by the vector-offset tests.
    use ndarray::{array, Array1, Array2};

    fn assert_differential_paths_close(
        reference: &BoostingPath,
        optimized: &BoostingPath,
        seed: u64,
        specification: &str,
        tolerance: f64,
    ) {
        assert_eq!(
            reference.selected, optimized.selected,
            "selection mismatch; seed={seed} spec={specification}"
        );
        let max_coefficient_difference = reference
            .coefs
            .iter()
            .zip(&optimized.coefs)
            .flat_map(|(left, right)| left.iter().zip(right))
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_coefficient_difference <= tolerance,
            "coefficient mismatch {max_coefficient_difference:.3e}; seed={seed} spec={specification}"
        );
        let risk_scale = reference
            .risk
            .first()
            .copied()
            .unwrap_or(1.0)
            .abs()
            .max(1.0);
        let max_relative_risk_difference = reference
            .risk
            .iter()
            .zip(&optimized.risk)
            .map(|(left, right)| (left - right).abs() / risk_scale)
            .fold(0.0_f64, f64::max);
        assert!(
            max_relative_risk_difference <= tolerance,
            "risk mismatch {max_relative_risk_difference:.3e}; seed={seed} spec={specification}"
        );
        let max_fitted_difference = reference
            .fitted
            .iter()
            .zip(&optimized.fitted)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_fitted_difference <= tolerance,
            "fitted mismatch {max_fitted_difference:.3e}; seed={seed} spec={specification}"
        );
    }

    #[test]
    fn fit_empty_learners_is_offset_only() {
        // No learners: zero boosting steps, no selections, every fitted value
        // is the family offset (the weighted mean for Gaussian).
        let mut learners: Vec<crate::learners::PreparedLearner> = Vec::new();
        let y = array![1.0, 2.0, 6.0];
        let w = array![1.0, 1.0, 1.0];
        let mut fam = Gaussian;
        let ctrl = BoostControl::new(5, 0.1).unwrap();
        let path = fit_with(
            &mut learners,
            y.view(),
            w.view(),
            &mut fam,
            &ctrl,
            None,
            true,
        );
        assert!(path.selected.is_empty());
        assert_eq!(path.mstop(), 0);
        assert_eq!(path.risk.len(), 1);
        assert!((path.offset - 3.0).abs() < 1e-12);
        assert!(path.fitted.iter().all(|&f| (f - 3.0).abs() < 1e-12));
    }

    #[test]
    fn try_fit_empty_learners_offset_respects_vector_offset() {
        let mut learners: Vec<crate::learners::PreparedLearner> = Vec::new();
        let y = array![1.0, 2.0, 6.0];
        let w = array![1.0, 1.0, 1.0];
        let offset = array![0.5, 1.5, 2.5];
        let mut fam = Gaussian;
        let ctrl = BoostControl::new(5, 0.1).unwrap();
        let path = try_fit_with_vector_offset(
            &mut learners,
            y.view(),
            w.view(),
            &mut fam,
            &ctrl,
            offset.view(),
            true,
        )
        .expect("offset-only fit");
        assert!(path.selected.is_empty());
        assert!(path
            .fitted
            .iter()
            .zip(offset.iter())
            .all(|(f, o)| (f - o).abs() < 1e-12));
    }

    #[test]
    fn margin_family_binary_codings_share_the_core_fit_path() {
        let x = array![-2.0, -1.0, 1.0, 2.0];
        let y01 = array![0.0, 0.0, 1.0, 1.0];
        let ypm = array![-1.0, -1.0, 1.0, 1.0];
        let weights = Array1::ones(4);
        let control = BoostControl::new(20, 0.1).unwrap();

        let fit = |y: &Array1<f64>, family: &mut dyn Family| {
            let mut learners = vec![prepare_scalar_bols(
                x.view(),
                weights.view(),
                0.0,
                true,
                "x",
                None,
            )];
            try_fit(
                &mut learners,
                y.view(),
                weights.view(),
                family,
                &control,
                None,
            )
            .unwrap()
        };

        for (mut raw_family, mut encoded_family) in [
            (
                Box::new(AdaExp) as Box<dyn Family>,
                Box::new(AdaExp) as Box<dyn Family>,
            ),
            (
                Box::new(HingeLoss::default()) as Box<dyn Family>,
                Box::new(HingeLoss::default()) as Box<dyn Family>,
            ),
        ] {
            let encoded_by_family = fit(&y01, raw_family.as_mut());
            let already_encoded = fit(&ypm, encoded_family.as_mut());
            assert_eq!(encoded_by_family.selected, already_encoded.selected);
            assert_eq!(encoded_by_family.risk, already_encoded.risk);
            assert_eq!(encoded_by_family.fitted, already_encoded.fitted);
        }
    }

    #[test]
    fn binned_scalar_selection_path_matches_exact_at_distinct_count() {
        let n = 400usize;
        let columns: Vec<Array1<f64>> = (0..6)
            .map(|column| {
                Array1::from_iter((0..n).map(|row| {
                    let phase = (row * (column + 3) + column * 17) as f64;
                    (phase * 0.017).sin() + 0.002 * row as f64 * (column + 1) as f64
                }))
            })
            .collect();
        let y = Array1::from_iter((0..n).map(|row| {
            1.4 * columns[0][row] - 0.9 * columns[2][row]
                + 0.35 * columns[4][row]
                + ((row * 37 % 101) as f64 / 101.0 - 0.5) * 0.2
        }));
        let weights = Array1::from_iter((0..n).map(|row| 0.5 + (row % 9) as f64 / 7.0));
        let mut exact = columns
            .iter()
            .enumerate()
            .map(|(index, column)| {
                prepare_scalar_bols(
                    column.view(),
                    weights.view(),
                    0.3,
                    true,
                    &format!("x{index}"),
                    None,
                )
            })
            .collect::<Vec<_>>();
        let mut binned = columns
            .iter()
            .enumerate()
            .map(|(index, column)| {
                let bins = quantile_bin_column(column.view(), n).unwrap();
                prepare_binned_scalar_bols(bins, weights.view(), 0.3, true, &format!("x{index}"))
            })
            .collect::<Vec<_>>();
        // Stop before the signal learners become numerically tied at the
        // floating-point floor. The two routes use different candidate-scan
        // kernels (packed GEMM versus one-learner bin scans), so once every
        // remaining improvement is roundoff-sized, a path identity assertion
        // tests the BLAS reduction order rather than the binned design.
        let control = BoostControl::new(100, 0.1).unwrap();
        let mut exact_family = Gaussian;
        let exact_path = try_fit(
            &mut exact,
            y.view(),
            weights.view(),
            &mut exact_family,
            &control,
            None,
        )
        .unwrap();
        let mut binned_family = Gaussian;
        let binned_path = try_fit(
            &mut binned,
            y.view(),
            weights.view(),
            &mut binned_family,
            &control,
            None,
        )
        .unwrap();

        assert_eq!(exact_path.selected, binned_path.selected);
        let fitted_difference = exact_path
            .fitted
            .iter()
            .zip(&binned_path.fitted)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f64, f64::max);
        let coefficient_difference = exact_path
            .coefs
            .iter()
            .zip(&binned_path.coefs)
            .flat_map(|(left, right)| left.iter().zip(right))
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f64, f64::max);
        let risk_difference = exact_path
            .risk
            .iter()
            .zip(&binned_path.risk)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f64, f64::max);
        assert!(
            fitted_difference < 1e-12,
            "fitted={fitted_difference}, coef={coefficient_difference}"
        );
        assert!(risk_difference < 1e-9, "{risk_difference}");
    }

    #[test]
    fn packed_scalar_rejects_a_round_without_a_finite_candidate() {
        // Squaring this finite Poisson gradient overflows `wu2`, so every
        // scalar candidate has a non-finite SSE. The packed path must report
        // that numerical failure instead of recording learner zero with a
        // zero coefficient for every requested iteration.
        let x1 = array![-2.0, -1.0, 1.0, 2.0];
        let x2 = array![1.0, -1.0, 1.0, -1.0];
        let y = Array1::from_elem(4, 1e300);
        let weights = Array1::ones(4);
        let mut learners = vec![
            prepare_scalar_bols(x1.view(), weights.view(), 0.0, true, "x1", None),
            prepare_scalar_bols(x2.view(), weights.view(), 0.0, true, "x2", None),
        ];
        let control = BoostControl::new(5, 0.1).unwrap();
        let mut family = Poisson;

        let error = try_fit_with(
            &mut learners,
            y.view(),
            weights.view(),
            &mut family,
            &control,
            None,
            false,
        )
        .expect_err("non-finite candidate scores must not produce a fake selection");

        assert!(
            matches!(error, MboostrsError::Solver(message) if message.contains("no finite candidate SSE"))
        );
    }

    #[test]
    fn a_non_finite_offset_only_risk_is_reported_rather_than_stored() {
        // Enhancement-hunt 7.2: `try_fit_lss` validated the initial risk and
        // every step's risk and returned a typed error; the single-parameter
        // engine did neither, so a fit whose very first risk evaluation
        // overflowed returned "successfully" with an infinite risk path.
        // Gaussian squares its residuals, and 1e200^2 overflows an f64.
        let x = array![-2.0, -1.0, 1.0, 2.0];
        let y = array![1e200, -1e200, 1e200, -1e200];
        let weights = Array1::ones(4);
        let mut learners = vec![prepare_scalar_bols(
            x.view(),
            weights.view(),
            0.0,
            true,
            "x",
            None,
        )];
        let control = BoostControl::new(5, 0.1).unwrap();
        let mut family = Gaussian;

        let error = try_fit_with(
            &mut learners,
            y.view(),
            weights.view(),
            &mut family,
            &control,
            None,
            false,
        )
        .expect_err("an offset-only risk that overflows must not produce a fit");

        assert!(
            matches!(&error, MboostrsError::Solver(message)
                     if message.contains("offset-only model is not finite")),
            "unexpected error: {error:?}"
        );
    }

    #[test]
    fn the_gram_fast_path_declines_rather_than_publishing_a_non_finite_risk() {
        // The Gram path returns `Option`, so it cannot carry the typed error.
        // It must decline instead, routing the fit to the general path, which
        // owns the check. Either way the caller never receives a path with a
        // non-finite risk in it.
        let x1 = array![1e200, -1e200, 1e200, -1e200, 1e200, -1e200];
        let x2 = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = array![1e200, -1e200, 1e200, -1e200, 1e200, -1e200];
        let weights = Array1::ones(6);
        let mut learners = vec![
            prepare_scalar_bols(x1.view(), weights.view(), 0.0, true, "x1", None),
            prepare_scalar_bols(x2.view(), weights.view(), 0.0, true, "x2", None),
        ];
        let control = BoostControl::new(20, 0.1).unwrap();
        let mut family = Gaussian;

        match try_fit_with(
            &mut learners,
            y.view(),
            weights.view(),
            &mut family,
            &control,
            None,
            false,
        ) {
            Err(MboostrsError::Solver(_)) => {}
            Err(other) => panic!("unexpected error kind: {other:?}"),
            Ok(path) => assert!(
                path.risk.iter().all(|value| value.is_finite()),
                "a successful fit must not carry a non-finite risk: {:?}",
                path.risk
            ),
        }
    }

    #[test]
    fn execution_report_routes_and_instrumentation_are_deterministic() {
        use crate::report::{EngineRoute, FallbackReason, InstrumentationLevel};

        let y = array![1.0, 2.0, 4.0, 3.0, 5.0, 7.0];
        let w = Array1::ones(y.len());
        let x = Array1::linspace(-1.0, 1.0, y.len());
        let scalar = prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None);

        let mut off_learners = vec![scalar.clone()];
        let mut summary_learners = vec![scalar];
        let off_control = BoostControl::new(5, 0.1).unwrap();
        let mut off_family = Gaussian;
        let mut summary_family = Gaussian;
        let off = try_fit_with(
            &mut off_learners,
            y.view(),
            w.view(),
            &mut off_family,
            &off_control,
            None,
            false,
        )
        .unwrap();
        let summary = try_fit_profiled(
            &mut summary_learners,
            y.view(),
            w.view(),
            &mut summary_family,
            &off_control,
            None,
            false,
        )
        .unwrap();

        assert_eq!(off.selected, summary.selected);
        assert_eq!(off.coefs, summary.coefs);
        assert_eq!(off.risk, summary.risk);
        assert_eq!(off.offset, summary.offset);
        assert_eq!(off.fitted, summary.fitted);
        assert_eq!(off.nuisance_path, summary.nuisance_path);
        assert_eq!(off.nuisance_vec_path, summary.nuisance_vec_path);
        assert_eq!(off.report.route, EngineRoute::PackedScalar);
        assert!(!off.report.gram_eligible);
        assert!(off
            .report
            .fallback_reasons
            .contains(&FallbackReason::PureScalarModel));
        assert_eq!(off.report.scalar_learner_count, 1);
        assert_eq!(off.report.block_learner_count, 0);
        assert_eq!(off.report.tree_learner_count, 0);
        assert_eq!(off.report.cholesky_solver_count, 0);
        assert_eq!(off.report.lu_solver_count, 0);
        assert_eq!(off.report.regularized_solver_count, 0);
        assert_eq!(off.report.max_solver_jitter, None);
        assert_eq!(off.report.factor_cache_bytes, 0);
        assert_eq!(off.report.gram_cache_bytes, 0);
        assert_eq!(off.report.calibration_count, 0);
        assert_eq!(off.report.total_ns, None);
        assert_eq!(off_control.instrumentation, InstrumentationLevel::Off);
        assert_eq!(
            summary.report.instrumentation,
            InstrumentationLevel::Summary
        );
        assert!(summary.report.setup_ns.is_some());
        assert!(summary.report.iteration_ns.is_some());
        assert!(summary.report.total_ns.is_some());
        let rows = off.path_rows();
        assert_eq!(rows.len(), 5);
        assert_eq!(rows[0].iteration, 1);
        assert_eq!(rows[0].learner_index, off.selected[0]);
        assert_eq!(rows[0].risk, Some(off.risk[1]));
        assert_eq!(rows[0].risk_change, Some(off.risk[0] - off.risk[1]));
        assert!(rows[0].coefficient_norm.is_finite());
    }

    #[test]
    fn seeded_packed_scalar_matches_reference_general() {
        // The tolerance covers only alternate floating-point summation order:
        // both routes solve the same scalar normal equations.
        const TOLERANCE: f64 = 2e-11;
        for seed in 1_u64..=12 {
            let n = 3 + (seed as usize % 29);
            let mstop = 1 + (seed as u32 % 19);
            let nu = [0.01, 0.1, 0.5, 1.0][seed as usize % 4];
            let mut state = seed;
            let mut random = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as f64 / u64::MAX as f64
            };
            let mut x0 = Array1::from_shape_fn(n, |_| 2.0 * random() - 1.0);
            x0[0] = -1.0;
            x0[n - 1] = 1.0;
            let x1 = x0.clone(); // exact tie: lowest learner index must win
            let x2 = Array1::from_elem(n, 0.25); // rank-deficient after centering
            let x3 = Array1::from_shape_fn(n, |_| 2.0 * random() - 1.0);
            let y = Array1::from_shape_fn(n, |index| {
                1.7 * x0[index] - 0.4 * x3[index] + 0.05 * (random() - 0.5)
            });
            let mut weights = Array1::from_shape_fn(n, |_| 0.1 + 1.9 * random());
            if seed % 2 == 0 {
                weights[0] = 0.0;
            }
            let build = || {
                [&x0, &x1, &x2, &x3]
                    .into_iter()
                    .enumerate()
                    .map(|(index, x)| {
                        prepare_scalar_bols(
                            x.view(),
                            weights.view(),
                            0.0,
                            true,
                            &format!("x{index}"),
                            None,
                        )
                    })
                    .collect::<Vec<_>>()
            };
            let control = BoostControl::new(mstop, nu).unwrap();
            let mut reference_learners = build();
            let mut reference_family = Gaussian;
            let reference = try_fit_with_engine_policy(
                &mut reference_learners,
                y.view(),
                weights.view(),
                &mut reference_family,
                &control,
                None,
                false,
                EnginePolicy::ReferenceGeneral,
            )
            .unwrap();
            let mut packed_learners = build();
            let mut packed_family = Gaussian;
            let packed = try_fit_with_engine_policy(
                &mut packed_learners,
                y.view(),
                weights.view(),
                &mut packed_family,
                &control,
                None,
                false,
                EnginePolicy::ForcePackedScalar,
            )
            .unwrap();
            let specification = format!("n={n},mstop={mstop},nu={nu}");
            assert_eq!(reference.report.route, EngineRoute::General);
            assert_eq!(packed.report.route, EngineRoute::PackedScalar);
            assert_differential_paths_close(&reference, &packed, seed, &specification, TOLERANCE);
        }
    }

    #[test]
    fn seeded_parallel_block_matches_sequential_block() {
        // Block solves are identical; the tolerance allows BLAS reduction
        // order differences between sequential and rayon execution.
        const TOLERANCE: f64 = 2e-10;
        for seed in 101_u64..=106 {
            let n = 24 + (seed as usize % 17);
            let mstop = 4 + (seed as u32 % 13);
            let nu = [0.05, 0.1, 0.3][seed as usize % 3];
            let mut state = seed;
            let mut random = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as f64 / u64::MAX as f64
            };
            let mut x0 = Array1::from_shape_fn(n, |_| random());
            x0[0] = 0.0;
            x0[n - 1] = 1.0; // exercise both spline boundary knots
            let x1 = x0.clone(); // exact tied smoothers
            let x2 = Array1::from_shape_fn(n, |_| random());
            let x3 = Array1::from_shape_fn(n, |_| random());
            let y = Array1::from_shape_fn(n, |index| {
                (std::f64::consts::TAU * x0[index]).sin() + x2[index] + 0.02 * (random() - 0.5)
            });
            let weights = Array1::from_shape_fn(n, |_| 0.2 + 1.8 * random());
            let build = || {
                [&x0, &x1, &x2, &x3]
                    .into_iter()
                    .enumerate()
                    .map(|(index, x)| {
                        prepare_bbs(
                            x.view(),
                            weights.view(),
                            6,
                            3,
                            2,
                            4.0,
                            true,
                            &format!("bbs(x{index})"),
                        )
                    })
                    .collect::<Vec<_>>()
            };
            let control = BoostControl::new(mstop, nu).unwrap();
            let mut sequential_learners = build();
            let mut sequential_family = Gaussian;
            let sequential = try_fit_with_engine_policy(
                &mut sequential_learners,
                y.view(),
                weights.view(),
                &mut sequential_family,
                &control,
                None,
                false,
                EnginePolicy::ForceSequentialBlock,
            )
            .unwrap();
            let mut parallel_learners = build();
            let mut parallel_family = Gaussian;
            let parallel = try_fit_with_engine_policy(
                &mut parallel_learners,
                y.view(),
                weights.view(),
                &mut parallel_family,
                &control,
                None,
                false,
                EnginePolicy::ForceParallelBlock,
            )
            .unwrap();
            let specification = format!("n={n},mstop={mstop},nu={nu}");
            assert_eq!(sequential.report.route, EngineRoute::General);
            assert_eq!(parallel.report.route, EngineRoute::ParallelBlock);
            assert_differential_paths_close(
                &sequential,
                &parallel,
                seed,
                &specification,
                TOLERANCE,
            );
        }
    }

    #[test]
    fn reusing_the_subsample_buffers_draws_the_same_learners() {
        // P3.4 item 3: the component-subsample draw allocated a shuffle
        // buffer and a mask on every boosting step -- `2 * mstop` allocations
        // for any fit that uses it, where nothing about the buffers changes
        // between steps except their contents.
        //
        // Reusing them is only safe if the draw is unchanged, and "unchanged"
        // here means the same learners, not a similar number of them: the
        // seed mixes the base seed with the iteration index, and a buffer
        // that was not cleared between steps would leave the previous
        // iteration's marks in place and quietly widen the candidate set.
        for fraction in [0.1_f64, 0.34, 0.5, 0.9] {
            for n_learners in [1_usize, 2, 7, 40] {
                let control = BoostControl::new(50, 0.1)
                    .unwrap()
                    .with_component_subsampling(fraction, 20260905)
                    .unwrap();
                let k = active_learner_count(&control, n_learners);
                let mut scratch = ComponentSubsampleScratch::with_capacity(n_learners);
                for iteration in 0..50usize {
                    let fresh = active_learner_mask(&control, n_learners, iteration);
                    match (k, fresh) {
                        (None, None) => {}
                        (Some(k), Some(expected)) => {
                            scratch.draw(&control, n_learners, iteration, k);
                            assert_eq!(
                                scratch.mask, expected,
                                "fraction={fraction} n={n_learners} iteration={iteration}"
                            );
                            assert_eq!(
                                scratch.mask.iter().filter(|&&on| on).count(),
                                k,
                                "the reused mask must mark exactly k learners"
                            );
                        }
                        (a, b) => panic!(
                            "the two paths disagree about whether to subsample: \
                             {a:?} against {}",
                            b.is_some()
                        ),
                    }
                }
            }
        }
    }

    #[test]
    fn seeded_hybrid_matches_reference_general() {
        // The route pairs already covered are packed-scalar against general,
        // parallel-block against sequential-block, and Gaussian-Gram against
        // general. `Hybrid` -- a learner set mixing packed scalars with dense
        // blocks -- had none, and it is the route a mixed `bols() + bbs()`
        // formula takes for every non-Gaussian family. P3.2 asks for
        // differential route tests at the module boundaries *before* the
        // monoliths are split, because a split that reordered selection
        // would otherwise be invisible.
        //
        // There is no `ForceHybrid`: `Auto` picks it from the learner mix, so
        // asserting the route came out `Hybrid` pins the routing decision as
        // well as the arithmetic.
        const TOLERANCE: f64 = 5e-11;
        for seed in 201_u64..=206 {
            let n = 30 + (seed as usize % 23);
            let mstop = 5 + (seed as u32 % 11);
            let nu = [0.05, 0.1, 0.3][seed as usize % 3];
            let mut state = seed;
            let mut random = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as f64 / u64::MAX as f64
            };
            let mut x0 = Array1::from_shape_fn(n, |_| random());
            x0[0] = 0.0;
            x0[n - 1] = 1.0;
            let x1 = Array1::from_shape_fn(n, |_| 2.0 * random() - 1.0);
            let x2 = x1.clone(); // an exact tie across the two kinds
            let x3 = Array1::from_shape_fn(n, |_| random());
            let y = Array1::from_shape_fn(n, |index| {
                (std::f64::consts::TAU * x0[index]).sin()
                    + 1.3 * x1[index]
                    + 0.02 * (random() - 0.5)
            });
            let weights = Array1::from_shape_fn(n, |_| 0.2 + 1.8 * random());
            let build = || {
                let mut learners = vec![
                    prepare_bbs(x0.view(), weights.view(), 6, 3, 2, 4.0, true, "bbs(x0)"),
                    prepare_bbs(x3.view(), weights.view(), 6, 3, 2, 4.0, true, "bbs(x3)"),
                ];
                learners.push(prepare_scalar_bols(
                    x1.view(),
                    weights.view(),
                    0.0,
                    true,
                    "x1",
                    None,
                ));
                learners.push(prepare_scalar_bols(
                    x2.view(),
                    weights.view(),
                    0.0,
                    true,
                    "x2",
                    None,
                ));
                learners
            };
            let control = BoostControl::new(mstop, nu).unwrap();

            let mut reference_learners = build();
            let mut reference_family = Huber::new(Some(1.0));
            let reference = try_fit_with_engine_policy(
                &mut reference_learners,
                y.view(),
                weights.view(),
                &mut reference_family,
                &control,
                None,
                false,
                EnginePolicy::ReferenceGeneral,
            )
            .unwrap();

            let mut hybrid_learners = build();
            let mut hybrid_family = Huber::new(Some(1.0));
            let hybrid = try_fit_with_engine_policy(
                &mut hybrid_learners,
                y.view(),
                weights.view(),
                &mut hybrid_family,
                &control,
                None,
                false,
                EnginePolicy::Auto,
            )
            .unwrap();

            let specification = format!("n={n},mstop={mstop},nu={nu}");
            assert_eq!(reference.report.route, EngineRoute::General);
            assert_eq!(
                hybrid.report.route,
                EngineRoute::Hybrid,
                "a mixed scalar/block set under a non-Gaussian family should \
                 route Hybrid; seed={seed} spec={specification}"
            );
            assert_differential_paths_close(&reference, &hybrid, seed, &specification, TOLERANCE);
        }
    }

    #[test]
    fn candidate_scan_merge_is_partition_independent() {
        // Enhancement-hunt 2.2 asked whether `winning_multiplicity` is
        // reduction-safe: Rayon merges partial accumulators over an arbitrary
        // binary tree, so if `merge_candidate_scans` disagreed with the serial
        // fold for some tree shape, the reported `tie_selection_count` would
        // depend on how the work was partitioned.
        //
        // Enumerate every binary tree shape over each score vector by
        // recursively splitting at every position, and require all of them to
        // equal the serial fold `observe_candidate` performs — index, score,
        // and multiplicity.
        fn serial(scores: &[f64]) -> (usize, f64, usize) {
            let mut best_index = 0usize;
            let mut best_sse = f64::INFINITY;
            let mut multiplicity = 0usize;
            for (index, &sse) in scores.iter().enumerate() {
                observe_candidate(
                    index,
                    sse,
                    &mut best_index,
                    &mut best_sse,
                    &mut multiplicity,
                );
            }
            (best_index, best_sse, multiplicity)
        }

        fn all_tree_shapes(leaves: &[(usize, f64, usize)]) -> Vec<(usize, f64, usize)> {
            if leaves.len() == 1 {
                return vec![leaves[0]];
            }
            let mut results = Vec::new();
            for split in 1..leaves.len() {
                for left in all_tree_shapes(&leaves[..split]) {
                    for right in all_tree_shapes(&leaves[split..]) {
                        results.push(merge_candidate_scans(left, right));
                    }
                }
            }
            results
        }

        let inf = f64::INFINITY;
        let nan = f64::NAN;
        let cases: Vec<Vec<f64>> = vec![
            vec![5.0, 5.0, 7.0],
            vec![7.0, 5.0, 5.0],
            vec![5.0, 7.0, 5.0],
            vec![3.0, 3.0, 3.0],
            vec![3.0, 5.0, 3.0, 5.0],
            vec![5.0, 3.0, 3.0, 5.0, 3.0],
            vec![inf, inf, inf],
            vec![inf, 4.0, inf, 4.0],
            vec![nan, 2.0, 2.0],
            vec![2.0, nan, 2.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![5.0, 4.0, 3.0, 2.0, 1.0],
        ];

        for scores in cases {
            let expected = serial(&scores);
            // Rayon may fold the identity in at any point, so include it.
            let mut leaves: Vec<(usize, f64, usize)> = vec![CANDIDATE_SCAN_IDENTITY];
            leaves.extend(
                scores
                    .iter()
                    .enumerate()
                    .map(|(index, &sse)| (index, sse, usize::from(sse.is_finite()))),
            );
            for actual in all_tree_shapes(&leaves) {
                assert_eq!(
                    actual.0, expected.0,
                    "index differs for {scores:?}: {actual:?} vs {expected:?}"
                );
                assert_eq!(
                    actual.2, expected.2,
                    "multiplicity differs for {scores:?}: {actual:?} vs {expected:?}"
                );
                assert!(
                    (actual.1 == expected.1) || (actual.1.is_nan() && expected.1.is_nan()),
                    "score differs for {scores:?}: {actual:?} vs {expected:?}"
                );
            }
        }
    }

    #[test]
    fn parallel_block_ties_match_sequential_across_learner_counts() {
        let n = 64;
        let x = Array1::linspace(-1.0, 1.0, n);
        let design = Array2::from_shape_fn((n, 3), |(row, column)| match column {
            0 => x[row],
            1 => x[row] * x[row],
            _ => x[row] * x[row] * x[row],
        });
        let y = x.mapv(|value| 1.2 * value - 0.7 * value * value);
        let weights = Array1::ones(n);
        let penalty = Array2::eye(3) * 0.01;
        let control = BoostControl::new(3, 0.1).unwrap();

        for learner_count in [1, 2, 4, 10, 50] {
            let build = || {
                (0..learner_count)
                    .map(|index| {
                        prepare_block_bols(
                            design.clone(),
                            weights.view(),
                            penalty.clone(),
                            false,
                            &format!("block{index}"),
                        )
                    })
                    .collect::<Vec<_>>()
            };
            let mut sequential_learners = build();
            let sequential = try_fit_with_engine_policy(
                &mut sequential_learners,
                y.view(),
                weights.view(),
                &mut Gaussian,
                &control,
                None,
                false,
                EnginePolicy::ForceSequentialBlock,
            )
            .unwrap();
            let mut parallel_learners = build();
            let parallel = try_fit_with_engine_policy(
                &mut parallel_learners,
                y.view(),
                weights.view(),
                &mut Gaussian,
                &control,
                None,
                false,
                EnginePolicy::ForceParallelBlock,
            )
            .unwrap();

            assert_eq!(sequential.selected, vec![0; 3], "J={learner_count}");
            assert_differential_paths_close(
                &sequential,
                &parallel,
                learner_count as u64,
                &format!("J={learner_count},exact_tie"),
                1e-12,
            );
            let expected_ties = if learner_count > 1 { 3 } else { 0 };
            assert_eq!(parallel.report.tie_selection_count, expected_ties);
        }
    }

    #[test]
    fn execution_report_counts_only_active_family_calibrations() {
        use crate::families::Huber;

        let y = array![1.0, 2.0, 4.0, 3.0, 5.0, 7.0];
        let w = Array1::ones(y.len());
        let x = Array1::linspace(-1.0, 1.0, y.len());
        let control = BoostControl::new(4, 0.1).unwrap();

        let mut adaptive_learners = vec![prepare_scalar_bols(
            x.view(),
            w.view(),
            0.0,
            true,
            "x",
            None,
        )];
        let mut adaptive_family = Huber::new(None);
        let adaptive = try_fit_with(
            &mut adaptive_learners,
            y.view(),
            w.view(),
            &mut adaptive_family,
            &control,
            None,
            false,
        )
        .unwrap();
        assert_eq!(adaptive.report.calibration_count, 5);

        let mut fixed_learners = vec![prepare_scalar_bols(
            x.view(),
            w.view(),
            0.0,
            true,
            "x",
            None,
        )];
        let mut fixed_family = Huber::new(Some(1.5));
        let fixed = try_fit_with(
            &mut fixed_learners,
            y.view(),
            w.view(),
            &mut fixed_family,
            &control,
            None,
            false,
        )
        .unwrap();
        assert_eq!(fixed.report.calibration_count, 0);
    }

    #[test]
    fn execution_report_tracks_propodds_calibration_stops() {
        use crate::families::PropOdds;

        let y = array![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 1.0, 2.0, 3.0];
        let w = Array1::ones(y.len());
        let x = Array1::linspace(-1.0, 1.0, y.len());
        let mut learners = vec![prepare_scalar_bols(
            x.view(),
            w.view(),
            0.0,
            true,
            "x",
            None,
        )];
        let mut family = PropOdds::new(3);
        let control = BoostControl::new(3, 0.1).unwrap();
        let path = try_fit_with(
            &mut learners,
            y.view(),
            w.view(),
            &mut family,
            &control,
            None,
            false,
        )
        .unwrap();

        assert_eq!(path.report.calibration_count, 4);
        assert_eq!(
            path.report.calibration_converged_count
                + path.report.calibration_limit_count
                + path.report.calibration_failure_count,
            path.report.calibration_count
        );
        assert_eq!(path.report.calibration_failure_count, 0);
    }

    #[test]
    fn execution_report_identifies_gaussian_gram_route() {
        use crate::report::{EngineRoute, InstrumentationLevel};

        let n = 20;
        let x = Array1::linspace(-1.0, 1.0, n);
        let y = x.mapv(|value| value * value + 0.5 * value);
        let w = Array1::ones(n);
        let design =
            Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { x[i] } else { x[i] * x[i] });
        let block = prepare_block_bols(design, w.view(), Array2::zeros((2, 2)), true, "block");
        let mut learners = vec![block];
        let mut family = Gaussian;
        let control = BoostControl::new(5, 0.1)
            .unwrap()
            .with_instrumentation(InstrumentationLevel::Summary);
        let path = try_fit_with(
            &mut learners,
            y.view(),
            w.view(),
            &mut family,
            &control,
            None,
            false,
        )
        .unwrap();

        assert_eq!(path.report.route, EngineRoute::GaussianGram);
        assert!(path.report.gram_eligible);
        assert!(path.report.fallback_reasons.is_empty());
        assert_eq!(path.report.total_basis_columns, 2);
        assert_eq!(path.report.scalar_learner_count, 0);
        assert_eq!(path.report.block_learner_count, 1);
        assert_eq!(path.report.cholesky_solver_count, 1);
        assert_eq!(path.report.lu_solver_count, 0);
        assert_eq!(path.report.regularized_solver_count, 0);
        assert_eq!(path.report.max_solver_jitter, None);
        assert_eq!(path.report.factor_cache_bytes, 2 * 2 * 8);
        assert_eq!(path.report.gram_cache_bytes, 2 * 2 * 8);
        assert!(path.report.setup_ns.is_some());
        assert!(path.report.cross_gram_ns.is_some());
        assert!(path.report.iteration_ns.is_some());
        assert!(path.report.reconstruction_ns.is_some());
        assert!(path.report.total_ns.is_some());
    }

    #[test]
    fn gaussian_gram_fit_never_builds_block_sparse_copies() {
        // #738: the Gram route works from the dense designs and the
        // cross-Grams only, so it must not trigger the lazy sparse-copy
        // build — otherwise an exact Gaussian fit retains about 46 MB per
        // bbs term at 10^6 rows for an index it never reads.
        use crate::learners::{prepare_bbs, PreparedLearner, SparseRowsCache};

        fn cache_of(learner: &PreparedLearner) -> &SparseRowsCache {
            match learner {
                PreparedLearner::BlockBols { sparse, .. } => sparse
                    .as_ref()
                    .expect("bbs design must carry a sparse cache"),
                _ => panic!("bbs must prepare as BlockBols"),
            }
        }

        let n = 3000;
        let x = Array1::from_shape_fn(n, |i| i as f64 / n as f64);
        let y = x.mapv(|v| (4.0 * v).sin() + v * v);
        let w = Array1::ones(n);
        let mut learners = vec![
            prepare_bbs(x.view(), w.view(), 20, 3, 2, 4.0, false, "bbs(x)"),
            prepare_bbs(x.view(), w.view(), 20, 3, 2, 4.0, false, "bbs(x) twin"),
        ];
        let control = BoostControl::new(10, 0.1).unwrap();
        try_fit_with_engine_policy(
            &mut learners,
            y.view(),
            w.view(),
            &mut Gaussian,
            &control,
            None,
            false,
            EnginePolicy::ForceGaussianGram,
        )
        .unwrap();
        for learner in &learners {
            assert!(
                !cache_of(learner).is_built(),
                "the Gram route must not build the sparse copy"
            );
        }
        // Prediction after the fit is still served by the copy, built on
        // that first use.
        let p = match &learners[0] {
            PreparedLearner::BlockBols { design, .. } => design.ncols(),
            _ => unreachable!(),
        };
        let mut out = vec![0.0; n];
        learners[0].predict_into(&vec![0.0; p], &mut out);
        assert!(
            cache_of(&learners[0]).is_built(),
            "prediction must build the sparse copy on first use"
        );
    }

    #[cfg(feature = "instrumentation-baseline")]
    #[test]
    fn no_report_benchmark_baseline_preserves_gram_fit() {
        let n = 30;
        let x = Array1::linspace(-1.0, 1.0, n);
        let y = x.mapv(|value| value * value + 0.5 * value);
        let w = Array1::ones(n);
        let build = || {
            let design =
                Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { x[i] } else { x[i] * x[i] });
            vec![prepare_block_bols(
                design,
                w.view(),
                Array2::zeros((2, 2)),
                true,
                "block",
            )]
        };
        let control = BoostControl::new(10, 0.1).unwrap();
        let mut reported_learners = build();
        let reported = try_fit_with(
            &mut reported_learners,
            y.view(),
            w.view(),
            &mut Gaussian,
            &control,
            None,
            false,
        )
        .unwrap();
        let mut baseline_learners = build();
        let baseline = try_fit_without_report(
            &mut baseline_learners,
            y.view(),
            w.view(),
            &mut Gaussian,
            &control,
            None,
        )
        .unwrap();

        assert_eq!(reported.selected, baseline.selected);
        assert_eq!(reported.coefs, baseline.coefs);
        assert_eq!(reported.risk, baseline.risk);
        assert_eq!(reported.fitted, baseline.fitted);
        assert_eq!(baseline.report.route, EngineRoute::Unknown);
    }

    #[test]
    fn execution_report_exposes_regularized_block_solver() {
        let n = 20;
        let x = Array1::linspace(-1.0, 1.0, n);
        let y = x.mapv(|value| 0.5 * value);
        let w = Array1::ones(n);
        // Rank-deficient by an exactly-zero second column, not by duplicating the
        // first. Duplicate columns make the Gram singular only up to accumulation
        // order: on a target with FMA (for example a local target-cpu=native build
        // with on Linux) the duplicated entries can differ by one ULP, leaving the
        // matrix merely ill-conditioned, so plain Cholesky succeeds with a tiny
        // pivot and the regularized route is never reached. A zero column puts an
        // exact zero on the diagonal, which no rounding mode can perturb, so the
        // fallback triggers identically on every platform.
        let design = Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { x[i] } else { 0.0 });
        let block = prepare_block_bols(design, w.view(), Array2::zeros((2, 2)), false, "rank1");
        let mut learners = vec![block];
        let control = BoostControl::new(3, 0.1).unwrap();
        let path = try_fit_with(
            &mut learners,
            y.view(),
            w.view(),
            &mut Gaussian,
            &control,
            None,
            false,
        )
        .unwrap();

        assert_eq!(path.report.regularized_solver_count, 1);
        assert_eq!(path.report.cholesky_solver_count, 0);
        assert_eq!(path.report.lu_solver_count, 0);
        assert!(path
            .report
            .max_solver_jitter
            .is_some_and(|value| value > 0.0));
        assert!(path.fitted.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn try_fit_y_weights_mismatch_returns_err() {
        let x = array![1.0, 2.0, 3.0];
        let w0 = array![0.0, 0.0, 0.0];
        let mut learners = vec![prepare_scalar_bols(
            x.view(),
            w0.view(),
            0.0,
            true,
            "x",
            None,
        )];
        let y = array![1.0, 2.0, 3.0];
        let w = array![1.0, 1.0]; // wrong length
        let mut fam = Gaussian;
        let ctrl = BoostControl::new(5, 0.1).unwrap();
        let result = try_fit_with(
            &mut learners,
            y.view(),
            w.view(),
            &mut fam,
            &ctrl,
            None,
            true,
        );
        assert!(matches!(result, Err(MboostrsError::Shape(_))));
    }

    #[test]
    fn try_fit_rejects_nonfinite_response_and_weights() {
        let x = array![1.0, 2.0, 3.0];
        let setup_w = array![1.0, 1.0, 1.0];
        let ctrl = BoostControl::new(5, 0.1).unwrap();

        for (y, w) in [
            (array![1.0, f64::NAN, 3.0], array![1.0, 1.0, 1.0]),
            (array![1.0, 2.0, 3.0], array![1.0, f64::INFINITY, 1.0]),
        ] {
            let mut learners = vec![prepare_scalar_bols(
                x.view(),
                setup_w.view(),
                0.0,
                true,
                "x",
                None,
            )];
            let result = try_fit_with(
                &mut learners,
                y.view(),
                w.view(),
                &mut Gaussian,
                &ctrl,
                None,
                true,
            );
            assert!(result.is_err());
        }
    }

    #[test]
    fn try_fit_rejects_negative_weights_nonfinite_offset_and_bad_learner_rows() {
        let x = array![1.0, 2.0, 3.0];
        let setup_w = array![1.0, 1.0, 1.0];
        let y = array![1.0, 2.0, 3.0];
        let ctrl = BoostControl::new(5, 0.1).unwrap();

        let mut learners = vec![prepare_scalar_bols(
            x.view(),
            setup_w.view(),
            0.0,
            true,
            "x",
            None,
        )];
        let negative_w = array![1.0, -1.0, 1.0];
        assert!(try_fit_with(
            &mut learners,
            y.view(),
            negative_w.view(),
            &mut Gaussian,
            &ctrl,
            None,
            true,
        )
        .is_err());

        assert!(try_fit_with(
            &mut learners,
            y.view(),
            setup_w.view(),
            &mut Gaussian,
            &ctrl,
            Some(f64::NAN),
            true,
        )
        .is_err());

        let short_x = array![1.0, 2.0];
        let short_w = array![1.0, 1.0];
        let mut short_learner = vec![prepare_scalar_bols(
            short_x.view(),
            short_w.view(),
            0.0,
            true,
            "short",
            None,
        )];
        assert!(matches!(
            try_fit_with(
                &mut short_learner,
                y.view(),
                setup_w.view(),
                &mut Gaussian,
                &ctrl,
                None,
                true,
            ),
            Err(MboostrsError::LearnerSetup(_, _))
        ));
    }

    #[test]
    fn try_fit_rejects_empty_response() {
        let x = array![1.0];
        let setup_w = array![1.0];
        let mut learners = vec![prepare_scalar_bols(
            x.view(),
            setup_w.view(),
            0.0,
            true,
            "x",
            None,
        )];
        let y = Array1::<f64>::zeros(0);
        let w = Array1::<f64>::zeros(0);
        let ctrl = BoostControl::new(5, 0.1).unwrap();
        assert!(try_fit_with(
            &mut learners,
            y.view(),
            w.view(),
            &mut Gaussian,
            &ctrl,
            None,
            true,
        )
        .is_err());
    }

    #[test]
    fn hybrid_parallel_block_ties_choose_lowest_index() {
        let n = 24;
        let y = Array1::from_iter((0..n).map(|i| (i as f64 + 1.0) / n as f64));
        let w = Array1::ones(n);
        let zero_scalar = Array1::zeros(n);
        let block_design = Array2::from_shape_vec((n, 1), y.to_vec()).unwrap();
        let penalty = Array2::zeros((1, 1));

        let mut learners = vec![prepare_scalar_bols(
            zero_scalar.view(),
            w.view(),
            0.0,
            false,
            "z",
            None,
        )];
        for idx in 0..4 {
            learners.push(prepare_block_bols(
                block_design.clone(),
                w.view(),
                penalty.clone(),
                false,
                &format!("b{idx}"),
            ));
        }

        let mut fam = Gaussian;
        let ctrl = BoostControl::new(1, 0.1).unwrap();
        let path = fit_with(
            &mut learners,
            y.view(),
            w.view(),
            &mut fam,
            &ctrl,
            None,
            false,
        );

        assert_eq!(path.selected, vec![1]);
        assert_eq!(path.report.tie_selection_count, 1);
    }

    #[test]
    fn sparse_gram_products_keep_exact_support_and_matrix_layout() {
        use ndarray::ShapeBuilder;
        for fortran in [false, true] {
            let mut matrix = if fortran {
                Array2::zeros((80, 90).f())
            } else {
                Array2::zeros((80, 90))
            };
            for i in 0..79 {
                matrix[[i, i]] = 2.0;
                matrix[[i, i + 3]] = -0.25;
            }
            // A completely empty row and a tiny remote entry exercise CSR
            // row offsets and ensure no numerical sparsification occurs.
            matrix[[0, 89]] = 1e-100;
            let product = GramMatVec::new(&matrix);
            assert!(product.sparse.is_some());
            let mut rhs = Array1::from_shape_fn(90, |j| (j as f64).sin());
            let mut actual = Array1::zeros(80);
            product.mul_into(&rhs, actual.view_mut());
            let expected = matrix.dot(&rhs);
            assert!((&actual - &expected).iter().all(|v| v.abs() < 1e-14));
            assert_eq!(actual[79], 0.0);
            rhs.fill(0.0);
            rhs[89] = 1.0;
            product.mul_into(&rhs, actual.view_mut());
            assert_eq!(actual[0], 1e-100);
        }
        let dense = Array2::from_elem((80, 90), 1.0);
        assert!(GramMatVec::new(&dense).sparse.is_none());
        assert!(GramMatVec::new(&Array2::eye(24)).sparse.is_none());
    }

    #[test]
    fn gaussian_gram_spatial_matches_general_with_weights_centering_and_by() {
        use crate::learners::{prepare_bbs, prepare_bspatial};
        let n = 240;
        let x = Array1::from_shape_fn(n, |i| ((i * 37) % 241) as f64 / 241.0);
        let z = Array1::from_shape_fn(n, |i| ((i * 83) % 239) as f64 / 239.0);
        let by = Array1::from_shape_fn(n, |i| 0.5 + x[i]);
        let y = Array1::from_shape_fn(n, |i| (6.0 * x[i]).sin() * (5.0 * z[i]).cos() + x[i]);
        let weights = Array1::from_shape_fn(n, |i| if i % 7 == 0 { 0.0 } else { 0.5 + z[i] });
        for center in [false, true] {
            for nu in [0.1, 0.5, 1.0] {
                for lambda in [None, Some(0.7)] {
                    let spatial = prepare_bspatial(
                        x.view(),
                        z.view(),
                        weights.view(),
                        5,
                        3,
                        2,
                        6.0,
                        lambda,
                        center,
                        "spatial",
                        Some(by.view()),
                    );
                    let smooth = prepare_bbs(x.view(), weights.view(), 8, 3, 2, 4.0, true, "x");
                    let learners = vec![spatial, smooth];
                    let control = BoostControl::new(200, nu).unwrap();
                    let run = |policy| {
                        try_fit_with_engine_policy(
                            &mut learners.clone(),
                            y.view(),
                            weights.view(),
                            &mut Gaussian,
                            &control,
                            None,
                            false,
                            policy,
                        )
                        .unwrap()
                    };
                    let reference = run(EnginePolicy::ReferenceGeneral);
                    let optimized = run(EnginePolicy::ForceGaussianGram);
                    assert_differential_paths_close(
                        &reference,
                        &optimized,
                        270,
                        &format!(
                            "spatial center={center}, nu={nu}, lambda={lambda:?}, weighted by"
                        ),
                        1e-8,
                    );
                }
            }
        }
    }

    #[test]
    fn gaussian_gram_sparse_twins_stay_tied() {
        use crate::learners::prepare_bspatial;
        let n = 180;
        let x = Array1::from_shape_fn(n, |i| ((i * 37) % 181) as f64 / 181.0);
        let z = Array1::from_shape_fn(n, |i| ((i * 83) % 179) as f64 / 179.0);
        let y = Array1::from_shape_fn(n, |i| (6.0 * x[i]).sin() * (5.0 * z[i]).cos());
        let weights = Array1::ones(n);
        let spatial = prepare_bspatial(
            x.view(),
            z.view(),
            weights.view(),
            8,
            3,
            2,
            6.0,
            Some(1.0),
            false,
            "spatial",
            None,
        );
        let mut learners = vec![spatial.clone(), spatial];
        let control = BoostControl::new(1200, 0.5).unwrap();
        let path = try_fit_with_engine_policy(
            &mut learners,
            y.view(),
            weights.view(),
            &mut Gaussian,
            &control,
            None,
            false,
            EnginePolicy::ForceGaussianGram,
        )
        .unwrap();
        assert!(path.selected.iter().all(|&j| j == 0));
    }

    #[test]
    fn gaussian_gram_duplicate_designs_share_blocks_and_stay_tied() {
        // Three copies of one bspatial learner — two clones (distinct
        // allocations, so the elementwise comparison path runs) and the
        // original. The Gram route must give all three bit-identical
        // blocks, so the fit is bit-identical to the two-learner fit and
        // the first learner wins every tie.
        use crate::learners::prepare_bspatial;
        let n = 180;
        let x = Array1::from_shape_fn(n, |i| ((i * 37) % 181) as f64 / 181.0);
        let z = Array1::from_shape_fn(n, |i| ((i * 83) % 179) as f64 / 179.0);
        let y = Array1::from_shape_fn(n, |i| (6.0 * x[i]).sin() * (5.0 * z[i]).cos());
        let weights = Array1::ones(n);
        let spatial = prepare_bspatial(
            x.view(),
            z.view(),
            weights.view(),
            8,
            3,
            2,
            6.0,
            Some(1.0),
            false,
            "spatial",
            None,
        );
        let control = BoostControl::new(1200, 0.5).unwrap();
        let run = |learners: &mut [PreparedLearner]| {
            try_fit_with_engine_policy(
                learners,
                y.view(),
                weights.view(),
                &mut Gaussian,
                &control,
                None,
                false,
                EnginePolicy::ForceGaussianGram,
            )
            .unwrap()
        };
        let pair = run(&mut [spatial.clone(), spatial.clone()]);
        let trio = run(&mut [spatial.clone(), spatial.clone(), spatial]);
        assert!(trio.selected.iter().all(|&j| j == 0));
        assert_eq!(pair.selected, trio.selected);
        assert_eq!(pair.coefs, trio.coefs);
        assert!(pair
            .fitted
            .iter()
            .zip(trio.fitted.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits()));

        // Same design, different penalties: here the duplicate is not a
        // tie partner — its weaker penalty should win selections the
        // general path also gives it. This exercises the filled blocks
        // in the duplicate's own row and column (the source's diagonal
        // and the transposed cross blocks), which the tie scenario above
        // never touches because its duplicates are never selected.
        use crate::learners::prepare_bbs;
        let xs1 = Array1::from_shape_fn(n, |i| ((i * 53) % 181) as f64 / 181.0);
        let ys = Array1::from_shape_fn(n, |i| (6.0 * xs1[i]).sin());
        let build = || {
            vec![
                prepare_bbs(xs1.view(), weights.view(), 20, 3, 2, 0.05, true, "bbs(x)"),
                prepare_bbs(xs1.view(), weights.view(), 20, 3, 2, 50.0, true, "bbs(x)"),
            ]
        };
        let ctrl = BoostControl::new(120, 0.1).unwrap();
        let mut lg = build();
        let general = try_fit_with_engine_policy(
            &mut lg,
            ys.view(),
            weights.view(),
            &mut Gaussian,
            &ctrl,
            None,
            false,
            EnginePolicy::ReferenceGeneral,
        )
        .unwrap();
        let mut lk = build();
        let gram = try_fit_with_engine_policy(
            &mut lk,
            ys.view(),
            weights.view(),
            &mut Gaussian,
            &ctrl,
            None,
            false,
            EnginePolicy::ForceGaussianGram,
        )
        .unwrap();
        assert_eq!(gram.report.route, EngineRoute::GaussianGram);
        assert!(gram.selected.contains(&1));
        assert_eq!(general.selected, gram.selected);
        let mut max_coef_d = 0.0f64;
        for (a, b) in general.coefs.iter().zip(&gram.coefs) {
            for k in 0..a.len().min(b.len()) {
                max_coef_d = max_coef_d.max((a[k] - b[k]).abs());
            }
        }
        assert!(max_coef_d < 1e-8, "coef mismatch: {max_coef_d:.3e}");
    }

    #[test]
    fn gaussian_gram_matches_general_path() {
        // Explicit policies keep route selection independent from path
        // storage choices.
        use crate::control::RiskKind;
        use crate::learners::prepare_bbs;

        let n = 300usize;
        let mut rng = 7u64;
        let mut xs = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng as f64) / (u64::MAX as f64)
        };
        let x1 = Array1::from_shape_fn(n, |_| xs());
        let x2 = Array1::from_shape_fn(n, |_| xs());
        let tau = std::f64::consts::TAU;
        let y = Array1::from_shape_fn(n, |i| {
            (tau * x1[i]).sin() + (tau * x2[i]).cos() + (xs() - 0.5) * 0.3
        });
        let ctrl = BoostControl::new(120, 0.1)
            .unwrap()
            .with_risk(RiskKind::Inbag);

        // Run with uniform and with non-uniform observation weights.
        let w_unit = Array1::ones(n);
        let w_rand = Array1::from_shape_fn(n, |_| 0.2 + 1.8 * xs());
        for w in [w_unit, w_rand] {
            let build = || {
                vec![
                    prepare_bbs(x1.view(), w.view(), 20, 3, 2, 4.0, true, "bbs(x1)"),
                    prepare_bbs(x2.view(), w.view(), 20, 3, 2, 4.0, true, "bbs(x2)"),
                ]
            };

            let mut lg = build();
            let mut fam_g = Gaussian;
            let general = try_fit_with_engine_policy(
                &mut lg,
                y.view(),
                w.view(),
                &mut fam_g,
                &ctrl,
                None,
                false,
                EnginePolicy::ReferenceGeneral,
            )
            .unwrap();

            let mut lk = build();
            let mut fam_k = Gaussian;
            let gram = try_fit_with_engine_policy(
                &mut lk,
                y.view(),
                w.view(),
                &mut fam_k,
                &ctrl,
                None,
                false,
                EnginePolicy::ForceGaussianGram,
            )
            .unwrap();

            assert_eq!(general.report.route, EngineRoute::General);
            assert_eq!(gram.report.route, EngineRoute::GaussianGram);
            assert!(general.increments.is_empty());
            assert!(gram.increments.is_empty());

            assert_eq!(
                general.selected, gram.selected,
                "selected sequence diverged; seed=7 spec=n=300,mstop=120,nu=0.1"
            );

            let mut max_coef_d = 0.0f64;
            for (a, b) in general.coefs.iter().zip(&gram.coefs) {
                for k in 0..a.len().min(b.len()) {
                    max_coef_d = max_coef_d.max((a[k] - b[k]).abs());
                }
            }
            assert!(
                max_coef_d < 1e-8,
                "coef mismatch: {max_coef_d:.3e}; seed=7 spec=n=300,mstop=120,nu=0.1"
            );

            let mut max_risk_d = 0.0f64;
            for (a, b) in general.risk.iter().zip(&gram.risk) {
                max_risk_d = max_risk_d.max((a - b).abs());
            }
            // risk magnitude scales with sum(w); use a relative bar.
            let risk0 = general.risk[0].abs().max(1.0);
            assert!(
                max_risk_d / risk0 < 1e-9,
                "risk mismatch: {max_risk_d:.3e}; seed=7 spec=n=300,mstop=120,nu=0.1"
            );

            let mut max_fit_d = 0.0f64;
            for i in 0..n {
                max_fit_d = max_fit_d.max((general.fitted[i] - gram.fitted[i]).abs());
            }
            assert!(
                max_fit_d < 1e-8,
                "fitted mismatch: {max_fit_d:.3e}; seed=7 spec=n=300,mstop=120,nu=0.1"
            );
        }
    }

    #[test]
    fn gaussian_gram_mixed_scalar_and_block() {
        // Mix of scalar bols + bbs learners. Exercises the ScalarBols
        // extraction branch of try_fit_gaussian_gram (1-column design, scalar
        // solve/penalty), which the all-bbs parity test does not cover.
        use crate::control::RiskKind;
        use crate::learners::{prepare_bbs, prepare_scalar_bols};

        let n = 300usize;
        let mut rng = 99u64;
        let mut xs = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng as f64) / (u64::MAX as f64)
        };
        let x1 = Array1::from_shape_fn(n, |_| xs());
        let x2 = Array1::from_shape_fn(n, |_| xs());
        let x3 = Array1::from_shape_fn(n, |_| xs());
        let tau = std::f64::consts::TAU;
        let y = Array1::from_shape_fn(n, |i| {
            (tau * x1[i]).sin() + 2.0 * x2[i] + (tau * x3[i]).cos() + (xs() - 0.5) * 0.3
        });
        let w = Array1::ones(n);
        let ctrl = BoostControl::new(100, 0.1)
            .unwrap()
            .with_risk(RiskKind::Inbag);

        let build = || {
            vec![
                prepare_scalar_bols(x2.view(), w.view(), 0.0, true, "x2", None),
                prepare_bbs(x1.view(), w.view(), 20, 3, 2, 4.0, true, "bbs(x1)"),
                prepare_bbs(x3.view(), w.view(), 20, 3, 2, 4.0, true, "bbs(x3)"),
            ]
        };

        let mut lg = build();
        let mut fam_g = Gaussian;
        let general = try_fit_with_engine_policy(
            &mut lg,
            y.view(),
            w.view(),
            &mut fam_g,
            &ctrl,
            None,
            false,
            EnginePolicy::ReferenceGeneral,
        )
        .unwrap();

        let mut lk = build();
        let mut fam_k = Gaussian;
        let gram = try_fit_with_engine_policy(
            &mut lk,
            y.view(),
            w.view(),
            &mut fam_k,
            &ctrl,
            None,
            false,
            EnginePolicy::ForceGaussianGram,
        )
        .unwrap();

        assert_eq!(general.report.route, EngineRoute::General);
        assert_eq!(gram.report.route, EngineRoute::GaussianGram);
        assert!(gram.increments.is_empty());
        assert_eq!(
            general.selected, gram.selected,
            "selected sequence diverged; seed=99 spec=n=300,mstop=100,nu=0.1"
        );

        let mut max_coef_d = 0.0f64;
        for (a, b) in general.coefs.iter().zip(&gram.coefs) {
            for k in 0..a.len().min(b.len()) {
                max_coef_d = max_coef_d.max((a[k] - b[k]).abs());
            }
        }
        assert!(
            max_coef_d < 1e-8,
            "coef mismatch: {max_coef_d:.3e}; seed=99 spec=n=300,mstop=100,nu=0.1"
        );

        let mut max_fit_d = 0.0f64;
        for (g, c) in general.fitted.iter().zip(gram.fitted.iter()) {
            max_fit_d = max_fit_d.max((g - c).abs());
        }
        assert!(
            max_fit_d < 1e-8,
            "fitted mismatch: {max_fit_d:.3e}; seed=99 spec=n=300,mstop=100,nu=0.1"
        );
    }

    #[test]
    fn gaussian_gram_twin_learners_stay_tied() {
        // Two learners over bit-identical columns have bit-identical candidate
        // SSEs at every iteration, so the selection criterion is exactly tied
        // and the lowest index must win each time — R's `which.min` behaviour.
        // The Gram path maintains rhsⱼ incrementally through cross-Gram
        // blocks; any asymmetry between the twins' blocks or updates breaks
        // the tie and sends the path to the wrong learner. Found by
        // tests/test_selection_sequence_parity.py (config GAUSS_TIE).
        use crate::control::RiskKind;

        let n = 200usize;
        let mut rng = 11u64;
        let mut xs = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng as f64) / (u64::MAX as f64)
        };
        let x1 = Array1::from_shape_fn(n, |_| 10.0 * xs());
        let y = Array1::from_shape_fn(n, |i| 2.0 * x1[i] + (xs() - 0.5) * 2.0);
        let w = Array1::ones(n);
        let ctrl = BoostControl::new(50, 0.1)
            .unwrap()
            .with_risk(RiskKind::Inbag);

        let build = || {
            vec![
                prepare_bbs(x1.view(), w.view(), 20, 3, 2, 4.0, false, "bbs(x1)"),
                prepare_bbs(x1.view(), w.view(), 20, 3, 2, 4.0, false, "bbs(x2)"),
            ]
        };
        let mut learners = build();
        let mut fam = Gaussian;
        let gram = try_fit_with_engine_policy(
            &mut learners,
            y.view(),
            w.view(),
            &mut fam,
            &ctrl,
            None,
            false,
            EnginePolicy::ForceGaussianGram,
        )
        .unwrap();
        assert_eq!(gram.report.route, EngineRoute::GaussianGram);
        assert!(
            gram.selected.iter().all(|&j| j == 0),
            "twin tie broken: learner 1 first picked at iteration {}",
            gram.selected.iter().position(|&j| j == 1).unwrap_or(0) + 1
        );
    }

    #[test]
    fn gaussian_gram_matches_general_path_with_trees() {
        // Two bbs terms and two trees (one a genuine step), Gaussian. The Gram
        // route fits the trees to an explicit residual beside the projections;
        // it must select the same sequence as the reference general route and
        // agree on risk, fitted values and coefficients.
        use crate::control::RiskKind;
        use crate::learners::{prepare_bbs, prepare_btree};

        let n = 300usize;
        let mut rng = 17u64;
        let mut xs = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng as f64) / (u64::MAX as f64)
        };
        let x1 = Array1::from_shape_fn(n, |_| xs());
        let x2 = Array1::from_shape_fn(n, |_| xs());
        let x3 = Array1::from_shape_fn(n, |_| xs());
        let x4 = Array1::from_shape_fn(n, |_| xs());
        let tau = std::f64::consts::TAU;
        let y = Array1::from_shape_fn(n, |i| {
            (tau * x1[i]).sin()
                + 0.5 * x2[i]
                + if x3[i] > 0.4 { 1.0 } else { 0.0 }
                + (xs() - 0.5) * 0.3
        });
        // Fold-style weights: a fifth of the rows held out.
        let w = Array1::from_shape_fn(n, |i| if i % 5 == 0 { 0.0 } else { 1.0 });
        let ctrl = BoostControl::new(200, 0.1)
            .unwrap()
            .with_risk(RiskKind::Inbag);
        let x3m = x3.clone().insert_axis(ndarray::Axis(1));
        let x4m = x4.clone().insert_axis(ndarray::Axis(1));
        let build = || {
            vec![
                prepare_bbs(x1.view(), w.view(), 20, 3, 2, 4.0, true, "bbs(x1)"),
                prepare_btree(x3m.view(), w.view(), 7, 20, 1, None, "btree(x3)"),
                prepare_bbs(x2.view(), w.view(), 20, 3, 2, 4.0, true, "bbs(x2)"),
                prepare_btree(x4m.view(), w.view(), 7, 20, 1, None, "btree(x4)"),
            ]
        };
        let run = |policy| {
            let mut learners = build();
            let mut family = Gaussian;
            try_fit_with_engine_policy(
                &mut learners,
                y.view(),
                w.view(),
                &mut family,
                &ctrl,
                None,
                false,
                policy,
            )
            .unwrap()
        };
        let general = run(EnginePolicy::ReferenceGeneral);
        let gram = run(EnginePolicy::ForceGaussianGram);
        assert_eq!(general.report.route, EngineRoute::General);
        assert_eq!(gram.report.route, EngineRoute::GaussianGram);
        assert_eq!(
            general.selected, gram.selected,
            "selected sequence diverged"
        );
        assert!(
            gram.selected.contains(&1),
            "the step tree was never selected"
        );
        for (a, b) in general.risk.iter().zip(&gram.risk) {
            assert!((a - b).abs() <= 1e-8 * a.abs().max(1.0), "risk {a} vs {b}");
        }
        for (a, b) in general.fitted.iter().zip(gram.fitted.iter()) {
            assert!((a - b).abs() <= 1e-9, "fitted {a} vs {b}");
        }
        for (a, b) in general.coefs.iter().zip(&gram.coefs) {
            assert_eq!(a.len(), b.len());
            for (u, v) in a.iter().zip(b) {
                assert!((u - v).abs() <= 1e-9 * u.abs().max(1.0), "coef {u} vs {v}");
            }
        }
    }

    #[test]
    fn gaussian_gram_matches_general_path_with_bmono() {
        // B1: 2×bbs + 2×bmono (one constrained), Gaussian. The explicitly
        // forced Gram path must match the reference-general route.
        use crate::control::RiskKind;
        use crate::learners::{prepare_bbs, prepare_bmono, MonoConstraint};

        let n = 300usize;
        let mut rng = 11u64;
        let mut xs = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng as f64) / (u64::MAX as f64)
        };
        let x1 = Array1::from_shape_fn(n, |_| xs());
        let x2 = Array1::from_shape_fn(n, |_| xs());
        let x3 = Array1::from_shape_fn(n, |_| xs());
        // x4: decreasing signal under an Increasing constraint, so the
        // constrained solve actually binds (beta genuinely non-stationary).
        let x4 = Array1::from_shape_fn(n, |i| i as f64 / (n - 1) as f64);
        let tau = std::f64::consts::TAU;
        let y = Array1::from_shape_fn(n, |i| {
            (tau * x1[i]).sin() + (tau * x2[i]).cos() - (2.0 * tau * x4[i]).sin()
                + 0.5 * x3[i]
                + (xs() - 0.5) * 0.3
        });
        let w = Array1::ones(n);
        let ctrl = BoostControl::new(150, 0.1)
            .unwrap()
            .with_risk(RiskKind::Inbag);

        let build = || {
            vec![
                prepare_bbs(x1.view(), w.view(), 20, 3, 2, 4.0, true, "bbs(x1)"),
                prepare_bbs(x2.view(), w.view(), 20, 3, 2, 4.0, true, "bbs(x2)"),
                prepare_bmono(
                    x3.view(),
                    w.view(),
                    20,
                    3,
                    2,
                    4.0,
                    MonoConstraint::Increasing,
                    BmonoSolver::Iterative,
                    1e6,
                    10,
                    "bmono(x3)",
                    None,
                    None,
                    false,
                ),
                prepare_bmono(
                    x4.view(),
                    w.view(),
                    20,
                    3,
                    2,
                    4.0,
                    MonoConstraint::Increasing,
                    BmonoSolver::Iterative,
                    1e6,
                    10,
                    "bmono(x4)",
                    None,
                    None,
                    false,
                ),
            ]
        };

        let mut lg = build();
        let mut fam_g = Gaussian;
        let general = try_fit_with_engine_policy(
            &mut lg,
            y.view(),
            w.view(),
            &mut fam_g,
            &ctrl,
            None,
            false,
            EnginePolicy::ReferenceGeneral,
        )
        .unwrap();

        let mut lk = build();
        let mut fam_k = Gaussian;
        let gram = try_fit_with_engine_policy(
            &mut lk,
            y.view(),
            w.view(),
            &mut fam_k,
            &ctrl,
            None,
            false,
            EnginePolicy::ForceGaussianGram,
        )
        .unwrap();

        assert_eq!(general.report.route, EngineRoute::General);
        assert_eq!(gram.report.route, EngineRoute::GaussianGram);
        assert!(general.increments.is_empty());
        assert!(gram.increments.is_empty());

        assert_eq!(
            general.selected, gram.selected,
            "selected sequence diverged; seed=11 spec=n=300,mstop=150,nu=0.1"
        );

        let mut max_coef_d = 0.0f64;
        for (a, b) in general.coefs.iter().zip(&gram.coefs) {
            for k in 0..a.len().min(b.len()) {
                max_coef_d = max_coef_d.max((a[k] - b[k]).abs());
            }
        }
        assert!(
            max_coef_d < 1e-8,
            "coef mismatch: {max_coef_d:.3e}; seed=11 spec=n=300,mstop=150,nu=0.1"
        );

        let mut max_risk_d = 0.0f64;
        for (a, b) in general.risk.iter().zip(&gram.risk) {
            max_risk_d = max_risk_d.max((a - b).abs());
        }
        assert!(
            max_risk_d < 1e-8,
            "risk mismatch: {max_risk_d:.3e}; seed=11 spec=n=300,mstop=150,nu=0.1"
        );

        let mut max_fit_d = 0.0f64;
        for (g, c) in general.fitted.iter().zip(gram.fitted.iter()) {
            max_fit_d = max_fit_d.max((g - c).abs());
        }
        assert!(
            max_fit_d < 1e-8,
            "fitted mismatch: {max_fit_d:.3e}; seed=11 spec=n=300,mstop=150,nu=0.1"
        );
    }

    #[test]
    fn gaussian_gram_selection_only_matches_reference_without_unused_paths() {
        use crate::control::RiskKind;
        use crate::learners::{prepare_bmono, MonoConstraint};

        let n = 180usize;
        let x1 = Array1::from_shape_fn(n, |i| i as f64 / (n - 1) as f64);
        let x2 = Array1::from_shape_fn(n, |i| ((i * 47) % n) as f64 / (n - 1) as f64);
        let x3 = Array1::from_shape_fn(n, |i| ((i * 89) % n) as f64 / (n - 1) as f64);
        let y = Array1::from_shape_fn(n, |i| {
            (std::f64::consts::TAU * x1[i]).sin() + 0.8 * x2[i] + 0.4 * x3[i]
        });
        let weights = Array1::from_shape_fn(n, |i| 0.5 + (i % 7) as f64 / 7.0);
        let control = BoostControl::new(75, 0.1)
            .unwrap()
            .with_risk(RiskKind::Inbag);

        let compare = |build: &dyn Fn() -> Vec<PreparedLearner>, label: &str| {
            let mut reference_learners = build();
            let reference = try_fit_selected_with_engine_policy(
                &mut reference_learners,
                y.view(),
                weights.view(),
                &mut Gaussian,
                &control,
                EnginePolicy::ReferenceGeneral,
            )
            .unwrap();
            let mut gram_learners = build();
            let gram = try_fit_selected_with_engine_policy(
                &mut gram_learners,
                y.view(),
                weights.view(),
                &mut Gaussian,
                &control,
                EnginePolicy::ForceGaussianGram,
            )
            .unwrap();

            assert_eq!(reference.report.route, EngineRoute::General, "{label}");
            assert_eq!(gram.report.route, EngineRoute::GaussianGram, "{label}");
            assert_eq!(reference.selected, gram.selected, "{label}");
            for path in [&reference, &gram] {
                assert!(path.coefs.is_empty(), "{label}");
                assert!(path.risk.is_empty(), "{label}");
                assert!(path.fitted.is_empty(), "{label}");
                assert!(path.increments.is_empty(), "{label}");
                assert!(path.nuisance_path.is_empty(), "{label}");
                assert!(path.nuisance_vec_path.is_empty(), "{label}");
                assert!(path.nuisance_vec.is_empty(), "{label}");
            }
        };

        compare(
            &|| {
                vec![
                    prepare_bbs(x1.view(), weights.view(), 12, 3, 2, 4.0, true, "bbs(x1)"),
                    prepare_bbs(x2.view(), weights.view(), 12, 3, 2, 4.0, true, "bbs(x2)"),
                ]
            },
            "block",
        );
        compare(
            &|| {
                vec![
                    prepare_scalar_bols(x2.view(), weights.view(), 0.0, true, "x2", None),
                    prepare_bbs(x1.view(), weights.view(), 12, 3, 2, 4.0, true, "bbs(x1)"),
                ]
            },
            "mixed",
        );
        compare(
            &|| {
                vec![
                    prepare_bbs(x1.view(), weights.view(), 12, 3, 2, 4.0, true, "bbs(x1)"),
                    prepare_bmono(
                        x3.view(),
                        weights.view(),
                        12,
                        3,
                        2,
                        4.0,
                        MonoConstraint::Increasing,
                        BmonoSolver::Iterative,
                        1e6,
                        10,
                        "bmono(x3)",
                        None,
                        None,
                        false,
                    ),
                ]
            },
            "monotone",
        );
    }

    #[test]
    fn fit_gaussian_two_scalar_predictors() {
        // y = [3, 5, 7], x1 = [1, 2, 3], x2 = [3, 1, 2]
        // offset = weighted mean of y = 5.0
        // x1 centered: [-1, 0, 1], x2 centered: [1, -1, 0]
        let y = array![3.0, 5.0, 7.0];
        let w = array![1.0, 1.0, 1.0];
        let x1 = array![1.0, 2.0, 3.0];
        let x2 = array![3.0, 1.0, 2.0];

        let l1 = prepare_scalar_bols(x1.view(), w.view(), 0.0, true, "x1", None);
        let l2 = prepare_scalar_bols(x2.view(), w.view(), 0.0, true, "x2", None);
        let mut learners = vec![l1, l2];

        let ctrl = BoostControl::new(3, 0.1).unwrap();
        let mut family = Gaussian;
        let path = fit(&mut learners, y.view(), w.view(), &mut family, &ctrl, None);

        assert!((path.offset - 5.0).abs() < 1e-12);
        assert_eq!(path.selected.len(), 3);
        assert_eq!(path.risk.len(), 4); // offset risk + 3 steps

        // Risk should decrease monotonically (or at least not increase)
        for i in 1..path.risk.len() {
            assert!(path.risk[i] <= path.risk[i - 1] + 1e-10);
        }
    }

    #[test]
    fn fit_tie_breaker_lowest_index() {
        // Two identical columns → lowest index always selected
        let y = array![1.0, 2.0, 3.0];
        let w = array![1.0, 1.0, 1.0];
        let x = array![1.0, 2.0, 3.0];

        let l1 = prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x1", None);
        let l2 = prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x2", None);
        let mut learners = vec![l1, l2];

        let ctrl = BoostControl::new(5, 0.1).unwrap();
        let mut family = Gaussian;
        let path = fit(&mut learners, y.view(), w.view(), &mut family, &ctrl, None);

        for &idx in &path.selected {
            assert_eq!(idx, 0);
        }
        assert_eq!(path.report.tie_selection_count, 5);
    }

    #[test]
    fn fit_single_step_hand_computed() {
        // y = [2, 4, 6], x = [1, 2, 3], w = [1, 1, 1], nu = 1.0
        // offset = 4.0, f0 = [4, 4, 4]
        // u = y - f = [-2, 0, 2]
        // x centered: [-1, 0, 1], denom = 2
        // numer = (-1)*(-2) + 0*0 + 1*2 = 4
        // beta = 4/2 = 2.0, pred = [-2, 0, 2]
        // f1 = f0 + 1.0 * pred = [2, 4, 6] = y
        // risk[1] = Σ (y-f)² = 0
        let y = array![2.0, 4.0, 6.0];
        let w = array![1.0, 1.0, 1.0];
        let x = array![1.0, 2.0, 3.0];

        let l = prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None);
        let mut learners = vec![l];

        let ctrl = BoostControl::new(1, 1.0).unwrap();
        let mut family = Gaussian;
        let path = fit(&mut learners, y.view(), w.view(), &mut family, &ctrl, None);

        assert!((path.offset - 4.0).abs() < 1e-12);
        assert_eq!(path.selected.len(), 1);
        assert!((path.coefs[0][0] - 2.0).abs() < 1e-12);
        assert!((path.risk[1]).abs() < 1e-10); // perfect fit after one step
        assert!((path.fitted[0] - 2.0).abs() < 1e-12);
        assert!((path.fitted[1] - 4.0).abs() < 1e-12);
        assert!((path.fitted[2] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn gaussian_incremental_risk_matches_full() {
        use crate::control::RiskKind;
        use crate::learners::prepare_scalar_bols;

        let n = 200usize;
        let mut rng = 42u64;
        let mut xs = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng as f64) / (u64::MAX as f64)
        };
        let x1 = Array1::from_shape_fn(n, |_| xs());
        let x2 = Array1::from_shape_fn(n, |_| xs());
        let y = Array1::from_shape_fn(n, |i| 2.0 * x1[i] - x2[i] + (xs() - 0.5));
        let w = Array1::ones(n);
        let ctrl = BoostControl::new(50, 0.1)
            .unwrap()
            .with_risk(RiskKind::Inbag);

        let build = || {
            vec![
                prepare_scalar_bols(x1.view(), w.view(), 0.0, true, "x1", None),
                prepare_scalar_bols(x2.view(), w.view(), 0.0, true, "x2", None),
            ]
        };

        // store_increments=true forces the general path (not Gram).
        let mut l1 = build();
        let mut fam1 = Gaussian;
        let path = fit_with(&mut l1, y.view(), w.view(), &mut fam1, &ctrl, None, true);

        // Recompute full risk from fitted values at each step.
        let mut f_check = Array1::from_elem(n, path.offset);
        let nu = 0.1f64;
        let _full_risks: Vec<f64> = vec![path.risk[0]];
        for m in 0..path.selected.len() {
            let sel = path.selected[m];
            let coef = &path.coefs[m];
            let col = match sel {
                0 => &x1,
                1 => &x2,
                _ => unreachable!(),
            };
            // Scalar bols with centering: the coef is the raw beta;
            // the increment is nu * (coef[0] * (x - mean_x)).
            // But the fitted values from the path already incorporate centering,
            // so we compare against path.fitted directly.
            for i in 0..n {
                f_check[i] += nu * coef[0] * col[i];
            }
        }
        // Full risk at final fitted values.
        let final_full_risk: f64 = y
            .iter()
            .zip(path.fitted.iter())
            .map(|(yi, fi)| (yi - fi) * (yi - fi))
            .sum();
        let final_incr_risk = *path.risk.last().unwrap();
        assert!(
            (final_full_risk - final_incr_risk).abs() < 1e-6,
            "final risk: full={final_full_risk:.6}, incr={final_incr_risk:.6}"
        );
    }

    #[test]
    fn hat_matrix_nonuniform_weights_maintains_df() {
        use crate::learners::prepare_bbs;
        use crate::learners::prepare_scalar_bols;

        let n = 100usize;
        let mut rng = 12345u64;
        let mut xs = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng as f64) / (u64::MAX as f64)
        };
        let x = Array1::from_shape_fn(n, |_| xs());
        let w = Array1::from_shape_fn(n, |_| 0.5 + xs()); // non-uniform
        let w_f64 = w.view();

        // === Scalar Bols without centering (lambda=0, df=1) ===
        let l = prepare_scalar_bols(x.view(), w_f64, 0.0, false, "x", None);
        let h = l.hat_matrix().unwrap();
        let tr: f64 = (0..n).map(|i| h[[i, i]]).sum();
        assert!(
            (tr - 1.0).abs() < 1e-12,
            "scalar bols no-center trace: {tr}"
        );

        // === Scalar Bols with centering (lambda=0, df=2) ===
        let lc = prepare_scalar_bols(x.view(), w_f64, 0.0, true, "x", None);
        let hc = lc.hat_matrix().unwrap();
        let trc: f64 = (0..n).map(|i| hc[[i, i]]).sum();
        assert!((trc - 2.0).abs() < 1e-12, "scalar bols center trace: {trc}");

        // === Scalar Bols with penalty (df < 1) ===
        let lp = prepare_scalar_bols(x.view(), w_f64, 10.0, false, "x", None);
        let hp = lp.hat_matrix().unwrap();
        let trp: f64 = (0..n).map(|i| hp[[i, i]]).sum();
        let xsq = x.mapv(|xi| xi * xi);
        let expected_df = w.dot(&xsq) / (w.dot(&xsq) + 10.0);
        assert!(
            (trp - expected_df).abs() < 1e-12,
            "scalar bols penalized trace: {trp} vs {expected_df}"
        );

        // === BlockBols: bbs with non-uniform weights ===
        let lb = prepare_bbs(x.view(), w_f64, 20, 3, 2, 4.0, false, "bbs(x)");
        let hb = lb.hat_matrix().unwrap();
        let trb: f64 = (0..n).map(|i| hb[[i, i]]).sum();
        assert!(trb > 0.0 && trb < n as f64, "bbs trace out of range: {trb}");
        assert!((trb - 4.0).abs() < 2.0, "bbs trace ~4: {trb}");

        // === AIC trace: 2 iterations, verify monotonic ===
        let l1 = prepare_scalar_bols(x.view(), w_f64, 0.0, true, "x", None);
        let l2 = prepare_scalar_bols(x.view(), w_f64, 0.0, false, "x", None);
        let learners = vec![l1, l2];
        let selected = vec![0usize, 1, 0, 1];
        let trace = try_compute_aic_trace(&learners, &selected, 0.1)
            .unwrap()
            .unwrap();
        assert!(
            trace.windows(2).all(|w| w[1] > w[0] - 1e-12),
            "AIC trace not monotonic: {trace:?}"
        );
        assert!(
            (trace[0] - 0.2).abs() < 1e-12,
            "first trace: {} vs 0.2",
            trace[0]
        );

        // === Verify hat matrix elements directly (penalized λ=10) ===
        let denom_pen = w.dot(&x.mapv(|xi| xi * xi)) + 10.0;
        let inv_denom_pen = 1.0 / denom_pen;
        for i in [0, 1, n / 2, n - 1] {
            for j in [0, n / 3, n - 1] {
                let expected = w[j] * x[i] * x[j] * inv_denom_pen;
                let actual = hp[[i, j]];
                assert!(
                    (actual - expected).abs() < 1e-14,
                    "H[{i},{j}]: expected {expected}, got {actual}"
                );
            }
        }
    }

    #[test]
    fn approximate_hat_diag_covers_multi_column_learners() {
        use crate::learners::{prepare_bbs, prepare_scalar_bols};

        let n = 80usize;
        let mut rng = 848u64;
        let mut xs = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng as f64) / (u64::MAX as f64)
        };
        let x = Array1::from_shape_fn(n, |_| xs());
        let z = Array1::from_shape_fn(n, |_| xs());
        let w = Array1::ones(n);
        let learners = vec![
            prepare_bbs(x.view(), w.view(), 20, 3, 2, 4.0, false, "bbs(x)"),
            prepare_scalar_bols(z.view(), w.view(), 0.0, true, "z", None),
        ];

        // The learner diagonal matches the full hat matrix's.
        let full = learners[0].hat_matrix().unwrap();
        let diag = learners[0].hat_diag().unwrap();
        for i in 0..n {
            assert!((diag[i] - full[[i, i]]).abs() < 1e-12);
        }

        // One step of the spline: the exact hat matrix is nu * H, so the
        // approximation is it plus the offset's 1/n.
        let nu = 0.1;
        let approx = try_approximate_hat_diag(&learners, &[0], nu)
            .unwrap()
            .unwrap();
        let exact = try_compute_hat_diag(&learners, &[0], nu).unwrap().unwrap();
        for i in 0..n {
            assert!((approx[i] - 1.0 / n as f64 - exact[i]).abs() < 1e-12);
        }
        let exact_trace = try_compute_aic_trace(&learners, &[0], nu).unwrap().unwrap();
        let approx_trace: f64 = approx.iter().sum();
        assert!((approx_trace - 1.0 - exact_trace[0]).abs() < 1e-10);

        // Several small steps of the spline: the exact trace is
        // sum_k 1 - (1 - nu e_k)^m over H's eigenvalues e_k, at most the
        // first-order m nu tr(H), and within the second-order term of it.
        let selected = [0usize, 0, 0, 0];
        let nu = 0.01;
        let approx = try_approximate_hat_diag(&learners, &selected, nu)
            .unwrap()
            .unwrap();
        let exact_trace = *try_compute_aic_trace(&learners, &selected, nu)
            .unwrap()
            .unwrap()
            .last()
            .unwrap();
        let approx_trace: f64 = approx.iter().sum::<f64>() - 1.0;
        assert!(approx_trace >= exact_trace - 1e-12);
        assert!(
            (approx_trace - exact_trace) / exact_trace < 0.05,
            "approximate {approx_trace} vs exact {exact_trace}"
        );

        // The single-column learner contributes its slope leverage only.
        let approx_z = try_approximate_hat_diag(&learners, &[1], 1.0)
            .unwrap()
            .unwrap();
        let zc = z.mapv(|v| v - z.mean().unwrap());
        let ss: f64 = zc.iter().map(|v| v * v).sum();
        for i in 0..n {
            let want = (1.0 / n as f64 + zc[i] * zc[i] / ss).clamp(0.0, 1.0);
            assert!((approx_z[i] - want).abs() < 1e-12);
        }
        assert!(try_approximate_hat_diag(&learners, &[2], 0.1).is_err());
    }

    // ─── AIC1: hat-factor verification ──────────────────────────────────

    /// The try_ AIC/hat variants must reject malformed inputs with a typed
    /// `Err` instead of panicking, and reproduce the wrapper's result on
    /// valid inputs.
    #[test]
    fn try_hat_trace_and_aic_variants_reject_bad_inputs() {
        use crate::learners::prepare_scalar_bols;

        let n = 20usize;
        let x = Array1::from_shape_fn(n, |i| (i as f64) / (n as f64));
        let w = Array1::ones(n);
        let l = prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None);
        let learners = vec![l];

        // Out-of-bounds selected index → Err, not a slice-index panic.
        assert!(matches!(
            try_compute_aic_trace(&learners, &[0, 7], 0.1),
            Err(MboostrsError::Solver(_))
        ));
        assert!(matches!(
            try_compute_hat_diag(&learners, &[7], 0.1),
            Err(MboostrsError::Solver(_))
        ));
        assert!(matches!(
            try_compute_aic_trace_fast(&learners, &[7], 0.1),
            Err(MboostrsError::Solver(_))
        ));

        // Empty learner set / empty selection → Err.
        assert!(matches!(
            try_compute_aic_trace(&[], &[0], 0.1),
            Err(MboostrsError::Solver(_))
        ));
        assert!(matches!(
            try_compute_aic_trace_fast(&learners, &[], 0.1),
            Err(MboostrsError::Solver(_))
        ));

        // Non-finite nu → Err.
        assert!(matches!(
            try_compute_aic_trace(&learners, &[0], f64::NAN),
            Err(MboostrsError::Solver(_))
        ));

        // from_hats: no hat matrices → Err.
        assert!(matches!(
            try_compute_aic_trace_from_hats(&[], &[0], 0.1),
            Err(MboostrsError::Solver(_))
        ));

        // compute_aic: risk too short for df (risk[i+1] would be out of
        // bounds), or a non-finite/negative sumw.
        assert!(matches!(
            try_compute_aic("corrected", &[1.0, 2.0], &[1.0, 1.0, 1.0], 10.0, 2.0, 100.0),
            Err(MboostrsError::Solver(_))
        ));
        assert!(matches!(
            try_compute_aic("corrected", &[1.0], &[1.0], 10.0, 2.0, 100.0),
            Err(MboostrsError::Solver(_))
        ));
        assert!(matches!(
            try_compute_aic("corrected", &[0.0, 5.0], &[1.0], -1.0, 2.0, 100.0),
            Err(MboostrsError::Solver(_))
        ));

        // Valid inputs reproduce the panicking wrapper's result exactly.
        // The wrapper is deprecated on the P0.8 schedule; this parity check
        // guards the shim until its 0.2.0 removal.
        let good = try_compute_aic("classical", &[0.0, 5.0], &[1.0], 10.0, 2.0, 100.0).unwrap();
        #[allow(deprecated)]
        let wrapper_result = compute_aic("classical", &[0.0, 5.0], &[1.0], 10.0, 2.0, 100.0);
        assert_eq!(wrapper_result, good);
        assert!(matches!(
            try_compute_hat_diag(&learners, &[0, 0, 0], 0.1),
            Ok(Some(_))
        ));
    }

    // ─── btree step hat matrices ─────────────────────────────────────

    /// One tree step's matrix, reconstructed independently of the code
    /// under test: decode the step's recorded tree, group the rows by
    /// the value the tree predicts for them (rows in one leaf share the
    /// leaf mean; the helper asserts the leaf means are distinct, so the
    /// groups ARE the leaves), and average the working response over
    /// each group with the learner's stored fit weights, in plain loops.
    fn brute_force_tree_step_matrix(learner: &PreparedLearner, coef: &[f64]) -> Array2<f64> {
        use crate::learners::FittedTree;
        let PreparedLearner::Btree {
            design, weights, ..
        } = learner
        else {
            panic!("brute_force_tree_step_matrix needs a tree learner");
        };
        let n = design.nrows();
        let tree = FittedTree::from_coef_vec(ndarray::ArrayView1::from(coef));
        let pred = tree.predict(design.view());
        // The leaf count, from the encoding's leaf tags.
        let mut leaves = 0usize;
        let mut idx = 1usize;
        for _ in 0..coef[0] as usize {
            match coef[idx] {
                0.0 => idx += 5,
                2.0 => idx += 5 + coef[idx + 2] as usize,
                _ => {
                    leaves += 1;
                    idx += 2;
                }
            }
        }
        let mut distinct: Vec<f64> = pred.to_vec();
        distinct.sort_by(|a, b| a.partial_cmp(b).unwrap());
        distinct.dedup();
        assert_eq!(
            distinct.len(),
            leaves,
            "leaf means must be distinct for grouping by prediction to recover the leaves"
        );
        let mut h = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            let total: f64 = (0..n)
                .filter(|&j| pred[j] == pred[i])
                .map(|j| weights[j])
                .sum();
            assert!(total > 0.0, "a fitted leaf has positive fit weight");
            for j in 0..n {
                if pred[j] == pred[i] {
                    h[[i, j]] = weights[j] / total;
                }
            }
        }
        h
    }

    /// The boosting hat recursion over a recorded path, composed in the
    /// test from per-step matrices: a smooth learner's established
    /// `hat_matrix()`, a tree step's brute-force matrix above. With
    /// `step_weights`, each step's update is row-scaled after the fact,
    /// as the IRLS-weighted recursion prescribes.
    fn brute_force_hat_path(
        learners: &[PreparedLearner],
        selected: &[usize],
        coefs: &[Vec<f64>],
        nu: f64,
        step_weights: Option<&[Vec<f64>]>,
    ) -> (Vec<f64>, Vec<f64>) {
        let n = learners[0].n();
        let mut b = Array2::<f64>::zeros((n, n));
        let mut trace = Vec::with_capacity(selected.len());
        for (step, &sel) in selected.iter().enumerate() {
            let h = match &learners[sel] {
                PreparedLearner::Btree { .. } => {
                    brute_force_tree_step_matrix(&learners[sel], &coefs[step])
                }
                learner => learner.hat_matrix().expect("smooth learner matrix"),
            };
            // z = (nu H)(I - B) = nu H - nu H B
            let mut z = Array2::<f64>::zeros((n, n));
            ndarray::linalg::general_mat_mul(-nu, &h, &b, 0.0, &mut z);
            z.zip_mut_with(&h, |zj, &hj| *zj += nu * hj);
            if let Some(all_w) = step_weights {
                let w = &all_w[step];
                for (i, mut row) in z.rows_mut().into_iter().enumerate() {
                    row.mapv_inplace(|v| v * w[i]);
                }
            }
            b += &z;
            trace.push(b.diag().sum());
        }
        (trace, b.diag().to_vec())
    }

    fn two_column_threshold_data(n: usize) -> (Array1<f64>, Array1<f64>, Array2<f64>) {
        let mut state = 0x1F123BB5u32;
        let mut next = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 8) as f64 / (1u32 << 24) as f64
        };
        let x = Array1::from_shape_fn(n, |_| next() * 4.0 - 2.0);
        let z = Array1::from_shape_fn(n, |_| next() * 2.0 - 1.0);
        let design = Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { x[i] } else { z[i] });
        (x, z, design)
    }

    /// A Gaussian fit whose path selects a tree repeatedly: the trace
    /// and the diagonal from the steps variants agree with the
    /// brute-force composition of the same recorded path, and the
    /// coefficient-free calls still refuse, as they did before.
    #[test]
    fn btree_steps_in_hat_recursion_match_brute_force() {
        use crate::learners::{prepare_btree, prepare_scalar_bols};

        let n = 36usize;
        let (x, z, design) = two_column_threshold_data(n);
        let mut noise = 0x0D15EA5Eu32;
        let mut next_noise = move || {
            noise = noise.wrapping_mul(1664525).wrapping_add(1013904223);
            (noise >> 8) as f64 / (1u32 << 24) as f64
        };
        let y = Array1::from_shape_fn(n, |i| {
            (if x[i] > 0.25 { 2.0 } else { -1.0 }) + 0.3 * z[i] + 0.05 * next_noise()
        });
        let w = Array1::ones(n);
        let mut learners = vec![
            prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None),
            prepare_btree(design.view(), w.view(), 3, 6, 2, None, "tree"),
        ];
        let nu = 0.3;
        let control = BoostControl::new(20, nu).unwrap();
        let mut family = Gaussian;
        let path = try_fit(
            &mut learners,
            y.view(),
            w.view(),
            &mut family,
            &control,
            None,
        )
        .unwrap();
        assert!(
            path.selected.iter().filter(|&&s| s == 1).count() >= 2,
            "the tree must be selected repeatedly for this test: {:?}",
            path.selected
        );

        let (trace_bf, diag_bf) =
            brute_force_hat_path(&learners, &path.selected, &path.coefs, nu, None);
        let trace = try_compute_aic_trace_steps(&learners, &path.selected, Some(&path.coefs), nu)
            .unwrap()
            .expect("tree steps resolve from the recorded path");
        let diag = try_compute_hat_diag_steps(&learners, &path.selected, Some(&path.coefs), nu)
            .unwrap()
            .expect("tree steps resolve from the recorded path");
        for (got, want) in trace.iter().zip(&trace_bf) {
            assert!(
                (got - want).abs() < 1e-10,
                "trace: {got} vs brute force {want}"
            );
        }
        for (got, want) in diag.iter().zip(&diag_bf) {
            assert!(
                (got - want).abs() < 1e-10,
                "diag: {got} vs brute force {want}"
            );
        }
        // The fast trace takes its dense fallback for a tree path and
        // lands on the same numbers.
        let fast =
            try_compute_aic_trace_fast_steps(&learners, &path.selected, Some(&path.coefs), nu)
                .unwrap()
                .expect("fast steps trace");
        for (got, want) in fast.iter().zip(&trace) {
            assert!(
                (got - want).abs() < 1e-12,
                "fast trace: {got} vs dense {want}"
            );
        }
        // Without the coefficients the same calls still refuse, as before.
        assert_eq!(
            try_compute_aic_trace(&learners, &path.selected, nu).unwrap(),
            None
        );
        assert_eq!(
            try_compute_hat_diag(&learners, &path.selected, nu).unwrap(),
            None
        );
    }

    /// The IRLS-weighted recursion over a Binomial fit with tree steps:
    /// the steps variant agrees with the brute-force composition using
    /// the step weights at the fit after each step, rebuilt from the
    /// path's offset and increments exactly as the bindings rebuild them.
    #[test]
    fn btree_steps_in_weighted_hat_match_brute_force() {
        use crate::families::Binomial;
        use crate::learners::{prepare_btree, prepare_scalar_bols};

        let n = 36usize;
        let (x, z, design) = two_column_threshold_data(n);
        let mut noise = 0x0BADF00Du32;
        let mut next_noise = move || {
            noise = noise.wrapping_mul(1664525).wrapping_add(1013904223);
            (noise >> 8) as f64 / (1u32 << 24) as f64
        };
        let y = Array1::from_shape_fn(n, |i| {
            let latent =
                (if x[i] > 0.25 { 1.3 } else { -1.3 }) + 0.3 * z[i] + (next_noise() - 0.5) * 0.8;
            if latent > 0.0 {
                1.0
            } else {
                0.0
            }
        });
        let w = Array1::ones(n);
        let mut learners = vec![
            prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None),
            prepare_btree(design.view(), w.view(), 3, 6, 2, None, "tree"),
        ];
        let nu = 0.3;
        let control = BoostControl::new(20, nu).unwrap();
        let mut family = Binomial;
        let path = try_fit(
            &mut learners,
            y.view(),
            w.view(),
            &mut family,
            &control,
            None,
        )
        .unwrap();
        assert!(
            path.selected.contains(&1),
            "the tree must be selected for this test: {:?}",
            path.selected
        );

        let mut f = vec![path.offset; n];
        let mut step_weights: Vec<Vec<f64>> = Vec::with_capacity(path.increments.len());
        for inc in &path.increments {
            for i in 0..n {
                f[i] += inc[i];
            }
            step_weights.push(
                f.iter()
                    .map(|&fi| glm_hat_weight("binomial", fi).expect("binomial weight"))
                    .collect(),
            );
        }
        let (trace_bf, diag_bf) = brute_force_hat_path(
            &learners,
            &path.selected,
            &path.coefs,
            nu,
            Some(&step_weights),
        );
        let (trace, diag) = try_compute_hat_weighted_steps(
            &learners,
            &path.selected,
            Some(&path.coefs),
            nu,
            &step_weights,
        )
        .unwrap()
        .expect("tree steps resolve from the recorded path");
        for (got, want) in trace.iter().zip(&trace_bf) {
            assert!(
                (got - want).abs() < 1e-10,
                "weighted trace: {got} vs brute force {want}"
            );
        }
        for (got, want) in diag.iter().zip(&diag_bf) {
            assert!(
                (got - want).abs() < 1e-10,
                "weighted diag: {got} vs brute force {want}"
            );
        }
        // Without the coefficients the weighted call still refuses.
        assert_eq!(
            try_compute_hat_weighted(&learners, &path.selected, nu, &step_weights).unwrap(),
            None
        );
    }

    /// A selected `bmono` learner still refuses in every steps variant:
    /// the tree convention does not extend to it.
    #[test]
    fn bmono_still_refuses_in_steps_variants() {
        use crate::learners::{prepare_bmono, prepare_scalar_bols, BmonoSolver, MonoConstraint};

        let n = 24usize;
        let x = Array1::from_shape_fn(n, |i| i as f64 / (n - 1) as f64);
        let w = Array1::ones(n);
        let learners = vec![
            prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None),
            prepare_bmono(
                x.view(),
                w.view(),
                10,
                3,
                2,
                4.0,
                MonoConstraint::Increasing,
                BmonoSolver::Iterative,
                1e6,
                10,
                "z",
                None,
                None,
                false,
            ),
        ];
        let selected = vec![0usize, 1, 0];
        let coefs: Vec<Vec<f64>> = vec![vec![0.0], vec![0.0; 8], vec![0.0]];
        let weights: Vec<Vec<f64>> = vec![vec![1.0; n]; 3];
        assert_eq!(
            try_compute_aic_trace_steps(&learners, &selected, Some(&coefs), 0.1).unwrap(),
            None
        );
        assert_eq!(
            try_compute_hat_diag_steps(&learners, &selected, Some(&coefs), 0.1).unwrap(),
            None
        );
        assert_eq!(
            try_compute_hat_weighted_steps(&learners, &selected, Some(&coefs), 0.1, &weights)
                .unwrap(),
            None
        );
    }

    /// Verify that the factor X·M·(XᵀW) reconstructs hat_matrix() for each
    /// learner variant (scalar, centered scalar, block, penalised).
    #[test]
    fn hat_factor_matches_hat_matrix() {
        use crate::learners::prepare_bbs;
        use crate::learners::prepare_scalar_bols;

        let n = 50usize;
        let x = Array1::from_shape_fn(n, |i| (i as f64) / (n as f64));
        let w = Array1::ones(n);
        let w_f64 = w.view();

        // Scalar Bols: no centering, no penalty
        let l = prepare_scalar_bols(x.view(), w_f64, 0.0, false, "x", None);
        verify_factor(&l, w.as_slice().unwrap());

        // Scalar Bols: with centering
        let lc = prepare_scalar_bols(x.view(), w_f64, 0.0, true, "x", None);
        verify_factor(&lc, w.as_slice().unwrap());

        // Scalar Bols: penalised
        let lp = prepare_scalar_bols(x.view(), w_f64, 5.0, false, "x", None);
        verify_factor(&lp, w.as_slice().unwrap());

        // Scalar Bols: penalised + centered
        let lpc = prepare_scalar_bols(x.view(), w_f64, 5.0, true, "x", None);
        verify_factor(&lpc, w.as_slice().unwrap());

        // BBS (BlockBols)
        let lb = prepare_bbs(x.view(), w_f64, 20, 3, 2, 4.0, false, "bbs(x)");
        verify_factor(&lb, w.as_slice().unwrap());

        // BBS with non-uniform weights
        let w2 = Array1::from_shape_fn(n, |i| 0.5 + (i as f64) / (n as f64));
        let lb2 = prepare_bbs(x.view(), w2.view(), 20, 3, 2, 4.0, false, "bbs(x)");
        verify_factor(&lb2, w2.as_slice().unwrap());
    }

    fn verify_factor(l: &PreparedLearner, w_slice: &[f64]) {
        let Some(factor) = l.hat_factor() else {
            panic!("hat_factor returned None for a learner that should support it");
        };
        let Some(h_mat) = l.hat_matrix() else {
            panic!("hat_matrix returned None");
        };
        let n = factor.design.nrows();
        let p = factor.design.ncols();
        // Reconstruct H_rebuilt = X · M · (XᵀW)
        // H_rebuilt[i,j] = Σ_k Σ_l X[i,k] · M[k,l] · (X[j,l] · w[j])
        //                = Σ_k X[i,k] · Σ_l M[k,l] · X[j,l] · w[j]
        let mut h_rebuilt = ndarray::Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let mut val = 0.0;
                for k in 0..p {
                    let xik = factor.design.read(i, k);
                    let mut inner = 0.0;
                    for l in 0..p {
                        inner += factor.hat_inv[k * p + l] * factor.design.read(j, l);
                    }
                    val += xik * inner * w_slice[j];
                }
                h_rebuilt[[i, j]] = val;
            }
        }
        for i in 0..n {
            for j in 0..n {
                let diff = (h_rebuilt[[i, j]] - h_mat[[i, j]]).abs();
                assert!(diff < 1e-12, "rebuild mismatch at ({i},{j}): {:.2e}", diff);
            }
        }
        // Also verify the trace matches
        let tr_rebuilt: f64 = (0..n).map(|i| h_rebuilt[[i, i]]).sum();
        assert!(
            (tr_rebuilt - factor.tr_h).abs() < 1e-12,
            "trace mismatch: rebuilt={tr_rebuilt}, factor={}",
            factor.tr_h
        );
    }

    /// `df()` / `df_mboost()` via the low-rank factor path equals the dense
    /// hat-matrix computation, and both match what `verify_factor` builds.
    #[test]
    fn df_fast_matches_dense_hat_matrix() {
        use crate::learners::prepare_bbs;
        use crate::learners::prepare_scalar_bols;

        let n = 50usize;
        let x = Array1::from_shape_fn(n, |i| (i as f64) / (n as f64));
        let w = Array1::ones(n);
        let w_f64 = w.view();

        let cases: Vec<PreparedLearner> = vec![
            prepare_scalar_bols(x.view(), w_f64, 0.0, false, "x", None),
            prepare_scalar_bols(x.view(), w_f64, 0.0, true, "x", None),
            prepare_scalar_bols(x.view(), w_f64, 5.0, false, "x", None),
            prepare_scalar_bols(x.view(), w_f64, 5.0, true, "x", None),
            prepare_bbs(x.view(), w_f64, 20, 3, 2, 4.0, false, "bbs(x)"),
        ];

        for l in &cases {
            let df_fast = l.df().expect("df fast None");
            let df_dense = dense_df(l).expect("df dense None");
            assert!(
                (df_fast - df_dense).abs() < 1e-9,
                "df fast {df_fast} != dense {df_dense} for {}",
                l.name()
            );

            let dfm_fast = l.df_mboost().expect("df_mboost fast None");
            let dfm_dense = dense_df_mboost(l).expect("df_mboost dense None");
            assert!(
                (dfm_fast - dfm_dense).abs() < 1e-9,
                "df_mboost fast {dfm_fast} != dense {dfm_dense} for {}",
                l.name()
            );
        }
    }

    fn dense_df(l: &PreparedLearner) -> Option<f64> {
        let h = l.hat_matrix()?;
        Some((0..h.nrows()).map(|i| h[[i, i]]).sum())
    }

    fn dense_df_mboost(l: &PreparedLearner) -> Option<f64> {
        let h = l.hat_matrix()?;
        let n = h.nrows();
        let mut value = 0.0;
        for i in 0..n {
            value += 2.0 * h[[i, i]];
            for j in 0..n {
                value -= h[[j, i]] * h[[j, i]];
            }
        }
        Some(value)
    }

    /// Verify that the fast AIC trace matches the dense reference trace.
    #[test]
    fn aic_trace_fast_matches_dense() {
        use crate::learners::prepare_bbs;
        use crate::learners::prepare_scalar_bols;

        let n = 100usize;
        let mut rng = 12345u64;
        let mut xs = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng as f64) / (u64::MAX as f64)
        };
        let x1 = Array1::from_shape_fn(n, |_| xs());
        let x2 = Array1::from_shape_fn(n, |_| xs());
        let w = Array1::ones(n);

        // Build a mixed learner set: centered scalar, uncentered scalar, BBS
        let l1 = prepare_scalar_bols(x1.view(), w.view(), 0.0, true, "x1", None);
        let l2 = prepare_scalar_bols(x2.view(), w.view(), 3.0, false, "x2", None);
        let l3 = prepare_bbs(x1.view(), w.view(), 20, 3, 2, 4.0, false, "bbs(x1)");
        let l4 = prepare_scalar_bols(x1.view(), w.view(), 0.0, false, "x1_plain", None);

        let learners = vec![l1, l2, l3, l4];
        let selected: Vec<usize> = (0..40).map(|i| i % 4).collect();

        let dense = try_compute_aic_trace(&learners, &selected, 0.1)
            .unwrap()
            .unwrap();
        let fast = try_compute_aic_trace_fast(&learners, &selected, 0.1)
            .unwrap()
            .unwrap();

        assert_eq!(
            dense.len(),
            fast.len(),
            "trace length mismatch: {} vs {}",
            dense.len(),
            fast.len()
        );
        for (i, (d, f)) in dense.iter().zip(fast.iter()).enumerate() {
            let diff = (d - f).abs();
            assert!(
                diff < 1e-10,
                "trace mismatch at step {i}: dense={d:.12e}, fast={f:.12e}, diff={diff:.2e}"
            );
        }

        // Also test with the fallback guard: construct enough learners to make P large
        let many_learners: Vec<_> = (0..10)
            .map(|i| {
                prepare_scalar_bols(
                    Array1::from_shape_fn(n, |_| xs()).view(),
                    w.view(),
                    if i % 2 == 0 { 0.0 } else { 1.0 },
                    i % 3 == 0,
                    &format!("x{i}"),
                    None,
                )
            })
            .collect();
        let selected2: Vec<usize> = (0..30).map(|i| i % 10).collect();
        let dense2 = try_compute_aic_trace(&many_learners, &selected2, 0.1)
            .unwrap()
            .unwrap();
        let fast2 = try_compute_aic_trace_fast(&many_learners, &selected2, 0.1)
            .unwrap()
            .unwrap();
        for (i, (d, f)) in dense2.iter().zip(fast2.iter()).enumerate() {
            let diff = (d - f).abs();
            assert!(
                diff < 1e-10,
                "many-learner trace mismatch at step {i}: dense={d:.12e}, fast={f:.12e}, diff={diff:.2e}"
            );
        }
    }

    /// Verify that compute_hat_diag returns a vector whose sum equals the
    /// final trace from compute_aic_trace.
    #[test]
    fn hat_diag_sum_matches_trace() {
        use crate::learners::prepare_bbs;
        use crate::learners::prepare_scalar_bols;

        let n = 50usize;
        let mut rng = 12345u64;
        let mut xs = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng as f64) / (u64::MAX as f64)
        };
        let x1 = Array1::from_shape_fn(n, |_| xs());
        let x2 = Array1::from_shape_fn(n, |_| xs());
        let w = Array1::ones(n);

        let l1 = prepare_scalar_bols(x1.view(), w.view(), 0.0, true, "x1", None);
        let l2 = prepare_scalar_bols(x2.view(), w.view(), 2.0, false, "x2", None);
        let l3 = prepare_bbs(x1.view(), w.view(), 10, 3, 2, 4.0, false, "bbs(x1)");

        let learners = vec![l1, l2, l3];
        let selected: Vec<usize> = (0..20).map(|i| i % 3).collect();
        let nu = 0.1f64;

        let trace = try_compute_aic_trace(&learners, &selected, nu)
            .unwrap()
            .unwrap();
        let diag = try_compute_hat_diag(&learners, &selected, nu)
            .unwrap()
            .unwrap();

        assert_eq!(diag.len(), n, "hat diag length mismatch");
        let diag_sum: f64 = diag.iter().sum();
        let final_trace = *trace.last().unwrap();
        assert!(
            (diag_sum - final_trace).abs() < 1e-12,
            "hat diag sum ({diag_sum:.12}) != final trace ({final_trace:.12})"
        );
        // All leverages should be non-negative and ≤ 1
        for (i, &h) in diag.iter().enumerate() {
            assert!(h >= 0.0, "negative leverage at obs {i}: {h}");
            assert!(h <= 1.0 + 1e-10, "leverage > 1 at obs {i}: {h}");
        }
    }

    // ─── Regression tests from Audit 1 & 2 ────────────────────────────────

    #[test]
    fn fit_constant_response_produces_zero_selection() {
        // T2 edge case: all y identical → offset is that value,
        // all residuals are zero → no learner should reduce SSE.
        let n = 20usize;
        let y = Array1::from_elem(n, 5.0);
        let w = Array1::ones(n);
        let x = Array1::from_shape_fn(n, |i| (i as f64) / (n as f64));

        let l = prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None);
        let mut learners = vec![l];
        let ctrl = BoostControl::new(10, 0.1).unwrap();
        let mut fam = Gaussian;
        let path = fit(&mut learners, y.view(), w.view(), &mut fam, &ctrl, None);

        assert!((path.offset - 5.0).abs() < 1e-12);
        // Risk should never increase (stays at ~0 since residuals are 0)
        for i in 1..path.risk.len() {
            assert!(
                path.risk[i] <= path.risk[i - 1] + 1e-10,
                "risk increased at step {i}: {} > {}",
                path.risk[i],
                path.risk[i - 1]
            );
        }
        // Fitted values should remain at the offset (residuals were 0)
        for i in 0..n {
            assert!(
                (path.fitted[i] - 5.0).abs() < 1e-10,
                "fitted[{i}] should stay at offset, got {}",
                path.fitted[i]
            );
        }
    }

    #[test]
    fn fit_single_observation() {
        // T2 edge case: n=1 — single obs with one predictor.
        let y = array![3.5];
        let w = array![1.0];
        let x = array![2.0];

        let l = prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None);
        let mut learners = vec![l];
        let ctrl = BoostControl::new(3, 0.1).unwrap();
        let mut fam = Gaussian;
        let path = fit(&mut learners, y.view(), w.view(), &mut fam, &ctrl, None);

        assert!((path.offset - 3.5).abs() < 1e-12);
        assert_eq!(path.selected.len(), 3);
        assert!(
            path.risk.iter().all(|r| r.is_finite()),
            "risk not finite: {:?}",
            path.risk
        );
        assert!(path.fitted.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn fit_all_zero_weights_produces_finite_output() {
        // T2 edge case: all weights zero → offset = 0, risk = 0, no update.
        let n = 10usize;
        let y = Array1::from_shape_fn(n, |i| i as f64);
        let w = Array1::zeros(n);
        let x = Array1::from_shape_fn(n, |i| (i as f64) * 2.0);

        let l = prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None);
        let mut learners = vec![l];
        let ctrl = BoostControl::new(5, 0.1).unwrap();
        let mut fam = Gaussian;
        let path = fit(&mut learners, y.view(), w.view(), &mut fam, &ctrl, None);

        assert!(path.offset.is_finite());
        assert!(path.risk.iter().all(|r| r.is_finite()));
        assert!(path.fitted.iter().all(|v| v.is_finite()));
        // All risk values should be 0 (zero-weight SSE = 0)
        for r in &path.risk {
            assert!(
                r.abs() < 1e-12,
                "risk should be 0 with zero weights, got {r}"
            );
        }
    }

    #[test]
    fn fit_zero_variance_predictor() {
        // T2 edge case: predictor is constant → centered design is all-zero
        // → denom = 0 → beta stays 0 → no update.
        let n = 20usize;
        let y = Array1::from_shape_fn(n, |i| i as f64 / n as f64);
        let w = Array1::ones(n);
        let x = Array1::from_elem(n, 5.0);

        let l = prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None);
        let mut learners = vec![l];
        let ctrl = BoostControl::new(10, 0.1).unwrap();
        let mut fam = Gaussian;
        let path = fit(&mut learners, y.view(), w.view(), &mut fam, &ctrl, None);

        // Fitted values should stay at offset (no update from zero-variance predictor)
        for i in 0..n {
            assert!(
                (path.fitted[i] - path.offset).abs() < 1e-12,
                "fitted[{i}] should equal offset, got {}",
                path.fitted[i]
            );
        }
    }

    #[test]
    fn fit_no_centering_zero_variance_predictor() {
        // Without centering, a constant predictor gets β = 0 because numer = 0
        // (x'Wu = 0 when x constant and u sums to 0 due to centered offset).
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let w = ndarray::Array1::from_elem(10, 1.0);
        let x = ndarray::Array1::from_elem(10, 5.0); // constant column

        let l = prepare_scalar_bols(x.view(), w.view(), 0.0, false, "x", None);
        let mut learners = vec![l];
        let ctrl = BoostControl::new(3, 1.0).unwrap();
        let mut fam = Gaussian;
        let path = fit(&mut learners, y.view(), w.view(), &mut fam, &ctrl, None);

        // β = numer/denom where numer = Σ w_i x_i u_i.
        // With x all 5 and spread parameter, numer = 5 * Σ w_i u_i.
        // The engine's centering inside fit_with ensures Σ u_i = 0 per step.
        // Since we're NOT centering here, check that coef is finite.
        assert!(path.coefs.iter().all(|c| c[0].is_finite()));
        assert!(path.risk.iter().all(|r| r.is_finite()));
    }

    #[test]
    fn test_compute_aic_edge_cases() {
        // sumw = 0 → INFINITY for all iterations
        let (aic, _) = try_compute_aic("corrected", &[0.0, 10.0], &[1.0], 0.0, 2.0, 100.0).unwrap();
        assert!(aic[0].is_infinite());

        // corrected: denominator approaches zero when sumw ≈ d + 2
        let (aic, _) = try_compute_aic("corrected", &[0.0, 5.0], &[8.0], 10.0, 2.0, 100.0).unwrap();
        assert!(
            aic[0].is_finite(),
            "corrected AIC should clamp denom -> finite"
        );

        // gmdl: d = 0 should not panic or NaN
        let (aic, _) = try_compute_aic("gmdl", &[0.0, 5.0], &[0.0], 10.0, 2.0, 100.0).unwrap();
        assert!(aic[0].is_finite());

        // classical: always finite for finite inputs
        let (aic, _) = try_compute_aic("classical", &[0.0, 5.0], &[1.0], 10.0, 2.0, 100.0).unwrap();
        assert!(aic[0].is_finite());

        // No NaN anywhere for any method
        for method in &["corrected", "classical", "gmdl"] {
            let (aic, _) =
                try_compute_aic(method, &[0.0, 5.0, 4.0], &[1.0, 1.5], 10.0, 2.0, 100.0).unwrap();
            assert!(
                aic.iter().all(|v| !v.is_nan()),
                "NaN in {method} AIC: {aic:?}"
            );
        }
    }

    #[test]
    fn test_compute_aic_finite_normal_case() {
        let risk = vec![0.0, 10.0, 9.0, 8.0];
        let df = vec![2.0, 2.0, 2.0];
        let (aic, _) = try_compute_aic("corrected", &risk, &df, 10.0, 2.0, 100.0).unwrap();
        assert!(aic.iter().all(|v| v.is_finite()));
    }

    // ── Stochastic component subsampling ────────────────────────────────
    mod subsampling {
        use super::*;
        use crate::learners::prepare_scalar_bols;

        fn scalar_dataset(n: usize, p: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
            let mut state = seed.max(1);
            let mut next = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 11) as f64 / (1u64 << 53) as f64
            };
            let cols = Array2::from_shape_fn((n, p), |_| next() - 0.5);
            // Only the first three columns carry signal.
            let y = Array1::from_shape_fn(n, |i| {
                1.5 * cols[[i, 0]] - 0.9 * cols[[i, 1]] + 0.6 * cols[[i, 2]] + 0.05 * (next() - 0.5)
            });
            (cols, y)
        }

        fn build(cols: &Array2<f64>, w: &Array1<f64>) -> Vec<PreparedLearner> {
            (0..cols.ncols())
                .map(|j| {
                    prepare_scalar_bols(cols.column(j), w.view(), 0.0, true, &format!("x{j}"), None)
                })
                .collect()
        }

        #[test]
        fn full_fraction_equals_no_subsampling() {
            // fraction so high it rounds to every learner -> None path -> identical.
            let (cols, y) = scalar_dataset(200, 8, 7);
            let w = Array1::ones(200);
            let base = BoostControl::new(50, 0.1).unwrap();
            let mut a = build(&cols, &w);
            let mut fam_a = Gaussian;
            let ref_path = try_fit(&mut a, y.view(), w.view(), &mut fam_a, &base, None).unwrap();

            // 0.99 * 8 rounds to 8 == n_learners, so the mask collapses to None.
            let sub = base.with_component_subsampling(0.99, 123).unwrap();
            let mut b = build(&cols, &w);
            let mut fam_b = Gaussian;
            let sub_path = try_fit(&mut b, y.view(), w.view(), &mut fam_b, &sub, None).unwrap();

            assert_eq!(ref_path.selected, sub_path.selected);
        }

        #[test]
        fn same_seed_gives_identical_path() {
            let (cols, y) = scalar_dataset(200, 12, 3);
            let w = Array1::ones(200);
            let control = BoostControl::new(80, 0.1)
                .unwrap()
                .with_component_subsampling(0.5, 999)
                .unwrap();

            let mut a = build(&cols, &w);
            let mut fam_a = Gaussian;
            let pa = try_fit(&mut a, y.view(), w.view(), &mut fam_a, &control, None).unwrap();
            let mut b = build(&cols, &w);
            let mut fam_b = Gaussian;
            let pb = try_fit(&mut b, y.view(), w.view(), &mut fam_b, &control, None).unwrap();

            assert_eq!(pa.selected, pb.selected);
            assert_eq!(pa.fitted, pb.fitted);
        }

        #[test]
        fn different_seed_changes_the_path() {
            let (cols, y) = scalar_dataset(200, 12, 5);
            let w = Array1::ones(200);
            let base = BoostControl::new(80, 0.1).unwrap();
            let c1 = base.clone().with_component_subsampling(0.4, 1).unwrap();
            let c2 = base.with_component_subsampling(0.4, 2).unwrap();

            let mut a = build(&cols, &w);
            let mut fam_a = Gaussian;
            let p1 = try_fit(&mut a, y.view(), w.view(), &mut fam_a, &c1, None).unwrap();
            let mut b = build(&cols, &w);
            let mut fam_b = Gaussian;
            let p2 = try_fit(&mut b, y.view(), w.view(), &mut fam_b, &c2, None).unwrap();

            assert_ne!(p1.selected, p2.selected);
        }

        #[test]
        fn subsampled_fit_still_reduces_risk_and_recovers_signal() {
            let (cols, y) = scalar_dataset(400, 20, 9);
            let w = Array1::ones(400);
            let control = BoostControl::new(300, 0.1)
                .unwrap()
                .with_component_subsampling(0.3, 42)
                .unwrap();
            let mut learners = build(&cols, &w);
            let mut fam = Gaussian;
            let path =
                try_fit(&mut learners, y.view(), w.view(), &mut fam, &control, None).unwrap();

            // Risk falls monotonically overall.
            assert!(path.risk.last().unwrap() < &path.risk[0]);
            // The three signal columns are among the most-selected.
            let mut counts = vec![0usize; 20];
            for &s in &path.selected {
                counts[s] += 1;
            }
            let top3: std::collections::HashSet<usize> = {
                let mut idx: Vec<usize> = (0..20).collect();
                idx.sort_by_key(|&i| std::cmp::Reverse(counts[i]));
                idx.into_iter().take(3).collect()
            };
            let signal: std::collections::HashSet<usize> = [0, 1, 2].into_iter().collect();
            assert!(
                top3.intersection(&signal).count() >= 2,
                "subsampled fit did not concentrate on the signal columns: {counts:?}"
            );
        }

        #[test]
        fn invalid_fraction_is_rejected() {
            let base = BoostControl::new(10, 0.1).unwrap();
            assert!(base.clone().with_component_subsampling(0.0, 1).is_err());
            assert!(base.clone().with_component_subsampling(1.0, 1).is_err());
            assert!(base.with_component_subsampling(f64::NAN, 1).is_err());
        }

        #[test]
        fn report_counts_subsampled_iterations() {
            let (cols, y) = scalar_dataset(200, 10, 4);
            let w = Array1::ones(200);
            let control = BoostControl::new(60, 0.1)
                .unwrap()
                .with_component_subsampling(0.5, 7)
                .unwrap();
            let mut learners = build(&cols, &w);
            let mut fam = Gaussian;
            let path =
                try_fit(&mut learners, y.view(), w.view(), &mut fam, &control, None).unwrap();
            assert_eq!(path.report.component_subsample_iterations, 60);
        }
    }
}

#[cfg(test)]
mod warm_continuation {
    //! A continued path must equal a cold path of the same length.
    //!
    //! Not approximately: the first `m` iterations of a boosting fit are a
    //! deterministic function of the data, so resuming from the state they
    //! left has to reproduce the rest bit for bit. Anything less and
    //! `continue_to` would return a different model depending on how the
    //! caller got there, which is worse than being slow.

    use super::*;
    use crate::families::{Gaussian, Huber, NBinomial, Poisson, PropOdds};
    use crate::learners::prepare_scalar_bols;
    use ndarray::{Array1, Array2};

    fn design(n: usize, p: usize, seed: u64) -> Array2<f64> {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64 / (1u64 << 31) as f64) - 1.0
        };
        Array2::from_shape_fn((n, p), |_| next())
    }

    fn learners(x: &Array2<f64>, w: ArrayView1<f64>) -> Vec<PreparedLearner> {
        (0..x.ncols())
            .map(|j| prepare_scalar_bols(x.column(j), w, 0.0, false, &format!("x{j}"), None))
            .collect()
    }

    fn assert_same_path(cold: &BoostingPath, warm: &BoostingPath) {
        assert_eq!(cold.selected, warm.selected, "selection sequence");
        assert_eq!(cold.offset.to_bits(), warm.offset.to_bits(), "offset");
        assert_eq!(cold.risk.len(), warm.risk.len(), "risk path length");
        for (index, (a, b)) in cold.risk.iter().zip(warm.risk.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "risk at iteration {index}");
        }
        assert_eq!(cold.coefs.len(), warm.coefs.len(), "coefficient count");
        for (index, (a, b)) in cold.coefs.iter().zip(warm.coefs.iter()).enumerate() {
            assert_eq!(a, b, "coefficients at iteration {index}");
        }
        for (index, (a, b)) in cold.fitted.iter().zip(warm.fitted.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "fitted value {index}");
        }
        assert_eq!(cold.nuisance_path, warm.nuisance_path, "nuisance path");
        assert_eq!(
            cold.nuisance_vec_path, warm.nuisance_vec_path,
            "vector nuisance path"
        );
    }

    /// Fit cold to `total`, fit cold to `part` then continue, compare.
    fn check<F: Family + Clone>(family: F, y: Array1<f64>, part: u32, total: u32) {
        let n = y.len();
        let x = design(n, 4, 7);
        let w = Array1::ones(n);
        let control_part = BoostControl::new(part, 0.1).unwrap();
        let control_total = BoostControl::new(total, 0.1).unwrap();

        let mut cold_family = family.clone();
        let mut cold_learners = learners(&x, w.view());
        let cold = try_fit(
            &mut cold_learners,
            y.view(),
            w.view(),
            &mut cold_family,
            &control_total,
            None,
        )
        .expect("cold fit");

        let mut warm_family = family.clone();
        let mut warm_learners = learners(&x, w.view());
        let partial = try_fit(
            &mut warm_learners,
            y.view(),
            w.view(),
            &mut warm_family,
            &control_part,
            None,
        )
        .expect("partial fit");
        assert_eq!(partial.selected.len(), part as usize);

        let mut continued_learners = learners(&x, w.view());
        let state = WarmStart::from_path(&partial);
        let continued = try_fit_continue(
            &mut continued_learners,
            y.view(),
            w.view(),
            &mut warm_family,
            &control_total,
            &state,
        )
        .expect("continued fit");

        assert_same_path(&cold, &continued);
    }

    fn response(n: usize, seed: u64) -> Array1<f64> {
        let x = design(n, 1, seed);
        Array1::from_shape_fn(n, |i| 2.0 * x[[i, 0]] + 0.3 * ((i % 7) as f64 - 3.0))
    }

    #[test]
    fn gaussian_continuation_equals_a_cold_path() {
        check(Gaussian, response(120, 3), 40, 100);
    }

    #[test]
    fn poisson_continuation_equals_a_cold_path() {
        let counts = Array1::from_shape_fn(120, |i| ((i * 7) % 5) as f64);
        check(Poisson, counts, 30, 90);
    }

    #[test]
    fn a_family_that_calibrates_continues_exactly() {
        // Huber recomputes its delta from the median absolute residual, a pure
        // function of `(y, f, w)`, so resuming reproduces it.
        check(Huber::new(None), response(120, 11), 25, 80);
    }

    #[test]
    fn a_family_with_a_fitted_nuisance_continues_exactly() {
        let counts = Array1::from_shape_fn(150, |i| ((i * 13) % 9) as f64);
        check(NBinomial::new(), counts, 20, 60);
    }

    #[test]
    fn a_path_dependent_calibration_continues_exactly() {
        // PropOdds seeds its coordinate descent from the current thresholds,
        // so the resumed fit restores them instead of recalibrating. Without
        // that restore this test fails on the nuisance path.
        let levels = Array1::from_shape_fn(200, |i| ((i % 4) + 1) as f64);
        check(PropOdds::new(4), levels, 15, 50);
    }

    #[test]
    fn continuing_past_the_retained_length_is_required() {
        let y = response(80, 5);
        let x = design(80, 3, 7);
        let w = Array1::ones(80);
        let control = BoostControl::new(30, 0.1).unwrap();
        let mut family = Gaussian;
        let mut prepared = learners(&x, w.view());
        let path = try_fit(
            &mut prepared,
            y.view(),
            w.view(),
            &mut family,
            &control,
            None,
        )
        .expect("fit");
        let state = WarmStart::from_path(&path);
        let mut again = learners(&x, w.view());
        let error = try_fit_continue(
            &mut again,
            y.view(),
            w.view(),
            &mut family,
            &BoostControl::new(30, 0.1).unwrap(),
            &state,
        )
        .expect_err("continuing to the same length must be refused");
        assert!(
            format!("{error}").contains("slice the retained path"),
            "unexpected message: {error}"
        );
    }

    #[test]
    fn a_warm_state_from_different_data_is_refused() {
        let y = response(90, 5);
        let other = response(90, 6);
        let x = design(90, 3, 7);
        let w = Array1::ones(90);
        let mut family = Gaussian;
        let mut prepared = learners(&x, w.view());
        let path = try_fit(
            &mut prepared,
            y.view(),
            w.view(),
            &mut family,
            &BoostControl::new(20, 0.1).unwrap(),
            None,
        )
        .expect("fit");

        let state = WarmStart::from_path(&path);
        let mut again = learners(&x, w.view());
        let error = try_fit_continue(
            &mut again,
            other.view(),
            w.view(),
            &mut family,
            &BoostControl::new(40, 0.1).unwrap(),
            &state,
        )
        .expect_err("a state fitted on other data must be refused");
        assert!(
            format!("{error}").contains("does not describe this fit"),
            "unexpected message: {error}"
        );
    }

    #[test]
    fn a_warm_state_of_the_wrong_length_is_refused() {
        let y = response(60, 5);
        let x = design(60, 2, 7);
        let w = Array1::ones(60);
        let mut family = Gaussian;
        let mut prepared = learners(&x, w.view());
        let path = try_fit(
            &mut prepared,
            y.view(),
            w.view(),
            &mut family,
            &BoostControl::new(10, 0.1).unwrap(),
            None,
        )
        .expect("fit");
        let short = Array1::<f64>::zeros(30);
        let state = WarmStart {
            fitted: short.view(),
            ..WarmStart::from_path(&path)
        };
        let mut again = learners(&x, w.view());
        let error = try_fit_continue(
            &mut again,
            y.view(),
            w.view(),
            &mut family,
            &BoostControl::new(20, 0.1).unwrap(),
            &state,
        )
        .expect_err("a mismatched fitted vector must be refused");
        assert!(
            format!("{error}").contains("fitted values"),
            "unexpected message: {error}"
        );
    }

    /// A `Gaussian` that counts how many times the engine asked it for a
    /// gradient, which is once per iteration plus once at setup.
    #[derive(Clone)]
    struct CountingGaussian {
        inner: Gaussian,
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl CountingGaussian {
        fn new() -> Self {
            Self {
                inner: Gaussian,
                calls: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            }
        }

        fn count(&self) -> usize {
            self.calls.load(std::sync::atomic::Ordering::Relaxed)
        }
    }

    impl Family for CountingGaussian {
        fn name(&self) -> &'static str {
            "counting-gaussian"
        }

        fn clone_family(&self) -> Box<dyn Family> {
            Box::new(self.clone())
        }

        fn offset(&self, y: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
            self.inner.offset(y, w)
        }

        fn negative_gradient(
            &mut self,
            y: ArrayView1<f64>,
            f: ArrayView1<f64>,
            w: ArrayView1<f64>,
            out: ndarray::ArrayViewMut1<f64>,
        ) {
            self.calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.inner.negative_gradient(y, f, w, out);
        }

        fn risk(&self, y: ArrayView1<f64>, f: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
            self.inner.risk(y, f, w)
        }

        fn held_out_risk(&self, y: ArrayView1<f64>, f: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
            self.inner.held_out_risk(y, f, w)
        }

        fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
            self.inner.check_y(y)
        }
    }

    #[test]
    fn a_continuation_runs_only_the_iterations_it_adds() {
        // The point of resuming: the retained iterations are not replayed.
        let y = response(100, 21);
        let x = design(100, 3, 7);
        let w = Array1::ones(100);
        let control_part = BoostControl::new(30, 0.1).unwrap();
        let control_total = BoostControl::new(100, 0.1).unwrap();

        let mut cold = CountingGaussian::new();
        let mut cold_learners = learners(&x, w.view());
        try_fit(
            &mut cold_learners,
            y.view(),
            w.view(),
            &mut cold,
            &control_total,
            None,
        )
        .expect("cold");
        let cold_calls = cold.count();

        let mut warm = CountingGaussian::new();
        let mut warm_learners = learners(&x, w.view());
        let partial = try_fit(
            &mut warm_learners,
            y.view(),
            w.view(),
            &mut warm,
            &control_part,
            None,
        )
        .expect("partial");
        let after_partial = warm.count();

        let state = WarmStart::from_path(&partial);
        let mut more = learners(&x, w.view());
        try_fit_continue(
            &mut more,
            y.view(),
            w.view(),
            &mut warm,
            &control_total,
            &state,
        )
        .expect("continue");
        let continuation_calls = warm.count() - after_partial;

        // 70 added iterations plus the one setup evaluation, against 101 for
        // the cold path.
        assert_eq!(cold_calls, 101);
        assert_eq!(continuation_calls, 71);
    }

    #[test]
    fn continuing_twice_equals_one_cold_path() {
        // The interactive loop: fit, look, extend, look, extend.
        let y = response(140, 9);
        let x = design(140, 4, 7);
        let w = Array1::ones(140);
        let mut family = Gaussian;

        let mut cold_learners = learners(&x, w.view());
        let cold = try_fit(
            &mut cold_learners,
            y.view(),
            w.view(),
            &mut family,
            &BoostControl::new(120, 0.1).unwrap(),
            None,
        )
        .expect("cold");

        let mut step = learners(&x, w.view());
        let mut path = try_fit(
            &mut step,
            y.view(),
            w.view(),
            &mut family,
            &BoostControl::new(30, 0.1).unwrap(),
            None,
        )
        .expect("first");
        for target in [70u32, 120] {
            let state = WarmStart::from_path(&path);
            let mut more = learners(&x, w.view());
            path = try_fit_continue(
                &mut more,
                y.view(),
                w.view(),
                &mut family,
                &BoostControl::new(target, 0.1).unwrap(),
                &state,
            )
            .expect("continue");
        }
        assert_same_path(&cold, &path);
    }

    // ─── line-search step rule (E1) ──────────────────────────────────────

    fn line_search(mstop: u32, nu: f64) -> BoostControl {
        BoostControl::new(mstop, nu)
            .unwrap()
            .with_step(crate::control::StepRule::LineSearch { max_iter: 4 })
    }

    #[test]
    fn line_search_gamma_reaches_the_weighted_log_linear_mean() {
        let n = 120;
        let x = Array1::linspace(-1.0, 1.0, n);
        let truth = x.mapv(|v: f64| 5000.0_f64.ln() + 0.2 * v);
        let y = truth.mapv(f64::exp);
        let w = Array1::from_shape_fn(n, |i| {
            if i < 8 {
                1000.0 + 100.0 * i as f64
            } else {
                1.0
            }
        });
        let ones = Array1::ones(n);
        let mut learners = vec![
            prepare_scalar_bols(ones.view(), w.view(), 0.0, false, "intercept", None),
            prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None),
        ];
        let path = try_fit(
            &mut learners,
            y.view(),
            w.view(),
            &mut crate::families::GammaReg::new(),
            &line_search(400, 1.0),
            None,
        )
        .unwrap();
        let worst = path
            .fitted
            .iter()
            .zip(&truth)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            worst < 1e-7,
            "weighted gamma mean did not converge: {worst}"
        );
        assert!(path.risk.windows(2).all(|p| p[1] <= p[0] + 1e-7));
    }

    #[test]
    fn line_search_leaves_a_squared_error_fit_bit_identical() {
        // For a squared-error family the least-squares step is already the
        // loss-optimal one, so enabling the search must change nothing --
        // not selection, not coefficients, not one bit of the fitted values.
        let n = 80usize;
        let x = Array1::from_shape_fn(n, |i| ((i as f64) / (n as f64)) * 2.0 - 1.0);
        let y = x.mapv(|v: f64| 1.0 + 0.7 * v)
            + Array1::from_shape_fn(n, |i| (((i * 37) % 19) as f64 / 19.0 - 0.5) * 0.4);
        let w = Array1::ones(n);
        let mut learners_fixed = vec![
            prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None),
            prepare_scalar_bols((2.0 * &x).view(), w.view(), 0.0, true, "x2", None),
        ];
        let mut learners_searched = vec![
            prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None),
            prepare_scalar_bols((2.0 * &x).view(), w.view(), 0.0, true, "x2", None),
        ];
        let mut fixed = Gaussian;
        let mut searched = Gaussian;
        let fixed_path = try_fit(
            &mut learners_fixed,
            y.view(),
            w.view(),
            &mut fixed,
            &BoostControl::new(200, 0.1).unwrap(),
            None,
        )
        .unwrap();
        let searched_path = try_fit(
            &mut learners_searched,
            y.view(),
            w.view(),
            &mut searched,
            &line_search(200, 0.1),
            None,
        )
        .unwrap();
        assert_eq!(fixed_path.selected, searched_path.selected);
        assert_eq!(fixed_path.coefs, searched_path.coefs);
        let worst = fixed_path
            .fitted
            .iter()
            .zip(&searched_path.fitted)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert_eq!(worst, 0.0, "the line search must be inactive for Gaussian");
    }

    #[test]
    fn line_search_keeps_a_large_count_poisson_fit_finite() {
        // Counts around 150 with nu=0.1: the constant step is about
        // nu*mean Newton steps, so iteration 2 pushes the linear predictor
        // past exp overflow and the fixed rule stops with the R-style
        // divergence error. Searching the step keeps every risk finite.
        let n = 200usize;
        let x = Array1::linspace(-1.0, 1.0, n);
        let y = x.mapv(|v| 150.0 + 40.0 * v) + Array1::from_shape_fn(n, |i| (i % 7) as f64);
        let w = Array1::ones(n);
        let make = || {
            vec![
                prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None),
                prepare_scalar_bols(x.mapv(|v| v * v).view(), w.view(), 0.0, true, "x2", None),
            ]
        };
        let control = BoostControl::new(50, 0.1).unwrap();
        let error = try_fit(
            &mut make(),
            y.view(),
            w.view(),
            &mut Poisson,
            &control,
            None,
        )
        .expect_err("the fixed step must diverge at these counts");
        assert!(
            error.to_string().contains("risk became non-finite"),
            "{error}"
        );
        let text = error.to_string();
        assert!(
            text.contains("Poisson risk became non-finite after boosting iteration "),
            "{text}"
        );
        assert!(text.contains("nu = 0.1"), "{text}");
        assert!(text.contains("step=\"line_search\""), "{text}");

        let path = try_fit(
            &mut make(),
            y.view(),
            w.view(),
            &mut Poisson,
            &line_search(50, 0.1),
            None,
        )
        .expect("the line search must keep the fit finite");
        assert!(path.risk.iter().all(|r| r.is_finite()));
        assert!(
            *path.risk.last().unwrap() < path.risk[0],
            "the searched path must reduce the risk: {} -> {}",
            path.risk[0],
            path.risk.last().unwrap()
        );
    }

    #[test]
    fn line_search_reaches_a_step_below_the_grid_for_counts_near_1e5() {
        // Counts around 2e5 (house values stored as whole numbers): the
        // loss-optimal multiple of the least-squares direction is about
        // 1/mu = 5e-6, far below the grid's smallest point 0.125, at which
        // every step overflows. The search must halve down to it.
        let n = 200usize;
        let x = Array1::linspace(-1.0, 1.0, n);
        let y = x.mapv(|v| 2e5 + 8e4 * v) + Array1::from_shape_fn(n, |i| (i % 7) as f64);
        let w = Array1::ones(n);
        let mut learners = vec![prepare_scalar_bols(
            x.view(),
            w.view(),
            0.0,
            true,
            "x",
            None,
        )];
        let path = try_fit(
            &mut learners,
            y.view(),
            w.view(),
            &mut Poisson,
            &line_search(50, 0.1),
            None,
        )
        .expect("the line search must keep the fit finite");
        assert!(path.risk.iter().all(|r| r.is_finite()));
        assert!(
            *path.risk.last().unwrap() < path.risk[0],
            "the searched path must reduce the risk: {} -> {}",
            path.risk[0],
            path.risk.last().unwrap()
        );
    }

    #[test]
    fn line_search_reaches_a_rare_event_risk_in_fewer_iterations() {
        // A 2% event rate: the curvature is 4p(1-p) ~ 0.08, so the fixed
        // step is about 12x smaller than a Newton step and the fixed rule
        // spends thousands of iterations. Searching the step must reach a
        // lower risk within 50 iterations than the fixed rule manages in
        // 300.
        let n = 500usize;
        let x = Array1::linspace(0.0, 1.0, n);
        let y = Array1::from_shape_fn(n, |i| if i < 10 { 1.0 } else { 0.0 });
        let w = Array1::ones(n);
        let mut fixed_learners = vec![prepare_scalar_bols(
            x.view(),
            w.view(),
            0.0,
            true,
            "x",
            None,
        )];
        let mut searched_learners = vec![prepare_scalar_bols(
            x.view(),
            w.view(),
            0.0,
            true,
            "x",
            None,
        )];
        let mut fixed = crate::families::Binomial;
        let mut searched = crate::families::Binomial;
        let fixed_path = try_fit(
            &mut fixed_learners,
            y.view(),
            w.view(),
            &mut fixed,
            &BoostControl::new(300, 0.1).unwrap(),
            None,
        )
        .unwrap();
        let searched_path = try_fit(
            &mut searched_learners,
            y.view(),
            w.view(),
            &mut searched,
            &line_search(50, 0.1),
            None,
        )
        .unwrap();
        // At an equal budget the searched path is strictly ahead, and 50
        // searched iterations reach the fixed rule's 300-iteration risk to
        // within a thousandth of the reduction either of them achieves.
        let mut short_learners = vec![prepare_scalar_bols(
            x.view(),
            w.view(),
            0.0,
            true,
            "x",
            None,
        )];
        let mut short_fixed = crate::families::Binomial;
        let short_path = try_fit(
            &mut short_learners,
            y.view(),
            w.view(),
            &mut short_fixed,
            &BoostControl::new(50, 0.1).unwrap(),
            None,
        )
        .unwrap();
        assert!(
            *searched_path.risk.last().unwrap() < *short_path.risk.last().unwrap(),
            "at an equal budget the searched risk ({}) must beat the fixed one ({})",
            searched_path.risk.last().unwrap(),
            short_path.risk.last().unwrap()
        );
        let total_reduction = fixed_path.risk[0] - *fixed_path.risk.last().unwrap();
        assert!(
            *searched_path.risk.last().unwrap()
                <= *fixed_path.risk.last().unwrap() + 1e-3 * total_reduction,
            "50 searched iterations ({}) must reach the 300 fixed ones ({})",
            searched_path.risk.last().unwrap(),
            fixed_path.risk.last().unwrap()
        );
        // Each searched step moves downhill on the risk it minimizes, so the
        // path the search walks is non-increasing.
        assert!(searched_path
            .risk
            .windows(2)
            .all(|step| step[1] <= step[0] + 1e-9 * step[0].abs()));
    }

    #[test]
    fn line_search_coefficients_replay_the_fitted_path() {
        // The stored coefficient is rho*beta, so replaying it with only `nu`
        // in front must reconstruct the fitted values exactly. A stored
        // coefficient missing rho would replay nu*beta and miss by nu*(1-rho).
        let n = 120usize;
        let x = Array1::linspace(0.0, 1.0, n);
        let y = x.mapv(|v: f64| 2.0 + (3.0 * v).sin())
            + Array1::from_shape_fn(n, |i| (i % 5) as f64 * 0.1);
        let w = Array1::ones(n);
        let mut learners = vec![prepare_scalar_bols(
            x.view(),
            w.view(),
            0.0,
            false,
            "x",
            None,
        )];
        let mut family = Poisson;
        let path = try_fit(
            &mut learners,
            y.view(),
            w.view(),
            &mut family,
            &line_search(40, 0.1),
            None,
        )
        .unwrap();
        let nu = 0.1;
        let replayed: Vec<f64> = (0..n)
            .map(|i| path.offset + nu * path.coefs.iter().map(|coef| coef[0] * x[i]).sum::<f64>())
            .collect();
        let worst = path
            .fitted
            .iter()
            .zip(&replayed)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(worst < 1e-9, "coefficient replay misses by {worst:.3e}");
    }

    #[test]
    fn line_search_keeps_a_tree_coefficient_decodable() {
        // A tree's coefficient vector encodes its node count, split variables
        // and split points. The line search used to multiply the whole vector
        // by rho, so a count of 3 nodes became 0.6 -> 0 nodes and prediction
        // indexed an empty tree. Only the leaf means carry rho.
        let n = 200usize;
        let x = Array1::linspace(0.0, 1.0, n);
        let y = x.mapv(|v: f64| 600.0 + 200.0 * f64::from(u8::from(v > 0.5)));
        let w = Array1::ones(n);
        let design = x.clone().insert_axis(ndarray::Axis(1));
        let mut learners = vec![crate::learners::prepare_btree(
            design.view(),
            w.view(),
            7,
            20,
            1,
            None,
            "x",
        )];
        let mut family = Poisson;
        let path = try_fit(
            &mut learners,
            y.view(),
            w.view(),
            &mut family,
            &line_search(30, 0.1),
            None,
        )
        .unwrap();
        let nu = 0.1;
        let mut replayed = vec![path.offset; n];
        for coef in &path.coefs {
            let tree = crate::learners::FittedTree::from_coef_vec(ndarray::ArrayView1::from(
                coef.as_slice(),
            ));
            let pred = tree.predict(design.view());
            for i in 0..n {
                replayed[i] += nu * pred[i];
            }
        }
        let worst = path
            .fitted
            .iter()
            .zip(&replayed)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(worst < 1e-9, "tree replay misses by {worst:.3e}");
    }
}

#[cfg(test)]
mod penalty_drift_tests {
    use super::*;
    use crate::learners::{prepare_bbs, smoother_spectrum};

    #[test]
    fn spectrum_reproduces_the_hat_trace_and_drift_rises_with_selections() {
        let n = 300;
        let x = Array1::from_shape_fn(n, |i| (i as f64 * 0.731).sin() * 2.0);
        let w = Array1::from_shape_fn(n, |i| 1.0 + (i % 3) as f64);
        let learner = prepare_bbs(x.view(), w.view(), 20, 3, 2, 4.0, false, "bbs(x)");
        let spectrum = smoother_spectrum(&learner).unwrap();
        let trace: f64 = spectrum.iter().sum();
        let df: f64 = spectrum.iter().map(|s| 2.0 * s - s * s).sum();
        assert!((trace - learner.hat_factor().unwrap().tr_h).abs() < 1e-8);
        // mboost's df2lambda definition, 2 tr(S) - tr(S^2), at which the
        // learner was calibrated.
        assert!((df - 4.0).abs() < 1e-4, "{df}");

        let learners = vec![learner];
        let at = |m: usize| penalty_drift(&learners, &vec![0; m], 0.1).unwrap()[0].unwrap();
        let (few, many, huge) = (at(10), at(1_000), at(10_000_000));
        assert_eq!(few.selections, 10);
        assert!(few.boosted < many.boosted && many.boosted < huge.boosted);
        assert_eq!(huge.unpenalized, 24.0);
        assert!(huge.share() > 0.9, "{huge:?}");
        assert!(few.share() == 0.0);
        assert!(penalty_drift(&learners, &[1], 0.1).is_err());
        assert!(penalty_drift(&learners, &[0], 0.0).is_err());
    }
}
