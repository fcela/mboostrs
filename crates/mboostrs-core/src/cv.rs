//! Cross-validation: fold generation and cvrisk computation.
//!
//! Implements R's `mboost::cv()` and `mboost::cvrisk()` semantics:
//! - `cv()`: generate (n, B) fold weight matrix (kfold, bootstrap, subsampling)
//! - `cvrisk()`: for each fold, re-prepare learners with fold weights, refit,
//!   compute OOB risk at every iteration, return (B, mstop+1) risk matrix

use ndarray::{Array1, Array2, ArrayView1};
use rand::RngExt;

/// Connected row groups: sharing any declared group keeps rows together.
/// Codes are data encodings, not numeric measurements. This supports crossed
/// grouping variables without leaking either marginal across folds.
pub fn group_components(
    codes: ndarray::ArrayView2<f64>,
) -> Result<Vec<usize>, crate::MboostrsError> {
    use std::collections::HashMap;
    let (n, p) = codes.dim();
    if n == 0 || p == 0 || codes.iter().any(|v| !v.is_finite()) {
        return Err(crate::MboostrsError::Shape(
            "group codes must be nonempty and finite".into(),
        ));
    }
    fn root(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    let mut parents: Vec<usize> = (0..n).collect();
    let mut seen = HashMap::new();
    for i in 0..n {
        for j in 0..p {
            let value = if codes[[i, j]] == 0.0 {
                0.0
            } else {
                codes[[i, j]]
            };
            if let Some(&previous) = seen.get(&(j, value.to_bits())) {
                let a = root(&mut parents, i);
                let b = root(&mut parents, previous);
                parents[a.max(b)] = a.min(b);
            } else {
                seen.insert((j, value.to_bits()), i);
            }
        }
    }
    let mut labels = HashMap::new();
    Ok((0..n)
        .map(|i| {
            let r = root(&mut parents, i);
            let next = labels.len();
            *labels.entry(r).or_insert(next)
        })
        .collect())
}
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use rayon::prelude::*;

use crate::control::BoostControl;
use crate::engine::{try_fit_selected, try_fit_with};
use crate::families::Family;
use crate::learners::PreparedLearner;
use crate::MboostrsError;

/// Fold type for cross-validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FoldType {
    /// `k`-fold: the data are split into `b` disjoint folds.
    KFold,
    /// Bootstrap: each fold draws `n` observations with replacement.
    Bootstrap,
    /// Subsampling: each fold draws a fraction of observations without
    /// replacement.
    Subsampling,
}

impl FoldType {
    /// Human-readable description of the resampling scheme.
    pub fn label(&self) -> &'static str {
        match self {
            FoldType::KFold => "cross-validation",
            FoldType::Bootstrap => "bootstrap",
            FoldType::Subsampling => "subsampling",
        }
    }
}

/// Specification for generating cross-validation folds.
#[derive(Clone, Debug)]
pub struct FoldSpec {
    /// Which resampling scheme generates the folds.
    pub fold_type: FoldType,
    /// Number of folds/replications, or if 0, the number of training-set
    /// observations for `Probabilistic` bootstrapping where probabilities are
    /// not fixed.
    pub b: usize,
    /// Expected training-set probability for `Bootstrap`/`Subsampling` folds,
    /// or fixed row probability matrix pointer for `KFold`.
    pub prob: f64,
    /// Seed for fold generation.
    pub seed: u64,
}

impl Default for FoldSpec {
    fn default() -> Self {
        Self {
            fold_type: FoldType::Bootstrap,
            b: 25,
            prob: 0.5,
            seed: 42,
        }
    }
}

impl FoldSpec {
    /// Create a fold specification, panicking if the parameters are invalid.
    pub fn new(fold_type: FoldType, b: usize, prob: f64, seed: u64) -> Self {
        Self::try_new(fold_type, b, prob, seed).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Construct a validated fold specification.
    pub fn try_new(
        fold_type: FoldType,
        b: usize,
        prob: f64,
        seed: u64,
    ) -> Result<Self, MboostrsError> {
        if b == 0 {
            return Err(MboostrsError::Control(
                "number of folds/replicates b must be > 0".to_string(),
            ));
        }
        if !prob.is_finite() {
            return Err(MboostrsError::Control(
                "fold sampling probability must be finite".to_string(),
            ));
        }
        if matches!(fold_type, FoldType::Subsampling) && !(0.0 < prob && prob < 1.0) {
            return Err(MboostrsError::Control(
                "subsampling probability must satisfy 0 < prob < 1".to_string(),
            ));
        }
        Ok(Self {
            fold_type,
            b,
            prob,
            seed,
        })
    }
}

/// Result of cross-validated risk estimation.
#[derive(Clone, Debug)]
pub struct CvRiskResult {
    /// (B, mstop+1) matrix of normalized OOB risks per fold per iteration.
    /// risk_matrix[[i, m]] = normalized OOB risk for fold i at iteration m.
    pub risk_matrix: Array2<f64>,
    /// Aggregated risk path: column means of risk_matrix, matching R's
    /// `cvrisk` (which aggregates with `colMeans`).
    pub aggregate: Array1<f64>,
    /// The grid of iteration indices (0..=mstop).
    pub grid: Vec<usize>,
    /// Description of the fold type.
    pub fold_type: String,
}

/// Held-out predictions from a fold whose preprocessing was fitted on its
/// training rows. The provider may rebuild a formula or a dataframe recipe;
/// family loss evaluation and cross-validation aggregation remain in the core.
pub struct CvPredictionFold {
    /// External response coding on the held-out rows.
    pub response: Array1<f64>,
    /// Held-out observation weights, before any multivariate expansion.
    pub weights: Array1<f64>,
    /// Link predictions at iterations 0 through mstop, one iteration per row.
    pub predictions: Array2<f64>,
    /// Training-fitted nuisance values at the same iterations: one value for
    /// a scalar nuisance or the full vector for a vector nuisance.
    pub nuisance: Vec<Vec<f64>>,
}

/// Cross-validated family risk when each fold has its own fitted recipe.
///
/// `provider` supplies one fold at a time, allowing a language frontend to
/// parse and replay its training-only transforms without retaining every
/// fold model. Fitting and prediction use the native engine; this function
/// validates, restores training-fitted nuisance state, scores the held-out
/// paths and aggregates them with equal fold weight.
pub fn try_cvrisk_foldwise(
    family: &dyn Family,
    n_folds: usize,
    mstop: usize,
    mut provider: impl FnMut(usize) -> Result<CvPredictionFold, MboostrsError>,
) -> Result<CvRiskResult, MboostrsError> {
    if n_folds == 0 || mstop == 0 {
        return Err(MboostrsError::Control(
            "CV needs folds and a positive mstop".into(),
        ));
    }
    let mut risks = Array2::zeros((n_folds, mstop + 1));
    for b in 0..n_folds {
        let fold = provider(b)?;
        let mut fam = family.clone_family();
        fam.check_y(fold.response.view())?;
        let encoded = fam.encode_response(fold.response.view());
        let y = encoded.as_ref().unwrap_or(&fold.response);
        let coords = fam.n_from_y(y.view());
        let span = fam.flat_response_span();
        if coords == 0
            || span == 0
            || coords % span != 0
            || fold.weights.len() != coords / span
            || fold.predictions.dim() != (mstop + 1, coords)
        {
            return Err(MboostrsError::Shape(format!(
                "CV fold {b}: response, weights and prediction path have incompatible shapes"
            )));
        }
        let sumw = fold.weights.sum();
        if !sumw.is_finite()
            || sumw <= 0.0
            || fold.weights.iter().any(|w| !w.is_finite() || *w < 0.0)
            || y.iter().any(|v| !v.is_finite())
        {
            return Err(MboostrsError::Control(format!(
                "CV fold {b}: responses must be finite and evaluation weights non-negative with positive sum"
            )));
        }
        let vector = fam.nuisance_vec().is_some();
        // Huber's adaptive delta is unset on a fresh family template until
        // calibration; its fitted path still carries a scalar nuisance.
        let scalar = fam.nuisance().is_some() || (fam.calibrates() && !vector);
        if (scalar || vector) && fold.nuisance.len() != mstop + 1 {
            return Err(MboostrsError::Shape(format!(
                "CV fold {b}: a training nuisance value is required at every iteration"
            )));
        }
        let weights = Array1::from_shape_fn(coords, |i| fold.weights[i / span]);
        for m in 0..=mstop {
            if let Some(values) = fold.nuisance.get(m) {
                if values.iter().any(|v| !v.is_finite()) || (scalar && values.len() != 1) {
                    return Err(MboostrsError::Shape(format!(
                        "CV fold {b}, iteration {m}: invalid training nuisance state"
                    )));
                }
                if scalar {
                    fam.set_nuisance(values[0]);
                }
                if vector {
                    fam.set_nuisance_vec(values.clone());
                }
            }
            risks[[b, m]] =
                fam.held_out_risk(y.view(), fold.predictions.row(m), weights.view()) / sumw;
        }
    }
    let (curve, _) = crate::scoring::select_score_path(risks.view())?;
    Ok(CvRiskResult {
        risk_matrix: risks,
        aggregate: Array1::from_vec(curve),
        grid: (0..=mstop).collect(),
        fold_type: "fold-local preprocessing".into(),
    })
}

/// Generate a cross-validation fold weight matrix, matching R's `cv()`.
///
/// Returns an (n, B) weight matrix where each column is one fold:
/// - **KFold**: each column has 0 for hold-out rows, original weights for training.
///   Each observation is held out exactly once across B=k folds.
/// - **Bootstrap**: each column is a multinomial draw (integer counts).
///   Observations with count 0 are OOB.
/// - **Subsampling**: each column has 0 for hold-out, original weights for training.
///   `floor(n * prob)` observations are in-bag per fold.
pub fn cv(spec: &FoldSpec, n: usize, weights: &[f64]) -> Array2<f64> {
    try_cv(spec, n, weights).unwrap_or_else(|e| panic!("{e}"))
}

/// Fallible fold generation with shape, weight, and sampling validation.
pub fn try_cv(spec: &FoldSpec, n: usize, weights: &[f64]) -> Result<Array2<f64>, MboostrsError> {
    // Validate specifications created as struct literals as well as via
    // FoldSpec::try_new.
    let spec = FoldSpec::try_new(spec.fold_type, spec.b, spec.prob, spec.seed)?;
    if n == 0 {
        return Err(MboostrsError::Shape(
            "cv requires at least one observation".to_string(),
        ));
    }
    if weights.len() != n {
        return Err(MboostrsError::Shape(format!(
            "cv: weights length {} does not match n {n}",
            weights.len()
        )));
    }
    if weights.iter().any(|value| !value.is_finite()) {
        return Err(MboostrsError::Shape(
            "cv: weights contain NaN or infinite values".to_string(),
        ));
    }
    if weights.iter().any(|&value| value < 0.0) {
        return Err(MboostrsError::Shape(
            "cv: weights must be non-negative".to_string(),
        ));
    }
    if matches!(spec.fold_type, FoldType::KFold) && spec.b > n {
        return Err(MboostrsError::Control(format!(
            "k-fold cross-validation requires k <= n; got k={} and n={n}",
            spec.b
        )));
    }
    if matches!(spec.fold_type, FoldType::Bootstrap) && weights.iter().all(|&w| w == 0.0) {
        return Err(MboostrsError::Control(
            "bootstrap folds are undefined when all observation weights are zero".to_string(),
        ));
    }

    let mut rng = ChaCha8Rng::seed_from_u64(spec.seed);
    let folds = match spec.fold_type {
        FoldType::KFold => cv_kfold(n, spec.b, weights, &mut rng),
        FoldType::Bootstrap => cv_bootstrap(n, spec.b, weights, &mut rng),
        FoldType::Subsampling => cv_subsampling(n, spec.b, spec.prob, weights, &mut rng),
    };
    Ok(folds)
}

fn cv_kfold(n: usize, k: usize, weights: &[f64], rng: &mut ChaCha8Rng) -> Array2<f64> {
    let mut folds = Array2::zeros((n, k));
    let holdout_per_fold = n / k;

    let mut perm: Vec<usize> = (0..n).collect();
    permute(&mut perm, rng);

    for fold_idx in 0..k {
        let start = fold_idx * holdout_per_fold;
        let end = if fold_idx == k - 1 {
            n
        } else {
            start + holdout_per_fold
        };
        for row in 0..n {
            if row >= start && row < end {
                folds[[perm[row], fold_idx]] = 0.0;
            } else {
                folds[[perm[row], fold_idx]] = weights[perm[row]];
            }
        }
    }
    folds
}

fn cv_bootstrap(n: usize, b: usize, weights: &[f64], rng: &mut ChaCha8Rng) -> Array2<f64> {
    let sw: f64 = weights.iter().sum();
    let mut probs = vec![0.0f64; n];
    if sw > 0.0 {
        for i in 0..n {
            probs[i] = weights[i] / sw;
        }
    }

    // Precompute the cumulative distribution once (same across all folds) so
    // each draw is an O(log n) binary search instead of an O(n) linear scan,
    // turning fold generation from O(B·n²) into O(B·n·log n). One
    // `rng.random::<f64>()` per draw and the same index selection are kept, so
    // the generated folds are bit-identical to the linear-scan version.
    let mut cdf = vec![0.0f64; n];
    let mut acc = 0.0f64;
    for i in 0..n {
        acc += probs[i];
        cdf[i] = acc;
    }

    let mut folds = Array2::zeros((n, b));
    for fold_idx in 0..b {
        for _ in 0..n {
            let draw = multinomial_draw(&cdf, rng);
            folds[[draw, fold_idx]] += 1.0;
        }
    }
    folds
}

fn cv_subsampling(
    n: usize,
    b: usize,
    prob: f64,
    weights: &[f64],
    rng: &mut ChaCha8Rng,
) -> Array2<f64> {
    let k = (n as f64 * prob).floor() as usize;
    let mut folds = Array2::zeros((n, b));

    for fold_idx in 0..b {
        let mut perm: Vec<usize> = (0..n).collect();
        permute(&mut perm, rng);
        for row in 0..n {
            if row < k {
                folds[[perm[row], fold_idx]] = weights[perm[row]];
            } else {
                folds[[perm[row], fold_idx]] = 0.0;
            }
        }
    }
    folds
}

fn permute(v: &mut [usize], rng: &mut ChaCha8Rng) {
    let n = v.len();
    for i in (1..n).rev() {
        let j = rng.random_range(0..=i);
        v.swap(i, j);
    }
}

/// Draw one index from a multinomial via binary search on a precomputed CDF.
/// `cdf[i]` is the cumulative probability through index `i` (non-decreasing).
/// Returns the first index whose CDF exceeds the uniform draw — identical to
/// a linear scan of the per-index probabilities, including the handling of
/// zero-probability entries (never selected) and the `u >= cdf[n-1]` tail.
fn multinomial_draw(cdf: &[f64], rng: &mut ChaCha8Rng) -> usize {
    cdf_index(cdf, rng.random::<f64>())
}

/// First index whose cumulative probability exceeds `u`, clamped to the last
/// index for the `u >= cdf[n-1]` tail. Equivalent to the first index where a
/// running sum of the per-index probabilities exceeds `u`.
#[inline]
fn cdf_index(cdf: &[f64], u: f64) -> usize {
    cdf.partition_point(|&c| c <= u).min(cdf.len() - 1)
}

/// Abstraction over the prepared learners used to refit during cross-
/// validation, so `cvrisk` can drive both GLM- and GAM-style fits.
pub trait LearnerPreparer: Sync + Send {
    /// Prepare the learners for the given observation weights.
    fn prepare(&self, weights: ArrayView1<f64>) -> Vec<PreparedLearner>;
    /// The family the prepared learners will be fit with.
    fn family(&self) -> Box<dyn Family>;
}

fn validate_cvrisk_inputs<F: Family + ?Sized>(
    family: &F,
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    folds: &Array2<f64>,
    offset_override: Option<f64>,
) -> Result<(), MboostrsError> {
    if y.iter().any(|value| !value.is_finite()) {
        return Err(MboostrsError::BadResponse(
            "cvrisk".to_string(),
            "response contains NaN or infinite values".to_string(),
        ));
    }
    family.check_y(y)?;
    if family.requires_unit_weights() {
        // cvrisk necessarily fits each fold with non-unit (bootstrap/
        // subsampling) case weights, which this family does not yet
        // support -- see `Family::requires_unit_weights` and `Auc`'s doc
        // comment. Fail fast here with a clear, cvrisk-specific message
        // rather than the generic per-fold "unit weights only" error each
        // fold's fit would otherwise raise.
        return Err(MboostrsError::BadResponse(
            family.name().to_string(),
            format!(
                "{} does not yet support cvrisk (bootstrap/subsampling case weights); \
                 fit it directly instead",
                family.name()
            ),
        ));
    }
    if family.requires_binary_weights() {
        if weights.iter().any(|&value| value != 0.0 && value != 1.0) {
            return Err(MboostrsError::BadResponse(
                family.name().to_string(),
                format!("weights must be either 0 or 1 for family {}", family.name()),
            ));
        }
        // Subsampling/k-fold cvrisk folds are exactly {0, 1} and are
        // supported; bootstrap-style folds carry integer resample counts
        // (values > 1), which violate this family's weight domain -- reject
        // with a clear message rather than the generic per-pair `w[i]*w[j]`
        // silently double-counting a resampled observation.
        if folds.iter().any(|&value| value != 0.0 && value != 1.0) {
            return Err(MboostrsError::BadResponse(
                family.name().to_string(),
                format!(
                    "{} requires {{0, 1}} cvrisk folds (subsampling/k-fold); \
                     bootstrap-style folds with integer resample counts are not supported",
                    family.name()
                ),
            ));
        }
    }
    let n = family.n_from_y(y);
    if n == 0 {
        return Err(MboostrsError::Shape(
            "cvrisk requires at least one observation".to_string(),
        ));
    }
    if weights.len() != n {
        return Err(MboostrsError::Shape(format!(
            "cvrisk: weights length {} does not match {n} observations",
            weights.len()
        )));
    }
    if weights.iter().any(|value| !value.is_finite())
        || folds.iter().any(|value| !value.is_finite())
    {
        return Err(MboostrsError::Shape(
            "cvrisk weights and folds must be finite".to_string(),
        ));
    }
    if weights.iter().any(|&value| value < 0.0) || folds.iter().any(|&value| value < 0.0) {
        return Err(MboostrsError::Shape(
            "cvrisk weights and folds must be non-negative".to_string(),
        ));
    }
    if folds.nrows() != n {
        return Err(MboostrsError::Shape(format!(
            "cvrisk fold matrix has {} rows but expected {n}",
            folds.nrows()
        )));
    }
    if folds.ncols() == 0 {
        return Err(MboostrsError::Shape(
            "cvrisk requires at least one fold column".to_string(),
        ));
    }
    if offset_override.is_some_and(|offset| !offset.is_finite()) {
        return Err(MboostrsError::Control(
            "cvrisk offset override must be finite".to_string(),
        ));
    }
    for fold_idx in 0..folds.ncols() {
        let fold = folds.column(fold_idx);
        if !fold.iter().any(|&value| value > 0.0) {
            return Err(MboostrsError::Control(format!(
                "cross-validation fold {fold_idx} has no in-bag observations"
            )));
        }
        if !(0..n).any(|i| fold[i] == 0.0 && weights[i] > 0.0) {
            return Err(MboostrsError::Control(format!(
                "cross-validation fold {fold_idx} has no positive-weight OOB observations"
            )));
        }
    }
    Ok(())
}

/// Cross-validated risk: one fit per column of `folds` (that fold's in-bag
/// weights), scored on the rows it leaves out, at every iteration up to `mstop`.
pub fn cvrisk(
    preparer: &dyn LearnerPreparer,
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    control: &BoostControl,
    folds: &Array2<f64>,
    offset_override: Option<f64>,
    parallel: bool,
    fold_label: &str,
) -> CvRiskResult {
    try_cvrisk(
        preparer,
        y,
        weights,
        control,
        folds,
        offset_override,
        None,
        None,
        parallel,
        fold_label,
    )
    .unwrap_or_else(|e| panic!("{e}"))
}

/// Fallible dynamically-dispatched cross-validation entry point.
#[allow(clippy::too_many_arguments)]
pub fn try_cvrisk(
    preparer: &dyn LearnerPreparer,
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    control: &BoostControl,
    folds: &Array2<f64>,
    offset_override: Option<f64>,
    offset_vec: Option<ArrayView1<f64>>,
    judge: Option<&Array2<bool>>,
    parallel: bool,
    fold_label: &str,
) -> Result<CvRiskResult, MboostrsError> {
    let validation_family = preparer.family();
    validate_cvrisk_inputs(
        validation_family.as_ref(),
        y,
        weights,
        folds,
        offset_override,
    )?;
    if let Some(mask) = judge {
        if mask.shape() != folds.shape() {
            return Err(MboostrsError::Shape(format!(
                "judge mask has shape {:?}, expected {:?} (the fold matrix)",
                mask.shape(),
                folds.shape()
            )));
        }
    }
    // Re-encode the response once (Binomial {0,1}->{-1,+1}; no-op otherwise) so
    // both the per-fold inbag fit AND the out-of-bag risk use the family's
    // internal coding. The OOB risk is computed here from `y` directly, so the
    // recoding must happen at this level — not only inside the inner fit.
    let y_encoded = preparer.family().encode_response(y);
    let y = match y_encoded.as_ref() {
        Some(a) => a.view(),
        None => y.view(),
    };
    let n = {
        let fam = preparer.family();
        fam.n_from_y(y)
    };
    if let Some(vector) = offset_vec {
        // The same contract the fit enforces: one known contribution per
        // fitted coordinate, all finite. A `log(0)` of zero exposure is
        // refused rather than scored as -inf.
        if vector.len() != n {
            return Err(MboostrsError::Shape(format!(
                "offset has {} entries but the response implies {} fitted coordinates",
                vector.len(),
                n
            )));
        }
        if vector.iter().any(|value| !value.is_finite()) {
            return Err(MboostrsError::Control(
                "offset must be finite in every entry; log(0) of a zero-exposure \
                 row is -inf, and such a row cannot enter the linear predictor"
                    .to_string(),
            ));
        }
    }
    let b = folds.ncols();
    let mstop = control.mstop as usize;
    let n_iter = mstop + 1;

    // `parallel=false` runs folds sequentially. Callers that already parallelize
    // an outer loop (e.g. confint's bootstrap thread pool) pass false to avoid
    // nested rayon-over-threads oversubscription.

    let compute_fold = |fold_idx: usize| -> Result<(usize, Vec<f64>), MboostrsError> {
        let fold_weights: Vec<f64> = (0..n).map(|i| folds[[i, fold_idx]]).collect();
        let fold_w = Array1::from_vec(fold_weights);

        let mut oob_weights = weights.to_owned();
        for i in 0..n {
            if fold_w[i] > 0.0 {
                oob_weights[i] = 0.0;
            }
            if let Some(mask) = judge {
                if !mask[[i, fold_idx]] {
                    oob_weights[i] = 0.0;
                }
            }
        }
        let oob_sum: f64 = oob_weights.sum();

        let mut learners = preparer.prepare(fold_w.view());
        let mut family = preparer.family();

        // Inbag fit. For Gaussian + Gram-eligible learners, use the fast
        // covariance-update path for the (expensive) inbag selection, then
        // build the per-iteration increments from the selected learner with
        // one matvec each — this is what the OOB-risk walk below consumes.
        // Otherwise fall back to the general path's stored increments.
        // A per-observation offset starts the fit at a vector the Gram
        // path's scalar-intercept algebra cannot represent, so a vector
        // offset forces the general path here too.
        let gram = if family.supports_gram_update() && offset_vec.is_none() {
            crate::engine::try_fit_gaussian_gram(
                &mut learners,
                y,
                fold_w.view(),
                control,
                offset_override,
                false,
                true,
                true,
            )
        } else {
            None
        };

        let nu = control.nu;
        #[allow(clippy::type_complexity)]
        let (start, increments, nuisance_path, nuisance_vec_path): (
            Vec<f64>,
            Vec<Vec<f64>>,
            Vec<Option<f64>>,
            Vec<Vec<f64>>,
        ) = match gram {
            Some(path) => {
                let mut incs: Vec<Vec<f64>> = Vec::with_capacity(mstop);
                for m in 0..mstop {
                    let idx = path.selected[m];
                    let coef = &path.coefs[m];
                    let mut inc = vec![0.0f64; n];
                    match &learners[idx] {
                        PreparedLearner::BlockBols { design, .. }
                        | PreparedLearner::BmonoBbs { design, .. } => {
                            // A view rather than a copy of the coefficients.
                            // `dot` still allocates its result: replacing it
                            // with a hand-written triple loop that writes
                            // into `inc` removes that allocation and is
                            // *slower* -- 28.1 ms against 26.3 on a
                            // 4,000-row, 12-learner, 150-iteration
                            // five-fold run -- because `dot` reaches
                            // matrixmultiply and the loop does not. Measured
                            // before keeping, per P3.4's own preamble.
                            let pred = design.dot(&ndarray::ArrayView1::from(&coef[..]));
                            for i in 0..n {
                                inc[i] = nu * pred[i];
                            }
                        }
                        PreparedLearner::ScalarBols { design, .. } => {
                            let c = coef[0];
                            for i in 0..n {
                                inc[i] = nu * design[i] * c;
                            }
                        }
                        tree @ PreparedLearner::Btree { .. } => {
                            tree.predict_into(coef, &mut inc);
                            for value in inc.iter_mut() {
                                *value *= nu;
                            }
                        }
                        _ => unreachable!("Gram path accepts linear learners and trees"),
                    }
                    incs.push(inc);
                }
                // Gaussian fast path has no scalar or vector nuisance.
                (vec![path.offset; n], incs, Vec::new(), Vec::new())
            }
            None => {
                let path = match offset_vec {
                    Some(vector) => crate::engine::try_fit_with_vector_offset(
                        &mut learners,
                        y,
                        fold_w.view(),
                        family.as_mut(),
                        control,
                        vector,
                        true,
                    )?,
                    None => try_fit_with(
                        &mut learners,
                        y,
                        fold_w.view(),
                        family.as_mut(),
                        control,
                        offset_override,
                        true,
                    )?,
                };
                let start = match offset_vec {
                    // The fold starts at the caller's known contribution; the
                    // path's scalar field describes a start this fit never
                    // used.
                    Some(vector) => vector.to_vec(),
                    None => vec![path.offset; n],
                };
                (
                    start,
                    path.increments,
                    path.nuisance_path,
                    path.nuisance_vec_path,
                )
            }
        };

        // Set the family's nuisance for the OOB risk at boosting step `k`.
        // R's cvrisk re-estimates the nuisance on the INBAG (fitting) sample
        // at each step, then evaluates the OOB risk with it; reusing the
        // stored inbag per-step value matches that. Families without a
        // scalar nuisance (path entry `None`, or the Gaussian fast path with
        // an empty path) fall back to calibrating on the OOB weights, which
        // for them is either a no-op or the correct unweighted statistic.
        // For families with vector nuisance (e.g. PropOdds), the per-step
        // threshold vector is restored from nuisance_vec_path so that
        // truncated-prediction CV uses the correct iteration-specific
        // thresholds rather than the final ones.
        let set_nuisance_at = |family: &mut dyn Family, k: usize, f: ArrayView1<f64>| {
            // Restore scalar nuisance from path if available.
            match nuisance_path.get(k).copied().flatten() {
                Some(v) => family.set_nuisance(v),
                None => family.calibrate(y, f, oob_weights.view()),
            }
            // Restore vector nuisance from path if available.
            if let Some(vv) = nuisance_vec_path.get(k) {
                if !vv.is_empty() {
                    family.set_nuisance_vec(vv.clone());
                }
            }
        };

        let mut oob_risks = Vec::with_capacity(n_iter);
        let mut f = Array1::from_vec(start);

        set_nuisance_at(family.as_mut(), 0, f.view());
        let raw_risk = family.held_out_risk(y, f.view(), oob_weights.view());
        // R's cvrisk divides every family's OOB risk by the sum of OOB
        // weights (`oobrisk / colSums(OOBweights)`), unconditionally.
        oob_risks.push(if oob_sum > 0.0 {
            raw_risk / oob_sum
        } else {
            f64::NAN
        });

        for m in 0..mstop {
            for i in 0..n {
                f[i] += increments[m][i];
            }
            set_nuisance_at(family.as_mut(), m + 1, f.view());
            let raw_risk = family.held_out_risk(y, f.view(), oob_weights.view());
            oob_risks.push(if oob_sum > 0.0 {
                raw_risk / oob_sum
            } else {
                f64::NAN
            });
        }

        Ok((fold_idx, oob_risks))
    };

    let fold_results: Vec<(usize, Vec<f64>)> = if parallel {
        (0..b)
            .into_par_iter()
            .map(&compute_fold)
            .collect::<Result<Vec<_>, _>>()?
    } else {
        (0..b).map(&compute_fold).collect::<Result<Vec<_>, _>>()?
    };

    let mut risk_matrix = Array2::zeros((b, n_iter));
    for (fold_idx, risks) in fold_results {
        for m in 0..n_iter {
            risk_matrix[[fold_idx, m]] = risks[m];
        }
    }

    let mut aggregate = Array1::zeros(n_iter);
    for m in 0..n_iter {
        let mut s = 0.0f64;
        for i in 0..b {
            s += risk_matrix[[i, m]];
        }
        aggregate[m] = s / b as f64;
    }

    let grid: Vec<usize> = (0..=mstop).collect();

    let fold_type = format!("{}-fold {fold_label}", b);

    Ok(CvRiskResult {
        risk_matrix,
        aggregate,
        grid,
        fold_type,
    })
}

/// What every fold of one cross-validation shares.
struct FoldInputs<'a, F: Family + Clone> {
    preparer: &'a dyn LearnerPreparer,
    y: ArrayView1<'a, f64>,
    weights: ArrayView1<'a, f64>,
    folds: &'a Array2<f64>,
    judge: Option<&'a Array2<bool>>,
    offset_override: Option<f64>,
    offset_vec: Option<ArrayView1<'a, f64>>,
    family_template: F,
    n: usize,
}

/// One fold's fit and its held-out risk path, kept so the path can be grown.
///
/// Learners are prepared once with the fold's in-bag weights, as R re-runs
/// the base-learner preprocessing (`dpp`) per fold: `df_to_lambda` depends on
/// the weights, so reusing a full-sample lambda across folds gives a small
/// systematic mismatch for penalized learners. `risks[m]` is the normalized
/// held-out risk after `m` steps; only held-out rows are evaluated, which
/// makes each step O(n_oob) rather than O(n).
struct FoldRun<F: Family + Clone> {
    learners: Vec<PreparedLearner>,
    fit_family: F,
    eval_family: F,
    fold_w: Array1<f64>,
    path: crate::path::BoostingPath,
    oob_idx: Vec<usize>,
    y_oob: Array1<f64>,
    w_oob: Array1<f64>,
    oob_sum: f64,
    /// Block, tree and factored designs restricted to held-out rows.
    compacted: Vec<Option<Array2<f64>>>,
    f_oob: Array1<f64>,
    nu: f64,
    risks: Vec<f64>,
}

impl<F: Family + Clone> FoldRun<F> {
    /// Fit fold `fold_idx` to `control.mstop` steps and score every step.
    fn start(
        inputs: &FoldInputs<'_, F>,
        fold_idx: usize,
        control: &BoostControl,
    ) -> Result<Self, MboostrsError> {
        let n = inputs.n;
        let y = inputs.y;
        let fold_w: Array1<f64> = (0..n).map(|i| inputs.folds[[i, fold_idx]]).collect();

        let mut oob_weights = inputs.weights.to_owned();
        for i in 0..n {
            if fold_w[i] > 0.0 {
                oob_weights[i] = 0.0;
            }
            if let Some(mask) = inputs.judge {
                if !mask[[i, fold_idx]] {
                    oob_weights[i] = 0.0;
                }
            }
        }
        let oob_sum: f64 = oob_weights.sum() / inputs.family_template.flat_response_span() as f64;

        // Some families (CoxPH, LogLog, Weibull) use a concatenated y of
        // length 2*n (first n = times, last n = events). For these we must
        // select from both halves.
        let oob_idx: Vec<usize> = (0..n).filter(|&i| oob_weights[i] > 0.0).collect();
        let n_oob = oob_idx.len();
        if n_oob == 0 {
            return Err(MboostrsError::Control(format!(
                "cross-validation fold {fold_idx} has no positive-weight OOB observations"
            )));
        }
        let y_oob = if y.len() == n {
            y.select(ndarray::Axis(0), &oob_idx)
        } else {
            let times = y.slice(ndarray::s![..n]);
            let events = y.slice(ndarray::s![n..]);
            ndarray::concatenate![
                ndarray::Axis(0),
                times.select(ndarray::Axis(0), &oob_idx),
                events.select(ndarray::Axis(0), &oob_idx),
            ]
        };
        let w_oob = oob_weights.select(ndarray::Axis(0), &oob_idx);

        let mut learners: Vec<PreparedLearner> = inputs.preparer.prepare(fold_w.view());
        let mut fit_family = inputs.family_template.clone();

        let gram = if fit_family.supports_gram_update() && inputs.offset_vec.is_none() {
            crate::engine::try_fit_gaussian_gram(
                &mut learners,
                y,
                fold_w.view(),
                control,
                inputs.offset_override,
                false,
                true,
                true,
            )
        } else {
            None
        };
        let path = match gram {
            Some(path) => path,
            None => match inputs.offset_vec {
                Some(vector) => crate::engine::try_fit_with_vector_offset(
                    &mut learners,
                    y,
                    fold_w.view(),
                    &mut fit_family,
                    control,
                    vector,
                    false,
                )?,
                None => try_fit_with(
                    &mut learners,
                    y,
                    fold_w.view(),
                    &mut fit_family,
                    control,
                    inputs.offset_override,
                    false,
                )?,
            },
        };
        // The walk evaluates only OOB rows, so the start is the caller's
        // offset on those rows; the path's scalar field describes a start
        // this fit never used.
        let start = match inputs.offset_vec {
            Some(vector) => vector.select(ndarray::Axis(0), &oob_idx),
            None => Array1::from_elem(n_oob, path.offset),
        };

        // ScalarBols indexes into the full design by OOB index; KroneckerBols
        // computes on the full design and selects. Btree prediction is
        // row-independent, so its design compacts too.
        let compacted: Vec<Option<Array2<f64>>> = learners
            .iter()
            .map(|l| match l {
                PreparedLearner::BlockBols { design, .. }
                | PreparedLearner::BmonoBbs { design, .. }
                | PreparedLearner::Btree { design, .. } => {
                    Some(design.select(ndarray::Axis(0), &oob_idx))
                }
                PreparedLearner::FactoredBlockBols { x_l, .. } => {
                    Some(x_l.select(ndarray::Axis(0), &oob_idx))
                }
                PreparedLearner::BspatialStreamed { .. } => {
                    Some(l.design_matrix()?.select(ndarray::Axis(0), &oob_idx))
                }
                _ => None,
            })
            .collect();

        let mut run = FoldRun {
            learners,
            fit_family,
            eval_family: inputs.family_template.clone(),
            fold_w,
            path,
            oob_idx,
            y_oob,
            w_oob,
            oob_sum,
            compacted,
            f_oob: start,
            nu: control.nu,
            risks: Vec::with_capacity(control.mstop as usize + 1),
        };
        run.push_risk(0);
        run.walk(0, run.path.selected.len());
        Ok(run)
    }

    /// Grow the fold's path to `control.mstop` steps by resuming it, and
    /// score the new steps. The resumed path equals a cold fit of the new
    /// length to machine precision (see `engine::WarmStart`).
    fn extend(&mut self, y: ArrayView1<f64>, control: &BoostControl) -> Result<(), MboostrsError> {
        let done = self.path.selected.len();
        if control.mstop as usize <= done {
            return Ok(());
        }
        let path = {
            let warm = crate::engine::WarmStart::from_path(&self.path);
            crate::engine::try_fit_continue(
                &mut self.learners,
                y,
                self.fold_w.view(),
                &mut self.fit_family,
                control,
                &warm,
            )?
        };
        self.path = path;
        self.walk(done, self.path.selected.len());
        Ok(())
    }

    /// Set the evaluation family's nuisance to its value after step `k` and
    /// record the held-out risk there.
    fn push_risk(&mut self, k: usize) {
        match self.path.nuisance_path.get(k).copied().flatten() {
            Some(v) => self.eval_family.set_nuisance(v),
            None => {
                self.eval_family
                    .calibrate(self.y_oob.view(), self.f_oob.view(), self.w_oob.view())
            }
        }
        if let Some(vv) = self.path.nuisance_vec_path.get(k) {
            if !vv.is_empty() {
                self.eval_family.set_nuisance_vec(vv.clone());
            }
        }
        let raw_risk =
            self.eval_family
                .held_out_risk(self.y_oob.view(), self.f_oob.view(), self.w_oob.view());
        self.risks.push(if self.oob_sum > 0.0 {
            raw_risk / self.oob_sum
        } else {
            f64::NAN
        });
    }

    /// Apply steps `from..to` of the path to the held-out predictor, scoring
    /// after each.
    fn walk(&mut self, from: usize, to: usize) {
        let nu = self.nu;
        for m in from..to {
            let idx = self.path.selected[m];
            let coef = &self.path.coefs[m];
            match &self.learners[idx] {
                PreparedLearner::ScalarBols { design, .. } => {
                    let c = nu * coef[0];
                    for (fi, &oi) in self.f_oob.iter_mut().zip(self.oob_idx.iter()) {
                        *fi += c * design[oi];
                    }
                }
                PreparedLearner::DiagBrandom { group_idx, .. } => {
                    for (fi, &oi) in self.f_oob.iter_mut().zip(self.oob_idx.iter()) {
                        *fi += nu * coef[group_idx[oi] as usize];
                    }
                }
                PreparedLearner::BinnedScalarBols {
                    bin_index,
                    bin_values,
                    ..
                } => {
                    let c = nu * coef[0];
                    for (fi, &oi) in self.f_oob.iter_mut().zip(self.oob_idx.iter()) {
                        *fi += c * bin_values[bin_index[oi] as usize];
                    }
                }
                PreparedLearner::BinnedBlockBols {
                    bin_index,
                    basis,
                    ..
                } => {
                    let k = basis.nrows();
                    let p = basis.ncols();
                    let mut bin_pred = vec![0.0f64; k];
                    for bin in 0..k {
                        let mut s = 0.0;
                        for col in 0..p {
                            s += basis[[bin, col]] * coef[col];
                        }
                        bin_pred[bin] = s;
                    }
                    for (fi, &oi) in self.f_oob.iter_mut().zip(self.oob_idx.iter()) {
                        *fi += nu * bin_pred[bin_index[oi] as usize];
                    }
                }
                PreparedLearner::BlockBols { .. }
                | PreparedLearner::BspatialStreamed { .. }
                | PreparedLearner::BmonoBbs { .. } => {
                    if let Some(ref design_oob) = self.compacted[idx] {
                        // The clone is gone; the product stays, for the
                        // reason above.
                        let pred = design_oob.dot(&ndarray::ArrayView1::from(&coef[..]));
                        for (fi, &pi) in self.f_oob.iter_mut().zip(pred.iter()) {
                            *fi += nu * pi;
                        }
                    }
                }
                PreparedLearner::Btree { .. } => {
                    if let Some(ref design_oob) = self.compacted[idx] {
                        // A view over the coefficients the path already
                        // owns; `from_coef_vec` reads them and keeps nothing.
                        let tree = crate::learners::FittedTree::from_coef_vec(
                            ndarray::ArrayView1::from(&coef[..]),
                        );
                        let pred = tree.predict(design_oob.view());
                        for (fi, &pi) in self.f_oob.iter_mut().zip(pred.iter()) {
                            *fi += nu * pi;
                        }
                    }
                }
                // An adaptive hinge's `[knot, direction, coefficient,
                // centering]` evaluates row-independently, so the full column
                // is read by hold-out index like ScalarBols.
                PreparedLearner::Bhinge { x, by, .. } => {
                    let (knot, direction, coefficient, centering) =
                        (coef[0], coef[1], coef[2], coef[3]);
                    let scale = by.as_ref();
                    for (fi, &oi) in self.f_oob.iter_mut().zip(self.oob_idx.iter()) {
                        let hinge = if direction > 0.0 {
                            (x[oi] - knot).max(0.0)
                        } else {
                            (knot - x[oi]).max(0.0)
                        };
                        let column = match scale {
                            Some(values) => hinge * values[oi],
                            None => hinge,
                        };
                        *fi += nu * coefficient * (column - centering);
                    }
                }
                PreparedLearner::FactoredBlockBols {
                    groups,
                    p_l,
                    n_groups,
                    ..
                } => {
                    if let Some(ref design_oob) = self.compacted[idx] {
                        // `design_oob` is the OOB-compacted left marginal x_l
                        // (row `fi` = observation oob_idx[fi]); `coef` is in the
                        // materialised column order l·ng + g.
                        for (row_fi, (fi, &oi)) in
                            self.f_oob.iter_mut().zip(self.oob_idx.iter()).enumerate()
                        {
                            let gr = groups[oi] as usize;
                            let mut s = 0.0;
                            for l in 0..*p_l {
                                s += design_oob[[row_fi, l]] * coef[l * *n_groups + gr];
                            }
                            *fi += nu * s;
                        }
                    }
                }
                PreparedLearner::KroneckerBols {
                    x1,
                    x2,
                    x2_is_identity,
                    n_resp,
                    ..
                } => {
                    // Multinomial-style flattened layout: entry e = obs·n_resp
                    // + class (matching fit_step_ref's pred_buf). The step
                    // increment at (i, k) is (X1·B·X2ᵀ)[i, k] with the stored
                    // coef reshaped column-major to B[j, l] = coef[l·p1 + j].
                    // Two matmuls — O(n_obs·p2·(p1 + n_resp)) — replace the
                    // former per-entry quadruple loop, which also mis-indexed
                    // f_oob for n_resp > 1 (treated flattened OOB entries as
                    // observation indices).
                    let p1 = x1.ncols();
                    let p2 = x2.ncols();
                    let nr = *n_resp;
                    if *x2_is_identity {
                        debug_assert_eq!(p2, nr);
                        for (fi, &e) in self.f_oob.iter_mut().zip(self.oob_idx.iter()) {
                            let obs = e / nr;
                            let k = e % nr;
                            let coef_k = &coef[k * p1..(k + 1) * p1];
                            let mut s = 0.0;
                            for j in 0..p1 {
                                s += x1[[obs, j]] * coef_k[j];
                            }
                            *fi += nu * s;
                        }
                    } else {
                        let b_mat =
                            ndarray::Array2::from_shape_fn((p1, p2), |(j, l)| coef[l * p1 + j]);
                        let pred = x1.dot(&b_mat).dot(&x2.t()); // (n_obs, n_resp)
                        for (fi, &e) in self.f_oob.iter_mut().zip(self.oob_idx.iter()) {
                            *fi += nu * pred[[e / nr, e % nr]];
                        }
                    }
                }
            }
            self.push_risk(m + 1);
        }
    }
}

/// Score a fitted in-bag [`BoostingPath`] on held-out observations across
/// iterations `0..=mstop`, returning normalized OOB risk at every step.
pub fn score_fold_oob_path<F: Family + Clone>(
    learners: Vec<PreparedLearner>,
    fit_family: F,
    eval_family: F,
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    fold_w: Array1<f64>,
    path: crate::path::BoostingPath,
    nu: f64,
    fold_idx: usize,
) -> Result<Vec<f64>, MboostrsError> {
    let n = eval_family.n_from_y(y);
    let mut oob_weights = weights.to_owned();
    for i in 0..n {
        if fold_w[i] > 0.0 {
            oob_weights[i] = 0.0;
        }
    }
    let oob_sum: f64 = oob_weights.sum() / eval_family.flat_response_span() as f64;
    let oob_idx: Vec<usize> = (0..n).filter(|&i| oob_weights[i] > 0.0).collect();
    let n_oob = oob_idx.len();
    if n_oob == 0 {
        return Err(MboostrsError::Control(format!(
            "cross-validation fold {fold_idx} has no positive-weight OOB observations"
        )));
    }
    let y_oob = if y.len() == n {
        y.select(ndarray::Axis(0), &oob_idx)
    } else {
        let times = y.slice(ndarray::s![..n]);
        let events = y.slice(ndarray::s![n..]);
        ndarray::concatenate![
            ndarray::Axis(0),
            times.select(ndarray::Axis(0), &oob_idx),
            events.select(ndarray::Axis(0), &oob_idx),
        ]
    };
    let w_oob = oob_weights.select(ndarray::Axis(0), &oob_idx);
    let start = Array1::from_elem(n_oob, path.offset);
    let compacted: Vec<Option<Array2<f64>>> = learners
        .iter()
        .map(|l| match l {
            PreparedLearner::BlockBols { design, .. }
            | PreparedLearner::BmonoBbs { design, .. }
            | PreparedLearner::Btree { design, .. } => {
                Some(design.select(ndarray::Axis(0), &oob_idx))
            }
            PreparedLearner::FactoredBlockBols { x_l, .. } => {
                Some(x_l.select(ndarray::Axis(0), &oob_idx))
            }
            PreparedLearner::BspatialStreamed { .. } => {
                Some(l.design_matrix()?.select(ndarray::Axis(0), &oob_idx))
            }
            _ => None,
        })
        .collect();
    let n_steps = path.selected.len();
    let mut run = FoldRun {
        learners,
        fit_family,
        eval_family,
        fold_w,
        path,
        oob_idx,
        y_oob,
        w_oob,
        oob_sum,
        compacted,
        f_oob: start,
        nu,
        risks: Vec::with_capacity(n_steps + 1),
    };
    run.push_risk(0);
    run.walk(0, n_steps);
    Ok(run.risks)
}

/// Monomorphised cvrisk: takes a concrete family type so the per-fold
/// `fit()` and `risk()` calls are statically dispatched and inlined.
/// The PyO3 layer dispatches via `with_concrete_family!`; the original
/// `cvrisk` remains for the dyn-Family test helper.
pub fn cvrisk_with<F: Family + Clone>(
    preparer: &dyn LearnerPreparer,
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    control: &BoostControl,
    folds: &Array2<f64>,
    offset_override: Option<f64>,
    parallel: bool,
    family_template: F,
    fold_label: &str,
) -> CvRiskResult {
    try_cvrisk_with(
        preparer,
        y,
        weights,
        control,
        folds,
        offset_override,
        None,
        None,
        parallel,
        family_template,
        fold_label,
    )
    .unwrap_or_else(|e| panic!("{e}"))
}

/// Fallible monomorphised cross-validation entry point.
#[allow(clippy::too_many_arguments)]
pub fn try_cvrisk_with<F: Family + Clone>(
    preparer: &dyn LearnerPreparer,
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    control: &BoostControl,
    folds: &Array2<f64>,
    offset_override: Option<f64>,
    offset_vec: Option<ArrayView1<f64>>,
    judge: Option<&Array2<bool>>,
    parallel: bool,
    family_template: F,
    fold_label: &str,
) -> Result<CvRiskResult, MboostrsError> {
    let (y_encoded, n) = check_cv_inputs(
        &family_template,
        y,
        weights,
        folds,
        offset_override,
        offset_vec,
        judge,
    )?;
    let y = match y_encoded.as_ref() {
        Some(a) => a.view(),
        None => y.view(),
    };
    let inputs = FoldInputs {
        preparer,
        y: y.view(),
        weights: weights.view(),
        folds,
        judge,
        offset_override,
        offset_vec: offset_vec.as_ref().map(|vector| vector.view()),
        family_template: family_template.clone(),
        n,
    };
    let (result, _) = run_folds(&inputs, control, parallel, fold_label, None)?;
    Ok(result)
}

/// How [`try_cvrisk_extending`] grows the iteration grid.
#[derive(Clone, Copy, Debug)]
pub struct ExtendRule {
    /// The first grid's largest iteration.
    pub start: u32,
    /// The largest iteration the grid may reach.
    pub max_mstop: u32,
    /// Stop once the aggregated held-out risk has not improved for this many
    /// iterations; the grid is grown only as far as that check needs. `None`
    /// uses the boundary rule instead: the grid doubles while the minimum
    /// lies in the last `boundary_share` of it.
    pub patience: Option<u32>,
    /// Share of the grid's end that counts as the boundary (boundary rule).
    pub boundary_share: f64,
    /// How many times the grid may grow; `None` is unlimited.
    pub max_extensions: Option<u32>,
}

/// What [`try_cvrisk_extending`] did to reach its grid.
#[derive(Clone, Debug)]
pub struct ExtensionReport {
    /// The grid's end after the first fit and after every extension.
    pub grid_ends: Vec<usize>,
    /// True when the search stopped with the minimum still at the end of the
    /// grid (within the patience window or the boundary share): the risk had
    /// not been seen to turn.
    pub at_boundary: bool,
    /// The iteration minimising the aggregated risk (never 0).
    pub best: usize,
}

/// Cross-validated risk on a grid that grows until the held-out risk turns.
///
/// Every fold is fitted to `rule.start` iterations. While the aggregated
/// held-out risk is still falling at the end of the grid, each fold's path is
/// resumed (`engine::try_fit_continue`) rather than refitted, so the total
/// work is that of one fit per fold to the final grid. With
/// `rule.patience = Some(p)` the search stops once the minimum is `p`
/// iterations behind the grid's end, and each extension runs only to
/// `best + p`; the iteration it selects equals the full grid's whenever no
/// lower risk appears more than `p` steps after a minimum. The risk matrix
/// covers the final grid, and the report lists every grid end.
#[allow(clippy::too_many_arguments)]
pub fn try_cvrisk_extending<F: Family + Clone>(
    preparer: &dyn LearnerPreparer,
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    control: &BoostControl,
    folds: &Array2<f64>,
    offset_override: Option<f64>,
    offset_vec: Option<ArrayView1<f64>>,
    judge: Option<&Array2<bool>>,
    parallel: bool,
    family_template: F,
    fold_label: &str,
    rule: &ExtendRule,
) -> Result<(CvRiskResult, ExtensionReport), MboostrsError> {
    if rule.start == 0 || rule.max_mstop == 0 {
        return Err(MboostrsError::Control(
            "start and max_mstop must be at least 1".to_string(),
        ));
    }
    if rule.patience == Some(0) {
        return Err(MboostrsError::Control(
            "patience must be at least 1".to_string(),
        ));
    }
    if !(rule.boundary_share.is_finite() && (0.0..1.0).contains(&rule.boundary_share)) {
        return Err(MboostrsError::Control(format!(
            "boundary_share must lie in [0, 1), got {}",
            rule.boundary_share
        )));
    }
    let (y_encoded, n) = check_cv_inputs(
        &family_template,
        y,
        weights,
        folds,
        offset_override,
        offset_vec,
        judge,
    )?;
    let y = match y_encoded.as_ref() {
        Some(a) => a.view(),
        None => y.view(),
    };
    let inputs = FoldInputs {
        preparer,
        y: y.view(),
        weights: weights.view(),
        folds,
        judge,
        offset_override,
        offset_vec: offset_vec.as_ref().map(|vector| vector.view()),
        family_template: family_template.clone(),
        n,
    };
    let (result, report) = run_folds(&inputs, control, parallel, fold_label, Some(rule))?;
    Ok((result, report.expect("an extension rule yields a report")))
}

/// Validate cross-validation inputs; return the family-encoded response
/// (when the family recodes it) and the number of fitted coordinates.
#[allow(clippy::too_many_arguments)]
fn check_cv_inputs<F: Family>(
    family_template: &F,
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    folds: &Array2<f64>,
    offset_override: Option<f64>,
    offset_vec: Option<ArrayView1<f64>>,
    judge: Option<&Array2<bool>>,
) -> Result<(Option<Array1<f64>>, usize), MboostrsError> {
    validate_cvrisk_inputs(family_template, y, weights, folds, offset_override)?;
    if let Some(mask) = judge {
        if mask.shape() != folds.shape() {
            return Err(MboostrsError::Shape(format!(
                "judge mask has shape {:?}, expected {:?} (the fold matrix)",
                mask.shape(),
                folds.shape()
            )));
        }
    }
    let y_encoded = family_template.encode_response(y);
    let n = match y_encoded.as_ref() {
        Some(a) => family_template.n_from_y(a.view()),
        None => family_template.n_from_y(y.view()),
    };
    if let Some(vector) = offset_vec {
        if vector.len() != n {
            return Err(MboostrsError::Shape(format!(
                "offset has {} entries but the response implies {} fitted coordinates",
                vector.len(),
                n
            )));
        }
        if vector.iter().any(|value| !value.is_finite()) {
            return Err(MboostrsError::Control(
                "offset must be finite in every entry; log(0) of a zero-exposure \
                 row is -inf, and such a row cannot enter the linear predictor"
                    .to_string(),
            ));
        }
    }
    Ok((y_encoded, n))
}

/// Fit every fold, grow the grid under `rule` when one is given, and
/// assemble the (folds x iterations) risk matrix.
fn run_folds<F: Family + Clone>(
    inputs: &FoldInputs<'_, F>,
    control: &BoostControl,
    parallel: bool,
    fold_label: &str,
    rule: Option<&ExtendRule>,
) -> Result<(CvRiskResult, Option<ExtensionReport>), MboostrsError> {
    let b = inputs.folds.ncols();
    let mut control = control.clone();
    if let Some(rule) = rule {
        control.mstop = rule.start.min(rule.max_mstop);
    }
    let start = |fold_idx: usize| FoldRun::start(inputs, fold_idx, &control);
    let mut runs: Vec<FoldRun<F>> = if parallel {
        (0..b)
            .into_par_iter()
            .map(start)
            .collect::<Result<Vec<_>, _>>()?
    } else {
        (0..b).map(start).collect::<Result<Vec<_>, _>>()?
    };

    let aggregate_of = |runs: &[FoldRun<F>]| -> Array1<f64> {
        let len = runs[0].risks.len();
        Array1::from_shape_fn(len, |m| {
            runs.iter().map(|run| run.risks[m]).sum::<f64>() / b as f64
        })
    };
    let mut aggregate = aggregate_of(&runs);
    let report = match rule {
        None => None,
        Some(rule) => {
            let mut grid_ends = vec![control.mstop as usize];
            let mut extensions = 0u32;
            loop {
                let end = control.mstop as usize;
                let best = best_iteration(&aggregate).ok_or_else(|| {
                    MboostrsError::Solver(
                        "no finite held-out risk at a positive iteration".to_string(),
                    )
                })?;
                let turned = match rule.patience {
                    Some(p) => end - best >= p as usize,
                    None => {
                        let tolerance =
                            ((rule.boundary_share * end as f64).round() as usize).max(1);
                        best + tolerance < end
                    }
                };
                let exhausted = end >= rule.max_mstop as usize
                    || rule.max_extensions.is_some_and(|limit| extensions >= limit);
                if turned || exhausted {
                    break Some(ExtensionReport {
                        grid_ends,
                        at_boundary: !turned,
                        best,
                    });
                }
                let next = match rule.patience {
                    Some(p) => best + p as usize,
                    None => 2 * end,
                }
                .min(rule.max_mstop as usize);
                control.mstop = next as u32;
                let y = inputs.y;
                let grow = |run: &mut FoldRun<F>| run.extend(y, &control);
                if parallel {
                    runs.par_iter_mut().try_for_each(grow)?;
                } else {
                    runs.iter_mut().try_for_each(grow)?;
                }
                extensions += 1;
                grid_ends.push(next);
                aggregate = aggregate_of(&runs);
            }
        }
    };

    let n_iter = aggregate.len();
    let mut risk_matrix = Array2::zeros((b, n_iter));
    for (fold_idx, run) in runs.iter().enumerate() {
        for m in 0..n_iter {
            risk_matrix[[fold_idx, m]] = run.risks[m];
        }
    }
    Ok((
        CvRiskResult {
            risk_matrix,
            aggregate,
            grid: (0..n_iter).collect(),
            fold_type: format!("{}-fold {fold_label}", b),
        },
        report,
    ))
}

/// The iteration (never 0) minimising a finite aggregated risk; ties go to
/// the lowest, as `CvRiskResult::optimal_mstop` and R's `mstop()` choose.
fn best_iteration(aggregate: &Array1<f64>) -> Option<usize> {
    (1..aggregate.len())
        .filter(|&m| aggregate[m].is_finite())
        .min_by(|&a, &b| aggregate[a].total_cmp(&aggregate[b]).then(a.cmp(&b)))
}

/// Monomorphised cvrisk_selected: takes a concrete family type for static dispatch.
///
/// Panicking compatibility wrapper around [`try_cvrisk_selected_with`],
/// deprecated on the P0.8 schedule. Removal: 0.2.0.
#[deprecated(
    since = "0.1.3",
    note = "use try_cvrisk_selected_with (returns MboostrsError instead of panicking); removed in 0.2.0"
)]
pub fn cvrisk_selected_with<F: Family + Clone>(
    preparer: &dyn LearnerPreparer,
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    control: &BoostControl,
    folds: &Array2<f64>,
    offset_override: Option<f64>,
    family_template: F,
) -> CvSelectedResult {
    try_cvrisk_selected_with(
        preparer,
        y,
        weights,
        control,
        folds,
        offset_override,
        family_template,
    )
    .unwrap_or_else(|e| panic!("{e}"))
}

/// Fallible selected-path cross-validation used by stability selection.
pub fn try_cvrisk_selected_with<F: Family + Clone>(
    preparer: &dyn LearnerPreparer,
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    control: &BoostControl,
    folds: &Array2<f64>,
    offset_override: Option<f64>,
    family_template: F,
) -> Result<CvSelectedResult, MboostrsError> {
    try_cvrisk_selected_with_parallel(
        preparer,
        y,
        weights,
        control,
        folds,
        offset_override,
        None,
        family_template,
        true,
    )
}

/// [`try_cvrisk_selected_with`] for a model fitted with a per-observation
/// offset: every fold starts from `offset_vec` (one entry per fitted
/// coordinate), as the full-data fit did, instead of a refitted constant.
#[allow(clippy::too_many_arguments)]
pub fn try_cvrisk_selected_with_offset_vec<F: Family + Clone>(
    preparer: &dyn LearnerPreparer,
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    control: &BoostControl,
    folds: &Array2<f64>,
    offset_vec: ArrayView1<f64>,
    family_template: F,
) -> Result<CvSelectedResult, MboostrsError> {
    try_cvrisk_selected_with_parallel(
        preparer,
        y,
        weights,
        control,
        folds,
        None,
        Some(offset_vec),
        family_template,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn try_cvrisk_selected_with_parallel<F: Family + Clone>(
    preparer: &dyn LearnerPreparer,
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    control: &BoostControl,
    folds: &Array2<f64>,
    offset_override: Option<f64>,
    offset_vec: Option<ArrayView1<f64>>,
    family_template: F,
    parallel: bool,
) -> Result<CvSelectedResult, MboostrsError> {
    let mstop = control.mstop as usize;
    let per_fold = fold_selected_paths(
        preparer,
        y,
        weights,
        control,
        folds,
        offset_override,
        offset_vec,
        family_template,
        parallel,
    )?;
    let mut selected_matrix = Array2::zeros((folds.ncols(), mstop));
    for (fold_idx, sel) in per_fold.into_iter().enumerate() {
        let sel = sel?;
        for m in 0..mstop.min(sel.len()) {
            selected_matrix[[fold_idx, m]] = sel[m];
        }
    }

    Ok(CvSelectedResult {
        selected: selected_matrix,
    })
}

/// The selection path of every fold for stability selection, one entry per
/// column of `folds`: the 0-based learner index chosen at each iteration, or
/// `None` when that fold's fit failed. A failed fit is kept apart from the
/// others so the caller can count it as a subsample that selected nothing.
///
/// Each fold is fitted on its column of `folds` as in-bag weights. The
/// offset is refitted on each fold unless `offset_override` (a fixed scalar)
/// or `offset_vec` (a per-observation offset such as `log(exposure)`) is
/// given. `Err` is returned only for inputs that fail the shared
/// cross-validation checks (response domain, weight and fold shapes, offset).
#[allow(clippy::too_many_arguments)]
pub fn try_stability_paths_with<F: Family + Clone>(
    preparer: &dyn LearnerPreparer,
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    control: &BoostControl,
    folds: &Array2<f64>,
    offset_override: Option<f64>,
    offset_vec: Option<ArrayView1<f64>>,
    family_template: F,
) -> Result<Vec<Option<Vec<usize>>>, MboostrsError> {
    let per_fold = fold_selected_paths(
        preparer,
        y,
        weights,
        control,
        folds,
        offset_override,
        offset_vec,
        family_template,
        true,
    )?;
    Ok(per_fold.into_iter().map(Result::ok).collect())
}

/// Fits every fold and returns each fold's selection path or its fit error,
/// in fold order. The outer `Err` is an input-validation failure.
#[allow(clippy::too_many_arguments)]
fn fold_selected_paths<F: Family + Clone>(
    preparer: &dyn LearnerPreparer,
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    control: &BoostControl,
    folds: &Array2<f64>,
    offset_override: Option<f64>,
    offset_vec: Option<ArrayView1<f64>>,
    family_template: F,
    parallel: bool,
) -> Result<Vec<Result<Vec<usize>, MboostrsError>>, MboostrsError> {
    validate_cvrisk_inputs(&family_template, y, weights, folds, offset_override)?;
    let y_encoded = family_template.encode_response(y);
    let y = match y_encoded.as_ref() {
        Some(a) => a.view(),
        None => y.view(),
    };
    let n = family_template.n_from_y(y);
    let b = folds.ncols();

    let compute_fold = |fold_idx| -> Result<Vec<usize>, MboostrsError> {
        let fold_weights: Vec<f64> = (0..n).map(|i| folds[[i, fold_idx]]).collect();
        let fold_w = Array1::from_vec(fold_weights);

        let mut learners = preparer.prepare(fold_w.view());
        let mut family = family_template.clone();

        let selected = match offset_vec {
            Some(vector) => crate::engine::try_fit_selected_with_vector_offset(
                &mut learners,
                y,
                fold_w.view(),
                &mut family,
                control,
                vector,
            )?,
            None => crate::engine::try_fit_selected(
                &mut learners,
                y,
                fold_w.view(),
                &mut family,
                control,
                offset_override,
            )?,
        };

        Ok(selected)
    };
    // `collect` on an indexed parallel iterator keeps fold order.
    Ok(if parallel {
        (0..b).into_par_iter().map(&compute_fold).collect()
    } else {
        (0..b).map(&compute_fold).collect()
    })
}
/// Result of `cvrisk_selected`: which base learner each fold selected at each
/// boosting iteration.
#[derive(Clone, Debug)]
pub struct CvSelectedResult {
    /// (B, mstop) matrix of selected base-learner indices.
    /// selected[[fold, iter]] = 0-based learner index at that iteration.
    pub selected: Array2<usize>,
}

/// Like cvrisk, but returns the selected base-learner index at each iteration
/// for each fold, instead of OOB risk values. Used by stabsel.
///
/// Compatibility wrapper around [`try_cvrisk_selected`]; panics on malformed
/// inputs and is deprecated on the P0.8 schedule.
#[deprecated(
    since = "0.1.3",
    note = "use try_cvrisk_selected (returns MboostrsError instead of panicking); removed in 0.2.0"
)]
pub fn cvrisk_selected(
    preparer: &dyn LearnerPreparer,
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    control: &BoostControl,
    folds: &Array2<f64>,
    offset_override: Option<f64>,
) -> CvSelectedResult {
    try_cvrisk_selected(preparer, y, weights, control, folds, offset_override)
        .expect("cvrisk_selected: valid inputs")
}

/// Fallible [`cvrisk_selected`]: never panics. Returns `Err` when the shared
/// CV input validation (response finiteness, family response domain, weight
/// and fold-matrix shape, offset finiteness) rejects the inputs, or when any
/// per-fold fit fails.
pub fn try_cvrisk_selected(
    preparer: &dyn LearnerPreparer,
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    control: &BoostControl,
    folds: &Array2<f64>,
    offset_override: Option<f64>,
) -> Result<CvSelectedResult, MboostrsError> {
    let fam = preparer.family();
    validate_cvrisk_inputs(fam.as_ref(), y, weights, folds, offset_override)?;
    // Re-encode the response (Binomial {0,1}->{-1,+1}; no-op otherwise) so the
    // per-fold fits use the family's internal coding, matching `fit`/`cvrisk`.
    let y_encoded = preparer.family().encode_response(y);
    let y = match y_encoded.as_ref() {
        Some(a) => a.view(),
        None => y.view(),
    };
    let n = {
        let fam = preparer.family();
        fam.n_from_y(y)
    };
    let b = folds.ncols();
    let mstop = control.mstop as usize;

    let fold_results: Vec<(usize, Vec<usize>)> = (0..b)
        .into_par_iter()
        .map(|fold_idx| -> Result<(usize, Vec<usize>), MboostrsError> {
            let fold_weights: Vec<f64> = (0..n).map(|i| folds[[i, fold_idx]]).collect();
            let fold_w = Array1::from_vec(fold_weights);

            let mut learners = preparer.prepare(fold_w.view());
            let mut family = preparer.family();

            let selected = try_fit_selected(
                &mut learners,
                y,
                fold_w.view(),
                family.as_mut(),
                control,
                offset_override,
            )?;

            Ok((fold_idx, selected))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut selected_matrix = Array2::zeros((b, mstop));
    for (fold_idx, sel) in fold_results {
        for m in 0..mstop.min(sel.len()) {
            selected_matrix[[fold_idx, m]] = sel[m];
        }
    }

    Ok(CvSelectedResult {
        selected: selected_matrix,
    })
}

impl CvRiskResult {
    /// Find the optimal stopping iteration (argmin of aggregate risk),
    /// excluding iteration 0 (offset-only) to match R's `mstop(cvrisk_object)`.
    ///
    /// A path with no boosting step (an offset-only `y ~ 1` model) has only
    /// iteration 0, so the answer is 0: there is no step to stop at.
    pub fn optimal_mstop(&self) -> usize {
        if self.aggregate.len() < 2 {
            return 0;
        }
        let mut best = 1usize;
        let mut best_val = self.aggregate[1];
        for m in 2..self.aggregate.len() {
            if self.aggregate[m] < best_val {
                best_val = self.aggregate[m];
                best = m;
            }
        }
        self.grid[best]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::Gaussian;
    use crate::learners::{prepare_bmono, prepare_scalar_bols, BmonoSolver, MonoConstraint};
    use ndarray::array;

    #[test]
    fn foldwise_scores_weighted_paths_and_excludes_zero() {
        let result = try_cvrisk_foldwise(&Gaussian, 1, 2, |_| {
            Ok(CvPredictionFold {
                response: array![1.0, 2.0],
                weights: array![1.0, 3.0],
                predictions: array![[1.0, 2.0], [0.0, 0.0], [1.0, 1.0]],
                nuisance: vec![],
            })
        })
        .unwrap();
        assert_eq!(result.aggregate.to_vec(), vec![0.0, 3.25, 0.75]);
        assert_eq!(result.optimal_mstop(), 2);
    }

    #[test]
    fn foldwise_requires_training_nuisance_and_valid_shapes() {
        let family = crate::families::GammaReg::new();
        let supply = || CvPredictionFold {
            response: array![2.0],
            weights: array![1.0],
            predictions: array![[2.0_f64.ln()], [2.0_f64.ln()]],
            nuisance: vec![vec![1.0], vec![3.0]],
        };
        let scored = try_cvrisk_foldwise(&family, 1, 1, |_| Ok(supply())).unwrap();
        let mut reference = family.clone();
        reference.set_nuisance(3.0);
        let expected = reference.risk(
            array![2.0].view(),
            array![2.0_f64.ln()].view(),
            array![1.0].view(),
        );
        assert_eq!(scored.aggregate[1], expected);
        assert!(try_cvrisk_foldwise(&family, 1, 1, |_| {
            let mut fold = supply();
            fold.nuisance.clear();
            Ok(fold)
        })
        .is_err());
        assert!(try_cvrisk_foldwise(&family, 1, 2, |_| Ok(supply())).is_err());
    }

    #[test]
    fn foldwise_restores_a_nuisance_unset_on_the_template() {
        let family = crate::families::Huber::new(None);
        assert_eq!(family.nuisance(), None);
        let result = try_cvrisk_foldwise(&family, 1, 1, |_| {
            Ok(CvPredictionFold {
                response: array![2.0],
                weights: array![1.0],
                predictions: array![[0.0], [0.0]],
                nuisance: vec![vec![0.5], vec![0.5]],
            })
        })
        .unwrap();
        assert_eq!(result.aggregate.to_vec(), vec![0.875, 0.875]);
    }

    #[test]
    fn cdf_index_matches_linear_scan() {
        // Binary-search draw must select the same index a linear scan of the
        // per-index probabilities would, so bootstrap folds stay bit-identical.
        // Includes zero-probability entries (never selectable) and the tail.
        fn linear(probs: &[f64], u: f64) -> usize {
            let mut cum = 0.0f64;
            for (i, &p) in probs.iter().enumerate() {
                cum += p;
                if u < cum {
                    return i;
                }
            }
            probs.len() - 1
        }
        let probs = [0.2, 0.0, 0.3, 0.0, 0.5];
        let mut cdf = vec![0.0; probs.len()];
        let mut acc = 0.0;
        for i in 0..probs.len() {
            acc += probs[i];
            cdf[i] = acc;
        }
        for k in 0..=1000 {
            let u = k as f64 / 1000.0;
            assert_eq!(cdf_index(&cdf, u), linear(&probs, u), "mismatch at u={u}");
        }
        // explicit boundary points
        for &u in &[0.0, 0.2, 0.5, 0.999_999, 1.0] {
            assert_eq!(cdf_index(&cdf, u), linear(&probs, u), "boundary u={u}");
        }
    }

    struct TestPreparer {
        x_cols: Vec<Array1<f64>>,
        names: Vec<String>,
    }

    impl LearnerPreparer for TestPreparer {
        fn prepare(&self, weights: ArrayView1<f64>) -> Vec<PreparedLearner> {
            let mut learners = Vec::with_capacity(self.x_cols.len());
            for (i, col) in self.x_cols.iter().enumerate() {
                learners.push(prepare_scalar_bols(
                    col.view(),
                    weights,
                    0.0,
                    true,
                    &self.names[i],
                    None,
                ));
            }
            learners
        }
        fn family(&self) -> Box<dyn Family> {
            Box::new(Gaussian)
        }
    }

    struct BmonoPreparer {
        x: Array1<f64>,
    }

    impl LearnerPreparer for BmonoPreparer {
        fn prepare(&self, weights: ArrayView1<f64>) -> Vec<PreparedLearner> {
            vec![prepare_bmono(
                self.x.view(),
                weights,
                10,
                3,
                2,
                4.0,
                MonoConstraint::Increasing,
                BmonoSolver::Iterative,
                1e6,
                10,
                "bmono(x)",
                None,
                None,
                false,
            )]
        }

        fn family(&self) -> Box<dyn Family> {
            Box::new(Gaussian)
        }
    }

    #[test]
    fn cv_kfold_basic() {
        let spec = FoldSpec::new(FoldType::KFold, 5, 0.5, 42);
        let n = 20;
        let weights = vec![1.0; n];
        let folds = cv(&spec, n, &weights);
        assert_eq!(folds.nrows(), 20);
        assert_eq!(folds.ncols(), 5);

        for fold_idx in 0..5 {
            let n_holdout: usize = folds.column(fold_idx).iter().filter(|&&v| v == 0.0).count();
            assert_eq!(n_holdout, 4);
        }
    }

    #[test]
    fn cv_bootstrap_basic() {
        let spec = FoldSpec::new(FoldType::Bootstrap, 10, 0.5, 42);
        let n = 50;
        let weights = vec![1.0; n];
        let folds = cv(&spec, n, &weights);
        assert_eq!(folds.nrows(), 50);
        assert_eq!(folds.ncols(), 10);

        for fold_idx in 0..10 {
            let sum: f64 = folds.column(fold_idx).sum();
            assert!((sum - n as f64).abs() < 1e-10);
        }
    }

    #[test]
    fn cv_subsampling_basic() {
        let spec = FoldSpec::new(FoldType::Subsampling, 10, 0.5, 42);
        let n = 20;
        let weights = vec![1.0; n];
        let folds = cv(&spec, n, &weights);
        assert_eq!(folds.nrows(), 20);
        assert_eq!(folds.ncols(), 10);

        for fold_idx in 0..10 {
            let n_inbag: usize = folds.column(fold_idx).iter().filter(|&&v| v > 0.0).count();
            assert_eq!(n_inbag, 10);
        }
    }

    #[test]
    fn fallible_fold_validation_rejects_invalid_specs_and_inputs() {
        assert!(FoldSpec::try_new(FoldType::KFold, 0, 0.5, 42).is_err());
        assert!(FoldSpec::try_new(FoldType::Subsampling, 5, f64::NAN, 42).is_err());
        assert!(FoldSpec::try_new(FoldType::Subsampling, 5, 0.0, 42).is_err());
        assert!(FoldSpec::try_new(FoldType::Subsampling, 5, 1.0, 42).is_err());

        let spec = FoldSpec::try_new(FoldType::KFold, 5, 0.5, 42).unwrap();
        assert!(try_cv(&spec, 0, &[]).is_err());
        assert!(try_cv(&spec, 4, &[1.0; 4]).is_err()); // k > n
        assert!(try_cv(&spec, 5, &[1.0; 4]).is_err());
        assert!(try_cv(&spec, 5, &[1.0, 1.0, -1.0, 1.0, 1.0]).is_err());
        assert!(try_cv(&spec, 5, &[1.0, 1.0, f64::NAN, 1.0, 1.0]).is_err());

        let bootstrap = FoldSpec::try_new(FoldType::Bootstrap, 2, 0.5, 42).unwrap();
        assert!(try_cv(&bootstrap, 5, &[0.0; 5]).is_err());
    }

    #[test]
    fn optimal_mstop_of_offset_only_path_is_zero() {
        // An offset-only fit has the single iteration 0 and no step to stop at.
        let result = CvRiskResult {
            risk_matrix: Array2::zeros((2, 1)),
            aggregate: array![3.0],
            grid: vec![0],
            fold_type: "2-fold test".to_string(),
        };
        assert_eq!(result.optimal_mstop(), 0);
    }

    #[test]
    fn cvrisk_gaussian_two_predictors() {
        let y = array![3.0, 5.0, 7.0, 4.0, 6.0, 8.0, 2.0, 5.0, 9.0, 1.0];
        let w = Array1::ones(10);
        let x1 = array![1.0, 2.0, 3.0, 1.5, 2.5, 3.5, 0.5, 2.0, 4.0, 0.0];
        let x2 = array![3.0, 1.0, 2.0, 2.5, 0.5, 1.5, 3.5, 1.0, 0.5, 4.0];

        let preparer = TestPreparer {
            x_cols: vec![x1, x2],
            names: vec!["x1".to_string(), "x2".to_string()],
        };

        let spec = FoldSpec::new(FoldType::Subsampling, 5, 0.5, 42);
        let n = 10;
        let weights = vec![1.0; n];
        let folds = cv(&spec, n, &weights);

        let ctrl = BoostControl::new(20, 0.1).unwrap();
        let result = cvrisk(
            &preparer,
            y.view(),
            w.view(),
            &ctrl,
            &folds,
            None,
            true,
            "subsampling",
        );

        assert_eq!(result.risk_matrix.nrows(), 5);
        assert_eq!(result.risk_matrix.ncols(), 21);
        assert_eq!(result.aggregate.len(), 21);
        assert!(result.optimal_mstop() <= 20);
    }

    #[test]
    fn seeded_parallel_cv_and_selection_match_sequential_folds() {
        for seed in 201_u64..=206 {
            let n = 15 + seed as usize % 19;
            let mstop = 3 + seed as u32 % 11;
            let nu = [0.05, 0.1, 0.4][seed as usize % 3];
            let mut state = seed;
            let mut random = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as f64 / u64::MAX as f64
            };
            let x0 = Array1::from_shape_fn(n, |_| 2.0 * random() - 1.0);
            let x1 = x0.clone(); // deterministic lowest-index tie
            let x2 = Array1::from_shape_fn(n, |_| 2.0 * random() - 1.0);
            let y = Array1::from_shape_fn(n, |index| {
                1.4 * x0[index] - 0.6 * x2[index] + 0.05 * (random() - 0.5)
            });
            let weights = Array1::from_shape_fn(n, |_| 0.2 + 1.8 * random());
            let preparer = TestPreparer {
                x_cols: vec![x0, x1, x2],
                names: vec!["x0".into(), "x1".into(), "x2".into()],
            };
            let folds = cv(
                &FoldSpec::new(FoldType::KFold, 3, 0.5, seed),
                n,
                weights.as_slice().unwrap(),
            );
            let control = BoostControl::new(mstop, nu).unwrap();
            let sequential = try_cvrisk(
                &preparer,
                y.view(),
                weights.view(),
                &control,
                &folds,
                None,
                None,
                None,
                false,
                "kfold",
            )
            .unwrap();
            let parallel = try_cvrisk(
                &preparer,
                y.view(),
                weights.view(),
                &control,
                &folds,
                None,
                None,
                None,
                true,
                "kfold",
            )
            .unwrap();
            assert_eq!(
                sequential.risk_matrix, parallel.risk_matrix,
                "CV risk mismatch; seed={seed} spec=n={n},mstop={mstop},nu={nu}"
            );
            assert_eq!(sequential.aggregate, parallel.aggregate);

            let sequential_selected = try_cvrisk_selected_with_parallel(
                &preparer,
                y.view(),
                weights.view(),
                &control,
                &folds,
                None,
                None,
                Gaussian,
                false,
            )
            .unwrap();
            let parallel_selected = try_cvrisk_selected_with_parallel(
                &preparer,
                y.view(),
                weights.view(),
                &control,
                &folds,
                None,
                None,
                Gaussian,
                true,
            )
            .unwrap();
            assert_eq!(
                sequential_selected.selected, parallel_selected.selected,
                "stability-selection path mismatch; seed={seed} \
                 spec=n={n},mstop={mstop},nu={nu}"
            );
        }
    }

    #[test]
    fn try_cvrisk_with_rejects_invalid_fold_matrices() {
        let y = array![1.0, 2.0, 3.0, 4.0];
        let w = Array1::ones(4);
        let preparer = TestPreparer {
            x_cols: vec![array![0.0, 1.0, 2.0, 3.0]],
            names: vec!["x".to_string()],
        };
        let ctrl = BoostControl::new(3, 0.1).unwrap();

        let no_oob = Array2::ones((4, 2));
        assert!(try_cvrisk_with(
            &preparer,
            y.view(),
            w.view(),
            &ctrl,
            &no_oob,
            None,
            None,
            None,
            false,
            Gaussian,
            "custom",
        )
        .is_err());

        let wrong_rows = Array2::zeros((3, 2));
        assert!(try_cvrisk_with(
            &preparer,
            y.view(),
            w.view(),
            &ctrl,
            &wrong_rows,
            None,
            None,
            None,
            false,
            Gaussian,
            "custom",
        )
        .is_err());
    }

    #[test]
    fn try_cvrisk_selected_errors_on_invalid_input_and_matches_wrapper() {
        let y = array![1.0, 2.0, 3.0, 4.0];
        let w = Array1::ones(4);
        let preparer = TestPreparer {
            x_cols: vec![array![0.0, 1.0, 2.0, 3.0]],
            names: vec!["x".to_string()],
        };
        let ctrl = BoostControl::new(3, 0.1).unwrap();
        let spec = FoldSpec::new(FoldType::KFold, 2, 0.5, 42);
        let folds = cv(&spec, 4, w.as_slice().unwrap());

        // Valid invocation returns the same result as the panicking wrapper.
        // The wrapper is deprecated on the P0.8 schedule; this parity check
        // guards the shim until its 0.2.0 removal.
        let good = try_cvrisk_selected(&preparer, y.view(), w.view(), &ctrl, &folds, None).unwrap();
        #[allow(deprecated)]
        let wrapper_selected =
            cvrisk_selected(&preparer, y.view(), w.view(), &ctrl, &folds, None).selected;
        assert_eq!(wrapper_selected, good.selected);

        // A NaN response fails with a typed error before numerical dispatch.
        let nan_y = array![1.0, f64::NAN, 3.0, 4.0];
        assert!(matches!(
            try_cvrisk_selected(&preparer, nan_y.view(), w.view(), &ctrl, &folds, None),
            Err(MboostrsError::BadResponse(_, _))
        ));

        // A fold matrix with the wrong row count is rejected.
        let wrong_rows = Array2::zeros((3, 2));
        assert!(matches!(
            try_cvrisk_selected(&preparer, y.view(), w.view(), &ctrl, &wrong_rows, None),
            Err(MboostrsError::Shape(_))
        ));
    }

    #[test]
    fn dynamic_cvrisk_bmono_gram_matches_monomorphised_path() {
        let n = 40;
        let x: Array1<f64> = Array1::linspace(0.0, 1.0, n);
        let y = x.mapv(|value| value * value + 0.1 * (8.0 * value).sin());
        let w = Array1::ones(n);
        let preparer = BmonoPreparer { x };
        let spec = FoldSpec::try_new(FoldType::KFold, 4, 0.5, 17).unwrap();
        let folds = try_cv(&spec, n, w.as_slice().unwrap()).unwrap();
        let ctrl = BoostControl::new(8, 0.1).unwrap();

        let dynamic = cvrisk(
            &preparer,
            y.view(),
            w.view(),
            &ctrl,
            &folds,
            None,
            false,
            "cross-validation",
        );
        let concrete = try_cvrisk_with(
            &preparer,
            y.view(),
            w.view(),
            &ctrl,
            &folds,
            None,
            None,
            None,
            false,
            Gaussian,
            "cross-validation",
        )
        .unwrap();

        for (a, b) in dynamic.risk_matrix.iter().zip(concrete.risk_matrix.iter()) {
            assert!((a - b).abs() < 1e-10, "dynamic={a}, concrete={b}");
        }
    }

    // ─── Regression tests from Audit 1 & 2 ────────────────────────────────

    #[test]
    fn cvrisk_returns_oob_risk_not_inbag() {
        // R3 guard: cvrisk must return out-of-bag risk, not the inbag risk
        // from the full-data fit.
        use crate::learners::prepare_scalar_bols;
        use ndarray::Array1;

        let n = 50usize;
        let mut rng = 123u64;
        let mut xs = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng as f64) / (u64::MAX as f64)
        };
        let x = Array1::from_shape_fn(n, |_| xs());
        let y = Array1::from_shape_fn(n, |i| 2.0 * x[i] + (xs() - 0.5));
        let w = Array1::ones(n);

        struct OobPreparer {
            x: Array1<f64>,
        }
        impl LearnerPreparer for OobPreparer {
            fn prepare(&self, weights: ArrayView1<f64>) -> Vec<PreparedLearner> {
                vec![prepare_scalar_bols(
                    self.x.view(),
                    weights,
                    0.0,
                    true,
                    "x",
                    None,
                )]
            }
            fn family(&self) -> Box<dyn Family> {
                Box::new(Gaussian)
            }
        }

        let spec = FoldSpec::new(FoldType::KFold, 5, 0.5, 42);
        let folds = cv(&spec, n, w.as_slice().unwrap());
        let preparer = OobPreparer { x };
        let ctrl = BoostControl::new(20, 0.1).unwrap();
        let result = cvrisk(
            &preparer,
            y.view(),
            w.view(),
            &ctrl,
            &folds,
            None,
            true,
            "cross-validation",
        );

        // OOB risk should be finite and non-negative.
        for &v in &result.aggregate {
            assert!(v.is_finite(), "OOB risk not finite: {v}");
            assert!(v >= 0.0, "OOB risk negative: {v}");
        }
        // OOB risk should improve (decrease) over iterations.
        let oob_improvement = result.aggregate.first().unwrap() - result.aggregate.last().unwrap();
        assert!(
            oob_improvement >= 0.0,
            "OOB risk increased: first={}, last={}",
            result.aggregate.first().unwrap(),
            result.aggregate.last().unwrap()
        );
    }

    #[test]
    fn cvrisk_kronecker_multinomial_matches_increment_walk() {
        // The Kronecker arm of the compacted OOB walk must reproduce the
        // reference computed by walking fit()'s stored full-length increments
        // and evaluating the OOB risk on the compacted entries. Guards both
        // the flattened (obs·n_resp + class) indexing — the pre-2026-06-11 arm
        // mis-indexed f_oob for n_resp > 1 — and the matmul restructuring.
        use crate::families::Multinomial;
        use crate::learners::prepare_kronecker_bols;

        let n_obs = 30usize;
        let nr = 2usize; // K = 3 classes
        let n_flat = n_obs * nr;

        // Deterministic data: x in [0,1), class from thirds of x.
        let x: Vec<f64> = (0..n_obs)
            .map(|i| ((i * 7919) % 100) as f64 / 100.0)
            .collect();
        let mut y = vec![0.0f64; n_flat];
        for i in 0..n_obs {
            let class = ((x[i] * 3.0) as usize).min(2);
            if class < nr {
                y[i * nr + class] = 1.0;
            } // class == 2 is the reference: all-zero indicators
        }
        let yv = Array1::from_vec(y);
        let w = Array1::from_elem(n_flat, 1.0);

        struct KronPreparer {
            x: Vec<f64>,
            nr: usize,
        }
        impl KronPreparer {
            fn build(&self, weights: ArrayView1<f64>) -> Vec<PreparedLearner> {
                let n_obs = self.x.len();
                // Fold weights are flattened like the response: one block of
                // `nr` equal entries per observation. The learner takes one
                // weight per observation, the first entry of each block.
                let obs_weights = Array1::from_iter(weights.iter().step_by(self.nr).copied());
                let mut x1 = ndarray::Array2::zeros((n_obs, 2));
                for i in 0..n_obs {
                    x1[[i, 0]] = 1.0;
                    x1[[i, 1]] = self.x[i];
                }
                vec![prepare_kronecker_bols(
                    x1,
                    obs_weights.view(),
                    ndarray::Array2::zeros((2, 2)),
                    ndarray::Array2::eye(self.nr),
                    false,
                    "kron(x)",
                )]
            }
        }
        impl LearnerPreparer for KronPreparer {
            fn prepare(&self, weights: ArrayView1<f64>) -> Vec<PreparedLearner> {
                self.build(weights)
            }
            fn family(&self) -> Box<dyn Family> {
                Box::new(Multinomial::new(3))
            }
        }

        // Folds: hold out whole observations (both flattened entries together).
        let b = 3usize;
        let mut folds = Array2::ones((n_flat, b));
        for fold in 0..b {
            for i in 0..n_obs {
                if i % b == fold {
                    folds[[i * nr, fold]] = 0.0;
                    folds[[i * nr + 1, fold]] = 0.0;
                }
            }
        }

        let preparer = KronPreparer { x, nr };
        let ctrl = BoostControl::new(15, 0.1).unwrap();
        let result = cvrisk_with(
            &preparer,
            yv.view(),
            w.view(),
            &ctrl,
            &folds,
            None,
            false,
            Multinomial::new(3),
            "cross-validation",
        );

        // Reference: per fold, general-path fit with stored increments, walk
        // the FULL flattened f, evaluate risk on the OOB-compacted entries.
        for fold in 0..b {
            let fold_w = Array1::from_vec((0..n_flat).map(|i| folds[[i, fold]]).collect());
            let oob_idx: Vec<usize> = (0..n_flat).filter(|&i| folds[[i, fold]] == 0.0).collect();
            // The observation-level normalization: the flattened OOB entries
            // repeat each observation `nr` times, so the divisor is the
            // observation count (`oob_idx.len() / nr`), matching
            // `flat_response_span` in `try_cvrisk_with`.
            let oob_sum = (oob_idx.len() / nr) as f64;
            let y_oob = yv.select(ndarray::Axis(0), &oob_idx);
            let w_oob = Array1::from_elem(oob_idx.len(), 1.0);

            let mut learners = preparer.build(fold_w.view());
            let mut fam = Multinomial::new(3);
            let path = crate::engine::fit(
                &mut learners,
                yv.view(),
                fold_w.view(),
                &mut fam,
                &ctrl,
                None,
            );

            let mut f_full = Array1::from_elem(n_flat, path.offset);
            let fam_eval = Multinomial::new(3);
            let f_oob = f_full.select(ndarray::Axis(0), &oob_idx);
            let want0 = fam_eval.held_out_risk(y_oob.view(), f_oob.view(), w_oob.view()) / oob_sum;
            assert!(
                (result.risk_matrix[[fold, 0]] - want0).abs() < 1e-12,
                "fold {fold} step 0"
            );
            for m in 0..path.increments.len() {
                for i in 0..n_flat {
                    f_full[i] += path.increments[m][i];
                }
                let f_oob = f_full.select(ndarray::Axis(0), &oob_idx);
                let want =
                    fam_eval.held_out_risk(y_oob.view(), f_oob.view(), w_oob.view()) / oob_sum;
                let got = result.risk_matrix[[fold, m + 1]];
                assert!(
                    (got - want).abs() < 1e-10,
                    "fold {fold} step {}: got {got}, want {want}",
                    m + 1
                );
            }
        }
    }

    #[test]
    fn fold_label_mentions_fold_type() {
        // S3 guard: fold_type string should reflect the actual fold method,
        // not always say "bootstrap".
        use crate::learners::prepare_scalar_bols;
        use ndarray::Array1;

        let n = 20usize;
        let y = Array1::from_shape_fn(n, |i| i as f64);
        let w = Array1::ones(n);
        let x_ref: Array1<f64> = Array1::from_shape_fn(n, |i| i as f64 * 2.0);

        struct FoldLabelPreparer {
            x: Array1<f64>,
        }
        impl LearnerPreparer for FoldLabelPreparer {
            fn prepare(&self, weights: ArrayView1<f64>) -> Vec<PreparedLearner> {
                vec![prepare_scalar_bols(
                    self.x.view(),
                    weights,
                    0.0,
                    true,
                    "x",
                    None,
                )]
            }
            fn family(&self) -> Box<dyn Family> {
                Box::new(Gaussian)
            }
        }

        let preparer = FoldLabelPreparer { x: x_ref };

        for (fold_type, label_fragment) in [
            (FoldType::KFold, "fold"),
            (FoldType::Bootstrap, "bootstrap"),
            (FoldType::Subsampling, "subsampl"),
        ] {
            let spec = FoldSpec::new(fold_type, 3, 0.5, 1);
            let folds = cv(&spec, n, w.as_slice().unwrap());
            let ctrl = BoostControl::new(5, 0.1).unwrap();
            let result = cvrisk(
                &preparer,
                y.view(),
                w.view(),
                &ctrl,
                &folds,
                None,
                true,
                fold_type.label(),
            );
            let label = result.fold_type.to_lowercase();
            assert!(
                label.contains(label_fragment),
                "fold_type='{label}' should contain '{label_fragment}'",
            );
        }
    }

    mod extending {
        use super::*;
        use crate::families::{Binomial, NBinomial, Poisson, PropOdds};

        fn fixture(n: usize) -> (TestPreparer, Array1<f64>) {
            let mut state = 7u64;
            let mut next = move || {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 11) as f64) / ((1u64 << 53) as f64)
            };
            let x1: Array1<f64> = (0..n).map(|_| next() * 2.0 - 1.0).collect();
            let x2: Array1<f64> = (0..n).map(|_| next() * 2.0 - 1.0).collect();
            let u: Array1<f64> = (0..n).map(|_| next()).collect();
            let preparer = TestPreparer {
                x_cols: vec![x1, x2],
                names: vec!["x1".into(), "x2".into()],
            };
            (preparer, u)
        }

        fn folds(n: usize) -> Array2<f64> {
            try_cv(
                &FoldSpec::try_new(FoldType::KFold, 4, 0.5, 3).unwrap(),
                n,
                &vec![1.0; n],
            )
            .unwrap()
        }

        fn compare<F: Family + Clone>(family: F, y: Array1<f64>, preparer: &TestPreparer) {
            let n = preparer.x_cols[0].len();
            let w = Array1::ones(n);
            let folds = folds(n);
            let rule = ExtendRule {
                start: 10,
                max_mstop: 400,
                patience: None,
                boundary_share: 0.02,
                max_extensions: None,
            };
            let ctrl = BoostControl::new(10, 0.1).unwrap();
            let (grown, report) = try_cvrisk_extending(
                preparer,
                y.view(),
                w.view(),
                &ctrl,
                &folds,
                None,
                None,
                None,
                false,
                family.clone(),
                "kfold",
                &rule,
            )
            .unwrap();
            let end = *report.grid_ends.last().unwrap();
            assert!(
                report.grid_ends.len() > 1,
                "the fixture should need an extension"
            );
            let full_ctrl = BoostControl::new(end as u32, 0.1).unwrap();
            let full = try_cvrisk_with(
                preparer,
                y.view(),
                w.view(),
                &full_ctrl,
                &folds,
                None,
                None,
                None,
                false,
                family,
                "kfold",
            )
            .unwrap();
            assert_eq!(grown.risk_matrix.dim(), full.risk_matrix.dim());
            for (a, b) in grown.risk_matrix.iter().zip(full.risk_matrix.iter()) {
                assert!((a - b).abs() <= 1e-10 * (1.0 + b.abs()), "{a} vs {b}");
            }
            assert_eq!(grown.optimal_mstop(), full.optimal_mstop());
            assert_eq!(report.best, full.optimal_mstop());
        }

        #[test]
        fn a_grown_grid_equals_the_full_grid_for_each_family() {
            let n = 200;
            let (preparer, u) = fixture(n);
            let eta: Array1<f64> = (0..n)
                .map(|i| 1.5 * preparer.x_cols[0][i] - preparer.x_cols[1][i])
                .collect();
            let gauss: Array1<f64> = (0..n).map(|i| eta[i] + (u[i] - 0.5)).collect();
            compare(Gaussian, gauss, &preparer);
            let binary: Array1<f64> = (0..n)
                .map(|i| {
                    if u[i] < 1.0 / (1.0 + (-eta[i]).exp()) {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect();
            compare(Binomial, binary, &preparer);
            let counts: Array1<f64> = (0..n)
                .map(|i| ((0.5 * eta[i]).exp() * 2.0 * u[i] * 2.0).floor())
                .collect();
            compare(Poisson, counts.clone(), &preparer);
            compare(NBinomial::new(), counts, &preparer);
            let ordered: Array1<f64> = (0..n)
                .map(|i| (eta[i] + u[i]).clamp(-1.0, 1.99).floor() + 2.0)
                .collect();
            compare(PropOdds::new(3), ordered, &preparer);
        }

        #[test]
        fn patience_selects_the_full_grid_minimum_with_a_shorter_grid() {
            let n = 200;
            let (preparer, u) = fixture(n);
            let y: Array1<f64> = (0..n)
                .map(|i| 1.5 * preparer.x_cols[0][i] + 4.0 * (u[i] - 0.5))
                .collect();
            let w = Array1::ones(n);
            let folds = folds(n);
            let ctrl = BoostControl::new(10, 0.1).unwrap();
            let full = try_cvrisk_with(
                &preparer,
                y.view(),
                w.view(),
                &BoostControl::new(1000, 0.1).unwrap(),
                &folds,
                None,
                None,
                None,
                false,
                Gaussian,
                "kfold",
            )
            .unwrap();
            let rule = ExtendRule {
                start: 10,
                max_mstop: 1000,
                patience: Some(25),
                boundary_share: 0.02,
                max_extensions: None,
            };
            let (grown, report) = try_cvrisk_extending(
                &preparer,
                y.view(),
                w.view(),
                &ctrl,
                &folds,
                None,
                None,
                None,
                true,
                Gaussian,
                "kfold",
                &rule,
            )
            .unwrap();
            assert_eq!(report.best, full.optimal_mstop());
            assert!(!report.at_boundary);
            assert_eq!(*report.grid_ends.last().unwrap(), report.best + 25);
            assert!(grown.aggregate.len() < full.aggregate.len());
        }

        #[test]
        fn a_search_that_reaches_max_mstop_reports_the_boundary() {
            let n = 120;
            let (preparer, _) = fixture(n);
            let y: Array1<f64> = preparer.x_cols[0].mapv(|v| 3.0 * v);
            let w = Array1::ones(n);
            let rule = ExtendRule {
                start: 5,
                max_mstop: 40,
                patience: None,
                boundary_share: 0.02,
                max_extensions: None,
            };
            let (grown, report) = try_cvrisk_extending(
                &preparer,
                y.view(),
                w.view(),
                &BoostControl::new(5, 0.1).unwrap(),
                &folds(n),
                None,
                None,
                None,
                false,
                Gaussian,
                "kfold",
                &rule,
            )
            .unwrap();
            assert!(report.at_boundary);
            assert_eq!(report.grid_ends, vec![5, 10, 20, 40]);
            assert_eq!(grown.aggregate.len(), 41);
        }
    }
}
