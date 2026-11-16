//! Boosting path: stores the trajectory of a boosting fit.

use crate::report::FitReport;
use crate::report::PathRow;
use crate::MboostrsError;
use ndarray::Array1;

/// The complete path of a component-wise boosting fit.
///
/// Contains the offset, per-iteration learner selections, coefficients,
/// empirical risk values, the final fitted values, and per-iteration
/// prediction increments (nu * best_pred) for model slicing.
#[derive(Clone, Debug)]
pub struct BoostingPath {
    /// Versioned description of the numerical route used for this fit.
    pub report: FitReport,
    /// The intercept / offset added to all predictions.
    pub offset: f64,
    /// Index of the selected base-learner at each iteration.
    pub selected: Vec<usize>,
    /// Coefficient vector for the selected learner at each iteration.
    pub coefs: Vec<Vec<f64>>,
    /// Empirical risk at iteration 0 (offset-only) through `mstop`.
    pub risk: Vec<f64>,
    /// Final fitted values on the link scale.
    pub fitted: Array1<f64>,
    /// Per-iteration prediction increments: nu * best_pred at each step.
    /// Used for model slicing: `fitted[:k] = offset + sum(increments[:k])`.
    pub increments: Vec<Vec<f64>>,
    /// Per-iteration scalar nuisance (e.g. NBinomial/Weibull/Loglog/GaussReg
    /// sigma), aligned with `risk` (length mstop+1). `None` for families
    /// without a scalar nuisance. Cross-validation reuses these INBAG per-step
    /// values for the out-of-bag risk, instead of re-estimating the nuisance on
    /// the OOB sample (which diverges from R's cvrisk semantics).
    pub nuisance_path: Vec<Option<f64>>,
    /// Per-iteration vector nuisance (e.g. PropOdds K-1 sigma thresholds),
    /// aligned with `risk` (length mstop+1). Each entry is the threshold vector
    /// after calibration at that iteration. Empty for families without a vector
    /// nuisance. Cross-validation reuses these INBAG per-step values for OOB
    /// risk, and truncated prediction uses the threshold vector at the requested
    /// iteration rather than the final one.
    pub nuisance_vec_path: Vec<Vec<f64>>,
    /// Final vector nuisance from the fitted family (e.g. PropOdds K-1 sigma
    /// thresholds). Empty for families without a vector nuisance. Kept for
    /// backward compatibility; prefer `nuisance_vec_path` for path-aware code.
    pub nuisance_vec: Vec<f64>,
    /// The negative gradient the loop held when it stopped -- the vector the
    /// next iteration would regress on.
    ///
    /// Retained so a path can be *continued* rather than replayed. For a
    /// squared-error family the loop maintains this incrementally
    /// (`u -= nu * pred`) instead of re-evaluating it, and the accumulated
    /// rounding of that subtraction is not what a fresh evaluation at the same
    /// fitted values gives. Recomputing it on resume therefore puts the
    /// continued path one unit in the last place away from the cold one at the
    /// resume step, and the difference compounds. Carrying it makes a
    /// continued path bit-for-bit equal to a cold one.
    ///
    /// Empty when the fit did not retain it (the Gaussian Gram route, and
    /// selection-only fits).
    pub residual: Array1<f64>,
}

impl BoostingPath {
    /// Return nuisance values in one vector-valued representation.
    ///
    /// Vector nuisances take precedence; scalar nuisances are represented as
    /// one-element vectors. Entries remain aligned with `risk`, including the
    /// offset-only state at index zero.
    pub fn nuisance_rows(&self) -> Vec<Vec<f64>> {
        let n = self.nuisance_vec_path.len().max(self.nuisance_path.len());
        (0..n)
            .map(|index| {
                self.nuisance_vec_path
                    .get(index)
                    .filter(|value| !value.is_empty())
                    .cloned()
                    .or_else(|| {
                        self.nuisance_path
                            .get(index)
                            .and_then(|value| value.map(|scalar| vec![scalar]))
                    })
                    .unwrap_or_default()
            })
            .collect()
    }

    /// Derive stable, one-based iteration rows without instrumenting the hot loop.
    pub fn path_rows(&self) -> Vec<PathRow> {
        let nuisance_rows = self.nuisance_rows();
        self.selected
            .iter()
            .enumerate()
            .map(|(step, &learner_index)| {
                let risk = self.risk.get(step + 1).copied().filter(|v| v.is_finite());
                let previous_risk = self.risk.get(step).copied().filter(|v| v.is_finite());
                let risk_change = previous_risk
                    .zip(risk)
                    .map(|(before, after)| before - after);
                let coefficient_norm = self
                    .coefs
                    .get(step)
                    .map(|coef| coef.iter().map(|value| value * value).sum::<f64>().sqrt())
                    .unwrap_or(0.0);
                let nuisance = nuisance_rows.get(step + 1).cloned().unwrap_or_default();
                PathRow {
                    iteration: step + 1,
                    cycle: None,
                    parameter_index: None,
                    learner_index,
                    risk,
                    risk_change,
                    coefficient_norm,
                    nuisance,
                }
            })
            .collect()
    }
}

impl BoostingPath {
    /// Number of boosting iterations actually performed.
    pub fn mstop(&self) -> usize {
        self.selected.len()
    }
}

/// Result of coefficient aggregation for a GLMBoost model.
///
/// `values[i]` is the aggregated coefficient for learner index `indices[i]`.
/// When `off2int` is true and learner 0 (intercept) is present, the
/// intercept already includes the offset and centering correction
/// (`intercept = raw_intercept - sum(beta_j * mean_j) + offset`).
#[derive(Clone, Debug)]
pub struct CoefResult {
    /// Learner indices (0 = intercept, 1..p = predictors).
    pub indices: Vec<usize>,
    /// Aggregated coefficient values.
    pub values: Vec<f64>,
}

/// Aggregate per-step coefficients into per-learner sums, matching R's
/// `coef.glmboost(..., aggregate = "sum")`.
///
/// # Arguments
/// * `selected` - Learner index selected at each boosting step (0 = intercept).
/// * `coefs_flat` - Scalar coefficient at each step.
/// * `nu` - Learning rate (shrinkage factor).
/// * `n_learners` - Total number of learners (1 intercept + p predictors).
/// * `centering_means` - Column means used for centering during fit (length p).
///   Index j corresponds to learner index j+1. May be empty (no centering).
/// * `off2int` - If true, fold offset and centering correction into the intercept.
/// * `offset` - The fitted offset value.
/// * `which` - Optional subset of learner indices to return. None returns all
///   with nonzero coefficients.
/// Check that a boosting path's parallel arrays describe the same path.
///
/// `selected[m]` and the coefficient at step `m` are two halves of one record,
/// and every function here walks them together. Nothing checked that they had
/// the same length, so a caller that built one from a truncated path and the
/// other from the full one got a panic from the indexing operator -- an
/// `index out of bounds` with no mention of which two things disagreed.
///
/// A slice that is *longer* than the path is not an error: `mstop` legitimately
/// truncates, and every function here reads only the first `max_m` entries.
fn check_path_lengths(
    selected_len: usize,
    coefficients_len: usize,
    what: &str,
) -> Result<(), MboostrsError> {
    if coefficients_len < selected_len {
        return Err(MboostrsError::Shape(format!(
            "the boosting path has {selected_len} selected steps but only \
             {coefficients_len} {what}; the two describe different paths"
        )));
    }
    Ok(())
}

/// Check that a design slice holds `n * p` values.
fn check_design(design_len: usize, n: usize, p: usize) -> Result<(), MboostrsError> {
    let wanted = n.saturating_mul(p);
    if design_len != wanted {
        return Err(MboostrsError::Shape(format!(
            "design has {design_len} values but {n} rows by {p} columns needs \
             {wanted}"
        )));
    }
    Ok(())
}

/// The linear model's fitted design: centered predictor columns and, when
/// requested, a leading column of ones. Uses the means retained at fitting;
/// observation weights do not change glmboost's column centering.
pub fn try_glmboost_design(
    x: ndarray::ArrayView2<f64>,
    centering_means: &[f64],
    intercept: bool,
) -> Result<ndarray::Array2<f64>, MboostrsError> {
    if !centering_means.is_empty() && centering_means.len() != x.ncols() {
        return Err(MboostrsError::Shape(format!(
            "centering_means has {} entries but the design has {} predictors",
            centering_means.len(),
            x.ncols()
        )));
    }
    let first = usize::from(intercept);
    let mut design = ndarray::Array2::ones((x.nrows(), x.ncols() + first));
    for j in 0..x.ncols() {
        let mean = centering_means.get(j).copied().unwrap_or(0.0);
        for i in 0..x.nrows() {
            design[[i, j + first]] = x[[i, j]] - mean;
        }
    }
    Ok(design)
}

/// [`coef_glmboost`], returning an error instead of panicking on a mismatch.
///
/// Prefer this at any boundary where the path arrays come from separate
/// sources -- a deserialized artifact, a binding, a caller's own bookkeeping.
/// The deprecated panicking form stays only as a compatibility shim.
///
/// ```
/// use mboostrs_core::path::try_coef_glmboost;
///
/// // A two-learner model whose path selected learner 1 twice at nu = 0.5.
/// // The result pairs learner indices with their aggregated values, and
/// // reports only the learners the path actually touched.
/// let coefficients = try_coef_glmboost(
///     &[1, 1], &[2.0, 1.0], 0.5, 2, &[], false, 0.0, None,
/// )?;
/// let position = coefficients.indices.iter().position(|&i| i == 1).unwrap();
/// assert_eq!(coefficients.values[position], 1.5);   // 0.5 * (2.0 + 1.0)
///
/// // A path array shorter than the selection sequence describes a different
/// // path, and is reported rather than indexed past the end.
/// let mismatch = try_coef_glmboost(
///     &[0, 1, 0], &[0.5, 0.25], 0.1, 3, &[], false, 0.0, None,
/// );
/// assert!(mismatch.unwrap_err().to_string().contains("different paths"));
/// # Ok::<(), mboostrs_core::MboostrsError>(())
/// ```
#[allow(clippy::too_many_arguments)]
pub fn try_coef_glmboost(
    selected: &[usize],
    coefs_flat: &[f64],
    nu: f64,
    n_learners: usize,
    centering_means: &[f64],
    off2int: bool,
    offset: f64,
    which: Option<&[usize]>,
) -> Result<CoefResult, MboostrsError> {
    check_path_lengths(selected.len(), coefs_flat.len(), "coefficients")?;
    if let Some(indices) = which {
        if let Some(&bad) = indices.iter().find(|&&index| index >= n_learners) {
            return Err(MboostrsError::Shape(format!(
                "which requests learner {bad}, but the model has {n_learners}"
            )));
        }
    }
    Ok(coef_glmboost_unchecked(
        selected,
        coefs_flat,
        nu,
        n_learners,
        centering_means,
        off2int,
        offset,
        which,
    ))
}

/// [`coef_gamboost`], returning an error instead of panicking on a mismatch.
pub fn try_coef_gamboost(
    selected: &[usize],
    coefs: &[Vec<f64>],
    nu: f64,
    n_learners: usize,
    which: Option<&[usize]>,
) -> Result<GAMCoefResult, MboostrsError> {
    check_path_lengths(selected.len(), coefs.len(), "coefficient vectors")?;
    if let Some(indices) = which {
        if let Some(&bad) = indices.iter().find(|&&index| index >= n_learners) {
            return Err(MboostrsError::Shape(format!(
                "which requests learner {bad}, but the model has {n_learners}"
            )));
        }
    }
    Ok(coef_gamboost_unchecked(
        selected, coefs, nu, n_learners, which,
    ))
}

/// [`predict_glmboost`], returning an error instead of panicking on a mismatch.
///
/// ```
/// use mboostrs_core::path::try_predict_glmboost;
///
/// // Three observations, two columns, stored row-major.
/// let design = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let eta = try_predict_glmboost(
///     1.0, &[1], &[2.0], 0.5, &design, &[0.0, 0.0], None, 3, 2,
/// )?;
/// assert_eq!(eta.len(), 3);
///
/// // A design whose length is not `n * p` is reported with both numbers.
/// let wrong = try_predict_glmboost(
///     0.0, &[1], &[0.5], 0.1, &[1.0; 5], &[], None, 3, 2,
/// );
/// assert!(wrong.unwrap_err().to_string().contains("needs 6"));
/// # Ok::<(), mboostrs_core::MboostrsError>(())
/// ```
#[allow(clippy::too_many_arguments)]
pub fn try_predict_glmboost(
    offset: f64,
    selected: &[usize],
    coefs_flat: &[f64],
    nu: f64,
    x_matrix: &[f64],
    centering_means: &[f64],
    mstop: Option<usize>,
    n: usize,
    p: usize,
) -> Result<Vec<f64>, MboostrsError> {
    try_predict_glmboost_with_offset_vec(
        offset,
        None,
        selected,
        coefs_flat,
        nu,
        x_matrix,
        centering_means,
        mstop,
        n,
        p,
    )
}

/// `try_predict_glmboost` with a per-observation offset (R's `offset=`
/// semantics): each row's linear predictor starts at its own known
/// contribution rather than at `offset`. `offset_vec` wins over `offset`
/// when both are supplied, and must have one entry per row.
pub fn try_predict_glmboost_with_offset_vec(
    offset: f64,
    offset_vec: Option<&[f64]>,
    selected: &[usize],
    coefs_flat: &[f64],
    nu: f64,
    x_matrix: &[f64],
    centering_means: &[f64],
    mstop: Option<usize>,
    n: usize,
    p: usize,
) -> Result<Vec<f64>, MboostrsError> {
    let max_m = mstop.unwrap_or(selected.len()).min(selected.len());
    check_path_lengths(max_m, coefs_flat.len(), "coefficients")?;
    check_design(x_matrix.len(), n, p)?;
    if let Some(vector) = offset_vec {
        if vector.len() != n {
            return Err(MboostrsError::Shape(format!(
                "offset has {} entries but {n} rows are predicted",
                vector.len()
            )));
        }
    }
    Ok(predict_glmboost_unchecked(
        offset,
        offset_vec,
        selected,
        coefs_flat,
        nu,
        x_matrix,
        centering_means,
        mstop,
        n,
        p,
    ))
}

/// Combine the flattened per-iteration coefficients into a per-learner
/// coefficient matrix for a `glmboost`.
///
/// Panicking compatibility wrapper around [`try_coef_glmboost`], deprecated
/// on the P0.8 schedule. Prefer the fallible form at any boundary where the
/// path arrays can disagree. Removal: 0.2.0.
#[deprecated(
    since = "0.1.3",
    note = "use try_coef_glmboost (returns MboostrsError instead of panicking); removed in 0.2.0"
)]
pub fn coef_glmboost(
    selected: &[usize],
    coefs_flat: &[f64],
    nu: f64,
    n_learners: usize,
    centering_means: &[f64],
    off2int: bool,
    offset: f64,
    which: Option<&[usize]>,
) -> CoefResult {
    try_coef_glmboost(
        selected,
        coefs_flat,
        nu,
        n_learners,
        centering_means,
        off2int,
        offset,
        which,
    )
    .unwrap_or_else(|e| panic!("coef_glmboost: {e}"))
}

/// Unchecked aggregation body behind [`try_coef_glmboost`]; the fallible form
/// validates `selected`/`coefs_flat` length agreement and `which` bounds
/// before this runs.
fn coef_glmboost_unchecked(
    selected: &[usize],
    coefs_flat: &[f64],
    nu: f64,
    n_learners: usize,
    centering_means: &[f64],
    off2int: bool,
    offset: f64,
    which: Option<&[usize]>,
) -> CoefResult {
    let mut agg = vec![0.0f64; n_learners];
    for (step, &sel) in selected.iter().enumerate() {
        if sel < n_learners {
            agg[sel] += nu * coefs_flat[step];
        }
    }

    // Centering correction: intercept -= sum(beta_j * mean_j)
    // This matches R's coef.glmboost logic where cm != 0 triggers the adjustment.
    let has_centering =
        !centering_means.is_empty() && centering_means.iter().any(|&m| m != 0.0) && n_learners > 1;

    if has_centering {
        for j in 0..centering_means.len().min(n_learners - 1) {
            agg[0] -= agg[j + 1] * centering_means[j];
        }
    }

    // off2int: add offset to intercept
    if off2int && n_learners > 0 {
        agg[0] += offset;
    }

    let indices: Vec<usize> = match which {
        Some(w) => {
            let mut v: Vec<usize> = w.to_vec();
            v.sort_unstable();
            v.dedup();
            v
        }
        None => {
            let mut v: Vec<usize> = (0..n_learners).filter(|&i| agg[i] != 0.0).collect();
            v.sort_unstable();
            v
        }
    };

    let values: Vec<f64> = indices.iter().map(|&i| agg[i]).collect();

    CoefResult { indices, values }
}

/// Effects at the levels of a centered indicator design.
///
/// `centering[j]` is the training mean of indicator column `j`, including
/// observation weights when the fit used them. Each level has effect
/// `coef[j] - sum_k centering[k] * coef[k]`. Empty centering denotes an
/// uncentered design; otherwise there must be one center per coefficient.
pub fn centered_group_effects(coef: &[f64], centering: &[f64]) -> Result<Vec<f64>, MboostrsError> {
    if !centering.is_empty() && centering.len() != coef.len() {
        return Err(MboostrsError::Shape(
            "group centering must match the coefficient count".into(),
        ));
    }
    if coef.iter().chain(centering).any(|v| !v.is_finite()) {
        return Err(MboostrsError::Control(
            "group coefficients and centering must be finite".into(),
        ));
    }
    let shift: f64 = coef.iter().zip(centering).map(|(b, c)| b * c).sum();
    Ok(coef.iter().map(|b| b - shift).collect())
}

/// Result of coefficient aggregation for a GAMBoost model.
///
/// `values[i]` is the aggregated coefficient vector for learner
/// `indices[i]`.
#[derive(Clone, Debug)]
pub struct GAMCoefResult {
    /// Learner indices.
    pub indices: Vec<usize>,
    /// Aggregated coefficient vectors (one per index).
    pub values: Vec<Vec<f64>>,
}

/// Aggregate per-step coefficient vectors into per-learner sums,
/// matching R's `coef.gamboost(..., aggregate = "sum")`.
///
/// # Arguments
/// * `selected` - Learner index selected at each boosting step.
/// * `coefs` - Coefficient vector at each step.
/// * `nu` - Learning rate (shrinkage factor).
/// * `n_learners` - Total number of learners.
/// * `which` - Optional subset of learner indices to return. None returns
///   all learners that were selected at least once.
pub fn coef_gamboost(
    selected: &[usize],
    coefs: &[Vec<f64>],
    nu: f64,
    n_learners: usize,
    which: Option<&[usize]>,
) -> GAMCoefResult {
    try_coef_gamboost(selected, coefs, nu, n_learners, which)
        .unwrap_or_else(|e| panic!("coef_gamboost: {e}"))
}

/// Unchecked aggregation body behind [`try_coef_gamboost`].
fn coef_gamboost_unchecked(
    selected: &[usize],
    coefs: &[Vec<f64>],
    nu: f64,
    n_learners: usize,
    which: Option<&[usize]>,
) -> GAMCoefResult {
    // Single-pass accumulation into per-learner buffers.
    let mut agg: Vec<Option<Vec<f64>>> = vec![None; n_learners];
    for (step, &sel) in selected.iter().enumerate() {
        if sel < n_learners {
            let c = &coefs[step];
            let acc = &mut agg[sel];
            if acc.is_none() {
                acc.replace(c.iter().map(|v| nu * v).collect());
            } else {
                let buf = acc.as_mut().unwrap();
                for (a, &b) in buf.iter_mut().zip(c.iter()) {
                    *a += nu * b;
                }
            }
        }
    }

    let indices: Vec<usize> = match which {
        Some(w) => {
            let mut v: Vec<usize> = w.to_vec();
            v.sort_unstable();
            v.dedup();
            v
        }
        None => {
            let mut v: Vec<usize> = agg
                .iter()
                .enumerate()
                .filter(|(_, v)| v.is_some())
                .map(|(i, _)| i)
                .collect();
            v.sort_unstable();
            v
        }
    };

    let values: Vec<Vec<f64>> = indices
        .iter()
        .map(|&i| match &agg[i] {
            Some(v) => v.clone(),
            None => vec![],
        })
        .collect();

    GAMCoefResult { indices, values }
}

/// Predict from a fitted GLMBoost model on new data.
///
/// Panicking compatibility wrapper around [`try_predict_glmboost`], deprecated
/// on the P0.8 schedule. Prefer the fallible form at any boundary where the
/// path arrays can disagree. Removal: 0.2.0.
///
/// # Arguments
/// * `offset` - Fitted offset value.
/// * `selected` - Learner indices selected at each step (0 = intercept, 1..p = predictors).
/// * `coefs_flat` - Scalar coefficient at each step.
/// * `nu` - Learning rate.
/// * `x_matrix` - New design matrix in row-major order (length n * p).
/// * `centering_means` - Column means for centering (length p).
/// * `mstop` - Maximum number of iterations. None means all.
/// * `n` - Number of observations (rows in x_matrix).
/// * `p` - Number of predictors (columns in x_matrix).
#[deprecated(
    since = "0.1.3",
    note = "use try_predict_glmboost (returns MboostrsError instead of panicking); removed in 0.2.0"
)]
pub fn predict_glmboost(
    offset: f64,
    selected: &[usize],
    coefs_flat: &[f64],
    nu: f64,
    x_matrix: &[f64],
    centering_means: &[f64],
    mstop: Option<usize>,
    n: usize,
    p: usize,
) -> Vec<f64> {
    try_predict_glmboost(
        offset,
        selected,
        coefs_flat,
        nu,
        x_matrix,
        centering_means,
        mstop,
        n,
        p,
    )
    .unwrap_or_else(|e| panic!("predict_glmboost: {e}"))
}

/// Unchecked prediction body behind [`try_predict_glmboost`]; the fallible
/// form validates path-array agreement and the design shape before this runs.
fn predict_glmboost_unchecked(
    offset: f64,
    offset_vec: Option<&[f64]>,
    selected: &[usize],
    coefs_flat: &[f64],
    nu: f64,
    x_matrix: &[f64],
    centering_means: &[f64],
    mstop: Option<usize>,
    n: usize,
    p: usize,
) -> Vec<f64> {
    let max_m = mstop.unwrap_or(selected.len()).min(selected.len());

    // The predictor is linear in the coefficients, so aggregate each column's
    // total first (one O(mstop) scalar pass), then apply with at most one
    // O(n) pass per selected column — instead of one O(n) pass per boosting
    // step. Same aggregation as `coef_glmboost`.
    let mut agg = vec![0.0f64; p + 1];
    for m in 0..max_m {
        let col_idx = selected[m];
        if col_idx <= p {
            agg[col_idx] += nu * coefs_flat[m];
        }
    }

    let mut eta = match offset_vec {
        Some(vector) => vector
            .iter()
            .map(|value| value + agg[0])
            .collect::<Vec<f64>>(),
        None => vec![offset + agg[0]; n],
    };
    for j in 0..p {
        let total = agg[j + 1];
        if total == 0.0 {
            continue;
        }
        let centering = centering_means.get(j).copied().unwrap_or(0.0);
        for i in 0..n {
            eta[i] += total * (x_matrix[i * p + j] - centering);
        }
    }

    eta
}

/// Fallible [`predict_glmboost_path`]: never panics.
///
/// Returns `Err(MboostrsError::Shape)` when the path arrays disagree
/// (`selected` has more steps than `coefs_flat` values) or the design does
/// not hold `n * p` values. Removal: 0.2.0.
pub fn try_predict_glmboost_path(
    offset: f64,
    selected: &[usize],
    coefs_flat: &[f64],
    nu: f64,
    x_matrix: &[f64],
    centering_means: &[f64],
    grid: &[usize],
    n: usize,
    p: usize,
) -> Result<Vec<Vec<f64>>, MboostrsError> {
    try_predict_glmboost_path_with_offset_vec(
        offset,
        None,
        selected,
        coefs_flat,
        nu,
        x_matrix,
        centering_means,
        grid,
        n,
        p,
    )
}

/// `try_predict_glmboost_path` with a per-observation offset: every snapshot
/// of the running linear predictor starts each row at its own known
/// contribution. See [`try_predict_glmboost_with_offset_vec`].
pub fn try_predict_glmboost_path_with_offset_vec(
    offset: f64,
    offset_vec: Option<&[f64]>,
    selected: &[usize],
    coefs_flat: &[f64],
    nu: f64,
    x_matrix: &[f64],
    centering_means: &[f64],
    grid: &[usize],
    n: usize,
    p: usize,
) -> Result<Vec<Vec<f64>>, MboostrsError> {
    check_path_lengths(selected.len(), coefs_flat.len(), "coefficients")?;
    check_design(x_matrix.len(), n, p)?;
    if let Some(vector) = offset_vec {
        if vector.len() != n {
            return Err(MboostrsError::Shape(format!(
                "offset has {} entries but {n} rows are predicted",
                vector.len()
            )));
        }
    }
    Ok(predict_glmboost_path_unchecked(
        offset,
        offset_vec,
        selected,
        coefs_flat,
        nu,
        x_matrix,
        centering_means,
        grid,
        n,
        p,
    ))
}

/// Predict at multiple `mstop` values in a single incremental pass.
///
/// Panicking compatibility wrapper around [`try_predict_glmboost_path`],
/// deprecated on the P0.8 schedule. Prefer the fallible form at any boundary
/// where the path arrays can disagree. Removal: 0.2.0.
///
/// A sweep over a grid of `k` values (path plots, AIC-style diagnostics,
/// manual early-stopping inspection) that calls [`predict_glmboost`] once
/// per grid point redoes the full `O(n·p)` design-application loop at
/// every point, even though each successive `k` only adds a handful of new
/// boosting steps. This function instead walks the boosting path once,
/// updating a running `eta` by a single `O(n)` column update per step
/// (touching only the column that step actually selected, not all `p`
/// columns), and snapshots `eta` whenever `m` reaches a requested grid
/// point. Total cost is `O(n·max(grid))` plus `O(n·|grid|)` for the
/// snapshots, instead of `O(|grid|·(n·p + mstop))`.
///
/// `grid` need not be sorted or unique; the result vector matches `grid`'s
/// order and length element-for-element. Values above the path length are
/// clamped to the full path, matching [`predict_glmboost`]'s `mstop=None`
/// behavior.
#[deprecated(
    since = "0.1.3",
    note = "use try_predict_glmboost_path (returns MboostrsError instead of panicking); removed in 0.2.0"
)]
pub fn predict_glmboost_path(
    offset: f64,
    selected: &[usize],
    coefs_flat: &[f64],
    nu: f64,
    x_matrix: &[f64],
    centering_means: &[f64],
    grid: &[usize],
    n: usize,
    p: usize,
) -> Vec<Vec<f64>> {
    try_predict_glmboost_path(
        offset,
        selected,
        coefs_flat,
        nu,
        x_matrix,
        centering_means,
        grid,
        n,
        p,
    )
    .unwrap_or_else(|e| panic!("predict_glmboost_path: {e}"))
}

/// Unchecked path-prediction body behind [`try_predict_glmboost_path`].
fn predict_glmboost_path_unchecked(
    offset: f64,
    offset_vec: Option<&[f64]>,
    selected: &[usize],
    coefs_flat: &[f64],
    nu: f64,
    x_matrix: &[f64],
    centering_means: &[f64],
    grid: &[usize],
    n: usize,
    p: usize,
) -> Vec<Vec<f64>> {
    if grid.is_empty() {
        return Vec::new();
    }
    let path_len = selected.len();

    // Ascending processing order so `eta` can be advanced incrementally;
    // `order[i]` is the index into `grid`/the output whose target mstop is
    // the i-th smallest.
    let mut order: Vec<usize> = (0..grid.len()).collect();
    order.sort_by_key(|&i| grid[i]);

    let mut eta = match offset_vec {
        Some(vector) => vector.to_vec(),
        None => vec![offset; n],
    };
    let mut results: Vec<Vec<f64>> = vec![Vec::new(); grid.len()];
    let mut m = 0usize;
    for out_idx in order {
        let target = grid[out_idx].min(path_len);
        while m < target {
            let col_idx = selected[m];
            let c = nu * coefs_flat[m];
            if col_idx == 0 {
                for e in eta.iter_mut() {
                    *e += c;
                }
            } else if col_idx <= p {
                let j = col_idx - 1;
                let centering = centering_means.get(j).copied().unwrap_or(0.0);
                for i in 0..n {
                    eta[i] += c * (x_matrix[i * p + j] - centering);
                }
            }
            m += 1;
        }
        results[out_idx] = eta.clone();
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glmboost_design_centers_and_retains_intercept() {
        let x = ndarray::arr2(&[[2.0, 5.0], [4.0, 9.0]]);
        let centered = try_glmboost_design(x.view(), &[3.0, 7.0], true).unwrap();
        assert_eq!(
            centered,
            ndarray::arr2(&[[1.0, -1.0, -2.0], [1.0, 1.0, 2.0]])
        );
        assert_eq!(try_glmboost_design(x.view(), &[], false).unwrap(), x);
        assert!(try_glmboost_design(x.view(), &[3.0], true).is_err());
    }

    /// Unwrap a fallible path helper in tests whose inputs are known valid.
    fn proven<T>(result: Result<T, MboostrsError>) -> T {
        result.unwrap()
    }

    #[test]
    fn test_coef_glmboost_basic() {
        let selected = vec![0, 1, 0, 2];
        let coefs_flat = vec![0.5, 1.0, 0.3, -0.7];
        let nu = 0.1;
        let n_learners = 3;
        let centering_means = vec![2.0, 3.0];
        let offset = 1.5;

        let result = proven(try_coef_glmboost(
            &selected,
            &coefs_flat,
            nu,
            n_learners,
            &centering_means,
            false,
            offset,
            None,
        ));

        assert_eq!(result.indices, vec![0, 1, 2]);
        let intercept_raw = nu * (0.5 + 0.3);
        let beta1 = nu * 1.0;
        let beta2 = nu * (-0.7);
        let centering_correction = beta1 * 2.0 + beta2 * 3.0;
        let expected_intercept = intercept_raw - centering_correction;
        assert!((result.values[0] - expected_intercept).abs() < 1e-12);
        assert!((result.values[1] - beta1).abs() < 1e-12);
        assert!((result.values[2] - beta2).abs() < 1e-12);
    }

    #[test]
    fn test_coef_glmboost_off2int() {
        let selected = vec![0, 1, 0, 2];
        let coefs_flat = vec![0.5, 1.0, 0.3, -0.7];
        let nu = 0.1;
        let n_learners = 3;
        let centering_means = vec![2.0, 3.0];
        let offset = 1.5;

        let result = proven(try_coef_glmboost(
            &selected,
            &coefs_flat,
            nu,
            n_learners,
            &centering_means,
            true,
            offset,
            None,
        ));

        assert_eq!(result.indices, vec![0, 1, 2]);
        let intercept_raw = nu * (0.5 + 0.3);
        let beta1 = nu * 1.0;
        let beta2 = nu * (-0.7);
        let centering_correction = beta1 * 2.0 + beta2 * 3.0;
        let expected_intercept = intercept_raw - centering_correction + offset;
        assert!((result.values[0] - expected_intercept).abs() < 1e-12);
        assert!((result.values[1] - beta1).abs() < 1e-12);
        assert!((result.values[2] - beta2).abs() < 1e-12);
    }

    #[test]
    fn test_coef_glmboost_no_centering() {
        let selected = vec![0, 1];
        let coefs_flat = vec![1.0, 2.0];
        let nu = 0.1;
        let n_learners = 2;

        let result_off2int = proven(try_coef_glmboost(
            &selected,
            &coefs_flat,
            nu,
            n_learners,
            &[],
            true,
            3.0,
            None,
        ));

        let intercept = nu * 1.0 + 3.0;
        assert!((result_off2int.values[0] - intercept).abs() < 1e-12);
    }

    #[test]
    fn test_predict_glmboost_basic() {
        let offset = 1.0;
        let selected = vec![0, 1];
        let coefs_flat = vec![0.5, 2.0];
        let nu = 0.1;
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let centering_means = vec![2.0];
        let n = 2;
        let p = 2;

        let eta = proven(try_predict_glmboost(
            offset,
            &selected,
            &coefs_flat,
            nu,
            &x,
            &centering_means,
            None,
            n,
            p,
        ));

        let expected_0 = offset + nu * 0.5 + nu * 2.0 * (1.0 - 2.0);
        let expected_1 = offset + nu * 0.5 + nu * 2.0 * (3.0 - 2.0);
        assert!((eta[0] - expected_0).abs() < 1e-12);
        assert!((eta[1] - expected_1).abs() < 1e-12);
    }

    #[test]
    fn seeded_aggregate_prediction_matches_step_replay() {
        // Aggregation changes addition order relative to literal step replay;
        // this bound allows only the resulting roundoff accumulation.
        const TOLERANCE: f64 = 5e-12;
        for seed in 1_u64..=16 {
            let n = 2 + seed as usize % 31;
            let p = 1 + seed as usize % 7;
            let path_len = 1 + seed as usize % 53;
            let mstop = seed as usize % (path_len + 4);
            let nu = [0.01, 0.1, 0.5, 1.0][seed as usize % 4];
            let mut state = seed;
            let mut random = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as f64 / u64::MAX as f64
            };
            let selected: Vec<usize> = (0..path_len)
                .map(|step| {
                    if step % 7 == 0 {
                        0
                    } else {
                        1 + (random() * p as f64).floor() as usize % p
                    }
                })
                .collect();
            let coefficients: Vec<f64> = (0..path_len).map(|_| 4.0 * random() - 2.0).collect();
            let x: Vec<f64> = (0..n * p).map(|_| 2.0 * random() - 1.0).collect();
            let means: Vec<f64> = (0..p).map(|_| random() - 0.5).collect();
            let offset = random() - 0.5;
            let aggregated = proven(try_predict_glmboost(
                offset,
                &selected,
                &coefficients,
                nu,
                &x,
                &means,
                Some(mstop),
                n,
                p,
            ));
            let mut replayed = vec![offset; n];
            for step in 0..mstop.min(path_len) {
                let learner = selected[step];
                let coefficient = nu * coefficients[step];
                if learner == 0 {
                    replayed.iter_mut().for_each(|value| *value += coefficient);
                } else {
                    let column = learner - 1;
                    for row in 0..n {
                        replayed[row] += coefficient * (x[row * p + column] - means[column]);
                    }
                }
            }
            let max_difference = aggregated
                .iter()
                .zip(&replayed)
                .map(|(left, right)| (left - right).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                max_difference <= TOLERANCE,
                "prediction mismatch {max_difference:.3e}; seed={seed} \
                 spec=n={n},p={p},path_len={path_len},mstop={mstop},nu={nu}"
            );
        }
    }

    #[test]
    fn test_coef_gamboost_basic() {
        let selected = vec![0, 0, 1, 1];
        let coefs = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ];
        let nu = 0.1;
        let n_learners = 2;

        let result = proven(try_coef_gamboost(&selected, &coefs, nu, n_learners, None));
        assert_eq!(result.indices, vec![0, 1]);
        for (i, idx) in result.indices.iter().enumerate() {
            for (j, &v) in result.values[i].iter().enumerate() {
                let expected: f64 = (0..selected.len())
                    .filter(|&s| selected[s] == *idx)
                    .map(|s| nu * coefs[s][j])
                    .sum();
                assert!(
                    (v - expected).abs() < 1e-12,
                    "idx={idx} j={j} got={v} expected={expected}"
                );
            }
        }
    }

    #[test]
    fn test_coef_gamboost_with_which() {
        let selected = vec![0, 1, 2, 0, 1];
        let coefs = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
        let nu = 0.5;
        let n_learners = 3;

        let result = proven(try_coef_gamboost(
            &selected,
            &coefs,
            nu,
            n_learners,
            Some(&[1]),
        ));
        assert_eq!(result.indices, vec![1]);
        let expected = nu * (coefs[1][0] + coefs[4][0]);
        assert!((result.values[0][0] - expected).abs() < 1e-12);
    }

    #[test]
    fn test_predict_glmboost_mstop() {
        let offset = 0.0;
        let selected = vec![0, 1, 2];
        let coefs_flat = vec![1.0, 2.0, 3.0];
        let nu = 1.0;
        let x = vec![5.0, 6.0, 7.0, 8.0];
        let centering_means = vec![0.0, 0.0];
        let n = 2;
        let p = 2;

        let eta = proven(try_predict_glmboost(
            offset,
            &selected,
            &coefs_flat,
            nu,
            &x,
            &centering_means,
            Some(2),
            n,
            p,
        ));

        let expected_0 = nu * 1.0 + nu * 2.0 * 5.0;
        let expected_1 = nu * 1.0 + nu * 2.0 * 7.0;
        assert!((eta[0] - expected_0).abs() < 1e-12);
        assert!((eta[1] - expected_1).abs() < 1e-12);
    }

    #[test]
    fn test_predict_glmboost_path_matches_single_calls() {
        // A pseudo-random path over 5 columns; compare predict_glmboost_path's
        // batch output against calling predict_glmboost once per grid point.
        let n = 6usize;
        let p = 5usize;
        let mstop = 40usize;
        let nu = 0.1;
        let offset = 0.7;

        let mut rng = 42u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };

        let x: Vec<f64> = (0..n * p).map(|_| (next() % 1000) as f64 / 100.0).collect();
        let centering_means: Vec<f64> = (0..p).map(|j| j as f64 * 0.3).collect();
        let selected: Vec<usize> = (0..mstop).map(|_| (next() as usize) % (p + 1)).collect();
        let coefs_flat: Vec<f64> = (0..mstop)
            .map(|_| (next() % 200) as f64 / 100.0 - 1.0)
            .collect();

        let grid = vec![0usize, 1, 3, 7, 15, 22, 40, 100]; // 100 clamps to mstop
        let batch = proven(try_predict_glmboost_path(
            offset,
            &selected,
            &coefs_flat,
            nu,
            &x,
            &centering_means,
            &grid,
            n,
            p,
        ));
        assert_eq!(batch.len(), grid.len());

        for (i, &k) in grid.iter().enumerate() {
            let single = proven(try_predict_glmboost(
                offset,
                &selected,
                &coefs_flat,
                nu,
                &x,
                &centering_means,
                Some(k),
                n,
                p,
            ));
            for j in 0..n {
                assert!(
                    (batch[i][j] - single[j]).abs() < 1e-10,
                    "grid point k={k} (index {i}) obs {j}: batch={} single={}",
                    batch[i][j],
                    single[j]
                );
            }
        }
    }

    #[test]
    fn test_predict_glmboost_path_empty_grid() {
        // Six design values: n=3, p=2 row-major.
        let x_full = vec![0.0; 6];
        let eta = proven(try_predict_glmboost_path(
            0.0,
            &[],
            &[],
            0.1,
            &x_full,
            &[],
            &[],
            3,
            2,
        ));
        assert!(eta.is_empty());
    }

    #[test]
    fn test_predict_glmboost_path_unsorted_grid_preserves_order() {
        let offset = 0.0;
        let selected = vec![1, 2, 1];
        let coefs_flat = vec![1.0, 2.0, 3.0];
        let nu = 1.0;
        let x = vec![1.0, 2.0, 3.0, 4.0]; // n=2, p=2, row-major
        let centering_means = vec![0.0, 0.0];

        // Deliberately out of order and with a repeat.
        let grid = vec![3usize, 1, 3, 0];
        let batch = proven(try_predict_glmboost_path(
            offset,
            &selected,
            &coefs_flat,
            nu,
            &x,
            &centering_means,
            &grid,
            2,
            2,
        ));

        for (i, &k) in grid.iter().enumerate() {
            let single = proven(try_predict_glmboost(
                offset,
                &selected,
                &coefs_flat,
                nu,
                &x,
                &centering_means,
                Some(k),
                2,
                2,
            ));
            assert_eq!(batch[i], single, "mismatch at grid[{i}]=k={k}");
        }
    }
}

#[cfg(test)]
mod fallible_path_tests {
    use super::*;

    /// The arrays a boosting path is made of are two halves of one record, and
    /// every function here walks them together. A mismatch used to reach the
    /// indexing operator and panic with `index out of bounds`, naming neither
    /// array.
    #[test]
    fn coef_glmboost_rejects_a_coefficient_array_shorter_than_the_path() {
        let error = try_coef_glmboost(&[0, 1, 0], &[0.5, 0.25], 0.1, 3, &[], false, 0.0, None)
            .expect_err("a 3-step path with 2 coefficients must be rejected");
        let message = error.to_string();
        assert!(message.contains("3 selected steps"), "{message}");
        assert!(message.contains("2 coefficients"), "{message}");
    }

    /// `mstop` truncation is ordinary, so a longer coefficient array is fine.
    #[test]
    fn coef_glmboost_accepts_more_coefficients_than_steps() {
        try_coef_glmboost(&[0, 1], &[0.5, 0.25, 0.125], 0.1, 3, &[], false, 0.0, None)
            .expect("a truncated path is not a mismatch");
    }

    #[test]
    fn coef_gamboost_rejects_a_short_coefficient_list() {
        let coefs = vec![vec![1.0, 2.0]];
        let error = try_coef_gamboost(&[0, 1], &coefs, 0.1, 2, None)
            .expect_err("a 2-step path with 1 coefficient vector must be rejected");
        assert!(error.to_string().contains("coefficient vectors"));
    }

    #[test]
    fn coef_rejects_a_which_index_past_the_learner_count() {
        let error = try_coef_glmboost(&[0], &[0.5], 0.1, 2, &[], false, 0.0, Some(&[7]))
            .expect_err("learner 7 does not exist in a 2-learner model");
        let message = error.to_string();
        assert!(message.contains("learner 7"), "{message}");
        assert!(message.contains("has 2"), "{message}");
    }

    #[test]
    fn predict_rejects_a_design_that_is_not_n_by_p() {
        let error = try_predict_glmboost(0.0, &[1], &[0.5], 0.1, &[1.0; 5], &[], None, 3, 2)
            .expect_err("5 values cannot be 3 rows by 2 columns");
        let message = error.to_string();
        assert!(message.contains("5 values"), "{message}");
        assert!(message.contains("needs 6"), "{message}");
    }

    #[test]
    fn predict_accepts_a_conformable_design() {
        let eta = try_predict_glmboost(
            1.0,
            &[1],
            &[2.0],
            0.5,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[0.0, 0.0],
            None,
            3,
            2,
        )
        .expect("a 3 by 2 design with 6 values is conformable");
        assert_eq!(eta.len(), 3);
    }
}

/// Steps at which an in-bag risk path rose, and the largest relative rise.
///
/// A component-wise boosting step fits the negative gradient by least
/// squares and moves `nu` along it, which lowers the training risk only
/// while `nu` is small enough for the loss's curvature (Schulte and
/// Rügamer, *Additive Model Boosting: New Insights and Path(ologie)s*,
/// AISTATS 2025): Poisson and other strongly curved losses can overshoot
/// and oscillate at a fixed step. A rise is counted when `risk[i+1]`
/// exceeds `risk[i]` by more than `rel_tol * max(1, |risk[i]|)`; rounding
/// on a flat path stays below that. Returns `(count, largest relative
/// rise)`. Only meaningful for an in-bag path of a family whose loss does
/// not change along the path (no re-estimated nuisance parameter).
///
/// ```
/// use mboostrs_core::risk_rises;
///
/// // One rise, from 8.0 to 9.0: an eighth of the risk before it.
/// let (count, largest) = risk_rises(&[10.0, 8.0, 9.0, 7.0], 1e-12);
/// assert_eq!(count, 1);
/// assert!((largest - 0.125).abs() < 1e-12);
/// ```
pub fn risk_rises(risk: &[f64], rel_tol: f64) -> (usize, f64) {
    let mut count = 0;
    let mut largest = 0.0f64;
    for pair in risk.windows(2) {
        let (before, after) = (pair[0], pair[1]);
        if !before.is_finite() || !after.is_finite() {
            continue;
        }
        let scale = before.abs().max(1.0);
        let rise = (after - before) / scale;
        if rise > rel_tol {
            count += 1;
            largest = largest.max(rise);
        }
    }
    (count, largest)
}

/// Material increases in a family's comparable fixed-step training risk.
/// Threshold selection and nuisance-policy handling are shared by bindings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RiskRiseWarning {
    /// Number of adjacent rises exceeding rounding tolerance.
    pub count: usize,
    /// Largest adjacent rise divided by max(1, abs(previous risk)).
    pub largest_step_rise: f64,
    /// Final risk minus the path minimum, divided by its absolute value.
    /// A zero minimum uses the smallest positive normal denominator.
    pub final_above_best: f64,
    /// Iteration of the path minimum, including the offset-only state at zero.
    pub best_step: usize,
}

/// Report material step increases, or a profiled likelihood ending more than
/// 0.1% above its path minimum. Recalibration must minimize the same objective
/// for the family to opt in through [`crate::families::Family::risk_rise_threshold`].
pub fn risk_rise_warning(
    family: &dyn crate::families::Family,
    risk: &[f64],
) -> Option<RiskRiseWarning> {
    let threshold = family.risk_rise_threshold()?;
    let final_risk = *risk.last()?;
    if !final_risk.is_finite() {
        return None;
    }
    let (best_step, &best_risk) = risk
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .min_by(|a, b| a.1.total_cmp(b.1))?;
    let final_gap = (final_risk - best_risk).max(0.0);
    let final_above_best = final_gap / best_risk.abs().max(f64::MIN_POSITIVE);
    let path_scale = risk
        .iter()
        .filter(|value| value.is_finite())
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    let (count, largest_step_rise) = risk_rises(risk, 1e-9);
    let material_step = count > 0 && largest_step_rise > threshold;
    let material_backslide =
        family.calibrates() && final_above_best > 0.001 && final_gap > 1e-9 * path_scale;
    (material_step || material_backslide).then_some(RiskRiseWarning {
        count,
        largest_step_rise,
        final_above_best,
        best_step,
    })
}

#[cfg(test)]
mod risk_rise_tests {
    use super::risk_rises;

    #[test]
    fn gamma_profiled_risk_rises_are_reported_below_one_percent() {
        let gamma = crate::families::GammaReg::new();
        assert!(super::risk_rise_warning(&gamma, &[-5818.0, -5810.0, -5775.0]).is_some());
        assert!(super::risk_rise_warning(&gamma, &[-5818.0, -5818.0 + 1e-7]).is_none());
        assert!(
            super::risk_rise_warning(&crate::families::Huber::new(None), &[1.0, 2.0]).is_none()
        );
    }

    #[test]
    fn profiled_likelihood_reports_accumulated_small_rises() {
        let families: Vec<Box<dyn crate::families::Family>> = vec![
            Box::new(crate::families::NBinomial::new()),
            Box::new(crate::families::Hurdle::new()),
            Box::new(crate::families::Weibull::new()),
            Box::new(crate::families::Loglog::new()),
            Box::new(crate::families::Lognormal::new()),
            Box::new(crate::families::PropOdds::new(3)),
        ];
        for family in families {
            let warning = super::risk_rise_warning(family.as_ref(), &[1000.0, 900.0, 900.5, 901.1])
                .expect("a profiled likelihood's accumulated backslide must be reported");
            assert_eq!(warning.count, 2);
            assert_eq!(warning.best_step, 1);
            assert!(warning.largest_step_rise < 0.01);
            assert!(warning.final_above_best > 0.001);
            let scaled = super::risk_rise_warning(family.as_ref(), &[1.0, 0.9, 0.9005, 0.9011])
                .expect("changing the scale of case weights must not hide backslide");
            assert!((scaled.final_above_best - warning.final_above_best).abs() < 1e-14);
            assert!(super::risk_rise_warning(family.as_ref(), &[1000.0, 900.0, 900.5]).is_none());
            assert!(
                super::risk_rise_warning(family.as_ref(), &[1000.0, 900.0, 901.1, 890.0]).is_none()
            );
        }
        assert!(super::risk_rise_warning(
            &crate::families::Gaussian,
            &[1000.0, 900.0, 900.5, 901.1],
        )
        .is_none());
    }

    #[test]
    fn counts_only_rises_above_rounding() {
        let risk = [10.0, 9.0, 9.0 + 1e-14, 9.5, 8.0, 8.4];
        let (count, largest) = risk_rises(&risk, 1e-9);
        assert_eq!(count, 2);
        assert!((largest - 0.5 / 9.0).abs() < 1e-12);
        assert_eq!(risk_rises(&[3.0, 2.0, 1.0], 1e-9).0, 0);
    }
}
