//! Discontinuity screening: locate regime jumps that a smooth fit absorbs.
//!
//! The scan tests for a step across each candidate boundary after accounting
//! for a smooth background along it. `reference_gap` measures how much
//! structure a smooth model leaves unexplained but does not locate that
//! structure or distinguish a jump from a steep curve. A diagonal jump can be
//! missed by marginal curves and axis-aligned step candidates.
//!
//! The statistics are selection-honest by sample splitting. Every candidate
//! is scored two ways on a random half split of the rows: `jump_fit` and
//! `t_fit` come from the fit half only and are what selection ranks on;
//! `jump`, `se` and `z` are the eval-half estimate of the same coefficient
//! and its standard error. Under the null of no jump anywhere, the winning
//! candidate's `z` follows a t distribution with `n_eval - (q + 1)` degrees
//! of freedom (asymptotically standard normal), because selection reads the
//! fit half only. `gain` is the
//! held-out mean-squared-error reduction of the fit-half step model against
//! the fit-half base model, measured on the eval half, on the same
//! per-observation scale as two out-of-fold risks.
//!
//! All fits are ordinary least squares on the small dense designs
//! `[background, indicator]`, solved by Cholesky factorization.

use crate::MboostrsError;
use ndarray::ArrayView2;

/// One evaluated candidate boundary from [`disc_scan`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiscCandidate {
    /// The threshold on the projected score: the candidate is `u > threshold`.
    pub threshold: f64,
    /// Rows on the `u > threshold` side, fit half plus eval half.
    pub n_above: usize,
    /// The fit-half jump estimate (coefficient of the step indicator).
    pub jump_fit: f64,
    /// The fit-half t-statistic of the jump; what selection ranks on.
    pub t_fit: f64,
    /// The eval-half jump estimate.
    pub jump: f64,
    /// The eval-half standard error of `jump`.
    pub se: f64,
    /// `jump / se`: t-distributed with `n_eval - (q + 1)` degrees of freedom
    /// under the null of no jump (asymptotically standard normal).
    pub z: f64,
    /// Held-out MSE gain on the eval half: the fit-half base model's error
    /// minus the fit-half step model's error, per eval observation.
    pub gain: f64,
}

/// Cholesky factor (lower triangular, row-major) of a small SPD matrix.
///
/// Returns `None` when the matrix is not positive definite, the analogue of
/// the reference implementation's `LinAlgError` skip.
fn cholesky(mat: &[f64], dim: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; dim * dim];
    for i in 0..dim {
        for j in 0..=i {
            let mut sum = mat[i * dim + j];
            for k in 0..j {
                sum -= l[i * dim + k] * l[j * dim + k];
            }
            if i == j {
                if sum <= 0.0 || !sum.is_finite() {
                    return None;
                }
                l[i * dim + i] = sum.sqrt();
            } else {
                l[i * dim + j] = sum / l[j * dim + j];
            }
        }
    }
    Some(l)
}

/// Solve `L Lᵀ x = rhs` given the Cholesky factor `L`.
fn chol_solve(l: &[f64], dim: usize, rhs: &[f64]) -> Option<Vec<f64>> {
    let mut y = vec![0.0; dim];
    for i in 0..dim {
        let mut sum = rhs[i];
        for k in 0..i {
            sum -= l[i * dim + k] * y[k];
        }
        y[i] = sum / l[i * dim + i];
    }
    let mut x = vec![0.0; dim];
    for i in (0..dim).rev() {
        let mut sum = y[i];
        for k in i + 1..dim {
            sum -= l[k * dim + i] * x[k];
        }
        x[i] = sum / l[i * dim + i];
        if !x[i].is_finite() {
            return None;
        }
    }
    Some(x)
}

/// The last diagonal element of the inverse: `(L Lᵀ)⁻¹[dim-1, dim-1]`,
/// the scale factor of the last coefficient's standard error.
fn chol_inv_last_diag(l: &[f64], dim: usize) -> Option<f64> {
    // Solving L L^T x = e_{dim-1} gives the last column of the inverse;
    // its last entry is the diagonal element asked for.
    let mut e = vec![0.0; dim];
    e[dim - 1] = 1.0;
    Some(chol_solve(l, dim, &e)?[dim - 1])
}

/// Evaluate every `u > c` step candidate against a smooth background.
///
/// `u` is the projected score (n), `background` the smooth part of the
/// design (n × q), `target` the response or response-scale residual (n),
/// `fit` the 0/1 fit-half membership (n), `cuts` the candidate thresholds
/// on `u`, and `min_side` the fewest rows a side of either half may hold.
/// For each cut the step model `[background, 1{u > c}]` is fitted by OLS on
/// the fit half (giving `jump_fit` and `t_fit`) and evaluated on the eval
/// half (giving `jump`, `se`, `z`), plus the held-out `gain` of the step
/// model over the fit-half base model. Candidates that leave fewer than
/// `min_side` rows on a side of either half, or whose small system is not
/// positive definite, are skipped; a base system that fails on the fit half
/// returns no candidates at all.
///
/// The prefix sums are built along the sort order of `u`, so every threshold
/// costs a fixed small amount after one O(n log n) sort and O(n · (q + 3))
/// accumulation. The scan is sequential and deterministic.
pub fn disc_scan(
    u: &[f64],
    background: ArrayView2<f64>,
    target: &[f64],
    fit: &[u8],
    cuts: &[f64],
    min_side: usize,
) -> Result<Vec<DiscCandidate>, MboostrsError> {
    let n = u.len();
    if background.nrows() != n || background.ncols() == 0 {
        return Err(MboostrsError::Shape(format!(
            "background is {} x {} but the projection has {n} rows",
            background.nrows(),
            background.ncols()
        )));
    }
    if target.len() != n || fit.len() != n {
        return Err(MboostrsError::Shape(format!(
            "target has {} and fit has {} entries; the projection has {n}",
            target.len(),
            fit.len()
        )));
    }
    let q = background.ncols();
    let streams = q + 3; // count, target, target^2, background columns

    // Stable sort by u: ties keep row order, as numpy's stable argsort.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| u[a].total_cmp(&u[b]).then_with(|| a.cmp(&b)));
    let us: Vec<f64> = order.iter().map(|&i| u[i]).collect();

    // Prefix sums along the sorted order, masked by half membership. Stream
    // 0 counts rows, stream 1 accumulates the target, stream 2 the squared
    // target, streams 3.. the background columns.
    let mut cum_fit = vec![0.0f64; streams * n];
    let mut cum_eval = vec![0.0f64; streams * n];
    let mut total_fit = vec![0.0f64; streams];
    let mut total_eval = vec![0.0f64; streams];
    let mut n_fit = 0usize;
    for (i, &row) in order.iter().enumerate() {
        let t = target[row];
        let mut row_streams = vec![0.0f64; streams];
        row_streams[0] = 1.0;
        row_streams[1] = t;
        row_streams[2] = t * t;
        for j in 0..q {
            row_streams[3 + j] = background[(row, j)];
        }
        if fit[row] != 0 {
            n_fit += 1;
        }
        // A prefix sum must exist at every sorted position, not only where
        // the row belongs to this half: a fit-row position still carries the
        // eval-half running total forward, as numpy's masked cumsum does.
        for s in 0..streams {
            let previous_fit = if i > 0 { cum_fit[s * n + i - 1] } else { 0.0 };
            let previous_eval = if i > 0 { cum_eval[s * n + i - 1] } else { 0.0 };
            if fit[row] != 0 {
                total_fit[s] += row_streams[s];
                cum_fit[s * n + i] = total_fit[s];
                cum_eval[s * n + i] = previous_eval;
            } else {
                total_eval[s] += row_streams[s];
                cum_eval[s * n + i] = total_eval[s];
                cum_fit[s * n + i] = previous_fit;
            }
        }
    }
    let n_eval = n - n_fit;

    // Base-model Gram and rhs on each half, in original row order.
    let mut gram_fit = vec![0.0f64; q * q];
    let mut gram_eval = vec![0.0f64; q * q];
    let mut rhs_fit = vec![0.0f64; q];
    let mut rhs_eval = vec![0.0f64; q];
    for row in 0..n {
        let (gram, rhs) = if fit[row] != 0 {
            (&mut gram_fit, &mut rhs_fit)
        } else {
            (&mut gram_eval, &mut rhs_eval)
        };
        let t = target[row];
        for a in 0..q {
            let va = background[(row, a)];
            rhs[a] += va * t;
            for b in 0..q {
                gram[a * q + b] += va * background[(row, b)];
            }
        }
    }
    let l_base = match cholesky(&gram_fit, q) {
        Some(l) => l,
        // A singular fit-half background (e.g. duplicated columns) fails
        // this unit gracefully with no candidates, exactly as the NumPy
        // reference does (`return []`); it must not kill the whole screen.
        None => return Ok(Vec::new()),
    };
    let gamma = chol_solve(&l_base, q, &rhs_fit)
        .ok_or_else(|| MboostrsError::Solver("the smooth background solve failed".into()))?;
    // Held-out MSE of the fit-half base model on the eval half, without the
    // step: t't - 2 gamma' A't + gamma' A'A gamma, per eval row.
    let mut mse_base_eval = total_eval[2];
    for a in 0..q {
        mse_base_eval -= 2.0 * gamma[a] * rhs_eval[a];
        for b in 0..q {
            mse_base_eval += gamma[a] * gram_eval[a * q + b] * gamma[b];
        }
    }
    mse_base_eval /= n_eval as f64;

    let mut out = Vec::with_capacity(cuts.len());
    for &cut in cuts {
        // Rows with u <= cut are the sorted prefix of length k.
        let k = us.partition_point(|&v| v <= cut);
        if k >= n {
            continue; // the u > cut side is empty
        }
        let mut suffix_fit = vec![0.0f64; streams];
        let mut suffix_eval = vec![0.0f64; streams];
        for s in 0..streams {
            let base_fit = if k > 0 { cum_fit[s * n + k - 1] } else { 0.0 };
            let base_eval = if k > 0 { cum_eval[s * n + k - 1] } else { 0.0 };
            suffix_fit[s] = total_fit[s] - base_fit;
            suffix_eval[s] = total_eval[s] - base_eval;
        }
        // min_side on both sides of both halves; the counts are stream 0.
        let n_g_fit = suffix_fit[0] as usize;
        let n_g_eval = suffix_eval[0] as usize;
        if n_g_fit < min_side
            || (n_fit - n_g_fit) < min_side
            || n_g_eval < min_side
            || (n_eval - n_g_eval) < min_side
        {
            continue;
        }
        let suffix = |half: &[f64], s: usize| half[s];

        // Gram of [background, step] and rhs on the fit half.
        let dim = q + 1;
        let mut gram = vec![0.0f64; dim * dim];
        let mut rhs = vec![0.0f64; dim];
        for a in 0..q {
            rhs[a] = rhs_fit[a];
            for b in 0..q {
                gram[a * dim + b] = gram_fit[a * q + b];
            }
            gram[a * dim + q] = suffix(&suffix_fit, 3 + a);
            gram[q * dim + a] = suffix(&suffix_fit, 3 + a);
        }
        gram[q * dim + q] = suffix(&suffix_fit, 0);
        rhs[q] = suffix(&suffix_fit, 1);
        let Some(l_fit) = cholesky(&gram, dim) else {
            continue;
        };
        let Some(theta_fit) = chol_solve(&l_fit, dim, &rhs) else {
            continue;
        };
        let rss_fit = total_fit[2] - theta_fit.iter().zip(&rhs).map(|(t, r)| t * r).sum::<f64>();
        // Mirror the reference: skip when the fit half has no more rows
        // than coefficients (dofF <= 0), not only when it is exactly zero.
        if n_fit <= dim || rss_fit <= 0.0 {
            continue;
        }
        let dof_fit = n_fit - dim;
        let Some(var_fit) = chol_inv_last_diag(&l_fit, dim) else {
            continue;
        };
        let var_fit = (rss_fit / dof_fit as f64) * var_fit;
        if var_fit <= 0.0 {
            continue;
        }
        let jump_fit = theta_fit[q];
        let t_fit = jump_fit / var_fit.sqrt();

        // The same system on the eval half: the honest estimate and SE.
        let mut gram_e = vec![0.0f64; dim * dim];
        let mut rhs_e = vec![0.0f64; dim];
        for a in 0..q {
            rhs_e[a] = rhs_eval[a];
            for b in 0..q {
                gram_e[a * dim + b] = gram_eval[a * q + b];
            }
            gram_e[a * dim + q] = suffix(&suffix_eval, 3 + a);
            gram_e[q * dim + a] = suffix(&suffix_eval, 3 + a);
        }
        gram_e[q * dim + q] = suffix(&suffix_eval, 0);
        rhs_e[q] = suffix(&suffix_eval, 1);
        let Some(l_eval) = cholesky(&gram_e, dim) else {
            continue;
        };
        let Some(theta_eval) = chol_solve(&l_eval, dim, &rhs_e) else {
            continue;
        };
        let rss_eval = total_eval[2]
            - theta_eval
                .iter()
                .zip(&rhs_e)
                .map(|(t, r)| t * r)
                .sum::<f64>();
        if n_eval <= dim || rss_eval <= 0.0 {
            continue;
        }
        let dof_eval = n_eval - dim;
        let Some(var_eval) = chol_inv_last_diag(&l_eval, dim) else {
            continue;
        };
        let var_eval = (rss_eval / dof_eval as f64) * var_eval;
        if var_eval <= 0.0 {
            continue;
        }
        let jump = theta_eval[q];
        let se = var_eval.sqrt();
        let z = jump / se;

        // Held-out gain: the fit-half step model against the fit-half base
        // model, both scored on the eval half.
        let mut mse_full_eval = total_eval[2];
        for a in 0..dim {
            mse_full_eval -= 2.0 * theta_fit[a] * rhs_e[a];
            for b in 0..dim {
                mse_full_eval += theta_fit[a] * gram_e[a * dim + b] * theta_fit[b];
            }
        }
        mse_full_eval /= n_eval as f64;
        let gain = mse_base_eval - mse_full_eval;

        out.push(DiscCandidate {
            threshold: cut,
            n_above: (suffix(&suffix_fit, 0) + suffix(&suffix_eval, 0)) as usize,
            jump_fit,
            t_fit,
            jump,
            se,
            z,
            gain,
        });
    }
    Ok(out)
}

/// The typing statistic: is the flagged step a jump, or a smooth crossing?
///
/// Fits `[background, indicator]` on the eval-half rows and returns the
/// t-statistic of the indicator coefficient. The caller passes a background
/// augmented with the smooth terms along the projection (a quadratic and
/// cubic in it), so a smooth crossing's along-projection marginal lies in
/// the background's span and the step coefficient vanishes, while a jump's
/// edge does not. Selection ranked candidates on fit-half statistics only,
/// so this eval-half t-statistic inherits the same selection-honest null
/// calibration as `z` (t-distributed, asymptotically normal).
///
/// Returns `None` when the small system is not positive definite, the
/// eval-half residual variance is zero, or there are fewer rows than
/// columns — the reference implementation's `NaN` case.
pub fn disc_step_z(
    background: ArrayView2<f64>,
    indicator: &[f64],
    target: &[f64],
    eval: &[u8],
) -> Result<Option<f64>, MboostrsError> {
    let n = eval.len();
    if background.nrows() != n || background.ncols() == 0 {
        return Err(MboostrsError::Shape(format!(
            "background is {} x {} but the eval mask has {n} rows",
            background.nrows(),
            background.ncols()
        )));
    }
    if indicator.len() != n || target.len() != n {
        return Err(MboostrsError::Shape(format!(
            "indicator has {} and target has {} entries; the eval mask has {n}",
            indicator.len(),
            target.len()
        )));
    }
    let q = background.ncols();
    let dim = q + 1;
    let rows: Vec<usize> = (0..n).filter(|&i| eval[i] != 0).collect();
    let n_eval = rows.len();
    if n_eval <= dim {
        return Ok(None);
    }
    let mut gram = vec![0.0f64; dim * dim];
    let mut rhs = vec![0.0f64; dim];
    let mut tt = 0.0f64;
    for &i in &rows {
        let g = indicator[i];
        let t = target[i];
        tt += t * t;
        for a in 0..q {
            let va = background[(i, a)];
            rhs[a] += va * t;
            for b in 0..q {
                gram[a * dim + b] += va * background[(i, b)];
            }
            gram[a * dim + q] += va * g;
            gram[q * dim + a] += va * g;
        }
        gram[q * dim + q] += g * g;
        rhs[q] += g * t;
    }
    let Some(l) = cholesky(&gram, dim) else {
        return Ok(None);
    };
    let Some(theta) = chol_solve(&l, dim, &rhs) else {
        return Ok(None);
    };
    let rss = tt - theta.iter().zip(&rhs).map(|(t, r)| t * r).sum::<f64>();
    let dof = n_eval - dim;
    if dof == 0 || rss <= 0.0 {
        return Ok(None);
    }
    let Some(var) = chol_inv_last_diag(&l, dim) else {
        return Ok(None);
    };
    let var = (rss / dof as f64) * var;
    if var <= 0.0 {
        return Ok(None);
    }
    Ok(Some(theta[q] / var.sqrt()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn scan_axis(u: &[f64], t: &[f64], fit: &[u8], cuts: &[f64]) -> Vec<DiscCandidate> {
        let n = u.len();
        let background = Array2::from_shape_fn((n, 3), |(i, j)| match j {
            0 => 1.0,
            1 => u[i],
            _ => u[i] * u[i],
        });
        disc_scan(u, background.view(), t, fit, cuts, 10).unwrap()
    }

    /// A cheap deterministic pseudo-noise in [-noise, noise]: one draw per
    /// row from a small xorshift generator, so tests never fit an exact
    /// zero-residual model (which the scan skips by design).
    fn jitter(i: u64, noise: f64) -> f64 {
        let mut z = i
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        z ^= z << 13;
        z ^= z >> 7;
        z ^= z << 17;
        let unit = (z >> 11) as f64 / (1u64 << 53) as f64;
        (2.0 * unit - 1.0) * noise
    }

    #[test]
    fn cholesky_solve_matches_inverse_on_small_spd() {
        // A known SPD matrix with an exact inverse: det = 18 and the adjugate
        // gives A^-1 = [[5,-2,1],[-2,8,-4],[1,-4,11]] / 18.
        let mat = [4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
        let l = cholesky(&mat, 3).unwrap();
        let x = chol_solve(&l, 3, &[1.0, 0.0, 0.0]).unwrap();
        assert!((x[0] - 5.0 / 18.0).abs() < 1e-12, "{x:?}");
        assert!((x[1] + 2.0 / 18.0).abs() < 1e-12, "{x:?}");
        assert!((x[2] - 1.0 / 18.0).abs() < 1e-12, "{x:?}");
        let last = chol_inv_last_diag(&l, 3).unwrap();
        assert!((last - 11.0 / 18.0).abs() < 1e-12, "{last}");
    }

    #[test]
    fn scan_finds_a_known_axis_jump() {
        let n = 400;
        let mut u = Vec::new();
        let mut t = Vec::new();
        for i in 0..n {
            let v = -1.0 + 2.0 * (i as f64) / (n as f64);
            u.push(v);
            // Jump of size 2 at u = 0 over a noisy background-free response.
            t.push((if v > 0.0 { 2.0 } else { 0.0 }) + jitter(i as u64, 0.3));
        }
        let fit: Vec<u8> = (0..n).map(|i| (i % 2 == 0) as u8).collect();
        let cuts: Vec<f64> = (-9..=9).map(|k| k as f64 / 10.0).collect();
        let rows = scan_axis(&u, &t, &fit, &cuts);
        let best = rows
            .iter()
            .max_by(|a, b| a.t_fit.abs().total_cmp(&b.t_fit.abs()))
            .expect("at least one cut evaluates");
        assert!(best.threshold.abs() <= 0.1, "{}", best.threshold);
        assert!((best.jump_fit - 2.0).abs() < 0.3, "{}", best.jump_fit);
        assert!(best.t_fit > 10.0, "{}", best.t_fit);
        assert!((best.jump - 2.0).abs() < 0.3, "{}", best.jump);
        assert!(best.z > 5.0, "{}", best.z);
        // The gain is the variance of the jump component beyond the
        // smooth background: for a size-2 jump it is about 0.25 here.
        assert!(best.gain > 0.1, "{}", best.gain);
    }

    #[test]
    fn scan_respects_min_side_and_skips_empty_above_side() {
        let n = 200;
        let u: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let t: Vec<f64> = (0..n)
            .map(|i| (if u[i] > 0.5 { 3.0 } else { 0.0 }) + jitter(i as u64, 0.05))
            .collect();
        let fit: Vec<u8> = (0..n).map(|i| (i % 3 != 0) as u8).collect();
        // Cuts at 0.99999 and 1.0 sit at or after the maximum (0.995), so
        // the u > cut side is empty and both are skipped.
        let cuts: Vec<f64> = vec![0.5, 0.99999, 1.0];
        let rows = scan_axis(&u, &t, &fit, &cuts);
        assert_eq!(rows.len(), 1);
        assert!((rows[0].threshold - 0.5).abs() < 1e-12);
    }

    #[test]
    fn scan_honours_min_side_on_both_halves() {
        // With min_side larger than either side of the split at every cut,
        // no candidate survives.
        let n = 200;
        let u: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let t: Vec<f64> = (0..n).map(|i| jitter(i as u64, 1.0)).collect();
        let fit: Vec<u8> = (0..n).map(|i| (i % 2 == 0) as u8).collect();
        let cuts: Vec<f64> = vec![0.5];
        let rows = disc_scan(
            &u,
            Array2::from_shape_fn((n, 1), |(i, _)| u[i]).view(),
            &t,
            &fit,
            &cuts,
            10_000,
        )
        .unwrap();
        assert!(rows.is_empty());
    }

    #[test]
    fn scan_noise_keeps_the_winning_z_in_its_null_band() {
        // With no jump anywhere, the winning candidate's eval-half z is a
        // draw from a t distribution with n_eval - (q+1) degrees of freedom
        // (near-normal at this n); this fixed construction is deterministic,
        // so the assertion pins one draw in a wide band.
        let n = 600;
        let u: Vec<f64> = (0..n).map(|i| -1.0 + 2.0 * (i as f64) / n as f64).collect();
        let t: Vec<f64> = (0..n).map(|i| jitter(i as u64, 1.0)).collect();
        let fit: Vec<u8> = (0..n).map(|i| (i % 5 != 0) as u8).collect();
        let cuts: Vec<f64> = (-9..=9).map(|k| k as f64 / 10.0).collect();
        let rows = scan_axis(&u, &t, &fit, &cuts);
        let best = rows
            .iter()
            .max_by(|a, b| a.t_fit.abs().total_cmp(&b.t_fit.abs()))
            .unwrap();
        assert!(best.z.abs() < 4.5, "{}", best.z);
    }

    #[test]
    fn step_z_separates_jump_from_smooth_crossing() {
        let n = 400;
        let u: Vec<f64> = (0..n).map(|i| -2.0 + 4.0 * (i as f64) / n as f64).collect();
        let eval: Vec<u8> = (0..n).map(|i| (i % 2 == 1) as u8).collect();
        // Background [1, u, u^2, u^3]: a jump at 0 is not in its span, but
        // any cubic crossing is, so the step coefficient collapses to zero.
        let bg = Array2::from_shape_fn((n, 4), |(i, j)| match j {
            0 => 1.0,
            1 => u[i],
            2 => u[i] * u[i],
            _ => u[i] * u[i] * u[i],
        });
        let jump: Vec<f64> = u.iter().map(|&v| if v > 0.0 { 1.0 } else { 0.0 }).collect();
        let t_jump: Vec<f64> = (0..n)
            .map(|i| (if u[i] > 0.0 { 2.0 } else { 0.0 }) + jitter(i as u64, 0.2))
            .collect();
        let z_jump = disc_step_z(bg.view(), &jump, &t_jump, &eval)
            .unwrap()
            .unwrap();
        assert!(z_jump > 5.0, "{z_jump}");
        // A steep but cubic crossing: in the background span up to noise, so
        // the step adds nothing and the typing statistic stays near zero.
        let t_smooth: Vec<f64> = (0..n)
            .map(|i| 3.0 * u[i] - u[i].powi(3) + jitter(i as u64 + 7, 0.2))
            .collect();
        let z_smooth = disc_step_z(bg.view(), &jump, &t_smooth, &eval)
            .unwrap()
            .unwrap();
        assert!(z_smooth.abs() < 2.0, "{z_smooth}");
    }

    #[test]
    fn singular_background_skips_the_unit_gracefully() {
        // Background with two identical columns: the fit-half Gram is
        // singular, so the unit contributes no candidates instead of
        // failing the whole screen (mirrors the NumPy reference).
        let n = 200;
        let u: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let t: Vec<f64> = vec![1.0; n];
        let mut fit = vec![0u8; n];
        for (i, f) in fit.iter_mut().enumerate() {
            *f = (i < n / 2) as u8;
        }
        let background = Array2::from_shape_fn((n, 2), |(_, _)| 1.0);
        let rows = disc_scan(&u, background.view(), &t, &fit, &[0.5], 10).unwrap();
        assert!(rows.is_empty());
        // Typing with one row: fewer rows than columns -> None.
        let bg1 = Array2::from_shape_fn((1, 4), |(i, j)| (i + j) as f64);
        assert_eq!(disc_step_z(bg1.view(), &[1.0], &[0.5], &[1]).unwrap(), None);
    }

    #[test]
    fn scan_guards_report_shape_mismatches() {
        let n = 100;
        let u = vec![0.5; n];
        let t = vec![0.0; n];
        let fit = vec![1u8; n];
        let background = Array2::zeros((n - 1, 3));
        assert!(disc_scan(&u, background.view(), &t, &fit, &[0.0], 5).is_err());
        assert!(disc_scan(
            &u,
            Array2::zeros((n, 3)).view(),
            &t[..n - 1],
            &fit,
            &[0.0],
            5
        )
        .is_err());
    }
}
