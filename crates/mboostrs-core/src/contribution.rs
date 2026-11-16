//! Shares and paired gains that report how much of a fitted prediction one
//! term carries (issue #124).
//!
//! A boosted prediction is a sum of learner contributions. These functions
//! take two vectors of contributions and report:
//!
//! * the covariance share of one part in the variance of the total, an exact
//!   decomposition (the shares of all parts sum to one);
//! * the absolute share, the mean absolute deviation of one part over the sum
//!   of both parts' mean absolute deviations;
//! * the mean and standard error of a paired difference between two vectors
//!   of held-out scores, computed on the same folds;
//! * the share of the response variance a fitted prediction leaves unexplained.

use crate::MboostrsError;

/// The two shares of one part of a prediction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContributionShares {
    /// `cov(part, total) / var(total)` with `total = part + rest`. The shares
    /// of the two parts sum to one. Negative when the part moves against the
    /// rest of the prediction.
    pub covariance_share: f64,
    /// `D(part) / (D(part) + D(rest))`, where `D(v)` is the mean absolute
    /// deviation of `v` from its mean. The part's share of the total absolute
    /// contribution; positive and negative contributions do not cancel.
    pub absolute_share: f64,
}

fn check_lengths(first: &[f64], second: &[f64], what: &str) -> Result<(), MboostrsError> {
    if first.len() != second.len() {
        return Err(MboostrsError::Shape(format!(
            "{what}: first has {} values but second has {}",
            first.len(),
            second.len()
        )));
    }
    if first.iter().chain(second).any(|v| !v.is_finite()) {
        return Err(MboostrsError::BadResponse(
            "contribution".to_string(),
            format!("{what}: values must be finite"),
        ));
    }
    Ok(())
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Population variance: the divisor is the number of rows.
fn variance(values: &[f64]) -> f64 {
    let m = mean(values);
    values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / values.len() as f64
}

/// Mean absolute deviation from the mean: `mean|v - mean(v)|`.
fn centred_abs_mean(values: &[f64]) -> f64 {
    let m = mean(values);
    values.iter().map(|v| (v - m).abs()).sum::<f64>() / values.len() as f64
}

fn covariance(a: &[f64], b: &[f64]) -> f64 {
    let (ma, mb) = (mean(a), mean(b));
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - ma) * (y - mb))
        .sum::<f64>()
        / a.len() as f64
}

/// Shares of `part` in the prediction `part + rest`.
///
/// Returns NaN shares when the total has no variance (covariance share) or
/// when both parts are identically zero (absolute share). Errors when the two
/// vectors differ in length, are empty, or contain non-finite values.
pub fn contribution_shares(
    part: &[f64],
    rest: &[f64],
) -> Result<ContributionShares, MboostrsError> {
    check_lengths(part, rest, "contribution_shares")?;
    if part.is_empty() {
        return Err(MboostrsError::Shape(
            "contribution_shares needs at least one row".to_string(),
        ));
    }
    let total: Vec<f64> = part.iter().zip(rest).map(|(a, b)| a + b).collect();
    let total_var = variance(&total);
    let covariance_share = if total_var > 0.0 {
        covariance(part, &total) / total_var
    } else {
        f64::NAN
    };
    // Each part is centred on its own mean first, so the intercept (the
    // constant offset of the linear predictor) is in neither part.
    let part_abs = centred_abs_mean(part);
    let rest_abs = centred_abs_mean(rest);
    let denominator = part_abs + rest_abs;
    let absolute_share = if denominator > 0.0 {
        part_abs / denominator
    } else {
        f64::NAN
    };
    Ok(ContributionShares {
        covariance_share,
        absolute_share,
    })
}

/// Mean and standard error of the paired differences `first - second`.
///
/// The standard error is the sample standard deviation of the differences
/// over the square root of their count. Both vectors hold paired scores on
/// the same folds or rows. Returns `(mean, standard_error)`; the standard
/// error is NaN with fewer than two pairs. Row losses must already include
/// any observation weights; no second weighting is applied. This descriptive
/// standard error does not correct for cross-fitting or selection dependence.
pub fn paired_mean_se(first: &[f64], second: &[f64]) -> Result<(f64, f64), MboostrsError> {
    check_lengths(first, second, "paired_mean_se")?;
    if first.is_empty() {
        return Err(MboostrsError::Shape(
            "paired_mean_se needs at least one fold".to_string(),
        ));
    }
    let diff: Vec<f64> = first.iter().zip(second).map(|(a, b)| a - b).collect();
    let n = diff.len();
    let m = mean(&diff);
    if n < 2 {
        return Ok((m, f64::NAN));
    }
    let sample_var = diff.iter().map(|d| (d - m).powi(2)).sum::<f64>() / (n - 1) as f64;
    Ok((m, (sample_var / n as f64).sqrt()))
}

/// Share of the response variance left unexplained by a fitted prediction:
/// `var(y - fitted) / var(y)`.
///
/// Computed on the rows it is given. On training rows this is the in-sample
/// leftover; on held-out rows it is the out-of-sample leftover. NaN when `y`
/// has no variance.
pub fn leftover_share(y: &[f64], fitted: &[f64]) -> Result<f64, MboostrsError> {
    check_lengths(y, fitted, "leftover_share")?;
    if y.is_empty() {
        return Err(MboostrsError::Shape(
            "leftover_share needs at least one row".to_string(),
        ));
    }
    let residual: Vec<f64> = y.iter().zip(fitted).map(|(a, b)| a - b).collect();
    let y_var = variance(y);
    if y_var > 0.0 {
        Ok(variance(&residual) / y_var)
    } else {
        Ok(f64::NAN)
    }
}

/// One learner's partial effect: its contribution to the additive predictor
/// at the rows of `design`.
///
/// The learner's coefficient vector is `b = nu * sum(coefs[m])` over the
/// steps `m` with `selected[m] == learner`. The effect at row `i` is
/// `(x_i - c)' b = x_i' b - c' b`, where `x_i` is row `i` of `design`
/// (row-major, `p` columns) and `c` is the column centering the fit
/// subtracted from the design (`centering` empty for none). For a grouped
/// learner (`brandom`, `bmrf`) evaluated at the identity design, row `g` is
/// group `g`'s effect `b_g - sum_j c_j b_j`.
///
/// Returns `Ok(None)` when a selected step carries a coefficient vector of
/// a length other than `p`: the fit's coefficients refer to a different
/// basis than `design` (a `center = TRUE` spline keeps only the penalized
/// directions), so no effect can be formed from them.
pub fn try_learner_effect(
    design: &[f64],
    p: usize,
    centering: &[f64],
    selected: &[usize],
    coefs: &[Vec<f64>],
    learner: usize,
    nu: f64,
) -> Result<Option<Vec<f64>>, MboostrsError> {
    if p == 0 || design.len() % p != 0 {
        return Err(MboostrsError::Shape(format!(
            "learner_effect: design has {} entries, not a multiple of p = {p}",
            design.len()
        )));
    }
    if !centering.is_empty() && centering.len() != p {
        return Err(MboostrsError::Shape(format!(
            "learner_effect: centering has {} entries for p = {p}",
            centering.len()
        )));
    }
    if coefs.len() != selected.len() {
        return Err(MboostrsError::Shape(format!(
            "learner_effect: {} coefficient vectors for {} steps",
            coefs.len(),
            selected.len()
        )));
    }
    if !nu.is_finite() {
        return Err(MboostrsError::Control(
            "learner_effect: nu must be finite".into(),
        ));
    }
    let mut b = vec![0.0; p];
    for (step, &sel) in selected.iter().enumerate() {
        if sel != learner {
            continue;
        }
        let step_coefs = &coefs[step];
        if step_coefs.len() != p {
            return Ok(None);
        }
        for (bj, &cj) in b.iter_mut().zip(step_coefs) {
            *bj += nu * cj;
        }
    }
    let shift: f64 = centering.iter().zip(&b).map(|(c, bj)| c * bj).sum();
    Ok(Some(
        design
            .chunks_exact(p)
            .map(|row| row.iter().zip(&b).map(|(x, bj)| x * bj).sum::<f64>() - shift)
            .collect(),
    ))
}

/// A grouped learner's (`brandom`, `bmrf`) contribution at rows given as
/// group indicators: [`try_learner_effect`] at the design the fit used for
/// those rows.
///
/// `indicators` is row-major with `n_groups` columns; row `i` is the
/// indicator of row `i`'s group (or a row of `1 / n_groups` for an unseen
/// level averaged over the groups). The design row is the indicator row
/// times `by[i]` (`by` empty for none), as the learner was prepared.
/// `centering` is the column centering the fit applied (empty for none;
/// nonzero for `brandom(center = TRUE)`). `spectral_penalty` is the
/// neighbourhood matrix `K` of a `bmrf(center = TRUE)` term, whose design is
/// [`crate::learners::center_spectral`] of the indicators: its coefficients
/// index the eigen-directions of `K` with a nonzero eigenvalue, so the
/// contribution of group `g` is row `g` of `Z b`, with `Z` the indicators'
/// projection onto those directions.
///
/// Returns `Ok(None)` when the coefficients do not match the design's
/// column count, as [`try_learner_effect`] does.
#[allow(clippy::too_many_arguments)]
pub fn try_grouped_learner_effect(
    indicators: &[f64],
    n_groups: usize,
    by: &[f64],
    centering: &[f64],
    spectral_penalty: Option<&ndarray::Array2<f64>>,
    selected: &[usize],
    coefs: &[Vec<f64>],
    learner: usize,
    nu: f64,
) -> Result<Option<Vec<f64>>, MboostrsError> {
    if n_groups == 0 || indicators.len() % n_groups != 0 {
        return Err(MboostrsError::Shape(format!(
            "grouped_learner_effect: {} indicator entries, not a multiple of {n_groups} groups",
            indicators.len()
        )));
    }
    let rows = indicators.len() / n_groups;
    if !by.is_empty() && by.len() != rows {
        return Err(MboostrsError::Shape(format!(
            "grouped_learner_effect: by has {} values for {rows} rows",
            by.len()
        )));
    }
    let mut design = indicators.to_vec();
    if !by.is_empty() {
        for (row, &z) in design.chunks_exact_mut(n_groups).zip(by) {
            row.iter_mut().for_each(|v| *v *= z);
        }
    }
    let Some(k) = spectral_penalty else {
        return try_learner_effect(&design, n_groups, centering, selected, coefs, learner, nu);
    };
    if k.dim() != (n_groups, n_groups) {
        return Err(MboostrsError::Shape(format!(
            "grouped_learner_effect: K is {:?} for {n_groups} groups",
            k.dim()
        )));
    }
    let design = ndarray::Array2::from_shape_vec((rows, n_groups), design)
        .map_err(|e| MboostrsError::Shape(format!("grouped_learner_effect: {e}")))?;
    let projected = crate::learners::center_spectral(&design, k, 1);
    let p = projected.ncols();
    if p == 0 {
        return Ok(Some(vec![0.0; rows]));
    }
    let flat: Vec<f64> = projected.iter().copied().collect();
    try_learner_effect(&flat, p, &[], selected, coefs, learner, nu)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learner_effect_of_a_grouped_learner_subtracts_the_centered_mean() {
        // Three groups, centering c = (0.5, 0.3, 0.2) (the group shares).
        // Two selected steps of learner 1 and one step of learner 0.
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let c = [0.5, 0.3, 0.2];
        let selected = [1, 0, 1];
        let coefs = vec![vec![1.0, 2.0, -1.0], vec![9.0], vec![1.0, 0.0, 3.0]];
        let effect = try_learner_effect(&identity, 3, &c, &selected, &coefs, 1, 0.5)
            .unwrap()
            .unwrap();
        // b = 0.5 * (2, 2, 2) = (1, 1, 1); c'b = 1; effect = b - 1 = 0.
        assert_eq!(effect, vec![0.0, 0.0, 0.0]);
        let coefs = vec![vec![2.0, 0.0, 0.0], vec![9.0], vec![0.0, 0.0, 0.0]];
        let effect = try_learner_effect(&identity, 3, &c, &selected, &coefs, 1, 0.5)
            .unwrap()
            .unwrap();
        // b = (1, 0, 0); c'b = 0.5; effect = (0.5, -0.5, -0.5). The
        // share-weighted mean of the effect is zero, as centering requires.
        assert_eq!(effect, vec![0.5, -0.5, -0.5]);
        let weighted_mean: f64 = effect.iter().zip(&c).map(|(e, w)| e * w).sum();
        assert!(weighted_mean.abs() < 1e-15);
        // The earlier R formula b * (1 - c) gives (0.5, 0, 0) here.
        assert_ne!(effect, vec![0.5, 0.0, 0.0]);
    }

    #[test]
    fn grouped_learner_effect_scales_centers_and_projects_the_indicators() {
        // Rows in groups 2, 0, 1 of three groups.
        let rows = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let c = [0.5, 0.3, 0.2];
        let coefs = vec![vec![2.0, 0.0, -2.0]];
        // b = (1, 0, -1), c'b = 0.3: row effects b_g - 0.3.
        let effect = try_grouped_learner_effect(&rows, 3, &[], &c, None, &[1], &coefs, 1, 0.5)
            .unwrap()
            .unwrap();
        for (got, want) in effect.iter().zip([-1.3, 0.7, -0.3]) {
            assert!((got - want).abs() < 1e-15);
        }
        // by scales the indicator row before the centering is subtracted.
        let effect =
            try_grouped_learner_effect(&rows, 3, &[2.0, 1.0, -1.0], &c, None, &[1], &coefs, 1, 0.5)
                .unwrap()
                .unwrap();
        for (got, want) in effect.iter().zip([-2.3, 0.7, -0.3]) {
            assert!((got - want).abs() < 1e-15);
        }
        // A chain neighbourhood: the centered design is the indicators
        // projected onto K's two directions with a nonzero eigenvalue, and
        // the coefficients have two entries.
        let k = ndarray::array![[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]];
        let design = ndarray::Array2::from_shape_vec((3, 3), rows.to_vec()).unwrap();
        let z = crate::learners::center_spectral(&design, &k, 1);
        assert_eq!(z.ncols(), 2);
        let coefs = vec![vec![1.0, -2.0], vec![0.5, 0.5]];
        let effect =
            try_grouped_learner_effect(&rows, 3, &[], &[], Some(&k), &[1, 1], &coefs, 1, 0.1)
                .unwrap()
                .unwrap();
        let b = [0.15, -0.15];
        for (i, got) in effect.iter().enumerate() {
            let want = z[[i, 0]] * b[0] + z[[i, 1]] * b[1];
            assert!((got - want).abs() < 1e-15);
        }
        // The projected effect sums to zero over the groups: the constant
        // is K's null space, which the projection removes.
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let per_group =
            try_grouped_learner_effect(&identity, 3, &[], &[], Some(&k), &[1], &coefs[..1], 1, 1.0)
                .unwrap()
                .unwrap();
        assert!(per_group.iter().sum::<f64>().abs() < 1e-12);
        // Coefficients indexing the groups do not match the projected design.
        assert!(try_grouped_learner_effect(
            &rows,
            3,
            &[],
            &[],
            Some(&k),
            &[1],
            &[vec![1.0; 3]],
            1,
            0.1
        )
        .unwrap()
        .is_none());
        assert!(try_grouped_learner_effect(&rows, 4, &[], &[], None, &[], &[], 1, 0.1).is_err());
        assert!(try_grouped_learner_effect(&rows, 3, &[1.0], &[], None, &[], &[], 1, 0.1).is_err());
    }

    #[test]
    fn learner_effect_is_the_centered_design_times_the_summed_coefficients() {
        // Two rows of a two-column design, centering (1, 2).
        let design = [1.0, 3.0, 2.0, 5.0];
        let effect = try_learner_effect(
            &design,
            2,
            &[1.0, 2.0],
            &[0, 0],
            &[vec![1.0, 1.0], vec![1.0, -3.0]],
            0,
            0.1,
        )
        .unwrap()
        .unwrap();
        // b = (0.2, -0.2): row 1 (0, 1)·b = -0.2, row 2 (1, 3)·b = -0.4.
        assert!((effect[0] + 0.2).abs() < 1e-15);
        assert!((effect[1] + 0.4).abs() < 1e-15);
        // No centering; an unselected learner has a zero effect.
        let none = try_learner_effect(&design, 2, &[], &[1], &[vec![5.0]], 0, 0.1)
            .unwrap()
            .unwrap();
        assert_eq!(none, vec![0.0, 0.0]);
        // A step whose coefficients do not match the design's columns.
        assert!(
            try_learner_effect(&design, 2, &[], &[0], &[vec![1.0]], 0, 0.1)
                .unwrap()
                .is_none()
        );
        assert!(try_learner_effect(&design, 3, &[], &[], &[], 0, 0.1).is_err());
        assert!(try_learner_effect(&design, 2, &[1.0], &[], &[], 0, 0.1).is_err());
    }

    #[test]
    fn covariance_shares_of_two_parts_sum_to_one() {
        let part: Vec<f64> = (0..50).map(|i| (i as f64 * 0.3).sin()).collect();
        let rest: Vec<f64> = (0..50)
            .map(|i| (i as f64 * 0.7).cos() + 0.1 * i as f64)
            .collect();
        let a = contribution_shares(&part, &rest).unwrap();
        let b = contribution_shares(&rest, &part).unwrap();
        assert!((a.covariance_share + b.covariance_share - 1.0).abs() < 1e-12);
        assert!(a.absolute_share > 0.0 && a.absolute_share < 1.0);
    }

    #[test]
    fn a_part_that_is_the_whole_prediction_has_share_one() {
        let part = vec![1.0, -2.0, 3.0, 0.5];
        let rest = vec![0.0; 4];
        let s = contribution_shares(&part, &rest).unwrap();
        assert!((s.covariance_share - 1.0).abs() < 1e-12);
        assert!((s.absolute_share - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_zero_part_has_zero_share() {
        let rest = vec![1.0, 2.0, 4.0];
        let s = contribution_shares(&[0.0; 3], &rest).unwrap();
        assert_eq!(s.covariance_share, 0.0);
        assert_eq!(s.absolute_share, 0.0);
    }

    #[test]
    fn constant_total_gives_nan_covariance_share() {
        let s = contribution_shares(&[1.0, -1.0], &[-1.0, 1.0]).unwrap();
        assert!(s.covariance_share.is_nan());
        assert!((s.absolute_share - 0.5).abs() < 1e-12);
    }

    #[test]
    fn paired_mean_and_standard_error_match_hand_values() {
        // Differences 1, 3, 2, 2: mean 2, sample variance 2/3, se sqrt((2/3)/4).
        let (m, se) = paired_mean_se(&[2.0, 4.0, 3.0, 5.0], &[1.0, 1.0, 1.0, 3.0]).unwrap();
        assert!((m - 2.0).abs() < 1e-12);
        assert!((se - (2.0_f64 / 3.0 / 4.0).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn one_fold_has_no_standard_error() {
        let (m, se) = paired_mean_se(&[3.0], &[1.0]).unwrap();
        assert_eq!(m, 2.0);
        assert!(se.is_nan());
    }

    #[test]
    fn leftover_share_is_one_for_a_zero_fit_and_zero_for_an_exact_fit() {
        let y = vec![1.0, 2.0, 3.0, 6.0];
        assert!((leftover_share(&y, &[0.0; 4]).unwrap() - 1.0).abs() < 1e-12);
        assert!(leftover_share(&y, &y).unwrap().abs() < 1e-12);
        assert!(leftover_share(&[2.0; 3], &[1.0; 3]).unwrap().is_nan());
    }

    #[test]
    fn mismatched_lengths_are_rejected() {
        assert!(contribution_shares(&[1.0, 2.0], &[1.0]).is_err());
        assert!(paired_mean_se(&[1.0], &[1.0, 2.0]).is_err());
        assert!(leftover_share(&[], &[]).is_err());
        assert!(paired_mean_se(&[f64::NAN], &[1.0]).is_err());
    }
}

/// Prefer a simpler specification within `multiple` paired fold standard errors.
/// Returns (prefer_simple, mean_simple_minus_rich, paired_standard_error).
/// Every paired score must be finite. Differences are scaled before computing
/// dispersion, avoiding squared-loss overflow without introducing a tie tolerance.
pub fn prefer_simpler_cv(
    simple: &[f64],
    rich: &[f64],
    multiple: f64,
) -> Result<(bool, f64, f64), MboostrsError> {
    check_lengths(simple, rich, "prefer_simpler_cv")?;
    if simple.len() < 2
        || !multiple.is_finite()
        || multiple < 0.0
        || simple.iter().chain(rich).any(|x| !x.is_finite())
    {
        return Err(MboostrsError::Control("paired complexity choice needs at least two finite scores and a nonnegative finite SE multiplier".into()));
    }
    let differences: Vec<f64> = simple.iter().zip(rich).map(|(a, b)| a - b).collect();
    if differences.iter().any(|x| !x.is_finite()) {
        return Err(MboostrsError::Control(
            "paired complexity differences overflowed; rescale the loss units".into(),
        ));
    }
    let scale = differences.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return Ok((true, 0.0, 0.0));
    }
    let normalized: Vec<f64> = differences.iter().map(|x| x / scale).collect();
    let (difference, se) = paired_mean_se(&normalized, &vec![0.0; simple.len()])?;
    Ok((difference <= multiple * se, difference * scale, se * scale))
}

#[cfg(test)]
mod complexity_choice_tests {
    use super::*;
    #[test]
    fn preference_and_scale_equivariance() {
        let simple = [1.0, 2.0, 1.5, 2.2, 1.9];
        let rich = [0.9, 2.2, 1.3, 2.3, 1.8];
        let (chosen, difference, se) = prefer_simpler_cv(&simple, &rich, 2.0).unwrap();
        let (reference, se_reference) = paired_mean_se(&simple, &rich).unwrap();
        assert!((difference - reference).abs() < 1e-15);
        assert!((se - se_reference).abs() < 1e-15);
        assert_eq!(chosen, reference <= 2.0 * se_reference);
        for scale in [1e-200, 1e200] {
            let a: Vec<_> = simple.iter().map(|x| x * scale).collect();
            let b: Vec<_> = rich.iter().map(|x| x * scale).collect();
            let (decision, d, s) = prefer_simpler_cv(&a, &b, 2.0).unwrap();
            assert_eq!(decision, chosen);
            assert!((d / scale - difference).abs() < 1e-14);
            assert!((s / scale - se).abs() < 1e-14);
        }
    }
    #[test]
    fn exact_ties_and_adjacent_distinct_scores() {
        let rich = [1.0; 5];
        assert!(prefer_simpler_cv(&rich, &rich, 2.0).unwrap().0);
        let adjacent = [f64::from_bits(1.0_f64.to_bits() + 1); 5];
        assert!(!prefer_simpler_cv(&adjacent, &rich, 2.0).unwrap().0);
        assert!(prefer_simpler_cv(&rich, &adjacent, 2.0).unwrap().0);
    }
    #[test]
    fn refuses_missing_failed_or_unpaired_folds() {
        assert!(prefer_simpler_cv(&[1.0], &[1.0], 2.0).is_err());
        assert!(prefer_simpler_cv(&[1.0, 2.0], &[1.0], 2.0).is_err());
        for value in [f64::NAN, f64::INFINITY] {
            assert!(prefer_simpler_cv(&[1.0, value], &[1.0, 2.0], 2.0).is_err());
        }
        assert!(prefer_simpler_cv(&[1.0, 2.0], &[1.0, 2.0], -1.0).is_err());
    }
}
