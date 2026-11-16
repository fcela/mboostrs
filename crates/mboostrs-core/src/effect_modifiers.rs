//! Tests and linear projections for effect modification, on AIPW scores.
//!
//! An augmented inverse-probability score `psi_i` has conditional mean
//! `tau(x_i)` given the covariates, so a regression of the scores on a
//! covariate estimates how the effect changes along that covariate. This
//! module provides the three numerical pieces of that analysis:
//!
//! * [`robust_wls`]: weighted least squares of the scores on a design matrix,
//!   with the heteroskedasticity-robust (HC1 sandwich) covariance of the
//!   coefficients. The slope on a standardized covariate is the best linear
//!   predictor of the effect's change per standard deviation of the covariate.
//! * [`wald_test`]: the joint chi-squared test that a block of coefficients
//!   is zero, used for categorical covariates (one block of dummy columns).
//! * [`gates`]: group means of the scores, the contrast between the top and
//!   bottom group, and the chi-squared test that the group means are equal.
//!
//! [`holm_adjust`] applies the Holm step-down correction across covariates,
//! and [`linear_predict`] evaluates a fitted projection on new rows.

use ndarray::{Array1, Array2};

use crate::causal::chi_square_sf;
use crate::linalg::cho_solve;
use crate::MboostrsError;

/// Two-sided p-value of a standard normal statistic.
fn normal_two_sided(z: f64) -> f64 {
    libm::erfc(z.abs() / std::f64::consts::SQRT_2)
}

/// Weighted least squares with a robust covariance.
#[derive(Debug, Clone, PartialEq)]
pub struct RobustFit {
    /// Coefficients, one per design column.
    pub coefficients: Vec<f64>,
    /// HC1 standard errors, the square roots of the diagonal of `covariance`.
    pub se: Vec<f64>,
    /// HC1 covariance matrix, row-major `(p, p)`.
    pub covariance: Vec<f64>,
    /// Rows with positive weight.
    pub n: usize,
    /// Design columns.
    pub p: usize,
}

/// Weighted least squares of `y` on the row-major `(n, p)` design `x`, with the
/// HC1 sandwich covariance `n/(n-p) B M B`, where `B = (X'WX)^-1` and
/// `M = sum_i w_i^2 e_i^2 x_i x_i'`. Rows with zero weight do not enter the fit.
///
/// The design must include an intercept column if one is wanted. The fit needs
/// more positive-weight rows than columns.
pub fn robust_wls(
    y: &[f64],
    x: &[f64],
    p: usize,
    weights: &[f64],
) -> Result<RobustFit, MboostrsError> {
    let n = y.len();
    if p == 0 {
        return Err(MboostrsError::Shape(
            "the design needs at least one column".into(),
        ));
    }
    if x.len() != n * p || weights.len() != n {
        return Err(MboostrsError::Shape(format!(
            "design has {} values for {n} rows and {p} columns; weights has {}",
            x.len(),
            weights.len()
        )));
    }
    if y.iter().chain(x).any(|v| !v.is_finite()) {
        return Err(MboostrsError::BadResponse(
            "robust_wls".into(),
            "response and design must be finite".into(),
        ));
    }
    if weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
        return Err(MboostrsError::BadResponse(
            "robust_wls".into(),
            "weights must be finite and non-negative".into(),
        ));
    }
    let positive = weights.iter().filter(|w| **w > 0.0).count();
    if positive <= p {
        return Err(MboostrsError::Shape(format!(
            "robust_wls needs more than {p} rows with positive weight; it has {positive}"
        )));
    }
    let mut xtwx = Array2::<f64>::zeros((p, p));
    let mut xtwy = Array1::<f64>::zeros(p);
    for i in 0..n {
        let w = weights[i];
        if w == 0.0 {
            continue;
        }
        let row = &x[i * p..(i + 1) * p];
        for a in 0..p {
            xtwy[a] += w * row[a] * y[i];
            for b in 0..p {
                xtwx[[a, b]] += w * row[a] * row[b];
            }
        }
    }
    let mut bread = Array2::<f64>::zeros((p, p));
    let mut unit = Array1::<f64>::zeros(p);
    for col in 0..p {
        unit.fill(0.0);
        unit[col] = 1.0;
        let column = cho_solve(&xtwx, &unit.view())?;
        bread.column_mut(col).assign(&column);
    }
    let beta = bread.dot(&xtwy);
    let mut meat = Array2::<f64>::zeros((p, p));
    for i in 0..n {
        let w = weights[i];
        if w == 0.0 {
            continue;
        }
        let row = &x[i * p..(i + 1) * p];
        let fitted: f64 = row.iter().zip(beta.iter()).map(|(a, b)| a * b).sum();
        let resid = y[i] - fitted;
        let scale = w * w * resid * resid;
        for a in 0..p {
            for b in 0..p {
                meat[[a, b]] += scale * row[a] * row[b];
            }
        }
    }
    let hc1 = positive as f64 / (positive - p) as f64;
    let cov = bread.dot(&meat).dot(&bread) * hc1;
    let se: Vec<f64> = (0..p).map(|a| cov[[a, a]].max(0.0).sqrt()).collect();
    let mut covariance = Vec::with_capacity(p * p);
    for a in 0..p {
        for b in 0..p {
            covariance.push(cov[[a, b]]);
        }
    }
    Ok(RobustFit {
        coefficients: beta.to_vec(),
        se,
        covariance,
        n: positive,
        p,
    })
}

/// Joint Wald test that the coefficients at `indices` are all zero.
///
/// Returns `(chi_squared, p_value)` with `indices.len()` degrees of freedom,
/// using the covariance from [`robust_wls`].
pub fn wald_test(
    coefficients: &[f64],
    covariance: &[f64],
    p: usize,
    indices: &[usize],
) -> Result<(f64, f64), MboostrsError> {
    let k = indices.len();
    if k == 0 {
        return Err(MboostrsError::Shape(
            "wald_test needs at least one coefficient index".into(),
        ));
    }
    if coefficients.len() != p || covariance.len() != p * p {
        return Err(MboostrsError::Shape(
            "coefficients and covariance do not match the column count".into(),
        ));
    }
    if indices.iter().any(|&i| i >= p) {
        return Err(MboostrsError::Shape(
            "wald_test index is outside the design columns".into(),
        ));
    }
    let mut sub = Array2::<f64>::zeros((k, k));
    let mut beta = Array1::<f64>::zeros(k);
    for (a, &ia) in indices.iter().enumerate() {
        beta[a] = coefficients[ia];
        for (b, &ib) in indices.iter().enumerate() {
            sub[[a, b]] = covariance[ia * p + ib];
        }
    }
    let solved = cho_solve(&sub, &beta.view())?;
    let chi2: f64 = beta.iter().zip(solved.iter()).map(|(b, s)| b * s).sum();
    Ok((chi2, chi_square_sf(chi2, k as f64)))
}

/// Linear predictor `x beta` for a row-major `(n, p)` design.
pub fn linear_predict(x: &[f64], p: usize, beta: &[f64]) -> Result<Vec<f64>, MboostrsError> {
    if p == 0 || beta.len() != p || x.len() % p != 0 {
        return Err(MboostrsError::Shape(
            "design rows and coefficients have different widths".into(),
        ));
    }
    Ok(x.chunks_exact(p)
        .map(|row| row.iter().zip(beta).map(|(a, b)| a * b).sum())
        .collect())
}

/// Group means of the scores and their contrasts, with uncertainty.
#[derive(Debug, Clone, PartialEq)]
pub struct Gates {
    /// Weighted mean of the scores in each group; NaN for an empty group.
    pub means: Vec<f64>,
    /// Standard error of each group mean; NaN for a group with fewer than two rows.
    pub se: Vec<f64>,
    /// Rows in each group.
    pub counts: Vec<usize>,
    /// Last nonempty group minus first nonempty group (the top-minus-bottom
    /// contrast when the groups are ordered by the covariate).
    pub ordered_contrast: f64,
    /// Standard error of `ordered_contrast`.
    pub ordered_se: f64,
    /// Two-sided normal p-value of `ordered_contrast`.
    pub ordered_p: f64,
    /// Index of the first nonempty group, the bottom of `ordered_contrast`.
    pub ordered_bottom: usize,
    /// Index of the last nonempty group, the top of `ordered_contrast`.
    pub ordered_top: usize,
    /// Highest group mean minus lowest group mean. The groups are chosen by
    /// their means, so `extreme_p` is optimistic; use `omnibus_p` as the test.
    pub extreme_contrast: f64,
    /// Standard error of `extreme_contrast`.
    pub extreme_se: f64,
    /// Two-sided normal p-value of `extreme_contrast`, before any selection correction.
    pub extreme_p: f64,
    /// Index of the group with the highest mean.
    pub extreme_top: usize,
    /// Index of the group with the lowest mean.
    pub extreme_bottom: usize,
    /// Chi-squared statistic that all nonempty group means are equal.
    pub omnibus_chi2: f64,
    /// Degrees of freedom of `omnibus_chi2`: nonempty groups minus one.
    pub omnibus_df: usize,
    /// p-value of `omnibus_chi2`.
    pub omnibus_p: f64,
}

/// GATES: weighted group means of `score` for group codes `groups[i]` in
/// `0..n_groups`, with the contrasts and the equal-means test described on
/// [`Gates`].
///
/// The mean of group `g` is `sum w psi / sum w`, and its variance is
/// `n_g/(n_g-1) sum w^2 (psi - mean_g)^2 / (sum w)^2`. The omnibus test weights
/// each mean by the inverse of its variance around the inverse-variance mean.
pub fn gates(
    score: &[f64],
    weights: &[f64],
    groups: &[usize],
    n_groups: usize,
) -> Result<Gates, MboostrsError> {
    let n = score.len();
    if weights.len() != n || groups.len() != n {
        return Err(MboostrsError::Shape(format!(
            "gates: score has {n} rows, weights {} and groups {}",
            weights.len(),
            groups.len()
        )));
    }
    if n_groups < 2 {
        return Err(MboostrsError::Shape(
            "gates needs at least two groups".into(),
        ));
    }
    if groups.iter().any(|&g| g >= n_groups) {
        return Err(MboostrsError::Shape(
            "gates group code is outside 0..n_groups".into(),
        ));
    }
    if score.iter().chain(weights).any(|v| !v.is_finite()) {
        return Err(MboostrsError::BadResponse(
            "gates".into(),
            "scores and weights must be finite".into(),
        ));
    }
    let mut wsum = vec![0.0; n_groups];
    let mut wpsi = vec![0.0; n_groups];
    let mut counts = vec![0usize; n_groups];
    for i in 0..n {
        let g = groups[i];
        wsum[g] += weights[i];
        wpsi[g] += weights[i] * score[i];
        if weights[i] > 0.0 {
            counts[g] += 1;
        }
    }
    let means: Vec<f64> = (0..n_groups)
        .map(|g| {
            if wsum[g] > 0.0 {
                wpsi[g] / wsum[g]
            } else {
                f64::NAN
            }
        })
        .collect();
    let mut sq = vec![0.0; n_groups];
    for i in 0..n {
        let g = groups[i];
        if weights[i] > 0.0 && wsum[g] > 0.0 {
            sq[g] += weights[i] * weights[i] * (score[i] - means[g]).powi(2);
        }
    }
    let se: Vec<f64> = (0..n_groups)
        .map(|g| {
            if counts[g] < 2 || wsum[g] <= 0.0 {
                return f64::NAN;
            }
            let nf = counts[g] as f64;
            (nf / (nf - 1.0) * sq[g]).sqrt() / wsum[g]
        })
        .collect();
    let nonempty: Vec<usize> = (0..n_groups).filter(|&g| counts[g] > 0).collect();
    if nonempty.len() < 2 {
        return Err(MboostrsError::Shape(
            "gates needs at least two nonempty groups".into(),
        ));
    }
    let contrast = |hi: usize, lo: usize| -> (f64, f64, f64) {
        let diff = means[hi] - means[lo];
        let s = (se[hi].powi(2) + se[lo].powi(2)).sqrt();
        (diff, s, normal_two_sided(diff / s))
    };
    let bottom = nonempty[0];
    let top = *nonempty.last().expect("two nonempty groups");
    let (ordered_contrast, ordered_se, ordered_p) = contrast(top, bottom);
    let (mut max_g, mut min_g) = (nonempty[0], nonempty[0]);
    for &g in &nonempty {
        if means[g] > means[max_g] {
            max_g = g;
        }
        if means[g] < means[min_g] {
            min_g = g;
        }
    }
    let (extreme_contrast, extreme_se, extreme_p) = contrast(max_g, min_g);
    // Inverse-variance mean of the nonempty group means, then the Wald statistic
    // for every group mean against it.
    let prec: f64 = nonempty.iter().map(|&g| 1.0 / se[g].powi(2)).sum();
    let mbar: f64 = nonempty
        .iter()
        .map(|&g| means[g] / se[g].powi(2))
        .sum::<f64>()
        / prec;
    let chi2: f64 = nonempty
        .iter()
        .map(|&g| (means[g] - mbar).powi(2) / se[g].powi(2))
        .sum();
    let df = nonempty.len() - 1;
    Ok(Gates {
        means,
        se,
        counts,
        ordered_contrast,
        ordered_se,
        ordered_p,
        ordered_bottom: bottom,
        ordered_top: top,
        extreme_contrast,
        extreme_se,
        extreme_p,
        extreme_top: max_g,
        extreme_bottom: min_g,
        omnibus_chi2: chi2,
        omnibus_df: df,
        omnibus_p: chi_square_sf(chi2, df as f64),
    })
}

/// Holm step-down adjusted p-values, in the input order.
///
/// Sorting the finite p-values ascending, the `j`-th smallest is multiplied by
/// `m - j + 1` (with `m` the number of finite p-values), and a running maximum
/// keeps the adjusted values monotone. NaN inputs stay NaN and do not count
/// toward `m`.
pub fn holm_adjust(p: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..p.len()).filter(|&i| p[i].is_finite()).collect();
    order.sort_by(|&a, &b| p[a].partial_cmp(&p[b]).expect("finite"));
    let m = order.len();
    let mut out = vec![f64::NAN; p.len()];
    let mut running = 0.0f64;
    for (rank, &i) in order.iter().enumerate() {
        let scaled = ((m - rank) as f64 * p[i]).min(1.0);
        running = running.max(scaled);
        out[i] = running;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ols_design(n: usize) -> (Vec<f64>, Vec<f64>) {
        // y = 1 + 2 x + deterministic noise, design [1, x].
        let mut x = Vec::with_capacity(2 * n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let xi = (i as f64) / (n as f64) - 0.5;
            let noise = ((i * 7919) % 101) as f64 / 101.0 - 0.5;
            x.extend([1.0, xi]);
            y.push(1.0 + 2.0 * xi + noise);
        }
        (y, x)
    }

    #[test]
    fn robust_wls_recovers_a_known_slope() {
        let (y, x) = ols_design(200);
        let w = vec![1.0; 200];
        let fit = robust_wls(&y, &x, 2, &w).unwrap();
        assert!((fit.coefficients[0] - 1.0).abs() < 0.05);
        assert!((fit.coefficients[1] - 2.0).abs() < 0.1);
        assert!(fit.se[1] > 0.0 && fit.se[1] < 0.5);
    }

    #[test]
    fn zero_weight_rows_do_not_change_the_fit() {
        let (y, x) = ols_design(120);
        let w = vec![1.0; 120];
        let base = robust_wls(&y[..100], &x[..200], 2, &w[..100]).unwrap();
        let mut w2 = w.clone();
        for v in &mut w2[100..] {
            *v = 0.0;
        }
        let padded = robust_wls(&y, &x, 2, &w2).unwrap();
        for k in 0..2 {
            assert!((base.coefficients[k] - padded.coefficients[k]).abs() < 1e-10);
        }
        assert_eq!(padded.n, 100);
    }

    #[test]
    fn wald_on_a_zero_slope_is_not_significant_and_on_a_large_one_is() {
        let (y, x) = ols_design(200);
        let w = vec![1.0; 200];
        let fit = robust_wls(&y, &x, 2, &w).unwrap();
        let (chi2, p) = wald_test(&fit.coefficients, &fit.covariance, 2, &[1]).unwrap();
        assert!(chi2 > 10.0 && p < 0.01);
        let y0: Vec<f64> = (0..200).map(|i| ((i * 37) % 11) as f64).collect();
        let null = robust_wls(&y0, &x, 2, &w).unwrap();
        let (_, p0) = wald_test(&null.coefficients, &null.covariance, 2, &[1]).unwrap();
        assert!(p0 > 0.01);
    }

    #[test]
    fn gates_contrast_is_top_minus_bottom() {
        let score = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
        let w = [1.0; 6];
        let g = [0, 0, 1, 1, 2, 2];
        let out = gates(&score, &w, &g, 3).unwrap();
        assert_eq!(out.means, vec![0.0, 1.0, 2.0]);
        assert!((out.ordered_contrast - 2.0).abs() < 1e-12);
        assert_eq!((out.ordered_top, out.ordered_bottom), (2, 0));
        assert_eq!(out.counts, vec![2, 2, 2]);
    }

    #[test]
    fn holm_matches_the_textbook_steps() {
        let adj = holm_adjust(&[0.01, 0.04, 0.03, 0.005]);
        assert!((adj[3] - 0.02).abs() < 1e-12);
        assert!((adj[0] - 0.03).abs() < 1e-12);
        assert!((adj[2] - 0.06).abs() < 1e-12);
        assert!((adj[1] - 0.06).abs() < 1e-12);
    }

    #[test]
    fn holm_keeps_nan_out_of_the_count() {
        let adj = holm_adjust(&[f64::NAN, 0.02, 0.5]);
        assert!(adj[0].is_nan());
        assert!((adj[1] - 0.04).abs() < 1e-12);
    }
}
