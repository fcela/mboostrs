//! Statistical summaries for sample-split inference and specification screens.
//!
//! The frontends assemble columns, model predictions and split records; these
//! kernels own the numerical comparisons shared by Python and R.

use faer::linalg::solvers::Cholesky;
use faer::prelude::SpSolver;
use faer::{Mat, Side};
use ndarray::{ArrayView1, ArrayView2};

use crate::{families::type7_quantile, special::normal_quantile, MboostrsError};

fn check_level(level: f64) -> Result<(), MboostrsError> {
    if !level.is_finite() || level <= 0.0 || level >= 1.0 {
        return Err(MboostrsError::Control("level must lie in (0, 1)".into()));
    }
    Ok(())
}

/// Quantile aggregation of split p-values: `min(1, quantile(p/gamma, gamma))`.
/// Uses the linearly interpolated (type-7) empirical quantile.
pub fn aggregate_split_p(p_values: &[f64], gamma: f64) -> Result<f64, MboostrsError> {
    if !gamma.is_finite() || gamma <= 0.0 || gamma >= 1.0 {
        return Err(MboostrsError::Control("gamma must lie in (0, 1)".into()));
    }
    if p_values.is_empty() {
        return Err(MboostrsError::Shape(
            "p_values must hold at least one p-value".into(),
        ));
    }
    if p_values
        .iter()
        .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
    {
        return Err(MboostrsError::Control(
            "p_values must be finite and lie in [0, 1]".into(),
        ));
    }
    let scaled: Vec<_> = p_values.iter().map(|p| p / gamma).collect();
    Ok(type7_quantile(ArrayView1::from(&scaled), gamma).min(1.0))
}

/// Two-sided normal p-value reconstructed from a symmetric Wald interval.
/// A zero-width interval gives one for a zero estimate and zero otherwise.
pub fn normal_interval_p(
    estimate: f64,
    lower: f64,
    upper: f64,
    level: f64,
) -> Result<f64, MboostrsError> {
    check_level(level)?;
    if [estimate, lower, upper].iter().any(|v| !v.is_finite()) || lower > upper {
        return Err(MboostrsError::Control(
            "estimate and ordered interval bounds must be finite".into(),
        ));
    }
    let se = (upper - lower) / (2.0 * normal_quantile(0.5 + level / 2.0));
    Ok(if se > 0.0 {
        libm::erfc((estimate / se).abs() / std::f64::consts::SQRT_2)
    } else if estimate == 0.0 {
        1.0
    } else {
        0.0
    })
}

/// Twice the smaller fraction of draws on either side of zero, capped at one.
/// Zero draws count on both sides; empty or non-finite draws are refused.
pub fn draw_p(draws: &[f64]) -> Result<f64, MboostrsError> {
    if draws.is_empty() || draws.iter().any(|v| !v.is_finite()) {
        return Err(MboostrsError::Control(
            "draws must be nonempty and finite".into(),
        ));
    }
    let below = draws.iter().filter(|&&v| v <= 0.0).count();
    let above = draws.iter().filter(|&&v| v >= 0.0).count();
    Ok((2.0 * below.min(above) as f64 / draws.len() as f64).min(1.0))
}

/// Average contrast, percentile bounds and two-sided draw p-value.
/// Draw matrices have one row per draw and one column per observation;
/// corresponding rows must use the same coefficient draw in both scenarios.
pub fn draw_contrast(
    alternative: &[f64],
    reference: &[f64],
    draws_alternative: ArrayView2<'_, f64>,
    draws_reference: ArrayView2<'_, f64>,
    level: f64,
) -> Result<(f64, f64, f64, f64), MboostrsError> {
    check_level(level)?;
    let n = alternative.len();
    if n == 0
        || reference.len() != n
        || draws_alternative.ncols() != n
        || draws_alternative.dim() != draws_reference.dim()
        || draws_alternative.nrows() == 0
    {
        return Err(MboostrsError::Shape(
            "contrast predictions and draw matrices must have matching nonempty dimensions".into(),
        ));
    }
    if alternative
        .iter()
        .chain(reference)
        .chain(draws_alternative.iter())
        .chain(draws_reference.iter())
        .any(|v| !v.is_finite())
    {
        return Err(MboostrsError::Control(
            "contrast predictions and draws must be finite".into(),
        ));
    }
    let estimate = alternative
        .iter()
        .zip(reference)
        .map(|(a, b)| a - b)
        .sum::<f64>()
        / n as f64;
    let draws: Vec<_> = draws_alternative
        .rows()
        .into_iter()
        .zip(draws_reference.rows())
        .map(|(a, b)| a.iter().zip(b).map(|(a, b)| a - b).sum::<f64>() / n as f64)
        .collect();
    let lower = type7_quantile(ArrayView1::from(&draws), (1.0 - level) / 2.0);
    let upper = type7_quantile(ArrayView1::from(&draws), (1.0 + level) / 2.0);
    Ok((estimate, lower, upper, draw_p(&draws)?))
}

/// OLS coefficients and known-variance statistics of a log-squared-residual screen.
#[derive(Debug)]
pub struct LogVarianceScreen {
    /// Slope for each supplied design column.
    pub slopes: Vec<f64>,
    /// Slopes divided by standard errors using `Var(log(chi^2_1)) = pi^2/2`.
    pub t_stats: Vec<f64>,
    /// Fraction of the centered log-squared-residual sum of squares explained.
    pub r_squared: f64,
    /// Rows with a positive finite squared residual and finite design.
    pub n_used: usize,
}

/// Regress centered `log(residual^2)` on column-centered covariates.
/// Columns are centered over their finite values before incomplete rows are
/// removed, preserving the original autospec screen's centering convention.
/// Returns no screen for a numerically rank-deficient design or fewer than
/// `p + 5` usable rows. A Cholesky pivot squared must exceed `p * eps` times
/// its column's squared norm, so roundoff cannot make duplicate columns usable.
pub fn log_variance_screen(
    residual: &[f64],
    design: ArrayView2<'_, f64>,
) -> Result<Option<LogVarianceScreen>, MboostrsError> {
    let (n, p) = design.dim();
    if residual.len() != n {
        return Err(MboostrsError::Shape(
            "residual and design rows must match".into(),
        ));
    }
    if p == 0 {
        return Ok(None);
    }
    let means: Vec<_> = design
        .columns()
        .into_iter()
        .map(|col| {
            let (sum, count) = col
                .iter()
                .filter(|v| v.is_finite())
                .fold((0.0, 0), |(s, n), v| (s + v, n + 1));
            sum / count as f64
        })
        .collect();
    let rows: Vec<_> = (0..n)
        .filter(|&i| {
            let sq = residual[i] * residual[i];
            sq.is_finite() && sq > 0.0 && design.row(i).iter().all(|v| v.is_finite())
        })
        .collect();
    if rows.len() < p + 5 {
        return Ok(None);
    }
    let y: Vec<_> = rows
        .iter()
        .map(|&i| (residual[i] * residual[i]).ln())
        .collect();
    let centre = y.iter().sum::<f64>() / y.len() as f64;
    let mut gram = Mat::<f64>::zeros(p, p);
    let mut rhs = Mat::<f64>::zeros(p, 1);
    for (&i, &yi) in rows.iter().zip(&y) {
        for j in 0..p {
            let xj = design[[i, j]] - means[j];
            rhs[(j, 0)] += xj * (yi - centre);
            for k in 0..=j {
                gram[(j, k)] += xj * (design[[i, k]] - means[k]);
            }
        }
    }
    let Ok(chol) = Cholesky::try_new(gram.as_ref(), Side::Lower) else {
        return Ok(None);
    };
    let factor = chol.compute_l();
    if (0..p).any(|j| {
        !factor[(j, j)].is_finite()
            || factor[(j, j)].powi(2) <= p as f64 * f64::EPSILON * gram[(j, j)]
    }) {
        return Ok(None);
    }
    chol.solve_in_place(rhs.as_mut());
    let slopes: Vec<_> = (0..p).map(|j| rhs[(j, 0)]).collect();
    let mut inverse = Mat::<f64>::identity(p, p);
    chol.solve_in_place(inverse.as_mut());
    let variance = std::f64::consts::PI.powi(2) / 2.0;
    let t_stats: Vec<_> = (0..p)
        .map(|j| {
            let se = (variance * inverse[(j, j)]).sqrt();
            if se > 0.0 {
                slopes[j] / se
            } else {
                0.0
            }
        })
        .collect();
    if slopes.iter().chain(&t_stats).any(|v| !v.is_finite()) {
        return Ok(None);
    }
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    for (&i, &yi) in rows.iter().zip(&y) {
        let fitted = (0..p)
            .map(|j| (design[[i, j]] - means[j]) * slopes[j])
            .sum::<f64>();
        ss_res += (yi - centre - fitted).powi(2);
        ss_tot += (yi - centre).powi(2);
    }
    Ok(Some(LogVarianceScreen {
        slopes,
        t_stats,
        r_squared: if ss_tot > 0.0 {
            1.0 - ss_res / ss_tot
        } else {
            0.0
        },
        n_used: rows.len(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contribution::paired_mean_se;
    use ndarray::array;

    #[test]
    fn split_p_values_and_normal_tail() {
        assert!((aggregate_split_p(&[0.01, 0.2, 1.0], 0.5).unwrap() - 0.4).abs() < 1e-15);
        assert!((aggregate_split_p(&[0.01, 0.02, 0.03, 0.04], 0.25).unwrap() - 0.07).abs() < 1e-15);
        assert_eq!(aggregate_split_p(&[0.8, 0.9], 0.5).unwrap(), 1.0);
        let z = normal_quantile(0.975);
        assert!((normal_interval_p(z, 0.0, 2.0 * z, 0.95).unwrap() - 0.05).abs() < 1e-14);
        let tail = normal_interval_p(10.0, 10.0 - z, 10.0 + z, 0.95).unwrap();
        assert!((tail / 1.523970604832105e-23 - 1.0).abs() < 1e-12);
        assert_eq!(normal_interval_p(0.0, 0.0, 0.0, 0.95).unwrap(), 1.0);
        assert_eq!(normal_interval_p(1.0, 1.0, 1.0, 0.95).unwrap(), 0.0);
        assert!(aggregate_split_p(&[], 0.5).is_err());
        assert!(aggregate_split_p(&[f64::NAN], 0.5).is_err());
        assert!(aggregate_split_p(&[-0.1], 0.5).is_err());
        assert!(aggregate_split_p(&[0.2], 1.0).is_err());
        assert!(normal_interval_p(1.0, 2.0, 0.0, 0.95).is_err());
        assert!(normal_interval_p(1.0, 0.0, 2.0, f64::NAN).is_err());
    }

    #[test]
    fn paired_losses_keep_zero_and_single_row_behavior() {
        let (gain, se) = paired_mean_se(&[2.0, 4.0, 6.0], &[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(gain, 2.0);
        assert!((se - (1.0_f64 / 3.0).sqrt()).abs() < 1e-15);
        assert_eq!(paired_mean_se(&[2.0; 3], &[1.0; 3]).unwrap(), (1.0, 0.0));
        assert!(paired_mean_se(&[2.0], &[1.0]).unwrap().1.is_nan());
        assert!(paired_mean_se(&[], &[]).is_err());
        assert!(paired_mean_se(&[1.0], &[0.0, 1.0]).is_err());
        assert!(paired_mean_se(&[f64::NAN], &[0.0]).is_err());
    }

    #[test]
    fn draw_contrast_averages_rows_before_quantiles() {
        let alt = array![[-2.0, 0.0], [0.0, 0.0], [0.0, 2.0], [2.0, 4.0]];
        let reference = ndarray::Array2::zeros((4, 2));
        let (estimate, lower, upper, p) =
            draw_contrast(&[2.0, 4.0], &[0.0, 0.0], alt.view(), reference.view(), 0.5).unwrap();
        assert_eq!(estimate, 3.0);
        assert_eq!((lower, upper, p), (-0.25, 1.5, 1.0));
        assert_eq!(draw_p(&[0.0; 4]).unwrap(), 1.0);
        assert_eq!(draw_p(&[1.0, 2.0]).unwrap(), 0.0);
        assert!(draw_p(&[]).is_err());
        assert!(draw_contrast(&[1.0], &[0.0], alt.view(), reference.view(), 0.95).is_err());
    }

    #[test]
    fn log_variance_recovers_an_exact_slope_and_refuses_singular_design() {
        let x = ndarray::Array2::from_shape_fn((9, 1), |(i, _)| i as f64 - 4.0);
        let residual: Vec<_> = (0..9)
            .map(|i| (0.5 * (2.0 + 0.3 * x[[i, 0]])).exp())
            .collect();
        let s = log_variance_screen(&residual, x.view()).unwrap().unwrap();
        assert!((s.slopes[0] - 0.3).abs() < 1e-14);
        assert!((s.t_stats[0] - 0.3 / (std::f64::consts::PI.powi(2) / 120.0).sqrt()).abs() < 1e-14);
        assert!((s.r_squared - 1.0).abs() < 1e-14);
        assert_eq!(s.n_used, 9);
        let singular = ndarray::Array2::<f64>::ones((9, 2));
        assert!(log_variance_screen(&residual, singular.view())
            .unwrap()
            .is_none());
        assert!(log_variance_screen(&[0.0; 9], x.view()).unwrap().is_none());
        assert!(log_variance_screen(&[1.0; 8], x.view()).is_err());
    }
}
