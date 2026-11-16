//! Agreement of a fitted association with a declared sign or shape.
//!
//! A declared constraint says something about the shape of one variable's
//! association with the response. The fitted model answers with a curve, and
//! this module measures how far the curve agrees with the declaration, then
//! turns the measurement into one of three verdicts.
//!
//! The two checks differ in what they read:
//!
//! * [`check_sign`] (declarations `positive` and `negative`) reads the contrast
//!   of the variable on each held-out fold: for each row, the link-scale
//!   prediction with the variable at its 90th percentile minus the prediction
//!   with it at its 10th, averaged over the fold's rows. The verdict needs the
//!   bootstrap percentile interval of the full-data contrast to exclude zero,
//!   with the same sign in at least four of five folds.
//! * [`check_shape`] (declarations `increasing`, `decreasing`, `convex`,
//!   `concave`) reads the average curve on a grid of the variable's values. It
//!   reports the share of grid steps (first differences for monotone
//!   declarations, second differences for convex or concave ones) that go the
//!   declared way, and the change in held-out risk when the constraint is
//!   removed.
//!
//! The inputs are predictions from the model with the declared constraint
//! removed. A monotone constraint forces its own contrast and steps to the
//! declared sign, so reading them from the constrained fit would agree by
//! construction. The caller fits the unconstrained model; this module only
//! measures what it is given.

use ndarray::{ArrayView1, ArrayView2, Axis};

use crate::families::type7_quantile;
use crate::MboostrsError;

/// Confidence level of the bootstrap percentile interval for the sign rule: the
/// interval is the 2.5% to 97.5% range of the bootstrap contrasts.
pub const SIGN_LEVEL: f64 = 0.95;

/// Share of folds, rounded up, whose contrast must share the sign of the
/// contrast before a sign verdict is reached: at least four of five folds.
pub const SIGN_FOLD_SHARE: f64 = 0.8;

/// At least this share of a fitted curve's steps must go the declared way for
/// a monotone or curvature declaration to agree.
pub const SHAPE_AGREE_SHARE: f64 = 0.9;

/// At most this share of steps going the declared way contradicts it.
pub const SHAPE_CONTRADICT_SHARE: f64 = 0.1;

/// Relative change in held-out risk, from removing the constraint, that a
/// shape verdict tolerates in either direction.
pub const RISK_CHANGE_TOLERANCE: f64 = 0.01;

/// The verdict on one declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// The fitted association has the declared direction or shape.
    Agrees,
    /// The fitted association has the opposite direction or shape.
    Contradicts,
    /// The measurements do not decide the question.
    Inconclusive,
}

impl Verdict {
    /// The label written into reports and records.
    pub fn as_str(self) -> &'static str {
        match self {
            Verdict::Agrees => "agrees",
            Verdict::Contradicts => "contradicts",
            Verdict::Inconclusive => "inconclusive",
        }
    }
}

/// Measurements behind a sign verdict.
#[derive(Debug, Clone, PartialEq)]
pub struct SignCheck {
    /// Contrast over all rows, 90th percentile minus 10th. The verdict is
    /// decided by the bootstrap interval and the folds.
    pub contrast: f64,
    /// Contrast over the held-out rows of each fold.
    pub fold_contrasts: Vec<f64>,
    /// Lower end of the bootstrap percentile interval of the contrast.
    pub lower: f64,
    /// Upper end of the bootstrap percentile interval of the contrast.
    pub upper: f64,
    /// Folds whose contrast has the sign of the full-data contrast.
    pub folds_agreeing: usize,
    /// The verdict under the rule documented on [`check_sign`].
    pub verdict: Verdict,
}

/// Measurements behind a monotone or curvature verdict.
#[derive(Debug, Clone, PartialEq)]
pub struct ShapeCheck {
    /// Average fitted curve over the grid, one value per grid point.
    pub curve: Vec<f64>,
    /// Steps larger than the numerical tolerance.
    pub steps: usize,
    /// Share of those steps that go the declared way; `None` when the curve
    /// has no step above tolerance (it is flat).
    pub share: Option<f64>,
    /// Mean held-out risk of the constrained model over folds.
    pub risk_constrained: f64,
    /// Mean held-out risk of the unconstrained model over folds.
    pub risk_unconstrained: f64,
    /// `(risk_unconstrained - risk_constrained) / |risk_constrained|`. Negative
    /// when removing the constraint lowers held-out risk. NaN when the
    /// constrained risk is zero.
    pub risk_change: f64,
    /// The verdict under the rule documented on [`check_shape`].
    pub verdict: Verdict,
}

fn mean(values: ArrayView1<f64>) -> f64 {
    values.sum() / values.len() as f64
}

/// Mean of `hi - lo` over rows.
pub fn average_contrast(hi: ArrayView1<f64>, lo: ArrayView1<f64>) -> Result<f64, MboostrsError> {
    if hi.len() != lo.len() {
        return Err(MboostrsError::Shape(format!(
            "contrast arrays differ in length: {} and {}",
            hi.len(),
            lo.len()
        )));
    }
    if hi.is_empty() {
        return Err(MboostrsError::Shape(
            "contrast needs at least one row".into(),
        ));
    }
    Ok(mean(hi) - mean(lo))
}

/// Sign verdict for a declared `positive` (`sign = 1.0`) or `negative`
/// (`sign = -1.0`) effect.
///
/// `hi` and `lo` are the full-data predictions with the variable at its 90th
/// and 10th percentile; the full-data contrast is `contrast`. `boot` holds the
/// same contrast recomputed on each bootstrap refit (rows resampled with
/// replacement and the unconstrained model refitted). Each entry of `folds`
/// holds the two prediction vectors on one fold's held-out rows.
///
/// Let `[lower, upper]` be the percentile interval of `boot` at [`SIGN_LEVEL`]
/// (2.5% and 97.5% for 0.95), and `direction` the sign of `contrast`. The
/// verdict is decided only when the interval excludes zero and at least
/// `ceil(0.8 * K)` of the `K` fold contrasts have the sign `direction`. A
/// decided verdict is `agrees` when `direction` is the declared sign and
/// `contradicts` when it is the opposite. Otherwise it is `inconclusive`.
pub fn check_sign(
    sign: f64,
    hi: ArrayView1<f64>,
    lo: ArrayView1<f64>,
    folds: &[(ArrayView1<f64>, ArrayView1<f64>)],
    boot: ArrayView1<f64>,
) -> Result<SignCheck, MboostrsError> {
    if sign != 1.0 && sign != -1.0 {
        return Err(MboostrsError::Control(format!(
            "sign must be 1 or -1, got {sign}"
        )));
    }
    if folds.is_empty() {
        return Err(MboostrsError::Shape(
            "sign check needs at least one fold".into(),
        ));
    }
    if boot.len() < 2 {
        return Err(MboostrsError::Shape(format!(
            "sign check needs at least two bootstrap contrasts, got {}",
            boot.len()
        )));
    }
    let contrast = average_contrast(hi, lo)?;
    let fold_contrasts = folds
        .iter()
        .map(|(h, l)| average_contrast(*h, *l))
        .collect::<Result<Vec<f64>, _>>()?;
    let tail = (1.0 - SIGN_LEVEL) / 2.0;
    let bounds = quantiles(boot, &[tail, 1.0 - tail])?;
    let (lower, upper) = (bounds[0], bounds[1]);
    let direction = if contrast >= 0.0 { 1.0 } else { -1.0 };
    let folds_agreeing = fold_contrasts
        .iter()
        .filter(|c| **c * direction > 0.0)
        .count();
    let k = fold_contrasts.len();
    // ceil(0.8 * k) in integer arithmetic: four of five folds, two of two.
    let needed = (4 * k + 4) / 5;
    let excludes_zero = lower > 0.0 || upper < 0.0;
    let decided = excludes_zero && folds_agreeing >= needed;
    let verdict = if !decided {
        Verdict::Inconclusive
    } else if direction == sign {
        Verdict::Agrees
    } else {
        Verdict::Contradicts
    };
    Ok(SignCheck {
        contrast,
        fold_contrasts,
        lower,
        upper,
        folds_agreeing,
        verdict,
    })
}

/// Shape verdict for a declared monotone (`second_difference = false`) or
/// curvature (`second_difference = true`) constraint.
///
/// `sign` is `1.0` for increasing or convex and `-1.0` for decreasing or
/// concave. `grid` holds one row per grid value of the variable and one column
/// per data row: the link-scale prediction of the unconstrained model with the
/// variable set to that value. `risk_constrained` and `risk_unconstrained` hold
/// the mean held-out loss of each model, one value per fold.
///
/// The verdict is `agrees` when at least [`SHAPE_AGREE_SHARE`] of the steps go
/// the declared way and removing the constraint does not lower held-out risk by
/// more than [`RISK_CHANGE_TOLERANCE`]. It is `contradicts` when at most
/// [`SHAPE_CONTRADICT_SHARE`] of the steps go the declared way and removing the
/// constraint lowers held-out risk by more than the tolerance. Otherwise, or
/// when the curve is flat, it is `inconclusive`.
pub fn check_shape(
    sign: f64,
    second_difference: bool,
    grid: ArrayView2<f64>,
    risk_constrained: &[f64],
    risk_unconstrained: &[f64],
) -> Result<ShapeCheck, MboostrsError> {
    if sign != 1.0 && sign != -1.0 {
        return Err(MboostrsError::Control(format!(
            "sign must be 1 or -1, got {sign}"
        )));
    }
    let points = grid.nrows();
    let min_points = if second_difference { 3 } else { 2 };
    if points < min_points {
        return Err(MboostrsError::Shape(format!(
            "shape check needs at least {min_points} grid points, got {points}"
        )));
    }
    if grid.ncols() == 0 {
        return Err(MboostrsError::Shape(
            "shape check needs fitted predictions".into(),
        ));
    }
    if risk_constrained.is_empty() || risk_constrained.len() != risk_unconstrained.len() {
        return Err(MboostrsError::Shape(
            "shape check needs one constrained and one unconstrained risk per fold".into(),
        ));
    }
    let curve: Vec<f64> = grid.axis_iter(Axis(0)).map(mean).collect();
    let mut steps: Vec<f64> = curve.windows(2).map(|w| w[1] - w[0]).collect();
    if second_difference {
        steps = steps.windows(2).map(|w| w[1] - w[0]).collect();
    }
    let largest = curve.iter().fold(1.0_f64, |acc, v| acc.max(v.abs()));
    let tolerance = 1e-9 * largest;
    let moving: Vec<f64> = steps.into_iter().filter(|d| d.abs() > tolerance).collect();
    let share = if moving.is_empty() {
        None
    } else {
        let declared = moving.iter().filter(|d| **d * sign > 0.0).count();
        Some(declared as f64 / moving.len() as f64)
    };

    let folds = risk_constrained.len() as f64;
    let risk_constrained_mean = risk_constrained.iter().sum::<f64>() / folds;
    let risk_unconstrained_mean = risk_unconstrained.iter().sum::<f64>() / folds;
    let risk_change = if risk_constrained_mean.abs() > 1e-12 {
        (risk_unconstrained_mean - risk_constrained_mean) / risk_constrained_mean.abs()
    } else {
        f64::NAN
    };

    let verdict = match share {
        None => Verdict::Inconclusive,
        Some(s) if s >= SHAPE_AGREE_SHARE && risk_change >= -RISK_CHANGE_TOLERANCE => {
            Verdict::Agrees
        }
        Some(s) if s <= SHAPE_CONTRADICT_SHARE && risk_change <= -RISK_CHANGE_TOLERANCE => {
            Verdict::Contradicts
        }
        Some(_) => Verdict::Inconclusive,
    };
    Ok(ShapeCheck {
        curve,
        steps: moving.len(),
        share,
        risk_constrained: risk_constrained_mean,
        risk_unconstrained: risk_unconstrained_mean,
        risk_change,
        verdict,
    })
}

/// Type-7 quantiles of `values` at each probability in `probs`.
pub fn quantiles(values: ArrayView1<f64>, probs: &[f64]) -> Result<Vec<f64>, MboostrsError> {
    if values.is_empty() {
        return Err(MboostrsError::Shape(
            "quantiles need at least one value".into(),
        ));
    }
    if let Some(p) = probs.iter().find(|p| !(0.0..=1.0).contains(*p)) {
        return Err(MboostrsError::Control(format!(
            "probability {p} is outside [0, 1]"
        )));
    }
    Ok(probs.iter().map(|&p| type7_quantile(values, p)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2, Array1, Array2};

    #[test]
    fn contrast_is_mean_of_row_differences() {
        let hi = arr1(&[3.0, 5.0, 7.0]);
        let lo = arr1(&[1.0, 1.0, 1.0]);
        assert_eq!(average_contrast(hi.view(), lo.view()).unwrap(), 4.0);
    }

    /// Five folds whose contrasts are the given values: each fold's two
    /// prediction vectors differ by `c` on every row.
    fn folds_with(contrasts: &[f64]) -> Vec<(Array1<f64>, Array1<f64>)> {
        contrasts
            .iter()
            .map(|c| (Array1::from_elem(4, *c), Array1::zeros(4)))
            .collect()
    }

    /// A bootstrap sample of 200 contrasts spread evenly over `[low, high]`.
    fn boot_between(low: f64, high: f64) -> Array1<f64> {
        Array1::from_shape_fn(200, |i| low + (high - low) * i as f64 / 199.0)
    }

    /// The sign check on full-data contrast 1 with the given fold contrasts and
    /// bootstrap contrasts.
    fn sign_of(sign: f64, folds: &[f64], boot: &Array1<f64>) -> SignCheck {
        let full_hi = Array1::from_elem(4, 1.0);
        let full_lo = Array1::zeros(4);
        let folds = folds_with(folds);
        let views: Vec<_> = folds.iter().map(|(h, l)| (h.view(), l.view())).collect();
        check_sign(sign, full_hi.view(), full_lo.view(), &views, boot.view()).unwrap()
    }

    #[test]
    fn sign_agrees_when_the_bootstrap_interval_excludes_zero_and_folds_agree() {
        let boot = boot_between(0.8, 1.2);
        let check = sign_of(1.0, &[1.0, 1.1, 0.9, 1.05, 0.95], &boot);
        assert_eq!(check.verdict, Verdict::Agrees);
        assert_eq!(check.folds_agreeing, 5);
        assert!(check.lower > 0.0 && check.upper > check.lower);
        assert_eq!(
            sign_of(-1.0, &[1.0, 1.1, 0.9, 1.05, 0.95], &boot).verdict,
            Verdict::Contradicts
        );
    }

    #[test]
    fn sign_is_inconclusive_when_the_bootstrap_interval_contains_zero() {
        let boot = boot_between(-0.5, 1.5);
        let check = sign_of(1.0, &[1.0, 1.1, 0.9, 1.05, 0.95], &boot);
        assert!(check.lower < 0.0 && check.upper > 0.0);
        assert_eq!(check.verdict, Verdict::Inconclusive);
    }

    #[test]
    fn sign_is_inconclusive_when_fewer_than_four_of_five_folds_agree() {
        // The interval excludes zero, but only three folds share the sign of
        // the full-data contrast.
        let boot = boot_between(0.8, 1.2);
        let check = sign_of(1.0, &[1.0, 1.0, 1.0, -1.0, -1.0], &boot);
        assert_eq!(check.folds_agreeing, 3);
        assert_eq!(check.verdict, Verdict::Inconclusive);
    }

    #[test]
    fn sign_is_inconclusive_when_four_of_five_folds_agree_but_the_interval_contains_zero() {
        let boot = boot_between(-0.2, 1.2);
        let check = sign_of(1.0, &[1.0, 1.0, 1.0, 1.0, -1.0], &boot);
        assert_eq!(check.folds_agreeing, 4);
        assert_eq!(check.verdict, Verdict::Inconclusive);
    }

    #[test]
    fn sign_needs_two_bootstrap_contrasts_and_a_valid_sign() {
        let hi = arr1(&[2.0, 2.0]);
        let lo = arr1(&[0.0, 0.0]);
        let fold = (hi.view(), lo.view());
        assert!(check_sign(1.0, hi.view(), lo.view(), &[fold], arr1(&[1.0]).view()).is_err());
        let boot = boot_between(0.8, 1.2);
        assert!(check_sign(0.0, hi.view(), lo.view(), &[fold], boot.view()).is_err());
    }

    #[test]
    fn increasing_curve_with_no_risk_cost_agrees() {
        let grid = Array2::from_shape_fn((5, 3), |(j, _)| j as f64);
        let check = check_shape(1.0, false, grid.view(), &[1.0, 1.0], &[1.0, 1.0]).unwrap();
        assert_eq!(check.share, Some(1.0));
        assert_eq!(check.verdict, Verdict::Agrees);
    }

    #[test]
    fn decreasing_curve_contradicts_an_increasing_declaration_when_constraint_costs_risk() {
        let grid = Array2::from_shape_fn((5, 3), |(j, _)| -(j as f64));
        let check = check_shape(1.0, false, grid.view(), &[2.0, 2.0], &[1.0, 1.0]).unwrap();
        assert_eq!(check.share, Some(0.0));
        assert!((check.risk_change + 0.5).abs() < 1e-12);
        assert_eq!(check.verdict, Verdict::Contradicts);
    }

    #[test]
    fn flat_curve_is_inconclusive() {
        let grid = arr2(&[[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]);
        let check = check_shape(1.0, false, grid.view(), &[1.0], &[1.0]).unwrap();
        assert_eq!(check.share, None);
        assert_eq!(check.verdict, Verdict::Inconclusive);
    }

    #[test]
    fn convex_curve_agrees_on_second_differences() {
        let grid = Array2::from_shape_fn((6, 2), |(j, _)| (j as f64) * (j as f64));
        let check = check_shape(1.0, true, grid.view(), &[1.0], &[1.0]).unwrap();
        assert_eq!(check.share, Some(1.0));
        assert_eq!(check.verdict, Verdict::Agrees);
    }

    #[test]
    fn quantiles_use_type_seven_interpolation() {
        let values = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let q = quantiles(values.view(), &[0.0, 0.5, 1.0]).unwrap();
        assert_eq!(q, vec![1.0, 3.0, 5.0]);
    }
}
