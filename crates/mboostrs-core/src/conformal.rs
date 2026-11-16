//! Conformal prediction intervals: calibration scores, finite-sample quantiles
//! and band construction for split, CQR, CV+ and distributional intervals.
//!
//! A conformity score measures how far an observation fell from its
//! prediction. The conformal quantile of the calibration scores is the
//! `ceil((n + 1) * level)`-th smallest score; widening a prediction by it
//! gives marginal coverage of at least `level` at finite `n` when the
//! calibration rows and a new row are exchangeable. The `+ 1` is what makes
//! the bound finite-sample. When that index exceeds `n` the bound cannot be
//! attained and the quantile is infinite.
//!
//! Fitting the models that produce the predictions (refits on subsets of the
//! training rows, the random split, the fold assignment and the tuning of
//! each refit's step count) is orchestrated by the frontends. This module
//! receives their predictions as arrays and returns the intervals.

use crate::MboostrsError;
use ndarray::ArrayView2;

/// Smallest step count a tuned quantile refit searches from.
///
/// A start of 200 stops the search where the pinball-loss CV curve first flattens,
/// while the interval width still falls at the same coverage. Starting at 1600
/// narrowed CQR intervals by 1.2 to 2.3% on fresh seeds (issue #559).
pub const TUNED_MSTOP_FLOOR: usize = 1600;

/// Largest step count a tuned quantile refit searches up to.
pub const TUNED_MSTOP_CAP: usize = 10_000;

/// Check that a miscoverage level is a nominal coverage in the open interval
/// (0, 1). Returns the error that frontends raise as a value error.
pub fn validate_level(level: f64) -> Result<(), MboostrsError> {
    if level.is_finite() && level > 0.0 && level < 1.0 {
        Ok(())
    } else {
        Err(MboostrsError::Control(format!(
            "level must be in (0, 1), got {level}"
        )))
    }
}

/// Relative distance from an integer within which a rank product counts as
/// that integer. `1 - 0.9` is `0.09999999999999998` in binary, so
/// `(1 - 0.9) * 10` evaluates to `0.9999999999999998`; a plain floor of it
/// gives 0 where the exact value 1 is meant.
const RANK_TOLERANCE: f64 = 1e-9;

/// `x` rounded to the nearest integer when it lies within [`RANK_TOLERANCE`]
/// (relative) of one, and `None` otherwise.
fn near_integer(x: f64) -> Option<f64> {
    let r = x.round();
    ((x - r).abs() <= RANK_TOLERANCE * x.abs().max(1.0)).then_some(r)
}

/// `floor(x)`, where an `x` within rounding error of an integer is that integer.
fn rank_floor(x: f64) -> usize {
    near_integer(x).unwrap_or_else(|| x.floor()) as usize
}

/// `ceil(x)`, where an `x` within rounding error of an integer is that integer.
fn rank_ceil(x: f64) -> usize {
    near_integer(x).unwrap_or_else(|| x.ceil()) as usize
}

/// The smallest calibration set size that can attain `level`. The conformal
/// rank `ceil((n + 1) * level)` is at most `n` exactly when
/// `n >= level / (1 - level)`, so the minimum is `ceil(level / (1 - level))`:
/// 9 at level 0.9, 4 at level 0.8, 19 at level 0.95. Below it the conformal
/// quantile is infinite.
pub fn min_calibration_size(level: f64) -> Result<usize, MboostrsError> {
    validate_level(level)?;
    Ok(rank_ceil(level / (1.0 - level)))
}

/// Number of rows held out for calibration: `n * fraction` rounded half to
/// even, which is how Python's `round` rounds. Fails when either side of the
/// split would be empty.
pub fn calibration_count(n: usize, fraction: f64) -> Result<usize, MboostrsError> {
    if !(fraction > 0.0 && fraction < 1.0) {
        return Err(MboostrsError::Control(format!(
            "calib_fraction must be in (0, 1), got {fraction}"
        )));
    }
    let count = (n as f64 * fraction).round_ties_even() as usize;
    if count < 1 || count >= n {
        return Err(MboostrsError::Control(format!(
            "calib_fraction={fraction} leaves an empty split for n={n}"
        )));
    }
    Ok(count)
}

/// The conformal quantile of `scores` at `level`: the
/// `ceil((n + 1) * level)`-th smallest score, or `+inf` when that index is
/// larger than `n`. An empty score set gives `+inf`.
pub fn conformal_quantile(scores: &[f64], level: f64) -> Result<f64, MboostrsError> {
    validate_level(level)?;
    reject_nan(scores, "calibration scores")?;
    let mut sorted = scores.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("NaN rejected above"));
    let n = sorted.len();
    let k = rank_ceil(((n as f64) + 1.0) * level);
    if k > n {
        return Ok(f64::INFINITY);
    }
    Ok(sorted[k.max(1) - 1])
}

/// Absolute residuals `|y - predicted|`, one per row. Split conformal scores.
pub fn absolute_residuals(y: &[f64], predicted: &[f64]) -> Result<Vec<f64>, MboostrsError> {
    same_length(y, predicted, "response", "predicted")?;
    reject_nan(y, "response")?;
    reject_nan(predicted, "predicted")?;
    Ok(y.iter()
        .zip(predicted)
        .map(|(y, p)| (y - p).abs())
        .collect())
}

/// The band `predicted - q` to `predicted + q`, used by split conformal.
pub fn symmetric_band(predicted: &[f64], q: f64) -> (Vec<f64>, Vec<f64>) {
    (
        predicted.iter().map(|p| p - q).collect(),
        predicted.iter().map(|p| p + q).collect(),
    )
}

/// Standardized location-scale residuals `|y-mu|/sigma`.
/// The location and positive scale must be fitted without calibration responses.
/// Split validity depends on exchangeability, not a Gaussian distribution.
pub fn location_scale_scores(
    y: &[f64],
    mu: &[f64],
    sigma: &[f64],
) -> Result<Vec<f64>, MboostrsError> {
    same_length(y, mu, "response", "location")?;
    location_scale_parameters(mu, sigma)?;
    if y.iter().any(|v| !v.is_finite()) {
        return Err(MboostrsError::Control(
            "location-scale scores need finite responses".into(),
        ));
    }
    Ok(y.iter()
        .zip(mu)
        .zip(sigma)
        .map(|((y, mu), sigma)| (y - mu).abs() / sigma)
        .collect())
}

/// Invert standardized scores as `mu +/- q*sigma`, allowing `q=+inf`.
pub fn location_scale_band(
    mu: &[f64],
    sigma: &[f64],
    q: f64,
) -> Result<(Vec<f64>, Vec<f64>), MboostrsError> {
    location_scale_parameters(mu, sigma)?;
    if q.is_nan() || q < 0.0 {
        return Err(MboostrsError::Control(
            "location-scale cutoff must be nonnegative".into(),
        ));
    }
    Ok((
        mu.iter()
            .zip(sigma)
            .map(|(mu, sigma)| mu - q * sigma)
            .collect(),
        mu.iter()
            .zip(sigma)
            .map(|(mu, sigma)| mu + q * sigma)
            .collect(),
    ))
}

fn location_scale_parameters(mu: &[f64], sigma: &[f64]) -> Result<(), MboostrsError> {
    same_length(mu, sigma, "location", "scale")?;
    if mu.iter().any(|v| !v.is_finite()) || sigma.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err(MboostrsError::Control(
            "location-scale calibration needs finite locations and strictly positive finite scales"
                .into(),
        ));
    }
    Ok(())
}

/// Count-response calibration scores using a fixed nonnegative mean fit.
/// `poisson_normalized` scales residuals by sqrt(max(mu, 1e-12));
/// `poisson_quantile` uses the fitted Poisson central quantile band.
/// Calibration validity needs exchangeability, not a correct Poisson model.
pub fn poisson_scores(
    y: &[f64],
    predicted: &[f64],
    level: f64,
    score: &str,
) -> Result<Vec<f64>, MboostrsError> {
    same_length(y, predicted, "response", "predicted")?;
    if y.iter()
        .any(|v| !v.is_finite() || *v < 0.0 || v.fract() != 0.0)
    {
        return Err(MboostrsError::BadResponse(
            "Poisson conformal".into(),
            "Poisson conformal scores need finite nonnegative integer responses".into(),
        ));
    }
    let (lower, upper) = poisson_base_band(predicted, level, score)?;
    if score == "poisson_normalized" {
        Ok(y.iter()
            .zip(predicted)
            .zip(upper)
            .map(|((y, mu), scale)| (y - mu).abs() / scale)
            .collect())
    } else {
        quantile_scores(&lower, &upper, y)
    }
}

/// Real endpoints for count-response split calibration. Known count support
/// is not projected here; callers may intersect all compared methods with it.
pub fn poisson_band(
    predicted: &[f64],
    level: f64,
    q: f64,
    score: &str,
) -> Result<(Vec<f64>, Vec<f64>), MboostrsError> {
    if q.is_nan() || (score == "poisson_normalized" && q < 0.0) {
        return Err(MboostrsError::Control(
            "invalid Poisson conformal correction".into(),
        ));
    }
    let (lower, upper) = poisson_base_band(predicted, level, score)?;
    if score == "poisson_normalized" {
        Ok((
            predicted
                .iter()
                .zip(&upper)
                .map(|(mu, scale)| mu - q * scale)
                .collect(),
            predicted
                .iter()
                .zip(&upper)
                .map(|(mu, scale)| mu + q * scale)
                .collect(),
        ))
    } else {
        widen_band(&lower, &upper, q)
    }
}

fn poisson_base_band(
    predicted: &[f64],
    level: f64,
    score: &str,
) -> Result<(Vec<f64>, Vec<f64>), MboostrsError> {
    validate_level(level)?;
    if predicted.iter().any(|mu| !mu.is_finite() || *mu < 0.0) {
        return Err(MboostrsError::BadResponse(
            "Poisson conformal".into(),
            "Poisson conformal scores need finite nonnegative predicted means".into(),
        ));
    }
    match score {
        "poisson_normalized" => Ok((
            Vec::new(),
            predicted.iter().map(|mu| mu.max(1e-12).sqrt()).collect(),
        )),
        "poisson_quantile" => {
            let alpha = 1.0 - level;
            let quantiles = |p| {
                predicted
                    .iter()
                    .map(|mu| {
                        if *mu >= 9_007_199_254_740_992.0 {
                            return Err(MboostrsError::Solver(
                                "Poisson conformal mean exceeds the exact-count quantile range"
                                    .into(),
                            ));
                        }
                        let result = crate::special::poisson_quantile_affine(p, *mu, 0.0, 1.0);
                        if !result.converged || !result.value.is_finite() {
                            Err(MboostrsError::Solver(
                                "Poisson conformal quantile did not converge".into(),
                            ))
                        } else {
                            Ok(result.value)
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()
            };
            Ok((quantiles(alpha / 2.0)?, quantiles(1.0 - alpha / 2.0)?))
        }
        _ => Err(MboostrsError::Control(format!(
            "unknown Poisson conformal score {score:?}"
        ))),
    }
}

/// Elementwise `(min(a, b), max(a, b))`. Sorts two crossed quantile
/// predictions so the lower one is first, as Chernozhukov, Fernandez-Val and
/// Galichon's rearrangement does for a pair of quantiles.
pub fn ordered_pair(a: &[f64], b: &[f64]) -> Result<(Vec<f64>, Vec<f64>), MboostrsError> {
    same_length(a, b, "first", "second")?;
    reject_nan(a, "first")?;
    reject_nan(b, "second")?;
    let n = a.len();
    let mut stacked = ndarray::Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        stacked[[i, 0]] = a[i];
        stacked[[i, 1]] = b[i];
    }
    let sorted = crate::rearrange::rearrange_levels(stacked.view())?.sorted;
    Ok((sorted.column(0).to_vec(), sorted.column(1).to_vec()))
}

/// Conformity scores of a quantile band: `max(lower - y, y - upper)`. The
/// score is how far outside the band the observation fell, and it is
/// negative when the observation is comfortably inside. Romano, Patterson and
/// Candes (2019) use it for CQR; the distributional method uses it with the
/// fitted distribution's quantiles.
pub fn quantile_scores(lower: &[f64], upper: &[f64], y: &[f64]) -> Result<Vec<f64>, MboostrsError> {
    same_length(lower, upper, "lower", "upper")?;
    same_length(lower, y, "lower", "response")?;
    reject_nan(lower, "lower")?;
    reject_nan(upper, "upper")?;
    reject_nan(y, "response")?;
    Ok(lower
        .iter()
        .zip(upper)
        .zip(y)
        .map(|((lo, hi), y)| (lo - y).max(y - hi))
        .collect())
}

/// Widen a band by the conformal correction `q`: `lower - q` and `upper + q`.
///
/// A negative `q` narrows the band, and where the two quantiles are closer
/// than `2|q|` it inverts it. An inverted row has no conformal set of its
/// own. Its endpoints are replaced by their midpoint, which contains the
/// empty set, so the coverage guarantee still holds.
pub fn widen_band(
    lower: &[f64],
    upper: &[f64],
    q: f64,
) -> Result<(Vec<f64>, Vec<f64>), MboostrsError> {
    same_length(lower, upper, "lower", "upper")?;
    reject_nan(lower, "lower")?;
    reject_nan(upper, "upper")?;
    let mut lo: Vec<f64> = lower.iter().map(|l| l - q).collect();
    let mut hi: Vec<f64> = upper.iter().map(|u| u + q).collect();
    for (l, h) in lo.iter_mut().zip(hi.iter_mut()) {
        if *l > *h {
            let middle = 0.5 * (*l + *h);
            *l = middle;
            *h = middle;
        }
    }
    Ok((lo, hi))
}

/// The midpoint `0.5 * (lower + upper)` of each band row.
pub fn band_midpoint(lower: &[f64], upper: &[f64]) -> Result<Vec<f64>, MboostrsError> {
    same_length(lower, upper, "lower", "upper")?;
    Ok(lower
        .iter()
        .zip(upper)
        .map(|(l, h)| 0.5 * (l + h))
        .collect())
}

/// `values` clipped into `[lower, upper]` row by row. The distributional
/// method reports the fitted median here, kept inside the widened band.
pub fn clip_to_band(
    values: &[f64],
    lower: &[f64],
    upper: &[f64],
) -> Result<Vec<f64>, MboostrsError> {
    same_length(values, lower, "values", "lower")?;
    same_length(values, upper, "values", "upper")?;
    Ok(values
        .iter()
        .zip(lower)
        .zip(upper)
        .map(|((v, l), h)| v.max(*l).min(*h))
        .collect())
}

/// The CV+ order-statistic ranks for `n` calibrated rows at `level`:
/// `(k_lo, k_hi)` with `k_lo = floor(alpha * (n + 1))` for the lower endpoint
/// and `k_hi = ceil(level * (n + 1))` for the upper, where `alpha = 1 - level`.
/// A rank of 0 for the lower endpoint, or one above `n` for the upper,
/// means that endpoint is unbounded. A product within rounding error of an
/// integer is taken as that integer, so `n = 9`, `level = 0.9` gives
/// `k_lo = 1`, the value of `floor(0.1 * 10)` in exact arithmetic.
pub fn cv_plus_ranks(n: usize, level: f64) -> Result<(usize, usize), MboostrsError> {
    validate_level(level)?;
    let alpha = 1.0 - level;
    let np1 = (n as f64) + 1.0;
    Ok((rank_floor(alpha * np1), rank_ceil(level * np1)))
}

/// Lower bound on the coverage of balanced K-fold CV+ (Barber, Candes,
/// Ramdas and Tibshirani 2021), for `n` rows in `k` folds of equal size.
/// Returns `None` when `n` is not a multiple of `k`, because the bound is
/// stated for equal folds.
pub fn cv_plus_guaranteed_coverage(
    n: usize,
    k: usize,
    level: f64,
) -> Result<Option<f64>, MboostrsError> {
    validate_level(level)?;
    if k == 0 {
        return Err(MboostrsError::Control("n_folds must be at least 2".into()));
    }
    if n % k != 0 {
        return Ok(None);
    }
    let alpha = 1.0 - level;
    let fold_size = n as f64 / k as f64;
    let kf = k as f64;
    let correction =
        (2.0 * (1.0 - 1.0 / kf) / (fold_size + 1.0)).min((1.0 - kf / n as f64) / (kf + 1.0));
    Ok(Some((1.0 - 2.0 * alpha - correction).max(0.0)))
}

/// CV+ interval endpoints (Barber et al. 2021).
///
/// `residuals[i]` is the out-of-fold absolute residual of calibrated row `i`.
/// `fold_of[i]` is the fold that held row `i` out, and `fold_predictions[[k, j]]`
/// is the prediction at target row `j` from the model fitted without fold
/// `k`. For each target row the lower endpoint is the `k_lo`-th smallest of
/// `prediction_i - residual_i` and the upper endpoint is the `k_hi`-th
/// smallest of `prediction_i + residual_i`, with the ranks from
/// [`cv_plus_ranks`]. An endpoint whose rank falls outside `1..=n` is
/// unbounded.
pub fn cv_plus_bounds(
    residuals: &[f64],
    fold_of: &[usize],
    fold_predictions: ArrayView2<'_, f64>,
    level: f64,
) -> Result<(Vec<f64>, Vec<f64>), MboostrsError> {
    let n = residuals.len();
    let k_folds = fold_predictions.nrows();
    let n_targets = fold_predictions.ncols();
    if fold_of.len() != n {
        return Err(MboostrsError::Shape(format!(
            "fold_of has {} entries, residuals have {n}",
            fold_of.len()
        )));
    }
    reject_nan(residuals, "residuals")?;
    if let Some(bad) = fold_of.iter().find(|&&f| f >= k_folds) {
        return Err(MboostrsError::Shape(format!(
            "fold index {bad} is out of range for {k_folds} folds"
        )));
    }
    if fold_predictions.iter().any(|v| v.is_nan()) {
        return Err(MboostrsError::Shape(
            "fold predictions must not contain NaN".into(),
        ));
    }
    let (k_lo, k_hi) = cv_plus_ranks(n, level)?;
    let mut lower = vec![f64::NEG_INFINITY; n_targets];
    let mut upper = vec![f64::INFINITY; n_targets];
    let lower_bounded = k_lo >= 1;
    let upper_bounded = k_hi <= n;
    if !lower_bounded && !upper_bounded {
        return Ok((lower, upper));
    }
    let mut buffer = vec![0.0_f64; n];
    for j in 0..n_targets {
        if lower_bounded {
            for i in 0..n {
                buffer[i] = fold_predictions[[fold_of[i], j]] - residuals[i];
            }
            lower[j] = nth_smallest(&mut buffer, k_lo - 1);
        }
        if upper_bounded {
            for i in 0..n {
                buffer[i] = fold_predictions[[fold_of[i], j]] + residuals[i];
            }
            upper[j] = nth_smallest(&mut buffer, k_hi - 1);
        }
    }
    Ok((lower, upper))
}

/// The step count a tuned quantile refit starts its grid at: the model's own
/// count, but at least [`TUNED_MSTOP_FLOOR`].
pub fn tuned_mstop_start(model_mstop: usize) -> usize {
    model_mstop.max(TUNED_MSTOP_FLOOR)
}

/// The step count a tuned refit reports: its cross-validated optimum, never the
/// offset-only iteration 0.
pub fn tuned_mstop_choice(optimal: usize) -> usize {
    optimal.max(1)
}

/// The next grid end after `largest` while cross-validated risk is still
/// falling at the end of the grid: double it, but not past `cap`.
pub fn tuned_mstop_next(largest: usize, cap: usize) -> usize {
    (2 * largest).min(cap)
}

/// The bin of one covariate whose interval coverage departs most from the
/// nominal level, with its binomial z-score.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorstBin {
    /// `(coverage - level) / sqrt(level (1 - level) / count)`; negative means
    /// the bin is under-covered.
    pub z: f64,
    /// Zero-based bin index, ordered by increasing covariate value.
    pub bin: usize,
    /// Share of rows in the bin whose response lies inside the interval.
    pub coverage: f64,
    /// Rows in the bin.
    pub count: usize,
}

/// Coverage of `[lower, upper]` across the values of one covariate.
///
/// The rows are sorted by `x` and cut into `bins` groups of nearly equal
/// size. Each group's coverage is compared with `level` as a binomial
/// z-score, and the group with the most negative z is returned. A
/// constant-width interval under noise whose scale grows with `x` covers the
/// high-noise group too rarely, so its z is strongly negative even when the
/// overall coverage is nominal.
pub fn worst_covariate_bin(
    x: &[f64],
    lower: &[f64],
    upper: &[f64],
    y: &[f64],
    level: f64,
    bins: usize,
) -> Result<WorstBin, MboostrsError> {
    validate_level(level)?;
    let n = x.len();
    same_length(x, lower, "x", "lower")?;
    same_length(x, upper, "x", "upper")?;
    same_length(x, y, "x", "y")?;
    if bins < 2 || bins > n {
        return Err(MboostrsError::Control(format!(
            "bins must be between 2 and the number of rows ({n}), got {bins}"
        )));
    }
    reject_nan(x, "x")?;
    reject_nan(lower, "lower")?;
    reject_nan(upper, "upper")?;
    reject_nan(y, "y")?;

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    let mut worst: Option<WorstBin> = None;
    for bin in 0..bins {
        let start = bin * n / bins;
        let stop = (bin + 1) * n / bins;
        let count = stop - start;
        let covered = order[start..stop]
            .iter()
            .filter(|&&i| lower[i] <= y[i] && y[i] <= upper[i])
            .count();
        let coverage = covered as f64 / count as f64;
        let se = (level * (1.0 - level) / count as f64).sqrt();
        let z = (coverage - level) / se;
        if worst.is_none_or(|w| z < w.z) {
            worst = Some(WorstBin {
                z,
                bin,
                coverage,
                count,
            });
        }
    }
    // bins >= 2 guarantees the loop ran at least once.
    Ok(worst.expect("at least one bin"))
}

/// Floor applied to a squared residual before its logarithm is taken.
pub const SCALE_TARGET_FLOOR: f64 = 1e-12;

/// The target of the fitted scale model: `ln(max((y - predicted)^2, 1e-12))`.
///
/// The floor keeps a residual of exactly zero from sending the logarithm to
/// minus infinity.
pub fn scale_target(y: &[f64], predicted: &[f64]) -> Result<Vec<f64>, MboostrsError> {
    same_length(y, predicted, "y", "predicted")?;
    Ok(y.iter()
        .zip(predicted)
        .map(|(&yi, &pi)| ((yi - pi) * (yi - pi)).max(SCALE_TARGET_FLOOR).ln())
        .collect())
}

fn median_of_values(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    }
}

/// Absolute correlation of `view` with the centred `target`, or 0 when
/// either has no spread.
fn absolute_correlation(view: &[f64], centred_target: &[f64], target_ss: f64) -> f64 {
    let n = view.len() as f64;
    let mean = view.iter().sum::<f64>() / n;
    let mut cross = 0.0;
    let mut ss = 0.0;
    for (&v, &t) in view.iter().zip(centred_target) {
        let c = v - mean;
        cross += c * t;
        ss += c * c;
    }
    let denominator = (ss * target_ss).sqrt();
    if denominator > 0.0 {
        (cross / denominator).abs()
    } else {
        0.0
    }
}

/// The columns most associated with the log residual, for the scale model.
///
/// A column's strength is the larger of the absolute correlations of the
/// target with its value and with its distance from its median, so noise
/// that grows toward either tail is ranked. The `keep` strongest columns are
/// returned as indices into `columns`, strongest first; columns of equal
/// strength keep their input order. With `keep` or fewer columns the input
/// order is returned unchanged.
pub fn rank_scale_columns(
    columns: &[Vec<f64>],
    target: &[f64],
    keep: usize,
) -> Result<Vec<usize>, MboostrsError> {
    for column in columns {
        if column.len() != target.len() {
            return Err(MboostrsError::Shape(format!(
                "every column must have {} rows, got {}",
                target.len(),
                column.len()
            )));
        }
    }
    if columns.len() <= keep {
        return Ok((0..columns.len()).collect());
    }
    let n = target.len() as f64;
    let target_mean = target.iter().sum::<f64>() / n;
    let centred: Vec<f64> = target.iter().map(|t| t - target_mean).collect();
    let target_ss: f64 = centred.iter().map(|t| t * t).sum();
    let strength: Vec<f64> = columns
        .iter()
        .map(|column| {
            let median = median_of_values(column);
            let distance: Vec<f64> = column.iter().map(|v| (v - median).abs()).collect();
            let mut best = 0.0;
            for view in [column.as_slice(), distance.as_slice()] {
                let value = absolute_correlation(view, &centred, target_ss);
                if value > best {
                    best = value;
                }
            }
            best
        })
        .collect();
    let mut order: Vec<usize> = (0..columns.len()).collect();
    order.sort_by(|&a, &b| strength[b].total_cmp(&strength[a]));
    order.truncate(keep);
    Ok(order)
}

fn nth_smallest(values: &mut [f64], k: usize) -> f64 {
    let (_, nth, _) = values.select_nth_unstable_by(k, |a, b| {
        a.partial_cmp(b).expect("NaN rejected before selection")
    });
    *nth
}

fn same_length(a: &[f64], b: &[f64], name_a: &str, name_b: &str) -> Result<(), MboostrsError> {
    if a.len() != b.len() {
        return Err(MboostrsError::Shape(format!(
            "{name_a} has {} rows and {name_b} has {}",
            a.len(),
            b.len()
        )));
    }
    Ok(())
}

fn reject_nan(values: &[f64], name: &str) -> Result<(), MboostrsError> {
    if values.iter().any(|v| v.is_nan()) {
        return Err(MboostrsError::BadResponse(
            name.into(),
            "values must not contain NaN; a model that predicts NaN cannot be calibrated".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poisson_normalization_and_zero_mean_floor() {
        let score = "poisson_normalized";
        let scores = poisson_scores(&[0.0, 8.0, 6.0], &[0.0, 4.0, 9.0], 0.9, score).unwrap();
        assert_eq!(scores, vec![0.0, 2.0, 1.0]);
        let (lo, hi) = poisson_band(&[0.0, 4.0, 9.0], 0.9, 2.0, score).unwrap();
        assert_eq!(lo, vec![-2e-6, 0.0, 3.0]);
        assert_eq!(hi, vec![2e-6, 8.0, 15.0]);
        assert_eq!(conformal_quantile(&[2.0, 2.0, 2.0], 0.5).unwrap(), 2.0);
    }

    #[test]
    fn poisson_quantile_scores_and_negative_correction() {
        let score = "poisson_quantile";
        assert_eq!(
            poisson_scores(&[0.0, 1.0, 5.0], &[1.0; 3], 0.9, score).unwrap(),
            vec![0.0, -1.0, 2.0]
        );
        assert_eq!(
            poisson_band(&[1.0], 0.9, 1.0, score).unwrap(),
            (vec![-1.0], vec![4.0])
        );
        // The existing real-band envelope does not insert an integer singleton.
        assert_eq!(
            poisson_band(&[1.0], 0.9, -2.0, score).unwrap(),
            (vec![1.5], vec![1.5])
        );
        assert_eq!(
            poisson_band(&[0.0], 0.9, 0.0, score).unwrap(),
            (vec![0.0], vec![0.0])
        );
    }

    #[test]
    fn poisson_scores_reject_invalid_inputs() {
        for score in ["poisson_normalized", "poisson_quantile"] {
            for y in [-1.0, 0.5, f64::NAN, f64::INFINITY] {
                assert!(poisson_scores(&[y], &[1.0], 0.9, score).is_err());
            }
            for mu in [-1.0, f64::NAN, f64::INFINITY] {
                assert!(poisson_scores(&[1.0], &[mu], 0.9, score).is_err());
            }
            assert!(poisson_scores(&[1.0], &[], 0.9, score).is_err());
            assert!(poisson_scores(&[1.0], &[1.0], 1.0, score).is_err());
            assert!(poisson_band(&[1.0], 0.9, f64::NAN, score).is_err());
        }
        assert!(poisson_scores(&[1.0], &[1.0], 0.9, "unknown").is_err());
        assert!(poisson_band(&[1.0], 0.9, -1.0, "poisson_normalized").is_err());
    }
    use ndarray::array;

    #[test]
    fn quantile_uses_ceiling_of_n_plus_one_times_level() {
        // n = 9, level = 0.9: ceil(10 * 0.9) = 9, the largest score.
        let scores = [5.0, 1.0, 9.0, 3.0, 7.0, 2.0, 8.0, 4.0, 6.0];
        assert_eq!(conformal_quantile(&scores, 0.9).unwrap(), 9.0);
        // n = 19, level = 0.9: ceil(20 * 0.9) = 18, the 18th of 1..=19.
        let scores: Vec<f64> = (1..=19).map(f64::from).collect();
        assert_eq!(conformal_quantile(&scores, 0.9).unwrap(), 18.0);
    }

    #[test]
    fn quantile_is_infinite_when_the_rank_exceeds_n() {
        let scores = [1.0, 2.0, 3.0];
        // ceil(4 * 0.95) = 4 > 3.
        assert_eq!(conformal_quantile(&scores, 0.95).unwrap(), f64::INFINITY);
        assert_eq!(conformal_quantile(&[], 0.5).unwrap(), f64::INFINITY);
    }

    #[test]
    fn quantile_ignores_input_order() {
        let a = [0.3, 0.1, 0.4, 0.15, 0.9, 0.26];
        let mut b = a;
        b.reverse();
        assert_eq!(
            conformal_quantile(&a, 0.8).unwrap(),
            conformal_quantile(&b, 0.8).unwrap()
        );
    }

    #[test]
    fn quantile_rejects_bad_level_and_nan_scores() {
        assert!(conformal_quantile(&[1.0], 1.0).is_err());
        assert!(conformal_quantile(&[1.0], 0.0).is_err());
        assert!(conformal_quantile(&[1.0], f64::NAN).is_err());
        assert!(conformal_quantile(&[1.0, f64::NAN], 0.5).is_err());
    }

    #[test]
    fn min_calibration_size_is_the_smallest_attainable_n() {
        // Was 10: 1 / (1 - 0.9) rounds to just above 10. n = 9 attains 0.9.
        assert_eq!(min_calibration_size(0.9).unwrap(), 9);
        assert_eq!(min_calibration_size(0.8).unwrap(), 4);
        assert_eq!(min_calibration_size(0.95).unwrap(), 19);
        assert_eq!(min_calibration_size(0.5).unwrap(), 1);
        assert!(min_calibration_size(1.5).is_err());
    }

    #[test]
    fn calibration_count_rounds_half_to_even_like_python() {
        assert_eq!(calibration_count(5, 0.5).unwrap(), 2); // 2.5 -> 2
        assert_eq!(calibration_count(7, 0.5).unwrap(), 4); // 3.5 -> 4
        assert_eq!(calibration_count(100, 0.3).unwrap(), 30);
        assert!(calibration_count(1, 0.5).is_err());
        assert!(calibration_count(10, 1.0).is_err());
    }

    #[test]
    fn residuals_and_symmetric_band() {
        let r = absolute_residuals(&[1.0, -2.0, 3.0], &[0.5, -1.0, 3.0]).unwrap();
        assert_eq!(r, vec![0.5, 1.0, 0.0]);
        let (lo, hi) = symmetric_band(&[10.0, -1.0], 0.25);
        assert_eq!(lo, vec![9.75, -1.25]);
        assert_eq!(hi, vec![10.25, -0.75]);
    }

    #[test]
    fn ordered_pair_sorts_crossed_quantiles() {
        let (lo, hi) = ordered_pair(&[1.0, 5.0], &[3.0, 2.0]).unwrap();
        assert_eq!(lo, vec![1.0, 2.0]);
        assert_eq!(hi, vec![3.0, 5.0]);
    }

    #[test]
    fn cqr_scores_are_signed_distance_outside_the_band() {
        let s = quantile_scores(&[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0], &[2.0, 0.5, -1.0]).unwrap();
        assert_eq!(s, vec![1.0, -0.5, 1.0]);
    }

    #[test]
    fn widen_band_inverts_to_midpoint_when_correction_is_negative() {
        // q = -1 turns [0, 1] into [1, 0], which is inverted, so both endpoints
        // become its midpoint 0.5. The row [0, 4] becomes [1, 3], which is not.
        let (lo, hi) = widen_band(&[0.0, 0.0], &[1.0, 4.0], -1.0).unwrap();
        assert_eq!(lo, vec![0.5, 1.0]);
        assert_eq!(hi, vec![0.5, 3.0]);
        let mid = band_midpoint(&lo, &hi).unwrap();
        assert_eq!(mid, vec![0.5, 2.0]);
    }

    #[test]
    fn clip_keeps_the_median_inside_the_band() {
        let c = clip_to_band(&[5.0, -3.0, 0.5], &[0.0, 0.0, 0.0], &[2.0, 2.0, 2.0]).unwrap();
        assert_eq!(c, vec![2.0, 0.0, 0.5]);
    }

    #[test]
    fn cv_plus_ranks_match_the_barber_formulas() {
        // Were (0, 9) and (0, 4): alpha * (n + 1) rounds to just below 1.
        assert_eq!(cv_plus_ranks(9, 0.9).unwrap(), (1, 9));
        assert_eq!(cv_plus_ranks(4, 0.8).unwrap(), (1, 4));
        // Rank 0 at a low level, past n at a high one.
        assert_eq!(cv_plus_ranks(3, 0.5).unwrap(), (2, 2));
        assert_eq!(cv_plus_ranks(2, 0.99).unwrap(), (0, 3));
    }

    #[test]
    fn cv_plus_ranks_are_exact_when_the_product_is_whole() {
        // alpha * (n + 1) is 1, 2, 10 and 2 in exact arithmetic.
        assert_eq!(cv_plus_ranks(9, 0.9).unwrap(), (1, 9));
        assert_eq!(cv_plus_ranks(19, 0.9).unwrap(), (2, 18));
        assert_eq!(cv_plus_ranks(99, 0.9).unwrap(), (10, 90));
        assert_eq!(cv_plus_ranks(9, 0.8).unwrap(), (2, 8));
        // floor(alpha (n + 1)) + ceil(level (n + 1)) = n + 1 in exact arithmetic.
        for n in 1..500 {
            for level in [0.5, 0.8, 0.9, 0.95, 0.99] {
                let (lo, hi) = cv_plus_ranks(n, level).unwrap();
                assert_eq!(lo + hi, n + 1, "n = {n}, level = {level}");
            }
        }
    }

    #[test]
    fn cv_plus_guarantee_matches_the_balanced_bound() {
        // n = 100, k = 10, level = 0.9: 1 - 0.2 - min(2*0.9/11, 0.9/11)
        let g = cv_plus_guaranteed_coverage(100, 10, 0.9).unwrap().unwrap();
        let expected = 1.0 - 0.2 - (2.0 * 0.9 / 11.0_f64).min(0.9 / 11.0);
        assert!((g - expected).abs() < 1e-15);
        assert_eq!(cv_plus_guaranteed_coverage(101, 10, 0.9).unwrap(), None);
        // The bound is clipped at zero.
        assert_eq!(cv_plus_guaranteed_coverage(20, 5, 0.5).unwrap(), Some(0.0));
    }

    #[test]
    fn cv_plus_bounds_take_order_statistics_over_folds() {
        // Four rows, two folds. Fold 0 holds rows 0 and 1, fold 1 rows 2 and 3.
        // Two targets; fold predictions for target 0 are 10 (fold 0) and 20 (fold 1).
        let residuals = [1.0, 2.0, 3.0, 4.0];
        let fold_of = [0, 0, 1, 1];
        let preds = array![[10.0, 0.0], [20.0, 0.0]];
        // level 0.5: k_lo = floor(0.5 * 5) = 2, k_hi = ceil(0.5 * 5) = 3.
        let (lo, hi) = cv_plus_bounds(&residuals, &fold_of, preds.view(), 0.5).unwrap();
        // Target 0: values 10-1, 10-2, 20-3, 20-4 = 9, 8, 17, 16 -> sorted 8, 9, 16, 17.
        // Lower rank 2 -> 9. Values 11, 12, 23, 24 -> upper rank 3 -> 23.
        assert_eq!(lo[0], 9.0);
        assert_eq!(hi[0], 23.0);
        // Target 1: all predictions 0. Lower values -1, -2, -3, -4 sorted are
        // -4, -3, -2, -1, so rank 2 gives -3; upper values 1, 2, 3, 4 give rank 3 -> 3.
        assert_eq!(lo[1], -3.0);
        assert_eq!(hi[1], 3.0);
    }

    #[test]
    fn cv_plus_bounds_are_unbounded_when_the_ranks_fall_outside() {
        let residuals = [1.0, 2.0];
        let fold_of = [0, 1];
        let preds = array![[0.0], [0.0]];
        // level 0.99 with n = 2: k_hi = ceil(0.99 * 3) = 3 > 2, upper unbounded.
        // alpha * 3 = 0.03 -> k_lo = 0, lower unbounded too.
        let (lo, hi) = cv_plus_bounds(&residuals, &fold_of, preds.view(), 0.99).unwrap();
        assert_eq!(lo, vec![f64::NEG_INFINITY]);
        assert_eq!(hi, vec![f64::INFINITY]);
    }

    #[test]
    fn cv_plus_bounds_reject_inconsistent_shapes() {
        let preds = array![[0.0], [1.0]];
        assert!(cv_plus_bounds(&[1.0, 2.0], &[0], preds.view(), 0.5).is_err());
        assert!(cv_plus_bounds(&[1.0, 2.0], &[0, 2], preds.view(), 0.5).is_err());
    }

    #[test]
    fn tuned_choice_is_never_iteration_zero() {
        assert_eq!(tuned_mstop_choice(0), 1);
        assert_eq!(tuned_mstop_choice(37), 37);
    }

    #[test]
    fn tuned_grid_starts_at_floor_and_doubles_to_cap() {
        assert_eq!(tuned_mstop_start(50), 1600);
        assert_eq!(tuned_mstop_start(2000), 2000);
        assert_eq!(tuned_mstop_next(200, TUNED_MSTOP_CAP), 400);
        assert_eq!(tuned_mstop_next(8000, TUNED_MSTOP_CAP), TUNED_MSTOP_CAP);
        assert_eq!(
            tuned_mstop_next(TUNED_MSTOP_CAP, TUNED_MSTOP_CAP),
            TUNED_MSTOP_CAP
        );
    }

    #[test]
    fn worst_bin_flags_the_high_noise_third() {
        // x in 0..30. Rows with x >= 20 have wide noise and miss their
        // interval; the others are covered.
        let x: Vec<f64> = (0..30).map(f64::from).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&v| if v >= 20.0 { 10.0 } else { 0.0 })
            .collect();
        let lower = vec![-1.0; 30];
        let upper: Vec<f64> = x
            .iter()
            .map(|&v| if v >= 20.0 { 0.5 } else { 1.0 })
            .collect();
        let w = worst_covariate_bin(&x, &lower, &upper, &y, 0.9, 3).unwrap();
        assert_eq!(w.bin, 2);
        assert_eq!(w.count, 10);
        assert_eq!(w.coverage, 0.0);
        assert!(w.z < -4.0);
    }

    #[test]
    fn worst_bin_is_near_zero_when_coverage_is_even() {
        let x: Vec<f64> = (0..300).map(f64::from).collect();
        // Cover 9 of every 10 rows in each third, in a fixed pattern.
        let y: Vec<f64> = x
            .iter()
            .map(|&v| if (v as usize) % 10 == 0 { 5.0 } else { 0.0 })
            .collect();
        let lower = vec![-1.0; 300];
        let upper: Vec<f64> = x
            .iter()
            .map(|&v| if (v as usize) % 10 == 0 { 0.0 } else { 1.0 })
            .collect();
        let w = worst_covariate_bin(&x, &lower, &upper, &y, 0.9, 3).unwrap();
        assert!(w.z.abs() < 1e-9, "{w:?}");
        assert!((w.coverage - 0.9).abs() < 1e-12);
    }

    #[test]
    fn worst_bin_rejects_one_bin_and_nan() {
        let x = [1.0, 2.0, 3.0];
        let y = [0.0; 3];
        let (lo, hi) = ([-1.0; 3], [1.0; 3]);
        assert!(worst_covariate_bin(&x, &lo, &hi, &y, 0.9, 1).is_err());
        assert!(worst_covariate_bin(&[1.0, f64::NAN, 3.0], &lo, &hi, &y, 0.9, 3).is_err());
    }
}

#[cfg(test)]
mod location_scale_tests {
    use super::*;

    #[test]
    fn normalized_score_set_and_affine_equivariance() {
        let y = [-4.0, 0.0, 4.0, 10.0];
        let mu = [-2.0, -1.0, 2.0, 4.0];
        let sigma = [2.0, 1.0, 4.0, 3.0];
        let scores = location_scale_scores(&y, &mu, &sigma).unwrap();
        assert_eq!(scores, vec![1.0, 1.0, 0.5, 2.0]);
        let (lo, hi) = location_scale_band(&mu, &sigma, 1.0).unwrap();
        for i in 0..y.len() {
            assert_eq!(scores[i] <= 1.0, lo[i] <= y[i] && y[i] <= hi[i]);
        }
        let yy: Vec<_> = y.iter().map(|v| 3.0 * v + 7.0).collect();
        let mm: Vec<_> = mu.iter().map(|v| 3.0 * v + 7.0).collect();
        let ss: Vec<_> = sigma.iter().map(|v| 3.0 * v).collect();
        assert_eq!(location_scale_scores(&yy, &mm, &ss).unwrap(), scores);
        let (ll, hh) = location_scale_band(&mm, &ss, 1.0).unwrap();
        assert_eq!(ll, lo.iter().map(|v| 3.0 * v + 7.0).collect::<Vec<_>>());
        assert_eq!(hh, hi.iter().map(|v| 3.0 * v + 7.0).collect::<Vec<_>>());
    }

    #[test]
    fn invalid_scales_and_unbounded_bands() {
        for scale in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(location_scale_scores(&[1.0], &[0.0], &[scale]).is_err());
            assert!(location_scale_band(&[0.0], &[scale], 1.0).is_err());
        }
        assert!(location_scale_scores(&[], &[0.0], &[1.0]).is_err());
        assert!(location_scale_scores(&[f64::INFINITY], &[0.0], &[1.0]).is_err());
        assert!(location_scale_band(&[0.0], &[1.0], -1.0).is_err());
        assert!(location_scale_band(&[0.0], &[1.0], f64::NAN).is_err());
        assert_eq!(
            location_scale_band(&[2.0], &[3.0], f64::INFINITY).unwrap(),
            (vec![f64::NEG_INFINITY], vec![f64::INFINITY])
        );
    }

    #[test]
    fn scale_target_floors_zero_residuals() {
        let t = scale_target(&[1.0, 2.0, 3.0], &[1.0, 0.0, 3.5]).unwrap();
        assert!((t[0] - 1e-12_f64.ln()).abs() < 1e-12);
        assert!((t[1] - 4.0_f64.ln()).abs() < 1e-12);
        assert!((t[2] - 0.25_f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn rank_scale_columns_uses_value_and_distance_from_median() {
        let n = 41;
        let x: Vec<f64> = (0..n).map(|i| i as f64 - 20.0).collect();
        let target: Vec<f64> = x.iter().map(|v| v.abs()).collect();
        let noise: Vec<f64> = (0..n).map(|i| ((i * 7919) % 13) as f64).collect();
        let columns = vec![noise.clone(), x, noise];
        // x relates to the target only through |x - median|.
        assert_eq!(rank_scale_columns(&columns, &target, 1).unwrap(), vec![1]);
        assert_eq!(
            rank_scale_columns(&columns, &target, 3).unwrap(),
            vec![0, 1, 2]
        );
    }
}

#[cfg(test)]
mod rank_tolerance_tests {
    use super::conformal_quantile;

    /// Scores 1..=n, so the quantile equals the 1-based rank it selects.
    fn rank_at(n: usize, level: f64) -> f64 {
        let scores: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        conformal_quantile(&scores, level).unwrap()
    }

    #[test]
    fn conformal_quantile_counts_near_integer_products_as_integers() {
        assert_eq!(rank_at(24, 0.56), 14.0);
        assert_eq!(rank_at(99, 0.55), 55.0);
        assert_eq!(rank_at(74, 0.68), 51.0);
    }
}
