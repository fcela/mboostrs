//! Count-rank split conformity with a proper-training reference and independent
//! auxiliary uniforms. The supplied mean is unchanged by reference fitting.

use crate::{families::minimize_brent, special, MboostrsError};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

const COUNT_CAP: f64 = 9_007_199_254_740_992.0;
const SEED_CAP: u64 = 9_007_199_254_740_991;
const TAIL_POINTS: usize = 17;

/// Score requested by the automatic standard-model policy. Reference
/// applicability is checked on the actual proper-training fit afterwards.
pub fn default_score(family: &str, y: &[f64]) -> &'static str {
    if matches!(family, "poisson" | "nbinomial")
        && !y.is_empty()
        && y.iter()
            .all(|v| v.is_finite() && *v >= 0.0 && v.fract() == 0.0)
    {
        "count_auto"
    } else {
        "absolute"
    }
}

fn count_response(y: &[f64]) -> Result<(), MboostrsError> {
    if y.iter()
        .any(|v| !v.is_finite() || *v < 0.0 || v.fract() != 0.0 || *v >= COUNT_CAP)
    {
        return Err(MboostrsError::BadResponse(
            "count-rank conformal".into(),
            "count-rank calibration requires finite nonnegative integer counts below 2^53".into(),
        ));
    }
    Ok(())
}

fn means(mean: &[f64], pi: f64) -> Result<(), MboostrsError> {
    if !pi.is_finite() || !(0.0..1.0).contains(&pi) {
        return Err(MboostrsError::Control(
            "zero probability must be finite in [0, 1)".into(),
        ));
    }
    if mean
        .iter()
        .any(|v| !v.is_finite() || *v < 0.0 || *v / (1.0 - pi) >= COUNT_CAP)
    {
        return Err(MboostrsError::Control(
            "count-rank reference requires finite nonnegative means within the exact-count range"
                .into(),
        ));
    }
    Ok(())
}

/// Fit zero mass at a fixed conditional mean. Return `(pi, D)`, where D is
/// the ratio of weighted squared residuals to the reference variance sum.
/// D is an applicability diagnostic, not a variance or specification test.
pub fn fit_reference(
    y: &[f64],
    mean: &[f64],
    weights: &[f64],
    zero_inflated: bool,
) -> Result<(f64, f64), MboostrsError> {
    count_response(y)?;
    means(mean, 0.0)?;
    if y.is_empty()
        || y.len() != mean.len()
        || y.len() != weights.len()
        || weights.iter().any(|w| !w.is_finite() || *w < 0.0)
        || !weights.iter().any(|w| *w > 0.0)
    {
        return Err(MboostrsError::Shape("reference responses, means and nonnegative weights must align and have positive total weight".into()));
    }
    let impossible = y
        .iter()
        .zip(mean)
        .zip(weights)
        .any(|((y, m), w)| *w > 0.0 && *y > 0.0 && *m == 0.0);
    let total = weights.iter().sum::<f64>();
    if !total.is_finite() {
        return Err(MboostrsError::Control(
            "reference total weight must be finite".into(),
        ));
    }
    let largest_active_mean = mean
        .iter()
        .zip(weights)
        .filter(|(_, w)| **w > 0.0)
        .map(|(mu, _)| *mu)
        .fold(0.0, f64::max);
    let upper = (1.0 - largest_active_mean / COUNT_CAP).min(1.0 - 1e-8);
    // Profile NLL difference from pi=0: the factorial cancels. This avoids
    // cancellation between large log-factorial terms at large counts.
    let objective = |pi: f64| -> f64 {
        if pi == 0.0 {
            return 0.0;
        }
        y.iter()
            .zip(mean)
            .zip(weights)
            .filter(|(_, w)| **w > 0.0)
            .map(|((y, mu), w)| {
                let difference = if *y == 0.0 {
                    let a = pi.ln();
                    let b = (-pi).ln_1p() - mu / (1.0 - pi);
                    let largest = a.max(b);
                    -(largest + ((a - largest).exp() + (b - largest).exp()).ln()) - mu
                } else {
                    (y - 1.0) * (-pi).ln_1p() + mu * pi / (1.0 - pi)
                };
                w / total * difference
            })
            .sum()
    };
    let degenerate = mean
        .iter()
        .zip(weights)
        .all(|(m, w)| *w == 0.0 || *m == 0.0);
    let candidate = if !zero_inflated || impossible || degenerate || upper <= 0.0 {
        0.0
    } else {
        minimize_brent(&objective, 0.0, upper, 1e-10, 200)
    };
    let pi = if objective(candidate) < 0.0 {
        candidate
    } else {
        0.0
    };
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for ((y, mu), w) in y.iter().zip(mean).zip(weights) {
        let weight = w / total;
        numerator += weight * (y - mu).powi(2);
        denominator += weight * (mu + pi / (1.0 - pi) * mu * mu);
    }
    let ratio = if denominator > 0.0 {
        numerator / denominator
    } else if numerator == 0.0 {
        0.0
    } else {
        f64::INFINITY
    };
    Ok((pi, ratio))
}

/// Prespecified proper-training applicability guard for automatic scoring.
pub fn automatic_score(dispersion_ratio: f64) -> &'static str {
    if dispersion_ratio > 2.0 {
        "poisson_normalized"
    } else {
        "count_rank"
    }
}

/// Reproducible separate calibration/query streams. The exact seed range is
/// shared with R doubles. Offsets permit chunking without changing a row's U.
pub fn uniforms(
    n: usize,
    seed: Option<u64>,
    stream: u64,
    offset: usize,
) -> Result<(u64, Vec<f64>), MboostrsError> {
    let seed = seed.unwrap_or_else(|| rand::rng().random::<u64>() >> 11);
    if seed > SEED_CAP {
        return Err(MboostrsError::Control(
            "count randomization seed must be an integer in [0, 2^53-1]".into(),
        ));
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(stream);
    rng.set_word_pos((offset as u128) * 2);
    Ok((seed, (0..n).map(|_| rng.random::<f64>()).collect()))
}

fn cdf(k: f64, mu: f64, pi: f64) -> Result<f64, MboostrsError> {
    if k < 0.0 {
        return Ok(0.0);
    }
    let value = special::poisson_cdf(k, mu / (1.0 - pi));
    if !value.converged || !value.value.is_finite() {
        return Err(MboostrsError::Solver(
            "count-rank Poisson CDF did not converge".into(),
        ));
    }
    Ok((pi + (1.0 - pi) * value.value).clamp(0.0, 1.0))
}

/// Continuous quantile of Y+U at an interior probability. Probability0 has
/// lower support0; probability1 removes the upper constraint, even when the
/// reference is degenerate, to retain validity under misspecification.
pub fn jitter_quantile(p: f64, mu: f64, pi: f64) -> Result<f64, MboostrsError> {
    means(&[mu], pi)?;
    if p.is_nan() {
        return Err(MboostrsError::Control(
            "count-rank probability must not be NaN".into(),
        ));
    }
    if p <= 0.0 {
        return Ok(0.0);
    }
    if p >= 1.0 {
        return Ok(f64::INFINITY);
    }
    let quantile = special::poisson_quantile_affine(p, mu / (1.0 - pi), pi, 1.0 - pi);
    if !quantile.converged || !quantile.value.is_finite() {
        return Err(MboostrsError::Solver(
            "count-rank reference quantile did not converge".into(),
        ));
    }
    let k = quantile.value;
    let below = cdf(k - 1.0, mu, pi)?;
    let above = cdf(k, mu, pi)?;
    if above <= below {
        return Err(MboostrsError::Solver(
            "count-rank reference has no resolvable probability mass at its quantile".into(),
        ));
    }
    Ok(k + ((p - below) / (above - below)).clamp(0.0, 1.0))
}

fn left_tail(mu: f64, pi: f64, level: f64) -> Result<f64, MboostrsError> {
    let first = cdf(0.0, mu, pi)?;
    // The zero jitter cell is globally most dense and contains the requested
    // mass. All shortest windows inside it are exactly equivalent; choose0.
    if first >= level && first >= 0.5 {
        return Ok(0.0);
    }
    let mut best = 0.0;
    let mut width = jitter_quantile(level, mu, pi)?;
    for index in 1..TAIL_POINTS {
        let a = (1.0 - level) * (index as f64) / ((TAIL_POINTS - 1) as f64);
        let candidate = jitter_quantile(a + level, mu, pi)? - jitter_quantile(a, mu, pi)?;
        if candidate < width {
            best = a;
            width = candidate;
        }
    }
    Ok(best)
}

fn inputs(mean: &[f64], pi: f64, uniform: &[f64], level: f64) -> Result<(), MboostrsError> {
    crate::conformal::validate_level(level)?;
    means(mean, pi)?;
    if mean.len() != uniform.len()
        || uniform
            .iter()
            .any(|u| !u.is_finite() || !(0.0..1.0).contains(u))
    {
        return Err(MboostrsError::Shape(
            "auxiliary uniforms must align with means and lie in [0, 1)".into(),
        ));
    }
    Ok(())
}

/// Fixed-reference count-rank scores, one per calibration row.
pub fn scores(
    y: &[f64],
    mean: &[f64],
    pi: f64,
    uniform: &[f64],
    level: f64,
) -> Result<Vec<f64>, MboostrsError> {
    inputs(mean, pi, uniform, level)?;
    count_response(y)?;
    if y.len() != mean.len() {
        return Err(MboostrsError::Shape(
            "calibration counts and means must align".into(),
        ));
    }
    y.iter()
        .zip(mean)
        .zip(uniform)
        .map(|((y, mu), u)| {
            let rank = (1.0 - u) * cdf(y - 1.0, *mu, pi)? + u * cdf(*y, *mu, pi)?;
            let a = left_tail(*mu, pi, level)?;
            Ok((a - rank).max(rank - a - level))
        })
        .collect()
}

/// Ordered real endpoints representing the exact accepted nonnegative count
/// set. If its support intersection is empty, use its negative upper endpoint
/// as a singleton. Count-set cardinality is zero and must remain reported.
pub fn band(
    mean: &[f64],
    pi: f64,
    q: f64,
    uniform: &[f64],
    level: f64,
) -> Result<(Vec<f64>, Vec<f64>), MboostrsError> {
    inputs(mean, pi, uniform, level)?;
    if q.is_nan() {
        return Err(MboostrsError::Control(
            "count-rank correction must not be NaN".into(),
        ));
    }
    let mut lower = Vec::with_capacity(mean.len());
    let mut upper = Vec::with_capacity(mean.len());
    for (mu, u) in mean.iter().zip(uniform) {
        if q < -level / 2.0 {
            lower.push(-1.0);
            upper.push(-1.0);
            continue;
        }
        let a = left_tail(*mu, pi, level)?;
        let lo = (jitter_quantile(a - q, *mu, pi)? - u).max(0.0);
        let hi = jitter_quantile(a + level + q, *mu, pi)? - u;
        lower.push(lo.min(hi));
        upper.push(hi);
    }
    Ok((lower, upper))
}

/// Cardinality after intersecting an ordered real interval with nonnegative
/// integer support. This applies equally to every compared method.
pub fn cardinalities(lower: &[f64], upper: &[f64]) -> Result<Vec<f64>, MboostrsError> {
    if lower.len() != upper.len()
        || lower
            .iter()
            .zip(upper)
            .any(|(a, b)| a.is_nan() || b.is_nan() || a > b)
    {
        return Err(MboostrsError::Shape(
            "ordered non-NaN endpoints must align".into(),
        ));
    }
    Ok(lower
        .iter()
        .zip(upper)
        .map(|(a, b)| (b.floor() - a.ceil().max(0.0) + 1.0).max(0.0))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_streams_and_chunks() {
        let (seed, whole) = uniforms(12, Some(53), 1, 0).unwrap();
        assert_eq!(seed, 53);
        assert_eq!(whole[4..9], uniforms(5, Some(seed), 1, 4).unwrap().1);
        assert_ne!(whole, uniforms(12, Some(seed), 0, 0).unwrap().1);
        assert!(uniforms(1, Some(SEED_CAP + 1), 0, 0).is_err());
    }

    #[test]
    fn score_set_equals_integer_interval_intersection() {
        for mu in [0.0, 0.02, 0.4, 1.5, 10.0, 100.0] {
            for pi in [0.0, 0.3] {
                for u in [0.0, 0.07, 0.43, 0.97] {
                    for q in [-0.03, 0.0, 0.02] {
                        let (lo, hi) = band(&[mu], pi, q, &[u], 0.9).unwrap();
                        for y in 0..180 {
                            let s = scores(&[y as f64], &[mu], pi, &[u], 0.9).unwrap()[0];
                            assert_eq!(
                                s <= q,
                                lo[0] <= y as f64 && y as f64 <= hi[0],
                                "mu={mu} pi={pi} u={u} q={q} y={y}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn zero_mean_has_random_empty_sets_and_valid_unbounded_limit() {
        let (lo, hi) = band(&[0.0, 0.0], 0.0, 0.0, &[0.3, 0.95], 0.9).unwrap();
        assert_eq!(cardinalities(&lo, &hi).unwrap(), vec![1.0, 0.0]);
        assert!(hi[1] < 0.0);
        let (lo, hi) = band(&[0.0], 0.0, f64::INFINITY, &[0.2], 0.9).unwrap();
        assert_eq!(lo, vec![0.0]);
        assert_eq!(hi, vec![f64::INFINITY]);
        assert_eq!(
            fit_reference(&[0.0; 4], &[0.0; 4], &[1.0; 4], true).unwrap(),
            (0.0, 0.0)
        );
    }

    #[test]
    fn invalid_inputs_refuse_without_panicking() {
        for y in [-1.0, 0.2, f64::NAN, f64::INFINITY] {
            assert!(scores(&[y], &[1.0], 0.0, &[0.5], 0.9).is_err());
        }
        assert!(fit_reference(&[1.0], &[1.0], &[0.0], true).is_err());
        assert!(band(&[1.0], 1.0, 0.0, &[0.5], 0.9).is_err());
        assert!(band(&[1.0], 0.0, 0.0, &[1.0], 0.9).is_err());
        assert!(band(&[f64::INFINITY], 0.0, 0.0, &[0.5], 0.9).is_err());
    }
}
