//! Fitted column transforms: the numbers behind a transform recipe.
//!
//! A recipe step such as `scale(x)`, `center(x)` or `ecdf(x)` stores a few
//! numbers computed from the training column (a mean and standard deviation,
//! or a grid of quantiles). The functions here compute those numbers and
//! apply a step's forward map, its inverse and its Jacobian, so that Python
//! and R replay a recipe identically.
//!
//! Quantiles and medians follow NumPy's definitions (`np.quantile` with
//! linear interpolation, `np.median`), including its interpolation formula,
//! so the results equal the previous Python implementation exactly.

use crate::MboostrsError;

/// Grid size of the `ecdf` map: at most this many training quantiles.
pub const ECDF_POINTS: usize = 257;

/// Mean and standard deviation with `ddof = 1` (`np.mean`, `np.std(ddof=1)`).
/// The standard deviation is NaN for fewer than two values.
pub fn mean_sd(values: &[f64]) -> (f64, f64) {
    let n = values.len();
    if n == 0 {
        return (f64::NAN, f64::NAN);
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    let sd = if n < 2 {
        f64::NAN
    } else {
        let ss: f64 = values.iter().map(|v| (v - mean) * (v - mean)).sum();
        (ss / (n as f64 - 1.0)).sqrt()
    };
    (mean, sd)
}

/// Median of `values` (`np.median`): NaN when the input is empty or holds a NaN.
pub fn median(values: &[f64]) -> f64 {
    let n = values.len();
    if n == 0 || values.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// `np.linspace(0, 1, m)`.
fn linspace01(m: usize) -> Vec<f64> {
    if m == 1 {
        return vec![0.0];
    }
    let step = 1.0 / (m as f64 - 1.0);
    let mut out: Vec<f64> = (0..m).map(|i| i as f64 * step).collect();
    out[m - 1] = 1.0;
    out
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    let diff = b - a;
    let mut r = a + diff * t;
    if t >= 0.5 {
        r = b - diff * (1.0 - t);
    }
    if t == 1.0 {
        r = b;
    }
    r
}

/// `np.quantile(sorted, p)` with linear interpolation, for ascending `sorted`.
pub fn quantile_sorted(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    let virtual_index = p * (n as f64 - 1.0);
    let prev = virtual_index.floor();
    let gamma = virtual_index - prev;
    let lo = (prev as usize).min(n - 1);
    let hi = (lo + 1).min(n - 1);
    lerp(sorted[lo], sorted[hi], gamma)
}

/// Training quantiles and their probabilities for `ecdf(x)`.
///
/// `ecdf(x)` maps a value to the share of training rows at or below it,
/// interpolated linearly between at most [`ECDF_POINTS`] quantiles. Tied
/// quantiles share one point at their average probability. Non-finite values
/// are dropped first. Returns `(knots, probabilities)`, both ascending.
pub fn ecdf_params(values: &[f64]) -> Result<(Vec<f64>, Vec<f64>), MboostrsError> {
    let mut finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    finite.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if finite.len() < 2 || finite[0] == finite[finite.len() - 1] {
        return Err(MboostrsError::Control(
            "ecdf needs at least two distinct finite training values".into(),
        ));
    }
    let probabilities = linspace01(ECDF_POINTS.min(finite.len()));
    let mut knots: Vec<f64> = Vec::new();
    let mut sums: Vec<f64> = Vec::new();
    let mut counts: Vec<f64> = Vec::new();
    for &p in &probabilities {
        let q = quantile_sorted(&finite, p);
        if knots.last() == Some(&q) {
            *sums.last_mut().unwrap() += p;
            *counts.last_mut().unwrap() += 1.0;
        } else {
            knots.push(q);
            sums.push(p);
            counts.push(1.0);
        }
    }
    let averaged = sums.iter().zip(&counts).map(|(s, c)| s / c).collect();
    Ok((knots, averaged))
}

/// Piecewise-linear interpolation with the end values held (`np.interp`).
/// `xp` is ascending; NaN input gives NaN.
pub fn interp(x: &[f64], xp: &[f64], fp: &[f64]) -> Result<Vec<f64>, MboostrsError> {
    if xp.is_empty() || xp.len() != fp.len() {
        return Err(MboostrsError::Shape(
            "interp needs knots and values of the same non-zero length".into(),
        ));
    }
    let last = xp.len() - 1;
    Ok(x.iter()
        .map(|&v| {
            if v.is_nan() {
                return f64::NAN;
            }
            if v < xp[0] {
                return fp[0];
            }
            if v >= xp[last] {
                return fp[last];
            }
            // Largest j with xp[j] <= v.
            let j = xp.partition_point(|&k| k <= v) - 1;
            let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
            let mut r = slope * (v - xp[j]) + fp[j];
            if r.is_nan() {
                r = slope * (v - xp[j + 1]) + fp[j + 1];
                if r.is_nan() && fp[j] == fp[j + 1] {
                    r = fp[j];
                }
            }
            r
        })
        .collect())
}

fn unknown(kind: &str) -> MboostrsError {
    MboostrsError::Control(format!("'{kind}' is not an invertible response transform"))
}

fn pair(params: &[f64]) -> Result<[f64; 2], MboostrsError> {
    match params {
        [m, s] => Ok([*m, *s]),
        // A `center` step from an older artifact keeps only the mean.
        [m] => Ok([*m, 1.0]),
        _ => Err(MboostrsError::Shape(
            "scale and center parameters are (mean, sd)".into(),
        )),
    }
}

/// Forward map `g(x)` of a stateless or fitted step.
///
/// `kind` is `log`, `log1p`, `sqrt`, `exp`, `scale`/`center` (`params =
/// [mean, sd]`) or `ecdf` (`params` = knots followed by probabilities, equal
/// halves).
pub fn forward(kind: &str, params: &[f64], x: &[f64]) -> Result<Vec<f64>, MboostrsError> {
    Ok(match kind {
        "log" => x.iter().map(|v| v.ln()).collect(),
        "log1p" => x.iter().map(|v| v.ln_1p()).collect(),
        "sqrt" => x.iter().map(|v| v.sqrt()).collect(),
        "exp" => x.iter().map(|v| v.exp()).collect(),
        "scale" | "center" => {
            let [mean, sd] = pair(params)?;
            x.iter().map(|v| (v - mean) / sd).collect()
        }
        "ecdf" => {
            let half = params.len() / 2;
            if params.len() % 2 != 0 || half == 0 {
                return Err(MboostrsError::Shape(
                    "ecdf parameters are the knots followed by the probabilities".into(),
                ));
            }
            interp(x, &params[..half], &params[half..])?
        }
        other => return Err(unknown(other)),
    })
}

/// Inverse map `g^-1(v)` of an invertible response transform.
pub fn inverse(kind: &str, params: &[f64], v: &[f64]) -> Result<Vec<f64>, MboostrsError> {
    Ok(match kind {
        "log" => v.iter().map(|t| t.exp()).collect(),
        "log1p" => v.iter().map(|t| t.exp_m1()).collect(),
        "sqrt" => v.iter().map(|t| t * t).collect(),
        "exp" => v.iter().map(|t| t.ln()).collect(),
        "scale" => {
            let [mean, sd] = pair(params)?;
            v.iter().map(|t| t * sd + mean).collect()
        }
        "center" => {
            let mean = *params
                .first()
                .ok_or_else(|| MboostrsError::Shape("center needs its mean".into()))?;
            v.iter().map(|t| t + mean).collect()
        }
        other => return Err(unknown(other)),
    })
}

/// `|dg/dy|` at original response values `y`.
pub fn jacobian(kind: &str, params: &[f64], y: &[f64]) -> Result<Vec<f64>, MboostrsError> {
    Ok(match kind {
        "log" => y.iter().map(|v| 1.0 / v).collect(),
        "log1p" => y.iter().map(|v| 1.0 / (1.0 + v)).collect(),
        "sqrt" => y.iter().map(|v| 0.5 / v.sqrt()).collect(),
        "exp" => y.iter().map(|v| v.exp()).collect(),
        "scale" => {
            let [_, sd] = pair(params)?;
            vec![1.0 / sd; y.len()]
        }
        "center" => vec![1.0; y.len()],
        other => return Err(unknown(other)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_even_odd_and_nan() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), 2.5);
        assert!(median(&[1.0, f64::NAN]).is_nan());
        assert!(median(&[]).is_nan());
    }

    #[test]
    fn mean_sd_small() {
        let (m, s) = mean_sd(&[1.0, 3.0]);
        assert_eq!(m, 2.0);
        assert!((s - 2f64.sqrt()).abs() < 1e-15);
        assert!(mean_sd(&[5.0]).1.is_nan());
    }

    #[test]
    fn ecdf_ties_average() {
        let (k, p) = ecdf_params(&[1.0, 1.0, 1.0, 2.0]).unwrap();
        assert_eq!(k, vec![1.0, 2.0]);
        assert_eq!(p[1], 1.0);
        assert!(p[0] > 0.0 && p[0] < 1.0);
        assert!(ecdf_params(&[1.0, 1.0]).is_err());
    }

    #[test]
    fn interp_holds_ends() {
        let r = interp(&[-1.0, 0.5, 2.0, f64::NAN], &[0.0, 1.0], &[0.0, 1.0]).unwrap();
        assert_eq!(r[0], 0.0);
        assert_eq!(r[1], 0.5);
        assert_eq!(r[2], 1.0);
        assert!(r[3].is_nan());
    }
}
