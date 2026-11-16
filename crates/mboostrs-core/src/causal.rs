//! Treatment-effect estimation from cross-fitted nuisance predictions.
//!
//! The R-learner of Nie and Wager (2021) estimates the conditional average
//! treatment effect `tau(x)` of a binary treatment `t` by minimizing
//!
//! ```text
//! sum_i ( y_i - m(x_i) - (t_i - e(x_i)) tau(x_i) )^2
//! ```
//!
//! where `m(x) = E[y | x]` and `e(x) = P(t = 1 | x)` are predicted out of
//! fold. Writing `r_i = t_i - e(x_i)`, each term equals
//! `r_i^2 (z_i - tau(x_i))^2` with `z_i = (y_i - m(x_i)) / r_i`, so the
//! R-loss is a weighted least-squares loss for the pseudo-outcome `z` with
//! weights `r^2`. Any weighted Gaussian learner, boosting included, then fits
//! `tau` directly. Rows whose treatment is nearly determined by `x` have
//! `r` near zero: their pseudo-outcome is large, but their weight is small
//! in proportion, and their contribution `r^2 z^2 = (y - m)^2` stays finite.

use crate::MboostrsError;

/// Pseudo-outcomes and weights for fitting `tau` under the R-loss.
#[derive(Debug, Clone)]
pub struct RLossTransform {
    /// `(y - m) / (t - e)`, with `e` clipped to the declared bounds.
    pub pseudo_outcome: Vec<f64>,
    /// `(t - e)^2`, with the same clipped `e`.
    pub weights: Vec<f64>,
    /// How many propensities the clipping bounds moved.
    pub n_clipped: usize,
}

fn check_inputs(
    y: &[f64],
    t: &[f64],
    m_hat: &[f64],
    e_hat: &[f64],
    clip: (f64, f64),
) -> Result<(), MboostrsError> {
    let n = y.len();
    for (name, len) in [
        ("t", t.len()),
        ("m_hat", m_hat.len()),
        ("e_hat", e_hat.len()),
    ] {
        if len != n {
            return Err(MboostrsError::Shape(format!(
                "{name} has {len} rows but y has {n}"
            )));
        }
    }
    if n == 0 {
        return Err(MboostrsError::Shape("no rows".to_string()));
    }
    let (lo, hi) = clip;
    if !(lo >= 0.0 && lo < hi && hi <= 1.0) {
        return Err(MboostrsError::Control(format!(
            "propensity clipping bounds must satisfy 0 <= lower < upper <= 1, got ({lo}, {hi})"
        )));
    }
    for i in 0..n {
        if !(y[i].is_finite() && m_hat[i].is_finite() && e_hat[i].is_finite()) {
            return Err(MboostrsError::BadResponse(
                "rlearner".to_string(),
                format!("row {i}: outcome and nuisance predictions must be finite"),
            ));
        }
        if t[i] != 0.0 && t[i] != 1.0 {
            return Err(MboostrsError::BadResponse(
                "rlearner".to_string(),
                format!("row {i}: treatment must be coded 0 or 1, got {}", t[i]),
            ));
        }
        if !(0.0..=1.0).contains(&e_hat[i]) {
            return Err(MboostrsError::BadResponse(
                "rlearner".to_string(),
                format!("row {i}: propensity {} is outside [0, 1]", e_hat[i]),
            ));
        }
    }
    Ok(())
}

/// The R-loss pseudo-outcomes and weights for cross-fitted nuisances.
///
/// A row whose clipped propensity equals its treatment (possible only with
/// bounds 0 or 1) has zero weight and pseudo-outcome 0: it carries no
/// information about `tau`.
pub fn rloss_transform(
    y: &[f64],
    t: &[f64],
    m_hat: &[f64],
    e_hat: &[f64],
    clip: (f64, f64),
) -> Result<RLossTransform, MboostrsError> {
    let ones = vec![1.0; y.len()];
    let out = rloss_transform_weighted(y, t, m_hat, e_hat, clip, &ones, &ones, &ones, None)?;
    Ok(RLossTransform {
        pseudo_outcome: out.pseudo_outcome,
        weights: out.weights,
        n_clipped: out.n_clipped,
    })
}

/// Pseudo-outcomes and weights for the R-loss with observation weights and
/// missing outcomes.
#[derive(Debug, Clone)]
pub struct WeightedRLoss {
    /// `(y - m) / (t - e)` on observed rows, 0 on unobserved rows.
    pub pseudo_outcome: Vec<f64>,
    /// `w * R * (t - e)^2 / pi`, where `w` is the observation weight, `R` the
    /// outcome-observed indicator and `pi` the response propensity.
    pub weights: Vec<f64>,
    /// How many propensities the `e` clipping bounds moved.
    pub n_clipped: usize,
    /// How many response propensities the `pi` clipping bounds moved.
    pub n_clipped_response: usize,
}

/// The R-loss transform with observation weights `w`, an outcome-observed
/// indicator `R` and a response propensity `pi = P(R = 1 | x, t)`.
///
/// Under outcome missingness at random given `(x, t)`, the inverse-probability
/// weight `R / pi` makes the weighted loss over observed rows an unbiased
/// estimate of the loss over all rows. Rows with `R = 0` carry no `y` and get
/// weight 0 and pseudo-outcome 0; `y` may be NaN there. `pi` is clipped to
/// `response_clip` when one is given and must lie in `(0, 1]` on the rows it
/// is used for.
#[allow(clippy::too_many_arguments)]
pub fn rloss_transform_weighted(
    y: &[f64],
    t: &[f64],
    m_hat: &[f64],
    e_hat: &[f64],
    clip: (f64, f64),
    sample_weight: &[f64],
    observed: &[f64],
    response: &[f64],
    response_clip: Option<(f64, f64)>,
) -> Result<WeightedRLoss, MboostrsError> {
    let n = y.len();
    if n == 0 {
        return Err(MboostrsError::Shape("no rows".to_string()));
    }
    for (name, len) in [
        ("t", t.len()),
        ("m_hat", m_hat.len()),
        ("e_hat", e_hat.len()),
        ("sample_weight", sample_weight.len()),
        ("observed", observed.len()),
        ("response", response.len()),
    ] {
        if len != n {
            return Err(MboostrsError::Shape(format!(
                "{name} has {len} rows but y has {n}"
            )));
        }
    }
    let (lo, hi) = clip;
    if !(lo >= 0.0 && lo < hi && hi <= 1.0) {
        return Err(MboostrsError::Control(format!(
            "propensity clipping bounds must satisfy 0 <= lower < upper <= 1, got ({lo}, {hi})"
        )));
    }
    if let Some((rlo, rhi)) = response_clip {
        if !(rlo > 0.0 && rlo < rhi && rhi <= 1.0) {
            return Err(MboostrsError::Control(format!(
                "response clipping bounds must satisfy 0 < lower < upper <= 1, got ({rlo}, {rhi})"
            )));
        }
    }
    let mut pseudo_outcome = Vec::with_capacity(n);
    let mut weights = Vec::with_capacity(n);
    let mut n_clipped = 0;
    let mut n_clipped_response = 0;
    for i in 0..n {
        let rejected = |what: &str| {
            MboostrsError::BadResponse("rlearner".to_string(), format!("row {i}: {what}"))
        };
        if t[i] != 0.0 && t[i] != 1.0 {
            return Err(rejected(&format!(
                "treatment must be coded 0 or 1, got {}",
                t[i]
            )));
        }
        if !(m_hat[i].is_finite() && e_hat[i].is_finite()) {
            return Err(rejected("nuisance predictions must be finite"));
        }
        if !(0.0..=1.0).contains(&e_hat[i]) {
            return Err(rejected(&format!(
                "propensity {} is outside [0, 1]",
                e_hat[i]
            )));
        }
        if !(sample_weight[i].is_finite() && sample_weight[i] >= 0.0) {
            return Err(rejected("sample weight must be finite and non-negative"));
        }
        if observed[i] != 0.0 && observed[i] != 1.0 {
            return Err(rejected(&format!(
                "observed must be 0 or 1, got {}",
                observed[i]
            )));
        }
        if observed[i] == 0.0 {
            pseudo_outcome.push(0.0);
            weights.push(0.0);
            continue;
        }
        if !y[i].is_finite() {
            return Err(rejected("an observed outcome must be finite"));
        }
        let e = e_hat[i].clamp(lo, hi);
        if e != e_hat[i] {
            n_clipped += 1;
        }
        let mut pi = response[i];
        if let Some((rlo, rhi)) = response_clip {
            let clipped = pi.clamp(rlo, rhi);
            if clipped != pi {
                n_clipped_response += 1;
            }
            pi = clipped;
        }
        if !(pi > 0.0 && pi <= 1.0) {
            return Err(rejected(&format!(
                "response propensity {pi} must lie in (0, 1]"
            )));
        }
        let r = t[i] - e;
        if r == 0.0 {
            pseudo_outcome.push(0.0);
            weights.push(0.0);
        } else {
            pseudo_outcome.push((y[i] - m_hat[i]) / r);
            weights.push(sample_weight[i] * r * r / pi);
        }
    }
    Ok(WeightedRLoss {
        pseudo_outcome,
        weights,
        n_clipped,
        n_clipped_response,
    })
}

/// Pseudo-outcomes clipped to quantile bounds.
#[derive(Debug, Clone)]
pub struct WinsorizedPseudo {
    /// The clipped pseudo-outcomes, same length as the input.
    pub values: Vec<f64>,
    /// Lower bound: the type-7 `tail` quantile of the positive-weight pseudo-outcomes.
    pub lower: f64,
    /// Upper bound: the type-7 `1 - tail` quantile of the positive-weight pseudo-outcomes.
    pub upper: f64,
    /// How many rows the bounds moved.
    pub n_clipped: usize,
}

/// Clip the R-loss pseudo-outcomes of rows with positive weight to their
/// `tail` and `1 - tail` quantiles.
///
/// The R-loss pseudo-outcome `(y - m) / (t - e)` grows without limit as the
/// overlap `|t - e|` shrinks, so a few rows near the propensity bounds can
/// carry pseudo-outcomes that are orders of magnitude larger than the rest.
/// A flexible second stage fits those rows, and its fitted effect goes far
/// outside the range of the outcome. Clipping at fixed quantiles bounds the
/// pseudo-outcome without choosing the rows to drop. The quantiles are
/// unweighted, over rows with `weight > 0`; rows with zero weight keep their
/// value, which is 0 for an unobserved outcome. `tail` must lie in `[0, 0.5)`;
/// `tail = 0` returns the input unchanged.
pub fn winsorize_pseudo_outcome(
    pseudo: &[f64],
    weights: &[f64],
    tail: f64,
) -> Result<WinsorizedPseudo, MboostrsError> {
    if pseudo.len() != weights.len() {
        return Err(MboostrsError::Shape(format!(
            "weights has {} rows but pseudo has {}",
            weights.len(),
            pseudo.len()
        )));
    }
    if !(0.0..0.5).contains(&tail) {
        return Err(MboostrsError::Control(format!(
            "winsorizing tail must lie in [0, 0.5), got {tail}"
        )));
    }
    let mut positive: Vec<f64> = pseudo
        .iter()
        .zip(weights)
        .filter(|(p, w)| **w > 0.0 && p.is_finite())
        .map(|(p, _)| *p)
        .collect();
    if positive.is_empty() {
        return Err(MboostrsError::Shape(
            "no row has a positive weight and a finite pseudo-outcome".to_string(),
        ));
    }
    positive.sort_by(|a, b| a.partial_cmp(b).expect("finite values are ordered"));
    let lower = type7_sorted(&positive, tail);
    let upper = type7_sorted(&positive, 1.0 - tail);
    let mut n_clipped = 0;
    let values = pseudo
        .iter()
        .zip(weights)
        .map(|(&p, &w)| {
            if w > 0.0 && p.is_finite() {
                let clipped = p.clamp(lower, upper);
                if clipped != p {
                    n_clipped += 1;
                }
                clipped
            } else {
                p
            }
        })
        .collect();
    Ok(WinsorizedPseudo {
        values,
        lower,
        upper,
        n_clipped,
    })
}

/// How a second-stage modifier is entered: its scale and the range its spline is
/// fitted over.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModifierScale {
    /// True when the modifier is entered as `ln(x)`: it is positive and heavy on the
    /// right, so an equally spaced basis over the raw scale spends its knots on the tail.
    pub log_scale: bool,
    /// Lower end of the central range, on the entered scale: the `tail` quantile.
    pub lower: f64,
    /// Upper end of the central range, on the entered scale: the `1 - tail` quantile.
    pub upper: f64,
}

/// The scale and central range of one modifier of the second stage.
///
/// A modifier is entered on the log scale when every value is positive and its
/// 99th percentile is at least `skew_ratio` times its median, which is the
/// heavy right tail of an earnings or distance column. The range is the `tail`
/// and `1 - tail` type-7 quantiles of the entered values. Each quantile is taken
/// on the scale the modifier is entered on, so the range is the same whether the
/// caller takes the logs first or not.
pub fn modifier_scale(
    values: &[f64],
    tail: f64,
    skew_ratio: f64,
) -> Result<ModifierScale, MboostrsError> {
    if !(0.0..0.5).contains(&tail) {
        return Err(MboostrsError::Control(format!(
            "trimming tail must lie in [0, 0.5), got {tail}"
        )));
    }
    // NaN fails this check as well as values below 1.
    if skew_ratio.is_nan() || skew_ratio < 1.0 {
        return Err(MboostrsError::Control(format!(
            "skew_ratio must be at least 1, got {skew_ratio}"
        )));
    }
    let mut finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return Err(MboostrsError::Shape(
            "a modifier needs at least one finite value".to_string(),
        ));
    }
    finite.sort_by(|a, b| a.partial_cmp(b).expect("finite values are ordered"));
    let median = type7_sorted(&finite, 0.5);
    let p99 = type7_sorted(&finite, 0.99);
    let positive = finite[0] > 0.0;
    let log_scale = positive && median > 0.0 && p99 / median >= skew_ratio;
    let mut entered: Vec<f64> = if log_scale {
        finite.iter().map(|v| v.ln()).collect()
    } else {
        finite
    };
    entered.sort_by(|a, b| a.partial_cmp(b).expect("finite values are ordered"));
    Ok(ModifierScale {
        log_scale,
        lower: type7_sorted(&entered, tail),
        upper: type7_sorted(&entered, 1.0 - tail),
    })
}

/// Type-7 (linear interpolation, R's default) quantile of sorted finite values.
fn type7_sorted(sorted: &[f64], p: f64) -> f64 {
    let position = p * (sorted.len() as f64 - 1.0);
    let low = position.floor() as usize;
    let high = position.ceil() as usize;
    let fraction = position - low as f64;
    sorted[low] + fraction * (sorted[high] - sorted[low])
}

/// The estimand of an augmented score: the average over all rows, or the
/// average effect among treated rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Estimand {
    /// The average treatment effect over all rows.
    Ate,
    /// The average treatment effect among treated rows.
    Att,
    /// The overlap-weighted average effect: each row counts with weight
    /// `e (1 - e)`, so rows with propensities near 0 or 1 count little.
    Overlap,
}

/// Per-row augmented scores and the weighted average they estimate.
#[derive(Debug, Clone)]
pub struct AugmentedScores {
    /// Row scores whose weighted mean is `estimate`.
    pub scores: Vec<f64>,
    /// Weighted mean of `scores`.
    pub estimate: f64,
    /// `scores - estimate` for the ATE; the ratio-estimator influence for the
    /// ATT, which includes the estimated treated share.
    pub influence: Vec<f64>,
    /// Weighted share of treated rows.
    pub treated_share: f64,
    /// Propensities after clipping.
    pub propensity_used: Vec<f64>,
    /// Response propensities after clipping.
    pub response_used: Vec<f64>,
    /// Rows whose propensity the clipping moved.
    pub n_clipped_propensity: usize,
    /// Rows whose response propensity the clipping moved.
    pub n_clipped_response: usize,
    /// Effective sample size of the inverse-probability weights `w R / pi`.
    pub response_ess: f64,
}

/// Augmented inverse-probability scores for a binary treatment, with
/// observation weights and outcomes missing at random given `(x, t)`.
///
/// For the ATE, row `i` contributes
///
/// ```text
/// psi_i = mu1 - mu0 + R A / (pi e) (Y - mu1) - R (1 - A) / (pi (1 - e)) (Y - mu0)
/// ```
///
/// with `e` the treatment propensity, `pi = P(R = 1 | x, A)` the response
/// propensity and `mu_a` the outcome means fitted on observed rows. The
/// estimate is the weighted mean of `psi`. With `R = 1`, `pi = 1` and unit
/// weights this is the complete-case AIPW score.
///
/// The ATT numerator of row `i` is
///
/// ```text
/// num_i = A (mu1 - mu0) + R A (Y - mu1) / pi - R (1 - A) e/(1 - e) (Y - mu0) / pi
/// ```
///
/// and its score is `num_i / P(A = 1)`, so the estimate targets all treated
/// rows, observed or not. With `R = 1` and `pi = 1` the numerator is the
/// complete-case ATT score `A (Y - mu0) - (1 - A) e/(1 - e) (Y - mu0)`.
///
/// The overlap estimand ([`Estimand::Overlap`]) weights each row by
/// `e (1 - e)`. Its estimate is `sum(w N) / sum(w e (1 - e))`, where `N` is
/// the ATE score without the `1 / e` factors: the treated residual has weight
/// `1 - e` and the control residual weight `e`, both at most 1.
///
/// `clip` bounds `e` and `response_clip` bounds `pi` when given. Without a
/// bound, `e` must lie strictly inside `(0, 1)` and `pi` in `(0, 1]`. Pass
/// `response_clip = None` for a complete-case analysis, where `pi = 1` and
/// clipping would change it. The overlap estimand does not need `clip`:
/// its weights are bounded, so the caller passes `clip = None`.
#[allow(clippy::too_many_arguments)]
pub fn augmented_scores(
    y: &[f64],
    a: &[f64],
    observed: &[f64],
    weights: &[f64],
    e_hat: &[f64],
    pi_hat: &[f64],
    mu0: &[f64],
    mu1: &[f64],
    estimand: Estimand,
    clip: Option<(f64, f64)>,
    response_clip: Option<(f64, f64)>,
) -> Result<AugmentedScores, MboostrsError> {
    let n = y.len();
    if n == 0 {
        return Err(MboostrsError::Shape("no rows".to_string()));
    }
    for (name, len) in [
        ("a", a.len()),
        ("observed", observed.len()),
        ("weights", weights.len()),
        ("e_hat", e_hat.len()),
        ("pi_hat", pi_hat.len()),
        ("mu0", mu0.len()),
        ("mu1", mu1.len()),
    ] {
        if len != n {
            return Err(MboostrsError::Shape(format!(
                "{name} has {len} rows but y has {n}"
            )));
        }
    }
    for (name, bounds) in [("propensity", clip), ("response propensity", response_clip)] {
        if let Some((lo, hi)) = bounds {
            if !(lo > 0.0 && lo < hi && hi < 1.0) {
                return Err(MboostrsError::Control(format!(
                    "{name} clipping bounds must satisfy 0 < lower < upper < 1, got ({lo}, {hi})"
                )));
            }
        }
    }
    let mut e_used = Vec::with_capacity(n);
    let mut pi_used = Vec::with_capacity(n);
    let mut n_clipped_propensity = 0;
    let mut n_clipped_response = 0;
    for i in 0..n {
        let rejected =
            |what: &str| MboostrsError::BadResponse("aipw".to_string(), format!("row {i}: {what}"));
        if a[i] != 0.0 && a[i] != 1.0 {
            return Err(rejected(&format!(
                "treatment must be coded 0 or 1, got {}",
                a[i]
            )));
        }
        if observed[i] != 0.0 && observed[i] != 1.0 {
            return Err(rejected(&format!(
                "observed must be 0 or 1, got {}",
                observed[i]
            )));
        }
        if observed[i] == 1.0 && !y[i].is_finite() {
            return Err(rejected("an observed outcome must be finite"));
        }
        if !(weights[i].is_finite() && weights[i] >= 0.0) {
            return Err(rejected("weight must be finite and non-negative"));
        }
        if !(mu0[i].is_finite() && mu1[i].is_finite() && e_hat[i].is_finite()) {
            return Err(rejected(
                "outcome and propensity predictions must be finite",
            ));
        }
        let mut e = e_hat[i];
        if let Some((lo, hi)) = clip {
            let clipped = e.clamp(lo, hi);
            if clipped != e {
                n_clipped_propensity += 1;
            }
            e = clipped;
        }
        if !(e > 0.0 && e < 1.0) {
            return Err(MboostrsError::BadResponse(
                "aipw".to_string(),
                "cross-fitted propensity reached 0 or 1; positivity is unsupported".to_string(),
            ));
        }
        let mut pi = pi_hat[i];
        if let Some((lo, hi)) = response_clip {
            let clipped = pi.clamp(lo, hi);
            if clipped != pi {
                n_clipped_response += 1;
            }
            pi = clipped;
        }
        if !(pi > 0.0 && pi <= 1.0) {
            return Err(rejected(&format!(
                "response propensity {pi} must lie in (0, 1]"
            )));
        }
        e_used.push(e);
        pi_used.push(pi);
    }
    let total: f64 = weights.iter().sum();
    if total.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return Err(MboostrsError::BadResponse(
            "aipw".to_string(),
            "the weights must have a positive sum".to_string(),
        ));
    }
    let weighted_treated: f64 = (0..n).map(|i| weights[i] * a[i]).sum();
    let treated_share = weighted_treated / total;
    let (scores, estimate, influence) = match estimand {
        Estimand::Ate => {
            let mut scores = Vec::with_capacity(n);
            for i in 0..n {
                let (ai, e, pi) = (a[i], e_used[i], pi_used[i]);
                let augmentation = if observed[i] == 1.0 {
                    ai * (y[i] - mu1[i]) / (pi * e)
                        - (1.0 - ai) * (y[i] - mu0[i]) / (pi * (1.0 - e))
                } else {
                    0.0
                };
                scores.push(mu1[i] - mu0[i] + augmentation);
            }
            let estimate = (0..n).map(|i| weights[i] * scores[i]).sum::<f64>() / total;
            let influence = scores.iter().map(|s| s - estimate).collect();
            (scores, estimate, influence)
        }
        Estimand::Att => {
            if treated_share <= 0.0 || treated_share >= 1.0 {
                return Err(MboostrsError::BadResponse(
                    "aipw".to_string(),
                    "both treatment levels must carry weight".to_string(),
                ));
            }
            // num_i / P(A = 1) is the row score; the ratio tau = E[num] / E[A]
            // is its weighted mean. Each residual is weighted by the response
            // propensity of its own row. An unobserved treated row still
            // contributes A (mu1 - mu0), so the target is all treated rows.
            let numerator: Vec<f64> = (0..n)
                .map(|i| {
                    let (ai, e, pi) = (a[i], e_used[i], pi_used[i]);
                    let (m0, m1) = (mu0[i], mu1[i]);
                    let mut num = ai * (m1 - m0);
                    if observed[i] == 1.0 {
                        let yi = y[i];
                        num += ai * (yi - m1) / pi - (1.0 - ai) * e / (1.0 - e) * (yi - m0) / pi;
                    }
                    num
                })
                .collect();
            let tau = (0..n).map(|i| weights[i] * numerator[i]).sum::<f64>() / weighted_treated;
            let scores: Vec<f64> = numerator.iter().map(|num| num / treated_share).collect();
            let estimate = (0..n).map(|i| weights[i] * scores[i]).sum::<f64>() / total;
            // The denominator E[A] is itself estimated, so its fluctuation
            // enters the influence through -tau (A - E[A]) / E[A].
            let influence = (0..n)
                .map(|i| (numerator[i] - tau * a[i]) / treated_share)
                .collect();
            (scores, estimate, influence)
        }
        Estimand::Overlap => {
            // The overlap weight h = e (1 - e) multiplies the ATE score.
            // Dividing the treated residual by e and multiplying by h leaves
            // (1 - e), and the control residual gives e, so neither term has
            // a propensity in its denominator and no clip is needed.
            let h: Vec<f64> = (0..n).map(|i| e_used[i] * (1.0 - e_used[i])).collect();
            let weighted_h: f64 = (0..n).map(|i| weights[i] * h[i]).sum();
            if weighted_h <= 0.0 {
                return Err(MboostrsError::BadResponse(
                    "aipw".to_string(),
                    "the overlap weights e (1 - e) must carry positive total weight".to_string(),
                ));
            }
            let numerator: Vec<f64> = (0..n)
                .map(|i| {
                    let (ai, e, pi) = (a[i], e_used[i], pi_used[i]);
                    let mut num = h[i] * (mu1[i] - mu0[i]);
                    if observed[i] == 1.0 {
                        let yi = y[i];
                        num +=
                            (ai * (1.0 - e) * (yi - mu1[i]) - (1.0 - ai) * e * (yi - mu0[i])) / pi;
                    }
                    num
                })
                .collect();
            // The estimate is the ratio sum(w num) / sum(w h). Its mean
            // overlap weight is h_bar, so the row scores are num / h_bar and
            // the influence is (num - tau h) / h_bar, as for the ATT ratio.
            let h_bar = weighted_h / total;
            let tau = (0..n).map(|i| weights[i] * numerator[i]).sum::<f64>() / weighted_h;
            let scores: Vec<f64> = numerator.iter().map(|num| num / h_bar).collect();
            let estimate = (0..n).map(|i| weights[i] * scores[i]).sum::<f64>() / total;
            // The ratio's influence: the denominator sum(w h) is estimated,
            // so tau h enters the influence alongside num.
            let influence = (0..n)
                .map(|i| (numerator[i] - tau * h[i]) / h_bar)
                .collect();
            (scores, estimate, influence)
        }
    };
    let response_weights: Vec<f64> = (0..n)
        .map(|i| weights[i] * observed[i] / pi_used[i])
        .collect();
    let response_ess = effective_sample_size(&response_weights)?;
    Ok(AugmentedScores {
        scores,
        estimate,
        influence,
        treated_share,
        propensity_used: e_used,
        response_used: pi_used,
        n_clipped_propensity,
        n_clipped_response,
        response_ess,
    })
}

/// Variance of a weighted mean of influence values.
///
/// Without `groups`, weights are frequency weights: a row with weight `w`
/// stands for `w` identical copies. The variance is then
/// `sum w (IF - mean)^2 / (W (W - 1))` with `W = sum w`, which equals the
/// unit-weight `var(IF, ddof = 1) / n` on unit weights and the same quantity
/// on the expanded data. With `groups` (dense codes `0..G`), rows in a group
/// are treated as one unit and the cluster sandwich
/// `G/(G-1) * sum_g (sum_{i in g} w_i (IF_i - mean))^2 / W^2` is returned.
pub fn influence_variance(
    influence: &[f64],
    weights: &[f64],
    groups: Option<&[usize]>,
) -> Result<f64, MboostrsError> {
    let n = influence.len();
    if n == 0 || weights.len() != n {
        return Err(MboostrsError::Shape(format!(
            "weights has {} rows but influence has {n}",
            weights.len()
        )));
    }
    if let Some(g) = groups {
        if g.len() != n {
            return Err(MboostrsError::Shape(format!(
                "groups has {} rows but influence has {n}",
                g.len()
            )));
        }
    }
    if influence.iter().any(|v| !v.is_finite())
        || weights.iter().any(|w| !(w.is_finite() && *w >= 0.0))
    {
        return Err(MboostrsError::BadResponse(
            "aipw".to_string(),
            "influence values must be finite and weights non-negative".to_string(),
        ));
    }
    let total: f64 = weights.iter().sum();
    if total.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return Err(MboostrsError::BadResponse(
            "aipw".to_string(),
            "the weights must have a positive sum".to_string(),
        ));
    }
    let mean = (0..n).map(|i| weights[i] * influence[i]).sum::<f64>() / total;
    let centred: Vec<f64> = influence.iter().map(|v| v - mean).collect();
    match groups {
        None => {
            if total <= 1.0 {
                return Err(MboostrsError::Shape(
                    "the variance needs a total weight above one".to_string(),
                ));
            }
            let ss: f64 = (0..n).map(|i| weights[i] * centred[i] * centred[i]).sum();
            Ok(ss / (total * (total - 1.0)))
        }
        Some(g) => {
            let n_labels = g.iter().max().map_or(0, |m| m + 1);
            let mut totals = vec![0.0; n_labels];
            let mut seen = vec![false; n_labels];
            for i in 0..n {
                totals[g[i]] += weights[i] * centred[i];
                seen[g[i]] = true;
            }
            let units = seen.iter().filter(|s| **s).count();
            if units < 2 {
                return Err(MboostrsError::Shape(
                    "a cluster variance needs at least two groups".to_string(),
                ));
            }
            let big_g = units as f64;
            let ss: f64 = totals.iter().map(|t| t * t).sum();
            Ok(big_g / (big_g - 1.0) * ss / (total * total))
        }
    }
}

/// Mean and variance of the augmented scores of one subgroup.
#[derive(Debug, Clone, PartialEq)]
pub struct SubgroupSummary {
    /// Rows in the subgroup.
    pub n: usize,
    /// Sum of the observation weights of the subgroup.
    pub weight: f64,
    /// Weighted mean of the subgroup's scores.
    pub mean: f64,
    /// Variance of that mean, computed as [`influence_variance`] computes the
    /// variance of the overall estimate (frequency weights, or the cluster
    /// sandwich when `groups` are given). `None` when the subgroup has total
    /// weight at most one, or fewer than two clusters.
    pub variance: Option<f64>,
    /// Clusters present in the subgroup; `None` without `groups`.
    pub n_clusters: Option<usize>,
}

/// Weighted mean of the augmented scores within each subgroup, with the
/// variance of that mean.
///
/// `codes[i]` is the subgroup of row `i`, in `0..n_levels`; a negative code
/// leaves the row out of every subgroup. `groups` holds dense cluster codes
/// as in [`influence_variance`]. The scores of a subgroup are used as their
/// own influence values, so a subgroup that holds every row reproduces the
/// mean and variance of [`influence_variance`] on the whole sample.
pub fn subgroup_scores(
    scores: &[f64],
    weights: &[f64],
    codes: &[i64],
    n_levels: usize,
    groups: Option<&[usize]>,
) -> Result<Vec<SubgroupSummary>, MboostrsError> {
    let n = scores.len();
    if weights.len() != n || codes.len() != n || groups.is_some_and(|g| g.len() != n) {
        return Err(MboostrsError::Shape(format!(
            "scores, weights, codes and groups must share {n} rows"
        )));
    }
    if codes.iter().any(|&c| c >= n_levels as i64) {
        return Err(MboostrsError::Shape(format!(
            "subgroup codes must be below n_levels = {n_levels}"
        )));
    }
    let mut out = Vec::with_capacity(n_levels);
    for level in 0..n_levels {
        let rows: Vec<usize> = (0..n).filter(|&i| codes[i] == level as i64).collect();
        if rows.is_empty() {
            return Err(MboostrsError::Shape(format!(
                "subgroup {level} holds no rows"
            )));
        }
        let s: Vec<f64> = rows.iter().map(|&i| scores[i]).collect();
        let w: Vec<f64> = rows.iter().map(|&i| weights[i]).collect();
        let g: Option<Vec<usize>> = groups.map(|g| rows.iter().map(|&i| g[i]).collect());
        let total: f64 = w.iter().sum();
        if total.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            return Err(MboostrsError::BadResponse(
                "aipw".to_string(),
                format!("subgroup {level} has no positive weight"),
            ));
        }
        let mean = s.iter().zip(&w).map(|(a, b)| a * b).sum::<f64>() / total;
        let n_clusters = g.as_ref().map(|g| {
            let mut seen = g.clone();
            seen.sort_unstable();
            seen.dedup();
            seen.len()
        });
        let estimable = match n_clusters {
            Some(k) => k >= 2,
            None => total > 1.0,
        };
        let variance = if estimable {
            Some(influence_variance(&s, &w, g.as_deref())?)
        } else {
            None
        };
        out.push(SubgroupSummary {
            n: rows.len(),
            weight: total,
            mean,
            variance,
            n_clusters,
        });
    }
    Ok(out)
}

/// Sample mean of `values` and its IID standard error, `sd(ddof = 1) / sqrt(n)`.
pub fn mean_and_iid_se(values: &[f64]) -> Result<(f64, f64), MboostrsError> {
    let n = values.len();
    if n < 2 {
        return Err(MboostrsError::Shape(
            "a standard error needs at least two scores".to_string(),
        ));
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(MboostrsError::BadResponse(
            "aipw".to_string(),
            "scores must be finite".to_string(),
        ));
    }
    let nf = n as f64;
    let mean = values.iter().sum::<f64>() / nf;
    let ss: f64 = values.iter().map(|v| (v - mean).powi(2)).sum();
    Ok((mean, (ss / (nf * (nf - 1.0))).sqrt()))
}

/// Effective sample size `(sum x)^2 / sum x^2` of non-negative weights `x`.
pub fn effective_sample_size(x: &[f64]) -> Result<f64, MboostrsError> {
    if x.is_empty() || x.iter().any(|v| !(v.is_finite() && *v >= 0.0)) {
        return Err(MboostrsError::BadResponse(
            "aipw".to_string(),
            "effective sample size needs finite, non-negative weights".to_string(),
        ));
    }
    let sum: f64 = x.iter().sum();
    let sum_sq: f64 = x.iter().map(|v| v * v).sum();
    if sum_sq <= 0.0 {
        return Err(MboostrsError::BadResponse(
            "aipw".to_string(),
            "effective sample size needs at least one positive weight".to_string(),
        ));
    }
    Ok(sum * sum / sum_sq)
}

/// Median of a non-empty list of finite values; for an even count, the mean of
/// the two middle values.
fn median_of(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).expect("finite values are ordered"));
    let k = values.len();
    if k % 2 == 1 {
        values[k / 2]
    } else {
        0.5 * (values[k / 2 - 1] + values[k / 2])
    }
}

/// Estimates and standard errors of repeated cross-fits, combined by the median
/// rule of Chernozhukov et al. (2018, "Double/debiased machine learning for
/// treatment and structural parameters", Journal of Econometrics): the estimate is the
/// median over repeats `r` of `estimate_r`, and the variance is the median over
/// repeats of `se_r^2 + (estimate_r - estimate)^2`. The second term adds the
/// spread of the repeats around their median to each repeat's own sampling
/// variance, so the split-to-split variation enters the interval.
#[derive(Debug, Clone, PartialEq)]
pub struct RepeatCombined {
    /// The median estimate of each row (or of the single estimate, length 1).
    pub estimate: Vec<f64>,
    /// The combined standard error of each row.
    pub se: Vec<f64>,
}

/// Combine `repeats` fits of `m` quantities by the median rule.
///
/// `estimates[r][j]` and `ses[r][j]` are the estimate and standard error of
/// quantity `j` from repeat `r`. Every repeat must have the same `m`, the
/// estimates and standard errors must be finite, and each standard error must
/// be non-negative. A single repeat is returned unchanged with its own standard
/// error.
pub fn median_repeat_combine(
    estimates: &[Vec<f64>],
    ses: &[Vec<f64>],
) -> Result<RepeatCombined, MboostrsError> {
    let repeats = estimates.len();
    if repeats == 0 {
        return Err(MboostrsError::Shape(
            "the median rule needs at least one repeat".to_string(),
        ));
    }
    if ses.len() != repeats {
        return Err(MboostrsError::Shape(format!(
            "estimates has {repeats} repeats but ses has {}",
            ses.len()
        )));
    }
    let m = estimates[0].len();
    for r in 0..repeats {
        if estimates[r].len() != m || ses[r].len() != m {
            return Err(MboostrsError::Shape(format!(
                "repeat {r} has {} estimates and {} standard errors; expected {m} of each",
                estimates[r].len(),
                ses[r].len()
            )));
        }
        if estimates[r].iter().any(|v| !v.is_finite())
            || ses[r].iter().any(|v| !(v.is_finite() && *v >= 0.0))
        {
            return Err(MboostrsError::BadResponse(
                "repeats".to_string(),
                "repeated estimates must be finite and their standard errors finite and non-negative"
                    .to_string(),
            ));
        }
    }
    let mut estimate = Vec::with_capacity(m);
    let mut se = Vec::with_capacity(m);
    let mut column = vec![0.0; repeats];
    for j in 0..m {
        for r in 0..repeats {
            column[r] = estimates[r][j];
        }
        let center = median_of(&mut column);
        for r in 0..repeats {
            column[r] = ses[r][j] * ses[r][j] + (estimates[r][j] - center).powi(2);
        }
        estimate.push(center);
        se.push(median_of(&mut column).sqrt());
    }
    Ok(RepeatCombined { estimate, se })
}

/// Percentile interval of `m` quantities from `B` bootstrap draws.
#[derive(Debug, Clone, PartialEq)]
pub struct PercentileInterval {
    /// Lower limit of each quantity: the type-7 `(1 - level) / 2` quantile of its draws.
    pub lower: Vec<f64>,
    /// Upper limit: the type-7 `(1 + level) / 2` quantile of its draws.
    pub upper: Vec<f64>,
    /// Standard deviation of each quantity over the draws (`ddof = 1`).
    pub se: Vec<f64>,
}

/// Percentile interval at `level` from bootstrap draws.
///
/// `draws[b][j]` is quantity `j` from bootstrap draw `b`. At least two draws
/// are needed, every value must be finite, and `level` must lie in `(0, 1)`.
/// The quantiles are type-7 (linear interpolation), as in R's `quantile()`.
pub fn bootstrap_percentile_interval(
    draws: &[Vec<f64>],
    level: f64,
) -> Result<PercentileInterval, MboostrsError> {
    let b = draws.len();
    if b < 2 {
        return Err(MboostrsError::Shape(
            "a percentile interval needs at least two bootstrap draws".to_string(),
        ));
    }
    if !(level > 0.0 && level < 1.0) {
        return Err(MboostrsError::Control(format!(
            "interval level must lie in (0, 1), got {level}"
        )));
    }
    let m = draws[0].len();
    if draws.iter().any(|d| d.len() != m) {
        return Err(MboostrsError::Shape(
            "every bootstrap draw must hold the same number of quantities".to_string(),
        ));
    }
    if draws.iter().any(|d| d.iter().any(|v| !v.is_finite())) {
        return Err(MboostrsError::BadResponse(
            "bootstrap".to_string(),
            "bootstrap draws must be finite".to_string(),
        ));
    }
    let lo_p = 0.5 * (1.0 - level);
    let hi_p = 1.0 - lo_p;
    let mut lower = Vec::with_capacity(m);
    let mut upper = Vec::with_capacity(m);
    let mut se = Vec::with_capacity(m);
    let mut column = vec![0.0; b];
    for j in 0..m {
        for (k, draw) in draws.iter().enumerate() {
            column[k] = draw[j];
        }
        let mean = column.iter().sum::<f64>() / b as f64;
        let ss: f64 = column.iter().map(|v| (v - mean).powi(2)).sum();
        se.push((ss / (b as f64 - 1.0)).sqrt());
        column.sort_by(|x, y| x.partial_cmp(y).expect("finite values are ordered"));
        lower.push(type7_sorted(&column, lo_p));
        upper.push(type7_sorted(&column, hi_p));
    }
    Ok(PercentileInterval { lower, upper, se })
}

/// Weighted mean binomial deviance of predicted probabilities:
/// `sum w (-2) [t log p + (1 - t) log(1 - p)] / sum w`, with `t` the 0/1
/// treatment and `p` the predicted probability.
///
/// The probabilities are held inside `[1e-12, 1 - 1e-12]` so that a
/// prediction at 0 or 1 that the outcome contradicts gives a large finite
/// deviance rather than infinity. Lower is better; this is the score used to
/// choose between propensity models fitted on the same folds.
pub fn binomial_deviance(t: &[f64], p: &[f64], w: &[f64]) -> Result<f64, MboostrsError> {
    let n = t.len();
    if n == 0 {
        return Err(MboostrsError::Shape("no rows".to_string()));
    }
    if p.len() != n || w.len() != n {
        return Err(MboostrsError::Shape(format!(
            "treatment has {n} rows, predictions {} and weights {}",
            p.len(),
            w.len()
        )));
    }
    const EPS: f64 = 1e-12;
    let mut total = 0.0;
    let mut weight_sum = 0.0;
    for i in 0..n {
        if !(t[i] == 0.0 || t[i] == 1.0) {
            return Err(MboostrsError::BadResponse(
                "binomial deviance".to_string(),
                format!("row {i}: treatment must be coded 0 or 1, got {}", t[i]),
            ));
        }
        if !(p[i].is_finite() && (0.0..=1.0).contains(&p[i])) {
            return Err(MboostrsError::BadResponse(
                "binomial deviance".to_string(),
                format!("row {i}: predicted probability {} is outside [0, 1]", p[i]),
            ));
        }
        if !(w[i].is_finite() && w[i] >= 0.0) {
            return Err(MboostrsError::BadResponse(
                "binomial deviance".to_string(),
                format!("row {i}: weight {} must be finite and non-negative", w[i]),
            ));
        }
        let q = p[i].clamp(EPS, 1.0 - EPS);
        let log_lik = if t[i] == 1.0 { q.ln() } else { (1.0 - q).ln() };
        total += -2.0 * w[i] * log_lik;
        weight_sum += w[i];
    }
    if weight_sum <= 0.0 {
        return Err(MboostrsError::Control(
            "binomial deviance needs positive total weight".to_string(),
        ));
    }
    Ok(total / weight_sum)
}

/// Weighted mean square of the residuals `y - fitted` of a log-scale outcome
/// model: `sum w (y - f)^2 / sum w`. It is the variance `s2` of the log outcome
/// in a two-part model.
pub fn log_residual_variance(
    y: &[f64],
    fitted: &[f64],
    weights: &[f64],
) -> Result<f64, MboostrsError> {
    let n = y.len();
    if n == 0 || fitted.len() != n || weights.len() != n {
        return Err(MboostrsError::BadResponse(
            "aipw".to_string(),
            "log residual variance needs equal, non-empty outcome, fitted and weight vectors"
                .to_string(),
        ));
    }
    if (0..n).any(|i| !(y[i].is_finite() && fitted[i].is_finite() && weights[i] >= 0.0)) {
        return Err(MboostrsError::BadResponse(
            "aipw".to_string(),
            "log residual variance needs finite outcomes and fitted values and non-negative weights"
                .to_string(),
        ));
    }
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return Err(MboostrsError::BadResponse(
            "aipw".to_string(),
            "log residual variance needs at least one positive weight".to_string(),
        ));
    }
    let ss: f64 = (0..n)
        .map(|i| weights[i] * (y[i] - fitted[i]) * (y[i] - fitted[i]))
        .sum();
    Ok(ss / total)
}

/// Mean of a positive, zero-inflated outcome from a two-part model:
/// `p * exp(m + s2 / 2)`, where `p` is the probability of a non-zero outcome,
/// `m` the mean of the log outcome among the non-zero rows and `s2` its
/// variance. `exp(m + s2 / 2)` is the mean of a log-normal with those
/// parameters, so the product is the unconditional mean of the outcome.
pub fn two_part_mean(
    employed: &[f64],
    log_mean: &[f64],
    log_variance: f64,
) -> Result<Vec<f64>, MboostrsError> {
    if employed.is_empty() || log_mean.len() != employed.len() {
        return Err(MboostrsError::BadResponse(
            "aipw".to_string(),
            "two-part mean needs non-empty probability and log-mean vectors of equal length"
                .to_string(),
        ));
    }
    if !(log_variance.is_finite() && log_variance >= 0.0) {
        return Err(MboostrsError::BadResponse(
            "aipw".to_string(),
            "two-part mean needs a finite, non-negative log variance".to_string(),
        ));
    }
    if employed.iter().any(|p| !p.is_finite()) || log_mean.iter().any(|m| !m.is_finite()) {
        return Err(MboostrsError::BadResponse(
            "aipw".to_string(),
            "two-part mean needs finite probabilities and log means".to_string(),
        ));
    }
    let half = 0.5 * log_variance;
    Ok(employed
        .iter()
        .zip(log_mean)
        .map(|(p, m)| p * (m + half).exp())
        .collect())
}

/// Mean R-loss of an effect prediction `tau_hat`, with `e` clipped as in
/// [`rloss_transform`]. Lower is better; it compares effect models fitted on
/// other rows, since the nuisances do not depend on `tau`.
pub fn rloss(
    y: &[f64],
    t: &[f64],
    m_hat: &[f64],
    e_hat: &[f64],
    tau_hat: &[f64],
    clip: (f64, f64),
) -> Result<f64, MboostrsError> {
    check_inputs(y, t, m_hat, e_hat, clip)?;
    if tau_hat.len() != y.len() {
        return Err(MboostrsError::Shape(format!(
            "tau_hat has {} rows but y has {}",
            tau_hat.len(),
            y.len()
        )));
    }
    let mut total = 0.0;
    for i in 0..y.len() {
        if !tau_hat[i].is_finite() {
            return Err(MboostrsError::BadResponse(
                "rlearner".to_string(),
                format!("row {i}: effect prediction must be finite"),
            ));
        }
        let e = e_hat[i].clamp(clip.0, clip.1);
        let residual = y[i] - m_hat[i] - (t[i] - e) * tau_hat[i];
        total += residual * residual;
    }
    Ok(total / y.len() as f64)
}

/// The generalised covariance measure test of `X` independent of `Y` given
/// `Z` (Shah and Peters, 2020), from out-of-fold residuals of `X` given `Z`
/// and of `Y` given `Z`.
#[derive(Debug, Clone)]
pub struct GcmTest {
    /// Normalised mean of the residual products; approximately standard
    /// normal when `X` and `Y` are independent given `Z` and both
    /// regressions are accurate enough.
    pub statistic: f64,
    /// Two-sided p-value from the standard normal.
    pub p_value: f64,
    /// Correlation of the two residual vectors: the partial association
    /// after `Z`.
    pub residual_correlation: f64,
    /// Rows used.
    pub n: usize,
}

/// The GCM statistic `T = sqrt(n) mean(R) / sd(R)` with `R = r_x * r_y`.
///
/// Its validity needs the product of the two regressions' errors to shrink
/// faster than `1/sqrt(n)`, which flexible learners fitted out of fold can
/// reach; it does not need either regression to be linear.
pub fn gcm_test(r_x: &[f64], r_y: &[f64]) -> Result<GcmTest, MboostrsError> {
    let n = r_x.len();
    if r_y.len() != n {
        return Err(MboostrsError::Shape(format!(
            "r_y has {} rows but r_x has {n}",
            r_y.len()
        )));
    }
    if n < 3 {
        return Err(MboostrsError::Shape(
            "the test needs at least three rows".to_string(),
        ));
    }
    if r_x.iter().chain(r_y).any(|v| !v.is_finite()) {
        return Err(MboostrsError::BadResponse(
            "gcm".to_string(),
            "residuals must be finite".to_string(),
        ));
    }
    let nf = n as f64;
    let prod: Vec<f64> = r_x.iter().zip(r_y).map(|(a, b)| a * b).collect();
    let mean = prod.iter().sum::<f64>() / nf;
    let var = prod.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / nf;
    let statistic = if var > 0.0 {
        nf.sqrt() * mean / var.sqrt()
    } else {
        0.0
    };
    let p_value = libm::erfc(statistic.abs() / std::f64::consts::SQRT_2);
    let mx = r_x.iter().sum::<f64>() / nf;
    let my = r_y.iter().sum::<f64>() / nf;
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (a, b) in r_x.iter().zip(r_y) {
        sxy += (a - mx) * (b - my);
        sxx += (a - mx).powi(2);
        syy += (b - my).powi(2);
    }
    let residual_correlation = if sxx > 0.0 && syy > 0.0 {
        sxy / (sxx * syy).sqrt()
    } else {
        0.0
    };
    Ok(GcmTest {
        statistic,
        p_value,
        residual_correlation,
        n,
    })
}

/// The multivariate GCM test: `r_x` holds `k` residual columns (row-major
/// `(n, k)`), the residuals of `k` functions of `X` (for example `x` and
/// hinge functions at its quartiles) given `Z`. With `R_i = r_x[i, :] * r_y[i]`,
/// `S = n m' C^-1 m` is approximately chi-squared with `k` degrees of
/// freedom under conditional independence, where `m` and `C` are the mean
/// and covariance of `R`. Several functions of `X` detect a dependence the
/// single covariance misses, such as `Y` depending on `X^2` for a symmetric
/// `X`.
pub fn gcm_test_multi(r_x: &[f64], k: usize, r_y: &[f64]) -> Result<GcmTest, MboostrsError> {
    let n = r_y.len();
    if k == 0 || r_x.len() != n * k {
        return Err(MboostrsError::Shape(format!(
            "r_x has {} entries, expected {n} rows times {k} columns",
            r_x.len()
        )));
    }
    if n <= k + 2 {
        return Err(MboostrsError::Shape(
            "too few rows for the test".to_string(),
        ));
    }
    if r_x.iter().chain(r_y).any(|v| !v.is_finite()) {
        return Err(MboostrsError::BadResponse(
            "gcm".to_string(),
            "residuals must be finite".to_string(),
        ));
    }
    let nf = n as f64;
    let mut prod = ndarray::Array2::<f64>::zeros((n, k));
    for i in 0..n {
        for j in 0..k {
            prod[[i, j]] = r_x[i * k + j] * r_y[i];
        }
    }
    let mean = prod.mean_axis(ndarray::Axis(0)).expect("n > 0");
    let centred = &prod - &mean;
    let cov = centred.t().dot(&centred) / nf;
    // Solve C a = m by a Cholesky factor, with a small ridge if C is
    // singular (two functions of X whose residual products coincide).
    let scale = (0..k)
        .map(|j| cov[[j, j]])
        .fold(0.0f64, f64::max)
        .max(1e-300);
    let mut solved = None;
    for ridge in [0.0, 1e-10, 1e-8, 1e-6] {
        let mut c = cov.clone();
        for j in 0..k {
            c[[j, j]] += ridge * scale;
        }
        if let Some(a) = cholesky_solve(&c, mean.view()) {
            solved = Some(a);
            break;
        }
    }
    let a = solved.ok_or_else(|| {
        MboostrsError::Solver("the residual-product covariance is singular".to_string())
    })?;
    let statistic = (nf * mean.dot(&a)).max(0.0);
    let p_value = chi_square_sf(statistic, k as f64);
    // The first column is X itself: report its partial correlation.
    let first: Vec<f64> = (0..n).map(|i| r_x[i * k]).collect();
    let single = gcm_test(&first, r_y)?;
    Ok(GcmTest {
        statistic,
        p_value,
        residual_correlation: single.residual_correlation,
        n,
    })
}

/// A column of a design that is an exact linear combination of the columns
/// before it.
#[derive(Debug, Clone, PartialEq)]
pub struct DependentColumn {
    /// Position of the column in the design, zero-based.
    pub column: usize,
    /// The earlier columns that reproduce it, as `(position, coefficient)`
    /// pairs over the columns that were kept (not over dropped ones). Empty
    /// for an all-zero column.
    pub depends_on: Vec<(usize, f64)>,
}

/// Columns of an `n x p` design that lie in the span of the columns before
/// them, found by Gram-Schmidt orthogonalization with a second pass.
///
/// `columns` holds the design column by column (column `j` is
/// `columns[j * n..(j + 1) * n]`). A column is dependent when its component
/// orthogonal to the kept columns is at most `tol` times its own norm, so `tol`
/// is a relative tolerance: `1e-10` flags a column that reproduces an earlier
/// one to about ten significant digits. Each dependent column reports the
/// coefficients of the kept columns that reproduce it, which is what a caller
/// needs to name the redundancy. Dependent columns are not added to the basis,
/// so a column dependent on a dropped one is reported against the kept ones.
pub fn linear_dependence(
    columns: &[f64],
    n: usize,
    p: usize,
    tol: f64,
) -> Result<Vec<DependentColumn>, MboostrsError> {
    if columns.len() != n * p {
        return Err(MboostrsError::Shape(format!(
            "linear_dependence expects {} values for an {n} x {p} design, got {}",
            n * p,
            columns.len()
        )));
    }
    if !(tol.is_finite() && tol > 0.0) {
        return Err(MboostrsError::Shape(format!(
            "tolerance must be a positive finite number, got {tol}"
        )));
    }
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    // Orthonormal basis of the kept columns, and for each basis vector q_k the
    // coefficients t[k][m] that express it through the kept original columns:
    // q_k = sum_m t[k][m] * column(kept[m]).
    let mut basis: Vec<Vec<f64>> = Vec::new();
    let mut kept: Vec<usize> = Vec::new();
    let mut t: Vec<Vec<f64>> = Vec::new();
    let mut out = Vec::new();
    for j in 0..p {
        let c = &columns[j * n..(j + 1) * n];
        let norm_c = dot(c, c).sqrt();
        let mut r = c.to_vec();
        let mut a = vec![0.0f64; basis.len()];
        for _pass in 0..2 {
            for (k, q) in basis.iter().enumerate() {
                let d = dot(q, &r);
                a[k] += d;
                for (ri, qi) in r.iter_mut().zip(q) {
                    *ri -= d * qi;
                }
            }
        }
        let residual_norm = dot(&r, &r).sqrt();
        let dependent = norm_c == 0.0 || residual_norm <= tol * norm_c;
        // c_j = sum_k a_k q_k + r, and q_k = sum_m t[k][m] c_kept[m].
        let coefficient_on_kept =
            |m: usize| -> f64 { a.iter().enumerate().map(|(k, ak)| ak * t[k][m]).sum() };
        if dependent {
            let depends_on = (0..kept.len())
                .map(|m| (kept[m], coefficient_on_kept(m)))
                .filter(|(_, coef)| *coef != 0.0)
                .collect();
            out.push(DependentColumn {
                column: j,
                depends_on,
            });
            continue;
        }
        let m_new = kept.len();
        let mut row: Vec<f64> = (0..m_new)
            .map(|m| -coefficient_on_kept(m) / residual_norm)
            .collect();
        row.push(1.0 / residual_norm);
        for earlier in t.iter_mut() {
            earlier.push(0.0);
        }
        t.push(row);
        basis.push(r.iter().map(|v| v / residual_norm).collect());
        kept.push(j);
    }
    Ok(out)
}

fn cholesky_solve(
    c: &ndarray::Array2<f64>,
    b: ndarray::ArrayView1<f64>,
) -> Option<ndarray::Array1<f64>> {
    let k = c.nrows();
    let mut l = ndarray::Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..=i {
            let mut sum = c[[i, j]];
            for m in 0..j {
                sum -= l[[i, m]] * l[[j, m]];
            }
            if i == j {
                if sum <= 0.0 || !sum.is_finite() {
                    return None;
                }
                l[[i, i]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }
    let mut y = ndarray::Array1::<f64>::zeros(k);
    for i in 0..k {
        let mut sum = b[i];
        for m in 0..i {
            sum -= l[[i, m]] * y[m];
        }
        y[i] = sum / l[[i, i]];
    }
    let mut x = ndarray::Array1::<f64>::zeros(k);
    for i in (0..k).rev() {
        let mut sum = y[i];
        for m in i + 1..k {
            sum -= l[[m, i]] * x[m];
        }
        x[i] = sum / l[[i, i]];
    }
    Some(x)
}

/// Upper tail of the chi-squared distribution: the regularized upper
/// incomplete gamma `Q(k/2, x/2)` (series below `a + 1`, continued
/// fraction above, as in Press et al., Numerical Recipes 6.2).
pub fn chi_square_sf(x: f64, k: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    let (a, x) = (k / 2.0, x / 2.0);
    let ln_gamma_a = libm::lgamma(a);
    if x < a + 1.0 {
        let (mut sum, mut term, mut ap) = (1.0 / a, 1.0 / a, a);
        for _ in 0..1000 {
            ap += 1.0;
            term *= x / ap;
            sum += term;
            if term.abs() < sum.abs() * 1e-15 {
                break;
            }
        }
        let p = sum * (-x + a * x.ln() - ln_gamma_a).exp();
        (1.0 - p).clamp(0.0, 1.0)
    } else {
        let tiny = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..1000 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-15 {
                break;
            }
        }
        (h * (-x + a * x.ln() - ln_gamma_a).exp()).clamp(0.0, 1.0)
    }
}

/// The partially linear instrumental-variable estimate from cross-fitted
/// residuals (`iv_ratio`).
#[derive(Debug, Clone)]
pub struct IvEstimate {
    /// `sum(z y) / sum(z d)` over the residuals: the ratio of the residual
    /// covariances of the instrument with the outcome and with the treatment.
    pub estimate: f64,
    /// Sandwich variance of `estimate`: `sum(psi^2) / n^2`, where `psi` is
    /// the influence function below.
    pub variance: f64,
    /// Influence function of the estimate, one value per row:
    /// `z_i (y_i - estimate * d_i) / mean(z d)`. Its sum is zero.
    pub influence: Vec<f64>,
    /// Squared correlation of the instrument residual and the treatment
    /// residual: the share of the treatment residual's variation that the
    /// instrument residual explains.
    pub first_stage_r2: f64,
    /// First-stage F statistic `n R2 / (1 - R2)`, the one-instrument form of
    /// the partial F. It is infinite when `R2` is one.
    pub first_stage_f: f64,
    /// Rows used.
    pub n: usize,
}

/// Partially linear IV ratio from out-of-fold residuals (DML-IV): `y_res` is
/// the outcome minus its prediction from the covariates, `d_res` the treatment
/// minus its prediction, and `z_res` the instrument minus its prediction.
///
/// The estimate is `sum(z y) / sum(z d)` over the residuals. Writing
/// `eps_i = y_i - estimate * d_i`, the influence function is
/// `z_i eps_i / mean(z d)`, and the variance is the sandwich
/// `sum((z eps)^2) / (sum(z d))^2`. The first stage is the squared
/// correlation of `z_res` and `d_res`.
pub fn iv_ratio(y_res: &[f64], d_res: &[f64], z_res: &[f64]) -> Result<IvEstimate, MboostrsError> {
    let ones = vec![1.0; y_res.len()];
    iv_ratio_weighted(y_res, d_res, z_res, &ones)
}

/// The DML-IV ratio with inverse response weights (`iv_ratio_weighted`).
///
/// `weights[i]` is `R_i / pi_i` for a row whose outcome is observed (`R_i = 1`)
/// and zero for a row whose outcome is missing, where `pi_i` is the
/// cross-fitted probability that the outcome is observed. The estimate is
/// `sum(w z y) / sum(w z d)`, the weighted moment of the residuals, and the
/// influence function is `w_i z_i (y_i - estimate * d_i) / mean(w z d)`. A row
/// with weight zero contributes nothing whatever its residuals are, so the
/// caller may pass any finite value for a missing outcome.
///
/// The first stage uses the rows with positive weight and their unweighted
/// residuals: `n_obs` of them, with `R2` the squared correlation of `z_res` and
/// `d_res` over those rows and `F = n_obs R2 / (1 - R2)`. With all weights one,
/// this is `iv_ratio`.
pub fn iv_ratio_weighted(
    y_res: &[f64],
    d_res: &[f64],
    z_res: &[f64],
    weights: &[f64],
) -> Result<IvEstimate, MboostrsError> {
    let n = y_res.len();
    if d_res.len() != n || z_res.len() != n || weights.len() != n {
        return Err(MboostrsError::Shape(format!(
            "residual vectors differ in length: y {n}, d {}, z {}, weights {}",
            d_res.len(),
            z_res.len(),
            weights.len()
        )));
    }
    if n < 3 {
        return Err(MboostrsError::Shape(
            "the IV ratio needs at least three rows".to_string(),
        ));
    }
    if y_res
        .iter()
        .chain(d_res)
        .chain(z_res)
        .any(|v| !v.is_finite())
    {
        return Err(MboostrsError::BadResponse(
            "iv".to_string(),
            "residuals must be finite".to_string(),
        ));
    }
    if weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
        return Err(MboostrsError::BadResponse(
            "iv".to_string(),
            "weights must be finite and non-negative".to_string(),
        ));
    }
    let observed: Vec<usize> = (0..n).filter(|&i| weights[i] > 0.0).collect();
    let n_obs = observed.len();
    if n_obs < 3 {
        return Err(MboostrsError::Shape(
            "the IV ratio needs at least three rows with a positive weight".to_string(),
        ));
    }
    let nf = n as f64;
    let zd: f64 = (0..n).map(|i| weights[i] * z_res[i] * d_res[i]).sum();
    let zy: f64 = (0..n).map(|i| weights[i] * z_res[i] * y_res[i]).sum();
    // First-stage moments over the rows that carry weight.
    let zz: f64 = observed.iter().map(|&i| z_res[i] * z_res[i]).sum();
    let dd: f64 = observed.iter().map(|&i| d_res[i] * d_res[i]).sum();
    let zd_obs: f64 = observed.iter().map(|&i| z_res[i] * d_res[i]).sum();
    if zz <= 0.0 || dd <= 0.0 || zd.abs() <= 1e-12 * (zz * dd).sqrt() {
        return Err(MboostrsError::Solver(
            "the residualized instrument has no covariance with the residualized treatment; \
             the instrument does not move the treatment after the covariates, so the IV \
             ratio is undefined"
                .to_string(),
        ));
    }
    let estimate = zy / zd;
    let scale = zd / nf;
    let influence: Vec<f64> = (0..n)
        .map(|i| weights[i] * z_res[i] * (y_res[i] - estimate * d_res[i]) / scale)
        .collect();
    let variance = influence.iter().map(|v| v * v).sum::<f64>() / (nf * nf);
    let r2 = (zd_obs * zd_obs / (zz * dd)).clamp(0.0, 1.0);
    let first_stage_f = if r2 < 1.0 {
        n_obs as f64 * r2 / (1.0 - r2)
    } else {
        f64::INFINITY
    };
    Ok(IvEstimate {
        estimate,
        variance,
        influence,
        first_stage_r2: r2,
        first_stage_f,
        n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subgroup_scores_full_group_matches_influence_variance() {
        let s = [1.0, 2.0, 4.0, 3.0, 0.5, 2.5];
        let w = [1.0, 2.0, 1.0, 1.0, 3.0, 1.0];
        let out = subgroup_scores(&s, &w, &[0; 6], 1, None).unwrap();
        let total: f64 = w.iter().sum();
        let mean: f64 = s.iter().zip(&w).map(|(a, b)| a * b).sum::<f64>() / total;
        assert!((out[0].mean - mean).abs() < 1e-14);
        let v = influence_variance(&s, &w, None).unwrap();
        assert!((out[0].variance.unwrap() - v).abs() < 1e-14);
        let g = [0usize, 0, 1, 1, 2, 2];
        let out = subgroup_scores(&s, &w, &[0; 6], 1, Some(&g)).unwrap();
        let v = influence_variance(&s, &w, Some(&g)).unwrap();
        assert!((out[0].variance.unwrap() - v).abs() < 1e-14);
        assert_eq!(out[0].n_clusters, Some(3));
    }

    #[test]
    fn subgroup_scores_excludes_negative_codes_and_flags_small_groups() {
        let s = [1.0, 2.0, 4.0, 3.0];
        let w = [1.0; 4];
        let out = subgroup_scores(&s, &w, &[0, 0, 1, -1], 2, None).unwrap();
        assert_eq!(out[0].n, 2);
        assert!((out[0].mean - 1.5).abs() < 1e-14);
        assert!(out[1].variance.is_none());
        assert!(subgroup_scores(&s, &w, &[0, 0, 0, 0], 2, None).is_err());
    }

    #[test]
    fn iv_ratio_is_exact_when_the_outcome_is_twice_the_treatment_residual() {
        // d = z / 2 and y = 2 d, so sum(z y) / sum(z d) = 2 and every residual
        // of y - 2 d is zero: the influence function and variance vanish.
        let z = [1.0, -1.0, 2.0, -2.0, 0.5];
        let d: Vec<f64> = z.iter().map(|v| v / 2.0).collect();
        let y: Vec<f64> = d.iter().map(|v| 2.0 * v).collect();
        let out = iv_ratio(&y, &d, &z).unwrap();
        assert!((out.estimate - 2.0).abs() < 1e-12);
        assert!(out.variance.abs() < 1e-24);
        assert!((out.first_stage_r2 - 1.0).abs() < 1e-12);
        assert!(out.first_stage_f.is_infinite());
    }

    #[test]
    fn iv_ratio_influence_sums_to_zero_and_refuses_an_irrelevant_instrument() {
        let z = [0.3, -1.2, 0.8, 1.5, -0.4, 0.9];
        let d = [0.1, -0.7, 0.5, 0.9, -0.2, 0.4];
        let y = [0.5, -0.9, 1.1, 1.7, -0.3, 0.2];
        let out = iv_ratio(&y, &d, &z).unwrap();
        let total: f64 = out.influence.iter().sum();
        assert!(total.abs() < 1e-12, "influence sums to {total}");
        // A constant instrument against an alternating treatment has no covariance.
        let alternating = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        assert!(iv_ratio(&y, &alternating, &[1.0; 6]).is_err());
    }

    #[test]
    fn iv_ratio_weighted_with_unit_weights_is_iv_ratio() {
        let z = [0.3, -1.2, 0.8, 1.5, -0.4, 0.9];
        let d = [0.1, -0.7, 0.5, 0.9, -0.2, 0.4];
        let y = [0.5, -0.9, 1.1, 1.7, -0.3, 0.2];
        let plain = iv_ratio(&y, &d, &z).unwrap();
        let unit = iv_ratio_weighted(&y, &d, &z, &[1.0; 6]).unwrap();
        assert!((plain.estimate - unit.estimate).abs() < 1e-12);
        assert!((plain.variance - unit.variance).abs() < 1e-12);
        assert!((plain.first_stage_f - unit.first_stage_f).abs() < 1e-9);
    }

    #[test]
    fn iv_ratio_weighted_ignores_zero_weight_rows_whatever_their_residuals() {
        // Rows 4 and 5 have weight zero; their residuals are arbitrary.
        let z = [0.3, -1.2, 0.8, 1.5, -0.4, 0.9];
        let d = [0.1, -0.7, 0.5, 0.9, -0.2, 0.4];
        let y = [0.5, -0.9, 1.1, 1.7, -0.3, 0.2];
        let w = [1.0, 2.0, 0.5, 1.0, 0.0, 0.0];
        let base = iv_ratio_weighted(&y, &d, &z, &w).unwrap();
        let mut y2 = y;
        let mut d2 = d;
        y2[4] = 250.0;
        d2[5] = -90.0;
        let moved = iv_ratio_weighted(&y2, &d2, &z, &w).unwrap();
        assert!((base.estimate - moved.estimate).abs() < 1e-12);
        assert!(moved.influence[4].abs() < 1e-12 && moved.influence[5].abs() < 1e-12);
        let total: f64 = base.influence.iter().sum();
        assert!(total.abs() < 1e-12, "weighted influence sums to {total}");
        assert!(iv_ratio_weighted(&y, &d, &z, &[1.0, -1.0, 1.0, 1.0, 1.0, 1.0]).is_err());
    }

    #[test]
    fn winsorizing_clips_to_type_seven_quantiles_of_positive_weight_rows() {
        // Sorted positive-weight values 1, 2, 3, 4, 100 with tail 0.2:
        // lower = position 0.8 -> 1.8, upper = position 3.2 -> 4 + 0.2 * 96 = 23.2.
        let pseudo = [1.0, 2.0, 3.0, 4.0, 100.0, 1e9];
        let weights = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0];
        let out = winsorize_pseudo_outcome(&pseudo, &weights, 0.2).unwrap();
        assert!((out.lower - 1.8).abs() < 1e-12);
        assert!((out.upper - 23.2).abs() < 1e-12);
        let expected = [1.8, 2.0, 3.0, 4.0, 23.2, 1e9];
        for (got, want) in out.values.iter().zip(expected) {
            assert!((got - want).abs() < 1e-9, "{got} != {want}");
        }
        assert_eq!(out.n_clipped, 2);
    }

    #[test]
    fn modifier_scale_logs_a_heavy_positive_column_and_keeps_the_rest_raw() {
        // 1..=100 has median 50.5 and p99 = 99.01: ratio 1.96, raw scale.
        let plain: Vec<f64> = (1..=100).map(f64::from).collect();
        let raw = modifier_scale(&plain, 0.01, 3.0).unwrap();
        assert!(!raw.log_scale);
        assert!((raw.lower - 1.99).abs() < 1e-9);
        assert!((raw.upper - 99.01).abs() < 1e-9);
        // A column with a zero is never logged, whatever its skew.
        let with_zero = [0.0, 1.0, 2.0, 3.0, 1000.0];
        assert!(!modifier_scale(&with_zero, 0.0, 3.0).unwrap().log_scale);
        // Positive and heavy: p99 / median = 10 / 1 >= 3, so the range is on ln(x).
        let heavy: Vec<f64> = (0..100).map(|i| (f64::from(i) / 10.0).exp()).collect();
        let logged = modifier_scale(&heavy, 0.0, 3.0).unwrap();
        assert!(logged.log_scale);
        let logs: Vec<f64> = heavy.iter().map(|v| v.ln()).collect();
        let min = logs.iter().copied().fold(f64::INFINITY, f64::min);
        let max = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!((logged.lower - min).abs() < 1e-9);
        assert!((logged.upper - max).abs() < 1e-9);
        assert!(modifier_scale(&plain, 0.5, 3.0).is_err());
        assert!(modifier_scale(&plain, 0.01, 0.5).is_err());
    }

    #[test]
    fn winsorizing_with_zero_tail_is_the_identity_and_bad_tails_are_refused() {
        let pseudo = [5.0, -7.0, 0.5];
        let weights = [1.0, 1.0, 1.0];
        let out = winsorize_pseudo_outcome(&pseudo, &weights, 0.0).unwrap();
        assert_eq!(out.values, pseudo.to_vec());
        assert_eq!(out.n_clipped, 0);
        assert!(winsorize_pseudo_outcome(&pseudo, &weights, 0.5).is_err());
        assert!(winsorize_pseudo_outcome(&pseudo, &weights[..2], 0.1).is_err());
    }

    #[test]
    fn a_column_that_is_an_exact_combination_is_reported_with_its_coefficients() {
        // Columns: intercept, age, education, experience = age - education - 6.
        let age = [30.0, 41.0, 25.0, 52.0, 38.0];
        let education = [12.0, 16.0, 9.0, 12.0, 14.0];
        let experience: Vec<f64> = (0..5).map(|i| age[i] - education[i] - 6.0).collect();
        let ones = [1.0; 5];
        let design: Vec<f64> = [&ones[..], &age[..], &education[..], &experience[..]].concat();
        let found = linear_dependence(&design, 5, 4, 1e-10).unwrap();
        assert_eq!(found.len(), 1, "{found:?}");
        assert_eq!(found[0].column, 3);
        let dependence = &found[0].depends_on;
        // The kept columns are intercept (0), age (1), education (2).
        let reproduce = |m: usize| {
            dependence
                .iter()
                .find(|(k, _)| *k == m)
                .map_or(0.0, |v| v.1)
        };
        assert!((reproduce(1) - 1.0).abs() < 1e-9, "{dependence:?}");
        assert!((reproduce(2) + 1.0).abs() < 1e-9, "{dependence:?}");
        assert!((reproduce(0) + 6.0).abs() < 1e-9, "{dependence:?}");
    }

    #[test]
    fn a_constant_column_depends_on_the_intercept_and_independent_columns_are_kept() {
        let ones = [1.0; 6];
        let constant = [2.5; 6];
        let x = [0.3, -1.2, 0.8, 2.0, -0.4, 1.1];
        let z = [1.0, 0.0, 2.0, -1.0, 0.5, 3.0];
        let design: Vec<f64> = [&ones[..], &constant[..], &x[..], &z[..]].concat();
        let found = linear_dependence(&design, 6, 4, 1e-10).unwrap();
        assert_eq!(found.len(), 1, "{found:?}");
        assert_eq!(found[0].column, 1);
        assert_eq!(found[0].depends_on.len(), 1);
        assert_eq!(found[0].depends_on[0].0, 0);
        assert!((found[0].depends_on[0].1 - 2.5).abs() < 1e-12);
    }

    #[test]
    fn a_wrong_shape_is_refused() {
        assert!(linear_dependence(&[1.0, 2.0, 3.0], 2, 2, 1e-10).is_err());
        assert!(linear_dependence(&[1.0, 2.0], 2, 1, 0.0).is_err());
    }

    #[test]
    fn weighted_pseudo_outcome_loss_equals_the_r_loss() {
        let y = [1.0, -0.5, 2.0, 0.3];
        let t = [1.0, 0.0, 1.0, 0.0];
        let m = [0.4, 0.1, 1.1, -0.2];
        let e = [0.7, 0.2, 0.995, 0.5];
        let tau = [0.8, 1.2, -0.3, 0.1];
        let clip = (0.01, 0.99);
        let tr = rloss_transform(&y, &t, &m, &e, clip).unwrap();
        assert_eq!(tr.n_clipped, 1);
        let weighted: f64 = (0..4)
            .map(|i| tr.weights[i] * (tr.pseudo_outcome[i] - tau[i]).powi(2))
            .sum::<f64>()
            / 4.0;
        let direct = rloss(&y, &t, &m, &e, &tau, clip).unwrap();
        assert!((weighted - direct).abs() < 1e-12, "{weighted} vs {direct}");
    }

    #[test]
    fn a_determined_treatment_has_zero_weight() {
        let tr = rloss_transform(&[1.0], &[1.0], &[0.0], &[1.0], (0.0, 1.0)).unwrap();
        assert_eq!(tr.weights, vec![0.0]);
        assert_eq!(tr.pseudo_outcome, vec![0.0]);
    }

    #[test]
    fn gcm_separates_dependent_from_independent_residuals() {
        let n = 2000;
        let mut state = 12345u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        let a: Vec<f64> = (0..n).map(|_| next()).collect();
        let b: Vec<f64> = (0..n).map(|_| next()).collect();
        let independent = gcm_test(&a, &b).unwrap();
        assert!(independent.p_value > 0.001, "{independent:?}");
        let c: Vec<f64> = a.iter().zip(&b).map(|(x, e)| x + 0.3 * e).collect();
        let dependent = gcm_test(&a, &c).unwrap();
        assert!(dependent.p_value < 1e-10 && dependent.residual_correlation > 0.9);
    }

    #[test]
    fn chi_square_tail_matches_known_values() {
        // qchisq(0.95, 1) = 3.841459, qchisq(0.99, 4) = 13.2767
        assert!((chi_square_sf(3.841459, 1.0) - 0.05).abs() < 1e-6);
        assert!((chi_square_sf(13.2767, 4.0) - 0.01).abs() < 1e-5);
        assert!((chi_square_sf(0.5, 3.0) - 0.918_891_4).abs() < 1e-6);
    }

    #[test]
    fn complete_case_ate_score_is_the_classic_aipw_score() {
        let y = [1.0, -0.5, 2.0, 0.3];
        let a = [1.0, 0.0, 1.0, 0.0];
        let obs = [1.0; 4];
        let w = [1.0; 4];
        let e = [0.7, 0.2, 0.6, 0.5];
        let pi = [1.0; 4];
        let mu0 = [0.4, 0.1, 1.1, -0.2];
        let mu1 = [1.2, 0.9, 0.5, 0.0];
        let out = augmented_scores(
            &y,
            &a,
            &obs,
            &w,
            &e,
            &pi,
            &mu0,
            &mu1,
            Estimand::Ate,
            None,
            None,
        )
        .unwrap();
        for i in 0..4 {
            let expected = mu1[i] - mu0[i] + a[i] * (y[i] - mu1[i]) / e[i]
                - (1.0 - a[i]) * (y[i] - mu0[i]) / (1.0 - e[i]);
            assert!((out.scores[i] - expected).abs() < 1e-12);
        }
        let mean = out.scores.iter().sum::<f64>() / 4.0;
        assert!((out.estimate - mean).abs() < 1e-12);
        assert!(out.influence.iter().sum::<f64>().abs() < 1e-12);
    }

    #[test]
    fn unobserved_rows_contribute_only_the_outcome_mean_difference() {
        // Row 1 has no outcome: its y is NaN and its augmentation must be 0.
        let y = [1.0, f64::NAN];
        let a = [1.0, 0.0];
        let obs = [1.0, 0.0];
        let w = [1.0, 1.0];
        let e = [0.5, 0.5];
        let pi = [0.8, 0.4];
        let mu0 = [0.0, 0.3];
        let mu1 = [0.6, 0.9];
        let out = augmented_scores(
            &y,
            &a,
            &obs,
            &w,
            &e,
            &pi,
            &mu0,
            &mu1,
            Estimand::Ate,
            None,
            None,
        )
        .unwrap();
        assert!((out.scores[1] - (0.9 - 0.3)).abs() < 1e-12);
        // row 0: mu1 - mu0 + (y - mu1) / (pi e)
        let expected0 = 0.6 + (1.0 - 0.6) / (0.8 * 0.5);
        assert!((out.scores[0] - expected0).abs() < 1e-12);
        assert!(out.scores[1].is_finite());
    }

    #[test]
    fn integer_weights_equal_replicated_rows() {
        let y = [1.0, -0.5, 2.0, 0.3, 0.9];
        let a = [1.0, 0.0, 1.0, 0.0, 1.0];
        let obs = [1.0, 1.0, 0.0, 1.0, 1.0];
        let w = [2.0, 1.0, 1.0, 3.0, 1.0];
        let e = [0.7, 0.2, 0.6, 0.5, 0.4];
        let pi = [0.9, 0.8, 0.5, 0.7, 0.6];
        let mu0 = [0.4, 0.1, 1.1, -0.2, 0.0];
        let mu1 = [1.2, 0.9, 0.5, 0.0, 0.3];
        let weighted = augmented_scores(
            &y,
            &a,
            &obs,
            &w,
            &e,
            &pi,
            &mu0,
            &mu1,
            Estimand::Ate,
            None,
            None,
        )
        .unwrap();
        let mut rep = Vec::new();
        for i in 0..5 {
            for _ in 0..(w[i] as usize) {
                rep.push(i);
            }
        }
        let pick = |v: &[f64]| rep.iter().map(|&i| v[i]).collect::<Vec<f64>>();
        let (yr, ar, or, er, pr, m0r, m1r) = (
            pick(&y),
            pick(&a),
            pick(&obs),
            pick(&e),
            pick(&pi),
            pick(&mu0),
            pick(&mu1),
        );
        let ones = vec![1.0; rep.len()];
        let plain = augmented_scores(
            &yr,
            &ar,
            &or,
            &ones,
            &er,
            &pr,
            &m0r,
            &m1r,
            Estimand::Ate,
            None,
            None,
        )
        .unwrap();
        assert!((weighted.estimate - plain.estimate).abs() < 1e-12);
        let v_weighted = influence_variance(&weighted.influence, &w, None).unwrap();
        let v_plain = influence_variance(&plain.influence, &ones, None).unwrap();
        assert!(
            (v_weighted - v_plain).abs() < 1e-12,
            "{v_weighted} vs {v_plain}"
        );
    }

    #[test]
    fn weighted_rloss_zeroes_unobserved_rows_and_inverse_weights_observed_ones() {
        let y = [1.0, f64::NAN, 2.0];
        let t = [1.0, 0.0, 1.0];
        let m = [0.4, 0.0, 1.1];
        let e = [0.7, 0.2, 0.5];
        let out = rloss_transform_weighted(
            &y,
            &t,
            &m,
            &e,
            (0.01, 0.99),
            &[1.0, 1.0, 2.0],
            &[1.0, 0.0, 1.0],
            &[0.5, 0.5, 0.25],
            Some((0.01, 0.99)),
        )
        .unwrap();
        assert_eq!(out.weights[1], 0.0);
        assert_eq!(out.pseudo_outcome[1], 0.0);
        // row 0: e = 0.7, so r = 0.3; weight r^2 / pi
        let r0 = 1.0 - 0.7;
        assert!((out.weights[0] - r0 * r0 / 0.5).abs() < 1e-12);
        // row 2: e = 0.5, sample weight 2, so weight 2 r^2 / pi
        let r2 = 1.0 - 0.5;
        assert!((out.weights[2] - 2.0 * r2 * r2 / 0.25).abs() < 1e-12);
    }

    #[test]
    fn cluster_variance_treats_a_group_as_one_unit() {
        let influence = [1.0, -1.0, 2.0, -2.0];
        let ones = [1.0; 4];
        let groups = [0, 0, 1, 1];
        let v = influence_variance(&influence, &ones, Some(&groups)).unwrap();
        // group totals are 0 and 0, so the cluster variance is zero
        assert!(v.abs() < 1e-15);
        let single = [0usize, 0, 0, 0];
        assert!(influence_variance(&influence, &ones, Some(&single)).is_err());
    }

    #[test]
    fn att_scores_on_unit_weights_match_the_definition() {
        let y = [1.0, 2.0, 0.5, -1.0];
        let a = [1.0, 1.0, 0.0, 0.0];
        let obs = [1.0; 4];
        let w = [1.0; 4];
        let e = [0.6, 0.4, 0.3, 0.7];
        let mu0 = [0.2, 0.1, 0.4, 0.0];
        let mu1 = [0.0; 4];
        let pi = [1.0; 4];
        let out = augmented_scores(
            &y,
            &a,
            &obs,
            &w,
            &e,
            &pi,
            &mu0,
            &mu1,
            Estimand::Att,
            None,
            None,
        )
        .unwrap();
        let num: Vec<f64> = (0..4)
            .map(|i| a[i] * (y[i] - mu0[i]) - (1.0 - a[i]) * e[i] / (1.0 - e[i]) * (y[i] - mu0[i]))
            .collect();
        // the mean of num / P(A = 1) with P(A = 1) = 0.5 over four rows
        let expected = num.iter().sum::<f64>() / 2.0;
        assert!((out.estimate - expected).abs() < 1e-12);
        assert!((out.treated_share - 0.5).abs() < 1e-15);
    }

    #[test]
    fn att_with_a_missing_treated_outcome_weights_the_residuals_by_response() {
        // Row 1 is a treated row whose outcome is missing: it still counts in
        // P(A = 1) and contributes A (mu1 - mu0) with no residual terms.
        let y = [1.0, f64::NAN, 2.0, 0.5];
        let a = [1.0, 1.0, 0.0, 0.0];
        let obs = [1.0, 0.0, 1.0, 1.0];
        let w = [1.0; 4];
        let e = [0.6, 0.4, 0.3, 0.7];
        let pi = [0.8, 0.5, 0.6, 0.9];
        let mu0 = [0.2, 0.1, 0.4, 0.0];
        let mu1 = [0.9, 0.7, 0.0, 0.0];
        let out = augmented_scores(
            &y,
            &a,
            &obs,
            &w,
            &e,
            &pi,
            &mu0,
            &mu1,
            Estimand::Att,
            None,
            None,
        )
        .unwrap();
        let num0 = (0.9 - 0.2) + (1.0 - 0.9) / 0.8;
        let num1 = 0.7 - 0.1;
        let num2 = -(0.3 / 0.7) * (2.0 - 0.4) / 0.6;
        let num3 = -(0.7 / 0.3) * (0.5 - 0.0) / 0.9;
        let expected = (num0 + num1 + num2 + num3) / 4.0 / 0.5;
        assert!(
            (out.estimate - expected).abs() < 1e-12,
            "{} vs {expected}",
            out.estimate
        );
        // The influence values of a weighted ratio average to zero.
        let mean_influence: f64 = out.influence.iter().sum::<f64>() / 4.0;
        assert!(mean_influence.abs() < 1e-12);
    }

    #[test]
    fn overlap_scores_with_exact_outcome_means_are_the_overlap_weighted_difference() {
        // With each observed outcome equal to its own arm's mean, the
        // residuals vanish and the estimate is sum(h (mu1 - mu0)) / sum(h),
        // h = e (1 - e), whatever the propensities are.
        let a = [1.0, 1.0, 0.0, 0.0, 1.0];
        let mu0 = [0.2, 0.1, 0.4, 0.0, 0.3];
        let mu1 = [0.9, 0.7, 0.5, 0.2, 0.6];
        let y: Vec<f64> = (0..5)
            .map(|i| if a[i] == 1.0 { mu1[i] } else { mu0[i] })
            .collect();
        let obs = [1.0; 5];
        let w = [1.0; 5];
        let e = [0.6, 0.02, 0.3, 0.7, 0.995];
        let pi = [1.0; 5];
        let out = augmented_scores(
            &y,
            &a,
            &obs,
            &w,
            &e,
            &pi,
            &mu0,
            &mu1,
            Estimand::Overlap,
            None,
            None,
        )
        .unwrap();
        let h: Vec<f64> = e.iter().map(|x| x * (1.0 - x)).collect();
        let expected: f64 =
            (0..5).map(|i| h[i] * (mu1[i] - mu0[i])).sum::<f64>() / h.iter().sum::<f64>();
        assert!((out.estimate - expected).abs() < 1e-12);
        let mean_influence: f64 = out.influence.iter().sum::<f64>() / 5.0;
        assert!(mean_influence.abs() < 1e-12);
    }

    #[test]
    fn invalid_inputs_are_refused() {
        assert!(rloss_transform(&[1.0], &[0.5], &[0.0], &[0.5], (0.01, 0.99)).is_err());
        assert!(rloss_transform(&[1.0], &[1.0], &[0.0], &[1.5], (0.01, 0.99)).is_err());
        assert!(rloss_transform(&[1.0], &[1.0], &[0.0], &[0.5], (0.6, 0.4)).is_err());
        assert!(rloss_transform(&[1.0, 2.0], &[1.0], &[0.0], &[0.5], (0.01, 0.99)).is_err());
    }

    #[test]
    fn median_rule_takes_median_estimate_and_median_of_variance_terms() {
        // Three repeats of one quantity: estimates 1, 2, 4 with ses 0.5, 0.5, 1.
        // Median estimate 2. Variance terms: 0.25 + 1 = 1.25, 0.25 + 0 = 0.25,
        // 1 + 4 = 5; their median is 1.25, so se = sqrt(1.25).
        let est = vec![vec![1.0], vec![2.0], vec![4.0]];
        let se = vec![vec![0.5], vec![0.5], vec![1.0]];
        let out = median_repeat_combine(&est, &se).unwrap();
        assert!((out.estimate[0] - 2.0).abs() < 1e-12);
        assert!((out.se[0] - 1.25_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn median_rule_with_one_repeat_returns_that_repeat() {
        let out = median_repeat_combine(&[vec![3.0, -1.0]], &[vec![0.2, 0.4]]).unwrap();
        assert_eq!(out.estimate, vec![3.0, -1.0]);
        assert!((out.se[0] - 0.2).abs() < 1e-12 && (out.se[1] - 0.4).abs() < 1e-12);
    }

    #[test]
    fn median_rule_averages_the_two_middle_repeats_for_an_even_count() {
        let est = vec![vec![1.0], vec![2.0], vec![3.0], vec![10.0]];
        let se = vec![vec![0.0], vec![0.0], vec![0.0], vec![0.0]];
        let out = median_repeat_combine(&est, &se).unwrap();
        assert!((out.estimate[0] - 2.5).abs() < 1e-12);
        // Variance terms (x - 2.5)^2: 2.25, 0.25, 0.25, 56.25; median (0.25 + 2.25)/2.
        assert!((out.se[0] - 1.25_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn median_rule_refuses_ragged_or_non_finite_input() {
        assert!(median_repeat_combine(&[], &[]).is_err());
        assert!(median_repeat_combine(&[vec![1.0]], &[vec![1.0], vec![1.0]]).is_err());
        assert!(
            median_repeat_combine(&[vec![1.0, 2.0], vec![1.0]], &[vec![1.0, 1.0], vec![1.0]])
                .is_err()
        );
        assert!(median_repeat_combine(&[vec![f64::NAN]], &[vec![1.0]]).is_err());
        assert!(median_repeat_combine(&[vec![1.0]], &[vec![-1.0]]).is_err());
    }

    #[test]
    fn bootstrap_percentile_interval_uses_type_seven_quantiles_and_sample_sd() {
        // Draws 1..=5 for one quantity. At level 0.8 the limits are the 0.1 and
        // 0.9 type-7 quantiles: positions 0.4 and 3.6 -> 1.4 and 4.6.
        let draws: Vec<Vec<f64>> = (1..=5).map(|v| vec![v as f64]).collect();
        let out = bootstrap_percentile_interval(&draws, 0.8).unwrap();
        assert!((out.lower[0] - 1.4).abs() < 1e-12);
        assert!((out.upper[0] - 4.6).abs() < 1e-12);
        // Sample variance of 1..5 is 2.5.
        assert!((out.se[0] - 2.5_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn bootstrap_percentile_interval_refuses_too_few_draws_and_bad_levels() {
        assert!(bootstrap_percentile_interval(&[vec![1.0]], 0.9).is_err());
        let draws = vec![vec![1.0], vec![2.0]];
        assert!(bootstrap_percentile_interval(&draws, 1.0).is_err());
        assert!(bootstrap_percentile_interval(&draws, 0.0).is_err());
        assert!(bootstrap_percentile_interval(&[vec![1.0], vec![f64::INFINITY]], 0.9).is_err());
    }
}

#[cfg(test)]
mod two_part_tests {
    use super::*;

    #[test]
    fn residual_variance_is_the_weighted_mean_square() {
        let y = [1.0, 3.0, 5.0];
        let f = [0.0, 3.0, 4.0];
        // residuals 1, 0, 1 with weights 1, 1, 2: (1 + 0 + 2) / 4
        let s2 = log_residual_variance(&y, &f, &[1.0, 1.0, 2.0]).unwrap();
        assert!((s2 - 0.75).abs() < 1e-12);
        assert!(log_residual_variance(&y, &f, &[0.0, 0.0, 0.0]).is_err());
        assert!(log_residual_variance(&y, &f[..2], &[1.0, 1.0, 1.0]).is_err());
    }

    #[test]
    fn two_part_mean_is_probability_times_lognormal_mean() {
        let mu = two_part_mean(&[0.5, 1.0], &[2.0, 0.0], 0.5).unwrap();
        assert!((mu[0] - 0.5 * (2.0_f64 + 0.25).exp()).abs() < 1e-12);
        assert!((mu[1] - 0.25_f64.exp()).abs() < 1e-12);
        assert!(two_part_mean(&[0.5], &[2.0], -1.0).is_err());
        assert!(two_part_mean(&[0.5, 0.5], &[2.0], 0.1).is_err());
    }

    #[test]
    fn binomial_deviance_is_weighted_mean_of_minus_twice_log_likelihood() {
        // Rows: t=1 with p=0.8, t=0 with p=0.25, weights 1 and 3.
        let d = binomial_deviance(&[1.0, 0.0], &[0.8, 0.25], &[1.0, 3.0]).unwrap();
        let expected = (-2.0 * 0.8_f64.ln() + 3.0 * (-2.0) * 0.75_f64.ln()) / 4.0;
        assert!((d - expected).abs() < 1e-12, "{d} vs {expected}");
        // A contradicted certainty is large and finite, not infinite.
        let extreme = binomial_deviance(&[1.0], &[0.0], &[1.0]).unwrap();
        assert!(extreme.is_finite() && extreme > 50.0);
        assert!(binomial_deviance(&[0.5], &[0.5], &[1.0]).is_err());
        assert!(binomial_deviance(&[1.0], &[1.5], &[1.0]).is_err());
        assert!(binomial_deviance(&[1.0, 0.0], &[0.5], &[1.0, 1.0]).is_err());
        assert!(binomial_deviance(&[1.0], &[0.5], &[0.0]).is_err());
    }
}
