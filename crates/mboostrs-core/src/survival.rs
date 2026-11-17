//! Survival summaries for Cox boosting fits.
//!
//! Four computations sit here: the Breslow baseline cumulative hazard of a
//! Cox linear predictor, survival curves at fixed horizons, a Kaplan–Meier
//! estimate of the censoring distribution, and two horizon-specific
//! evaluation scores that weight observations by the inverse probability of
//! being uncensored (IPCW): the Brier score and the cumulative/dynamic AUC.

use ndarray::Array2;

use crate::MboostrsError;

/// Smallest risk-set sum used in the Breslow hazard. A smaller sum is treated
/// as an empty risk set, so the hazard increment at that time is zero.
const RISK_SET_FLOOR: f64 = 1e-6;

/// Breslow baseline cumulative hazard, stored at the distinct event times.
///
/// The linear predictor is centered at the mean of the training predictor
/// (`center`), so `cumhaz` is the baseline on that centered scale. A new
/// subject's relative risk is `exp(linpred - center)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Breslow {
    /// Distinct times with at least one event, ascending.
    pub event_times: Vec<f64>,
    /// Weighted event count at each entry of `event_times`.
    pub n_event: Vec<f64>,
    /// Cumulative baseline hazard `H0(t)` at each entry of `event_times`.
    pub cumhaz: Vec<f64>,
    /// Mean of the training linear predictor that the baseline is centered on.
    pub center: f64,
}

fn check_finite(name: &str, values: &[f64]) -> Result<(), MboostrsError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(MboostrsError::Shape(format!("{name} must be finite")))
    }
}

fn check_same_length(name: &str, len: usize, expected: usize) -> Result<(), MboostrsError> {
    if len == expected {
        Ok(())
    } else {
        Err(MboostrsError::Shape(format!(
            "{name} has {len} entries, expected {expected}"
        )))
    }
}

/// Indices that sort `time` ascending. The sort is stable, so tied times keep
/// their input order.
fn time_order(time: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..time.len()).collect();
    order.sort_by(|&a, &b| time[a].total_cmp(&time[b]));
    order
}

/// Breslow baseline cumulative hazard from observed times, event indicators
/// and the training linear predictor.
///
/// For each distinct time `u`, the hazard increment is `d(u) / R(u)` where
/// `d(u)` is the weighted event count at `u` and `R(u)` is the sum of
/// `exp(linpred - mean(linpred))` over rows with `time >= u`. A risk-set sum
/// below `1e-6` is treated as empty and contributes no hazard.
pub fn breslow(
    time: &[f64],
    event: &[f64],
    linpred_train: &[f64],
) -> Result<Breslow, MboostrsError> {
    let n = time.len();
    if n == 0 {
        return Err(MboostrsError::Shape(
            "survival data must not be empty".into(),
        ));
    }
    check_same_length("event", event.len(), n)?;
    check_same_length("linpred_train", linpred_train.len(), n)?;
    check_finite("time", time)?;
    check_finite("linpred_train", linpred_train)?;

    let center = linpred_train.iter().sum::<f64>() / n as f64;
    let order = time_order(time);
    let time_sorted: Vec<f64> = order.iter().map(|&i| time[i]).collect();

    // Suffix sums of exp(centered predictor) in time order: suffix[k] is the
    // sum over sorted rows k..n. A risk set at a time is the suffix that
    // starts at that time's first row.
    let mut suffix = vec![0.0; n];
    let mut running = 0.0;
    for k in (0..n).rev() {
        running += (linpred_train[order[k]] - center).exp();
        suffix[k] = running;
    }

    let mut event_times = Vec::new();
    let mut n_event = Vec::new();
    let mut cumhaz = Vec::new();
    let mut hazard = 0.0;
    let mut start = 0;
    while start < n {
        let current = time_sorted[start];
        let mut end = start;
        let mut events = 0.0;
        while end < n && time_sorted[end] == current {
            events += event[order[end]];
            end += 1;
        }
        let risk = if suffix[start] < RISK_SET_FLOOR {
            f64::INFINITY
        } else {
            suffix[start]
        };
        hazard += events / risk;
        if events > 0.0 {
            event_times.push(current);
            n_event.push(events);
            cumhaz.push(hazard);
        }
        start = end;
    }
    Ok(Breslow {
        event_times,
        n_event,
        cumhaz,
        center,
    })
}

impl Breslow {
    /// Number of event times in the baseline.
    pub fn len(&self) -> usize {
        self.event_times.len()
    }

    /// Whether the baseline has no event times.
    pub fn is_empty(&self) -> bool {
        self.event_times.is_empty()
    }

    /// Survival curves at every event time, one column per subject.
    ///
    /// With `linpred_new = None` the single column is the baseline curve
    /// `exp(-H0(t))`. Otherwise column `j` is
    /// `exp(-H0(t) * exp(linpred_new[j] - center))`. The result has shape
    /// `(n_event_times, n_curves)`.
    pub fn curves(&self, linpred_new: Option<&[f64]>) -> Result<Array2<f64>, MboostrsError> {
        match linpred_new {
            None => Ok(Array2::from_shape_fn((self.len(), 1), |(i, _)| {
                (-self.cumhaz[i]).exp()
            })),
            Some(values) => {
                check_finite("linpred_new", values)?;
                let relative: Vec<f64> = values
                    .iter()
                    .map(|value| (value - self.center).exp())
                    .collect();
                Ok(Array2::from_shape_fn(
                    (self.len(), relative.len()),
                    |(i, j)| (-(self.cumhaz[i] * relative[j])).exp(),
                ))
            }
        }
    }

    /// Survival probabilities `S(t | x)` at fixed horizons for each subject.
    ///
    /// `H0(t)` is the cumulative hazard at the largest event time that is at
    /// most `t`, and zero before the first event, so the step function is
    /// right-continuous. The result has shape `(n_subjects, n_horizons)`.
    pub fn survival_at(
        &self,
        horizons: &[f64],
        linpred_new: &[f64],
    ) -> Result<Array2<f64>, MboostrsError> {
        check_finite("horizons", horizons)?;
        check_finite("linpred_new", linpred_new)?;
        let baseline: Vec<f64> = horizons
            .iter()
            .map(|&t| {
                let passed = self
                    .event_times
                    .partition_point(|&event_time| event_time <= t);
                if passed == 0 {
                    0.0
                } else {
                    self.cumhaz[passed - 1]
                }
            })
            .collect();
        let relative: Vec<f64> = linpred_new
            .iter()
            .map(|value| (value - self.center).exp())
            .collect();
        Ok(Array2::from_shape_fn(
            (relative.len(), horizons.len()),
            |(i, h)| (-(baseline[h] * relative[i])).exp(),
        ))
    }
}

/// Kaplan–Meier estimate of a survival function from weighted right-censored
/// data. Applied with the censoring indicator reversed, it estimates the
/// censoring survival `G(t)`.
#[derive(Clone, Debug, PartialEq)]
pub struct KaplanMeier {
    times: Vec<f64>,
    surv: Vec<f64>,
}

/// Kaplan–Meier estimate. `event[i] > 0` marks row `i` as the event of
/// interest. Weights multiply each row's contribution to the risk set and to
/// the event count.
pub fn kaplan_meier(
    time: &[f64],
    event: &[f64],
    weights: &[f64],
) -> Result<KaplanMeier, MboostrsError> {
    let n = time.len();
    check_same_length("event", event.len(), n)?;
    check_same_length("weights", weights.len(), n)?;
    check_finite("time", time)?;
    check_finite("weights", weights)?;
    if weights.iter().any(|&w| w < 0.0) {
        return Err(MboostrsError::Shape("weights must be non-negative".into()));
    }
    let order = time_order(time);
    let mut suffix_weight = vec![0.0; n + 1];
    for k in (0..n).rev() {
        suffix_weight[k] = suffix_weight[k + 1] + weights[order[k]];
    }
    let mut times = Vec::new();
    let mut surv = Vec::new();
    let mut current_surv = 1.0;
    let mut start = 0;
    while start < n {
        let current = time[order[start]];
        let mut end = start;
        let mut events = 0.0;
        while end < n && time[order[end]] == current {
            if event[order[end]] > 0.0 {
                events += weights[order[end]];
            }
            end += 1;
        }
        if events > 0.0 {
            let at_risk = suffix_weight[start];
            if at_risk <= 0.0 {
                return Err(MboostrsError::Control(
                    "Kaplan-Meier risk set has no weight at an event time".into(),
                ));
            }
            current_surv *= 1.0 - events / at_risk;
            times.push(current);
            surv.push(current_surv);
        }
        start = end;
    }
    Ok(KaplanMeier { times, surv })
}

impl KaplanMeier {
    /// Right-continuous value `S(t)`, the probability of surviving past `t`.
    pub fn at(&self, t: f64) -> f64 {
        let passed = self.times.partition_point(|&time| time <= t);
        if passed == 0 {
            1.0
        } else {
            self.surv[passed - 1]
        }
    }

    /// Left limit `S(t-)`, the probability of surviving up to but not
    /// including `t`.
    pub fn left(&self, t: f64) -> f64 {
        let passed = self.times.partition_point(|&time| time < t);
        if passed == 0 {
            1.0
        } else {
            self.surv[passed - 1]
        }
    }
}

/// An IPCW score at one horizon, with the censoring positivity diagnostic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IpcwScore {
    /// The score.
    pub value: f64,
    /// Kaplan–Meier censoring survival `G(t)` at the horizon.
    pub censoring_at_horizon: f64,
    /// Whether `G(t)` is below the censoring floor the caller set.
    pub below_floor: bool,
}

fn censoring_km(
    time: &[f64],
    event: &[f64],
    weights: &[f64],
    horizon: f64,
    censoring_floor: f64,
) -> Result<(KaplanMeier, f64, bool), MboostrsError> {
    if !horizon.is_finite() {
        return Err(MboostrsError::Control("horizon must be finite".into()));
    }
    if !(0.0..1.0).contains(&censoring_floor) {
        return Err(MboostrsError::Control(
            "censoring_floor must lie in [0, 1)".into(),
        ));
    }
    let censored: Vec<f64> = event
        .iter()
        .map(|&value| if value > 0.0 { 0.0 } else { 1.0 })
        .collect();
    let censor = kaplan_meier(time, &censored, weights)?;
    let g_horizon = censor.at(horizon);
    if g_horizon <= 0.0 {
        return Err(MboostrsError::Control(format!(
            "censoring survival at horizon {horizon} is zero, so IPCW weights are infinite; \
             evaluate at an earlier horizon"
        )));
    }
    Ok((censor, g_horizon, g_horizon < censoring_floor))
}

/// IPCW Brier score at horizon `t` (Graf et al., 1999).
///
/// `survival_at_horizon[i]` is the predicted `S_i(t)`. A row whose event
/// occurred by `t` contributes `S_i(t)^2 / G(T_i-)`, a row still at risk at
/// `t` contributes `(1 - S_i(t))^2 / G(t)`, and a row censored by `t`
/// contributes nothing. The sum is divided by the total weight. `G` is the
/// Kaplan–Meier censoring survival estimated from the same rows.
pub fn ipcw_brier(
    time: &[f64],
    event: &[f64],
    weights: &[f64],
    survival_at_horizon: &[f64],
    horizon: f64,
    censoring_floor: f64,
) -> Result<IpcwScore, MboostrsError> {
    let n = time.len();
    check_same_length("survival_at_horizon", survival_at_horizon.len(), n)?;
    check_finite("survival_at_horizon", survival_at_horizon)?;
    let total: f64 = weights.iter().sum();
    if total.is_nan() || total <= 0.0 {
        return Err(MboostrsError::Control(
            "weights must have positive total".into(),
        ));
    }
    let (censor, g_horizon, below_floor) =
        censoring_km(time, event, weights, horizon, censoring_floor)?;
    let mut sum = 0.0;
    for i in 0..n {
        let s = survival_at_horizon[i];
        let term = if time[i] <= horizon {
            if event[i] > 0.0 {
                s * s / censor.left(time[i])
            } else {
                0.0
            }
        } else {
            (1.0 - s) * (1.0 - s) / g_horizon
        };
        sum += weights[i] * term;
    }
    Ok(IpcwScore {
        value: sum / total,
        censoring_at_horizon: g_horizon,
        below_floor,
    })
}

/// Result of [`ipcw_auc`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AucScore {
    /// Cumulative/dynamic AUC at the horizon, with ties counted one half.
    pub auc: f64,
    /// Kaplan–Meier censoring survival `G(t)` at the horizon.
    pub censoring_at_horizon: f64,
    /// Whether `G(t)` is below the censoring floor the caller set.
    pub below_floor: bool,
    /// Sum of IPCW-weighted case weights, `sum_i w_i / G(T_i-)`.
    pub case_weight: f64,
    /// Sum of evaluation weights over subjects still event-free at `t`.
    pub control_weight: f64,
}

/// Cumulative/dynamic IPCW AUC at horizon `t` (Heagerty and Zheng, 2005).
///
/// Cases are rows with an event at or before `t`, weighted by `w_i /
/// G(T_i-)`. Controls are rows with `time > t`, weighted by `w_j`. The AUC is
/// the weighted fraction of case–control pairs in which the case has the
/// higher `score`, with a tied pair counted one half. `score` is a risk
/// ranking, so higher values mean higher hazard. The estimate ranks subjects
/// only and does not use the absolute survival probabilities.
pub fn ipcw_auc(
    time: &[f64],
    event: &[f64],
    weights: &[f64],
    score: &[f64],
    horizon: f64,
    censoring_floor: f64,
) -> Result<AucScore, MboostrsError> {
    let n = time.len();
    check_same_length("score", score.len(), n)?;
    check_finite("score", score)?;
    let (censor, g_horizon, below_floor) =
        censoring_km(time, event, weights, horizon, censoring_floor)?;

    let mut controls: Vec<(f64, f64)> = Vec::new();
    let mut cases: Vec<(f64, f64)> = Vec::new();
    for i in 0..n {
        if time[i] > horizon {
            controls.push((score[i], weights[i]));
        } else if event[i] > 0.0 {
            let case_weight = weights[i] / censor.left(time[i]);
            cases.push((score[i], case_weight));
        }
    }
    let control_weight: f64 = controls.iter().map(|(_, w)| w).sum();
    let case_weight: f64 = cases.iter().map(|(_, w)| w).sum();
    if case_weight <= 0.0 || control_weight <= 0.0 {
        return Err(MboostrsError::Control(format!(
            "AUC at horizon {horizon} needs events by the horizon and subjects still at risk \
             after it; choose a horizon inside the follow-up"
        )));
    }

    controls.sort_by(|a, b| a.0.total_cmp(&b.0));
    let control_scores: Vec<f64> = controls.iter().map(|(s, _)| *s).collect();
    let mut prefix = vec![0.0; controls.len() + 1];
    for (k, (_, w)) in controls.iter().enumerate() {
        prefix[k + 1] = prefix[k] + w;
    }
    let mut concordant = 0.0;
    for (case_score, case_w) in &cases {
        let below = control_scores.partition_point(|s| s < case_score);
        let at_or_below = control_scores.partition_point(|s| s <= case_score);
        let lower = prefix[below];
        let tied = prefix[at_or_below] - prefix[below];
        concordant += case_w * (lower + 0.5 * tied);
    }
    Ok(AucScore {
        auc: concordant / (case_weight * control_weight),
        censoring_at_horizon: g_horizon,
        below_floor,
        case_weight,
        control_weight,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn breslow_matches_hand_computed_hazard() {
        // Times 1, 2, 3, 4 with events at 1 and 3. With linpred 0 everywhere,
        // the risk sets are 4 at t=1 and 2 at t=3.
        let time = [1.0, 2.0, 3.0, 4.0];
        let event = [1.0, 0.0, 1.0, 0.0];
        let fit = breslow(&time, &event, &[0.0; 4]).unwrap();
        assert_eq!(fit.event_times, vec![1.0, 3.0]);
        assert_eq!(fit.n_event, vec![1.0, 1.0]);
        assert!((fit.cumhaz[0] - 0.25).abs() < 1e-15);
        assert!((fit.cumhaz[1] - (0.25 + 0.5)).abs() < 1e-15);
    }

    #[test]
    fn horizon_survival_is_a_right_continuous_step() {
        let time = [1.0, 2.0, 3.0, 4.0];
        let event = [1.0, 0.0, 1.0, 0.0];
        let fit = breslow(&time, &event, &[0.0; 4]).unwrap();
        let s = fit.survival_at(&[0.5, 1.0, 2.9, 3.0], &[0.0]).unwrap();
        assert!((s[[0, 0]] - 1.0).abs() < 1e-15);
        assert!((s[[0, 1]] - (-0.25f64).exp()).abs() < 1e-15);
        assert!((s[[0, 2]] - (-0.25f64).exp()).abs() < 1e-15);
        assert!((s[[0, 3]] - (-0.75f64).exp()).abs() < 1e-15);
    }

    #[test]
    fn kaplan_meier_left_limit_differs_at_event_time() {
        let km = kaplan_meier(&[1.0, 2.0, 3.0], &[1.0, 1.0, 0.0], &[1.0; 3]).unwrap();
        assert!((km.at(2.0) - (2.0 / 3.0) * 0.5).abs() < 1e-15);
        assert!((km.left(2.0) - 2.0 / 3.0).abs() < 1e-15);
    }

    #[test]
    fn brier_refuses_zero_censoring_at_horizon() {
        // Row 2 is censored at t=2 and is the last row at risk, so the
        // censoring survival falls to zero at the horizon.
        let err = ipcw_brier(
            &[1.0, 2.0],
            &[1.0, 0.0],
            &[1.0, 1.0],
            &[0.5, 0.5],
            2.0,
            0.05,
        );
        assert!(err.is_err());
    }
}
