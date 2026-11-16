//! Response-scale scoring shared by evaluation and stopping selection.

use crate::MboostrsError;
use ndarray::{ArrayView2, Axis};

/// Return predictions from a log or log1p response fit to the original scale.
/// These are back-transformed point predictions, not bias-corrected means.
pub fn inverse_response_transform(values: &[f64], kind: &str) -> Result<Vec<f64>, MboostrsError> {
    if !matches!(kind, "identity" | "log" | "log1p") {
        return Err(MboostrsError::Control(format!(
            "unsupported response transform {kind:?}"
        )));
    }
    values
        .iter()
        .map(|value| {
            let result = match kind {
                "identity" => *value,
                "log" => value.exp(),
                "log1p" => value.exp_m1(),
                _ => unreachable!(),
            };
            if result.is_finite() {
                Ok(result)
            } else {
                Err(MboostrsError::BadResponse(
                    "response_transform".into(),
                    "back-transformed prediction must be finite".into(),
                ))
            }
        })
        .collect()
}

#[derive(Clone, Copy)]
enum Metric {
    Mse,
    Brier,
    LogLoss,
    Pinball(f64),
    Expectile(f64),
}

impl Metric {
    fn parse(name: &str, parameter: Option<f64>) -> Result<Self, MboostrsError> {
        if matches!(name, "pinball" | "expectile") {
            let tau = parameter
                .ok_or_else(|| MboostrsError::Control("score_parameter is required".into()))?;
            if !tau.is_finite() || tau <= 0.0 || tau >= 1.0 {
                return Err(MboostrsError::Control(
                    "score_parameter must lie strictly between zero and one".into(),
                ));
            }
            return Ok(if name == "pinball" {
                Self::Pinball(tau)
            } else {
                Self::Expectile(tau)
            });
        }
        if parameter.is_some() {
            return Err(MboostrsError::Control(
                "score_parameter is only used by pinball or expectile loss".into(),
            ));
        }
        match name {
            "mse" => Ok(Self::Mse),
            "brier" => Ok(Self::Brier),
            "log_loss" => Ok(Self::LogLoss),
            _ => Err(MboostrsError::Control(format!(
                "unknown response score {name:?}"
            ))),
        }
    }
}

fn score_inner(
    y: &[f64],
    predicted: ArrayView2<f64>,
    weights: &[f64],
    metric: Metric,
    retain: bool,
) -> Result<(f64, Vec<f64>), MboostrsError> {
    let (n, classes) = predicted.dim();
    if n == 0 || classes == 0 || y.len() != n || weights.len() != n {
        return Err(MboostrsError::Shape(
            "score inputs must have matching nonempty rows".into(),
        ));
    }
    let sumw: f64 = weights.iter().sum();
    if weights.iter().any(|w| !w.is_finite() || *w < 0.0) || !sumw.is_finite() || sumw <= 0.0 {
        return Err(MboostrsError::Control(
            "score weights must be finite, nonnegative and have positive finite sum".into(),
        ));
    }
    if classes > 1 && !matches!(metric, Metric::Brier | Metric::LogLoss) {
        return Err(MboostrsError::Control(
            "multiclass probabilities require brier or log_loss".into(),
        ));
    }
    let mut losses = if retain { vec![0.0; n] } else { Vec::new() };
    let mut mean = 0.0;
    for i in 0..n {
        if weights[i] == 0.0 {
            continue;
        }
        if !y[i].is_finite() || predicted.row(i).iter().any(|p| !p.is_finite()) {
            return Err(MboostrsError::BadResponse(
                "response_score".into(),
                "score observations and predictions must be finite".into(),
            ));
        }
        let probability = classes > 1 || matches!(metric, Metric::LogLoss | Metric::Brier);
        if probability && predicted.row(i).iter().any(|p| *p < 0.0 || *p > 1.0) {
            return Err(MboostrsError::BadResponse(
                "response_score".into(),
                "probability scores require predictions in [0, 1]".into(),
            ));
        }
        let loss = if classes > 1 {
            if y[i] < 0.0 || y[i] >= classes as f64 || y[i].fract() != 0.0 {
                return Err(MboostrsError::BadResponse(
                    "response_score".into(),
                    "multiclass observations must be class indices".into(),
                ));
            }
            if (predicted.row(i).sum() - 1.0).abs() > 1e-8 {
                return Err(MboostrsError::BadResponse(
                    "response_score".into(),
                    "class probabilities must sum to one".into(),
                ));
            }
            let class = y[i] as usize;
            match metric {
                Metric::LogLoss => -predicted[[i, class]].max(f64::MIN_POSITIVE).ln(),
                _ => (0..classes)
                    .map(|j| (predicted[[i, j]] - if j == class { 1.0 } else { 0.0 }).powi(2))
                    .sum(),
            }
        } else {
            let residual = y[i] - predicted[[i, 0]];
            match metric {
                Metric::Mse => residual * residual,
                Metric::Brier | Metric::LogLoss => {
                    if y[i] != 0.0 && y[i] != 1.0 {
                        return Err(MboostrsError::BadResponse(
                            "response_score".into(),
                            "probability scores require binary labels coded 0/1".into(),
                        ));
                    }
                    if matches!(metric, Metric::Brier) {
                        residual * residual
                    } else {
                        let p = predicted[[i, 0]].clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
                        -y[i] * p.ln() - (1.0 - y[i]) * (-p).ln_1p()
                    }
                }
                Metric::Pinball(tau) => {
                    if residual >= 0.0 {
                        tau * residual
                    } else {
                        (tau - 1.0) * residual
                    }
                }
                Metric::Expectile(tau) => {
                    if residual >= 0.0 {
                        tau * residual * residual
                    } else {
                        (1.0 - tau) * residual * residual
                    }
                }
            }
        };
        if !loss.is_finite() {
            return Err(MboostrsError::Solver("response loss overflowed".into()));
        }
        if retain {
            losses[i] = loss;
        }
        mean += (weights[i] / sumw) * loss;
    }
    if !mean.is_finite() {
        return Err(MboostrsError::Solver(
            "aggregate response loss overflowed".into(),
        ));
    }
    Ok((mean, losses))
}

/// Weighted score and observation losses. Predictions are rows by quantities.
pub fn response_score(
    y: &[f64],
    predicted: ArrayView2<f64>,
    weights: &[f64],
    metric: &str,
    parameter: Option<f64>,
) -> Result<(f64, Vec<f64>), MboostrsError> {
    score_inner(
        y,
        predicted,
        weights,
        Metric::parse(metric, parameter)?,
        true,
    )
}

/// Scores for scalar predictions arranged as path positions by observations.
pub fn response_score_path(
    y: &[f64],
    predictions: ArrayView2<f64>,
    weights: &[f64],
    metric: &str,
    parameter: Option<f64>,
) -> Result<Vec<f64>, MboostrsError> {
    let metric = Metric::parse(metric, parameter)?;
    predictions
        .rows()
        .into_iter()
        .map(|row| score_inner(y, row.insert_axis(Axis(1)), weights, metric, false).map(|v| v.0))
        .collect()
}

/// Equal-fold mean path and its first minimum after the offset-only model.
/// A position with a nonfinite fold value is ineligible; no fold is dropped.
pub fn select_score_path(scores: ArrayView2<f64>) -> Result<(Vec<f64>, usize), MboostrsError> {
    let (folds, positions) = scores.dim();
    if folds == 0 || positions < 2 {
        return Err(MboostrsError::Shape(
            "a score path needs folds and at least iterations 0 and 1".into(),
        ));
    }
    let mut curve = vec![0.0; positions];
    for j in 0..positions {
        curve[j] = if scores.column(j).iter().all(|v| v.is_finite()) {
            scores.column(j).iter().map(|v| v / folds as f64).sum()
        } else {
            f64::INFINITY
        };
    }
    let best = (1..positions)
        .filter(|j| curve[*j].is_finite())
        .min_by(|a, b| curve[*a].total_cmp(&curve[*b]).then(a.cmp(b)))
        .ok_or_else(|| {
            MboostrsError::Solver("no finite positive-iteration score is available".into())
        })?;
    Ok((curve, best))
}

/// Score a scalar fitted path without materializing an iterations-by-rows
/// prediction matrix. Basis recipes and coefficients come from the same fit.
/// Small spline designs are cached after applying their fitted replay recipe;
/// other learners replay only the newly selected increment. Cached designs
/// are capped at eight million cells across the path.
pub fn response_prediction_path(
    bases: &[crate::BasisInfo],
    selected: &[usize],
    coefs: &[Vec<f64>],
    x: &[Vec<f64>],
    by: &[Option<Vec<f64>>],
    offset: &[f64],
    nu: f64,
    y: &[f64],
    weights: &[f64],
    family: &dyn crate::Family,
    metric: &str,
    parameter: Option<f64>,
    unseen: crate::UnseenLevels,
) -> Result<Vec<f64>, MboostrsError> {
    use crate::BasisInfo;
    use ndarray::{ArrayView1, ArrayViewMut1};
    if selected.len() != coefs.len()
        || x.len() != bases.len()
        || offset.len() != y.len()
        || selected.iter().any(|j| *j >= bases.len())
        || !nu.is_finite()
    {
        return Err(MboostrsError::Shape(
            "prediction path dimensions or step length disagree".into(),
        ));
    }
    let n = y.len();
    if family.flat_response_span() != 1 || family.n_from_y(ndarray::ArrayView1::from(y)) != n {
        return Err(MboostrsError::Control(
            "streamed response scores require a scalar response family".into(),
        ));
    }
    let metric = Metric::parse(metric, parameter)?;
    let mut eta = offset.to_vec();
    let mut prediction = vec![0.; n];
    let mut cache: Vec<Option<Vec<Vec<f64>>>> = vec![None; bases.len()];
    let mut counts = vec![0; bases.len()];
    for &j in selected {
        counts[j] += 1;
    }
    fn cacheable(basis: &BasisInfo) -> bool {
        match basis {
            BasisInfo::BbsCentered { .. }
            | BasisInfo::BbsUncentered { .. }
            | BasisInfo::BmonoCentered { .. }
            | BasisInfo::BmonoUncentered { .. }
            | BasisInfo::Bns { .. } => true,
            BasisInfo::Binned { inner, .. } => cacheable(inner),
            _ => false,
        }
    }
    let mut available = 8_000_000usize;
    let mut result = Vec::with_capacity(selected.len() + 1);
    for position in 0..=selected.len() {
        if position > 0 {
            let j = selected[position - 1];
            let coef = &coefs[position - 1];
            let by = by.get(j).and_then(|v| v.as_deref());
            if x[j].len() < n || coef.is_empty() || by.is_some_and(|b| b.len() != n) {
                return Err(MboostrsError::Shape(
                    "invalid prediction path inputs".into(),
                ));
            }
            if cache[j].is_none()
                && cacheable(&bases[j])
                && coef.len() <= 64
                && counts[j] >= coef.len()
                && n.saturating_mul(coef.len()) <= available
            {
                let mut design = Vec::with_capacity(coef.len());
                for k in 0..coef.len() {
                    let mut unit = vec![0.; coef.len()];
                    unit[k] = 1.;
                    let mut column = vec![0.; n];
                    bases[j].predict_into_with(&unit, &x[j], &mut column, 1., by, unseen)?;
                    design.push(column);
                }
                available -= n * coef.len();
                cache[j] = Some(design);
            }
            if let Some(design) = &cache[j] {
                if coef.len() != design.len() {
                    return Err(MboostrsError::Shape(
                        "coefficient width changed along a fixed basis".into(),
                    ));
                }
                for (column, &c) in design.iter().zip(coef) {
                    for (value, b) in eta.iter_mut().zip(column) {
                        *value += nu * c * b;
                    }
                }
            } else {
                bases[j].predict_into_with(coef, &x[j], &mut eta, nu, by, unseen)?;
            }
        }
        family.response(ArrayView1::from(&eta), ArrayViewMut1::from(&mut prediction));
        let values = ArrayView2::from_shape((n, 1), &prediction)
            .map_err(|e| MboostrsError::Shape(e.to_string()))?;
        result.push(score_inner(y, values, weights, metric, false)?.0);
    }
    Ok(result)
}

/// Whether a learner's basis is a function of one covariate value per row.
///
/// Only these learners can be evaluated at a single value and broadcast to
/// every row. A tree, hinge, multi-column or kernel basis reads a different
/// shape of input and is refused by [`predict_over_grid`].
fn univariate_basis(basis: &crate::BasisInfo) -> bool {
    use crate::BasisInfo;
    match basis {
        BasisInfo::ScalarBols { .. }
        | BasisInfo::Bols { .. }
        | BasisInfo::BbsCentered { .. }
        | BasisInfo::BbsUncentered { .. }
        | BasisInfo::Bns { .. }
        | BasisInfo::BmonoUncentered { .. }
        | BasisInfo::BmonoCentered { .. }
        | BasisInfo::Brandom { .. } => true,
        BasisInfo::Binned { inner, .. } => univariate_basis(inner),
        _ => false,
    }
}

/// Predictions at each value of one covariate, with every other learner held
/// at each row's own values.
///
/// `start` is the offset, one entry per row. `x` and `by` are the replay
/// inputs of the base frame, one entry per learner in learner order; `by` may
/// be empty when no learner has a `by=` term. `grid_x` lists the learners
/// whose covariate is the grid variable, each with the value that covariate
/// takes at every grid position. `grid_by` does the same for a learner whose
/// `by=` is the grid variable. `grid_len` is the number of grid positions.
///
/// Each learner's coefficients are summed over the steps up to `mstop`, as
/// `predict_newdata_grouped` does, so a learner is evaluated once per call
/// rather than once per step. A learner that does not read the variable is
/// evaluated once over the rows. A learner whose covariate is replaced and
/// that has no per-row `by=` is evaluated at the single grid value and the
/// result is added to every row. Any other learner that reads the variable
/// is evaluated over the rows at each grid position.
///
/// Returns one linear predictor per grid position, each of length `n`. The
/// caller applies the extrapolation policy to the grid values and to the base
/// rows before the call.
pub fn predict_over_grid(
    bases: &[crate::BasisInfo],
    selected: &[usize],
    coefs: &[Vec<f64>],
    start: &[f64],
    x: &[Vec<f64>],
    by: &[Option<Vec<f64>>],
    nu: f64,
    mstop: usize,
    grid_x: &[(usize, Vec<f64>)],
    grid_by: &[(usize, Vec<f64>)],
    grid_len: usize,
    unseen: crate::UnseenLevels,
) -> Result<Vec<Vec<f64>>, MboostrsError> {
    use crate::BasisInfo;
    let n = start.len();
    if selected.len() != coefs.len()
        || x.len() != bases.len()
        || (!by.is_empty() && by.len() != bases.len())
        || selected.iter().any(|j| *j >= bases.len())
        || !nu.is_finite()
    {
        return Err(MboostrsError::Shape(
            "grid prediction dimensions or step length disagree".into(),
        ));
    }
    let replaced_x = |j: usize| grid_x.iter().find(|(k, _)| *k == j);
    let replaced_by = |j: usize| grid_by.iter().find(|(k, _)| *k == j);
    let mut dependent = vec![false; bases.len()];
    for (j, values) in grid_x.iter().chain(grid_by) {
        if *j >= bases.len() || values.len() != grid_len {
            return Err(MboostrsError::Shape(format!(
                "grid values for learner {j} must have {grid_len} entries"
            )));
        }
        if !univariate_basis(&bases[*j]) {
            return Err(MboostrsError::Control(format!(
                "learner {j} cannot be evaluated over a value grid: its basis is not a \
                 function of one covariate value"
            )));
        }
        dependent[*j] = true;
    }
    let base_by = |j: usize| by.get(j).and_then(|values| values.as_deref());
    for j in 0..bases.len() {
        if replaced_x(j).is_none() && x[j].len() < n {
            return Err(MboostrsError::Shape(format!(
                "x for learner {j} has {} entries; {n} rows are predicted",
                x[j].len()
            )));
        }
        if replaced_by(j).is_none() {
            if let Some(values) = base_by(j) {
                if values.len() != n {
                    return Err(MboostrsError::Shape(format!(
                        "by for learner {j} has {} entries; {n} rows are predicted",
                        values.len()
                    )));
                }
            }
        }
    }

    // Tree and hinge steps replay one step at a time and add into `fixed`
    // first, as in `predict_newdata_grouped`; the rest are summed per learner.
    let mut fixed = start.to_vec();
    let mut totals: Vec<Option<Vec<f64>>> = vec![None; bases.len()];
    for (step, &j) in selected.iter().take(mstop).enumerate() {
        let coef = &coefs[step];
        if matches!(bases[j], BasisInfo::Btree { .. } | BasisInfo::Bhinge { .. }) {
            if dependent[j] {
                return Err(MboostrsError::Control(format!(
                    "learner {j} is a tree or hinge and cannot be evaluated over a value grid"
                )));
            }
            bases[j].predict_into_with(coef, &x[j], &mut fixed, nu, base_by(j), unseen)?;
            continue;
        }
        match &mut totals[j] {
            Some(total) => {
                if total.len() != coef.len() {
                    return Err(MboostrsError::Shape(format!(
                        "learner {j} produced coefficient vectors of differing lengths \
                         ({} then {}) along the path",
                        total.len(),
                        coef.len()
                    )));
                }
                for (slot, value) in total.iter_mut().zip(coef) {
                    *slot += value;
                }
            }
            None => totals[j] = Some(coef.clone()),
        }
    }
    for j in 0..bases.len() {
        if dependent[j] {
            continue;
        }
        if let Some(total) = &totals[j] {
            bases[j].predict_into_with(total, &x[j], &mut fixed, nu, base_by(j), unseen)?;
        }
    }

    let mut out = Vec::with_capacity(grid_len);
    for position in 0..grid_len {
        let mut eta = fixed.clone();
        for j in 0..bases.len() {
            let Some(total) = &totals[j] else { continue };
            if !dependent[j] {
                continue;
            }
            let x_value = replaced_x(j).map(|(_, values)| values[position]);
            let by_value = replaced_by(j).map(|(_, values)| values[position]);
            match (x_value, by_value, base_by(j)) {
                (Some(value), None, None) => {
                    let mut single = [0.0f64];
                    bases[j].predict_into_with(total, &[value], &mut single, nu, None, unseen)?;
                    for row in eta.iter_mut() {
                        *row += single[0];
                    }
                }
                _ => {
                    let x_rows = x_value.map(|value| vec![value; n]);
                    let by_rows = by_value.map(|value| vec![value; n]);
                    let xs: &[f64] = x_rows.as_deref().unwrap_or(&x[j][..n]);
                    let by_slice: Option<&[f64]> = match &by_rows {
                        Some(values) => Some(values.as_slice()),
                        None => base_by(j),
                    };
                    bases[j].predict_into_with(total, xs, &mut eta, nu, by_slice, unseen)?;
                }
            }
        }
        out.push(eta);
    }
    Ok(out)
}

/// Closed-form CRPS of a normal predictive distribution `N(mu, sigma^2)`:
/// `sigma * (z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi))` with `z = (y - mu) / sigma`.
pub fn crps_normal(y: &[f64], mu: &[f64], sigma: &[f64]) -> Result<Vec<f64>, MboostrsError> {
    if mu.len() != y.len() || sigma.len() != y.len() {
        return Err(MboostrsError::Shape(
            "crps_normal inputs must have matching lengths".into(),
        ));
    }
    let inv_sqrt_pi = 1.0 / std::f64::consts::PI.sqrt();
    let inv_sqrt_2pi = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
    let mut out = Vec::with_capacity(y.len());
    for i in 0..y.len() {
        if !(y[i].is_finite() && mu[i].is_finite() && sigma[i].is_finite() && sigma[i] > 0.0) {
            return Err(MboostrsError::Control(
                "crps_normal requires finite y and mu and positive finite sigma".into(),
            ));
        }
        let z = (y[i] - mu[i]) / sigma[i];
        let density = (-0.5 * z * z).exp() * inv_sqrt_2pi;
        let cdf = crate::special::normal_cdf(z);
        out.push(sigma[i] * (z * (2.0 * cdf - 1.0) + 2.0 * density - inv_sqrt_pi));
    }
    Ok(out)
}

/// CRPS of an integer-valued predictive distribution, as the sum over integer
/// thresholds `k = 0..K` of `(F(k) - 1{y <= k})^2`. Column `k` of `cdf` holds
/// `F(k)`, so the sum is exact when `F` is flat between the integers and the
/// columns reach the support's upper tail (the tail beyond column `K-1`
/// contributes the probability mass above it, which the caller keeps below
/// the score's reporting precision). Each `y` must be a non-negative integer
/// below `K`.
pub fn crps_discrete_sum(y: &[f64], cdf: ArrayView2<f64>) -> Result<Vec<f64>, MboostrsError> {
    let (n, k_count) = cdf.dim();
    if y.len() != n || k_count == 0 {
        return Err(MboostrsError::Shape(
            "crps_discrete_sum needs one cdf row per observation and at least one column".into(),
        ));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        if !y[i].is_finite() || y[i] < 0.0 || y[i].fract() != 0.0 || y[i] >= k_count as f64 {
            return Err(MboostrsError::Control(format!(
                "crps_discrete_sum requires integer observations in 0..{k_count}; row {i} is {}",
                y[i]
            )));
        }
        let mut total = 0.0;
        let mut previous = 0.0;
        for k in 0..k_count {
            let value = cdf[[i, k]];
            if !value.is_finite() || !(0.0..=1.0).contains(&value) || value < previous {
                return Err(MboostrsError::Control(
                    "crps_discrete_sum requires a finite, non-decreasing cdf in [0, 1]".into(),
                ));
            }
            previous = value;
            let indicator = if y[i] <= k as f64 { 1.0 } else { 0.0 };
            total += (value - indicator) * (value - indicator);
        }
        out.push(total);
    }
    Ok(out)
}

/// CRPS of a continuous or mixed predictive distribution by Simpson quadrature
/// of `integral (F(t) - 1{y <= t})^2 dt`, split at the observation:
/// the integral over `t < y` of `F(t)^2` plus the integral over `t >= y` of
/// `(1 - F(t))^2`.
///
/// `cdf_below[i, k]` is `F` at `y_i - (M-1-k) * step_below[i]`, so column
/// `M-1` sits at `y_i` and column `0` at the lower end of the integral.
/// `cdf_above[i, k]` is `F` at `y_i + k * step_above[i]`, so column `0` sits
/// at `y_i` and column `M-1` at the upper end. `M` must be odd and at least 3.
/// A cdf that jumps at a point is accurate only when the jump lies at an
/// endpoint, such as a point mass at the lower support bound.
pub fn crps_cdf_quadrature(
    cdf_below: ArrayView2<f64>,
    cdf_above: ArrayView2<f64>,
    step_below: &[f64],
    step_above: &[f64],
) -> Result<Vec<f64>, MboostrsError> {
    let (n, m) = cdf_below.dim();
    if cdf_above.dim() != (n, m) || step_below.len() != n || step_above.len() != n {
        return Err(MboostrsError::Shape(
            "crps_cdf_quadrature inputs must share one row per observation and one column per node"
                .into(),
        ));
    }
    if m < 3 || m % 2 == 0 {
        return Err(MboostrsError::Control(
            "crps_cdf_quadrature needs an odd number of nodes, at least 3".into(),
        ));
    }
    let simpson = |h: f64, values: &dyn Fn(usize) -> f64| -> f64 {
        let mut sum = 0.0;
        for k in 0..m {
            let weight = if k == 0 || k == m - 1 {
                1.0
            } else if k % 2 == 1 {
                4.0
            } else {
                2.0
            };
            sum += weight * values(k);
        }
        sum * h / 3.0
    };
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let (hb, ha) = (step_below[i], step_above[i]);
        if !(hb.is_finite() && hb >= 0.0 && ha.is_finite() && ha >= 0.0) {
            return Err(MboostrsError::Control(
                "crps_cdf_quadrature steps must be finite and non-negative".into(),
            ));
        }
        let below = |k: usize| {
            let value = cdf_below[[i, k]];
            value * value
        };
        let above = |k: usize| {
            let value = cdf_above[[i, k]];
            (1.0 - value) * (1.0 - value)
        };
        for value in cdf_below.row(i).iter().chain(cdf_above.row(i).iter()) {
            if !value.is_finite() || !(0.0..=1.0).contains(value) {
                return Err(MboostrsError::Control(
                    "crps_cdf_quadrature requires cdf values in [0, 1]".into(),
                ));
            }
        }
        out.push(simpson(hb, &below) + simpson(ha, &above));
    }
    Ok(out)
}

/// Weighted AUC and its DeLong covariance for `m` score columns.
///
/// `observed` holds 0/1 outcomes, `scores` one column per candidate score, and
/// `weights` nonnegative frequency weights. The AUC of a column is the weighted
/// probability that a positive row outscores a negative row, with ties counted
/// as one half. The covariance uses the placement values of DeLong, DeLong and
/// Clarke-Pearson (1988) with frequency weights: a positive row's placement is
/// its weighted share of negatives it outscores, and the variance of the AUC
/// is `sum_pos w (v - A)^2 / (W1 (W1 - 1)) + sum_neg w (u - A)^2 / (W0 (W0 - 1))`.
/// Unit weights reproduce the unweighted DeLong estimate. Each class needs
/// total weight above one for the estimate to exist.
///
/// Placements come from sorting each column once, so the cost is
/// `O(m n log n + m^2 n)`.
pub fn auc_delong(
    observed: &[f64],
    scores: ArrayView2<f64>,
    weights: &[f64],
) -> Result<(Vec<f64>, ndarray::Array2<f64>), MboostrsError> {
    let (n, m) = scores.dim();
    if n == 0 || m == 0 || observed.len() != n || weights.len() != n {
        return Err(MboostrsError::Shape(
            "auc_delong needs one observed value, one weight and one score row per observation"
                .into(),
        ));
    }
    if observed.iter().any(|y| *y != 0.0 && *y != 1.0) {
        return Err(MboostrsError::Control(
            "auc_delong requires outcomes coded 0 and 1".into(),
        ));
    }
    if weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
        return Err(MboostrsError::Control(
            "auc_delong weights must be finite and nonnegative".into(),
        ));
    }
    if scores.iter().any(|s| !s.is_finite()) {
        return Err(MboostrsError::Control(
            "auc_delong scores must be finite".into(),
        ));
    }
    let positives: Vec<usize> = (0..n).filter(|&i| observed[i] == 1.0).collect();
    let negatives: Vec<usize> = (0..n).filter(|&i| observed[i] == 0.0).collect();
    let w1: f64 = positives.iter().map(|&i| weights[i]).sum();
    let w0: f64 = negatives.iter().map(|&i| weights[i]).sum();
    if w1 <= 1.0 || w0 <= 1.0 {
        return Err(MboostrsError::Control(
            "auc_delong needs total weight above one in each outcome class".into(),
        ));
    }

    let mut aucs = Vec::with_capacity(m);
    let mut placements_pos: Vec<Vec<f64>> = Vec::with_capacity(m);
    let mut placements_neg: Vec<Vec<f64>> = Vec::with_capacity(m);
    for c in 0..m {
        let score = |i: usize| scores[[i, c]];
        // Negatives sorted by score with the cumulative weight below each rank.
        let mut neg_sorted: Vec<(f64, f64)> =
            negatives.iter().map(|&j| (score(j), weights[j])).collect();
        neg_sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
        let neg_scores: Vec<f64> = neg_sorted.iter().map(|p| p.0).collect();
        let mut neg_cum = vec![0.0; neg_sorted.len() + 1];
        for (k, p) in neg_sorted.iter().enumerate() {
            neg_cum[k + 1] = neg_cum[k] + p.1;
        }
        let pos_placement: Vec<f64> = positives
            .iter()
            .map(|&i| {
                let s = score(i);
                let lo = neg_scores.partition_point(|&t| t < s);
                let hi = neg_scores.partition_point(|&t| t <= s);
                (neg_cum[lo] + 0.5 * (neg_cum[hi] - neg_cum[lo])) / w0
            })
            .collect();

        let mut pos_sorted: Vec<(f64, f64)> =
            positives.iter().map(|&i| (score(i), weights[i])).collect();
        pos_sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
        let pos_scores: Vec<f64> = pos_sorted.iter().map(|p| p.0).collect();
        let mut pos_cum = vec![0.0; pos_sorted.len() + 1];
        for (k, p) in pos_sorted.iter().enumerate() {
            pos_cum[k + 1] = pos_cum[k] + p.1;
        }
        let neg_placement: Vec<f64> = negatives
            .iter()
            .map(|&j| {
                let s = score(j);
                let lo = pos_scores.partition_point(|&t| t < s);
                let hi = pos_scores.partition_point(|&t| t <= s);
                // Positives scoring above s: total minus those at or below s,
                // with half credit for the tie block.
                let below_or_equal = pos_cum[hi];
                let tie = pos_cum[hi] - pos_cum[lo];
                let above = w1 - below_or_equal;
                (above + 0.5 * tie) / w1
            })
            .collect();

        let auc: f64 = positives
            .iter()
            .zip(&pos_placement)
            .map(|(&i, v)| weights[i] * v)
            .sum::<f64>()
            / w1;
        aucs.push(auc);
        placements_pos.push(pos_placement);
        placements_neg.push(neg_placement);
    }

    let mut covariance = ndarray::Array2::<f64>::zeros((m, m));
    for a in 0..m {
        for b in a..m {
            let mut pos_term = 0.0;
            for (k, &i) in positives.iter().enumerate() {
                pos_term += weights[i]
                    * (placements_pos[a][k] - aucs[a])
                    * (placements_pos[b][k] - aucs[b]);
            }
            let mut neg_term = 0.0;
            for (k, &j) in negatives.iter().enumerate() {
                neg_term += weights[j]
                    * (placements_neg[a][k] - aucs[a])
                    * (placements_neg[b][k] - aucs[b]);
            }
            let value = pos_term / (w1 * (w1 - 1.0)) + neg_term / (w0 * (w0 - 1.0));
            covariance[[a, b]] = value;
            covariance[[b, a]] = value;
        }
    }
    Ok((aucs, covariance))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn crps_normal_matches_the_standard_normal_value_at_zero() {
        // CRPS of N(0, 1) at y = 0 is 2/sqrt(2 pi) - 1/sqrt(pi).
        let expected =
            2.0 / (2.0 * std::f64::consts::PI).sqrt() - 1.0 / std::f64::consts::PI.sqrt();
        let got = crps_normal(&[0.0], &[0.0], &[1.0]).unwrap()[0];
        assert!((got - expected).abs() < 1e-14);
        let scaled = crps_normal(&[0.0], &[0.0], &[3.0]).unwrap()[0];
        assert!((scaled - 3.0 * expected).abs() < 1e-14);
    }

    #[test]
    fn crps_discrete_sum_matches_the_bernoulli_closed_form() {
        // Bernoulli(p): CRPS is p^2 when y = 0 and (1 - p)^2 when y = 1.
        let p = 0.3;
        let cdf = array![[1.0 - p, 1.0], [1.0 - p, 1.0]];
        let got = crps_discrete_sum(&[0.0, 1.0], cdf.view()).unwrap();
        assert!((got[0] - p * p).abs() < 1e-15);
        assert!((got[1] - (1.0 - p) * (1.0 - p)).abs() < 1e-15);
    }

    #[test]
    fn crps_cdf_quadrature_is_exact_for_the_uniform_distribution() {
        // F(t) = t on [0, 1] and y = 0.3: the integral is 0.3^3/3 + 0.7^3/3.
        let h = 0.3 / 4.0;
        let below: Vec<f64> = (0..5).map(|k| k as f64 * h).collect();
        let above: Vec<f64> = (0..5).map(|k| 0.3 + k as f64 * 0.7 / 4.0).collect();
        let got = crps_cdf_quadrature(
            ndarray::Array2::from_shape_vec((1, 5), below)
                .unwrap()
                .view(),
            ndarray::Array2::from_shape_vec((1, 5), above)
                .unwrap()
                .view(),
            &[h],
            &[0.7 / 4.0],
        )
        .unwrap()[0];
        let expected = 0.3f64.powi(3) / 3.0 + 0.7f64.powi(3) / 3.0;
        assert!((got - expected).abs() < 1e-14);
    }

    #[test]
    fn auc_delong_matches_a_hand_placement_calculation() {
        // Positives score 2 and 3; negatives score 1 and 2. Pairs: 3.5 wins of 4.
        let observed = [0.0, 1.0, 1.0, 0.0];
        let scores = array![[1.0], [2.0], [3.0], [2.0]];
        let (auc, covariance) = auc_delong(&observed, scores.view(), &[1.0; 4]).unwrap();
        assert!((auc[0] - 0.875).abs() < 1e-15);
        assert!((covariance[[0, 0]] - 0.03125).abs() < 1e-15);
    }

    #[test]
    fn auc_delong_rejects_a_class_without_weight_above_one() {
        let scores = array![[1.0], [2.0]];
        assert!(auc_delong(&[0.0, 1.0], scores.view(), &[1.0, 1.0]).is_err());
    }

    #[test]
    fn weighted_scores_match_closed_forms() {
        let p = array![[0.25], [0.75]];
        assert_eq!(
            response_score(&[0., 1.], p.view(), &[1., 3.], "brier", None)
                .unwrap()
                .0,
            0.0625
        );
        assert!(
            (response_score(&[0., 1.], p.view(), &[1., 3.], "log_loss", None)
                .unwrap()
                .0
                + 0.75f64.ln())
            .abs()
                < 1e-14
        );
        let p = array![[1.], [1.]];
        assert!(
            (response_score(&[0., 3.], p.view(), &[1., 3.], "pinball", Some(0.2))
                .unwrap()
                .0
                - 0.5)
                .abs()
                < 1e-14
        );
        assert!(
            (response_score(&[0., 3.], p.view(), &[1., 3.], "expectile", Some(0.2))
                .unwrap()
                .0
                - 0.8)
                .abs()
                < 1e-14
        );
    }
    #[test]
    fn multiclass_brier_is_sum_over_classes() {
        let p = array![[0.6, 0.3, 0.1], [0.1, 0.2, 0.7]];
        assert!(
            (response_score(&[0., 2.], p.view(), &[1., 3.], "brier", None)
                .unwrap()
                .0
                - 0.17)
                .abs()
                < 1e-14
        );
        assert!(response_score(&[0., 2.], p.view(), &[1., 3.], "mse", None).is_err());
    }
    #[test]
    fn path_skips_zero_and_preserves_first_tie() {
        let scores = array![[0., 9., 3., 4.], [0., 1., 5., 4.]];
        let (curve, best) = select_score_path(scores.view()).unwrap();
        assert_eq!(curve, vec![0., 5., 4., 4.]);
        assert_eq!(best, 2);
    }
    #[test]
    fn invalid_shapes_values_and_parameters_are_errors() {
        let p = array![[0.5], [f64::NAN]];
        assert!(response_score(&[0., 1.], p.view(), &[1., 1.], "brier", None).is_err());
        assert!(response_score(&[0., 1.], p.view(), &[1., 0.], "brier", None).is_ok());
        assert!(response_score(&[0.], p.view(), &[1.], "mse", None).is_err());
        assert!(response_score(&[0., 1.], p.view(), &[1., 1.], "pinball", Some(1.)).is_err());
        assert!(response_score(&[0., 1.], p.view(), &[1., 1.], "mse", Some(0.5)).is_err());
    }

    #[test]
    fn inverse_response_transform_uses_the_declared_original_scale() {
        let values = [0.0, 2.0f64.ln()];
        assert_eq!(
            inverse_response_transform(&values, "identity").unwrap(),
            values
        );
        assert_eq!(
            inverse_response_transform(&values, "log").unwrap(),
            vec![1.0, 2.0]
        );
        assert_eq!(
            inverse_response_transform(&values, "log1p").unwrap(),
            vec![0.0, 1.0]
        );
        assert!(inverse_response_transform(&values, "sqrt").is_err());
        assert!(inverse_response_transform(&[1000.0], "log").is_err());
    }

    #[test]
    fn streamed_scalar_path_matches_closed_form() {
        let bases = [crate::BasisInfo::ScalarBols {
            centering: 0.,
            by: None,
        }];
        let result = response_prediction_path(
            &bases,
            &[0; 40],
            &vec![vec![0.1]; 40],
            &[vec![1., 2.]],
            &[],
            &[0., 0.],
            0.1,
            &[1., 2.],
            &[1., 1.],
            &crate::Gaussian,
            "mse",
            None,
            crate::UnseenLevels::Error,
        )
        .unwrap();
        assert_eq!(result.len(), 41);
        assert!((result[0] - 2.5).abs() < 1e-14);
        assert!((result[40] - 0.9).abs() < 1e-14);
        assert!(response_prediction_path(
            &bases,
            &[1],
            &[vec![0.1]],
            &[vec![1., 2.]],
            &[],
            &[0., 0.],
            0.1,
            &[1., 2.],
            &[1., 1.],
            &crate::Gaussian,
            "mse",
            None,
            crate::UnseenLevels::Error
        )
        .is_err());
    }

    #[test]
    fn predict_over_grid_matches_hand_computed_eta() {
        use crate::BasisInfo;
        // Learner 0 is a fixed centred-at-zero linear term on `z`; learner 1 is
        // an intercept-and-slope term on `x`, which the grid replaces. Learner
        // 1 is selected twice, so its coefficients are summed before evaluation.
        let bases = [
            BasisInfo::ScalarBols {
                centering: 0.0,
                by: None,
            },
            BasisInfo::Bols {
                intercept: true,
                by: None,
            },
        ];
        let selected = [0, 1, 1];
        let coefs = vec![vec![0.5], vec![1.0, 2.0], vec![0.25, -1.0]];
        let start = [0.1, -0.2];
        let z = vec![2.0, 4.0];
        let x = vec![0.0, 0.0];
        let grid = vec![3.0, -1.0];
        let nu = 0.5;

        // Without a per-row by=, the grid evaluation adds one value to every
        // row. Learner 1's summed coefficients are [1.25, 1.0].
        let out = predict_over_grid(
            &bases,
            &selected,
            &coefs,
            &start,
            &[z.clone(), x.clone()],
            &[],
            nu,
            3,
            &[(1, grid.clone())],
            &[],
            grid.len(),
            crate::UnseenLevels::Error,
        )
        .unwrap();
        for (position, value) in grid.iter().enumerate() {
            let slope_term = 1.25 + 1.0 * value;
            for row in 0..2 {
                let expected = start[row] + nu * 0.5 * z[row] + nu * slope_term;
                assert!(
                    (out[position][row] - expected).abs() < 1e-14,
                    "grid {position} row {row}: {} vs {expected}",
                    out[position][row]
                );
            }
        }

        // A per-row by= multiplies the grid-evaluated term; the grid replaces by.
        let by_base = vec![None, Some(vec![0.0, 0.0])];
        let out = predict_over_grid(
            &bases,
            &selected,
            &coefs,
            &start,
            &[z.clone(), x.clone()],
            &by_base,
            nu,
            3,
            &[],
            &[(1, grid.clone())],
            grid.len(),
            crate::UnseenLevels::Error,
        )
        .unwrap();
        for (position, value) in grid.iter().enumerate() {
            for row in 0..2 {
                // The base by= is 0, which the grid value replaces.
                let expected = start[row] + nu * 0.5 * z[row] + nu * value * 1.25;
                assert!(
                    (out[position][row] - expected).abs() < 1e-14,
                    "by grid {position} row {row}: {} vs {expected}",
                    out[position][row]
                );
            }
        }
    }
}
