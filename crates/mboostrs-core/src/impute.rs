//! Model-based imputation of a numeric covariate's missing values.
//!
//! A penalized additive regression predicts the incomplete column from the
//! other covariates. Each numeric predictor enters as a cubic B-spline with a
//! second-difference penalty and 20 interior knots at training quantiles;
//! each factor predictor enters as a level design with a ridge penalty; each
//! predictor that is missing somewhere in training adds a 0/1 missingness
//! column. The smoothing parameter is chosen by generalized cross-validation
//! over a grid. The model is fitted on the rows where the target is observed
//! and never sees the response. A predictor that is missing on a row enters at
//! its training median for that row, with its missingness column carrying the
//! difference, as the `na_action="impute"` fill does.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

use crate::linalg::CachedCholesky;
use crate::MboostrsError;

/// Layout version written as the first entry of the parameter vector.
const PARAMS_VERSION: f64 = 1.0;
/// Interior knots per numeric predictor (the basis has `knots + 4` columns).
const INTERIOR_KNOTS: usize = 20;
/// Smoothing grid: `lambda = scale * 10^(k / 2 - 4)` for `k = 0..LAMBDA_GRID`.
const LAMBDA_GRID: usize = 17;

/// The fitted layout of one predictor.
#[derive(Clone, Debug, PartialEq)]
pub enum PredictorSpec {
    /// A numeric predictor: `fill` replaces a missing value, `lo..hi` is the
    /// training range the B-spline covers (new values are held inside it), and
    /// `interior` are the interior knots. `indicator` adds a missingness column.
    Numeric {
        /// Training median, used for a missing value.
        fill: f64,
        /// Smallest training value.
        lo: f64,
        /// Largest training value.
        hi: f64,
        /// Interior knots, strictly inside `(lo, hi)`.
        interior: Vec<f64>,
        /// Whether a missingness column is part of the design.
        indicator: bool,
    },
    /// A factor predictor with `levels` codes `0..levels`. Code 0 is the
    /// reference; a missing or unseen code (NaN, or outside `0..levels`) gives
    /// an all-zero dummy row, and `indicator` adds a missingness column.
    Factor {
        /// Number of codes.
        levels: usize,
        /// Whether a missingness column is part of the design.
        indicator: bool,
    },
}

impl PredictorSpec {
    /// Number of design columns this predictor contributes.
    fn width(&self) -> usize {
        match self {
            PredictorSpec::Numeric {
                lo,
                hi,
                interior,
                indicator,
                ..
            } => {
                let basis = if hi > lo { interior.len() + 4 } else { 0 };
                basis + usize::from(*indicator)
            }
            PredictorSpec::Factor { levels, indicator } => {
                levels.saturating_sub(1) + usize::from(*indicator)
            }
        }
    }
}

/// A fitted imputation model for one incomplete column.
#[derive(Clone, Debug, PartialEq)]
pub struct ImputationModel {
    /// Predictor layout, in the order of the input columns.
    pub predictors: Vec<PredictorSpec>,
    /// Coefficients: the intercept first, then each predictor's columns.
    pub coef: Vec<f64>,
    /// The selected smoothing parameter.
    pub lambda: f64,
    /// R-squared of the conditional mean on the training rows where the target
    /// was observed. A diagnostic only: `from_params` does not restore it, and
    /// it is NaN for a model read back from parameters.
    pub r2: f64,
}

/// Cubic B-spline basis values at `u` for knot vector `knots` (de Boor, as in
/// Piegl and Tiller's FindSpan and BasisFuns). Returns the first basis column
/// index and the four nonzero basis values.
fn spline_row(knots: &[f64], nb: usize, u: f64) -> (usize, [f64; 4]) {
    const P: usize = 3;
    let n = nb - 1;
    let span = if u >= knots[n + 1] {
        n
    } else if u <= knots[P] {
        P
    } else {
        let (mut low, mut high) = (P, n + 1);
        let mut mid = (low + high) / 2;
        while u < knots[mid] || u >= knots[mid + 1] {
            if u < knots[mid] {
                high = mid;
            } else {
                low = mid;
            }
            mid = (low + high) / 2;
        }
        mid
    };
    let mut basis = [0.0f64; 4];
    let mut left = [0.0f64; 4];
    let mut right = [0.0f64; 4];
    basis[0] = 1.0;
    for j in 1..=P {
        left[j] = u - knots[span + 1 - j];
        right[j] = knots[span + j] - u;
        let mut saved = 0.0;
        for r in 0..j {
            let denom = right[r + 1] + left[j - r];
            let temp = if denom != 0.0 { basis[r] / denom } else { 0.0 };
            basis[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        basis[j] = saved;
    }
    (span - P, basis)
}

/// Knot vector with the boundary knots repeated four times.
fn full_knots(lo: f64, hi: f64, interior: &[f64]) -> Vec<f64> {
    let mut knots = vec![lo; 4];
    knots.extend_from_slice(interior);
    knots.extend(std::iter::repeat_n(hi, 4));
    knots
}

/// Quantile of sorted `values` at probability `p` (linear interpolation).
fn quantile_sorted(values: &[f64], p: f64) -> f64 {
    let pos = p * (values.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    values[lo] + frac * (values[hi] - values[lo])
}

/// Predictor layout learned from the training columns.
fn fit_specs(x: ArrayView2<f64>, levels: &[usize]) -> Vec<PredictorSpec> {
    let mut specs = Vec::with_capacity(levels.len());
    for (j, &lev) in levels.iter().enumerate() {
        let col = x.column(j);
        let indicator = col.iter().any(|v| !v.is_finite());
        if lev == 0 {
            let mut present: Vec<f64> = col.iter().copied().filter(|v| v.is_finite()).collect();
            present.sort_by(|a, b| a.partial_cmp(b).expect("finite values"));
            if present.is_empty() {
                // A predictor missing everywhere carries no information.
                specs.push(PredictorSpec::Numeric {
                    fill: 0.0,
                    lo: 0.0,
                    hi: 0.0,
                    interior: Vec::new(),
                    indicator,
                });
                continue;
            }
            let lo = present[0];
            let hi = present[present.len() - 1];
            let mut interior: Vec<f64> = Vec::new();
            if hi > lo {
                for k in 1..=INTERIOR_KNOTS {
                    let q = quantile_sorted(&present, k as f64 / (INTERIOR_KNOTS + 1) as f64);
                    if q > lo && q < hi && interior.last().is_none_or(|&last| q > last) {
                        interior.push(q);
                    }
                }
            }
            specs.push(PredictorSpec::Numeric {
                fill: quantile_sorted(&present, 0.5),
                lo,
                hi,
                interior,
                indicator,
            });
        } else {
            specs.push(PredictorSpec::Factor {
                levels: lev,
                indicator,
            });
        }
    }
    specs
}

/// Fill one design row (the columns after the intercept) for the predictor
/// values in `x`.
fn design_row(specs: &[PredictorSpec], x: ArrayView1<f64>, out: &mut [f64]) {
    let mut at = 0usize;
    for (j, spec) in specs.iter().enumerate() {
        let v = x[j];
        match spec {
            PredictorSpec::Numeric {
                fill,
                lo,
                hi,
                interior,
                indicator,
            } => {
                let missing = !v.is_finite();
                if hi > lo {
                    let value = if missing { *fill } else { v.clamp(*lo, *hi) };
                    let knots = full_knots(*lo, *hi, interior);
                    let nb = interior.len() + 4;
                    let (first, basis) = spline_row(&knots, nb, value);
                    for (k, b) in basis.iter().enumerate() {
                        out[at + first + k] = *b;
                    }
                    at += nb;
                }
                if *indicator {
                    out[at] = f64::from(u8::from(missing));
                    at += 1;
                }
            }
            PredictorSpec::Factor { levels, indicator } => {
                let width = levels.saturating_sub(1);
                if v.is_finite() {
                    let code = v as i64;
                    if code >= 1 && (code as usize) < *levels {
                        out[at + code as usize - 1] = 1.0;
                    }
                }
                at += width;
                if *indicator {
                    out[at] = f64::from(u8::from(!v.is_finite()));
                    at += 1;
                }
            }
        }
    }
}

fn total_width(specs: &[PredictorSpec]) -> usize {
    1 + specs.iter().map(PredictorSpec::width).sum::<usize>()
}

/// Design matrix with the intercept as its first column.
fn design(specs: &[PredictorSpec], x: ArrayView2<f64>) -> Array2<f64> {
    let p = total_width(specs);
    let mut out = Array2::<f64>::zeros((x.nrows(), p));
    let mut buf = vec![0.0f64; p - 1];
    for (i, row) in x.rows().into_iter().enumerate() {
        buf.iter_mut().for_each(|b| *b = 0.0);
        design_row(specs, row, &mut buf);
        out[[i, 0]] = 1.0;
        for (k, b) in buf.iter().enumerate() {
            out[[i, k + 1]] = *b;
        }
    }
    out
}

/// Penalty matrix: a second-difference penalty on each numeric spline block,
/// a ridge penalty on factor and indicator columns, none on the intercept.
fn penalty(specs: &[PredictorSpec]) -> Array2<f64> {
    let p = total_width(specs);
    let mut pen = Array2::<f64>::zeros((p, p));
    let mut at = 1usize;
    for spec in specs {
        match spec {
            PredictorSpec::Numeric {
                lo,
                hi,
                interior,
                indicator,
                ..
            } => {
                if hi > lo {
                    let nb = interior.len() + 4;
                    // D'D for the second differences of the basis coefficients.
                    for r in 0..nb.saturating_sub(2) {
                        let stencil = [(r, 1.0), (r + 1, -2.0), (r + 2, 1.0)];
                        for &(a, da) in &stencil {
                            for &(b, db) in &stencil {
                                pen[[at + a, at + b]] += da * db;
                            }
                        }
                    }
                    at += nb;
                }
                if *indicator {
                    pen[[at, at]] += 1.0;
                    at += 1;
                }
            }
            PredictorSpec::Factor { levels, indicator } => {
                for k in 0..levels.saturating_sub(1) {
                    pen[[at + k, at + k]] += 1.0;
                }
                at += levels.saturating_sub(1);
                if *indicator {
                    pen[[at, at]] += 1.0;
                    at += 1;
                }
            }
        }
    }
    pen
}

impl ImputationModel {
    /// Fit the conditional-mean model of `target` on the predictors `x`.
    ///
    /// `target` holds the incomplete column, NaN where missing. `x` has one
    /// column per predictor. `levels[j]` is 0 for a numeric predictor, and the
    /// number of codes for a factor predictor, whose values are the codes
    /// `0.0..levels` (NaN for missing or unseen).
    pub fn fit(
        target: ArrayView1<f64>,
        x: ArrayView2<f64>,
        levels: &[usize],
    ) -> Result<Self, MboostrsError> {
        if x.ncols() != levels.len() {
            return Err(MboostrsError::Shape(format!(
                "imputation model: {} predictor columns but {} level entries",
                x.ncols(),
                levels.len()
            )));
        }
        if x.nrows() != target.len() {
            return Err(MboostrsError::Shape(
                "imputation model: target and predictors have different lengths".into(),
            ));
        }
        let rows: Vec<usize> = (0..target.len())
            .filter(|&i| target[i].is_finite())
            .collect();
        let n = rows.len();
        if n < 2 {
            return Err(MboostrsError::Solver(
                "imputation model: fewer than two observed values to fit from; drop this \
                 column or fill it yourself before fitting"
                    .into(),
            ));
        }
        let observed_x = x.select(Axis(0), &rows);
        let y = Array1::from_iter(rows.iter().map(|&i| target[i]));
        let specs = fit_specs(observed_x.view(), levels);
        let xd = design(&specs, observed_x.view());
        let pen = penalty(&specs);
        let xtx = xd.t().dot(&xd);
        let xty = xd.t().dot(&y);
        let p = xtx.nrows();
        let scale = (xtx.diag().sum() / p as f64).max(f64::MIN_POSITIVE);
        let ybar = y.sum() / n as f64;
        let tss: f64 = y.iter().map(|v| (v - ybar).powi(2)).sum();

        // (gcv, lambda, coefficients, r2) of the best grid point so far.
        let mut best: Option<(f64, f64, Array1<f64>, f64)> = None;
        for k in 0..LAMBDA_GRID {
            let lambda = scale * 10f64.powf(k as f64 / 2.0 - 4.0);
            let mut a = &xtx + &(&pen * lambda);
            for d in 0..p {
                a[[d, d]] += 1e-10 * scale;
            }
            let chol = match CachedCholesky::new(&a) {
                Ok(c) => c,
                Err(_) => continue,
            };
            let beta = chol.solve(&xty.view());
            // Effective degrees of freedom: trace of (X'X + lambda P)^-1 X'X.
            let edf: f64 = chol.solve_mat(&xtx).diag().sum();
            let fitted = xd.dot(&beta);
            let rss: f64 = fitted
                .iter()
                .zip(y.iter())
                .map(|(f, v)| (v - f).powi(2))
                .sum();
            let denom = n as f64 - edf;
            if denom.partial_cmp(&1e-8) != Some(std::cmp::Ordering::Greater) || !rss.is_finite() {
                continue;
            }
            let gcv = n as f64 * rss / (denom * denom);
            if best.as_ref().is_none_or(|b| gcv < b.0) {
                let r2 = if tss > 0.0 { 1.0 - rss / tss } else { 0.0 };
                best = Some((gcv, lambda, beta, r2));
            }
        }
        let (_, lambda, beta, r2) = best.ok_or_else(|| {
            MboostrsError::Solver(
                "imputation model: no smoothing parameter gave a finite fit".into(),
            )
        })?;
        Ok(ImputationModel {
            predictors: specs,
            coef: beta.to_vec(),
            lambda,
            r2,
        })
    }

    /// Predicted values of the target for the rows of `x`.
    pub fn predict(&self, x: ArrayView2<f64>) -> Result<Vec<f64>, MboostrsError> {
        if x.ncols() != self.predictors.len() {
            return Err(MboostrsError::Shape(format!(
                "imputation model: {} predictor columns given, {} expected",
                x.ncols(),
                self.predictors.len()
            )));
        }
        let xd = design(&self.predictors, x);
        Ok(xd.dot(&Array1::from(self.coef.clone())).to_vec())
    }

    /// The parameter vector that `from_params` reads back.
    pub fn to_params(&self) -> Vec<f64> {
        let mut out = vec![PARAMS_VERSION, self.lambda, self.predictors.len() as f64];
        for spec in &self.predictors {
            match spec {
                PredictorSpec::Numeric {
                    fill,
                    lo,
                    hi,
                    interior,
                    indicator,
                } => {
                    out.extend([
                        0.0,
                        *fill,
                        *lo,
                        *hi,
                        interior.len() as f64,
                        f64::from(u8::from(*indicator)),
                    ]);
                    out.extend(interior.iter().copied());
                }
                PredictorSpec::Factor { levels, indicator } => {
                    out.extend([1.0, *levels as f64, f64::from(u8::from(*indicator))]);
                }
            }
        }
        out.push(self.coef.len() as f64);
        out.extend(self.coef.iter().copied());
        out
    }

    /// Read a parameter vector written by `to_params`.
    pub fn from_params(params: &[f64]) -> Result<Self, MboostrsError> {
        let mut pos = 0usize;
        let mut take = |what: &str| -> Result<f64, MboostrsError> {
            let v = params.get(pos).copied().ok_or_else(|| {
                MboostrsError::Shape(format!("imputation model parameters end before {what}"))
            })?;
            pos += 1;
            Ok(v)
        };
        let version = take("the version")?;
        if version != PARAMS_VERSION {
            return Err(MboostrsError::Shape(format!(
                "imputation model parameters have version {version}, expected {PARAMS_VERSION}"
            )));
        }
        let lambda = take("lambda")?;
        let count = take("the predictor count")? as usize;
        let mut predictors = Vec::with_capacity(count);
        for _ in 0..count {
            let kind = take("a predictor kind")?;
            if kind == 0.0 {
                let fill = take("a fill")?;
                let lo = take("the lower bound")?;
                let hi = take("the upper bound")?;
                let knots = take("the knot count")? as usize;
                let indicator = take("an indicator flag")? != 0.0;
                let mut interior = Vec::with_capacity(knots);
                for _ in 0..knots {
                    interior.push(take("a knot")?);
                }
                predictors.push(PredictorSpec::Numeric {
                    fill,
                    lo,
                    hi,
                    interior,
                    indicator,
                });
            } else {
                let levels = take("a level count")? as usize;
                let indicator = take("an indicator flag")? != 0.0;
                predictors.push(PredictorSpec::Factor { levels, indicator });
            }
        }
        let width = take("the coefficient count")? as usize;
        let mut coef = Vec::with_capacity(width);
        for _ in 0..width {
            coef.push(take("a coefficient")?);
        }
        if coef.len() != total_width(&predictors) {
            return Err(MboostrsError::Shape(
                "imputation model parameters: coefficient count does not match the predictors"
                    .into(),
            ));
        }
        Ok(ImputationModel {
            predictors,
            coef,
            lambda,
            r2: f64::NAN,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn basis_is_a_partition_of_unity() {
        let interior = vec![0.2, 0.5, 0.8];
        let knots = full_knots(0.0, 1.0, &interior);
        let nb = interior.len() + 4;
        for k in 0..=20 {
            let u = k as f64 / 20.0;
            let (first, basis) = spline_row(&knots, nb, u);
            let total: f64 = basis.iter().sum();
            assert!((total - 1.0).abs() < 1e-12, "u={u}: sum {total}");
            assert!(first + 3 < nb);
        }
    }

    #[test]
    fn recovers_a_linear_conditional_mean() {
        let n = 200;
        let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let z: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
        // target = 2x + 1 + 0.5 * z, with z a factor whose reference is code 0.
        let target: Vec<f64> = x
            .iter()
            .zip(&z)
            .map(|(a, b)| 2.0 * a + 1.0 + 0.5 * b)
            .collect();
        let mut cols = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            cols[[i, 0]] = x[i];
            cols[[i, 1]] = z[i];
        }
        let model = ImputationModel::fit(arr1(&target).view(), cols.view(), &[0, 3]).unwrap();
        assert!(model.r2 > 0.999, "r2 {}", model.r2);
        let mut newx = Array2::<f64>::zeros((1, 2));
        newx[[0, 0]] = 0.5;
        newx[[0, 1]] = 2.0;
        let pred = model.predict(newx.view()).unwrap();
        assert!(
            (pred[0] - (2.0 * 0.5 + 1.0 + 1.0)).abs() < 0.05,
            "pred {}",
            pred[0]
        );
    }

    #[test]
    fn params_round_trip_predicts_identically() {
        let n = 120;
        let mut cols = Array2::<f64>::zeros((n, 2));
        let mut target = vec![0.0; n];
        for i in 0..n {
            let x = (i as f64).sin();
            cols[[i, 0]] = if i % 7 == 0 { f64::NAN } else { x };
            cols[[i, 1]] = (i % 4) as f64;
            target[i] = if i % 9 == 0 {
                f64::NAN
            } else {
                x * x + (i % 4) as f64
            };
        }
        let model = ImputationModel::fit(arr1(&target).view(), cols.view(), &[0, 4]).unwrap();
        let back = ImputationModel::from_params(&model.to_params()).unwrap();
        assert_eq!(
            model.predict(cols.view()).unwrap(),
            back.predict(cols.view()).unwrap()
        );
    }
}
