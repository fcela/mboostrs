//! Penalized GAMLSS refit with smoothing parameters chosen by the
//! Laplace-approximate restricted likelihood (Wood, Pya and Saefken 2016).
//!
//! A distributional model has `K` linear predictors `eta_1..eta_K`, one per
//! distribution parameter (mean, scale, shape...), each with its own terms.
//! Each parameter's terms get the same treatment as in the single-predictor
//! refit (sum-to-zero constraints, one smoothing parameter per penalty), and
//! all coefficients `theta = (beta_1, ..., beta_K)` are estimated jointly by
//! maximizing
//!
//! ```text
//! l(theta) - theta' S_lambda theta / 2,     l = sum_i w_i log f(y_i | eta_i1..eta_iK)
//! ```
//!
//! with Newton's method on the full Hessian, including the cross-parameter
//! blocks `X_a' diag(w_i H_i,ab) X_b` (`H_i` is the family's per-row negative
//! Hessian in the linear predictors, `LssFamily::neg_hessian`). For the
//! Newton direction each row's `H_i` has its eigenvalues floored, so every
//! step is an ascent direction; the criterion below uses the unfloored sum.
//!
//! The smoothing parameters minimize
//!
//! ```text
//! V(rho) = -2 l(theta_hat) + theta_hat' S theta_hat + log|H + S| - log|S|+
//! ```
//!
//! by BFGS. Its gradient is exact except for the change of `H` with
//! `theta_hat`, whose third derivatives are taken as a finite-difference
//! directional derivative of `neg_hessian` along `X dtheta/drho_k`.

use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};

use super::laml::{self, LamlEval};
use super::reml::{penalty_logdet, symmetric_eigen};
use super::{build_model, Model, RefitBlock, RefitControl, RefitTerm};
use crate::lss::LssFamily;
use crate::MboostrsError;

/// One distribution parameter's part of the result.
#[derive(Clone, Debug)]
pub struct LssRefitParam {
    /// Parameter name (`mu`, `sigma`, ...).
    pub name: String,
    /// Columns of the joint coefficient vector for this parameter
    /// (`start` is its intercept).
    pub start: usize,
    /// End (exclusive) of the parameter's columns.
    pub end: usize,
    /// The parameter's terms; their `start`/`end` index the joint vector.
    pub blocks: Vec<RefitBlock>,
}

/// The fitted GAMLSS refit.
#[derive(Clone, Debug)]
pub struct LssRefitResult {
    /// One entry per distribution parameter.
    pub params: Vec<LssRefitParam>,
    /// Joint coefficients: each parameter's intercept then its terms.
    pub coef: Array1<f64>,
    /// Bayesian covariance `(H + S)^-1` of `coef`, across all parameters.
    pub cov: Array2<f64>,
    /// `cov` corrected, to first order, for smoothing-parameter uncertainty.
    pub cov_corrected: Array2<f64>,
    /// Laplace-approximate REML criterion at the optimum.
    pub reml_score: f64,
    /// Internal log smoothing parameters, parameter by parameter.
    pub rho: Array1<f64>,
    /// Hessian of the criterion in `rho` (central differences of the
    /// gradient).
    pub rho_hessian: Array2<f64>,
    /// Largest absolute free gradient component at the end.
    pub gradient_norm: f64,
    /// BFGS iterations.
    pub iterations: usize,
    /// Whether BFGS converged.
    pub converged: bool,
    /// Why the BFGS iteration that produced the optimum stopped.
    pub stop_reason: &'static str,
    /// Evaluations of the criterion (each one an inner Newton solve for the
    /// coefficients), over both BFGS passes and the Hessian differences.
    pub evaluations: usize,
    /// Whether a second BFGS pass was run from the interior because some
    /// smoothing parameters ended at their upper bound.
    pub restarted: bool,
    /// BFGS iterations of that second pass (0 when it did not run).
    pub restart_iterations: usize,
    /// Wall-clock seconds the refit took.
    pub seconds: f64,
    /// Inner Newton iterations for the coefficients, summed over every
    /// criterion evaluation.
    pub inner_iterations: usize,
    /// Whether some rows' information was not positive definite at the
    /// optimum and was floored (the Laplace approximation then uses the
    /// floored matrix).
    pub information_floored: bool,
    /// Total effective degrees of freedom (intercepts included).
    pub edf_total: f64,
    /// Number of observations.
    pub n: usize,
}

struct Joint<'a> {
    models: Vec<Model>,
    /// Start of each parameter in the joint vector.
    offsets: Vec<usize>,
    /// Start of each parameter's smoothing parameters in `rho`.
    rho_offsets: Vec<usize>,
    p: usize,
    m: usize,
    family: &'a dyn LssFamily,
    y: ArrayView1<'a, f64>,
    w: ArrayView1<'a, f64>,
    /// Whether an inner solve that reaches its iteration cap counts as
    /// infeasible (see `penalized_mle`). Set by the second pass of a refit.
    strict: std::sync::atomic::AtomicBool,
    /// Inner Newton iterations of `penalized_mle` since the refit started;
    /// reported in the result so a slow refit shows where its time went.
    /// Atomic because criterion evaluations run on the rayon pool.
    inner: std::sync::atomic::AtomicUsize,
}

/// Rows per block of the weighted cross-products. The blocks are summed in
/// order, so the result does not depend on the number of threads.
const CROSS_ROWS: usize = 1024;

/// `X_a' diag(v) X_b` over the rows `start..end`, sequential.
fn weighted_cross_block(
    xa: ArrayView2<f64>,
    xb: ArrayView2<f64>,
    v: &[f64],
    start: usize,
    end: usize,
) -> Array2<f64> {
    let rows = end - start;
    let (pa, pb) = (xa.ncols(), xb.ncols());
    let a = faer::Mat::from_fn(rows, pa, |i, j| xa[[start + i, j]]);
    let b = faer::Mat::from_fn(rows, pb, |i, j| v[start + i] * xb[[start + i, j]]);
    let mut acc = faer::Mat::<f64>::zeros(pa, pb);
    faer::linalg::matmul::matmul(
        acc.as_mut(),
        a.transpose(),
        b.as_ref(),
        Some(1.0),
        1.0,
        faer::Parallelism::None,
    );
    Array2::from_shape_fn((pa, pb), |(i, j)| acc.read(i, j))
}

/// `X_a' diag(v) X_b`: one partial product per block of rows, computed on
/// the rayon pool, then summed in block order.
fn weighted_cross(xa: ArrayView2<f64>, xb: ArrayView2<f64>, v: &[f64]) -> Array2<f64> {
    use rayon::prelude::*;
    let (n, pa) = xa.dim();
    let pb = xb.ncols();
    let starts: Vec<usize> = (0..n).step_by(CROSS_ROWS).collect();
    let parts: Vec<Array2<f64>> = starts
        .par_iter()
        .map(|&start| weighted_cross_block(xa, xb, v, start, (start + CROSS_ROWS).min(n)))
        .collect();
    let mut acc = Array2::<f64>::zeros((pa, pb));
    for part in &parts {
        acc += part;
    }
    acc
}

fn cholesky_inverse(a: &Array2<f64>) -> Option<(Array2<f64>, f64)> {
    use faer::linalg::solvers::Cholesky;
    use faer::prelude::SolverCore;
    let p = a.nrows();
    let fa = faer::Mat::from_fn(p, p, |i, j| a[[i, j]]);
    let chol = Cholesky::try_new(fa.as_ref(), faer::Side::Lower).ok()?;
    let l = chol.compute_l();
    let mut logdet = 0.0;
    for i in 0..p {
        let d = l.read(i, i);
        if d.is_nan() || d <= 0.0 {
            return None;
        }
        logdet += 2.0 * d.ln();
    }
    let inv = chol.inverse();
    Some((
        Array2::from_shape_fn((p, p), |(i, j)| inv.read(i, j)),
        logdet,
    ))
}

/// Floor the eigenvalues of a small symmetric matrix at `floor`; returns
/// whether anything changed.
fn floor_psd(block: &mut [f64], k: usize) -> bool {
    // Fast path: positive definite by an attempted Cholesky.
    let mut l = vec![0.0; k * k];
    let mut ok = true;
    'outer: for i in 0..k {
        for j in 0..=i {
            let mut sum = block[i * k + j];
            for r in 0..j {
                sum -= l[i * k + r] * l[j * k + r];
            }
            if i == j {
                if sum <= 1e-12 * (1.0 + block[i * k + i].abs()) {
                    ok = false;
                    break 'outer;
                }
                l[i * k + i] = sum.sqrt();
            } else {
                l[i * k + j] = sum / l[j * k + j];
            }
        }
    }
    if ok {
        return false;
    }
    let m = Array2::from_shape_fn((k, k), |(i, j)| block[i * k + j]);
    let (vals, vecs) = symmetric_eigen(&m);
    let top = vals.iter().fold(0.0f64, |a, v| a.max(v.abs())).max(1e-12);
    let floored = vals.mapv(|v| v.max(1e-6 * top));
    for i in 0..k {
        for j in 0..k {
            block[i * k + j] = (0..k)
                .map(|r| vecs[[i, r]] * floored[r] * vecs[[j, r]])
                .sum();
        }
    }
    true
}

impl Joint<'_> {
    fn k(&self) -> usize {
        self.models.len()
    }

    fn etas(&self, theta: &Array1<f64>) -> Vec<Array1<f64>> {
        self.models
            .iter()
            .zip(&self.offsets)
            .map(|(model, &off)| model.x.dot(&theta.slice(s![off..off + model.x.ncols()])))
            .collect()
    }

    /// `(S theta, theta' S theta)`.
    fn penalty(&self, theta: &Array1<f64>, rho: &Array1<f64>) -> (Array1<f64>, f64) {
        let mut st = Array1::<f64>::zeros(self.p);
        for (q, model) in self.models.iter().enumerate() {
            for (k, pen) in model.penalties.iter().enumerate() {
                let block = &model.blocks[pen.block];
                let r = self.offsets[q] + block.start..self.offsets[q] + block.end;
                let lambda = rho[self.rho_offsets[q] + k].exp();
                let b = pen.matrix.dot(&theta.slice(s![r.clone()])) * lambda;
                let mut dst = st.slice_mut(s![r]);
                dst += &b;
            }
        }
        let value = theta.dot(&st);
        (st, value)
    }

    fn add_penalty(&self, a: &mut Array2<f64>, rho: &Array1<f64>) {
        for (q, model) in self.models.iter().enumerate() {
            if let Some(fixed) = &model.fixed_penalty {
                let r = self.offsets[q]..self.offsets[q] + fixed.nrows();
                let mut view = a.slice_mut(s![r.clone(), r]);
                view += fixed;
            }
            for (k, pen) in model.penalties.iter().enumerate() {
                let block = &model.blocks[pen.block];
                let r = self.offsets[q] + block.start..self.offsets[q] + block.end;
                let mut view = a.slice_mut(s![r.clone(), r]);
                view.scaled_add(rho[self.rho_offsets[q] + k].exp(), &pen.matrix);
            }
        }
    }

    fn rho_part(&self, rho: &Array1<f64>, q: usize) -> Array1<f64> {
        let n = self.models[q].penalties.len();
        rho.slice(s![self.rho_offsets[q]..self.rho_offsets[q] + n])
            .to_owned()
    }

    /// `log|S|+` and its gradient.
    fn logdet_penalty(&self, rho: &Array1<f64>) -> Option<(f64, Array1<f64>)> {
        let mut value = 0.0;
        let mut grad = Array1::<f64>::zeros(self.m);
        for (q, model) in self.models.iter().enumerate() {
            let (v, g, _) = penalty_logdet(model, &self.rho_part(rho, q), true)?;
            value += v;
            let o = self.rho_offsets[q];
            grad.slice_mut(s![o..o + g.len()]).assign(&g);
        }
        Some((value, grad))
    }

    /// Weighted per-row negative Hessian (`n * K * K`).
    fn row_information(&self, etas: &[Array1<f64>]) -> Vec<f64> {
        let k = self.k();
        let n = self.y.len() / self.family.response_span();
        let mut h = vec![0.0; n * k * k];
        self.family
            .neg_hessian(self.y.as_slice().expect("contiguous y"), etas, &mut h);
        for i in 0..n {
            for v in &mut h[i * k * k..(i + 1) * k * k] {
                *v *= self.w[i];
            }
        }
        h
    }

    /// The joint `X' W X` from per-row information blocks.
    fn information(&self, rows: &[f64]) -> Array2<f64> {
        let k = self.k();
        let n = rows.len() / (k * k);
        let mut h = Array2::<f64>::zeros((self.p, self.p));
        for a in 0..k {
            for b in a..k {
                let v: Vec<f64> = (0..n).map(|i| rows[i * k * k + a * k + b]).collect();
                let block = weighted_cross(self.models[a].x.view(), self.models[b].x.view(), &v);
                let (ra, rb) = (
                    self.offsets[a]..self.offsets[a] + self.models[a].x.ncols(),
                    self.offsets[b]..self.offsets[b] + self.models[b].x.ncols(),
                );
                h.slice_mut(s![ra.clone(), rb.clone()]).assign(&block);
                if a != b {
                    h.slice_mut(s![rb, ra]).assign(&block.t());
                }
            }
        }
        h
    }

    fn loglik(&self, etas: &[Array1<f64>]) -> f64 {
        -self.family.risk(
            self.y.as_slice().expect("contiguous y"),
            etas,
            self.w.as_slice().expect("contiguous w"),
        )
    }

    fn score(&self, etas: &[Array1<f64>], st: &Array1<f64>) -> Array1<f64> {
        let n = etas[0].len();
        let ones = vec![1.0; n];
        let mut u = vec![0.0; n];
        let mut g = Array1::<f64>::zeros(self.p);
        for (q, model) in self.models.iter().enumerate() {
            self.family.ngradient(
                q,
                self.y.as_slice().expect("contiguous y"),
                etas,
                &ones,
                &mut u,
            );
            let wu = Array1::from_shape_fn(n, |i| self.w[i] * u[i]);
            let gq = model.x.t().dot(&wu);
            let o = self.offsets[q];
            g.slice_mut(s![o..o + gq.len()]).assign(&gq);
        }
        g - st
    }

    /// Penalized MLE at `rho` by Newton's method from `theta0`.
    fn penalized_mle(&self, rho: &Array1<f64>, theta0: &Array1<f64>) -> Option<Array1<f64>> {
        let k = self.k();
        let mut theta = theta0.clone();
        let mut etas = self.etas(&theta);
        let (mut st, mut pen) = self.penalty(&theta, rho);
        let mut objective = self.loglik(&etas) - 0.5 * pen;
        if !objective.is_finite() {
            return None;
        }
        // Consecutive iterations whose gain was below the objective's own
        // rounding. Near the optimum the decrement test can stay just above
        // its tolerance while every step gains nothing measurable; without
        // this the loop ran to its cap and one criterion evaluation took
        // minutes (claims-discovery benchmark, replicate H2).
        let mut stalled = 0;
        for _ in 0..200 {
            self.inner
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // Newton on the observed information when the penalized sum is
            // positive definite; otherwise on rows floored to positive
            // definite, which still gives an ascent direction.
            let rows = self.row_information(&etas);
            let mut a = self.information(&rows);
            self.add_penalty(&mut a, rho);
            let ainv = match cholesky_inverse(&a) {
                Some((inv, _)) => inv,
                None => {
                    let mut fixed = rows;
                    for block in fixed.chunks_mut(k * k) {
                        floor_psd(block, k);
                    }
                    let mut af = self.information(&fixed);
                    self.add_penalty(&mut af, rho);
                    let ridge = 1e-8 * (af.diag().sum() / self.p as f64).max(1e-12);
                    af.diag_mut().mapv_inplace(|v| v + ridge);
                    cholesky_inverse(&af)?.0
                }
            };
            let g = self.score(&etas, &st);
            let step = ainv.dot(&g);
            // Newton decrement: the predicted gain of a full step.
            let decrement = g.dot(&step);
            if decrement.abs() < 1e-11 * (1.0 + objective.abs()) {
                return Some(theta);
            }
            let mut alpha = 1.0;
            let mut moved = false;
            for _ in 0..40 {
                let trial = &theta + &(&step * alpha);
                let te = self.etas(&trial);
                let (tst, tpen) = self.penalty(&trial, rho);
                let obj = self.loglik(&te) - 0.5 * tpen;
                if obj.is_finite() && obj >= objective - 1e-12 * objective.abs() {
                    if obj - objective <= 1e-12 * (1.0 + objective.abs()) {
                        stalled += 1;
                    } else {
                        stalled = 0;
                    }
                    theta = trial;
                    etas = te;
                    st = tst;
                    pen = tpen;
                    objective = obj;
                    moved = true;
                    break;
                }
                alpha *= 0.5;
            }
            if !moved || stalled >= 3 {
                return Some(theta);
            }
        }
        // The cap was reached before the Newton decrement met its tolerance.
        // The penalized likelihood can have no finite maximum at a small
        // smoothing parameter (a smooth on a covariate the data cannot
        // identify, such as a null term in sigma): the objective keeps rising
        // while the coefficients grow. In strict mode such a `rho` is
        // infeasible, so the criterion is not evaluated there. The default
        // pass returns the point as it always has, so fits that converge
        // with it are unchanged.
        let _ = pen;
        if self.strict.load(std::sync::atomic::Ordering::Relaxed) {
            return None;
        }
        Some(theta)
    }

    /// The criterion, its gradient, and the pieces the result needs.
    fn evaluate(&self, rho: &Array1<f64>, theta0: &Array1<f64>) -> Option<(LamlEval, bool)> {
        let k = self.k();
        let theta = self.penalized_mle(rho, theta0)?;
        let etas = self.etas(&theta);
        let (_, pen) = self.penalty(&theta, rho);
        // The criterion uses the observed information summed over rows. A
        // single row's information need not be positive definite (for the
        // Gaussian location-scale family it never is), but the sum is at a
        // maximum. Only if the penalized sum is not positive definite are the
        // rows floored, and the result flagged.
        let rows = self.row_information(&etas);
        let mut floored = false;
        let mut h = self.information(&rows);
        let mut a = h.clone();
        self.add_penalty(&mut a, rho);
        let (ainv, logdet_a) = match cholesky_inverse(&a) {
            Some(found) => found,
            None => {
                let mut fixed = rows.clone();
                for block in fixed.chunks_mut(k * k) {
                    floor_psd(block, k);
                }
                floored = true;
                h = self.information(&fixed);
                a = h.clone();
                self.add_penalty(&mut a, rho);
                cholesky_inverse(&a)?
            }
        };
        let (logdet_s, lds_grad) = self.logdet_penalty(rho)?;
        let value = -2.0 * self.loglik(&etas) + pen + logdet_a - logdet_s;
        if !value.is_finite() {
            return None;
        }
        let n = etas[0].len();
        // C_i[a][b] = x_ia' A^-1_ab x_ib for every row and parameter pair.
        // Each (parameter pair, block of rows) is an independent task.
        use rayon::prelude::*;
        const ROW_BLOCK: usize = 1024;
        let pairs: Vec<(usize, usize)> = (0..k)
            .flat_map(|qa| (qa..k).map(move |qb| (qa, qb)))
            .collect();
        let tasks: Vec<(usize, usize, usize)> = (0..pairs.len())
            .flat_map(|pi| {
                (0..n)
                    .step_by(ROW_BLOCK)
                    .map(move |start| (pi, start, (start + ROW_BLOCK).min(n)))
            })
            .collect();
        let task_values: Vec<Vec<f64>> = tasks
            .par_iter()
            .map(|&(pi, start, end)| {
                let (qa, qb) = pairs[pi];
                let ra = self.offsets[qa]..self.offsets[qa] + self.models[qa].x.ncols();
                let rb = self.offsets[qb]..self.offsets[qb] + self.models[qb].x.ncols();
                let xa = self.models[qa].x.slice(s![start..end, ..]);
                let xb = self.models[qb].x.slice(s![start..end, ..]);
                let m = super::reml::matmul(xa, ainv.slice(s![ra, rb]));
                (0..end - start).map(|i| m.row(i).dot(&xb.row(i))).collect()
            })
            .collect();
        let mut c = vec![0.0; n * k * k];
        for (&(pi, start, _), values) in tasks.iter().zip(&task_values) {
            let (qa, qb) = pairs[pi];
            for (i, &v) in values.iter().enumerate() {
                c[(start + i) * k * k + qa * k + qb] = v;
                c[(start + i) * k * k + qb * k + qa] = v;
            }
        }
        let max_eta = etas
            .iter()
            .flat_map(|e| e.iter())
            .fold(0.0f64, |m, v| m.max(v.abs()));
        // Each smoothing parameter's derivative is independent of the others,
        // so the components are computed on the rayon pool and then stored
        // by index.
        let slots: Vec<(usize, usize)> = self
            .models
            .iter()
            .enumerate()
            .flat_map(|(q, model)| (0..model.penalties.len()).map(move |kk| (q, kk)))
            .collect();
        let components: Vec<f64> = slots
            .par_iter()
            .map(|&(q, kk)| {
                let model = &self.models[q];
                let pen = &model.penalties[kk];
                let g = self.rho_offsets[q] + kk;
                let block = &model.blocks[pen.block];
                let r = self.offsets[q] + block.start..self.offsets[q] + block.end;
                let lambda = rho[g].exp();
                let bk = pen.matrix.dot(&theta.slice(s![r.clone()])) * lambda;
                let explicit = theta.slice(s![r.clone()]).dot(&bk);
                let trace = (&ainv.slice(s![r.clone(), r.clone()]) * &pen.matrix).sum() * lambda;
                // dtheta/drho = -A^-1 b_k; the implied change of each eta.
                let dtheta = ainv.slice(s![.., r]).dot(&bk).mapv(|v| -v);
                let deta = self.etas(&dtheta);
                let size = deta
                    .iter()
                    .flat_map(|e| e.iter())
                    .fold(0.0f64, |m, v| m.max(v.abs()));
                let mut weight_change = 0.0;
                if size > 0.0 {
                    let eps = 1e-5 * (1.0 + max_eta) / size;
                    let up: Vec<Array1<f64>> = etas
                        .iter()
                        .zip(&deta)
                        .map(|(e, d)| e + &(d * eps))
                        .collect();
                    let down: Vec<Array1<f64>> = etas
                        .iter()
                        .zip(&deta)
                        .map(|(e, d)| e - &(d * eps))
                        .collect();
                    let (hu, hd) = (self.row_information(&up), self.row_information(&down));
                    for idx in 0..n * k * k {
                        weight_change += (hu[idx] - hd[idx]) / (2.0 * eps) * c[idx];
                    }
                }
                explicit + trace + weight_change - lds_grad[g]
            })
            .collect();
        let mut grad = Array1::<f64>::zeros(self.m);
        for (&(q, kk), value) in slots.iter().zip(components) {
            grad[self.rho_offsets[q] + kk] = value;
        }
        Some((
            LamlEval {
                value,
                grad,
                beta: theta,
                ainv,
                xtwx: h,
            },
            floored,
        ))
    }
}

/// Refit a distributional model: `terms[q]` are the terms of parameter `q`,
/// `param_names[q]` its name. `start_etas`, when given, are the boosted
/// fit's linear predictors, projected onto the refit's bases as the start of
/// Newton's method.
pub fn try_gamlss_refit(
    terms: Vec<Vec<RefitTerm>>,
    param_names: Vec<String>,
    y: ArrayView1<f64>,
    weights: Option<ArrayView1<f64>>,
    family: &dyn LssFamily,
    start_etas: Option<&[Array1<f64>]>,
    control: &RefitControl,
) -> Result<LssRefitResult, MboostrsError> {
    let k = family.n_params();
    if terms.len() != k || param_names.len() != k {
        return Err(MboostrsError::Shape(format!(
            "{} has {k} parameters; got terms for {} and {} names",
            family.name(),
            terms.len(),
            param_names.len()
        )));
    }
    family.check_y(y)?;
    let n = y.len() / family.response_span();
    let w = match weights {
        Some(w) if w.len() == n => w.to_owned(),
        Some(w) => {
            return Err(MboostrsError::Shape(format!(
                "weights have {} entries for {n} observations",
                w.len()
            )))
        }
        None => Array1::ones(n),
    };
    let y = y.to_owned();
    let mut models = Vec::with_capacity(k);
    for q_terms in terms {
        if q_terms.is_empty() {
            // An intercept-only parameter: a model with no blocks.
            models.push(Model {
                x: Array2::ones((n, 1)),
                full_p: 1,
                blocks: Vec::new(),
                penalties: Vec::new(),
                null_dim: 1,
                fixed_penalty: None,
                unidentified: 0,
                unidentified_terms: Vec::new(),
            });
        } else {
            models.push(build_model(q_terms, w.view(), n)?);
        }
    }
    let mut offsets = Vec::with_capacity(k);
    let mut rho_offsets = Vec::with_capacity(k);
    let (mut p, mut m) = (0, 0);
    for model in &models {
        offsets.push(p);
        rho_offsets.push(m);
        p += model.x.ncols();
        m += model.penalties.len();
    }
    let joint = Joint {
        models,
        offsets,
        rho_offsets,
        p,
        m,
        family,
        y: y.view(),
        w: w.view(),
        strict: std::sync::atomic::AtomicBool::new(false),
        inner: std::sync::atomic::AtomicUsize::new(0),
    };

    // Start: the boosted predictors projected on each parameter's basis, or
    // the family's offsets as intercepts.
    let mut theta0 = Array1::<f64>::zeros(p);
    for q in 0..k {
        let model = &joint.models[q];
        let o = joint.offsets[q];
        match start_etas {
            Some(etas) if etas.len() == k && etas[q].len() == n => {
                let x = &model.x;
                let mut xtx = super::reml::weighted_gram(x.view(), w.view());
                let ridge = 1e-8 * (xtx.diag().sum() / x.ncols() as f64).max(1e-12);
                xtx.diag_mut().mapv_inplace(|v| v + ridge);
                let rhs = x.t().dot(&(&etas[q] * &w));
                let (inv, _) = cholesky_inverse(&xtx).ok_or_else(|| {
                    MboostrsError::Solver("cannot project the start onto the basis".into())
                })?;
                theta0
                    .slice_mut(s![o..o + x.ncols()])
                    .assign(&inv.dot(&rhs));
            }
            _ => {
                let prior: Vec<f64> = (0..q).map(|j| theta0[joint.offsets[j]]).collect();
                theta0[o] = family.offset(
                    q,
                    y.as_slice().expect("contiguous y"),
                    w.as_slice().expect("contiguous w"),
                    &prior,
                );
            }
        }
    }

    let started = std::time::Instant::now();
    joint.inner.store(0, std::sync::atomic::Ordering::Relaxed);
    let floored_flag = std::cell::Cell::new(false);
    let evaluations = std::cell::Cell::new(0usize);
    let bound = control.rho_bound;
    // One pass over the smoothing parameters, ending with the criterion at
    // the optimum. The default pass starts at rho = 0 and keeps the point of
    // an inner solve that hit its cap, as it always has. The strict pass
    // starts at the first of rho = 0, 2, 4, ... (up to the bound) whose
    // penalized likelihood has a finite maximum, and counts capped solves as
    // infeasible. The strict pass runs only when the default pass fails.
    let pass =
        |strict: bool| -> Result<(laml::LamlOptimum, bool, usize, LamlEval, bool), MboostrsError> {
            joint
                .strict
                .store(strict, std::sync::atomic::Ordering::Relaxed);
            let evaluate = |r: &Array1<f64>, t: &Array1<f64>| {
                evaluations.set(evaluations.get() + 1);
                joint.evaluate(r, t).map(|(e, f)| {
                    floored_flag.set(f);
                    e
                })
            };
            let solve = |start: &Array1<f64>| {
                laml::optimize(
                    &evaluate,
                    start,
                    &Array2::eye(m),
                    &theta0,
                    bound,
                    control.max_newton,
                    control.newton_tol,
                )
            };
            let mut found = solve(&Array1::zeros(m));
            let mut s = 2.0;
            while strict && found.is_none() && s <= bound {
                found = solve(&Array1::from_elem(m, s));
                s += 2.0;
            }
            let mut opt = found.ok_or_else(|| {
                let smooth: Vec<String> = joint
                    .models
                    .iter()
                    .zip(&param_names)
                    .filter(|(model, _)| !model.penalties.is_empty())
                    .map(|(model, name)| {
                        let terms: Vec<&str> =
                            model.blocks.iter().map(|b| b.name.as_str()).collect();
                        format!("{name} ({})", terms.join(", "))
                    })
                    .collect();
                MboostrsError::Solver(format!(
                    "the GAMLSS refit has no finite penalized maximum at any starting smoothing \
                 parameter up to {bound}; the smooth terms are {}. A smooth term the data \
                 cannot identify makes the penalized likelihood unbounded at small smoothing: \
                 remove that term from its parameter's formula, or lower its df",
                    smooth.join("; ")
                ))
            })?;
            // Restart smoothing parameters that ended at their upper bound from
            // the interior; keep the lower criterion (as in the single-predictor
            // refit).
            let at_bound: Vec<usize> = (0..m).filter(|&j| opt.rho[j] >= bound - 1e-6).collect();
            let mut restarted = false;
            let mut restart_iterations = 0;
            if !at_bound.is_empty() {
                restarted = true;
                let mut restart = opt.rho.clone();
                for &j in &at_bound {
                    restart[j] = 0.0;
                }
                if let Some(alt) = laml::optimize(
                    &evaluate,
                    &restart,
                    &Array2::eye(m),
                    &opt.eval.beta,
                    bound,
                    control.max_newton,
                    control.newton_tol,
                ) {
                    restart_iterations = alt.iterations;
                    if alt.eval.value < opt.eval.value {
                        opt = alt;
                    }
                }
            }
            // Re-evaluate at the optimum to record the floored flag for it.
            let (final_eval, floored) =
                joint.evaluate(&opt.rho, &opt.eval.beta).ok_or_else(|| {
                    MboostrsError::Solver("the refit optimum cannot be re-evaluated".into())
                })?;
            Ok((opt, restarted, restart_iterations, final_eval, floored))
        };
    // The default pass is final only when it converged. A default pass that
    // stops short (for example `no_descent`) can instead end at a point whose
    // re-evaluation succeeds on one CPU and fails on another, because the
    // criterion is evaluated at the edge of what the inner solve can resolve;
    // which of the two happened then decided whether the strict pass ran at
    // all. The strict pass now runs whenever the default pass did not
    // converge, and the converged (or else lower-criterion) result is kept.
    let default_pass = pass(false);
    let (opt, restarted, restart_iterations, final_eval, floored) = match default_pass {
        Ok(done) if done.0.converged => done,
        other => match (other, pass(true)) {
            (Ok(default), Ok(strict)) => {
                if strict.0.converged || strict.3.value < default.3.value {
                    strict
                } else {
                    joint
                        .strict
                        .store(false, std::sync::atomic::Ordering::Relaxed);
                    default
                }
            }
            (Ok(default), Err(_)) => {
                joint
                    .strict
                    .store(false, std::sync::atomic::Ordering::Relaxed);
                default
            }
            (Err(_), strict) => strict?,
        },
    };

    // Hessian of V in rho by central differences of the gradient.
    // Each row of the difference quotient is an independent pair of
    // criterion evaluations, so the rows are computed on the rayon pool.
    use rayon::prelude::*;
    let mut hess = Array2::<f64>::zeros((m, m));
    evaluations.set(evaluations.get() + 2 * m);
    let rows: Vec<Option<Array1<f64>>> = (0..m)
        .into_par_iter()
        .map(|j| {
            let h = 1e-4;
            let mut up = opt.rho.clone();
            let mut down = opt.rho.clone();
            up[j] += h;
            down[j] -= h;
            match (
                joint.evaluate(&up, &final_eval.beta),
                joint.evaluate(&down, &final_eval.beta),
            ) {
                (Some((gu, _)), Some((gd, _))) => Some((&gu.grad - &gd.grad) / (2.0 * h)),
                _ => None,
            }
        })
        .collect();
    for (j, row) in rows.iter().enumerate() {
        if let Some(row) = row {
            hess.row_mut(j).assign(row);
        }
    }
    let hess = (&hess + &hess.t()) / 2.0;

    let cov = final_eval.ainv.clone();
    let cov_corrected = corrected(&joint, &final_eval, &opt.rho, &hess, bound);
    let influence: Vec<f64> = (0..p)
        .map(|i| final_eval.ainv.row(i).dot(&final_eval.xtwx.column(i)))
        .collect();
    // Each parameter's start in the full-width joint vector, and the full
    // position of every fitted column (a side constraint may have removed
    // some; they are reported with zero coefficient and variance).
    let mut full_offsets = Vec::with_capacity(k);
    let mut positions = Vec::with_capacity(p);
    let mut full_p = 0;
    for model in &joint.models {
        full_offsets.push(full_p);
        positions.extend(model.full_positions().into_iter().map(|c| full_p + c));
        full_p += model.full_p;
    }
    let params = joint
        .models
        .iter()
        .enumerate()
        .map(|(q, model)| {
            let o = joint.offsets[q];
            let fo = full_offsets[q];
            LssRefitParam {
                name: param_names[q].clone(),
                start: fo,
                end: fo + model.full_p,
                blocks: model
                    .blocks
                    .iter()
                    .map(|b| RefitBlock {
                        name: b.name.clone(),
                        start: fo + b.full_start,
                        end: fo + b.full_end,
                        householder: b.householder.clone(),
                        has_by: b.has_by,
                        rank: b.rank,
                        edf: influence[o + b.start..o + b.end].iter().sum(),
                        lambda: b
                            .penalty_ids
                            .iter()
                            .map(|&j| {
                                model.penalties[j].scale * opt.rho[joint.rho_offsets[q] + j].exp()
                            })
                            .collect(),
                        test: None,
                        concurvity: None,
                    })
                    .collect(),
            }
        })
        .collect();
    Ok(LssRefitResult {
        params,
        coef: super::expand_vector(&final_eval.beta, &positions, full_p),
        cov: super::expand_matrix(&cov, &positions, full_p),
        cov_corrected: super::expand_matrix(&cov_corrected, &positions, full_p),
        reml_score: final_eval.value,
        rho: opt.rho.clone(),
        rho_hessian: hess,
        gradient_norm: opt.gradient_norm,
        iterations: opt.iterations,
        converged: opt.converged,
        stop_reason: opt.stop,
        evaluations: evaluations.get(),
        restarted,
        restart_iterations,
        seconds: started.elapsed().as_secs_f64(),
        inner_iterations: joint.inner.load(std::sync::atomic::Ordering::Relaxed),
        information_floored: floored || floored_flag.get(),
        edf_total: influence.iter().sum(),
        n,
    })
}

/// One term's raw design at new rows, with its `by=` values and whether an
/// all-zero row (an unseen group) has no effect.
pub struct NewRowsTerm {
    /// The learner's raw design at the new rows.
    pub raw: Array2<f64>,
    /// The `by=` values at the new rows, for a varying-coefficient term.
    pub by: Option<Array1<f64>>,
    /// Give all-zero raw rows zero effect (a random-effect level the refit
    /// never saw).
    pub zero_rows_have_no_effect: bool,
}

/// A linear predictor at new rows and its standard error, as `(eta, se)`.
pub type Predictor = (Array1<f64>, Array1<f64>);

impl LssRefitResult {
    /// Each parameter's design at new rows: an intercept column, then its
    /// constrained terms in the columns `start..end` of the joint vector
    /// (relative to the parameter's own first column).
    fn parameter_designs(
        &self,
        terms: &[Vec<NewRowsTerm>],
        intercept: bool,
    ) -> Result<Vec<Array2<f64>>, MboostrsError> {
        let k = self.params.len();
        if terms.len() != k {
            return Err(MboostrsError::Shape(format!(
                "terms for {} parameters, expected {k}",
                terms.len()
            )));
        }
        let n_rows = match terms.iter().flatten().next() {
            Some(t) => t.raw.nrows(),
            None => {
                return Err(MboostrsError::Control(
                    "no terms: the new rows' count is unknown".into(),
                ))
            }
        };
        let mut designs = Vec::with_capacity(k);
        for (q, param) in self.params.iter().enumerate() {
            if terms[q].len() != param.blocks.len() {
                return Err(MboostrsError::Shape(format!(
                    "parameter {} has {} terms, got {}",
                    param.name,
                    param.blocks.len(),
                    terms[q].len()
                )));
            }
            let mut x = Array2::<f64>::zeros((n_rows, param.end - param.start));
            if intercept {
                x.column_mut(0).fill(1.0);
            }
            for (block, term) in param.blocks.iter().zip(&terms[q]) {
                if term.raw.nrows() != n_rows {
                    return Err(MboostrsError::Shape(
                        "terms differ in their row counts".into(),
                    ));
                }
                let xb = super::constrained_design(
                    term.raw.view(),
                    block.householder.view(),
                    block.end - block.start,
                    block.has_by,
                    term.by.as_ref().map(|b| b.view()),
                    term.zero_rows_have_no_effect,
                )?;
                x.slice_mut(s![.., block.start - param.start..block.end - param.start])
                    .assign(&xb);
            }
            designs.push(x);
        }
        Ok(designs)
    }

    /// The linear predictor of every distribution parameter at new rows, with
    /// its standard error from the covariance of that parameter's own
    /// coefficients (intercept and terms of the parameter). Returns one
    /// `(eta, se)` pair per parameter, in `params` order; `terms` is as in
    /// [`parameter_draws`](Self::parameter_draws).
    pub fn parameter_predictors(
        &self,
        terms: &[Vec<NewRowsTerm>],
        corrected: bool,
    ) -> Result<Vec<Predictor>, MboostrsError> {
        let designs = self.parameter_designs(terms, true)?;
        let cov = if corrected {
            &self.cov_corrected
        } else {
            &self.cov
        };
        self.params
            .iter()
            .zip(&designs)
            .map(|(param, x)| {
                let r = param.start..param.end;
                super::predictor_with_se(
                    x,
                    self.coef.slice(s![r.clone()]),
                    cov.slice(s![r.clone(), r]),
                )
            })
            .collect()
    }

    /// The average change in each distribution parameter's linear predictor
    /// from the rows `lo` to the rows `hi`, as one `(estimate, se)` pair per
    /// parameter in `params` order.
    ///
    /// `hi[q]` and `lo[q]` hold parameter `q`'s terms as for
    /// [`parameter_predictors`](Self::parameter_predictors), with the same row
    /// count in both sets. Each pair is the contrast of
    /// [`LssRefitResult`]'s parameter alone, with the standard error from that
    /// parameter's own covariance block, as `parameter_predictors` gives it.
    pub fn parameter_contrasts(
        &self,
        hi: &[Vec<NewRowsTerm>],
        lo: &[Vec<NewRowsTerm>],
        corrected: bool,
    ) -> Result<Vec<(f64, f64)>, MboostrsError> {
        let x_hi = self.parameter_designs(hi, false)?;
        let x_lo = self.parameter_designs(lo, false)?;
        let cov = if corrected {
            &self.cov_corrected
        } else {
            &self.cov
        };
        self.params
            .iter()
            .zip(x_hi.iter().zip(&x_lo))
            .map(|(param, (a, b))| {
                if a.nrows() != b.nrows() || a.nrows() == 0 {
                    return Err(MboostrsError::Shape(format!(
                        "the contrast of parameter {} needs the same nonzero row count in both \
                         row sets, got {} and {}",
                        param.name,
                        a.nrows(),
                        b.nrows()
                    )));
                }
                let d = (a - b).sum_axis(ndarray::Axis(0)) / a.nrows() as f64;
                let d = d.insert_axis(ndarray::Axis(0));
                let r = param.start..param.end;
                let (eta, se) = super::predictor_with_se(
                    &d,
                    self.coef.slice(s![r.clone()]),
                    cov.slice(s![r.clone(), r]),
                )?;
                Ok((eta[0], se[0]))
            })
            .collect()
    }

    /// Draws of every distribution parameter at new rows from the
    /// approximate posterior of the joint coefficients, `N(coef, cov)` (or
    /// the corrected covariance), on each parameter's natural scale.
    ///
    /// `terms[q]` holds parameter `q`'s terms at the new rows, in the order
    /// of `params[q].blocks`. Returns one `n_draws x n_rows` matrix per
    /// parameter. A quantity that combines parameters, such as a centile of
    /// the response distribution, has its posterior draws computed from
    /// these, and an interval from their quantiles; the draws carry the
    /// correlation between the parameters' coefficients.
    pub fn parameter_draws(
        &self,
        family: &dyn LssFamily,
        terms: &[Vec<NewRowsTerm>],
        n_draws: usize,
        seed: u64,
        corrected: bool,
    ) -> Result<Vec<Array2<f64>>, MboostrsError> {
        use rand::RngExt;
        use rand::SeedableRng;
        let k = self.params.len();
        if terms.len() != k {
            return Err(MboostrsError::Shape(format!(
                "terms for {} parameters, expected {k}",
                terms.len()
            )));
        }
        let designs = self.parameter_designs(terms, true)?;
        let n_rows = designs.first().map_or(0, |x| x.nrows());
        let cov = if corrected {
            &self.cov_corrected
        } else {
            &self.cov
        };
        // A lower factor of the covariance by its eigendecomposition, which
        // tolerates a numerically semi-definite matrix.
        let (vals, vecs) = symmetric_eigen(cov);
        let root = &vecs
            * &vals
                .mapv(|v| v.max(0.0).sqrt())
                .view()
                .insert_axis(ndarray::Axis(0));
        let p = self.coef.len();
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let mut normal = || {
            // Box-Muller.
            let u1: f64 = rng.random::<f64>().max(1e-300);
            let u2: f64 = rng.random::<f64>();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };
        // The normal deviates are drawn in order, as one stream, so the draws
        // do not depend on the thread count. The draws themselves are then
        // computed on the rayon pool in chunks, which bounds the memory held
        // for the per-draw parameter vectors.
        use rayon::prelude::*;
        let zs: Vec<Array1<f64>> = (0..n_draws)
            .map(|_| Array1::from_shape_fn(p, |_| normal()))
            .collect();
        let mut out: Vec<Array2<f64>> = (0..k).map(|_| Array2::zeros((n_draws, n_rows))).collect();
        const CHUNK: usize = 64;
        for start in (0..n_draws).step_by(CHUNK) {
            let end = (start + CHUNK).min(n_draws);
            let natural_chunk: Vec<Vec<Vec<f64>>> = zs[start..end]
                .par_iter()
                .map(|z| {
                    let theta = &self.coef + &root.dot(z);
                    (0..k)
                        .map(|q| {
                            let param = &self.params[q];
                            let eta = designs[q].dot(&theta.slice(s![param.start..param.end]));
                            let mut natural = vec![0.0; n_rows];
                            family.response(q, eta.as_slice().expect("contiguous"), &mut natural);
                            natural
                        })
                        .collect()
                })
                .collect();
            for (offset, per_param) in natural_chunk.iter().enumerate() {
                for (q, natural) in per_param.iter().enumerate() {
                    out[q]
                        .row_mut(start + offset)
                        .assign(&ArrayView1::from(natural.as_slice()));
                }
            }
        }
        Ok(out)
    }
}

/// `Vp + J V_rho J'` over the smoothing parameters not at a bound.
fn corrected(
    joint: &Joint<'_>,
    eval: &LamlEval,
    rho: &Array1<f64>,
    hess: &Array2<f64>,
    bound: f64,
) -> Array2<f64> {
    let vp = eval.ainv.clone();
    let m = rho.len();
    let free: Vec<usize> = (0..m).filter(|&j| rho[j].abs() < bound - 1e-6).collect();
    if free.is_empty() {
        return vp;
    }
    let f = free.len();
    let h = Array2::from_shape_fn((f, f), |(a, b)| hess[[free[a], free[b]]]);
    let (vals, vecs) = symmetric_eigen(&h);
    let top = vals.iter().fold(0.0f64, |a, v| a.max(*v));
    if top.is_nan() || top <= 0.0 {
        return vp;
    }
    let inv = vals.mapv(|v| if v > 1e-7 * top { 2.0 / v } else { 0.0 });
    let v_rho = (&vecs * &inv.view().insert_axis(ndarray::Axis(0))).dot(&vecs.t());
    let mut jac = Array2::<f64>::zeros((joint.p, f));
    let mut col = 0;
    for (q, model) in joint.models.iter().enumerate() {
        for (kk, pen) in model.penalties.iter().enumerate() {
            let g = joint.rho_offsets[q] + kk;
            let Some(pos) = free.iter().position(|&j| j == g) else {
                continue;
            };
            let block = &model.blocks[pen.block];
            let r = joint.offsets[q] + block.start..joint.offsets[q] + block.end;
            let bk = pen.matrix.dot(&eval.beta.slice(s![r.clone()])) * rho[g].exp();
            let dtheta = eval.ainv.slice(s![.., r]).dot(&bk).mapv(|v| -v);
            jac.column_mut(pos).assign(&dtheta);
            col += 1;
        }
    }
    let _ = col;
    vp + jac.dot(&v_rho).dot(&jac.t())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gam::TermBasis;
    use crate::lss::{GaussianLSS, StudentTLSS};

    fn uniform(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 11) as f64) / ((1u64 << 53) as f64)
            })
            .collect()
    }

    fn normal(n: usize, seed: u64) -> Vec<f64> {
        let (a, b) = (uniform(n, seed), uniform(n, seed + 99));
        a.iter()
            .zip(&b)
            .map(|(u, v)| {
                (-2.0 * u.max(1e-12).ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
            })
            .collect()
    }

    fn bbs(name: &str, x: &[f64], knots: usize) -> RefitTerm {
        let basis = TermBasis::Bbs {
            boundary: (0.0, 1.0),
            n_knots: knots,
            degree: 3,
            differences: 2,
            cyclic: false,
        };
        let xa = Array1::from(x.to_vec());
        RefitTerm {
            name: name.into(),
            design: basis.design(&[xa.view()], false).unwrap(),
            penalties: basis.penalties(),
            by: None,
            variables: vec![name.into()],
        }
    }

    /// Gaussian location-scale data: mean sin(2 pi x1), log sd -0.7 + x2.
    fn data(n: usize) -> (Vec<f64>, Vec<f64>, Array1<f64>) {
        let (x1, x2, e) = (uniform(n, 3), uniform(n, 4), normal(n, 5));
        let y = Array1::from_shape_fn(n, |i| {
            (2.0 * std::f64::consts::PI * x1[i]).sin() + (-0.7 + x2[i]).exp() * e[i]
        });
        (x1, x2, y)
    }

    fn joint<'a>(
        family: &'a dyn LssFamily,
        y: &'a Array1<f64>,
        w: &'a Array1<f64>,
        x1: &[f64],
        x2: &[f64],
    ) -> Joint<'a> {
        let n = y.len();
        let mut models = vec![
            build_model(vec![bbs("x1", x1, 8)], w.view(), n).unwrap(),
            build_model(vec![bbs("x2", x2, 8)], w.view(), n).unwrap(),
        ];
        for _ in 2..family.n_params() {
            models.push(Model {
                x: Array2::ones((n, 1)),
                full_p: 1,
                blocks: Vec::new(),
                penalties: Vec::new(),
                null_dim: 1,
                fixed_penalty: None,
                unidentified: 0,
                unidentified_terms: Vec::new(),
            });
        }
        let (mut offsets, mut rho_offsets, mut p, mut m) = (vec![], vec![], 0, 0);
        for model in &models {
            offsets.push(p);
            rho_offsets.push(m);
            p += model.x.ncols();
            m += model.penalties.len();
        }
        Joint {
            models,
            offsets,
            rho_offsets,
            p,
            m,
            family,
            y: y.view(),
            w: w.view(),
            strict: std::sync::atomic::AtomicBool::new(false),
            inner: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    #[test]
    fn gamlss_criterion_gradient_matches_finite_differences() {
        let n = 300;
        let (x1, x2, y) = data(n);
        let w = Array1::ones(n);
        let j = joint(&GaussianLSS, &y, &w, &x1, &x2);
        let mut theta0 = Array1::zeros(j.p);
        theta0[j.offsets[1]] = -0.3;
        let rho = Array1::from(vec![0.5, 1.5]);
        let (e, _) = j.evaluate(&rho, &theta0).unwrap();
        // The inner penalized fit converges to about 1e-11 relative, so a
        // step of 1e-5 would measure its tolerance; 1e-4 measures the slope.
        for k in 0..rho.len() {
            let h = 1e-4;
            let mut up = rho.clone();
            let mut down = rho.clone();
            up[k] += h;
            down[k] -= h;
            let vu = j.evaluate(&up, &e.beta).unwrap().0.value;
            let vd = j.evaluate(&down, &e.beta).unwrap().0.value;
            let fd = (vu - vd) / (2.0 * h);
            assert!(
                (fd - e.grad[k]).abs() < 1e-4 * (1.0 + fd.abs()),
                "gradient {k}: analytic {} vs finite difference {fd}",
                e.grad[k]
            );
        }
    }

    #[test]
    fn gamlss_refit_recovers_mean_and_scale_effects() {
        let n = 800;
        let (x1, x2, y) = data(n);
        let terms = vec![vec![bbs("x1", &x1, 12)], vec![bbs("x2", &x2, 12)]];
        let fit = try_gamlss_refit(
            terms,
            vec!["mu".into(), "sigma".into()],
            y.view(),
            None,
            &GaussianLSS,
            None,
            &RefitControl::default(),
        )
        .unwrap();
        assert!(fit.converged, "gradient norm {}", fit.gradient_norm);
        // The log-sd effect is linear in x2 (slope 1): its term should be
        // close to a straight line with slope near 1.
        let sigma_block = &fit.params[1].blocks[0];
        assert!(sigma_block.edf < 2.5, "sigma edf {}", sigma_block.edf);
        let grid: Vec<f64> = (0..21).map(|i| 0.025 + 0.95 * i as f64 / 20.0).collect();
        let raw = TermBasis::Bbs {
            boundary: (0.0, 1.0),
            n_knots: 12,
            degree: 3,
            differences: 2,
            cyclic: false,
        }
        .design(&[Array1::from(grid.clone()).view()], false)
        .unwrap();
        let r = sigma_block.start..sigma_block.end;
        let (effect, _) = super::super::block_effect(
            raw.view(),
            sigma_block.householder.view(),
            fit.coef.slice(s![r.clone()]),
            fit.cov.slice(s![r.clone(), r]),
            false,
            None,
            false,
        )
        .unwrap();
        let slope = (effect[20] - effect[0]) / (grid[20] - grid[0]);
        assert!((slope - 1.0).abs() < 0.3, "log-sd slope {slope}");
        // The mean term follows the sine.
        let mu_block = &fit.params[0].blocks[0];
        assert!(mu_block.edf > 4.0, "mu edf {}", mu_block.edf);
    }

    #[test]
    fn parameter_draws_centre_on_the_fit_with_its_standard_error() {
        let n = 600;
        let (x1, x2, y) = data(n);
        let terms = vec![vec![bbs("x1", &x1, 10)], vec![bbs("x2", &x2, 10)]];
        let fit = try_gamlss_refit(
            terms,
            vec!["mu".into(), "sigma".into()],
            y.view(),
            None,
            &GaussianLSS,
            None,
            &RefitControl::default(),
        )
        .unwrap();
        let grid = Array1::from(vec![0.2, 0.5, 0.8]);
        let basis = TermBasis::Bbs {
            boundary: (0.0, 1.0),
            n_knots: 10,
            degree: 3,
            differences: 2,
            cyclic: false,
        };
        let raw = basis.design(&[grid.view()], false).unwrap();
        let new = |r: &Array2<f64>| NewRowsTerm {
            raw: r.clone(),
            by: None,
            zero_rows_have_no_effect: false,
        };
        let draws = fit
            .parameter_draws(
                &GaussianLSS,
                &[vec![new(&raw)], vec![new(&raw)]],
                4000,
                7,
                false,
            )
            .unwrap();
        assert_eq!(draws[0].dim(), (4000, 3));
        // mu is on the identity scale: the draws' mean is the fitted mean and
        // their sd the pointwise sd of intercept + effect.
        let block = &fit.params[0].blocks[0];
        let xb = crate::gam::constrained_design(
            raw.view(),
            block.householder.view(),
            block.end - block.start,
            false,
            None,
            false,
        )
        .unwrap();
        let mut x = Array2::<f64>::ones((3, fit.params[0].end - fit.params[0].start));
        x.slice_mut(s![.., 1..]).assign(&xb);
        let r = fit.params[0].start..fit.params[0].end;
        let mean = x.dot(&fit.coef.slice(s![r.clone()]));
        let cov = fit.cov.slice(s![r.clone(), r]);
        for i in 0..3 {
            let col = draws[0].column(i);
            let m = col.mean().unwrap();
            let sd = col.std(1.0);
            let want_sd = x.row(i).dot(&cov.dot(&x.row(i))).sqrt();
            assert!(
                (m - mean[i]).abs() < 4.0 * want_sd / (4000f64).sqrt(),
                "mean {m} vs {}",
                mean[i]
            );
            assert!((sd / want_sd - 1.0).abs() < 0.06, "sd {sd} vs {want_sd}");
        }
        // sigma draws are positive (natural scale).
        assert!(draws[1].iter().all(|v| *v > 0.0));
    }

    /// The refit and its posterior draws on a pool of `threads` rayon threads:
    /// the coefficients, the criterion, the smoothing parameters and the draws.
    fn refit_on_pool(threads: usize) -> (Array1<f64>, f64, Array1<f64>, Vec<Array2<f64>>) {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                let n = 600;
                let (x1, x2, y) = data(n);
                let terms = vec![vec![bbs("x1", &x1, 10)], vec![bbs("x2", &x2, 10)]];
                let fit = try_gamlss_refit(
                    terms,
                    vec!["mu".into(), "sigma".into()],
                    y.view(),
                    None,
                    &GaussianLSS,
                    None,
                    &RefitControl::default(),
                )
                .unwrap();
                let grid = Array1::from_shape_fn(50, |i| (i as f64 + 0.5) / 50.0);
                let raw = TermBasis::Bbs {
                    boundary: (0.0, 1.0),
                    n_knots: 10,
                    degree: 3,
                    differences: 2,
                    cyclic: false,
                }
                .design(&[grid.view()], false)
                .unwrap();
                let new_rows = || {
                    vec![NewRowsTerm {
                        raw: raw.clone(),
                        by: None,
                        zero_rows_have_no_effect: false,
                    }]
                };
                let draws = fit
                    .parameter_draws(&GaussianLSS, &[new_rows(), new_rows()], 200, 7, false)
                    .unwrap();
                (fit.coef.clone(), fit.reml_score, fit.rho.clone(), draws)
            })
    }

    /// Largest difference relative to the larger magnitude (at least 1).
    fn max_rel_diff(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs() / x.abs().max(y.abs()).max(1.0))
            .fold(0.0, f64::max)
    }

    #[test]
    fn parallel_and_sequential_refits_agree() {
        let (coef1, score1, rho1, draws1) = refit_on_pool(1);
        let (coef4, score4, rho4, draws4) = refit_on_pool(4);
        assert!(max_rel_diff(coef1.as_slice().unwrap(), coef4.as_slice().unwrap()) <= 1e-12);
        assert!((score1 - score4).abs() <= 1e-12 * score1.abs().max(1.0));
        assert!(max_rel_diff(rho1.as_slice().unwrap(), rho4.as_slice().unwrap()) <= 1e-12);
        for (a, b) in draws1.iter().zip(&draws4) {
            assert!(max_rel_diff(a.as_slice().unwrap(), b.as_slice().unwrap()) <= 1e-12);
        }
        // A fixed thread count gives bit-identical results on every run.
        let (coef4b, score4b, _, draws4b) = refit_on_pool(4);
        assert_eq!(coef4.as_slice(), coef4b.as_slice());
        assert_eq!(score4.to_bits(), score4b.to_bits());
        for (a, b) in draws4.iter().zip(&draws4b) {
            assert_eq!(a.as_slice(), b.as_slice());
        }
    }

    #[test]
    fn gamlss_refit_fits_a_three_parameter_family() {
        let n = 500;
        let (x1, x2, y) = data(n);
        let terms = vec![vec![bbs("x1", &x1, 8)], vec![bbs("x2", &x2, 8)], vec![]];
        let fit = try_gamlss_refit(
            terms,
            vec!["mu".into(), "sigma".into(), "df".into()],
            y.view(),
            None,
            &StudentTLSS,
            None,
            &RefitControl::default(),
        )
        .unwrap();
        assert_eq!(fit.params.len(), 3);
        assert!(fit.params[2].blocks.is_empty());
        assert!(fit.reml_score.is_finite());
    }

    /// Small-sample data for a null smooth in sigma (issue #374): mean
    /// `sin(3 x)`, log sd `-1 + 0.8 z`, and `w` unrelated to the response.
    fn small_sample(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Array1<f64>) {
        let (x, z, w) = (
            uniform(n, 10 * seed + 1),
            uniform(n, 10 * seed + 2),
            uniform(n, 10 * seed + 3),
        );
        let e = normal(n, 10 * seed + 4);
        let y = Array1::from_shape_fn(n, |i| (3.0 * x[i]).sin() + (-1.0 + 0.8 * z[i]).exp() * e[i]);
        (x, z, w, y)
    }

    #[test]
    fn gamlss_refit_converges_with_a_null_smooth_in_sigma_on_small_samples() {
        // Seed 2 fails the default pass: at rho = 0 the inner solve has no
        // finite maximum. The refit must still converge (issue #374).
        let (x, z, w, y) = small_sample(80, 2);
        let fit = try_gamlss_refit(
            vec![
                vec![bbs("x", &x, 20)],
                vec![bbs("z", &z, 20), bbs("w", &w, 20)],
            ],
            vec!["mu".into(), "sigma".into()],
            y.view(),
            None,
            &GaussianLSS,
            None,
            &RefitControl::default(),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        assert!(fit.converged, "stopped at {}", fit.stop_reason);
        assert!(fit.reml_score.is_finite());
    }

    #[test]
    fn gamlss_refit_refuses_when_no_finite_maximum_exists() {
        // Residuals of 1e200 overflow when squared, so the log-likelihood is
        // not finite at any smoothing parameter.
        let (x, z, w, y) = small_sample(80, 0);
        let y = y.mapv(|v| v * 1e200);
        let err = try_gamlss_refit(
            vec![
                vec![bbs("x", &x, 8)],
                vec![bbs("z", &z, 8), bbs("w", &w, 8)],
            ],
            vec!["mu".into(), "sigma".into()],
            y.view(),
            None,
            &GaussianLSS,
            None,
            &RefitControl::default(),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("no finite penalized maximum"), "{err}");
        assert!(err.contains("sigma (z, w)"), "{err}");
        assert!(err.contains("remove that term"), "{err}");
    }
}
