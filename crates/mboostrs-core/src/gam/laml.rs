//! Outer iteration on the Laplace-approximate restricted likelihood (LAML)
//! for known-scale likelihoods (Wood 2011, section 3; mgcv's default for
//! generalized models).
//!
//! At fixed log smoothing parameters `rho`, Newton's method finds the
//! penalized maximum likelihood estimate `beta`, maximizing
//! `l(beta) - beta' S beta / 2`. The criterion is
//!
//! ```text
//! V(rho) = -2 l(beta) + beta' S beta + log|X'WX + S| - log|S|+
//! ```
//!
//! with `W` the observed information weights `-d2 l_i / d eta_i^2` at
//! `beta`. Its gradient is exact:
//!
//! ```text
//! dV/drho_k = lambda_k beta' S_k beta + lambda_k tr(A^-1 S_k)
//!           + sum_i (dW_i/deta_i) (X dbeta/drho_k)_i h_i - dlog|S|+/drho_k
//! ```
//!
//! where `A = X'WX + S`, `h_i = x_i' A^-1 x_i` and
//! `dbeta/drho_k = -lambda_k A^-1 S_k beta` (the first-order terms in
//! `dbeta/drho` vanish because `beta` maximizes the penalized likelihood).
//! `rho` is found by BFGS started from the performance-iteration solution
//! and its Hessian.
//!
//! The performance iteration re-estimates `rho` on each working model; its
//! fixed point is not in general the LAML optimum, and on the Cox fixture of
//! the test suite it left one term at edf 1.0 where the LAML optimum (and
//! mgcv) has 2.6.

use ndarray::{Array1, Array2, ArrayView1};

use super::reml::{add_penalty, matmul, penalty_logdet, symmetric_eigen, weighted_gram};
use super::{approximate_wolfe, criterion_resolution, Model, SlopeTest};

/// A known-scale log-likelihood with the derivatives the LAML needs.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Likelihood {
    /// 0/1 response, logit link.
    Binomial,
    /// Counts, log link.
    Poisson,
    /// Counts, log link, variance `mu + mu^2 / theta`.
    NegativeBinomial(f64),
    /// Positive response, log link, gamma with the given shape (the scale
    /// parameter is its inverse).
    Gamma(f64),
}

impl Likelihood {
    /// The dispersion-type parameter the criterion also estimates: the
    /// negative binomial `theta` or the gamma shape.
    pub(crate) fn dispersion(&self) -> Option<f64> {
        match self {
            Likelihood::NegativeBinomial(v) | Likelihood::Gamma(v) => Some(*v),
            _ => None,
        }
    }

    /// The same likelihood with another dispersion value.
    pub(crate) fn with_dispersion(&self, v: f64) -> Likelihood {
        match self {
            Likelihood::NegativeBinomial(_) => Likelihood::NegativeBinomial(v),
            Likelihood::Gamma(_) => Likelihood::Gamma(v),
            other => *other,
        }
    }
}

/// Per-row log-likelihood contributions and derivatives in `eta`.
struct Terms {
    loglik: f64,
    /// `dl_i/deta_i`.
    d1: Array1<f64>,
    /// `W_i = -d2 l_i / deta_i^2` (observed information).
    w: Array1<f64>,
    /// `dW_i / deta_i`.
    d3: Array1<f64>,
}

fn ln_gamma(x: f64) -> f64 {
    libm::lgamma(x)
}

impl Likelihood {
    fn terms(&self, y: ArrayView1<f64>, eta: &Array1<f64>, prior: ArrayView1<f64>) -> Terms {
        let n = y.len();
        let mut d1 = Array1::zeros(n);
        let mut w = Array1::zeros(n);
        let mut d3 = Array1::zeros(n);
        let mut loglik = 0.0;
        for i in 0..n {
            let (pw, yi, e) = (prior[i], y[i], eta[i]);
            if pw == 0.0 {
                continue;
            }
            match self {
                Likelihood::Binomial => {
                    let mu = 1.0 / (1.0 + (-e).exp());
                    // log(1 + e^eta), stable for large |eta|.
                    let softplus = if e > 0.0 {
                        e + (-e).exp().ln_1p()
                    } else {
                        e.exp().ln_1p()
                    };
                    loglik += pw * (yi * e - softplus);
                    d1[i] = pw * (yi - mu);
                    let v = mu * (1.0 - mu);
                    w[i] = pw * v;
                    d3[i] = pw * v * (1.0 - 2.0 * mu);
                }
                Likelihood::Poisson => {
                    let mu = e.exp();
                    loglik += pw * (yi * e - mu - ln_gamma(yi + 1.0));
                    d1[i] = pw * (yi - mu);
                    w[i] = pw * mu;
                    d3[i] = pw * mu;
                }
                Likelihood::NegativeBinomial(theta) => {
                    let mu = e.exp();
                    let s = mu + theta;
                    loglik += pw
                        * (ln_gamma(yi + theta) - ln_gamma(*theta) - ln_gamma(yi + 1.0)
                            + theta * theta.ln()
                            + yi * e
                            - (yi + theta) * s.ln());
                    d1[i] = pw * theta * (yi - mu) / s;
                    w[i] = pw * (yi + theta) * mu * theta / (s * s);
                    d3[i] = pw * (yi + theta) * theta * mu * (theta - mu) / (s * s * s);
                }
                Likelihood::Gamma(shape) => {
                    // l = shape log(shape) - lgamma(shape) + (shape - 1) log y
                    //     - shape eta - shape y e^-eta
                    let ratio = yi * (-e).exp();
                    loglik += pw
                        * (shape * shape.ln() - ln_gamma(*shape) + (shape - 1.0) * yi.ln()
                            - shape * e
                            - shape * ratio);
                    d1[i] = pw * shape * (ratio - 1.0);
                    w[i] = pw * shape * ratio;
                    d3[i] = -pw * shape * ratio;
                }
            }
        }
        Terms { loglik, d1, w, d3 }
    }
}

/// The penalized MLE at one `rho` and the LAML there.
pub(crate) struct LamlEval {
    pub value: f64,
    pub grad: Array1<f64>,
    pub beta: Array1<f64>,
    pub ainv: Array2<f64>,
    pub xtwx: Array2<f64>,
}

pub(crate) struct Data<'a> {
    pub model: &'a Model,
    pub y: ArrayView1<'a, f64>,
    pub prior: ArrayView1<'a, f64>,
    pub offset: ArrayView1<'a, f64>,
    pub likelihood: Likelihood,
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

impl Data<'_> {
    fn eta(&self, beta: &Array1<f64>) -> Array1<f64> {
        self.model.x.dot(beta) + self.offset
    }

    fn penalty_value(&self, beta: &Array1<f64>, rho: &Array1<f64>) -> f64 {
        let mut total = 0.0;
        for (k, pen) in self.model.penalties.iter().enumerate() {
            let block = &self.model.blocks[pen.block];
            let b = beta.slice(ndarray::s![block.start..block.end]);
            total += rho[k].exp() * b.dot(&pen.matrix.dot(&b));
        }
        total
    }

    /// Newton's method for the penalized MLE at `rho`, from `beta0`.
    fn penalized_mle(&self, rho: &Array1<f64>, beta0: &Array1<f64>) -> Option<Array1<f64>> {
        let mut beta = beta0.clone();
        let mut t = self.likelihood.terms(self.y, &self.eta(&beta), self.prior);
        let mut objective = t.loglik - 0.5 * self.penalty_value(&beta, rho);
        for _ in 0..200 {
            let mut a = weighted_gram(self.model.x.view(), t.w.view());
            add_penalty(self.model, &mut a, rho);
            let mut sbeta = Array1::<f64>::zeros(beta.len());
            for (k, pen) in self.model.penalties.iter().enumerate() {
                let block = &self.model.blocks[pen.block];
                let r = block.start..block.end;
                let b = beta.slice(ndarray::s![r.clone()]);
                let mut dst = sbeta.slice_mut(ndarray::s![r]);
                dst.scaled_add(rho[k].exp(), &pen.matrix.dot(&b));
            }
            let score = self.model.x.t().dot(&t.d1) - &sbeta;
            let (ainv, _) = cholesky_inverse(&a)?;
            let step = ainv.dot(&score);
            let mut alpha = 1.0;
            let mut moved = false;
            for _ in 0..40 {
                let trial = &beta + &(&step * alpha);
                let tt = self.likelihood.terms(self.y, &self.eta(&trial), self.prior);
                let obj = tt.loglik - 0.5 * self.penalty_value(&trial, rho);
                if obj.is_finite() && obj >= objective - 1e-12 * objective.abs() {
                    let change = (obj - objective).abs();
                    beta = trial;
                    t = tt;
                    objective = obj;
                    moved = true;
                    let size = step.iter().fold(0.0f64, |m, v| m.max(v.abs())) * alpha;
                    if size < 1e-9 * (1.0 + beta.iter().fold(0.0f64, |m, v| m.max(v.abs())))
                        || change < 1e-12 * (1.0 + objective.abs())
                    {
                        return Some(beta);
                    }
                    break;
                }
                alpha *= 0.5;
            }
            if !moved {
                return Some(beta);
            }
        }
        Some(beta)
    }

    /// The LAML and its gradient at `rho`.
    pub fn evaluate(&self, rho: &Array1<f64>, beta0: &Array1<f64>) -> Option<LamlEval> {
        let beta = self.penalized_mle(rho, beta0)?;
        let t = self.likelihood.terms(self.y, &self.eta(&beta), self.prior);
        let xtwx = weighted_gram(self.model.x.view(), t.w.view());
        let mut a = xtwx.clone();
        add_penalty(self.model, &mut a, rho);
        let (ainv, logdet_a) = cholesky_inverse(&a)?;
        let (logdet_s, lds_grad, _) = penalty_logdet(self.model, rho, true)?;
        let pen = self.penalty_value(&beta, rho);
        let value = -2.0 * t.loglik + pen + logdet_a - logdet_s;
        if !value.is_finite() {
            return None;
        }
        // h_i = x_i' A^-1 x_i
        let b = matmul(self.model.x.view(), ainv.view());
        let h = (&b * &self.model.x).sum_axis(ndarray::Axis(1));
        let m = self.model.penalties.len();
        let mut grad = Array1::<f64>::zeros(m);
        for (k, pen) in self.model.penalties.iter().enumerate() {
            let block = &self.model.blocks[pen.block];
            let r = block.start..block.end;
            let lambda = rho[k].exp();
            let bk = pen.matrix.dot(&beta.slice(ndarray::s![r.clone()])) * lambda;
            let explicit = beta.slice(ndarray::s![r.clone()]).dot(&bk);
            let trace = {
                let sub = ainv.slice(ndarray::s![r.clone(), r.clone()]);
                (&sub * &pen.matrix).sum() * lambda
            };
            // dbeta/drho_k = -A^-1 b_k; deta = X dbeta.
            let dbeta = ainv.slice(ndarray::s![.., r]).dot(&bk).mapv(|v| -v);
            let deta = self.model.x.dot(&dbeta);
            let weight_change: f64 = (0..h.len()).map(|i| t.d3[i] * deta[i] * h[i]).sum();
            grad[k] = explicit + trace + weight_change - lds_grad[k];
        }
        Some(LamlEval {
            value,
            grad,
            beta,
            ainv,
            xtwx,
        })
    }
}

impl Data<'_> {
    /// Hessian of the criterion in `rho` by central differences of the
    /// exact gradient, symmetrized.
    pub fn hessian_fd(&self, rho: &Array1<f64>, beta: &Array1<f64>) -> Option<Array2<f64>> {
        let m = rho.len();
        let h = 1e-4;
        let mut hess = Array2::<f64>::zeros((m, m));
        for k in 0..m {
            let mut up = rho.clone();
            let mut down = rho.clone();
            up[k] += h;
            down[k] -= h;
            let gu = self.evaluate(&up, beta)?.grad;
            let gd = self.evaluate(&down, beta)?.grad;
            for j in 0..m {
                hess[[k, j]] = (gu[j] - gd[j]) / (2.0 * h);
            }
        }
        Some((&hess + &hess.t()) / 2.0)
    }
}

/// A criterion evaluator: the value and gradient at `rho`, with the inner
/// optimization started from the given coefficients.
pub(crate) type Evaluator<'a> = dyn Fn(&Array1<f64>, &Array1<f64>) -> Option<LamlEval> + 'a;

/// The LAML optimum.
pub(crate) struct LamlOptimum {
    pub rho: Array1<f64>,
    pub eval: LamlEval,
    pub iterations: usize,
    pub converged: bool,
    pub gradient_norm: f64,
    /// Why the iteration stopped; see [`STOP_REASONS`](super::STOP_REASONS).
    pub stop: &'static str,
    /// BFGS approximation of the Hessian of V in rho at the optimum.
    pub hessian: Array2<f64>,
}

/// BFGS on `rho` in `[-bound, bound]^m`, from `rho0` with starting Hessian
/// `hess0` (made positive definite) and coefficients `beta0`. `evaluate`
/// returns the criterion and its gradient at `rho`, starting its inner
/// optimization from the coefficients it is given.
pub(crate) fn optimize(
    evaluate: &Evaluator<'_>,
    rho0: &Array1<f64>,
    hess0: &Array2<f64>,
    beta0: &Array1<f64>,
    bound: f64,
    max_iter: usize,
    tol: f64,
) -> Option<LamlOptimum> {
    let m = rho0.len();
    let mut rho = rho0.mapv(|v| v.clamp(-bound, bound));
    let mut eval = evaluate(&rho, beta0)?;
    // Inverse Hessian from the working-model Hessian, eigenvalues floored.
    let mut hinv = if m > 0 {
        let (vals, vecs) = symmetric_eigen(hess0);
        let top = vals.iter().fold(0.0f64, |a, v| a.max(v.abs())).max(1.0);
        let floor = top * 1e-6;
        let inv = Array1::from_shape_fn(m, |i| 1.0 / vals[i].abs().max(floor));
        let scaled = &vecs * &inv.view().insert_axis(ndarray::Axis(0));
        scaled.dot(&vecs.t())
    } else {
        Array2::zeros((0, 0))
    };
    let mut iterations = 0;
    let mut converged = m == 0;
    let mut gradient_norm = 0.0;
    let mut stop = if m == 0 {
        "no_smoothing_parameters"
    } else {
        "iteration_limit"
    };
    // MBOOSTRS_TRACE_REML=1 prints one line per BFGS iteration to stderr:
    // iteration, criterion, free-gradient norm, accepted step length and the
    // number of line-search trials, so a slow refit shows where it spends
    // its evaluations.
    let trace = std::env::var_os("MBOOSTRS_TRACE_REML").is_some();
    if trace {
        eprintln!("reml start rho {:?}", rho.to_vec());
    }
    let started = std::time::Instant::now();
    while !converged && iterations < max_iter {
        iterations += 1;
        let free: Vec<bool> = (0..m)
            .map(|k| {
                let g = eval.grad[k];
                !((rho[k] <= -bound + 1e-9 && g > 0.0) || (rho[k] >= bound - 1e-9 && g < 0.0))
            })
            .collect();
        let g_free = Array1::from_shape_fn(m, |k| if free[k] { eval.grad[k] } else { 0.0 });
        gradient_norm = g_free.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        if gradient_norm < tol * (1.0 + eval.value.abs()) {
            converged = true;
            stop = "gradient";
            break;
        }
        let mut step = hinv.dot(&g_free).mapv(|v| -v);
        for k in 0..m {
            if !free[k] {
                step[k] = 0.0;
            }
        }
        if step.dot(&g_free) >= 0.0 {
            // Not a descent direction: fall back to steepest descent.
            step = g_free.mapv(|v| -v);
        }
        let longest = step.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        if longest > 5.0 {
            step *= 5.0 / longest;
        }
        let mut alpha = 1.0;
        let mut accepted = None;
        let mut trials = 0;
        for _ in 0..30 {
            trials += 1;
            let trial = (&rho + &(&step * alpha)).mapv(|v| v.clamp(-bound, bound));
            if let Some(e) = evaluate(&trial, &eval.beta) {
                if e.value <= eval.value + 1e-4 * alpha * step.dot(&g_free) {
                    accepted = Some((trial, e));
                    break;
                }
                // A rise smaller than the criterion's rounding does not show
                // that the step failed; the slope at the trial point decides.
                let s = &trial - &rho;
                let slope0 = g_free.dot(&s);
                if slope0 < 0.0 && e.value - eval.value <= criterion_resolution(eval.value) {
                    let test = approximate_wolfe(slope0, e.grad.dot(&s));
                    if trace {
                        eprintln!("reml   alpha {alpha:.3e} dV {:.3e}: {test:?}", e.value - eval.value);
                    }
                    match test {
                        SlopeTest::Accept => {
                            accepted = Some((trial, e));
                            break;
                        }
                        SlopeTest::TooShort => break,
                        SlopeTest::Overshoot => {}
                    }
                }
            }
            alpha *= 0.5;
        }
        if trace {
            eprintln!(
                "reml iter {iterations:4} V {:.6} |g| {gradient_norm:.3e} alpha {alpha:.3e} \
                 trials {trials:2} max|step| {longest:.3} t {:.1}s",
                eval.value,
                started.elapsed().as_secs_f64()
            );
        }
        let Some((trial, new_eval)) = accepted else {
            converged = gradient_norm < 1e-4 * (1.0 + eval.value.abs());
            stop = if converged { "flat" } else { "no_descent" };
            break;
        };
        let s = &trial - &rho;
        let yv = &new_eval.grad - &eval.grad;
        let sy = s.dot(&yv);
        if sy > 1e-12 {
            // BFGS update of the inverse Hessian.
            let hy = hinv.dot(&yv);
            let yhy = yv.dot(&hy);
            for i in 0..m {
                for j in 0..m {
                    hinv[[i, j]] +=
                        ((sy + yhy) * s[i] * s[j]) / (sy * sy) - (hy[i] * s[j] + s[i] * hy[j]) / sy;
                }
            }
        }
        let previous = eval.value;
        rho = trial;
        eval = new_eval;
        if (previous - eval.value).abs() < 1e-10 * (1.0 + previous.abs())
            && s.iter().fold(0.0f64, |a, v| a.max(v.abs())) < 1e-6
        {
            converged = true;
            stop = "stationary";
        }
    }
    if trace {
        eprintln!(
            "reml stop {stop} rho {:?} bound {bound} V {:.6}",
            rho.to_vec(),
            eval.value
        );
    }
    let hessian = if m > 0 {
        let (vals, vecs) = symmetric_eigen(&hinv);
        let inv = vals.mapv(|v| if v.abs() > 1e-300 { 1.0 / v } else { 0.0 });
        let scaled = &vecs * &inv.view().insert_axis(ndarray::Axis(0));
        scaled.dot(&vecs.t())
    } else {
        Array2::zeros((0, 0))
    };
    Some(LamlOptimum {
        rho,
        eval,
        iterations,
        converged,
        gradient_norm,
        stop,
        hessian,
    })
}
