//! The REML criterion of a weighted penalized least-squares problem, its
//! exact gradient and Hessian in the log smoothing parameters, and Newton's
//! method on it (Wood 2011, sections 2-3; Wood 2017, section 6.6).
//!
//! For a working response `z`, weights `W`, model matrix `X` and penalty
//! `S(rho) = sum_k exp(rho_k) S_k`, with `A = X'WX + S`:
//!
//! ```text
//! beta  = A^-1 X'Wz
//! P     = ||z - X beta||^2_W + beta' S beta  =  z'Wz - beta' X'Wz
//! V     = P + log|A| - log|S|+                         (scale known, = 1)
//! V     = (n - M) log P + log|A| - log|S|+             (scale profiled)
//! ```
//!
//! `V` is -2 x the restricted log-likelihood up to a constant; `M` is the
//! dimension of the penalty's null space. With `lambda_k = exp(rho_k)` and
//! `d beta / d rho_k = -lambda_k A^-1 S_k beta`:
//!
//! ```text
//! dP/drho_k         = lambda_k beta' S_k beta
//! d2P/drho_k drho_j = delta_kj dP/drho_k - 2 lambda_k lambda_j beta' S_k A^-1 S_j beta
//! dlog|A|/drho_k    = lambda_k tr(A^-1 S_k)
//! d2log|A|          = delta_kj dlog|A|/drho_k - lambda_k lambda_j tr(A^-1 S_k A^-1 S_j)
//! ```
//!
//! and the same two formulas for `log|S|+` on the range space of each
//! term's summed penalty. A single-penalty term has `log|lambda S|+ =
//! rank * rho + const`.

use faer::linalg::solvers::Cholesky;
use faer::prelude::{SolverCore, SpSolver};
use faer::{Mat, Side};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};

use super::{Model, RefitControl};

/// `X'WX`, accumulated over blocks of 4,096 rows with faer's parallel
/// product, so no weighted copy of the whole design is formed.
pub(crate) fn weighted_gram(x: ArrayView2<f64>, w: ArrayView1<f64>) -> Array2<f64> {
    const ROWS: usize = 4096;
    let (n, p) = x.dim();
    let mut acc = Mat::<f64>::zeros(p, p);
    let mut start = 0;
    while start < n {
        let end = (start + ROWS).min(n);
        let rows = end - start;
        let xb = Mat::from_fn(rows, p, |i, j| x[[start + i, j]]);
        let wxb = Mat::from_fn(rows, p, |i, j| w[start + i] * x[[start + i, j]]);
        faer::linalg::matmul::matmul(
            acc.as_mut(),
            xb.transpose(),
            wxb.as_ref(),
            Some(1.0),
            1.0,
            crate::experiment_faer_par(),
        );
        start = end;
    }
    from_faer(&acc)
}

/// `a' W b` for two designs with the same rows, accumulated over blocks of
/// 4,096 rows exactly as [`weighted_gram`] accumulates `X'WX`.
pub(crate) fn weighted_cross(
    a: ArrayView2<f64>,
    b: ArrayView2<f64>,
    w: ArrayView1<f64>,
) -> Array2<f64> {
    const ROWS: usize = 4096;
    let (n, pa) = a.dim();
    let pb = b.ncols();
    let mut acc = Mat::<f64>::zeros(pa, pb);
    let mut start = 0;
    while start < n {
        let end = (start + ROWS).min(n);
        let rows = end - start;
        let ab = Mat::from_fn(rows, pa, |i, j| a[[start + i, j]]);
        let bw = Mat::from_fn(rows, pb, |i, j| w[start + i] * b[[start + i, j]]);
        faer::linalg::matmul::matmul(
            acc.as_mut(),
            ab.transpose(),
            bw.as_ref(),
            Some(1.0),
            1.0,
            crate::experiment_faer_par(),
        );
        start = end;
    }
    from_faer(&acc)
}

/// Eigenvalues (ascending) and eigenvectors (columns) of a symmetric matrix.
pub(crate) fn symmetric_eigen(m: &Array2<f64>) -> (Array1<f64>, Array2<f64>) {
    let n = m.nrows();
    let fm = Mat::from_fn(n, n, |i, j| 0.5 * (m[[i, j]] + m[[j, i]]));
    let eig = fm.selfadjoint_eigendecomposition(Side::Lower);
    let s = eig.s().column_vector();
    let u = eig.u();
    let vals = Array1::from_shape_fn(n, |i| s.read(i));
    let vecs = Array2::from_shape_fn((n, n), |(i, j)| u.read(i, j));
    (vals, vecs)
}

fn to_faer(m: &Array2<f64>) -> Mat<f64> {
    Mat::from_fn(m.nrows(), m.ncols(), |i, j| m[[i, j]])
}

fn from_faer(m: &Mat<f64>) -> Array2<f64> {
    Array2::from_shape_fn((m.nrows(), m.ncols()), |(i, j)| m.read(i, j))
}

/// `a b` through faer's blocked, parallel product (the P x P products of the
/// derivatives dominate the refit's cost).
pub(crate) fn matmul(a: ArrayView2<f64>, b: ArrayView2<f64>) -> Array2<f64> {
    let fa = Mat::from_fn(a.nrows(), a.ncols(), |i, j| a[[i, j]]);
    let fb = Mat::from_fn(b.nrows(), b.ncols(), |i, j| b[[i, j]]);
    // faer 0.20 ignores beta = 0 (it writes nothing), so accumulate into
    // zeros with beta = 1.
    let mut out = Mat::<f64>::zeros(a.nrows(), b.ncols());
    faer::linalg::matmul::matmul(
        out.as_mut(),
        fa.as_ref(),
        fb.as_ref(),
        Some(1.0),
        1.0,
        crate::experiment_faer_par(),
    );
    from_faer(&out)
}

/// Square-root form of a working model for the stable REML evaluation
/// (Wood 2011, section 3): `X'WX` is never formed, because forming it squares
/// the condition number of the design.
///
/// `r0` is the `p x p` triangular factor of `sqrt(W) X` (`r0' r0 = X'WX`),
/// reduced over blocks of 4,096 rows so no copy of the design is made.
/// `roots[k]` satisfies `roots[k]' roots[k] = S_k` on its block's columns,
/// and `fixed_root` the same for `Model::fixed_penalty`. For each `rho` the
/// QR factor `R` of the stacked rows `[r0; sqrt(lambda_k) roots[k]; fixed]`
/// satisfies `R'R = A` exactly, so `log|A| = 2 sum log|R_ii|` and `beta` and
/// `A^-1` come from triangular solves with `R`, whose condition number is
/// the square root of `A`'s.
pub(crate) struct SquareRoot {
    pub r0: Array2<f64>,
    pub roots: Vec<Array2<f64>>,
    pub fixed_root: Option<Array2<f64>>,
}

/// Rows `B` with `B'B = m` for a symmetric positive semidefinite `m`: the
/// eigenvectors of its positive eigenvalues, scaled by their square roots.
fn psd_root(m: &Array2<f64>) -> Array2<f64> {
    let (vals, vecs) = symmetric_eigen(m);
    let top = vals.iter().cloned().fold(0.0f64, f64::max);
    let tol = top * (m.nrows() as f64) * f64::EPSILON * 10.0;
    let keep: Vec<usize> = (0..vals.len()).filter(|&i| vals[i] > tol).collect();
    Array2::from_shape_fn((keep.len(), m.ncols()), |(r, j)| {
        vals[keep[r]].sqrt() * vecs[[j, keep[r]]]
    })
}

/// The thin triangular QR factor of `m`, padded with zero rows to `ncols`.
fn thin_r(m: &Mat<f64>) -> Mat<f64> {
    let p = m.ncols();
    let r = m.qr().compute_thin_r();
    if r.nrows() == p {
        return r;
    }
    let mut out = Mat::<f64>::zeros(p, p);
    for i in 0..r.nrows() {
        for j in 0..p {
            out.write(i, j, r.read(i, j));
        }
    }
    out
}

impl SquareRoot {
    pub(crate) fn new(model: &Model, w: ArrayView1<f64>) -> Self {
        const ROWS: usize = 4096;
        let x = &model.x;
        let (n, p) = x.dim();
        let mut acc = Mat::<f64>::zeros(0, p);
        let mut start = 0;
        while start < n {
            let end = (start + ROWS).min(n);
            let top = acc.nrows();
            let block = Mat::from_fn(top + end - start, p, |i, j| {
                if i < top {
                    acc.read(i, j)
                } else {
                    let row = start + i - top;
                    w[row].max(0.0).sqrt() * x[[row, j]]
                }
            });
            let r = block.qr().compute_thin_r();
            acc = r;
            start = end;
        }
        let r0 = thin_r(&acc);
        SquareRoot {
            r0: from_faer(&r0),
            roots: model
                .penalties
                .iter()
                .map(|pen| psd_root(&pen.matrix))
                .collect(),
            fixed_root: model.fixed_penalty.as_ref().map(psd_root),
        }
    }

    /// `R` with `R'R = A(rho)`.
    fn factor(&self, model: &Model, rho: &Array1<f64>) -> Mat<f64> {
        let p = self.r0.ncols();
        let extra: usize = self.roots.iter().map(|b| b.nrows()).sum::<usize>()
            + self.fixed_root.as_ref().map_or(0, |b| b.nrows());
        let mut stacked = Mat::<f64>::zeros(p + extra, p);
        for i in 0..p {
            for j in i..p {
                stacked.write(i, j, self.r0[[i, j]]);
            }
        }
        let mut row = p;
        for (k, (pen, root)) in model.penalties.iter().zip(&self.roots).enumerate() {
            let block = &model.blocks[pen.block];
            let scale = (0.5 * rho[k]).exp();
            for r in 0..root.nrows() {
                for j in 0..root.ncols() {
                    stacked.write(row, block.start + j, scale * root[[r, j]]);
                }
                row += 1;
            }
        }
        if let Some(root) = &self.fixed_root {
            for r in 0..root.nrows() {
                for j in 0..p {
                    stacked.write(row, j, root[[r, j]]);
                }
                row += 1;
            }
        }
        thin_r(&stacked)
    }
}

/// The sufficient statistics of one working model.
pub(crate) struct Problem<'a> {
    pub model: &'a Model,
    pub xtwx: Array2<f64>,
    pub xtwz: Array1<f64>,
    pub zwz: f64,
    pub n: usize,
    pub known_scale: bool,
    /// The working response and weights, when available. With them the
    /// penalized residual sum of squares is computed from the residuals,
    /// `||z - X beta||^2_W + beta' S beta`, rather than as
    /// `z'Wz - beta' X'Wz`. The second form subtracts two nearly equal numbers
    /// when the fit is close to the data, and it is wrong to first order in
    /// any error in `beta`; the first is stationary in `beta`, so a solve
    /// error enters only to second order. On a nearly singular `A` (collinear
    /// smooths, a response fitted to R^2 = 0.99999) the subtraction made the
    /// criterion noisy enough that Newton's line search found no descent.
    pub data: Option<(ArrayView1<'a, f64>, ArrayView1<'a, f64>)>,
    /// The square-root form; when present the criterion is evaluated from
    /// QR factors instead of a Cholesky factor of `X'WX + S`.
    pub root: Option<SquareRoot>,
}

/// The criterion and, on request, its derivatives at one `rho`.
pub(crate) struct Eval {
    pub value: f64,
    pub grad: Array1<f64>,
    pub hess: Array2<f64>,
    pub beta: Array1<f64>,
    /// `A^-1` (only when derivatives were requested).
    pub ainv: Option<Array2<f64>>,
    /// Penalized residual sum of squares `P`.
    pub pen_rss: f64,
}

/// Add `S(rho) = sum_k exp(rho_k) S_k` to `a` in place.
pub(crate) fn add_penalty(model: &Model, a: &mut Array2<f64>, rho: &Array1<f64>) {
    if let Some(fixed) = &model.fixed_penalty {
        *a += fixed;
    }
    for (k, pen) in model.penalties.iter().enumerate() {
        let block = &model.blocks[pen.block];
        let mut view = a.slice_mut(s![block.start..block.end, block.start..block.end]);
        view.scaled_add(rho[k].exp(), &pen.matrix);
    }
}

/// `log|S|+` and, with `derivs`, its gradient and Hessian.
pub(crate) fn penalty_logdet(
    model: &Model,
    rho: &Array1<f64>,
    derivs: bool,
) -> Option<(f64, Array1<f64>, Array2<f64>)> {
    let m = model.penalties.len();
    let mut value = 0.0;
    let mut grad = Array1::zeros(if derivs { m } else { 0 });
    let mut hess = Array2::zeros(if derivs { (m, m) } else { (0, 0) });
    for block in &model.blocks {
        let ids = &block.penalty_ids;
        if ids.is_empty() {
            continue;
        }
        if ids.len() == 1 {
            let k = ids[0];
            value += block.rank as f64 * rho[k] + block.logdet_const;
            if derivs {
                grad[k] = block.rank as f64;
            }
            continue;
        }
        if let Some(pair) = &block.pair {
            let (k1, k2) = (ids[0], ids[1]);
            let (l1, l2) = (rho[k1].exp(), rho[k2].exp());
            value += pair.logdet_t;
            for &mu in pair.mu.iter() {
                let (a, b) = (l1 * mu, l2 * (1.0 - mu));
                let d = a + b;
                value += d.ln();
                if derivs {
                    let (fa, fb) = (a / d, b / d);
                    grad[k1] += fa;
                    grad[k2] += fb;
                    hess[[k1, k1]] += fa - fa * fa;
                    hess[[k2, k2]] += fb - fb * fb;
                    hess[[k1, k2]] -= fa * fb;
                    hess[[k2, k1]] -= fa * fb;
                }
            }
            continue;
        }
        let r = block.rank;
        let mut sum = Array2::<f64>::zeros((r, r));
        for &k in ids {
            let proj = model.penalties[k].projected.as_ref()?;
            sum.scaled_add(rho[k].exp(), proj);
        }
        // log|S~| and S~^-1 from a Cholesky factor: S~ is positive
        // definite on the range space it was projected onto.
        let chol = Cholesky::try_new(to_faer(&sum).as_ref(), Side::Lower).ok()?;
        let l = chol.compute_l();
        for i in 0..r {
            let d = l.read(i, i);
            if d.is_nan() || d <= 0.0 {
                return None;
            }
            value += 2.0 * d.ln();
        }
        if derivs {
            let sinv = from_faer(&chol.inverse());
            let mut prods = Vec::with_capacity(ids.len());
            for &k in ids {
                let proj = model.penalties[k].projected.as_ref()?;
                let prod = matmul(sinv.view(), proj.view()).mapv(|v| v * rho[k].exp());
                grad[k] = trace(&prod);
                prods.push(prod);
            }
            for (a, &k) in ids.iter().enumerate() {
                hess[[k, k]] += grad[k];
                for (b, &j) in ids.iter().enumerate() {
                    hess[[k, j]] -= trace_product(&prods[a], &prods[b]);
                }
            }
        }
    }
    Some((value, grad, hess))
}

/// How `A` was factored at one `rho`.
enum Factor {
    Cholesky(Cholesky<f64>),
    /// Upper triangular `R` with `R'R = A`.
    Triangular(Mat<f64>),
}

impl Problem<'_> {
    /// `(beta, log|A|, factor)`; `None` when `A` is not positive definite.
    fn solve(&self, rho: &Array1<f64>) -> Option<(Array1<f64>, f64, Factor)> {
        let p = self.xtwz.len();
        if let Some(root) = &self.root {
            let r = root.factor(self.model, rho);
            let top = (0..p).fold(0.0f64, |a, i| a.max(r.read(i, i).abs()));
            let mut logdet = 0.0;
            for i in 0..p {
                let d = r.read(i, i).abs();
                // A pivot below sqrt(eps) of the largest (mgcv's rank.tol)
                // is a direction neither the data nor the penalty determines
                // to working precision: A's eigenvalue there is below eps.
                if !d.is_finite() || d <= top * f64::EPSILON.sqrt() {
                    return None;
                }
                logdet += 2.0 * d.ln();
            }
            // R'R beta = X'Wz: solve R' y = X'Wz, then R beta = y.
            let mut sol = Mat::from_fn(p, 1, |i, _| self.xtwz[i]);
            faer::linalg::triangular_solve::solve_lower_triangular_in_place(
                r.transpose(),
                sol.as_mut(),
                faer::Parallelism::None,
            );
            faer::linalg::triangular_solve::solve_upper_triangular_in_place(
                r.as_ref(),
                sol.as_mut(),
                faer::Parallelism::None,
            );
            let beta = Array1::from_shape_fn(p, |i| sol.read(i, 0));
            return Some((beta, logdet, Factor::Triangular(r)));
        }
        let a = self.penalized_gram(rho);
        let chol = Cholesky::try_new(to_faer(&a).as_ref(), Side::Lower).ok()?;
        let l = chol.compute_l();
        let mut logdet = 0.0;
        for i in 0..p {
            let d = l.read(i, i);
            if d.is_nan() || d <= 0.0 {
                return None;
            }
            logdet += 2.0 * d.ln();
        }
        let rhs = Mat::from_fn(p, 1, |i, _| self.xtwz[i]);
        let sol = chol.solve(&rhs);
        let beta = Array1::from_shape_fn(p, |i| sol.read(i, 0));
        Some((beta, logdet, Factor::Cholesky(chol)))
    }

    fn restricted_n(&self) -> f64 {
        (self.n as f64 - self.model.null_dim as f64).max(1.0)
    }

    /// `A = X'WX + S(rho)`.
    fn penalized_gram(&self, rho: &Array1<f64>) -> Array2<f64> {
        let mut a = self.xtwx.clone();
        add_penalty(self.model, &mut a, rho);
        a
    }

    /// The criterion at `rho`; `None` when `A` is not positive definite.
    pub fn evaluate(&self, rho: &Array1<f64>, derivs: bool) -> Option<Eval> {
        let (beta, logdet_a, factor) = self.solve(rho)?;
        let p = beta.len();
        let pen_rss = match self.data {
            Some((z, w)) => {
                let fitted = self.model.x.dot(&beta);
                let rss: f64 = z
                    .iter()
                    .zip(fitted.iter())
                    .zip(w.iter())
                    .map(|((zi, fi), wi)| wi * (zi - fi) * (zi - fi))
                    .sum();
                let mut penalty = 0.0;
                for (k, pen) in self.model.penalties.iter().enumerate() {
                    let block = &self.model.blocks[pen.block];
                    let b = beta.slice(s![block.start..block.end]);
                    penalty += rho[k].exp() * b.dot(&pen.matrix.dot(&b));
                }
                rss + penalty
            }
            None => self.zwz - beta.dot(&self.xtwz),
        }
        .max(f64::MIN_POSITIVE);
        let (logdet_s, lds_grad, lds_hess) = penalty_logdet(self.model, rho, derivs)?;
        let rn = self.restricted_n();
        let head = if self.known_scale {
            pen_rss
        } else {
            rn * pen_rss.ln()
        };
        let value = head + logdet_a - logdet_s;
        if !value.is_finite() {
            return None;
        }
        if !derivs {
            return Some(Eval {
                value,
                grad: Array1::zeros(0),
                hess: Array2::zeros((0, 0)),
                beta,
                ainv: None,
                pen_rss,
            });
        }
        let ainv = match factor {
            Factor::Cholesky(chol) => from_faer(&chol.inverse()),
            Factor::Triangular(r) => {
                // A^-1 = R^-1 R^-T.
                let mut rinv = Mat::<f64>::identity(p, p);
                faer::linalg::triangular_solve::solve_upper_triangular_in_place(
                    r.as_ref(),
                    rinv.as_mut(),
                    faer::Parallelism::None,
                );
                let mut out = Mat::<f64>::zeros(p, p);
                faer::linalg::matmul::matmul(
                    out.as_mut(),
                    rinv.as_ref(),
                    rinv.transpose(),
                    Some(1.0),
                    1.0,
                    crate::experiment_faer_par(),
                );
                from_faer(&out)
            }
        };
        let m = self.model.penalties.len();
        let mut p_grad = Array1::<f64>::zeros(m);
        let mut ld_grad = Array1::<f64>::zeros(m);
        // b_k = lambda_k S_k beta_blk (on the block), u_k = A^-1 b_k (all p),
        // G_k = lambda_k A^-1[:, blk] S_k (p x q_k).
        let mut bs = Vec::with_capacity(m);
        let mut us = Vec::with_capacity(m);
        let mut gs = Vec::with_capacity(m);
        for (k, pen) in self.model.penalties.iter().enumerate() {
            let block = &self.model.blocks[pen.block];
            let range = block.start..block.end;
            let lambda = rho[k].exp();
            let bk = pen.matrix.dot(&beta.slice(s![range.clone()])) * lambda;
            p_grad[k] = beta.slice(s![range.clone()]).dot(&bk);
            let ainv_cols = ainv.slice(s![.., range.clone()]);
            let uk = ainv_cols.dot(&bk);
            let gk = matmul(ainv_cols, pen.matrix.view()) * lambda;
            ld_grad[k] = trace(&gk.slice(s![range, ..]).to_owned());
            bs.push(bk);
            us.push(uk);
            gs.push(gk);
        }
        let mut p_hess = Array2::<f64>::zeros((m, m));
        let mut ld_hess = Array2::<f64>::zeros((m, m));
        for k in 0..m {
            let bk_range = {
                let b = &self.model.blocks[self.model.penalties[k].block];
                b.start..b.end
            };
            for j in 0..m {
                let bj_range = {
                    let b = &self.model.blocks[self.model.penalties[j].block];
                    b.start..b.end
                };
                let cross = bs[k].dot(&us[j].slice(s![bk_range.clone()]));
                p_hess[[k, j]] = -2.0 * cross;
                // tr(G_k G_j) over the nonzero columns of each.
                let left = gs[k].slice(s![bj_range.clone(), ..]);
                let right = gs[j].slice(s![bk_range.clone(), ..]);
                let mut t = 0.0;
                for (a_idx, row) in left.rows().into_iter().enumerate() {
                    for (b_idx, v) in row.iter().enumerate() {
                        t += v * right[[b_idx, a_idx]];
                    }
                }
                ld_hess[[k, j]] = -t;
            }
            p_hess[[k, k]] += p_grad[k];
            ld_hess[[k, k]] += ld_grad[k];
        }
        let (grad, hess) = if self.known_scale {
            (
                &p_grad + &ld_grad - &lds_grad,
                &p_hess + &ld_hess - &lds_hess,
            )
        } else {
            let g = p_grad.mapv(|v| rn * v / pen_rss) + &ld_grad - &lds_grad;
            let mut h = &ld_hess - &lds_hess;
            for k in 0..m {
                for j in 0..m {
                    h[[k, j]] += rn
                        * (p_hess[[k, j]] / pen_rss - p_grad[k] * p_grad[j] / (pen_rss * pen_rss));
                }
            }
            (g, h)
        };
        Some(Eval {
            value,
            grad,
            hess,
            beta,
            ainv: Some(ainv),
            pen_rss,
        })
    }
}

fn trace(m: &Array2<f64>) -> f64 {
    m.diag().sum()
}

/// `tr(A B)` for square `A`, `B` of the same size.
fn trace_product(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let mut t = 0.0;
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            t += a[[i, j]] * b[[j, i]];
        }
    }
    t
}

/// The optimum found by [`optimize`].
pub(crate) struct Optimum {
    pub rho: Array1<f64>,
    pub eval: Eval,
    pub iterations: usize,
    pub converged: bool,
    pub gradient_norm: f64,
    /// Why the iteration stopped; see [`STOP_REASONS`](super::STOP_REASONS).
    pub stop: &'static str,
}

/// Minimize the criterion over `rho` in `[-bound, bound]^m` by Newton's
/// method: the Hessian's eigenvalues are made positive (their absolute
/// values, floored) so every step is a descent direction, steps longer than
/// 5 in any coordinate are shortened, and a step that does not lower the
/// criterion is halved. A parameter at a bound whose gradient points out of
/// the box is held there (a term smoothed to its null space, or left
/// unpenalized).
pub(crate) fn optimize(
    problem: &Problem<'_>,
    rho0: &Array1<f64>,
    control: &RefitControl,
) -> Option<Optimum> {
    let bound = control.rho_bound;
    let m = problem.model.penalties.len();
    let mut rho = rho0.mapv(|v| v.clamp(-bound, bound));
    // Start from a feasible point: a singular A at the start (a basis column
    // with no data) is cured by more smoothing.
    let mut eval = None;
    for _ in 0..40 {
        if let Some(e) = problem.evaluate(&rho, true) {
            eval = Some(e);
            break;
        }
        rho.mapv_inplace(|v| (v + 2.0).min(bound));
    }
    let mut eval = eval?;
    if m == 0 {
        return Some(Optimum {
            rho,
            eval,
            iterations: 0,
            converged: true,
            gradient_norm: 0.0,
            stop: "no_smoothing_parameters",
        });
    }
    let mut iterations = 0;
    let mut converged = false;
    let mut gradient_norm = f64::INFINITY;
    let mut stop = "iteration_limit";
    while iterations < control.max_newton {
        iterations += 1;
        let free: Vec<usize> = (0..m)
            .filter(|&k| {
                let g = eval.grad[k];
                !((rho[k] <= -bound + 1e-9 && g > 0.0) || (rho[k] >= bound - 1e-9 && g < 0.0))
            })
            .collect();
        gradient_norm = free.iter().map(|&k| eval.grad[k].abs()).fold(0.0, f64::max);
        if gradient_norm < control.newton_tol * (1.0 + eval.value.abs()) {
            converged = true;
            stop = "gradient";
            break;
        }
        let step = newton_step(&eval, &free, m);
        let mut alpha = 1.0;
        let mut accepted = None;
        // A smoothing parameter heading to 0 or infinity sits on an
        // exponential tail of the criterion (V ~ c + a exp(-+rho), so the
        // gradient and the diagonal Hessian have equal size), where Newton
        // advances one unit of rho per iteration. Try moving such a
        // coordinate straight to its bound; keep it only if V does not rise.
        let tail: Vec<usize> = free
            .iter()
            .copied()
            .filter(|&k| {
                let (g, h) = (eval.grad[k], eval.hess[[k, k]]);
                g != 0.0 && g.abs() < 1.0 && (h - g.abs()).abs() < 0.2 * g.abs()
            })
            .collect();
        if !tail.is_empty() {
            let mut trial = (&rho + &step).mapv(|v| v.clamp(-bound, bound));
            for &k in &tail {
                trial[k] = if eval.grad[k] < 0.0 { bound } else { -bound };
            }
            if let Some(e) = problem.evaluate(&trial, false) {
                if e.value <= eval.value {
                    accepted = Some(trial);
                }
            }
        }
        for _ in 0..30 {
            if accepted.is_some() {
                break;
            }
            let trial = (&rho + &(&step * alpha)).mapv(|v| v.clamp(-bound, bound));
            if let Some(e) = problem.evaluate(&trial, false) {
                if e.value <= eval.value {
                    accepted = Some(trial);
                    break;
                }
            }
            alpha *= 0.5;
        }
        let Some(trial) = accepted else {
            // No descent along the Newton direction: the criterion is flat to
            // rounding here, which is convergence in all but name.
            converged = gradient_norm < 1e-4 * (1.0 + eval.value.abs());
            stop = if converged { "flat" } else { "no_descent" };
            break;
        };
        let previous = eval.value;
        let moved = (&trial - &rho)
            .iter()
            .fold(0.0f64, |acc, v| acc.max(v.abs()));
        rho = trial;
        eval = problem.evaluate(&rho, true)?;
        if moved < 1e-10 && (previous - eval.value).abs() < 1e-12 * (1.0 + previous.abs()) {
            converged = true;
            stop = "stationary";
            break;
        }
    }
    Some(Optimum {
        rho,
        eval,
        iterations,
        converged,
        gradient_norm,
        stop,
    })
}

/// `-H~^-1 g` on the free coordinates, `H~` the Hessian with its eigenvalues
/// replaced by their floored absolute values; at most 5 in any coordinate.
fn newton_step(eval: &Eval, free: &[usize], m: usize) -> Array1<f64> {
    let f = free.len();
    let mut step = Array1::<f64>::zeros(m);
    if f == 0 {
        return step;
    }
    let h = Array2::from_shape_fn((f, f), |(a, b)| eval.hess[[free[a], free[b]]]);
    let g = Array1::from_shape_fn(f, |a| eval.grad[free[a]]);
    let (vals, vecs) = symmetric_eigen(&h);
    let top = vals.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
    let floor = (top * 1e-7).max(1e-10);
    let coords = vecs.t().dot(&g);
    let scaled = Array1::from_shape_fn(f, |i| coords[i] / vals[i].abs().max(floor));
    let local = vecs.dot(&scaled).mapv(|v| -v);
    let longest = local.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
    let shrink = if longest > 5.0 { 5.0 / longest } else { 1.0 };
    for (a, &k) in free.iter().enumerate() {
        step[k] = local[a] * shrink;
    }
    step
}
