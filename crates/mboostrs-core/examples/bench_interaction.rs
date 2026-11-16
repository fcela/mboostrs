//! P3.4 item 7 decision benchmark: factored vs materialised `bbs %X% brandom`.
//!
//! A `bbs(x, df=4) %X% brandom(g, df=4)` interaction is today rewritten into a
//! `buser` term over the materialised row-wise tensor `T` (24 spline columns x
//! 20 group indicators = 480 columns; at 20,000 rows that is 76.8 MB of f64).
//! The tensor's Kronecker-sum penalty `K = kron(K_spline, I) + I` is block
//! diagonal over groups with an identical `K_spline + I` block, so with the
//! interaction's `center = False` default the normal equations decouple per
//! group: each group solves its own 24x24 ridge system. This bench builds both
//! arms and asserts they agree before timing anything:
//!
//!   1. single-solve beta / prediction / SSE equality,
//!   2. calibration agreement (lambda from the per-group trace union vs
//!      `df_to_lambda` on the materialised tensor),
//!   3. selected-sequence and cumulative-risk equality over 200 boosting
//!      steps against a shared scalar competitor.
//!
//! Then it times cold setup and one hot candidate step (paired alternating
//! rounds) and reports setup bytes/count through a counting allocator.

use faer::Side;
use faer::{Mat, MatRef};
use mboostrs_core::learners::{
    build_bbs_template, df_to_lambda, prepare_block_bols, prepare_scalar_bols,
};
use mboostrs_core::linalg::GramSolver;
use ndarray::{Array1, Array2};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct CountingAllocator;
static TRACK: AtomicBool = AtomicBool::new(false);
static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACK.load(Ordering::Relaxed) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if TRACK.load(Ordering::Relaxed) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if TRACK.load(Ordering::Relaxed) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

fn xorshift(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed;
    move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state as f64) / (u64::MAX as f64)
    }
}

/// The factored arm: per-group ridge systems sharing one lambda.
struct Factored {
    solvers: Vec<GramSolver>,
    x_l: Array2<f64>,
    groups: Vec<usize>,
    weights: Array1<f64>,
    p_l: usize,
    beta: Vec<f64>,
    rhs: Array2<f64>,
    pred: Array1<f64>,
}

impl Factored {
    /// `gram_g = X_g'W X_g + lambda * (K_spline + I)` per group, factored once.
    fn new(
        x_l: Array2<f64>,
        groups: Vec<usize>,
        weights: Array1<f64>,
        k_spline: &Array2<f64>,
        lambda: f64,
    ) -> Self {
        let n = x_l.nrows();
        let p_l = x_l.ncols();
        let n_groups = groups.iter().copied().max().unwrap_or(0) + 1;
        let mut solvers = Vec::with_capacity(n_groups);
        for gr in 0..n_groups {
            let mut gram = Array2::<f64>::zeros((p_l, p_l));
            for i in 0..n {
                if groups[i] != gr {
                    continue;
                }
                let wi = weights[i];
                for jj in 0..p_l {
                    let xj = x_l[[i, jj]];
                    for kk in 0..p_l {
                        gram[[jj, kk]] += wi * xj * x_l[[i, kk]];
                    }
                }
            }
            for jj in 0..p_l {
                gram[[jj, jj]] += lambda * (k_spline[[jj, jj]] + 1.0);
                for kk in (jj + 1)..p_l {
                    let v = lambda * k_spline[[jj, kk]];
                    gram[[jj, kk]] += v;
                    gram[[kk, jj]] += v;
                }
            }
            solvers.push(GramSolver::try_new(&gram).expect("per-group gram is solvable"));
        }
        Self {
            solvers,
            x_l,
            groups,
            weights,
            p_l,
            beta: vec![0.0; n_groups * p_l],
            rhs: Array2::zeros((n_groups, p_l)),
            pred: Array1::zeros(n),
        }
    }

    /// One candidate step: rhs per group, solve, predict, weighted SSE.
    fn step(&mut self, u: &Array1<f64>) -> f64 {
        let n = self.x_l.nrows();
        self.rhs.fill(0.0);
        for i in 0..n {
            let gr = self.groups[i];
            let wi = self.weights[i];
            let ui = u[i];
            for jj in 0..self.p_l {
                self.rhs[[gr, jj]] += wi * self.x_l[[i, jj]] * ui;
            }
        }
        for gr in 0..self.solvers.len() {
            let lo = gr * self.p_l;
            let rhs = self.rhs.row(gr);
            let beta_slice = &mut self.beta[lo..lo + self.p_l];
            self.solvers[gr].solve_into_slice(rhs.as_slice().unwrap(), beta_slice);
        }
        let mut sse = 0.0;
        for i in 0..n {
            let gr = self.groups[i];
            let base = gr * self.p_l;
            let mut p = 0.0;
            for jj in 0..self.p_l {
                p += self.x_l[[i, jj]] * self.beta[base + jj];
            }
            self.pred[i] = p;
            let r = u[i] - p;
            sse += self.weights[i] * r * r;
        }
        sse
    }

    fn pred(&self, i: usize) -> f64 {
        let gr = self.groups[i];
        let base = gr * self.p_l;
        let mut p = 0.0;
        for jj in 0..self.p_l {
            p += self.x_l[[i, jj]] * self.beta[base + jj];
        }
        p
    }
}

/// Per-group Gram matrices, reused by the calibration gate.
fn per_group_grams(
    x_l: &Array2<f64>,
    groups: &[usize],
    weights: &Array1<f64>,
    n_groups: usize,
) -> Vec<Array2<f64>> {
    let n = x_l.nrows();
    let p_l = x_l.ncols();
    let mut grams = Vec::with_capacity(n_groups);
    for gr in 0..n_groups {
        let mut gram = Array2::<f64>::zeros((p_l, p_l));
        for i in 0..n {
            if groups[i] != gr {
                continue;
            }
            let wi = weights[i];
            for jj in 0..p_l {
                let xj = x_l[[i, jj]];
                for kk in 0..p_l {
                    gram[[jj, kk]] += wi * xj * x_l[[i, kk]];
                }
            }
        }
        grams.push(gram);
    }
    grams
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        0.5 * (values[mid - 1] + values[mid])
    } else {
        values[mid]
    }
}

/// Inverse-transpose backsolve on the lower Cholesky factor, replicating the
/// production `df_to_lambda`'s `rm = chol_backsolve_inv_t(a)` so the per-block
/// spectrum is exactly comparable to the full-tensor one.
fn chol_backsolve_inv_t_mat(a: MatRef<'_, f64>) -> Mat<f64> {
    let p = a.nrows();
    let l = faer::linalg::solvers::Cholesky::try_new(a, Side::Lower)
        .expect("PSD gram")
        .compute_l();
    let mut rm = Mat::zeros(p, p);
    for j in 0..p {
        for i in (0..p).rev() {
            let mut s = if i == j { 1.0 } else { 0.0 };
            for k in i + 1..p {
                s -= l.read(k, i) * rm.read(k, j);
            }
            rm.write(i, j, s / l.read(i, i));
        }
    }
    rm
}

fn main() {
    let n = 20_000usize;
    let n_groups = 20usize;
    let mut rng = xorshift(20260908);
    let x: Array1<f64> = (0..n).map(|_| rng() * 5.0 - 2.0).collect();
    let groups: Vec<usize> = (0..n).map(|_| (rng() * n_groups as f64) as usize).collect();
    let weights = Array1::ones(n);
    let y: Array1<f64> = (0..n)
        .map(|i| {
            let gr = groups[i];
            (x[i] * 3.0).sin() + 0.1 * gr as f64 + (rng() - 0.5) * 0.4
        })
        .collect();

    TRACK.store(true, Ordering::SeqCst);
    ALLOC_COUNT.store(0, Ordering::SeqCst);
    ALLOC_BYTES.store(0, Ordering::SeqCst);

    let (x_l, k_spline) = build_bbs_template(x.view(), 20, 3, 2, false, None, false, None, 0);
    let tensor_build_bytes = ALLOC_BYTES.load(Ordering::SeqCst);
    let tensor_build_count = ALLOC_COUNT.load(Ordering::SeqCst);

    // Materialised tensor T (n x p_l*n_groups) and its Kronecker-sum penalty K.
    let mut tensor = Array2::<f64>::zeros((n, 24 * n_groups));
    for i in 0..n {
        let gr = groups[i];
        for jl in 0..24 {
            let base = x_l[[i, jl]];
            if base == 0.0 {
                continue;
            }
            tensor[[i, jl * n_groups + gr]] = base;
        }
    }
    let p = 24 * n_groups;
    let mut k = Array2::<f64>::zeros((p, p));
    for jl in 0..24 {
        for jk in 0..24 {
            let v = k_spline[[jl, jk]];
            if v == 0.0 {
                continue;
            }
            for jr in 0..n_groups {
                k[[jl * n_groups + jr, jk * n_groups + jr]] += v;
            }
        }
    }
    for d in 0..p {
        k[[d, d]] += 1.0;
    }
    let tensor_bytes = ALLOC_BYTES.load(Ordering::SeqCst) - tensor_build_bytes;
    ALLOC_COUNT.store(0, Ordering::SeqCst);

    // arm A (baseline): materialised tensor through the production BlockBols path.
    ALLOC_COUNT.store(0, Ordering::SeqCst);
    ALLOC_BYTES.store(0, Ordering::SeqCst);
    let lambda = df_to_lambda(&tensor, &k, &weights.view(), 16.0);
    let t0 = std::time::Instant::now();
    let mut arm_a = prepare_block_bols(
        tensor.clone(),
        weights.view(),
        k.mapv(|v| lambda * v),
        false,
        "interaction",
    );
    let arm_a_setup = t0.elapsed();
    let arm_a_bytes = ALLOC_BYTES.load(Ordering::SeqCst);
    let arm_a_count = ALLOC_COUNT.load(Ordering::SeqCst);

    // arm B (factored): per-group ridge systems sharing the same lambda.
    ALLOC_COUNT.store(0, Ordering::SeqCst);
    ALLOC_BYTES.store(0, Ordering::SeqCst);
    let t0 = std::time::Instant::now();
    let mut arm_b = Factored::new(
        x_l.clone(),
        groups.clone(),
        weights.clone(),
        &k_spline,
        lambda,
    );
    let arm_b_setup = t0.elapsed();
    let arm_b_bytes = ALLOC_BYTES.load(Ordering::SeqCst);
    let arm_b_count = ALLOC_COUNT.load(Ordering::SeqCst);
    TRACK.store(false, Ordering::SeqCst);

    // gate 1: single-solve beta / pred / sse equality.
    let u = y.clone();
    let (beta_a, pred_a, sse_a) = arm_a.fit_step_ref(u.view(), weights.view());
    let sse_b = arm_b.step(&u);
    let mut norm2 = 0.0f64;
    let mut beta_max = 0.0f64;
    for gr in 0..n_groups {
        for jl in 0..24 {
            // materialised column order (jl, jr) at jl*n_groups + jr
            let b_ml = beta_a[jl * n_groups + gr];
            let b_f = arm_b.beta[gr * 24 + jl];
            let d = (b_ml - b_f).abs();
            beta_max = beta_max.max(d);
            norm2 += d * d;
        }
    }
    let beta_rmse = (norm2 / (n_groups * 24) as f64).sqrt();
    let mut pred_max = 0.0f64;
    for (i, pa) in pred_a.iter().enumerate() {
        pred_max = pred_max.max((pa - arm_b.pred(i)).abs());
    }
    println!(
        "single-solve  beta_rmse={beta_rmse:.3e} beta_max={beta_max:.3e} pred_max={pred_max:.3e}"
    );
    println!("              sse_materialised={sse_a:.9e} sse_factored={sse_b:.9e}");
    if beta_rmse > 1e-8 || pred_max > 1e-8 || (sse_a - sse_b).abs() / sse_a > 1e-8 {
        println!("GATE 1 FAILED: factored and materialised single solve diverge");
        std::process::exit(1);
    }
    println!("gate 1 ok");

    // gate 2: calibration must factor. `df_to_lambda` reads the spectrum of
    // RᵀKR (R = inverse-transpose Cholesky of X'WX + εK) through
    // `sum 2/(1+λd) - 1/(1+λd)²`. For a Kronecker-sum penalty the system is
    // block diagonal, so that spectrum is the union of the per-block spectra:
    // a factored arm can derive the same λ from 20 small 24x24 factorizations.
    // Compute the per-block spectra with the production formula and bisect the
    // df curve to the same target; the root must match `df_to_lambda`'s.
    let grams = per_group_grams(&x_l, &groups, &weights, n_groups);
    let gl = 24usize;
    let mboost_eps = 1e-9;
    let mut k_g = Array2::<f64>::zeros((gl, gl));
    for jj in 0..gl {
        for kk in 0..gl {
            k_g[[jj, kk]] = k_spline[[jj, kk]];
        }
        k_g[[jj, jj]] += 1.0;
    }
    let mut spectra: Vec<f64> = Vec::with_capacity(n_groups * gl);
    for gram in &grams {
        let mut a = gram.clone();
        for jj in 0..gl {
            for l in 0..gl {
                a[[jj, l]] += mboost_eps * k_g[[jj, l]];
            }
        }
        let a_mat = Mat::from_fn(gl, gl, |i, j| a[[i, j]]);
        let rm = chol_backsolve_inv_t_mat(a_mat.as_ref());
        let mut krm = Mat::zeros(gl, gl);
        let k_mat = Mat::from_fn(gl, gl, |i, j| k_g[[i, j]]);
        faer::linalg::matmul::matmul(&mut krm, &k_mat, &rm, None, 1.0, faer::Parallelism::None);
        let mut kmrm = Mat::zeros(gl, gl);
        faer::linalg::matmul::matmul(
            &mut kmrm,
            rm.transpose(),
            &krm,
            None,
            1.0,
            faer::Parallelism::None,
        );
        let mut d = kmrm.singular_values();
        d.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        spectra.extend(d);
    }
    let df_fun = |lambda_val: f64| -> f64 {
        let mut df = 0.0;
        for &dk in &spectra {
            let v = 1.0 / (1.0 + lambda_val * dk);
            df += 2.0 * v - v * v;
        }
        df
    };
    let d_target = 16.0;
    if df_fun(0.0) <= d_target {
        println!("GATE 2 FAILED: df(0) already at/below target");
        std::process::exit(1);
    }
    let root_at = |df_fun: &dyn Fn(f64) -> f64| -> f64 {
        let mut lo = 0.0f64;
        let mut hi = 1e15f64;
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if df_fun(mid) > d_target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    };
    // Cross-check: the per-block spectrum union must reproduce the full-tensor
    // spectrum's curve. `df_to_lambda`'s returned λ already encodes the full
    // tensor, so compare the bisected root directly against it.
    let lambda_f = root_at(&df_fun);
    let lambda_rel = (lambda_f - lambda).abs() / lambda.max(1e-300);
    println!("calibration   lambda_materialised={lambda:.6e} lambda_factored={lambda_f:.6e} rel={lambda_rel:.3e}");
    if lambda_rel > 1e-6 || !lambda_rel.is_finite() {
        println!("GATE 2 FAILED: factored calibration diverges from df_to_lambda");
        std::process::exit(1);
    }
    println!("gate 2 ok");

    // gate 3: 200 boosting steps against a shared scalar competitor — the
    // selection sequence and cumulative (discounted) risk must agree.
    let z: Array1<f64> = (0..n).map(|_| rng() * 2.0 - 1.0).collect();
    let mut scalar_a = prepare_scalar_bols(z.view(), weights.view(), 0.0, true, "z", None);
    let mut scalar_b = prepare_scalar_bols(z.view(), weights.view(), 0.0, true, "z", None);
    let mstep = 200usize;
    let nu = 0.1f64;
    let mut u_a = y.clone();
    let mut u_b = y.clone();
    let mut mismatch = 0usize;
    let mut risk_a = 0.0f64;
    let mut risk_b = 0.0f64;
    for _ in 0..mstep {
        let (_, pred_a_i, sse_a_i) = arm_a.fit_step_ref(u_a.view(), weights.view());
        let (_, pred_z_a, sse_z_a) = scalar_a.fit_step_ref(u_a.view(), weights.view());
        let sse_b_i = arm_b.step(&u_b);
        let (_, pred_z_b, sse_z_b) = scalar_b.fit_step_ref(u_b.view(), weights.view());
        let (win_a, pred_a_used, sse_a_used) = if sse_a_i <= sse_z_a {
            (0, pred_a_i.to_vec(), sse_a_i)
        } else {
            (1, pred_z_a.to_vec(), sse_z_a)
        };
        let (win_b, pred_b_used, sse_b_used) = if sse_b_i <= sse_z_b {
            (
                0,
                (0..n).map(|i| arm_b.pred(i)).collect::<Vec<_>>(),
                sse_b_i,
            )
        } else {
            (1, pred_z_b.to_vec(), sse_z_b)
        };
        mismatch += usize::from(win_a != win_b);
        risk_a += (1.0 - nu) * 0.0 + nu * sse_a_used / mstep as f64;
        risk_b += nu * sse_b_used / mstep as f64;
        for i in 0..n {
            u_a[i] -= nu * pred_a_used[i];
            u_b[i] -= nu * pred_b_used[i];
        }
    }
    println!("selection     mismatches={mismatch}/{mstep} mean_risk_a={risk_a:.9e} mean_risk_b={risk_b:.9e}");
    if mismatch > 0 || (risk_a - risk_b).abs() / risk_a > 1e-6 {
        println!("GATE 3 FAILED: selection or risk diverge");
        std::process::exit(1);
    }
    println!("gate 3 ok");

    // timing: paired alternating rounds of one hot candidate step.
    let warmup = 5;
    let rounds = 20;
    let mut ta = Vec::with_capacity(rounds);
    let mut tb = Vec::with_capacity(rounds);
    let swap = (warmup + rounds) % 2;
    let rounds_total = warmup + rounds;
    for r in 0..rounds_total {
        if (r + swap) % 2 == 0 {
            let t0 = std::time::Instant::now();
            let _ = black_box(arm_a.fit_step_ref(u_a.view(), weights.view()));
            if r >= warmup {
                ta.push(t0.elapsed().as_secs_f64() * 1e3);
            }
        } else {
            let t0 = std::time::Instant::now();
            let _ = black_box(arm_b.step(&u_b));
            if r >= warmup {
                tb.push(t0.elapsed().as_secs_f64() * 1e3);
            }
        }
    }
    let mut ta_sorted = ta.clone();
    let mut tb_sorted = tb.clone();
    let med_a = median(&mut ta_sorted);
    let med_b = median(&mut tb_sorted);
    let max_a = ta.iter().copied().fold(0.0, f64::max);
    let max_b = tb.iter().copied().fold(0.0, f64::max);
    let ratio = med_a / med_b;
    println!("hot step ms   materialised med={med_a:.4} max={max_a:.4} | factored med={med_b:.4} max={max_b:.4}");
    println!("ratio         materialised/factored = {ratio:.1}x");
    println!(
        "setup ms      materialised={:.3} ({arm_a_count} allocs, {arm_a_bytes} B) | factored={:.3} ({arm_b_count} allocs, {arm_b_bytes} B)",
        arm_a_setup.as_secs_f64() * 1e3,
        arm_b_setup.as_secs_f64() * 1e3
    );
    println!(
        "tensor        prebuild {tensor_build_count} allocs, {tensor_build_bytes} B; materialised T+K {tensor_bytes} B"
    );
}
