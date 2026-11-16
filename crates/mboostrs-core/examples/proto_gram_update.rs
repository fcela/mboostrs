// PROTOTYPE (go/no-go speed signal) for the Gaussian covariance/Gram-update
// boosting loop — see .remember/experiment-gaussian-gram-update-2026-05-30.md
//
// Builds 2 centered bbs learners, then runs two Gaussian component-wise
// boosting loops on the SAME data:
//   1. NAIVE: recompute rhs_j = B_j' u every iteration (what the engine does)
//   2. GRAM : maintain rhs_j incrementally via precomputed cross-Grams G_jw
// Cross-checks they produce identical selected[]/coefs/risk, then times both.
//
// Self-contained: no engine internals, no Python, no brandom files.
use mboostrs_core::linalg::GramSolver;
use mboostrs_core::{bspline_basis, bspline_knots, center_spectral, df_to_lambda, diff_penalty};
use ndarray::{Array1, Array2};
use std::time::Instant;

fn build_centered_bbs(
    x: &Array1<f64>,
    n_knots: usize,
    degree: usize,
    diff: usize,
    df: f64,
) -> (Array2<f64>, f64) // (centered design B (n,p), lambda)
{
    let bk = (
        x.iter().cloned().fold(f64::INFINITY, f64::min),
        x.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );
    let knots = bspline_knots(bk, n_knots, degree);
    let basis = bspline_basis(&x.to_vec(), &knots, degree);
    let k = diff_penalty(basis.ncols(), diff);
    let centered = center_spectral(&basis, &k, diff);
    let p = centered.ncols();
    let eye = Array2::<f64>::eye(p);
    let w = Array1::<f64>::ones(x.len());
    let lambda = df_to_lambda(&centered, &eye, &w, df);
    (centered, lambda)
}

struct Learner {
    b: Array2<f64>,       // (n, p) centered design
    lambda: f64,          // ridge penalty (penalty P = lambda * I)
    a_solver: GramSolver, // factorization of A = B'B + lambda I
    p: usize,
}

fn prep(b: Array2<f64>, lambda: f64) -> Learner {
    let p = b.ncols();
    let mut a = b.t().dot(&b);
    for i in 0..p {
        a[[i, i]] += lambda;
    }
    let a_solver = GramSolver::new(&a);
    Learner {
        b,
        lambda,
        a_solver,
        p,
    }
}

fn main() {
    let n = 10_000usize;
    let mstop: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let nu = 0.1f64;

    // deterministic fixture
    let mut rng = 42u64;
    let mut xs = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng as f64) / (u64::MAX as f64)
    };
    let x1: Array1<f64> = (0..n).map(|_| xs()).collect();
    let x2: Array1<f64> = (0..n).map(|_| xs()).collect();
    let tau = std::f64::consts::TAU;
    let y: Array1<f64> = (0..n)
        .map(|i| (tau * x1[i]).sin() + (tau * x2[i]).cos() + (xs() - 0.5) * 0.4)
        .collect();

    let (b0, l0) = build_centered_bbs(&x1, 20, 3, 2, 4.0);
    let (b1, l1) = build_centered_bbs(&x2, 20, 3, 2, 4.0);
    let learners = [prep(b0, l0), prep(b1, l1)];
    println!("p per learner = {}", learners[0].p);

    let offset = y.iter().sum::<f64>() / n as f64;

    // ---------- NAIVE loop (reference) ----------
    let t_naive = Instant::now();
    let (sel_n, coef_n, risk_n) = {
        let mut u = &y - offset;
        let mut sel = Vec::with_capacity(mstop);
        let mut coef: Vec<Array1<f64>> = Vec::with_capacity(mstop);
        let mut risk = Vec::with_capacity(mstop + 1);
        risk.push(u.dot(&u));
        for _ in 0..mstop {
            let mut best = 0usize;
            let mut best_sse = f64::INFINITY;
            let mut best_pred = Array1::zeros(n);
            let mut best_beta = Array1::zeros(1);
            for (j, lr) in learners.iter().enumerate() {
                let rhs = lr.b.t().dot(&u);
                let beta = lr.a_solver.solve(&rhs.view());
                let pred = lr.b.dot(&beta);
                let resid = &u - &pred;
                let sse = resid.dot(&resid);
                if sse < best_sse {
                    best_sse = sse;
                    best = j;
                    best_pred = pred;
                    best_beta = beta;
                }
            }
            u = &u - &(nu * &best_pred);
            sel.push(best);
            coef.push(best_beta);
            risk.push(u.dot(&u));
        }
        (sel, coef, risk)
    };
    let naive_ms = t_naive.elapsed().as_secs_f64() * 1e3;

    // ---------- GRAM-update loop ----------
    let t_setup = Instant::now();
    // cross-Grams G[j][w] = B_j' B_w (p x p)
    let l = learners.len();
    let mut g = vec![vec![Array2::<f64>::zeros((0, 0)); l]; l];
    for j in 0..l {
        for w in 0..l {
            g[j][w] = learners[j].b.t().dot(&learners[w].b);
        }
    }
    let u0 = &y - offset;
    let mut rhs: Vec<Array1<f64>> = learners.iter().map(|lr| lr.b.t().dot(&u0)).collect();
    let mut wu2 = u0.dot(&u0);
    let setup_ms = t_setup.elapsed().as_secs_f64() * 1e3;

    let t_gram = Instant::now();
    let (sel_g, coef_g, risk_g) = {
        let mut sel = Vec::with_capacity(mstop);
        let mut coef: Vec<Array1<f64>> = Vec::with_capacity(mstop);
        let mut risk = Vec::with_capacity(mstop + 1);
        risk.push(wu2);
        for _ in 0..mstop {
            let mut best = 0usize;
            let mut best_sse = f64::INFINITY;
            let mut best_beta = Array1::zeros(1);
            for (j, lr) in learners.iter().enumerate() {
                let beta = lr.a_solver.solve(&rhs[j].view());
                // SSE = wu2 - beta·rhs - lambda*||beta||^2   (P = lambda I)
                let sse = wu2 - beta.dot(&rhs[j]) - lr.lambda * beta.dot(&beta);
                if sse < best_sse {
                    best_sse = sse;
                    best = j;
                    best_beta = beta;
                }
            }
            let w = best;
            // wu2 update: -2nu beta_w·rhs_w + nu^2 beta_w' (B_w'B_w) beta_w
            let gww_beta = g[w][w].dot(&best_beta);
            wu2 = wu2 - 2.0 * nu * best_beta.dot(&rhs[w]) + nu * nu * best_beta.dot(&gww_beta);
            // rhs_j -= nu * G_jw beta_w   for all j
            for j in 0..l {
                let upd = g[j][w].dot(&best_beta);
                rhs[j] = &rhs[j] - &(nu * &upd);
            }
            sel.push(w);
            coef.push(best_beta);
            risk.push(wu2);
        }
        (sel, coef, risk)
    };
    let gram_ms = t_gram.elapsed().as_secs_f64() * 1e3;

    // ---------- cross-check ----------
    let sel_match = sel_n == sel_g;
    let mut max_coef_d = 0.0f64;
    for (a, b) in coef_n.iter().zip(&coef_g) {
        for k in 0..a.len().min(b.len()) {
            max_coef_d = max_coef_d.max((a[k] - b[k]).abs());
        }
    }
    let mut max_risk_d = 0.0f64;
    for (a, b) in risk_n.iter().zip(&risk_g) {
        max_risk_d = max_risk_d.max((a - b).abs());
    }
    let final_risk_rel = (risk_n[mstop] - risk_g[mstop]).abs() / risk_n[mstop].abs();

    println!("--- correctness (naive vs gram, mstop={mstop}) ---");
    println!("selected sequence identical : {sel_match}");
    println!("max |coef diff|             : {max_coef_d:.3e}");
    println!("max |risk diff| (abs)       : {max_risk_d:.3e}");
    println!("final risk rel diff (drift) : {final_risk_rel:.3e}");
    println!("--- speed ---");
    println!(
        "NAIVE total  : {naive_ms:.2} ms  ({:.1} us/iter)",
        naive_ms / mstop as f64 * 1e3
    );
    println!("GRAM  setup  : {setup_ms:.2} ms");
    println!(
        "GRAM  loop   : {gram_ms:.2} ms  ({:.1} us/iter)",
        gram_ms / mstop as f64 * 1e3
    );
    println!("GRAM  total  : {:.2} ms", setup_ms + gram_ms);
    println!(
        "loop speedup (naive/gram inner) : {:.1}x",
        naive_ms / gram_ms
    );
    println!(
        "total speedup (naive / gram+setup): {:.2}x",
        naive_ms / (setup_ms + gram_ms)
    );
}
