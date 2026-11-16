//! Timing probe for `gam::try_gam_refit` on a 24 x 24 tensor surface
//! (576 columns, two penalties), the size of a default `bspatial` term.
//!
//! cargo run --release -p mboostrs-core --example gam_refit_timing
use mboostrs_core::gam::{try_gam_refit, RefitControl, RefitFamily, RefitTerm};
use mboostrs_core::learners::{bspline_basis, bspline_knots, diff_penalty};
use ndarray::{Array1, Array2};
use std::time::Instant;

fn main() {
    let n = 1000;
    let mut state = 42u64;
    let mut unif = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let lon: Vec<f64> = (0..n).map(|_| unif()).collect();
    let lat: Vec<f64> = (0..n).map(|_| unif()).collect();
    let y = Array1::from_shape_fn(n, |i| (4.0 * lon[i]).sin() * lat[i] + 0.3 * (unif() - 0.5));
    let margin = |x: &[f64]| {
        let knots = bspline_knots((0.0, 1.0), 20, 3);
        bspline_basis(x, &knots, 3)
    };
    let (a, b) = (margin(&lon), margin(&lat));
    let (qa, qb) = (a.ncols(), b.ncols());
    let design = Array2::from_shape_fn((n, qa * qb), |(i, j)| a[[i, j / qb]] * b[[i, j % qb]]);
    let kron = |l: &Array2<f64>, r: &Array2<f64>| {
        let (r1, c1) = l.dim();
        let (r2, c2) = r.dim();
        Array2::from_shape_fn((r1 * r2, c1 * c2), |(i, j)| {
            l[[i / r2, j / c2]] * r[[i % r2, j % c2]]
        })
    };
    let penalties = vec![
        kron(&diff_penalty(qa, 2), &Array2::eye(qb)),
        kron(&Array2::eye(qa), &diff_penalty(qb, 2)),
    ];
    let term = RefitTerm {
        name: "te".into(),
        design,
        penalties,
        by: None,
        variables: Vec::new(),
    };
    for _ in 0..3 {
        let start = Instant::now();
        let fit = try_gam_refit(
            vec![term.clone()],
            y.view(),
            None,
            None,
            &RefitFamily::Gaussian,
            &RefitControl::default(),
        )
        .unwrap();
        println!(
            "{:.3} s, edf {:.3}, newton {}",
            start.elapsed().as_secs_f64(),
            fit.blocks[0].edf,
            fit.newton_iterations
        );
    }
}
