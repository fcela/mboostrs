//! Fixed-cost profile for a 30-knot bspatial + six bbs design on the
//! California housing spatial-cell split (#276): where per-fit preparation
//! time goes, and how much of a 2,000-step fit (the notebook protocol's
//! cap) is fixed setup rather than boosting loop.
//!
//! Usage: cargo run --release -p mboostrs-core --example bench_spatial_prep
//!        [-- /path/to/train.csv]
//!
//! The CSV needs columns lon,lat,log_value plus six covariates
//! (log_income, age, log_rooms, bed_share, log_occupancy, log_households),
//! as written by benches/bench_spatial_gram.py's split.

use ndarray::Array1;
use std::env;
use std::fs;
use std::time::Instant;

use mboostrs_core::{
    df_to_lambda, fit_with, prepare_bbs, prepare_block_bols, BoostControl, Gaussian,
    LearnerBlueprint, PreparationTimer, PreparedLearner,
};

fn main() {
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/issue276_train.csv".to_string());
    let raw = fs::read_to_string(&path).expect("read csv");
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for line in raw.lines().skip(1) {
        rows.push(line.split(',').map(|v| v.parse().unwrap()).collect());
    }
    let n = rows.len();
    assert_eq!(rows[0].len(), 9, "expected lon,lat,y + 6 covariates");
    let col = |j: usize| Array1::from_iter((0..n).map(|i| rows[i][j]));
    let lon = col(0);
    let lat = col(1);
    let y = col(2);
    let weights = Array1::ones(n);
    let covariates: Vec<Array1<f64>> = (3..9).map(col).collect();

    // 1. Blueprint construction: tensor basis + Kronecker penalty.
    let started = Instant::now();
    let blueprint = LearnerBlueprint::bspatial(
        lon.clone(),
        lat.clone(),
        "spatial",
        30,
        3,
        2,
        6.0,
        None,
        false,
        None,
        None,
        None,
    )
    .unwrap();
    let basis_ms = started.elapsed().as_secs_f64() * 1e3;

    let (design, penalty) = match &blueprint {
        LearnerBlueprint::Bspatial {
            design, penalty, ..
        } => (design.clone(), penalty.clone()),
        _ => unreachable!(),
    };
    let p = design.ncols();

    // 2. df -> lambda alone: gram + Cholesky + inverse backsolve + two
    //    matrix products + spectrum.
    let started = Instant::now();
    let lambda = df_to_lambda(&design, &penalty, &weights.view(), 6.0);
    let df_ms = started.elapsed().as_secs_f64() * 1e3;

    // 3. prepare_block_bols alone: design clone + penalized gram + solver.
    let started = Instant::now();
    let penalized = penalty.mapv(|value| value * lambda);
    let _block = prepare_block_bols(
        design.clone(),
        weights.view(),
        penalized.clone(),
        false,
        "spatial",
    );
    let block_ms = started.elapsed().as_secs_f64() * 1e3;

    // 4. The production prepare path with the binding's own timer split.
    let timer = PreparationTimer::start();
    let (spatial, _) = blueprint.prepare(weights.view()).unwrap();
    let timing = timer.finish();
    let branch_lambda = match &spatial {
        PreparedLearner::BlockBols { lambda, .. } => lambda.as_ref().copied(),
        _ => None,
    };
    println!(
        "lambda old df_to_lambda: {:.17e} | branch prepare: {:.17e} | rel diff: {:.3e}",
        lambda,
        branch_lambda.unwrap_or(f64::NAN),
        (branch_lambda.unwrap_or(f64::NAN) - lambda).abs() / lambda,
    );

    println!("n = {n}, spatial p = {p}, lambda(df=6) = {lambda:.6e}");
    println!("blueprint (tensor basis + penalty): {basis_ms:8.1} ms");
    println!("df_to_lambda alone:                 {df_ms:8.1} ms");
    println!("prepare_block_bols alone:           {block_ms:8.1} ms");
    println!(
        "blueprint.prepare (production):     {:8.1} ms  (basis {:8.1} ms, factorization {:8.1} ms)",
        timing.total_ns as f64 / 1e6,
        timing.basis_ns as f64 / 1e6,
        timing.factorization_ns as f64 / 1e6,
    );

    // 5. The notebook protocol's fit: cap 2000, nu 0.5.
    let names = [
        "log_income",
        "age",
        "log_rooms",
        "bed_share",
        "log_occupancy",
        "log_households",
    ];
    let mut learners: Vec<PreparedLearner> = vec![spatial];
    for (x, name) in covariates.iter().zip(names) {
        learners.push(prepare_bbs(
            x.view(),
            weights.view(),
            20,
            3,
            2,
            4.0,
            true,
            name,
        ));
    }
    let control = BoostControl::new(2000, 0.5).unwrap();
    let started = Instant::now();
    let path = fit_with(
        &mut learners,
        y.view(),
        weights.view(),
        &mut Gaussian,
        &control,
        None,
        false,
    );
    let fit_ms = started.elapsed().as_secs_f64() * 1e3;
    let report = &path.report;
    println!(
        "fit(cap=2000, nu=0.5): total {fit_ms:8.1} ms | route {:?} | fallbacks {:?} | gram setup {:8.1} ms | loop {:8.1} ms ({} iters) | reconstruction {:8.1} ms",
        report.route,
        report.fallback_reasons,
        report.setup_ns.unwrap_or(0) as f64 / 1e6,
        report.iteration_ns.unwrap_or(0) as f64 / 1e6,
        path.selected.len(),
        report.reconstruction_ns.unwrap_or(0) as f64 / 1e6,
    );
    let per_iter = fit_ms / path.selected.len() as f64;
    println!("wall ms/iteration (notebook metric): {per_iter:.3}");
    let fixed = fit_ms - report.iteration_ns.unwrap_or(0) as f64 / 1e6;
    println!("fixed (prep + setup + reconstruction): {fixed:.0} ms of {fit_ms:.0} ms total");
}
