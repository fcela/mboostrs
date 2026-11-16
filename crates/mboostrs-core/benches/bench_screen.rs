//! Screening pass cost (#298 phase 1): the held-out residual-MSE pair and
//! main screens on the three frames EVAL_298.md's cost model priced.
//!
//! The gate before timing (the `bench_ranking` pattern): every fixture
//! carries a planted x0*x1 interaction and asserts that pair wins its
//! ranking first, so the numbers describe the screen finding structure,
//! not scanning garbage.
//!
//! Frames and targets, from EVAL_298.md §"Cost model":
//! - 24k x 13 (Beijing shape, 78 pairs): pair screen under 10 ms.
//! - 24k x 100 (4950 pairs, one planted interaction): well under 1 s,
//!   measured single-threaded and across pairs with rayon.
//! - 1e6 x 20 (190 pairs): the long-frame case, to confirm the pass is
//!   linear in n and memory-bandwidth-bound.

use criterion::{criterion_group, criterion_main, Criterion};
use mboostrs_core::screen::{screen_main_effects, screen_pairs};

fn xorshift(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed;
    move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state as f64) / (u64::MAX as f64)
    }
}

/// An n x p frame of iid uniform(-1, 1) columns whose residual carries a
/// planted x0*x1 interaction plus uniform noise.
fn frame(seed: u64, n: usize, p: usize) -> (ndarray::Array2<f64>, Vec<f64>) {
    let mut rng = xorshift(seed);
    let mut x = ndarray::Array2::<f64>::zeros((n, p));
    let mut residual = Vec::with_capacity(n);
    for i in 0..n {
        for j in 0..p {
            x[[i, j]] = rng() * 2.0 - 1.0;
        }
        residual.push(x[[i, 0]] * x[[i, 1]] + (rng() - 0.5) * 0.2);
    }
    (x, residual)
}

/// Assert the planted pair won the last frame's ranking, then hand the
/// frame to the caller for timing.
fn gate(x: &ndarray::Array2<f64>, residual: &[f64]) {
    let ranked = screen_pairs(x.view(), residual, 4, 0, &vec![0; x.ncols()], &[], None)
        .expect("the fixture must screen cleanly");
    assert_eq!(
        (ranked[0].left, ranked[0].right),
        (0, 1),
        "the planted pair must win the ranking, got {:?}",
        ranked[0]
    );
}

fn bench_screen_pass(c: &mut Criterion) {
    let mut group = c.benchmark_group("screen");

    // 24k x 13, the Beijing-shaped frame: 78 pairs, plus the p-column main
    // pass on the same frame.
    let (x13, r13) = frame(7, 24_000, 13);
    gate(&x13, &r13);
    group.bench_function("pairs_24k_x_13", |b| {
        b.iter(|| {
            let scored = screen_pairs(x13.view(), &r13, 4, 0, &[0; 13], &[], None).unwrap();
            std::hint::black_box(scored);
        })
    });
    group.bench_function("mains_24k_x_13", |b| {
        b.iter(|| {
            let scored = screen_main_effects(x13.view(), &r13, 4, 0, &[0; 13], None).unwrap();
            std::hint::black_box(scored);
        })
    });

    // 24k x 100: 4950 pairs, one planted interaction. Serial is a one-thread
    // rayon pool via `install`; the rayon arm runs on the global pool.
    let (x100, r100) = frame(11, 24_000, 100);
    gate(&x100, &r100);
    let serial = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("the one-thread pool must build");
    group.bench_function("pairs_24k_x_100_serial", |b| {
        b.iter(|| {
            let scored = serial
                .install(|| screen_pairs(x100.view(), &r100, 4, 0, &[0; 100], &[], None))
                .unwrap();
            std::hint::black_box(scored);
        })
    });
    group.bench_function("pairs_24k_x_100_rayon", |b| {
        b.iter(|| {
            let scored = screen_pairs(x100.view(), &r100, 4, 0, &[0; 100], &[], None).unwrap();
            std::hint::black_box(scored);
        })
    });

    // 1e6 x 20: the long frame, linear in n at fixed p.
    let (xlong, rlong) = frame(13, 1_000_000, 20);
    gate(&xlong, &rlong);
    group.bench_function("pairs_1e6_x_20", |b| {
        b.iter(|| {
            let scored = screen_pairs(xlong.view(), &rlong, 4, 0, &[0; 20], &[], None).unwrap();
            std::hint::black_box(scored);
        })
    });

    group.finish();
}

criterion_group!(benches, bench_screen_pass);
criterion_main!(benches);
