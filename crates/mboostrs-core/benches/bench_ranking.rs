//! Gehan ranked vs pairwise kernel cost (P3.4 item 5).
//!
//! The two kernels must agree before anything is timed: the bench builds the
//! fixture, computes both gradients and both risks, and asserts agreement
//! within summing-order rounding. Only then does it time each arm, so a
//! ranked rewrite that changed the result would fail the gate in setup
//! rather than produce a number for a different computation.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use mboostrs_core::families::{
    gehan_ngradient_core, gehan_ngradient_ranked, gehan_risk_core, gehan_risk_ranked,
};

fn xorshift(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed;
    move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state as f64) / (u64::MAX as f64)
    }
}

/// Residuals spread over ~[-3, 3], ~60% events, weights in (0, 2] with a
/// 5% fraction of zeros -- the shape a survival fit actually sees.
fn gehan_fixture(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = xorshift(7);
    let res: Vec<f64> = (0..n).map(|_| rng() * 6.0 - 3.0).collect();
    let del: Vec<f64> = (0..n)
        .map(|_| if rng() < 0.6 { 1.0 } else { 0.0 })
        .collect();
    let w: Vec<f64> = (0..n)
        .map(|_| if rng() < 0.05 { 0.0 } else { 1.0 + rng() })
        .collect();
    (res, del, w)
}

fn bench_gehan_ranked_vs_pairwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("gehan_ranked_vs_pairwise");
    for &n in &[400usize, 2000, 6000, 12000] {
        let (res, del, w) = gehan_fixture(n);

        let mut ranked_out = vec![0.0; n];
        gehan_ngradient_ranked(&res, &del, &w, &mut ranked_out);
        let mut pairwise_out = vec![0.0; n];
        gehan_ngradient_core(&res, &del, &w, &mut pairwise_out);
        for (a, b) in pairwise_out.iter().zip(&ranked_out) {
            assert!(
                (a - b).abs() <= 1e-9 * a.abs().max(1.0),
                "ranked gradient diverged from pairwise at n = {n}"
            );
        }
        let pairwise_risk = gehan_risk_core(&res, &del, &w);
        let ranked_risk = gehan_risk_ranked(&res, &del, &w);
        assert!(
            (pairwise_risk - ranked_risk).abs() <= 1e-9 * pairwise_risk.abs().max(1.0),
            "ranked risk diverged from pairwise at n = {n}"
        );

        group.bench_with_input(
            BenchmarkId::new("pairwise", n),
            &(res.clone(), del.clone(), w.clone()),
            |b, (res, del, w)| {
                b.iter(|| {
                    let mut out = vec![0.0; res.len()];
                    gehan_ngradient_core(res, del, w, &mut out);
                    std::hint::black_box(out);
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("ranked", n),
            &(res, del, w),
            |b, (res, del, w)| {
                b.iter(|| {
                    let mut out = vec![0.0; res.len()];
                    gehan_ngradient_ranked(res, del, w, &mut out);
                    std::hint::black_box(out);
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_gehan_ranked_vs_pairwise);
criterion_main!(benches);
