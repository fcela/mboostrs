// Times a 2× bbs gamboost fit at n=10k, mstop=200 — the workload where the
// per-iter block-learner GEMV dominates. Used to measure Phase 3a/1 changes
// to the engine block path without going through the (currently broken)
// Python layer.
use mboostrs_core::{
    control::BoostControl, families::Gaussian, fit, fit_with, learners::prepare_bbs,
};
use ndarray::Array1;
use std::time::Instant;

fn main() {
    let n: usize = 10_000;
    let mstop: u32 = 200;
    let nu = 0.1;

    let mut rng = 42u64;
    let mut xorshift = || -> f64 {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng as f64) / (u64::MAX as f64)
    };

    let x1: Array1<f64> = (0..n).map(|_| xorshift()).collect();
    let x2: Array1<f64> = (0..n).map(|_| xorshift()).collect();
    let two_pi = std::f64::consts::TAU;
    let y: Array1<f64> = (0..n)
        .map(|i| (two_pi * x1[i]).sin() + (two_pi * x2[i]).cos() + (xorshift() - 0.5) * 0.4)
        .collect();
    let w = Array1::ones(n);

    let ctrl = BoostControl::new(mstop, nu).unwrap();

    let build = || {
        vec![
            prepare_bbs(x1.view(), w.view(), 20, 3, 2, 4.0, true, "bbs(x1)"),
            prepare_bbs(x2.view(), w.view(), 20, 3, 2, 4.0, true, "bbs(x2)"),
        ]
    };

    let reps = 5;
    // `fit` uses store_increments=true → general path. `fit_with(.., false)`
    // takes the Gaussian Gram fast path.
    let bench = |gram: bool| -> f64 {
        // warm
        {
            let mut learners = build();
            let mut fam = Gaussian;
            if gram {
                let _ = fit_with(
                    &mut learners,
                    y.view(),
                    w.view(),
                    &mut fam,
                    &ctrl,
                    None,
                    false,
                );
            } else {
                let _ = fit(&mut learners, y.view(), w.view(), &mut fam, &ctrl, None);
            }
        }
        let mut times = Vec::with_capacity(reps);
        for _ in 0..reps {
            let mut learners = build();
            let mut fam = Gaussian;
            let t0 = Instant::now();
            if gram {
                let _ = fit_with(
                    &mut learners,
                    y.view(),
                    w.view(),
                    &mut fam,
                    &ctrl,
                    None,
                    false,
                );
            } else {
                let _ = fit(&mut learners, y.view(), w.view(), &mut fam, &ctrl, None);
            }
            times.push(t0.elapsed().as_secs_f64());
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        times[reps / 2]
    };

    let general = bench(false);
    let gram = bench(true);
    println!(
        "general  : {:.2} ms total ({:.0} us/iter)",
        general * 1e3,
        general / mstop as f64 * 1e6
    );
    println!(
        "gram     : {:.2} ms total ({:.0} us/iter)",
        gram * 1e3,
        gram / mstop as f64 * 1e6
    );
    println!("speedup  : {:.1}x", general / gram);
}
