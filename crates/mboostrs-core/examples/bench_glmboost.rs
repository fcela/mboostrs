use mboostrs_core::{
    control::BoostControl, families::Gaussian, fit, learners::prepare_scalar_bols,
};
use ndarray::Array1;

fn main() {
    let n: usize = 50_000;
    let mstop: u32 = 100;
    let nu = 0.1;

    let mut rng = 42u64;
    let mut xorshift = || -> f64 {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng as f64) / (u64::MAX as f64)
    };

    let x1: Array1<f64> = (0..n).map(|_| xorshift() * 2.0 - 1.0).collect();
    let x2: Array1<f64> = (0..n).map(|_| xorshift() * 2.0 - 1.0).collect();
    let y: Array1<f64> = (0..n)
        .map(|i| 1.5 * x1[i] + 0.8 * x2[i] + (xorshift() - 0.5) * 0.6)
        .collect();
    let w = Array1::ones(n);

    let l1 = prepare_scalar_bols(x1.view(), w.view(), 0.0, true, "x1", None);
    let l2 = prepare_scalar_bols(x2.view(), w.view(), 0.0, true, "x2", None);
    let mut learners = vec![l1, l2];

    let ctrl = BoostControl::new(mstop, nu).unwrap();

    // Warmup
    let mut fam = Gaussian;
    let _ = fit(&mut learners, y.view(), w.view(), &mut fam, &ctrl, None);

    // Timed runs
    for _ in 0..5 {
        let mut fam = Gaussian;
        let t0 = std::time::Instant::now();
        let _path = fit(&mut learners, y.view(), w.view(), &mut fam, &ctrl, None);
        let elapsed = t0.elapsed();
        eprintln!("fit: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    }
}
