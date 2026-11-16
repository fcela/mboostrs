// Peak memory of preparing ten bbs learners (run under `/usr/bin/time -l`).
use mboostrs_core::learners::prepare_bbs;
use ndarray::Array1;

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(200_000);
    let j: usize = std::env::args()
        .nth(2)
        .and_then(|a| a.parse().ok())
        .unwrap_or(10);
    let w = Array1::<f64>::ones(n);
    let learners: Vec<_> = (0..j)
        .map(|k| {
            let x: Array1<f64> = (0..n)
                .map(|i| ((i * (k + 7)) % 997) as f64 / 997.0)
                .collect();
            prepare_bbs(
                x.view(),
                w.view(),
                20,
                3,
                2,
                4.0,
                std::env::args().nth(3).as_deref() == Some("center"),
                "bbs",
            )
        })
        .collect();
    let design_mb = (n * 24 * 8) as f64 / 1048576.0;
    println!(
        "prepared {} learners; one design = {design_mb:.0} MB",
        learners.len()
    );
}
