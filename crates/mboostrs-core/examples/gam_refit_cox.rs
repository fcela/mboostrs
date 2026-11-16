//! Cox refit on `tests/fixtures/cox_refit.csv` (eight bbs terms); prints the
//! edf per term. mgcv's cox.ph gives 5.512 1 2.601 1.766 1.001 1.722 1 1.001.
//!
//! cargo run --release -p mboostrs-core --example gam_refit_cox
use mboostrs_core::gam::{try_gam_refit, RefitControl, RefitFamily, RefitTerm, TermBasis};
use ndarray::Array1;

fn main() {
    let text =
        std::fs::read_to_string("tests/fixtures/cox_refit.csv").expect("run from the repo root");
    let mut lines = text.lines();
    let header: Vec<&str> = lines
        .next()
        .unwrap()
        .split(',')
        .map(|s| s.trim_matches('"'))
        .collect();
    let rows: Vec<Vec<f64>> = lines
        .map(|l| l.split(',').map(|v| v.parse().unwrap()).collect())
        .collect();
    let col = |name: &str| {
        let j = header.iter().position(|h| *h == name).unwrap();
        Array1::from_iter(rows.iter().map(|r| r[j]))
    };
    let terms: Vec<RefitTerm> = (1..=8)
        .map(|j| {
            let x = col(&format!("x{j}"));
            let lo = x.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let basis = TermBasis::Bbs {
                boundary: (lo, hi),
                n_knots: 20,
                degree: 3,
                differences: 2,
                cyclic: false,
            };
            RefitTerm {
                name: format!("x{j}"),
                design: basis.design(&[x.view()], false).unwrap(),
                penalties: basis.penalties(),
                by: None,
                variables: Vec::new(),
            }
        })
        .collect();
    let family = RefitFamily::CoxPH { time: col("time") };
    let status = col("status");
    for outer in [false, true] {
        let control = RefitControl {
            outer_iteration: outer,
            ..RefitControl::default()
        };
        let fit =
            try_gam_refit(terms.clone(), status.view(), None, None, &family, &control).unwrap();
        let edf: Vec<String> = fit.blocks.iter().map(|b| format!("{:.3}", b.edf)).collect();
        println!(
            "{} V={:.4} edf {}",
            fit.method,
            fit.reml_score,
            edf.join(" ")
        );
    }
}
