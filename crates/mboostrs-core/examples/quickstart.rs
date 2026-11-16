use mboostrs_core::{prepare_scalar_bols, try_fit, BoostControl, Gaussian};
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Array1::from_vec(vec![-1.0, -0.5, 0.0, 0.5, 1.0]);
    let y = Array1::from_vec(vec![-1.4, -0.8, 0.1, 0.7, 1.6]);
    let weights = Array1::ones(x.len());

    let learner = prepare_scalar_bols(x.view(), weights.view(), 0.0, true, "x", None);
    let control = BoostControl::new(25, 0.1)?;
    let path = try_fit(
        &mut [learner],
        y.view(),
        weights.view(),
        &mut Gaussian,
        &control,
        None,
    )?;

    println!("selected steps: {}", path.selected.len());
    println!("final risk: {:.6}", path.risk.last().unwrap());
    Ok(())
}
