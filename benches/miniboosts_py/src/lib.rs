//! `fit_predict`: train one miniboosts booster with depth-limited decision
//! trees on a CSV file (target coded -1/+1) and return its confidence (the
//! weighted vote, in [-1, 1]) on the rows of a second CSV file.

use miniboosts::prelude::*;
use pyo3::prelude::*;

fn read(path: &str, target: &str) -> PyResult<Sample> {
    SampleReader::new()
        .file(path)
        .has_header(true)
        .target_feature(target)
        .read()
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{path}: {e}")))
}

#[pyfunction]
#[pyo3(signature = (algorithm, train_csv, test_csv, target, max_depth=2, tolerance=0.01, nu_share=0.2, max_rounds=500))]
#[allow(clippy::too_many_arguments)]
fn fit_predict(
    algorithm: &str,
    train_csv: &str,
    test_csv: &str,
    target: &str,
    max_depth: usize,
    tolerance: f64,
    nu_share: f64,
    max_rounds: usize,
) -> PyResult<Vec<f64>> {
    let train = read(train_csv, target)?;
    let test = read(test_csv, target)?;
    let n = train.shape().0 as f64;
    let nu = (nu_share * n).clamp(1.0, n);
    let tree = DecisionTreeBuilder::new(&train)
        .max_depth(max_depth)
        .criterion(Criterion::Entropy)
        .build();
    let scores = match algorithm {
        "adaboost" => AdaBoost::init(&train)
            .tolerance(tolerance)
            .force_quit_at(max_rounds)
            .run(&tree)
            .confidence_all(&test),
        "lpboost" => LPBoost::init(&train)
            .tolerance(tolerance)
            .nu(nu)
            .run(&tree)
            .confidence_all(&test),
        "erlpboost" => ERLPBoost::init(&train)
            .tolerance(tolerance)
            .nu(nu)
            .run(&tree)
            .confidence_all(&test),
        "softboost" => SoftBoost::init(&train)
            .tolerance(tolerance)
            .nu(nu)
            .run(&tree)
            .confidence_all(&test),
        "mlpboost" => MLPBoost::init(&train)
            .tolerance(tolerance)
            .nu(nu)
            .run(&tree)
            .confidence_all(&test),
        "smoothboost" => SmoothBoost::init(&train)
            .tolerance(tolerance)
            .run(&tree)
            .confidence_all(&test),
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown algorithm {other:?}"
            )))
        }
    };
    Ok(scores)
}

#[pymodule]
fn miniboosts_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit_predict, m)?)?;
    Ok(())
}
