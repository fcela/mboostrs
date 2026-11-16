//! Python bindings of the scale-model column ranking in
//! `mboostrs_core::conformal` and, below it, the deselection and probing
//! arithmetic in `mboostrs_core::selection`.

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::mboostrs_error_to_pyerr;

/// `log(max((y - predicted)^2, 1e-12))`, the fitted scale model's target.
#[pyfunction]
pub fn conformal_scale_target(
    y: PyReadonlyArray1<'_, f64>,
    predicted: PyReadonlyArray1<'_, f64>,
) -> PyResult<Vec<f64>> {
    mboostrs_core::conformal::scale_target(&y.as_array().to_vec(), &predicted.as_array().to_vec())
        .map_err(mboostrs_error_to_pyerr)
}

/// Indices of the `keep` columns (rows of `columns`) most associated with
/// the target, strongest first.
#[pyfunction]
pub fn conformal_rank_scale_columns(
    columns: PyReadonlyArray2<'_, f64>,
    target: PyReadonlyArray1<'_, f64>,
    keep: usize,
) -> PyResult<Vec<usize>> {
    let view = columns.as_array();
    let rows: Vec<Vec<f64>> = view.rows().into_iter().map(|r| r.to_vec()).collect();
    mboostrs_core::conformal::rank_scale_columns(&rows, &target.as_array().to_vec(), keep)
        .map_err(mboostrs_error_to_pyerr)
}

/// Register every function of this file on the extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(conformal_scale_target, m)?)?;
    m.add_function(wrap_pyfunction!(conformal_rank_scale_columns, m)?)?;
    m.add_function(wrap_pyfunction!(deselect_lss_shares, m)?)?;
    m.add_function(wrap_pyfunction!(deselect_group_sums, m)?)?;
    m.add_function(wrap_pyfunction!(deselect_keep, m)?)?;
    m.add_function(wrap_pyfunction!(probe_permutations, m)?)?;
    m.add_function(wrap_pyfunction!(probe_stop, m)?)?;
    Ok(())
}

/// Shares of the total risk reduction per `(parameter, learner)` of a
/// non-cyclic distributional fit, as `(parameter, learner, share)` rows.
#[pyfunction]
pub fn deselect_lss_shares(
    parameter: Vec<usize>,
    learner: Vec<usize>,
    risk_change: Vec<f64>,
) -> PyResult<Vec<(usize, usize, f64)>> {
    let shares = mboostrs_core::selection::lss_learner_shares(&parameter, &learner, &risk_change)
        .map_err(mboostrs_error_to_pyerr)?;
    Ok(shares
        .into_iter()
        .map(|s| (s.parameter, s.learner, s.share))
        .collect())
}

/// Sums of consecutive groups of values (the shares of a term's learners).
#[pyfunction]
pub fn deselect_group_sums(values: Vec<f64>, lengths: Vec<usize>) -> PyResult<Vec<f64>> {
    mboostrs_core::selection::group_sums(&values, &lengths).map_err(mboostrs_error_to_pyerr)
}

/// Which terms have `share >= threshold`.
#[pyfunction]
pub fn deselect_keep(shares: Vec<f64>, threshold: f64) -> PyResult<Vec<bool>> {
    mboostrs_core::selection::deselect_keep(&shares, threshold).map_err(mboostrs_error_to_pyerr)
}

/// `count` seeded permutations of `0..n`, one per shadow column.
#[pyfunction]
pub fn probe_permutations(n: usize, count: usize, seed: u64) -> Vec<Vec<usize>> {
    mboostrs_core::selection::probe_permutations(n, count, seed)
}

/// `(stop, first_shadow, selected)` read off the selected-learner sequence.
#[pyfunction]
pub fn probe_stop(
    selected: Vec<usize>,
    is_shadow: Vec<bool>,
) -> PyResult<(Option<usize>, Option<usize>, Vec<usize>)> {
    let r = mboostrs_core::selection::probe_stop(&selected, &is_shadow)
        .map_err(mboostrs_error_to_pyerr)?;
    Ok((r.stop, r.first_shadow, r.selected))
}
