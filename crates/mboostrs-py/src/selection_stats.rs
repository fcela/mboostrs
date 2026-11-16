//! Thin bindings for split-inference summaries and specification screens.

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::mboostrs_error_to_pyerr;
use mboostrs_core::selection_stats as stats;

#[pyfunction]
fn aggregate_split_p(p_values: PyReadonlyArray1<'_, f64>, gamma: f64) -> PyResult<f64> {
    stats::aggregate_split_p(&p_values.as_array().to_vec(), gamma).map_err(mboostrs_error_to_pyerr)
}

#[pyfunction]
fn normal_interval_p(estimate: f64, lower: f64, upper: f64, level: f64) -> PyResult<f64> {
    stats::normal_interval_p(estimate, lower, upper, level).map_err(mboostrs_error_to_pyerr)
}

#[pyfunction]
fn draw_p(draws: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    stats::draw_p(&draws.as_array().to_vec()).map_err(mboostrs_error_to_pyerr)
}

#[pyfunction]
fn draw_contrast(
    alternative: PyReadonlyArray1<'_, f64>,
    reference: PyReadonlyArray1<'_, f64>,
    draws_alternative: PyReadonlyArray2<'_, f64>,
    draws_reference: PyReadonlyArray2<'_, f64>,
    level: f64,
) -> PyResult<(f64, f64, f64, f64)> {
    stats::draw_contrast(
        &alternative.as_array().to_vec(),
        &reference.as_array().to_vec(),
        draws_alternative.as_array(),
        draws_reference.as_array(),
        level,
    )
    .map_err(mboostrs_error_to_pyerr)
}

#[pyfunction]
fn log_variance_screen(
    py: Python<'_>,
    residual: PyReadonlyArray1<'_, f64>,
    design: PyReadonlyArray2<'_, f64>,
) -> PyResult<Option<(Vec<f64>, Vec<f64>, f64, usize)>> {
    let residual = residual.as_array().to_vec();
    let design = design.as_array().to_owned();
    py.allow_threads(|| stats::log_variance_screen(&residual, design.view()))
        .map(|r| r.map(|s| (s.slopes, s.t_stats, s.r_squared, s.n_used)))
        .map_err(mboostrs_error_to_pyerr)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(aggregate_split_p, m)?)?;
    m.add_function(wrap_pyfunction!(normal_interval_p, m)?)?;
    m.add_function(wrap_pyfunction!(draw_p, m)?)?;
    m.add_function(wrap_pyfunction!(draw_contrast, m)?)?;
    m.add_function(wrap_pyfunction!(log_variance_screen, m)?)?;
    Ok(())
}
