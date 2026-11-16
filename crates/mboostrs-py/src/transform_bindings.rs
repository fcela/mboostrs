//! Python binding of `mboostrs_core::transform`: fitted column transforms.

use mboostrs_core::transform as t;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use crate::mboostrs_error_to_pyerr;

/// Mean and standard deviation (`ddof = 1`) of a training column.
#[pyfunction]
pub fn transform_mean_sd(values: PyReadonlyArray1<'_, f64>) -> (f64, f64) {
    t::mean_sd(&values.as_array().to_vec())
}

/// Median of a column; NaN when it is empty or holds a NaN.
#[pyfunction]
pub fn transform_median(values: PyReadonlyArray1<'_, f64>) -> f64 {
    t::median(&values.as_array().to_vec())
}

/// Training quantile knots and probabilities of `ecdf(x)`.
#[pyfunction]
pub fn transform_ecdf_params(values: PyReadonlyArray1<'_, f64>) -> PyResult<(Vec<f64>, Vec<f64>)> {
    t::ecdf_params(&values.as_array().to_vec()).map_err(mboostrs_error_to_pyerr)
}

/// Forward map of a recipe step.
#[pyfunction]
pub fn transform_forward(
    kind: &str,
    params: Vec<f64>,
    x: PyReadonlyArray1<'_, f64>,
) -> PyResult<Vec<f64>> {
    t::forward(kind, &params, &x.as_array().to_vec()).map_err(mboostrs_error_to_pyerr)
}

/// Inverse map of an invertible response transform.
#[pyfunction]
pub fn transform_inverse(
    kind: &str,
    params: Vec<f64>,
    v: PyReadonlyArray1<'_, f64>,
) -> PyResult<Vec<f64>> {
    t::inverse(kind, &params, &v.as_array().to_vec()).map_err(mboostrs_error_to_pyerr)
}

/// `|dg/dy|` of a response transform at original response values.
#[pyfunction]
pub fn transform_jacobian(
    kind: &str,
    params: Vec<f64>,
    y: PyReadonlyArray1<'_, f64>,
) -> PyResult<Vec<f64>> {
    t::jacobian(kind, &params, &y.as_array().to_vec()).map_err(mboostrs_error_to_pyerr)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(transform_mean_sd, m)?)?;
    m.add_function(wrap_pyfunction!(transform_median, m)?)?;
    m.add_function(wrap_pyfunction!(transform_ecdf_params, m)?)?;
    m.add_function(wrap_pyfunction!(transform_forward, m)?)?;
    m.add_function(wrap_pyfunction!(transform_inverse, m)?)?;
    m.add_function(wrap_pyfunction!(transform_jacobian, m)?)?;
    Ok(())
}
