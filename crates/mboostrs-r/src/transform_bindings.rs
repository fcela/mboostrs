//! R binding of `mboostrs_core::transform`: fitted column transforms.

use extendr_api::prelude::*;
use mboostrs_core::transform as t;

#[extendr]
fn r_transform_mean_sd(values: &[f64]) -> Vec<f64> {
    let (mean, sd) = t::mean_sd(values);
    vec![mean, sd]
}

#[extendr]
fn r_transform_median(values: &[f64]) -> f64 {
    t::median(values)
}

/// Knots followed by probabilities of `ecdf(x)`.
#[extendr]
fn r_transform_ecdf_params(values: &[f64]) -> List {
    let (knots, probabilities) =
        t::ecdf_params(values).unwrap_or_else(|e| throw_r_error(e.to_string()));
    list!(knots = knots, probabilities = probabilities)
}

#[extendr]
fn r_transform_forward(kind: &str, params: &[f64], x: &[f64]) -> Vec<f64> {
    t::forward(kind, params, x).unwrap_or_else(|e| throw_r_error(e.to_string()))
}

#[extendr]
fn r_transform_inverse(kind: &str, params: &[f64], v: &[f64]) -> Vec<f64> {
    t::inverse(kind, params, v).unwrap_or_else(|e| throw_r_error(e.to_string()))
}

#[extendr]
fn r_transform_jacobian(kind: &str, params: &[f64], y: &[f64]) -> Vec<f64> {
    t::jacobian(kind, params, y).unwrap_or_else(|e| throw_r_error(e.to_string()))
}

extendr_module! {
    mod transform_bindings;
    fn r_transform_mean_sd;
    fn r_transform_median;
    fn r_transform_ecdf_params;
    fn r_transform_forward;
    fn r_transform_inverse;
    fn r_transform_jacobian;
}
