//! R bindings of the same split-inference and screening kernels as Python.

use extendr_api::prelude::*;
use mboostrs_core::selection_stats as stats;
use ndarray::Array2;

fn matrix(x: &RMatrix<f64>) -> Array2<f64> {
    Array2::from_shape_fn((x.nrows(), x.ncols()), |(i, j)| x[[i, j]])
}

#[extendr]
fn r_aggregate_split_p(p_values: &[f64], gamma: f64) -> f64 {
    stats::aggregate_split_p(p_values, gamma).unwrap_or_else(|e| throw_r_error(e.to_string()))
}

#[extendr]
fn r_normal_interval_p(estimate: f64, lower: f64, upper: f64, level: f64) -> f64 {
    stats::normal_interval_p(estimate, lower, upper, level)
        .unwrap_or_else(|e| throw_r_error(e.to_string()))
}

#[extendr]
fn r_draw_p(draws: &[f64]) -> f64 {
    stats::draw_p(draws).unwrap_or_else(|e| throw_r_error(e.to_string()))
}

#[extendr]
fn r_draw_contrast(
    alternative: &[f64],
    reference: &[f64],
    draws_alternative: RMatrix<f64>,
    draws_reference: RMatrix<f64>,
    level: f64,
) -> List {
    let (estimate, lower, upper, p_value) = stats::draw_contrast(
        alternative,
        reference,
        matrix(&draws_alternative).view(),
        matrix(&draws_reference).view(),
        level,
    )
    .unwrap_or_else(|e| throw_r_error(e.to_string()));
    list!(
        estimate = estimate,
        lower = lower,
        upper = upper,
        p_value = p_value
    )
}

#[extendr]
fn r_paired_mean_se(before: &[f64], after: &[f64]) -> List {
    let (gain, se) = mboostrs_core::contribution::paired_mean_se(before, after)
        .unwrap_or_else(|e| throw_r_error(e.to_string()));
    list!(gain = gain, se = se)
}

#[extendr]
fn r_log_variance_screen(residual: &[f64], design: RMatrix<f64>) -> Robj {
    match stats::log_variance_screen(residual, matrix(&design).view())
        .unwrap_or_else(|e| throw_r_error(e.to_string()))
    {
        Some(s) => list!(
            slopes = s.slopes,
            t_stats = s.t_stats,
            r_squared = s.r_squared,
            n_used = s.n_used as i32
        )
        .into(),
        None => r!(NULL),
    }
}

extendr_module! {
    mod selection_stats;
    fn r_aggregate_split_p;
    fn r_normal_interval_p;
    fn r_draw_p;
    fn r_draw_contrast;
    fn r_paired_mean_se;
    fn r_log_variance_screen;
}
