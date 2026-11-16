//! Python binding of `mboostrs_core::gam`: the REML refit.
//!
//! Python builds each term's raw design and penalties (it holds the formula
//! and the data frame); everything numeric happens in the core.

use mboostrs_core::gam::{try_gam_refit, RefitControl, RefitFamily, RefitResult, RefitTerm};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::mboostrs_error_to_pyerr;

/// A fitted REML refit held in Rust.
#[pyclass(module = "mboostrs._rust", name = "GamRefitCore")]
pub struct GamRefitCore {
    inner: RefitResult,
}

fn family_from(
    family: &str,
    theta: Option<f64>,
    time: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<RefitFamily> {
    Ok(match family {
        "Gaussian" => RefitFamily::Gaussian,
        "Binomial" => RefitFamily::Binomial,
        "Poisson" => RefitFamily::Poisson,
        "GammaReg" => RefitFamily::Gamma,
        "NBinomial" => RefitFamily::NegativeBinomial {
            theta: theta.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("an NBinomial refit needs theta")
            })?,
        },
        "CoxPH" => RefitFamily::CoxPH {
            time: time
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "a CoxPH refit needs the survival times",
                    )
                })?
                .as_array()
                .to_owned(),
        },
        other => {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                "gam_refit supports Gaussian, Binomial, Poisson, NBinomial, GammaReg and CoxPH; \
                 got {other}"
            )))
        }
    })
}

/// Refit terms as one penalized GAM with REML smoothing parameters.
///
/// `designs[i]` is term `i`'s raw design at the fitting rows, `penalties[i]`
/// its penalty matrices, `bys[i]` its `by=` column or `None`.
#[pyfunction]
#[pyo3(signature = (names, designs, penalties, bys, y, weights=None, offset=None,
                    family="Gaussian", theta=None, time=None, variables=None, gram=false))]
#[allow(clippy::too_many_arguments)]
pub fn gam_refit_core(
    py: Python<'_>,
    names: Vec<String>,
    designs: Vec<PyReadonlyArray2<'_, f64>>,
    penalties: Vec<Vec<PyReadonlyArray2<'_, f64>>>,
    bys: Vec<Option<PyReadonlyArray1<'_, f64>>>,
    y: PyReadonlyArray1<'_, f64>,
    weights: Option<PyReadonlyArray1<'_, f64>>,
    offset: Option<PyReadonlyArray1<'_, f64>>,
    family: &str,
    theta: Option<f64>,
    time: Option<PyReadonlyArray1<'_, f64>>,
    variables: Option<Vec<Vec<String>>>,
    gram: bool,
) -> PyResult<GamRefitCore> {
    let k = names.len();
    let variables = variables.unwrap_or_else(|| vec![Vec::new(); k]);
    if designs.len() != k || penalties.len() != k || bys.len() != k || variables.len() != k {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "names, designs, penalties, bys and variables must have one entry per term",
        ));
    }
    let family = family_from(family, theta, time)?;
    let terms: Vec<RefitTerm> = names
        .into_iter()
        .zip(designs)
        .zip(penalties)
        .zip(bys)
        .zip(variables)
        .map(|((((name, design), pens), by), variables)| RefitTerm {
            name,
            design: design.as_array().to_owned(),
            penalties: pens.iter().map(|p| p.as_array().to_owned()).collect(),
            by: by.map(|b| b.as_array().to_owned()),
            variables,
        })
        .collect();
    let y = y.as_array().to_owned();
    let weights = weights.map(|w| w.as_array().to_owned());
    let offset = offset.map(|o| o.as_array().to_owned());
    let result = py.allow_threads(|| {
        try_gam_refit(
            terms,
            y.view(),
            weights.as_ref().map(|w| w.view()),
            offset.as_ref().map(|o| o.view()),
            &family,
            &RefitControl {
                gram,
                ..RefitControl::default()
            },
        )
    });
    Ok(GamRefitCore {
        inner: result.map_err(mboostrs_error_to_pyerr)?,
    })
}

/// The REML refits of a row bootstrap, one per draw, held in Rust.
#[pyclass(module = "mboostrs._rust", name = "GamRefitBootstrapCore")]
pub struct GamRefitBootstrapCore {
    draws: Vec<RefitResult>,
}

#[pymethods]
impl GamRefitBootstrapCore {
    /// The number of draws.
    #[getter]
    fn n_draws(&self) -> usize {
        self.draws.len()
    }

    /// Whether each draw's REML optimization and PIRLS converged.
    fn converged(&self) -> Vec<bool> {
        self.draws.iter().map(|d| d.converged).collect()
    }

    /// The linear predictor of every draw at new rows, as a `draws x rows`
    /// array. The new rows' designs are those of `prediction`.
    fn linear_predictor<'py>(
        &self,
        py: Python<'py>,
        raws: Vec<PyReadonlyArray2<'_, f64>>,
        bys: Vec<Option<PyReadonlyArray1<'_, f64>>>,
        zero_rows: Vec<bool>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let terms = new_rows_terms(&raws, &bys, &zero_rows)?;
        let eta = mboostrs_core::gam::bootstrap_linear_predictor(&self.draws, &terms)
            .map_err(mboostrs_error_to_pyerr)?;
        Ok(eta.into_pyarray(py))
    }
}

/// Refit the terms once per row of multiplicities `counts` (`B x n`), with the
/// full-sample designs and penalties and the Gaussian family.
///
/// Draw `b` has prior weights `weights * counts[b]`; see
/// `mboostrs_core::gam::refit_row_bootstrap`.
#[pyfunction]
#[pyo3(signature = (names, designs, penalties, bys, y, counts, weights=None, variables=None))]
#[allow(clippy::too_many_arguments)]
pub fn gam_refit_bootstrap_core(
    py: Python<'_>,
    names: Vec<String>,
    designs: Vec<PyReadonlyArray2<'_, f64>>,
    penalties: Vec<Vec<PyReadonlyArray2<'_, f64>>>,
    bys: Vec<Option<PyReadonlyArray1<'_, f64>>>,
    y: PyReadonlyArray1<'_, f64>,
    counts: PyReadonlyArray2<'_, f64>,
    weights: Option<PyReadonlyArray1<'_, f64>>,
    variables: Option<Vec<Vec<String>>>,
) -> PyResult<GamRefitBootstrapCore> {
    let k = names.len();
    let variables = variables.unwrap_or_else(|| vec![Vec::new(); k]);
    if designs.len() != k || penalties.len() != k || bys.len() != k || variables.len() != k {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "names, designs, penalties, bys and variables must have one entry per term",
        ));
    }
    let terms: Vec<RefitTerm> = names
        .into_iter()
        .zip(designs)
        .zip(penalties)
        .zip(bys)
        .zip(variables)
        .map(|((((name, design), pens), by), variables)| RefitTerm {
            name,
            design: design.as_array().to_owned(),
            penalties: pens.iter().map(|p| p.as_array().to_owned()).collect(),
            by: by.map(|b| b.as_array().to_owned()),
            variables,
        })
        .collect();
    let y = y.as_array().to_owned();
    let counts = counts.as_array().to_owned();
    let weights = weights.map(|w| w.as_array().to_owned());
    let draws = py.allow_threads(|| {
        mboostrs_core::gam::refit_row_bootstrap(
            &terms,
            y.view(),
            weights.as_ref().map(|w| w.view()),
            counts.view(),
            &RefitFamily::Gaussian,
            &RefitControl::default(),
        )
    });
    Ok(GamRefitBootstrapCore {
        draws: draws.map_err(mboostrs_error_to_pyerr)?,
    })
}

/// Rebuild a `GamRefitCore` from its JSON state (used by pickle).
#[pyfunction]
pub fn gam_refit_core_from_json(state: &str) -> PyResult<GamRefitCore> {
    let value: serde_json::Value = serde_json::from_str(state)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(GamRefitCore {
        inner: RefitResult::from_json(&value).map_err(mboostrs_error_to_pyerr)?,
    })
}

/// Rebuild a `GamlssRefitCore` from its JSON state (used by pickle).
#[pyfunction]
pub fn gamlss_refit_core_from_json(family: String, state: &str) -> PyResult<GamlssRefitCore> {
    let value: serde_json::Value = serde_json::from_str(state)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(GamlssRefitCore {
        inner: mboostrs_core::gam::LssRefitResult::from_json(&value)
            .map_err(mboostrs_error_to_pyerr)?,
        family_name: family,
    })
}

#[pymethods]
impl GamRefitCore {
    /// Pickle support: rebuilt by `gam_refit_core_from_json`.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, (String,))> {
        let module = py.import("mboostrs._rust")?;
        Ok((
            module.getattr("gam_refit_core_from_json")?,
            (self.inner.to_json().to_string(),),
        ))
    }

    /// The refit as a JSON string.
    fn to_json(&self) -> String {
        self.inner.to_json().to_string()
    }

    /// Coefficients: intercept, then each term's constrained coefficients.
    #[getter]
    fn coef<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.coef.clone().into_pyarray(py)
    }

    /// Bayesian covariance of the coefficients.
    #[getter]
    fn cov<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.cov.clone().into_pyarray(py)
    }

    /// Covariance corrected for smoothing-parameter uncertainty.
    #[getter]
    fn cov_corrected<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.cov_corrected.clone().into_pyarray(py)
    }

    /// Negative binomial theta used by the refit (estimated by default).
    #[getter]
    fn theta(&self) -> Option<f64> {
        self.inner.theta
    }

    #[getter]
    fn scale(&self) -> f64 {
        self.inner.scale
    }

    #[getter]
    fn reml_score(&self) -> f64 {
        self.inner.reml_score
    }

    #[getter]
    fn rho<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.rho.clone().into_pyarray(py)
    }

    #[getter]
    fn rho_hessian<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.rho_hessian.clone().into_pyarray(py)
    }

    #[getter]
    fn edf_total(&self) -> f64 {
        self.inner.edf_total
    }

    /// Log-likelihood of the fitted model (mgcv's `logLik.gam` convention).
    #[getter]
    fn loglik(&self) -> f64 {
        self.inner.loglik
    }

    /// Degrees of freedom for AIC (edf plus an estimated scale or dispersion).
    #[getter]
    fn df(&self) -> f64 {
        self.inner.df
    }

    #[getter]
    fn link(&self) -> &'static str {
        self.inner.link
    }

    #[getter]
    fn intercept(&self) -> f64 {
        self.inner.intercept()
    }

    /// Convergence report: iterations, gradient norm and whether both loops
    /// converged.
    fn convergence<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("converged", self.inner.converged)?;
        d.set_item("stop_reason", self.inner.stop_reason)?;
        d.set_item("newton_iterations", self.inner.newton_iterations)?;
        d.set_item("pirls_iterations", self.inner.pirls_iterations)?;
        d.set_item("gradient_norm", self.inner.gradient_norm)?;
        d.set_item(
            "unidentified_directions",
            self.inner.unidentified_directions,
        )?;
        d.set_item("unidentified_terms", self.inner.unidentified_terms.clone())?;
        d.set_item("method", self.inner.method)?;
        Ok(d)
    }

    /// One dict per term: name, columns, edf, lambda, rank, has_by.
    fn blocks<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.inner
            .blocks
            .iter()
            .map(|b| {
                let d = PyDict::new(py);
                d.set_item("name", &b.name)?;
                d.set_item("start", b.start)?;
                d.set_item("end", b.end)?;
                d.set_item("edf", b.edf)?;
                d.set_item("lambda", b.lambda.clone())?;
                d.set_item("rank", b.rank)?;
                d.set_item("has_by", b.has_by)?;
                if let Some(t) = &b.test {
                    d.set_item("ref_df", t.ref_df)?;
                    d.set_item("statistic", t.statistic)?;
                    d.set_item("p_value", t.p_value)?;
                }
                if let Some(c) = &b.concurvity {
                    d.set_item("concurvity_worst", c.worst)?;
                    d.set_item("concurvity_observed", c.observed)?;
                    d.set_item("concurvity_estimate", c.estimate)?;
                }
                Ok(d)
            })
            .collect()
    }

    /// Residual degrees of freedom when the scale was estimated, else None.
    #[getter]
    fn residual_df(&self) -> Option<f64> {
        self.inner.residual_df
    }

    /// Number of observations the refit used.
    #[getter]
    fn n(&self) -> usize {
        self.inner.n
    }

    /// Term `b`'s effect and standard error at new rows from its raw design;
    /// `corrected` uses the covariance corrected for smoothing-parameter
    /// uncertainty.
    #[pyo3(signature = (b, raw, by=None, zero_rows_have_no_effect=false, corrected=false))]
    fn effect<'py>(
        &self,
        py: Python<'py>,
        b: usize,
        raw: PyReadonlyArray2<'_, f64>,
        by: Option<PyReadonlyArray1<'_, f64>>,
        zero_rows_have_no_effect: bool,
        corrected: bool,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        let block =
            self.inner.blocks.get(b).ok_or_else(|| {
                pyo3::exceptions::PyIndexError::new_err(format!("no refit term {b}"))
            })?;
        let r = block.start..block.end;
        let cov = if corrected {
            &self.inner.cov_corrected
        } else {
            &self.inner.cov
        };
        let (estimate, se) = mboostrs_core::gam::block_effect(
            raw.as_array(),
            block.householder.view(),
            self.inner.coef.slice(ndarray::s![r.clone()]),
            cov.slice(ndarray::s![r.clone(), r]),
            block.has_by,
            by.as_ref().map(|v| v.as_array()),
            zero_rows_have_no_effect,
        )
        .map_err(mboostrs_error_to_pyerr)?;
        Ok((estimate.into_pyarray(py), se.into_pyarray(py)))
    }

    /// The summed linear predictor (intercept plus every term) at new rows and
    /// its standard error from the full covariance. `raws[b]`, `bys[b]` and
    /// `zero_rows[b]` describe term `b` (in the order of `blocks()`) there.
    #[pyo3(signature = (raws, bys, zero_rows, corrected=false))]
    fn prediction<'py>(
        &self,
        py: Python<'py>,
        raws: Vec<PyReadonlyArray2<'_, f64>>,
        bys: Vec<Option<PyReadonlyArray1<'_, f64>>>,
        zero_rows: Vec<bool>,
        corrected: bool,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        let terms = new_rows_terms(&raws, &bys, &zero_rows)?;
        let (eta, se) = self
            .inner
            .linear_predictor(&terms, corrected)
            .map_err(mboostrs_error_to_pyerr)?;
        Ok((eta.into_pyarray(py), se.into_pyarray(py)))
    }

    /// The average change in the linear predictor from the rows `lo` to the
    /// rows `hi`, as `(estimate, se)`. The arguments of `prediction` are given
    /// once for each row set, with the same row count in both.
    #[pyo3(signature = (hi_raws, hi_bys, hi_zero, lo_raws, lo_bys, lo_zero, corrected=false))]
    fn contrast(
        &self,
        hi_raws: Vec<PyReadonlyArray2<'_, f64>>,
        hi_bys: Vec<Option<PyReadonlyArray1<'_, f64>>>,
        hi_zero: Vec<bool>,
        lo_raws: Vec<PyReadonlyArray2<'_, f64>>,
        lo_bys: Vec<Option<PyReadonlyArray1<'_, f64>>>,
        lo_zero: Vec<bool>,
        corrected: bool,
    ) -> PyResult<(f64, f64)> {
        let hi = new_rows_terms(&hi_raws, &hi_bys, &hi_zero)?;
        let lo = new_rows_terms(&lo_raws, &lo_bys, &lo_zero)?;
        self.inner
            .linear_contrast(&hi, &lo, corrected)
            .map_err(mboostrs_error_to_pyerr)
    }
}

/// The terms at new rows, one per entry of the three parallel lists.
fn new_rows_terms(
    raws: &[PyReadonlyArray2<'_, f64>],
    bys: &[Option<PyReadonlyArray1<'_, f64>>],
    zero_rows: &[bool],
) -> PyResult<Vec<mboostrs_core::gam::NewRowsTerm>> {
    if raws.len() != bys.len() || raws.len() != zero_rows.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "raws, bys and zero_rows must have one entry per term",
        ));
    }
    Ok(raws
        .iter()
        .zip(bys)
        .zip(zero_rows)
        .map(|((r, b), z)| mboostrs_core::gam::NewRowsTerm {
            raw: r.as_array().to_owned(),
            by: b.as_ref().map(|v| v.as_array().to_owned()),
            zero_rows_have_no_effect: *z,
        })
        .collect())
}

/// The terms at new rows for each distribution parameter.
fn nested_new_rows_terms(
    raws: &[Vec<PyReadonlyArray2<'_, f64>>],
    bys: &[Vec<Option<PyReadonlyArray1<'_, f64>>>],
    zero_rows: &[Vec<bool>],
) -> PyResult<Vec<Vec<mboostrs_core::gam::NewRowsTerm>>> {
    if raws.len() != bys.len() || raws.len() != zero_rows.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "raws, bys and zero_rows must have one entry per distribution parameter",
        ));
    }
    raws.iter()
        .zip(bys)
        .zip(zero_rows)
        .map(|((r, b), z)| new_rows_terms(r, b, z))
        .collect()
}

/// A fitted GAMLSS REML refit held in Rust.
#[pyclass(module = "mboostrs._rust", name = "GamlssRefitCore")]
pub struct GamlssRefitCore {
    inner: mboostrs_core::gam::LssRefitResult,
    family_name: String,
}

type TermsPerParam<'py> = Vec<Vec<PyReadonlyArray2<'py, f64>>>;

/// Refit a distributional model's terms, parameter by parameter, as one
/// penalized likelihood with REML smoothing parameters.
///
/// `names[q]`, `designs[q]`, `penalties[q]` and `bys[q]` describe parameter
/// `q`'s terms as in `gam_refit_core`; `start_etas[q]` is the boosted fit's
/// link-scale predictor for parameter `q` (or `None`).
#[pyfunction]
#[pyo3(signature = (family, param_names, names, designs, penalties, bys, y, weights=None,
                    start_etas=None, variables=None))]
#[allow(clippy::too_many_arguments)]
pub fn gamlss_refit_core(
    py: Python<'_>,
    family: &str,
    param_names: Vec<String>,
    names: Vec<Vec<String>>,
    designs: TermsPerParam<'_>,
    penalties: Vec<Vec<Vec<PyReadonlyArray2<'_, f64>>>>,
    bys: Vec<Vec<Option<PyReadonlyArray1<'_, f64>>>>,
    y: PyReadonlyArray1<'_, f64>,
    weights: Option<PyReadonlyArray1<'_, f64>>,
    start_etas: Option<Vec<PyReadonlyArray1<'_, f64>>>,
    variables: Option<Vec<Vec<Vec<String>>>>,
) -> PyResult<GamlssRefitCore> {
    let k = param_names.len();
    let variables =
        variables.unwrap_or_else(|| names.iter().map(|qn| vec![Vec::new(); qn.len()]).collect());
    if names.len() != k || designs.len() != k || penalties.len() != k || bys.len() != k {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "names, designs, penalties and bys need one list per distribution parameter",
        ));
    }
    let fam = mboostrs_core::lss::make_family(family, k).map_err(mboostrs_error_to_pyerr)?;
    let mut terms = Vec::with_capacity(k);
    if variables.len() != k {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "variables needs one list per distribution parameter",
        ));
    }
    for ((((qn, qd), qp), qb), qv) in names
        .into_iter()
        .zip(designs)
        .zip(penalties)
        .zip(bys)
        .zip(variables)
    {
        if qd.len() != qn.len()
            || qp.len() != qn.len()
            || qb.len() != qn.len()
            || qv.len() != qn.len()
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "a parameter's names, designs, penalties, bys and variables differ in length",
            ));
        }
        terms.push(
            qn.into_iter()
                .zip(qd)
                .zip(qp)
                .zip(qb)
                .zip(qv)
                .map(|((((name, design), pens), by), variables)| RefitTerm {
                    name,
                    design: design.as_array().to_owned(),
                    penalties: pens.iter().map(|p| p.as_array().to_owned()).collect(),
                    by: by.map(|b| b.as_array().to_owned()),
                    variables,
                })
                .collect::<Vec<_>>(),
        );
    }
    let y = y.as_array().to_owned();
    let weights = weights.map(|w| w.as_array().to_owned());
    let start: Option<Vec<ndarray::Array1<f64>>> =
        start_etas.map(|v| v.iter().map(|e| e.as_array().to_owned()).collect());
    let family_name = family.to_string();
    let result = py.allow_threads(|| {
        mboostrs_core::gam::try_gamlss_refit(
            terms,
            param_names,
            y.view(),
            weights.as_ref().map(|w| w.view()),
            fam.as_ref(),
            start.as_deref(),
            &RefitControl::default(),
        )
    });
    Ok(GamlssRefitCore {
        inner: result.map_err(mboostrs_error_to_pyerr)?,
        family_name,
    })
}

#[pymethods]
impl GamlssRefitCore {
    /// Pickle support: rebuilt by `gamlss_refit_core_from_json`.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, (String, String))> {
        let module = py.import("mboostrs._rust")?;
        Ok((
            module.getattr("gamlss_refit_core_from_json")?,
            (self.family_name.clone(), self.inner.to_json().to_string()),
        ))
    }

    /// The refit as a JSON string.
    fn to_json(&self) -> String {
        self.inner.to_json().to_string()
    }

    /// Joint coefficients (each parameter's intercept, then its terms).
    #[getter]
    fn coef<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.coef.clone().into_pyarray(py)
    }

    /// Bayesian covariance of the joint coefficients.
    #[getter]
    fn cov<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.cov.clone().into_pyarray(py)
    }

    /// Covariance corrected for smoothing-parameter uncertainty.
    #[getter]
    fn cov_corrected<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.cov_corrected.clone().into_pyarray(py)
    }

    #[getter]
    fn reml_score(&self) -> f64 {
        self.inner.reml_score
    }

    #[getter]
    fn edf_total(&self) -> f64 {
        self.inner.edf_total
    }

    /// Convergence report.
    fn convergence<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("converged", self.inner.converged)?;
        d.set_item("iterations", self.inner.iterations)?;
        d.set_item("stop_reason", self.inner.stop_reason)?;
        d.set_item("evaluations", self.inner.evaluations)?;
        d.set_item("restarted", self.inner.restarted)?;
        d.set_item("restart_iterations", self.inner.restart_iterations)?;
        d.set_item("seconds", self.inner.seconds)?;
        d.set_item("inner_iterations", self.inner.inner_iterations)?;
        d.set_item("gradient_norm", self.inner.gradient_norm)?;
        d.set_item("information_floored", self.inner.information_floored)?;
        d.set_item("method", "outer")?;
        Ok(d)
    }

    /// One dict per parameter: name, start, end, and its terms (name, edf,
    /// lambda, rank, has_by).
    fn params<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.inner
            .params
            .iter()
            .map(|p| {
                let d = PyDict::new(py);
                d.set_item("name", &p.name)?;
                d.set_item("start", p.start)?;
                d.set_item("end", p.end)?;
                let blocks: PyResult<Vec<Bound<'py, PyDict>>> = p
                    .blocks
                    .iter()
                    .map(|b| {
                        let e = PyDict::new(py);
                        e.set_item("name", &b.name)?;
                        e.set_item("edf", b.edf)?;
                        e.set_item("lambda", b.lambda.clone())?;
                        e.set_item("rank", b.rank)?;
                        e.set_item("has_by", b.has_by)?;
                        Ok(e)
                    })
                    .collect();
                d.set_item("blocks", blocks?)?;
                Ok(d)
            })
            .collect()
    }

    /// Intercept of parameter `q`.
    fn intercept(&self, q: usize) -> PyResult<f64> {
        let p =
            self.inner.params.get(q).ok_or_else(|| {
                pyo3::exceptions::PyIndexError::new_err(format!("no parameter {q}"))
            })?;
        Ok(self.inner.coef[p.start])
    }

    /// Term `b` of parameter `q`: effect and standard error at new rows.
    #[pyo3(signature = (q, b, raw, by=None, zero_rows_have_no_effect=false, corrected=false))]
    #[allow(clippy::too_many_arguments)]
    fn effect<'py>(
        &self,
        py: Python<'py>,
        q: usize,
        b: usize,
        raw: PyReadonlyArray2<'_, f64>,
        by: Option<PyReadonlyArray1<'_, f64>>,
        zero_rows_have_no_effect: bool,
        corrected: bool,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        let block = self
            .inner
            .params
            .get(q)
            .and_then(|p| p.blocks.get(b))
            .ok_or_else(|| {
                pyo3::exceptions::PyIndexError::new_err(format!("no term {b} of parameter {q}"))
            })?;
        let r = block.start..block.end;
        let cov = if corrected {
            &self.inner.cov_corrected
        } else {
            &self.inner.cov
        };
        let (estimate, se) = mboostrs_core::gam::block_effect(
            raw.as_array(),
            block.householder.view(),
            self.inner.coef.slice(ndarray::s![r.clone()]),
            cov.slice(ndarray::s![r.clone(), r]),
            block.has_by,
            by.as_ref().map(|v| v.as_array()),
            zero_rows_have_no_effect,
        )
        .map_err(mboostrs_error_to_pyerr)?;
        Ok((estimate.into_pyarray(py), se.into_pyarray(py)))
    }

    /// Posterior draws of every parameter on its natural scale at new rows:
    /// `raws[q][b]`, `bys[q][b]` and `zero_rows[q][b]` describe term `b` of
    /// parameter `q` there. Returns one `(n_draws, n_rows)` array per
    /// parameter.
    #[pyo3(signature = (raws, bys, zero_rows, n_draws=1000, seed=0, corrected=false))]
    #[allow(clippy::too_many_arguments)]
    fn parameter_draws<'py>(
        &self,
        py: Python<'py>,
        raws: Vec<Vec<PyReadonlyArray2<'_, f64>>>,
        bys: Vec<Vec<Option<PyReadonlyArray1<'_, f64>>>>,
        zero_rows: Vec<Vec<bool>>,
        n_draws: usize,
        seed: u64,
        corrected: bool,
    ) -> PyResult<Vec<Bound<'py, PyArray2<f64>>>> {
        let k = self.inner.params.len();
        let fam = mboostrs_core::lss::make_family(&self.family_name, k)
            .map_err(mboostrs_error_to_pyerr)?;
        let terms = nested_new_rows_terms(&raws, &bys, &zero_rows)?;
        let draws = self
            .inner
            .parameter_draws(fam.as_ref(), &terms, n_draws, seed, corrected)
            .map_err(mboostrs_error_to_pyerr)?;
        Ok(draws.into_iter().map(|d| d.into_pyarray(py)).collect())
    }

    /// The linear predictor of every parameter at new rows, with the standard
    /// error from that parameter's coefficient covariance. The arguments are
    /// as for `parameter_draws`; returns one `(eta, se)` pair per parameter.
    #[pyo3(signature = (raws, bys, zero_rows, corrected=false))]
    fn parameter_predictions<'py>(
        &self,
        py: Python<'py>,
        raws: Vec<Vec<PyReadonlyArray2<'_, f64>>>,
        bys: Vec<Vec<Option<PyReadonlyArray1<'_, f64>>>>,
        zero_rows: Vec<Vec<bool>>,
        corrected: bool,
    ) -> PyResult<Vec<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)>> {
        let terms = nested_new_rows_terms(&raws, &bys, &zero_rows)?;
        let pairs = self
            .inner
            .parameter_predictors(&terms, corrected)
            .map_err(mboostrs_error_to_pyerr)?;
        Ok(pairs
            .into_iter()
            .map(|(eta, se)| (eta.into_pyarray(py), se.into_pyarray(py)))
            .collect())
    }

    /// The average change in every parameter's linear predictor from the rows
    /// `lo` to the rows `hi`, as one `(estimate, se)` pair per parameter. The
    /// arguments are as for `parameter_predictions`, once for each row set.
    #[pyo3(signature = (hi_raws, hi_bys, hi_zero, lo_raws, lo_bys, lo_zero, corrected=false))]
    fn parameter_contrasts(
        &self,
        hi_raws: Vec<Vec<PyReadonlyArray2<'_, f64>>>,
        hi_bys: Vec<Vec<Option<PyReadonlyArray1<'_, f64>>>>,
        hi_zero: Vec<Vec<bool>>,
        lo_raws: Vec<Vec<PyReadonlyArray2<'_, f64>>>,
        lo_bys: Vec<Vec<Option<PyReadonlyArray1<'_, f64>>>>,
        lo_zero: Vec<Vec<bool>>,
        corrected: bool,
    ) -> PyResult<Vec<(f64, f64)>> {
        let hi = nested_new_rows_terms(&hi_raws, &hi_bys, &hi_zero)?;
        let lo = nested_new_rows_terms(&lo_raws, &lo_bys, &lo_zero)?;
        self.inner
            .parameter_contrasts(&hi, &lo, corrected)
            .map_err(mboostrs_error_to_pyerr)
    }

    /// Parameter `q` on its natural scale from its linear predictor, by the
    /// family's inverse link.
    fn response<'py>(
        &self,
        py: Python<'py>,
        q: usize,
        eta: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let k = self.inner.params.len();
        if q >= k {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "no parameter {q}"
            )));
        }
        let fam = mboostrs_core::lss::make_family(&self.family_name, k)
            .map_err(mboostrs_error_to_pyerr)?;
        let eta = eta.as_array().to_vec();
        let mut out = vec![0.0; eta.len()];
        fam.response(q, &eta, &mut out);
        Ok(ndarray::Array1::from_vec(out).into_pyarray(py))
    }
}

fn basis_from_dict(recipe: &Bound<'_, PyDict>) -> PyResult<mboostrs_core::gam::TermBasis> {
    use mboostrs_core::gam::TermBasis;
    let get = |key: &str| -> PyResult<Bound<'_, PyAny>> {
        recipe.get_item(key)?.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("basis recipe needs {key:?}"))
        })
    };
    let kind: String = get("kind")?.extract()?;
    Ok(match kind.as_str() {
        "bbs" => {
            let (lo, hi): (f64, f64) = get("boundary")?.extract()?;
            // "cyclic" is optional: recipes recorded before the cyclic
            // bbs-by-factor work carry no such key and mean the open basis.
            let cyclic: bool = recipe
                .get_item("cyclic")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(false);
            TermBasis::Bbs {
                boundary: (lo, hi),
                n_knots: get("n_knots")?.extract()?,
                degree: get("degree")?.extract()?,
                differences: get("differences")?.extract()?,
                cyclic,
            }
        }
        "bols" => TermBasis::Bols {
            intercept: get("intercept")?.extract()?,
            ridge: get("ridge")?.extract()?,
        },
        "brandom" => TermBasis::Brandom {
            levels: get("levels")?.extract()?,
        },
        "byfactor" => TermBasis::ByFactor {
            levels: get("levels")?.extract()?,
        },
        "tensor" => TermBasis::Tensor(
            Box::new(basis_from_dict(get("left")?.downcast()?)?),
            Box::new(basis_from_dict(get("right")?.downcast()?)?),
        ),
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown basis kind {other:?}"
            )))
        }
    })
}

/// A term's raw design at the given variable values and its penalties,
/// built by the core from a basis recipe (`{"kind": "bbs", "boundary": (lo,
/// hi), "n_knots", "degree", "differences", "cyclic" (optional, default
/// false)}`, `{"kind": "bols",
/// "intercept", "ridge"}`, `{"kind": "brandom" | "byfactor", "levels"}` with
/// level codes, or `{"kind": "tensor", "left", "right"}`).
#[pyfunction]
#[pyo3(signature = (recipe, vars, unseen_zero=false))]
pub fn term_basis_design<'py>(
    py: Python<'py>,
    recipe: &Bound<'py, PyDict>,
    vars: Vec<PyReadonlyArray1<'py, f64>>,
    unseen_zero: bool,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Vec<Bound<'py, PyArray2<f64>>>)> {
    let basis = basis_from_dict(recipe)?;
    let views: Vec<_> = vars.iter().map(|v| v.as_array()).collect();
    let design = basis
        .design(&views, unseen_zero)
        .map_err(mboostrs_error_to_pyerr)?;
    let penalties = basis
        .penalties()
        .into_iter()
        .map(|p| p.into_pyarray(py))
        .collect();
    Ok((design.into_pyarray(py), penalties))
}

/// Row-wise (Khatri-Rao) tensor product of two designs.
#[pyfunction]
pub fn row_tensor<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if a.as_array().nrows() != b.as_array().nrows() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "row_tensor needs designs with the same number of rows",
        ));
    }
    Ok(mboostrs_core::gam::row_tensor(a.as_array(), b.as_array()).into_pyarray(py))
}

/// Kronecker product of two matrices.
#[pyfunction]
pub fn kron<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray2<'py, f64>,
) -> Bound<'py, PyArray2<f64>> {
    mboostrs_core::gam::kron(&a.as_array().to_owned(), &b.as_array().to_owned()).into_pyarray(py)
}
