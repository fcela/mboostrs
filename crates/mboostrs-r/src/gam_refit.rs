//! R binding of `mboostrs_core::gam`: the REML refit.
//!
//! R describes each term by a basis recipe (a named list: `kind` and the
//! basis parameters) and its variables; the designs, the fit and the effects
//! at new rows are computed in Rust. The result is returned as plain R data,
//! so a refit object survives `saveRDS`.

use extendr_api::prelude::*;
use mboostrs_core::gam::{
    block_effect, try_gam_refit, RefitControl, RefitFamily, RefitTerm, TermBasis,
};
use ndarray::{Array1, Array2};

use crate::{list_named_elt, list_optional_elt};

fn real_vec(robj: &Robj, what: &str) -> Vec<f64> {
    robj.as_real_vector()
        .or_else(|| {
            robj.as_integer_vector()
                .map(|v| v.iter().map(|&x| x as f64).collect())
        })
        .unwrap_or_else(|| throw_r_error(format!("{what} must be numeric")))
}

fn usize_field(list: &List, name: &str) -> usize {
    let v = real_vec(&list_named_elt(list, name), name);
    match v.as_slice() {
        [x] if *x >= 0.0 && x.fract() == 0.0 => *x as usize,
        _ => throw_r_error(format!("{name} must be one non-negative whole number")),
    }
}

/// Parse a basis recipe: `list(kind = "bbs", boundary = c(lo, hi), n_knots,
/// degree, differences)`, `list(kind = "bols", intercept = TRUE)`,
/// `list(kind = "brandom", levels = c(...))` or
/// `list(kind = "tensor", left = <recipe>, right = <recipe>)`.
fn parse_basis(robj: &Robj) -> TermBasis {
    let list = robj
        .as_list()
        .unwrap_or_else(|| throw_r_error("a term basis must be a list"));
    let kind = list_named_elt(&list, "kind")
        .as_str()
        .unwrap_or_else(|| throw_r_error("a term basis needs a character `kind`"))
        .to_string();
    match kind.as_str() {
        "bbs" => {
            let b = real_vec(&list_named_elt(&list, "boundary"), "boundary");
            if b.len() != 2 {
                throw_r_error("bbs boundary must have two values");
            }
            TermBasis::Bbs {
                boundary: (b[0], b[1]),
                n_knots: usize_field(&list, "n_knots"),
                degree: usize_field(&list, "degree"),
                differences: usize_field(&list, "differences"),
                cyclic: false,
            }
        }
        "bols" => TermBasis::Bols {
            intercept: list_named_elt(&list, "intercept").as_bool().unwrap_or(true),
            ridge: list_optional_elt(&list, "ridge")
                .and_then(|r| r.as_bool())
                .unwrap_or(false),
        },
        "brandom" => TermBasis::Brandom {
            levels: real_vec(&list_named_elt(&list, "levels"), "levels"),
        },
        "tensor" => TermBasis::Tensor(
            Box::new(parse_basis(&list_named_elt(&list, "left"))),
            Box::new(parse_basis(&list_named_elt(&list, "right"))),
        ),
        other => throw_r_error(format!(
            "gam_refit supports bbs, bols, brandom and tensor bases; got {other}"
        )),
    }
}

fn parse_vars(robj: &Robj) -> Vec<Array1<f64>> {
    let list = robj
        .as_list()
        .unwrap_or_else(|| throw_r_error("term variables must be a list of numeric vectors"));
    list.values()
        .map(|v| Array1::from_vec(real_vec(&v, "a term variable")))
        .collect()
}

fn optional_vec(robj: Option<Robj>) -> Option<Vec<f64>> {
    robj.filter(|r| !r.is_null()).map(|r| real_vec(&r, "by"))
}

/// Refit terms as one penalized GAM with REML smoothing parameters.
///
/// `terms` is a list of `list(name, basis, vars, by)`; `family` one of
/// Gaussian, Binomial, Poisson, NBinomial, GammaReg, CoxPH. `theta` is used
/// by NBinomial, `time` by CoxPH (pass `numeric(0)` otherwise), `offset`
/// may be `numeric(0)`.
#[extendr]
fn r_gam_refit(
    terms: List,
    y: &[f64],
    weights: &[f64],
    offset: &[f64],
    family: &str,
    theta: f64,
    time: &[f64],
) -> List {
    let family = match family {
        "Gaussian" => RefitFamily::Gaussian,
        "Binomial" => RefitFamily::Binomial,
        "Poisson" => RefitFamily::Poisson,
        "GammaReg" => RefitFamily::Gamma,
        "NBinomial" => RefitFamily::NegativeBinomial { theta },
        "CoxPH" => RefitFamily::CoxPH {
            time: Array1::from_vec(time.to_vec()),
        },
        other => throw_r_error(format!(
            "gam_refit supports Gaussian, Binomial, Poisson, NBinomial, GammaReg and CoxPH; \
             got {other}"
        )),
    };
    let mut refit_terms = Vec::with_capacity(terms.len());
    for (i, term) in terms.values().enumerate() {
        let term = term
            .as_list()
            .unwrap_or_else(|| throw_r_error(format!("terms[[{}]] is not a list", i + 1)));
        let name = list_named_elt(&term, "name")
            .as_str()
            .unwrap_or_else(|| throw_r_error("a term needs a character name"))
            .to_string();
        let basis = parse_basis(&list_named_elt(&term, "basis"));
        let vars = parse_vars(&list_named_elt(&term, "vars"));
        let views: Vec<_> = vars.iter().map(|v| v.view()).collect();
        let design = basis
            .design(&views, false)
            .unwrap_or_else(|e| throw_r_error(format!("term {name}: {e}")));
        refit_terms.push(RefitTerm {
            name,
            design,
            penalties: basis.penalties(),
            by: optional_vec(list_optional_elt(&term, "by")).map(Array1::from_vec),
            variables: term_columns(&term),
        });
    }
    let y = Array1::from_vec(y.to_vec());
    let w = Array1::from_vec(weights.to_vec());
    let offset = (!offset.is_empty()).then(|| Array1::from_vec(offset.to_vec()));
    let fit = try_gam_refit(
        refit_terms,
        y.view(),
        Some(w.view()),
        offset.as_ref().map(|o| o.view()),
        &family,
        &RefitControl::default(),
    )
    .unwrap_or_else(|e| throw_r_error(format!("gam_refit: {e}")));

    let p = fit.coef.len();
    let cov = RMatrix::new_matrix(p, p, |i, j| fit.cov[[i, j]]);
    let cov_corrected = RMatrix::new_matrix(p, p, |i, j| fit.cov_corrected[[i, j]]);
    let blocks: Vec<Robj> = fit
        .blocks
        .iter()
        .map(|b| {
            list!(
                name = b.name.clone(),
                start = b.start as i32,
                end = b.end as i32,
                edf = b.edf,
                lambda = b.lambda.clone(),
                rank = b.rank as i32,
                householder = b.householder.to_vec(),
                has_by = b.has_by,
                ref_df = b.test.as_ref().map_or(f64::NAN, |t| t.ref_df),
                statistic = b.test.as_ref().map_or(f64::NAN, |t| t.statistic),
                p_value = b.test.as_ref().map_or(f64::NAN, |t| t.p_value),
                concurvity_worst = b.concurvity.as_ref().map_or(f64::NAN, |c| c.worst),
                concurvity_observed = b.concurvity.as_ref().map_or(f64::NAN, |c| c.observed),
                concurvity_estimate = b.concurvity.as_ref().map_or(f64::NAN, |c| c.estimate)
            )
            .into()
        })
        .collect();
    list!(
        residual_df = fit.residual_df.unwrap_or(f64::NAN),
        blocks = List::from_values(blocks),
        coef = fit.coef.to_vec(),
        cov = cov,
        cov_corrected = cov_corrected,
        scale = fit.scale,
        reml_score = fit.reml_score,
        rho = fit.rho.to_vec(),
        edf_total = fit.edf_total,
        loglik = fit.loglik,
        df = fit.df,
        converged = fit.converged,
        newton_iterations = fit.newton_iterations as i32,
        pirls_iterations = fit.pirls_iterations as i32,
        gradient_norm = fit.gradient_norm,
        method = fit.method,
        theta = fit.theta.unwrap_or(f64::NAN),
        link = fit.link,
        n = fit.n as i32
    )
}

/// A term's effect and standard error at new rows: `basis` and `vars` as in
/// `r_gam_refit`, and the term's stored `householder`, coefficients and
/// covariance block.
#[extendr]
fn r_gam_refit_effect(
    basis: Robj,
    vars: Robj,
    householder: &[f64],
    coef: &[f64],
    cov: RMatrix<f64>,
    has_by: bool,
    by: Robj,
    unseen_zero: bool,
) -> List {
    let basis = parse_basis(&basis);
    let vars = parse_vars(&vars);
    let views: Vec<_> = vars.iter().map(|v| v.view()).collect();
    let raw = basis
        .design(&views, unseen_zero)
        .unwrap_or_else(|e| throw_r_error(format!("{e}")));
    let k = coef.len();
    if cov.nrows() != k || cov.ncols() != k {
        throw_r_error("the covariance block does not match the coefficients");
    }
    let cov = Array2::from_shape_fn((k, k), |(i, j)| cov[[i, j]]);
    let by = optional_vec(Some(by)).map(Array1::from_vec);
    let (estimate, se) = block_effect(
        raw.view(),
        Array1::from_vec(householder.to_vec()).view(),
        Array1::from_vec(coef.to_vec()).view(),
        cov.view(),
        has_by,
        by.as_ref().map(|b| b.view()),
        unseen_zero && matches!(basis, TermBasis::Brandom { .. }),
    )
    .unwrap_or_else(|e| throw_r_error(format!("{e}")));
    list!(estimate = estimate.to_vec(), se = se.to_vec())
}

/// The data columns a term reads: its `var_names` and `by_name` (empty when
/// absent, and the term then takes no side constraint).
fn term_columns(term: &List) -> Vec<String> {
    let names = |key: &str| -> Vec<String> {
        list_optional_elt(term, key)
            .and_then(|robj| {
                robj.as_str_vector()
                    .map(|v| v.iter().map(|s| s.to_string()).collect())
            })
            .unwrap_or_default()
    };
    let mut out = names("var_names");
    for name in names("by_name") {
        if !out.contains(&name) {
            out.push(name);
        }
    }
    out
}

fn parse_terms(robj: &Robj) -> Vec<RefitTerm> {
    let list = robj
        .as_list()
        .unwrap_or_else(|| throw_r_error("a parameter's terms must be a list"));
    list.values()
        .enumerate()
        .map(|(i, term)| {
            let term = term
                .as_list()
                .unwrap_or_else(|| throw_r_error(format!("terms[[{}]] is not a list", i + 1)));
            let name = list_named_elt(&term, "name")
                .as_str()
                .unwrap_or_else(|| throw_r_error("a term needs a character name"))
                .to_string();
            let basis = parse_basis(&list_named_elt(&term, "basis"));
            let vars = parse_vars(&list_named_elt(&term, "vars"));
            let views: Vec<_> = vars.iter().map(|v| v.view()).collect();
            let design = basis
                .design(&views, false)
                .unwrap_or_else(|e| throw_r_error(format!("term {name}: {e}")));
            RefitTerm {
                name,
                design,
                penalties: basis.penalties(),
                by: optional_vec(list_optional_elt(&term, "by")).map(Array1::from_vec),
                variables: term_columns(&term),
            }
        })
        .collect()
}

/// GAMLSS REML refit: `terms` is a list with one list of terms per
/// distribution parameter (as in `r_gam_refit`), `family` the core family
/// name (`"gaussianlss"`, ...), `start_etas` a list of link-scale predictors
/// or an empty list.
#[extendr]
fn r_gamlss_refit(
    terms: List,
    param_names: Vec<String>,
    family: &str,
    y: &[f64],
    weights: &[f64],
    start_etas: List,
) -> List {
    let k = param_names.len();
    let fam = mboostrs_core::lss::make_family(family, k)
        .unwrap_or_else(|e| throw_r_error(format!("gamlss_refit: {e}")));
    if terms.len() != k {
        throw_r_error(format!(
            "expected terms for {k} parameters, got {}",
            terms.len()
        ));
    }
    let per_param: Vec<Vec<RefitTerm>> = terms.values().map(|t| parse_terms(&t)).collect();
    let start: Vec<Array1<f64>> = start_etas
        .values()
        .map(|v| Array1::from_vec(real_vec(&v, "start_etas")))
        .collect();
    let y = Array1::from_vec(y.to_vec());
    let w = Array1::from_vec(weights.to_vec());
    let fit = mboostrs_core::gam::try_gamlss_refit(
        per_param,
        param_names,
        y.view(),
        Some(w.view()),
        fam.as_ref(),
        (start.len() == k).then_some(start.as_slice()),
        &RefitControl::default(),
    )
    .unwrap_or_else(|e| throw_r_error(format!("gamlss_refit: {e}")));
    let p = fit.coef.len();
    let cov = RMatrix::new_matrix(p, p, |i, j| fit.cov[[i, j]]);
    let cov_corrected = RMatrix::new_matrix(p, p, |i, j| fit.cov_corrected[[i, j]]);
    let params: Vec<Robj> = fit
        .params
        .iter()
        .map(|q| {
            let blocks: Vec<Robj> = q
                .blocks
                .iter()
                .map(|b| {
                    list!(
                        name = b.name.clone(),
                        start = b.start as i32,
                        end = b.end as i32,
                        edf = b.edf,
                        lambda = b.lambda.clone(),
                        rank = b.rank as i32,
                        householder = b.householder.to_vec(),
                        has_by = b.has_by
                    )
                    .into()
                })
                .collect();
            list!(
                name = q.name.clone(),
                start = q.start as i32,
                end = q.end as i32,
                blocks = List::from_values(blocks)
            )
            .into()
        })
        .collect();
    list!(
        params = List::from_values(params),
        coef = fit.coef.to_vec(),
        cov = cov,
        cov_corrected = cov_corrected,
        reml_score = fit.reml_score,
        edf_total = fit.edf_total,
        converged = fit.converged,
        iterations = fit.iterations as i32,
        gradient_norm = fit.gradient_norm,
        information_floored = fit.information_floored,
        state = fit.to_json().to_string()
    )
}

/// Posterior draws of every distribution parameter at new rows. `state` is
/// the refit's JSON state (`r_gamlss_refit(...)$state`), `terms[[q]]` a list
/// of `list(basis, vars, by, zero)` for parameter `q`'s terms at the new
/// rows. Returns a list of `n_draws x n_rows` matrices.
#[extendr]
fn r_gamlss_draws(
    state: &str,
    family: &str,
    terms: List,
    n_draws: i32,
    seed: f64,
    corrected: bool,
) -> List {
    let value: serde_json::Value =
        serde_json::from_str(state).unwrap_or_else(|e| throw_r_error(format!("{e}")));
    let fit = mboostrs_core::gam::LssRefitResult::from_json(&value)
        .unwrap_or_else(|e| throw_r_error(format!("{e}")));
    let k = fit.params.len();
    let fam = mboostrs_core::lss::make_family(family, k)
        .unwrap_or_else(|e| throw_r_error(format!("{e}")));
    let per_param: Vec<Vec<mboostrs_core::gam::NewRowsTerm>> = terms
        .values()
        .map(|q| {
            q.as_list()
                .unwrap_or_else(|| throw_r_error("terms must be a list per parameter"))
                .values()
                .map(|term| {
                    let term = term
                        .as_list()
                        .unwrap_or_else(|| throw_r_error("a term must be a list"));
                    let basis = parse_basis(&list_named_elt(&term, "basis"));
                    let zero = list_named_elt(&term, "zero").as_bool().unwrap_or(false);
                    let vars = parse_vars(&list_named_elt(&term, "vars"));
                    let views: Vec<_> = vars.iter().map(|v| v.view()).collect();
                    let raw = basis
                        .design(&views, zero)
                        .unwrap_or_else(|e| throw_r_error(format!("{e}")));
                    mboostrs_core::gam::NewRowsTerm {
                        raw,
                        by: optional_vec(list_optional_elt(&term, "by")).map(Array1::from_vec),
                        zero_rows_have_no_effect: zero
                            && matches!(basis, TermBasis::Brandom { .. }),
                    }
                })
                .collect()
        })
        .collect();
    let draws = fit
        .parameter_draws(
            fam.as_ref(),
            &per_param,
            n_draws.max(1) as usize,
            seed as u64,
            corrected,
        )
        .unwrap_or_else(|e| throw_r_error(format!("{e}")));
    List::from_values(
        draws
            .iter()
            .map(|d| Robj::from(RMatrix::new_matrix(d.nrows(), d.ncols(), |i, j| d[[i, j]]))),
    )
}

/// Parameter `q` (1-based) of an LSS family on its natural scale.
#[extendr]
fn r_lss_response(family: &str, n_params: i32, q: i32, eta: &[f64]) -> Vec<f64> {
    let fam = mboostrs_core::lss::make_family(family, n_params as usize)
        .unwrap_or_else(|e| throw_r_error(format!("{e}")));
    if q < 1 || q > n_params {
        throw_r_error(format!("parameter index {q} out of range"));
    }
    let mut out = vec![0.0; eta.len()];
    fam.response((q - 1) as usize, eta, &mut out);
    out
}

extendr_module! {
    mod gam_refit;
    fn r_gam_refit;
    fn r_gam_refit_effect;
    fn r_gamlss_refit;
    fn r_lss_response;
    fn r_gamlss_draws;
}
