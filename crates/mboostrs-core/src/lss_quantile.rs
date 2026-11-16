//! Quantile functions of the location-scale and survival families.
//!
//! The quantile of a distribution at probability `p` is the value `q` with
//! `P(Y <= q) = p`. For each row, the functions here take the probability and
//! that row's distribution parameters, in the response scale that the
//! families use, and return `q`. The Python `quantile` methods, the prediction
//! intervals and the residual and centile diagnostics all call this function,
//! so one implementation serves them all.
//!
//! The input is a list of equal-length columns. The first column holds the
//! probabilities. The next columns hold the family's parameters in the order
//! its `param_names` lists them, and a family that takes an extra per-row
//! value (the known sampling variance of `GaussianKnownVarLSS`, the trial
//! count of `BinomialLSS`) takes it as the last column. The family is named
//! by its Python class name, such as `"GammaLSS"`.

use crate::special::{self as sp, Special};
use crate::MboostrsError;

/// Quantiles for every row, and how many of them the underlying root search or
/// series did not converge within its iteration limit.
#[derive(Debug, Clone, PartialEq)]
pub struct LssQuantiles {
    /// One quantile per row.
    pub values: Vec<f64>,
    /// Rows whose special-function evaluation reached its iteration limit.
    pub unconverged: usize,
}

/// Parameter columns (after the probability column) that each family takes.
pub(crate) fn n_columns(family: Family) -> usize {
    match family {
        Family::Poisson | Family::Geometric => 1,
        Family::Gaussian
        | Family::Gamma
        | Family::Beta
        | Family::NBinomial
        | Family::ZIPo
        | Family::NegBinomial
        | Family::Binomial
        | Family::InvGauss
        | Family::Logistic
        | Family::Gumbel
        | Family::Pareto2
        | Family::ZAP
        | Family::LogNormal
        | Family::LogLog
        | Family::Weibull => 2,
        Family::KnownVar | Family::StudentT | Family::ZINB | Family::ZAGA | Family::BCCG => 3,
        Family::JSU | Family::BCT | Family::BCPE => 4,
    }
}

// The variants keep the acronyms of the Python family names (ZINB, BCCG, ...),
// so that a name in an error message and a name in the code match.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Family {
    Gaussian,
    KnownVar,
    Gamma,
    StudentT,
    Beta,
    NBinomial,
    ZIPo,
    ZINB,
    Poisson,
    Geometric,
    Binomial,
    NegBinomial,
    ZAGA,
    InvGauss,
    Logistic,
    Gumbel,
    Pareto2,
    ZAP,
    BCCG,
    JSU,
    BCT,
    BCPE,
    LogNormal,
    LogLog,
    Weibull,
}

impl Family {
    pub(crate) fn from_class(name: &str) -> Result<Self, MboostrsError> {
        Ok(match name {
            "GaussianLSS" => Family::Gaussian,
            "GaussianKnownVarLSS" => Family::KnownVar,
            "GammaLSS" => Family::Gamma,
            "StudentTLSS" => Family::StudentT,
            "BetaLSS" => Family::Beta,
            "NBinomialLSS" => Family::NBinomial,
            "ZIPoLSS" => Family::ZIPo,
            "ZINBLSS" => Family::ZINB,
            "PoissonLSS" => Family::Poisson,
            "GeometricLSS" => Family::Geometric,
            "BinomialLSS" => Family::Binomial,
            "NegativeBinomialLSS" => Family::NegBinomial,
            "ZeroAdjustedGammaLSS" => Family::ZAGA,
            "InverseGaussianLSS" => Family::InvGauss,
            "LogisticLSS" => Family::Logistic,
            "GumbelLSS" => Family::Gumbel,
            "Pareto2LSS" => Family::Pareto2,
            "ZeroAdjustedPoissonLSS" => Family::ZAP,
            "BCCGLSS" => Family::BCCG,
            "JSULSS" => Family::JSU,
            "BCTLSS" => Family::BCT,
            "BCPELSS" => Family::BCPE,
            "LogNormalLSS" => Family::LogNormal,
            "LogLogLSS" => Family::LogLog,
            "WeibullLSS" => Family::Weibull,
            other => {
                return Err(MboostrsError::Control(format!(
                    "no quantile function for family '{other}'; the quantile families are \
                     the LSS families with a closed-form or tabulated quantile"
                )))
            }
        })
    }

    /// The message name a refusal carries: the Python class name.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Family::Gaussian => "GaussianLSS",
            Family::KnownVar => "GaussianKnownVarLSS",
            Family::Gamma => "GammaLSS",
            Family::StudentT => "StudentTLSS",
            Family::Beta => "BetaLSS",
            Family::NBinomial => "NBinomialLSS",
            Family::ZIPo => "ZIPoLSS",
            Family::ZINB => "ZINBLSS",
            Family::Poisson => "PoissonLSS",
            Family::Geometric => "GeometricLSS",
            Family::Binomial => "BinomialLSS",
            Family::NegBinomial => "NegativeBinomialLSS",
            Family::ZAGA => "ZeroAdjustedGammaLSS",
            Family::InvGauss => "InverseGaussianLSS",
            Family::Logistic => "LogisticLSS",
            Family::Gumbel => "GumbelLSS",
            Family::Pareto2 => "Pareto2LSS",
            Family::ZAP => "ZeroAdjustedPoissonLSS",
            Family::BCCG => "BCCGLSS",
            Family::JSU => "JSULSS",
            Family::BCT => "BCTLSS",
            Family::BCPE => "BCPELSS",
            Family::LogNormal => "LogNormalLSS",
            Family::LogLog => "LogLogLSS",
            Family::Weibull => "WeibullLSS",
        }
    }
}

/// Quantile of each row of a family.
///
/// `columns[0]` holds the probabilities and `columns[1..]` the parameters and
/// extras described in the module documentation. Every column must have the
/// same length. Validation runs over all rows before any value is computed,
/// and a refusal names the family and the violated requirement, as the Python
/// families did.
pub fn lss_quantile(family: &str, columns: &[Vec<f64>]) -> Result<LssQuantiles, MboostrsError> {
    let family = Family::from_class(family)?;
    let expected = 1 + n_columns(family);
    if columns.len() != expected {
        return Err(MboostrsError::Shape(format!(
            "{} quantile takes {} columns (probabilities and {} parameters), got {}",
            family.name(),
            expected,
            expected - 1,
            columns.len()
        )));
    }
    let n = columns[0].len();
    if columns.iter().any(|c| c.len() != n) {
        return Err(MboostrsError::Shape(format!(
            "{} quantile: every column must have the same length",
            family.name()
        )));
    }
    let p = &columns[0];
    if p.iter().any(|&v| !v.is_finite() || v <= 0.0 || v >= 1.0) {
        return Err(MboostrsError::Control(format!(
            "{} quantile probabilities must be finite and in (0, 1)",
            family.name()
        )));
    }
    validate(family, &columns[1..])?;

    let mut values = Vec::with_capacity(n);
    let mut unconverged = 0usize;
    for i in 0..n {
        let row: Vec<f64> = columns.iter().map(|c| c[i]).collect();
        let special = evaluate(family, &row);
        if !special.converged {
            unconverged += 1;
        }
        values.push(special.value);
    }
    Ok(LssQuantiles {
        values,
        unconverged,
    })
}

fn any(column: &[f64], test: impl Fn(f64) -> bool) -> bool {
    column.iter().any(|&v| test(v))
}

/// The refusals of each family, in the order the Python quantile methods
/// checked them. A NaN parameter passes a `<= 0` test, as it did in NumPy.
pub(crate) fn validate(family: Family, params: &[Vec<f64>]) -> Result<(), MboostrsError> {
    let refuse = |message: String| Err(MboostrsError::Control(message));
    let name = family.name();
    match family {
        Family::Gaussian => {
            if any(&params[1], |s| s <= 0.0) {
                return refuse(format!("{name} sigma must be positive"));
            }
        }
        Family::KnownVar => {}
        Family::Gamma
        | Family::ZAGA
        | Family::NegBinomial
        | Family::InvGauss
        | Family::Pareto2
        | Family::BCCG => {
            if any(&params[0], |m| m <= 0.0) || any(&params[1], |s| s <= 0.0) {
                let message = match family {
                    Family::Gamma => "GammaLSS requires mu and sigma to be positive",
                    Family::ZAGA => "ZeroAdjustedGammaLSS requires mu and sigma positive",
                    Family::NegBinomial => {
                        "NegativeBinomialLSS requires mu and sigma to be positive"
                    }
                    Family::InvGauss => "InverseGaussianLSS requires mu and sigma positive",
                    Family::Pareto2 => "Pareto2LSS requires mu and sigma positive",
                    _ => "BCCGLSS requires mu and sigma positive",
                };
                return refuse(message.to_string());
            }
        }
        Family::StudentT => {
            if any(&params[1], |s| s <= 0.0) || any(&params[2], |d| d <= 0.0) {
                return refuse("StudentTLSS requires sigma > 0 and df > 0".to_string());
            }
        }
        Family::Beta => {
            if any(&params[0], |m| m <= 0.0 || m >= 1.0) || any(&params[1], |f| f <= 0.0) {
                return refuse("BetaLSS requires 0 < mu < 1 and phi > 0".to_string());
            }
        }
        Family::NBinomial | Family::Poisson | Family::Geometric | Family::ZAP => {
            if any(&params[0], |m| m <= 0.0) {
                let message = match family {
                    Family::Poisson => "PoissonLSS requires mu to be positive",
                    Family::Geometric => "GeometricLSS requires mu to be positive",
                    Family::ZAP => "ZeroAdjustedPoissonLSS mu must be positive",
                    _ => "NBinomialLSS requires mu and sigma to be positive",
                };
                return refuse(message.to_string());
            }
            if family == Family::NBinomial && any(&params[1], |s| s <= 0.0) {
                return refuse("NBinomialLSS requires mu and sigma to be positive".to_string());
            }
        }
        Family::ZIPo => {
            if any(&params[0], |m| m <= 0.0) || any(&params[1], |s| !(0.0..=1.0).contains(&s)) {
                return refuse("ZIPoLSS requires mu > 0 and 0 <= sigma <= 1".to_string());
            }
        }
        Family::ZINB => {
            if any(&params[0], |m| m <= 0.0)
                || any(&params[1], |s| s <= 0.0)
                || any(&params[2], |nu| !(0.0..=1.0).contains(&nu))
            {
                return refuse("ZINBLSS requires mu,sigma > 0 and 0 <= nu <= 1".to_string());
            }
        }
        Family::Binomial => {
            if any(&params[0], |m| m <= 0.0 || m >= 1.0) {
                return refuse("BinomialLSS requires 0 < mu < 1".to_string());
            }
        }
        Family::Logistic | Family::Gumbel => {
            if any(&params[1], |s| s <= 0.0) {
                let message = if family == Family::Logistic {
                    "LogisticLSS sigma must be positive"
                } else {
                    "GumbelLSS sigma must be positive"
                };
                return refuse(message.to_string());
            }
        }
        Family::JSU => {
            if any(&params[1], |s| s <= 0.0) || any(&params[3], |t| t <= 0.0) {
                return refuse("JSULSS requires sigma and tau positive".to_string());
            }
        }
        Family::BCT | Family::BCPE => {
            let finite = params.iter().all(|c| c.iter().all(|v| v.is_finite()));
            if !finite
                || any(&params[0], |m| m <= 0.0)
                || any(&params[1], |s| s <= 0.0)
                || any(&params[3], |t| t <= 0.0)
            {
                let label = if family == Family::BCT {
                    "BCTLSS"
                } else {
                    "BCPELSS"
                };
                return refuse(format!(
                    "{label} requires finite mu > 0, sigma > 0, nu, and tau > 0"
                ));
            }
        }
        Family::LogNormal | Family::LogLog | Family::Weibull => {
            if any(&params[1], |s| s <= 0.0) {
                return refuse(format!("{name} sigma must be positive"));
            }
        }
    }
    Ok(())
}

/// Value of the quantile for one row: `row[0]` is the probability and
/// `row[1..]` the parameters, in the order `n_columns` describes.
fn evaluate(family: Family, row: &[f64]) -> Special {
    let p = row[0];
    let exact = |value: f64| Special {
        value,
        converged: true,
    };
    match family {
        Family::Gaussian => exact(row[1] + row[2] * sp::normal_quantile(p)),
        Family::KnownVar => {
            let (mu, sigma, s2) = (row[1], row[2], row[3]);
            let sd = (s2 + sigma * sigma).sqrt();
            exact(mu + sd * sp::normal_quantile(p))
        }
        Family::Gamma => {
            let (mu, sigma) = (row[1], row[2]);
            let q = sp::gamma_quantile(p, sigma);
            Special {
                value: q.value * mu / sigma,
                converged: q.converged,
            }
        }
        Family::StudentT => {
            let (mu, sigma, df) = (row[1], row[2], row[3]);
            let q = sp::student_t_quantile(p, df);
            Special {
                value: mu + sigma * q.value,
                converged: q.converged,
            }
        }
        Family::Beta => {
            let (mu, phi) = (row[1], row[2]);
            sp::beta_quantile(p, mu * phi, (1.0 - mu) * phi)
        }
        Family::NBinomial => {
            let (mu, sigma) = (row[1], row[2]);
            sp::nbinom_quantile_affine(p, mu, sigma, 0.0, 1.0)
        }
        Family::ZIPo => {
            let (mu, sigma) = (row[1], row[2]);
            sp::poisson_quantile_affine(p, mu, sigma, 1.0 - sigma)
        }
        Family::ZINB => {
            let (mu, sigma, nu) = (row[1], row[2], row[3]);
            sp::nbinom_quantile_affine(p, mu, 1.0 / sigma, nu, 1.0 - nu)
        }
        Family::Poisson => sp::poisson_quantile_affine(p, row[1], 0.0, 1.0),
        Family::Geometric => {
            let mu = row[1];
            let prob = 1.0 / (1.0 + mu);
            let k = (((1.0 - p).ln() / (1.0 - prob).ln()) - 1e-12).ceil() - 1.0;
            exact(if k < 0.0 { 0.0 } else { k })
        }
        Family::Binomial => {
            let (mu, size) = (row[1], row[2]);
            sp::binom_quantile(p, size, mu)
        }
        Family::NegBinomial => {
            let (mu, sigma) = (row[1], row[2]);
            sp::nbinom_quantile_affine(p, mu, 1.0 / sigma, 0.0, 1.0)
        }
        Family::ZAGA => {
            let (mu, sigma, nu) = (row[1], row[2], row[3]);
            if p > nu {
                // The atom takes nu of the mass; the rest is a gamma with shape
                // 1 / sigma^2 and scale mu * sigma^2.
                let target = (p - nu) / (1.0 - nu);
                let q = sp::gamma_quantile(target, 1.0 / (sigma * sigma));
                Special {
                    value: q.value * mu * (sigma * sigma),
                    converged: q.converged,
                }
            } else {
                exact(0.0)
            }
        }
        Family::InvGauss => sp::inverse_gaussian_quantile(p, row[1], row[2]),
        Family::Logistic => exact(row[1] + row[2] * (p / (1.0 - p)).ln()),
        Family::Gumbel => exact(row[1] - row[2] * (-p.ln()).ln()),
        Family::Pareto2 => {
            let (mu, sigma) = (row[1], row[2]);
            exact(mu * ((1.0 - p).powf(-1.0 / sigma) - 1.0))
        }
        Family::ZAP => {
            let (mu, sigma) = (row[1], row[2]);
            let zero = (-mu).exp();
            let slope = (1.0 - sigma) / (1.0 - zero);
            sp::poisson_quantile_affine(p, mu, sigma - slope * zero, slope)
        }
        Family::BCCG => {
            let (mu, sigma, nu) = (row[1], row[2], row[3]);
            exact(bccg_quantile(p, mu, sigma, nu))
        }
        Family::JSU => {
            let (mu, sigma, nu, tau) = (row[1], row[2], row[3], row[4]);
            let rtau = 1.0 / tau;
            let w = if rtau < 1e-7 {
                1.0
            } else {
                (rtau * rtau).exp()
            };
            let omega = -nu * rtau;
            let c = (0.5 * (w - 1.0) * (w * (2.0 * omega).cosh() + 1.0)).powf(-0.5);
            let r = sp::normal_quantile(p);
            let z = (rtau * (r + nu)).sinh();
            exact(mu + c * sigma * (w.sqrt() * omega.sinh() + z))
        }
        Family::BCT => {
            let (mu, sigma, nu, tau) = (row[1], row[2], row[3], row[4]);
            let edge = boxcox_edge(sigma, nu);
            let f3 = sp::student_t_cdf(edge, tau);
            let target = if nu <= 0.0 {
                p * f3.value
            } else {
                1.0 - (1.0 - p) * f3.value
            };
            let z = sp::student_t_quantile(target, tau);
            Special {
                value: boxcox_inverse(z.value, mu, sigma, nu),
                converged: z.converged && f3.converged,
            }
        }
        Family::BCPE => {
            let (mu, sigma, nu, tau) = (row[1], row[2], row[3], row[4]);
            let edge = boxcox_edge(sigma, nu);
            let f3 = if edge.is_finite() {
                bcpe_cdf_t(edge, tau)
            } else {
                1.0
            };
            let near_zero = nu.abs() < 1e-10;
            let target = if nu < 0.0 {
                p * f3
            } else if near_zero {
                p
            } else {
                1.0 - (1.0 - p) * f3
            };
            exact(boxcox_inverse(bcpe_ppf_t(target, tau), mu, sigma, nu))
        }
        Family::LogNormal => exact((row[1] + row[2] * sp::normal_quantile(p)).exp()),
        Family::LogLog => exact((row[1] + row[2] * (p / (1.0 - p)).ln()).exp()),
        Family::Weibull => exact((row[1] + row[2] * (-(-p).ln_1p()).ln()).exp()),
    }
}

/// Box-Cox-Cole-Green quantile, `gamlss.dist::qBCCG`.
fn bccg_quantile(p: f64, mu: f64, sigma: f64, nu: f64) -> f64 {
    let small = nu.abs() < 1e-10;
    let safe_nu = if small { 1.0 } else { nu };
    let edge = 1.0 / (sigma * if small { 1.0 } else { nu.abs() });
    let lower = if small || nu <= 0.0 {
        0.0
    } else {
        sp::normal_cdf(-edge)
    };
    let scale = if small { 1.0 } else { sp::normal_cdf(edge) };
    let z = sp::normal_quantile((lower + p * scale).clamp(1e-15, 1.0 - 1e-15));
    if small {
        mu * (sigma * z).exp()
    } else {
        let base = nu * sigma * z + 1.0;
        let base = if base < 0.0 { 0.0 } else { base };
        mu * base.powf(1.0 / safe_nu)
    }
}

/// Positive-response truncation edge, `1 / (sigma |nu|)`, infinite at nu = 0.
pub(crate) fn boxcox_edge(sigma: f64, nu: f64) -> f64 {
    if nu.abs() < 1e-10 {
        f64::INFINITY
    } else {
        1.0 / (sigma * nu.abs())
    }
}

/// The response from a Box-Cox standardized value `z`.
fn boxcox_inverse(z: f64, mu: f64, sigma: f64, nu: f64) -> f64 {
    if nu.abs() < 1e-10 {
        return mu * (sigma * z).exp();
    }
    let base = nu * sigma * z + 1.0;
    let base = if base < 0.0 { 0.0 } else { base };
    mu * base.powf(1.0 / nu)
}

/// `log c` of the standardized power-exponential kernel of BCPE.
pub(crate) fn bcpe_log_c(tau: f64) -> f64 {
    0.5 * (-(2.0 / tau) * std::f64::consts::LN_2 + libm::lgamma(1.0 / tau)
        - libm::lgamma(3.0 / tau))
}

/// CDF of the standardized power-exponential kernel used by BCPE.
pub(crate) fn bcpe_cdf_t(t: f64, tau: f64) -> f64 {
    let c = bcpe_log_c(tau).exp();
    let s = 0.5 * (t / c).abs().powf(tau);
    let sign = if t > 0.0 {
        1.0
    } else if t < 0.0 {
        -1.0
    } else {
        0.0
    };
    0.5 * (1.0 + sign * sp::gamma_p(1.0 / tau, s).value)
}

/// Quantile of the power-exponential kernel, in closed form through the gamma
/// quantile of shape `1 / tau`.
fn bcpe_ppf_t(p: f64, tau: f64) -> f64 {
    if p == 0.5 {
        return 0.0;
    }
    let c = bcpe_log_c(tau).exp();
    let q = sp::gamma_quantile((2.0 * p - 1.0).abs(), 1.0 / tau).value;
    let sign = if p - 0.5 > 0.0 { 1.0 } else { -1.0 };
    sign * c * (2.0 * q).powf(1.0 / tau)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cols(values: &[&[f64]]) -> Vec<Vec<f64>> {
        values.iter().map(|c| c.to_vec()).collect()
    }

    #[test]
    fn gaussian_quantile_matches_the_closed_form() {
        // Median is mu, and the 97.5% point of N(1, 2^2) is 1 + 2 * 1.959963984540054.
        let out = lss_quantile(
            "GaussianLSS",
            &cols(&[&[0.5, 0.975], &[1.0, 1.0], &[2.0, 2.0]]),
        )
        .unwrap();
        assert!((out.values[0] - 1.0).abs() < 1e-14);
        assert!((out.values[1] - (1.0 + 2.0 * 1.959963984540054)).abs() < 1e-12);
        assert_eq!(out.unconverged, 0);
    }

    #[test]
    fn unknown_family_is_refused_with_the_name() {
        let err = lss_quantile("NoSuchLSS", &cols(&[&[0.5], &[0.0]])).unwrap_err();
        assert!(err.to_string().contains("NoSuchLSS"));
    }

    #[test]
    fn invalid_probability_is_refused_before_parameters_are_read() {
        let err = lss_quantile("GaussianLSS", &cols(&[&[0.0], &[0.0], &[1.0]])).unwrap_err();
        assert!(err
            .to_string()
            .contains("GaussianLSS quantile probabilities must be finite and in (0, 1)"));
    }

    #[test]
    fn sigma_refusal_names_the_family() {
        let err = lss_quantile("GaussianLSS", &cols(&[&[0.5], &[0.0], &[0.0]])).unwrap_err();
        assert!(err
            .to_string()
            .contains("GaussianLSS sigma must be positive"));
    }

    #[test]
    fn wrong_column_count_is_a_shape_error() {
        let err = lss_quantile("GaussianLSS", &cols(&[&[0.5], &[0.0]])).unwrap_err();
        assert!(matches!(err, MboostrsError::Shape(_)));
    }

    #[test]
    fn beta_quantile_inverts_its_own_cdf() {
        let out = lss_quantile("BetaLSS", &cols(&[&[0.3], &[0.4], &[5.0]])).unwrap();
        let back = sp::beta_i(0.4 * 5.0, 0.6 * 5.0, out.values[0]).value;
        assert!((back - 0.3).abs() < 1e-9);
    }

    #[test]
    fn bcpe_median_is_mu_when_nu_is_zero() {
        // With nu = 0 the response is mu * exp(sigma * z), so the median is mu.
        let out =
            lss_quantile("BCPELSS", &cols(&[&[0.5], &[2.0], &[0.3], &[0.0], &[2.0]])).unwrap();
        assert!((out.values[0] - 2.0).abs() < 1e-12);
    }
}
