//! Densities, CDFs, means and deviance contributions of the location-scale
//! families.
//!
//! For each row, the functions here take a response value and that row's
//! distribution parameters on the response scale, and return the density (a
//! probability mass for a count family), the cumulative distribution function
//! `P(Y <= y)` or the deviance contribution `-2 log f(y)`. The R distribution
//! helpers (`lss_pdf`, `lss_cdf`, `lss_deviance_increment`) call them, so the
//! R package evaluates the same formulas as the core. [`lss_mean`] gives the
//! conditional response mean from the parameters alone.
//!
//! The input follows [`crate::lss_quantile`]: a list of equal-length columns,
//! the first holding the response values and the next the family's parameters
//! in the order its `param_names` lists them. `BinomialLSS` takes the number
//! of trials as a last column. The family is named by its class name, such as
//! `"GammaLSS"`.
//!
//! A count family's mass is zero at a value that is not a non-negative
//! integer, as R's `dpois` and `dnbinom` give, and its CDF at a non-integer
//! `y` is the CDF at `floor(y)`. A continuous family's density is zero
//! outside its support.

use crate::lss::student_t_pdf;
use crate::lss_quantile::{bcpe_cdf_t, bcpe_log_c, boxcox_edge, n_columns, validate, Family};
use crate::special::{self as sp};
use crate::MboostrsError;
use libm::lgamma;
use std::f64::consts::PI;

/// Values for every row, and how many of them came from a special-function
/// evaluation that reached its iteration limit before its tolerance.
#[derive(Debug, Clone, PartialEq)]
pub struct LssValues {
    /// One value per row.
    pub values: Vec<f64>,
    /// Rows whose special-function evaluation reached its iteration limit.
    pub unconverged: usize,
}

/// `ln(sqrt(2 pi))`.
const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_8;

/// The families with a density here, in the order R's `lss_pdf` lists them.
fn has_density(family: Family) -> bool {
    matches!(
        family,
        Family::Gaussian
            | Family::Gamma
            | Family::Beta
            | Family::NBinomial
            | Family::NegBinomial
            | Family::Poisson
            | Family::Geometric
            | Family::Binomial
            | Family::ZIPo
            | Family::ZINB
            | Family::StudentT
            | Family::ZAGA
            | Family::BCT
            | Family::BCPE
    )
}

/// Parse the family and check the column layout shared by the density and
/// the CDF: the response column, then the parameter and extra columns.
fn checked_family(
    family: &str,
    columns: &[Vec<f64>],
    what: &str,
    supported: fn(Family) -> bool,
) -> Result<Family, MboostrsError> {
    let parsed = Family::from_class(family)
        .map_err(|_| MboostrsError::Control(format!("no {what} for family '{family}'")))?;
    if !supported(parsed) {
        return Err(MboostrsError::Control(format!(
            "no {what} for family '{}'",
            parsed.name()
        )));
    }
    let expected = 1 + n_columns(parsed);
    if columns.len() != expected {
        return Err(MboostrsError::Shape(format!(
            "{} {what} takes {} columns (responses and {} parameters), got {}",
            parsed.name(),
            expected,
            expected - 1,
            columns.len()
        )));
    }
    let n = columns[0].len();
    if columns.iter().any(|c| c.len() != n) {
        return Err(MboostrsError::Shape(format!(
            "{} {what}: every column must have the same length",
            parsed.name()
        )));
    }
    Ok(parsed)
}

/// The parameter checks of the density and CDF. They are those of the
/// quantile, except that `BinomialLSS` accepts a probability of exactly 0 or
/// 1 (a saturated logit gives one) and `ZeroAdjustedGammaLSS` also checks its
/// zero probability `nu`.
fn validate_density(family: Family, params: &[Vec<f64>]) -> Result<(), MboostrsError> {
    match family {
        Family::Binomial => {
            if params[0]
                .iter()
                .any(|&m| !(0.0..=1.0).contains(&m) && !m.is_nan())
            {
                return Err(MboostrsError::Control(
                    "BinomialLSS requires 0 <= mu <= 1".to_string(),
                ));
            }
            Ok(())
        }
        Family::ZAGA => {
            validate(family, params)?;
            if params[2]
                .iter()
                .any(|&nu| !(0.0..=1.0).contains(&nu) && !nu.is_nan())
            {
                return Err(MboostrsError::Control(
                    "ZeroAdjustedGammaLSS nu must be in [0, 1]".to_string(),
                ));
            }
            Ok(())
        }
        _ => validate(family, params),
    }
}

/// Density or mass of each row of a family.
///
/// `columns[0]` holds the response values and `columns[1..]` the parameters
/// described in the module documentation. A refusal names the family and the
/// violated requirement.
///
/// ```
/// use mboostrs_core::lss_distribution::lss_pdf;
///
/// // R: dnorm(1, mean = 0, sd = 2) = 0.17603266338214976
/// let out = lss_pdf("GaussianLSS", &[vec![1.0], vec![0.0], vec![2.0]]).unwrap();
/// assert!((out.values[0] - 0.17603266338214976).abs() < 1e-16);
/// ```
pub fn lss_pdf(family: &str, columns: &[Vec<f64>]) -> Result<LssValues, MboostrsError> {
    let family = checked_family(family, columns, "density", has_density)?;
    validate_density(family, &columns[1..])?;
    let n = columns[0].len();
    if family == Family::StudentT {
        let values = student_t_pdf(&columns[0], &columns[1], &columns[2], &columns[3])?;
        return Ok(LssValues {
            values,
            unconverged: 0,
        });
    }
    let mut values = Vec::with_capacity(n);
    let mut unconverged = 0usize;
    for i in 0..n {
        let row: Vec<f64> = columns.iter().map(|c| c[i]).collect();
        let (value, converged) = density(family, &row);
        if !converged {
            unconverged += 1;
        }
        values.push(value);
    }
    Ok(LssValues {
        values,
        unconverged,
    })
}

/// Deviance contribution `-2 log f(y)` of each row, with the density floored
/// at the smallest positive normal double so that a zero density gives a large
/// finite value. The columns are those of [`lss_pdf`].
pub fn lss_deviance_increment(
    family: &str,
    columns: &[Vec<f64>],
) -> Result<LssValues, MboostrsError> {
    let mut out = lss_pdf(family, columns)?;
    for v in out.values.iter_mut() {
        *v = -2.0 * v.max(f64::MIN_POSITIVE).ln();
    }
    Ok(out)
}

/// The count value at `y`, when `y` is a non-negative integer up to R's
/// tolerance (`|y - round(y)| <= 1e-7 max(1, |y|)`).
fn count_value(y: f64) -> Option<f64> {
    let r = y.round();
    if y >= 0.0 && (y - r).abs() <= 1e-7 * y.abs().max(1.0) {
        Some(r)
    } else {
        None
    }
}

/// Density of one row: `row[0]` is the response and `row[1..]` the parameters.
/// Returns the value and whether every special function in it converged.
fn density(family: Family, row: &[f64]) -> (f64, bool) {
    let y = row[0];
    let exact = |v: f64| (v, true);
    match family {
        Family::Gaussian => {
            let (mu, sigma) = (row[1], row[2]);
            let z = (y - mu) / sigma;
            exact((-0.5 * z * z - LN_SQRT_2PI).exp() / sigma)
        }
        Family::Gamma => {
            let (mu, sigma) = (row[1], row[2]);
            exact(dgamma(y, sigma, mu / sigma))
        }
        Family::Beta => {
            let (mu, phi) = (row[1], row[2]);
            exact(dbeta(y, mu * phi, (1.0 - mu) * phi))
        }
        Family::NBinomial => exact(count_value(y).map_or(0.0, |k| dnbinom_mu(k, row[2], row[1]))),
        Family::NegBinomial => {
            exact(count_value(y).map_or(0.0, |k| dnbinom_mu(k, 1.0 / row[2], row[1])))
        }
        Family::Poisson => exact(count_value(y).map_or(0.0, |k| dpois_raw(k, row[1]))),
        Family::Geometric => {
            let prob = 1.0 / (1.0 + row[1]);
            exact(count_value(y).map_or(0.0, |k| prob * (1.0 - prob).powf(k)))
        }
        Family::Binomial => {
            let (mu, size) = (row[1], row[2]);
            exact(count_value(y).map_or(0.0, |k| dbinom_raw(k, size, mu, 1.0 - mu)))
        }
        Family::ZIPo => {
            let (mu, sigma) = (row[1], row[2]);
            exact(match count_value(y) {
                Some(0.0) => sigma + (1.0 - sigma) * (-mu).exp(),
                Some(k) => (1.0 - sigma) * dpois_raw(k, mu),
                None => 0.0,
            })
        }
        Family::ZINB => {
            let (mu, sigma, nu) = (row[1], row[2], row[3]);
            exact(match count_value(y) {
                Some(k) => {
                    // ZINBI: sigma is the dispersion, so the size is 1 / sigma.
                    let mass = dnbinom_mu(k, 1.0 / sigma, mu);
                    if k == 0.0 {
                        nu + (1.0 - nu) * mass
                    } else {
                        (1.0 - nu) * mass
                    }
                }
                None => 0.0,
            })
        }
        Family::ZAGA => {
            let (mu, sigma, nu) = (row[1], row[2], row[3]);
            exact(if y < 0.0 {
                0.0
            } else if y == 0.0 {
                nu
            } else {
                let s2 = sigma * sigma;
                (1.0 - nu) * dgamma(y, 1.0 / s2, mu * s2)
            })
        }
        Family::BCT => {
            if y <= 0.0 {
                return exact(0.0);
            }
            let (mu, sigma, nu, tau) = (row[1], row[2], row[3], row[4]);
            let z = boxcox_z(y, mu, sigma, nu);
            let edge = boxcox_edge(sigma, nu);
            let (f3, converged) = if edge.is_finite() {
                let s = sp::student_t_cdf(edge, tau);
                (s.value, s.converged)
            } else {
                (1.0, true)
            };
            let log_det = (nu - 1.0) * y.ln() - nu * mu.ln() - sigma.ln();
            (
                (log_det + student_t_log_pdf(z, tau) - f3.ln()).exp(),
                converged,
            )
        }
        Family::BCPE => {
            if y <= 0.0 {
                return exact(0.0);
            }
            let (mu, sigma, nu, tau) = (row[1], row[2], row[3], row[4]);
            let z = boxcox_z(y, mu, sigma, nu);
            let log_c = bcpe_log_c(tau);
            let c = log_c.exp();
            let log_fz = tau.ln()
                - log_c
                - 0.5 * (z / c).abs().powf(tau)
                - (1.0 + 1.0 / tau) * std::f64::consts::LN_2
                - lgamma(1.0 / tau);
            let log_det = (nu - 1.0) * y.ln() - nu * mu.ln() - sigma.ln();
            let edge = boxcox_edge(sigma, nu);
            let f3 = if edge.is_finite() {
                bcpe_cdf_t(edge, tau)
            } else {
                1.0
            };
            exact((log_det + log_fz - f3.ln()).exp())
        }
        // `has_density` admits no other family.
        _ => exact(f64::NAN),
    }
}

/// The families with a CDF here, in the order R's `lss_cdf` lists them.
fn has_cdf(family: Family) -> bool {
    matches!(
        family,
        Family::Gaussian
            | Family::NBinomial
            | Family::NegBinomial
            | Family::Poisson
            | Family::Geometric
            | Family::Binomial
            | Family::ZIPo
            | Family::ZINB
            | Family::ZAGA
            | Family::BCT
            | Family::BCPE
    )
}

/// Cumulative distribution function `P(Y <= y)` of each row of a family. The
/// columns are those of [`lss_pdf`]. A count family's CDF at a non-integer
/// `y` is its CDF at `floor(y)`, and every CDF is 0 below the support.
///
/// ```
/// use mboostrs_core::lss_distribution::lss_cdf;
///
/// // R: ppois(3, 2.5) = 0.7575761331330658
/// let out = lss_cdf("PoissonLSS", &[vec![3.0], vec![2.5]]).unwrap();
/// assert!((out.values[0] - 0.7575761331330658).abs() < 1e-14);
/// ```
pub fn lss_cdf(family: &str, columns: &[Vec<f64>]) -> Result<LssValues, MboostrsError> {
    let family = checked_family(family, columns, "CDF", has_cdf)?;
    validate_density(family, &columns[1..])?;
    let n = columns[0].len();
    let mut values = Vec::with_capacity(n);
    let mut unconverged = 0usize;
    for i in 0..n {
        let row: Vec<f64> = columns.iter().map(|c| c[i]).collect();
        let (value, converged) = cdf(family, &row);
        if !converged {
            unconverged += 1;
        }
        values.push(value);
    }
    Ok(LssValues {
        values,
        unconverged,
    })
}

/// CDF of one row: `row[0]` is the response and `row[1..]` the parameters.
fn cdf(family: Family, row: &[f64]) -> (f64, bool) {
    let y = row[0];
    let special = |s: sp::Special| (s.value, s.converged);
    match family {
        Family::Gaussian => (sp::normal_cdf((y - row[1]) / row[2]), true),
        Family::NBinomial => special(sp::nbinom_cdf(y, row[1], row[2])),
        Family::NegBinomial => special(sp::nbinom_cdf(y, row[1], 1.0 / row[2])),
        Family::Poisson => special(sp::poisson_cdf(y, row[1])),
        Family::Geometric => {
            if y < 0.0 {
                return (0.0, true);
            }
            let prob = 1.0 / (1.0 + row[1]);
            (1.0 - (1.0 - prob).powf(y.floor() + 1.0), true)
        }
        Family::Binomial => special(sp::binom_cdf(y, row[2], row[1])),
        Family::ZIPo => {
            if y < 0.0 {
                return (0.0, true);
            }
            let (mu, sigma) = (row[1], row[2]);
            let f = sp::poisson_cdf(y.floor(), mu);
            (sigma + (1.0 - sigma) * f.value, f.converged)
        }
        Family::ZINB => {
            if y < 0.0 {
                return (0.0, true);
            }
            let (mu, sigma, nu) = (row[1], row[2], row[3]);
            let f = sp::nbinom_cdf(y.floor(), mu, 1.0 / sigma);
            (nu + (1.0 - nu) * f.value, f.converged)
        }
        Family::ZAGA => {
            let (mu, sigma, nu) = (row[1], row[2], row[3]);
            if y < 0.0 {
                return (0.0, true);
            }
            if y == 0.0 {
                return (nu, true);
            }
            let s2 = sigma * sigma;
            let f = sp::gamma_p(1.0 / s2, y / (mu * s2));
            (nu + (1.0 - nu) * f.value, f.converged)
        }
        Family::BCT => {
            if y <= 0.0 {
                return (0.0, true);
            }
            let (mu, sigma, nu, tau) = (row[1], row[2], row[3], row[4]);
            let z = boxcox_z(y, mu, sigma, nu);
            let edge = boxcox_edge(sigma, nu);
            let f1 = sp::student_t_cdf(z, tau);
            let (f2, f3, converged) = if edge.is_finite() {
                let upper = sp::student_t_cdf(edge, tau);
                let lower = if nu > 0.0 {
                    sp::student_t_cdf(-edge, tau)
                } else {
                    sp::Special {
                        value: 0.0,
                        converged: true,
                    }
                };
                (lower.value, upper.value, lower.converged && upper.converged)
            } else {
                (0.0, 1.0, true)
            };
            ((f1.value - f2) / f3, f1.converged && converged)
        }
        Family::BCPE => {
            if y <= 0.0 {
                return (0.0, true);
            }
            let (mu, sigma, nu, tau) = (row[1], row[2], row[3], row[4]);
            let z = boxcox_z(y, mu, sigma, nu);
            let edge = boxcox_edge(sigma, nu);
            let f1 = bcpe_cdf_t(z, tau);
            let (f2, f3) = if edge.is_finite() {
                let lower = if nu > 0.0 {
                    bcpe_cdf_t(-edge, tau)
                } else {
                    0.0
                };
                (lower, bcpe_cdf_t(edge, tau))
            } else {
                (0.0, 1.0)
            };
            ((f1 - f2) / f3, true)
        }
        // `has_cdf` admits no other family.
        _ => (f64::NAN, true),
    }
}

/// Conditional response mean `E(Y)` of each row of a family.
///
/// `columns` holds the family's parameters in `param_names` order, without a
/// response column and without the extra column the density takes. For most
/// families the mean is the parameter `mu`; for the zero-inflated and
/// zero-adjusted families it is the mixture mean, for the accelerated
/// failure-time families (`mu` and `sigma` on the log-time scale) the mean
/// time, and for `GumbelLSS` `mu + gamma sigma` with Euler's constant
/// `gamma`. Families whose mean does not exist for the supplied parameters
/// (`StudentTLSS` at `df <= 1`, `Pareto2LSS` at `sigma <= 1`) are refused, as
/// are those whose `mu` is a median or location with no closed-form mean
/// (`BCCGLSS`, `BCTLSS`, `BCPELSS`) and `DirichletLSS`, which has one mean
/// per component.
///
/// ```
/// use mboostrs_core::lss_distribution::lss_mean;
///
/// // ZIPoLSS: (1 - sigma) * mu
/// let out = lss_mean("ZIPoLSS", &[vec![2.0], vec![0.25]]).unwrap();
/// assert_eq!(out, vec![1.5]);
/// ```
pub fn lss_mean(family: &str, columns: &[Vec<f64>]) -> Result<Vec<f64>, MboostrsError> {
    let refuse = |message: &str| Err(MboostrsError::Control(message.to_string()));
    let (name, expected) = match family {
        "PIGLSS" => ("PIGLSS", 2),
        "DirichletLSS" => {
            return refuse(
                "DirichletLSS has one compositional mean per component; scalar lss_mean is \
                 not applicable",
            )
        }
        other => {
            let parsed = Family::from_class(other).map_err(|_| {
                MboostrsError::Control(format!("{other} response mean helper is not implemented"))
            })?;
            let extras = usize::from(matches!(parsed, Family::KnownVar | Family::Binomial));
            (parsed.name(), n_columns(parsed) - extras)
        }
    };
    if columns.len() != expected {
        return Err(MboostrsError::Shape(format!(
            "{name} mean takes {expected} parameter columns, got {}",
            columns.len()
        )));
    }
    let n = columns[0].len();
    if columns.iter().any(|c| c.len() != n) {
        return Err(MboostrsError::Shape(format!(
            "{name} mean: every column must have the same length"
        )));
    }
    let mu = &columns[0];
    let any = |c: &[f64], test: fn(f64) -> bool| c.iter().any(|&v| test(v));
    let map2 = |f: &dyn Fn(f64, f64) -> f64| -> Vec<f64> {
        mu.iter().zip(&columns[1]).map(|(&m, &s)| f(m, s)).collect()
    };
    if name == "PIGLSS" {
        return Ok(mu.clone());
    }
    let family = Family::from_class(name)?;
    Ok(match family {
        Family::Gaussian
        | Family::KnownVar
        | Family::Gamma
        | Family::Beta
        | Family::NBinomial
        | Family::NegBinomial
        | Family::Poisson
        | Family::Geometric
        | Family::Binomial
        | Family::InvGauss
        | Family::Logistic
        | Family::JSU => mu.clone(),
        Family::StudentT => {
            if any(&columns[2], |d| d <= 1.0) {
                return refuse("StudentTLSS mean requires df > 1");
            }
            mu.clone()
        }
        // digamma(1) is minus Euler's constant.
        Family::Gumbel => map2(&|m, s| m + EULER_GAMMA * s),
        Family::ZIPo => map2(&|m, s| (1.0 - s) * m),
        Family::ZINB | Family::ZAGA => mu
            .iter()
            .zip(&columns[2])
            .map(|(&m, &nu)| (1.0 - nu) * m)
            .collect(),
        Family::ZAP => {
            if any(mu, |m| m <= 0.0) || any(&columns[1], |s| !(0.0..=1.0).contains(&s)) {
                return refuse("ZeroAdjustedPoissonLSS mean requires mu > 0 and sigma in [0, 1]");
            }
            map2(&|m, s| (1.0 - s) * m / -(-m).exp_m1())
        }
        Family::Pareto2 => {
            if any(mu, |m| m <= 0.0) || any(&columns[1], |s| s <= 1.0) {
                return refuse("Pareto2LSS mean requires mu > 0 and sigma > 1");
            }
            map2(&|m, s| m / (s - 1.0))
        }
        Family::LogNormal => {
            if any(&columns[1], |s| s <= 0.0) {
                return refuse("LogNormalLSS mean requires sigma > 0");
            }
            map2(&|m, s| (m + s * s / 2.0).exp())
        }
        Family::LogLog => {
            if any(&columns[1], |s| s <= 0.0 || s >= 1.0) {
                return refuse("LogLogLSS mean requires 0 < sigma < 1");
            }
            map2(&|m, s| m.exp() * PI * s / (PI * s).sin())
        }
        Family::Weibull => {
            if any(&columns[1], |s| s <= 0.0) {
                return refuse("WeibullLSS mean requires sigma > 0");
            }
            map2(&|m, s| (m + lgamma(1.0 + s)).exp())
        }
        Family::BCCG => {
            return refuse(
                "BCCGLSS mu is the median; its general truncated Box-Cox mean is not implemented",
            )
        }
        Family::BCT => {
            return refuse(
                "BCTLSS mu is a location parameter; its truncated Box-Cox-t mean is not \
                 implemented",
            )
        }
        Family::BCPE => {
            return refuse(
                "BCPELSS mu is a location parameter; its truncated Box-Cox mean is not implemented",
            )
        }
    })
}

/// Euler's constant, `-digamma(1)`.
const EULER_GAMMA: f64 = std::f64::consts::EULER_GAMMA;

/// Box-Cox standardized value `((y/mu)^nu - 1) / (nu sigma)`, with its
/// `log(y/mu) / sigma` limit for `|nu| < 1e-10`.
pub(crate) fn boxcox_z(y: f64, mu: f64, sigma: f64, nu: f64) -> f64 {
    if nu.abs() < 1e-10 {
        (y / mu).ln() / sigma
    } else {
        ((y / mu).powf(nu) - 1.0) / (nu * sigma)
    }
}

/// Log density of the standard Student-t distribution with `df` degrees of
/// freedom, in log-gamma form so that it stays finite for a large `df`.
fn student_t_log_pdf(t: f64, df: f64) -> f64 {
    lgamma(0.5 * (df + 1.0))
        - lgamma(0.5 * df)
        - 0.5 * (df * PI).ln()
        - 0.5 * (df + 1.0) * (t * t / df).ln_1p()
}

// ─── Loader's saddle-point densities ──────────────────────────────────────
// R computes its binomial, Poisson, negative binomial, gamma and beta
// densities through Catherine Loader's saddle-point expansion ("Fast and
// accurate computation of binomial probabilities", 2000). It writes the log
// density as a Stirling-series error term `stirlerr` plus a deviance term
// `bd0` evaluated without cancellation, which keeps full relative accuracy
// where the log-gamma form loses digits (large counts, large shapes). The
// functions below follow R's `nmath` sources.

/// `ln(n!) - ln(sqrt(2 pi n) (n/e)^n)`, the error of Stirling's formula.
fn stirlerr(n: f64) -> f64 {
    const S0: f64 = 1.0 / 12.0;
    const S1: f64 = 1.0 / 360.0;
    const S2: f64 = 1.0 / 1260.0;
    const S3: f64 = 1.0 / 1680.0;
    const S4: f64 = 1.0 / 1188.0;
    if n <= 15.0 {
        return lgamma(n + 1.0) - (n + 0.5) * n.ln() + n - LN_SQRT_2PI;
    }
    let nn = n * n;
    if n > 500.0 {
        (S0 - S1 / nn) / n
    } else if n > 80.0 {
        (S0 - (S1 - S2 / nn) / nn) / n
    } else if n > 35.0 {
        (S0 - (S1 - (S2 - S3 / nn) / nn) / nn) / n
    } else {
        (S0 - (S1 - (S2 - (S3 - S4 / nn) / nn) / nn) / nn) / n
    }
}

/// The deviance term `x ln(x / np) + np - x`, by a series when `x` is near
/// `np` so that the two large terms do not cancel.
fn bd0(x: f64, np: f64) -> f64 {
    if (x - np).abs() < 0.1 * (x + np) {
        let mut v = (x - np) / (x + np);
        let mut s = (x - np) * v;
        if s.abs() < f64::MIN_POSITIVE {
            return s;
        }
        let mut ej = 2.0 * x * v;
        v *= v;
        for j in 1..1000 {
            ej *= v;
            let s1 = s + ej / (2 * j + 1) as f64;
            if s1 == s {
                return s1;
            }
            s = s1;
        }
    }
    x * (x / np).ln() + np - x
}

/// Poisson mass at `x >= 0` (not necessarily an integer) for mean `lambda`.
fn dpois_raw(x: f64, lambda: f64) -> f64 {
    if lambda == 0.0 {
        return if x == 0.0 { 1.0 } else { 0.0 };
    }
    if !lambda.is_finite() || x < 0.0 {
        return 0.0;
    }
    if x <= lambda * f64::MIN_POSITIVE {
        return (-lambda).exp();
    }
    if lambda < x * f64::MIN_POSITIVE {
        return (-lambda + x * lambda.ln() - lgamma(x + 1.0)).exp();
    }
    (-stirlerr(x) - bd0(x, lambda)).exp() / (2.0 * PI * x).sqrt()
}

/// Binomial mass at `x` for `n` trials, success probability `p` and its
/// complement `q`.
fn dbinom_raw(x: f64, n: f64, p: f64, q: f64) -> f64 {
    if p == 0.0 {
        return if x == 0.0 { 1.0 } else { 0.0 };
    }
    if q == 0.0 {
        return if x == n { 1.0 } else { 0.0 };
    }
    if x == 0.0 {
        if n == 0.0 {
            return 1.0;
        }
        let lc = if p < 0.1 {
            -bd0(n, n * q) - n * p
        } else {
            n * q.ln()
        };
        return lc.exp();
    }
    if x == n {
        let lc = if q < 0.1 {
            -bd0(n, n * p) - n * q
        } else {
            n * p.ln()
        };
        return lc.exp();
    }
    if x < 0.0 || x > n {
        return 0.0;
    }
    let lc = stirlerr(n) - stirlerr(x) - stirlerr(n - x) - bd0(x, n * p) - bd0(n - x, n * q);
    let lf = (2.0 * PI).ln() + x.ln() + (-x / n).ln_1p();
    (lc - 0.5 * lf).exp()
}

/// Negative binomial mass at count `x` for size `size` and mean `mu`, as R's
/// `dnbinom(x, size = size, mu = mu)`.
fn dnbinom_mu(x: f64, size: f64, mu: f64) -> f64 {
    if x == 0.0 {
        let log_p0 = if size < mu {
            (size / (size + mu)).ln()
        } else {
            (-mu / (size + mu)).ln_1p()
        };
        return (size * log_p0).exp();
    }
    if x < 1e-10 * size {
        let p = if size < mu {
            (size / (1.0 + size / mu)).ln()
        } else {
            (mu / (1.0 + mu / size)).ln()
        };
        return (x * p - mu - lgamma(x + 1.0) + (x * (x - 1.0) / (2.0 * size)).ln_1p()).exp();
    }
    let ratio = size / (size + x);
    ratio * dbinom_raw(size, x + size, size / (size + mu), mu / (size + mu))
}

/// Gamma density at `x` for shape `shape` and scale `scale`.
fn dgamma(x: f64, shape: f64, scale: f64) -> f64 {
    if x < 0.0 {
        return 0.0;
    }
    if x == 0.0 {
        return if shape < 1.0 {
            f64::INFINITY
        } else if shape > 1.0 {
            0.0
        } else {
            1.0 / scale
        };
    }
    if shape < 1.0 {
        dpois_raw(shape, x / scale) * shape / x
    } else {
        dpois_raw(shape - 1.0, x / scale) / scale
    }
}

/// Beta density at `x` for shapes `a` and `b`.
fn dbeta(x: f64, a: f64, b: f64) -> f64 {
    if !(0.0..=1.0).contains(&x) {
        return 0.0;
    }
    if x == 0.0 {
        return if a > 1.0 {
            0.0
        } else if a < 1.0 {
            f64::INFINITY
        } else {
            b
        };
    }
    if x == 1.0 {
        return if b > 1.0 {
            0.0
        } else if b < 1.0 {
            f64::INFINITY
        } else {
            a
        };
    }
    if a <= 2.0 || b <= 2.0 {
        let lbeta = lgamma(a) + lgamma(b) - lgamma(a + b);
        ((a - 1.0) * x.ln() + (b - 1.0) * (-x).ln_1p() - lbeta).exp()
    } else {
        (a + b - 1.0) * dbinom_raw(a - 1.0, a + b - 2.0, x, 1.0 - x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cols(values: &[&[f64]]) -> Vec<Vec<f64>> {
        values.iter().map(|c| c.to_vec()).collect()
    }

    fn rel(a: f64, b: f64) -> f64 {
        ((a - b) / b).abs()
    }

    // Reference values are R 4.5.1's d* functions.

    #[test]
    fn gamma_density_matches_r_dgamma() {
        // dgamma(c(0.5, 2, 7), shape = c(0.5, 3, 400), scale = c(2, 1/3, 3/400))
        let out = lss_pdf(
            "GammaLSS",
            &cols(&[&[0.5, 2.0, 7.0], &[1.0, 1.0, 3.0], &[0.5, 3.0, 400.0]]),
        )
        .unwrap();
        let r = [
            0.43939128946772243,
            0.13385261753998337,
            4.2047674304935883e-85,
        ];
        assert!(rel(out.values[0], r[0]) < 1e-13);
        assert!(rel(out.values[1], r[1]) < 1e-13);
        assert!(rel(out.values[2], r[2]) < 1e-11);
    }

    #[test]
    fn negative_binomial_mass_matches_r_dnbinom() {
        // dnbinom(c(0, 2, 30), mu = c(1, 4, 20), size = c(2, 3, 1e4))
        let out = lss_pdf(
            "NBinomialLSS",
            &cols(&[&[0.0, 2.0, 30.0], &[1.0, 4.0, 20.0], &[2.0, 3.0, 1e4]]),
        )
        .unwrap();
        let r = [
            0.444_444_444_444_444_5,
            0.15422145534598675,
            0.008_372_710_926_346_933,
        ];
        for (v, e) in out.values.iter().zip(r) {
            assert!(rel(*v, e) < 1e-12, "{v} vs {e}");
        }
    }

    #[test]
    fn count_mass_is_zero_off_the_integers() {
        let out = lss_pdf("PoissonLSS", &cols(&[&[-1.0, 1.5, 2.0], &[2.0, 2.0, 2.0]])).unwrap();
        assert_eq!(out.values[0], 0.0);
        assert_eq!(out.values[1], 0.0);
        assert!(rel(out.values[2], 2.0 * (-2.0f64).exp()) < 1e-14);
    }

    #[test]
    fn stirlerr_series_meets_the_log_gamma_form_at_fifteen() {
        let n = 15.0 + 1e-9;
        let series = stirlerr(n);
        let direct = lgamma(n + 1.0) - (n + 0.5) * n.ln() + n - LN_SQRT_2PI;
        assert!(rel(series, direct) < 1e-10);
    }

    #[test]
    fn zero_adjusted_gamma_puts_nu_at_zero() {
        let out = lss_pdf(
            "ZeroAdjustedGammaLSS",
            &cols(&[&[-1.0, 0.0], &[2.0, 2.0], &[0.5, 0.5], &[0.3, 0.3]]),
        )
        .unwrap();
        assert_eq!(out.values, vec![0.0, 0.3]);
    }

    #[test]
    fn student_t_density_is_finite_at_large_df() {
        let out = lss_pdf("StudentTLSS", &cols(&[&[0.0], &[1.0], &[2.0], &[400.0]])).unwrap();
        // R: dt(-0.5, 400) / 2
        assert!(rel(out.values[0], 0.1758745944821675) < 1e-12);
    }

    #[test]
    fn deviance_floors_a_zero_density() {
        let out = lss_deviance_increment("PoissonLSS", &cols(&[&[1.5], &[2.0]])).unwrap();
        assert_eq!(out.values[0], -2.0 * f64::MIN_POSITIVE.ln());
    }

    #[test]
    fn count_cdf_floors_a_non_integer_response() {
        // R: pnbinom(2, mu = 4, size = 2) = 0.40740740740740744
        let at = |y: f64| {
            lss_cdf("NBinomialLSS", &cols(&[&[y], &[4.0], &[2.0]]))
                .unwrap()
                .values[0]
        };
        assert!(rel(at(2.0), 0.40740740740740744) < 1e-13);
        assert_eq!(at(2.5), at(2.0));
        assert_eq!(at(-0.5), 0.0);
    }

    #[test]
    fn box_cox_cdf_is_zero_at_and_below_zero_and_one_at_infinity() {
        for family in ["BCTLSS", "BCPELSS"] {
            let out = lss_cdf(
                family,
                &cols(&[
                    &[-1.0, 0.0, f64::INFINITY],
                    &[1.7; 3],
                    &[0.4; 3],
                    &[0.3; 3],
                    &[5.0; 3],
                ]),
            )
            .unwrap();
            assert_eq!(out.values[0], 0.0);
            assert_eq!(out.values[1], 0.0);
            assert!(
                (out.values[2] - 1.0).abs() < 1e-12,
                "{family}: {}",
                out.values[2]
            );
        }
    }

    #[test]
    fn mixture_and_failure_time_means() {
        let zap = lss_mean("ZeroAdjustedPoissonLSS", &cols(&[&[2.0], &[0.25]])).unwrap();
        assert!(rel(zap[0], 0.75 * 2.0 / (1.0 - (-2.0f64).exp())) < 1e-15);
        let weibull = lss_mean("WeibullLSS", &cols(&[&[0.5], &[1.0]])).unwrap();
        assert!(rel(weibull[0], 0.5f64.exp()) < 1e-15);
        // Binomial takes its parameter alone; the trial count is not a column.
        assert_eq!(
            lss_mean("BinomialLSS", &cols(&[&[0.3]])).unwrap(),
            vec![0.3]
        );
    }

    #[test]
    fn undefined_means_are_refused() {
        let t = lss_mean("StudentTLSS", &cols(&[&[0.0], &[1.0], &[1.0]])).unwrap_err();
        assert!(t.to_string().contains("StudentTLSS mean requires df > 1"));
        let p = lss_mean("Pareto2LSS", &cols(&[&[2.0], &[1.0]])).unwrap_err();
        assert!(p.to_string().contains("sigma > 1"));
        assert!(lss_mean("BCTLSS", &cols(&[&[1.0], &[1.0], &[0.1], &[5.0]])).is_err());
    }

    #[test]
    fn family_without_density_is_refused() {
        let err = lss_pdf("GumbelLSS", &cols(&[&[0.0], &[0.0], &[1.0]])).unwrap_err();
        assert!(err
            .to_string()
            .contains("no density for family 'GumbelLSS'"));
    }
}
