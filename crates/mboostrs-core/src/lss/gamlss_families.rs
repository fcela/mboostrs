#![allow(unused_imports)]
use super::*;

// ─── PoissonLSS (PO) — 1 param, log link ──────────────────────────

/// Poisson distribution (single parameter mu, mean, log link).
/// y >= 0 integer. Matches `gamlss.dist::PO()`.
pub struct PoissonLSS;

impl LssFamily for PoissonLSS {
    fn name(&self) -> &'static str {
        "PoissonLSS"
    }
    fn n_params(&self) -> usize {
        1
    }
    fn param_names(&self) -> &'static [&'static str] {
        &["mu"]
    }
    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        for (i, &v) in y.iter().enumerate() {
            if !v.is_finite() || v < 0.0 || v.fract() != 0.0 {
                return Err(MboostrsError::BadResponse(
                    "PoissonLSS".into(),
                    format!("y must be non-negative integer at {i}"),
                ));
            }
        }
        Ok(())
    }
    fn offset(&self, _k: usize, y: &[f64], w: &[f64], _prior: &[f64]) -> f64 {
        let n = y.len();
        let (sw, sy) = (0..n).fold((0.0, 0.0), |(s, sy), i| (s + w[i], sy + w[i] * y[i]));
        if sw > 0.0 {
            (sy / sw).ln()
        } else {
            0.0
        }
    }
    fn ngradient(&self, _k: usize, y: &[f64], etas: &[Array1<f64>], _w: &[f64], out: &mut [f64]) {
        let mu = etas[0].as_slice().expect("eta contiguous");
        for i in 0..y.len() {
            out[i] = y[i] - mu[i].exp();
        }
    }
    fn risk(&self, y: &[f64], etas: &[Array1<f64>], w: &[f64]) -> f64 {
        let mu = etas[0].as_slice().expect("eta contiguous");
        let mut s = 0.0;
        for i in 0..y.len() {
            let m = mu[i].exp();
            s += w[i] * (-y[i] * m.ln() + m + lgamma(y[i] + 1.0));
        }
        s
    }
    fn response(&self, _k: usize, eta: &[f64], out: &mut [f64]) {
        for i in 0..eta.len() {
            out[i] = eta[i].exp();
        }
    }
    fn clone_box(&self) -> Box<dyn LssFamily> {
        Box::new(PoissonLSS)
    }
}

// ─── GeometricLSS (GEOM) — 1 param, log link ─────────────────────

/// Geometric distribution (single parameter mu, mean, log link).
/// y >= 0 integer. Matches `gamlss.dist::GEOM()`.
pub struct GeometricLSS;

impl LssFamily for GeometricLSS {
    fn name(&self) -> &'static str {
        "GeometricLSS"
    }
    fn n_params(&self) -> usize {
        1
    }
    fn param_names(&self) -> &'static [&'static str] {
        &["mu"]
    }
    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        for (i, &v) in y.iter().enumerate() {
            if !v.is_finite() || v < 0.0 || v.fract() != 0.0 {
                return Err(MboostrsError::BadResponse(
                    "GeometricLSS".into(),
                    format!("y must be non-negative integer at {i}"),
                ));
            }
        }
        Ok(())
    }
    fn offset(&self, _k: usize, y: &[f64], w: &[f64], _prior: &[f64]) -> f64 {
        let n = y.len();
        let (sw, sy) = (0..n).fold((0.0, 0.0), |(s, sy), i| (s + w[i], sy + w[i] * y[i]));
        if sw > 0.0 {
            (sy / sw).ln()
        } else {
            0.0
        }
    }
    fn ngradient(&self, _k: usize, y: &[f64], etas: &[Array1<f64>], _w: &[f64], out: &mut [f64]) {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        for i in 0..y.len() {
            let mu = eta_mu[i].exp();
            out[i] = y[i] - (y[i] + 1.0) * mu / (1.0 + mu);
        }
    }
    fn risk(&self, y: &[f64], etas: &[Array1<f64>], w: &[f64]) -> f64 {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let mut s = 0.0;
        for i in 0..y.len() {
            let mu = eta_mu[i].exp();
            s += w[i] * (-y[i] * mu.ln() + (y[i] + 1.0) * (1.0 + mu).ln());
        }
        s
    }
    fn response(&self, _k: usize, eta: &[f64], out: &mut [f64]) {
        for i in 0..eta.len() {
            out[i] = eta[i].exp();
        }
    }
    fn clone_box(&self) -> Box<dyn LssFamily> {
        Box::new(GeometricLSS)
    }
}

// ─── BinomialLSS (BI) — 1 param, logit link ──────────────────────

/// Binomial distribution for proportion data (single parameter mu = probability,
/// logit link). `y` in `[0, 1]`, a proportion. Matches `gamlss.dist::BI()`.
pub struct BinomialLSS;

impl LssFamily for BinomialLSS {
    fn name(&self) -> &'static str {
        "BinomialLSS"
    }
    fn n_params(&self) -> usize {
        1
    }
    fn param_names(&self) -> &'static [&'static str] {
        &["mu"]
    }
    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        for (i, &v) in y.iter().enumerate() {
            if !v.is_finite() || v < 0.0 || v > 1.0 {
                return Err(MboostrsError::BadResponse(
                    "BinomialLSS".into(),
                    format!("y must be proportion in [0,1] at {i}"),
                ));
            }
        }
        Ok(())
    }
    fn offset(&self, _k: usize, y: &[f64], w: &[f64], _prior: &[f64]) -> f64 {
        let n = y.len();
        let (sw, sy) = (0..n).fold((0.0, 0.0), |(s, sy), i| (s + w[i], sy + w[i] * y[i]));
        let p = if sw > 0.0 {
            (sy / sw).clamp(1e-8, 1.0 - 1e-8)
        } else {
            0.5
        };
        (p / (1.0 - p)).ln()
    }
    fn ngradient(&self, _k: usize, y: &[f64], etas: &[Array1<f64>], _w: &[f64], out: &mut [f64]) {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        for i in 0..y.len() {
            let mu = 1.0 / (1.0 + (-eta_mu[i]).exp());
            out[i] = y[i] - mu;
        }
    }
    fn risk(&self, y: &[f64], etas: &[Array1<f64>], w: &[f64]) -> f64 {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let mut s = 0.0;
        for i in 0..y.len() {
            let mu = 1.0 / (1.0 + (-eta_mu[i]).exp());
            let eps = 1e-15;
            s += w[i] * (-y[i] * (mu + eps).ln() - (1.0 - y[i]) * (1.0 - mu + eps).ln());
        }
        s
    }
    fn response(&self, _k: usize, eta: &[f64], out: &mut [f64]) {
        for i in 0..eta.len() {
            out[i] = 1.0 / (1.0 + (-eta[i]).exp());
        }
    }
    fn clone_box(&self) -> Box<dyn LssFamily> {
        Box::new(BinomialLSS)
    }
}

// ─── NegativeBinomialLSS (NBI) — 2 param, mu log, sigma log ─────

/// Negative Binomial type I (variance = mu + sigma * mu^2). Matches `gamlss.dist::NBI()`.
pub struct NegativeBinomialLSS;

impl LssFamily for NegativeBinomialLSS {
    fn name(&self) -> &'static str {
        "NegativeBinomialLSS"
    }
    fn n_params(&self) -> usize {
        2
    }
    fn param_names(&self) -> &'static [&'static str] {
        &["mu", "sigma"]
    }
    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        for (i, &v) in y.iter().enumerate() {
            if !v.is_finite() || v < 0.0 || v.fract() != 0.0 {
                return Err(MboostrsError::BadResponse(
                    "NegativeBinomialLSS".into(),
                    format!("y must be non-negative integer at {i}"),
                ));
            }
        }
        Ok(())
    }
    fn offset(&self, k: usize, y: &[f64], w: &[f64], _prior: &[f64]) -> f64 {
        let n = y.len();
        match k {
            0 => {
                let (sw, sy) = (0..n).fold((0.0, 0.0), |(s, sy), i| (s + w[i], sy + w[i] * y[i]));
                if sw > 0.0 {
                    (sy / sw).ln()
                } else {
                    0.0
                }
            }
            1 => {
                // gamlss NBI sigma.initial, as in the ZINBLSS offset:
                // max((var(y) - mean(y)) / mean(y)^2, 0.1), from variance
                // mu + sigma * mu^2; var has the n - 1 denominator, here the
                // weighted `sw - 1`.
                let (sw, sy) = (0..n).fold((0.0, 0.0), |(s, sy), i| (s + w[i], sy + w[i] * y[i]));
                let mean = sy / sw.max(1e-300);
                let ss = (0..n).fold(0.0, |s, i| s + w[i] * (y[i] - mean) * (y[i] - mean));
                let var = if sw > 1.0 { ss / (sw - 1.0) } else { 0.0 };
                let sigma = ((var - mean) / (mean * mean).max(1e-300)).max(0.1);
                sigma.ln().clamp(-10.0, 10.0)
            }
            _ => unreachable!("NBI has 2 parameters"),
        }
    }
    fn ngradient(&self, k: usize, y: &[f64], etas: &[Array1<f64>], _w: &[f64], out: &mut [f64]) {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        let n = y.len();
        match k {
            0 => {
                for i in 0..n {
                    let mu = eta_mu[i].exp();
                    let sig = eta_sigma[i].exp();
                    // NLL = -y*log(mu) + (y + 1/sig)*log(1 + sig*mu) + const
                    // d(NLL)/d(eta_mu) = -y + (y*sigma + 1)*mu/(1 + sigma*mu)
                    out[i] = y[i] - (y[i] * sig + 1.0) * mu / (1.0 + sig * mu);
                }
            }
            1 => {
                for i in 0..n {
                    let mu = eta_mu[i].exp();
                    let sig = eta_sigma[i].exp();
                    let t = sig * mu;
                    out[i] = y[i]
                        - (y[i] * sig + 1.0) * mu / (1.0 + t)
                        - digamma(y[i] + 1.0 / sig) / sig
                        + digamma(1.0 / sig) / sig
                        + (1.0 + t).ln() / sig;
                }
            }
            _ => unreachable!("NBI has 2 parameters"),
        }
    }
    fn risk(&self, y: &[f64], etas: &[Array1<f64>], w: &[f64]) -> f64 {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        let mut s = 0.0;
        for i in 0..y.len() {
            let mu = eta_mu[i].exp();
            let sig = eta_sigma[i].exp();
            let t = sig * mu;
            s += w[i]
                * (-y[i] * t.ln() + (t + 1.0).ln() * (y[i] + 1.0 / sig) - lgamma(y[i] + 1.0 / sig)
                    + lgamma(1.0 / sig)
                    + lgamma(y[i] + 1.0));
        }
        s
    }
    fn response(&self, k: usize, eta: &[f64], out: &mut [f64]) {
        match k {
            0 => {
                for i in 0..eta.len() {
                    out[i] = eta[i].exp();
                }
            }
            1 => {
                for i in 0..eta.len() {
                    out[i] = eta[i].exp();
                }
            }
            _ => unreachable!("NBI has 2 parameters"),
        }
    }
    fn clone_box(&self) -> Box<dyn LssFamily> {
        Box::new(NegativeBinomialLSS)
    }
}

// ─── GumbelLSS (gamlss RG) — 2 param, identity/log links ───────

/// Gumbel distribution for maxima — the standard extreme-value type I
/// (location mu, identity link; scale sigma, log link). y ∈ ℝ.
///
/// Matches `gamlss.dist::RG()`, which gamlss labels "Reverse Gumbel", **not**
/// `gamlss.dist::GU()`. gamlss's naming is the reverse of the common one: its
/// `GU` is the Gumbel for *minima* (left-skewed), and its `RG` is the
/// right-skewed form almost everyone means by "Gumbel". This implementation is
/// the latter, so the mboostrs name is the conventional one and the gamlss
/// cross-reference is `RG`.
///
/// Verified numerically rather than by name: on
/// `y = [0.4, 1.3, -0.7, 2.1, 0.0]` with `mu = 0.5, sigma = 2`, this `risk()`
/// returns 9.042800226406863, which is `-sum(dRG(..., log = TRUE))` to all
/// printed digits; `-sum(dGU(...))` is 9.161943. The doc comment previously
/// claimed `GU()` parity, which was wrong in exactly the direction that would
/// have sent someone comparing against R to the wrong reference.
///
/// gamlss's `GU()` — the mirrored form — is not implemented.
pub struct GumbelLSS;

impl LssFamily for GumbelLSS {
    fn scale_map(&self) -> Option<&'static [ParamScale]> {
        const MAP: &[ParamScale] = &[ParamScale::Multiply, ParamScale::Shift(1.0)];
        Some(MAP)
    }

    fn name(&self) -> &'static str {
        "GumbelLSS"
    }
    fn n_params(&self) -> usize {
        2
    }
    fn param_names(&self) -> &'static [&'static str] {
        &["mu", "sigma"]
    }
    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        for (i, &v) in y.iter().enumerate() {
            if !v.is_finite() {
                return Err(MboostrsError::BadResponse(
                    "GumbelLSS".into(),
                    format!("y must be finite at {i}"),
                ));
            }
        }
        Ok(())
    }
    fn offset(&self, k: usize, y: &[f64], w: &[f64], _prior: &[f64]) -> f64 {
        let n = y.len();
        match k {
            0 => {
                // mu: identity link, weighted mean
                let (sw, sy) = (0..n).fold((0.0, 0.0), |(s, sy), i| (s + w[i], sy + w[i] * y[i]));
                if sw > 0.0 {
                    sy / sw
                } else {
                    0.0
                }
            }
            1 => {
                // sigma: log link, approximate via std dev
                let (sw, sy, syy) = (0..n).fold((0.0, 0.0, 0.0), |(s, sy, syy), i| {
                    (s + w[i], sy + w[i] * y[i], syy + w[i] * y[i] * y[i])
                });
                let mean = sy / sw.max(1e-300);
                let var = (syy / sw - mean * mean).max(1e-12);
                // sigma: log link, offset = log(weighted.sd). The leading 0.5 was a
                // factor-of-2 bug (it produced 0.5*log(std) instead of log(std)),
                // inconsistent with GaussianLSS (lss.rs:635) and JSULSS (lss.rs:2973).
                var.sqrt().ln().clamp(-10.0, 10.0)
            }
            _ => unreachable!("GumbelLSS has 2 parameters"),
        }
    }
    fn ngradient(&self, k: usize, y: &[f64], etas: &[Array1<f64>], _w: &[f64], out: &mut [f64]) {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        let n = y.len();
        match k {
            0 => {
                for i in 0..n {
                    let sig = eta_sigma[i].exp();
                    out[i] = (1.0 - (-(y[i] - eta_mu[i]) / sig).exp()) / sig;
                }
            }
            1 => {
                for i in 0..n {
                    let x = y[i] - eta_mu[i];
                    let sig = eta_sigma[i].exp();
                    let z = (-x / sig).exp();
                    out[i] = x / sig * (1.0 - z) - 1.0;
                }
            }
            _ => unreachable!("GumbelLSS has 2 parameters"),
        }
    }
    fn risk(&self, y: &[f64], etas: &[Array1<f64>], w: &[f64]) -> f64 {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        let mut s = 0.0;
        for i in 0..y.len() {
            let x = y[i] - eta_mu[i];
            let sig = eta_sigma[i].exp();
            s += w[i] * ((-x / sig).exp() + x / sig + eta_sigma[i]);
        }
        s
    }
    fn response(&self, k: usize, eta: &[f64], out: &mut [f64]) {
        match k {
            0 => out.copy_from_slice(eta),
            1 => {
                for i in 0..eta.len() {
                    out[i] = eta[i].exp();
                }
            }
            _ => unreachable!("GumbelLSS has 2 parameters"),
        }
    }
    fn clone_box(&self) -> Box<dyn LssFamily> {
        Box::new(GumbelLSS)
    }
}

// ─── Pareto2LSS (PARETO2) — 2 param, log link ──────────────────

/// Pareto type 2 (Lomax) distribution: mu = scale (log link), sigma = shape
/// (log link), y >= 0. Density `f(y) = sigma * mu^sigma / (y + mu)^(sigma+1)`,
/// so `NLL = (sigma+1)*ln(1 + y/mu) + ln(mu) - ln(sigma)`.
///
/// The shape parameterization differs from `gamlss.dist::PARETO2()`, which
/// uses `1/sigma`. For `y = [0.15, 0.8, 1.0, 2.6, 7.4]`, `mu = 1.6`, and
/// `sigma = 2.3`, this family's `risk()` is 10.305999389708, while the R
/// density evaluated at `sigma = 2.3` gives a negative log-likelihood of
/// 11.784358047059. Evaluating the R density at `sigma = 1/2.3` gives the
/// same distribution as this family's parameterization.
pub struct Pareto2LSS;

impl LssFamily for Pareto2LSS {
    fn scale_map(&self) -> Option<&'static [ParamScale]> {
        const MAP: &[ParamScale] = &[ParamScale::Shift(1.0), ParamScale::Invariant];
        Some(MAP)
    }

    fn name(&self) -> &'static str {
        "Pareto2LSS"
    }
    fn n_params(&self) -> usize {
        2
    }
    fn param_names(&self) -> &'static [&'static str] {
        &["mu", "sigma"]
    }
    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        for (i, &v) in y.iter().enumerate() {
            if !v.is_finite() || v < 0.0 {
                return Err(MboostrsError::BadResponse(
                    "Pareto2LSS".into(),
                    format!("y must be >= 0 at {i}"),
                ));
            }
        }
        Ok(())
    }
    fn offset(&self, k: usize, y: &[f64], w: &[f64], _prior: &[f64]) -> f64 {
        let n = y.len();
        match k {
            0 => {
                let (sw, sy) = (0..n).fold((0.0, 0.0), |(s, sy), i| (s + w[i], sy + w[i] * y[i]));
                if sw > 0.0 {
                    (sy / sw).ln().clamp(-10.0, 10.0)
                } else {
                    0.0
                }
            }
            1 => {
                let (sw, sly) = (0..n).fold((0.0, 0.0), |(s, sly), i| {
                    (s + w[i], sly + w[i] * (y[i] / (y[i] + 1.0).max(1.0)).ln())
                });
                if sw > 0.0 {
                    (-sly / sw).ln().clamp(-10.0, 10.0)
                } else {
                    0.0
                }
            }
            _ => unreachable!("Pareto2LSS has 2 parameters"),
        }
    }
    fn ngradient(&self, k: usize, y: &[f64], etas: &[Array1<f64>], _w: &[f64], out: &mut [f64]) {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        let n = y.len();
        match k {
            0 => {
                // d(NLL)/d(eta_mu) = (mu - sig*y) / (mu+y)
                // ngradient_mu = (sig*y - mu) / (mu+y)
                for i in 0..n {
                    let mu = eta_mu[i].exp();
                    let sig = eta_sigma[i].exp();
                    out[i] = (sig * y[i] - mu) / (mu + y[i]);
                }
            }
            1 => {
                // d(NLL)/d(eta_sig) = sig*ln(1+y/mu) - 1
                // ngradient_sig = 1 - sig*ln(1+y/mu)
                for i in 0..n {
                    let mu = eta_mu[i].exp();
                    let sig = eta_sigma[i].exp();
                    out[i] = 1.0 - sig * (1.0 + y[i] / mu).ln();
                }
            }
            _ => unreachable!("Pareto2LSS has 2 parameters"),
        }
    }
    fn risk(&self, y: &[f64], etas: &[Array1<f64>], w: &[f64]) -> f64 {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        let mut s = 0.0;
        for i in 0..y.len() {
            let mu = eta_mu[i].exp();
            let sig = eta_sigma[i].exp();
            // NLL = (sig+1)*ln(1+y/mu) + ln(mu) - ln(sig)
            s += w[i] * ((sig + 1.0) * (1.0 + y[i] / mu).ln() + mu.ln() - sig.ln());
        }
        s
    }
    fn response(&self, k: usize, eta: &[f64], out: &mut [f64]) {
        match k {
            0 => {
                for i in 0..eta.len() {
                    out[i] = eta[i].exp();
                }
            }
            1 => {
                for i in 0..eta.len() {
                    out[i] = eta[i].exp();
                }
            }
            _ => unreachable!("Pareto2LSS has 2 parameters"),
        }
    }
    fn clone_box(&self) -> Box<dyn LssFamily> {
        Box::new(Pareto2LSS)
    }
}

// ─── ZeroAdjustedPoissonLSS (ZAP) — 2 param ─────────────────────

/// Zero-Adjusted Poisson (ZAP): mixture of a point mass at zero and a
/// zero-truncated Poisson. mu = mean of Poisson part (log link),
/// sigma = probability of zero (logit link). y >= 0 integer.
/// Matches `gamlss.dist::ZAP()`.
pub struct ZeroAdjustedPoissonLSS;

impl LssFamily for ZeroAdjustedPoissonLSS {
    fn name(&self) -> &'static str {
        "ZeroAdjustedPoissonLSS"
    }
    fn n_params(&self) -> usize {
        2
    }
    fn param_names(&self) -> &'static [&'static str] {
        &["mu", "sigma"]
    }
    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        for (i, &v) in y.iter().enumerate() {
            if !v.is_finite() || v < 0.0 || v.fract() != 0.0 {
                return Err(MboostrsError::BadResponse(
                    "ZeroAdjustedPoissonLSS".into(),
                    format!("y must be non-negative integer at {i}"),
                ));
            }
        }
        Ok(())
    }
    fn offset(&self, k: usize, y: &[f64], w: &[f64], _prior: &[f64]) -> f64 {
        let n = y.len();
        let (sw, _, sz) = (0..n).fold((0.0, 0.0, 0.0), |(sw, sy, sz), i| {
            (
                sw + w[i],
                sy + w[i] * y[i],
                sz + if y[i] < 0.5 { w[i] } else { 0.0 },
            )
        });
        match k {
            0 => {
                let nonzero_mean = (0..n)
                    .filter(|&i| y[i] > 0.5)
                    .fold((0.0, 0.0), |(s, sw), i| (s + w[i] * y[i], sw + w[i]));
                if nonzero_mean.1 > 0.0 {
                    (nonzero_mean.0 / nonzero_mean.1).ln()
                } else {
                    0.0
                }
            }
            1 => {
                let p0 = if sw > 0.0 { sz / sw } else { 0.0 };
                (p0 / (1.0 - p0).max(1e-10)).ln().clamp(-10.0, 10.0)
            }
            _ => unreachable!("ZAP has 2 parameters"),
        }
    }
    fn ngradient(&self, k: usize, y: &[f64], etas: &[Array1<f64>], _w: &[f64], out: &mut [f64]) {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        let n = y.len();
        match k {
            0 => {
                // ZAP mu gradient (log link):
                //   y=0: NLL = -ln(sig), does not depend on mu => ngradient = 0
                //   y>0: NLL = -ln(1-sig) + mu - y*ln(mu) + lgamma(y+1) - ln(1-exp(-mu))
                //         dNLL/dmu = 1 - y/mu + exp(-mu)/(1-exp(-mu))
                //   ngradient_mu = y/mu - 1 - exp(-mu)/(1-exp(-mu))
                //   But with log link eta_mu = ln(mu), d/d(eta_mu) = d/dm * mu
                //   So ngradient_eta_mu = (y/mu - 1 - exp(-mu)/(1-exp(-mu))) * mu
                //                      = y - mu - mu/(exp(mu)-1)
                for i in 0..n {
                    if y[i] < 0.5 {
                        out[i] = 0.0;
                    } else {
                        let mu = eta_mu[i].exp();
                        out[i] = y[i] - mu - mu / (mu.exp() - 1.0);
                    }
                }
            }
            1 => {
                // ZAP sigma gradient (logit link):
                //   y=0: NLL = -ln(sig), dNLL/dsig = -1/sig, ngradient_raw = 1/sig
                //   y>0: NLL = -ln(1-sig) + ..., dNLL/dsig = 1/(1-sig), ngradient_raw = -1/(1-sig)
                //   Chain rule: d/d(eta_sig) = d/d(sig) * sig*(1-sig)
                for i in 0..n {
                    let sig = 1.0 / (1.0 + (-eta_sigma[i]).exp());
                    if y[i] < 0.5 {
                        out[i] = 1.0 - sig; // (1/sig) * sig*(1-sig)
                    } else {
                        out[i] = -sig; // (-1/(1-sig)) * sig*(1-sig)
                    }
                }
            }
            _ => unreachable!("ZAP has 2 parameters"),
        }
    }
    fn risk(&self, y: &[f64], etas: &[Array1<f64>], w: &[f64]) -> f64 {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        let mut s = 0.0;
        for i in 0..y.len() {
            let mu = eta_mu[i].exp();
            let sig = 1.0 / (1.0 + (-eta_sigma[i]).exp());
            if y[i] < 0.5 {
                // P(Y=0) = sig  (point mass only, no Poisson zero)
                s += w[i] * (-sig.ln());
            } else {
                // P(Y=y) = (1-sig) * exp(-mu) * mu^y / (y! * (1-exp(-mu)))
                let lpois = y[i] * mu.ln() - mu - lgamma(y[i] + 1.0);
                s += w[i] * (-(1.0 - sig).ln() - lpois + (1.0 - (-mu).exp()).ln());
            }
        }
        s
    }
    fn response(&self, k: usize, eta: &[f64], out: &mut [f64]) {
        match k {
            0 => {
                for i in 0..eta.len() {
                    out[i] = eta[i].exp();
                }
            }
            1 => {
                for i in 0..eta.len() {
                    out[i] = 1.0 / (1.0 + (-eta[i]).exp());
                }
            }
            _ => unreachable!("ZAP has 2 parameters"),
        }
    }
    fn clone_box(&self) -> Box<dyn LssFamily> {
        Box::new(ZeroAdjustedPoissonLSS)
    }
}

/// `log K_{i - 1/2}(x)` for `i = 0 ..= order_max`, where `K` is the modified
/// Bessel function of the second kind.
///
/// Half-integer orders have a closed form and an exact three-term recurrence,
/// so no series or continued fraction is needed:
///
/// * `K_{1/2}(x) = K_{-1/2}(x) = sqrt(pi / 2x) exp(-x)`;
/// * `K_{v+1}(x) = K_{v-1}(x) + (2v / x) K_v(x)`.
///
/// The recurrence adds two positive terms, so it is run in logs -- `K_{y-1/2}`
/// overflows for a count of a few hundred at a small `x`, and a PIG fit on
/// count data reaches that.
///
/// Used by [`PIGLSS`], whose density needs `K_{y-1/2}` and whose gradient
/// needs the ratio `K_{y+1/2} / K_{y-1/2}`.
pub(super) fn log_bessel_k_half_integer(x: f64, order_max: usize) -> Vec<f64> {
    let mut values = vec![f64::NEG_INFINITY; order_max + 1];
    if !(x.is_finite() && x > 0.0) {
        return values;
    }
    let base = 0.5 * (std::f64::consts::PI / (2.0 * x)).ln() - x;
    values[0] = base;
    if order_max == 0 {
        return values;
    }
    values[1] = base;
    for index in 2..=order_max {
        // `v = index - 3/2`, so this step produces order `index - 1/2` from
        // orders `index - 5/2` and `index - 3/2`.
        let v = index as f64 - 1.5;
        let scaled = (2.0 * v / x).ln() + values[index - 1];
        values[index] = log_add_exp(values[index - 2], scaled);
    }
    values
}

/// `ln(exp(a) + exp(b))` without overflowing.
pub(super) fn log_add_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let (high, low) = if a > b { (a, b) } else { (b, a) };
    high + (low - high).exp().ln_1p()
}

/// The PIG quantities every one of its methods needs, computed once.
///
/// `alpha = sqrt(1/sigma^2 + 2 mu / sigma)`, and
/// `ty = (mu / (alpha sigma)) K_{y+1/2}(alpha) / K_{y-1/2}(alpha)`, which is
/// what `gamlss.dist`'s `tofyPIG1` returns and what both of its derivatives
/// are written in terms of.
pub(super) struct PigTerms {
    pub(super) log_density: f64,
    pub(super) ty: f64,
}

pub(super) fn pig_terms(y: f64, mu: f64, sigma: f64) -> PigTerms {
    let count = y.max(0.0).round();
    let index = count as usize;
    let alpha = (1.0 / (sigma * sigma) + 2.0 * mu / sigma).sqrt();
    let logs = log_bessel_k_half_integer(alpha, index + 1);
    let log_k_lower = logs[index];
    let log_k_upper = logs[index + 1];
    let log_density =
        0.5 * (2.0 * alpha / std::f64::consts::PI).ln() + count * mu.ln() + 1.0 / sigma
            - count * (alpha * sigma).ln()
            - lgamma(count + 1.0)
            + log_k_lower;
    let ty = (mu / (alpha * sigma)) * (log_k_upper - log_k_lower).exp();
    PigTerms { log_density, ty }
}

// ─── PIGLSS (PIG) — 2 param, log links ─────────────────────────

/// Poisson-Inverse Gaussian (variance = mu + sigma * mu^2, so a larger
/// sigma means more overdispersion). y >= 0 integer. Matches
/// `gamlss.dist::PIG()`.
pub struct PIGLSS;

impl LssFamily for PIGLSS {
    fn name(&self) -> &'static str {
        "PIGLSS"
    }
    fn n_params(&self) -> usize {
        2
    }
    fn param_names(&self) -> &'static [&'static str] {
        &["mu", "sigma"]
    }
    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        for (index, &value) in y.iter().enumerate() {
            if !value.is_finite() || value < 0.0 || value != value.round() {
                return Err(MboostrsError::BadResponse(
                    "PIGLSS".into(),
                    format!(
                        "y must be a non-negative whole count, got {value} at \
                         index {index}"
                    ),
                ));
            }
        }
        Ok(())
    }
    fn offset(&self, k: usize, y: &[f64], w: &[f64], _prior: &[f64]) -> f64 {
        let n = y.len();
        let (sw, sy) = (0..n).fold((0.0, 0.0), |(s, sy), i| (s + w[i], sy + w[i] * y[i]));
        match k {
            0 => {
                if sw > 0.0 {
                    (sy / sw).ln()
                } else {
                    0.0
                }
            }
            1 => {
                // gamlss PIG sigma.initial: max((var(y) - mean(y)) / mean(y)^2,
                // 0.1), from variance = mu + sigma * mu^2; var has the n - 1
                // denominator, here the weighted `sw - 1`.
                let mean = sy / sw.max(1e-300);
                let sy2 = (0..n).fold(0.0, |s2, i| s2 + w[i] * (y[i] - mean) * (y[i] - mean));
                let var = if sw > 1.0 { sy2 / (sw - 1.0) } else { 0.0 };
                let sigma = ((var - mean) / (mean * mean).max(1e-300)).max(0.1);
                sigma.ln().clamp(-10.0, 10.0)
            }
            _ => unreachable!("PIG has 2 parameters"),
        }
    }
    fn ngradient(&self, k: usize, y: &[f64], etas: &[Array1<f64>], _w: &[f64], out: &mut [f64]) {
        // `gamlss.dist::PIG()` writes both derivatives in terms of one
        // quantity, `ty` from its `tofyPIG1`:
        //
        //   dl/dmu    = (y - ty) / mu
        //   dl/dsigma = (ty (1 + sigma mu) / mu - (1 + sigma y)) / sigma^2
        //
        // Both links are logs here, so the gradient with respect to eta picks
        // up `dmu/deta = mu` and `dsigma/deta = sigma`. The mu case reduces to
        // `y - ty` exactly, which is why the old approximation `y - mu` looked
        // plausible: it is this with `ty` replaced by the Poisson mean.
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        let n = y.len();
        match k {
            0 => {
                for i in 0..n {
                    let mu = eta_mu[i].exp();
                    let sigma = eta_sigma[i].exp();
                    out[i] = y[i] - pig_terms(y[i], mu, sigma).ty;
                }
            }
            1 => {
                for i in 0..n {
                    let mu = eta_mu[i].exp();
                    let sigma = eta_sigma[i].exp();
                    let ty = pig_terms(y[i], mu, sigma).ty;
                    out[i] = (ty * (1.0 + sigma * mu) / mu - (1.0 + sigma * y[i])) / sigma;
                }
            }
            _ => unreachable!("PIG has 2 parameters"),
        }
    }
    fn risk(&self, y: &[f64], etas: &[Array1<f64>], w: &[f64]) -> f64 {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        let mut total = 0.0;
        for i in 0..y.len() {
            let mu = eta_mu[i].exp();
            let sigma = eta_sigma[i].exp();
            total -= w[i] * pig_terms(y[i], mu, sigma).log_density;
        }
        total
    }
    fn response(&self, k: usize, eta: &[f64], out: &mut [f64]) {
        match k {
            0 => {
                for i in 0..eta.len() {
                    out[i] = eta[i].exp();
                }
            }
            1 => {
                for i in 0..eta.len() {
                    out[i] = eta[i].exp();
                }
            }
            _ => unreachable!("PIG has 2 parameters"),
        }
    }
    fn clone_box(&self) -> Box<dyn LssFamily> {
        Box::new(PIGLSS)
    }
}

// ─── BCCG (Box-Cox Cole-Green) — 3 param ───────────────────────

/// Box-Cox Cole-Green distribution (mu = median, sigma = CV, nu = skewness).
/// y > 0. Matches `gamlss.dist::BCCG()`.
pub struct BCCGLSS;

impl LssFamily for BCCGLSS {
    fn scale_map(&self) -> Option<&'static [ParamScale]> {
        const MAP: &[ParamScale] = &[
            ParamScale::Shift(1.0),
            ParamScale::Invariant,
            ParamScale::Invariant,
        ];
        Some(MAP)
    }

    fn name(&self) -> &'static str {
        "BCCGLSS"
    }
    fn n_params(&self) -> usize {
        3
    }
    fn param_names(&self) -> &'static [&'static str] {
        &["mu", "sigma", "nu"]
    }
    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        for (i, &v) in y.iter().enumerate() {
            if !v.is_finite() || v <= 0.0 {
                return Err(MboostrsError::BadResponse(
                    "BCCGLSS".into(),
                    format!("y must be > 0 at {i}"),
                ));
            }
        }
        Ok(())
    }
    fn offset(&self, k: usize, y: &[f64], w: &[f64], _prior: &[f64]) -> f64 {
        let n = y.len();
        match k {
            0 => {
                let (sw, sly) =
                    (0..n).fold((0.0, 0.0), |(s, sly), i| (s + w[i], sly + w[i] * y[i].ln()));
                if sw > 0.0 {
                    sly / sw
                } else {
                    0.0
                }
            }
            1 => {
                let (sw, sly) =
                    (0..n).fold((0.0, 0.0), |(s, sly), i| (s + w[i], sly + w[i] * y[i].ln()));
                let lmu = if sw > 0.0 { sly / sw } else { 0.0 };
                let slv = (0..n)
                    .map(|i| w[i] * (y[i].ln() - lmu) * (y[i].ln() - lmu))
                    .sum::<f64>();
                (slv / (sw - 1.0).max(1.0)).sqrt().ln().clamp(-5.0, 5.0)
            }
            2 => 0.0, // nu offset = 0 (symmetric)
            _ => unreachable!("BCCG has 3 parameters"),
        }
    }
    fn ngradient(&self, k: usize, y: &[f64], etas: &[Array1<f64>], _w: &[f64], out: &mut [f64]) {
        let n = y.len();
        let e_mu = etas[0].as_slice().expect("eta contiguous");
        let e_sig = etas[1].as_slice().expect("eta contiguous");
        let e_nu = etas[2].as_slice().expect("eta contiguous");
        match k {
            0 => {
                // mu link = log: mu = exp(eta_mu)
                // ngradient = -d(NLL)/d(eta_mu) = z/sig + nu*(z^2 - 1).
                //
                // The nu = 0 branch is the limit of that expression, which is
                // z/sig with z = log(y/mu)/sig -- the nu*(z^2 - 1) term simply
                // vanishes. It used to subtract a further 1, making the mu
                // gradient wrong by exactly 1 per observation at nu = 0. Since
                // nu's offset is 0, that is where every BCCG fit starts, so
                // the error was present from the first iteration and biased mu
                // for the whole path.
                for i in 0..n {
                    let mu = e_mu[i].exp();
                    let sig = e_sig[i].exp();
                    let nu = e_nu[i];
                    if nu.abs() > 1e-10 {
                        let z = ((nu * (y[i] / mu).ln()).exp_m1()) / (nu * sig);
                        out[i] = z / sig + nu * (z * z - 1.0);
                    } else {
                        let z = (y[i] / mu).ln() / sig;
                        out[i] = z / sig;
                    }
                }
            }
            1 => {
                for i in 0..n {
                    let sig = e_sig[i].exp();
                    let mu = e_mu[i].exp();
                    let nu = e_nu[i];
                    let z = if nu.abs() > 1e-10 {
                        ((nu * (y[i] / mu).ln()).exp_m1()) / (nu * sig)
                    } else {
                        (y[i] / mu).ln() / sig
                    };
                    // gamlss.dist::BCCG$dldd on the eta scale (log link, so
                    // multiplied by sigma): (z^2 - 1) + h / (sigma * |nu|).
                    let h = bccg_truncation_ratio(sig, nu);
                    out[i] = z * z - 1.0
                        + if nu.abs() > 1e-10 {
                            h / (sig * nu.abs())
                        } else {
                            0.0
                        };
                }
            }
            2 => {
                for i in 0..n {
                    let mu = e_mu[i].exp();
                    let sig = e_sig[i].exp();
                    let nu = e_nu[i];
                    if nu.abs() > 1e-10 {
                        let z = ((nu * (y[i] / mu).ln()).exp_m1()) / (nu * sig);
                        let l = (y[i] / mu).ln();
                        // gamlss.dist::BCCG$dldv; nu has an identity link so
                        // the eta scale is the parameter scale.
                        let h = bccg_truncation_ratio(sig, nu);
                        out[i] = (z - l / sig) * (z / nu) - l * (z * z - 1.0)
                            + nu.signum() * h / (sig * nu * nu);
                    } else {
                        // nu = 0 is a removable singularity, not a stationary
                        // point. Expanding z = (exp(nu*l) - 1)/(nu*sigma) about
                        // nu = 0 gives z -> l/sigma and dz/dnu -> l^2/(2*sigma),
                        // so dl/dnu -> l - l^3/(2*sigma^2). The truncation term
                        // vanishes here because 1/(sigma*|nu|) -> infinity.
                        //
                        // Returning 0 instead made this a fixed point: nu's
                        // offset is 0, so a zero gradient there meant nu could
                        // never move and every BCCG fit was silently its
                        // nu = 0 special case, the log-normal.
                        let l = (y[i] / mu).ln();
                        out[i] = l - l * l * l / (2.0 * sig * sig);
                    }
                }
            }
            _ => unreachable!("BCCG has 3 parameters"),
        }
    }
    fn risk(&self, y: &[f64], etas: &[Array1<f64>], w: &[f64]) -> f64 {
        let n = y.len();
        let e_mu = etas[0].as_slice().expect("eta contiguous");
        let e_sig = etas[1].as_slice().expect("eta contiguous");
        let e_nu = etas[2].as_slice().expect("eta contiguous");
        let mut s = 0.0;
        for i in 0..n {
            let mu = e_mu[i].exp(); // log link
            let sig = e_sig[i].exp();
            let nu = e_nu[i];
            let z = if nu.abs() > 1e-10 {
                ((nu * (y[i] / mu).ln()).exp_m1()) / (nu * sig)
            } else {
                (y[i] / mu).ln() / sig
            };
            // NLL = 0.5*z^2 + ln(sig) + ln(mu) - (nu-1)*ln(y/mu) + ln(2*pi)/2
            // The ln(mu) is the Jacobian for the BCCG density on log(y) scale.
            //
            // The ln(2*pi)/2 was missing. It is constant in the parameters, so
            // no gradient and no selected path changes, but it made risk(),
            // logLik() and AIC differ from `gamlss.dist::dBCCG` by
            // n*0.9189385, and — the reason it matters — made BCCG's risk
            // incomparable with every other family's, which is exactly what a
            // family-ranking scan does. Same defect class as the log(y!) that
            // was missing from NegativeBinomialLSS.
            // The truncation normalizer, which depends on sigma and nu and so
            // also appears in their gradients above.
            let truncation = if nu.abs() > 1e-10 {
                let a = 1.0 / (sig * nu.abs());
                (0.5 * libm::erfc(-a / std::f64::consts::SQRT_2)).ln()
            } else {
                0.0
            };
            s += w[i]
                * (0.5 * z * z + e_sig[i] + e_mu[i] - (nu - 1.0) * (y[i] / mu).ln()
                    + 0.5 * (2.0 * std::f64::consts::PI).ln()
                    + truncation);
        }
        s
    }
    fn response(&self, k: usize, eta: &[f64], out: &mut [f64]) {
        match k {
            0 => {
                // mu link = log
                for i in 0..eta.len() {
                    out[i] = eta[i].exp();
                }
            }
            1 => {
                for i in 0..eta.len() {
                    out[i] = eta[i].exp();
                }
            }
            2 => out.copy_from_slice(eta),
            _ => unreachable!("BCCG has 3 parameters"),
        }
    }
    fn clone_box(&self) -> Box<dyn LssFamily> {
        Box::new(BCCGLSS)
    }
}

// ─── JSU (Johnson's SU) — 4 param ──────────────────────────────

/// Johnson's SU distribution. Centered parameterization matching
/// `gamlss.dist::JSU()`:
///
///   rtau = 1/tau
///   w = exp(rtau^2)                  (w=1 if rtau < 1e-7)
///   omega = -nu * rtau
///   c = 1/sqrt(0.5*(w-1)*(w*cosh(2*omega)+1))
///   mu_c = mu + c * sigma * sqrt(w) * sinh(omega)
///   sigma_c = c * sigma
///   z = (y - mu_c) / sigma_c
///   r = -nu + asinh(z) / rtau
///
/// f(y) = tau/(c * sigma * sqrt(2*pi)) * 1/sqrt(1+z^2) * exp(-0.5*r^2)
///
/// Links: mu=identity, sigma=log, nu=identity, tau=log.
pub struct JSULSS;

impl JSULSS {
    /// Common centering computation: returns (z, r, 1+z^2, sqrt(1+z^2), c, sig_c, ww, omega).
    #[inline]
    fn centered_z_r(
        y_i: f64,
        mu: f64,
        sig: f64,
        nu: f64,
        tau: f64,
    ) -> (f64, f64, f64, f64, f64, f64, f64, f64) {
        let rtau = 1.0 / tau;
        let ww = if rtau < 1e-7 {
            1.0
        } else {
            (rtau * rtau).exp()
        };
        let omega = -nu * rtau;
        let c = if rtau < 1e-7 {
            1.0 / rtau
        } else {
            (0.5 * (ww - 1.0) * (ww * (2.0 * omega).cosh() + 1.0))
                .sqrt()
                .recip()
        };
        let sqrt_w = ww.sqrt();
        let mu_c = mu + c * sig * sqrt_w * omega.sinh();
        let sig_c = c * sig;
        let z = (y_i - mu_c) / sig_c;
        let r = -nu + z.asinh() / rtau;
        let s1 = 1.0 + z * z;
        let s1_sqrt = s1.sqrt();
        (z, r, s1, s1_sqrt, c, sig_c, ww, omega)
    }
}

impl LssFamily for JSULSS {
    fn scale_map(&self) -> Option<&'static [ParamScale]> {
        const MAP: &[ParamScale] = &[
            ParamScale::Multiply,
            ParamScale::Shift(1.0),
            ParamScale::Invariant,
            ParamScale::Invariant,
        ];
        Some(MAP)
    }

    fn name(&self) -> &'static str {
        "JSULSS"
    }
    fn n_params(&self) -> usize {
        4
    }
    fn param_names(&self) -> &'static [&'static str] {
        &["mu", "sigma", "nu", "tau"]
    }
    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        for (i, &v) in y.iter().enumerate() {
            if !v.is_finite() {
                return Err(MboostrsError::BadResponse(
                    "JSULSS".into(),
                    format!("y must be finite at {i}"),
                ));
            }
        }
        Ok(())
    }
    fn offset(&self, k: usize, y: &[f64], w: &[f64], _prior: &[f64]) -> f64 {
        let n = y.len();
        let (sw, sy, _, syxx) = (0..n).fold((0.0, 0.0, 0.0, 0.0), |(sw, sy, _, sxx), i| {
            (sw + w[i], sy + w[i] * y[i], 0.0, sxx + w[i] * y[i] * y[i])
        });
        match k {
            0 => {
                if sw > 0.0 {
                    sy / sw
                } else {
                    0.0
                }
            }
            1 => {
                let mean = sy / sw.max(1e-300);
                let var = (syxx / sw - mean * mean).max(1e-12);
                0.5 * var.ln().clamp(-10.0, 10.0)
            }
            2 => 0.0,
            3 => 0.0,
            _ => unreachable!("JSU has 4 parameters"),
        }
    }
    fn ngradient(&self, k: usize, y: &[f64], etas: &[Array1<f64>], _w: &[f64], out: &mut [f64]) {
        let n = y.len();
        let e_mu = etas[0].as_slice().expect("eta contiguous");
        let e_sig = etas[1].as_slice().expect("eta contiguous");
        let e_nu = etas[2].as_slice().expect("eta contiguous");
        let e_tau = etas[3].as_slice().expect("eta contiguous");
        match k {
            0 => {
                // mu gradient: dldm = base / (c * sig), identity link
                for i in 0..n {
                    let sig = e_sig[i].exp();
                    let tau = e_tau[i].exp();
                    let (z, r, s1, s1_sqrt, _c, sig_c, _ww, _omega) =
                        Self::centered_z_r(y[i], e_mu[i], sig, e_nu[i], tau);
                    let base = z / s1 + r * tau / s1_sqrt; // r/(rtau*s1_sqrt) = r*tau/s1_sqrt
                    out[i] = base / sig_c;
                }
            }
            1 => {
                // sigma gradient: dldd_raw - 1 (since sigma.link = "log")
                for i in 0..n {
                    let sig = e_sig[i].exp();
                    let tau = e_tau[i].exp();
                    let (z, r, s1, s1_sqrt, _c, _sig_c, ww, omega) =
                        Self::centered_z_r(y[i], e_mu[i], sig, e_nu[i], tau);
                    let base = z / s1 + r * tau / s1_sqrt;
                    let dldd_raw = (z + ww.sqrt() * omega.sinh()) * base;
                    out[i] = dldd_raw - 1.0;
                }
            }
            2 => {
                // nu gradient: dldv (identity link)
                for i in 0..n {
                    let sig = e_sig[i].exp();
                    let tau = e_tau[i].exp();
                    let (z, r, s1, s1_sqrt, _c, _sig_c, ww, omega) =
                        Self::centered_z_r(y[i], e_mu[i], sig, e_nu[i], tau);
                    let rtau = 1.0 / tau;
                    let base = z / s1 + r * tau / s1_sqrt;
                    let dlogcdv =
                        rtau * ww * (2.0 * omega).sinh() / (ww * (2.0 * omega).cosh() + 1.0);
                    let dzdv =
                        -(z + ww.sqrt() * omega.sinh()) * dlogcdv + rtau * ww.sqrt() * omega.cosh();
                    out[i] = -dlogcdv - base * dzdv + r;
                }
            }
            3 => {
                // tau gradient: dldt * tau (since tau.link = "log")
                for i in 0..n {
                    let sig = e_sig[i].exp();
                    let tau = e_tau[i].exp();
                    let (z, r, s1, s1_sqrt, _c, _sig_c, ww, omega) =
                        Self::centered_z_r(y[i], e_mu[i], sig, e_nu[i], tau);
                    let rtau = 1.0 / tau;
                    let rtau2 = rtau * rtau;
                    let base = z / s1 + r * tau / s1_sqrt;
                    let denom = ww * (2.0 * omega).cosh() + 1.0;
                    let dlogcdt = -rtau * ww * (1.0 / (ww - 1.0) + (2.0 * omega).cosh() / denom)
                        + (e_nu[i] * ww * (2.0 * omega).sinh()) / denom;
                    let dzdt = -(z + ww.sqrt() * omega.sinh()) * dlogcdt
                        - rtau * ww.sqrt() * omega.sinh()
                        + e_nu[i] * ww.sqrt() * omega.cosh();
                    let dldt_inner = -dlogcdt - 1.0 / rtau - base * dzdt + r * (r + e_nu[i]) / rtau;
                    let dldt = -dldt_inner * rtau2;
                    out[i] = dldt * tau;
                }
            }
            _ => unreachable!("JSU has 4 parameters"),
        }
    }
    fn risk(&self, y: &[f64], etas: &[Array1<f64>], w: &[f64]) -> f64 {
        let n = y.len();
        let e_mu = etas[0].as_slice().expect("eta contiguous");
        let e_sig = etas[1].as_slice().expect("eta contiguous");
        let e_nu = etas[2].as_slice().expect("eta contiguous");
        let e_tau = etas[3].as_slice().expect("eta contiguous");
        let half_log_2pi = 0.5 * (2.0 * std::f64::consts::PI).ln();
        let mut s = 0.0;
        for i in 0..n {
            let sig = e_sig[i].exp();
            let tau = e_tau[i].exp();
            let (_z, r, s1, _s1_sqrt, c, _sig_c, _ww, _omega) =
                Self::centered_z_r(y[i], e_mu[i], sig, e_nu[i], tau);
            let log_c = c.ln();
            // NLL = log(sigma) + log(c) + log(rtau) + 0.5*(1+z^2).ln() + 0.5*r^2 + const
            //     = e_sig[i] + log_c - e_tau[i] + 0.5*s1.ln() + 0.5*r*r + half_log_2pi
            // where log(rtau) = -log(tau) = -e_tau[i].
            s += w[i] * (0.5 * r * r + e_sig[i] + log_c - e_tau[i] + 0.5 * s1.ln() + half_log_2pi);
        }
        s
    }
    fn response(&self, k: usize, eta: &[f64], out: &mut [f64]) {
        match k {
            0 => out.copy_from_slice(eta),
            1 => {
                for i in 0..eta.len() {
                    out[i] = eta[i].exp();
                }
            }
            2 => out.copy_from_slice(eta),
            3 => {
                for i in 0..eta.len() {
                    out[i] = eta[i].exp();
                }
            }
            _ => unreachable!("JSU has 4 parameters"),
        }
    }
    fn clone_box(&self) -> Box<dyn LssFamily> {
        Box::new(JSULSS)
    }
}

// ─── BCT (Box-Cox t) and BCPE (Box-Cox Power Exponential) — 4 param ──

/// The shared Box-Cox standardized variate behind BCT and BCPE:
///
///   z = ((y/mu)^nu - 1) / (nu sigma),          nu != 0
///   z = log(y/mu) / sigma,                    nu == 0
///
/// `nu` and the log ratio are precomputed by the callers (the `nu == 0` case is
/// handled by a guard whose threshold matches gamlss' `ifelse(nu != 0, ...)`).
#[inline]
pub(super) fn boxcox_z(log_ratio: f64, nu: f64, sigma: f64) -> f64 {
    if nu.abs() < 1e-10 {
        log_ratio / sigma
    } else {
        ((nu * log_ratio).exp_m1()) / (nu * sigma)
    }
}

/// The log-normalizing constant `c` of the BCPE kernel, from `gamlss.dist`:
/// `log.c = 0.5*(-(2/tau) log2 + lgamma(1/tau) - lgamma(3/tau))`.
#[inline]
pub(super) fn bcpe_log_c(tau: f64) -> f64 {
    0.5 * (-(2.0 / tau) * std::f64::consts::LN_2 + lgamma(1.0 / tau) - lgamma(3.0 / tau))
}

/// BCPE power-exponential kernel: `log f_T(t)`, matching `gamlss.dist`'s
/// inner `f.T(t, log = TRUE)`.
#[inline]
pub(super) fn bcpe_log_f_t(t: f64, tau: f64, log_c: f64) -> f64 {
    tau.ln()
        - log_c
        - 0.5 * (t / log_c.exp()).abs().powf(tau)
        - (1.0 + 1.0 / tau) * std::f64::consts::LN_2
        - lgamma(1.0 / tau)
}

/// `|z/c|^power`, returning `0` at `z == 0` instead of the literal value.
///
/// BCPE's mu, nu and tau scores each multiply this factor by a term that
/// vanishes at least as fast as the power grows, so the product tends to zero
/// as `z → 0` for every `tau > 0`. Evaluated literally at `z == 0` — a
/// response exactly equal to its fitted `mu`, which the mean-valued mu offset
/// makes reachable on the first iteration — a negative `power` gives `inf`
/// and the product becomes NaN, poisoning the whole candidate scan. Returning
/// the limit keeps the score finite. For `tau < 1` the mu-score is genuinely
/// unbounded at the kernel's cusp, and zero is the subgradient there.
#[inline]
pub(super) fn bcpe_kernel_power(z: f64, log_c: f64, power: f64) -> f64 {
    if z == 0.0 {
        return 0.0;
    }
    (z / log_c.exp()).abs().powf(power)
}

/// BCPE power-exponential CDF `F_T(t)`, matching `gamlss.dist`'s `F.T`:
/// `0.5*(1 + sign(t) * pgamma(0.5 |t/c|^tau, 1/tau))`.
#[inline]
pub(super) fn bcpe_cdf_t(t: f64, tau: f64, log_c: f64) -> f64 {
    let s = 0.5 * (t / log_c.exp()).abs().powf(tau);
    let p = regularized_incomplete_gamma_p(1.0 / tau, s);
    0.5 * (1.0 + t.signum() * p)
}

/// BCT truncation ratio `h = dt(t0, tau) / pt(t0, tau)` at `t0 = 1/(sigma|nu|)`.
/// This is what gamlss' `dldd`/`dldv`/`dldt` add to fold in the truncation to
/// `y > 0`. Returns 0 at `nu == 0` (the truncation point recedes to infinity).
#[inline]
pub(super) fn bct_truncation_ratio(sigma: f64, nu: f64, tau: f64) -> f64 {
    if nu.abs() < 1e-10 {
        return 0.0;
    }
    let t0 = 1.0 / (sigma * nu.abs());
    let log_density = lgamma((tau + 1.0) / 2.0)
        - lgamma(tau / 2.0)
        - 0.5 * (tau * std::f64::consts::PI).ln()
        - (tau + 1.0) / 2.0 * (1.0 + t0 * t0 / tau).ln();
    let cdf = student_t_cdf(t0, tau);
    (log_density - cdf.clamp(1e-300, 1.0).ln()).exp()
}

/// BCPE truncation ratio `h = f_T(t0) / F_T(t0)` at `t0 = 1/(sigma|nu|)`.
#[inline]
pub(super) fn bcpe_truncation_ratio(sigma: f64, nu: f64, tau: f64) -> f64 {
    if nu.abs() < 1e-10 {
        return 0.0;
    }
    let log_c = bcpe_log_c(tau);
    let t0 = 1.0 / (sigma * nu.abs());
    let log_f = bcpe_log_f_t(t0, tau, log_c);
    let cdf = bcpe_cdf_t(t0, tau, log_c);
    (log_f - cdf.clamp(1e-300, 1.0).ln()).exp()
}

/// Box-Cox t distribution. `y > 0`. Links: mu=identity, sigma=log, nu=identity,
/// tau=log. Matches `gamlss.dist::BCT()`.
pub struct BCTLSS;

impl BCTLSS {
    /// Full per-observation negative log-density = -`dBCT(..., log = TRUE)`.
    #[inline]
    fn nll(y: f64, mu: f64, sigma: f64, nu: f64, tau: f64) -> f64 {
        let log_ratio = (y / mu).ln();
        let z = boxcox_z(log_ratio, nu, sigma);
        let kernel = lgamma((tau + 1.0) / 2.0)
            - lgamma(tau / 2.0)
            - 0.5 * tau.ln()
            - lgamma(0.5)
            - (tau + 1.0) / 2.0 * (1.0 + z * z / tau).ln();
        let log_det = (nu - 1.0) * y.ln() - nu * mu.ln() - sigma.ln();
        let truncation = if nu.abs() < 1e-10 {
            0.0
        } else {
            student_t_cdf(1.0 / (sigma * nu.abs()), tau).ln()
        };
        -(log_det + kernel - truncation)
    }
}

impl LssFamily for BCTLSS {
    fn scale_map(&self) -> Option<&'static [ParamScale]> {
        const MAP: &[ParamScale] = &[
            ParamScale::Multiply,
            ParamScale::Invariant,
            ParamScale::Invariant,
            ParamScale::Invariant,
        ];
        Some(MAP)
    }

    fn name(&self) -> &'static str {
        "BCTLSS"
    }
    fn n_params(&self) -> usize {
        4
    }
    fn param_names(&self) -> &'static [&'static str] {
        &["mu", "sigma", "nu", "tau"]
    }
    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        for (i, &v) in y.iter().enumerate() {
            if !v.is_finite() || v <= 0.0 {
                return Err(MboostrsError::BadResponse(
                    "BCTLSS".into(),
                    format!("y must be > 0 at {i}"),
                ));
            }
        }
        Ok(())
    }
    fn offset(&self, k: usize, y: &[f64], w: &[f64], _prior: &[f64]) -> f64 {
        let n = y.len();
        let sw: f64 = (0..n).map(|i| w[i]).sum();
        let mean = if sw > 0.0 {
            (0..n).map(|i| w[i] * y[i]).sum::<f64>() / sw
        } else {
            0.0
        };
        match k {
            0 => mean,
            1 => -std::f64::consts::LN_10, // sigma.initial = 0.1, log link
            2 => 0.5,                      // nu.initial = 0.5, identity link
            3 => std::f64::consts::LN_10,  // tau (= 10 on natural scale, log link)
            _ => unreachable!("BCT has 4 parameters"),
        }
    }
    fn ngradient(&self, k: usize, y: &[f64], etas: &[Array1<f64>], _w: &[f64], out: &mut [f64]) {
        let n = y.len();
        let e_mu = etas[0].as_slice().expect("eta contiguous");
        let e_sig = etas[1].as_slice().expect("eta contiguous");
        let e_nu = etas[2].as_slice().expect("eta contiguous");
        let e_tau = etas[3].as_slice().expect("eta contiguous");
        match k {
            0 => {
                for i in 0..n {
                    let mu = e_mu[i];
                    let sigma = e_sig[i].exp();
                    let nu = e_nu[i];
                    let tau = e_tau[i].exp();
                    let log_ratio = (y[i] / mu).ln();
                    let z = boxcox_z(log_ratio, nu, sigma);
                    let w = (tau + 1.0) / (tau + z * z);
                    out[i] = (w * z) / (mu * sigma) + (nu / mu) * (w * z * z - 1.0);
                }
            }
            1 => {
                for i in 0..n {
                    let mu = e_mu[i];
                    let sigma = e_sig[i].exp();
                    let nu = e_nu[i];
                    let tau = e_tau[i].exp();
                    let log_ratio = (y[i] / mu).ln();
                    let z = boxcox_z(log_ratio, nu, sigma);
                    let w = (tau + 1.0) / (tau + z * z);
                    let h = bct_truncation_ratio(sigma, nu, tau);
                    // log link: multiply the raw dldd by sigma.
                    let truncation = if nu.abs() < 1e-10 {
                        0.0
                    } else {
                        h / (sigma * sigma * nu.abs())
                    };
                    let dldd = (w * z * z - 1.0) / sigma + truncation;
                    out[i] = dldd * sigma;
                }
            }
            2 => {
                for i in 0..n {
                    let mu = e_mu[i];
                    let sigma = e_sig[i].exp();
                    let nu = e_nu[i];
                    let tau = e_tau[i].exp();
                    let log_ratio = (y[i] / mu).ln();
                    let z = boxcox_z(log_ratio, nu, sigma);
                    let w = (tau + 1.0) / (tau + z * z);
                    if nu.abs() < 1e-7 {
                        // The analytic expression has cancelling 1/nu terms.
                        // Differentiate the exact normalized density locally
                        // rather than turn the removable singularity into NaN.
                        let step = 1e-5;
                        let plus = Self::nll(y[i], mu, sigma, nu + step, tau);
                        let minus = Self::nll(y[i], mu, sigma, nu - step, tau);
                        out[i] = -(plus - minus) / (2.0 * step);
                    } else {
                        let h = bct_truncation_ratio(sigma, nu, tau);
                        out[i] = (w * z * z) / nu
                            - log_ratio * (w * z * z + (w * z) / (sigma * nu) - 1.0)
                            + nu.signum() * h / (sigma * nu * nu);
                    }
                }
            }
            3 => {
                for i in 0..n {
                    let mu = e_mu[i];
                    let sigma = e_sig[i].exp();
                    let nu = e_nu[i];
                    let tau = e_tau[i].exp();
                    let log_ratio = (y[i] / mu).ln();
                    let z = boxcox_z(log_ratio, nu, sigma);
                    let w = (tau + 1.0) / (tau + z * z);
                    // j = d/dt log pt(t0, tau), by gamlss' forward
                    // difference at h=0.01.
                    let t0 = if nu.abs() < 1e-10 {
                        f64::INFINITY
                    } else {
                        1.0 / (sigma * nu.abs())
                    };
                    let j = if t0.is_finite() {
                        let inner = |df: f64| -> f64 { student_t_cdf(t0, df).ln() };
                        (inner(tau + 0.01) - inner(tau)) / 0.01
                    } else {
                        0.0
                    };
                    let dldt = -0.5 * (1.0 + z * z / tau).ln()
                        + (w * z * z) / (2.0 * tau)
                        + 0.5 * digamma((tau + 1.0) / 2.0)
                        - 0.5 * digamma(tau / 2.0)
                        - 1.0 / (2.0 * tau)
                        - j;
                    // tau log link: multiply by tau.
                    out[i] = dldt * tau;
                }
            }
            _ => unreachable!("BCT has 4 parameters"),
        }
    }
    fn risk(&self, y: &[f64], etas: &[Array1<f64>], w: &[f64]) -> f64 {
        let n = y.len();
        let e_mu = etas[0].as_slice().expect("eta contiguous");
        let e_sig = etas[1].as_slice().expect("eta contiguous");
        let e_nu = etas[2].as_slice().expect("eta contiguous");
        let e_tau = etas[3].as_slice().expect("eta contiguous");
        let mut s = 0.0;
        for i in 0..n {
            s += w[i] * Self::nll(y[i], e_mu[i], e_sig[i].exp(), e_nu[i], e_tau[i].exp());
        }
        s
    }
    fn response(&self, k: usize, eta: &[f64], out: &mut [f64]) {
        match k {
            0 => out.copy_from_slice(eta),
            1 => {
                for i in 0..eta.len() {
                    out[i] = eta[i].exp();
                }
            }
            2 => out.copy_from_slice(eta),
            3 => {
                for i in 0..eta.len() {
                    out[i] = eta[i].exp();
                }
            }
            _ => unreachable!("BCT has 4 parameters"),
        }
    }
    fn clone_box(&self) -> Box<dyn LssFamily> {
        Box::new(BCTLSS)
    }
}

/// Box-Cox Power Exponential distribution. `y > 0`. Links: mu=identity,
/// sigma=log, nu=identity, tau=log. Matches `gamlss.dist::BCPE()`.
pub struct BCPELSS;

impl BCPELSS {
    /// Full per-observation negative log-density = -`dBCPE(..., log = TRUE)`.
    #[inline]
    fn nll(y: f64, mu: f64, sigma: f64, nu: f64, tau: f64) -> f64 {
        let log_c = bcpe_log_c(tau);
        let log_ratio = (y / mu).ln();
        let z = boxcox_z(log_ratio, nu, sigma);
        let log_f_z = bcpe_log_f_t(z, tau, log_c);
        let truncation = if nu.abs() < 1e-10 {
            0.0
        } else {
            bcpe_cdf_t(1.0 / (sigma * nu.abs()), tau, log_c).ln()
        };
        let log_det = (nu - 1.0) * y.ln() - nu * mu.ln() - sigma.ln();
        -(log_det + log_f_z - truncation)
    }
}

impl LssFamily for BCPELSS {
    fn scale_map(&self) -> Option<&'static [ParamScale]> {
        const MAP: &[ParamScale] = &[
            ParamScale::Multiply,
            ParamScale::Invariant,
            ParamScale::Invariant,
            ParamScale::Invariant,
        ];
        Some(MAP)
    }

    fn name(&self) -> &'static str {
        "BCPELSS"
    }
    fn n_params(&self) -> usize {
        4
    }
    fn param_names(&self) -> &'static [&'static str] {
        &["mu", "sigma", "nu", "tau"]
    }
    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        for (i, &v) in y.iter().enumerate() {
            if !v.is_finite() || v <= 0.0 {
                return Err(MboostrsError::BadResponse(
                    "BCPELSS".into(),
                    format!("y must be > 0 at {i}"),
                ));
            }
        }
        Ok(())
    }
    fn offset(&self, k: usize, y: &[f64], w: &[f64], _prior: &[f64]) -> f64 {
        let n = y.len();
        let sw: f64 = (0..n).map(|i| w[i]).sum();
        let mean = if sw > 0.0 {
            (0..n).map(|i| w[i] * y[i]).sum::<f64>() / sw
        } else {
            0.0
        };
        match k {
            0 => mean,
            1 => -std::f64::consts::LN_10, // sigma.initial = 0.1, log link
            2 => 1.0,                      // nu.initial = 1, identity link
            3 => std::f64::consts::LN_2,   // tau (= 2 on natural scale, log link)
            _ => unreachable!("BCPE has 4 parameters"),
        }
    }
    fn ngradient(&self, k: usize, y: &[f64], etas: &[Array1<f64>], _w: &[f64], out: &mut [f64]) {
        let n = y.len();
        let e_mu = etas[0].as_slice().expect("eta contiguous");
        let e_sig = etas[1].as_slice().expect("eta contiguous");
        let e_nu = etas[2].as_slice().expect("eta contiguous");
        let e_tau = etas[3].as_slice().expect("eta contiguous");
        match k {
            0 => {
                for i in 0..n {
                    let mu = e_mu[i];
                    let sigma = e_sig[i].exp();
                    let nu = e_nu[i];
                    let tau = e_tau[i].exp();
                    let log_c = bcpe_log_c(tau);
                    let c2 = (log_c.exp()).powi(2);
                    let log_ratio = (y[i] / mu).ln();
                    let z = boxcox_z(log_ratio, nu, sigma);
                    out[i] = (tau / (2.0 * mu * sigma * c2))
                        * (z + sigma * nu * z * z)
                        * bcpe_kernel_power(z, log_c, tau - 2.0)
                        - nu / mu;
                }
            }
            1 => {
                for i in 0..n {
                    let mu = e_mu[i];
                    let sigma = e_sig[i].exp();
                    let nu = e_nu[i];
                    let tau = e_tau[i].exp();
                    let log_c = bcpe_log_c(tau);
                    let log_ratio = (y[i] / mu).ln();
                    let z = boxcox_z(log_ratio, nu, sigma);
                    let h = bcpe_truncation_ratio(sigma, nu, tau);
                    let truncation = if nu.abs() < 1e-10 {
                        0.0
                    } else {
                        h / (sigma * sigma * nu.abs())
                    };
                    let dldd = (1.0 / sigma)
                        * ((tau / 2.0) * (z / log_c.exp()).abs().powf(tau) - 1.0)
                        + truncation;
                    out[i] = dldd * sigma;
                }
            }
            2 => {
                for i in 0..n {
                    let mu = e_mu[i];
                    let sigma = e_sig[i].exp();
                    let nu = e_nu[i];
                    let tau = e_tau[i].exp();
                    let log_c = bcpe_log_c(tau);
                    let c2 = log_c.exp().powi(2);
                    let log_ratio = (y[i] / mu).ln();
                    let z = boxcox_z(log_ratio, nu, sigma);
                    if nu.abs() < 1e-7 {
                        let step = 1e-5;
                        let plus = Self::nll(y[i], mu, sigma, nu + step, tau);
                        let minus = Self::nll(y[i], mu, sigma, nu - step, tau);
                        out[i] = -(plus - minus) / (2.0 * step);
                    } else {
                        let h = bcpe_truncation_ratio(sigma, nu, tau);
                        out[i] = -(tau / (2.0 * nu * c2))
                            * bcpe_kernel_power(z, log_c, tau - 2.0)
                            * z
                            * ((nu * z + 1.0 / sigma) * log_ratio - z)
                            + log_ratio
                            + nu.signum() * h / (sigma * nu * nu);
                    }
                }
            }
            3 => {
                for i in 0..n {
                    let mu = e_mu[i];
                    let sigma = e_sig[i].exp();
                    let nu = e_nu[i];
                    let tau = e_tau[i].exp();
                    let log_c = bcpe_log_c(tau);
                    let log_ratio = (y[i] / mu).ln();
                    let z = boxcox_z(log_ratio, nu, sigma);
                    let dlogc_dt = (1.0 / (2.0 * tau * tau))
                        * (2.0 * std::f64::consts::LN_2 - digamma(1.0 / tau)
                            + 3.0 * digamma(3.0 / tau));
                    // j = d/dt log F_T(t0, tau) at h=0.001 (matches gamlss dldt).
                    let t0 = if nu.abs() < 1e-10 {
                        f64::INFINITY
                    } else {
                        1.0 / (sigma * nu.abs())
                    };
                    let j = if t0.is_finite() {
                        let inner = |df: f64| -> f64 { bcpe_cdf_t(t0, df, bcpe_log_c(df)).ln() };
                        (inner(tau + 0.001) - inner(tau)) / 0.001
                    } else {
                        0.0
                    };
                    // `|z/c|^tau · ln|z/c|` also tends to zero as z → 0, but
                    // literally it is `0 · -inf`; take the limit there too.
                    let scaled_power = bcpe_kernel_power(z, log_c, tau);
                    let log_term = if z == 0.0 {
                        0.0
                    } else {
                        -0.5 * (z / log_c.exp()).abs().ln() * scaled_power
                    };
                    let dldt = 1.0 / tau
                        + log_term
                        + (1.0 / (tau * tau)) * (std::f64::consts::LN_2 + digamma(1.0 / tau))
                        + ((tau / 2.0) * scaled_power - 1.0) * dlogc_dt
                        - j;
                    out[i] = dldt * tau;
                }
            }
            _ => unreachable!("BCPE has 4 parameters"),
        }
    }
    fn risk(&self, y: &[f64], etas: &[Array1<f64>], w: &[f64]) -> f64 {
        let n = y.len();
        let e_mu = etas[0].as_slice().expect("eta contiguous");
        let e_sig = etas[1].as_slice().expect("eta contiguous");
        let e_nu = etas[2].as_slice().expect("eta contiguous");
        let e_tau = etas[3].as_slice().expect("eta contiguous");
        let mut s = 0.0;
        for i in 0..n {
            s += w[i] * Self::nll(y[i], e_mu[i], e_sig[i].exp(), e_nu[i], e_tau[i].exp());
        }
        s
    }
    fn response(&self, k: usize, eta: &[f64], out: &mut [f64]) {
        match k {
            0 => out.copy_from_slice(eta),
            1 => {
                for i in 0..eta.len() {
                    out[i] = eta[i].exp();
                }
            }
            2 => out.copy_from_slice(eta),
            3 => {
                for i in 0..eta.len() {
                    out[i] = eta[i].exp();
                }
            }
            _ => unreachable!("BCPE has 4 parameters"),
        }
    }
    fn clone_box(&self) -> Box<dyn LssFamily> {
        Box::new(BCPELSS)
    }
}

// ─── InverseGaussianLSS (IG) — 2 param, log/log links ──────────

/// Inverse Gaussian distribution (mean mu, log link; dispersion sigma, log
/// link). y > 0. Matches `gamlss.dist::IG()`.
///
/// Density `f(y) = (2*pi*sigma^2*y^3)^(-1/2) * exp(-(y-mu)^2 / (2*mu^2*sigma^2*y))`,
/// so the negative log-likelihood is
/// `0.5*log(2*pi) + log(sigma) + 1.5*log(y) + (y-mu)^2/(2*mu^2*sigma^2*y)`.
///
/// The gradients below are `gamlss.dist::IG()`'s `dldm`/`dldd` with the log
/// link's chain rule applied (`d/deta = d/dparam * param`), verified against
/// the oracle before implementation.
pub struct InverseGaussianLSS;

impl LssFamily for InverseGaussianLSS {
    fn scale_map(&self) -> Option<&'static [ParamScale]> {
        const MAP: &[ParamScale] = &[ParamScale::Shift(1.0), ParamScale::Shift(-0.5)];
        Some(MAP)
    }

    fn name(&self) -> &'static str {
        "InverseGaussianLSS"
    }
    fn n_params(&self) -> usize {
        2
    }
    fn param_names(&self) -> &'static [&'static str] {
        &["mu", "sigma"]
    }
    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        for (i, &v) in y.iter().enumerate() {
            if !v.is_finite() || v <= 0.0 {
                return Err(MboostrsError::BadResponse(
                    "InverseGaussianLSS".into(),
                    format!("y must be finite and strictly positive at {i}, got {v}"),
                ));
            }
        }
        Ok(())
    }
    fn offset(&self, k: usize, y: &[f64], w: &[f64], _prior: &[f64]) -> f64 {
        let n = y.len();
        let (sw, sy) = (0..n).fold((0.0, 0.0), |(s, sy), i| (s + w[i], sy + w[i] * y[i]));
        let mean = if sw > 0.0 { sy / sw } else { 1.0 };
        match k {
            // mu: log link on the weighted mean.
            0 => mean.max(1e-10).ln().clamp(-10.0, 10.0),
            // sigma: log link. The IG dispersion satisfies Var = sigma^2 * mu^3,
            // so a moment start is sigma = sqrt(Var / mu^3).
            1 => {
                let syy = (0..n).fold(0.0, |acc, i| acc + w[i] * y[i] * y[i]);
                let var = (syy / sw.max(1e-300) - mean * mean).max(1e-12);
                let sigma = (var / mean.max(1e-10).powi(3)).max(1e-12).sqrt();
                sigma.ln().clamp(-10.0, 10.0)
            }
            _ => unreachable!("InverseGaussianLSS has 2 parameters"),
        }
    }
    fn ngradient(&self, k: usize, y: &[f64], etas: &[Array1<f64>], _w: &[f64], out: &mut [f64]) {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        for i in 0..y.len() {
            let mu = eta_mu[i].exp();
            let sigma = eta_sigma[i].exp();
            match k {
                // dl/dmu * mu = (y-mu)/(sigma^2 * mu^2)
                0 => out[i] = (y[i] - mu) / (sigma * sigma * mu * mu),
                // dl/dsigma * sigma = -1 + (y-mu)^2/(y * sigma^2 * mu^2)
                1 => {
                    let d = y[i] - mu;
                    out[i] = -1.0 + d * d / (y[i] * sigma * sigma * mu * mu);
                }
                _ => unreachable!("InverseGaussianLSS has 2 parameters"),
            }
        }
    }
    fn risk(&self, y: &[f64], etas: &[Array1<f64>], w: &[f64]) -> f64 {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        let half_log_two_pi = 0.5 * (2.0 * std::f64::consts::PI).ln();
        let mut total = 0.0;
        for i in 0..y.len() {
            let mu = eta_mu[i].exp();
            let sigma = eta_sigma[i].exp();
            let d = y[i] - mu;
            total += w[i]
                * (half_log_two_pi
                    + eta_sigma[i]
                    + 1.5 * y[i].ln()
                    + d * d / (2.0 * mu * mu * sigma * sigma * y[i]));
        }
        total
    }
    fn response(&self, k: usize, eta: &[f64], out: &mut [f64]) {
        match k {
            0 | 1 => {
                for i in 0..eta.len() {
                    out[i] = eta[i].exp();
                }
            }
            _ => unreachable!("InverseGaussianLSS has 2 parameters"),
        }
    }
    fn clone_box(&self) -> Box<dyn LssFamily> {
        Box::new(InverseGaussianLSS)
    }
}

// ─── LogisticLSS (LO) — 2 param, identity/log links ────────────

/// Logistic distribution (location mu, identity link; scale sigma, log link).
/// y ∈ ℝ. Matches `gamlss.dist::LO()`.
///
/// With `z = (y - mu)/sigma`, the negative log-likelihood is
/// `z + log(sigma) + 2*log(1 + exp(-z))`. Heavier-tailed than Gaussian at the
/// same scale, so it is the usual drop-in when residuals are mildly
/// outlier-prone but symmetric.
pub struct LogisticLSS;

impl LssFamily for LogisticLSS {
    fn scale_map(&self) -> Option<&'static [ParamScale]> {
        const MAP: &[ParamScale] = &[ParamScale::Multiply, ParamScale::Shift(1.0)];
        Some(MAP)
    }

    fn name(&self) -> &'static str {
        "LogisticLSS"
    }
    fn n_params(&self) -> usize {
        2
    }
    fn param_names(&self) -> &'static [&'static str] {
        &["mu", "sigma"]
    }
    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        for (i, &v) in y.iter().enumerate() {
            if !v.is_finite() {
                return Err(MboostrsError::BadResponse(
                    "LogisticLSS".into(),
                    format!("y must be finite at {i}"),
                ));
            }
        }
        Ok(())
    }
    fn offset(&self, k: usize, y: &[f64], w: &[f64], _prior: &[f64]) -> f64 {
        let n = y.len();
        let (sw, sy, syy) = (0..n).fold((0.0, 0.0, 0.0), |(s, sy, syy), i| {
            (s + w[i], sy + w[i] * y[i], syy + w[i] * y[i] * y[i])
        });
        let mean = if sw > 0.0 { sy / sw } else { 0.0 };
        match k {
            0 => mean,
            1 => {
                // Var = sigma^2 * pi^2 / 3, so sigma = sd * sqrt(3)/pi.
                let var = (syy / sw.max(1e-300) - mean * mean).max(1e-12);
                let sigma = var.sqrt() * 3.0f64.sqrt() / std::f64::consts::PI;
                sigma.max(1e-12).ln().clamp(-10.0, 10.0)
            }
            _ => unreachable!("LogisticLSS has 2 parameters"),
        }
    }
    fn ngradient(&self, k: usize, y: &[f64], etas: &[Array1<f64>], _w: &[f64], out: &mut [f64]) {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        for i in 0..y.len() {
            let sigma = eta_sigma[i].exp();
            let z = (y[i] - eta_mu[i]) / sigma;
            // exp(z)/(1+exp(z)) = logistic(z), computed stably for large |z|.
            let logistic = if z >= 0.0 {
                1.0 / (1.0 + (-z).exp())
            } else {
                let e = z.exp();
                e / (1.0 + e)
            };
            match k {
                // dl/dmu = (1/sigma)*(exp(z)-1)/(1+exp(z)) = (2*logistic - 1)/sigma
                0 => out[i] = (2.0 * logistic - 1.0) / sigma,
                // dl/dsigma * sigma = -1 - z + 2*z*logistic
                1 => out[i] = -1.0 - z + 2.0 * z * logistic,
                _ => unreachable!("LogisticLSS has 2 parameters"),
            }
        }
    }
    fn risk(&self, y: &[f64], etas: &[Array1<f64>], w: &[f64]) -> f64 {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        let mut total = 0.0;
        for i in 0..y.len() {
            let z = (y[i] - eta_mu[i]) / eta_sigma[i].exp();
            // log(1 + exp(-z)) via the numerically stable softplus.
            let softplus = if -z > 0.0 {
                -z + (1.0 + z.exp()).ln()
            } else {
                (1.0 + (-z).exp()).ln()
            };
            total += w[i] * (z + eta_sigma[i] + 2.0 * softplus);
        }
        total
    }
    fn response(&self, k: usize, eta: &[f64], out: &mut [f64]) {
        match k {
            0 => out.copy_from_slice(eta),
            1 => {
                for i in 0..eta.len() {
                    out[i] = eta[i].exp();
                }
            }
            _ => unreachable!("LogisticLSS has 2 parameters"),
        }
    }
    fn clone_box(&self) -> Box<dyn LssFamily> {
        Box::new(LogisticLSS)
    }
}

// ─── ZeroAdjustedGammaLSS (ZAGA) — 3 param, log/log/logit ──────

/// Zero-adjusted Gamma (mu log, sigma log, nu logit). `y >= 0`, with `nu` the
/// probability of an exact zero. Matches `gamlss.dist::ZAGA()`.
///
/// This family models semicontinuous responses such as claim amounts, rainfall,
/// or household spending, which can be zero or positive and right-skewed. The
/// parameter `nu` models the probability of an exact zero; `mu` and `sigma`
/// model the positive response distribution.
///
/// Density is a two-part mixture:
/// `f(0) = nu`, and `f(y) = (1 - nu) * Gamma(y; mu, sigma)` for `y > 0`.
///
/// The `nu` gradient collapses under the logit link: `dl/dnu` is `1/nu` at a
/// zero and `-1/(1-nu)` otherwise, and multiplying by `dnu/deta = nu(1-nu)`
/// leaves `1 - nu` and `-nu`. Verified against `gamlss.dist` before
/// implementation, along with the other two and the risk.
pub struct ZeroAdjustedGammaLSS;

impl LssFamily for ZeroAdjustedGammaLSS {
    fn scale_map(&self) -> Option<&'static [ParamScale]> {
        const MAP: &[ParamScale] = &[
            ParamScale::Shift(1.0),
            ParamScale::Invariant,
            ParamScale::Invariant,
        ];
        Some(MAP)
    }

    fn scale_jacobian_weight(&self, y: &[f64], w: &[f64]) -> f64 {
        // A zero is a probability mass, not a density on y.
        y.iter()
            .zip(w)
            .filter(|(v, _)| **v > 0.0)
            .map(|(_, wi)| wi)
            .sum()
    }

    fn name(&self) -> &'static str {
        "ZeroAdjustedGammaLSS"
    }
    fn n_params(&self) -> usize {
        3
    }
    fn param_names(&self) -> &'static [&'static str] {
        &["mu", "sigma", "nu"]
    }
    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        let mut positives = 0usize;
        for (i, &v) in y.iter().enumerate() {
            if !v.is_finite() || v < 0.0 {
                return Err(MboostrsError::BadResponse(
                    "ZeroAdjustedGammaLSS".into(),
                    format!("y must be finite and non-negative at {i}, got {v}"),
                ));
            }
            if v > 0.0 {
                positives += 1;
            }
        }
        // The Gamma part is estimated from the positives alone, so an all-zero
        // response leaves mu and sigma unidentified. Say so here rather than
        // returning a fit whose two Gamma parameters mean nothing.
        if positives == 0 {
            return Err(MboostrsError::BadResponse(
                "ZeroAdjustedGammaLSS".into(),
                "y is all zeros; the Gamma part has no data to estimate mu and \
                 sigma from"
                    .to_string(),
            ));
        }
        Ok(())
    }
    fn offset(&self, k: usize, y: &[f64], w: &[f64], _prior: &[f64]) -> f64 {
        let n = y.len();
        match k {
            // mu, sigma: the Gamma part, from the positive observations only.
            0 | 1 => {
                let (sw, sy, syy) = (0..n)
                    .filter(|&i| y[i] > 0.0)
                    .fold((0.0, 0.0, 0.0), |(s, sy, syy), i| {
                        (s + w[i], sy + w[i] * y[i], syy + w[i] * y[i] * y[i])
                    });
                let mean = if sw > 0.0 { sy / sw } else { 1.0 };
                if k == 0 {
                    mean.max(1e-10).ln().clamp(-10.0, 10.0)
                } else {
                    // Gamma: Var = sigma^2 * mu^2, so sigma = sd / mean.
                    let var = (syy / sw.max(1e-300) - mean * mean).max(1e-12);
                    (var.sqrt() / mean.max(1e-10))
                        .max(1e-12)
                        .ln()
                        .clamp(-10.0, 10.0)
                }
            }
            // nu: logit of the observed zero proportion.
            2 => {
                let (sw, sz) = (0..n).fold((0.0, 0.0), |(s, sz), i| {
                    (s + w[i], sz + if y[i] == 0.0 { w[i] } else { 0.0 })
                });
                let p = (sz / sw.max(1e-300)).clamp(1e-6, 1.0 - 1e-6);
                (p / (1.0 - p)).ln().clamp(-10.0, 10.0)
            }
            _ => unreachable!("ZeroAdjustedGammaLSS has 3 parameters"),
        }
    }
    fn ngradient(&self, k: usize, y: &[f64], etas: &[Array1<f64>], _w: &[f64], out: &mut [f64]) {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        let eta_nu = etas[2].as_slice().expect("eta contiguous");
        for i in 0..y.len() {
            let is_zero = y[i] == 0.0;
            match k {
                0 => {
                    if is_zero {
                        out[i] = 0.0;
                    } else {
                        let mu = eta_mu[i].exp();
                        let sigma = eta_sigma[i].exp();
                        out[i] = (y[i] - mu) / (sigma * sigma * mu);
                    }
                }
                1 => {
                    if is_zero {
                        out[i] = 0.0;
                    } else {
                        let mu = eta_mu[i].exp();
                        let s2 = (2.0 * eta_sigma[i]).exp();
                        out[i] = (2.0 / s2)
                            * (y[i] / mu - y[i].ln() + mu.ln() + s2.ln() - 1.0 + digamma(1.0 / s2));
                    }
                }
                2 => {
                    // logistic(eta) computed stably.
                    let e = eta_nu[i];
                    let nu = if e >= 0.0 {
                        1.0 / (1.0 + (-e).exp())
                    } else {
                        let t = e.exp();
                        t / (1.0 + t)
                    };
                    out[i] = if is_zero { 1.0 - nu } else { -nu };
                }
                _ => unreachable!("ZeroAdjustedGammaLSS has 3 parameters"),
            }
        }
    }
    fn risk(&self, y: &[f64], etas: &[Array1<f64>], w: &[f64]) -> f64 {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        let eta_nu = etas[2].as_slice().expect("eta contiguous");
        let mut total = 0.0;
        for i in 0..y.len() {
            let e = eta_nu[i];
            // log(nu) and log(1 - nu) from the logit eta, without forming nu
            // first: log(nu) = -softplus(-eta), log(1-nu) = -softplus(eta).
            let softplus = |v: f64| {
                if v > 0.0 {
                    v + (1.0 + (-v).exp()).ln()
                } else {
                    (1.0 + v.exp()).ln()
                }
            };
            if y[i] == 0.0 {
                total += w[i] * softplus(-e);
            } else {
                let mu = eta_mu[i].exp();
                let s2 = (2.0 * eta_sigma[i]).exp();
                let inv = 1.0 / s2;
                let log_gamma_density =
                    inv * (y[i] / (mu * s2)).ln() - y[i] / (mu * s2) - y[i].ln() - lgamma(inv);
                total += w[i] * (softplus(e) - log_gamma_density);
            }
        }
        total
    }
    fn response(&self, k: usize, eta: &[f64], out: &mut [f64]) {
        match k {
            0 | 1 => {
                for i in 0..eta.len() {
                    out[i] = eta[i].exp();
                }
            }
            2 => {
                for i in 0..eta.len() {
                    let e = eta[i];
                    out[i] = if e >= 0.0 {
                        1.0 / (1.0 + (-e).exp())
                    } else {
                        let t = e.exp();
                        t / (1.0 + t)
                    };
                }
            }
            _ => unreachable!("ZeroAdjustedGammaLSS has 3 parameters"),
        }
    }
    fn clone_box(&self) -> Box<dyn LssFamily> {
        Box::new(ZeroAdjustedGammaLSS)
    }
}
