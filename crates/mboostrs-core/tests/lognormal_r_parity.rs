//! Golden-value parity test for the `Lognormal` AFT family (bug LOGN1, see
//! `.remember/performance-enhancement-opportunities.md` §4e).
//!
//! The censored branch of `Lognormal` is currently wrong in both the loss and
//! the negative gradient (gradient off by exactly −1/σ from R; loss has
//! spurious η-dependent terms; `log_phi_neg_eta` asymptotic branches wrong).
//! This test pins the CORRECT values, generated 2026-06-11 from R
//! `mboost::Lognormal()` (mboost 2.9-11, R 4.5):
//!
//! - risk: `sum(environment(fam@ngradient)$plloss(sigma, cbind(time, event), f))`
//! - ngradient (fam@ngradient with sigma held fixed, deparse-verified):
//!   `(event * eta + (1 - event) * dnorm(eta)/(1 - pnorm(eta)))/sigma`,
//!   `eta = (log(time) - f)/sigma`
//! - offset: `fam@offset(cbind(time, event), w = 1)`
//!
//! It FAILS until LOGN1 is fixed — that is its purpose. Do not adjust the
//! golden values or tolerances to make it pass; fix `families.rs::Lognormal`.

use mboostrs_core::{Family, Lognormal};
use ndarray::Array1;

const TIME: [f64; 10] = [0.5, 1.2, 2.0, 3.5, 0.8, 4.1, 1.7, 2.9, 2.0, 0.5];
const EVENT: [f64; 10] = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
const F: [f64; 10] = [0.2, -0.5, 1.0, 0.4, 1.5, 0.9, -1.2, 2.0, -3.5, 4.0];
const SIGMA: f64 = 1.2;

// η = (log t − f)/σ spans [−3.91, 3.49]: covers censored observations in both
// tails as well as the central range.
const R_RISK: f64 = 17.578039488364915;
const R_NGRADIENT: [f64; 10] = [
    -0.620_241_097_611_073_3,
    0.993_031_732_015_072_7,
    -0.21309223572226024,
    0.592_196_505_899_561_2,
    0.12825564159365524,
    0.354_852_065_076_570_9,
    1.574_711_185_655_551,
    0.31371691503265164,
    3.1216717432869716,
    0.00015861947283044554,
];
const R_OFFSET: f64 = 1.2025132779925765;

/// Concatenated response [time..., event...] as the Rust family expects.
fn y_concat() -> Array1<f64> {
    let mut y = Vec::with_capacity(2 * TIME.len());
    y.extend_from_slice(&TIME);
    y.extend_from_slice(&EVENT);
    Array1::from_vec(y)
}

#[test]
fn lognormal_risk_matches_r() {
    let mut fam = Lognormal::new();
    fam.set_nuisance(SIGMA);
    let y = y_concat();
    let f = Array1::from_vec(F.to_vec());
    let w = Array1::from_elem(TIME.len(), 1.0);
    let risk = fam.risk(y.view(), f.view(), w.view());
    assert!(
        (risk - R_RISK).abs() < 1e-10,
        "Lognormal risk at sigma={SIGMA}: got {risk}, R gives {R_RISK} \
         (diff {:.3e}). Censored plloss must be -log(Phi(-eta)); event plloss \
         must include the 0.5*ln(2*pi) of log(dnorm).",
        (risk - R_RISK).abs()
    );
}

#[test]
fn lognormal_ngradient_matches_r() {
    let mut fam = Lognormal::new();
    fam.set_nuisance(SIGMA);
    let y = y_concat();
    let f = Array1::from_vec(F.to_vec());
    let w = Array1::from_elem(TIME.len(), 1.0);
    let mut out = Array1::zeros(TIME.len());
    fam.negative_gradient(y.view(), f.view(), w.view(), out.view_mut());
    for i in 0..TIME.len() {
        assert!(
            (out[i] - R_NGRADIENT[i]).abs() < 1e-10,
            "Lognormal ngradient obs {i} (event={}, eta={:.3}): got {}, R gives {} \
             (diff {:.3e}). Censored gradient must be phi(eta)/(sigma*Phi(-eta)), \
             with no extra -1/sigma term.",
            EVENT[i],
            (TIME[i].ln() - F[i]) / SIGMA,
            out[i],
            R_NGRADIENT[i],
            (out[i] - R_NGRADIENT[i]).abs()
        );
    }
}

#[test]
fn lognormal_offset_matches_r() {
    let fam = Lognormal::new();
    let y = y_concat();
    let w = Array1::from_elem(TIME.len(), 1.0);
    let off = fam.offset(y.view(), w.view());
    // The lognormal NLL is smooth in f (no kink), so Brent converges well
    // inside R's optimize() tolerance; 1e-6 separates a correct objective
    // from the current wrong one (which lands ~0.1 away on this data).
    assert!(
        (off - R_OFFSET).abs() < 1e-6,
        "Lognormal offset: got {off}, R gives {R_OFFSET} (diff {:.3e}). \
         The offset Brent search must minimize the correct plloss.",
        (off - R_OFFSET).abs()
    );
}
