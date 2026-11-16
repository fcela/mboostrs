//! Cross-language artifact parity: read the JSON fixtures in
//! `mboostrs/tests/fixtures/` (written by the Python `save_model` writer, the
//! same fixtures the R reader's `test_load_model.R` scores against) and
//! verify the Rust reader reproduces Python's predictions bit-for-bit.
//!
//! `expected.json` holds `mb.predict(..., type=...)` output computed by
//! Python for the shared `newdata`; `glm_<family>.json` are the artifacts.
//! Run with `cargo test -p mboostrs-core --test artifact_predict_parity`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use mboostrs_core::{Artifact, NewData, Scale};
use serde_json::Value;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../mboostrs/tests/fixtures")
}

fn load_expected() -> Value {
    let text = std::fs::read_to_string(fixtures_dir().join("expected.json"))
        .expect("expected.json must exist alongside the artifact fixtures");
    serde_json::from_str(&text).expect("expected.json must be valid JSON")
}

fn newdata_from(expected: &Value) -> NewData {
    let cols = expected["newdata"].as_object().unwrap();
    let mut newdata = NewData::new();
    for (name, values) in cols {
        let values: Vec<f64> = values
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        newdata = newdata.with_column(name.clone(), values);
    }
    newdata
}

fn expected_vec(expected: &Value, family: &str, scale: &str) -> Vec<f64> {
    expected["expected"][family][scale]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}

/// Bit-identity check: the artifact's predictions must equal Python's exactly
/// (max |diff| == 0.0), not merely within a tolerance — this is the same bar
/// the R reader was held to (`mboostrs/tests/test_load_model.R`).
#[test]
fn glm_families_predict_bit_identically_to_python() {
    let expected = load_expected();
    let newdata = newdata_from(&expected);

    let cases: BTreeMap<&str, &str> = BTreeMap::from([
        ("gaussian", "glm_gaussian.json"),
        ("poisson", "glm_poisson.json"),
        ("binomial", "glm_binomial.json"),
    ]);

    for (family, file) in cases {
        let artifact = Artifact::load(fixtures_dir().join(file))
            .unwrap_or_else(|e| panic!("failed to load {file}: {e}"));
        assert_eq!(artifact.model_class, "GLMBoostModel");

        for (scale, scale_key, want) in [
            (Scale::Link, "link", expected_vec(&expected, family, "link")),
            (
                Scale::Response,
                "response",
                expected_vec(&expected, family, "response"),
            ),
        ] {
            let got = artifact
                .predict(&newdata, scale)
                .unwrap_or_else(|e| panic!("{family} {scale_key} predict failed: {e}"));
            assert_eq!(
                got.len(),
                want.len(),
                "{family} {scale_key}: length mismatch"
            );
            let maxdiff = got
                .iter()
                .zip(&want)
                .map(|(g, w)| (g - w).abs())
                .fold(0.0_f64, f64::max);
            assert_eq!(
                got, want,
                "{family} {scale_key}: not bit-identical to Python (maxdiff={maxdiff:e})"
            );
        }
    }
}

/// A spline (GAMBoost) artifact loads for inspection but is refused at
/// predict time with a message pointing at the Python reader.
#[test]
fn gamboost_artifact_loads_but_predict_is_refused() {
    let expected = load_expected();
    let newdata = newdata_from(&expected);

    let artifact = Artifact::load(fixtures_dir().join("gam_spline.json"))
        .expect("gam_spline.json must load for inspection");
    assert_eq!(artifact.model_class, "GAMBoostModel");
    assert_eq!(artifact.formula().unwrap(), "yg ~ bbs(x, df=4)");

    let err = artifact
        .predict(&newdata, Scale::Link)
        .expect_err("GAMBoost artifacts must not be scorable by this reader");
    let message = err.to_string();
    assert!(
        message.contains("linear glmboost models only"),
        "unexpected error message: {message}"
    );
    assert!(
        message.contains("mboostrs.load_model"),
        "error should point at the Python reader: {message}"
    );
}

/// An artifact with an unsupported schema version is rejected explicitly
/// rather than silently misread.
#[test]
fn unsupported_schema_version_is_rejected() {
    let text = std::fs::read_to_string(fixtures_dir().join("glm_gaussian.json")).unwrap();
    let mut document: Value = serde_json::from_str(&text).unwrap();
    document["schema_version"] = Value::from(999);
    let tampered = serde_json::to_string(&document).unwrap();

    let err = Artifact::from_json_str(&tampered)
        .expect_err("an artifact from an unknown schema version must be rejected");
    assert!(
        err.to_string().contains("schema version"),
        "unexpected error message: {err}"
    );
}

/// Loading an artifact also exposes the execution report as typed data
/// (reusing the engine's own `FitReport`), not just the fields predict()
/// needs.
#[test]
fn execution_report_decodes_to_typed_fit_report() {
    let artifact = Artifact::load(fixtures_dir().join("glm_gaussian.json")).unwrap();
    let report = artifact
        .execution_report()
        .unwrap()
        .expect("glm_gaussian.json carries an execution report");
    assert_eq!(report.n_observations, 150);
    assert_eq!(report.n_learners, 3);
    assert_eq!(report.route.as_str(), "packed_scalar");
    assert_eq!(report.instrumentation.as_str(), "off");
}
