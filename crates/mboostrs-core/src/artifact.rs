//! Reader for the portable model artifact written by Python's `mb.save_model`
//! (`python/mboostrs/serialize.py`).
//!
//! The artifact is schema-versioned JSON, independent of Python and the
//! compiled extension. This reader loads any artifact for inspection and
//! scores linear (`glmboost`) models directly in Rust, using the same
//! [`crate::path::predict_glmboost`] the Python binding calls for its own
//! `predict()` — so link-scale predictions are not merely close to Python's,
//! they are produced by literally the same aggregation routine, run on the
//! same numbers. Models whose prediction needs the spline / tensor /
//! transform machinery (gamboost, interactions, inline transforms) are loaded
//! but not scored here; [`Artifact::predict`] refuses with a message pointing
//! at the Python reader (`mboostrs.load_model`).
//!
//! This mirrors the R reader (`mboostrs/R/mboostrs_load.R`, committed
//! alongside this module) field-for-field: both decode the writer's
//! `{"__type__": ...}` tagged values generically first, then pull the small
//! set of state fields that linear prediction needs. Any deliberate
//! deviations from the R reader's behavior are called out inline below.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::{Map, Value};

use crate::report::{EngineRoute, FallbackReason, FitReport, InstrumentationLevel};

/// Schema version this reader understands. An artifact written with a
/// different version is rejected outright rather than guessed at — see
/// [`Artifact::load`].
pub const SCHEMA_VERSION: i64 = 1;

/// Errors from reading or scoring a portable model artifact.
#[derive(Debug, thiserror::Error)]
pub enum ArtifactError {
    #[error("failed to read artifact at {path}: {source}")]
    /// Reading the artifact file failed.
    Io {
        /// Filesystem path the artifact was being read from.
        path: String,
        /// Underlying I/O error.
        source: std::io::Error,
    },

    #[error("failed to parse artifact JSON: {0}")]
    /// The artifact's JSON payload could not be parsed.
    Json(#[from] serde_json::Error),

    #[error(
        "artifact schema version {found} is not supported by this reader (expected {SCHEMA_VERSION}); \
         re-export it with a matching mboostrs release"
    )]
    /// The artifact's schema version is newer than this reader supports.
    UnsupportedSchemaVersion {
        /// Version tag found in the artifact.
        found: String,
    },

    #[error("unknown model class {0:?} in artifact")]
    /// The model class string in the artifact is not one this reader knows.
    UnknownModelClass(String),

    #[error("unknown encoded type {0:?} in artifact")]
    /// A typed-value `"__type__"` marker is not recognized.
    UnknownEncodedType(String),

    #[error("artifact is missing required field {0:?}")]
    /// A required top-level field is absent from the artifact.
    MissingField(&'static str),

    #[error("artifact field {field:?} is malformed: {message}")]
    /// An artifact field holds a value of the wrong shape for its contract.
    InvalidField {
        /// Name of the malformed field.
        field: String,
        /// Detail of why the value is invalid.
        message: String,
    },

    #[error(
        "the Rust reader scores linear glmboost models only; this artifact is a {model_class}{recipe_note}. \
         Load it with the Python reader (mboostrs.load_model) to predict."
    )]
    /// The artifact is a model this reader cannot score (non-linear family,
    /// structural formula, or transform recipe).
    NotLinearlyScorable {
        /// Model class recorded in the artifact.
        model_class: String,
        /// Adds ", built from a non-structural formula" when a recipe is present.
        recipe_note: &'static str,
    },

    #[error("response-scale prediction is not implemented for family {0:?} in the Rust reader")]
    /// Response-scale prediction unsupported for this family.
    UnsupportedFamily(String),

    #[error("newdata is missing predictor {0:?}")]
    /// The newdata supplied to predict is missing a named predictor.
    MissingPredictor(String),

    #[error(
        "newdata column {name:?} has {got} rows, expected {expected} (columns must have equal length)"
    )]
    /// Newdata columns with differing row counts.
    RaggedNewData {
        /// Column with the unexpected row count.
        name: String,
        /// Expected number of rows.
        expected: usize,
        /// Observed number of rows.
        got: usize,
    },
}

// ---------------------------------------------------------------------------
// typed-value decoding
// ---------------------------------------------------------------------------
//
// The Python writer tags non-trivial values as `{"__type__": ...}`. This
// walks the parsed JSON tree back into plain `serde_json::Value`s: ndarrays
// become number arrays (or, for arrays with more than one dimension, an
// object retaining `shape` and flat `data` — see the note in `decode_ndarray`
// on why this reader does not attempt R's naive reshape), tuples and dicts
// become plain arrays/objects, and family/learner/execution-report/transform
// wrappers become plain objects carrying just their payload.

fn decode(value: &Value) -> Result<Value, ArtifactError> {
    match value {
        Value::Array(items) => Ok(Value::Array(
            items.iter().map(decode).collect::<Result<_, _>>()?,
        )),
        Value::Object(map) => match map.get("__type__").and_then(Value::as_str) {
            None => {
                let mut out = Map::with_capacity(map.len());
                for (k, v) in map {
                    out.insert(k.clone(), decode(v)?);
                }
                Ok(Value::Object(out))
            }
            Some("ndarray") => decode_ndarray(map),
            Some("tuple") => {
                let items = map
                    .get("items")
                    .and_then(Value::as_array)
                    .ok_or_else(|| tagged_field_error("tuple", "items"))?;
                Ok(Value::Array(
                    items.iter().map(decode).collect::<Result<_, _>>()?,
                ))
            }
            Some("dict") => {
                let items = map
                    .get("items")
                    .and_then(Value::as_object)
                    .ok_or_else(|| tagged_field_error("dict", "items"))?;
                let mut out = Map::with_capacity(items.len());
                for (k, v) in items {
                    out.insert(k.clone(), decode(v)?);
                }
                Ok(Value::Object(out))
            }
            Some("pairs") => {
                let items = map
                    .get("items")
                    .and_then(Value::as_array)
                    .ok_or_else(|| tagged_field_error("pairs", "items"))?;
                let mut out = Vec::with_capacity(items.len());
                for kv in items {
                    let kv = kv
                        .as_array()
                        .filter(|kv| kv.len() == 2)
                        .ok_or_else(|| tagged_field_error("pairs", "items[i] as [k, v]"))?;
                    out.push(Value::Array(vec![decode(&kv[0])?, decode(&kv[1])?]));
                }
                Ok(Value::Array(out))
            }
            Some("family") | Some("learner") => {
                let name = map
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| tagged_field_error("family/learner", "name"))?;
                let params = decode(
                    map.get("params")
                        .ok_or_else(|| tagged_field_error("family/learner", "params"))?,
                )?;
                let mut out = Map::with_capacity(2);
                out.insert("name".to_string(), Value::String(name.to_string()));
                out.insert("params".to_string(), params);
                Ok(Value::Object(out))
            }
            Some("execution_report") => decode(
                map.get("fields")
                    .ok_or_else(|| tagged_field_error("execution_report", "fields"))?,
            ),
            Some("transform_recipe") => {
                let steps = decode(
                    map.get("steps")
                        .ok_or_else(|| tagged_field_error("transform_recipe", "steps"))?,
                )?;
                let mut out = Map::with_capacity(1);
                out.insert("steps".to_string(), steps);
                Ok(Value::Object(out))
            }
            Some(other) => Err(ArtifactError::UnknownEncodedType(other.to_string())),
        },
        other => Ok(other.clone()),
    }
}

fn tagged_field_error(kind: &str, field: &str) -> ArtifactError {
    ArtifactError::InvalidField {
        field: kind.to_string(),
        message: format!("missing or malformed {field:?}"),
    }
}

/// Decode an `{"__type__": "ndarray", "dtype", "shape", "data"}` value.
///
/// 1-D (and 0-D) arrays decode to a flat `Value::Array` of numbers, which is
/// all `Artifact`'s typed accessors need. The R reader reshapes higher-rank
/// arrays with `array(data, dim = rev(shape))`, which is *not* a transpose of
/// numpy's row-major layout and is explicitly flagged there as untested ("the
/// reader only needs 1-D arrays, so keep this simple"). Nothing this reader
/// exposes needs rank > 1, so rather than reproduce that same shortcut this
/// keeps `shape` and the flat, row-major `data` both available under an
/// object, and leaves reshaping to a caller that actually needs it.
fn decode_ndarray(map: &Map<String, Value>) -> Result<Value, ArtifactError> {
    let data = map
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| tagged_field_error("ndarray", "data"))?;
    let shape = map
        .get("shape")
        .and_then(Value::as_array)
        .ok_or_else(|| tagged_field_error("ndarray", "shape"))?;

    if shape.len() <= 1 {
        return Ok(Value::Array(data.clone()));
    }

    let mut out = Map::with_capacity(2);
    out.insert("shape".to_string(), Value::Array(shape.clone()));
    out.insert("data".to_string(), Value::Array(data.clone()));
    Ok(Value::Object(out))
}

// ---------------------------------------------------------------------------
// generic Value helpers
// ---------------------------------------------------------------------------

fn get<'a>(obj: &'a Value, field: &'static str) -> Option<&'a Value> {
    obj.as_object()?.get(field)
}

fn require<'a>(obj: &'a Value, field: &'static str) -> Result<&'a Value, ArtifactError> {
    get(obj, field).ok_or(ArtifactError::MissingField(field))
}

fn as_f64_field(obj: &Value, field: &'static str) -> Result<f64, ArtifactError> {
    require(obj, field)?
        .as_f64()
        .ok_or_else(|| invalid(field, "expected a number"))
}

fn as_str_field<'a>(obj: &'a Value, field: &'static str) -> Result<&'a str, ArtifactError> {
    require(obj, field)?
        .as_str()
        .ok_or_else(|| invalid(field, "expected a string"))
}

fn as_f64_vec_field(obj: &Value, field: &'static str) -> Result<Vec<f64>, ArtifactError> {
    let value = require(obj, field)?;
    if value.is_null() {
        return Ok(Vec::new());
    }
    value
        .as_array()
        .ok_or_else(|| invalid(field, "expected an array"))?
        .iter()
        .map(|v| v.as_f64().ok_or_else(|| invalid(field, "expected numbers")))
        .collect()
}

fn as_string_vec_field(obj: &Value, field: &'static str) -> Result<Vec<String>, ArtifactError> {
    let value = require(obj, field)?;
    if value.is_null() {
        return Ok(Vec::new());
    }
    value
        .as_array()
        .ok_or_else(|| invalid(field, "expected an array"))?
        .iter()
        .map(|v| {
            v.as_str()
                .map(str::to_string)
                .ok_or_else(|| invalid(field, "expected strings"))
        })
        .collect()
}

fn as_usize_vec_field(obj: &Value, field: &'static str) -> Result<Vec<usize>, ArtifactError> {
    let value = require(obj, field)?;
    if value.is_null() {
        return Ok(Vec::new());
    }
    value
        .as_array()
        .ok_or_else(|| invalid(field, "expected an array"))?
        .iter()
        .map(|v| {
            v.as_u64()
                .map(|u| u as usize)
                .ok_or_else(|| invalid(field, "expected non-negative integers"))
        })
        .collect()
}

fn invalid(field: &str, message: &str) -> ArtifactError {
    ArtifactError::InvalidField {
        field: field.to_string(),
        message: message.to_string(),
    }
}

// ---------------------------------------------------------------------------
// public typed views
// ---------------------------------------------------------------------------

/// A model's family, as recorded in the artifact: the Python class name
/// (`"Gaussian"`, `"Quantile"`, ...) plus its constructor parameters.
#[derive(Clone, Debug)]
pub struct FamilySpec {
    /// Python family class name (`"Gaussian"`, `"Quantile"`, ...).
    pub name: String,
    /// Extra constructor parameters (e.g. `tau` for `Quantile`), decoded but
    /// left as JSON — the set of parameters is family-specific and this
    /// reader does not need to interpret them to score a linear model.
    pub params: Value,
}

/// Scale on which to produce predictions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scale {
    /// The linear predictor `eta`.
    Link,
    /// `eta` passed through the family's inverse link.
    Response,
}

/// New observations to score, as named columns of equal length.
///
/// Column names must match the artifact's `predictor_names`.
#[derive(Clone, Debug, Default)]
pub struct NewData(BTreeMap<String, Vec<f64>>);

impl NewData {
    /// Create an empty set of columns.
    pub fn new() -> Self {
        Self(BTreeMap::new())
    }

    /// Add a predictor column, replacing any existing column of the same name.
    pub fn with_column(mut self, name: impl Into<String>, values: Vec<f64>) -> Self {
        self.0.insert(name.into(), values);
        self
    }

    /// Return the predictor column with the given name, if present.
    pub fn column(&self, name: &str) -> Option<&[f64]> {
        self.0.get(name).map(Vec::as_slice)
    }
}

/// A loaded portable model artifact.
///
/// [`Artifact::load`] accepts any artifact this reader's schema version
/// supports, for either `GLMBoostModel` or `GAMBoostModel`. The typed
/// accessors below (`formula`, `family`, `execution_report`, ...) work for
/// both; [`Artifact::predict`] only scores linear `GLMBoostModel` fits with
/// no transform recipe, matching the scope of the R reader.
#[derive(Debug)]
pub struct Artifact {
    /// Schema version the artifact was written with.
    pub schema_version: i64,
    /// `mboostrs` package version that produced the artifact.
    pub mboostrs_version: String,
    /// Model class: `GLMBoostModel` or `GAMBoostModel`.
    pub model_class: String,
    /// The decoded `state` object (tag-stripped, but otherwise the writer's
    /// full field set), for fields this module doesn't surface a dedicated
    /// accessor for.
    state: Value,
    transform_recipe: Option<Value>,
}

impl Artifact {
    /// Read and validate an artifact written by `mb.save_model`.
    ///
    /// Rejects a schema version this reader does not understand (rather than
    /// guessing at forward-compatibility) and a `model_class` other than
    /// `GLMBoostModel` / `GAMBoostModel`.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ArtifactError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| ArtifactError::Io {
            path: path.display().to_string(),
            source,
        })?;
        Self::from_json_str(&text)
    }

    /// Parse an artifact already read into memory.
    pub fn from_json_str(text: &str) -> Result<Self, ArtifactError> {
        let document: Value = serde_json::from_str(text)?;

        let schema_version = document
            .get("schema_version")
            .and_then(Value::as_i64)
            .ok_or(ArtifactError::MissingField("schema_version"))?;
        if schema_version != SCHEMA_VERSION {
            return Err(ArtifactError::UnsupportedSchemaVersion {
                found: document
                    .get("schema_version")
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "null".to_string()),
            });
        }

        let model_class = document
            .get("model_class")
            .and_then(Value::as_str)
            .ok_or(ArtifactError::MissingField("model_class"))?
            .to_string();
        if model_class != "GLMBoostModel" && model_class != "GAMBoostModel" {
            return Err(ArtifactError::UnknownModelClass(model_class));
        }

        let mboostrs_version = document
            .get("mboostrs_version")
            .and_then(Value::as_str)
            .ok_or(ArtifactError::MissingField("mboostrs_version"))?
            .to_string();

        let raw_state = document
            .get("state")
            .ok_or(ArtifactError::MissingField("state"))?;
        let state = decode(raw_state)?;

        let transform_recipe = match document.get("transform_recipe") {
            None | Some(Value::Null) => None,
            Some(raw) => Some(decode(raw)?),
        };

        Ok(Self {
            schema_version,
            mboostrs_version,
            model_class,
            state,
            transform_recipe,
        })
    }

    /// The full decoded `state` object, for inspecting fields this module
    /// does not expose a dedicated accessor for (e.g. GAMBoost's per-learner
    /// basis data). Tags (`{"__type__": ...}`) have already been stripped;
    /// see the module-level `decode` for the mapping.
    pub fn raw_state(&self) -> &Value {
        &self.state
    }

    /// Whether the artifact records a transform recipe.
    pub fn has_transform_recipe(&self) -> bool {
        self.transform_recipe.is_some()
    }

    /// The decoded transform recipe steps, if the model has one. Left as
    /// generic JSON deliberately: replaying `scale`/`log`/factor transforms
    /// is the Python reader's job, not this one's — see the module docs.
    pub fn transform_recipe_steps(&self) -> Option<&Value> {
        self.transform_recipe.as_ref().and_then(|v| get(v, "steps"))
    }

    /// The model formula string recorded at fit time.
    pub fn formula(&self) -> Result<&str, ArtifactError> {
        as_str_field(&self.state, "formula")
    }

    /// The response column name recorded at fit time.
    pub fn response_name(&self) -> Result<&str, ArtifactError> {
        as_str_field(&self.state, "response_name")
    }

    /// The fitted family (Python class name + decoded constructor params).
    pub fn family(&self) -> Result<FamilySpec, ArtifactError> {
        let family = require(&self.state, "family")?;
        let name = as_str_field(family, "name")?.to_string();
        let params = get(family, "params").cloned().unwrap_or(Value::Null);
        Ok(FamilySpec { name, params })
    }

    /// The offset used in the fit.
    pub fn offset(&self) -> Result<f64, ArtifactError> {
        as_f64_field(&self.state, "offset")
    }

    /// The boosting step size `nu`.
    pub fn nu(&self) -> Result<f64, ArtifactError> {
        as_f64_field(&self.state, "nu")
    }

    /// The base-learner names, in fit order.
    pub fn learner_names(&self) -> Result<Vec<String>, ArtifactError> {
        as_string_vec_field(&self.state, "learner_names")
    }

    /// The predictor column names, in fit order.
    pub fn predictor_names(&self) -> Result<Vec<String>, ArtifactError> {
        as_string_vec_field(&self.state, "predictor_names")
    }

    /// Per-predictor centering means subtracted before fitting (linear fits).
    pub fn centering_means(&self) -> Result<Vec<f64>, ArtifactError> {
        as_f64_vec_field(&self.state, "centering_means")
    }

    /// The base-learner selected at each boosting step.
    pub fn selected(&self) -> Result<Vec<usize>, ArtifactError> {
        as_usize_vec_field(&self.state, "selected")
    }

    /// Per-step coefficients along the boosting path. Linear (`glmboost`)
    /// models store one scalar per step (as a length-1 vector); GAMBoost
    /// spline coefficients are longer. [`Artifact::predict`] uses only the
    /// first entry of each, matching the R reader and `coef_glmboost`.
    pub fn coefficients(&self) -> Result<Vec<Vec<f64>>, ArtifactError> {
        require(&self.state, "coefficients")?
            .as_array()
            .ok_or_else(|| invalid("coefficients", "expected an array"))?
            .iter()
            .map(|c| {
                c.as_array()
                    .ok_or_else(|| invalid("coefficients", "expected an array of arrays"))?
                    .iter()
                    .map(|v| {
                        v.as_f64()
                            .ok_or_else(|| invalid("coefficients", "expected numbers"))
                    })
                    .collect()
            })
            .collect()
    }

    /// The versioned execution report recorded at fit time, decoded into the
    /// same [`FitReport`] type the fitting engine itself produces.
    pub fn execution_report(&self) -> Result<Option<FitReport>, ArtifactError> {
        match get(&self.state, "execution_report") {
            None | Some(Value::Null) => Ok(None),
            Some(report) => Ok(Some(parse_execution_report(report)?)),
        }
    }

    fn ensure_linearly_scorable(&self) -> Result<(), ArtifactError> {
        if self.model_class == "GLMBoostModel" && self.transform_recipe.is_none() {
            Ok(())
        } else {
            Err(ArtifactError::NotLinearlyScorable {
                model_class: self.model_class.clone(),
                recipe_note: if self.transform_recipe.is_some() {
                    " with a transform recipe"
                } else {
                    ""
                },
            })
        }
    }

    /// Score a linear `glmboost` model on `newdata`.
    ///
    /// Aggregates `nu * coefficient` per learner across the selected path —
    /// the same reduction [`crate::path::coef_glmboost`] performs — then
    /// reconstructs `eta` with [`crate::path::predict_glmboost`], the exact
    /// function the Python binding calls for its own `predict()`. `Scale::
    /// Response` then applies the inverse link keyed by the artifact's family
    /// name, matching `mboostrs.families._inverse_link` and the R reader's
    /// `.reader_response_link` table.
    ///
    /// Returns [`ArtifactError::NotLinearlyScorable`] for a `GAMBoostModel`
    /// or any model carrying a transform recipe — matching the R reader's
    /// scope exactly. Predicting those requires the Python reader.
    pub fn predict(&self, newdata: &NewData, scale: Scale) -> Result<Vec<f64>, ArtifactError> {
        self.ensure_linearly_scorable()?;

        let offset = self.offset()?;
        let nu = self.nu()?;
        let selected = self.selected()?;
        let coefficients = self.coefficients()?;
        let predictor_names = self.predictor_names()?;
        let centering_means = self.centering_means()?;

        if coefficients.len() != selected.len() {
            return Err(invalid(
                "coefficients",
                &format!(
                    "length {} does not match 'selected' length {}",
                    coefficients.len(),
                    selected.len()
                ),
            ));
        }
        let coefs_flat: Vec<f64> = coefficients
            .iter()
            .map(|c| c.first().copied().unwrap_or(0.0))
            .collect();

        let p = predictor_names.len();
        let n = match predictor_names.first() {
            Some(first) => newdata
                .column(first)
                .ok_or_else(|| ArtifactError::MissingPredictor(first.clone()))?
                .len(),
            None => 0,
        };

        let mut x_matrix = vec![0.0f64; n * p];
        for (j, name) in predictor_names.iter().enumerate() {
            let col = newdata
                .column(name)
                .ok_or_else(|| ArtifactError::MissingPredictor(name.clone()))?;
            if col.len() != n {
                return Err(ArtifactError::RaggedNewData {
                    name: name.clone(),
                    expected: n,
                    got: col.len(),
                });
            }
            for i in 0..n {
                x_matrix[i * p + j] = col[i];
            }
        }

        let eta = crate::path::try_predict_glmboost(
            offset,
            &selected,
            &coefs_flat,
            nu,
            &x_matrix,
            &centering_means,
            None,
            n,
            p,
        )
        .map_err(|e| invalid("prediction", &e.to_string()))?;

        match scale {
            Scale::Link => Ok(eta),
            Scale::Response => {
                let family = self.family()?;
                apply_inverse_link(&family.name, &eta)
            }
        }
    }
}

/// Response-scale inverse link, keyed by the artifact's family name (the
/// Python class name). Matches `mboostrs.families._inverse_link` for every
/// family it implements without needing the fitted per-observation state
/// that e.g. `PropOdds`/`Multinomial` response scale requires — those two
/// (and any other family not listed here) are rejected rather than silently
/// returning the link scale, matching the R reader's `.reader_response_link`
/// table exactly (same family set, same formulas).
fn apply_inverse_link(family_name: &str, eta: &[f64]) -> Result<Vec<f64>, ArtifactError> {
    match family_name {
        "Gaussian" | "GaussReg" | "Laplace" | "Huber" | "Quantile" | "Expectile" => {
            Ok(eta.to_vec())
        }
        "Binomial" | "AdaExp" | "HingeLoss" => Ok(eta
            .iter()
            .map(|&f| 1.0 / (1.0 + (-2.0 * f).exp()))
            .collect()),
        "Poisson" | "GammaReg" | "NBinomial" => Ok(eta.iter().map(|&f| f.exp()).collect()),
        other => Err(ArtifactError::UnsupportedFamily(other.to_string())),
    }
}

// ---------------------------------------------------------------------------
// execution report decoding
// ---------------------------------------------------------------------------

fn opt_u64_field(obj: &Value, field: &'static str) -> Result<Option<u64>, ArtifactError> {
    match get(obj, field) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_u64()
            .map(Some)
            .ok_or_else(|| invalid(field, "expected a non-negative integer or null")),
    }
}

fn opt_f64_field(obj: &Value, field: &'static str) -> Result<Option<f64>, ArtifactError> {
    match get(obj, field) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_f64()
            .map(Some)
            .ok_or_else(|| invalid(field, "expected a number or null")),
    }
}

fn usize_field(obj: &Value, field: &'static str) -> Result<usize, ArtifactError> {
    require(obj, field)?
        .as_u64()
        .map(|u| u as usize)
        .ok_or_else(|| invalid(field, "expected a non-negative integer"))
}

fn usize_field_or_zero(obj: &Value, field: &'static str) -> Result<usize, ArtifactError> {
    match get(obj, field) {
        None => Ok(0),
        Some(value) => value
            .as_u64()
            .map(|number| number as usize)
            .ok_or_else(|| invalid(field, "expected a non-negative integer")),
    }
}

fn bool_field(obj: &Value, field: &'static str) -> Result<bool, ArtifactError> {
    require(obj, field)?
        .as_bool()
        .ok_or_else(|| invalid(field, "expected a boolean"))
}

fn parse_execution_report(report: &Value) -> Result<FitReport, ArtifactError> {
    use std::str::FromStr;

    let instrumentation_str = as_str_field(report, "instrumentation")?;
    let instrumentation = InstrumentationLevel::from_str(instrumentation_str)
        .map_err(|message| invalid("instrumentation", &message))?;

    let route_str = as_str_field(report, "route")?;
    let route = EngineRoute::from_str(route_str).map_err(|message| invalid("route", &message))?;

    let fallback_reasons = require(report, "fallback_reasons")?
        .as_array()
        .ok_or_else(|| invalid("fallback_reasons", "expected an array"))?
        .iter()
        .map(|v| {
            let s = v
                .as_str()
                .ok_or_else(|| invalid("fallback_reasons", "expected strings"))?;
            FallbackReason::from_str(s).map_err(|message| invalid("fallback_reasons", &message))
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(FitReport {
        schema_version: usize_field(report, "schema_version")? as u32,
        instrumentation,
        route,
        n_observations: usize_field(report, "n_observations")?,
        n_learners: usize_field(report, "n_learners")?,
        total_basis_columns: usize_field(report, "total_basis_columns")?,
        requested_mstop: usize_field(report, "requested_mstop")?,
        completed_mstop: usize_field(report, "completed_mstop")?,
        effective_threads: usize_field(report, "effective_threads")?,
        gram_eligible: bool_field(report, "gram_eligible")?,
        fallback_reasons,
        scalar_learner_count: usize_field(report, "scalar_learner_count")?,
        block_learner_count: usize_field(report, "block_learner_count")?,
        tree_learner_count: usize_field(report, "tree_learner_count")?,
        cholesky_solver_count: usize_field(report, "cholesky_solver_count")?,
        lu_solver_count: usize_field(report, "lu_solver_count")?,
        regularized_solver_count: usize_field(report, "regularized_solver_count")?,
        max_solver_jitter: opt_f64_field(report, "max_solver_jitter")?,
        factor_cache_bytes: usize_field(report, "factor_cache_bytes")?,
        retained_learner_bytes: usize_field_or_zero(report, "retained_learner_bytes")?,
        gram_cache_bytes: usize_field(report, "gram_cache_bytes")?,
        calibration_count: usize_field(report, "calibration_count")?,
        tie_selection_count: usize_field(report, "tie_selection_count")?,
        component_subsample_iterations: usize_field(report, "component_subsample_iterations")?,
        calibration_converged_count: usize_field(report, "calibration_converged_count")?,
        calibration_limit_count: usize_field(report, "calibration_limit_count")?,
        calibration_failure_count: usize_field(report, "calibration_failure_count")?,
        nonfinite_candidate_fallbacks: usize_field(report, "nonfinite_candidate_fallbacks")?,
        setup_ns: opt_u64_field(report, "setup_ns")?,
        basis_ns: opt_u64_field(report, "basis_ns")?,
        factorization_ns: opt_u64_field(report, "factorization_ns")?,
        cross_gram_ns: opt_u64_field(report, "cross_gram_ns")?,
        iteration_ns: opt_u64_field(report, "iteration_ns")?,
        reconstruction_ns: opt_u64_field(report, "reconstruction_ns")?,
        binding_ns: opt_u64_field(report, "binding_ns")?,
        total_ns: opt_u64_field(report, "total_ns")?,
    })
}
