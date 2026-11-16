//! JSON round trips of refit results, for pickling in Python and for model
//! artifacts. Arrays are stored as flat row-major lists with their shape.

use ndarray::{Array1, Array2};
use serde_json::{json, Map, Value};

use super::{LssRefitParam, LssRefitResult, RefitBlock, RefitResult, TermTest};
use crate::MboostrsError;

fn bad(what: &str) -> MboostrsError {
    MboostrsError::Control(format!("refit JSON: missing or invalid field {what:?}"))
}

fn vec_of(v: &Value, what: &str) -> Result<Vec<f64>, MboostrsError> {
    v.get(what)
        .and_then(Value::as_array)
        .ok_or_else(|| bad(what))?
        .iter()
        .map(|x| x.as_f64().ok_or_else(|| bad(what)))
        .collect()
}

fn f64_of(v: &Value, what: &str) -> Result<f64, MboostrsError> {
    v.get(what).and_then(Value::as_f64).ok_or_else(|| bad(what))
}

fn usize_of(v: &Value, what: &str) -> Result<usize, MboostrsError> {
    v.get(what)
        .and_then(Value::as_u64)
        .map(|x| x as usize)
        .ok_or_else(|| bad(what))
}

fn bool_of(v: &Value, what: &str) -> Result<bool, MboostrsError> {
    v.get(what)
        .and_then(Value::as_bool)
        .ok_or_else(|| bad(what))
}

fn matrix(m: &Array2<f64>) -> Value {
    json!({"shape": [m.nrows(), m.ncols()], "data": m.iter().copied().collect::<Vec<f64>>()})
}

fn matrix_of(v: &Value, what: &str) -> Result<Array2<f64>, MboostrsError> {
    let m = v.get(what).ok_or_else(|| bad(what))?;
    let shape = m
        .get("shape")
        .and_then(Value::as_array)
        .ok_or_else(|| bad(what))?;
    let (r, c) = match shape.as_slice() {
        [r, c] => (
            r.as_u64().ok_or_else(|| bad(what))? as usize,
            c.as_u64().ok_or_else(|| bad(what))? as usize,
        ),
        _ => return Err(bad(what)),
    };
    let data = vec_of(m, "data")?;
    Array2::from_shape_vec((r, c), data).map_err(|_| bad(what))
}

fn block_json(b: &RefitBlock) -> Value {
    json!({
        "name": b.name, "start": b.start, "end": b.end,
        "householder": b.householder.to_vec(), "has_by": b.has_by,
        "rank": b.rank, "edf": b.edf, "lambda": b.lambda,
        "test": b.test.as_ref().map(|t| json!({
            "ref_df": t.ref_df, "statistic": t.statistic, "p_value": t.p_value,
        })),
        "concurvity": b.concurvity.as_ref().map(|c| json!({
            "worst": c.worst, "observed": c.observed, "estimate": c.estimate,
        })),
    })
}

fn block_of(v: &Value) -> Result<RefitBlock, MboostrsError> {
    Ok(RefitBlock {
        name: v
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| bad("name"))?
            .to_string(),
        start: usize_of(v, "start")?,
        end: usize_of(v, "end")?,
        householder: Array1::from_vec(vec_of(v, "householder")?),
        has_by: bool_of(v, "has_by")?,
        rank: usize_of(v, "rank")?,
        edf: f64_of(v, "edf")?,
        lambda: vec_of(v, "lambda")?,
        // Artifacts written before the term tests existed carry none.
        test: match v.get("test") {
            Some(t) if !t.is_null() => Some(TermTest {
                ref_df: f64_of(t, "ref_df")?,
                statistic: f64_of(t, "statistic")?,
                p_value: f64_of(t, "p_value")?,
            }),
            _ => None,
        },
        // Artifacts written before concurvity was recorded carry none.
        concurvity: match v.get("concurvity") {
            Some(c) if !c.is_null() => Some(super::Concurvity {
                worst: c.get("worst").and_then(Value::as_f64).unwrap_or(f64::NAN),
                observed: c
                    .get("observed")
                    .and_then(Value::as_f64)
                    .unwrap_or(f64::NAN),
                estimate: c
                    .get("estimate")
                    .and_then(Value::as_f64)
                    .unwrap_or(f64::NAN),
            }),
            _ => None,
        },
    })
}

fn static_str(
    v: &Value,
    what: &str,
    allowed: &[&'static str],
) -> Result<&'static str, MboostrsError> {
    let s = v
        .get(what)
        .and_then(Value::as_str)
        .ok_or_else(|| bad(what))?;
    allowed
        .iter()
        .copied()
        .find(|a| *a == s)
        .ok_or_else(|| bad(what))
}

impl RefitResult {
    /// The result as JSON.
    pub fn to_json(&self) -> Value {
        let mut m = Map::new();
        m.insert("kind".into(), json!("gam_refit"));
        m.insert(
            "blocks".into(),
            Value::Array(self.blocks.iter().map(block_json).collect()),
        );
        m.insert("coef".into(), json!(self.coef.to_vec()));
        m.insert("cov".into(), matrix(&self.cov));
        m.insert("cov_corrected".into(), matrix(&self.cov_corrected));
        m.insert("scale".into(), json!(self.scale));
        m.insert("reml_score".into(), json!(self.reml_score));
        m.insert("rho".into(), json!(self.rho.to_vec()));
        m.insert("rho_hessian".into(), matrix(&self.rho_hessian));
        m.insert("gradient_norm".into(), json!(self.gradient_norm));
        m.insert("edf_total".into(), json!(self.edf_total));
        m.insert("loglik".into(), json!(self.loglik));
        m.insert("df".into(), json!(self.df));
        m.insert("newton_iterations".into(), json!(self.newton_iterations));
        m.insert("pirls_iterations".into(), json!(self.pirls_iterations));
        m.insert("converged".into(), json!(self.converged));
        m.insert("stop_reason".into(), json!(self.stop_reason));
        m.insert(
            "unidentified_directions".into(),
            json!(self.unidentified_directions),
        );
        m.insert("unidentified_terms".into(), json!(self.unidentified_terms));
        m.insert("method".into(), json!(self.method));
        m.insert("theta".into(), json!(self.theta));
        m.insert("link".into(), json!(self.link));
        m.insert("n".into(), json!(self.n));
        m.insert("residual_df".into(), json!(self.residual_df));
        Value::Object(m)
    }

    /// Rebuild a result written by [`to_json`](Self::to_json).
    pub fn from_json(v: &Value) -> Result<Self, MboostrsError> {
        let blocks = v
            .get("blocks")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("blocks"))?
            .iter()
            .map(block_of)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(RefitResult {
            blocks,
            coef: Array1::from_vec(vec_of(v, "coef")?),
            cov: matrix_of(v, "cov")?,
            cov_corrected: matrix_of(v, "cov_corrected")?,
            scale: f64_of(v, "scale")?,
            reml_score: f64_of(v, "reml_score")?,
            rho: Array1::from_vec(vec_of(v, "rho")?),
            rho_hessian: matrix_of(v, "rho_hessian")?,
            gradient_norm: f64_of(v, "gradient_norm")?,
            edf_total: f64_of(v, "edf_total")?,
            loglik: f64_of(v, "loglik")?,
            df: f64_of(v, "df")?,
            newton_iterations: usize_of(v, "newton_iterations")?,
            pirls_iterations: usize_of(v, "pirls_iterations")?,
            converged: bool_of(v, "converged")?,
            // Absent from artifacts written before the field existed.
            stop_reason: match v.get("stop_reason") {
                Some(_) => static_str(v, "stop_reason", &super::STOP_REASONS)?,
                None => "unrecorded",
            },
            unidentified_directions: match v.get("unidentified_directions") {
                Some(_) => usize_of(v, "unidentified_directions")?,
                None => 0,
            },
            unidentified_terms: match v.get("unidentified_terms").and_then(Value::as_array) {
                Some(items) => items
                    .iter()
                    .filter_map(|item| item.as_str().map(str::to_owned))
                    .collect(),
                None => Vec::new(),
            },
            method: static_str(v, "method", &["performance", "outer"])?,
            theta: v.get("theta").and_then(Value::as_f64),
            link: static_str(v, "link", &["identity", "logit", "log"])?,
            n: usize_of(v, "n")?,
            residual_df: v.get("residual_df").and_then(Value::as_f64),
        })
    }
}

impl LssRefitResult {
    /// The result as JSON.
    pub fn to_json(&self) -> Value {
        let params: Vec<Value> = self
            .params
            .iter()
            .map(|p| {
                json!({"name": p.name, "start": p.start, "end": p.end,
                       "blocks": p.blocks.iter().map(block_json).collect::<Vec<_>>()})
            })
            .collect();
        json!({
            "kind": "gamlss_refit", "params": params, "coef": self.coef.to_vec(),
            "cov": matrix(&self.cov), "cov_corrected": matrix(&self.cov_corrected),
            "reml_score": self.reml_score, "rho": self.rho.to_vec(),
            "rho_hessian": matrix(&self.rho_hessian), "gradient_norm": self.gradient_norm,
            "iterations": self.iterations, "converged": self.converged,
            "stop_reason": self.stop_reason, "evaluations": self.evaluations,
            "restarted": self.restarted, "restart_iterations": self.restart_iterations,
            "seconds": self.seconds, "inner_iterations": self.inner_iterations,
            "information_floored": self.information_floored, "edf_total": self.edf_total,
            "n": self.n,
        })
    }

    /// Rebuild a result written by [`to_json`](Self::to_json).
    pub fn from_json(v: &Value) -> Result<Self, MboostrsError> {
        let params = v
            .get("params")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("params"))?
            .iter()
            .map(|p| {
                Ok(LssRefitParam {
                    name: p
                        .get("name")
                        .and_then(Value::as_str)
                        .ok_or_else(|| bad("name"))?
                        .to_string(),
                    start: usize_of(p, "start")?,
                    end: usize_of(p, "end")?,
                    blocks: p
                        .get("blocks")
                        .and_then(Value::as_array)
                        .ok_or_else(|| bad("blocks"))?
                        .iter()
                        .map(block_of)
                        .collect::<Result<Vec<_>, _>>()?,
                })
            })
            .collect::<Result<Vec<_>, MboostrsError>>()?;
        Ok(LssRefitResult {
            params,
            coef: Array1::from_vec(vec_of(v, "coef")?),
            cov: matrix_of(v, "cov")?,
            cov_corrected: matrix_of(v, "cov_corrected")?,
            reml_score: f64_of(v, "reml_score")?,
            rho: Array1::from_vec(vec_of(v, "rho")?),
            rho_hessian: matrix_of(v, "rho_hessian")?,
            gradient_norm: f64_of(v, "gradient_norm")?,
            iterations: usize_of(v, "iterations")?,
            converged: bool_of(v, "converged")?,
            // The run diagnostics below were added after the format; an
            // artifact written before them reads them as unknown.
            stop_reason: v
                .get("stop_reason")
                .and_then(Value::as_str)
                .and_then(|s| super::STOP_REASONS.iter().find(|r| **r == s).copied())
                .unwrap_or("unknown"),
            evaluations: v.get("evaluations").and_then(Value::as_u64).unwrap_or(0) as usize,
            restarted: v.get("restarted").and_then(Value::as_bool).unwrap_or(false),
            restart_iterations: v
                .get("restart_iterations")
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize,
            seconds: v.get("seconds").and_then(Value::as_f64).unwrap_or(f64::NAN),
            inner_iterations: v
                .get("inner_iterations")
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize,
            information_floored: bool_of(v, "information_floored")?,
            edf_total: f64_of(v, "edf_total")?,
            n: usize_of(v, "n")?,
        })
    }
}
