//! Preliminary tree and neural exploration shared by both language bindings.
//!
//! Permutation importance, partial dependence, marginal diagnostics and pair
//! H statistics propose comparisons; none is a test of a selected model.
//! Callers must reserve evaluation data outside the entire specification search.

use crate::learners::{prepare_btree_categorical, FittedTree};
use crate::screen::{column_diagnostics, h_statistic_with_variance};
use crate::{BoostControl, MboostrsError};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rand::{seq::SliceRandom, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde_json::{json, Value};
use std::collections::BTreeMap;

mod neural;

enum Surrogate {
    Neural(neural::Neural),
    Native {
        offset: f64,
        trees: Vec<FittedTree>,
    },
    // hessboost 0.2.4: `learner::BoostedModel` moved to `model::BoostedModel`
    // (path fix for #298 phase 1, 2026-10-08).
    #[cfg(feature = "hessboost")]
    Hessboost(Box<hessboost::model::BoostedModel>),
}

impl Surrogate {
    fn predict(&self, x: ArrayView2<f64>) -> Result<Vec<f64>, MboostrsError> {
        match self {
            Self::Neural(model) => Ok(model.predict(x)),
            Self::Native { offset, trees } => Ok(x
                .rows()
                .into_iter()
                .map(|row| offset + 0.3 * trees.iter().map(|t| t.predict_row(row)).sum::<f64>())
                .collect()),
            #[cfg(feature = "hessboost")]
            Self::Hessboost(model) => {
                let values: Vec<f32> = x.iter().map(|v| *v as f32).collect();
                let data = hessboost::data::DMatrix::from_dense(&values, x.nrows(), x.ncols())
                    .map_err(hess_error)?;
                // hessboost 0.2.4: `predict` selects iterations via
                // `Iterations` and returns a `Predictions` container, not an
                // iterator; `Iterations::Best` is every round without early
                // stopping, matching the old behavior (path fix for #298
                // phase 1, 2026-10-08).
                Ok(model
                    .predict(&data, hessboost::model::Iterations::Best)
                    .map_err(hess_error)?
                    .into_vec()
                    .into_iter()
                    .map(f64::from)
                    .collect())
            }
        }
    }
}

#[cfg(feature = "hessboost")]
fn hess_error(error: hessboost::error::HessboostError) -> MboostrsError {
    MboostrsError::Control(format!("hessboost pre-exploration: {error}"))
}

fn fit(
    x: &Array2<f64>,
    y: &[f64],
    categorical: &[usize],
    engine: &str,
    seed: u64,
    rounds: u32,
) -> Result<Surrogate, MboostrsError> {
    match engine {
        "neural" => Ok(Surrogate::Neural(neural::Neural::fit(
            x.view(),
            y,
            categorical,
            seed,
            rounds,
        )?)),
        "native" => {
            let w = Array1::ones(y.len());
            let learner = prepare_btree_categorical(
                x.view(),
                w.view(),
                7,
                20,
                4,
                None,
                "surrogate",
                categorical.to_vec(),
            );
            let control = BoostControl::new(rounds, 0.3)?;
            let path = crate::engine::try_fit_with(
                &mut [learner],
                ArrayView1::from(y),
                w.view(),
                &mut crate::families::Gaussian,
                &control,
                None,
                false,
            )?;
            Ok(Surrogate::Native {
                offset: path.offset,
                trees: path
                    .coefs
                    .iter()
                    .map(|v| FittedTree::from_coef_vec(ArrayView1::from(v)))
                    .collect(),
            })
        }
        "hessboost" => {
            #[cfg(feature = "hessboost")]
            {
                // hessboost 0.2.4: `FeatureType` left the prelude; it is
                // re-exported from `hessboost::data` (path fix for #298
                // phase 1, 2026-10-08).
                use hessboost::data::FeatureType;
                use hessboost::prelude::*;
                let values: Vec<f32> = x.iter().map(|v| *v as f32).collect();
                let labels: Vec<f32> = y.iter().map(|v| *v as f32).collect();
                let data = DMatrix::from_dense(&values, x.nrows(), x.ncols())
                    .map_err(hess_error)?
                    .with_labels(&labels)
                    .map_err(hess_error)?
                    .with_feature_types(
                        &categorical
                            .iter()
                            .map(|n| {
                                if *n == 0 {
                                    FeatureType::Numerical
                                } else {
                                    FeatureType::Categorical
                                }
                            })
                            .collect::<Vec<_>>(),
                    )
                    .map_err(hess_error)?;
                let params = TrainingParams::builder()
                    // hessboost 0.2.4: `objective()` takes the `Objective`
                    // enum, not an XGBoost-style string; the squared-loss
                    // variant is `Objective::SquaredError(RegLoss::default())`.
                    .objective(Objective::SquaredError(RegLoss::default()))
                    .tree_method(TreeMethod::Hist)
                    .max_depth(4)
                    .eta(0.3)
                    .seed(seed)
                    .nthread(1)
                    .build()
                    .map_err(hess_error)?;
                Ok(Surrogate::Hessboost(Box::new(
                    // hessboost 0.2.4: `learner::train` moved to
                    // `training::train` (same signature).
                    hessboost::training::train(&params, &data, rounds as usize)
                        .map_err(hess_error)?,
                )))
            }
            #[cfg(not(feature = "hessboost"))]
            {
                let _ = seed;
                Err(MboostrsError::Control(
                    "hessboost pre-exploration requires a build with the hessboost Cargo feature"
                        .into(),
                ))
            }
        }
        _ => Err(MboostrsError::Control(
            "pre-exploration engine must be native, neural, dual or hessboost".into(),
        )),
    }
}

fn variance(y: &[f64]) -> f64 {
    let mean = y.iter().sum::<f64>() / y.len() as f64;
    y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / y.len() as f64
}

fn mse(y: &[f64], pred: &[f64]) -> f64 {
    y.iter()
        .zip(pred)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / y.len() as f64
}

fn pd(model: &Surrogate, rows: &Array2<f64>, columns: &[usize]) -> Result<Vec<f64>, MboostrsError> {
    let m = rows.nrows();
    let mut grid = Array2::from_shape_fn((m * m, rows.ncols()), |(i, j)| rows[[i % m, j]]);
    for i in 0..m * m {
        for &j in columns {
            grid[[i, j]] = rows[[i / m, j]];
        }
    }
    Ok(model
        .predict(grid.view())?
        .chunks(m)
        .map(|v| v.iter().sum::<f64>() / m as f64)
        .collect())
}

fn dual_report(mut tree: Value, neural: Result<Value, MboostrsError>) -> Value {
    tree["engine"] = json!("dual");
    tree["fits"] = json!(2);
    tree["shape_source"] = json!("native");
    let neural = match neural {
        Ok(report) => report,
        Err(error) => {
            tree["neural_failure"] = json!(error.to_string());
            tree["fit_accounting"] =
                json!("two probe attempts; failed neural attempt charged conservatively");
            return tree;
        }
    };
    type PairSources = (Value, Vec<&'static str>, Value, Value);
    let mut merged: BTreeMap<(usize, usize), PairSources> = BTreeMap::new();
    for (source, report) in [("tree", &tree), ("neural", &neural)] {
        for pair in report["pairs"].as_array().unwrap() {
            let a = pair[0].as_u64().unwrap() as usize;
            let b = pair[1].as_u64().unwrap() as usize;
            let key = (a.min(b), a.max(b));
            let entry = merged
                .entry(key)
                .or_insert_with(|| (pair.clone(), vec![], Value::Null, Value::Null));
            let rank = |row: &Value| (row[2].as_f64().unwrap() >= 0.05, row[3].as_f64().unwrap());
            if rank(pair) > rank(&entry.0) {
                entry.0 = pair.clone();
            }
            if pair[2].as_f64().unwrap() >= 0.05 {
                entry.1.push(source);
            }
            if source == "tree" {
                entry.2 = pair.clone();
            } else {
                entry.3 = pair.clone();
            }
        }
    }
    let mut pairs: Vec<Value> = merged.values().map(|v| v.0.clone()).collect();
    pairs.sort_by(|a, b| b[3].as_f64().unwrap().total_cmp(&a[3].as_f64().unwrap()));
    tree["pairs"] = json!(pairs);
    tree["pair_sources"] = json!(merged
        .iter()
        .map(|((a, b), (_, sources, t, n))| {
            json!({"indices":[a,b],"sources":sources,"tree":t,"neural":n})
        })
        .collect::<Vec<_>>());
    let mut priority = Vec::new();
    for j in 0..tree["columns"].as_array().unwrap().len() {
        let score = tree["columns"][j]["importance_fraction"]
            .as_f64()
            .unwrap()
            .max(
                neural["columns"][j]["importance_fraction"]
                    .as_f64()
                    .unwrap(),
            );
        tree["columns"][j]["neural_importance_fraction"] =
            neural["columns"][j]["importance_fraction"].clone();
        priority.push((j, score));
    }
    priority.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    tree["priority"] = json!(priority.iter().map(|v| v.0).collect::<Vec<_>>());
    tree["subsets"] = neural["subsets"].clone();
    tree["neural"] = json!({"training_r2":neural["training_r2"],"fit":neural["neural_fit"],
        "columns":neural["columns"],"pd_rows":neural["pd_rows"]});
    tree
}

/// Greedy subsets scored for total nonadditivity, including lower-order terms.
/// The extension score is a diagnostic difference, not a pure order-k H statistic.
fn subset_proposals(
    model: &Surrogate,
    rows: &Array2<f64>,
    singles: &[Vec<f64>],
    ranked: &[usize],
    pairs: &[(usize, usize, f64, f64)],
    scale: f64,
) -> Result<Vec<Value>, MboostrsError> {
    let mut parents: Vec<(Vec<usize>, f64)> = pairs
        .iter()
        .take(4)
        .map(|p| {
            let mut columns = vec![p.0, p.1];
            columns.sort_unstable();
            (columns, p.3)
        })
        .collect();
    let mut proposals = Vec::new();
    for size in 3..=ranked.len().min(5) {
        let mut candidates: BTreeMap<Vec<usize>, f64> = BTreeMap::new();
        for (parent, variance) in &parents {
            for &j in ranked {
                if !parent.contains(&j) {
                    let mut columns = parent.clone();
                    columns.push(j);
                    columns.sort_unstable();
                    candidates
                        .entry(columns)
                        .and_modify(|v| *v = v.max(*variance))
                        .or_insert(*variance);
                }
            }
        }
        let mut scored = Vec::new();
        for (columns, parent_variance) in candidates {
            let joint = pd(model, rows, &columns)?;
            let contrast: Vec<f64> = (0..joint.len())
                .map(|i| joint[i] - columns.iter().map(|j| singles[*j][i]).sum::<f64>())
                .collect();
            let interaction_variance = variance(&contrast) * scale * scale;
            let denominator = variance(&joint) * scale * scale;
            let h2 = if denominator > 1e-15 * scale * scale {
                (interaction_variance / denominator).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let extension = interaction_variance - parent_variance;
            proposals.push(json!({"indices":columns,"order":size,"h2_total":h2,
                "interaction_variance":interaction_variance,"extension_variance":extension,
                "source":"neural","nominated":h2>=0.05 && extension>0.01*scale*scale,
                "interpretation":"total nonadditivity; may contain lower-order interactions"}));
            scored.push((columns, interaction_variance));
        }
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        parents = scored.into_iter().take(4).collect();
    }
    Ok(proposals)
}

/// Fit one preliminary ensemble and return a portable report.
///
/// `x` is rows by columns; factors use integer codes `0..categorical[j]`,
/// with zero level count denoting numeric data. `transforms` describes the
/// incumbent's scale for shape diagnostics: 0 raw, 1 log, 2 square root.
/// Nonfinite predictors are imputed with a training median (factor: mode).
/// Response standardization stabilizes both engines and preserves H^2;
/// reported importance and interaction variance are restored to response units.
/// At most 1,000 rows measure permutation importance and 100 rows measure
/// partial dependence for at most 32 important columns. Other columns retain
/// every shape alternative. Pair enumeration is capped at eight columns.
/// Shape hints retain uncertainty: categorical, weak, transformed-domain and
/// marginal/partial-dependence disagreements never remove comparisons.
pub fn preexplore(
    x: ArrayView2<f64>,
    y: &[f64],
    categorical: &[usize],
    transforms: &[u8],
    engine: &str,
    seed: u64,
    rounds: u32,
    pairs: bool,
) -> Result<Value, MboostrsError> {
    if engine == "dual" {
        let (tree, neural) = rayon::join(
            || preexplore(x, y, categorical, transforms, "native", seed, rounds, pairs),
            || preexplore(x, y, categorical, transforms, "neural", seed, rounds, pairs),
        );
        return Ok(dual_report(tree?, neural));
    }
    let (n, p) = x.dim();
    if n < 20 || p == 0 || y.len() != n || categorical.len() != p || transforms.len() != p {
        return Err(MboostrsError::Shape("pre-exploration needs at least 20 rows, one column, and matching response/column metadata".into()));
    }
    if rounds == 0 || transforms.iter().any(|v| *v > 2) || y.iter().any(|v| !v.is_finite()) {
        return Err(MboostrsError::Control(
            "pre-exploration needs finite responses, positive rounds, and transforms 0, 1 or 2"
                .into(),
        ));
    }
    let mean = y.iter().sum::<f64>() / n as f64;
    let scale = variance(y).sqrt();
    if !scale.is_finite() || scale <= 0.0 {
        return Err(MboostrsError::Control(
            "pre-exploration response must have positive finite variance".into(),
        ));
    }
    let mut x = x.to_owned();
    for j in 0..p {
        let mut finite: Vec<f64> = x
            .column(j)
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .collect();
        finite.sort_by(f64::total_cmp);
        if finite.is_empty() {
            return Err(MboostrsError::Shape(
                "pre-exploration column has no finite values".into(),
            ));
        }
        if categorical[j] > 0
            && finite
                .iter()
                .any(|v| *v < 0.0 || v.fract() != 0.0 || *v >= categorical[j] as f64)
        {
            return Err(MboostrsError::Shape(
                "categorical codes must lie in 0..level_count".into(),
            ));
        }
        let fill = if categorical[j] > 0 {
            let mut counts = vec![0; categorical[j]];
            for v in &finite {
                counts[*v as usize] += 1;
            }
            counts
                .iter()
                .enumerate()
                .max_by_key(|(i, c)| (**c, std::cmp::Reverse(*i)))
                .unwrap()
                .0 as f64
        } else {
            finite[finite.len() / 2]
        };
        for v in x.column_mut(j) {
            if !v.is_finite() {
                *v = fill;
            }
        }
    }
    let y: Vec<f64> = y.iter().map(|v| (v - mean) / scale).collect();
    let model = fit(&x, &y, categorical, engine, seed, rounds)?;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut order: Vec<usize> = (0..n).collect();
    order.shuffle(&mut rng);
    let indices = &order[..n.min(1000)];
    let mut sample = x.select(Axis(0), indices);
    let ys: Vec<f64> = indices.iter().map(|i| y[*i]).collect();
    let base = mse(&ys, &model.predict(sample.view())?);
    let mut importance = Vec::with_capacity(p);
    for j in 0..p {
        let original = sample.column(j).to_owned();
        let mut perm = original.to_vec();
        perm.shuffle(&mut rng);
        sample.column_mut(j).assign(&ArrayView1::from(&perm));
        importance.push(mse(&ys, &model.predict(sample.view())?) - base);
        sample.column_mut(j).assign(&original);
    }
    let mut ranked: Vec<usize> = (0..p).collect();
    ranked.sort_by(|a, b| importance[*b].total_cmp(&importance[*a]).then(a.cmp(b)));
    let evaluated = &ranked[..p.min(32)];
    let rows = x.select(
        Axis(0),
        &order[..n.min(if engine == "neural" { 40 } else { 100 })],
    );
    let mut singles = Vec::with_capacity(p);
    let mut columns = Vec::with_capacity(p);
    for j in 0..p {
        let partial = if evaluated.contains(&j) {
            pd(&model, &rows, &[j])?
        } else {
            vec![0.0; rows.nrows()]
        };
        let transformed = |v: f64| match transforms[j] {
            1 => v.ln(),
            2 => v.sqrt(),
            _ => v,
        };
        let all: Vec<f64> = x.column(j).iter().copied().map(transformed).collect();
        let grid: Vec<f64> = rows.column(j).iter().copied().map(transformed).collect();
        let domain = all.iter().all(|v| v.is_finite());
        let marginal = if domain {
            Some(column_diagnostics(&all, &y)?)
        } else {
            None
        };
        let diagnostic = if domain && evaluated.contains(&j) {
            Some(column_diagnostics(&grid, &partial)?)
        } else {
            None
        };
        let mut shapes = vec!["bols", "bbs", "btree"];
        let mut reason = "ambiguous";
        // Conservative agreement: a tree approximates even a line by steps.
        // Require marginal evidence as well as a nearly linear PD to skip trees.
        if categorical[j] == 0 && engine != "neural" {
            if let (Some(m), Some(d)) = (&marginal, &diagnostic) {
                if importance[j] > 0.01 && variance(&partial) > 0.005 {
                    if d.linear_r2 > 0.9 && m.linear_r2 > 0.5 * m.curve_r2 {
                        shapes = vec!["bols", "bbs"];
                        reason = "linear partial dependence";
                    } else if d.curvature() > 0.1 && d.step_share() > 0.9 && m.step_share() > 0.8 {
                        shapes = vec!["bbs", "btree"];
                        reason = "step partial dependence";
                    } else if d.curvature() > 0.15
                        && d.step_share() < 0.65
                        && m.curvature() > 0.02
                        && m.step_share() < 0.65
                    {
                        shapes = vec!["bols", "bbs"];
                        reason = "smooth partial dependence";
                    }
                }
            }
        }
        columns.push(json!({"index":j,"importance":importance[j]*scale*scale,
            "importance_fraction":importance[j],"pd_variance":variance(&partial)*scale*scale,
            "shapes":shapes,"reason":reason,
            "marginal":marginal.as_ref().map(|d|json!({"linear_r2":d.linear_r2,"curve_r2":d.curve_r2,"step_share":d.step_share()})),
            "partial":diagnostic.as_ref().map(|d|json!({"linear_r2":d.linear_r2,"curve_r2":d.curve_r2,"step_share":d.step_share()}))}));
        singles.push(partial);
    }
    let mut pair_rows = Vec::new();
    if pairs {
        let top = &ranked[..p.min(8)];
        for (a, &j) in top.iter().enumerate() {
            for &k in &top[a + 1..] {
                let (h2, var) = h_statistic_with_variance(
                    &pd(&model, &rows, &[j, k])?,
                    &singles[j],
                    &singles[k],
                )?;
                pair_rows.push((j, k, h2, var * scale * scale));
            }
        }
        pair_rows.sort_by(|a, b| b.3.total_cmp(&a.3));
    }
    let subsets = if pairs && engine == "neural" {
        subset_proposals(
            &model,
            &rows,
            &singles,
            &ranked[..p.min(8)],
            &pair_rows,
            scale,
        )?
    } else {
        vec![]
    };
    let neural_fit = match &model {
        Surrogate::Neural(net) => net.metadata.clone(),
        _ => Value::Null,
    };
    Ok(json!({"engine":engine,"fits":1,"seed":seed,"rounds":rounds,
        "columns":columns,"priority":ranked,"pairs":pair_rows,
        "subsets":subsets,"neural_fit":neural_fit,
        "sample_rows":indices.len(),"pd_rows":rows.nrows(),
        "training_r2":1.0-base/variance(&ys)}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngExt;

    #[test]
    fn neural_probe_recovers_null_marginal_product_and_is_seeded() {
        let mut rng = ChaCha8Rng::seed_from_u64(15);
        let x = Array2::from_shape_fn((600, 3), |_| rng.random_range(-1.0..1.0));
        let y: Vec<f64> = x.rows().into_iter().map(|r| r[0] * r[1]).collect();
        let report = preexplore(x.view(), &y, &[0; 3], &[0; 3], "neural", 7, 250, true).unwrap();
        assert!(report["training_r2"].as_f64().unwrap() > 0.9, "{report}");
        let pair = &report["pairs"][0];
        let mut indices = vec![pair[0].as_u64().unwrap(), pair[1].as_u64().unwrap()];
        indices.sort_unstable();
        assert_eq!(indices, vec![0, 1]);
        assert!(pair[2].as_f64().unwrap() > 0.7);
        assert_eq!(
            report,
            preexplore(x.view(), &y, &[0; 3], &[0; 3], "neural", 7, 250, true).unwrap()
        );
        assert_eq!(
            report["columns"][0]["shapes"],
            json!(["bols", "bbs", "btree"])
        );
    }

    #[test]
    fn neural_probe_reports_greedy_three_way_nonadditivity() {
        let mut rng = ChaCha8Rng::seed_from_u64(18);
        let x = Array2::from_shape_fn((1000, 3), |_| rng.random_range(-1.0..1.0));
        let y: Vec<f64> = x.rows().into_iter().map(|r| r[0] * r[1] * r[2]).collect();
        let report = preexplore(x.view(), &y, &[0; 3], &[0; 3], "neural", 3, 350, true).unwrap();
        assert!(report["training_r2"].as_f64().unwrap() > 0.75, "{report}");
        assert_eq!(report["subsets"][0]["indices"], json!([0, 1, 2]));
        assert_eq!(report["subsets"][0]["nominated"], true);
    }

    #[test]
    fn dual_probe_retains_sources_and_charges_both_fits() {
        let (x, _) = line_data();
        let y: Vec<f64> = x.rows().into_iter().map(|r| r[0] * r[1]).collect();
        let report = preexplore(x.view(), &y, &[0; 2], &[0; 2], "dual", 2, 150, true).unwrap();
        assert_eq!(report["fits"], 2);
        assert_eq!(report["shape_source"], "native");
        assert!(report["pair_sources"][0]["sources"]
            .as_array()
            .unwrap()
            .contains(&json!("neural")));
        let fallback = dual_report(report, Err(MboostrsError::Control("unavailable".into())));
        assert!(fallback["neural_failure"]
            .as_str()
            .unwrap()
            .contains("unavailable"));
        assert!(!fallback["pairs"].as_array().unwrap().is_empty());
    }

    #[test]
    fn dual_merge_does_not_hide_a_valid_neural_nomination() {
        let tree = json!({"columns":[],"pairs":[[0,1,0.01,2.0]]});
        let neural = json!({"pairs":[[0,1,0.8,1.0]],"subsets":[]});
        let merged = dual_report(tree, Ok(neural));
        assert_eq!(merged["pairs"][0], json!([0, 1, 0.8, 1.0]));
        assert_eq!(merged["pair_sources"][0]["sources"], json!(["neural"]));
    }

    #[test]
    fn neural_categories_use_one_hot_inputs_and_enforce_width_limit() {
        let (mut x, _) = line_data();
        let y: Vec<f64> = (0..x.nrows())
            .map(|i| {
                x[[i, 1]] = (i % 3) as f64;
                x[[i, 0]] * if i % 3 == 0 { -1.0 } else { 1.0 }
            })
            .collect();
        let report = preexplore(x.view(), &y, &[0, 3], &[0, 0], "neural", 4, 150, true).unwrap();
        assert_eq!(report["neural_fit"]["encoded_inputs"], 4);
        assert!(report["training_r2"].as_f64().unwrap() > 0.85);
        assert!(preexplore(x.view(), &y, &[0, 300], &[0, 0], "neural", 4, 1, false).is_err());
    }

    fn line_data() -> (Array2<f64>, Vec<f64>) {
        let x = Array2::from_shape_fn((400, 2), |(i, j)| {
            if j == 0 {
                (i % 20) as f64 / 10.0 - 1.0
            } else {
                (i / 20) as f64 / 10.0 - 1.0
            }
        });
        let y = x.column(0).iter().map(|v| 3.0 * v).collect();
        (x, y)
    }

    #[test]
    fn native_exploration_recognizes_a_line_and_retains_ambiguous_noise() {
        let (x, y) = line_data();
        let report = preexplore(x.view(), &y, &[0, 0], &[0, 0], "native", 7, 60, true).unwrap();
        assert!(report["training_r2"].as_f64().unwrap() > 0.99);
        assert_eq!(report["priority"][0], 0);
        assert_eq!(report["columns"][0]["shapes"], json!(["bols", "bbs"]));
        assert_eq!(
            report["columns"][1]["shapes"],
            json!(["bols", "bbs", "btree"])
        );
        assert_eq!(
            report,
            preexplore(x.view(), &y, &[0, 0], &[0, 0], "native", 7, 60, true).unwrap()
        );
    }

    #[test]
    fn pair_proposals_find_a_product_with_no_marginal_signal() {
        let (x, _) = line_data();
        let y: Vec<f64> = x.rows().into_iter().map(|r| r[0] * r[1]).collect();
        let report = preexplore(x.view(), &y, &[0, 0], &[0, 0], "native", 2, 100, true).unwrap();
        assert!(report["pairs"][0][2].as_f64().unwrap() > 0.7);
        assert!(report["pairs"][0][3].as_f64().unwrap() > 0.03);
        // A pure interaction is never discarded for weak marginal association.
        assert_eq!(report["columns"][0]["shapes"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn transformed_scale_and_categorical_levels_remain_distinct() {
        let (mut x, mut y) = line_data();
        for i in 0..x.nrows() {
            x[[i, 0]] = (2.0 * x[[i, 0]]).exp();
            x[[i, 1]] = (i % 3) as f64;
            y[i] = x[[i, 0]].ln();
        }
        let report = preexplore(x.view(), &y, &[0, 3], &[1, 0], "native", 1, 70, false).unwrap();
        assert_eq!(report["columns"][0]["shapes"], json!(["bols", "bbs"]));
        assert_eq!(report["columns"][1]["reason"], "ambiguous");
        assert_eq!(report["pairs"], json!([]));
        x[[0, 1]] = 3.0;
        assert!(preexplore(x.view(), &y, &[0, 3], &[0, 0], "native", 0, 10, false).is_err());
    }

    #[test]
    fn missing_predictors_are_imputed_but_bad_controls_are_errors() {
        let (mut x, y) = line_data();
        x[[0, 0]] = f64::NAN;
        assert!(preexplore(x.view(), &y, &[0, 0], &[0, 0], "native", 0, 20, false).is_ok());
        assert!(preexplore(x.view(), &y[..10], &[0, 0], &[0, 0], "native", 0, 20, false).is_err());
        assert!(preexplore(x.view(), &y, &[0, 0], &[0, 0], "missing", 0, 20, false).is_err());
        assert!(preexplore(x.view(), &y, &[0, 0], &[0, 0], "native", 0, 0, false).is_err());
    }

    #[cfg(not(feature = "hessboost"))]
    #[test]
    fn optional_engine_is_explicit_when_not_compiled() {
        let (x, y) = line_data();
        assert!(
            preexplore(x.view(), &y, &[0, 0], &[0, 0], "hessboost", 0, 10, false)
                .unwrap_err()
                .to_string()
                .contains("Cargo feature")
        );
    }

    #[cfg(feature = "hessboost")]
    #[test]
    fn optional_engine_uses_the_same_report_and_shape_policy() {
        let (x, y) = line_data();
        let report = preexplore(x.view(), &y, &[0, 0], &[0, 0], "hessboost", 7, 60, true).unwrap();
        assert!(report["training_r2"].as_f64().unwrap() > 0.99);
        assert_eq!(report["columns"][0]["shapes"], json!(["bols", "bbs"]));
        assert_eq!(report["fits"], 1);
    }
}
