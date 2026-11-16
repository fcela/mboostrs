//! `df -> lambda` against R's `mboost:::df2lambda`.
//!
//! Fixture: `tests/fixtures/df_lambda.json`, written by
//! `tests/fixtures/generate_df_lambda_fixtures.R` (mboost 2.9.11, R 4.5.1).
//!
//! The grid runs down to `df` below the rank of the penalty's null space, which
//! is where the two implementations part company, and the tests are split at
//! the point where they stop agreeing.
//!
//! `df(lambda)` falls towards the rank of the null space as `lambda` grows and
//! never reaches it, so `lambda` behaves like `c / (df - rank)` near the floor:
//! it is identified, but its sensitivity to `c` — a function of the smallest
//! non-zero singular values of the penalty in the metric of the design — grows
//! without bound. R computes those singular values through LAPACK and this
//! crate through faer, and they agree to about 1e-11. Divided by `df - rank`,
//! that is a 1e-11 relative gap in `lambda` at `df - rank = 1`, 1e-7 at 1e-4,
//! and no agreement at all at the floor itself. No choice of root-finder
//! changes this; it is the conditioning of the problem being inverted.

use std::fs;
use std::path::PathBuf;

use mboostrs_core::{bspline_basis, bspline_knots, df_to_lambda, diff_penalty, kron};
use ndarray::{Array1, Array2};
use serde_json::Value;

fn fixture() -> Value {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/df_lambda.json");
    let text =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    serde_json::from_str(&text).expect("df_lambda.json must be valid JSON")
}

fn numbers(value: &Value, key: &str) -> Vec<f64> {
    value[key]
        .as_array()
        .unwrap_or_else(|| panic!("{key} must be an array"))
        .iter()
        .map(|v| v.as_f64().expect("numeric"))
        .collect()
}

/// Relative difference, with an absolute fallback at zero.
fn relative(a: f64, b: f64) -> f64 {
    let scale = a.abs().max(b.abs());
    if scale == 0.0 {
        0.0
    } else {
        (a - b).abs() / scale
    }
}

#[test]
fn df_to_lambda_matches_r_where_lambda_is_well_conditioned() {
    let document = fixture();
    let mut worst = 0.0f64;
    let mut worst_label = String::new();
    let mut compared = 0usize;

    for case in document["cases"].as_array().expect("cases array") {
        let (label, design, penalty, weights, null_rank) = unpack(case);
        for (df, expected) in numbers(case, "df").iter().zip(numbers(case, "lambda")) {
            if *df - null_rank < WELL_CONDITIONED_MARGIN {
                continue;
            }
            let actual = df_to_lambda(&design, &penalty, &weights, *df);
            let gap = relative(actual, expected);
            compared += 1;
            if gap > worst {
                worst = gap;
                worst_label =
                    format!("{label} at df={df}: mboostrs {actual:.12e}, R {expected:.12e}");
            }
        }
    }

    assert!(compared > 0, "fixture holds no well-conditioned df values");
    // Measured worst: 3.1e-7, on `clustered_200`, whose basis is close to
    // singular over the gap in its covariate.
    assert!(
        worst < 1e-6,
        "worst relative lambda gap {worst:.3e} over {compared} comparisons — {worst_label}"
    );
}

#[test]
fn df_between_the_rank_and_the_margin_agrees_as_well_as_conditioning_allows() {
    // `lambda` behaves like `c / (df - rank)` here, so a relative gap in `c`
    // arrives multiplied by `1 / (df - rank)`. Deflating the observed gap by
    // that factor recovers the underlying disagreement in the spectrum, which
    // is what stays bounded.
    let document = fixture();
    let mut worst = 0.0f64;
    let mut worst_label = String::new();
    let mut compared = 0usize;

    for case in document["cases"].as_array().expect("cases array") {
        let (label, design, penalty, weights, null_rank) = unpack(case);
        for (df, expected) in numbers(case, "df").iter().zip(numbers(case, "lambda")) {
            let above = *df - null_rank;
            if above <= 0.0 || above >= WELL_CONDITIONED_MARGIN {
                continue;
            }
            let actual = df_to_lambda(&design, &penalty, &weights, *df);
            let deflated = relative(actual, expected) * above;
            compared += 1;
            if deflated > worst {
                worst = deflated;
                worst_label = format!(
                    "{label} at df={df} (df - rank = {above:.1e}): \
                     mboostrs {actual:.6e}, R {expected:.6e}"
                );
            }
        }
    }

    assert!(compared > 0, "fixture holds no near-rank df values");
    // Measured worst: 1.7e-5, on `clustered_200` at `df - rank = 1e-4`.
    assert!(
        worst < 1e-4,
        "worst conditioning-deflated lambda gap {worst:.3e} over {compared} \
         comparisons — {worst_label}"
    );
}

#[test]
fn at_the_floor_the_learner_can_only_represent_its_null_space() {
    // At `df` equal to the rank of the penalty's null space, R's lambda and
    // this crate's differ by decades and both are arbitrary: no finite lambda
    // attains the target. The fitted values nonetheless agree, and it is worth
    // being precise about why, because a parity test pinned here passes for a
    // reason that has nothing to do with either implementation being right.
    //
    // What the learner returns at either lambda is the projection of the
    // response onto the penalty's null space, and nothing else. For a
    // second-difference penalty that is a straight line; for the Kronecker sum
    // of two of them it is the bilinear span {1, a, b, ab}. The restriction is
    // qualitative, not a matter of precision: a curved surface is not fitted
    // slightly worse at the floor, it is not fitted at all.
    let document = fixture();
    let mut checked = 0usize;

    for case in document["cases"].as_array().expect("cases array") {
        let (label, design, penalty, weights, null_rank) = unpack(case);
        let differences = case["differences"].as_u64().unwrap() as usize;
        let basis = null_space(case, design.ncols(), differences);
        let null_columns = design.dot(&basis);

        let response = curved_response(case);

        for (df, r_lambda) in numbers(case, "df").iter().zip(numbers(case, "lambda")) {
            if *df != null_rank {
                continue;
            }
            for (source, lambda) in [
                ("mboostrs", df_to_lambda(&design, &penalty, &weights, *df)),
                ("R", r_lambda),
            ] {
                let weighted = &design * &weights.view().insert_axis(ndarray::Axis(1));
                let gram = design.t().dot(&weighted) + &penalty * lambda;
                let coefficients = solve_spd(&gram, &design.t().dot(&response));
                let fitted = design.dot(&coefficients);
                // Measured worst: 1.3e-10, at R's lambda on `clustered_200`,
                // whose lambda at the floor is three decades below ours and so
                // leaves marginally more outside. Ours reach 1.9e-14.
                let outside = 1.0 - explained_share(&fitted, &null_columns);
                assert!(
                    outside < 1e-8,
                    "{label} at df={df} ({source} lambda {lambda:.3e}): the fit should lie \
                     in the penalty's null space, but {outside:.3e} of it lies outside"
                );
                checked += 1;
            }
        }
    }

    assert!(checked > 0, "fixture holds no df at a null-space rank");
}

#[test]
fn above_the_floor_the_learner_escapes_its_null_space() {
    // The counterpart to the test above, and what stops it being vacuous.
    //
    // The floor is exact and design-independent: at `df` equal to the rank, the
    // fit lies in the null space to machine precision on every design here.
    // How fast it escapes above the floor is neither. Share of the fit outside
    // the null space, half a degree of freedom up:
    //
    //   first_differences    6.0e-1
    //   tbill_rate           5.3e-2
    //   uniform_60           2.5e-3
    //   clustered_200        8.5e-5
    //
    // Four orders of magnitude, same offset. A design whose penalized spectrum
    // is concentrated — `clustered_200` puts 150 of its 200 points in one
    // narrow band — is shrunk far harder at the same `df` than one that spreads
    // it. So no number here is a usability threshold that transfers to another
    // design; the only claim that does is the floor itself. The assertion is
    // set from the worst design rather than the typical one for that reason.
    //
    // Sample size, by contrast, does not enter. `df` fixes the trace of the
    // smoother and `lambda` adapts to hold it there, so `lambda` grows in
    // proportion to `n` and the fit is unchanged. Measured on the tensor at
    // rank + 0.2, holding `df` fixed and letting `df_to_lambda` do its work:
    // n = 200 gives lambda 5.7e2 and 8.6e-4 outside, n = 2000 gives lambda
    // 5.6e3 and 1.0e-3 outside. Ten times the data, ten times the penalty, the
    // same fit. A pipeline where `n` does move the answer is varying something
    // other than `df` — a fixed `lambda`, or a fixed iteration count over a
    // boosted path, where realized flexibility is set by `mstop` and not by
    // `df` alone.
    let document = fixture();
    let mut checked = 0usize;

    for case in document["cases"].as_array().expect("cases array") {
        let (label, design, penalty, weights, null_rank) = unpack(case);
        let differences = case["differences"].as_u64().unwrap() as usize;
        let basis = null_space(case, design.ncols(), differences);
        let null_columns = design.dot(&basis);
        let response = curved_response(case);

        let Some(df) = numbers(case, "df")
            .into_iter()
            .find(|df| *df - null_rank >= 0.5)
        else {
            continue;
        };

        let lambda = df_to_lambda(&design, &penalty, &weights, df);
        let weighted = &design * &weights.view().insert_axis(ndarray::Axis(1));
        let gram = design.t().dot(&weighted) + &penalty * lambda;
        let coefficients = solve_spd(&gram, &design.t().dot(&response));
        let fitted = design.dot(&coefficients);
        let outside = 1.0 - explained_share(&fitted, &null_columns);
        assert!(
            outside > 1e-5,
            "{label} at df={df} (rank {null_rank} + {:.2}): the fit should carry structure \
             outside the null space, but only {outside:.3e} of it does",
            df - null_rank
        );
        checked += 1;
    }

    assert!(
        checked > 0,
        "fixture holds no df half a unit above a null-space rank"
    );
}

/// How far above the null-space rank `df` must sit for `lambda` to be
/// recoverable to better than a part in a million. Set from the fixture: at
/// `df - rank = 1e-2` the worst design agrees to 3.1e-7, at 1e-3 to 3.0e-4,
/// and at 1e-4 not at all.
const WELL_CONDITIONED_MARGIN: f64 = 1e-2;

/// Design, penalty and weights for one fixture case, plus the rank of the
/// penalty's null space.
///
/// A case carrying `a` and `b` is the two-margin tensor: its design is the
/// row-wise Kronecker product of the marginal bases and its penalty the
/// Kronecker sum of the marginal penalties, as `bspatial` and `%X%` build them.
fn unpack(case: &Value) -> (String, Array2<f64>, Array2<f64>, Array1<f64>, f64) {
    let label = case["label"].as_str().unwrap().to_string();
    let weights = Array1::from(numbers(case, "weights"));
    let n_knots = case["knots"].as_u64().unwrap() as usize;
    let degree = case["degree"].as_u64().unwrap() as usize;
    let differences = case["differences"].as_u64().unwrap() as usize;

    let (design, penalty, null_rank) = if case.get("a").is_some() {
        let a = numbers(case, "a");
        let b = numbers(case, "b");
        let upper_a = case["boundary_a"].as_f64().unwrap();
        let upper_b = case["boundary_b"].as_f64().unwrap();
        let b1 = bspline_basis(&a, &bspline_knots((0.0, upper_a), n_knots, degree), degree);
        let b2 = bspline_basis(&b, &bspline_knots((0.0, upper_b), n_knots, degree), degree);
        let (n, p1, p2) = (b1.nrows(), b1.ncols(), b2.ncols());
        let mut design = Array2::<f64>::zeros((n, p1 * p2));
        for i in 0..n {
            for u in 0..p1 {
                for v in 0..p2 {
                    design[[i, u * p2 + v]] = b1[[i, u]] * b2[[i, v]];
                }
            }
        }
        let k1 = diff_penalty(p1, differences);
        let k2 = diff_penalty(p2, differences);
        let penalty = kron(&k1, &Array2::<f64>::eye(p2)) + kron(&Array2::<f64>::eye(p1), &k2);
        (design, penalty, (differences * differences) as f64)
    } else {
        let x = numbers(case, "x");
        let boundary = (
            x.iter().copied().fold(f64::INFINITY, f64::min),
            x.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        );
        let design = bspline_basis(&x, &bspline_knots(boundary, n_knots, degree), degree);
        let penalty = diff_penalty(design.ncols(), differences);
        (design, penalty, differences as f64)
    };

    assert_eq!(
        design.ncols(),
        case["n_columns"].as_u64().unwrap() as usize,
        "{label}: basis width differs from R's"
    );
    assert_eq!(
        null_rank,
        case["null_space_rank"].as_f64().unwrap(),
        "{label}: null-space rank differs from the fixture's"
    );
    (label, design, penalty, weights, null_rank)
}

/// A response with structure in every covariate the design carries.
///
/// It has to be a function of the covariates, not of the row index: the
/// covariates are not sorted, and in the tensor case a response built from the
/// index carries no structure in the second margin at all, which understates
/// how much a learner just above the floor can pick up.
fn curved_response(case: &Value) -> Array1<f64> {
    if case.get("a").is_some() {
        let a = numbers(case, "a");
        let b = numbers(case, "b");
        Array1::from_shape_fn(a.len(), |i| (1.5 * a[i]).sin() * (1.5 * b[i]).cos())
    } else {
        let x = numbers(case, "x");
        let scale = {
            let lo = x.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if hi > lo {
                6.0 / (hi - lo)
            } else {
                1.0
            }
        };
        Array1::from_shape_fn(x.len(), |i| {
            (scale * x[i]).sin() + 0.5 * (1.7 * scale * x[i]).cos()
        })
    }
}

/// A basis for the penalty's null space, in coefficient space.
///
/// For a `d`-th difference penalty on `p` columns that is the polynomials of
/// degree below `d` in the coefficient index; for the Kronecker sum of two such
/// penalties it is the products of the marginal null spaces.
fn null_space(case: &Value, p: usize, differences: usize) -> Array2<f64> {
    let poly = |len: usize| -> Array2<f64> {
        Array2::from_shape_fn((len, differences), |(j, power)| {
            (j as f64).powi(power as i32)
        })
    };
    if case.get("a").is_some() {
        let p2 = ((p as f64).sqrt().round()) as usize;
        let p1 = p / p2;
        let (n1, n2) = (poly(p1), poly(p2));
        let mut out = Array2::<f64>::zeros((p, differences * differences));
        for u in 0..p1 {
            for v in 0..p2 {
                for c1 in 0..differences {
                    for c2 in 0..differences {
                        out[[u * p2 + v, c1 * differences + c2]] = n1[[u, c1]] * n2[[v, c2]];
                    }
                }
            }
        }
        out
    } else {
        poly(p)
    }
}

/// Solve the symmetric positive-definite system `a x = b` by Cholesky.
fn solve_spd(a: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let p = a.nrows();
    let mut l = Array2::<f64>::zeros((p, p));
    for i in 0..p {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                l[[i, j]] = sum.max(1e-300).sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }
    let mut y = Array1::<f64>::zeros(p);
    for i in 0..p {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[[i, k]] * y[k];
        }
        y[i] = sum / l[[i, i]];
    }
    let mut x = Array1::<f64>::zeros(p);
    for i in (0..p).rev() {
        let mut sum = y[i];
        for k in (i + 1)..p {
            sum -= l[[k, i]] * x[k];
        }
        x[i] = sum / l[[i, i]];
    }
    x
}

/// Share of `fitted` that a least-squares fit on `columns` reproduces.
///
/// The denominator is the fitted values, not the response, because the question
/// these tests ask is what the learner's output can contain, not how much of the
/// signal it recovered. That makes the ratio conservative for the floor test —
/// a small fit inflates it, so the test cannot pass spuriously — and it is
/// stable here: across the `df` range these tests use, the fitted norm moves
/// hardly at all within a design (`tbill_rate` 44.0 to 54.8 from the rank to a
/// full degree of freedom above it, `clustered_200` 110.4 to 110.4). A ratio
/// against the fit is only treacherous where the fit itself can collapse, which
/// is worth knowing before reusing this measure elsewhere: on the tensor the
/// fit carries under 1% of the response's variance, so the same quantity
/// against the response reads 2.5e-4 where against the fit it reads 3.2e-2.
fn explained_share(fitted: &Array1<f64>, columns: &Array2<f64>) -> f64 {
    let gram = columns.t().dot(columns);
    let rhs = columns.t().dot(fitted);
    let coefficients = solve_spd(&(gram + Array2::<f64>::eye(columns.ncols()) * 1e-10), &rhs);
    let approximation = columns.dot(&coefficients);
    let residual: f64 = (fitted - &approximation).iter().map(|v| v * v).sum();
    let total: f64 = fitted.iter().map(|v| v * v).sum();
    if total == 0.0 {
        1.0
    } else {
        1.0 - residual / total
    }
}
