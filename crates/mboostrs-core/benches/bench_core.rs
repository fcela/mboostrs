use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use mboostrs_core::{
    control::BoostControl,
    cv::{cv, try_cvrisk_selected_with, FoldSpec, FoldType, LearnerPreparer},
    families::{weighted_quantile, Binomial, Family, Gaussian, Poisson, PropOdds},
    fit, fit_with,
    learners::{
        df_to_lambda, prepare_bbs, prepare_binned_scalar_bols, prepare_block_bols,
        prepare_block_bols_shared_gram, prepare_bspatial_streamed, prepare_btree,
        prepare_scalar_bols, quantile_bin_column, LearnerBlueprint, PreparedLearner, SplitMethod,
    },
    linalg::CachedLu,
    try_fit_profiled, CachedCholesky, PreparationTimer,
};
use ndarray::{Array1, Array2, Axis};
use rayon::prelude::*;
use std::hint::black_box;

#[cfg(feature = "instrumentation-baseline")]
use mboostrs_core::try_fit_without_report;

fn xorshift_rng(seed: u64) -> impl FnMut() -> f64 {
    let mut rng = seed;
    move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng as f64) / (u64::MAX as f64)
    }
}

fn make_glmboost_data(n: usize, j: usize) -> (Vec<Array1<f64>>, Array1<f64>, Array1<f64>) {
    let mut rng = xorshift_rng(42);
    let xs: Vec<Array1<f64>> = (0..j)
        .map(|_| (0..n).map(|_| rng() * 2.0 - 1.0).collect())
        .collect();
    let true_betas: Vec<f64> = (0..j).map(|i| 1.5 - (i as f64) * 0.3).collect();
    let y: Array1<f64> = (0..n)
        .map(|i| {
            let mut mu = 0.0;
            for k in 0..j {
                mu += true_betas[k] * xs[k][i];
            }
            mu + (rng() - 0.5) * 0.6
        })
        .collect();
    let w = Array1::ones(n);
    (xs, y, w)
}

fn make_gamboost_data(n: usize, j: usize) -> (Vec<Array1<f64>>, Array1<f64>, Array1<f64>) {
    let mut rng = xorshift_rng(42);
    let xs: Vec<Array1<f64>> = (0..j).map(|_| (0..n).map(|_| rng()).collect()).collect();
    let two_pi = std::f64::consts::TAU;
    let y: Array1<f64> = (0..n)
        .map(|i| {
            let mu: f64 = xs
                .iter()
                .enumerate()
                .map(|(k, x)| (two_pi * x[i] * (k as f64 + 1.0)).sin())
                .sum();
            mu + (rng() - 0.5) * 0.4
        })
        .collect();
    let w = Array1::ones(n);
    (xs, y, w)
}

fn prepare_scalar_learners(xs: &[Array1<f64>], w: &Array1<f64>) -> Vec<PreparedLearner> {
    xs.iter()
        .enumerate()
        .map(|(k, x)| prepare_scalar_bols(x.view(), w.view(), 0.0, true, &format!("x{k}"), None))
        .collect()
}

fn prepare_binned_scalar_learners(
    xs: &[Array1<f64>],
    w: &Array1<f64>,
    bins: usize,
) -> Vec<PreparedLearner> {
    xs.iter()
        .enumerate()
        .map(|(index, x)| {
            prepare_binned_scalar_bols(
                quantile_bin_column(x.view(), bins).unwrap(),
                w.view(),
                0.0,
                true,
                &format!("x{index}"),
            )
        })
        .collect()
}

fn prepare_bbs_learners(xs: &[Array1<f64>], w: &Array1<f64>) -> Vec<PreparedLearner> {
    xs.iter()
        .enumerate()
        .map(|(k, x)| {
            prepare_bbs(
                x.view(),
                w.view(),
                20,
                3,
                2,
                4.0,
                true,
                &format!("bbs(x{k})"),
            )
        })
        .collect()
}

fn prepare_binned_bbs_learners(
    xs: &[Array1<f64>],
    weights: &Array1<f64>,
    bins: usize,
) -> Vec<PreparedLearner> {
    xs.iter()
        .enumerate()
        .map(|(index, x)| {
            LearnerBlueprint::bbs_binned(
                x.clone(),
                format!("bbs(x{index})"),
                20,
                3,
                2,
                4.0,
                None,
                true,
                None,
                false,
                None,
                0,
                bins,
            )
            .unwrap()
            .prepare(weights.view())
            .unwrap()
            .0
        })
        .collect()
}

struct ScalarBenchPreparer {
    xs: Vec<Array1<f64>>,
    names: Vec<String>,
}

impl LearnerPreparer for ScalarBenchPreparer {
    fn prepare(&self, weights: ndarray::ArrayView1<f64>) -> Vec<PreparedLearner> {
        self.xs
            .iter()
            .zip(&self.names)
            .map(|(x, name)| prepare_scalar_bols(x.view(), weights, 0.0, true, name, None))
            .collect()
    }

    fn family(&self) -> Box<dyn Family> {
        Box::new(Gaussian)
    }
}

struct BbsBenchPreparer {
    xs: Vec<Array1<f64>>,
    names: Vec<String>,
}

impl LearnerPreparer for BbsBenchPreparer {
    fn prepare(&self, weights: ndarray::ArrayView1<f64>) -> Vec<PreparedLearner> {
        self.xs
            .iter()
            .zip(&self.names)
            .map(|(x, name)| prepare_bbs(x.view(), weights, 20, 3, 2, 4.0, true, name))
            .collect()
    }

    fn family(&self) -> Box<dyn Family> {
        Box::new(Gaussian)
    }
}

fn bench_prepare_learners(c: &mut Criterion) {
    let mut group = c.benchmark_group("setup_prepare_learners");

    for &(n, j) in &[(1_000, 10), (10_000, 10)] {
        let (xs, _y, w) = make_glmboost_data(n, j);
        group.bench_function(format!("scalar_n={n}_j={j}"), |b| {
            b.iter(|| black_box(prepare_scalar_learners(black_box(&xs), black_box(&w))))
        });
    }

    for &(n, j) in &[(1_000, 2), (10_000, 2), (10_000, 10)] {
        let (xs, _y, w) = make_gamboost_data(n, j);
        group.bench_function(format!("bbs_n={n}_j={j}"), |b| {
            b.iter(|| black_box(prepare_bbs_learners(black_box(&xs), black_box(&w))))
        });
    }

    let n = 5_000;
    let mut rng = xorshift_rng(42);
    let x1: Array1<f64> = (0..n).map(|_| rng()).collect();
    let x2: Array1<f64> = (0..n).map(|_| rng()).collect();
    let design = ndarray::stack(Axis(1), &[x1.view(), x2.view()]).unwrap();
    let w = Array1::ones(n);
    group.bench_function("btree_n=5k_p=2", |b| {
        b.iter(|| {
            black_box(prepare_btree(
                black_box(design.view()),
                black_box(w.view()),
                7,
                7,
                1,
                None,
                "btree(x1,x2)",
            ))
        })
    });

    group.finish();
}

fn bench_cvrisk_selected(c: &mut Criterion) {
    let mut group = c.benchmark_group("cvrisk_selected");

    let n = 1_000;
    let b = 20;
    let ctrl = BoostControl::new(50, 0.1).unwrap();

    let (xs, y, w) = make_glmboost_data(n, 10);
    let folds = cv(
        &FoldSpec::new(FoldType::Subsampling, b, 0.5, 42),
        n,
        w.as_slice().unwrap(),
    );
    let scalar_preparer = ScalarBenchPreparer {
        names: (0..xs.len()).map(|k| format!("x{k}")).collect(),
        xs,
    };
    group.bench_function("scalar_n=1k_j=10_b=20_mstop=50", |bench| {
        bench.iter(|| {
            try_cvrisk_selected_with(
                black_box(&scalar_preparer),
                y.view(),
                w.view(),
                &ctrl,
                black_box(&folds),
                None,
                Gaussian,
            )
            .unwrap()
        })
    });

    let (xs, y, w) = make_gamboost_data(n, 4);
    let folds = cv(
        &FoldSpec::new(FoldType::Subsampling, b, 0.5, 42),
        n,
        w.as_slice().unwrap(),
    );
    let bbs_preparer = BbsBenchPreparer {
        names: (0..xs.len()).map(|k| format!("bbs(x{k})")).collect(),
        xs,
    };
    for mstop in [10, 50] {
        let bbs_control = BoostControl::new(mstop, 0.1).unwrap();
        group.bench_function(format!("bbs_n=1k_j=4_b=20_mstop={mstop}"), |bench| {
            bench.iter(|| {
                try_cvrisk_selected_with(
                    black_box(&bbs_preparer),
                    y.view(),
                    w.view(),
                    &bbs_control,
                    black_box(&folds),
                    None,
                    Gaussian,
                )
                .unwrap()
            })
        });
    }

    group.finish();
}

fn bench_glmboost_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("glmboost_scalar");

    for &(n, j, mstop) in &[(1_000, 10, 100), (10_000, 10, 100), (50_000, 2, 100)] {
        let (xs, y, w) = make_glmboost_data(n, j);
        let ctrl = BoostControl::new(mstop, 0.1).unwrap();
        let learners_template = prepare_scalar_learners(&xs, &w);

        group.bench_function(BenchmarkId::new(format!("n={n}_j={j}"), mstop), |b| {
            b.iter_batched(
                || learners_template.clone(),
                |mut learners| {
                    let mut fam = Gaussian;
                    fit(
                        black_box(&mut learners),
                        y.view(),
                        w.view(),
                        black_box(&mut fam),
                        &ctrl,
                        None,
                    )
                },
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

fn bench_binned_scalar(c: &mut Criterion) {
    let (n, learners, mstop) = (100_000, 50, 25);
    let (xs, y, weights) = make_glmboost_data(n, learners);
    let control = BoostControl::new(mstop, 0.1).unwrap();
    let exact_template = prepare_scalar_learners(&xs, &weights);
    let binned_100 = prepare_binned_scalar_learners(&xs, &weights, 100);
    let binned_400 = prepare_binned_scalar_learners(&xs, &weights, 400);
    let mut group = c.benchmark_group("binned_scalar");
    group.sample_size(10);
    for (label, template) in [
        ("exact", exact_template),
        ("K=100", binned_100),
        ("K=400", binned_400),
    ] {
        group.bench_function(label, |bench| {
            bench.iter_batched(
                || template.clone(),
                |mut prepared| {
                    let mut family = Gaussian;
                    fit(
                        black_box(&mut prepared),
                        y.view(),
                        weights.view(),
                        black_box(&mut family),
                        &control,
                        None,
                    )
                },
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

fn bench_gamboost_bbs(c: &mut Criterion) {
    let mut group = c.benchmark_group("gamboost_bbs");

    for &(n, j, mstop) in &[(1_000, 2, 100), (10_000, 2, 200), (10_000, 10, 200)] {
        let (xs, y, w) = make_gamboost_data(n, j);
        let ctrl = BoostControl::new(mstop, 0.1).unwrap();
        let learners_template = prepare_bbs_learners(&xs, &w);

        group.bench_function(BenchmarkId::new(format!("n={n}_j={j}"), mstop), |b| {
            b.iter_batched(
                || learners_template.clone(),
                |mut learners| {
                    let mut fam = Gaussian;
                    fit(
                        black_box(&mut learners),
                        y.view(),
                        w.view(),
                        black_box(&mut fam),
                        &ctrl,
                        None,
                    )
                },
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

fn bench_binned_bbs(c: &mut Criterion) {
    let (n, learners, mstop) = (100_000, 10, 25);
    let (xs, y, weights) = make_gamboost_data(n, learners);
    let control = BoostControl::new(mstop, 0.1).unwrap();
    let exact_template = prepare_bbs_learners(&xs, &weights);
    let binned_100 = prepare_binned_bbs_learners(&xs, &weights, 100);
    let binned_400 = prepare_binned_bbs_learners(&xs, &weights, 400);
    let mut group = c.benchmark_group("binned_bbs");
    group.sample_size(10);
    for (label, template) in [
        ("exact", exact_template),
        ("K=100", binned_100),
        ("K=400", binned_400),
    ] {
        group.bench_function(label, |bench| {
            bench.iter_batched(
                || template.clone(),
                |mut prepared| {
                    let mut family = Gaussian;
                    fit(
                        black_box(&mut prepared),
                        y.view(),
                        weights.view(),
                        black_box(&mut family),
                        &control,
                        None,
                    )
                },
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

/// Benchmark: fit-only (Gram path) vs fit-only (general path) for gamboost
/// bbs learners, across j=2 and j=10, to measure the Gram speedup and the
/// crossover guard impact. Setup is excluded (learners are pre-prepared).
fn bench_gamboost_gram_vs_general(c: &mut Criterion) {
    for &(n, j, mstop) in &[(10_000, 2, 200), (10_000, 10, 200), (10_000, 10, 500)] {
        let (xs, y, w) = make_gamboost_data(n, j);
        let ctrl = BoostControl::new(mstop, 0.1).unwrap();
        let learners_template = prepare_bbs_learners(&xs, &w);

        let mut group = c.benchmark_group(format!("gram_vs_general/j={j}"));

        group.bench_function(format!("gram_mstop={mstop}"), |b| {
            b.iter_batched(
                || learners_template.clone(),
                |mut learners| {
                    let mut fam = Gaussian;
                    fit_with(
                        black_box(&mut learners),
                        y.view(),
                        w.view(),
                        black_box(&mut fam),
                        &ctrl,
                        None,
                        false,
                    )
                },
                BatchSize::SmallInput,
            )
        });

        group.bench_function(format!("general_mstop={mstop}"), |b| {
            b.iter_batched(
                || learners_template.clone(),
                |mut learners| {
                    let mut fam = Gaussian;
                    fit(
                        black_box(&mut learners),
                        y.view(),
                        w.view(),
                        black_box(&mut fam),
                        &ctrl,
                        None,
                    )
                },
                BatchSize::SmallInput,
            )
        });

        group.finish();
    }
}

fn bench_binomial(c: &mut Criterion) {
    let n = 10_000;
    let mut rng = xorshift_rng(42);
    let x: Array1<f64> = (0..n).map(|_| rng()).collect();
    let w = Array1::ones(n);
    let y: Array1<f64> = (0..n).map(|i| if x[i] > 0.5 { 1.0 } else { 0.0 }).collect();
    let ctrl = BoostControl::new(100, 0.1).unwrap();
    let learners_template = vec![prepare_bbs(
        x.view(),
        w.view(),
        20,
        3,
        2,
        4.0,
        true,
        "bbs(x)",
    )];

    let mut group = c.benchmark_group("binomial");
    group.bench_function("bbs_n10k", |b| {
        b.iter_batched(
            || learners_template.clone(),
            |mut learners| {
                let mut fam = Binomial;
                fit(
                    black_box(&mut learners),
                    y.view(),
                    w.view(),
                    black_box(&mut fam),
                    &ctrl,
                    None,
                )
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

fn bench_poisson(c: &mut Criterion) {
    let n = 10_000;
    let mut rng = xorshift_rng(42);
    let x: Array1<f64> = (0..n).map(|_| rng()).collect();
    let w = Array1::ones(n);
    let y: Array1<f64> = (0..n).map(|_| (rng() * 10.0).floor()).collect();
    let ctrl = BoostControl::new(100, 0.1).unwrap();
    let learners_template = vec![prepare_bbs(
        x.view(),
        w.view(),
        20,
        3,
        2,
        4.0,
        true,
        "bbs(x)",
    )];

    let mut group = c.benchmark_group("poisson");
    group.bench_function("bbs_n10k", |b| {
        b.iter_batched(
            || learners_template.clone(),
            |mut learners| {
                let mut fam = Poisson;
                fit(
                    black_box(&mut learners),
                    y.view(),
                    w.view(),
                    black_box(&mut fam),
                    &ctrl,
                    None,
                )
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

fn bench_btree(c: &mut Criterion) {
    let n = 5_000;
    let mut rng = xorshift_rng(42);
    let x1: Array1<f64> = (0..n).map(|_| rng()).collect();
    let x2: Array1<f64> = (0..n).map(|_| rng()).collect();
    let design = ndarray::stack(Axis(1), &[x1.view(), x2.view()]).unwrap();
    let w = Array1::ones(n);
    let y: Array1<f64> = (0..n)
        .map(|i| {
            let v = x1[i] + x2[i];
            if v > 1.0 {
                2.0
            } else {
                -1.0
            }
        })
        .collect();
    let ctrl = BoostControl::new(50, 0.1).unwrap();
    let learners_template = vec![prepare_btree(
        design.view(),
        w.view(),
        7,
        7,
        1,
        None,
        "btree(x1,x2)",
    )];

    let mut group = c.benchmark_group("btree");
    group.bench_function("n5k_mstop50", |b| {
        b.iter_batched(
            || learners_template.clone(),
            |mut learners| {
                let mut fam = Gaussian;
                fit(
                    black_box(&mut learners),
                    y.view(),
                    w.view(),
                    black_box(&mut fam),
                    &ctrl,
                    None,
                )
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

/// Exact split scanning versus histogram split finding for a depth-6 btree, at
/// n=10k and n=100k. Preparation is excluded; only the boosting fit is timed.
/// The histogram path replaces the exact scan's random gathered candidates
/// (through the sort-index indirection) with a sequential scatter into
/// L1-resident bins, so its per-column cost is far lower and its scaling is
/// near-linear; the fixed per-node passes (node totals, marking, row
/// partition) are shared by both methods.
fn bench_btree_split_method(c: &mut Criterion) {
    let mut group = c.benchmark_group("btree_split_method");
    group.sample_size(10);
    for &n in &[10_000usize, 100_000] {
        let p = 4usize;
        let mut rng = xorshift_rng(24680);
        let data: Vec<f64> = (0..n * p).map(|_| rng()).collect();
        let design = Array2::from_shape_vec((n, p), data).unwrap();
        let y: Array1<f64> = Array1::from_shape_fn(n, |i| {
            let mut mu = 0.0;
            for j in 0..p {
                mu += (design[[i, j]] * 3.0).sin();
            }
            mu + 0.1 * (rng() - 0.5)
        });
        let w = Array1::ones(n);
        let control = BoostControl::new(30, 0.1).unwrap();
        for (label, method) in [
            ("exact", SplitMethod::Exact),
            ("histogram", SplitMethod::Histogram),
        ] {
            let template = vec![
                LearnerBlueprint::btree(design.clone(), "btree", 5, 20, 6, None)
                    .unwrap()
                    .with_split_method(method)
                    .prepare(w.view())
                    .unwrap()
                    .0,
            ];
            group.bench_function(format!("n={n}_{label}"), |b| {
                b.iter_batched(
                    || template.clone(),
                    |mut learners| {
                        let mut family = Gaussian;
                        fit(
                            black_box(&mut learners),
                            y.view(),
                            w.view(),
                            black_box(&mut family),
                            &control,
                            None,
                        )
                    },
                    BatchSize::SmallInput,
                )
            });
        }
    }
    group.finish();
}

/// `bspatial` setup (#284): the old dense tail (`df_to_lambda` +
/// `prepare_block_bols`) pays two O(n·p²) Gram passes; the shared tail
/// (`X'WX + λK`) pays one. The `streamed_*` entries additionally use the
/// marginal-eigen centering and one-eig / pencil df→λ calibrations, which
/// replace the streamed Gram's full O(p³) spectral-centering chain with two
/// marginal eigensolves, four mode contractions and one eigenproblem.
fn bench_bspatial_setup(c: &mut Criterion) {
    let mut group = c.benchmark_group("bspatial_setup");
    group.sample_size(10);
    let n = 10_000usize;
    let mut rng = xorshift_rng(13579);
    let x1 = Array1::from_iter((0..n).map(|_| rng()));
    let x2 = Array1::from_iter((0..n).map(|_| rng()));
    let w = Array1::ones(n);
    let blueprint = LearnerBlueprint::bspatial(
        x1.clone(),
        x2.clone(),
        "spatial",
        20,
        3,
        2,
        6.0,
        None,
        false,
        None,
        None,
        None,
    )
    .unwrap();
    let (design, penalty) = match &blueprint {
        LearnerBlueprint::Bspatial {
            design, penalty, ..
        } => (design.clone(), penalty.clone()),
        _ => unreachable!(),
    };

    group.bench_function("n=10000_dense_old", |b| {
        b.iter_batched(
            || (design.clone(), penalty.clone()),
            |(design, penalty)| {
                let lambda = df_to_lambda(&design, &penalty, &w, 6.0);
                black_box(prepare_block_bols(
                    design,
                    w.view(),
                    penalty.mapv(|v| v * lambda),
                    false,
                    "spatial",
                ))
            },
            BatchSize::LargeInput,
        )
    });
    group.bench_function("n=10_000_shared_new", |b| {
        b.iter_batched(
            || (design.clone(), penalty.clone()),
            |(design, penalty)| {
                prepare_block_bols_shared_gram(design, w.view(), penalty, 6.0, None, "spatial")
            },
            BatchSize::LargeInput,
        )
    });
    group.bench_function("n=10_000_full_prepare", |b| {
        b.iter(|| blueprint.prepare(w.view()).unwrap())
    });
    group.bench_function("n=10_000_streamed_new", |b| {
        b.iter(|| {
            prepare_bspatial_streamed(
                x1.view(),
                x2.view(),
                w.view(),
                20,
                3,
                2,
                6.0,
                None,
                false,
                "spatial",
                None,
            )
        })
    });
    group.bench_function("n=10_000_streamed_full_prepare", |b| {
        b.iter(|| {
            prepare_bspatial_streamed(
                x1.view(),
                x2.view(),
                w.view(),
                20,
                3,
                2,
                6.0,
                None,
                true,
                "spatial",
                None,
            )
        })
    });
    group.finish();
}

fn bench_propodds(c: &mut Criterion) {
    let n = 5_000;
    let mut rng = xorshift_rng(42);
    let x: Array1<f64> = (0..n).map(|_| rng()).collect();
    let w = Array1::ones(n);
    let y: Array1<f64> = (0..n)
        .map(|i| {
            let u = rng();
            if x[i] < 0.33 {
                if u < 0.6 {
                    1.0
                } else if u < 0.9 {
                    2.0
                } else {
                    3.0
                }
            } else {
                if u < 0.3 {
                    1.0
                } else if u < 0.7 {
                    2.0
                } else {
                    3.0
                }
            }
        })
        .collect();
    let ctrl = BoostControl::new(50, 0.1).unwrap();
    let learners_template = vec![prepare_bbs(
        x.view(),
        w.view(),
        20,
        3,
        2,
        4.0,
        true,
        "bbs(x)",
    )];

    let mut group = c.benchmark_group("propodds");
    group.bench_function("bbs_n5k", |b| {
        b.iter_batched(
            || learners_template.clone(),
            |mut learners| {
                let mut fam = PropOdds::new(3);
                fit(
                    black_box(&mut learners),
                    y.view(),
                    w.view(),
                    black_box(&mut fam),
                    &ctrl,
                    None,
                )
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

fn bench_many_learners(c: &mut Criterion) {
    let mut group = c.benchmark_group("many_learners");

    for &j in &[2, 10, 50] {
        let n = 1_000;
        let (xs, y, w) = make_glmboost_data(n, j);
        let ctrl = BoostControl::new(100, 0.1).unwrap();
        let learners_template = prepare_scalar_learners(&xs, &w);

        group.bench_function(format!("j={j}_scalar"), |b| {
            b.iter_batched(
                || learners_template.clone(),
                |mut learners| {
                    let mut fam = Gaussian;
                    fit(
                        black_box(&mut learners),
                        y.view(),
                        w.view(),
                        black_box(&mut fam),
                        &ctrl,
                        None,
                    )
                },
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

fn bench_block_selection_crossover(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_selection_crossover");
    for j in [2, 3, 4, 6, 8, 10, 16, 24, 32, 50] {
        let (xs, y, weights) = make_gamboost_data(2_000, j);
        let learners_template = prepare_bbs_learners(&xs, &weights);
        let offset = y.mean().unwrap_or(0.0);
        let residual = y.mapv(|value| value - offset);
        let wu2 = residual
            .iter()
            .zip(&weights)
            .map(|(&value, &weight)| weight * value * value)
            .sum();

        group.bench_function(format!("bbs_sequential_j={j}"), |bench| {
            bench.iter_batched(
                || learners_template.clone(),
                |mut learners| {
                    black_box(
                        learners
                            .iter_mut()
                            .enumerate()
                            .map(|(index, learner)| {
                                (
                                    index,
                                    learner.fit_step_select_sse(
                                        residual.view(),
                                        weights.view(),
                                        wu2,
                                    ),
                                )
                            })
                            .min_by(|left, right| {
                                left.1
                                    .partial_cmp(&right.1)
                                    .unwrap_or(std::cmp::Ordering::Greater)
                                    .then_with(|| left.0.cmp(&right.0))
                            }),
                    )
                },
                BatchSize::SmallInput,
            )
        });
        group.bench_function(format!("bbs_parallel_score_only_j={j}"), |bench| {
            bench.iter_batched(
                || learners_template.clone(),
                |mut learners| {
                    black_box(
                        learners
                            .par_iter_mut()
                            .enumerate()
                            .map(|(index, learner)| {
                                (
                                    index,
                                    learner.fit_step_select_sse(
                                        residual.view(),
                                        weights.view(),
                                        wu2,
                                    ),
                                )
                            })
                            .reduce_with(|left, right| {
                                if right.1 < left.1 || (right.1 == left.1 && right.0 < left.0) {
                                    right
                                } else {
                                    left
                                }
                            }),
                    )
                },
                BatchSize::SmallInput,
            )
        });
        group.bench_function(format!("bbs_parallel_legacy_clone_j={j}"), |bench| {
            bench.iter_batched(
                || learners_template.clone(),
                |mut learners| {
                    black_box(
                        learners
                            .par_iter_mut()
                            .enumerate()
                            .map(|(index, learner)| {
                                let (beta, sse) =
                                    learner.fit_step_select(residual.view(), weights.view(), wu2);
                                (index, sse, beta.to_vec())
                            })
                            .reduce_with(|left, right| {
                                if right.1 < left.1 || (right.1 == left.1 && right.0 < left.0) {
                                    right
                                } else {
                                    left
                                }
                            }),
                    )
                },
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

fn bench_non_gaussian_bbs_crossover(c: &mut Criterion) {
    let mut group = c.benchmark_group("non_gaussian_bbs_crossover");
    for j in [2, 3, 10, 50] {
        let (xs, _, weights) = make_gamboost_data(2_000, j);
        let y = Array1::from_shape_fn(2_000, |row| {
            (1.0 + 3.0 * xs[0][row] + 2.0 * xs[j - 1][row]).floor()
        });
        let learners_template = prepare_bbs_learners(&xs, &weights);
        let control = BoostControl::new(100, 0.1).unwrap();
        group.bench_function(format!("poisson_j={j}"), |bench| {
            bench.iter_batched(
                || learners_template.clone(),
                |mut learners| {
                    fit_with(
                        black_box(&mut learners),
                        y.view(),
                        weights.view(),
                        black_box(&mut Poisson),
                        &control,
                        None,
                        false,
                    )
                },
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

/// Compare report-free, clock-free report, and summary timing on representative
/// packed-general and Gaussian-Gram macro fits, plus learner preparation.
/// Learner cloning is batched as setup and excluded from fit measurements.
fn bench_instrumentation_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("instrumentation_overhead");
    group.sample_size(30);

    let (scalar_xs, scalar_signal, scalar_w) = make_glmboost_data(10_000, 10);
    // Poisson deliberately exercises the packed general engine rather than the
    // Gaussian Gram route benchmarked below.
    let scalar_y = scalar_signal.mapv(|value| value.exp().round());
    let scalar_template = prepare_scalar_learners(&scalar_xs, &scalar_w);
    let scalar_control = BoostControl::new(200, 0.1).unwrap();
    for (label, summary) in [("packed_off", false), ("packed_summary", true)] {
        group.bench_function(label, |b| {
            b.iter_batched(
                || scalar_template.clone(),
                |mut learners| {
                    let mut family = Poisson;
                    if summary {
                        try_fit_profiled(
                            black_box(&mut learners),
                            scalar_y.view(),
                            scalar_w.view(),
                            black_box(&mut family),
                            &scalar_control,
                            None,
                            false,
                        )
                        .unwrap()
                    } else {
                        fit_with(
                            black_box(&mut learners),
                            scalar_y.view(),
                            scalar_w.view(),
                            black_box(&mut family),
                            &scalar_control,
                            None,
                            false,
                        )
                    }
                },
                BatchSize::SmallInput,
            )
        });
    }
    #[cfg(feature = "instrumentation-baseline")]
    group.bench_function("packed_no_report_baseline", |b| {
        b.iter_batched(
            || scalar_template.clone(),
            |mut learners| {
                let mut family = Poisson;
                try_fit_without_report(
                    black_box(&mut learners),
                    scalar_y.view(),
                    scalar_w.view(),
                    black_box(&mut family),
                    &scalar_control,
                    None,
                )
                .unwrap()
            },
            BatchSize::SmallInput,
        )
    });

    let (gram_xs, gram_y, gram_w) = make_gamboost_data(10_000, 10);
    let gram_template = prepare_bbs_learners(&gram_xs, &gram_w);
    let gram_control = BoostControl::new(200, 0.1).unwrap();
    for (label, summary) in [("gram_off", false), ("gram_summary", true)] {
        group.bench_function(label, |b| {
            b.iter_batched(
                || gram_template.clone(),
                |mut learners| {
                    let mut family = Gaussian;
                    if summary {
                        try_fit_profiled(
                            black_box(&mut learners),
                            gram_y.view(),
                            gram_w.view(),
                            black_box(&mut family),
                            &gram_control,
                            None,
                            false,
                        )
                        .unwrap()
                    } else {
                        fit_with(
                            black_box(&mut learners),
                            gram_y.view(),
                            gram_w.view(),
                            black_box(&mut family),
                            &gram_control,
                            None,
                            false,
                        )
                    }
                },
                BatchSize::SmallInput,
            )
        });
    }
    #[cfg(feature = "instrumentation-baseline")]
    group.bench_function("gram_no_report_baseline", |b| {
        b.iter_batched(
            || gram_template.clone(),
            |mut learners| {
                let mut family = Gaussian;
                try_fit_without_report(
                    black_box(&mut learners),
                    gram_y.view(),
                    gram_w.view(),
                    black_box(&mut family),
                    &gram_control,
                    None,
                )
                .unwrap()
            },
            BatchSize::SmallInput,
        )
    });

    let (preparation_xs, _preparation_y, preparation_w) = make_gamboost_data(2_000, 4);
    for (label, summary) in [("preparation_off", false), ("preparation_summary", true)] {
        group.bench_function(label, |b| {
            b.iter(|| {
                if summary {
                    let timer = PreparationTimer::start();
                    let learners =
                        prepare_bbs_learners(black_box(&preparation_xs), black_box(&preparation_w));
                    black_box((learners, timer.finish()))
                } else {
                    black_box((
                        prepare_bbs_learners(black_box(&preparation_xs), black_box(&preparation_w)),
                        Default::default(),
                    ))
                }
            })
        });
    }
    group.finish();
}

fn bench_cached_solvers(c: &mut Criterion) {
    let dimension = 64;
    let mut random = xorshift_rng(20260715);
    let design = Array2::from_shape_fn((dimension, dimension), |_| random() * 2.0 - 1.0);
    let mut positive_definite = design.t().dot(&design);
    for index in 0..dimension {
        positive_definite[[index, index]] += dimension as f64;
    }
    let cholesky = CachedCholesky::new(&positive_definite).unwrap();

    let mut indefinite = Array2::zeros((dimension, dimension));
    for row in 0..dimension {
        indefinite[[row, row]] = if row % 2 == 0 { 1.0 } else { -1.0 } * (1.0 + row as f64 / 64.0);
        for column in 0..row {
            let value = 0.01 * (random() * 2.0 - 1.0);
            indefinite[[row, column]] = value;
            indefinite[[column, row]] = value;
        }
    }
    let lu = CachedLu::new(&indefinite).unwrap();
    let rhs: Vec<f64> = (0..dimension).map(|_| random() * 2.0 - 1.0).collect();
    let mut output = vec![0.0; dimension];

    let mut group = c.benchmark_group("cached_solvers");
    group.bench_function("cholesky_dim=64", |bench| {
        bench.iter(|| cholesky.solve_into_slice(black_box(&rhs), black_box(&mut output)))
    });
    group.bench_function("indefinite_lu_dim=64", |bench| {
        bench.iter(|| lu.solve_into_slice(black_box(&rhs), black_box(&mut output)))
    });
    group.finish();
}

fn bench_weighted_quantile(c: &mut Criterion) {
    let n = 10_000;
    let mut random = xorshift_rng(20260715);
    let values = Array1::from_shape_fn(n, |_| random() * 10.0);
    let unit_weights = Array1::ones(n);
    let huge_weights = Array1::from_elem(n, 1.0e9);
    let mut group = c.benchmark_group("weighted_quantile_frequency");
    group.bench_function("n=10k_weight=1", |bench| {
        bench.iter(|| weighted_quantile(black_box(values.view()), unit_weights.view(), 0.5))
    });
    group.bench_function("n=10k_weight=1e9", |bench| {
        bench.iter(|| weighted_quantile(black_box(values.view()), huge_weights.view(), 0.5))
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_prepare_learners,
    bench_cvrisk_selected,
    bench_glmboost_scalar,
    bench_binned_scalar,
    bench_gamboost_bbs,
    bench_binned_bbs,
    bench_gamboost_gram_vs_general,
    bench_binomial,
    bench_poisson,
    bench_btree,
    bench_btree_split_method,
    bench_bspatial_setup,
    bench_propodds,
    bench_many_learners,
    bench_block_selection_crossover,
    bench_non_gaussian_bbs_crossover,
    bench_instrumentation_overhead,
    bench_cached_solvers,
    bench_weighted_quantile,
);
criterion_main!(benches);
