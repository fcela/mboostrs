use mboostrs_core::learners::{
    prepare_block_bols, prepare_bmono, prepare_btree, BmonoSolver, MonoConstraint, PreparedLearner,
};
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

/// The global counting allocator sees every thread in the process. A Rayon
/// pool started by one test spawns worker threads whose first allocations
/// (thread-local malloc arenas) land in another test's measured rounds.
/// Serialising the two tests removes that cross-talk instead of bounding it.
static SERIALIZE_MEASUREMENTS: Mutex<()> = Mutex::new(());

struct CountingAllocator;

static TRACK: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACK.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if TRACK.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if TRACK.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn run_selection(learners: &mut [PreparedLearner], u: &Array1<f64>, weights: &Array1<f64>) {
    let wu2 = u
        .iter()
        .zip(weights)
        .map(|(&residual, &weight)| weight * residual * residual)
        .sum();
    let winner = learners
        .par_iter_mut()
        .enumerate()
        .map(|(index, learner)| {
            (
                index,
                learner.fit_step_select_sse(u.view(), weights.view(), wu2),
            )
        })
        .reduce(
            || (0, f64::INFINITY),
            |left, right| {
                if right.1 < left.1 || (right.1 == left.1 && right.0 < left.0) {
                    right
                } else {
                    left
                }
            },
        );
    black_box(winner);
}

fn run_candidate_solves(learners: &mut [PreparedLearner], u: &Array1<f64>, weights: &Array1<f64>) {
    let wu2 = u
        .iter()
        .zip(weights)
        .map(|(&residual, &weight)| weight * residual * residual)
        .sum();
    for learner in learners {
        black_box(learner.fit_step_select_sse(u.view(), weights.view(), wu2));
    }
}

#[test]
fn parallel_candidate_reduction_has_no_loser_beta_allocations() {
    let _measurement_guard = SERIALIZE_MEASUREMENTS.lock().unwrap();
    let n = 256;
    let weights = Array1::ones(n);
    let residual = Array1::from_shape_fn(n, |row| ((row * 37) % n) as f64 / n as f64 - 0.5);
    let learners_template = (0..50)
        .map(|learner| {
            let design = Array2::from_shape_fn((n, 8), |(row, column)| {
                let phase = (row * (learner + 3) + column * 17) % n;
                phase as f64 / n as f64 - 0.5
            });
            prepare_block_bols(
                design,
                weights.view(),
                Array2::eye(8) * 0.01,
                false,
                &format!("block{learner}"),
            )
        })
        .collect::<Vec<_>>();
    let mut learners = learners_template;

    // What this guards: a candidate solve reuses its beta buffer instead of
    // cloning one per learner per iteration. What it measures: process-wide
    // allocations, through a global counting allocator.
    //
    // Those are not the same thing, and the gap is where this test used to
    // give an answer nobody could explain. It ran `run_selection` -- which
    // starts Rayon's thread pool -- warmed up with one serial round, then
    // asserted zero allocations over the next hundred. That passes in a debug
    // build and fails in a release one, at `0167ac3` and today alike, so it
    // is a property of the build profile and not of any commit.
    //
    // Measured per round in release, the allocations land in a short burst a
    // few rounds in and never return: rounds 3 and 4 on this fixture, 6 and
    // 14 of them. Removing the `run_selection` call removes them entirely --
    // zero across all hundred rounds. **They are Rayon's worker threads
    // settling**, caught by a process-global counter while a serial loop runs
    // beside them. A debug build is slow enough that the pool has quietened
    // before measurement starts; a release build is not.
    //
    // A process-global counter cannot attribute an allocation to a thread, so
    // the parallel reduction's own allocation behaviour is not measurable
    // this way and this test does not claim to measure it. What it does
    // measure -- and what its name is about -- is that the serial candidate
    // solves reuse their buffers. That is now asserted with no pool running
    // alongside, which makes it exact in both profiles rather than accidental
    // in one.
    run_selection(&mut learners, &residual, &weights);

    // Give the pool time to finish whatever it does after a `par_iter`, so
    // its allocations are not counted against the loop below. The check that
    // this worked is the assertion itself: if the pool is still active, the
    // count is not zero.
    std::thread::sleep(std::time::Duration::from_millis(250));
    run_candidate_solves(&mut learners, &residual, &weights);

    const ROUNDS: usize = 100;
    let mut per_round = [0usize; ROUNDS];
    for slot in per_round.iter_mut() {
        ALLOCATIONS.store(0, Ordering::SeqCst);
        TRACK.store(true, Ordering::SeqCst);
        run_candidate_solves(&mut learners, &residual, &weights);
        TRACK.store(false, Ordering::SeqCst);
        *slot = ALLOCATIONS.load(Ordering::SeqCst);
    }

    let total: usize = per_round.iter().sum();
    let nonzero: Vec<(usize, usize)> = per_round
        .iter()
        .enumerate()
        .filter(|(_, &value)| value > 0)
        .map(|(index, &value)| (index, value))
        .collect();
    assert_eq!(
        total, 0,
        "candidate solves must retain beta buffers without cloning; \
         allocations by round: {nonzero:?}"
    );

    // Rayon may occasionally allocate scheduler bookkeeping, but allocation
    // must not scale as iterations × losing learners (4,900 here). A bound
    // below one allocation per reduction cleanly distinguishes scheduler
    // traffic from the 4,900 removed per-candidate Vec clones.
    ALLOCATIONS.store(0, Ordering::SeqCst);
    TRACK.store(true, Ordering::SeqCst);
    for _ in 0..100 {
        run_selection(&mut learners, &residual, &weights);
    }
    TRACK.store(false, Ordering::SeqCst);
    let parallel_allocations = ALLOCATIONS.load(Ordering::SeqCst);
    assert!(
        parallel_allocations < 100,
        "parallel reduction allocated {parallel_allocations} times; loser betas appear to be cloned"
    );
}

/// Serial candidate-solve allocations per learner kind (release build;
/// attributable because the Rayon pool is never started on this path).
///
/// Every row of a piecewise-parallel gamboostLSS candidate scan funnels
/// through this same call, so a per-candidate allocation here is per
/// learner per iteration. The kinds with owned solver returns (BmonoBbs,
/// Btree) used to pay that; the kinds writing into prepared scratch
/// (BlockBols, KroneckerBols) never did.
fn measure_serial_solves(
    make: impl Fn(&Array1<f64>) -> PreparedLearner,
    u: &Array1<f64>,
    weights: &Array1<f64>,
) -> Vec<usize> {
    let mut learner = make(weights);
    let wu2 = u
        .iter()
        .zip(weights)
        .map(|(&residual, &weight)| weight * residual * residual)
        .sum::<f64>();

    // Settle one round so buffer sizing is warm, then count steady-state.
    black_box(learner.fit_step_select_sse(u.view(), weights.view(), wu2));
    let rounds: usize = 100;
    let mut per_round = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        ALLOCATIONS.store(0, Ordering::SeqCst);
        TRACK.store(true, Ordering::SeqCst);
        black_box(learner.fit_step_select_sse(u.view(), weights.view(), wu2));
        TRACK.store(false, Ordering::SeqCst);
        per_round.push(ALLOCATIONS.load(Ordering::SeqCst));
    }
    per_round
}

fn bmono_fixture(weights: &Array1<f64>, solver: BmonoSolver) -> PreparedLearner {
    let n = weights.len();
    let x = Array1::from_shape_fn(n, |row| {
        let t = row as f64 / (n as f64 - 1.0);
        t - 0.5
    });
    prepare_bmono(
        x.view(),
        weights.view(),
        8,
        2,
        1,
        4.0,
        MonoConstraint::Positive,
        solver,
        1e6,
        100,
        "bm",
        None,
        None,
        false,
    )
}

fn btree_fixture(weights: &Array1<f64>) -> PreparedLearner {
    let n = weights.len();
    let design: Array2<f64> = Array2::from_shape_fn((n, 3), |(row, column)| {
        let phase = (row * 7 + column * 11) % n;
        phase as f64 / n as f64 - 0.5
    });
    prepare_btree(design.view(), weights.view(), 8, 8, 4, None, "bt")
}

/// The bmono and btree candidate solves are serialised here so the per-round
/// count is attributable (no Rayon pool). P3.4 item 2 closed the block-bols
/// path at zero and this pins the remaining learner kinds.
#[test]
fn candidate_solve_allocations_by_learner_kind() {
    let _measurement_guard = SERIALIZE_MEASUREMENTS.lock().unwrap();
    let n = 256;
    let weights = Array1::ones(n);
    let residual = Array1::from_shape_fn(n, |row| ((row * 37) % n) as f64 / n as f64 - 0.5);

    // A freshly prepared learner allocates its scratch lazily on the first
    // solve (Vec::new buffers, solver setup). Both are one-time; measured
    // rounds below re-prepare and therefore count them, so the stable shape
    // is what the per-round counts bound.
    let kinds = [
        (
            "block_bols",
            measure_serial_solves(
                |w| {
                    let design = Array2::from_shape_fn((n, 8), |(row, column)| {
                        let phase = (row * 3 + column * 17) % n;
                        phase as f64 / n as f64 - 0.5
                    });
                    prepare_block_bols(design, w.view(), Array2::eye(8) * 0.01, false, "blk")
                },
                &residual,
                &weights,
            ),
        ),
        (
            "bmono_iterative",
            measure_serial_solves(
                |w| bmono_fixture(w, BmonoSolver::Iterative),
                &residual,
                &weights,
            ),
        ),
        (
            "bmono_quadprog",
            measure_serial_solves(
                |w| bmono_fixture(w, BmonoSolver::QuadProg),
                &residual,
                &weights,
            ),
        ),
        (
            "btree",
            measure_serial_solves(btree_fixture, &residual, &weights),
        ),
    ];

    let block_max = kinds[0].1.iter().copied().max().unwrap_or(0);
    for (name, per_round) in kinds {
        let distinct = {
            let mut set = per_round.clone();
            set.sort_unstable();
            set.dedup();
            set
        };
        let max = per_round.iter().copied().max().unwrap_or(0);
        eprintln!(
            "{name}: max {max} allocs/round across {} rounds; distinct {distinct:?}",
            per_round.len()
        );
        match name {
            // The bmono solvers re-factor the p×p augmented/collision matrix
            // into their own buffer each iteration, taking faer's working
            // memory from the learner's `chol_scratch`, so the steady state is
            // zero. The small budget left here absorbs the global counting
            // allocator seeing other threads, not a per-solve allocation: the
            // owned-return solver paid ~85/round and the earlier
            // `Cholesky::try_new` route ~10, both of which fail this bound, as
            // does a per-candidate beta clone in the thousands.
            "bmono_iterative" | "bmono_quadprog" => assert!(
                max <= block_max + 6,
                "{name}: {max} allocations/round exceeds the block_bols baseline {block_max} + cross-thread budget; distinct {distinct:?}"
            ),
            // The tree re-builds its node storage per call by design (a
            // fresh prediction tree is the learner's output, not a reusable
            // buffer); its row buffers are allocated once per tree and
            // partitioned in place, 9 allocations measured. The bound
            // rejects per-node buffers (163 when each node copied its
            // children's sorted lists) and the beta-buffer clone the old
            // `*beta_buf = coef.into_raw_vec_and_offset().0` paid, which is
            // unbounded.
            "btree" => assert!(
                max <= 20,
                "{name}: {max} allocations/round; the tree fit allocates about 9, a per-node buffer or beta clone far more; distinct {distinct:?}"
            ),
            // block_bols is the zero-clone baseline; its own band is noise
            // (process-visible, non-candidate), reported for context only.
            _ => {}
        }
    }
}
