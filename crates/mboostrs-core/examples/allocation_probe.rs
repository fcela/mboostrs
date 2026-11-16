//! Count heap allocations across representative complete core fits.

use mboostrs_core::{prepare_bbs, prepare_scalar_bols, try_fit_with, BoostControl, Gaussian};
use ndarray::Array1;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct CountingAllocator;

static TRACK: AtomicBool = AtomicBool::new(false);
static COUNT: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACK.load(Ordering::Relaxed) {
            COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if TRACK.load(Ordering::Relaxed) {
            COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if TRACK.load(Ordering::Relaxed) {
            COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn rng(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed;
    move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state as f64 / u64::MAX as f64
    }
}

fn count_allocations(mut operation: impl FnMut()) -> usize {
    COUNT.store(0, Ordering::SeqCst);
    TRACK.store(true, Ordering::SeqCst);
    operation();
    TRACK.store(false, Ordering::SeqCst);
    COUNT.load(Ordering::SeqCst)
}

fn main() {
    let n = 10_000;
    let mstop = 200;
    let weights = Array1::ones(n);
    let control = BoostControl::new(mstop, 0.1).unwrap();
    let mut random = rng(20260715);

    let scalar_x: Vec<Array1<f64>> = (0..10)
        .map(|_| (0..n).map(|_| random() * 2.0 - 1.0).collect())
        .collect();
    let scalar_y: Array1<f64> = (0..n)
        .map(|row| scalar_x.iter().map(|column| column[row]).sum::<f64>() + random() * 0.1)
        .collect();
    let scalar_template: Vec<_> = scalar_x
        .iter()
        .enumerate()
        .map(|(index, x)| {
            prepare_scalar_bols(
                x.view(),
                weights.view(),
                0.0,
                true,
                &format!("x{index}"),
                None,
            )
        })
        .collect();
    let mut scalar_warmup = scalar_template.clone();
    try_fit_with(
        &mut scalar_warmup,
        scalar_y.view(),
        weights.view(),
        &mut Gaussian,
        &control,
        None,
        false,
    )
    .unwrap();
    let mut scalar_learners = scalar_template.clone();
    let scalar_allocations = count_allocations(|| {
        try_fit_with(
            &mut scalar_learners,
            scalar_y.view(),
            weights.view(),
            &mut Gaussian,
            &control,
            None,
            false,
        )
        .unwrap();
    });

    let smooth_x: Vec<Array1<f64>> = (0..10)
        .map(|_| (0..n).map(|_| random() * 2.0 - 1.0).collect())
        .collect();
    let smooth_y: Array1<f64> = (0..n)
        .map(|row| smooth_x.iter().map(|column| column[row].sin()).sum::<f64>() + random() * 0.1)
        .collect();
    let smooth_template: Vec<_> = smooth_x
        .iter()
        .enumerate()
        .map(|(index, x)| {
            prepare_bbs(
                x.view(),
                weights.view(),
                20,
                3,
                2,
                4.0,
                true,
                &format!("bbs(x{index})"),
            )
        })
        .collect();
    let mut gram_warmup = smooth_template.clone();
    try_fit_with(
        &mut gram_warmup,
        smooth_y.view(),
        weights.view(),
        &mut Gaussian,
        &control,
        None,
        false,
    )
    .unwrap();
    let mut gram_learners = smooth_template.clone();
    let gram_allocations = count_allocations(|| {
        try_fit_with(
            &mut gram_learners,
            smooth_y.view(),
            weights.view(),
            &mut Gaussian,
            &control,
            None,
            false,
        )
        .unwrap();
    });

    println!(
        "{{\"schema_version\":1,\"kind\":\"mboostrs-allocation-result\",\
         \"n\":{},\"mstop\":{},\
         \"packed_scalar\":{{\"learners\":10,\"allocations\":{}}},\
         \"gaussian_gram\":{{\"learners\":10,\"allocations\":{}}}}}",
        n, mstop, scalar_allocations, gram_allocations
    );
}
