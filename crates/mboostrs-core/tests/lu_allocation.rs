use mboostrs_core::linalg::CachedLu;
use ndarray::array;
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct CountingAllocator;

static TRACK_ALLOCATIONS: AtomicBool = AtomicBool::new(false);
static ALLOCATION_COUNT: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACK_ALLOCATIONS.load(Ordering::Relaxed) {
            ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        System.alloc(layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if TRACK_ALLOCATIONS.load(Ordering::Relaxed) {
            ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        System.alloc_zeroed(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if TRACK_ALLOCATIONS.load(Ordering::Relaxed) {
            ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn cached_lu_slice_solve_has_no_steady_state_heap_allocations() {
    let a = array![[0.0, 2.0, 1.0], [2.0, -1.0, 0.0], [1.0, 0.0, 3.0]];
    let lu = CachedLu::new(&a).unwrap();
    let b = [1.0, -2.0, 0.5];
    let mut x = [0.0; 3];

    // Warm up any lazily initialized runtime state before measuring.
    lu.solve_into_slice(&b, &mut x);
    ALLOCATION_COUNT.store(0, Ordering::SeqCst);
    TRACK_ALLOCATIONS.store(true, Ordering::SeqCst);
    for _ in 0..1_000 {
        lu.solve_into_slice(black_box(&b), black_box(&mut x));
    }
    TRACK_ALLOCATIONS.store(false, Ordering::SeqCst);

    assert_eq!(ALLOCATION_COUNT.load(Ordering::SeqCst), 0);
    let residual = a.dot(&ndarray::ArrayView1::from(&x)) - ndarray::ArrayView1::from(&b);
    assert!(residual.iter().all(|value| value.abs() < 1e-11));
}
