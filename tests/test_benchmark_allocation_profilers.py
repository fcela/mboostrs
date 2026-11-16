"""Executable checks for cross-language allocation evidence."""

from __future__ import annotations

import sys

import pytest

from benches.bench_all import _python_allocation_count, _r_allocation_count


def test_memray_counter_records_python_and_native_allocation_scope():
    pytest.importorskip("memray")
    count, method, scope = _python_allocation_count(
        lambda: [bytearray(1024) for _ in range(25)]
    )
    assert count >= 25
    assert method.startswith("Memray ")
    assert "Rust/native extensions" in scope


def test_rprofmem_counter_records_serial_r_allocator_scope():
    ro = pytest.importorskip("rpy2.robjects")
    if not bool(ro.r('capabilities("profmem")')[0]):
        pytest.skip("R was built without memory profiling")
    count, method, scope = _r_allocation_count(
        ro, 'lapply(seq_len(25), function(i) raw(1024))'
    )
    assert count >= 25
    assert method == "Rprofmem threshold=0"
    assert "parent" in scope


@pytest.mark.skipif(sys.platform == "win32", reason="parallel::mclapply is unavailable")
def test_rprofmem_counter_collects_fork_worker_profiles():
    ro = pytest.importorskip("rpy2.robjects")
    if not bool(ro.r('capabilities("profmem")')[0]):
        pytest.skip("R was built without memory profiling")
    count, _, scope = _r_allocation_count(
        ro,
        "bench_profiled_mclapply(as.list(seq_len(4)), function(i) raw(1024), mc.cores = 2)",
        parallel=True,
    )
    assert count >= 4
    assert "all fork workers" in scope
