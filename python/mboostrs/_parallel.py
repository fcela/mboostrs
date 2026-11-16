"""Internal helper for parallelizing embarrassingly-parallel bootstrap loops.

The Rust fit/cvrisk functions release the GIL (`py.allow_threads`), so a thread
pool gives real parallelism for these loops without process/pickling overhead.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor


def parallel_map(fn, items, n_jobs):
    """Apply ``fn`` to each item, preserving order.

    ``n_jobs`` in ``(None, 1)`` runs sequentially; ``n_jobs < 0`` uses all CPUs;
    otherwise uses ``n_jobs`` worker threads. Sequential when there is <= 1 item.
    """
    items = list(items)
    if n_jobs in (None, 1) or len(items) <= 1:
        return [fn(x) for x in items]
    workers = (os.cpu_count() or 1) if n_jobs < 0 else n_jobs
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))
