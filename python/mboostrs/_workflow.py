"""Context-local admission and accounting for Python-to-Rust fit calls.

This module schedules work; all fitting remains in the native kernels.
Fit charges are hard admission limits. A native CV batch reserves all its
folds before starting; on failure that reservation is retained because a
binding exception does not say how many folds actually ran. Time and memory
are cooperative checkpoints, not interruption or allocator limits.
"""

from __future__ import annotations

import contextvars
import sys
import threading
import time
from functools import wraps

from mboostrs._errors import MboostrsRuntimeError
from mboostrs._validate import check_int_at_least

_CURRENT = contextvars.ContextVar("mboostrs_workflow", default=None)


class BudgetExhausted(MboostrsRuntimeError):
    """A workflow cannot admit its next fit; no partial fit is returned."""

    def __init__(self, reason, operation, requested, report):
        super().__init__(f"workflow {reason} budget cannot admit {operation} "
                         f"({requested} fit charge(s))", code="workflow_budget",
                         subject=operation)
        self.reason = reason
        self.requested_fit_charges = requested
        self.report = report


def _peak_memory_mb():
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux and other supported Unix builds use KiB.
    return float(peak) / (1024 * 1024 if sys.platform == "darwin" else 1024)


class Workflow:
    """One scope, shared by its nested synchronous/native-parallel fits."""

    def __init__(self, budget, progress=None):
        self.budget = budget
        self.progress = progress
        self.started = time.perf_counter()
        self.charges = 0
        self.events = []
        self.stopped = None
        self.blocked_request = None
        self._lock = threading.RLock()
        self._token = None
        if budget.memory_mb is not None and _peak_memory_mb() is None:
            raise ValueError("memory budgets require process high-water memory telemetry on this platform")

    def __enter__(self):
        if _CURRENT.get() is not None:
            raise ValueError("nested workflow budgets are not supported; use one outer workflow")
        self._token = _CURRENT.set(self)
        return self

    def __exit__(self, *_):
        _CURRENT.reset(self._token)

    def report(self):
        return {
            "budget": self.budget.to_dict(), "fit_charges": self.charges,
            "elapsed_seconds": time.perf_counter() - self.started,
            "peak_memory_mb": _peak_memory_mb(),
            "memory_scope": "absolute process-lifetime high-water RSS, MiB",
            "limits": "fit admission is hard; time and memory checked between native calls",
            "failed_fit_accounting": ("failed native batches retain their full reserved charge; "
                                      "actual completed folds unknown"),
            "stopped": self.stopped, "blocked_request": self.blocked_request,
            "operations": [dict(e) for e in self.events],
        }

    def checkpoint(self, operation="checkpoint", requested=0):
        with self._lock:
            reason = self.stopped
            if reason is None and self.budget.fits is not None and self.charges + requested > self.budget.fits:
                reason = "fits"
            if reason is None and self.budget.seconds is not None:
                if time.perf_counter() - self.started >= self.budget.seconds:
                    reason = "seconds"
            if reason is None and self.budget.memory_mb is not None:
                if _peak_memory_mb() >= self.budget.memory_mb:
                    reason = "memory_mb"
            if reason is not None:
                if self.blocked_request is None:
                    self.blocked_request = {"operation": operation, "fit_charges": requested,
                                            "reason": reason}
                self.stopped = reason
                raise BudgetExhausted(reason, operation, requested, self.report())

    def call(self, function, args, kwargs, fits):
        name = function.__name__
        with self._lock:
            self.checkpoint(name, fits)
            self.charges += fits
            event = {"operation": name, "fit_charges": fits, "status": "running",
                     "cumulative_fit_charges": self.charges}
            self.events.append(event)
        started = time.perf_counter()
        try:
            value = function(*args, **kwargs)
        except BaseException as exc:
            event.update(status="failed", error_type=type(exc).__name__,
                         error_code=getattr(exc, "code", None), error=str(exc)[:500],
                         actual_completed_fits=None)
            raise
        else:
            event["status"] = "returned"
            return value
        finally:
            event["seconds"] = time.perf_counter() - started
            event["peak_memory_mb"] = _peak_memory_mb()
            if self.progress is not None:
                self.progress(dict(event))


def _is_native_panic(exc):
    """Whether ``exc`` is pyo3's ``PanicException`` (a Rust panic).

    pyo3 raises it as a ``BaseException`` subclass that no public module
    exports, so it is matched by class name rather than imported.
    """
    return any(kind.__name__ == "PanicException" for kind in type(exc).__mro__)


def call_fit(function, *args, _fit_count=1, **kwargs):
    """Dispatch one native operation, charging it only inside a workflow.

    A Rust panic inside the native core surfaces from pyo3 as a
    ``PanicException``, which derives from ``BaseException`` and so escapes
    every ``except Exception`` a caller writes. It is re-raised here as an
    ``MboostrsRuntimeError`` (``code="native_panic"``) so a probe that
    compares candidate models can record the failed candidate and move on.
    The original exception is chained as ``__cause__``.
    """
    try:
        workflow = _CURRENT.get()
        if workflow is None:
            return function(*args, **kwargs)
        count = check_int_at_least("fit charge", _fit_count, 1)
        return workflow.call(function, args, kwargs, count)
    except BaseException as exc:
        if not _is_native_panic(exc):
            raise
        operation = getattr(function, "__name__", "native call")
        raise MboostrsRuntimeError(
            f"the native core aborted during {operation}: {exc}. This is a "
            "bug in mboostrs; the fit was abandoned and no partial result "
            "is available.",
            code="native_panic", subject=operation, details=str(exc),
        ) from exc


def checkpoint(operation="checkpoint", requested=0):
    """Propagate a budget stop even when a probe caught a fit exception."""
    workflow = _CURRENT.get()
    if workflow is not None:
        workflow.checkpoint(operation, requested)


def summary():
    """Bounded resource evidence for a judge, without the operation ledger."""
    workflow = _CURRENT.get()
    if workflow is None:
        return None
    limit = workflow.budget.fits
    return {"budget": workflow.budget.to_dict(), "fit_charges": workflow.charges,
            "remaining_fit_charges": None if limit is None else max(0, limit - workflow.charges),
            "elapsed_seconds": time.perf_counter() - workflow.started,
            "memory_scope": "absolute process-lifetime peak RSS, MiB"}


def fit_checkpoint(function):
    """Check admission before Python constructs a fit's designs and recipes."""
    @wraps(function)
    def checked(*args, **kwargs):
        workflow = _CURRENT.get()
        if workflow is not None:
            workflow.checkpoint(f"prepare_{function.__name__}", requested=1)
        return function(*args, **kwargs)
    return checked
