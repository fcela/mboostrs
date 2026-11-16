"""Copy semantics for the accessors that hand out a fit's stored state.

Three forms of the same quantity differed in what a caller could do to the
model through the value they got back. ``fit.fitted()`` returned a fresh
array, while ``fit.fitted_`` and ``fit.risk_`` returned the model's own
arrays. So::

    f = fit.fitted_
    f -= f.mean()          # a caller centring "their" copy

changed what ``fit.residuals()`` reported afterwards -- measured on a
100-row Gaussian fit, the first three response residuals moved from
``-0.5211, -0.4967, -1.4113`` to ``-0.4969, -0.4725, -1.3871`` -- while
``fit.predict()`` recomputed from the coefficients and returned the original
values. The model was then internally inconsistent, and nothing said so.
``fit.risk_.sort()`` did the same to the risk path.

The rule:

* A trailing-underscore property (``fitted_``, ``risk_``) is the fit's stored
  state, and returns a **read-only view**. No copy is made, so reading stays
  free; a write raises ``ValueError: assignment destination is read-only``
  rather than silently corrupting the fit. ``np.asarray(v)`` still works and
  ``v.copy()`` gives a writable array.
* A method that computes a quantity (``fitted()``, ``predict()``,
  ``model_weights()``) returns a fresh writable array. There is no stored
  object behind it to protect.
"""

from __future__ import annotations

import numpy as np

__all__ = ["CallableArray", "callable_readonly", "readonly"]


def readonly(array) -> np.ndarray:
    """A read-only view of ``array``, without copying it.

    A view is taken so that the caller cannot reach the model's array through
    it. When ``array`` is not already an ndarray, ``np.asarray`` may itself
    copy; the result is marked read-only either way, so the promise the
    caller gets does not depend on which happened.
    """
    view = np.asarray(array).view()
    view.flags.writeable = False
    return view


class CallableArray(np.ndarray):
    """A read-only ndarray that also accepts a call.

    ``fit.risk`` returns one of these, so ``fit.risk`` and ``fit.risk()`` both
    give the risk path. Everything else about the value is a plain ndarray:
    arithmetic, comparisons, numpy functions and indexing return plain arrays
    (not this class), ``repr`` is the array's own ``repr``, and pickling
    produces a plain writable array. Only ``type(x) is CallableArray`` and
    calling the value differ.

    The call takes no arguments unless the model supplied a ``call`` target
    (the LSS classes do, for their ``parameter=`` and ``merge=`` options); with
    no target it returns a fresh writable copy of the path.
    """

    _mboostrs_call = None

    def __call__(self, *args, **kwargs):
        if self._mboostrs_call is None:
            if args or kwargs:
                raise TypeError(
                    "the risk path takes no arguments; write fit.risk or "
                    "fit.risk() for the joint risk")
            return np.array(self, copy=True)
        return self._mboostrs_call(*args, **kwargs)

    def __array_wrap__(self, obj, context=None, return_scalar=False):
        # Results of ufuncs and reductions are plain arrays, so arithmetic on
        # the path does not carry the call target along with it.
        plain = np.asarray(obj)
        return plain[()] if return_scalar else plain

    def __getitem__(self, key):
        return np.asarray(self)[key]

    def __repr__(self):
        return repr(np.asarray(self))

    def __str__(self):
        return str(np.asarray(self))

    def __reduce_ex__(self, protocol):
        # Pickle the values as the plain writable array the path has always
        # unpickled to; the call target is a bound method and is not state.
        return np.array(self, copy=True).__reduce_ex__(protocol)


def callable_readonly(array, call=None) -> CallableArray:
    """A :class:`CallableArray` view of ``array``, read-only, without a copy.

    ``call``, when given, is what calling the value does (it is passed the
    call's arguments). Without it, calling returns a copy of the path.
    """
    view = readonly(array).view(CallableArray)
    view._mboostrs_call = call
    return view
