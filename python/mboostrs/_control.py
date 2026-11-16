"""`boost_control`, as a named record rather than an anonymous 4-tuple.

The Rust binding returns `(mstop, nu, risk, trace)` and nothing says so:
``help(mb.boost_control)`` was empty, ``type(c)`` was `tuple`, and reading a
field meant knowing that `nu` is index 1. Passing it around, storing it on a
fitted model and comparing two of them all worked; understanding one did not.

:class:`BoostControl` is a `NamedTuple` over exactly those four values, so it
**is** a tuple: the bindings, the artifact format and every `control[0]` in
the codebase keep working unchanged, and `c.nu` now reads as what it is.
`boost_control()` validates through the Rust constructor first -- that is
where `mstop >= 1` and `0 < nu <= 1` are enforced, and duplicating the rules
here would be a second place for them to drift.
"""

from __future__ import annotations

from typing import NamedTuple

from mboostrs._rust import boost_control as _rust_boost_control

__all__ = ["BoostControl", "as_control", "boost_control", "step_kwargs"]


class BoostControl(NamedTuple):
    """How long to boost, and how far each step goes.

    Attributes
    ----------
    mstop : int
        Iterations to run. Each one selects a single base-learner and adds
        `nu` times its fit. Small values may leave the path underfit, while
        large values may continue fitting noise. Choose it with `cvrisk`.
    nu : float
        The step length, between 0 and 1. Smaller steps make each update
        smaller and usually require more iterations; they do not guarantee a
        better fit. The default is 0.1.
    risk : {"inbag", "oobag", "none"}
        Which risk the path records. `"inbag"` is the fitted sample's own
        risk, which is what `mboost` stores and what the reported path means.
    trace : bool
        Print the risk at each iteration to stderr while fitting.
    step : {"fixed", "line_search"}, default "fixed"
        How the step length is chosen after the winning learner is selected.
        `"fixed"` takes the classic constant step `nu * h` that `mboost` uses.
        `"line_search"` first finds the multiple `rho` of the winner's
        direction that minimizes the training loss, then takes `nu * rho * h`:
        `nu` becomes a fraction of the loss-optimal step, so its meaning no
        longer depends on the response scale. This removes the divergence a
        fixed step suffers on large-count Poisson responses and the crawl it
        suffers at rare binary events. Selection is unchanged, and the
        squared-error families are bit-identical either way (their
        least-squares step is already optimal, `rho = 1`).
    step_iterations : int, default 4
        Golden-section refinement steps per boosting iteration under
        `step="line_search"`. The grid has already bracketed the minimum, so
        small values suffice.
    """

    mstop: int
    nu: float
    risk: str
    trace: bool
    step: str = "fixed"
    step_iterations: int = 4

    def describe(self) -> str:
        """One-line text description of the control.

        Returns
        -------
        str
            For example ``"100 iterations at nu=0.1, risk='inbag'"``; the step
            rule is appended when it is not ``"fixed"``.
        """
        step = "" if self.step == "fixed" else f", step={self.step!r}"
        return (f"{self.mstop} iterations at nu={self.nu:g}, "
                f"risk={self.risk!r}" + (", tracing" if self.trace else "") + step)


def boost_control(mstop: int = 100, nu: float = 0.1, risk: str = "inbag",
                  trace: bool = False, step: str = "fixed",
                  step_iterations: int = 4) -> BoostControl:
    """The control record a fitting function takes.

    ``boost_control(100, 0.1)`` is the default: a hundred iterations at a
    step length of a tenth. See :class:`BoostControl` for what each field
    means and how to choose it.

    `mstop` must be at least 1, `nu` in `(0, 1]`, and `risk` one of
    `"inbag"`, `"oobag"`, `"none"`. `step` must be `"fixed"` or
    `"line_search"`, and `step_iterations` at least 1.

    Parameters
    ----------
    mstop : int, default 100
        Number of boosting iterations; at least 1.
    nu : float, default 0.1
        Step length in ``(0, 1]``.
    risk : {"inbag", "oobag", "none"}, default "inbag"
        Which risk the recorded path holds.
    trace : bool, default False
        Print the risk at each iteration to stderr.
    step : {"fixed", "line_search"}, default "fixed"
        Step-length rule; see :class:`BoostControl`.
    step_iterations : int, default 4
        Golden-section steps per iteration under ``step="line_search"``;
        at least 1.

    Returns
    -------
    BoostControl
        Named tuple with the fields above.

    Raises
    ------
    ValueError
        If any argument is outside the ranges above (``mstop``, ``nu`` and
        ``risk`` are checked by the Rust constructor).
    """
    control = BoostControl(*_rust_boost_control(mstop, nu, risk, trace))
    if step not in ("fixed", "line_search"):
        raise ValueError(
            f"step must be 'fixed' or 'line_search', got {step!r}")
    if not isinstance(step_iterations, int) or isinstance(step_iterations, bool):
        raise ValueError(
            f"step_iterations must be an integer, got {step_iterations!r}")
    if step_iterations < 1:
        raise ValueError(
            f"step_iterations must be at least 1, got {step_iterations}")
    return control._replace(step=step, step_iterations=step_iterations)


def as_control(control) -> BoostControl:
    """A control record, accepting the bare tuple forms callers may pass.

    A 4-tuple (or the 2-tuple a fitting function has already normalized)
    becomes a record with the step rule at its default; a record passes
    through unchanged, so its rule survives every rebuild.
    """
    if isinstance(control, BoostControl):
        return control
    return BoostControl(*control)


def step_kwargs(control) -> dict:
    """Keyword arguments carrying a control's step rule to a binding call.

    `getattr` defaults keep controls written before the step rule existed --
    shorter tuples from hand-built code, records saved by older versions --
    reading as the classic fixed step.
    """
    return {
        "step": getattr(control, "step", "fixed"),
        "step_iterations": getattr(control, "step_iterations", 4),
    }
