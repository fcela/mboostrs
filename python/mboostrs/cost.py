"""`estimate_cost`: how many model fits a call performs, and what one fit costs.

Each call's fit count comes from the same function its own code uses to plan
the work, so the estimate and a `fit_budget=` refusal read one number. The
time comes from timing one ordinary fit of the call's specification on the
data, which is what the call's fits cost on this machine.
"""
from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field

from mboostrs._names import quote_response

#: Calls `estimate_cost` can cost.
CALLS = ("gamboost", "gamboostLSS", "tune_mstop", "compare_specifications",
         "term_report", "stabsel", "conformal_interval", "aipw", "effect_model",
         "ci_test", "iv_effect", "confounding_benchmark", "discover")

#: Step count of the timed fit when the call tunes its own step count.
_TIMING_STEPS = 100


@dataclass(frozen=True)
class CostEstimate:
    """The cost of one call before it runs.

    ``fits`` is the number of model fits the call performs, counted the way
    the call counts them for its own ``fit_budget=`` check. ``seconds_per_fit``
    is the wall-clock time of one timed ordinary fit on this machine.
    ``estimated_seconds`` is ``fits * seconds_per_fit``. ``explanation`` is one
    line saying what the count is made of. ``stages`` splits ``fits`` by stage
    for a call that runs in stages (``discover``) and is empty otherwise.
    """

    call: str
    fits: int
    seconds_per_fit: float
    estimated_seconds: float
    explanation: str
    stages: dict = field(default_factory=dict)


def estimate_cost(call: str, *args, **kwargs) -> CostEstimate:
    """Count the fits of a call and time one fit, without running the call.

    Parameters
    ----------
    call : str
        One of ``"gamboost"``, ``"gamboostLSS"``, ``"tune_mstop"``,
        ``"compare_specifications"``, ``"term_report"``, ``"stabsel"``,
        ``"conformal_interval"``, ``"aipw"``, ``"effect_model"``,
        ``"ci_test"``, ``"iv_effect"``, ``"confounding_benchmark"`` or
        ``"discover"``.
    *args, **kwargs
        The arguments the call would take, exactly as they would be passed to
        it. For ``"stabsel"`` and ``"conformal_interval"`` the first argument
        is the fitted model.

    Returns
    -------
    CostEstimate
        The fit count, the time of one fit, the product of the two, and a
        one-line explanation of the count. ``stages`` holds the count of each
        stage of a staged call (``discover``).

    Notes
    -----
    ``"gamboost"`` and ``"gamboostLSS"`` are one fit, so the estimate runs the
    call once and reports its time. Every other call is timed on one fit of its
    own specification: the same formula, family and step control the call
    would use. A call that tunes its step count (``mstop="cv"``) cannot know
    that count before it runs, so the timed fit uses 100 steps; a tuning that
    settles above 100 steps makes the estimate low.

    The causal calls (``aipw``, ``effect_model``, ``ci_test``, ``iv_effect``,
    ``confounding_benchmark``) count a tuned nuisance model as the fits of
    :func:`~mboostrs.explore.tune_mstop` (its inner folds plus one full-data
    fit) and an untuned one as a single fit. The counts follow the arguments
    as the call binds them, with defaults read from the call's signature.
    Two counts are upper bounds because the call decides after fitting:
    ``aipw(outcome_model="auto")`` counts separate outcome models for the two
    arms, and ``effect_model(terms="auto")`` counts the final smooth fit.
    With ``outcome_model="cv"`` every candidate outcome model is counted.

    ``"discover"`` counts the fits each stage makes on its first pass. A stage
    that repeats or branches on the data (rounds of backward elimination,
    interaction candidates, spread tests, the tuning of a spread model) adds
    more. ``CostEstimate.stages`` gives the first-pass count of each stage, and
    ``DiscoveryResult.stage_seconds`` the measured seconds after a run.

    Examples
    --------
    >>> import numpy as np, mboostrs as mb
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(-2, 2, 300)
    >>> data = {"x": x, "y": np.sin(x) + rng.normal(scale=0.3, size=300)}
    >>> cost = mb.estimate_cost("tune_mstop", "y ~ bbs(x, df=4)", data, folds=5)
    >>> cost.fits
    6
    """
    if call not in CALLS:
        raise ValueError(
            f"estimate_cost cannot cost {call!r}; pass one of {', '.join(CALLS)}")
    planned = _PLANNERS[call](args, kwargs)
    fits, run, explanation = planned[:3]
    stages = planned[3] if len(planned) > 3 else {}
    seconds = _seconds(run)
    return CostEstimate(call=call, fits=int(fits), seconds_per_fit=seconds,
                        estimated_seconds=fits * seconds,
                        explanation=explanation, stages=dict(stages))


def _seconds(run) -> float:
    start = time.perf_counter()
    run()
    return time.perf_counter() - start


def _bound(func, args, kwargs):
    """The arguments a call binds, by name, with defaults filled in, and the
    keyword arguments collected by a ``**`` parameter (such as ``fit_kwargs``)."""
    signature = inspect.signature(func)
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    values = dict(bound.arguments)
    extra = {}
    for name, parameter in signature.parameters.items():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            extra = dict(values.pop(name, {}))
    return values, extra


def _is_distributional(family) -> bool:
    return hasattr(family, "param_names") and hasattr(family, "pdf")


def _steps(mstop) -> int:
    return int(mstop) if isinstance(mstop, int) and not isinstance(mstop, bool) \
        else _TIMING_STEPS


def _fit_once(formula, data, family, control, extra):
    """One ordinary fit: `gamboostLSS` for a distributional family or a
    mapping of parameter formulas, `gamboost` otherwise."""
    from mboostrs.gamboost import gamboost
    from mboostrs.gamlss import gamboostLSS

    if isinstance(formula, dict) or _is_distributional(family):
        return gamboostLSS(formula, data, family=family, control=control, **extra)
    return gamboost(formula, data=data, family=family, control=control, **extra)


def _plan_single_fit(call, args, kwargs):
    from mboostrs.gamboost import gamboost
    from mboostrs.gamlss import gamboostLSS

    fit = gamboost if call == "gamboost" else gamboostLSS
    return 1, lambda: fit(*args, **kwargs), (
        f"one fit: the {call} call is itself a single model fit")


def _plan_tune_mstop(args, kwargs):
    from mboostrs._control import boost_control
    from mboostrs.explore import _doubling_ends, _tune_mstop_fits, tune_mstop
    from mboostrs.families import Gaussian

    a, extra = _bound(tune_mstop, args, kwargs)
    start = min(a["start"], a["max_mstop"])
    response_route = (a["score"] != "family" or a["fold_local"]) and not isinstance(a["formula"], dict)
    grids = len(_doubling_ends(start, a["max_mstop"], a["extensions"])) if response_route else 1
    fits = _tune_mstop_fits(a["folds"], grids=grids)
    family = Gaussian() if a["family"] is None else a["family"]
    control = boost_control(start, a["nu"], step=a["step"])
    n_folds = _tune_mstop_fits(a["folds"]) - 1
    return fits, lambda: _fit_once(a["formula"], a["data"], family, control, extra), (
        f"{n_folds} fold fits, each resumed from grid to grid, plus one full-data fit; "
        f"a response-score fold on the Gaussian Gram route is refitted at each grid, so "
        f"the worst case counts up to {grids} grids; the first grid is {start} steps")


def _plan_compare(args, kwargs):
    from mboostrs._control import boost_control
    from mboostrs.explore import _compare_plan, _lss_formulas, compare_specifications

    a, _extra = _bound(compare_specifications, args, kwargs)
    plan = _compare_plan(a["specifications"], a["data"], a["response"], a["family"],
                         a["folds"], a["mstop"], a["seed"], a["weights"],
                         a["na_action"], a["observation_ids"],
                         max_mstop=a["max_mstop"], score=a["score"])
    name, rhs = plan.pairs[0]
    if _is_distributional(a["family"]):
        formula = _lss_formulas(rhs, a["response"], a["family"])
    else:
        formula = f"{a['response']} ~ {rhs}"
    extra = {"na_action": a["na_action"]}
    if plan.weights is not None:
        extra["weights"] = plan.weights
    control = boost_control(_steps(a["mstop"]), a["nu"])
    return plan.planned_total, lambda: _fit_once(
        formula, plan.data, a["family"], control, extra), (
        f"{len(plan.pairs)} specifications scored on the same folds, "
        f"{plan.planned_total} fits in all: the folds of each specification, plus "
        f"the inner tuning fits when mstop='cv'")


def _plan_term_report(args, kwargs):
    from mboostrs._control import boost_control
    from mboostrs.explore import (
        _term_report_plan_for,
        _term_report_render,
        term_report,
    )

    a, _extra = _bound(term_report, args, kwargs)
    setup, plan, fits = _term_report_plan_for(
        a["formula"], a["data"], a["response"], a["family"], folds=a["folds"],
        mstop=a["mstop"], seed=a["seed"], weights=a["weights"],
        na_action=a["na_action"], shapes=a["shapes"], df=a["df"], score=a["score"])
    distributional, parts, _rhs, _text, response = setup
    spec = _term_report_render(parts)
    if distributional:
        formula = {name: f"{quote_response(response)} ~ {rhs}" for name, rhs in spec.items()}
    else:
        formula = f"{quote_response(response)} ~ {spec}"
    removals = sum(c.kind == "without" for c in plan)
    shapes = sum(c.kind == "shape" for c in plan)
    extra = {"na_action": a["na_action"]}
    if a["weights"] is not None:
        extra["weights"] = a["weights"]
    control = boost_control(_steps(a["mstop"]), a["nu"])
    return fits, lambda: _fit_once(formula, a["data"], a["family"], control, extra), (
        f"{len(plan)} candidate specifications ({removals} with one term removed, "
        f"{shapes} shape candidates), each a cross-validation of its folds")


def _plan_stabsel(args, kwargs):
    from mboostrs._refit import refit_spec
    from mboostrs.stabsel import stabsel, stabsel_planned_fits

    a, _extra = _bound(stabsel, args, kwargs)
    model = a["model"]
    fits = stabsel_planned_fits(a["B"], a["folds"], a["sampling_type"])
    paired = a["sampling_type"] == "SS"
    return fits, lambda: refit_spec(model).build(), (
        f"{fits} subsample fits: {fits // 2 if paired else fits} "
        f"{'complementary pairs (two fits each)' if paired else 'half-samples'}")


def _plan_conformal(args, kwargs):
    from mboostrs._refit import refit_spec
    from mboostrs.conformal import conformal_fit_count, conformal_interval

    a, _extra = _bound(conformal_interval, args, kwargs)
    model = a["model"]
    fits = conformal_fit_count(model, method=a["method"], mstop=a["mstop"],
                               n_folds=a["n_folds"], calibration=a["calibration"])
    if a["calibration"] is not None:
        why = "calibration rows were given, so the fitted model is used without a refit"
    elif a["mstop"] == "tune" or (a["method"] == "cqr" and a["mstop"] is None):
        why = ("each tuned refit costs one fit plus 5 inner fits; the step grid "
               "doubles while the risk falls, so this is the first pass")
    else:
        why = f"one refit per calibration fold or split under method={a['method']!r}"
    return fits, lambda: refit_spec(model).build(), f"{fits} refit fits: {why}"


def _causal_planner(name):
    """Planner of a causal call: its bound arguments go to the plan of the same
    name in ``_cost_plans``."""
    def planner(args, kwargs):
        from mboostrs import _cost_plans, causal

        a, _extra = _bound(getattr(causal, name), args, kwargs)
        return getattr(_cost_plans, f"plan_{name}")(a)
    return planner


def _plan_discover(args, kwargs):
    from mboostrs.discover import plan_discover_cost

    return plan_discover_cost(args, kwargs, _bound)


_PLANNERS = {
    **{name: _causal_planner(name) for name in (
        "aipw", "effect_model", "ci_test", "iv_effect", "confounding_benchmark")},
    "discover": _plan_discover,
    "gamboost": lambda args, kwargs: _plan_single_fit("gamboost", args, kwargs),
    "gamboostLSS": lambda args, kwargs: _plan_single_fit("gamboostLSS", args, kwargs),
    "tune_mstop": _plan_tune_mstop,
    "compare_specifications": _plan_compare,
    "term_report": _plan_term_report,
    "stabsel": _plan_stabsel,
    "conformal_interval": _plan_conformal,
}
