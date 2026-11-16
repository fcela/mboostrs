"""Residuals named for what they are.

A residual kind is a definition, not a label for whatever vector is available.
Four were being conflated:

``response``
    ``y - mu``, with ``mu`` on the response's own scale. Defined when the fit
    produces such a ``mu`` and censoring does not make the difference
    meaningless. A ranking loss produces a score, not a ``mu``, so subtracting
    an observed time from it is not a residual of anything.

``pearson``
    ``(y - mu) / sqrt(V(mu))``, where ``V`` is the family's variance function.
    Defined only for families that declare one. `Laplace`, `Quantile`,
    `Expectile` and `Huber` do not -- they are not probability models -- and
    ``type="pearson"`` on them returned the *working gradient* instead:
    ``sign(y - mu)`` for `Laplace`, which takes two distinct values across the
    whole sample and is not a Pearson residual under any definition.

``working``
    The fitted family's negative gradient: the quantity used as the response
    for the next boosting step. Its scale follows the family's gradient
    convention. For Binomial, mboostrs uses the `1 / log(2)` factor in
    mboost's gradient, so the working residual is about 1.44 times
    `2 (y - p)`, the derivative scale of the natural-log risk.

``deviance``
    The signed square root of each observation's contribution to the deviance,
    ``sign(y - mu) * sqrt(2 * (l_saturated - l_fitted))``. Defined where the
    family has a likelihood and a saturated fit.

``martingale``
    For `CoxPH`: ``delta_i - H0(t_i) exp(eta_i)`` with Breslow's baseline
    cumulative hazard. The natural residual for a partial-likelihood fit,
    which has no ``mu`` on the time scale at all.

The available kinds per family come from ``mboostrs._likelihood``, so the
declaration and the implementation cannot drift apart: a kind offered there
must be computable here, and one refused here must not be offered there.
"""
from __future__ import annotations

import numpy as np

from mboostrs._likelihood import family_likelihood

#: Every kind this module can produce. ``working`` is available for every
#: family, because every family has a negative gradient by construction.
RESIDUAL_KINDS = ("response", "pearson", "working", "deviance", "martingale")

#: Families whose fitted value is a ranking score rather than a location.
#: Subtracting an observation from one is not a residual: the score is
#: identified only up to a monotone transformation, so the difference has no
#: units and no zero.
_RANKING_FAMILIES = frozenset({"AUC", "Gehan", "Cindex"})

#: Accelerated-failure-time families. Their `eta` is a location on the
#: *log-time* scale and the response is censored, so `y - exp(eta)` is neither
#: on a common scale nor meaningful for the censored rows. A censored residual
#: for these needs an explicit derivation -- a Cox-Snell or standardised
#: residual carrying the censoring indicator -- which is not implemented; the
#: refusal says so rather than offering the subtraction.
_AFT_FAMILIES = frozenset({"Weibull", "Loglog", "Lognormal"})


def _family_name(model) -> str:
    return type(getattr(model, "_family", None)).__name__


def available_kinds(model) -> tuple[str, ...]:
    """Residual kinds defined for this model's family, in a stable order."""
    contract = family_likelihood(getattr(model, "_family", None))
    declared = () if contract is None else contract.residual_kinds
    # `working` is always available: it is the negative gradient, which every
    # family computes in order to be fitted at all.
    return tuple(dict.fromkeys((*declared, "working")))


def _refuse(model, kind: str) -> str:
    """Why ``kind`` is undefined here, and what to ask for instead."""
    name = _family_name(model)
    contract = family_likelihood(getattr(model, "_family", None))
    offered = ", ".join(available_kinds(model))

    if kind == "pearson":
        if contract is not None and not contract.is_likelihood:
            return (
                f"Pearson residuals are undefined for {name}: they are "
                f"(y - mu) / sqrt(V(mu)), and {name} is a loss rather than a "
                f"probability model, so it declares no variance function V. "
                f"Use kind='working' for the family's negative gradient."
            )
        return (
            f"Pearson residuals are undefined for {name}: it declares no "
            f"variance function. Available kinds: {offered}."
        )

    if kind == "response":
        if name in _RANKING_FAMILIES:
            return (
                f"Response residuals are undefined for {name}: its fitted "
                f"value is a ranking score, identified only up to a monotone "
                f"transformation, so y - score has no units and no zero. "
                f"kind='working' gives the negative gradient."
            )
        if name == "CoxPH":
            return (
                "Response residuals are undefined for CoxPH: exp(eta) is a "
                "relative risk, not a time, so subtracting the observed time "
                "from it is not a residual. Use kind='martingale' or "
                "kind='deviance'."
            )
        if name in _AFT_FAMILIES:
            return (
                f"Response residuals are undefined for {name}: eta is a "
                f"location on the log-time scale, and the response is "
                f"censored, so y - exp(eta) is neither on a common scale nor "
                f"meaningful for a censored observation. A censored residual "
                f"for an AFT fit needs an explicit derivation carrying the "
                f"censoring indicator, which mboostrs does not implement. Use "
                f"kind='working' for the negative gradient."
            )
        return f"Response residuals are undefined for {name}. Available: {offered}."

    if kind == "martingale":
        return (
            f"Martingale residuals are defined for a proportional-hazards fit; "
            f"{name} is not one. Available kinds: {offered}."
        )

    if kind == "deviance":
        return (
            f"Deviance residuals are undefined for {name}: they need a "
            f"likelihood and a saturated fit to compare against. Available "
            f"kinds: {offered}."
        )

    return f"kind={kind!r} is undefined for {name}. Available: {offered}."


def _observed(model, newdata=None):
    """The response as the fit stored it.

    A survival response is a ``Surv(time, event)`` pair, which the model keeps
    as one stacked ``[times..., events...]`` vector; the generic accessor
    treats the response name as a single column and raises on the tuple.
    """
    from mboostrs.diagnostics import _observed_response
    from mboostrs.gamboost import _extract_response

    response_name = getattr(model, "_response_name", None)
    if isinstance(response_name, tuple):
        data = getattr(model, "_data", {}) if newdata is None else newdata
        return np.asarray(
            _extract_response(data, response_name, model._family),
            dtype=np.float64,
        )
    return np.asarray(_observed_response(model, newdata), dtype=np.float64)


def _survival_parts(model, newdata=None):
    """``(time, event)`` for a survival response stored as one stacked vector."""
    stacked = _observed(model, newdata)
    half = stacked.size // 2
    return stacked[:half], stacked[half:]


def _cox_baseline(model, mstop=None):
    """Training Breslow event times, scaled cumulative hazard, and eta shift.

    Tied events share one risk-set denominator.  The cumulative hazard is kept
    on the ``exp(eta - shift)`` scale, avoiding overflow while preserving the
    product ``H0(t) * exp(eta)`` used by a martingale residual.
    """
    time, event = _survival_parts(model, None)
    eta = np.asarray(model.predict(type="link", mstop=mstop),
                     dtype=np.float64).ravel()
    weights = np.asarray(model.weights, dtype=np.float64).ravel()
    shift = float(np.max(eta)) if eta.size else 0.0
    scaled_risk = np.exp(eta - shift)

    event_times = np.unique(time[(event > 0) & (weights > 0)])
    increments = np.empty(event_times.size, dtype=np.float64)
    for index, value in enumerate(event_times):
        deaths = float(np.sum(weights[time == value] * event[time == value]))
        denominator = float(np.sum(weights[time >= value] * scaled_risk[time >= value]))
        increments[index] = deaths / denominator if denominator > 0 else 0.0
    return event_times, np.cumsum(increments), shift


def _martingale(model, newdata=None, mstop=None):
    """Breslow martingale residuals for a Cox fit.

    ``M_i = delta_i - H0(t_i) exp(eta_i)``, with ``H0`` the Breslow estimate of
    the baseline cumulative hazard. They sum to zero over the sample by
    construction, are bounded above by 1 and unbounded below, and a large
    negative one marks an observation that survived far longer than its
    covariates predicted.
    """
    event_times, cumulative, shift = _cox_baseline(model, mstop)
    time, event = _survival_parts(model, newdata)
    eta = np.asarray(model.predict(newdata=newdata, type="link", mstop=mstop),
                     dtype=np.float64).ravel()
    positions = np.searchsorted(event_times, time, side="right") - 1
    baseline = np.zeros(time.size, dtype=np.float64)
    available = positions >= 0
    baseline[available] = cumulative[positions[available]]
    return event - baseline * np.exp(np.clip(eta - shift, -745.0, 709.0))


def _deviance_from_martingale(martingale, event):
    """The usual deviance transform of martingale residuals.

    ``sign(M) * sqrt(-2 (M + delta log(delta - M)))``, which spreads the
    martingale residuals' skewed distribution toward symmetry.
    """
    inner = martingale + np.where(
        event > 0,
        event * np.log(np.maximum(event - martingale, 1e-300)),
        0.0,
    )
    return np.sign(martingale) * np.sqrt(np.maximum(-2.0 * inner, 0.0))


def _deviance(model, newdata=None, mstop=None):
    """Signed square-root deviance contributions for a likelihood family."""
    name = _family_name(model)
    if name == "CoxPH":
        _time, event = _survival_parts(model, newdata)
        return _deviance_from_martingale(
            _martingale(model, newdata, mstop), event
        )

    y = _observed(model, newdata)
    mu = np.asarray(model.predict(newdata=newdata, type="response", mstop=mstop),
                    dtype=np.float64).ravel()
    y = y[: mu.size]
    sign = np.sign(y - mu)

    if name in ("Gaussian", "GaussReg"):
        # The saturated Gaussian fit reproduces y exactly, so the deviance
        # contribution is the squared residual and the signed root is the
        # residual itself.
        return y - mu
    if name == "Poisson":
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(y > 0, y * np.log(y / np.maximum(mu, 1e-300)), 0.0)
        return sign * np.sqrt(np.maximum(2.0 * (term - (y - mu)), 0.0))
    if name == "Binomial":
        mu = np.clip(mu, 1e-12, 1 - 1e-12)
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(y > 0, y * np.log(y / mu), 0.0) + np.where(
                y < 1, (1 - y) * np.log((1 - y) / (1 - mu)), 0.0
            )
        return sign * np.sqrt(np.maximum(2.0 * term, 0.0))

    raise NotImplementedError(_refuse(model, "deviance"))


def _working(model, newdata=None, mstop=None):
    """The fitted family's negative gradient at the current fit.

    Computed by the same core routine the boosting loop uses, so it is the
    vector the next step would regress on rather than a re-derivation of it.
    """
    from mboostrs._rust import family_diagnostics
    from mboostrs.explore import _fitted_nuisance
    from mboostrs.families import _family_params, _family_rust_name

    family = model._family
    eta = np.asarray(model.predict(newdata=newdata, type="link", mstop=mstop),
                     dtype=np.float64).ravel()
    y = _observed(model, newdata)
    weights = np.asarray(model.weights, dtype=np.float64)
    if weights.size != eta.size:
        weights = np.ones(eta.size, dtype=np.float64)

    # The next step's gradient is taken at the fit's own nuisance estimate
    # (NBinomial dispersion, GammaReg shape, ...), not the constructor default.
    scalar, vector = (_fitted_nuisance(model, mstop)
                      if hasattr(model, "nuisance") else (None, None))
    _offset, gradient, _risk = family_diagnostics(
        _family_rust_name(family),
        _family_params(family),
        np.ascontiguousarray(y),
        np.ascontiguousarray(eta),
        np.ascontiguousarray(weights),
        scalar,
        vector,
    )
    return np.asarray(gradient, dtype=np.float64)


def residuals(model, kind="response", newdata=None, mstop=None):
    """Residuals of the named kind, or a refusal saying why there are none.

    See the module docstring for what each kind means. ``kind`` is checked
    against the family's declared set, so a kind that is undefined raises with
    the alternative rather than returning a different quantity under the
    requested name.
    """
    if kind not in RESIDUAL_KINDS:
        raise ValueError(
            f"kind must be one of {RESIDUAL_KINDS}, got {kind!r}"
        )

    name = _family_name(model)
    contract = family_likelihood(getattr(model, "_family", None))

    if kind == "working":
        return _working(model, newdata, mstop)

    if kind == "martingale":
        if name != "CoxPH":
            raise NotImplementedError(_refuse(model, "martingale"))
        return _martingale(model, newdata, mstop)

    if kind == "deviance":
        if contract is None or "deviance" not in contract.residual_kinds:
            raise NotImplementedError(_refuse(model, "deviance"))
        return _deviance(model, newdata, mstop)

    if kind == "pearson":
        if contract is None or contract.variance_function is None:
            raise NotImplementedError(_refuse(model, "pearson"))
        y = _observed(model, newdata)
        mu = np.asarray(
            model.predict(newdata=newdata, type="response", mstop=mstop),
            dtype=np.float64,
        ).ravel()
        y = y[: mu.size]
        variance = np.asarray(contract.variance_function(mu), dtype=np.float64)
        return (y - mu) / np.sqrt(np.maximum(variance, 1e-300))

    # kind == "response"
    if contract is None or "response" not in contract.residual_kinds:
        raise NotImplementedError(_refuse(model, "response"))
    y = _observed(model, newdata)
    mu = np.asarray(
        model.predict(newdata=newdata, type="response", mstop=mstop),
        dtype=np.float64,
    )
    if mu.ndim > 1:
        # An ordinal fit predicts a category-probability matrix; the response
        # residual is against its expected category, using the coding the fit
        # actually accepted.
        categories = np.arange(1, mu.shape[1] + 1, dtype=np.float64)
        mu = mu @ categories
    y = y[: mu.size]
    return y - mu
