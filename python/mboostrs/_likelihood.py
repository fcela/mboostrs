"""What each family's risk is, so `logLik` and AIC can refuse what they cannot do.

Component-wise boosting minimises an empirical risk. For some families that
risk is exactly a negative log-likelihood, and `-risk` is a number an
information criterion can use. For others it is a loss chosen for its
robustness or its ranking behaviour, with no probability model behind it, and
negating it produces a number that looks like a log-likelihood and compares
like one without being one.

Three properties settle what a caller may do with a fitted risk.

`is_likelihood`
    Whether the loss comes from a probability model at all. `Laplace`,
    `Quantile`, `Expectile`, `Huber`, `AdaExp`, `HingeLoss`, `AUC`, `Gehan`
    and `Cindex` do not: their risks are robust or ranking losses. An AIC over
    one of these is not an AIC.

`has_normalized_loglik`
    Whether `-risk` is the *complete* log-likelihood, constants included. A
    loss can come from a probability model and still drop terms that do not
    depend on the fitted parameter, because the optimiser does not need them.
    `Gaussian` is the case that matters: its risk is the residual sum of
    squares, which is proportional to the Gaussian NLL for a fixed variance
    but is not one -- it carries neither the variance nor the `2*pi` term, and
    is off by a factor of two besides. The log-likelihood has to be
    reconstructed with an estimated `sigma` rather than read off the risk.
    `GammaReg` re-estimates its shape at every step. The risk omits `log(y)`
    on each row, and `logLik` adds that term back.

`loglik_measure`
    Which quantity the density is over. Two normalized log-likelihoods are
    comparable only if they describe the same observations on the same scale.
    The AFT families integrate against `d log(t)`, not `dt` -- their risk is
    the negative log-likelihood of `log(T)`, missing the `1/t` Jacobian that
    would put it on the time scale -- so an AFT AIC is not comparable with a
    `Gaussian` AIC on the same `time` column, even though both are proper
    likelihoods. `CoxPH` is a *partial* likelihood: valid for comparing Cox
    models to each other, not a density of the data.

Every value here was measured against an independently computed reference NLL
at the fitted `eta`, not read off the loss definitions; see
`tests/test_likelihood_contract.py`, which recomputes them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Measures a normalized log-likelihood can be taken with respect to. Only
#: likelihoods sharing a measure may be compared by an information criterion.
#: ``"response"`` is a density of the response as given; ``"log_response"`` a
#: density of its logarithm; ``"partial"`` a partial likelihood, which is not a
#: density at all and is comparable only within its own family.
LOGLIK_MEASURES = ("response", "log_response", "partial")


@dataclass(frozen=True)
class FamilyLikelihood:
    """What a family's risk is, and what may therefore be computed from it."""

    #: Whether the loss comes from a probability model.
    is_likelihood: bool
    #: Whether ``-risk`` is the complete log-likelihood, constants included.
    has_normalized_loglik: bool
    #: What the density is over. ``None`` when there is no likelihood.
    loglik_measure: str | None = None
    #: Where a log-likelihood comes from, when one is available at all.
    #: ``"negative_risk"`` -- the risk is already the complete NLL, so
    #: ``-risk`` is the answer. ``"gaussian_mle"`` -- the risk is ``0.5 * RSS``
    #: and the log-likelihood is reconstructed by plugging in the MLE of
    #: ``sigma``. ``None`` -- no log-likelihood is available, and
    #: ``unavailable_reason`` says why.
    #:
    #: This is separate from ``has_normalized_loglik``, which is a fact about
    #: the *risk*. Gaussian has a log-likelihood and a risk that is not one.
    loglik_source: str | None = None
    #: Residual kinds the family defines, beyond the working gradient.
    residual_kinds: tuple[str, ...] = ()
    #: ``V(mu)`` for the Pearson residual, or ``None`` when the family declares
    #: no variance function and Pearson residuals are therefore undefined.
    variance_function: object = None
    #: Why the log-likelihood is unavailable, phrased as the alternative the
    #: caller should use. Only set when ``has_normalized_loglik`` is False.
    unavailable_reason: str = ""


def _binomial_variance(mu):
    return mu * (1.0 - mu)


def _identity_variance(mu):
    return np.ones_like(np.asarray(mu, dtype=np.float64))


def _poisson_variance(mu):
    return np.asarray(mu, dtype=np.float64)


def _gamma_variance(mu):
    mu = np.asarray(mu, dtype=np.float64)
    return mu * mu


#: One row per family, keyed by the Python class name.
#:
#: A family absent from this table has no declared contract, which
#: :func:`family_likelihood` reports as such rather than guessing: a missing
#: row must not read as "not a likelihood", because the two lead to different
#: corrections.
FAMILY_LIKELIHOOD: dict[str, FamilyLikelihood] = {
    # ── likelihoods whose risk is the complete NLL ──────────────────
    "Binomial": FamilyLikelihood(
        is_likelihood=True,
        has_normalized_loglik=True,
        loglik_source="negative_risk",
        loglik_measure="response",
        residual_kinds=("response", "pearson", "deviance"),
        variance_function=_binomial_variance,
    ),
    "Poisson": FamilyLikelihood(
        is_likelihood=True,
        has_normalized_loglik=True,
        loglik_source="negative_risk",
        loglik_measure="response",
        residual_kinds=("response", "pearson", "deviance"),
        variance_function=_poisson_variance,
    ),
    "NBinomial": FamilyLikelihood(
        is_likelihood=True,
        has_normalized_loglik=True,
        loglik_source="negative_risk",
        loglik_measure="response",
        residual_kinds=("response",),
        # The variance is mu + mu^2/sigma, and sigma is profiled out per fit,
        # so it is not a function of mu alone and cannot live in this table.
        variance_function=None,
    ),
    "Hurdle": FamilyLikelihood(
        is_likelihood=True,
        has_normalized_loglik=True,
        loglik_source="negative_risk",
        # The risk is the complete NLL of the zero-truncated negative
        # binomial, lgamma(y + 1) included, so it is a mass on the positive
        # counts and compares with other fits to those same counts.
        loglik_measure="response",
        residual_kinds=("response",),
        variance_function=None,
    ),
    # ── likelihoods on the log scale ────────────────────────────────
    #
    # The AFT risks are the NLL of log(T): the `1/t` Jacobian that would put
    # them on the time scale is absent (measured). They are proper
    # likelihoods, but for a different quantity than a `Gaussian` fit of the
    # same `time` column, so an information criterion may not compare them.
    "Weibull": FamilyLikelihood(
        is_likelihood=True,
        has_normalized_loglik=True,
        loglik_source="negative_risk",
        loglik_measure="log_response",
        residual_kinds=(),
    ),
    "Loglog": FamilyLikelihood(
        is_likelihood=True,
        has_normalized_loglik=True,
        loglik_source="negative_risk",
        loglik_measure="log_response",
        residual_kinds=(),
    ),
    "Lognormal": FamilyLikelihood(
        is_likelihood=True,
        has_normalized_loglik=True,
        loglik_source="negative_risk",
        loglik_measure="log_response",
        residual_kinds=(),
    ),
    # ── partial likelihood ──────────────────────────────────────────
    "CoxPH": FamilyLikelihood(
        is_likelihood=True,
        # Measured: the risk equals the Breslow negative log partial
        # likelihood, to 2e-13. A partial likelihood is not a density of the
        # data -- the baseline hazard is profiled out, not estimated -- so it
        # ranks Cox models against each other and nothing else.
        has_normalized_loglik=True,
        loglik_source="negative_risk",
        loglik_measure="partial",
        residual_kinds=("martingale", "deviance"),
    ),
    # ── likelihoods whose risk drops terms ──────────────────────────
    "Gaussian": FamilyLikelihood(
        is_likelihood=True,
        # The risk is the residual sum of squares (measured; `mboost`'s
        # Gaussian loss is `(y - f)^2`). The log-likelihood exists but has to
        # be reconstructed with an estimated sigma; `logLik` does that, so this
        # flag describes the *risk*, not the availability of the number.
        has_normalized_loglik=False,
        loglik_source="gaussian_mle",
        loglik_measure="response",
        residual_kinds=("response", "pearson", "deviance"),
        variance_function=_identity_variance,
        unavailable_reason=(
            "the Gaussian risk is the residual sum of squares, which omits "
            "the variance and the 2*pi term and is twice the corresponding "
            "NLL term besides; the log-likelihood is reconstructed with the "
            "MLE of sigma rather than read off the risk"
        ),
    ),
    "GaussReg": FamilyLikelihood(
        is_likelihood=True,
        has_normalized_loglik=False,
        loglik_source="gaussian_mle",
        loglik_measure="response",
        residual_kinds=("response", "pearson", "deviance"),
        variance_function=_identity_variance,
        unavailable_reason=(
            "the squared-error risk omits the variance and the 2*pi term; the "
            "log-likelihood is reconstructed with the MLE of sigma"
        ),
    ),
    "GammaReg": FamilyLikelihood(
        is_likelihood=True,
        # The risk is R's plloss, lgamma(s) + s y e^-f - s log y - s log s
        # + s f, with the shape s re-estimated at every step: the Gamma NLL
        # minus the log y of each row. `gamma_shape` adds that term back.
        has_normalized_loglik=False,
        loglik_source="gamma_shape",
        loglik_measure="response",
        residual_kinds=("response", "pearson"),
        variance_function=_gamma_variance,
    ),
    # ── ordinal and multi-class ─────────────────────────────────────
    "PropOdds": FamilyLikelihood(
        is_likelihood=True,
        # Measured: the risk equals the categorical NLL of the observed
        # category under the fitted cumulative probabilities, to 3e-13.
        has_normalized_loglik=True,
        loglik_source="negative_risk",
        loglik_measure="response",
        residual_kinds=("response",),
    ),
    "Multinomial": FamilyLikelihood(
        is_likelihood=True,
        # The risk sums `-y_k log(p_k)` over the K-1 non-reference columns
        # only. An observation in the reference class has `y_k = 0` in every
        # one of them, so it contributes exactly zero -- measured: the risk
        # equals the categorical NLL summed over non-reference rows alone, to
        # 3e-14. The omitted term is `-log(p_reference)`, which moves with the
        # fit, so it is a missing part of the objective rather than a constant
        # that cancels.
        #
        # `mboost::Multinomial` has the same loss (`-y * log(p)` over the same
        # K-1 columns), so this is upstream behaviour and is kept for parity.
        # The gradient is `y_k - p_k`, which is correct for reference-class
        # observations, so the *fit* is unaffected; what the omission reaches
        # is every risk-valued output -- the reported risk, the risk path, and
        # any information criterion. `cvrisk` and stability selection refuse
        # outright rather than select `mstop` on the partial risk.
        has_normalized_loglik=False,
        loglik_measure="response",
        residual_kinds=(),
        unavailable_reason=(
            "the multinomial risk omits every observation in the reference "
            "class, because its loss is summed over the K-1 non-reference "
            "columns where such an observation is zero. The omitted term "
            "moves with the fit, so -risk is not the multinomial "
            "log-likelihood and differs from it by an amount that depends on "
            "how many observations fall in the reference class. This matches "
            "mboost::Multinomial; the fit still descends the full categorical "
            "NLL, whose negative gradient (y_k - p_k) is also correct on "
            "reference-class observations. negative_risk() returns the "
            "mboost-comparable reported value, which is NOT the quantity the "
            "fit minimised"
        ),
    ),
    # ── losses that are not likelihoods ─────────────────────────────
    #
    # Each names the quantity it targets, because the useful correction is
    # "this fit estimates a median, so compare medians", not "no AIC".
    "Laplace": FamilyLikelihood(
        is_likelihood=False,
        has_normalized_loglik=False,
        residual_kinds=("response",),
        unavailable_reason=(
            "absolute-error loss is not a log-likelihood: it targets the "
            "conditional median with no distributional assumption behind it. "
            "Compare fits by cross-validated risk (cvrisk), or use a "
            "likelihood family if an information criterion is required"
        ),
    ),
    "Quantile": FamilyLikelihood(
        is_likelihood=False,
        has_normalized_loglik=False,
        residual_kinds=("response",),
        unavailable_reason=(
            "the check loss is not a log-likelihood: it targets a conditional "
            "quantile with no distributional assumption behind it. Compare "
            "fits by cross-validated risk (cvrisk) at the same tau"
        ),
    ),
    "Expectile": FamilyLikelihood(
        is_likelihood=False,
        has_normalized_loglik=False,
        residual_kinds=("response",),
        unavailable_reason=(
            "the asymmetric squared-error loss is not a log-likelihood: it "
            "targets a conditional expectile. Compare fits by cross-validated "
            "risk (cvrisk) at the same tau"
        ),
    ),
    "Huber": FamilyLikelihood(
        is_likelihood=False,
        has_normalized_loglik=False,
        residual_kinds=("response",),
        unavailable_reason=(
            "the Huber loss is not a log-likelihood: it is a robustified "
            "squared-error loss whose changeover point is re-estimated as the "
            "fit proceeds. Compare fits by cross-validated risk (cvrisk)"
        ),
    ),
    "AdaExp": FamilyLikelihood(
        is_likelihood=False,
        has_normalized_loglik=False,
        residual_kinds=(),
        unavailable_reason=(
            "the exponential loss is not a log-likelihood. Use Binomial for a "
            "classification model with one, or compare fits by cross-validated "
            "risk (cvrisk)"
        ),
    ),
    "HingeLoss": FamilyLikelihood(
        is_likelihood=False,
        has_normalized_loglik=False,
        residual_kinds=(),
        unavailable_reason=(
            "the hinge loss is not a log-likelihood: it is a margin loss with "
            "no probability model. Use Binomial for a classification model "
            "with one, or compare fits by cross-validated risk (cvrisk)"
        ),
    ),
    "AUC": FamilyLikelihood(
        is_likelihood=False,
        has_normalized_loglik=False,
        residual_kinds=(),
        unavailable_reason=(
            "the smoothed 1 - AUC loss is a ranking objective, not a "
            "log-likelihood: it depends only on the order of the fitted "
            "values. Compare fits by cross-validated risk (cvrisk)"
        ),
    ),
    "Gehan": FamilyLikelihood(
        is_likelihood=False,
        has_normalized_loglik=False,
        residual_kinds=(),
        unavailable_reason=(
            "the Gehan rank loss is a ranking objective, not a "
            "log-likelihood. Use Weibull, Loglog or Lognormal for an AFT "
            "model with one, or CoxPH for a partial likelihood"
        ),
    ),
    "Cindex": FamilyLikelihood(
        is_likelihood=False,
        has_normalized_loglik=False,
        residual_kinds=(),
        unavailable_reason=(
            "the smoothed concordance loss is a ranking objective, not a "
            "log-likelihood: it depends only on the order of the fitted "
            "values. Use Weibull, Loglog or Lognormal for an AFT model with "
            "one, or CoxPH for a partial likelihood"
        ),
    ),
}


#: Distributional (LSS) families fit every parameter of a named distribution by
#: minimising its negative log-likelihood, so their risk *is* that NLL with no
#: dropped constants -- there is no nuisance parameter left implicit for a
#: constant to hide in. Measured on `GaussianLSS`, `NBinomialLSS` and
#: `LogNormalLSS`, each agreeing with an independently computed reference to
#: better than 1e-12; `tests/test_likelihood_contract.py` re-checks them.
#:
#: The survival ones carry the same caveat as their standard counterparts:
#: their risk is the NLL of `log(T)`, not of `T`.
_LSS_SURVIVAL_MEASURE = "log_response"
_LSS_RESPONSE_MEASURE = "response"


def _lss_contract(family) -> FamilyLikelihood | None:
    """The contract for a distributional family, derived from its own metadata.

    Enumerating two dozen LSS classes here would go stale the moment one is
    added; the properties that matter are already on the family object.
    """
    if not hasattr(family, "n_params") or not hasattr(family, "param_names"):
        return None
    measure = (_LSS_SURVIVAL_MEASURE if getattr(family, "_survival", False)
               else _LSS_RESPONSE_MEASURE)
    return FamilyLikelihood(
        is_likelihood=True,
        has_normalized_loglik=True,
        loglik_measure=measure,
        loglik_source="negative_risk",
        residual_kinds=("response", "quantile"),
    )


def family_likelihood(family) -> FamilyLikelihood | None:
    """The likelihood contract for ``family``, or ``None`` if undeclared.

    ``None`` means the family has no row in the table, which is different from
    "not a likelihood": the first is a gap in this file, the second is a fact
    about the loss. Callers must distinguish them, because a missing row is
    fixed by adding one and a non-likelihood is not.
    """
    name = type(family).__name__ if not isinstance(family, str) else family
    declared = FAMILY_LIKELIHOOD.get(name)
    if declared is not None:
        return declared
    if not isinstance(family, str):
        return _lss_contract(family)
    return None


def loglik_refusal(family) -> str:
    """The message explaining why this family has no usable log-likelihood.

    Returns an empty string when one is available.
    """
    name = type(family).__name__ if not isinstance(family, str) else family
    contract = family_likelihood(family)
    if contract is None:
        return (
            f"family {name!r} has no declared likelihood contract, so "
            f"mboostrs cannot say whether its risk is a log-likelihood. This "
            f"is a gap in mboostrs, not a property of the family: add a row "
            f"for {name!r} to mboostrs._likelihood.FAMILY_LIKELIHOOD. Use "
            f"negative_risk() for the quantity the fit minimised."
        )
    if contract.loglik_source is not None:
        return ""
    reason = contract.unavailable_reason or "its risk is not a log-likelihood"
    return f"logLik is not available for {name}: {reason}."


#: Families whose likelihood is a probability mass: the value at an observed
#: count or category is a probability between 0 and 1.
_COUNTING_FAMILIES = frozenset({
    "Binomial", "Poisson", "NBinomial", "Hurdle", "PropOdds", "Multinomial",
    "PoissonLSS", "GeometricLSS", "NBinomialLSS", "NegativeBinomialLSS", "ZIPoLSS",
    "ZINBLSS", "ZeroAdjustedPoissonLSS", "PIGLSS", "BinomialLSS",
})
#: Families whose likelihood is a probability at ``y = 0`` and a density
#: elsewhere.
_ZERO_ATOM_FAMILIES = frozenset({"ZeroAdjustedGammaLSS"})

#: Names of the reference measures :func:`reference_measure` returns.
REFERENCE_MEASURES = ("counting", "lebesgue", "zero_atom_lebesgue", "partial")


def reference_measure(family) -> str | None:
    """The measure a family's likelihood is a density with respect to, or None.

    ``"counting"`` for a probability mass function (Poisson, Binomial, the
    count LSS families): each observation contributes a probability.
    ``"lebesgue"`` for a density of a continuous response (Gaussian, Gamma,
    the AFT families): each observation contributes a density value, which
    depends on the units of ``y`` and can exceed 1. ``"zero_atom_lebesgue"``
    for a zero-adjusted continuous family, a probability at zero and a density
    elsewhere. ``"partial"`` for the Cox partial likelihood. None for a loss
    that is not a likelihood, or a family with no declared contract.

    A log probability and a log density of the same counts differ by a term
    that depends on the units, so an information criterion or a mean negative
    log-likelihood can rank two families only when this value is the same.
    """
    contract = family_likelihood(family)
    if contract is None or not contract.is_likelihood:
        return None
    if contract.loglik_measure == "partial":
        return "partial"
    name = type(family).__name__ if not isinstance(family, str) else family
    if name in _COUNTING_FAMILIES:
        return "counting"
    if name in _ZERO_ATOM_FAMILIES:
        return "zero_atom_lebesgue"
    return "lebesgue"


def comparable_measures(families) -> tuple[bool, str]:
    """Whether an information criterion may compare these fitted families.

    Returns ``(True, "")`` when every family is a likelihood on the same
    measure, and ``(False, reason)`` otherwise. The reason names which family
    broke the comparison and why, because "incomparable" alone leaves the
    caller to work out which of several models is the problem. The measure is
    both the quantity the likelihood describes (the response or its
    logarithm) and the reference measure of :func:`reference_measure`: a
    Poisson mass and a Gaussian density of the same counts are not ranked.
    """
    contracts = {}
    for family in families:
        name = type(family).__name__ if not isinstance(family, str) else family
        contracts[name] = family_likelihood(family)

    missing = sorted(n for n, c in contracts.items() if c is None)
    if missing:
        return False, (
            f"no declared likelihood contract for {', '.join(missing)}; add a "
            f"row to mboostrs._likelihood.FAMILY_LIKELIHOOD"
        )

    not_likelihood = sorted(n for n, c in contracts.items() if not c.is_likelihood)
    if not_likelihood:
        return False, (
            f"{', '.join(not_likelihood)} minimise a loss that is not a "
            f"log-likelihood, so an information criterion over them is not an "
            f"information criterion. Compare by cross-validated risk instead"
        )

    measures = {c.loglik_measure for c in contracts.values()}
    if len(measures) > 1:
        by_measure = {}
        for name, contract in contracts.items():
            by_measure.setdefault(contract.loglik_measure, []).append(name)
        parts = "; ".join(
            f"{measure}: {', '.join(sorted(names))}"
            for measure, names in sorted(by_measure.items())
        )
        return False, (
            f"these log-likelihoods are taken with respect to different "
            f"quantities, so their constants do not cancel and the difference "
            f"between them is not a likelihood ratio ({parts})"
        )

    references = {}
    for family in families:
        name = type(family).__name__ if not isinstance(family, str) else family
        references.setdefault(reference_measure(family), set()).add(name)
    if len(references) > 1:
        parts = "; ".join(
            f"{measure}: {', '.join(sorted(names))}"
            for measure, names in sorted(references.items(), key=lambda item: str(item[0]))
        )
        return False, (
            f"these likelihoods use different reference measures: a probability "
            f"mass and a probability density of the same observations differ by "
            f"a term that depends on the units of the response, so their "
            f"log-likelihoods are not on one scale ({parts})"
        )

    if measures == {"partial"}:
        return True, ""
    return True, ""


def normalized_loglik(model) -> float:
    """The fitted model's log-likelihood, or raise saying why there is none.

    ``"negative_risk"`` families hand back ``-risk`` directly, because their
    risk is already the complete NLL. ``"gaussian_mle"`` reconstructs the
    Gaussian log-likelihood by plugging in ``sigma^2 = RSS / n``, the MLE the
    squared-error risk implies but does not carry -- so the number is a real
    log-likelihood rather than a negated loss, and is smaller than the
    ``mboost`` value for the same fit by exactly the constants the risk drops.
    """
    family = getattr(model, "_family", None)
    refusal = loglik_refusal(family)
    if refusal:
        raise NotImplementedError(
            refusal + " Use negative_risk() if the value you want is the "
            "quantity the fit minimised, negated."
        )

    contract = family_likelihood(family)
    risk = np.asarray(model._risk, dtype=np.float64)
    mstop = model.mstop

    if contract.loglik_source == "negative_risk":
        return float(-risk[mstop])

    if contract.loglik_source == "gamma_shape":
        observed = _observed_for_loglik(model)
        weights = getattr(model, "_weights", None)
        weights = (np.ones(observed.size) if weights is None
                   else np.asarray(weights, dtype=np.float64))
        return float(-risk[mstop] - np.sum(weights * np.log(observed)))

    if contract.loglik_source == "gaussian_mle":
        fitted = np.asarray(model.fitted(), dtype=np.float64).ravel()
        n = fitted.size
        observed = _observed_for_loglik(model)
        rss = float(np.sum((observed - fitted) ** 2))
        sigma2 = rss / n
        if sigma2 <= 0.0:
            return float("-inf")
        return float(-n / 2.0 * np.log(2 * np.pi * sigma2) - rss / (2 * sigma2))

    raise AssertionError(
        f"unreachable: loglik_source {contract.loglik_source!r} has no branch"
    )


def _observed_for_loglik(model):
    """The response vector, aligned with ``model.fitted()``."""
    from mboostrs.diagnostics import _observed_response

    observed = np.asarray(_observed_response(model), dtype=np.float64).ravel()
    fitted = np.asarray(model.fitted(), dtype=np.float64).ravel()
    if observed.size != fitted.size:
        observed = observed[: fitted.size]
    return observed
