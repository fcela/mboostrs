import warnings

import numpy as np

from mboostrs._control import boost_control
from mboostrs._validate import (
    check_int_at_least,
    check_positive,
    check_unit_interval,
)


class Gaussian:
    """Squared error loss, ``0.5 * (y - f)**2``, with an identity link.

    The default family. The negative gradient is the raw residual ``y - f``,
    and the offset is the weighted mean of the response. Use it for a
    continuous response whose conditional distribution is roughly symmetric
    with no heavy tails; ``Huber`` or ``Quantile(0.5)`` are the robust
    alternatives when it is not.
    """


class Binomial:
    """Binomial log-likelihood with a logit link, for a two-class response.

    The response is coded ``{0, 1}``. ``predict(type="response")`` returns
    ``P(y = 1) = (1 + exp(-2f))**-1`` — note the factor of two, which follows
    `mboost`'s internal ``±1`` coding, so the linear predictor is half the
    usual log-odds. The offset is half the log-odds of the weighted class
    frequency.
    """


class Poisson:
    """Poisson log-likelihood with a log link, for counts.

    ``mu = exp(f)`` and the negative gradient is the response-scale residual
    ``y - mu``. The offset is the log of the weighted mean rate. Poisson fixes
    the variance equal to the mean; when the data is overdispersed, use
    :class:`NBinomial`, and check with a PIT histogram rather than assuming.
    """


def _check_family_response_scale(value):
    """``"auto"``, ``"none"`` (also ``None``) or a positive number."""
    if value is None:
        return "none"
    if isinstance(value, str):
        if value not in ("auto", "none"):
            raise ValueError(
                f'response_scale must be "auto", "none" (or None) or a positive number, '
                f'got {value!r}')
        return value
    scale = float(value)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"response_scale must be finite and positive, got {value!r}")
    return scale


def _response_scale_param(value):
    """The binding's encoding: absent for the raw response, 0 for ``"auto"``."""
    if value == "none":
        return {}
    return {"response_scale": 0.0 if value == "auto" else float(value)}





class Laplace:
    """Absolute error loss, ``|y - f|``, with an identity link.

    The negative gradient is ``sign(y - f)``, so every observation contributes
    the same magnitude regardless of how far off it is, and the fit targets the
    conditional **median** rather than the mean. The offset is the weighted
    median. Use it when outliers should not pull the fit.

    Parameters
    ----------
    response_scale : {"auto", "none"} or float, default "auto"
        The scale the response is divided by before fitting. The negative
        gradient of this loss does not grow with the response, so each
        boosting step moves the fit by about ``nu`` times a constant in
        response units: without a scale the same data in dollars and in
        thousands of dollars give different fits at the same ``mstop``, and a
        response in large units barely leaves its offset. ``"auto"`` fits the
        response divided by its weighted SD (the SD of the rows each fit sees)
        and maps the model back exactly, so predictions, coefficients and the
        risk are in the original units. ``"none"`` (or ``None``) is R mboost's
        raw-units path; a number fixes the scale.
    """

    def __init__(self, response_scale="auto"):
        self.response_scale = _check_family_response_scale(response_scale)


class GammaReg:
    """Gamma log-likelihood with a log link, for a positive continuous response.

    ``mu = exp(f)``, so the response is strictly positive and its variance
    grows with the square of the mean — constant coefficient of variation.
    ``nuisance()`` is the shape in ``Var(Y) = mu**2 / shape``. A Gamma GLM
    reports the dispersion, which is the reciprocal of that shape.
    Being right-skewed does not by itself make a response Gamma; check the fit
    with a PIT histogram before committing to it.
    """


class GaussReg:
    """Squared error loss with an identity link, without R's `mboost` equivalent.

    Fits the same model as :class:`Gaussian`. It exists because `mboost` names
    its squared-error family `Gaussian` while some references reserve that name
    for the likelihood rather than the loss; both are provided so a translated
    script keeps working.
    """


class AdaExp:
    """Exponential loss, ``exp(-(2y - 1) f)`` — the AdaBoost loss.

    Classification with a two-class response coded ``{0, 1}``. It penalises
    confident mistakes far more sharply than :class:`Binomial` does, which
    makes it more sensitive to mislabelled observations.
    """


class Quantile:
    """Check (pinball) loss for quantile regression.

    Targets the conditional ``tau``-quantile: residuals above the fit are
    weighted ``tau`` and those below ``1 - tau``, so ``tau=0.5`` gives median
    regression. The offset is the weighted ``qoffset`` quantile of the
    response, by default the quantile being fitted.

    Fitting several ``tau`` values gives a picture of the whole conditional
    distribution without assuming its shape, which is the alternative to the
    distributional families when only a few quantiles are needed.

    Three defaults differ from R mboost's ``QuantReg``, chosen on 18
    quantile fits (six OpenML-CTR23 data sets, ``tau`` 0.1, 0.5, 0.9): the
    offset starts at ``tau`` rather than the median; a fit given no
    ``control`` (and ``tune_mstop`` given no ``nu``) takes steps of
    ``nu = 0.3`` rather than 0.1, which cut the median time 2.7-fold with
    the pinball loss 1.6% lower on average; and the response is scaled (see
    ``response_scale``). ``Quantile(tau, qoffset=0.5, response_scale=None)``
    with ``boost_control(..., nu=0.1)`` is R's ``QuantReg``.

    Parameters
    ----------
    tau : float
        The quantile to fit, in (0, 1).
    qoffset : float, optional
        Quantile of the response used for the starting offset. ``None``
        (default) is ``tau``; R's ``QuantReg`` uses 0.5 whatever ``tau``.
    response_scale : {"auto", "none"} or float, default "auto"
        The scale the response is divided by before fitting. The negative
        gradient of this loss does not grow with the response, so each
        boosting step moves the fit by about ``nu`` times a constant in
        response units: without a scale the same data in dollars and in
        thousands of dollars give different fits at the same ``mstop``, and a
        response in large units barely leaves its offset. ``"auto"`` fits the
        response divided by its weighted SD (the SD of the rows each fit sees)
        and maps the model back exactly, so predictions, coefficients and the
        risk are in the original units. ``"none"`` (or ``None``) is R mboost's
        raw-units path; a number fixes the scale.
    """

    #: The step length of a fit given no control (R mboost uses 0.1).
    default_nu = 0.3

    def __init__(self, tau=0.5, qoffset=None, response_scale="auto"):
        self.tau = check_unit_interval("tau", tau)
        self.qoffset = check_unit_interval("qoffset", self.tau if qoffset is None else qoffset)
        self.response_scale = _check_family_response_scale(response_scale)


class Expectile:
    """Asymmetric squared error loss for expectile regression.

    The expectile analogue of :class:`Quantile`: squared rather than absolute
    deviations, weighted ``tau`` above the fit and ``1 - tau`` below. Expectiles
    are smoother in the data than quantiles and cheaper to fit, at the cost of
    a less direct interpretation — the ``tau``-expectile is not the
    ``tau``-quantile.

    Parameters
    ----------
    tau : float
        The expectile to fit, in (0, 1). ``0.5`` recovers the mean.
    """

    def __init__(self, tau=0.5):
        self.tau = check_unit_interval("tau", tau)


class Huber:
    """Huber's robust loss: quadratic near zero, linear in the tails.

    Behaves like :class:`Gaussian` for residuals within ``delta`` and like
    :class:`Laplace` beyond it, so it targets the mean while limiting the
    influence of outliers. The negative gradient is ``y - f`` clipped to
    ``[-delta, delta]``.

    Parameters
    ----------
    delta : float, optional
        The changeover point. When omitted it is set to the median absolute
        deviation of the current residuals and re-estimated as the fit
        proceeds, which is R `mboost`'s adaptive default.
    """

    def __init__(self, delta=None):
        # A non-positive delta makes the loss decrease with the residual and
        # reports a negative value as risk.
        self.delta = check_positive("delta", delta, allow_none=True)


class NBinomial:
    """Negative binomial log-likelihood with a log link, for overdispersed counts.

    ``nuisance()`` is the size ``sigma`` in ``Var(Y) = mu + mu**2 / sigma``,
    profiled at each iteration by a one-dimensional Brent search matching R's
    ``optimize``. :class:`Hurdle` uses the reciprocal parameterization (its
    ``sigma`` is the dispersion, and the size is ``1 / sigma``). A statsmodels
    NB2 fit reports the reciprocal as ``alpha``
    (``Var = mu + alpha * mu**2``); using ``nuisance()`` as that alpha
    overstates the extra-Poisson variance by ``sigma**2``. The negative
    gradient depends on ``sigma``, so it is re-estimated as the fit proceeds
    rather than fixed in advance.

    The fitted model's ``size`` property names this value explicitly.
    ``NBinomial()`` takes no fixed size, ``alpha`` or ``dispersion`` argument;
    the core estimates the size at every step.
    """

    def __init__(self, **kwargs):
        if kwargs:
            raise TypeError(
                f"NBinomial does not accept {', '.join(sorted(kwargs))}: "
                "the fitted nuisance is the negative-binomial size, estimated "
                "during fitting and available as model.size. statsmodels NB2 "
                "alpha is its reciprocal (alpha = 1/size); alpha, dispersion "
                "and fixed size arguments are unsupported.")


class Hurdle:
    """The count part of a hurdle model: a zero-truncated negative binomial.

    A negative binomial with a log link, conditioned on the count being
    positive, as ``mboost::Hurdle()``. Fit it to the positive counts only;
    whether a count is positive is a separate binary model (``Binomial`` on
    ``y > 0``), and the two together describe counts with more or fewer zeros
    than one negative binomial allows. The dispersion ``sigma`` is
    re-estimated at every step by a Brent search on (0, 100), matching R's
    ``optimize``. ``sigma`` is the gamlss dispersion, so the variance is
    ``mu + sigma * mu**2`` and the size is ``1 / sigma``. That is the
    reciprocal of :class:`NBinomial`, whose ``sigma`` is the size
    (``mu + mu**2 / sigma``). ``predict(type="response")`` returns ``exp(f)``,
    the mean of the untruncated negative binomial, as mboost does; the mean of
    the truncated count is that divided by ``1 - (1 + exp(f) * sigma) ** (-1 / sigma)``.
    """


class PropOdds:
    """Proportional-odds cumulative model for an ordered categorical response.

    A single linear predictor ``f`` shifts a set of ordered thresholds, giving
    cumulative probabilities ``P(y <= k) = (1 + exp(-(theta_k - f)))**-1``. The
    thresholds are nuisance parameters estimated alongside the boosting steps.

    The response is given in one of three ways:

    * an **ordered categorical column** (a pandas ``Categorical`` with
      ``ordered=True`` or a polars ``Enum``), whose declared order is used;
    * any column with ``levels=``, the categories in order, lowest first:
      ``PropOdds(levels=["low", "mid", "high"])`` or, for a zero-based integer
      code, ``PropOdds(levels=[0, 1, 2])``;
    * integers ``{1, ..., K}`` with ``n_levels=K``, R's 1-based factor codes.

    The order has to be declared rather than read from the values: a subset
    of the rows (a cross-validation fold, a conformal split) can lack the
    lowest class, and a coding inferred from the values it holds would shift
    every class by one. ``predict(type="response")`` returns an ``(n, K)``
    probability matrix whose column ``k`` is the probability of the ``k``-th
    level.

    Parameters
    ----------
    n_levels : int, optional
        Number of ordered categories, at least 3. Taken from ``levels`` or
        from an ordered categorical column when those give it. (Two ordered
        categories is a binary response, which :class:`Binomial` fits.)
    levels : sequence, optional
        The categories in order, lowest first.
    """

    def __init__(self, n_levels=None, levels=None):
        self.levels = None if levels is None else tuple(levels)
        if self.levels is not None:
            if len(set(self.levels)) != len(self.levels):
                raise ValueError("PropOdds levels must be distinct")
            if n_levels is not None and n_levels != len(self.levels):
                raise ValueError(
                    f"n_levels={n_levels} but levels names {len(self.levels)} categories")
            n_levels = len(self.levels)
        # Two ordered categories is a binary response, which Binomial fits;
        # one is not a categorical response at all.
        self.n_levels = check_int_at_least("n_levels", n_levels, 3,
                                           allow_none=True)


class CoxPH:
    """Cox partial likelihood for right-censored survival data.

    Takes a ``Surv(time, status)`` response, with ``status`` 1 for an observed
    event and 0 for a censored observation. Fits the log hazard ratio, leaving
    the baseline hazard unspecified, so predictions order patients by risk
    rather than giving survival times. For an absolute time scale, use one of
    the accelerated-failure-time families instead.
    """


class Weibull:
    """Weibull accelerated-failure-time model for right-censored data.

    Takes a ``Surv(time, status)`` response and models log survival time
    directly, so the fit is a statement about time rather than about a hazard
    ratio. The scale parameter is a nuisance estimated alongside the path.
    """


class Loglog:
    """Log-logistic accelerated-failure-time model for right-censored data.

    Takes a ``Surv(time, status)`` response. Like :class:`Lognormal` it allows
    a non-monotone hazard, with heavier tails, and its survival function has a
    closed form.
    """


class Lognormal:
    """Log-normal accelerated-failure-time model for right-censored data.

    Takes a ``Surv(time, status)`` response and models log survival time as
    normal, which gives a hazard that rises and then falls — appropriate when
    risk peaks some time after the origin rather than decaying monotonically.
    """


class HingeLoss:
    """Hinge loss, ``max(0, 1 - y f)`` — the support vector loss.

    Classification with a two-class response. Observations already classified
    with margin contribute nothing to the gradient, so the fit is driven by the
    points near the boundary.

    Its ``offset()`` returns the AdaBoost log-odds rather than the minimiser of
    its own risk. The risk is piecewise linear in a constant ``c`` with its
    minimum at the boundary ``c = ±1``, so no interior offset can minimise it;
    R's internal `HingeLoss` returns the identical value.

    Parameters
    ----------
    alpha_cost : float
        Asymmetric misclassification cost. ``0.5``, the default, is the
        symmetric hinge loss.
    """

    def __init__(self, alpha_cost=0.5):
        if not 0.0 < alpha_cost < 1.0:
            raise ValueError(f"alpha_cost must be in (0, 1), got {alpha_cost}")
        self.alpha_cost = alpha_cost


class Multinomial:
    """Multinomial logistic likelihood for an unordered K-class response.

    Generalises :class:`Binomial` to more than two classes using a
    Kronecker-structured design: one linear predictor per class, sharing the
    base learners. ``predict(type="response")`` returns an ``(n, K)``
    probability matrix whose rows sum to one.

    Available on :func:`~mboostrs.gamboost` only. :func:`~mboostrs.glmboost`
    rejects any response outside ``{0, 1}`` whatever ``n_classes`` says,
    because the Kronecker-structured design this family needs is built only on
    the ``gamboost`` path. Write ``gamboost("y ~ bols(x)")`` for a linear
    multi-class fit.

    Parameters
    ----------
    n_classes : int, optional
        Number of classes, at least 3. Inferred from the response when
        omitted, on the ``gamboost`` path.
    """

    def __init__(self, n_classes=None):
        # Two classes is a binary response, which Binomial fits; one is not a
        # classification problem.
        self.n_classes = check_int_at_least("n_classes", n_classes, 3,
                                            allow_none=True)


class AUC:
    """Smoothed 1 - AUC ranking loss (mboost::AUC()).

    Response must be a two-level (binary) column; any two distinct raw
    values are accepted -- the Rust core recodes the larger to +1 and the
    smaller to -1, mirroring R's factor level ordering.
    """

    pass


class Gehan:
    """Gehan rank loss for AFT survival models (mboost::Gehan()).

    Response must be a ``Surv(time, event)`` pair (see :class:`CoxPH`).
    Only unit case weights are supported.

    The Gehan loss depends only on pairwise residual differences, so a
    constant offset cancels and the risk is flat in it -- the offset is
    unidentifiable. mboostrs therefore defines ``offset() == 0.0`` rather
    than reproducing R's arbitrary ``optimize()``-on-a-flat-function value;
    fitted values consequently differ from R's by an additive constant
    (gradient, coefficient increments, ``selected``, and risk all match R
    exactly). See ``crates/mboostrs-core/src/families.rs`` (``Gehan``) for
    the full derivation.
    """

    pass


class Cindex:
    """Uno's IPCW-weighted concordance ("C-index") loss (mboost::Cindex()).

    Response must be a ``Surv(time, event)`` pair (see :class:`CoxPH`).
    Case weights must be either 0 or 1 (this is the valid domain, not a
    limitation -- it matches the subsampling/k-fold ``cvrisk()``/``stabsel()``
    fold format, both of which are supported for this family; bootstrap-style
    integer-count folds are not).

    Unlike :class:`Gehan`, the offset is identically 0 in R (verified), so
    raw fitted values match R directly -- no centering deviation.

    Parameters
    ----------
    sigma : float
        Width of the sigmoid smoothing of the (otherwise non-differentiable)
        pairwise concordance indicator. Default 0.1, matching R.
    """

    def __init__(self, sigma=0.1):
        # A non-positive sigma inverts the smoothing sigmoid, so the loss
        # rewards discordant pairs.
        self.sigma = check_positive("sigma", sigma)


__all__ = [
    "Gaussian", "Binomial", "Poisson", "Laplace", "GammaReg", "GaussReg",
    "AdaExp", "Quantile", "Expectile", "Huber", "NBinomial", "Hurdle", "PropOdds",
    "CoxPH", "Weibull", "Loglog", "Lognormal", "HingeLoss", "Multinomial", "AUC", "Gehan",
    "Cindex",
    "boost_control",
]


_FAMILY_NAME_MAP = {
    "gaussian": "gaussian",
    "gaussreg": "gaussreg",
    "binomial": "binomial",
    "poisson": "poisson",
    "laplace": "laplace",
    "gammareg": "gammasreg",
    "gamma": "gammasreg",
    "adaexp": "adaexp",
    "quantile": "quantreg",
    "expectile": "expectreg",
    "huber": "huber",
    "nbinomial": "nbinomial",
    "hurdle": "hurdle",
    "propodds": "propodds",
    "coxph": "coxph",
    "weibull": "weibull",
    "loglog": "loglog",
    "lognormal": "lognormal",
    "hingeloss": "hingeloss",
    "multinomial": "multinomial",
    "auc": "auc",
    "gehan": "gehan",
    "cindex": "cindex",
}


def require_standard_family(family, caller):
    """Reject a distributional family passed to ``glmboost`` or ``gamboost``.

    ``ZIPoLSS`` and the other GAMLSS families are real families, but they have
    one predictor per parameter. Sending one to a single-predictor fit used to
    fail inside the engine as ``unknown family: zipolss``, which reads as
    "this family does not exist" and does not name ``gamboostLSS``.
    """
    if getattr(family, "n_params", None) is not None and hasattr(family, "param_names"):
        name = type(family).__name__
        params = ", ".join(family.param_names)
        raise TypeError(
            f"{name} is a distributional family (parameters: {params}). "
            f"{caller}() fits one predictor. Use glmboostLSS() or gamboostLSS(), "
            f"for example mb.gamboostLSS(formula, data, family=mb.{name}())."
        )


def require_lss_family(family, caller):
    """Reject a single-predictor family passed to an LSS fit.

    ``Poisson()`` on ``gamboostLSS`` used to raise ``AttributeError: 'Poisson'
    object has no attribute 'n_params'``, which does not say to use ``gamboost``.
    """
    if getattr(family, "n_params", None) is None or not hasattr(family, "param_names"):
        name = type(family).__name__
        raise TypeError(
            f"{name} is a single-predictor family. {caller}() needs a distributional "
            f"family such as GaussianLSS or ZIPoLSS. Fit {name} with glmboost() or "
            f"gamboost(), for example mb.gamboost(formula, data, family=mb.{name}())."
        )


def _family_rust_name(family):
    python_name = family.__class__.__name__.lower()
    return _FAMILY_NAME_MAP.get(python_name, python_name)


def _class_prediction(eta, family):
    """Binary predictions as 0/1 codes, using the core decision rule."""
    import numpy as np

    from mboostrs._rust import family_class

    values = np.asarray(eta, dtype=np.float64)
    codes = family_class(_family_rust_name(family), _family_params(family),
                         np.ascontiguousarray(values.ravel()))
    return np.asarray(codes, dtype=np.intp).reshape(values.shape)


def _family_repr(self):
    """``Quantile(tau=0.9, qoffset=0.5)``: the class and its set arguments."""
    args = ", ".join(f"{k}={v!r}" for k, v in vars(self).items()
                     if not k.startswith("_") and v is not None)
    return f"{type(self).__name__}({args})"


for _cls in (Gaussian, Binomial, Poisson, Laplace, GammaReg, GaussReg, AdaExp, Quantile,
             Expectile, Huber, NBinomial, Hurdle, PropOdds, CoxPH, Weibull, Loglog, Lognormal,
             HingeLoss, Multinomial, AUC, Gehan, Cindex):
    if "__repr__" not in _cls.__dict__:
        _cls.__repr__ = _family_repr
del _cls


def _family_params(family):
    if isinstance(family, Quantile):
        return {"tau": family.tau, "qoffset": family.qoffset,
                **_response_scale_param(getattr(family, "response_scale", "none"))}
    if isinstance(family, Laplace):
        return _response_scale_param(getattr(family, "response_scale", "none"))
    if isinstance(family, Expectile):
        return {"tau": family.tau}
    if isinstance(family, Huber):
        if family.delta is not None:
            return {"delta": family.delta}
        return {}
    if isinstance(family, PropOdds):
        # Omit the key when unset rather than sending a sentinel: the binding
        # then applies its own default and can check every value it is given
        # against the >= 3 rule without a magic number to exempt.
        if family.n_levels is not None:
            return {"n_levels": float(family.n_levels)}
        return {}
    if isinstance(family, Multinomial):
        if family.n_classes is not None:
            return {"n_classes": float(family.n_classes)}
        return {}
    if isinstance(family, HingeLoss):
        return {"alpha_cost": family.alpha_cost}
    if isinstance(family, Cindex):
        return {"sigma": family.sigma}
    return {}


def _inverse_link(eta, family):
    import numpy as np
    if isinstance(family, Gaussian | GaussReg):
        return eta.copy()
    if isinstance(family, Binomial | AdaExp | HingeLoss):
        return 1.0 / (1.0 + np.exp(-2.0 * eta))
    if isinstance(family, Poisson | GammaReg | NBinomial | Hurdle):
        return np.exp(eta)
    if isinstance(family, PropOdds):
        # PropOdds response is the per-category probability matrix P(Y=k), which
        # needs the fitted K-1 thresholds (not available here). The model's
        # predict(type="response") computes it from the stored thresholds via the
        # core's propodds_response; this generic link helper cannot.
        raise NotImplementedError(
            "PropOdds response (per-category probabilities) requires the fitted "
            "thresholds; use model.predict(type='response') instead of _inverse_link."
        )
    if isinstance(family, Multinomial):
        # Multinomial response is the softmax probability matrix.
        # Requires the full K-dimensional eta; use model.predict(type="response")
        # instead of this generic helper.
        raise NotImplementedError(
            "Multinomial response (softmax probabilities) requires the full "
            "class structure; use model.predict(type='response') instead."
        )
    if isinstance(family, CoxPH | Weibull | Loglog | Lognormal):
        # AFT survival families use a log link: response = exp(f) (the time/scale).
        # (Loglog is log-logistic AFT, NOT a logistic model -- exp(f), not sigmoid.
        # R's Loglog()@response is exp(f). CoxPH@response is NA in R; exp(f) here
        # gives the relative risk, which is the useful scale.)
        return np.exp(eta)
    return eta.copy()


def log_odds_factor(family):
    """The factor that takes a family's link scale to the log-odds.

    `Binomial` boosts eta with p = 1 / (1 + exp(-2 * eta)) (mboost's -1/+1
    coding), so the log-odds are 2 * eta. Every other family has no log-odds
    scale, and asking for one raises.
    """
    if type(family).__name__ == "Binomial":
        return 2.0
    raise ValueError(
        f"the log-odds scale is defined for Binomial; this model's family is "
        f"{type(family).__name__}, whose link scale is type='link'")


def scale_output(value, factor):
    """Multiply a prediction (array, list or dict of them) by `factor`."""
    import numpy as np

    if isinstance(value, dict):
        return {key: scale_output(item, factor) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(scale_output(item, factor) for item in value)
    return np.asarray(value, dtype=np.float64) * factor


def ordinal_response(data, response_name, family):
    """Recode an ordered response to PropOdds' 1..K codes, at fit entry.

    Returns ``(data, family)``. With ``levels=`` or an ordered categorical
    column the response column is replaced by its 1-based codes (a copy of
    the frame; the caller's data are unchanged) and the family's
    ``n_levels`` is filled in. A numeric response with neither is returned
    unchanged and validated as 1..K by the engine.
    """
    if not isinstance(family, PropOdds) or not isinstance(response_name, str):
        return data, family
    column = data[response_name]
    levels = family.levels
    if levels is None:
        dtype = getattr(column, "dtype", None)
        # pandas ordered Categorical
        if getattr(dtype, "ordered", False):
            levels = tuple(dtype.categories)
        # polars Enum
        elif dtype is not None and type(dtype).__name__ == "Enum":
            levels = tuple(dtype.categories.to_list())
        else:
            values = column.to_numpy() if hasattr(column, "to_numpy") else np.asarray(column)
            if np.asarray(values).dtype.kind in "OUS":
                raise ValueError(
                    f"column '{response_name}' holds labels, and their order is not "
                    f"known; pass PropOdds(levels=[...]) with the categories lowest "
                    f"first, or make the column an ordered categorical")
            return data, family
    values = column.to_list() if hasattr(column, "to_list") else list(np.asarray(column))
    index = {level: k + 1 for k, level in enumerate(levels)}
    unknown = sorted({str(v) for v in values if v not in index})
    if unknown:
        raise ValueError(
            f"column '{response_name}' has value(s) {unknown[:5]} not among the "
            f"PropOdds levels {list(levels)}")
    codes = np.array([index[v] for v in values], dtype=np.float64)
    if isinstance(data, dict):
        data = {**data, response_name: codes}
    elif hasattr(data, "with_columns"):
        import polars as pl

        data = data.with_columns(pl.Series(response_name, codes))
    else:
        data = data.copy()
        data[response_name] = codes
    return data, PropOdds(n_levels=len(levels))


def link_scale_label(family) -> "str | None":
    """What ``predict(type="link")`` returns, in words — or None when the link
    scale is the publishable object.

    The default ``predict()`` returns link-scale values and nothing in the
    returned array says so. On a log-link family that is the log mean, not
    ``E[Y]``; on ``Binomial`` it is half the log-odds; on a location-scale
    family some parameters are means and others are logs. Callers who treat
    "the prediction" as the mean publish the wrong number with no error.

    Returns None for families where the link scale is what a caller should
    publish: the identity-link families, and ``Quantile``, whose link is the
    quantile itself. Everywhere else it returns a short label naming the
    scale, so a warning can quote it. LSS families are matched by class name
    (importing gamlss here would be circular) with one label per parameter.
    """
    name = type(family).__name__
    # Identity link, or the link is itself the estimand: nothing to warn about.
    if name in ("Gaussian", "GaussReg", "Laplace", "Huber", "Expectile", "Quantile"):
        return None
    single = {
        "Poisson": "log-mean",
        "GammaReg": "log-mean",
        "NBinomial": "log-mean",
        "Hurdle": "log-mean",
        "Binomial": "half log-odds (mboost +/-1 coding)",
        "AdaExp": "link scale (not a probability)",
        "HingeLoss": "link scale (not a probability)",
        "PropOdds": "threshold scale (not the per-level probability matrix)",
        "Multinomial": "multinomial-logit scale",
        "CoxPH": "log hazard ratio (larger = shorter survival)",
        "Weibull": "location of log-time",
        "Loglog": "location of log-time",
        "Lognormal": "location of log-time",
        "AUC": "link scale",
        "Cindex": "link scale",
        "Gehan": "link scale",
    }
    if name in single:
        return single[name]
    lss = {
        "GaussianLSS": "mu: mean; sigma: log-scale",
        "GaussianKnownVarLSS": "mu: mean",
        "StudentTLSS": "mu: mean; sigma: log t-scale (not a standard deviation); "
                       "df: log degrees of freedom",
        "BetaLSS": "mu: logit-mean; phi: log precision",
        "PoissonLSS": "mu: log-mean",
        "GammaLSS": "mu: log-mean; sigma: log-shape",
        "NBinomialLSS": "mu: log-mean; sigma: log-size (theta; larger = less overdispersion)",
        "NegativeBinomialLSS": "mu: log-mean; sigma: log-dispersion (larger = more overdispersion)",
        "GeometricLSS": "mu: log-mean",
        "ZIPoLSS": "mu: log-mean; sigma: logit zero-probability",
        "ZINBLSS": "mu: log-mean; sigma: log-dispersion (larger = more overdispersion); "
                  "nu: logit zero-probability",
        "ZeroAdjustedPoissonLSS": "mu: log-mean; sigma: logit zero-probability",
        "ZeroAdjustedGammaLSS": "mu: log-mean; sigma: log-shape; nu: logit zero-probability",
        "BinomialLSS": "mu: logit-probability",
        "InverseGaussianLSS": "mu: log-mean; sigma: log-shape",
        "LogisticLSS": "mu: mean; sigma: log-scale",
        "GumbelLSS": "mu: mean; sigma: log-scale",
        "Pareto2LSS": "mu: log-mean; sigma: log-shape",
        "PIGLSS": "mu: log-mean; sigma: log-dispersion (larger = more overdispersion)",
        "BCCGLSS": "mu: mean; sigma: log-scale; nu: identity",
        "JSULSS": "mu: mean; sigma: log-scale; nu: identity; tau: log-shape",
        "BCTLSS": "mu: mean; sigma: log-scale; nu: identity; tau: log-df",
        "BCPELSS": "mu: mean; sigma: log-scale; nu: identity; tau: log-shape",
        "LogNormalLSS": "mu: location of log-time; sigma: log-scale",
        "LogLogLSS": "mu: location of log-time; sigma: log-scale",
        "WeibullLSS": "mu: location of log-time; sigma: log-scale",
        "DirichletLSS": "log-concentrations",
    }
    return lss.get(name, "link scale")


#: Sentinel default for ``predict(type=...)``. ``predict`` still defaults to
#: the link scale (R ``mboost`` parity), but the method needs to know whether
#: the caller asked for it or just omitted it: only the omitted case warns
#: (#229). ``predict_parameters`` reads the same sentinel the other way
#: (#326): an omitted ``type`` means the response scale there, so it replaces
#: the sentinel and never reaches the warning.
_TYPE_DEFAULT = object()


def maybe_warn_link_scale(model, type_was_default, method="predict"):
    """Warn once per fitted model when ``predict`` runs on the default link scale.

    Fires only when the caller did not pass ``type`` explicitly and the
    family's link scale is not the publishable object
    (:func:`link_scale_label` returns a label). The default itself is unchanged
    — R parity stays — but the first silent link-scale prediction now names the
    scale instead of letting a log-mean pass as a mean.

    ``method`` is the name of the method the user actually called, so the
    message points at the right call: every caller passes it explicitly
    (``"predict"``; ``predict_parameters`` resolves its own default to
    ``type="response"`` and does not reach this warning, #326).
    """
    if not type_was_default:
        return
    if getattr(model, "_mboostrs_scale_warned", False):
        return
    label = link_scale_label(model._family)
    if label is None:
        return
    model._mboostrs_scale_warned = True
    warnings.warn(
        f"{method}() with default type='link' returns the {label} for "
        f"{type(model._family).__name__}; pass type='response' for response scale.",
        UserWarning,
        stacklevel=3,
    )


def prediction_scale_text(family) -> str:
    """One-line description of what ``predict()`` returns by default.

    Used by ``summary()`` so the default scale is visible next to the family:
    ``"link (identity: same as response)"`` when the link is the publishable
    object, otherwise ``"link (<scale>)"`` with the label from
    :func:`link_scale_label`.
    """
    label = link_scale_label(family)
    if label is None:
        return "link (identity: same as response)"
    return f"link ({label})"
