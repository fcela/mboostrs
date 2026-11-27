"""Reusable diagnostic data builders for standard and LSS models.

These functions build tables that can be consumed by plot functions,
notebooks, tests, and bug reports.  They delegate to ``LSSModel`` methods
when the model is an LSS model and synthesise equivalent data for
``GLMBoostModel`` and ``GAMBoostModel`` from existing fitted/predicted
values.

All functions return a list of ``dict`` records when neither pandas nor
polars is installed.
"""

from __future__ import annotations

import inspect

import numpy as np

from mboostrs._dispatch import is_lss as _is_lss
from mboostrs._same_family import SameFamilyGap, same_family_gap  # noqa: F401
from mboostrs.gamlss import _normal_ppf_array, _prob_col, _records_frame

__all__ = [
    "SameFamilyGap",
    "acf_residuals",
    "calibration_frame",
    "centile_fan_data",
    "coef_all",
    "coverage",
    "density_overlay_data",
    "deviance_increment",
    "deviance_profile",
    "interval_metrics",
    "ipc_weights",
    "outlier_frame",
    "pit",
    "predict_all",
    "q_stats",
    "quantile_residuals",
    "residual_frame",
    "same_family_gap",
    "worm_data",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fitted_values(model, newdata=None, mstop=None, type="link"):
    return model.predict(newdata=newdata, type=type, mstop=mstop)


def _observed_response(model, newdata=None):
    if _is_lss(model):
        return model._observed_response(model._data if newdata is None else newdata)
    if newdata is None:
        resp_name = getattr(model, "_response_name", "y")
        data = getattr(model, "_data", {})
        return np.asarray(data[resp_name], dtype=np.float64)
    resp_name = getattr(model, "_response_name", "y")
    return np.asarray(newdata[resp_name], dtype=np.float64)


# ---------------------------------------------------------------------------
# Predict-all
# ---------------------------------------------------------------------------

def predict_all(model, newdata=None, type="link", mstop=None,
                include_response=True, output="auto", extrapolation="warn"):
    """Return a table of fitted values (and response for in-sample).

    For LSS models, one column per parameter plus ``y``.
    For standard models, ``fitted`` plus ``y``. ``extrapolation`` is as in
    the model's ``predict``; a linear model has no range to check and ignores it.
    """
    if _is_lss(model):
        return model.predict_all(
            newdata=newdata, type=type, mstop=mstop,
            include_response=include_response, output=output,
            extrapolation=extrapolation,
        )
    policy = ({"extrapolation": extrapolation}
              if "extrapolation" in inspect.signature(model.predict).parameters else {})
    fitted = model.predict(newdata=newdata, type=type, mstop=mstop, **policy)
    n = len(fitted)
    response = None
    if include_response:
        try:
            response = _observed_response(model, newdata)
        except KeyError:
            # Predictor-only newdata is valid for prediction. In that case the
            # output simply omits the unavailable observed response.
            if newdata is None:
                raise
    rows = []
    for i in range(n):
        row = {"index": i, "fitted": float(fitted[i])}
        if response is not None:
            row["y"] = float(response[i])
        rows.append(row)
    return _records_frame(rows, output)


# ---------------------------------------------------------------------------
# Coef-all
# ---------------------------------------------------------------------------

def coef_all(model, output="auto"):
    """Return one coefficient row per learner, boosting step and component."""
    # "dataframe" was this function's own default and is not a value the
    # shared policy knows. Map it rather than break a caller who wrote it.
    if output == "dataframe":
        output = "auto"
    if _is_lss(model):
        return model.coef_all(output=output)
    result = model.coef(aggregate="none")
    rows = []
    # GLMBoostModel.coef returns a dict {name: [coefs...]}; GAMBoostModel similar
    if isinstance(result, dict):
        for name, coefs in result.items():
            if np.isscalar(coefs):
                rows.append({"learner": name, "step": 0, "coefficient": float(coefs)})
            else:
                for step, val in enumerate(coefs):
                    arr = np.asarray(val)
                    if arr.ndim == 0:
                        rows.append({
                            "learner": name,
                            "step": step,
                            "component": 0,
                            "coefficient": float(arr),
                        })
                    else:
                        for component, coefficient in enumerate(arr.ravel()):
                            rows.append({
                                "learner": name,
                                "step": step,
                                "component": component,
                                "coefficient": float(coefficient),
                            })
    else:
        names = getattr(model, "_learner_names", [])
        for name, coefs in zip(names, result, strict=False):
            for step, val in enumerate(coefs):
                rows.append({"learner": name, "step": step, "coefficient": float(val)})
    return _records_frame(rows, output)


# ---------------------------------------------------------------------------
# Deviance increment
# ---------------------------------------------------------------------------

def deviance_increment(model, newdata=None):
    """Per-observation deviance contributions.

    For LSS families with ``deviance_increment``, delegates to the family.
    For supported standard exponential-family models, dispatches on family
    name and computes the corresponding deviance contribution. Families whose
    loss is not a likelihood deviance raise ``NotImplementedError``.
    """
    if _is_lss(model):
        return model.deviance_increment(newdata=newdata)
    fitted = model.predict(newdata=newdata, type="response")
    y = _observed_response(model, newdata)
    family_name = type(getattr(model, "_family", None)).__name__
    if family_name in ("Gaussian", "GaussReg"):
        return (y - fitted) ** 2
    if family_name == "Poisson":
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(y > 0, y / np.maximum(fitted, 1e-300), 0.0)
        return 2.0 * (y * np.log(np.maximum(ratio, 1e-300)) - (y - fitted))
    if family_name == "Binomial":
        mu = np.clip(fitted, 1e-10, 1.0 - 1e-10)
        yc = np.clip(y, 1e-10, 1.0 - 1e-10)
        return 2.0 * (yc * np.log(yc / mu) + (1.0 - yc) * np.log((1.0 - yc) / (1.0 - mu)))
    if family_name == "GammaReg":
        ratio = np.maximum(y, 1e-300) / np.maximum(fitted, 1e-300)
        return 2.0 * (-np.log(ratio) + (ratio - 1.0))
    raise NotImplementedError(
        f"deviance increments are not implemented for family {family_name!r}"
    )


# ---------------------------------------------------------------------------
# PIT
# ---------------------------------------------------------------------------

def pit(model, newdata=None, mstop=None, randomized=True, random_state=None):
    """Probability integral transform for LSS models with CDF support.

    Standard (non-LSS) models do not have a full distribution CDF and raise
    ``NotImplementedError``.
    """
    if not _is_lss(model):
        raise NotImplementedError(
            "PIT requires an LSS model with family CDF support."
            " Check mb.family_capabilities() for supported diagnostics."
        )
    return model.pit(
        newdata=newdata, mstop=mstop, randomized=randomized, random_state=random_state
    )


# ---------------------------------------------------------------------------
# Quantile residuals
# ---------------------------------------------------------------------------

def quantile_residuals(model, newdata=None, mstop=None, randomized=True, random_state=None):
    """Randomized quantile residuals.

    For LSS families with CDF support, this is ``Phi^{-1}(F(y))``. For
    standard Gaussian models, it reduces to normalized response residuals.
    For other standard families, raises ``NotImplementedError``.
    """
    if _is_lss(model):
        return model.quantile_residuals(
            newdata=newdata, mstop=mstop, randomized=randomized,
            random_state=random_state,
        )
    family_name = type(getattr(model, "_family", None)).__name__
    if family_name in ("Gaussian", "GaussReg"):
        fitted = model.predict(newdata=newdata, type="response", mstop=mstop)
        y = _observed_response(model, newdata)
        resid = y - fitted
        sigma = np.sqrt(np.mean(resid**2)) if len(resid) > 1 else 1.0
        if sigma > 0:
            return resid / sigma
        return resid
    raise NotImplementedError(
        f"quantile_residuals is not implemented for standard {family_name} models."
        " Check mb.family_capabilities() for supported diagnostics."
    )


# ---------------------------------------------------------------------------
# Residual frame
# ---------------------------------------------------------------------------

def residual_frame(model, kind="response", newdata=None, mstop=None,
                   randomized=True, random_state=None, output="auto"):
    """Diagnostic residual table for any model.

    ``kind`` is one of the kinds :mod:`mboostrs._residuals` defines --
    ``"response"``, ``"pearson"``, ``"working"``, ``"deviance"``,
    ``"martingale"`` -- plus ``"quantile"`` for randomized quantile residuals,
    which are uniform under a correct fit and so are the one kind comparable
    across families.

    The arithmetic is the same as :meth:`residuals` on the model. The two used
    to disagree: for a Gaussian fit this divided by the residual RMS while the
    method did not, and both called the result Pearson.
    """
    from mboostrs._residuals import RESIDUAL_KINDS
    from mboostrs._residuals import residuals as _residuals

    if _is_lss(model):
        return model.residual_frame(
            kind=kind, newdata=newdata, mstop=mstop,
            randomized=randomized, random_state=random_state, output=output,
        )

    valid = (*RESIDUAL_KINDS, "quantile")
    if kind not in valid:
        raise ValueError(f"kind must be one of {valid}, got {kind!r}")

    if kind == "quantile":
        residual = quantile_residuals(
            model, newdata=newdata, mstop=mstop,
            randomized=randomized, random_state=random_state,
        )
    else:
        residual = _residuals(model, kind=kind, newdata=newdata, mstop=mstop)
    residual = np.asarray(residual, dtype=np.float64).ravel()

    y = np.asarray(_observed_response(model, newdata), dtype=np.float64).ravel()
    fitted = np.asarray(
        model.predict(newdata=newdata, type="response", mstop=mstop),
        dtype=np.float64,
    )
    if fitted.ndim > 1:
        # An ordinal or multi-class fit predicts a matrix; the table reports
        # the expected category rather than one arbitrary column.
        fitted = fitted @ np.arange(1, fitted.shape[1] + 1, dtype=np.float64)
    fitted = fitted.ravel()

    rows = []
    for i in range(residual.size):
        rows.append({
            "index": i,
            "y": float(y[i]) if i < y.size else float("nan"),
            "fitted": float(fitted[i]) if i < fitted.size else float("nan"),
            "residual_kind": kind,
            "residual": float(residual[i]),
        })
    return _records_frame(rows, output)


# ---------------------------------------------------------------------------
# Q-stats
# ---------------------------------------------------------------------------

def _binned_residual_moments(residual, x, xvar_name, n_intervals):
    """Moments of ``residual`` within ``n_intervals`` equal-count bins of ``x``.

    One implementation, used by both the module function and the model
    methods. The two used to differ in ways that changed the numbers on the
    same fit: equal-width bins against equal-count ones, and a population
    variance against a sample one, so a caller reading `q_stats(model)` and
    `model.q_stats()` side by side saw different variances for what the
    columns claimed were the same bins.

    The conventions settled here:

    **Equal-count bins.** The sorted order of ``x`` is split into groups of
    near-equal size, so every bin's moments are estimated from a comparable
    number of observations. Equal-width bins on a skewed ``x`` leave some bins
    with a handful of points and others with most of the data, and the
    resulting moments are not comparable down the column -- which is what a
    reader scanning for a departure from N(0, 1) is doing.

    **Sample (unbiased) variance**, and the same ``sqrt`` of it as the scale
    for skewness and kurtosis, so the three are consistent with each other.
    The diagnostic compares against a standard normal, where the unbiased
    estimate is the one whose sampling distribution is known.

    **Excess kurtosis**: 0 for a normal, not 3.
    """
    residual = np.asarray(residual, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n_intervals = int(n_intervals)
    if n_intervals <= 0:
        raise ValueError(f"n_intervals must be positive, got {n_intervals}")

    order = np.argsort(x, kind="stable")
    groups = np.array_split(order, min(n_intervals, max(residual.size, 1)))

    rows = []
    for interval, index in enumerate(groups, start=1):
        if index.size == 0:
            continue
        block = residual[index]
        mean = float(np.mean(block))
        variance = float(np.var(block, ddof=1)) if index.size > 1 else 0.0
        if variance > 0.0:
            centered = block - mean
            deviation = np.sqrt(variance)
            skewness = float(np.mean(centered ** 3) / deviation ** 3)
            kurtosis = float(np.mean(centered ** 4) / deviation ** 4 - 3.0)
        else:
            skewness = 0.0
            kurtosis = 0.0
        rows.append({
            "interval": interval,
            "xvar": xvar_name,
            "x_min": float(np.min(x[index])),
            "x_max": float(np.max(x[index])),
            "n": int(index.size),
            "mean": mean,
            "variance": variance,
            "skewness": skewness,
            "kurtosis": kurtosis,
        })
    return rows


def q_stats(model, xvar=None, n_intervals=10, newdata=None, mstop=None,
            randomized=True, random_state=None, output="auto"):
    """Binned quantile-residual moment summaries.

    One row per bin: the interval number, the binning variable's name and
    range, the count, and the first four moments of the (randomized) quantile
    residuals in that bin. Under a correctly specified model the residuals are
    standard normal, so a bin's mean should sit near 0, its variance near 1,
    and its skewness and excess kurtosis near 0; a column that drifts with
    ``xvar`` points at the part of the predictor space the distribution is
    getting wrong.

    ``xvar=None`` bins by observation order, which detects a departure that
    tracks the data's own ordering and nothing else. Pass the variable you
    suspect.

    Descriptive moments only. GAMLSS's ``Q.stats`` also reports Z-statistics
    and p-values against them; those need the exact reference definition and
    its assumptions reproduced before they would mean anything, which is a
    separate piece of work.
    """
    if _is_lss(model):
        return model.q_stats(
            xvar=xvar, n_intervals=n_intervals, newdata=newdata, mstop=mstop,
            randomized=randomized, random_state=random_state, output=output,
        )

    residual = quantile_residuals(
        model, newdata=newdata, mstop=mstop,
        randomized=randomized, random_state=random_state,
    )
    if xvar is None:
        x = np.arange(len(residual), dtype=np.float64)
        xvar_name = "index"
    else:
        data = getattr(model, "_data", {}) if newdata is None else newdata
        x = np.asarray(data[xvar], dtype=np.float64)
        xvar_name = str(xvar)

    rows = _binned_residual_moments(residual, x, xvar_name, n_intervals)
    return _records_frame(rows, output)


# ---------------------------------------------------------------------------
# Worm data
# ---------------------------------------------------------------------------

def worm_data(model, xvar=None, n_intervals=4, newdata=None, mstop=None,
              randomized=True, random_state=None, output="auto"):
    """Return worm-plot deviation data.

    Bins observations by ``xvar`` into ``n_intervals`` groups, then
    computes expected vs theoretical normal quantiles within each bin.
    """
    residual = quantile_residuals(
        model, newdata=newdata, mstop=mstop,
        randomized=randomized, random_state=random_state,
    )
    n = len(residual)
    if xvar is None:
        x = np.arange(n, dtype=float)
    else:
        data = getattr(model, "_data", {})
        if newdata is not None:
            data = newdata
        x = np.asarray(data[xvar], dtype=np.float64)
    if n_intervals > n:
        n_intervals = n
    edges = np.linspace(x.min(), x.max() + 1e-12, n_intervals + 1)
    rows = []
    for k in range(n_intervals):
        mask = (x >= edges[k]) & (x < edges[k + 1])
        r_k = residual[mask]
        if len(r_k) == 0:
            continue
        nk = len(r_k)
        sorted_r = np.sort(r_k)
        theoretical = _normal_ppf_array((np.arange(1, nk + 1) - 0.5) / nk)
        deviation = sorted_r - theoretical
        for j in range(nk):
            rows.append({
                "bin": k + 1,
                "rank": j + 1,
                "residual": float(sorted_r[j]),
                "expected": float(theoretical[j]),
                "deviation": float(deviation[j]),
            })
    return _records_frame(rows, output)


# ---------------------------------------------------------------------------
# Distribution profiles
# ---------------------------------------------------------------------------

def _profile_weights(model, n, newdata, weights):
    if weights is None:
        if newdata is None and hasattr(model, "model_weights"):
            weights = model.model_weights()
        else:
            weights = np.ones(n, dtype=np.float64)
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or values.size != n:
        raise ValueError(f"weights must be a length-{n} vector")
    if np.any(~np.isfinite(values)) or np.any(values < 0) or not np.any(values > 0):
        raise ValueError("weights must be finite, non-negative, and contain a positive value")
    return values


def density_overlay_data(model, newdata=None, mstop=None, grid=None,
                         n_grid=200, weights=None, output="auto"):
    """Observed and average fitted response distributions on one grid.

    The fitted curve is the weighted average of the row-specific conditional
    densities or masses, not the density evaluated at average parameters.
    This distinction matters whenever scale, skewness, or tail shape varies by
    row. The observed continuous curve is a Gaussian kernel estimate; for a
    discrete response it is the weighted empirical frequency.

    The overlay is descriptive in-sample unless ``newdata`` is held out. It
    can reveal a marginal mismatch while hiding offsetting conditional errors;
    use :func:`worm_data` or :func:`q_stats` against relevant covariates for
    those errors.
    """
    if not _is_lss(model):
        raise TypeError("density_overlay_data requires an LSSModel")
    family = model._family
    density = getattr(family, "pdf", None)
    mass = getattr(family, "pmf", None)
    if density is None and mass is None:
        raise NotImplementedError(
            f"{type(family).__name__} does not expose a density or mass function"
        )
    y = np.asarray(_observed_response(model, newdata), dtype=np.float64)
    if y.ndim != 1 or y.size == 0 or np.any(~np.isfinite(y)):
        raise ValueError("the observed response must be a non-empty finite vector")
    params = model.params(newdata=newdata, type="response", mstop=mstop)
    row_weights = _profile_weights(model, y.size, newdata, weights)
    total_weight = float(row_weights.sum())

    n_grid = int(n_grid)
    if n_grid < 2:
        raise ValueError("n_grid must be at least 2")
    is_discrete = bool(getattr(family, "_discrete", False))
    mixed_atom = getattr(family, "_mixed_atom", None)
    if is_discrete and mass is None:
        mass = density
    if grid is None and is_discrete:
        lo = int(np.floor(np.min(y)))
        hi = int(np.ceil(np.max(y)))
        if hasattr(family, "quantile"):
            try:
                upper = np.asarray(
                    family.quantile(np.full(y.size, 0.995), params), dtype=np.float64
                )
                if np.any(np.isfinite(upper)):
                    hi = max(hi, int(np.nanmax(upper[np.isfinite(upper)])))
            except (ArithmeticError, RuntimeError, ValueError):
                pass
        if hi - lo + 1 > n_grid:
            raise ValueError(
                f"the discrete support from {lo} to {hi} has {hi - lo + 1} points; "
                f"raise n_grid above that count or pass an explicit grid"
            )
        x_grid = np.arange(lo, hi + 1, dtype=np.float64)
    elif grid is None:
        continuous_y = y if mixed_atom is None else y[y != mixed_atom]
        if continuous_y.size == 0:
            raise ValueError("a mixed distribution overlay needs at least one non-atom response")
        limits = [continuous_y]
        if hasattr(family, "quantile"):
            for probability in (0.005, 0.995):
                try:
                    limits.append(np.asarray(
                        family.quantile(np.full(y.size, probability), params),
                        dtype=np.float64,
                    ))
                except (ArithmeticError, RuntimeError, ValueError):
                    pass
        finite = np.concatenate([item.ravel() for item in limits])
        finite = finite[np.isfinite(finite)]
        lo, hi = float(np.min(finite)), float(np.max(finite))
        if hi <= lo:
            padding = max(abs(lo), 1.0) * 0.1
        else:
            padding = 0.03 * (hi - lo)
        x_grid = np.linspace(lo - padding, hi + padding, n_grid)
        if mixed_atom is not None:
            x_grid = x_grid[x_grid > mixed_atom]
    else:
        x_grid = np.asarray(grid, dtype=np.float64)
        if x_grid.ndim != 1 or x_grid.size < 2 or np.any(~np.isfinite(x_grid)):
            raise ValueError("grid must be a finite one-dimensional vector with at least 2 values")
        if np.any(np.diff(x_grid) <= 0):
            raise ValueError("grid must be strictly increasing")

    rows = []
    if is_discrete:
        for value in x_grid:
            fitted = np.asarray(mass(np.full(y.size, value), params), dtype=np.float64)
            observed = float(row_weights[y == value].sum() / total_weight)
            rows.extend((
                {"x": float(value), "value": observed, "series": "observed", "kind": "mass"},
                {"x": float(value), "value": float(np.dot(row_weights, fitted) / total_weight),
                 "series": "fitted", "kind": "mass"},
            ))
    else:
        continuous = row_weights > 0
        if mixed_atom is not None:
            continuous &= y != mixed_atom
        observed_y = y[continuous]
        observed_weights = row_weights[continuous]
        observed_weight = float(observed_weights.sum())
        if observed_weight <= 0:
            raise ValueError("a continuous density overlay needs positive weight off the atom")
        mean = float(np.dot(observed_weights, observed_y) / observed_weight)
        variance = float(
            np.dot(observed_weights, (observed_y - mean) ** 2) / observed_weight
        )
        effective_n = observed_weight**2 / float(np.dot(observed_weights, observed_weights))
        iqr = float(np.quantile(observed_y, 0.75) - np.quantile(observed_y, 0.25))
        scale = min(np.sqrt(max(variance, 0.0)), iqr / 1.34) if iqr > 0 else np.sqrt(max(variance, 0.0))
        if not np.isfinite(scale) or scale <= 0:
            scale = max(float(np.ptp(observed_y)), abs(mean), 1.0) / 4.0
        bandwidth = max(0.9 * scale * effective_n ** (-0.2), np.finfo(float).eps)
        normalizer = np.sqrt(2.0 * np.pi) * bandwidth * total_weight
        for value in x_grid:
            if mixed_atom is not None and value == mixed_atom:
                continue
            fitted = np.asarray(density(np.full(y.size, value), params), dtype=np.float64)
            kernel = np.exp(-0.5 * ((value - observed_y) / bandwidth) ** 2)
            observed = float(np.dot(observed_weights, kernel) / normalizer)
            rows.extend((
                {"x": float(value), "value": observed, "series": "observed", "kind": "density"},
                {"x": float(value), "value": float(np.dot(row_weights, fitted) / total_weight),
                 "series": "fitted", "kind": "density"},
            ))
        if mixed_atom is not None:
            fitted_atom = np.asarray(
                density(np.full(y.size, mixed_atom), params), dtype=np.float64
            )
            rows.extend((
                {"x": float(mixed_atom), "value": float(row_weights[y == mixed_atom].sum() / total_weight),
                 "series": "observed", "kind": "atom"},
                {"x": float(mixed_atom), "value": float(np.dot(row_weights, fitted_atom) / total_weight),
                 "series": "fitted", "kind": "atom"},
            ))
    return _records_frame(rows, output)


def centile_fan_data(model, xvar, probs=(0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975),
                     split_by=None, newdata=None, mstop=None, output="auto"):
    """Observed responses and fitted centiles ordered by one numeric covariate.

    ``probs`` must contain 0.5 and symmetric lower/upper pairs so a fan has an
    unambiguous median and nested central bands. ``split_by`` retains a group
    label for faceting; it does not refit separate models.
    """
    if not _is_lss(model):
        raise TypeError("centile_fan_data requires an LSSModel")
    probabilities = np.asarray(probs, dtype=np.float64)
    if (probabilities.ndim != 1 or probabilities.size < 3
            or np.any(~np.isfinite(probabilities))
            or np.any((probabilities <= 0) | (probabilities >= 1))):
        raise ValueError(
            "probs must contain at least three finite probabilities in (0, 1), including 0.5"
        )
    probabilities = np.unique(probabilities)
    if not np.any(np.isclose(probabilities, 0.5, atol=1e-12, rtol=0)):
        raise ValueError("probs must contain 0.5 for the fan median")
    lower = probabilities[probabilities < 0.5]
    upper = probabilities[probabilities > 0.5]
    if lower.size != upper.size or not np.allclose(lower, 1.0 - upper[::-1], atol=1e-12, rtol=0):
        raise ValueError("probs below and above 0.5 must form symmetric pairs")

    data = model._data if newdata is None else newdata
    x = np.asarray(data[xvar], dtype=np.float64)
    y = np.asarray(_observed_response(model, newdata), dtype=np.float64)
    if x.ndim != 1 or x.size != y.size or np.any(~np.isfinite(x)):
        raise ValueError(f"{xvar!r} must be a finite numeric column aligned with the response")
    groups = np.repeat("all", y.size) if split_by is None else np.asarray(data[split_by]).astype(str)
    if groups.ndim != 1 or groups.size != y.size:
        raise ValueError(f"{split_by!r} must be a one-dimensional column aligned with the response")
    params = model.params(newdata=newdata, type="response", mstop=mstop)
    quantiles = {
        _prob_col(float(probability)): np.asarray(
            model._family.quantile(np.full(y.size, probability), params),
            dtype=np.float64,
        )
        for probability in probabilities
    }
    order = np.lexsort((x, groups))
    rows = []
    for index in order:
        row = {
            "index": int(index), "x": float(x[index]), "observed": float(y[index]),
            "split": str(groups[index]), "xvar": str(xvar),
        }
        for column, values in quantiles.items():
            row[column] = float(values[index])
        rows.append(row)
    return _records_frame(rows, output)


def deviance_profile(model, xvar=None, bins=10, newdata=None, mstop=None,
                     weights=None, output="auto"):
    """Bin joint deviance contributions along a numeric covariate.

    LSS deviance is joint across parameters and cannot be assigned to a single
    parameter update. The profile therefore localizes observations where the
    fitted distribution performs poorly; it is not a per-term attribution.
    Equal-count bins keep the support per row comparable.
    """
    if not _is_lss(model):
        raise TypeError("deviance_profile requires an LSSModel")
    y = np.asarray(_observed_response(model, newdata), dtype=np.float64)
    params = model.params(newdata=newdata, type="response", mstop=mstop)
    contribution = np.asarray(model._family.deviance_increment(y, params), dtype=np.float64)
    if contribution.shape != y.shape or np.any(~np.isfinite(contribution)):
        raise ValueError("deviance contributions must be finite and aligned with the response")
    data = model._data if newdata is None else newdata
    if xvar is None:
        x = np.arange(y.size, dtype=np.float64)
        name = "index"
    else:
        x = np.asarray(data[xvar], dtype=np.float64)
        name = str(xvar)
    if x.ndim != 1 or x.size != y.size or np.any(~np.isfinite(x)):
        raise ValueError(f"{name!r} must be a finite numeric column aligned with the response")
    row_weights = _profile_weights(model, y.size, newdata, weights)
    bins = int(bins)
    if bins <= 0:
        raise ValueError("bins must be positive")
    order = np.argsort(x, kind="stable")
    # The first ``n % k`` intervals take one observation more than the rest.
    # R's ``deviance_profile`` reproduces this rule explicitly rather than
    # using ``cut``, so both frontends report the same bin membership when
    # ``n`` is not a multiple of ``bins``.
    groups = np.array_split(order, min(bins, y.size))
    rows = []
    for interval, index in enumerate(groups, start=1):
        w = row_weights[index]
        total = float(w.sum())
        if total <= 0:
            continue
        d = contribution[index]
        rows.append({
            "interval": interval,
            "xvar": name,
            "x_min": float(np.min(x[index])),
            "x_max": float(np.max(x[index])),
            "x_mid": float(np.dot(w, x[index]) / total),
            "n": int(index.size),
            "weight": total,
            "mean_deviance": float(np.dot(w, d) / total),
            "total_deviance": float(np.dot(w, d)),
            "median_deviance": float(np.median(d)),
            "q90_deviance": float(np.quantile(d, 0.9)),
        })
    return _records_frame(rows, output)


# ---------------------------------------------------------------------------
# Calibration frame
# ---------------------------------------------------------------------------

def calibration_frame(model, xvar=None, bins=10, newdata=None, mstop=None,
                      output="auto"):
    """Compare observed and fitted means within bins.

    Bins the observations by ``xvar`` (or by observation order when it is
    ``None``) and reports the mean response and the mean fitted value in each.
    A well-calibrated fit puts the two columns on top of each other across
    every bin; a systematic gap that moves with ``xvar`` is the fit getting
    that part of the predictor space wrong.

    On a distributional model, this computes the family's conditional response
    mean from all fitted parameters. It refuses families without a declared
    finite mean. For example, zero-adjusted Gamma uses ``(1-nu)*mu`` and Gumbel
    uses ``mu + EulerGamma*sigma``; the first parameter alone is not either
    mean. The result column is ``fitted_mean`` on both standard and LSS models.

    A distribution-free calibration check that does not depend on which
    parameter means what is :func:`pit` or :func:`worm_data`: both ask whether
    the fitted distribution is right as a whole, rather than whether one of
    its parameters tracks the response.
    """
    y = _observed_response(model, newdata)
    n = len(y)

    if _is_lss(model):
        family = model._family
        if not hasattr(family, "mean"):
            raise NotImplementedError(
                f"calibration_frame needs a conditional response mean, but "
                f"{type(family).__name__} does not declare one. Use pit() or "
                "worm_data() to assess the fitted distribution."
            )
        params = model.params(newdata=newdata, type="response", mstop=mstop)
        try:
            fitted = np.asarray(family.mean(params), dtype=np.float64)
        except ValueError as exc:
            raise NotImplementedError(
                f"calibration_frame cannot compute a finite conditional mean: {exc}. "
                "Use pit() or worm_data()."
            ) from exc
        fitted_column = "fitted_mean"
    else:
        fitted = np.asarray(
            model.predict(newdata=newdata, type="response", mstop=mstop),
            dtype=np.float64,
        )
        if fitted.ndim > 1:
            raise NotImplementedError(
                f"calibration_frame needs one fitted value per observation, "
                f"but {type(model._family).__name__} predicts a "
                f"{fitted.shape[1]}-column probability matrix. Compare the "
                f"predicted class probabilities against observed frequencies "
                f"directly, or use pit() for a distribution-level check."
            )
        fitted_column = "fitted_mean"

    if xvar is None:
        x = np.arange(n, dtype=np.float64)
    else:
        data = getattr(model, "_data", {}) if newdata is None else newdata
        x = np.asarray(data[xvar], dtype=np.float64)

    bins = int(bins)
    if bins <= 0:
        raise ValueError(f"bins must be positive, got {bins}")
    if bins > n:
        bins = n

    # Equal-count bins, matching q_stats: on a skewed x, equal-width bins put
    # most of the data in one interval and leave the rest with a handful of
    # points, so the means down the column are not comparable with each other.
    order = np.argsort(x, kind="stable")
    rows = []
    for interval, index in enumerate(np.array_split(order, bins), start=1):
        if index.size == 0:
            continue
        rows.append({
            "interval": interval,
            "n": int(index.size),
            "x_min": float(np.min(x[index])),
            "x_max": float(np.max(x[index])),
            "observed_mean": float(np.mean(y[index])),
            fitted_column: float(np.mean(fitted[index])),
        })
    return _records_frame(rows, output)


# ---------------------------------------------------------------------------
# Outlier frame
# ---------------------------------------------------------------------------

def outlier_frame(model, newdata=None, mstop=None, output="auto"):
    """Observations ranked by absolute residual, largest first.

    Records are ranked before the requested frame is built, so row order is
    consistent across pandas and Polars output.
    """
    records = residual_frame(
        model, kind="response", newdata=newdata, mstop=mstop, output="records"
    )
    ranked = sorted(records, key=lambda row: abs(row["residual"]), reverse=True)
    return _records_frame(ranked, output)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def coverage(observed, lower, upper):
    """Compute empirical coverage of prediction intervals.

    Returns ``(coverage_rate, avg_width)`` using inclusive bounds. Inputs
    must be non-empty one-dimensional arrays of identical shape. Observed
    values must be finite; unbounded intervals may use negative infinity for
    lower bounds and positive infinity for upper bounds. NaN or reversed
    bounds are refused. Coverage on training observations does not estimate
    future-outcome coverage.
    """
    observed = np.asarray(observed, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if (observed.ndim != 1 or observed.size == 0
            or lower.shape != observed.shape or upper.shape != observed.shape):
        raise ValueError("observed, lower and upper must be non-empty one-dimensional arrays of the same shape")
    if not np.all(np.isfinite(observed)):
        raise ValueError("observed values must be finite")
    if (np.any(np.isnan(lower)) or np.any(np.isnan(upper))
            or np.any(np.isposinf(lower)) or np.any(np.isneginf(upper))):
        raise ValueError("interval bounds must be finite or use -inf for lower and +inf for upper")
    if np.any(lower > upper):
        raise ValueError("each lower bound must be less than or equal to its upper bound")
    inside = (observed >= lower) & (observed <= upper)
    rate = float(np.mean(inside))
    avg_width = float(np.mean(upper - lower))
    return rate, avg_width


def interval_metrics(observed, lower, upper, *, level=.95, weights=None,
                     retain_scores=False):
    """Weighted coverage, width, and proper central-interval score.

    For nominal coverage ``level`` and ``alpha = 1-level``, the observation's
    interval score is

    ``(upper-lower) + 2/alpha*(lower-y)*I(y<lower) + 2/alpha*(y-upper)*I(y>upper)``.

    Width rewards sharp intervals while the miss penalties preserve calibration;
    comparing width alone favors intervals that cover nothing. The empirical
    coverage and lower/upper miss rates describe these observations and are not
    a future-coverage guarantee. ``weights`` define the evaluation population.
    """
    from mboostrs._validate import check_unit_interval
    from mboostrs._weights import normalize_weights

    level = check_unit_interval("level", level)
    observed = np.asarray(observed, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    # Reuse coverage's complete shape, support, NaN, infinity, and ordering
    # contract before applying weighted aggregation.
    coverage(observed, lower, upper)
    w = normalize_weights(weights, len(observed), policy="raw").fit
    alpha = 1 - level
    below = observed < lower
    above = observed > upper
    width = upper - lower
    score = (
        width
        + 2 / alpha * (lower - observed) * below
        + 2 / alpha * (observed - upper) * above
    )
    total = float(w.sum())
    result = {
        "nominal_coverage": level,
        "empirical_coverage": float(np.sum(w * ~(below | above)) / total),
        "lower_miss_rate": float(np.sum(w * below) / total),
        "upper_miss_rate": float(np.sum(w * above) / total),
        "mean_width": float(np.sum(w * width) / total),
        "mean_interval_score": float(np.sum(w * score) / total),
        "weight_total": total,
        "n": len(observed),
        "score_definition": "central_interval_score",
    }
    if retain_scores:
        values = score.copy()
        values.setflags(write=False)
        result["interval_scores"] = values
    return result


# ---------------------------------------------------------------------------
# Inverse probability of censoring weights
# ---------------------------------------------------------------------------

def _reverse_km_survival(time: np.ndarray, flip_event: np.ndarray) -> dict[float, float]:
    """Kaplan-Meier survival value at each distinct observed time.

    Standard right-continuous KM: at each distinct time, the risk set is
    every subject with ``time >= t`` (ties between an event and a censoring
    at the same ``t`` keep the censored subject at risk for that step, the
    convention used by R's ``survival::survfit``).
    """
    order = np.argsort(time, kind="stable")
    t_sorted = time[order]
    e_sorted = flip_event[order]
    n = len(t_sorted)
    surv_at: dict[float, float] = {}
    s = 1.0
    i = 0
    while i < n:
        t = t_sorted[i]
        j = i
        deaths = 0
        while j < n and t_sorted[j] == t:
            if e_sorted[j]:
                deaths += 1
            j += 1
        at_risk = n - i
        if deaths > 0:
            s *= 1.0 - deaths / at_risk
        surv_at[float(t)] = s
        i = j
    return surv_at


def ipc_weights(time, status, maxweight: float = 5.0) -> np.ndarray:
    """Inverse probability of censoring weights (mboost ``IPCweights``).

    Fits the Kaplan-Meier estimate of the *censoring* distribution (event and
    censoring indicators swapped), evaluates it at each subject's own
    observed time, and returns ``status / G(time)`` clipped to
    ``maxweight``. Censored observations (``status == 0``) always receive
    weight 0. Weighting a standard-family fit by these values approximates
    the outcome for the censoring-free population, the standard IPCW
    construction used with :class:`Weibull`, :class:`Loglog`, and other
    non-survival mboost families on right-censored data.

    Parameters
    ----------
    time : array-like
        Non-empty vector of finite, non-negative observed event times.
    status : array-like
        1 if the event was observed, 0 if the observation was censored.
    maxweight : float
        Positive, finite upper clip applied to the raw inverse-probability weights.
    """
    from mboostrs._validate import check_positive

    maxweight = check_positive("maxweight", maxweight)
    time = np.asarray(time, dtype=np.float64)
    status = np.asarray(status, dtype=np.float64)
    if time.ndim != 1 or status.ndim != 1 or time.size == 0 or status.size == 0:
        raise ValueError("time and status must be non-empty one-dimensional arrays")
    if time.shape != status.shape:
        raise ValueError("time and status must have the same shape")
    if np.any(time < 0) or not np.all(np.isfinite(time)):
        raise ValueError("time must be finite and non-negative")
    if not np.all(np.isin(status, (0.0, 1.0))):
        raise ValueError("status must be 0 (censored) or 1 (event)")

    surv_at = _reverse_km_survival(time, 1.0 - status)
    ghat = np.array([surv_at[float(t)] for t in time], dtype=np.float64)
    ghat[status == 0.0] = 1.0
    weights = status / ghat
    weights[weights > maxweight] = maxweight
    return weights


def acf_residuals(model, max_lag=None, powers=(1, 2, 3, 4), kind=None,
                  newdata=None, mstop=None, output="auto"):
    """Compute residual autocorrelation and autocorrelation of selected powers.

    The counterpart of ``gamlss::acfResid``. Boosting assumes the observations
    are independent given the predictors; when they are ordered in time, any
    structure the model failed to absorb shows up as correlation between
    neighbouring residuals. Checking the *powers* as well as the residuals
    themselves separates which moment is still dependent: correlation in ``r``
    is unmodelled level, in ``r**2`` unmodelled variance, and in ``r**3`` and
    ``r**4`` unmodelled skewness and kurtosis.

    The result contains the values used in the residual diagnostic panels.

    Parameters
    ----------
    model : fitted model
    max_lag : int, optional
        Highest lag. Defaults to ``10 * log10(n)``, as R's ``acf`` does.
    powers : sequence of int
        Which powers of the centered residual to examine.
    kind : str, optional
        Residual type, passed to :func:`residual_frame`. Defaults to
        ``"quantile"`` where the family supports it and ``"response"``
        otherwise, since only quantile residuals are comparable across
        families.
    output : str
        Frame format, as elsewhere.

    Returns
    -------
    frame
        One row per (power, lag): ``power``, ``lag``, ``acf``, ``pacf``, and
        ``bound`` — the two-sided 95% white-noise band ``1.96 / sqrt(n)``.
        A value outside the band at a given lag is the signal to look at.

    Notes
    -----
    Order matters and is taken as given: the rows are used in the order they
    appear in the data. For an unordered sample the result is meaningless
    rather than merely uninformative, so only apply it when the rows have a
    sequence.
    """
    if kind is None:
        family = getattr(model, "_family", None)
        kind = "quantile" if hasattr(family, "cdf") else "response"
    records = residual_frame(model, kind=kind, newdata=newdata, mstop=mstop,
                             output="records")
    residual = np.asarray([row["residual"] for row in records], dtype=np.float64)
    n = residual.size
    if n < 3:
        raise ValueError("acf_residuals needs at least three observations")
    if max_lag is None:
        max_lag = int(np.floor(10.0 * np.log10(n)))
    max_lag = int(min(max_lag, n - 2))
    if max_lag < 1:
        raise ValueError("acf_residuals needs max_lag of at least 1")

    centered = residual - residual.mean()
    bound = 1.96 / np.sqrt(n)

    rows = []
    for power in powers:
        series = centered**power
        series = series - series.mean()
        denominator = float(np.dot(series, series))
        if denominator <= 0:
            autocorrelation = np.zeros(max_lag + 1)
        else:
            autocorrelation = np.array([
                float(np.dot(series[: n - lag], series[lag:])) / denominator
                for lag in range(max_lag + 1)
            ])
        partial = _durbin_levinson(autocorrelation, max_lag)
        for lag in range(1, max_lag + 1):
            rows.append({
                "power": int(power),
                "lag": lag,
                "acf": float(autocorrelation[lag]),
                "pacf": float(partial[lag]),
                "bound": float(bound),
            })
    return _records_frame(rows, output)


def _durbin_levinson(autocorrelation: np.ndarray, max_lag: int) -> np.ndarray:
    """Partial autocorrelations from autocorrelations, by Durbin-Levinson.

    The same recursion R's ``pacf`` uses. Returns an array indexed by lag, with
    index 0 unused.
    """
    partial = np.zeros(max_lag + 1)
    phi = np.zeros((max_lag + 1, max_lag + 1))
    if max_lag >= 1:
        phi[1, 1] = autocorrelation[1]
        partial[1] = phi[1, 1]
    for k in range(2, max_lag + 1):
        numerator = autocorrelation[k] - sum(
            phi[k - 1, j] * autocorrelation[k - j] for j in range(1, k)
        )
        denominator = 1.0 - sum(
            phi[k - 1, j] * autocorrelation[j] for j in range(1, k)
        )
        phi[k, k] = numerator / denominator if denominator != 0 else 0.0
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
        partial[k] = phi[k, k]
    return partial

