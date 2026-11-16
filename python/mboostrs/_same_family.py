"""In-sample gap between a linear boost and the same-family GLM.

Cross-validation can stop while the linear predictor is still a shrunk
version of the maximum-likelihood fit of the same family. :func:`same_family_gap`
reports that. It does not move ``mstop``, it does not change ``nu``, and it
does not keep boosting until the likelihood stops rising.

The comparison is fit in the Rust core (``mboostrs_core::same_family``),
which maximizes the family's own risk, the quantity behind
:meth:`~mboostrs.glmboost.GLMBoostModel.logLik`, for Gaussian, Binomial,
Poisson, GammaReg, NBinomial, PropOdds, and a linear ``BetaLSS``. A smooth
``gamboost`` basis is not a linear GLM, so that fit is reported as having no
comparison rather than matched against a different model. A caller who
already has the comparison log-likelihood can pass it and skip the fit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from mboostrs._results import MappingResultMixin

_TOL = 1.0
_SUPPORTED = {
    "Gaussian": "Gaussian linear model",
    "GaussReg": "Gaussian linear model",
    "Binomial": "binomial logit GLM",
    "Poisson": "Poisson log-linear GLM",
    "GammaReg": "gamma log-linear GLM",
    "NBinomial": "negative-binomial log-linear GLM",
    "PropOdds": "proportional-odds model",
    "BetaLSS": "beta regression",
}


@dataclass
class SameFamilyGap(MappingResultMixin):
    """Log-likelihood gap between a boost and the same-family GLM.

    ``gap`` is ``comparison_loglik - boost_loglik``, in nats, on the same
    rows. Positive means the boost is short of the comparison. ``material``
    is true only when that gap exceeds ``tol`` and the comparison is a
    completed maximum of the same likelihood ``logLik`` reports. The selected
    step is never changed.

    ``mstop`` is the cross-validated step when ``cv`` was passed, otherwise
    the fit's own step. ``mstop_source`` says which.
    """

    available: bool
    family: str
    mstop: object
    mstop_source: str
    boost_loglik: float | None
    comparison_loglik: float | None
    gap: float | None
    material: bool
    comparison: str
    message: str
    tol: float = _TOL
    n_obs: int | None = None

    _MAPPING_KEYS = (
        "available", "family", "mstop", "mstop_source", "boost_loglik",
        "comparison_loglik", "gap", "material", "comparison", "message",
        "tol", "n_obs",
    )

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        gap = "NA" if self.gap is None else f"{self.gap:.4f}"
        return (
            f"SameFamilyGap(family={self.family!r}, mstop={self.mstop!r}, "
            f"gap={gap}, material={self.material})"
        )

    def to_dict(self, *, include_arrays: bool = False) -> dict:
        mstop = self.mstop
        if isinstance(mstop, (list, tuple)):
            mstop = [int(v) for v in mstop]
        elif isinstance(mstop, dict):
            mstop = {str(k): int(v) for k, v in mstop.items()}
        elif mstop is not None:
            mstop = int(mstop)
        return {
            "available": self.available,
            "family": self.family,
            "mstop": mstop,
            "mstop_source": self.mstop_source,
            "boost_loglik": self.boost_loglik,
            "comparison_loglik": self.comparison_loglik,
            "gap": self.gap,
            "gap_per_observation": (
                None if self.gap is None or not self.n_obs
                else self.gap / self.n_obs
            ),
            "material": self.material,
            "comparison": self.comparison,
            "message": self.message,
            "tol": self.tol,
            "n_obs": self.n_obs,
        }

    def to_frame(self, output="auto"):
        """One row. ``mstop`` is written as text when it is not a single step."""
        from mboostrs._output import output_frame

        row = self.to_dict()
        if not isinstance(row["mstop"], int):
            row["mstop"] = str(row["mstop"])
        return output_frame([row], output)


def same_family_gap(model, cv=None, *, comparison_loglik=None, tol=_TOL):
    """Compare a linear boost at its cross-validated step with the same-family GLM.

    Reports the chosen ``mstop``, the boost ``logLik`` there, the comparison
    log-likelihood, and the gap. When the gap is larger than ``tol`` (nats,
    total, default 1), the message says cross-validation stopped while the
    likelihood was still short of the GLM. The fit is not extended toward
    the maximum, and the step cross-validation chose is left where it is.

    Parameters
    ----------
    model
        A fitted ``glmboost`` or linear ``glmboostLSS`` model. The path must
        already reach the step being judged; this function does not call
        ``continue_to``.
    cv : cvrisk result, optional
        The result of ``model.cvrisk(...)``. The step judged is
        ``cv.optimal_mstop``. Without it, the model's own ``mstop`` is used
        and the message does not attribute the step to cross-validation.
    comparison_loglik : float, optional
        A same-family log-likelihood the caller already computed, on the same
        rows and with the same constants as ``logLik``. When given, no GLM
        is fit.
    tol : float
        A gap above this, in nats, is material. A boost that has reached the
        GLM sits well below it; the default is 1.

    Returns
    -------
    SameFamilyGap
        ``available`` is false when this family has no likelihood, or when no
        cheap same-family GLM can be fit and none was supplied. ``material``
        is then false too.
    """
    if tol < 0:
        raise ValueError(f"tol must be non-negative, got {tol}")
    family = type(getattr(model, "_family", None)).__name__
    mstop, source = _chosen_mstop(model, cv)
    _require_reached(model, mstop)
    view = _view_at(model, mstop)
    n_obs = _n_obs(view)
    label = _SUPPORTED.get(family, "")

    try:
        boost_ll = float(view.logLik())
    except NotImplementedError as exc:
        return _make(
            available=False, family=family, mstop=mstop, mstop_source=source,
            boost_loglik=None, comparison_loglik=None, gap=None, material=False,
            comparison="", message=str(exc), tol=tol, n_obs=n_obs,
        )

    if comparison_loglik is not None:
        return _judge(
            family=family, mstop=mstop, source=source, boost_ll=boost_ll,
            comparison_ll=float(comparison_loglik), tol=tol, n_obs=n_obs,
            comparison="supplied log-likelihood",
        )

    if family not in _SUPPORTED:
        return _unavailable(
            family, mstop, source, boost_ll, tol, n_obs,
            f"no same-family GLM is implemented for {family}",
        )

    refusal = _scope_refusal(model, family)
    if refusal:
        return _unavailable(family, mstop, source, boost_ll, tol, n_obs, refusal)

    try:
        comparison_ll, converged = _fit_comparison(view, family)
    except (ValueError, np.linalg.LinAlgError) as exc:
        return _unavailable(
            family, mstop, source, boost_ll, tol, n_obs,
            f"the same-family GLM could not be fit ({exc})",
        )

    if not converged or not math.isfinite(comparison_ll):
        return _unavailable(
            family, mstop, source, boost_ll, tol, n_obs,
            "the same-family GLM fit did not converge, so no shortfall is reported",
        )
    if comparison_ll < boost_ll - max(tol, 1e-4):
        return _unavailable(
            family, mstop, source, boost_ll, tol, n_obs,
            "the same-family GLM fit finished below the boost log-likelihood "
            f"({comparison_ll:.4f} against {boost_ll:.4f}), so it is not a "
            "completed maximum and no shortfall is reported",
            comparison_loglik=comparison_ll, comparison=label,
        )
    return _judge(
        family=family, mstop=mstop, source=source, boost_ll=boost_ll,
        comparison_ll=comparison_ll, tol=tol, n_obs=n_obs,
        comparison=label,
    )


# ── result text ────────────────────────────────────────────────────


def _fmt_mstop(mstop) -> str:
    if isinstance(mstop, dict):
        body = ", ".join(f"{key}: {int(value)}" for key, value in mstop.items())
        return "{" + body + "}"
    if isinstance(mstop, (list, tuple)):
        return "[" + ", ".join(str(int(v)) for v in mstop) + "]"
    return str(int(mstop))


def _make(**kwargs) -> SameFamilyGap:
    return SameFamilyGap(**kwargs)


def _unavailable(family, mstop, source, boost_ll, tol, n_obs, reason, *,
                 comparison_loglik=None, comparison=""):
    boost_txt = "unavailable" if boost_ll is None else f"{boost_ll:.4f}"
    message = (
        f"No same-family GLM comparison for {family}: {reason}. "
        f"Boost logLik at mstop={_fmt_mstop(mstop)} is {boost_txt}. "
        "The selected step was not changed."
    )
    gap = None
    if comparison_loglik is not None and boost_ll is not None:
        gap = float(comparison_loglik) - float(boost_ll)
    return _make(
        available=False, family=family, mstop=mstop, mstop_source=source,
        boost_loglik=boost_ll, comparison_loglik=comparison_loglik, gap=gap,
        material=False, comparison=comparison, message=message, tol=tol,
        n_obs=n_obs,
    )


def _judge(*, family, mstop, source, boost_ll, comparison_ll, tol, n_obs,
           comparison):
    gap = float(comparison_ll) - float(boost_ll)
    # A comparison we fit ourselves that lands below the boost is rejected by
    # the caller. A supplied number below the boost is not a shortfall.
    material = bool(gap > tol)
    where = _fmt_mstop(mstop)
    if material and source == "cvrisk":
        message = (
            f"Cross-validation stopped at mstop={where} while the likelihood "
            f"was still short of the same-family GLM ({comparison}): "
            f"boost logLik {boost_ll:.4f}, comparison logLik {comparison_ll:.4f}, "
            f"gap {gap:.4f}. The selected step was not changed."
        )
    elif material:
        message = (
            f"At mstop={where} the likelihood is still short of the same-family "
            f"GLM ({comparison}): boost logLik {boost_ll:.4f}, comparison "
            f"logLik {comparison_ll:.4f}, gap {gap:.4f}. The selected step "
            "was not changed."
        )
    elif gap < -tol:
        message = (
            f"The comparison log-likelihood ({comparison_ll:.4f}) is below the "
            f"boost ({boost_ll:.4f}) at mstop={where}, so it is not reported "
            "as a shortfall. The selected step was not changed."
        )
    else:
        whose = "Cross-validation's chosen step" if source == "cvrisk" else "This step"
        message = (
            f"{whose} mstop={where} is not short of the same-family GLM "
            f"({comparison}): boost logLik {boost_ll:.4f}, comparison logLik "
            f"{comparison_ll:.4f}, gap {gap:.4f}, within tol={tol:g}. "
            "The selected step was not changed."
        )
    return _make(
        available=True, family=family, mstop=mstop, mstop_source=source,
        boost_loglik=float(boost_ll), comparison_loglik=float(comparison_ll),
        gap=gap, material=material, comparison=comparison, message=message,
        tol=tol, n_obs=n_obs,
    )


# ── which step, and the fit already there ──────────────────────────


def _is_lss(model) -> bool:
    return type(model).__name__ == "LSSModel"


def _chosen_mstop(model, cv):
    if cv is None:
        return _native_mstop(model), "model"
    optimal = getattr(cv, "optimal_mstop", None)
    if optimal is None:
        raise TypeError(
            "cv must be a cvrisk result with optimal_mstop, or None. "
            f"Got {type(cv).__name__}."
        )
    if _is_lss(model):
        return _lss_budget(model, optimal), "cvrisk"
    if isinstance(optimal, (list, tuple, dict)):
        raise TypeError(
            "a standard model's cvrisk optimal_mstop is one step count; "
            f"got {optimal!r}"
        )
    return int(optimal), "cvrisk"


def _native_mstop(model):
    if not _is_lss(model):
        return int(model.mstop)
    return _lss_budget(model, model.mstop)


def _lss_budget(model, mstop) -> list[int]:
    names = list(model.param_names)
    if isinstance(mstop, dict):
        missing = [name for name in names if name not in mstop]
        if missing:
            raise ValueError(
                f"mstop is missing {missing}; this fit's parameters are {names}"
            )
        return [int(mstop[name]) for name in names]
    if isinstance(mstop, (list, tuple, np.ndarray)):
        values = [int(v) for v in np.asarray(mstop).ravel()]
        if len(values) == 1 and len(names) > 1:
            return values * len(names)
        if len(values) != len(names):
            raise ValueError(
                f"mstop has length {len(values)} but this fit has parameters {names}"
            )
        return values
    return [int(mstop)] * len(names)


def _fitted_budget(model) -> list[int]:
    counts = model.stopping_budget().per_parameter
    return [int(counts[name]) for name in model.param_names]


def _require_reached(model, mstop) -> None:
    if _is_lss(model):
        have = _fitted_budget(model)
        if any(int(want) > int(got) for want, got in zip(mstop, have, strict=True)):
            raise ValueError(
                f"the chosen mstop is {list(map(int, mstop))}, but this fit "
                f"only has {have} steps per parameter. same_family_gap does "
                "not extend the path."
            )
        return
    if int(mstop) > int(model.mstop):
        raise ValueError(
            f"the chosen mstop is {int(mstop)}, but this fit only has "
            f"{int(model.mstop)} steps. same_family_gap does not extend the path."
        )
    if int(mstop) < 1:
        raise ValueError(f"mstop must be at least 1, got {int(mstop)}")


def _view_at(model, mstop):
    """The fit at ``mstop``, without extending it and without mutating ``model``."""
    if _is_lss(model):
        if _fitted_budget(model) == [int(v) for v in mstop]:
            return model
        # A cyclic distributional path is not a prefix of itself, so the
        # likelihood at an earlier budget is a refit to that budget, not a
        # slice and not a run past it.
        return model[list(map(int, mstop))]
    if int(mstop) == int(model.mstop):
        return model
    return model[: int(mstop)]


def _n_obs(model) -> int:
    fitted = getattr(model, "_fitted", None)
    if fitted is not None:
        return int(len(np.asarray(fitted)))
    if _is_lss(model):
        return int(len(np.asarray(model._params[0]["fitted"])))
    return 0


def _scope_refusal(model, family) -> str:
    kind = type(model).__name__
    if family == "BetaLSS":
        if kind != "LSSModel" or getattr(model, "_model_kind", "") != "glmboostLSS":
            return (
                "beta regression is the comparison for a linear glmboostLSS "
                "BetaLSS fit; a smooth basis is a different model"
            )
        if any(param.get("kind") == "gam" for param in model._params):
            return "a parameter is fit with a smooth learner, not a linear beta regression"
        return ""
    if kind == "GAMBoostModel":
        return (
            "this is a gamboost fit. A same-family GLM is comparable only "
            "when it uses the same basis, and a smooth basis is not the "
            "linear specification this check fits"
        )
    if kind == "LSSModel":
        return (
            f"{family} here is distributional. The same-family GLM check "
            "covers linear glmboost families, and BetaLSS against a beta "
            "regression"
        )
    if kind != "GLMBoostModel":
        return f"same_family_gap does not compare a {kind}"
    return ""


# ── likelihoods, matching logLik ───────────────────────────────────


def _weights(model) -> np.ndarray:
    weights = getattr(model, "_weights", None)
    n = _n_obs(model)
    if weights is None:
        return np.ones(n, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).ravel()
    if weights.size != n:
        raise ValueError(
            f"weights have length {weights.size} but the fit has {n} rows"
        )
    if np.any(weights < 0) or not np.all(np.isfinite(weights)):
        raise ValueError("weights must be finite and non-negative")
    return weights


def _response(model) -> np.ndarray:
    from mboostrs.diagnostics import _observed_response

    return np.asarray(_observed_response(model), dtype=np.float64).ravel()


def _as_offset(offset, n) -> np.ndarray:
    if offset is None:
        return np.zeros(n, dtype=np.float64)
    array = np.asarray(offset, dtype=np.float64)
    if array.ndim == 0 or array.size == 1:
        return np.full(n, float(array.reshape(-1)[0]))
    if array.size != n:
        raise ValueError(f"offset has length {array.size}, expected {n}")
    return array.ravel().astype(np.float64, copy=False)


def _design_matrix(x, centering_means, intercept) -> np.ndarray:
    from mboostrs._rust import glmboost_design

    x = np.ascontiguousarray(x, dtype=np.float64)
    means = [float(v) for v in (centering_means or [])]
    design = np.asarray(glmboost_design(x, means, bool(intercept)), dtype=np.float64)
    if design.ndim == 1:
        design = design.reshape(-1, 1)
    return design


def _glm_parts(model):
    """Centered linear design and an offset that is not collinear with it.

    A constant offset is the intercept the design already has, so it is
    dropped. A varying offset stays, because it is not in the column space.
    """
    from mboostrs.glmboost import _extract_column

    names = list(model._predictor_names or [])
    n = _n_obs(model)
    if names:
        x = np.column_stack([_extract_column(model._data, name) for name in names])
    else:
        x = np.zeros((n, 0), dtype=np.float64)
    intercept = bool(getattr(model, "_intercept", True))
    # Learner 0 is the intercept when the formula has one. `_intercept` is
    # the authority; the design builder adds the column.
    design = _design_matrix(x, model._centering_means or [], intercept)
    offset = _as_offset(model._offset, n)
    if intercept and np.allclose(offset, offset.flat[0]):
        offset = np.zeros(n, dtype=np.float64)
    return design, offset, intercept


def _lss_parts(model, k):
    from mboostrs.glmboost import _extract_column

    param = model._params[k]
    names = list(param.get("predictor_names") or [])
    n = _n_obs(model)
    if names:
        x = np.column_stack([_extract_column(model._data, name) for name in names])
    else:
        x = np.zeros((n, 0), dtype=np.float64)
    labels = model.base_learner_names(model.param_names[k])
    intercept = bool(labels) and labels[0] == "(Intercept)"
    design = _design_matrix(x, param.get("centering_means") or [], intercept)
    offset = _as_offset(param.get("offset"), n)
    if intercept and np.allclose(offset, offset.flat[0]):
        offset = np.zeros(n, dtype=np.float64)
    return design, offset, intercept


def _link(model) -> np.ndarray:
    return np.asarray(model.predict(type="link"), dtype=np.float64).ravel()


def _guard(view, family, ll_at_boost: float) -> None:
    """The comparison formula has to agree with ``logLik`` on this fit.

    A gap against a different likelihood is not the shortfall this check
    reports. The tolerance is absolute plus a hair of the magnitude, which
    is what the likelihood contract tests themselves use.
    """
    boost = float(view.logLik())
    if not math.isfinite(ll_at_boost) or not math.isfinite(boost):
        raise ValueError(f"{family} logLik and the comparison formula are not both finite")
    if abs(ll_at_boost - boost) > 1e-5 + 1e-8 * abs(boost):
        raise ValueError(
            f"the {family} comparison likelihood ({ll_at_boost:.6f}) does not "
            f"match logLik() ({boost:.6f}) on this fit"
        )


def _fit_comparison(view, family) -> tuple[float, bool]:
    """Fit the comparison in the core and check its likelihood against logLik.

    ``same_family_glm_fit`` maximizes the family's own risk over the
    coefficients and nuisance; ``start_loglik`` is that likelihood at the
    boost's link predictor and nuisance, which must equal ``logLik()``.
    """
    from mboostrs._rust import same_family_glm_fit

    weights = _weights(view)
    y = _response(view)
    nuisance: list[float] = []
    if family == "BetaLSS":
        names = list(view.param_names)
        designs, offsets, starts = [], [], []
        for parameter in ("mu", "phi"):
            design, offset, _intercept = _lss_parts(view, names.index(parameter))
            designs.append(design)
            offsets.append(offset)
            starts.append(np.asarray(
                view.predict(parameter=parameter, type="link"), dtype=np.float64,
            ).ravel())
    else:
        design, offset, intercept = _glm_parts(view)
        if family == "PropOdds":
            n_levels = int(getattr(view._family, "n_levels", 0) or 0)
            if n_levels < 3:
                n_levels = int(np.max(y))
            codes = set(np.unique(np.rint(y)).astype(int).tolist())
            if not codes <= set(range(1, n_levels + 1)):
                raise ValueError(
                    f"PropOdds comparison expects category codes 1..{n_levels}, "
                    f"got {sorted(codes)}"
                )
            y = np.rint(y)
            # The thresholds take the intercept's role, and a constant offset
            # is absorbed by them.
            if intercept:
                design = design[:, 1:]
            if np.allclose(offset, offset.flat[0]):
                offset = np.zeros(len(y), dtype=np.float64)
            nuisance = [float(v) for v in np.asarray(view.nuisance()).ravel()]
            if len(nuisance) != n_levels - 1:
                raise ValueError(
                    f"PropOdds has {len(nuisance)} thresholds but the response "
                    f"implies {n_levels - 1}"
                )
        elif family in ("GammaReg", "NBinomial"):
            nuisance = [float(np.asarray(view.nuisance()).ravel()[0])]
        designs, offsets, starts = [design], [offset], [_link(view)]
    fit = same_family_glm_fit(
        family, np.ascontiguousarray(y, dtype=np.float64), weights,
        [np.ascontiguousarray(d, dtype=np.float64) for d in designs],
        [np.ascontiguousarray(o, dtype=np.float64) for o in offsets],
        [np.ascontiguousarray(s, dtype=np.float64) for s in starts],
        nuisance,
    )
    _guard(view, family, float(fit["start_loglik"]))
    return float(fit["loglik"]), bool(fit["converged"])
