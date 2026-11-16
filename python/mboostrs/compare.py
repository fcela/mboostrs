"""R functionality diff harness (Milestone 10).

Provides a structured comparison between mboostrs (Python) and mboost (R)
for a given model specification. The comparison runs both backends on
identical data and reports numerical diffs for fields with matching semantics.
Standard models compare fitted values, coefficients, risk, and offsets. LSS
models compare fitted parameter predictors; upstream's parameter-specific risk
lists are deliberately not compared with mboostrs' joint NLL path.

When R/rpy2 are unavailable, the tool still validates the spec and produces
an empty report skeleton with ``r_available=False``.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from typing import Any

import numpy as np

__all__ = ["CompareSpec", "CompareReport", "compare_to_r"]


def _has_r() -> bool:
    """Check whether R is installed."""
    try:
        subprocess.run(["R", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _has_rpy2() -> bool:
    return importlib.util.find_spec("rpy2") is not None


class CompareSpec:
    """Specification for a single model comparison.

    Parameters
    ----------
    model_type : str
        One of "glmboost", "gamboost", "glmboostLSS", "gamboostLSS".
    formula : str
        Model formula.
    family : str
        Family name (e.g. "Gaussian", "Binomial", "GaussianLSS").
    mstop : int
        Number of boosting iterations.
    nu : float
        Step length.
    n : int
        Number of observations for synthetic data.
    seed : int
        Random seed for reproducible synthetic data.
    risk : str
        Risk type ("inbag" or "oobag").
    weights : optional
        Observation weights (if None, equal weights).
    """

    def __init__(
        self,
        model_type: str = "glmboost",
        formula: str = "y ~ x1 + x2",
        family: str = "Gaussian",
        mstop: int = 50,
        nu: float = 0.1,
        n: int = 200,
        seed: int = 42,
        risk: str = "inbag",
        weights: list[float] | None = None,
    ):
        if model_type not in ("glmboost", "gamboost", "glmboostLSS", "gamboostLSS"):
            raise ValueError(f"Unknown model_type: {model_type!r}")
        self.model_type = model_type
        self.formula = formula
        self.family = family
        self.mstop = mstop
        self.nu = nu
        self.n = n
        self.seed = seed
        self.risk = risk
        self.weights = weights

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        """The specification as a JSON-safe mapping (it holds no arrays)."""
        return {
            "model_type": self.model_type,
            "formula": self.formula,
            "family": self.family,
            "mstop": self.mstop,
            "nu": self.nu,
            "n": self.n,
            "seed": self.seed,
            "risk": self.risk,
            "weights": self.weights,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CompareSpec:
        return cls(**{k: v for k, v in d.items() if k in cls.__init__.__code__.co_varnames})

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class CompareReport:
    """Structured comparison report between Python and R.

    Attributes
    ----------
    spec : dict
        The model specification used.
    r_available : bool
        Whether R + rpy2 were available to run the comparison.
    py_result : dict
        Python-side results (fitted, risk, coef, selected).
    r_result : dict or None
        R-side results, or None if R unavailable.
    diffs : dict
        Numerical diffs for each compared quantity.
    max_abs_diff : dict
        Maximum absolute difference for each quantity.
    status : str
        "ok", "r_unavailable", "error", or "mismatch".
    error : str or None
        Error message if status is "error".
    """

    def __init__(self, spec: CompareSpec):
        self.spec = spec.to_dict()
        self.r_available = False
        self.py_result: dict[str, Any] = {}
        self.r_result: dict[str, Any] | None = None
        self.diffs: dict[str, Any] = {}
        self.max_abs_diff: dict[str, float] = {}
        self.status = "pending"
        self.error: str | None = None

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        """The report as a mapping. ``diffs`` holds per-quantity summary
        numbers, not arrays, so ``include_arrays`` changes nothing here."""
        return {
            "spec": self.spec,
            "r_available": self.r_available,
            "status": self.status,
            "error": self.error,
            "max_abs_diff": self.max_abs_diff,
            "diffs": self.diffs,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=_json_default)

    def summary(self) -> str:
        """One-line summary string."""
        if self.status == "r_unavailable":
            return f"[{self.spec['model_type']}/{self.spec['family']}] R unavailable — Python-only"
        if self.status == "error":
            return f"[{self.spec['model_type']}/{self.spec['family']}] ERROR: {self.error}"
        parts = []
        for key, val in sorted(self.max_abs_diff.items()):
            parts.append(f"{key}={val:.2e}")
        return f"[{self.spec['model_type']}/{self.spec['family']}] {' '.join(parts)}"


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def _generate_data(spec: CompareSpec) -> dict[str, np.ndarray]:
    """Generate reproducible synthetic data matching R's set.seed."""
    rng = np.random.default_rng(spec.seed)
    n = spec.n
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    # Use the same generating process as the parity tests
    y = 1.5 * x1 + 0.8 * x2 + rng.standard_normal(n) * 0.3
    return {"x1": x1, "x2": x2, "y": y}


def _fit_python(spec: CompareSpec, data: dict[str, np.ndarray]) -> dict[str, Any]:
    """Fit the model in Python and extract results."""
    import mboostrs as mb

    formula = spec.formula
    df = {"y": data["y"], "x1": data["x1"], "x2": data["x2"]}
    control = mb.boost_control(spec.mstop, spec.nu, spec.risk)
    weights = spec.weights

    if spec.model_type == "glmboost":
        family = _resolve_family_py(spec.family)
        mod = mb.glmboost(formula, data=df, family=family, control=control, weights=weights)
        return {
            "fitted": np.asarray(mod.fitted_, dtype=np.float64),
            "risk": np.asarray(mod.risk_, dtype=np.float64),
            "coef": mod.coef(),
            "offset": float(mod.offset_),
        }

    if spec.model_type == "gamboost":
        family = _resolve_family_py(spec.family)
        mod = mb.gamboost(
            formula, data=df, family=family, control=control, weights=weights
        )
        return {
            "fitted": np.asarray(mod.fitted_, dtype=np.float64),
            "risk": np.asarray(mod.risk_, dtype=np.float64),
            "coef": mod.coef(),
            "offset": float(mod.offset_),
        }

    if spec.model_type == "glmboostLSS":
        family = _resolve_lss_family_py(spec.family)
        mod = mb.glmboostLSS(formula, data=df, families=family, control=control, weights=weights)
        return {
            "fitted": np.concatenate([
                np.asarray(mod.fitted(parameter=p), dtype=np.float64)
                for p in mod.param_names
            ]),
        }

    if spec.model_type == "gamboostLSS":
        family = _resolve_lss_family_py(spec.family)
        mod = mb.gamboostLSS(formula, data=df, families=family, control=control, weights=weights)
        return {
            "fitted": np.concatenate([
                np.asarray(mod.fitted(parameter=p), dtype=np.float64)
                for p in mod.param_names
            ]),
        }

    raise NotImplementedError(f"model_type {spec.model_type!r} not yet supported")


def _resolve_family_py(name: str):
    import mboostrs as mb

    families = {
        "Gaussian": mb.Gaussian, "Binomial": mb.Binomial, "Poisson": mb.Poisson,
        "NBinomial": mb.NBinomial, "PropOdds": mb.PropOdds, "Laplace": mb.Laplace,
        "Huber": mb.Huber, "GaussReg": mb.GaussReg, "GammaReg": mb.GammaReg,
        "Quantile": mb.Quantile, "Expectile": mb.Expectile, "AdaExp": mb.AdaExp,
    }
    if name not in families:
        raise ValueError(f"Unknown family: {name!r}. Available: {list(families)}")
    return families[name]()


def _resolve_lss_family_py(name: str):
    from mboostrs.gamlss import GammaLSS, GaussianLSS

    families = {"GaussianLSS": GaussianLSS, "GammaLSS": GammaLSS}
    if name not in families:
        raise ValueError(f"Unknown LSS family: {name!r}. Available: {list(families)}")
    return families[name]()


def _fit_r(spec: CompareSpec, data: dict[str, np.ndarray]) -> dict[str, Any]:
    """Fit the model in R via rpy2 and extract results."""
    import rpy2.robjects as ro
    from rpy2.robjects import default_converter, numpy2ri
    from rpy2.robjects.conversion import localconverter

    np_conv = default_converter + numpy2ri.converter

    with localconverter(np_conv):
        ro.globalenv["x1"] = data["x1"]
        ro.globalenv["x2"] = data["x2"]
        ro.globalenv["y"] = data["y"]
        weights_values = (
            np.ones(len(data["y"]), dtype=np.float64)
            if spec.weights is None
            else np.asarray(spec.weights, dtype=np.float64)
        )
        # numpy2ri preserves a one-dimensional ndarray's `dim` attribute;
        # mboost requires a plain numeric vector for case weights.
        weights = ro.FloatVector(weights_values.tolist())

        is_lss = spec.model_type in ("glmboostLSS", "gamboostLSS")

        if is_lss:
            family_arg = f"families = {spec.family}()"
            fit_call = f"{spec.model_type}("
        else:
            family_arg = f"family = mboost::{spec.family}()"
            fit_call = f"mboost::{spec.model_type}("

        # LSS constructors and fit functions live in gamboostLSS; loading it
        # also attaches mboost and its base-learners.
        lib_line = "library(gamboostLSS)" if is_lss else "library(mboost)"

        # center= is only valid for glmboost, not gamboost
        center_arg = "center = TRUE," if spec.model_type == "glmboost" else ""
        weights_arg = "weights = weights," if spec.weights is not None else ""
        # mboost's Binomial() accepts only a two-level factor response (the
        # second level is the event). The 0/1 numeric vector is the same data
        # the Python fit receives, so the factor is built on the R side only.
        factor_line = (
            "dat$y <- factor(dat$y)" if spec.family == "Binomial" else ""
        )

        r_code = f"""
        function(x1, x2, y, weights, mstop, nu, seed) {{
            set.seed(seed)
            {lib_line}
            # mboost's Binomial() rejects a numeric response ("response is not a
            # factor"), so a 0/1 vector is handed to R as a two-level factor.
            # The levels are "0" and "1", so the success class is still 1.
            yv <- if ("{spec.family}" == "Binomial") factor(y, levels = c(0, 1)) else y
            dat <- data.frame(x1=x1, x2=x2, y=yv)
            {factor_line}
            fit <- {fit_call}
                as.formula({spec.formula!r}),
                data = dat,
                {family_arg},
                {center_arg}
                {weights_arg}
                control = mboost::boost_control(mstop = mstop, nu = nu)
            )
            if ("{spec.model_type}" %in% c("glmboostLSS", "gamboostLSS")) {{
                fitted_mat <- sapply(fit, function(m) as.numeric(stats::predict(m, type='link')))
                # Upstream risk(fit) is a list of parameter-specific paths,
                # whereas mboostrs stores a joint NLL path. Do not pretend
                # these differently defined quantities are comparable.
                list(fitted = as.numeric(fitted_mat))
            }} else {{
                cc <- if ("{spec.model_type}" == "glmboost") {{
                    stats::coef(fit, off2int = FALSE)
                }} else {{
                    stats::coef(fit)
                }}
                off <- attr(cc, "offset")
                list(
                    fitted = as.numeric(stats::predict(fit, type='link')),
                    risk = as.numeric(mboost::risk(fit)),
                    coef = as.numeric(unlist(cc, recursive = TRUE, use.names = FALSE)),
                    offset = as.numeric(off)
                )
            }}
        }}
        """
        r_func = ro.r(r_code)
        result = r_func(
            data["x1"], data["x2"], data["y"], weights,
            spec.mstop, spec.nu, spec.seed,
        )

    out: dict[str, Any] = {}
    # R named list may not carry names through rpy2; map by position.
    if is_lss:
        keys = ["fitted"]
    else:
        keys = ["fitted", "risk", "coef", "offset"]
    for i, nm in enumerate(keys):
        try:
            val = result[i]
        except (IndexError, TypeError):
            continue
        try:
            length = len(val)
        except TypeError:
            length = 1
        if length <= 1:
            out[nm] = float(val[0]) if length == 1 else 0.0
        else:
            out[nm] = np.asarray(val, dtype=np.float64)
    return out


def _compute_diffs(py: dict, r: dict) -> tuple[dict[str, Any], dict[str, float]]:
    """Compute numerical diffs between Python and R results."""
    diffs: dict[str, Any] = {}
    max_diffs: dict[str, float] = {}

    def numeric_array(value):
        if isinstance(value, dict):
            parts = [np.asarray(v, dtype=np.float64).ravel() for v in value.values()]
            return np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)
        return np.asarray(value, dtype=np.float64).ravel()

    for key in ("fitted", "risk", "coef"):
        if key not in py or key not in r:
            if key in py or key in r or key == "fitted":
                diffs[key] = {
                    "mismatch": True,
                    "missing": "python" if key not in py else "r",
                }
                max_diffs[key] = float("inf")
            continue
        try:
            py_val = numeric_array(py[key])
            r_val = numeric_array(r[key])
            if py_val.shape == r_val.shape:
                d = np.abs(py_val - r_val)
                max_diff = float(np.max(d)) if d.size else 0.0
                mean_diff = float(np.mean(d)) if d.size else 0.0
                diffs[key] = {"max_abs": max_diff, "mean_abs": mean_diff}
                max_diffs[key] = max_diff
            else:
                diffs[key] = {
                    "shape_mismatch": True,
                    "py_shape": list(py_val.shape),
                    "r_shape": list(r_val.shape),
                }
                max_diffs[key] = float("inf")
        except (TypeError, ValueError) as exc:
            diffs[key] = {"mismatch": True, "error": str(exc)}
            max_diffs[key] = float("inf")

    if "offset" in py or "offset" in r:
        if "offset" not in py or "offset" not in r:
            diffs["offset"] = {
                "mismatch": True,
                "missing": "python" if "offset" not in py else "r",
            }
            max_diffs["offset"] = float("inf")
            return diffs, max_diffs
        try:
            py_offset = numeric_array(py["offset"])
            r_offset = numeric_array(r["offset"])
            if py_offset.shape != r_offset.shape:
                diffs["offset"] = {
                    "shape_mismatch": True,
                    "py_shape": list(py_offset.shape),
                    "r_shape": list(r_offset.shape),
                }
                max_diffs["offset"] = float("inf")
            else:
                d = np.abs(py_offset - r_offset)
                max_diff = float(np.max(d)) if d.size else 0.0
                diffs["offset"] = {"max_abs": max_diff}
                max_diffs["offset"] = max_diff
        except (TypeError, ValueError):
            diffs["offset"] = {"mismatch": True}
            max_diffs["offset"] = float("inf")

    return diffs, max_diffs


def compare_to_r(
    spec: CompareSpec,
    data: dict[str, np.ndarray] | None = None,
    backend: str = "mboost",
) -> CompareReport:
    """Compare a mboostrs fit against R mboost.

    Parameters
    ----------
    spec : CompareSpec
        Model specification.
    data : dict, optional
        Pre-generated data. If None, synthetic data is generated from the spec.
    backend : str
        R backend to compare against (currently only "mboost").

    Returns
    -------
    CompareReport
        Structured report with diffs.
    """
    if backend != "mboost":
        raise ValueError(f"unsupported R comparison backend: {backend!r}")
    report = CompareReport(spec)

    # Generate or use provided data
    if data is None:
        data = _generate_data(spec)

    # Fit Python side
    try:
        report.py_result = _fit_python(spec, data)
    # Broad on purpose: the report *is* the typed missing value, carrying
    # `status="error"` and the message. Narrowing would turn an unexpected
    # failure on one side of the comparison into an aborted comparison, which
    # is the outcome this function exists to avoid.
    except Exception as exc:
        report.status = "error"
        report.error = f"Python fit failed: {exc!s}"
        return report

    # Check R availability
    if not (_has_r() and _has_rpy2()):
        report.r_available = False
        report.status = "r_unavailable"
        return report

    report.r_available = True

    # Fit R side. Broad for the same reason, and because rpy2 raises
    # `RRuntimeError`, which is outside this package's exception hierarchy.
    try:
        report.r_result = _fit_r(spec, data)
    except Exception as exc:
        report.status = "error"
        report.error = f"R fit failed: {exc!s}"
        return report

    # Compute diffs
    report.diffs, report.max_abs_diff = _compute_diffs(report.py_result, report.r_result)
    if any(
        isinstance(v, dict) and (v.get("shape_mismatch") or v.get("mismatch"))
        for v in report.diffs.values()
    ):
        report.status = "mismatch"
    else:
        report.status = "ok"
    return report
