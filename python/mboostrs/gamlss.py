"""glmboostLSS / gamboostLSS: boosting for GAMLSS (location/scale/shape).

Called as a function, ``mboostrs.gamlss(formulas, data)`` fits a
distributional regression by REML with no boosting; its arguments are those
of ``mboostrs.direct_gam.gamlss``. Workflow, pitfalls and worked analyses:
``mboostrs.agent_guide()``, or ``python -m mboostrs guide index`` from a
shell. After a fit, run ``mboostrs.reference_gap`` to measure what a
flexible model still gains.

The cyclic and non-cyclic fitting loops live in the Rust core
(``mboostrs_core::lss``); this module provides the Python front-end, result
objects, prediction helpers, and inspection methods for linear and smooth LSS
models.

Binding contract (``mboostrs._rust.fit_glmboostlss``), pinned here:

    fit_glmboostlss(
        x_all,            # (n, sum(param_ncols)) f64, columns grouped by param
        param_ncols,      # list[int], #predictors per parameter (in order)
        learner_names,    # list[str], flat predictor names (len == sum(param_ncols))
        y, weights,       # (n,) f64
        mstop, nu,        # int, float
        family_name,      # str, e.g. "gaussianlss"
        offsets,          # Optional[list[float]] per-parameter offset override
    ) -> (results, risk, selection_order, execution_report)
        results = list per parameter of (offset, selected, coefs_flat, fitted)
        risk    = joint risk path (len mstop+1)

Each parameter is fitted with an intercept learner + one scalar-bols learner per
column (matching ``glmboost``), so prediction reuses ``predict_glmboost``.
"""
from __future__ import annotations

import sys
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from math import lgamma

import numpy as np

from mboostrs._comparison_identity import fit_identity, record_data_identity
from mboostrs._errors import MboostrsKeyError
from mboostrs._model_api import capture_fit_warnings
from mboostrs._names import partition_tilde, quote_name
from mboostrs._output import export_result
from mboostrs._results import MappingResultMixin, Provenance
from mboostrs._validate import (
    _model_extrapolation,
    _model_unseen_levels,
)
from mboostrs._weights import (
    WeightPolicyMixin,
    normalize_weights,
    rescale_fold_weights,
)
from mboostrs._workflow import call_fit, fit_checkpoint
from mboostrs.families import _TYPE_DEFAULT, maybe_warn_link_scale, prediction_scale_text
from mboostrs.formula import parse_formula, parse_gamboost_formula, parse_response
from mboostrs.glmboost import (
    _extract_column,
    _merge_part,
    _part_of,
    _replay_interactions,
    _tensor_margin_clamp,
    _tensor_margin_rows,
)
from mboostrs.interactions import reject_factor_by
from mboostrs.learners import (
    Bbs,
    Bconstrained,
    Bkernel,
    Bmono,
    Bmrf,
    Bns,
    Bols,
    Brad,
    Brandom,
    Bspatial,
    Btree,
    Buser,
    Hinge,
)
from mboostrs.report import ExecutionReport, lss_path_rows, path_rows_output
from mboostrs.transforms import (
    TransformRecipe,
    apply_recipe,
    encode_factors,
    expand_factor_dummies,
    extract_transforms,
    factor_term_labels,
    fit_recipe,
    impute_missing,
    lower_categorical_bols,
    response_inverse,
)

try:
    from mboostrs._control import boost_control, step_kwargs
    from mboostrs._rust import cvrisk_gamboostlss as _cvrisk_gamboostlss
    from mboostrs._rust import cvrisk_gamboostlss_mixed as _cvrisk_gamboostlss_mixed
    from mboostrs._rust import cvrisk_glmboostlss as _cvrisk_glmboostlss
    from mboostrs._rust import fit_gamboostlss as _fit_gamboostlss
    from mboostrs._rust import fit_gamboostlss_mixed as _fit_gamboostlss_mixed
    from mboostrs._rust import fit_glmboostlss as _fit_glmboostlss
    from mboostrs._rust import fit_glmboostlss_noncyclic as _fit_glmboostlss_noncyclic
    from mboostrs._rust import predict_glmboost as _predict_glmboost
    from mboostrs._rust import prepare_gamboostlss_stores as _prepare_gamboostlss_stores
    from mboostrs._rust import search_gamboostlss_grid as _search_gamboostlss_grid
    from mboostrs._rust import stabsel_selected_gamboostlss as _stabsel_selected_gamboostlss
    from mboostrs._rust import (
        stabsel_selected_gamboostlss_mixed as _stabsel_selected_gamboostlss_mixed,
    )
    from mboostrs._rust import stabsel_selected_glmboostlss as _stabsel_selected_glmboostlss
    _HAS_LSS_ENGINE = True
except ImportError:  # pragma: no cover - engine always present in a real install
    _HAS_LSS_ENGINE = False
    from mboostrs._control import step_kwargs
    _fit_glmboostlss = None
    _fit_glmboostlss_noncyclic = None
    _fit_gamboostlss = None
    _fit_gamboostlss_mixed = None
    _prepare_gamboostlss_stores = None
    _cvrisk_glmboostlss = None
    _cvrisk_gamboostlss = None
    _cvrisk_gamboostlss_mixed = None
    _search_gamboostlss_grid = None
    _stabsel_selected_glmboostlss = None
    _stabsel_selected_gamboostlss = None
    _stabsel_selected_gamboostlss_mixed = None
    _predict_glmboost = None
    boost_control = None


_SQRT_2PI = np.sqrt(2.0 * np.pi)
_LGAMMA = np.vectorize(lgamma, otypes=[np.float64])


def _on_count_support(y) -> np.ndarray:
    """Whether each ``y`` is a non-negative integer, where a count mass is defined.

    A count family's mass is zero elsewhere, as R's ``dpois``/``dnbinom``
    give; evaluating the log-gamma formula there interpolates a positive
    "mass" between integers and fails at negative integers.
    """
    y = np.asarray(y, dtype=np.float64)
    return (y >= 0) & (y == np.floor(y))
_CAPABILITY_HINT = " Check mb.family_capabilities() for supported diagnostics."


def _special(name: str, *args) -> np.ndarray:
    """Evaluate a special function of the core (`mboostrs_core::special`)
    elementwise over broadcast arguments.

    The series, continued fractions and root searches there stop after a
    fixed number of iterations; when any element reaches that limit before
    its tolerance this warns, naming the function and the count, instead of
    returning a value of unknown accuracy without notice.
    """
    from mboostrs import _rust

    arrays = np.broadcast_arrays(*[np.asarray(a, dtype=np.float64) for a in args])
    shape = arrays[0].shape
    values, unconverged = _rust.special_function(
        name, [np.ascontiguousarray(a).ravel().tolist() for a in arrays])
    if unconverged:
        warnings.warn(
            f"{name}: {unconverged} of {arrays[0].size} value(s) reached the "
            "iteration limit before converging; they may be inaccurate.",
            RuntimeWarning, stacklevel=3)
    return np.asarray(values, dtype=np.float64).reshape(shape)


def _core_quantile(family: str, p, params, names, extra=()) -> np.ndarray:
    """Quantiles of an LSS family at probabilities ``p``, computed in the core.

    ``names`` are the family's distribution parameters in the order its
    ``param_names`` lists them; ``extra`` are per-row values the family takes
    after them (the known variance of ``GaussianKnownVarLSS``, the trial count
    of ``BinomialLSS``). The inputs broadcast as NumPy broadcasts them, and the
    result has the broadcast shape. Probabilities outside (0, 1) and invalid
    parameters are refused by the core, which names the family.
    """
    from mboostrs import _rust

    columns = [np.asarray(p, dtype=np.float64)]
    columns += [_param_array(params, name) for name in names]
    columns += [np.asarray(value, dtype=np.float64) for value in extra]
    arrays = np.broadcast_arrays(*columns)
    shape = arrays[0].shape
    values, unconverged = _rust.lss_quantile(
        family, [np.ascontiguousarray(a, dtype=np.float64).ravel().tolist() for a in arrays])
    if unconverged:
        warnings.warn(
            f"{family} quantile: {unconverged} of {arrays[0].size} value(s) reached "
            "the iteration limit before converging; they may be inaccurate.",
            RuntimeWarning, stacklevel=3)
    return np.asarray(values, dtype=np.float64).reshape(shape)


def _family_quantile(family, p, params) -> np.ndarray:
    """Quantiles of an LSS family object, through the core.

    The per-row extra argument is filled with its default: a one-trial
    binomial (``size=1``), the value :meth:`BinomialLSS.quantile` uses when
    called without one.
    """
    name = type(family).__name__
    extra = (1.0,) if name == "BinomialLSS" else ()
    return _core_quantile(name, p, params, family.param_names, extra=extra)


def _normal_cdf_array(z: np.ndarray) -> np.ndarray:
    return _special("normal_cdf", z)


def _normal_ppf_array(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    if np.any(~((p > 0.0) & (p < 1.0))):
        from statistics import StatisticsError

        raise StatisticsError("p must be in the range 0.0 < p < 1.0")
    return _special("normal_quantile", p)


def _normal_log_cdf_array(z: np.ndarray) -> np.ndarray:
    """``log(Phi(z))``, accurate in the far left tail.

    `InverseGaussianLSS.cdf` multiplies a normal tail by ``exp(2/(mu*sigma^2))``,
    which overflows on its own for a small ``mu*sigma^2`` while the product
    stays finite. Forming the second term in logs and exponentiating once keeps
    it representable. Below ``z = -8`` the core uses the asymptotic expansion
    of the Mills ratio, since the CDF itself underflows there.
    """
    return _special("normal_log_cdf", z)


def _cdf_left(family, y, params) -> np.ndarray:
    """The left limit F(y-), including atoms and lattice distributions."""
    y = np.asarray(y, dtype=np.float64)
    if hasattr(family, "cdf_left"):
        return np.asarray(family.cdf_left(y, params), dtype=np.float64)
    if getattr(family, "_discrete", False):
        return np.asarray(family.cdf(y - 1.0, params), dtype=np.float64)
    return np.asarray(family.cdf(y, params), dtype=np.float64)


# ─── Regularized incomplete gamma / beta ────────────────────────────────────
# The Gamma, Student-t, and Beta CDFs need the regularized incomplete gamma and
# beta functions. They are computed by the core (`mboostrs_core::special`,
# reached through `_special`), which reports an iteration limit instead of
# returning silently; `tests/test_lss_special_functions.py` checks them against
# SciPy, which is available in the development and CI environments.


def _gammainc_p(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a, x)."""
    if x < 0.0 or a <= 0.0:
        raise ValueError("_gammainc_p requires a > 0 and x >= 0")
    return float(_special("gamma_p", a, x))


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return float(_special("beta_i", a, b, x))


def _student_t_cdf_scalar(t: float, df: float) -> float:
    """Student-t CDF via the incomplete beta identity."""
    if df <= 0.0:
        raise ValueError("_student_t_cdf_scalar requires df > 0")
    return float(_special("t_cdf", t, df))


def _boxcox_z(y, mu, sigma, nu) -> np.ndarray:
    """Box-Cox standardized value with its continuous ``nu -> 0`` limit."""
    y, mu, sigma, nu = np.broadcast_arrays(
        np.asarray(y, dtype=np.float64),
        np.asarray(mu, dtype=np.float64),
        np.asarray(sigma, dtype=np.float64),
        np.asarray(nu, dtype=np.float64),
    )
    small = np.abs(nu) < 1e-10
    out = np.empty(y.shape, dtype=np.float64)
    out[small] = np.log(y[small] / mu[small]) / sigma[small]
    out[~small] = (
        np.power(y[~small] / mu[~small], nu[~small]) - 1
    ) / (nu[~small] * sigma[~small])
    return out


def _boxcox_edge(sigma, nu) -> np.ndarray:
    """Positive-response truncation edge without dividing on the ``nu=0`` limit."""
    sigma, nu = np.broadcast_arrays(
        np.asarray(sigma, dtype=np.float64), np.asarray(nu, dtype=np.float64)
    )
    out = np.full(sigma.shape, np.inf, dtype=np.float64)
    finite = np.abs(nu) >= 1e-10
    out[finite] = 1 / (sigma[finite] * np.abs(nu[finite]))
    return out


def _bcpe_log_c(tau) -> np.ndarray:
    tau = np.asarray(tau, dtype=np.float64)
    return .5 * (
        -(2 / tau) * np.log(2)
        + _LGAMMA(1 / tau)
        - _LGAMMA(3 / tau)
    )


def _bcpe_cdf_t(t, tau) -> np.ndarray:
    """CDF of the standardized power-exponential kernel used by BCPE."""
    t, tau = np.broadcast_arrays(
        np.asarray(t, dtype=np.float64), np.asarray(tau, dtype=np.float64)
    )
    c = np.exp(_bcpe_log_c(tau))
    s = .5 * np.power(np.abs(t / c), tau)
    return .5 * (1 + np.sign(t) * _special("gamma_p", 1 / tau, s))


def _records_frame(records: list[dict[str, object]], output: str = "auto",
                   *, columns: tuple[str, ...] | None = None):
    from mboostrs._output import output_frame
    return output_frame(records, output, columns=columns)


def _merge_selected(by_parameter: dict) -> list[str]:
    """Flatten ``{parameter: [learner, ...]}`` to ``["learner.parameter", ...]``.

    Two parameters can select learners of the same name, so counting across
    the whole fit needs the qualified form.
    """
    merged: list[str] = []
    for parameter, path in by_parameter.items():
        merged.extend(f"{learner}.{parameter}" for learner in path)
    return merged


def _prob_col(prob: float) -> str:
    return f"q{100.0 * prob:g}".replace(".", "_")


def _param_array(params, name: str) -> np.ndarray:
    try:
        value = params[name]
    except (KeyError, TypeError) as exc:
        raise MboostrsKeyError(
            f"missing LSS parameter {name!r}",
            code="missing_value",
            subject=name,
            details=f"missing LSS parameter {name!r}",
        ) from exc
    return np.asarray(value, dtype=np.float64)


def _build_lss_response(data, response, survival, multivariate=False,
                        known_variance=False):
    """Return ``(y, n_obs)``. For survival families the response is a
    ``(time, status)`` tuple and y is the concatenated ``[t0..t_{n-1},
    d0..d_{n-1}]`` slice the Rust survival families expect (matching R's
    model.response ordering). For multivariate families (DirichletLSS) the
    response is a ``(y1, ..., yK)`` tuple and y is the concatenated
    ``[col0.., .., col_{K-1}..]`` slice. For known-variance families
    (GaussianKnownVarLSS) the response is an ``(outcome, s2)`` tuple and y is
    the concatenated ``[outcome.., s2..]`` slice the Rust family expects —
    the auxiliary vector rides inside y so reweighting-based resampling
    (cvrisk, stabsel) keeps it aligned. Otherwise y is the plain column."""
    if known_variance:
        o = np.asarray(_extract_column(data, response[0]), dtype=np.float64)
        s2 = np.asarray(_extract_column(data, response[1]), dtype=np.float64)
        return np.ascontiguousarray(np.concatenate([o, s2]), dtype=np.float64), len(o)
    if survival:
        t = np.asarray(_extract_column(data, response[0]), dtype=np.float64)
        d = np.asarray(_extract_column(data, response[1]), dtype=np.float64)
        # Concatenated [times..., status...], matching R's model.response ordering
        # and the Rust survival families.
        return np.ascontiguousarray(np.concatenate([t, d]), dtype=np.float64), len(t)
    if multivariate:
        cols = [np.asarray(_extract_column(data, nm), dtype=np.float64) for nm in response]
        n = len(cols[0])
        return np.ascontiguousarray(np.concatenate(cols), dtype=np.float64), n
    y = np.ascontiguousarray(_extract_column(data, response), dtype=np.float64)
    return y, len(y)


def _uncensored_survival_family(family, formulas, data):
    """The uncensored variant of a survival family, when the response is plain.

    A survival family given a plain numeric response (not ``Surv(time, status)``)
    treats every observation as an event, so its risk is the negative
    log-likelihood of the response itself under the family's distribution. The
    returned variant reports itself as a plain-response family, which keeps the
    response layout, the likelihood contract and the distribution methods on the
    response scale, the same as the other distributional families. A mixed set
    of responses (some ``Surv``, some plain) is left for the caller to refuse.
    """
    if not getattr(family, "_survival", False) or not hasattr(family, "uncensored"):
        return family
    responses = [parse_response(f) for f in formulas]
    if any(isinstance(r, tuple) for r in responses):
        return family
    name = type(family).__name__
    y = np.asarray(_extract_column(data, responses[0]), dtype=np.float64)
    bad = np.flatnonzero(~np.isfinite(y) | (y <= 0))
    if bad.size:
        i = int(bad[0])
        raise ValueError(
            f"{name} needs a positive response: {bad.size} value(s) are zero, "
            f"negative or missing (first at row {i}: {y[i]!r}). Remove those rows, "
            f"or, if some times are right-censored, pass Surv(time, status) with "
            f"status 0 for the censored rows."
        )
    return family.uncensored()


def learner_spec(learner, data, *, bols_form="block", kronecker_n_resp=None):
    """One self-describing record for a base learner, as Rust reads it.

    This is the Python side of the learner boundary. A learner crosses into
    Rust as a single type-tagged dict — the shape the R binding has always used
    — parsed by `parse_learner_blueprint_list` into a core `LearnerBlueprint`.

    `gamboost`, `gamboostLSS`, and `glmboostLSS` all build their specs here, so
    a new learner attribute is added in one place rather than threaded through
    parallel arrays at three call sites. The alternative this replaced kept ~80
    lists index-aligned by hand, which is the hazard
    `Family::encode_response`'s doc comment records the cost of.
    """
    # `Brad` is a construction over `Buser` — a Matérn design and an identity
    # penalty — so it lowers here and every downstream path sees a `buser`.
    if isinstance(learner, Brad):
        learner = learner.as_buser(data)
    if not isinstance(
        learner,
        (
            Bols,
            Bbs,
            Bns,
            Brandom,
            Bmrf,
            Bspatial,
            Bmono,
            Bconstrained,
            Btree,
            Hinge,
            Bkernel,
            Buser,
        ),
    ):
        raise ValueError(
            "gamboostLSS currently supports bols(), bbs(), bns(), brandom(), bmrf(), "
            "bspatial(), bmono(), bconstrained(), btree(), hinge(), bkernel(), and "
            f"buser() base-learners; got {learner.__class__.__name__}"
        )
    if learner.by is not None:
        from mboostrs.interactions import reject_factor_by

        reject_factor_by(f"by={quote_name(learner.by)}", data, "an LSS model")
    by = (
        np.ascontiguousarray(_extract_column(data, learner.by), dtype=np.float64)
        if learner.by is not None
        else None
    )
    spec = {
        "kind": (
            "bols"
            if isinstance(learner, Bols)
            # `Bbs(constraint=...)` is the formula spelling of a shape
            # constraint. It has to lower to the bmono engine learner; routing
            # it to "bbs" drops the constraint and fits unconstrained.
            else "bmono"
            if isinstance(learner, Bbs) and learner.constraint is not None
            else "bbs"
            if isinstance(learner, Bbs)
            else "bns"
            if isinstance(learner, Bns)
            else "brandom"
            if isinstance(learner, Brandom)
            else "bmrf"
            if isinstance(learner, Bmrf)
            else "bspatial"
            if isinstance(learner, Bspatial)
            else "bmono"
            if isinstance(learner, Bmono)
            else "bconstrained"
            if isinstance(learner, Bconstrained)
            else "bkernel"
            if isinstance(learner, Bkernel)
            else "buser"
            if isinstance(learner, Buser)
            else "bhinge"
            if isinstance(learner, Hinge)
            else "btree"
        ),
        "name": learner.name,
        "x": (
            np.ascontiguousarray(
                np.asarray(learner.X, dtype=np.float64).ravel(),
                dtype=np.float64,
            )
            if isinstance(learner, Buser)
            else np.ascontiguousarray(
                np.column_stack(
                    [_extract_column(data, name) for name in learner.names]
                ).ravel(),
                dtype=np.float64,
            )
            if isinstance(learner, Btree)
            else np.ascontiguousarray(
                _extract_column(
                    data,
                    learner.name1
                    if isinstance(learner, Bspatial)
                    else learner.name,
                ),
                dtype=np.float64,
            )
        ),
        "by": by,
        "lambda": (
            None
            if not hasattr(learner, "lambda_value")
            or learner.lambda_value is None
            else float(learner.lambda_value)
        ),
    }
    if isinstance(learner, Bols):
        # `bols(x, intercept=True)` means two different things by frontend.
        # An LSS predictor gets its own constant term, so the learner is the
        # two-column block `[1, x]` ("block"). `gamboost` puts the constant in
        # the model offset and fits one weight-centered column ("centered"),
        # which is why the same term has one coefficient there and two here.
        #
        # The two forms fit the same thing only because a constant is available
        # either way — inside the learner, or from the offset. A `by=` breaks
        # that: R multiplies the whole design by it, so the learner is
        # `[by, x*by]` and carries the `by` main effect, where a centered
        # `x*by` spans `{1, x*by}` and cannot represent it. Once `by` is
        # present the block form is the only one that matches R, whatever the
        # frontend.
        # A penalty forces the same choice for the same reason: R shrinks the
        # learner's own intercept along with its slope, and a constant supplied
        # by the offset is not shrinkable. The centered single column stays the
        # `gamboost` form only where it is genuinely equivalent — an
        # unpenalized `bols` with no `by`.
        #
        # That equivalence is a property of the *whole menu*, not of one
        # learner. The centered form leans on the model offset for its
        # constant; a block learner carries its own. Mix the two and the
        # constant is represented twice, and the two representations compete
        # along the path: measured on `bols(x, lambda=2) + bols(z)`, fitted
        # values differed from `mboost` by 1.1e-3 while an all-centered or
        # all-block menu agreed to 3e-15, and the selection sequence was
        # identical throughout. `bols_form="block"` is how the caller says a
        # penalized term is present, so every `bols` in the menu takes the
        # block form together.
        needs_block = (
            bols_form == "block"
            or learner.by is not None
            or learner.df is not None
            or learner.lambda_value is not None
        )
        spec.update(
            center=bool(learner.intercept) and not needs_block,
            intercept=bool(learner.intercept) and needs_block,
            df=None if learner.df is None else float(learner.df),
            bins=learner.bins,
        )
    elif isinstance(learner, Bbs) and learner.constraint is not None:
        spec.update(
            knots=int(learner.knots),
            degree=int(learner.degree),
            differences=int(learner.differences),
            df=float(learner.df),
            constraint=str(learner.constraint),
            center=bool(learner.center),
            # `Bbs` exposes no solver choice, so use the exact one: the
            # approximate solver leaves a same-signed violation per step that
            # boosting accumulates.
            type="quad.prog",
            lambda2=1e6,
            niter=10,
        )
    elif isinstance(learner, Bbs) and kronecker_n_resp is not None:
        # Multinomial bbs: lower to the Kronecker-wrapped recipe
        # (`prepare_kronecker_bbs`, raw basis + I+K1 penalty) the array
        # boundary built directly. The spec is tagged "kronecker_bbs" so the
        # core folds it into `KroneckerBols` with x2 = I_{K-1}.
        spec["kind"] = "kronecker_bbs"
        spec["x"] = np.ascontiguousarray(
            _extract_column(data, learner.name), dtype=np.float64,
        )
        spec.update(
            knots=int(learner.knots),
            degree=int(learner.degree),
            differences=int(learner.differences),
            df=float(learner.df),
            deriv=int(learner.deriv),
            n_resp=int(kronecker_n_resp),
            center=bool(learner.center),
        )
    elif isinstance(learner, Bbs):
        spec.update(
            knots=int(learner.knots),
            degree=int(learner.degree),
            differences=int(learner.differences),
            df=float(learner.df),
            center=bool(learner.center),
            cyclic=bool(learner.cyclic),
            boundary_knots=(
                None
                if learner.boundary_knots is None
                else tuple(float(value) for value in learner.boundary_knots)
            ),
            deriv=int(learner.deriv),
            bins=learner.bins,
            knot_placement=getattr(learner, "knot_placement", "equidistant"),
        )
    elif isinstance(learner, Bns):
        x_values = np.asarray(_extract_column(data, learner.name), dtype=np.float64)
        spec.update(
            knots=int(learner.knots),
            degree=int(learner.degree),
            df=None if learner.df is None else float(learner.df),
            center=bool(learner.center),
            boundary_knots=(float(x_values.min()), float(x_values.max())),
        )
    elif isinstance(learner, Brandom):
        spec.update(
            df=float(learner.df),
            center=bool(learner.center),
            bins=learner.bins,
        )
    elif isinstance(learner, Bmrf):
        if learner.K is None:
            raise ValueError(
                f"Bmrf learner for {learner.name!r} requires a neighborhood "
                "penalty matrix K; pass it through env, for example "
                "env={'K_regions': K} with bmrf(region, K=K_regions)"
            )
        spec.update(
            K=np.ascontiguousarray(learner.K, dtype=np.float64),
            df=float(learner.df),
            center=bool(learner.center),
        )
    elif isinstance(learner, Bspatial):
        x1_values = np.asarray(
            _extract_column(data, learner.name1), dtype=np.float64
        )
        x2_values = np.asarray(
            _extract_column(data, learner.name2), dtype=np.float64
        )
        spec.update(
            x2=np.ascontiguousarray(x2_values, dtype=np.float64),
            knots=int(learner.knots),
            degree=int(learner.degree),
            differences=int(learner.differences),
            df=float(learner.df),
            center=bool(learner.center),
            bins=learner.bins,
            boundary_knots1=(float(x1_values.min()), float(x1_values.max())),
            boundary_knots2=(float(x2_values.min()), float(x2_values.max())),
        )
    elif isinstance(learner, Bmono):
        spec.update(
            knots=int(learner.knots),
            degree=int(learner.degree),
            differences=int(learner.differences),
            df=float(learner.df),
            constraint=str(learner.constraint),
            center=bool(learner.center),
            type=str(learner.type),
            lambda2=float(learner.lambda2),
            niter=int(learner.niter),
        )
    elif isinstance(learner, Bconstrained):
        if learner.C is None:
            raise ValueError(
                f"Bconstrained learner for {learner.name!r} requires a constraint "
                "matrix C; pass it through env, for example env={'C_shape': C} "
                "with bconstrained(x, C=C_shape)"
            )
        spec.update(
            C=np.ascontiguousarray(learner.C, dtype=np.float64),
            knots=int(learner.knots),
            degree=int(learner.degree),
            differences=int(learner.differences),
            df=float(learner.df),
            type=str(getattr(learner, "type", "quad.prog")),
            lambda2=float(learner.lambda2),
            niter=int(learner.niter),
        )
    elif isinstance(learner, Btree):
        spec.update(
            n_features=len(learner.names),
            minbucket=int(learner.minbucket),
            minsplit=int(learner.minsplit),
            maxdepth=int(learner.maxdepth),
        )
        categorical = list(getattr(learner, "categorical", None) or ())
        if any(categorical):
            spec["categorical"] = [int(v) for v in categorical]
        spec["selection"] = getattr(learner, "selection", "ctree")
        spec["split_method"] = getattr(learner, "split_method", "exact")
        spec["lambda_leaf"] = float(getattr(learner, "lambda_leaf", 0.0))
        spec["colsample"] = float(getattr(learner, "colsample", 1.0))
        spec["colsample_seed"] = int(getattr(learner, "colsample_seed", 0))
    elif isinstance(learner, Bkernel):
        if learner.landmarks is not None:
            spec["kind"] = "bkernel_nystrom"
            spec.update(
                bandwidth=float(learner.bandwidth),
                df=float(learner.df),
                kernel=str(learner.kernel),
                landmarks=int(learner.landmarks),
                center=bool(learner.center),
            )
        else:
            spec.update(
                bandwidth=float(learner.bandwidth),
                df=float(learner.df),
                kernel=str(learner.kernel),
                center=bool(learner.center),
            )
    elif isinstance(learner, Hinge):
        spec.update(
            df=None if learner.df is None else float(learner.df),
            n_candidates=int(learner.n_candidates),
            center=bool(learner.center),
        )
    else:
        spec.update(
            K=np.ascontiguousarray(learner.K, dtype=np.float64),
            n_features=int(learner.X.shape[1]),
            df=float(learner.df),
            center=bool(learner.center),
        )
    return spec


def _apply_lss_global_bins(per_param_learners, bins):
    """Apply a frontend ``bins=`` to every parameter's learner list.

    An LSS fit builds the candidate set once per distribution parameter, so a
    two-parameter fit holds two copies of every spline design and a
    three-parameter Student-t fit holds three. Binning therefore saves more
    here than on a `gamboost` fit of the same formula. The per-parameter lists
    come from the same formula whenever one formula is reused, so the skipped
    terms are reported once rather than once per parameter.
    """
    if bins is None:
        return per_param_learners
    from mboostrs.learners import apply_global_bins, warn_skipped_bins

    binned, skipped, seen = [], [], set()
    for learners in per_param_learners:
        compact, param_skipped = apply_global_bins(learners, bins)
        binned.append(compact)
        for entry in param_skipped:
            if entry not in seen:
                seen.add(entry)
                skipped.append(entry)
    warn_skipped_bins(skipped, stacklevel=4)
    return binned


def _gamboostlss_learner_specs(per_param_learners, data):
    """Normalize supported learners for the generic Rust LSS adapter.

    The nested list retains formula order separately for each distribution
    parameter. Rust selection indices refer to this order even when different
    prepared learner variants compete in the same parameter predictor.
    """
    per_param_specs = []
    for learners in per_param_learners:
        parameter_specs = []
        for learner in learners:
            parameter_specs.append(learner_spec(learner, data))
        per_param_specs.append(parameter_specs)
    return per_param_specs


def _resolve_buser_design(learner, buser_newdata, *, require_external, data=None):
    if (learner._radial_basis is not None and data is not None
            and (buser_newdata is None or learner.name not in buser_newdata)):
        from mboostrs.learners import _radial_design

        return _radial_design(data, learner._radial_basis)
    if buser_newdata is None:
        if require_external:
            raise ValueError(
                f"new-data prediction for Buser learner {learner.name!r} requires "
                f"buser_newdata={{'{learner.name}': X_new}}"
            )
        matrix = learner.X
    else:
        try:
            matrix = buser_newdata[learner.name]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"buser_newdata is missing an explicit design matrix for {learner.name!r}"
            ) from exc
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != learner.X.shape[1]:
        raise ValueError(
            f"Buser design for {learner.name!r} must have shape (n, {learner.X.shape[1]}); "
            f"got {matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"Buser design for {learner.name!r} contains NaN or infinite values")
    return np.ascontiguousarray(matrix, dtype=np.float64)


def _gamboostlss_x_data(
    data,
    learner,
    *,
    buser_newdata=None,
    require_external_buser=False,
):
    """Raw replay values for one structured LSS learner.

    Bspatial packs its two coordinate vectors consecutively so the common Rust
    prediction store can retain one raw vector per formula-level learner.
    """
    if isinstance(learner, Brad):
        learner = learner.as_buser()
    if isinstance(learner, Bspatial):
        x1 = np.asarray(_extract_column(data, learner.name1), dtype=np.float64)
        x2 = np.asarray(_extract_column(data, learner.name2), dtype=np.float64)
        if len(x1) != len(x2):
            raise ValueError(
                f"Bspatial coordinates have different lengths: {len(x1)} and {len(x2)}"
            )
        return np.ascontiguousarray(np.concatenate((x1, x2)), dtype=np.float64)
    if isinstance(learner, Btree):
        return np.ascontiguousarray(
            np.column_stack([_extract_column(data, name) for name in learner.names]).ravel(),
            dtype=np.float64,
        )
    if isinstance(learner, Buser):
        matrix = _resolve_buser_design(
            learner,
            buser_newdata,
            require_external=require_external_buser,
            data=data,
        )
        return np.ascontiguousarray(matrix.ravel(), dtype=np.float64)
    return np.ascontiguousarray(_extract_column(data, learner.name), dtype=np.float64)


def _gamboostlss_by_data(source, learners):
    """Fresh newdata by-values per learner (None where by= is unused).

    Mirrors :func:`_gamboostlss_x_data`'s column extraction so the Rust
    prediction store can override a learner's training-length ``by`` values
    with the ones observed in ``source`` (newdata) rather than replaying the
    training-length values positionally.
    """
    out = []
    for learner in learners:
        by = getattr(learner, "by", None)
        out.append(
            _extract_column(source, by).astype(np.float64).tolist()
            if by is not None
            else None
        )
    return out


def _gamboostlss_effect_coordinates(
    data,
    learner,
    *,
    buser_newdata=None,
    require_external_buser=False,
):
    if isinstance(learner, Brad):
        learner = learner.as_buser()
    if isinstance(learner, Bspatial):
        return np.column_stack(
            (
                np.asarray(_extract_column(data, learner.name1), dtype=np.float64),
                np.asarray(_extract_column(data, learner.name2), dtype=np.float64),
            )
        )
    if isinstance(learner, Btree):
        # A one-covariate tree reports its covariate as a vector, as gamboost's
        # partial_effects does and as every other single-covariate learner here
        # does; an `(n, 1)` column broke the row masks llm_explain_prompt builds.
        if len(learner.names) == 1:
            return np.ascontiguousarray(
                _extract_column(data, learner.names[0]), dtype=np.float64
            )
        return np.ascontiguousarray(
            np.column_stack([_extract_column(data, name) for name in learner.names]),
            dtype=np.float64,
        )
    if isinstance(learner, Buser):
        matrix = _resolve_buser_design(
            learner,
            buser_newdata,
            require_external=require_external_buser,
            data=data,
        )
        return np.arange(matrix.shape[0], dtype=np.float64)
    return np.ascontiguousarray(_extract_column(data, learner.name), dtype=np.float64)


def _poisson_cdf(q, mu) -> np.ndarray:
    """Poisson CDF ``P(X <= q)``, elementwise in the core (R's ``ppois``)."""
    return _special("poisson_cdf", q, mu)


def _nbinom_cdf(q, mu, size) -> np.ndarray:
    """Negative binomial CDF with mean ``mu`` and size ``size`` (R's
    ``pnbinom(q, size = size, mu = mu)``), elementwise in the core."""
    return _special("nbinom_cdf", q, mu, size)


def _check_stabilization(value):
    """Validate a gradient-stabilization setting, returning the canonical spelling.

    ``"none"``, ``"MAD"`` and ``"L2"`` are the three values upstream
    ``gamboostLSS`` accepts (``check_stabilization``), and the same three are
    accepted here at both levels that can set them. Casing is normalized
    (``"mad"`` -> ``"MAD"``), matching the Rust parser's behaviour and the
    lowercase spelling R's deprecation message shows.
    """
    normalized = str(value).upper() if str(value).upper() != "NONE" else "none"
    if normalized not in ("none", "MAD", "L2"):
        raise ValueError(
            f'stabilization must be "none", "MAD" or "L2", got {value!r}')
    return normalized


class _StabilizedNGradient:
    """Family-level gradient stabilization, upstream's spelling.

    gamboostLSS 2.2-0 deprecated its global ``options(gamboostLSS_stab_ngrad)``
    in favour of the family constructor — ``GaussianLSS(stabilization = "MAD")``
    threads the setting into every parameter's partial gradient. Every LSS
    family here takes the same argument with the same three values and the
    same default, and the fitting functions use it unless their own
    ``stabilization=`` argument is passed explicitly (see :func:`glmboostLSS`
    for the precedence rule).
    """

    #: Gradient stabilization applied by the fit: "none" (default), "MAD", "L2".
    stabilization: str = "none"

    def __init__(self, stabilization: str = "none"):
        self.stabilization = _check_stabilization(stabilization)

    def __repr__(self):
        """``GaussianLSS()``: the class and its set arguments.

        The same shape the mean families print (`GammaReg()`), so reports
        and recorded probes read `BetaLSS()` instead of a memory address.
        The default stabilization is left out; a set one is printed.
        """
        args = ", ".join(
            f"{key}={value!r}" for key, value in vars(self).items()
            if not key.startswith("_") and value is not None
            and not (key == "stabilization" and value == "none"))
        return f"{type(self).__name__}({args})"


def _check_response_scale(value):
    """A valid ``response_scale`` in its stored form: ``"auto"``, ``"none"``
    (also spelled ``None``) or a positive number.

    Boosted LSS fitting is not unit-free: once gradients are stabilized each
    step moves a location parameter by about ``nu`` in response units, so a
    response in dollars barely leaves its offset while the same response in
    thousands of dollars fits. A scaled fit divides the response by ``s`` and
    maps the model back, so it is the same in any units and reports
    everything in the original units. ``"auto"`` takes ``s`` as the weighted
    SD of the response each fit sees -- a cross-validation fold, a weighted
    refit and a fit on the same rows agree -- and leaves families with no
    units to change (counts, proportions, compositions) unscaled. ``None``
    fits the raw response, the path R gamboostLSS takes; it is stored as
    ``"none"`` so that a refit that drops unset arguments keeps it.
    """
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


def _response_scale_used(value, family, y, w):
    """The scale a fit with ``response_scale=value`` divided ``y`` by, or ``None``."""
    if value is None or value == "none":
        return None
    if isinstance(value, str):
        from mboostrs import _rust

        return _rust.lss_auto_response_scale(
            family._rust_name, int(family.n_params),
            np.ascontiguousarray(y, dtype=np.float64),
            np.ascontiguousarray(w, dtype=np.float64))
    return float(value)


def _resolve_stabilization(fit_level, family):
    """The stabilization the fit runs with.

    The family's own setting is the default; a fit-level ``stabilization=``
    argument overrides it, which is an mboostrs extension — upstream sets the
    option on the family constructor only. The fit-level default is therefore
    ``None`` meaning "use the family's", not ``"none"``: a ``"none"`` default
    would silently ignore a family constructed with ``stabilization="MAD"``.
    An explicit fit-level ``"none"`` does override the family, so both levels
    stay controllable.
    """
    if fit_level is None:
        return _check_stabilization(getattr(family, "stabilization", "none"))
    return _check_stabilization(fit_level)


def _resolve_nu_per_param(nu, family):
    """Resolve a per-parameter learning-rate request into family order.

    Upstream ``gamboostLSS`` takes the vector in ``boost_control(nu = ...)``:
    ``mboostLSS_fit`` runs it through ``check()``, which recycles a scalar,
    takes an unnamed length-``k`` vector in family order, or a named vector
    reordered to family order. This resolver follows the same contract:
    ``None`` keeps the control's scalar for every parameter, a float sets the
    same rate for all, a list is family order, and a ``{param_name: float}``
    dict must name every parameter. Range validation (``0 < nu <= 1``) happens
    in the engine, which is the one place the check cannot be bypassed.
    """
    if nu is None:
        return None
    names = list(family.param_names)
    if isinstance(nu, dict):
        missing = [name for name in names if name not in nu]
        if missing:
            raise ValueError(
                f"nu dict is missing {missing}; expected one entry per "
                f"parameter: {names}"
            )
        return [float(nu[name]) for name in names]
    if isinstance(nu, (list, tuple, np.ndarray)):
        values = [float(v) for v in nu]
        if len(values) != len(names):
            raise ValueError(
                f"nu must have one entry per parameter; expected "
                f"{len(names)}, got {len(values)}"
            )
        return values
    return [float(nu)] * len(names)


class _DensityDeviance:
    """The deviance contribution implied by a family's own density.

    `-2 * log(pdf(y))` is what every family that defines this computes; it was
    copied into each of them. Families that need something else — the
    zero-inflated ones, whose deviance is defined against a saturated model —
    override it.
    """

    def deviance_increment(self, y, params) -> np.ndarray:
        """Per-observation deviance contribution, ``-2 * log(pdf(y))``."""
        density = self.pdf(y, params)
        return -2.0 * np.log(np.maximum(density, np.finfo(np.float64).tiny))


class GaussianLSS(_StabilizedNGradient):
    """Normal location-scale family: ``mu`` (identity link), ``sigma`` (log link).

    Matches R ``gamboostLSS::GaussianLSS()``.
    """

    n_params = 2
    param_names = ("mu", "sigma")
    _rust_name = "gaussianlss"

    def response(self, k: int, eta: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=np.float64)
        if k == 0:  # mu: identity link
            return eta.copy()
        if k == 1:  # sigma: log link
            return np.exp(eta)
        raise IndexError(f"GaussianLSS has 2 parameters, got k={k}")

    def mean(self, params) -> np.ndarray:
        """Conditional response mean."""
        return _param_array(params, "mu")

    def pdf(self, y, params) -> np.ndarray:
        """Normal density at ``y`` for response-scale ``mu`` and ``sigma``."""
        y = np.asarray(y, dtype=np.float64)
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        if np.any(sigma <= 0):
            raise ValueError("GaussianLSS sigma must be positive")
        z = (y - mu) / sigma
        return np.exp(-0.5 * z * z) / (sigma * _SQRT_2PI)

    def cdf(self, y, params) -> np.ndarray:
        """Normal CDF at ``y`` for response-scale ``mu`` and ``sigma``."""
        y = np.asarray(y, dtype=np.float64)
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        if np.any(sigma <= 0):
            raise ValueError("GaussianLSS sigma must be positive")
        return _normal_cdf_array((y - mu) / sigma)

    def quantile(self, p, params) -> np.ndarray:
        """Normal quantile for probabilities ``p``."""
        return _core_quantile("GaussianLSS", p, params, ("mu", "sigma"))

    def simulate(self, params, rng=None, size=None) -> np.ndarray:
        """Draw Normal samples for response-scale ``mu`` and ``sigma``."""
        rng = np.random.default_rng() if rng is None else rng
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        if np.any(sigma <= 0):
            raise ValueError("GaussianLSS sigma must be positive")
        return rng.normal(loc=mu, scale=sigma, size=size)

    def deviance_increment(self, y, params) -> np.ndarray:
        """Per-observation deviance contribution, ``-2 * log(pdf(y))``."""
        dens = self.pdf(y, params)
        return -2.0 * np.log(np.maximum(dens, np.finfo(np.float64).tiny))


class GaussianKnownVarLSS(_StabilizedNGradient):
    """Normal location-scale family for outcomes that carry a known sampling
    variance: ``v_i = s2_i + exp(eta_sigma_i)``, the known sampling variance
    plus a boosted *excess* variance component.

    Made for regressing published small-area estimates (life expectancy,
    county means), whose standard errors vary by orders of magnitude across
    areas: a plain :class:`GaussianLSS` fit mistakes that sampling noise for
    heterogeneity and its sigma-submodel selects proxies for sample size.

    The response is a two-column ``cbind`` tuple — pass the formulas as
    ``"cbind(le, s2) ~ ..."`` with ``s2`` the squared standard error —
    mirroring how survival families take ``Surv(time, status)``.
    Both submodels are boosted: ``mu`` carries its own additive predictor
    as in :class:`GaussianLSS`, and ``eta_sigma`` models only the
    *unknown excess* variance on top of the known ``s2``. The gradient
    toward ``eta_sigma`` is attenuated by ``exp(eta_sigma) / v`` wherever
    the known component dominates, which is the correction. The
    natural-scale ``sigma`` parameter is the excess standard deviation
    ``exp(eta_sigma / 2)``; the total standard deviation of a new
    observation is ``sqrt(s2 + exp(eta_sigma))`` and is what
    :meth:`pdf`, :meth:`cdf`, and :meth:`quantile` compute when given ``s2``.

    This family is available in both the Python and R frontends as
    ``GaussianKnownVarLSS()``. It has no upstream ``gamboostLSS``
    counterpart: it is an mboostrs addition, so there is no upstream
    implementation to match or to check results against.
    """

    n_params = 2
    param_names = ("mu", "sigma")
    _rust_name = "gaussianknownvarlss"
    _known_variance = True

    def response(self, k: int, eta: np.ndarray) -> np.ndarray:
        """Link-scale to natural-scale: ``mu`` identity, ``sigma`` the excess
        standard deviation ``exp(eta / 2)``."""
        eta = np.asarray(eta, dtype=np.float64)
        if k == 0:
            return eta.copy()
        if k == 1:
            return np.exp(0.5 * eta)
        raise IndexError(f"GaussianKnownVarLSS has 2 parameters, got k={k}")

    def mean(self, params) -> np.ndarray:
        """Conditional response mean; known and excess variance do not shift it."""
        return _param_array(params, "mu")

    def total_sd(self, params, s2) -> np.ndarray:
        """Total standard deviation ``sqrt(s2 + excess variance)``."""
        s2 = np.asarray(s2, dtype=np.float64)
        excess = _param_array(params, "sigma") ** 2
        return np.sqrt(s2 + excess)

    def pdf(self, y, params, s2) -> np.ndarray:
        """Normal density at ``y`` under the total variance."""
        y = np.asarray(y, dtype=np.float64)
        sd = self.total_sd(params, s2)
        z = (y - _param_array(params, "mu")) / sd
        return np.exp(-0.5 * z * z) / (sd * _SQRT_2PI)

    def cdf(self, y, params, s2) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        sd = self.total_sd(params, s2)
        return _normal_cdf_array((y - _param_array(params, "mu")) / sd)

    def quantile(self, p, params, s2) -> np.ndarray:
        return _core_quantile("GaussianKnownVarLSS", p, params, ("mu", "sigma"), extra=(s2,))


class GammaLSS(_StabilizedNGradient):
    """Gamma location-shape family: ``mu`` (log link, mean), ``sigma`` (log link,
    shape). Matches R ``gamboostLSS::GammaLSS()``. Requires ``y > 0``.
    """

    n_params = 2
    param_names = ("mu", "sigma")
    _rust_name = "gammalss"

    def response(self, k: int, eta: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=np.float64)
        if k in (0, 1):  # mu and sigma both use the log link
            return np.exp(eta)
        raise IndexError(f"GammaLSS has 2 parameters, got k={k}")

    def mean(self, params) -> np.ndarray:
        """Conditional response mean."""
        return _param_array(params, "mu")

    def pdf(self, y, params) -> np.ndarray:
        """Gamma density with response-scale ``mu`` (mean) and ``sigma`` (shape)."""
        y = np.asarray(y, dtype=np.float64)
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        if np.any(y <= 0) or np.any(mu <= 0) or np.any(sigma <= 0):
            raise ValueError("GammaLSS requires y, mu, and sigma to be positive")
        scale = mu / sigma
        logpdf = (sigma - 1.0) * np.log(y) - y / scale - _LGAMMA(sigma) - sigma * np.log(scale)
        return np.exp(logpdf)

    def cdf(self, y, params) -> np.ndarray:
        """Gamma CDF with shape ``sigma`` and scale ``mu / sigma``."""
        y = np.asarray(y, dtype=np.float64)
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        if np.any(y < 0) or np.any(mu <= 0) or np.any(sigma <= 0):
            raise ValueError("GammaLSS requires y >= 0 and mu, sigma > 0")
        return _special("gamma_p", sigma, y * sigma / mu)

    def quantile(self, p, params) -> np.ndarray:
        """Gamma quantile, obtained by inverting :meth:`cdf`."""
        return _core_quantile("GammaLSS", p, params, ("mu", "sigma"))

    def simulate(self, params, rng=None, size=None) -> np.ndarray:
        """Draw Gamma samples with response-scale ``mu`` and ``sigma``."""
        rng = np.random.default_rng() if rng is None else rng
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        if np.any(mu <= 0) or np.any(sigma <= 0):
            raise ValueError("GammaLSS requires mu and sigma to be positive")
        return rng.gamma(shape=sigma, scale=mu / sigma, size=size)

    def deviance_increment(self, y, params) -> np.ndarray:
        dens = self.pdf(y, params)
        return -2.0 * np.log(np.maximum(dens, np.finfo(np.float64).tiny))


class StudentTLSS(_StabilizedNGradient):
    """Student's t family: ``mu`` (identity link), ``sigma`` (log link, scale),
    ``df`` (log link, degrees of freedom). Matches R ``gamboostLSS::StudentTLSS()``.
    """

    n_params = 3
    param_names = ("mu", "sigma", "df")
    _rust_name = "studenttlss"

    def response(self, k: int, eta: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=np.float64)
        if k == 0:  # mu: identity
            return eta.copy()
        if k in (1, 2):  # sigma, df: log link
            return np.exp(eta)
        raise IndexError(f"StudentTLSS has 3 parameters, got k={k}")

    def mean(self, params) -> np.ndarray:
        """Conditional response mean, defined only for df > 1."""
        df = _param_array(params, "df")
        if np.any(df <= 1):
            raise ValueError("StudentTLSS has no finite mean where df <= 1")
        return _param_array(params, "mu")

    def pdf(self, y, params) -> np.ndarray:
        """Student-t density: location ``mu``, scale ``sigma``, df ``df``."""
        from mboostrs import _rust

        y = np.asarray(y, dtype=np.float64)
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        df = _param_array(params, "df")
        columns = np.broadcast_arrays(y, mu, sigma, df)
        values = _rust.student_t_pdf(*(c.ravel().tolist() for c in columns))
        return np.asarray(values).reshape(columns[0].shape)

    def cdf(self, y, params) -> np.ndarray:
        """Student-t CDF for location ``mu``, scale ``sigma``, and ``df``."""
        y = np.asarray(y, dtype=np.float64)
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        df = _param_array(params, "df")
        if np.any(sigma <= 0) or np.any(df <= 0):
            raise ValueError("StudentTLSS requires sigma > 0 and df > 0")
        return _special("t_cdf", (y - mu) / sigma, df)

    def quantile(self, p, params) -> np.ndarray:
        """Student-t quantile, obtained by inverting the standardized CDF."""
        return _core_quantile("StudentTLSS", p, params, ("mu", "sigma", "df"))

    def simulate(self, params, rng=None, size=None) -> np.ndarray:
        """Draw Student-t samples: mu + sigma * standard_t(df)."""
        rng = np.random.default_rng() if rng is None else rng
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        df = _param_array(params, "df")
        if np.any(sigma <= 0) or np.any(df <= 0):
            raise ValueError("StudentTLSS requires sigma > 0 and df > 0")
        return mu + sigma * rng.standard_t(df=df, size=size)

    def deviance_increment(self, y, params) -> np.ndarray:
        dens = self.pdf(y, params)
        return -2.0 * np.log(np.maximum(dens, np.finfo(np.float64).tiny))


class BetaLSS(_StabilizedNGradient):
    """Beta location-precision family: ``mu`` (logit link, mean in (0,1)),
    ``phi`` (log link, precision). Matches R ``gamboostLSS::BetaLSS()``. Requires
    ``0 < y < 1``.
    """

    n_params = 2
    param_names = ("mu", "phi")
    _rust_name = "betalss"

    def response(self, k: int, eta: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=np.float64)
        if k == 0:  # mu: logit link
            return 1.0 / (1.0 + np.exp(-eta))
        if k == 1:  # phi: log link
            return np.exp(eta)
        raise IndexError(f"BetaLSS has 2 parameters, got k={k}")

    def mean(self, params) -> np.ndarray:
        """Conditional response mean."""
        return _param_array(params, "mu")

    def pdf(self, y, params) -> np.ndarray:
        """Beta density with response-scale ``mu`` and precision ``phi``."""
        y = np.asarray(y, dtype=np.float64)
        mu = _param_array(params, "mu")
        phi = _param_array(params, "phi")
        if np.any((y <= 0) | (y >= 1)) or np.any((mu <= 0) | (mu >= 1)) or np.any(phi <= 0):
            raise ValueError("BetaLSS requires 0 < y, mu < 1 and phi > 0")
        alpha = mu * phi
        beta = (1.0 - mu) * phi
        logpdf = (
            (alpha - 1.0) * np.log(y)
            + (beta - 1.0) * np.log1p(-y)
            + _LGAMMA(alpha + beta)
            - _LGAMMA(alpha)
            - _LGAMMA(beta)
        )
        return np.exp(logpdf)

    def cdf(self, y, params) -> np.ndarray:
        """Beta CDF with shapes ``mu * phi`` and ``(1 - mu) * phi``."""
        y = np.asarray(y, dtype=np.float64)
        mu = _param_array(params, "mu")
        phi = _param_array(params, "phi")
        if np.any((y < 0) | (y > 1)) or np.any((mu <= 0) | (mu >= 1)) or np.any(phi <= 0):
            raise ValueError("BetaLSS requires 0 <= y <= 1, 0 < mu < 1, and phi > 0")
        return _special("beta_i", mu * phi, (1.0 - mu) * phi, y)

    def quantile(self, p, params) -> np.ndarray:
        """Beta quantile, obtained by inverting :meth:`cdf` on (0, 1)."""
        return _core_quantile("BetaLSS", p, params, ("mu", "phi"))

    def simulate(self, params, rng=None, size=None) -> np.ndarray:
        """Draw Beta samples with response-scale ``mu`` and precision ``phi``."""
        rng = np.random.default_rng() if rng is None else rng
        mu = _param_array(params, "mu")
        phi = _param_array(params, "phi")
        if np.any((mu <= 0) | (mu >= 1)) or np.any(phi <= 0):
            raise ValueError("BetaLSS requires 0 < mu < 1 and phi > 0")
        return rng.beta(a=mu * phi, b=(1.0 - mu) * phi, size=size)

    def deviance_increment(self, y, params) -> np.ndarray:
        dens = self.pdf(y, params)
        return -2.0 * np.log(np.maximum(dens, np.finfo(np.float64).tiny))


class NBinomialLSS(_StabilizedNGradient):
    """Negative-binomial family: ``mu`` (log link, mean), ``sigma`` (log link,
    size/dispersion). Matches R ``gamboostLSS::NBinomialLSS()``. Requires
    non-negative integer counts.
    """

    n_params = 2
    param_names = ("mu", "sigma")
    _rust_name = "nbinomiallss"
    _discrete = True

    def response(self, k: int, eta: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=np.float64)
        if k in (0, 1):  # mu and sigma both use the log link
            return np.exp(eta)
        raise IndexError(f"NBinomialLSS has 2 parameters, got k={k}")

    def mean(self, params) -> np.ndarray:
        """Conditional count mean ``mu``."""
        return _param_array(params, "mu")

    def pmf(self, y, params) -> np.ndarray:
        """Negative-binomial mass with response-scale mean ``mu`` and size ``sigma``."""
        y = np.asarray(y, dtype=np.float64)
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        if np.any(y < 0) or np.any(y != np.floor(y)):
            raise ValueError("NBinomialLSS requires non-negative integer y")
        if np.any(mu <= 0) or np.any(sigma <= 0):
            raise ValueError("NBinomialLSS requires mu and sigma to be positive")
        log_p = np.log(sigma) - np.log(sigma + mu)
        log_1mp = np.log(mu) - np.log(sigma + mu)
        logpmf = _LGAMMA(y + sigma) - _LGAMMA(sigma) - _LGAMMA(y + 1.0) + sigma * log_p + y * log_1mp
        return np.exp(logpmf)

    def cdf(self, y, params) -> np.ndarray:
        """Negative-binomial CDF with response-scale mean ``mu`` and size ``sigma``."""
        y = np.asarray(y, dtype=np.float64)
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        if np.any(mu <= 0) or np.any(sigma <= 0):
            raise ValueError("NBinomialLSS requires mu and sigma to be positive")
        return _nbinom_cdf(y, mu, sigma)

    def quantile(self, p, params) -> np.ndarray:
        """Negative-binomial quantile for probabilities ``p``."""
        return _core_quantile("NBinomialLSS", p, params, ("mu", "sigma"))

    def simulate(self, params, rng=None, size=None) -> np.ndarray:
        """Draw negative-binomial samples with response-scale mean and size."""
        rng = np.random.default_rng() if rng is None else rng
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        if np.any(mu <= 0) or np.any(sigma <= 0):
            raise ValueError("NBinomialLSS requires mu and sigma to be positive")
        return rng.negative_binomial(n=sigma, p=sigma / (sigma + mu), size=size)

    def deviance_increment(self, y, params) -> np.ndarray:
        mass = self.pmf(y, params)
        return -2.0 * np.log(np.maximum(mass, np.finfo(np.float64).tiny))

    def pdf(self, y, params) -> np.ndarray:
        """Negative-binomial probability mass with ``sigma`` naming the size,
        so the variance is ``mu + mu**2 / sigma``.

        Matches ``gamlss.dist::dNBI(y, mu, 1 / sigma)``. See
        :class:`NegativeBinomialLSS`, which is the same distribution with
        ``sigma`` naming gamlss' dispersion instead.
        """
        y = np.asarray(y, dtype=np.float64)
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        if np.any(mu <= 0) or np.any(sigma <= 0):
            raise ValueError("NBinomialLSS requires mu and sigma positive")
        y, mu, sigma = np.broadcast_arrays(y, mu, sigma)
        support = _on_count_support(y)
        safe = np.where(support, y, 0.0)
        log_mass = (
            _LGAMMA(safe + sigma) - _LGAMMA(sigma) - _LGAMMA(safe + 1.0)
            + sigma * np.log(sigma / (sigma + mu))
            + safe * np.log(mu / (sigma + mu))
        )
        return np.where(support, np.exp(log_mass), 0.0)


class ZIPoLSS(_StabilizedNGradient):
    """Zero-inflated Poisson family: ``mu`` (log link, Poisson mean), ``sigma``
    (logit link, zero-inflation probability). Matches R ``gamboostLSS::ZIPoLSS()``
    (wrapping gamlss.dist ``ZIP``). Requires non-negative integer counts.
    """

    n_params = 2
    param_names = ("mu", "sigma")
    _rust_name = "zipolss"
    _discrete = True

    _EPS = 2.220446049250313e-16
    _THRESH = 36.04365338911715  # -qlogis(eps)

    def response(self, k: int, eta: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=np.float64)
        if k == 0:  # mu: log link
            return np.maximum(np.exp(eta), self._EPS)
        if k == 1:  # sigma: logit link (clamped, matching gamlss.dist)
            clipped = np.clip(eta, -self._THRESH, self._THRESH)
            return 1.0 / (1.0 + np.exp(-clipped))
        raise IndexError(f"ZIPoLSS has 2 parameters, got k={k}")

    def mean(self, params) -> np.ndarray:
        """Mixture mean ``(1 - sigma) * mu``."""
        return (1 - _param_array(params, "sigma")) * _param_array(params, "mu")

    def pmf(self, y, params) -> np.ndarray:
        """ZIP mass: P(y=0) = sigma + (1-sigma)*exp(-mu);
        P(y>0) = (1-sigma)*Poisson(y; mu)."""
        y = np.asarray(y, dtype=np.float64)
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        if np.any(y < 0) or np.any(y != np.floor(y)):
            raise ValueError("ZIPoLSS requires non-negative integer y")
        if np.any(mu <= 0) or np.any((sigma < 0) | (sigma > 1)):
            raise ValueError("ZIPoLSS requires mu > 0 and 0 <= sigma <= 1")
        pois_0 = np.exp(-mu)
        pois_y = np.where(y == 0, 1.0, np.exp(y * np.log(mu) - mu - _LGAMMA(y + 1.0)))
        return np.where(y == 0, sigma + (1.0 - sigma) * pois_0, (1.0 - sigma) * pois_y)

    def cdf(self, y, params) -> np.ndarray:
        """ZIP CDF."""
        y = np.asarray(y, dtype=np.float64)
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        if np.any(mu <= 0) or np.any((sigma < 0) | (sigma > 1)):
            raise ValueError("ZIPoLSS requires mu > 0 and 0 <= sigma <= 1")
        y, mu, sigma = np.broadcast_arrays(y, mu, sigma)
        mixed = np.minimum(sigma + (1.0 - sigma) * _poisson_cdf(y, mu), 1.0)
        return np.where(y < 0, 0.0, mixed)

    def quantile(self, p, params) -> np.ndarray:
        """ZIP quantile: smallest k with CDF(k) >= p."""
        return _core_quantile("ZIPoLSS", p, params, ("mu", "sigma"))

    def simulate(self, params, rng=None, size=None) -> np.ndarray:
        """Draw ZIP samples: with probability sigma, y=0; else Poisson(mu)."""
        rng = np.random.default_rng() if rng is None else rng
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        if np.any(mu <= 0) or np.any((sigma < 0) | (sigma > 1)):
            raise ValueError("ZIPoLSS requires mu > 0 and 0 <= sigma <= 1")
        mask = rng.uniform(size=len(mu) if size is None else size) < sigma
        pois = rng.poisson(lam=mu, size=size)
        return np.where(mask, 0.0, pois)

    def deviance_increment(self, y, params) -> np.ndarray:
        mass = self.pmf(y, params)
        return -2.0 * np.log(np.maximum(mass, np.finfo(np.float64).tiny))

    def pdf(self, y, params) -> np.ndarray:
        """Zero-inflated Poisson mass: ``sigma`` is the extra-zero probability,
        so ``P(0) = sigma + (1-sigma) exp(-mu)``. Matches ``gamlss.dist::dZIP``."""
        y = np.asarray(y, dtype=np.float64)
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        if np.any(mu <= 0):
            raise ValueError("ZIPoLSS mu must be positive")
        y, mu, sigma = np.broadcast_arrays(y, mu, sigma)
        poisson = np.exp(y * np.log(mu) - mu - _LGAMMA(np.maximum(y, 0.0) + 1.0))
        mass = np.where(y < 0.5, sigma + (1.0 - sigma) * np.exp(-mu),
                        (1.0 - sigma) * poisson)
        return np.where(_on_count_support(y), mass, 0.0)


class ZINBLSS(_StabilizedNGradient):
    """Zero-inflated negative-binomial (type I) family: ``mu`` (log link, mean),
    ``sigma`` (log link, dispersion), ``nu`` (logit link, zero-inflation
    probability). Matches R ``gamboostLSS::ZINBLSS()`` (wrapping gamlss.dist
    ``ZINBI``). Requires non-negative integer counts.
    """

    n_params = 3
    param_names = ("mu", "sigma", "nu")
    _rust_name = "zinblss"
    _discrete = True

    _EPS = 2.220446049250313e-16
    _THRESH = 36.04365338911715  # -qlogis(eps)

    def response(self, k: int, eta: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=np.float64)
        if k in (0, 1):  # mu, sigma: log link
            return np.maximum(np.exp(eta), self._EPS)
        if k == 2:  # nu: logit link (clamped, matching gamlss.dist)
            clipped = np.clip(eta, -self._THRESH, self._THRESH)
            return 1.0 / (1.0 + np.exp(-clipped))
        raise IndexError(f"ZINBLSS has 3 parameters, got k={k}")

    def mean(self, params) -> np.ndarray:
        """Mixture mean ``(1 - nu) * mu``."""
        return (1 - _param_array(params, "nu")) * _param_array(params, "mu")

    def pmf(self, y, params) -> np.ndarray:
        """Return the ZINBI probability mass using the family parameterization.

        This is the same response-scale distribution used by :meth:`pdf`.
        """
        return self.pdf(y, params)

    def cdf(self, y, params) -> np.ndarray:
        """ZINB CDF."""
        y = np.asarray(y, dtype=np.float64)
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        nu = _param_array(params, "nu")
        if np.any(mu <= 0) or np.any(sigma <= 0) or np.any((nu < 0) | (nu > 1)):
            raise ValueError("ZINBLSS requires mu,sigma > 0 and 0 <= nu <= 1")
        y, mu, sigma, nu = np.broadcast_arrays(y, mu, sigma, nu)
        mixed = np.minimum(nu + (1.0 - nu) * _nbinom_cdf(y, mu, 1.0 / sigma), 1.0)
        return np.where(y < 0, 0.0, mixed)

    def quantile(self, p, params) -> np.ndarray:
        """ZINBI quantile: smallest k with nu + (1 - nu) * NBI_CDF(k) >= p (size 1/sigma)."""
        return _core_quantile("ZINBLSS", p, params, ("mu", "sigma", "nu"))

    def simulate(self, params, rng=None, size=None) -> np.ndarray:
        """Draw ZINBI samples: with probability nu, y=0; else NBI with dispersion sigma."""
        rng = np.random.default_rng() if rng is None else rng
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        nu = _param_array(params, "nu")
        if np.any(mu <= 0) or np.any(sigma <= 0) or np.any((nu < 0) | (nu > 1)):
            raise ValueError("ZINBLSS requires mu,sigma > 0 and 0 <= nu <= 1")
        mask = rng.uniform(size=len(mu) if size is None else size) < nu
        nb_size = 1.0 / sigma
        nb = rng.negative_binomial(n=nb_size, p=nb_size / (nb_size + mu), size=size)
        return np.where(mask, 0.0, nb)

    def deviance_increment(self, y, params) -> np.ndarray:
        mass = self.pmf(y, params)
        return -2.0 * np.log(np.maximum(mass, np.finfo(np.float64).tiny))

    def pdf(self, y, params) -> np.ndarray:
        """Zero-inflated negative-binomial mass.

        ``nu`` is the extra-zero probability, so ``P(0)`` mixes it with the
        negative binomial's own zero. ``sigma`` is gamlss' dispersion, as in
        :class:`NegativeBinomialLSS`, so the size is its reciprocal.
        """
        y = np.asarray(y, dtype=np.float64)
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        nu = _param_array(params, "nu")
        if np.any(mu <= 0) or np.any(sigma <= 0):
            raise ValueError("ZINBLSS requires mu and sigma positive")
        y, mu, sigma, nu = np.broadcast_arrays(y, mu, sigma, nu)
        size = 1.0 / sigma
        safe = np.maximum(y, 0.0)
        log_nb = (
            _LGAMMA(safe + size) - _LGAMMA(size) - _LGAMMA(safe + 1.0)
            + size * (np.log(size) - np.log(size + mu))
            + safe * (np.log(mu) - np.log(size + mu))
        )
        nb_zero = np.exp(
            size * (np.log(size) - np.log(size + mu))
        )
        mass = np.where(y < 0.5, nu + (1.0 - nu) * nb_zero,
                        (1.0 - nu) * np.exp(log_nb))
        return np.where(_on_count_support(y), mass, 0.0)


class PoissonLSS(_StabilizedNGradient):
    """:math:`P(\\mu)` Poisson location family: ``mu`` (log link, mean).
    Matches R ``gamboostLSS::PoissonLSS()`` (wrapping gamlss.dist ``PO``).
    Requires non-negative integer counts.
    """

    n_params = 1
    param_names = ("mu",)
    _rust_name = "poissonlss"
    _discrete = True

    _EPS = 2.220446049250313e-16

    def response(self, k: int, eta: np.ndarray) -> np.ndarray:
        return np.maximum(np.exp(np.asarray(eta, dtype=np.float64)), self._EPS)

    def mean(self, params) -> np.ndarray:
        """Conditional count mean ``mu``."""
        return _param_array(params, "mu")

    def pmf(self, y, params) -> np.ndarray:
        """Poisson mass with response-scale mean ``mu``."""
        y = np.asarray(y, dtype=np.float64)
        mu = _param_array(params, "mu")
        if np.any(y < 0) or np.any(y != np.floor(y)):
            raise ValueError("PoissonLSS requires non-negative integer y")
        if np.any(mu <= 0):
            raise ValueError("PoissonLSS requires mu to be positive")
        return np.exp(y * np.log(mu) - mu - _LGAMMA(y + 1.0))

    def cdf(self, y, params) -> np.ndarray:
        """Poisson CDF with response-scale mean ``mu``."""
        y = np.asarray(y, dtype=np.float64)
        mu = _param_array(params, "mu")
        if np.any(mu <= 0):
            raise ValueError("PoissonLSS requires mu to be positive")
        return _poisson_cdf(y, mu)

    def quantile(self, p, params) -> np.ndarray:
        """Poisson quantile for probabilities ``p``."""
        return _core_quantile("PoissonLSS", p, params, ("mu",))

    def simulate(self, params, rng=None, size=None) -> np.ndarray:
        """Draw Poisson samples with response-scale mean ``mu``."""
        rng = np.random.default_rng() if rng is None else rng
        mu = _param_array(params, "mu")
        if np.any(mu <= 0):
            raise ValueError("PoissonLSS requires mu to be positive")
        return rng.poisson(lam=mu, size=size)

    def deviance_increment(self, y, params) -> np.ndarray:
        mass = self.pmf(y, params)
        return -2.0 * np.log(np.maximum(mass, np.finfo(np.float64).tiny))

    def pdf(self, y, params) -> np.ndarray:
        """Poisson probability mass at the integer counts ``y``."""
        y = np.asarray(y, dtype=np.float64)
        mu = _param_array(params, "mu")
        if np.any(mu <= 0):
            raise ValueError("PoissonLSS mu must be positive")
        y, mu = np.broadcast_arrays(y, mu)
        support = _on_count_support(y)
        safe = np.where(support, y, 0.0)
        log_mass = safe * np.log(mu) - mu - _LGAMMA(safe + 1.0)
        return np.where(support, np.exp(log_mass), 0.0)


class GeometricLSS(_StabilizedNGradient):
    r""":math:`\text{Geom}(\pi)` geometric location family: ``mu`` (log link,
    mean). Matches R ``gamboostLSS::GeometricLSS()`` (wrapping gamlss.dist
    ``GEOM``). Requires non-negative integer counts.
    """

    n_params = 1
    param_names = ("mu",)
    _rust_name = "geometriclss"
    _discrete = True

    _EPS = 2.220446049250313e-16

    def response(self, k: int, eta: np.ndarray) -> np.ndarray:
        return np.maximum(np.exp(np.asarray(eta, dtype=np.float64)), self._EPS)

    def mean(self, params) -> np.ndarray:
        """Mean failures before the first success, ``mu``."""
        return _param_array(params, "mu")

    def pmf(self, y, params) -> np.ndarray:
        """Geometric mass with response-scale mean ``mu`` (number of failures
        before the first success)."""
        y = np.asarray(y, dtype=np.float64)
        mu = _param_array(params, "mu")
        if np.any(y < 0) or np.any(y != np.floor(y)):
            raise ValueError("GeometricLSS requires non-negative integer y")
        if np.any(mu <= 0):
            raise ValueError("GeometricLSS requires mu to be positive")
        prob = 1.0 / (1.0 + mu)  # success probability from mean
        return prob * (1.0 - prob) ** y

    def cdf(self, y, params) -> np.ndarray:
        """Geometric CDF: :math:`1 - (1-\\pi)^{\\lfloor y \\rfloor + 1}`."""
        y = np.asarray(y, dtype=np.float64)
        mu = _param_array(params, "mu")
        if np.any(mu <= 0):
            raise ValueError("GeometricLSS requires mu to be positive")
        y, mu = np.broadcast_arrays(y, mu)
        result = np.empty_like(y, dtype=np.float64)
        prob = 1.0 / (1.0 + mu)
        k = np.floor(np.maximum(y, 0.0)).astype(np.int64)
        result = 1.0 - (1.0 - prob) ** (k + 1.0)
        result = np.where(y < 0, 0.0, result)
        return np.minimum(result, 1.0)

    def quantile(self, p, params) -> np.ndarray:
        """Geometric quantile: smallest ``k`` with ``CDF(k) >= p``."""
        return _core_quantile("GeometricLSS", p, params, ("mu",))

    def simulate(self, params, rng=None, size=None) -> np.ndarray:
        """Draw geometric samples (number of failures before first success)."""
        rng = np.random.default_rng() if rng is None else rng
        mu = _param_array(params, "mu")
        if np.any(mu <= 0):
            raise ValueError("GeometricLSS requires mu to be positive")
        prob = 1.0 / (1.0 + mu)
        return rng.geometric(p=prob, size=size) - 1  # shift: 0-indexed

    def deviance_increment(self, y, params) -> np.ndarray:
        mass = self.pmf(y, params)
        return -2.0 * np.log(np.maximum(mass, np.finfo(np.float64).tiny))

    def pdf(self, y, params) -> np.ndarray:
        """Geometric probability mass. Matches ``gamlss.dist::dGEOM``, whose
        ``mu`` is the mean, so the success probability is ``1 / (1 + mu)``."""
        y = np.asarray(y, dtype=np.float64)
        mu = _param_array(params, "mu")
        if np.any(mu <= 0):
            raise ValueError("GeometricLSS mu must be positive")
        y, mu = np.broadcast_arrays(y, mu)
        # In logs: mu**y / (1+mu)**(y+1) overflows both halves well before the
        # ratio itself becomes small, giving inf/inf for a large count.
        safe = np.maximum(y, 0.0)
        log_mass = safe * np.log(mu) - (safe + 1.0) * np.log1p(mu)
        return np.where(_on_count_support(y), np.exp(log_mass), 0.0)


class BinomialLSS(_StabilizedNGradient):
    """Binomial location family: ``mu`` (logit link, probability).
    Matches R ``gamboostLSS::BinomialLSS()`` (wrapping gamlss.dist ``BI``).
    Requires ``y`` in ``[0, 1]`` (proportions).
    """

    n_params = 1
    param_names = ("mu",)
    _rust_name = "binomiallss"
    _discrete = True

    def response(self, k: int, eta: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-eta))

    def mean(self, params) -> np.ndarray:
        """Single-trial response mean, the success probability ``mu``."""
        return _param_array(params, "mu")

    def pmf(self, y, params, size=1) -> np.ndarray:
        """Binomial mass for ``size`` trials and response-scale probability ``mu``."""
        y = np.asarray(y, dtype=np.float64)
        mu = _param_array(params, "mu")
        if np.any((y < 0) | (y > size)) or np.any(y != np.floor(y)):
            raise ValueError("BinomialLSS requires integer y in [0, size]")
        if np.any((mu <= 0) | (mu >= 1)):
            raise ValueError("BinomialLSS requires 0 < mu < 1")
        n = float(size)
        logpmf = (
            _LGAMMA(n + 1.0)
            - _LGAMMA(y + 1.0)
            - _LGAMMA(n - y + 1.0)
            + y * np.log(mu)
            + (n - y) * np.log1p(-mu)
        )
        return np.exp(logpmf)

    def cdf(self, y, params, size=1) -> np.ndarray:
        """Binomial CDF with ``size`` trials (R's ``pbinom``)."""
        y = np.asarray(y, dtype=np.float64)
        mu = _param_array(params, "mu")
        if np.any((mu <= 0) | (mu >= 1)):
            raise ValueError("BinomialLSS requires 0 < mu < 1")
        return _special("binom_cdf", y, float(int(size)), mu)

    def quantile(self, p, params, size=1) -> np.ndarray:
        """Binomial quantile: smallest ``k`` with ``CDF(k) >= p``."""
        return _core_quantile("BinomialLSS", p, params, ("mu",), extra=(float(int(size)),))

    def simulate(self, params, rng=None, size=None, n_trials=1) -> np.ndarray:
        """Draw binomial samples with ``n_trials`` and response-scale ``mu``."""
        rng = np.random.default_rng() if rng is None else rng
        mu = _param_array(params, "mu")
        if np.any((mu <= 0) | (mu >= 1)):
            raise ValueError("BinomialLSS requires 0 < mu < 1")
        out_size = size if size is not None else mu.shape
        return rng.binomial(n=n_trials, p=mu, size=out_size)

    def deviance_increment(self, y, params, size=1) -> np.ndarray:
        mass = self.pmf(y, params, size=size)
        return -2.0 * np.log(np.maximum(mass, np.finfo(np.float64).tiny))

    def pdf(self, y, params) -> np.ndarray:
        """Bernoulli mass for a response in ``{0, 1}``.

        ``mu`` is the success probability, so this is ``mu`` at 1 and
        ``1 - mu`` at 0. The family models a single trial; a count of
        successes out of several needs a `Binomial` standard family instead.
        """
        y = np.asarray(y, dtype=np.float64)
        mu = _param_array(params, "mu")
        if np.any((mu <= 0) | (mu >= 1)):
            raise ValueError("BinomialLSS mu must be in (0, 1)")
        y, mu = np.broadcast_arrays(y, mu)
        # Mass at the accepted codings, 1 for a success and 0 or -1 for a
        # failure; zero at any other value.
        return np.where(y == 1, mu, np.where((y == 0) | (y == -1), 1.0 - mu, 0.0))


class NegativeBinomialLSS(_StabilizedNGradient):
    """Negative-binomial location-scale family: ``mu`` (log link, mean), ``sigma``
    (log link, dispersion), variance ``mu + sigma * mu**2``. This is
    gamlss.dist ``NBI``: ``pdf`` equals ``dNBI(y, mu, sigma)``. gamboostLSS
    has no family of this name; its ``NBinomialLSS`` (:class:`NBinomialLSS`
    here) is the same distribution with ``sigma`` naming the size
    ``1/sigma``. Requires non-negative integer counts.
    """

    n_params = 2
    param_names = ("mu", "sigma")
    _rust_name = "negativebinomiallss"
    _discrete = True

    _EPS = 2.220446049250313e-16

    def response(self, k: int, eta: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=np.float64)
        if k in (0, 1):
            return np.maximum(np.exp(eta), self._EPS)
        raise IndexError(f"NegativeBinomialLSS has 2 parameters, got k={k}")

    def mean(self, params) -> np.ndarray:
        """Conditional count mean ``mu``."""
        return _param_array(params, "mu")

    def pmf(self, y, params) -> np.ndarray:
        """NBI probability mass with mean ``mu`` and variance ``mu + sigma * mu**2``.

        The same function as :meth:`pdf`, which documents the
        parameterization. It used the size ``sigma`` instead of ``1/sigma``,
        the NBinomialLSS parameterization, so mass, CDF, quantiles and draws
        described a different distribution from the one the fit estimates.
        """
        return self.pdf(y, params)

    def cdf(self, y, params) -> np.ndarray:
        """NBI CDF with mean ``mu`` and dispersion ``sigma`` (size ``1/sigma``)."""
        y = np.asarray(y, dtype=np.float64)
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        if np.any(mu <= 0) or np.any(sigma <= 0):
            raise ValueError("NegativeBinomialLSS requires mu and sigma to be positive")
        return _nbinom_cdf(y, mu, 1.0 / sigma)

    def quantile(self, p, params) -> np.ndarray:
        """NBI quantile for probabilities ``p`` (size ``1/sigma``)."""
        return _core_quantile("NegativeBinomialLSS", p, params, ("mu", "sigma"))

    def simulate(self, params, rng=None, size=None) -> np.ndarray:
        """Draw NBI samples with mean ``mu`` and dispersion ``sigma`` (size ``1/sigma``)."""
        rng = np.random.default_rng() if rng is None else rng
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        if np.any(mu <= 0) or np.any(sigma <= 0):
            raise ValueError("NegativeBinomialLSS requires mu and sigma to be positive")
        nb_size = 1.0 / sigma
        return rng.negative_binomial(n=nb_size, p=nb_size / (nb_size + mu), size=size)

    def deviance_increment(self, y, params) -> np.ndarray:
        mass = self.pmf(y, params)
        return -2.0 * np.log(np.maximum(mass, np.finfo(np.float64).tiny))

    def pdf(self, y, params) -> np.ndarray:
        """NBI probability mass, variance ``mu + sigma * mu**2``.

        Matches ``gamlss.dist::dNBI(y, mu, sigma)`` with ``sigma`` passed
        straight through: here ``sigma`` is gamlss' dispersion, so the size is
        its reciprocal. :class:`NBinomialLSS` is the same distribution with
        ``sigma`` naming the size instead — the two are separate families for
        that reason, not aliases.
        """
        y = np.asarray(y, dtype=np.float64)
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        if np.any(mu <= 0) or np.any(sigma <= 0):
            raise ValueError("NegativeBinomialLSS requires mu and sigma positive")
        y, mu, sigma = np.broadcast_arrays(y, mu, sigma)
        size = 1.0 / sigma
        support = _on_count_support(y)
        safe = np.where(support, y, 0.0)
        log_mass = (
            _LGAMMA(safe + size) - _LGAMMA(size) - _LGAMMA(safe + 1.0)
            + size * np.log(size / (size + mu))
            + safe * np.log(mu / (size + mu))
        )
        return np.where(support, np.exp(log_mass), 0.0)


class ZeroAdjustedGammaLSS(_StabilizedNGradient, _DensityDeviance):
    """Zero-adjusted Gamma: ``mu`` (log, mean of the positive part), ``sigma``
    (log, Gamma dispersion), ``nu`` (logit, probability of an exact zero).
    Requires ``y >= 0``. Matches ``gamlss.dist::ZAGA()``.

    The semicontinuous case: insurance claims, daily rainfall, household spend
    — positive and right-skewed *when it happens*, exactly zero otherwise.
    Because ``nu`` is a fitted parameter you can put learners on, "what drives
    whether it happens" and "what drives how large it is" become two separate
    predictors rather than one blended variance assumption.

    An all-zero response is refused: the Gamma part would have no data to
    identify ``mu`` and ``sigma`` from.
    """

    n_params = 3
    param_names = ("mu", "sigma", "nu")
    _rust_name = "zeroadjustedgammalss"
    _mixed_atom = 0.0

    def response(self, k: int, eta: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=np.float64)
        if k in (0, 1):
            return np.exp(eta)
        if k == 2:
            return 1.0 / (1.0 + np.exp(-eta))
        raise IndexError(f"ZeroAdjustedGammaLSS has 3 parameters, got k={k}")

    def mean(self, params) -> np.ndarray:
        """Unconditional response mean, ``(1 - nu) * mu``."""
        return (1.0 - _param_array(params, "nu")) * _param_array(params, "mu")

    def pdf(self, y, params):
        """Mixture density: ``nu`` at zero, ``(1-nu) * Gamma`` above it."""
        y = np.asarray(y, dtype=np.float64)
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        nu = _param_array(params, "nu")
        if np.any(mu <= 0) or np.any(sigma <= 0):
            raise ValueError("ZeroAdjustedGammaLSS requires mu and sigma positive")
        if np.any((nu < 0) | (nu > 1)):
            raise ValueError("ZeroAdjustedGammaLSS nu must be in [0, 1]")
        y, mu, sigma, nu = np.broadcast_arrays(y, mu, sigma, nu)
        s2 = sigma**2
        inv = 1.0 / s2
        with np.errstate(divide="ignore", invalid="ignore"):
            log_gamma = (
                inv * np.log(np.where(y > 0, y, 1.0) / (mu * s2))
                - np.where(y > 0, y, 0.0) / (mu * s2)
                - np.log(np.where(y > 0, y, 1.0))
                - _LGAMMA(inv)
            )
        return np.where(y < 0.0, 0.0,
                        np.where(y == 0.0, nu, (1.0 - nu) * np.exp(log_gamma)))

    def cdf(self, y, params) -> np.ndarray:
        """Zero-adjusted gamma CDF: ``nu`` at zero, then ``nu + (1 - nu) * F_gamma``.

        Note the parameterization: here ``sigma`` is the gamma coefficient of
        variation, as in ``gamlss.dist::ZAGA``, so the shape is ``1 / sigma^2``
        and the scale ``mu * sigma^2``. This differs from :class:`GammaLSS`,
        whose ``sigma`` is the shape itself.
        """
        y = np.asarray(y, dtype=np.float64)
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        nu = _param_array(params, "nu")
        if np.any(mu <= 0) or np.any(sigma <= 0):
            raise ValueError("ZeroAdjustedGammaLSS requires mu and sigma positive")
        if np.any((nu < 0) | (nu > 1)):
            raise ValueError("ZeroAdjustedGammaLSS nu must be in [0, 1]")
        y, mu, sigma, nu = np.broadcast_arrays(y, mu, sigma, nu)
        out = np.zeros(y.shape, dtype=np.float64)
        at_zero = y == 0
        out[at_zero] = nu[at_zero]
        positive = y > 0
        if np.any(positive):
            shape = 1.0 / sigma[positive] ** 2
            scale = mu[positive] * sigma[positive] ** 2
            gamma_cdf = _special("gamma_p", shape, y[positive] / scale)
            out[positive] = nu[positive] + (1.0 - nu[positive]) * gamma_cdf
        return np.clip(out, 0.0, 1.0)

    def cdf_left(self, y, params) -> np.ndarray:
        """Left CDF limit: zero at the atom, ordinary CDF above zero."""
        y = np.asarray(y, dtype=np.float64)
        return np.where(y <= 0.0, 0.0, self.cdf(y, params))

    def quantile(self, p, params) -> np.ndarray:
        """Zero-adjusted gamma quantile; zero for ``p <= nu``."""
        return _core_quantile("ZeroAdjustedGammaLSS", p, params, ("mu", "sigma", "nu"))


class InverseGaussianLSS(_StabilizedNGradient, _DensityDeviance):
    """Inverse Gaussian family: ``mu`` (log link, mean), ``sigma`` (log link,
    dispersion). Requires ``y > 0``. Matches ``gamlss.dist::IG()``.

    ``Var(y) = sigma**2 * mu**3``, so the variance grows faster with the mean
    than Gamma's does — the usual choice for right-skewed positive data with
    heavier mean-dependence than Gamma can express.
    """

    n_params = 2
    param_names = ("mu", "sigma")
    _rust_name = "inversegaussianlss"

    def response(self, k: int, eta: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=np.float64)
        if k in (0, 1):
            return np.exp(eta)
        raise IndexError(f"InverseGaussianLSS has 2 parameters, got k={k}")

    def mean(self, params) -> np.ndarray:
        """Conditional response mean."""
        return _param_array(params, "mu")

    def pdf(self, y, params):
        mu = np.asarray(params["mu"], dtype=np.float64)
        sigma = np.asarray(params["sigma"], dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        y, mu, sigma = np.broadcast_arrays(y, mu, sigma)
        # Zero below the support (y <= 0), where the formula gives NaN; the
        # CDF is already 0 there.
        positive = y > 0
        safe = np.where(positive, y, 1.0)
        density = np.exp(
            -0.5 * np.log(2.0 * np.pi)
            - np.log(sigma)
            - 1.5 * np.log(safe)
            - (safe - mu) ** 2 / (2.0 * mu**2 * sigma**2 * safe)
        )
        return np.where(positive, density, 0.0)

    def cdf(self, y, params) -> np.ndarray:
        """Inverse-Gaussian CDF. Matches ``gamlss.dist::pIG``.

        The closed form pairs a normal tail with a reflected one:
        ``Phi(a) + exp(2 / (mu * sigma^2)) * Phi(b)``. The exponential factor
        overflows for a small ``mu * sigma^2``, so the second term is formed in
        logs and exponentiated once.
        """
        y = np.asarray(y, dtype=np.float64)
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        if np.any(mu <= 0) or np.any(sigma <= 0):
            raise ValueError("InverseGaussianLSS requires mu and sigma positive")
        return _special("inverse_gaussian_cdf", y, mu, sigma)

    def quantile(self, p, params) -> np.ndarray:
        """Inverse-Gaussian quantile, obtained by inverting :meth:`cdf`."""
        return _core_quantile("InverseGaussianLSS", p, params, ("mu", "sigma"))


class LogisticLSS(_StabilizedNGradient, _DensityDeviance):
    """Logistic family: ``mu`` (identity link, location), ``sigma`` (log link,
    scale). ``y`` unrestricted. Matches ``gamlss.dist::LO()``.

    Heavier-tailed than Gaussian at the same scale, so it is the usual drop-in
    when residuals are symmetric but mildly outlier-prone. Note ``sigma`` is the
    logistic scale, not the standard deviation: ``sd = sigma * pi / sqrt(3)``.
    """

    n_params = 2
    param_names = ("mu", "sigma")
    _rust_name = "logisticlss"

    def response(self, k: int, eta: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=np.float64)
        if k == 0:
            return eta.copy()
        if k == 1:
            return np.exp(eta)
        raise IndexError(f"LogisticLSS has 2 parameters, got k={k}")

    def mean(self, params) -> np.ndarray:
        """Conditional response mean."""
        return _param_array(params, "mu")

    def pdf(self, y, params):
        mu = np.asarray(params["mu"], dtype=np.float64)
        sigma = np.asarray(params["sigma"], dtype=np.float64)
        if np.any(sigma <= 0):
            raise ValueError("LogisticLSS sigma must be positive")
        z = (np.asarray(y, dtype=np.float64) - mu) / sigma
        tail = np.exp(-np.abs(z))
        return tail / (sigma * (1.0 + tail) ** 2)

    def cdf(self, y, params) -> np.ndarray:
        """Logistic CDF, ``1 / (1 + exp(-(y - mu) / sigma))``."""
        y = np.asarray(y, dtype=np.float64)
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        if np.any(sigma <= 0):
            raise ValueError("LogisticLSS sigma must be positive")
        z = (y - mu) / sigma
        # expit written to avoid overflow for large |z|.
        return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)),
                        np.exp(np.minimum(z, 0.0)) / (1.0 + np.exp(np.minimum(z, 0.0))))

    def quantile(self, p, params) -> np.ndarray:
        """Logistic quantile, ``mu + sigma * log(p / (1 - p))``."""
        return _core_quantile("LogisticLSS", p, params, ("mu", "sigma"))


class GumbelLSS(_StabilizedNGradient, _DensityDeviance):
    """Gumbel location-scale family for maxima — the standard extreme-value
    type I: ``mu`` (identity link, location), ``sigma`` (log link, scale).

    Matches ``gamlss.dist::RG()``, which gamlss labels "Reverse Gumbel", **not**
    ``GU()``. gamlss's naming inverts the common one: its ``GU`` is the Gumbel
    for *minima*, and its ``RG`` is the right-skewed form usually meant by
    "Gumbel". For ``y = [0.4, 1.3, -0.7, 2.1, 0.0]`` at
    ``mu = 0.5, sigma = 2``, the risk is 9.042800226406863, matching
    ``-sum(dRG(..., log = TRUE))``; ``dGU`` gives 9.161943.
    """

    n_params = 2
    param_names = ("mu", "sigma")
    _rust_name = "gumbellss"

    def response(self, k: int, eta: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=np.float64)
        if k == 0:
            return eta.copy()
        if k == 1:
            return np.exp(eta)
        raise IndexError(f"GumbelLSS has 2 parameters, got k={k}")

    def mean(self, params) -> np.ndarray:
        """Conditional response mean, ``mu + EulerGamma * sigma``."""
        return _param_array(params, "mu") + np.euler_gamma * _param_array(params, "sigma")

    def cdf(self, y, params) -> np.ndarray:
        """CDF of the extreme-value distribution for maxima,
        ``exp(-exp(-(y - mu) / sigma))``. Matches ``gamlss.dist::pRG``."""
        y = np.asarray(y, dtype=np.float64)
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        if np.any(sigma <= 0):
            raise ValueError("GumbelLSS sigma must be positive")
        return np.exp(-np.exp(-(y - mu) / sigma))

    def quantile(self, p, params) -> np.ndarray:
        """Gumbel (maxima) quantile, ``mu - sigma * log(-log(p))``."""
        return _core_quantile("GumbelLSS", p, params, ("mu", "sigma"))

    def pdf(self, y, params) -> np.ndarray:
        """Density of the extreme-value distribution for maxima. Matches
        ``gamlss.dist::dRG``."""
        y = np.asarray(y, dtype=np.float64)
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        if np.any(sigma <= 0):
            raise ValueError("GumbelLSS sigma must be positive")
        z = (y - mu) / sigma
        return np.exp(-z - np.exp(-z)) / sigma


class Pareto2LSS(_StabilizedNGradient, _DensityDeviance):
    """Pareto type-2 (Lomax) family: ``mu`` (log link, scale), ``sigma`` (log
    link, tail shape). The density is ``sigma * mu**sigma / (y + mu)**(sigma + 1)``
    for ``y > 0``; a smaller ``sigma`` is a heavier tail, and the mean
    ``mu / (sigma - 1)`` exists only for ``sigma > 1``.

    gamlss.dist ``PARETO2`` uses the reciprocal shape: this family's
    ``(mu, sigma)`` is ``PARETO2(mu, 1 / sigma)``, so ``pdf`` equals
    ``dPARETO2(y, mu, 1 / sigma)``. Requires ``y > 0``.
    """

    n_params = 2
    param_names = ("mu", "sigma")
    _rust_name = "pareto2lss"

    def response(self, k: int, eta: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=np.float64)
        if k in (0, 1):
            return np.exp(eta)
        raise IndexError(f"Pareto2LSS has 2 parameters, got k={k}")

    def mean(self, params) -> np.ndarray:
        """Lomax mean ``mu / (sigma - 1)``, defined only for ``sigma > 1``."""
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        if np.any(mu <= 0) or np.any(sigma <= 1):
            raise ValueError("Pareto2LSS mean requires mu > 0 and sigma > 1")
        return mu / (sigma - 1)

    def cdf(self, y, params) -> np.ndarray:
        """Pareto-II (Lomax) CDF, ``1 - (mu / (y + mu))**sigma``.

        This is mboostrs' own parameterization, in which ``sigma`` is the shape.
        ``gamlss.dist::PARETO2`` uses ``1 / sigma`` as the shape, so its CDF at
        the same nominal ``sigma`` differs; see the class docstring.
        """
        y = np.asarray(y, dtype=np.float64)
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        if np.any(mu <= 0) or np.any(sigma <= 0):
            raise ValueError("Pareto2LSS requires mu and sigma positive")
        y, mu, sigma = np.broadcast_arrays(y, mu, sigma)
        return np.where(y <= 0, 0.0, 1.0 - (mu / (y + mu)) ** sigma)

    def quantile(self, p, params) -> np.ndarray:
        """Pareto-II quantile, ``mu * ((1 - p)**(-1 / sigma) - 1)``."""
        return _core_quantile("Pareto2LSS", p, params, ('mu', 'sigma',))

    def pdf(self, y, params) -> np.ndarray:
        """Pareto-II (Lomax) density, ``sigma * mu**sigma / (y + mu)**(sigma+1)``.

        mboostrs' parameterization, in which ``sigma`` is the shape; see the
        class docstring for how it relates to ``gamlss.dist::PARETO2``.
        """
        y = np.asarray(y, dtype=np.float64)
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        if np.any(mu <= 0) or np.any(sigma <= 0):
            raise ValueError("Pareto2LSS requires mu and sigma positive")
        y, mu, sigma = np.broadcast_arrays(y, mu, sigma)
        return np.where(y < 0, 0.0, sigma * mu**sigma / (y + mu) ** (sigma + 1.0))


class ZeroAdjustedPoissonLSS(_StabilizedNGradient, _DensityDeviance):
    """Zero-adjusted Poisson family: ``mu`` (log link, Poisson mean), ``sigma``
    (logit link, zero-inflation probability). Matches R
    ``gamboostLSS::ZeroAdjustedPoissonLSS()`` (wrapping gamlss.dist ``ZAP``).
    Requires non-negative integer counts.
    """

    n_params = 2
    param_names = ("mu", "sigma")
    _rust_name = "zeroadjustedpoissonlss"

    _EPS = 2.220446049250313e-16
    _THRESH = 36.04365338911715

    def response(self, k: int, eta: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=np.float64)
        if k == 0:
            return np.maximum(np.exp(eta), self._EPS)
        if k == 1:
            clipped = np.clip(eta, -self._THRESH, self._THRESH)
            return 1.0 / (1.0 + np.exp(-clipped))
        raise IndexError(f"ZeroAdjustedPoissonLSS has 2 parameters, got k={k}")

    def mean(self, params) -> np.ndarray:
        """Mixture mean ``(1-sigma) * mu / (1-exp(-mu))``."""
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        if np.any(mu <= 0) or np.any((sigma < 0) | (sigma > 1)):
            raise ValueError("ZeroAdjustedPoissonLSS mean requires mu > 0 and sigma in [0, 1]")
        return (1 - sigma) * mu / -np.expm1(-mu)

    def cdf(self, y, params) -> np.ndarray:
        """Zero-adjusted Poisson CDF: ``sigma`` at zero, then the
        zero-truncated Poisson scaled by ``1 - sigma``."""
        y = np.asarray(y, dtype=np.float64)
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        if np.any(mu <= 0):
            raise ValueError("ZeroAdjustedPoissonLSS mu must be positive")
        if np.any((sigma <= 0) | (sigma >= 1)):
            raise ValueError("ZeroAdjustedPoissonLSS sigma must be in (0, 1)")
        y, mu, sigma = np.broadcast_arrays(y, mu, sigma)
        zero = np.exp(-mu)
        truncated = np.maximum((_poisson_cdf(y, mu) - zero) / (1.0 - zero), 0.0)
        return np.clip(np.where(y < 0, 0.0, sigma + (1.0 - sigma) * truncated), 0.0, 1.0)

    def quantile(self, p, params) -> np.ndarray:
        """Zero-adjusted Poisson quantile; zero for ``p <= sigma``."""
        return _core_quantile("ZeroAdjustedPoissonLSS", p, params, ("mu", "sigma"))

    def pdf(self, y, params) -> np.ndarray:
        """Zero-adjusted Poisson mass: ``sigma`` at zero, the zero-truncated
        Poisson scaled by ``1 - sigma`` above. Matches ``gamlss.dist::dZAP``."""
        y = np.asarray(y, dtype=np.float64)
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        if np.any(mu <= 0):
            raise ValueError("ZeroAdjustedPoissonLSS mu must be positive")
        y, mu, sigma = np.broadcast_arrays(y, mu, sigma)
        poisson = np.exp(y * np.log(mu) - mu - _LGAMMA(np.maximum(y, 0.0) + 1.0))
        truncated = poisson / (1.0 - np.exp(-mu))
        mass = np.where(y < 0.5, sigma, (1.0 - sigma) * truncated)
        return np.where(_on_count_support(y), mass, 0.0)


class PIGLSS(_StabilizedNGradient):
    """Poisson-inverse-Gaussian family: ``mu`` (log link, mean), ``sigma``
    (log link, dispersion). Matches R ``gamboostLSS::PIGLSS()`` (wrapping
    gamlss.dist ``PIG``). Requires non-negative integer counts.
    """

    n_params = 2
    param_names = ("mu", "sigma")
    _rust_name = "piglss"

    _EPS = 2.220446049250313e-16

    def response(self, k: int, eta: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=np.float64)
        if k in (0, 1):
            return np.maximum(np.exp(eta), self._EPS)
        raise IndexError(f"PIGLSS has 2 parameters, got k={k}")

    def mean(self, params) -> np.ndarray:
        """Conditional count mean ``mu``."""
        return _param_array(params, "mu")


class BCCGLSS(_StabilizedNGradient, _DensityDeviance):
    """Box-Cox-Cole-Green family: ``mu`` (log link, median), ``sigma``
    (log link, scale), ``nu`` (identity link, Box-Cox power).
    Matches R ``gamboostLSS::BCCGLSS()`` (wrapping gamlss.dist ``BCCG``).
    Requires ``y > 0``.
    """

    n_params = 3
    param_names = ("mu", "sigma", "nu")
    _rust_name = "bccglss"

    def response(self, k: int, eta: np.ndarray) -> np.ndarray:
        """Return response-scale parameters: log-linked mu and sigma, identity nu."""
        eta = np.asarray(eta, dtype=np.float64)
        if k in (0, 1):
            return np.exp(eta)
        if k == 2:
            return eta.copy()
        raise IndexError(f"BCCGLSS has 3 parameters, got k={k}")

    def cdf(self, y, params) -> np.ndarray:
        """Box-Cox-Cole-Green CDF. Matches ``gamlss.dist::pBCCG``.

        The Box-Cox transform maps ``y > 0`` to a normal variate truncated
        below, so the normal CDF is renormalized by ``Phi(1 / (sigma |nu|))``.
        """
        y = np.asarray(y, dtype=np.float64)
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        nu = _param_array(params, "nu")
        if np.any(mu <= 0) or np.any(sigma <= 0):
            raise ValueError("BCCGLSS requires mu and sigma positive")
        y, mu, sigma, nu = np.broadcast_arrays(y, mu, sigma, nu)
        ratio = np.where(y > 0, y / mu, 1.0)
        small = np.abs(nu) < 1e-10
        z = np.where(small, np.log(ratio) / sigma,
                     (ratio ** np.where(small, 1.0, nu) - 1.0) / (np.where(small, 1.0, nu) * sigma))
        edge = 1.0 / (sigma * np.where(small, 1.0, np.abs(nu)))
        lower = np.where(small, 0.0, np.where(nu > 0, _normal_cdf_array(-edge), 0.0))
        scale = np.where(small, 1.0, _normal_cdf_array(edge))
        value = (_normal_cdf_array(z) - lower) / scale
        return np.where(y > 0, np.clip(value, 0.0, 1.0), 0.0)

    def quantile(self, p, params) -> np.ndarray:
        """Box-Cox-Cole-Green quantile. Matches ``gamlss.dist::qBCCG``."""
        return _core_quantile("BCCGLSS", p, params, ("mu", "sigma", "nu"))

    def pdf(self, y, params) -> np.ndarray:
        """Box-Cox-Cole-Green density. Matches ``gamlss.dist::dBCCG``.

        Includes the truncation normalizer ``Phi(1 / (sigma |nu|))`` that makes
        it integrate to one over ``y > 0``.
        """
        y = np.asarray(y, dtype=np.float64)
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        nu = _param_array(params, "nu")
        if np.any(mu <= 0) or np.any(sigma <= 0):
            raise ValueError("BCCGLSS requires mu and sigma positive")
        y, mu, sigma, nu = np.broadcast_arrays(y, mu, sigma, nu)
        out = np.zeros(np.shape(y), dtype=np.float64)
        positive = y > 0
        if np.any(positive):
            yy, mm, ss, nn = y[positive], mu[positive], sigma[positive], nu[positive]
            ratio = yy / mm
            small = np.abs(nn) < 1e-10
            z = np.where(small, np.log(ratio) / ss,
                         (ratio ** np.where(small, 1.0, nn) - 1.0)
                         / (np.where(small, 1.0, nn) * ss))
            log_density = (
                nn * np.log(ratio) - np.log(ss) - 0.5 * z * z - np.log(yy)
                - 0.5 * np.log(2.0 * np.pi)
            )
            edge = 1.0 / (ss * np.where(small, 1.0, np.abs(nn)))
            log_density = log_density - np.where(
                small, 0.0, np.log(_normal_cdf_array(edge))
            )
            out[positive] = np.exp(log_density)
        return out


class JSULSS(_StabilizedNGradient, _DensityDeviance):
    """Johnson's SU family: ``mu`` (identity link, location), ``sigma``
    (log link, scale), ``nu`` (identity link, skewness), ``tau`` (log link,
    kurtosis). Matches R ``gamboostLSS::JSULSS()`` (wrapping gamlss.dist
    ``JSU``).
    """

    n_params = 4
    param_names = ("mu", "sigma", "nu", "tau")
    _rust_name = "jsulss"

    def response(self, k: int, eta: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=np.float64)
        if k in (0, 2):
            return eta.copy()
        if k in (1, 3):
            return np.exp(eta)
        raise IndexError(f"JSULSS has 4 parameters, got k={k}")

    def mean(self, params) -> np.ndarray:
        """Conditional mean ``mu`` under the centered JSU parameterization."""
        return _param_array(params, "mu")

    def _jsu_shape(self, params):
        """The centering constants gamlss' JSU parameterization applies.

        ``mu`` and ``sigma`` are the mean and standard deviation, so the raw
        Johnson SU location and scale are shifted and rescaled by these.
        """
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        nu = _param_array(params, "nu")
        tau = _param_array(params, "tau")
        if np.any(sigma <= 0) or np.any(tau <= 0):
            raise ValueError("JSULSS requires sigma and tau positive")
        rtau = 1.0 / tau
        w = np.where(rtau < 1e-7, 1.0, np.exp(rtau * rtau))
        omega = -nu * rtau
        c = (0.5 * (w - 1.0) * (w * np.cosh(2.0 * omega) + 1.0)) ** -0.5
        return mu, sigma, nu, rtau, w, omega, c

    def cdf(self, y, params) -> np.ndarray:
        """Johnson SU CDF in the centered parameterization. Matches
        ``gamlss.dist::pJSU``."""
        y = np.asarray(y, dtype=np.float64)
        mu, sigma, nu, rtau, w, omega, c = self._jsu_shape(params)
        y, mu, sigma, nu, rtau, w, omega, c = np.broadcast_arrays(
            y, mu, sigma, nu, rtau, w, omega, c
        )
        z = (y - (mu + c * sigma * np.sqrt(w) * np.sinh(omega))) / (c * sigma)
        return _normal_cdf_array(-nu + np.arcsinh(z) / rtau)

    def quantile(self, p, params) -> np.ndarray:
        """Johnson SU quantile. Matches ``gamlss.dist::qJSU``."""
        return _core_quantile("JSULSS", p, params, ("mu", "sigma", "nu", "tau"))

    def pdf(self, y, params) -> np.ndarray:
        """Johnson SU density in the centered parameterization. Matches
        ``gamlss.dist::dJSU``."""
        y = np.asarray(y, dtype=np.float64)
        mu, sigma, nu, rtau, w, omega, c = self._jsu_shape(params)
        y, mu, sigma, nu, rtau, w, omega, c = np.broadcast_arrays(
            y, mu, sigma, nu, rtau, w, omega, c
        )
        z = (y - (mu + c * sigma * np.sqrt(w) * np.sinh(omega))) / (c * sigma)
        r = -nu + np.arcsinh(z) / rtau
        return (
            np.exp(-0.5 * r * r)
            / (_SQRT_2PI * c * sigma * rtau * np.sqrt(z * z + 1.0))
        )


class BCTLSS(_StabilizedNGradient, _DensityDeviance):
    """Box-Cox t family with ``mu``, ``sigma``, ``nu``, and ``tau``.

    ``mu`` and ``nu`` use identity links; ``sigma`` and Student-t degrees of
    freedom ``tau`` use log links. The response must be positive, as must the
    response-scale ``mu``, ``sigma``, and ``tau``. The Box-Cox transform is
    truncated at the positive-response boundary and the density is normalized
    by the corresponding Student-t probability, matching
    ``gamlss.dist::BCT``.

    ``mu`` is a location parameter, not generally the response mean. Use
    quantiles and centile curves for interpretation; :meth:`mean` refuses.
    """

    n_params = 4
    param_names = ("mu", "sigma", "nu", "tau")
    _rust_name = "bctlss"

    def response(self, k: int, eta: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=np.float64)
        if k in (0, 2):
            return eta.copy()
        if k in (1, 3):
            return np.exp(eta)
        raise IndexError(f"BCTLSS has 4 parameters, got k={k}")

    def mean(self, params) -> np.ndarray:
        raise NotImplementedError(
            "BCTLSS mu is a location parameter; the truncated Box-Cox-t "
            "response mean is not implemented"
        )

    @staticmethod
    def _params(params):
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        nu = _param_array(params, "nu")
        tau = _param_array(params, "tau")
        if (np.any(~np.isfinite(mu)) or np.any(~np.isfinite(sigma))
                or np.any(~np.isfinite(nu)) or np.any(~np.isfinite(tau))
                or np.any(mu <= 0) or np.any(sigma <= 0) or np.any(tau <= 0)):
            raise ValueError("BCTLSS requires finite mu > 0, sigma > 0, nu, and tau > 0")
        return mu, sigma, nu, tau

    def pdf(self, y, params) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        mu, sigma, nu, tau = self._params(params)
        y, mu, sigma, nu, tau = np.broadcast_arrays(y, mu, sigma, nu, tau)
        out = np.zeros(y.shape, dtype=np.float64)
        positive = y > 0
        if not np.any(positive):
            return out
        yy, mm, ss, nn, tt = (
            value[positive] for value in (y, mu, sigma, nu, tau)
        )
        z = _boxcox_z(yy, mm, ss, nn)
        log_t = (
            _LGAMMA((tt + 1) / 2)
            - _LGAMMA(tt / 2)
            - .5 * np.log(tt)
            - lgamma(.5)
            - ((tt + 1) / 2) * np.log1p(z * z / tt)
        )
        t0 = _boxcox_edge(ss, nn)
        # t_cdf(inf) is 1: no truncation when nu is 0.
        normalizer = _special("t_cdf", t0, tt)
        log_jacobian = (nn - 1) * np.log(yy) - nn * np.log(mm) - np.log(ss)
        out[positive] = np.exp(log_jacobian + log_t - np.log(normalizer))
        return out

    def cdf(self, y, params) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        mu, sigma, nu, tau = self._params(params)
        y, mu, sigma, nu, tau = np.broadcast_arrays(y, mu, sigma, nu, tau)
        out = np.zeros(y.shape, dtype=np.float64)
        positive = y > 0
        if not np.any(positive):
            return out
        yy, mm, ss, nn, tt = (
            value[positive] for value in (y, mu, sigma, nu, tau)
        )
        z = _boxcox_z(yy, mm, ss, nn)
        t0 = _boxcox_edge(ss, nn)
        f1 = _special("t_cdf", z, tt)
        lower_edge = (nn > 0) & np.isfinite(t0)
        f2 = np.where(lower_edge, _special("t_cdf", -np.where(lower_edge, t0, 0.0), tt), 0.0)
        f3 = _special("t_cdf", t0, tt)
        out[positive] = np.clip((f1 - f2) / f3, 0.0, 1.0)
        return out

    def quantile(self, p, params) -> np.ndarray:
        return _core_quantile("BCTLSS", p, params, ("mu", "sigma", "nu", "tau"))

    def simulate(self, params, rng=None, size=None) -> np.ndarray:
        rng = np.random.default_rng() if rng is None else rng
        mu, sigma, nu, tau = self._params(params)
        shape = np.broadcast(mu, sigma, nu, tau).shape if size is None else size
        return self.quantile(rng.uniform(size=shape), params)


class BCPELSS(_StabilizedNGradient, _DensityDeviance):
    """Box-Cox power-exponential family with four fitted parameters.

    Links and response support match :class:`BCTLSS`. ``tau`` controls the
    power-exponential kernel rather than Student-t tails. The truncated
    positive-response density, CDF, quantile, and inverse-CDF simulation match
    ``gamlss.dist::BCPE``. ``mu`` is a location parameter; no general response
    mean is declared.
    """

    n_params = 4
    param_names = ("mu", "sigma", "nu", "tau")
    _rust_name = "bcpelss"

    def response(self, k: int, eta: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=np.float64)
        if k in (0, 2):
            return eta.copy()
        if k in (1, 3):
            return np.exp(eta)
        raise IndexError(f"BCPELSS has 4 parameters, got k={k}")

    def mean(self, params) -> np.ndarray:
        raise NotImplementedError(
            "BCPELSS mu is a location parameter; the truncated Box-Cox "
            "power-exponential response mean is not implemented"
        )

    @staticmethod
    def _params(params):
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        nu = _param_array(params, "nu")
        tau = _param_array(params, "tau")
        if (np.any(~np.isfinite(mu)) or np.any(~np.isfinite(sigma))
                or np.any(~np.isfinite(nu)) or np.any(~np.isfinite(tau))
                or np.any(mu <= 0) or np.any(sigma <= 0) or np.any(tau <= 0)):
            raise ValueError("BCPELSS requires finite mu > 0, sigma > 0, nu, and tau > 0")
        return mu, sigma, nu, tau

    def pdf(self, y, params) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        mu, sigma, nu, tau = self._params(params)
        y, mu, sigma, nu, tau = np.broadcast_arrays(y, mu, sigma, nu, tau)
        out = np.zeros(y.shape, dtype=np.float64)
        positive = y > 0
        if not np.any(positive):
            return out
        yy, mm, ss, nn, tt = (
            value[positive] for value in (y, mu, sigma, nu, tau)
        )
        z = _boxcox_z(yy, mm, ss, nn)
        log_c = _bcpe_log_c(tt)
        c = np.exp(log_c)
        log_kernel = (
            np.log(tt) - log_c - .5 * np.power(np.abs(z / c), tt)
            - (1 + 1 / tt) * np.log(2) - _LGAMMA(1 / tt)
        )
        t0 = _boxcox_edge(ss, nn)
        normalizer = np.ones(t0.shape, dtype=np.float64)
        finite_edge = np.isfinite(t0)
        if np.any(finite_edge):
            normalizer[finite_edge] = _bcpe_cdf_t(
                t0[finite_edge], tt[finite_edge]
            )
        log_jacobian = (nn - 1) * np.log(yy) - nn * np.log(mm) - np.log(ss)
        out[positive] = np.exp(log_jacobian + log_kernel - np.log(normalizer))
        return out

    def cdf(self, y, params) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        mu, sigma, nu, tau = self._params(params)
        y, mu, sigma, nu, tau = np.broadcast_arrays(y, mu, sigma, nu, tau)
        out = np.zeros(y.shape, dtype=np.float64)
        positive = y > 0
        if not np.any(positive):
            return out
        yy, mm, ss, nn, tt = (
            value[positive] for value in (y, mu, sigma, nu, tau)
        )
        z = _boxcox_z(yy, mm, ss, nn)
        t0 = _boxcox_edge(ss, nn)
        f1 = _bcpe_cdf_t(z, tt)
        f2 = np.zeros(t0.shape, dtype=np.float64)
        lower_edge = (nn > 0) & np.isfinite(t0)
        if np.any(lower_edge):
            f2[lower_edge] = _bcpe_cdf_t(-t0[lower_edge], tt[lower_edge])
        f3 = np.ones(t0.shape, dtype=np.float64)
        finite_edge = np.isfinite(t0)
        if np.any(finite_edge):
            f3[finite_edge] = _bcpe_cdf_t(t0[finite_edge], tt[finite_edge])
        out[positive] = np.clip((f1 - f2) / f3, 0.0, 1.0)
        return out

    def quantile(self, p, params) -> np.ndarray:
        return _core_quantile("BCPELSS", p, params, ("mu", "sigma", "nu", "tau"))

    def simulate(self, params, rng=None, size=None) -> np.ndarray:
        rng = np.random.default_rng() if rng is None else rng
        mu, sigma, nu, tau = self._params(params)
        shape = np.broadcast(mu, sigma, nu, tau).shape if size is None else size
        return self.quantile(rng.uniform(size=shape), params)


class DirichletLSS(_StabilizedNGradient):
    """Dirichlet family for compositional responses: ``K`` concentration
    parameters ``alpha_k = exp(eta_k)`` (log links). The response is a ``K``-column
    composition specified as ``cbind(y1, ..., yK) ~ ...`` (each row sums to 1).
    Matches R ``gamboostLSS::DirichletLSS(K)``. Requires ``0 < y < 1``.
    """

    _rust_name = "dirichletlss"
    _multivariate = True

    def __init__(self, K: int, stabilization: str = "none"):
        super().__init__(stabilization)
        if not 2 <= K <= 20:
            raise ValueError("DirichletLSS supports K in 2..=20")
        self.K = K
        self.n_params = K
        self.param_names = tuple(f"a{j + 1}" for j in range(K))

    def response(self, k: int, eta: np.ndarray) -> np.ndarray:
        if not 0 <= k < self.K:
            raise IndexError(f"DirichletLSS has {self.K} parameters, got k={k}")
        return np.exp(np.asarray(eta, dtype=np.float64))  # all params: log link

    def pdf(self, y, params) -> np.ndarray:
        """Dirichlet density for a K-column compositional response."""
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(self.K, -1).T
        alphas = np.array([_param_array(params, name) for name in self.param_names])
        if alphas.ndim == 1:
            alphas = alphas[:, np.newaxis]
        alpha0 = alphas.sum(axis=0)
        log_norm = _LGAMMA(alpha0) - np.sum(_LGAMMA(alphas), axis=0)
        log_y = np.sum((alphas.T - 1.0) * np.log(np.maximum(y, np.finfo(np.float64).tiny)), axis=1)
        return np.exp(log_norm + log_y)

    def simulate(self, params, rng=None, size=None) -> np.ndarray:
        """Draw Dirichlet samples with concentration ``alpha_k``."""
        rng = np.random.default_rng() if rng is None else rng
        alphas = np.array([_param_array(params, name) for name in self.param_names])
        if np.any(alphas <= 0):
            raise ValueError("DirichletLSS requires alpha > 0")
        return rng.dirichlet(alpha=alphas.T[0] if alphas.ndim > 1 else alphas, size=size)

    def deviance_increment(self, y, params) -> np.ndarray:
        dens = self.pdf(y, params)
        return -2.0 * np.log(np.maximum(dens, np.finfo(np.float64).tiny))


class _SurvivalLSS(_StabilizedNGradient, _DensityDeviance):
    """Base for accelerated-failure-time LSS families with a Surv(time, status)
    response. mu (identity link, log-location), sigma (log link, scale)."""

    n_params = 2
    param_names = ("mu", "sigma")
    _survival = True

    def response(self, k: int, eta: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=np.float64)
        if k == 0:  # mu: identity
            return eta.copy()
        if k == 1:  # sigma: log link
            return np.exp(eta)
        raise IndexError(f"{type(self).__name__} has 2 parameters, got k={k}")

    # --- the survival-time distribution ------------------------------------
    #
    # All three families share the accelerated-failure-time form
    # ``e = (log t - mu) / sigma``, differing only in the standard distribution
    # of ``e``. Subclasses supply that distribution's CDF and its inverse; the
    # time CDF and quantile follow.

    def _standard_cdf(self, e: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _standard_quantile(self, p: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def cdf(self, t, params) -> np.ndarray:
        """``P(T <= t)`` for the survival time.

        This is the distribution of the event time, not of the observed
        response: a right-censored observation contributes only the information
        that its event time exceeds the recorded one. `pit()` and
        `quantile_residuals()` therefore still refuse for these families, and
        say why — a censored PIT is not uniform.
        """
        t = np.asarray(t, dtype=np.float64)
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        if np.any(sigma <= 0):
            raise ValueError(f"{type(self).__name__} sigma must be positive")
        t, mu, sigma = np.broadcast_arrays(t, mu, sigma)
        out = np.zeros(np.shape(t), dtype=np.float64)
        positive = t > 0
        if np.any(positive):
            e = (np.log(t[positive]) - mu[positive]) / sigma[positive]
            out[positive] = self._standard_cdf(e)
        return np.clip(out, 0.0, 1.0)

    def _standard_pdf(self, e: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def pdf(self, t, params) -> np.ndarray:
        """Density of the survival time.

        The AFT change of variable ``e = (log t - mu)/sigma`` contributes a
        Jacobian of ``1/(sigma * t)``.
        """
        t = np.asarray(t, dtype=np.float64)
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        if np.any(sigma <= 0):
            raise ValueError(f"{type(self).__name__} sigma must be positive")
        t, mu, sigma = np.broadcast_arrays(t, mu, sigma)
        out = np.zeros(np.shape(t), dtype=np.float64)
        positive = t > 0
        if np.any(positive):
            tt, mm, ss = t[positive], mu[positive], sigma[positive]
            e = (np.log(tt) - mm) / ss
            out[positive] = self._standard_pdf(e) / (ss * tt)
        return out

    def quantile(self, p, params) -> np.ndarray:
        """The survival time ``t`` with ``P(T <= t) = p``.

        ``quantile(0.5, ...)`` is the median survival time, which is the
        quantity a survival model is usually read for.
        """
        return _core_quantile(type(self).__name__, p, params, ("mu", "sigma"))

    def uncensored(self):
        """This family for a plain positive response, with every observation an event.

        The returned object fits the response itself (its risk is the negative
        log-likelihood of ``y``, not of ``log(y)``) and reports itself as a
        plain-response family, so ``cdf``, ``quantile`` and the other
        distribution methods apply to ``y`` directly.
        """
        clone = type(self)(stabilization=self.stabilization)
        clone._survival = False
        clone._rust_name = f"{self._rust_name}_uncensored"
        return clone


class LogNormalLSS(_SurvivalLSS):
    """Log-normal AFT family. Matches R ``gamboostLSS::LogNormalLSS()`` for a
    right-censored ``Surv(time, status)`` response.

    A plain positive response is treated as uncensored (every status 1). Its
    risk is the negative log-likelihood of the response on its own scale,
    comparable with the other ``realplus`` families."""

    _rust_name = "lognormallss"

    def mean(self, params) -> np.ndarray:
        """Uncensored survival-time mean ``exp(mu + sigma**2 / 2)``."""
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        if np.any(sigma <= 0):
            raise ValueError("LogNormalLSS mean requires sigma > 0")
        return np.exp(mu + .5 * sigma**2)

    def _standard_cdf(self, e):
        return _normal_cdf_array(e)

    def _standard_pdf(self, e):
        e = np.asarray(e, dtype=np.float64)
        return np.exp(-0.5 * e * e) / _SQRT_2PI

    def _standard_quantile(self, p):
        return _normal_ppf_array(p)


class LogLogLSS(_SurvivalLSS):
    """Log-logistic AFT family. Takes a right-censored ``Surv(time, status)``
    response, or a plain positive response treated as uncensored.

    Negative log-likelihood, gradients, and risk match R
    ``gamboostLSS::LogLogLSS()``. For censored data, the automatic sigma
    offset minimizes the log-logistic negative log-likelihood; its value can
    differ from the initialization used by R's ``gamboostLSS``.
    """

    _rust_name = "logloglss"

    def mean(self, params) -> np.ndarray:
        """Uncensored log-logistic mean, defined for ``0 < sigma < 1``."""
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        if np.any((sigma <= 0) | (sigma >= 1)):
            raise ValueError("LogLogLSS mean requires 0 < sigma < 1")
        return np.exp(mu) * np.pi * sigma / np.sin(np.pi * sigma)

    def _standard_cdf(self, e):
        e = np.asarray(e, dtype=np.float64)
        return np.where(e >= 0, 1.0 / (1.0 + np.exp(-e)),
                        np.exp(np.minimum(e, 0.0)) / (1.0 + np.exp(np.minimum(e, 0.0))))

    def _standard_pdf(self, e):
        e = np.asarray(e, dtype=np.float64)
        # exp(e) / (1 + exp(e))^2, written to survive a large |e|.
        half = np.exp(-np.abs(e))
        return half / (1.0 + half) ** 2

    def _standard_quantile(self, p):
        return np.log(p / (1.0 - p))


class WeibullLSS(_SurvivalLSS):
    """Weibull AFT family. Takes a right-censored ``Surv(time, status)``
    response, or a plain positive response treated as uncensored.

    Negative log-likelihood, gradients, and risk match R
    ``gamboostLSS::WeibullLSS()``. The automatic ``mu`` offset is an
    identity-link location minimizing the Weibull negative log-likelihood;
    its value can differ from the log-location initialization used by R's
    ``gamboostLSS``.
    """

    _rust_name = "weibulllss"

    def mean(self, params) -> np.ndarray:
        """Uncensored AFT Weibull mean ``exp(mu) * Gamma(1 + sigma)``."""
        mu = _param_array(params, "mu")
        sigma = _param_array(params, "sigma")
        if np.any(sigma <= 0):
            raise ValueError("WeibullLSS mean requires sigma > 0")
        return np.exp(mu + _LGAMMA(1 + sigma))

    def _standard_cdf(self, e):
        return -np.expm1(-np.exp(np.asarray(e, dtype=np.float64)))

    def _standard_pdf(self, e):
        e = np.asarray(e, dtype=np.float64)
        return np.exp(e - np.exp(e))

    def _standard_quantile(self, p):
        return np.log(-np.log1p(-np.asarray(p, dtype=np.float64)))


_FAMILIES = {"GaussianLSS": GaussianLSS, "GammaLSS": GammaLSS,
             "StudentTLSS": StudentTLSS, "BetaLSS": BetaLSS,
             "NBinomialLSS": NBinomialLSS, "LogNormalLSS": LogNormalLSS,
             "LogLogLSS": LogLogLSS, "WeibullLSS": WeibullLSS,
             "ZIPoLSS": ZIPoLSS, "ZINBLSS": ZINBLSS,
             "PoissonLSS": PoissonLSS, "GeometricLSS": GeometricLSS,
             "BinomialLSS": BinomialLSS, "NegativeBinomialLSS": NegativeBinomialLSS,
             "GumbelLSS": GumbelLSS, "Pareto2LSS": Pareto2LSS,
             "ZeroAdjustedPoissonLSS": ZeroAdjustedPoissonLSS, "PIGLSS": PIGLSS,
             "BCCGLSS": BCCGLSS, "BCTLSS": BCTLSS, "BCPELSS": BCPELSS,
             "JSULSS": JSULSS}


# ── LSS residual kinds, declared as a table ───────────────────────── 
# Enough for ``LSSModel.residual_kinds()`` to be a claim that the 
# implementation checks in both directions, like the standard-model table in 
# ``mboostrs._residuals``. ``working`` is implicit -- every LSS parameter has 
# a negative gradient by construction -- and ``response``/``pearson`` are the 
# kinds a family can honestly produce on its location parameter.
#
# ``response``: ``y - mu`` with ``mu`` on the location's response scale. 
#   Defined for every univariate LSS family whose response is not censored. 
#   Survival responses (time, scale on the log-time link) refuse: the 
#   response is censored, and ``y - mu`` mixes a censored time with a scale. 
#   ``DirichletLSS`` refuses because its response is a whole composition 
#   vector, not a scalar.
# ``pearson``: ``(y - mu) / sqrt(V(mu))`` where the family declares a 
#   variance function V. Only ``GaussianLSS`` (V = sigma^2, so the residual 
#   is ``(y - mu)/sigma``) and ``PoissonLSS`` (V = mu) declare one in the 
#   current implementation; everything else refuses and says what the 
#   alternatives are.
#: Families whose quantile is computed by the core (`lss_quantile`).
_CORE_QUANTILE_FAMILIES = frozenset({
    "GaussianLSS", "GaussianKnownVarLSS", "GammaLSS", "StudentTLSS", "BetaLSS",
    "NBinomialLSS", "ZIPoLSS", "ZINBLSS", "PoissonLSS", "GeometricLSS",
    "BinomialLSS", "NegativeBinomialLSS", "ZeroAdjustedGammaLSS",
    "InverseGaussianLSS", "LogisticLSS", "GumbelLSS", "Pareto2LSS",
    "ZeroAdjustedPoissonLSS", "BCCGLSS", "JSULSS", "BCTLSS", "BCPELSS",
    "LogNormalLSS", "LogLogLSS", "WeibullLSS",
})
_NO_RESPONSE_LSS = frozenset({"LogNormalLSS", "LogLogLSS", "WeibullLSS",
                              "DirichletLSS"})
_PEARSON_LSS = frozenset({"GaussianLSS", "PoissonLSS"})


def lss_residual_kinds(family) -> tuple[str, ...]:
    """Declared residual kinds for an LSS family, ``working`` always last."""
    name = family.__class__.__name__
    declared = ("response",) if name not in _NO_RESPONSE_LSS else ()
    if name in _PEARSON_LSS:
        declared += ("pearson",)
    return tuple(dict.fromkeys((*declared, "working")))


def _lss_refuse(family, kind: str) -> str:
    """Why ``kind`` is undefined for this LSS family, and what to ask for."""
    name = family.__class__.__name__
    offered = ", ".join(lss_residual_kinds(family))

    if kind == "response" and name in _NO_RESPONSE_LSS:
        if name in ("LogNormalLSS", "LogLogLSS", "WeibullLSS") and not getattr(
                family, "_survival", True):
            return (
                f"Response residuals are undefined for {name}: its location "
                f"parameter is on the log scale of y, not the mean of y, so "
                f"y - mu subtracts a log from a value. Use "
                f"residual_frame(kind='quantile') or quantile_residuals() "
                f"for a distributional residual."
            )
        if name in ("LogNormalLSS", "LogLogLSS", "WeibullLSS"):
            return (
                f"Response residuals are undefined for {name}: the response "
                f"is censored and the location eta is on the log-time scale, "
                f"so y - mu mixes a censored time with a scale. Use "
                f"residual_frame(kind='quantile') or quantile_residuals() "
                f"for residuals that carry the censoring contract."
            )
        return (
            f"Response residuals are undefined for {name}: the response is a "
            f"composition vector, not a scalar, so y - mu is not one "
            f"residual. Use type='working' for the negative gradient of a "
            f"given parameter."
        )

    if kind == "pearson" and name not in _PEARSON_LSS:
        return (
            f"Pearson residuals are undefined for {name}: they are "
            f"(y - mu) / sqrt(V(mu)), and {name} declares no variance "
            f"function (only GaussianLSS and PoissonLSS do). type='working' "
            f"gives the negative gradient, and residual_frame(kind='quantile') "
            f"a distributional residual."
        )

    return f"{kind!r} residuals are undefined for {name}. Available: {offered}."


class LSSModel(WeightPolicyMixin):
    """Fitted gamboostLSS/glmboostLSS model (one predictor per parameter)."""

    #: Warnings raised while this model was fitted; set by the fit functions.
    _fit_warnings: tuple = ()

    @property
    def fit_warnings(self) -> tuple:
        """Warnings the fit showed, as ``warnings.WarningMessage`` objects.

        Empty for a quiet fit. The warnings are also emitted normally during
        the fit call itself, so reading this attribute afterwards is an
        alternative to wrapping the call with ``warnings.catch_warnings``.
        A warning your filters suppress is not recorded: under Python's
        default filters that includes ``ResourceWarning`` and a warning the
        same line already showed once.

        Not persisted: ``save_model``/``load_model`` restore the model without
        the fit-time warnings, because the artifact format carries plain data
        only and the warning types are interpreter-specific.
        """
        return self._fit_warnings

    def __init__(
        self,
        family,
        nu,
        risk,
        params,
        data,
        response_name=None,
        *,
        weights=None,
        formula=None,
        control=None,
        offset=None,
        method="cyclic",
        model_kind="glmboostLSS",
        execution_report=None,
        selection_order=None,
        bins=None,
        env=None,
        nu_per_param=None,
        stabilization=None,
    ):
        # params: list (per parameter, in family order) of dicts with keys
        #   offset, selected, coefs_flat, predictor_names, centering_means, fitted
        self._family = family
        self._nu_per_param = None if nu_per_param is None else list(nu_per_param)
        self._stabilization = stabilization
        self._nu = nu
        self._risk = np.asarray(risk, dtype=np.float64)
        self._params = params
        self._data = data
        self._response_name = response_name
        self._weights = None if weights is None else np.asarray(weights, dtype=np.float64)
        self._formula = formula
        self._control = control
        self._offset_arg = offset
        self._method = method
        self._model_kind = model_kind
        self._bins = bins
        self._env = env
        self._execution_report = (
            execution_report
            if isinstance(execution_report, ExecutionReport)
            else ExecutionReport.from_mapping(execution_report)
            if execution_report is not None
            else None
        )
        self._selection_order = list(selection_order or [])
        self.param_names = list(family.param_names)

    def __getstate__(self):
        """Serialize Python-owned model state, excluding opaque Rust stores.

        The stores cache prepared learners for prediction but contain no unique
        fitted information. They are rebuilt from the retained learner
        definitions, training data, weights, and step size on deserialization.
        """
        state = dict(self.__dict__)
        # `_fit_warnings` is a live-only diagnostic (see `fit_warnings`); the
        # model artifact format cannot encode `WarningMessage` objects.
        state.pop("_fit_warnings", None)
        state["_params"] = []
        for parameter in self._params:
            saved = dict(parameter)
            saved.pop("learner_store", None)
            state["_params"].append(saved)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        gam_parameters = [
            parameter for parameter in self._params if parameter.get("kind") == "gam"
        ]
        if not gam_parameters:
            return
        if _prepare_gamboostlss_stores is None:
            raise RuntimeError(
                "this serialized gamboostLSS model requires a mixed-learner "
                "mboostrs Rust extension"
            )
        learner_sets = [parameter["learners"] for parameter in self._params]
        specs = _gamboostlss_learner_specs(learner_sets, self._data)
        weights = self._weights
        if weights is None:
            weights = np.ones(len(self._params[0]["fitted"]), dtype=np.float64)
        stores = _prepare_gamboostlss_stores(
            specs, weights, self._nu,
            nu_per_param=getattr(self, "_nu_per_param", None),
        )
        for parameter, store in zip(self._params, stores, strict=True):
            parameter["learner_store"] = store

    def _parameter_nu(self, k: int) -> float:
        rates = getattr(self, "_nu_per_param", None)
        return self._nu if rates is None else rates[k]

    def execution_report(self) -> ExecutionReport | None:
        """Return aggregate numerical-route metadata for this LSS fit."""
        return self._execution_report

    def path_frame(self, output="auto"):
        """Return globally ordered parameter-update rows for this LSS fit."""
        rows = lss_path_rows(
            params=self._params,
            param_names=self.param_names,
            selection_order=self._selection_order,
            risk=self._risk,
            nu=self._nu,
            noncyclic=self._method == "noncyclic",
            nu_per_param=getattr(self, "_nu_per_param", None),
            labels=self._interaction_labels(),
        )
        return path_rows_output(rows, output)

    @property
    def risk_(self) -> np.ndarray:
        """The joint risk after each iteration.

        Read-only: this is the fit's own array, not a copy. See
        :mod:`mboostrs._state`.
        """
        from mboostrs._state import readonly

        return readonly(self._risk)

    @property
    def mstop(self):
        # Per-parameter number of boosting steps (R's mstop vector). Returns an
        # int if all parameters share the same count, else a {param: count} dict.
        counts = {name: len(self._params[k]["selected"])
                  for k, name in enumerate(self.param_names)}
        uniq = set(counts.values())
        return next(iter(uniq)) if len(uniq) == 1 else counts

    def stopping_report(self) -> dict:
        """Where the boosting path stopped, and whether that looks finished.

        Same report as :meth:`ModelAPIMixin.stopping_report`: best in-sample
        risk step, risk there versus at the end, whether the risk rose after
        its best step, and a same-family GLM gap when one can be fit cheaply.
        Reads only the stored joint risk path; nothing is refit.
        """
        from mboostrs._model_api import _stopping_report_for

        return _stopping_report_for(self)

    def stopping_budget(self):
        """How far this model was boosted, as
        :class:`mboostrs._model_api.StoppingBudget`: per-parameter update
        counts, their total (the budget a non-cyclic refit takes) and whether
        the parameters were updated cyclically or non-cyclically."""
        from mboostrs._model_api import StoppingBudget

        counts = {name: len(self._params[k]["selected"])
                  for k, name in enumerate(self.param_names)}
        return StoppingBudget(scheme=self._method, total=int(sum(counts.values())),
                              per_parameter=counts)

    def selected_terms(self, output="auto"):
        """One row per boosting update, in the order the fit made them:
        ``order`` (1-based, across all parameters), ``parameter``,
        ``iteration`` (1-based, within the parameter) and ``learner``. A
        cyclic fit updates every parameter once per iteration, in parameter
        order; a non-cyclic fit records the parameter it chose at each update.
        """
        from mboostrs._output import output_frame

        paths = [self._parameter_selected_names(k) for k in range(len(self.param_names))]
        if self._method == "noncyclic":
            sequence = [int(k) for k in self._selection_order]
        else:
            sequence = [k for i in range(max((len(p) for p in paths), default=0))
                        for k in range(len(paths)) if i < len(paths[k])]
        seen = [0] * len(paths)
        rows = []
        for order, k in enumerate(sequence, start=1):
            rows.append({"order": order, "parameter": self.param_names[k],
                         "iteration": seen[k] + 1, "learner": paths[k][seen[k]]})
            seen[k] += 1
        return output_frame(rows, output)

    def predict_parameters(self, newdata=None, type=_TYPE_DEFAULT,
                           extrapolation=None, unseen_levels=None) -> dict:
        """Return response-scale predictions by distribution parameter.

        An LSS model returns ``{parameter: array}``. A standard model has
        one linear predictor and returns it as ``{"f": array}``.

        The default is the response/natural scale: each parameter comes
        back as the quantity it is named for — an ``NBinomialLSS`` ``sigma``
        is the size, a ``GaussianLSS`` ``sigma`` a standard deviation, a
        logit parameter a probability — and no link-scale warning fires for
        an omitted ``type``. Pass ``type="link"`` for the linear predictors;
        ``predict()`` keeps the link scale as its own (warning) default.
        ``extrapolation`` and ``unseen_levels`` are as in :meth:`predict`;
        pass ``extrapolation="clip"`` to hold covariates at the training range
        and ``unseen_levels="average"`` to score categories absent from training.
        """
        unseen_levels = _model_unseen_levels(self, unseen_levels)
        if type is _TYPE_DEFAULT:
            type = "response"
        out = self.predict(newdata=newdata, type=type, extrapolation=extrapolation,
                           unseen_levels=unseen_levels)
        return {name: np.asarray(out[name]) for name in self.param_names}

    def _budget(self):
        """The budget that refits this model to its current path.

        A non-cyclic fit takes one total and decides step by step which
        parameter to update, so its budget is the number of updates across
        all parameters. `mstop` reports per-parameter counts and collapses
        equal counts to one integer, which for a non-cyclic fit (45 updates
        of each parameter) would read as a total half the size (45 instead
        of 90). A cyclic fit's budget is `mstop` itself.
        """
        if self._method == "noncyclic":
            return int(sum(len(self._params[k]["selected"])
                           for k in range(len(self.param_names))))
        return self.mstop

    def digest(self) -> str:
        """Return a concise, deterministic summary card of this LSSModel."""
        n_obs = len(self._params[0]["fitted"]) if self._params else 0
        risk_str = f"{float(self._risk[-1]):.6g}" if len(self._risk) else "nan"
        if isinstance(self._formula, dict):
            f_text = "; ".join(f"{p}: {self._formula.get(p, '')}" for p in self.param_names)
        else:
            f_text = str(self._formula)
        lines = [
            f"[mboostrs LSSModel] family={type(self._family).__name__}, n={n_obs}, "
            f"method={self._method}, formulas=({f_text})",
            f"stopping: mstop={self.mstop}, nu={self._nu}, final_risk={risk_str}",
        ]
        for k, pname in enumerate(self.param_names):
            names_k = self._parameter_selected_names(k)
            counts: dict[str, int] = {}
            for nm in names_k:
                counts[nm] = counts.get(nm, 0) + 1
            total_k = len(self._params[k].get("learners")
                          or self._params[k].get("predictor_names") or ())
            ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            shown = ", ".join(f"{nm}({c})" for nm, c in ranked[:6]) or "none"
            if len(ranked) > 6:
                shown += f", (+{len(ranked) - 6} more)"
            lines.append(f"active[{pname}]: {shown}; inactive={max(0, total_k - len(counts))}")
        lines.append(f"diagnostics: fit_warnings={len(self._fit_warnings)}")
        return "\n".join(lines[:25])

    def _param_index(self, parameter) -> int:
        if isinstance(parameter, int):
            if not 0 <= parameter < len(self.param_names):
                raise IndexError(f"parameter index {parameter} out of range")
            return parameter
        try:
            return self.param_names.index(parameter)
        except ValueError as exc:
            raise ValueError(f"unknown parameter {parameter!r}; expected one of {self.param_names}") from exc

    def _parameter_mstop(self, k: int, mstop):
        if mstop is None:
            return None
        if isinstance(mstop, dict):
            return int(mstop[self.param_names[k]])
        if isinstance(mstop, (list, tuple, np.ndarray)):
            if len(mstop) != len(self.param_names):
                raise ValueError("mstop must contain one budget per LSS parameter")
            return int(mstop[k])
        value = int(mstop)
        if value < 0:
            raise ValueError("mstop must be non-negative")
        if self._method == "noncyclic":
            prefix = self._selection_order[:value]
            return sum(parameter == k for parameter in prefix)
        return value

    def _interaction_labels(self):
        """Each ``%X%`` term's derived column name mapped to the term as written."""
        return {ix.derived: ix.name for ix in getattr(self, "_interactions", None) or ()
                if str(ix.derived).startswith(("__ix", "__by"))}

    def _learner_labels(self, k: int) -> list[str]:
        """Parameter ``k``'s learner names as reported: a ``%X%`` pair, fitted
        as a derived design (``__ix0``), or a factor ``bols`` lowered to a
        derived design (``__by0``), is named by the term the user wrote; a
        categorical column's learner is named by its term as written too,
        as gamboost names it."""
        labels = self._interaction_labels()
        factor = self._params[k].get("factor_labels") or {}
        names = [labels.get(name, factor.get(name, name))
                 for name in self._params[k]["predictor_names"]]
        # The observed and filled parts of a switched impute_model term share a
        # column name; they are labelled by part, e.g. "bbs(x) [observed]".
        from mboostrs.transforms import switched_term_labels

        switched = switched_term_labels(self._params[k].get("learners") or [],
                                        getattr(self, "_transform_recipe", None))
        return [switched.get(position, name) for position, name in enumerate(names)]

    def _parameter_selected_names(self, k: int) -> list[str]:
        p = self._params[k]
        names = self._learner_labels(k)
        if p.get("kind") != "gam":
            names = ["(Intercept)", *names]
        return [names[int(sel)] for sel in p["selected"]]

    @property
    def selected(self):
        """Selected base-learners at each iteration, as ``{parameter: path}``.

        The call forms `fit.selected(parameter=...)` and
        `fit.selected(merge=True)` still work and warn; they are
        `fit.selected_which(parameter=...)` and
        `fit.selected_which(merge=True)`.
        """
        from mboostrs._compat import still_callable

        return still_callable(self._selected_by_parameter(),
                              call=self._selected_call,
                              deprecation="lss-selected-call")

    def _selected_by_parameter(self) -> dict:
        return {
            name: self._parameter_selected_names(k)
            for k, name in enumerate(self.param_names)
        }

    def _selected_call(self, parameter=None, merge=False):
        """The deprecated call form of :attr:`selected`."""
        if parameter is not None:
            return self._parameter_selected_names(self._param_index(parameter))
        by_param = self._selected_by_parameter()
        if not merge:
            return by_param
        return _merge_selected(by_param)

    def fitted(self, parameter=None, mstop=None, type="link"):
        """In-sample fitted values, mirroring ``predict(newdata=None, ...)``."""
        return self.predict(parameter=parameter, type=type, mstop=mstop)

    def _replay_newdata(self, newdata, unseen_levels="error"):
        """New rows with the predictor recipe applied, so derived columns
        (`log(x)`, factors) exist before any learner reads its column.

        The response step is not in the recipe: prediction data needs no
        response, and the model carries that step separately. Under
        ``unseen_levels="average"`` or ``"zero_effect"`` a label the fit never
        saw is let through in a factor column that only ``brandom`` and
        ``btree`` terms read, in any parameter, as `GAMBoostModel.predict`
        does; the learners then apply the policy. A label that differs from
        a fitted one only in dtype (fitted as strings, arriving as
        integers) is refused under every policy, because it is not a new
        level and the policy would silently average or zero it (#334).
        """
        recipe = getattr(self, "_transform_recipe", None)
        if recipe and newdata is not None:
            from mboostrs.transforms import refuse_unfilled_missing

            refuse_unfilled_missing(self, recipe, newdata)
            columns = frozenset()
            if unseen_levels in ("zero_effect", "average"):
                from mboostrs.glmboost import _brandom_only_columns

                learners = [spec for parameter in getattr(self, "_params", None) or ()
                            for spec in parameter.get("learners") or ()]
                allowed = set(_brandom_only_columns(learners))
                for parameter in getattr(self, "_params", None) or ():
                    for interaction in parameter.get("interactions") or ():
                        for margin in (interaction.left, interaction.right):
                            if margin.kind in ("contrasts", "brandom", "byfactor"):
                                allowed.add(margin.column)
                columns = frozenset(allowed)
            return apply_recipe(recipe, newdata, unseen_level_columns=columns,
                                unseen_levels=unseen_levels)
        return newdata

    def _transformed_response(self, data):
        """The observed response on the scale the family was fitted to.

        With no response transform this is the column unchanged. With
        `log(rent) ~ ...` it is log(rent): a PIT and a residual compare the
        observed outcome with the fitted distribution, and that distribution
        lives on the transformed scale.
        """
        step = getattr(self, "_response_transform", None)
        if step is None:
            return self._observed_response(data)
        # The stored response name is the derived column the fit read; the
        # outcome the caller measured is the step's source column, and the
        # forward transform maps it to the fitted scale.
        from mboostrs.transforms import response_forward

        y = np.asarray(_extract_column(data, step.source), dtype=np.float64)
        return np.asarray(response_forward(step, y), dtype=np.float64)

    def params(self, newdata=None, type="response", mstop=None, buser_newdata=None,
               unseen_levels=None, extrapolation=None):
        """Predicted LSS parameters for all family parameters.

        ``type="response"`` returns distribution parameters such as ``sigma`` on
        their natural scale. ``type="link"`` returns the linear predictors.
        ``unseen_levels`` is as in :meth:`predict`: how a row whose
        ``brandom`` level the fit never saw is predicted — ``"average"`` as a
        member of an average group, ``"zero_effect"`` with a zero coefficient,
        ``"error"`` (the default) refused. ``extrapolation`` is as in
        :meth:`predict`; pass ``"clip"`` to hold covariates at the training
        range instead of continuing the spline edge slope.
        """
        unseen_levels = _model_unseen_levels(self, unseen_levels)
        return self.predict(
            newdata=newdata,
            type=type,
            mstop=mstop,
            buser_newdata=buser_newdata,
            unseen_levels=unseen_levels,
            extrapolation=extrapolation,
        )

    def predict_all(
        self,
        newdata=None,
        type="response",
        mstop=None,
        output="auto",
        include_response=True,
        buser_newdata=None,
        unseen_levels=None,
        extrapolation=None,
    ):
        """All-parameter prediction table, analogous to GAMLSS ``predictAll``.

        Parameters
        ----------
        output : {"auto", "pandas", "polars", "records", "dict", "dataframe"}
            ``"dataframe"`` is a deprecated alias for ``"auto"``.
        unseen_levels : {"average", "zero_effect", "error"}
            As in :meth:`predict`; ``"average"`` predicts a group the fit
            never saw as the average group.
        extrapolation : {"warn", "clip", "error", "allow"}, default "warn"
            As in :meth:`predict`, applied to every parameter's column.
        """
        unseen_levels = _model_unseen_levels(self, unseen_levels)
        if output == "dataframe":
            output = "auto"
        params = self.params(
            newdata=newdata,
            type=type,
            mstop=mstop,
            buser_newdata=buser_newdata,
            unseen_levels=unseen_levels,
            extrapolation=extrapolation,
        )
        if output == "dict":
            return {name: np.asarray(value, dtype=np.float64).copy() for name, value in params.items()}
        if output not in ("auto", "pandas", "polars", "records"):
            raise ValueError("output must be 'auto', 'pandas', 'polars', 'records', or 'dict'")

        first_param = np.asarray(next(iter(params.values())), dtype=np.float64)
        rows = []
        y = None
        if include_response:
            try:
                y = self._observed_response(self._data if newdata is None else newdata)
            except (KeyError, ValueError, NotImplementedError):
                y = None
        for i in range(first_param.size):
            row = {"index": i}
            if y is not None:
                row["y"] = float(y.ravel()[i])
            for name, value in params.items():
                row[name] = float(np.asarray(value, dtype=np.float64).ravel()[i])
            rows.append(row)
        return _records_frame(rows, output)


    def coef_all(self, output="auto"):
        """All-parameter coefficient table, analogous to GAMLSS ``coefAll``.

        Parameters
        ----------
        output : {"auto", "pandas", "polars", "records", "dict", "dataframe"}
            ``"dataframe"`` is a deprecated alias for ``"auto"``.
        """
        if output == "dataframe":
            output = "auto"
        coefs = self.coef()
        if output == "dict":
            return coefs
        if output not in ("auto", "pandas", "polars", "records"):
            raise ValueError("output must be 'auto', 'pandas', 'polars', 'records', or 'dict'")
        rows = []
        for parameter, by_learner in coefs.items():
            for learner, value in by_learner.items():
                arr = np.asarray(value)
                row = {"parameter": parameter, "learner": learner}
                if arr.ndim == 0 or arr.size == 1:
                    row["coefficient"] = float(arr.item())
                else:
                    row["coefficient"] = [float(v) for v in arr.ravel()]
                rows.append(row)
        return _records_frame(rows, output)


    @property
    def risk(self):
        """The joint risk after each iteration (each full cycle for a cyclic
        fit, each update for a non-cyclic one), as a read-only array.

        A property, like `risk_` and like the standard classes' `risk`.
        Calling it, ``fit.risk()``, returns the same values as a fresh writable
        ndarray. The joint negative log-likelihood after each parameter's own
        updates, which gamboostLSS's ``risk()`` reports, is
        :meth:`parameter_risk`.
        """
        from mboostrs._state import callable_readonly

        return callable_readonly(self._risk, call=self._risk_call)

    def parameter_risk(self, merge=False):
        """The joint risk after each parameter's own updates, as
        gamboostLSS's ``risk(merge = FALSE)`` and ``risk(merge = TRUE)``.

        ``merge=False`` returns a dict keyed by parameter name; each array
        starts with the offset-only risk and then holds the joint risk after
        each update of that parameter. A cyclic fit's last parameter entry
        per cycle is the cycle's risk (:attr:`risk`). ``merge=True`` returns
        one array and the parameter each entry belongs to: for a cyclic fit
        the per-parameter arrays interleaved by position, for a non-cyclic
        fit the offset-only risk once per parameter followed by every step's
        risk in order.
        """
        names = list(self.param_names)
        paths = {name: np.asarray(self._params[k].get("risk", ()), dtype=np.float64)
                 for k, name in enumerate(names)}
        if any(len(path) == 0 for path in paths.values()):
            raise ValueError(
                "this model was fitted before per-parameter risk was recorded; "
                "refit it to obtain parameter_risk()")
        if not merge:
            return paths
        if self._method == "noncyclic":
            values = [paths[name][0] for name in names]
            labels = list(names)
            positions = {name: 1 for name in names}
            for k in self._selection_order or ():
                name = names[int(k)]
                values.append(paths[name][positions[name]])
                labels.append(name)
                positions[name] += 1
            return np.asarray(values), labels
        longest = max(len(path) for path in paths.values())
        values, labels = [], []
        for index in range(longest):
            for name in names:
                if index < len(paths[name]):
                    values.append(paths[name][index])
                    labels.append(name)
        return np.asarray(values), labels

    def _risk_call(self, parameter=None, merge=False):
        """What calling :attr:`risk` does: the joint risk, as a fresh array."""
        if parameter is not None:
            raise NotImplementedError(
                "parameter-specific LSS risk paths are parameter_risk(); "
                "risk is the joint risk after each iteration")
        if merge not in (False, True):
            raise ValueError("merge must be a boolean")
        return self._risk.copy()

    def model_weights(self):
        """Observation weights used for fitting."""
        if self._weights is not None:
            return self._weights.copy()
        n = len(self._params[0]["fitted"]) if self._params else 0
        return np.ones(n, dtype=np.float64)

    def _observed_response(self, data):
        resp = self._response_name
        if resp is None:
            raise ValueError("response name is not stored in this LSS model")
        if isinstance(resp, tuple):
            raise NotImplementedError(
                "distribution diagnostics are not implemented for "
                "survival/multivariate LSS responses"
            )
        return np.asarray(_extract_column(data, resp), dtype=np.float64)

    def deviance_increment(self, newdata=None):
        """Per-observation deviance contribution for supported LSS families."""
        if not hasattr(self._family, "deviance_increment"):
            raise NotImplementedError(
                f"{self._family.__class__.__name__} does not expose distribution diagnostics yet.{_CAPABILITY_HINT}"
            )
        data = self._data if newdata is None else newdata
        y = self._observed_response(data)
        params = self.params(newdata=newdata, type="response")
        return self._family.deviance_increment(y, params)

    def centiles(self, probs=(0.025, 0.5, 0.975), newdata=None, mstop=None):
        """Distribution centile table for fitted LSS parameters.

        Returns one row per observation, in fitting order, as a frame with an
        ``index`` column followed by one column per requested probability. A
        column is named for the probability as a **percentage**, with any
        decimal point replaced by an underscore: ``probs=[0.5]`` gives
        ``q50``, ``probs=[0.025, 0.975]`` gives ``q2_5`` and ``q97_5``.

        Select by name rather than position; converting the frame to a bare
        array puts ``index`` in column 0, where a caller reading column 0 as
        the first centile gets row numbers instead.

        Every family that exposes ``quantile()`` supports this, which is all of
        them except ``DirichletLSS`` and ``PIGLSS``; the rest raise
        NotImplementedError naming the family.
        """
        if not hasattr(self._family, "quantile"):
            raise NotImplementedError(
                f"{self._family.__class__.__name__} does not expose quantile diagnostics yet.{_CAPABILITY_HINT}"
            )
        probs_arr = np.asarray(probs, dtype=np.float64)
        if probs_arr.ndim != 1 or np.any((probs_arr <= 0.0) | (probs_arr >= 1.0)):
            raise ValueError("centile probabilities must be a 1-D sequence in (0, 1)")
        params = self.params(newdata=newdata, type="response", mstop=mstop)
        first_param = np.asarray(next(iter(params.values())), dtype=np.float64)
        quantiles = {
            _prob_col(float(prob)): np.asarray(
                self._family.quantile(np.full(first_param.shape, float(prob)), params),
                dtype=np.float64,
            )
            for prob in probs_arr
        }
        rows = []
        for i in range(first_param.size):
            row = {"index": i}
            for col, values in quantiles.items():
                row[col] = float(values.ravel()[i])
            rows.append(row)
        return _records_frame(rows)


    def _marginal_group_sd(self, group_sd, newdata):
        """Validate the marginal-policy contract and return the widened scale.

        A marginal new-group prediction assumes the held-out group's effect
        follows a centered normal distribution on the linear predictor,
        with the standard deviation the caller states. The fitted
        `brandom` coefficients cannot supply it: they are shrunk by the
        ridge and by the boosting path, and on a synthetic frame with a
        true group standard deviation of 1.5 the dispersion of the fitted
        level coefficients was 0.9 — intervals built from it would be far
        too narrow, so the model refuses rather than guess. The caller
        supplies the value from design knowledge or from a purpose-fit
        variance-components model.

        Supported shape: exactly one `brandom` intercept term, in the
        location parameter of a `GaussianLSS` fit. A random slope
        (`by=`), a group term in a non-location parameter, several group
        terms, and non-Gaussian families are refused with the reason —
        each needs a different integration, and the Gaussian convolution
        is the one this release has.
        """
        if group_sd is None:
            raise ValueError(
                "unseen_levels='marginal' needs group_sd=, the standard "
                "deviation of a new group's effect on the location scale. "
                "The fitted brandom coefficients cannot supply it: ridge "
                "and boosting shrink them, and the dispersion of the "
                "fitted level coefficients understates the group spread "
                "(measured 0.9 where the truth was 1.5 on a synthetic "
                "frame). Supply the value from design knowledge or a "
                "variance-components fit.")
        group_sd = float(group_sd)
        if not np.isfinite(group_sd) or group_sd <= 0.0:
            raise ValueError(f"group_sd must be positive, got {group_sd!r}")
        if type(self._family).__name__ != "GaussianLSS":
            raise ValueError(
                "the marginal policy integrates the Gaussian location-scale "
                f"convolution in closed form; {type(self._family).__name__} "
                "would need per-row quadrature and is not supported here")
        from mboostrs.learners import Brandom

        location = self._params[self._param_index(self.param_names[0])]
        terms = [term for term in location["learners"]
                 if isinstance(term, Brandom)]
        if not terms:
            raise ValueError(
                "the marginal policy needs a brandom term in the location "
                "parameter; this fit has none")
        for term in terms:
            if term.by is not None:
                raise ValueError(
                    f"the marginal policy supports a group intercept; "
                    f"{term.name!r} carries a random slope (by=), whose "
                    "multivariate group effect would need a joint "
                    "distribution, not the one-dimensional convolution "
                    "implemented here")
        if len(terms) > 1:
            raise ValueError(
                f"the marginal policy supports one group term; this fit "
                f"has {len(terms)}, whose joint distribution is not the "
                "independent convolution implemented here")
        # A group term in a non-location parameter changes the shape of the
        # response distribution, not just its location; refused above by the
        # family scan unless it sits in the location parameter alone.
        for k, parameter in enumerate(self._params):
            if k == 0:
                continue
            from mboostrs.learners import Brandom as _Brandom

            if any(isinstance(term, _Brandom) for term in parameter["learners"]):
                raise ValueError(
                    f"the marginal policy supports a group term in the "
                    f"location parameter only; parameter "
                    f"{self.param_names[k]} carries one")
        return terms[0], group_sd

    def prediction_interval(self, level=0.95, newdata=None, mstop=None,
                            unseen_levels=None, group_sd=None):
        """Central prediction interval table for the fitted LSS distribution.

        ``unseen_levels`` is as in :meth:`predict`; the interval is for the
        conditional distribution at the point prediction the policy makes,
        and does not integrate the uncertainty of a group the fit never saw.

        ``unseen_levels="marginal"`` changes that: rows whose group the fit
        never saw get the predictive distribution convolved with the
        assumed group effect, ``group_sd=`` being its standard deviation on
        the location scale. For a Gaussian location-scale fit the
        convolution is closed form: the scale of an unseen-group row widens
        from ``sigma(x)`` to ``sqrt(sigma(x)^2 + group_sd^2)``. The
        group-effect distribution is an assumption the caller states, not
        something the fitted brandom coefficients supply — they are shrunk
        by design, and the model refuses to guess their dispersion.
        Rows whose group the fit saw keep their conditional interval.
        """
        unseen_levels = _model_unseen_levels(self, unseen_levels)
        level = float(level)
        if not 0.0 < level < 1.0:
            raise ValueError("level must be in (0, 1)")
        if unseen_levels == "marginal":
            return self._marginal_prediction_interval(
                level=level, newdata=newdata, mstop=mstop,
                group_sd=group_sd)
        family_name = type(self._family).__name__
        if family_name not in _CORE_QUANTILE_FAMILIES:
            raise NotImplementedError(
                f"{family_name} does not expose quantile diagnostics yet.{_CAPABILITY_HINT}"
            )
        if family_name == "GaussianKnownVarLSS":
            raise NotImplementedError(
                "prediction_interval does not take the known sampling variance of "
                "GaussianKnownVarLSS yet; the quantile needs one value per row. "
                "Use the family's quantile(p, params, s2) directly."
            )
        alpha = (1.0 - level) / 2.0
        params = self.params(newdata=newdata, type="response", mstop=mstop,
                             unseen_levels=unseen_levels)
        first_param = np.asarray(next(iter(params.values())), dtype=np.float64)
        def at(prob):
            return np.asarray(_family_quantile(
                self._family, np.full(first_param.shape, prob), params), dtype=np.float64)

        lower, median, upper = at(alpha), at(0.5), at(1.0 - alpha)
        # A quantile commutes with the response transform, so a `log(rent)`
        # fit's interval on the original scale is exactly the inverse
        # transform of its log-scale quantiles.
        step = getattr(self, "_response_transform", None)
        if step is not None:
            lower = np.asarray(response_inverse(step, lower), dtype=np.float64)
            median = np.asarray(response_inverse(step, median), dtype=np.float64)
            upper = np.asarray(response_inverse(step, upper), dtype=np.float64)
        rows = [
            {
                "index": i,
                "lower": float(lower.ravel()[i]),
                "median": float(median.ravel()[i]),
                "upper": float(upper.ravel()[i]),
                "level": level,
                "lower_prob": alpha,
                "upper_prob": 1.0 - alpha,
            }
            for i in range(first_param.size)
        ]
        return _records_frame(rows)


    def _marginal_prediction_interval(self, level, newdata, mstop, group_sd):
        """Interval table for ``unseen_levels="marginal"``.

        The point prediction for every row is the ``"zero_effect"`` one: an
        unseen group gets a zero ``brandom`` coefficient. That is the mean of
        the assumed group effect, which is centered at zero, so the location
        of an unseen row is the population level. Rows whose group the fit
        saw are unchanged. For an unseen row the scale is widened from
        ``sigma`` to ``sqrt(sigma^2 + group_sd^2)``, the standard deviation of
        a Gaussian convolved with an independent ``N(0, group_sd^2)`` location
        effect. The quantiles come from the same core route as the conditional
        interval.
        """
        term, group_sd = self._marginal_group_sd(group_sd, newdata)
        alpha = (1.0 - level) / 2.0
        params = self.params(newdata=newdata, type="response", mstop=mstop,
                             unseen_levels="zero_effect")
        mu_name, sigma_name = self._family.param_names[0], self._family.param_names[1]
        mu = np.asarray(params[mu_name], dtype=np.float64)
        sigma = np.asarray(params[sigma_name], dtype=np.float64)
        # A row is unseen when its group value is absent from the training
        # rows. With newdata=None the rows are the training rows, all seen.
        if newdata is None:
            unseen = np.zeros(mu.shape, dtype=bool)
        else:
            trained = _extract_column(self._replay_newdata(self._data, "zero_effect"), term.name)
            current = _extract_column(self._replay_newdata(newdata, "zero_effect"), term.name)
            unseen = (~np.isin(current, trained)).reshape(mu.shape)
        scale = np.where(unseen, np.sqrt(sigma ** 2 + group_sd ** 2), sigma)
        quantile_params = dict(params)
        quantile_params[sigma_name] = scale

        def at(prob):
            return np.asarray(_family_quantile(
                self._family, np.full(mu.shape, prob), quantile_params), dtype=np.float64)

        lower, median, upper = at(alpha), at(0.5), at(1.0 - alpha)
        # Same back-mapping as the conditional interval: quantiles commute
        # with the response transform.
        step = getattr(self, "_response_transform", None)
        if step is not None:
            lower = np.asarray(response_inverse(step, lower), dtype=np.float64)
            median = np.asarray(response_inverse(step, median), dtype=np.float64)
            upper = np.asarray(response_inverse(step, upper), dtype=np.float64)
        rows = [
            {
                "index": i,
                "lower": float(lower.ravel()[i]),
                "median": float(median.ravel()[i]),
                "upper": float(upper.ravel()[i]),
                "level": level,
                "lower_prob": alpha,
                "upper_prob": 1.0 - alpha,
            }
            for i in range(mu.size)
        ]
        return _records_frame(rows)

    def predint(self, level=0.95, newdata=None, mstop=None, unseen_levels=None):
        """Alias for :meth:`prediction_interval`, matching R ``predint`` naming."""
        unseen_levels = _model_unseen_levels(self, unseen_levels)
        return self.prediction_interval(level=level, newdata=newdata, mstop=mstop,
                                        unseen_levels=unseen_levels)

    def PI(self, level=0.95, newdata=None, mstop=None, unseen_levels=None):  # noqa: N802 - R parity alias
        """Alias for :meth:`prediction_interval`, matching R ``PI`` naming."""
        return self.prediction_interval(level=level, newdata=newdata, mstop=mstop,
                                         unseen_levels=unseen_levels)

    def pit(self, newdata=None, mstop=None, randomized=True, random_state=None,
            unseen_levels=None):
        """Probability integral transform values for fitted LSS distributions."""
        unseen_levels = _model_unseen_levels(self, unseen_levels)
        unseen_levels = _model_unseen_levels(self, unseen_levels)
        if not hasattr(self._family, "cdf"):
            raise NotImplementedError(
                f"{self._family.__class__.__name__} does not expose CDF diagnostics yet.{_CAPABILITY_HINT}"
            )
        data = self._data if newdata is None else newdata
        # The fitted distribution lives on the transformed scale, so the PIT
        # compares g(y) with it: F_y(y) = F_g(g(y)), with no Jacobian,
        # because a CDF is invariant under a monotone transform.
        y = self._transformed_response(data)
        params = self.params(newdata=newdata, type="response", mstop=mstop,
                             unseen_levels=unseen_levels)
        upper = np.asarray(self._family.cdf(y, params), dtype=np.float64)
        lower = _cdf_left(self._family, y, params)
        if randomized:
            rng = (random_state if isinstance(random_state, np.random.Generator)
                   else np.random.default_rng(random_state))
            u = rng.uniform(lower, upper)
        else:
            u = 0.5 * (lower + upper)
        eps = np.finfo(np.float64).eps
        return np.clip(u, eps, 1.0 - eps)

    def quantile_residuals(self, newdata=None, mstop=None, randomized=True,
                           random_state=None, unseen_levels=None):
        """Randomized quantile residuals for fitted LSS distributions.

        For continuous families this is ``Phi^-1(F(y))``. For discrete
        families, ``randomized=True`` draws uniformly between
        ``F(y - 1)`` and ``F(y)``; ``randomized=False`` uses the midpoint of
        that interval for deterministic diagnostics.
        """
        unseen_levels = _model_unseen_levels(self, unseen_levels)
        if not hasattr(self._family, "cdf"):
            raise NotImplementedError(
                f"{self._family.__class__.__name__} does not expose CDF diagnostics yet.{_CAPABILITY_HINT}"
            )
        return _normal_ppf_array(
            self.pit(newdata=newdata, mstop=mstop, randomized=randomized,
                     random_state=random_state, unseen_levels=unseen_levels)
        )

    def residual_frame(self, kind="quantile", newdata=None, mstop=None,
                       randomized=True, random_state=None, output="auto",
                       unseen_levels=None):
        """Diagnostic residual table for LSS models.

        ``kind="quantile"`` returns randomized quantile residuals and requires a
        family CDF. ``kind="response"`` returns the location response residual
        ``y - mu`` and is refused for survival and multivariate responses,
        exactly as :meth:`residuals` refuses them.
        """
        unseen_levels = _model_unseen_levels(self, unseen_levels)
        if kind not in ("quantile", "response"):
            raise ValueError("kind must be 'quantile' or 'response'")
        if kind == "response":
            declared = lss_residual_kinds(self._family)
            if "response" not in declared:
                raise NotImplementedError(_lss_refuse(self._family, "response"))
        data = self._data if newdata is None else newdata
        # Both residual kinds compare the outcome with the fitted
        # distribution, which lives on the transformed scale.
        y = self._transformed_response(data)
        params = self.params(newdata=newdata, type="response", mstop=mstop,
                             unseen_levels=unseen_levels)
        pit = None
        if hasattr(self._family, "cdf"):
            pit = self.pit(newdata=newdata, mstop=mstop, randomized=randomized,
                           random_state=random_state, unseen_levels=unseen_levels)
        if kind == "quantile":
            if pit is None:
                raise NotImplementedError(
                    f"{self._family.__class__.__name__} does not expose CDF diagnostics yet.{_CAPABILITY_HINT}"
                )
            residual = _normal_ppf_array(pit)
        else:
            residual = y - np.asarray(params[self.param_names[0]], dtype=np.float64)

        deviance = None
        if hasattr(self._family, "deviance_increment"):
            deviance = self._family.deviance_increment(y, params)

        rows = []
        for i in range(len(y)):
            row = {
                "index": i,
                "y": float(y.ravel()[i]),
                "residual_kind": kind,
                "residual": float(np.asarray(residual, dtype=np.float64).ravel()[i]),
            }
            for name, value in params.items():
                row[name] = float(np.asarray(value, dtype=np.float64).ravel()[i])
            if pit is not None:
                row["pit"] = float(np.asarray(pit, dtype=np.float64).ravel()[i])
            if deviance is not None:
                row["deviance"] = float(np.asarray(deviance, dtype=np.float64).ravel()[i])
            rows.append(row)
        return _records_frame(rows, output)

    def worm_data(self, xvar=None, n_intervals=4, newdata=None, mstop=None,
                  randomized=True, random_state=None, output="auto"):
        """Worm-plot data. See :func:`mboostrs.worm_data`.

        The table :meth:`worm_plot` draws from, for a caller who wants the
        numbers rather than the figure.
        """
        from mboostrs.diagnostics import worm_data

        return worm_data(
            self, xvar, n_intervals, newdata, mstop, randomized, random_state,
            output,
        )

    def calibration_frame(self, xvar=None, bins=10, newdata=None, mstop=None,
                          output="auto"):
        """Calibration table. See :func:`mboostrs.calibration_frame`.

        Reports the location parameter's mean per bin, under the name
        ``location_mean``: on a distributional fit there is no single fitted
        vector, and the first parameter is the mean only for some families.
        """
        from mboostrs.diagnostics import calibration_frame

        return calibration_frame(self, xvar, bins, newdata, mstop, output)

    def q_stats(self, xvar=None, n_intervals=10, newdata=None, mstop=None,
                randomized=True, random_state=None, output="auto"):
        """Binned quantile-residual moment summaries.

        A data-table analogue of the descriptive part of GAMLSS ``Q.stats``.
        It uses the same equal-count bins and sample-variance convention as
        :func:`mboostrs.q_stats`.
        """
        from mboostrs.diagnostics import _binned_residual_moments

        data = self._data if newdata is None else newdata
        residuals = np.asarray(
            self.quantile_residuals(
                newdata=newdata,
                mstop=mstop,
                randomized=randomized,
                random_state=random_state,
            ),
            dtype=np.float64,
        )
        if xvar is None:
            x = np.arange(residuals.size, dtype=np.float64)
            xvar_name = "index"
        else:
            x = np.asarray(_extract_column(data, xvar), dtype=np.float64)
            xvar_name = str(xvar)

        rows = _binned_residual_moments(residuals, x, xvar_name, n_intervals)
        return _records_frame(rows, output)

    def density_overlay_data(self, newdata=None, mstop=None, grid=None,
                             n_grid=200, weights=None, output="auto"):
        """Observed and fitted marginal-distribution curves as a table."""
        from mboostrs.diagnostics import density_overlay_data

        return density_overlay_data(
            self, newdata=newdata, mstop=mstop, grid=grid, n_grid=n_grid,
            weights=weights, output=output,
        )

    def centile_fan_data(self, xvar,
                         probs=(0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975),
                         split_by=None, newdata=None, mstop=None, output="auto"):
        """Observed responses and fitted centiles ordered by ``xvar``."""
        from mboostrs.diagnostics import centile_fan_data

        return centile_fan_data(
            self, xvar=xvar, probs=probs, split_by=split_by,
            newdata=newdata, mstop=mstop, output=output,
        )

    def deviance_profile(self, xvar=None, bins=10, newdata=None, mstop=None,
                         weights=None, output="auto"):
        """Binned joint deviance contributions along a numeric covariate."""
        from mboostrs.diagnostics import deviance_profile

        return deviance_profile(
            self, xvar=xvar, bins=bins, newdata=newdata, mstop=mstop,
            weights=weights, output=output,
        )

    def worm_plot(self, xvar=None, n_intervals=4, newdata=None, mstop=None,
                  randomized=True, random_state=None):
        """Worm plot(s) of (randomized) quantile residuals (R's ``gamlss::wp``).

        Thin wrapper around :func:`mboostrs.plot.plot_worm`; see there for
        the full description of the panels.
        """
        from mboostrs.plot import plot_worm

        return plot_worm(
            self, xvar=xvar, n_intervals=n_intervals, newdata=newdata,
            mstop=mstop, randomized=randomized, random_state=random_state,
        )

    def logLik(self):
        """Joint log-likelihood at the current LSS fit.

        A distributional family fits every parameter of a named distribution by
        minimising its negative log-likelihood, so unlike the standard
        families' losses this risk drops no constants and ``-risk`` is the
        log-likelihood exactly.

        For the survival families the density is over ``log(T)``, not ``T``:
        the ``1/t`` Jacobian is absent, as it is in the standard AFT families.
        The number is a proper log-likelihood, but not one comparable with a
        fit of the untransformed time.

        Returns a plain ``float``. Effective degrees of freedom for the
        adaptive LSS fit are not implemented. The update count used by
        :meth:`gaic` is a separate heuristic and must not be reported as
        likelihood degrees of freedom.
        """
        self._logLik_update_count = (
            sum(len(self._params[k]["selected"]) for k in range(len(self._params))) + 1
        )
        self._logLik_df = None
        self._logLik_nobs = len(self.model_weights())
        return float(-self._risk[-1])

    def negative_risk(self):
        """The negated joint risk at the current fit.

        Equal to :meth:`logLik` for every distributional family, because their
        risk is the joint negative log-likelihood. It exists so that a caller
        writing family-agnostic code can ask for the quantity the fit minimised
        without asserting it is a likelihood; on a standard model the two
        differ.
        """
        return float(-self._risk[-1])

    def parity_notes(self, *, include_global=False, output="auto"):
        """Where this fit differs from what its names refer to.

        See :meth:`GLMBoostModel.parity_notes`. A distributional fit collects
        notes from its family and from every learner kind in its formulas.
        """
        from mboostrs._parity_notes import format_notes, notes_for

        notes = notes_for(self, include_global=include_global)
        if output == "text":
            return format_notes(notes)
        records = [
            {
                "name": note.name,
                "mechanism": note.mechanism,
                "consequence": note.consequence,
                "shared_with_upstream": note.shared_with_upstream,
                "evidence": note.evidence,
            }
            for note in notes
        ]
        if output == "records":
            return records
        from mboostrs._output import output_frame

        return output_frame(records, "auto" if output == "auto" else output)

    def likelihood_contract(self):
        """What this family's risk is: a likelihood, a loss, or neither.

        Returns the :class:`~mboostrs._likelihood.FamilyLikelihood` row.
        """
        from mboostrs._likelihood import family_likelihood

        return family_likelihood(self._family)

    def gaic(self, k=2.0):
        """Update-count penalized likelihood heuristic for LSS models.

        The historical name is retained for compatibility. The penalty is
        the number of boosting updates across parameters plus one, not
        effective degrees of freedom. Repeated selection of the same learner
        increases it at every step. It is not a normalized AIC or BIC and
        does not estimate the complexity of adaptive distributional fitting.
        ``gaic_table`` labels it ``lss_update_count_heuristic`` and refuses
        to rank it against standard-model criteria. Prefer held-out scoring
        for family selection.
        """
        from mboostrs._validate import check_positive

        k = check_positive("k", k)
        risk_final = float(self._risk[-1])
        df = sum(len(self._params[k]["selected"]) for k in range(len(self._params))) + 1
        return 2.0 * risk_final + k * df

    def llm_explain_prompt(self, **kwargs):
        """Render this fit as a prompt for a language model.

        One block per distribution parameter. See
        :func:`mboostrs.llm_explain_prompt` for the arguments (``parameter``,
        ``which``, ``top``, ``n_points``, ``dataset_description``, ``output``,
        ...).
        """
        from mboostrs.explain import llm_explain_prompt

        return llm_explain_prompt(self, **kwargs)

    def llm_describe(self, **kwargs):
        """Send this fit's description to a language model and return the reply.

        Makes a network request. See :func:`mboostrs.llm_describe` for the
        arguments (``llm``, ``base_url``, ``api_key``, ``client``, ``reasoning``,
        plus everything :func:`mboostrs.llm_explain_prompt` takes).
        """
        from mboostrs.explain import llm_describe

        return llm_describe(self, **kwargs)

    def to_dict(self, *, include_arrays=False):
        """The ``summary(as_dict=True)`` record plus a ``"provenance"`` entry:
        the library build and environment, and the fingerprint of the
        training data. ``include_arrays`` is accepted for the uniform export
        signature; the summary holds no per-row arrays. A model loaded from
        an artifact keeps no training frame, so its provenance has no
        ``"data"`` entry."""
        from mboostrs._output import export_model

        return export_model(self, training_data=self._data)

    def summary(self, as_dict=False, cv=None):
        """Summarise the fitted LSS model.

        Returns a :class:`~mboostrs.summary.Summary`, which prints as
        fixed-width text and renders as a table in a notebook, with one
        selection table per distribution parameter. Use ``as_dict=True`` for
        the machine-readable form.
        """
        weights = self.model_weights()
        selected_counts = {
            name: len(self._params[k]["selected"])
            for k, name in enumerate(self.param_names)
        }
        info = {
            "model_class": "LSSModel",
            "model_kind": self._model_kind,
            "family": self._family.__class__.__name__,
            "prediction_scale": prediction_scale_text(self._family),
            "parameters": list(self.param_names),
            "mstop": self.mstop,
            "nu": float(self._nu),
            "method": self._method,
            "n": int(len(weights)),
            "risk_initial": float(self._risk[0]),
            "risk_final": float(self._risk[-1]),
            "selected_counts": selected_counts,
            "weights": {
                "sum": float(np.sum(weights)),
                "min": float(np.min(weights)) if len(weights) else 0.0,
                "max": float(np.max(weights)) if len(weights) else 0.0,
            },
        }
        if as_dict:
            return info
        from mboostrs.summary import build_summary  # noqa: PLC0415

        return build_summary(self, cv=cv)

    # ── method forms ─────────────────────────────────────────────────────
    #
    # `GLMBoostModel` and `GAMBoostModel` got method forms for these operations
    # (the `ModelAPIMixin` work); `LSSModel` did not, so a user moving from
    # `gamboost` to `gamboostLSS` had to drop back to module-level functions and
    # name the fit function a second time. The model already knows which one
    # produced it — `_model_kind` is what `update()` switches on — so the
    # methods dispatch internally and the four module-level names stay as thin
    # delegates.

    def _refit_arguments(self):
        """The stored specification, as the module-level entry points take it."""
        if self._formula is None or self._control is None:
            raise ValueError(
                "this model did not retain its formula and control, so it "
                "cannot be refit; fit it again rather than unpickling"
            )
        # A transformed response stores the formula as written (`log(rent)`)
        # while `self._data` carries the derived column the fit read; a refit
        # parses the rewritten formulas against that frame.
        rewritten = getattr(self, "_rewritten_formulas", None)
        arguments = {
            "formula": rewritten if rewritten is not None else self._formula,
            "data": self._data,
            "families": self._family,
            "control": self._control_object(),
            "weight_policy": self._weight_policy,
        }
        # A refit that dropped `bins=` would build the exact design the fit
        # deliberately did not, so `cvrisk`/`stabsel` on a binned model would
        # cost what the binning was chosen to avoid.
        if getattr(self, "_bins", None) is not None and self._model_kind == "gamboostLSS":
            arguments["bins"] = self._bins
        # Per-parameter rates change every step; a refit without them is a
        # different model, exactly as a dropped stabilization would be.
        if getattr(self, "_nu_per_param", None) is not None:
            arguments["nu"] = self._nu_per_param
        # An interaction model's rewritten formulas name `buser` symbols
        # that resolve through `env`; a refit without the matrices cannot
        # rebuild the tensor designs. The linear LSS entry takes no `env`,
        # so this is the mixed frontend's contract.
        if getattr(self, "_env", None) is not None and \
                self._model_kind == "gamboostLSS":
            arguments["env"] = self._env
        # A fit-level `stabilization=` override (or the resolved family value)
        # must reach the refit; the module cvrisk/stabsel functions resolve it
        # the same way, and copying only the family would drop a fit-level
        # override silently.
        resolved = getattr(self, "_stabilization", None)
        if resolved is not None:
            arguments["stabilization"] = resolved
        # The response scale as requested: an automatic scale is taken again
        # from the rows each refit, fold or stability-selection subsample
        # sees, as a fit on those rows would take it. A model saved before the
        # option existed fitted the raw response.
        arguments["response_scale"] = getattr(self, "_response_scale", "none")
        return arguments

    def _control_object(self):
        stored_risk = self._control[2] if len(self._control) > 2 else "inbag"
        stored_trace = self._control[3] if len(self._control) > 3 else False
        return boost_control(self._control[0], self._nu, stored_risk, stored_trace,
                             **step_kwargs(self._control))

    def cvrisk(self, folds=None, grid=None, *, B=5, seed=42, length_out=10,
               unseen_levels="error", **kwargs):
        """Cross-validated risk for this model's own specification.

        Dispatches on the fitting function that produced the model, so
        ``fit.cvrisk()`` replaces
        ``mb.cvrisk_gamboostLSS(formula, data, families, ..., folds=...)`` and
        cannot disagree with the fit about which of the two it is.

        If `folds` or `grid` is omitted, the method builds them from this
        model's weights, step count and the supplied seed.

        `folds` defaults to `B`-fold in-bag weights drawn from this model's
        own weights, and `grid` to the **diagonal**: every parameter given the
        same budget, `length_out` points up to the fitted `mstop`. The
        diagonal is not a shortcut but the cheap direction -- the engine reads
        every diagonal point off one fitted path per fold, where each
        off-diagonal point needs its own fit, because a cyclic path is not
        prefix-consistent. `mb.diagonal_grid` builds the same thing, and
        `LssCvRiskResult.grid_cost()` reports what a given grid cost.

        Parameters
        ----------
        folds : FoldScheme or array_like of shape (n, B), optional
            In-bag weight matrix or scheme. ``None`` draws ``B``-fold k-fold
            weights from the model's weights with ``seed``.
        grid : sequence, optional
            Stopping budgets to evaluate. ``None`` uses the diagonal grid.
        B : int, default 5
            Number of folds when ``folds`` is ``None``.
        seed : int, default 42
            Seed for generated folds.
        length_out : int, default 10
            Number of diagonal grid points when ``grid`` is ``None``.
        unseen_levels : {"error", "average", "zero_effect"}, optional
            ``None`` (default) uses the model's policy: ``"error"``, except
            for a model returned by ``mb.fit``, which predicts an unseen
            level as the average group.
            Policy for factor or group levels held out completely in a fold.
            Passed to held-out prediction in non-cyclic cross-validation.
        **kwargs
            Overrides of the arguments the model is refitted with, passed to
            ``cvrisk_gamboostLSS`` or ``cvrisk_glmboostLSS``.

        Returns
        -------
        LssCvRiskResult
            Cross-validated risk over the grid.

        Raises
        ------
        ValueError
            If ``folds`` is not a two-dimensional matrix or ``unseen_levels``
            is invalid.
        """
        if unseen_levels not in ("error", "average", "zero_effect"):
            raise ValueError(
                "unseen_levels must be 'error', 'average', or 'zero_effect', "
                f"got {unseen_levels!r}"
            )
        n = len(np.asarray(self._params[0]["fitted"], dtype=np.float64))
        if folds is None:
            from mboostrs.cvrisk import cv_weights

            folds = cv_weights(n, type="kfold", B=int(B), seed=int(seed),
                               weights=self.model_weights())
        if self._method == "noncyclic":
            return self._cvrisk_noncyclic(folds, grid, length_out, kwargs,
                                          unseen_levels=unseen_levels)
        if grid is None:
            budget = self.mstop
            largest = max(budget.values()) if isinstance(budget, dict) else budget
            grid = diagonal_grid(int(largest), len(self.param_names),
                                 length_out=int(length_out))
        folds_matrix = np.asarray(folds, dtype=np.float64)
        if folds_matrix.ndim != 2:
            raise ValueError(
                f"folds must be an (n, B) matrix of fold weights, got "
                f"{'a scalar' if folds_matrix.ndim == 0 else f'a {folds_matrix.ndim}-D array'}"
                f". mb.cv_weights(n, type='kfold', B={int(folds) if folds_matrix.ndim == 0 else 5}) "
                f"builds one."
            )
        arguments = self._refit_arguments()
        if getattr(self, "_input_weights", None) is not None:
            # Fold refits and held-out scoring use the model's weights.
            arguments["weights"] = self.model_weights()
        arguments.update(kwargs)
        entry = (
            cvrisk_gamboostLSS
            if self._model_kind == "gamboostLSS"
            else cvrisk_glmboostLSS
        )
        from mboostrs.folds import FoldScheme

        # A scheme travels whole, so its judge mask reaches the scoring.
        return entry(folds=folds if isinstance(folds, FoldScheme) else folds_matrix,
                     grid=grid, **arguments)

    def _cvrisk_search(self, folds, ends, grids, boundary_share):
        """The doubling grid search of `tune_mstop` over this fit's specification.

        Cross-validates on the grids `grids`, one per grid end in `ends`, with
        the folds and arguments :meth:`cvrisk` uses. Each fold is fitted once
        and its path is resumed from one grid to the next, so a grid that
        doubles costs no refit. Returns the dict that
        :func:`cvrisk_gamboostLSS` builds for a search (``risk``, ``curve``,
        ``grid``, ``mstop``, ``at_boundary``, ``ends``).
        """
        if self._model_kind != "gamboostLSS":
            raise NotImplementedError(
                "the doubling search resumes the paths of a gamboostLSS fit")
        arguments = self._refit_arguments()
        if getattr(self, "_input_weights", None) is not None:
            arguments["weights"] = self.model_weights()
        from mboostrs.folds import FoldScheme

        folds_matrix = np.asarray(folds, dtype=np.float64)
        return cvrisk_gamboostLSS(
            folds=folds if isinstance(folds, FoldScheme) else folds_matrix,
            grid=None,
            _search={"ends": list(ends), "grids": [list(g) for g in grids],
                     "boundary_share": float(boundary_share),
                     "method": self._method},
            **arguments)

    def _cvrisk_noncyclic(self, folds, grid, length_out, kwargs, *, unseen_levels="error"):
        """Cross-validate a non-cyclic fit over its total step count.

        A non-cyclic fit takes one budget and chooses, step by step, which
        parameter to update, so the quantity to tune is the total number of
        steps, and its path is prefix-consistent: the fit to 50 steps is the
        first 50 steps of the fit to 200. Each fold is fitted once, to the
        largest total in the grid, and every smaller total is read off that
        path. The held-out risk is the family's negative log-likelihood
        averaged over the held-out rows with the model's weights, the scale
        the cyclic engine reports.
        """
        if kwargs:
            raise TypeError(
                "a non-cyclic LSSModel.cvrisk() refits through update() and "
                f"takes no extra fit arguments; got {sorted(kwargs)}"
            )
        from mboostrs.cvrisk import _fold_design

        design = _fold_design(folds, np.asarray(folds, dtype=np.float64))
        folds_matrix, judge = _lss_fold_matrix(folds, where="non-cyclic LSS cvrisk")
        if folds_matrix.ndim != 2:
            raise ValueError(
                "folds must be an (n, B) matrix of fold weights; "
                "mb.cv_weights(n, type='kfold', B=5) builds one"
            )
        total = self._budget()
        if grid is None:
            totals = [point[0] for point in
                      diagonal_grid(int(total), 1, length_out=int(length_out))]
        else:
            totals = []
            for point in grid:
                values = (list(point) if isinstance(point, (list, tuple, np.ndarray))
                          else [point])
                if len(values) != 1:
                    raise ValueError(
                        "a non-cyclic fit is tuned over its total step count, "
                        "so each grid point is one integer (or a one-element "
                        f"list), got {point!r}; the fit decides how to divide "
                        "each total between the parameters"
                    )
                totals.append(int(values[0]))
        totals = sorted(set(totals))
        largest = max(totals)
        base_weights = self.model_weights()
        if getattr(self, "_input_weights", None) is not None:
            from mboostrs.cvrisk import _with_model_weights

            # A 0/1 fold mask refits with the model's weights, as in cvrisk.
            folds_matrix = _with_model_weights(folds_matrix, base_weights)
        risk = np.empty((folds_matrix.shape[1], len(totals)), dtype=np.float64)
        for b in range(folds_matrix.shape[1]):
            column = folds_matrix[:, b]
            held = (column == 0) & (base_weights > 0)
            if judge is not None:
                held &= judge[:, b]
            if not held.any():
                raise ValueError(
                    f"fold {b} holds out no rows with positive weight, so it "
                    "has no held-out risk to report"
                )
            held_weights = np.where(held, base_weights, 0.0)
            fold_fit = self.update(mstop=largest, weights=column)
            for j, k in enumerate(totals):
                risk[b, j] = fold_fit._weighted_risk(held_weights, mstop=k,
                                                     unseen_levels=unseen_levels)
        mean_risk = risk.mean(axis=0)
        best = totals[int(np.argmin(mean_risk))]
        result = LssCvRiskResult(
            risk_matrix=risk,
            aggregate=mean_risk,
            grid=[[k] for k in totals],
            optimal_mstop=[best],
            # A non-cyclic search has one budget, the total update count.
            param_names=("total",),
            data_identity=record_data_identity(getattr(self, "_comparison_identity", None)),
            provenance=Provenance(
                operation="cvrisk", model=self._model_kind,
                family=type(self._family).__name__, n_obs=len(base_weights),
                mstop=largest, weight_policy=self._weight_policy,
                effective_weight_total=float(base_weights.sum()),
                folds=design,
                n_fits=int(folds_matrix.shape[1]),
            ),
        )
        if result.at_boundary():
            warnings.warn(
                f"the lowest cross-validated risk is at {largest} total steps, "
                "the largest budget in the grid, so the reported optimum is "
                "where the search stopped rather than where the risk turns. "
                "Refit with a larger control mstop, or pass a grid of totals "
                "that extends past it.",
                RuntimeWarning,
                stacklevel=3,
            )
        return result

    def _weighted_risk(self, weights, mstop=None, *, unseen_levels="error"):
        """Weighted mean negative log-likelihood on the training rows."""
        from mboostrs._rust import lss_family_diagnostics

        data = self._data
        etas = [
            np.asarray(self.predict(newdata=data, parameter=name, type="link",
                                    mstop=mstop, unseen_levels=unseen_levels), dtype=np.float64)
            for name in self.param_names
        ]
        if isinstance(self._response_name, tuple):
            y = np.concatenate([
                np.asarray(_extract_column(data, name), dtype=np.float64)
                for name in self._response_name
            ])
        else:
            y = np.asarray(self._observed_response(data), dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        _offsets, _grads, total = lss_family_diagnostics(
            self._family._rust_name, len(self.param_names),
            np.ascontiguousarray(y), [np.ascontiguousarray(e) for e in etas],
            np.ascontiguousarray(weights),
        )
        return float(total) / float(weights.sum())

    def stabsel(self, **kwargs):
        """Stability selection for this model's own specification.

        Replaces ``mb.stabsel_glmboostLSS`` / ``mb.stabsel_gamboostLSS``; see
        those for the error-control parameters.

        The model's own weights go with it. Without that the subsamples were
        drawn with unit weights whatever the fit used, so a weighted model's
        selection frequencies described a sample the caller never fitted --
        and nothing said so. `cvrisk` has always passed them, by building its
        folds from `model_weights()`; this path built its own inside the entry
        point and had no way to receive them.
        """
        arguments = self._refit_arguments()
        arguments["weights"] = self._weights
        arguments.update(kwargs)
        entry = (
            stabsel_gamboostLSS
            if self._model_kind == "gamboostLSS"
            else stabsel_glmboostLSS
        )
        return entry(**arguments)

    def profile_mstop(self, grid=None, metric="risk"):
        """Refit over a grid of stopping points and report `metric` at each."""
        from mboostrs.model_selection import profile_mstop as _profile

        return _profile(self, grid=grid, metric=metric)

    @property
    def control(self):
        """The `boost_control` this model was fitted with."""
        return self._control_object()

    @property
    def offset(self):
        """Per-parameter offsets, as ``{parameter: value}``.

        Same mapping as :attr:`offset_`, with one value per distribution
        parameter.
        """
        return self.offset_

    def plot_risk(self, **kwargs):
        """Plot the joint risk path."""
        from mboostrs import plot as _plot

        return _plot.plot_lss_risk(self, **kwargs)

    def plot_selected(self, **kwargs):
        """Plot selection frequency per base learner and parameter."""
        from mboostrs import plot as _plot

        return _plot.plot_lss_selected(self, **kwargs)

    def plot_density(self, **kwargs):
        """Overlay the observed and average fitted response distributions."""
        from mboostrs import plot as _plot

        return _plot.plot_distribution_density(self, **kwargs)

    def plot_centile_fan(self, xvar, **kwargs):
        """Plot nested fitted centile bands against ``xvar``."""
        from mboostrs import plot as _plot

        return _plot.plot_centile_fan(self, xvar=xvar, **kwargs)

    def plot_deviance_profile(self, **kwargs):
        """Plot binned joint deviance along a covariate or row order."""
        from mboostrs import plot as _plot

        return _plot.plot_deviance_profile(self, **kwargs)

    def continue_to(self, mstop):
        """Return the model grown to a larger `mstop`, per parameter.

        The standard models' `continue_to` grows an existing path; an LSS path
        cannot be extended in place, because each parameter has its own risk
        path and the cyclic schedule interleaves them — which is why
        `mstop<-` for the LSS classes in R delegates to `update()` too. This
        refits, and is exact rather than warm-started.

        `mstop` is either a scalar, broadcast to every parameter, or a mapping
        from parameter name to iteration count.
        """
        if self._method == "noncyclic":
            if isinstance(mstop, Mapping):
                raise ValueError(
                    "a non-cyclic fit is grown by its total number of updates; "
                    "pass one integer")
            total = self._budget()
            if mstop < total:
                raise ValueError(
                    f"continue_to only grows a path: this non-cyclic model has "
                    f"{total} updates in total, cannot go to {mstop}. Use "
                    "with_mstop() to truncate.")
            return self.update(mstop=mstop)
        current = self.mstop
        if isinstance(mstop, Mapping):
            for name, value in mstop.items():
                if name in current and value < current[name]:
                    raise ValueError(
                        f"continue_to only grows a path: parameter {name!r} is "
                        f"already at {current[name]}, cannot go to {value}"
                    )
        else:
            smallest = min(current.values()) if isinstance(current, Mapping) else current
            if mstop < smallest:
                raise ValueError(
                    f"continue_to only grows a path: this model is at {current}, "
                    f"cannot go to {mstop}. Use with_mstop() to truncate."
                )
        return self.update(mstop=mstop)

    def varimp(self, percent=True, parameter=None):
        """Variable importance from in-bag risk reduction, per parameter.

        The standard models attribute each iteration's risk drop to the learner
        selected at that iteration. An LSS fit interleaves updates across
        parameters and, in a cyclic fit, records risk once per *cycle* rather
        than once per update — so this reads `path_frame()`, which already
        resolves that indexing and the per-parameter learner names, and sums
        its `risk_change` per learner.

        Results are keyed by ``"<learner>.<parameter>"``, the same combined name
        stability selection uses. Returns a dict with ``reduction`` and
        ``selfreq``, matching the standard models' shape. ``parameter=``
        restricts to one distribution parameter.

        Requires ``method="noncyclic"``. A cyclic fit records risk once per
        cycle rather than once per update, so there is nothing to attribute
        per learner; it raises `NotImplementedError` saying so rather than
        returning a table in which the dominant predictor scores zero.
        """
        if self._method != "noncyclic":
            # A cyclic fit records the joint risk once per *cycle*, after every
            # parameter has been updated, so there is no per-update risk change
            # to attribute — the whole cycle's drop would land on whichever
            # update happened to come last. Summing it anyway produces a
            # confident-looking table in which the dominant predictor scores
            # 0.0, which is worse than no table. Refusing is the same choice
            # `Family::requires_unit_weights` makes: a clear rejection instead
            # of a quietly wrong answer.
            raise NotImplementedError(
                "varimp needs a risk change per update, and a cyclic LSS fit "
                "records risk once per cycle. Refit with method='noncyclic' "
                "for per-update risk attribution, or use selected() and "
                "path_frame() for selection frequencies, which are exact "
                "either way."
            )
        from mboostrs._attribution import attribute_by_name

        rows = [row for row in self.path_frame(output="records")
                if parameter is None or row["parameter"] == parameter]
        keys = [f"{row['learner']}.{row['parameter']}" for row in rows]
        changes = [float(row["risk_change"]) if row.get("risk_change") is not None else 0.0
                   for row in rows]
        reduction, selfreq = attribute_by_name(changes, keys, keys, percent=percent)
        return {"reduction": reduction, "selfreq": selfreq}

    # ── naming and shape accessors ───────────────────────────────────
    #
    # The standard models' spellings, answered per distribution parameter.
    # These were absent (enhancement-hunt 3.2), so a user moving from
    # `gamboost` to `gamboostLSS` lost the vocabulary as well as the methods.

    def base_learner_names(self, parameter=None):
        """Base-learner names, as a `{parameter: [name, ...]}` mapping.

        `parameter=` restricts to one distribution parameter and returns a
        plain list, matching what the standard models return.
        """
        names = {}
        for index, name in enumerate(self.param_names):
            param = self._params[index]
            learners = self._learner_labels(index)
            if "coefs_flat" in param:
                learners = ["(Intercept)", *learners]
            names[str(name)] = learners
        if parameter is None:
            return names
        if parameter not in names:
            raise ValueError(
                f"unknown parameter {parameter!r}; expected one of "
                f"{list(names)}"
            )
        return names[parameter]

    def bnames(self, parameter=None):
        """Alias for :meth:`base_learner_names`."""
        return self.base_learner_names(parameter)

    def variable_names(self, parameter=None):
        """Distinct predictor names, in formula order."""
        names = self.base_learner_names(parameter)
        if isinstance(names, list):
            return [n for n in dict.fromkeys(names) if n != "(Intercept)"]
        return {
            key: [n for n in dict.fromkeys(value) if n != "(Intercept)"]
            for key, value in names.items()
        }

    def selected_which(self, which=None, parameter=None, merge=False):
        """Selected learner names per parameter, optionally filtered.

        `which` accepts a learner name or a list of them. Selections are
        returned as `{parameter: [name, ...]}`, or a plain list when
        `parameter=` names one. `merge=True` flattens the mapping to one list
        of parameter-qualified names such as ``"x.mu"``, which is the form
        that lets two parameters' paths be counted together.
        """
        selections = {
            str(name): list(self._parameter_selected_names(index))
            for index, name in enumerate(self.param_names)
        }
        if which is not None:
            wanted = {which} if isinstance(which, str) else set(which)
            selections = {
                key: [n for n in value if n in wanted]
                for key, value in selections.items()
            }
        if parameter is not None:
            if parameter not in selections:
                raise ValueError(
                    f"unknown parameter {parameter!r}; expected one of "
                    f"{list(selections)}"
                )
            if merge:
                raise ValueError(
                    "merge=True flattens the per-parameter mapping, so it has "
                    "nothing to do once parameter= has picked one out"
                )
            return selections[parameter]
        return _merge_selected(selections) if merge else selections

    @property
    def fitted_(self):
        """Fitted link-scale values per parameter, as a mapping.

        The standard models' `fitted_` is one array; an LSS fit has one per
        distribution parameter, so this mirrors `fitted()`'s shape.
        """
        from mboostrs._state import readonly

        return {
            str(name): readonly(
                np.asarray(self._params[index]["fitted"], dtype=np.float64)
            )
            for index, name in enumerate(self.param_names)
        }

    @property
    def offset_(self):
        """Per-parameter offsets, as a mapping. See also :attr:`offset`."""
        return {
            str(name): float(self._params[index]["offset"])
            for index, name in enumerate(self.param_names)
        }

    def save(self, path, *, indent=None):
        """Write a portable JSON artifact. See :func:`mboostrs.save_model`.

        Load it with :func:`mboostrs.load_model`. A distributional artifact
        carries one learner block per distribution parameter and the family's
        own state; the Rust and R readers score linear and additive models
        only and refuse it by name.
        """
        from mboostrs.serialize import save_model

        return save_model(self, path, indent=indent)

    def with_mstop(self, k):
        """Return a copy refit to ``k`` boosting iterations.

        Equivalent to ``model.update(mstop=k)`` but more readable.
        Scalar ``k`` is broadcast to all parameters.
        """
        return self.update(mstop=k)

    def __getitem__(self, key):
        """``fit[:k]``, the same truncation spelling the standard models use.

        It differs in cost and says so: a standard fit is *sliced*, since its
        path is a prefix of itself, while a cyclic distributional path is not
        prefix-consistent -- freezing one parameter changes the later updates
        of the others -- so this refits to `k`. The result is what a cold fit
        to `k` gives, which is the same guarantee slicing offers; only the
        price differs.

        ``fit[:{"mu": 40, "sigma": 10}]`` gives the parameters different
        budgets, which is the thing a distributional path can be truncated to
        that a single-parameter one cannot.
        """
        if isinstance(key, slice):
            if key.start is not None and key.start != 0:
                raise ValueError(
                    "only slicing from the start (model[:k]) is supported: a "
                    "boosting path has no meaning cut from the middle"
                )
            if key.step is not None:
                raise ValueError("model[:k] takes no step")
            k = key.stop
            if k is None:
                return self
        else:
            k = key
        return self.update(mstop=k)

    def update(self, mstop=None, weights=None, oobweights=None, risk=None, trace=None):
        """Refit the stored LSS model specification.

        Supports changing ``mstop``, ``weights``, ``risk``, and ``trace``. The
        latter two are forwarded to :func:`boost_control`; when omitted, the
        original fit's settings are carried over.

        ``oobweights`` is unsupported because the LSS fit functions do not
        take a separate out-of-bag weight vector. Use ``risk="oobag"`` with
        zero weights on held-out rows instead.
        """
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64).ravel()
            if np.any(weights < 0):
                raise ValueError("weights must be non-negative")
        else:
            weights = self._weights
        if oobweights is not None:
            raise NotImplementedError(
                "LSS update(oobweights=...) is not supported: the LSS fit "
                "functions take no out-of-bag weight vector. Use "
                "risk='oobag' with zero weights on the held-out rows instead."
            )
        if risk is not None and risk not in ("inbag", "oobag", "none"):
            raise ValueError(
                f"risk must be 'inbag', 'oobag', or 'none'; got {risk!r}"
            )
        if mstop is None:
            weights_changed = (
                not np.array_equal(
                    np.asarray(weights, dtype=np.float64),
                    np.asarray(self._weights, dtype=np.float64),
                )
            )
            if not weights_changed and risk is None and trace is None:
                return self
            mstop = self._budget()
        if self._formula is None or self._control is None:
            raise ValueError(
                "update() requires a model fitted by glmboostLSS() or "
                "gamboostLSS() with stored formula/control."
            )

        new_mstop = mstop
        requested_allocation = None
        if self._method == "noncyclic":
            # A non-cyclic fit is given one budget and decides how to spend it
            # between the parameters, so `mstop` *reports* an allocation --
            # `{"mu": 20, "sigma": 10}` -- while the control takes the total.
            # The two are different quantities under one name, and
            # `fit.with_mstop(fit.mstop)` used to raise on a fit's own value.
            # A mapping is read as the allocation it is: refit to its total,
            # then check the algorithm actually spends it that way.
            if isinstance(new_mstop, dict):
                requested_allocation = {str(name): int(value)
                                        for name, value in new_mstop.items()}
                control_mstop = sum(requested_allocation.values())
            elif (isinstance(new_mstop, (list, tuple, np.ndarray))
                  and len(new_mstop) == 1 and len(self.param_names) > 1):
                # `LssCvRiskResult.optimal_mstop` for a non-cyclic fit is a
                # one-element list holding the total step count.
                control_mstop = int(new_mstop[0])
            elif isinstance(new_mstop, (list, tuple, np.ndarray)):
                requested_allocation = {
                    name: int(value)
                    for name, value in zip(self.param_names, new_mstop,
                                           strict=True)
                }
                control_mstop = sum(requested_allocation.values())
            else:
                control_mstop = int(new_mstop)
            mstop_arg = None
        elif isinstance(new_mstop, dict):
            control_mstop = max(int(v) for v in new_mstop.values())
            mstop_arg = new_mstop
        elif isinstance(new_mstop, (list, tuple, np.ndarray)):
            control_mstop = max(int(v) for v in new_mstop)
            mstop_arg = [int(v) for v in new_mstop]
        else:
            control_mstop = int(new_mstop)
            mstop_arg = int(new_mstop)

        # Fall back to the original fit's settings when the caller does not
        # override them, so update(mstop=...) alone is unchanged in behaviour.
        stored_risk = self._control[2] if len(self._control) > 2 else "inbag"
        stored_trace = self._control[3] if len(self._control) > 3 else False
        risk_kind = stored_risk if risk is None else risk
        trace_flag = stored_trace if trace is None else bool(trace)
        control = boost_control(control_mstop, self._nu, risk_kind, trace_flag,
                                 **step_kwargs(self._control))
        # `self._data` carries the derived columns the fit read, so a
        # transformed response (`log(rent) ~ ...`) must refit from the
        # rewritten formulas; re-extracting the original formula would
        # collide with the derived column already in the frame.
        formula_for_refit = (getattr(self, "_rewritten_formulas", None)
                             if getattr(self, "_rewritten_formulas", None) is not None
                             else self._formula)
        if self._model_kind == "gamboostLSS":
            refit = gamboostLSS(
                formula_for_refit,
                data=self._data,
                families=self._family,
                control=control,
                offset=self._offset_arg,
                mstop=mstop_arg,
                weights=weights,
                method=self._method,
                bins=getattr(self, "_bins", None),
                env=self._env,
                stabilization=self._stabilization,
                nu=self._nu_per_param,
                response_scale=getattr(self, "_response_scale", "none"),
            )
        else:
            refit = glmboostLSS(
                formula_for_refit,
                data=self._data,
                families=self._family,
                control=control,
                offset=self._offset_arg,
                mstop=mstop_arg,
                method=self._method,
                weights=weights,
                stabilization=self._stabilization,
                nu=self._nu_per_param,
                response_scale=getattr(self, "_response_scale", "none"),
            )
        # The refit parsed rewritten formulas against the processed frame,
        # so it sees no transform to record; the carried contract belongs to
        # the original fit and must survive the truncation.
        refit._transform_recipe = getattr(self, "_transform_recipe", None)
        refit._response_transform = getattr(self, "_response_transform", None)
        refit._rewritten_formulas = getattr(self, "_rewritten_formulas", None)
        # The tensor recipes and the env matrices they rebuild from are part
        # of the same carried contract: the refit's formulas are the
        # rewritten ones, so nothing re-expands them.
        refit._interactions = getattr(self, "_interactions", None)
        if self._env is not None:
            refit._env = self._env
        if requested_allocation is not None:
            produced = refit.mstop
            if produced != requested_allocation:
                raise ValueError(
                    f"a noncyclic fit chooses how to divide its budget "
                    f"between parameters, so an allocation cannot be asked "
                    f"for directly. Spending {sum(requested_allocation.values())} "
                    f"iterations gives {produced}, not {requested_allocation}. "
                    f"Pass the total as a number, or use method='cyclic', "
                    f"which takes a per-parameter mstop."
                )
        return refit

    def _extrapolation_rows(self, learners, x_data, by_data):
        """Training range of each spline learner's covariate, and of its `by=`.

        Returns `(rows, index_of)`; `index_of` maps each row to `("x", position)`
        or `("by", position)`, as :meth:`GAMBoostModel._range_rows` does.
        """
        from mboostrs._extrapolation import row_for

        rows, index_of = [], {}
        for position, (spec, values) in enumerate(zip(learners, x_data, strict=False)):
            if not isinstance(spec, (Bbs, Bns, Bmono, Bconstrained, Bspatial)):
                continue
            label = f"{type(spec).__name__.lower()}({spec.name})"
            by = getattr(spec, "by", None)
            if by is not None:
                label = f"{label}, by={quote_name(by)}"
            if isinstance(spec, Bspatial):
                # Packed as [coordinate 1, coordinate 2]; each margin has its
                # own column's training range.
                half = len(values) // 2
                margins = [
                    (spec.name1, slice(0, half), self._column_range(spec.name1)),
                    (spec.name2, slice(half, None), self._column_range(spec.name2)),
                ]
            else:
                given = getattr(spec, "boundary_knots", None)
                if given is not None:
                    low, high = float(given[0]), float(given[1])
                else:
                    low, high = self._column_range(spec.name)
                margins = [(spec.name, None, (low, high))]
            for variable, part, (low, high) in margins:
                index_of[len(rows)] = ("x", position, part)
                rows.append(row_for(label, variable,
                                    values if part is None else values[part], low, high))
            if by_data is not None and by_data[position] is not None:
                try:
                    by_low, by_high = self._column_range(by)
                except (KeyError, ValueError, TypeError):
                    # No numeric training column for this `by=`: nothing to check.
                    continue
                index_of[len(rows)] = ("by", position, None)
                rows.append(row_for(label, by, by_data[position], by_low, by_high))
        return rows, index_of

    def _column_range(self, name):
        """Training minimum and maximum of one column, as floats."""
        column = np.asarray(_extract_column(self._data, name), dtype=np.float64)
        return float(np.min(column)), float(np.max(column))

    def _extrapolation(self, learners, x_data, by_data, policy, parameter):
        """Apply `extrapolation=` to one parameter's spline learners.

        The same policy and messages as `GAMBoostModel.predict`: each spline
        learner's training interval (its explicit boundary knots, or the
        training column's range) and each `by=` variable's training range
        against the new values; ``"clip"`` returns clamped copies. Returns
        `(x_data, by_data)`.
        """
        from dataclasses import replace

        from mboostrs._extrapolation import apply_policy

        rows, index_of = self._extrapolation_rows(learners, x_data, by_data)
        # The warning names the parameter, so each row carries it as a suffix.
        rows = [replace(row, learner=f"{row.learner} [{parameter}]") for row in rows]
        values = {row: np.asarray(
                      _part_of(x_data if kind == "x" else by_data, position, part),
                      dtype=np.float64)
                  for row, (kind, position, part) in index_of.items()}
        updated = apply_policy(rows, values, policy, stacklevel=5)
        if updated is values:
            return x_data, by_data
        out_x = list(x_data)
        out_by = list(by_data) if by_data is not None else None
        for row, (kind, position, part) in index_of.items():
            target = out_x if kind == "x" else out_by
            target[position] = _merge_part(target[position], part, updated[row]).tolist()
        return out_x, out_by

    def extrapolation_report(self, newdata, output="auto"):
        """Terms and covariates with rows outside their training range, by parameter.

        Columns are those of :meth:`GAMBoostModel.extrapolation_report`, with
        `parameter` naming the distribution parameter whose predictor holds the
        term. Only spline terms appear: `predict(extrapolation=)` applies to
        spline learners, and a parameter whose predictor is linear has no rows.
        """
        from mboostrs._extrapolation import REPORT_COLUMNS, report_records
        from mboostrs._output import output_frame

        raw = newdata
        newdata = self._replay_newdata(newdata, "average")
        newdata, buser_newdata = _replay_interactions(
            self, newdata, None, unseen_levels="average")
        entries = []
        for k, name in enumerate(self.param_names):
            p = self._params[k]
            if p.get("kind") != "gam":
                continue
            learners = p["learners"]
            x_data = [
                _gamboostlss_x_data(newdata, learner, buser_newdata=buser_newdata,
                                    require_external_buser=True)
                for learner in learners
            ]
            by_data = _gamboostlss_by_data(newdata, learners)
            rows, _index = self._extrapolation_rows(learners, x_data, by_data)
            entries.extend((name, "spline basis", row) for row in rows)
            entries.extend((name, "spline basis", row) for row in _tensor_margin_rows(
                self, raw, self._parameter_interactions(learners))[0])
        return output_frame(report_records(entries), output, columns=REPORT_COLUMNS)

    def _parameter_interactions(self, learners):
        """The ``%X%`` terms whose derived column one of ``learners`` reads."""
        derived = {getattr(spec, "name", None) for spec in learners}
        return [ix for ix in getattr(self, "_interactions", None) or ()
                if ix.derived in derived]

    def _eta(self, k: int, newdata, mstop, buser_newdata=None,
             unseen_levels="error", extrapolation="warn") -> np.ndarray:
        p = self._params[k]
        if newdata is None and (mstop is None or mstop >= len(p["selected"])):
            return np.asarray(p["fitted"], dtype=np.float64)
        if p.get("kind") == "gam":
            store = p["learner_store"]
            selected = list(p["selected"])
            coefs = [np.asarray(value, dtype=np.float64).tolist() for value in p["coefs"]]
            if newdata is None:
                return np.asarray(
                    store.predict(p["offset"], selected, coefs, mstop),
                    dtype=np.float64,
                )
            learners = p["learners"]
            x_data = [
                _gamboostlss_x_data(
                    newdata,
                    learner,
                    buser_newdata=buser_newdata,
                    require_external_buser=True,
                )
                for learner in learners
            ]
            by_data = _gamboostlss_by_data(newdata, learners)
            x_data, by_data = self._extrapolation(
                learners, x_data, by_data, extrapolation, self.param_names[k])
            # `predict_newdata_grouped` sums each learner's coefficients over the
            # steps that chose it and evaluates its basis once, where
            # `predict_newdata` rebuilds the basis on every step. Both give the
            # same eta -- every learner but `Btree` is linear in its
            # coefficients, and the grouped path keeps tree steps per-step --
            # but the per-step cost is O(mstop) B-spline evaluations per row,
            # which on a long path is most of the time a `predict(newdata=)`
            # spends. `gamboost.predict` has taken the grouped path since it
            # was added.
            return np.asarray(
                store.predict_newdata_grouped(
                    p["offset"], selected, coefs, x_data, mstop, by_data,
                    unseen_levels=unseen_levels,
                ),
                dtype=np.float64,
            )
        source = newdata if newdata is not None else self._data
        names = p["predictor_names"]
        if names:
            x_matrix = np.column_stack([_extract_column(source, nm) for nm in names])
        else:
            n = len(p["fitted"]) if newdata is None else len(_extract_column(source, self.param_names[0]))
            x_matrix = np.empty((n, 0), dtype=np.float64)
        eta = _predict_glmboost(
            p["offset"], p["selected"], list(p["coefs_flat"]), self._parameter_nu(k),
            np.ascontiguousarray(x_matrix, dtype=np.float64), mstop, p["centering_means"],
        )
        return np.asarray(eta, dtype=np.float64)

    def mean(
        self,
        newdata=None,
        *,
        mstop=None,
        buser_newdata=None,
        unseen_levels=None,
        extrapolation=None,
    ):
        """Expected response under the fitted distribution.

        For a zero-inflated family this is the mixture mean, not the mean of
        the count component: ``ZIPoLSS`` returns ``(1 - sigma) * mu`` and
        ``ZINBLSS`` returns ``(1 - nu) * mu``. ``predict(type="response")``
        still returns the parameters themselves.
        """
        unseen_levels = _model_unseen_levels(self, unseen_levels)
        if not hasattr(self._family, "mean"):
            raise TypeError(
                f"{type(self._family).__name__} does not define a mean of the response"
            )
        predicted = self.predict(
            newdata=newdata,
            type="response",
            mstop=mstop,
            buser_newdata=buser_newdata,
            unseen_levels=unseen_levels,
            extrapolation=extrapolation,
        )
        return np.asarray(self._family.mean(predicted), dtype=np.float64)

    def predict(
        self,
        newdata=None,
        *,
        parameter=None,
        type=_TYPE_DEFAULT,
        mstop=None,
        buser_newdata=None,
        unseen_levels=None,
        extrapolation=None,
    ):
        """Predictions for one distribution parameter, or for all of them.

        ``type="link"`` is the linear predictor. ``type="response"`` applies
        each parameter's link, so for ``ZIPoLSS`` ``mu`` is the Poisson mean
        of the count component and ``sigma`` is the extra-zero probability —
        not the expected count. ``type="mean"`` is that expected response
        (``(1 - sigma) * mu`` for ``ZIPoLSS``); it returns one array and
        ignores ``parameter``.

        Returns an array when ``parameter`` names a parameter or ``type`` is
        ``"mean"``, otherwise a dict keyed by parameter name. ``unseen_levels`` is as in
        :meth:`GAMBoostModel.predict`: ``"average"`` predicts a ``brandom``
        level the fit never saw as an average group, and ``"zero_effect"``
        with a zero coefficient, instead of refusing it.

        ``extrapolation`` is as in :meth:`GAMBoostModel.predict`, applied to
        every parameter's spline learners (``bbs``, ``bns``, ``bmono``,
        constrained splines), to both coordinates of ``bspatial``, and to the
        B-spline margins of ``%X%`` terms. ``btree``, ``bkernel``, ``buser``
        and ``brad`` terms have no training-range boundary, so the policy does
        not apply to them. ``"warn"`` (default) warns when a new value lies
        materially outside a learner's training range, ``"error"`` refuses any
        value outside it, ``"clip"`` holds the effect flat at the boundary, and
        ``"allow"`` continues the basis silently. A spline basis continues at
        its edge slope outside the boundary knots, which the fit does not
        constrain, and a scale or shape parameter extrapolated that way can
        reach implausible values quickly.

        Parameters
        ----------
        newdata : dict or DataFrame, optional
            Rows to predict. ``None`` predicts the training rows.
        parameter : str, optional
            One distribution parameter (for example ``"mu"``). ``None`` returns
            every parameter. Not allowed with ``type="mean"``.
        type : {"link", "response", "mean"}, default "link"
            ``"link"`` is each parameter's linear predictor, ``"response"``
            applies the inverse link, and ``"mean"`` is the expected response
            E[Y]. Leaving ``type`` unset predicts on the link scale and
            emits a one-time ``UserWarning`` per model when that scale is not
            the response scale.
        mstop : int or sequence of int, optional
            Truncate the boosting path. A scalar is a per-parameter budget for
            a cyclic model and a global update budget for a non-cyclic one;
            ``None`` uses the fitted stopping iterations.
        buser_newdata : dict, optional
            Design matrices for ``buser`` learners, which cannot be rebuilt
            from a column of ``newdata``.
        unseen_levels : {"error", "average", "zero_effect"}, optional
            ``None`` (default) uses the model's policy: ``"error"``, except
            for a model returned by ``mb.fit``, which predicts an unseen
            level as the average group.
            Treatment of categorical levels absent from the fit, as in
            :meth:`GAMBoostModel.predict`.
        extrapolation : {"warn", "error", "allow", "clip"}, default "warn"
            Treatment of rows outside a spline learner's training range.

        Returns
        -------
        ndarray or dict of ndarray
            Shape ``(n,)`` arrays, one value per row of ``newdata``. An array
            when ``parameter`` is given or ``type="mean"``; otherwise a dict
            keyed by parameter name.

        Raises
        ------
        ValueError
            If ``type`` or ``extrapolation`` is not one of the listed values,
            or ``parameter`` is combined with ``type="mean"``.
        """
        unseen_levels = _model_unseen_levels(self, unseen_levels)
        extrapolation = _model_extrapolation(self, extrapolation)
        from mboostrs._extrapolation import POLICIES
        from mboostrs.gamboost import _check_unseen_levels

        if type_was_default := type is _TYPE_DEFAULT:
            type = "link"
        unseen_levels = _check_unseen_levels(unseen_levels)
        if extrapolation not in POLICIES:
            raise ValueError(f"extrapolation must be one of {POLICIES}, got {extrapolation!r}")
        if type == "mean":
            if parameter is not None:
                raise ValueError(
                    "type='mean' is the expected response, not one parameter; "
                    "drop parameter= or ask for type='response'"
                )
            return self.mean(
                newdata=newdata, mstop=mstop, buser_newdata=buser_newdata,
                unseen_levels=unseen_levels, extrapolation=extrapolation,
            )
        if type not in ("link", "response"):
            raise ValueError(
                "type must be 'link', 'response', or 'mean', got "
                f"{type!r}. 'response' is each distribution parameter on its "
                "own scale, not the expected response; use type='mean' for E[Y]"
            )
        maybe_warn_link_scale(self, type_was_default, method="predict")
        clamp = _tensor_margin_clamp(self, newdata, extrapolation)
        newdata = self._replay_newdata(newdata, unseen_levels)
        newdata, buser_newdata = _replay_interactions(
            self, newdata, buser_newdata, unseen_levels=unseen_levels, clamp=clamp)
        if parameter is not None:
            k = self._param_index(parameter)
            eta = self._eta(
                k,
                newdata,
                self._parameter_mstop(k, mstop),
                buser_newdata=buser_newdata,
                unseen_levels=unseen_levels,
                extrapolation=extrapolation,
            )
            return self._family.response(k, eta) if type == "response" else eta
        out = {}
        for k, name in enumerate(self.param_names):
            eta = self._eta(
                k,
                newdata,
                self._parameter_mstop(k, mstop),
                buser_newdata=buser_newdata,
                unseen_levels=unseen_levels,
                extrapolation=extrapolation,
            )
            out[name] = self._family.response(k, eta) if type == "response" else eta
        return out

    def partial_effects(
        self,
        parameter,
        which=None,
        newdata=None,
        mstop=None,
        buser_newdata=None,
        unseen_levels="error",
        extrapolation="allow",
    ):
        """Return formula-ordered learner contributions for one LSS parameter.

        Each effect excludes the parameter offset. Adding the returned effect
        vectors to that offset reconstructs the parameter's link predictor.
        ``mstop`` follows :meth:`predict`: a scalar is a per-parameter budget for
        cyclic models and a global update budget for noncyclic models.
        ``unseen_levels`` is as in :meth:`predict`.

        ``extrapolation`` is as in :meth:`predict`, applied to the spline and
        ``bspatial`` terms of ``newdata`` before their effects are evaluated.
        It defaults to ``"allow"``, as :meth:`GAMBoostModel.partial_effects`
        does. The ``%X%`` margins are not checked here: this method reads the
        raw ``newdata`` and does not rebuild the ``%X%`` designs that
        :meth:`predict` builds. Under ``"clip"`` the effect is evaluated at the
        clamped values, while the returned coordinates are the values supplied.
        """
        from mboostrs.gamboost import _check_unseen_levels

        unseen_levels = _check_unseen_levels(unseen_levels)
        k = self._param_index(parameter)
        p = self._params[k]
        if p.get("kind") != "gam":
            raise NotImplementedError(
                "partial_effects() currently requires a gamboostLSS "
                "Bols/Bbs/Bns/Brandom/Bmrf/Bspatial/Bmono/Bconstrained/Btree/"
                "Bkernel/Buser model"
            )
        names = self._learner_labels(k)
        scalar = isinstance(which, (str, int))
        if which is None:
            indices = list(dict.fromkeys(int(value) for value in p["selected"]))
        else:
            requested = [which] if scalar else list(which)
            indices = []
            for value in requested:
                if isinstance(value, int):
                    if not 0 <= value < len(names):
                        raise IndexError(f"learner index {value} out of range")
                    indices.append(value)
                else:
                    # The derived name of a `%X%` pair (`__ix0`) is still accepted.
                    value = self._interaction_labels().get(value, value)
                    try:
                        indices.append(names.index(value))
                    except ValueError as exc:
                        raise ValueError(f"unknown base learner {value!r}") from exc

        limit = self._parameter_mstop(k, mstop)
        limit = len(p["selected"]) if limit is None else min(limit, len(p["selected"]))
        learners = p["learners"]
        source = self._data if newdata is None else newdata
        x_data = [
            _gamboostlss_x_data(
                source,
                learner,
                buser_newdata=buser_newdata,
                require_external_buser=newdata is not None,
            )
            for learner in learners
        ]
        by_data = _gamboostlss_by_data(source, learners) if newdata is not None else None
        if newdata is not None:
            x_data, by_data = self._extrapolation(
                learners, x_data, by_data, extrapolation, self.param_names[k])
        store = p["learner_store"]
        result = {}
        for index in indices:
            step_coefs = [
                np.asarray(p["coefs"][step], dtype=np.float64).tolist()
                for step in range(limit)
                if int(p["selected"][step]) == index
            ]
            selected = [index] * len(step_coefs)
            if newdata is None:
                effect = store.predict(0.0, selected, step_coefs, None)
            else:
                effect = store.predict_newdata_grouped(
                    0.0, selected, step_coefs, x_data, None, by_data,
                    unseen_levels=unseen_levels,
                )
            result[names[index]] = (
                np.asarray(
                    _gamboostlss_effect_coordinates(
                        source,
                        learners[index],
                        buser_newdata=buser_newdata,
                        require_external_buser=newdata is not None,
                    ),
                    dtype=np.float64,
                ),
                np.asarray(effect, dtype=np.float64),
            )
        if scalar:
            return result[names[indices[0]]]
        return result

    def coef(self, parameter=None):
        """Per-parameter aggregated coefficients (sum of nu*coef per learner).

        Returns a mapping from each parameter to learner coefficients, or
        one parameter's mapping when `parameter` is given. Coefficients sum
        the updates after multiplying each by that parameter's learning rate;
        the intercept includes the shifts induced by centered predictors.
        """
        def agg(k):
            p = self._params[k]
            nu = self._parameter_nu(k)
            if p.get("kind") == "gam":
                learner_names = self._learner_labels(k)
                n_learners = len(learner_names)
                acc = {i: None for i in range(n_learners)}
                for sel, coef in zip(p["selected"], p["coefs"], strict=False):
                    c = nu * coef
                    if isinstance(p["learners"][sel], Btree):
                        if acc[sel] is None:
                            acc[sel] = []
                        acc[sel].append(c.copy())
                        continue
                    if acc[sel] is None:
                        acc[sel] = c.copy()
                    else:
                        acc[sel] = acc[sel] + c
                result = {}
                for i in range(n_learners):
                    name = learner_names[i]
                    v = acc[i]
                    if v is None:
                        result[name] = 0.0
                    elif isinstance(v, list):
                        result[name] = v
                    else:
                        result[name] = v.item() if v.ndim == 0 or v.size == 1 else v
                result["(offset)"] = p["offset"]
                return result
            predictors = list(p["predictor_names"])
            names = ["(Intercept)"] + predictors
            totals = dict.fromkeys(names, 0.0)
            for step, sel in enumerate(p["selected"]):
                totals[names[sel]] += nu * p["coefs_flat"][step]

            # Each predictor is fitted on its centred column, so its slope
            # carries an implied shift of -mean_j * beta_j that belongs to the
            # intercept. glmboost folds this in; the LSS path accumulated the
            # intercept learner's own updates only, leaving the reported
            # coefficients unable to reproduce the linear predictor — off by
            # a constant, while predict() stayed correct because the engine
            # applies the centring itself.
            means = p.get("centering_means") or []
            for j, predictor in enumerate(predictors):
                if j < len(means):
                    totals["(Intercept)"] -= float(means[j]) * totals[predictor]

            return {n: v for n, v in totals.items() if v != 0.0}

        if parameter is not None:
            return agg(self._param_index(parameter))
        return {name: agg(k) for k, name in enumerate(self.param_names)}

    def residual_kinds(self) -> tuple[str, ...]:
        """Residual kinds defined for the fitted LSS family.

        ``working`` is always present (the negative gradient exists for every
        parameter); ``response`` needs an uncensored univariate response and a
        response-scale ``mu``; ``pearson`` needs a declared variance function,
        which only ``GaussianLSS`` and ``PoissonLSS`` have.
        """
        return lss_residual_kinds(self._family)

    def residuals(self, parameter=None, type="response", newdata=None, mstop=None):
        """Residuals of the named kind for an LSS parameter, or a refusal.

        ``type`` is checked against the family's declared kinds, so a kind that
        is undefined raises with the alternative instead of returning a
        different quantity under the requested name:

        - ``"response"``: ``y - mu`` with ``mu`` on the location's response
          scale. Only defined for the location parameter, and refused for
          survival (censored) and multivariate (composition) responses.
        - ``"pearson"``: ``(y - mu) / sqrt(V(mu))``, only where a variance
          function is declared: ``GaussianLSS`` divides by ``sigma``,
          ``PoissonLSS`` by ``sqrt(mu)``.
        - ``"working"``: the fitted negative gradient for the named parameter
          -- the quantity the next boosting step actually regresses on,
          computed by the same core routine used by the fitting loop.

        ``newdata`` and ``mstop`` behave as for :meth:`predict`. When
        ``parameter`` is `None` the result is a dict ``{location_name:
        residuals}``; otherwise a plain array.
        """
        if type not in ("response", "pearson", "working"):
            raise ValueError(
                f"type must be one of 'response', 'pearson', 'working', "
                f"got {type!r}"
            )
        if parameter is not None and not isinstance(parameter, str):
            parameter = self.param_names[parameter]

        loc_name = self.param_names[0]
        data = self._data if newdata is None else newdata
        family_name = self._family.__class__.__name__

        if type == "working":
            from mboostrs._rust import lss_family_diagnostics

            k = self._param_index(loc_name if parameter is None else parameter)
            etas = [
                np.asarray(
                    self.predict(newdata=data, parameter=name, type="link",
                                 mstop=mstop),
                    dtype=np.float64,
                )
                for name in self.param_names
            ]
            n = etas[0].size
            if isinstance(self._response_name, tuple):
                # Survival (time, event) and Dirichlet composition responses
                # are stored as concatenated per-column vectors in the core.
                y = np.concatenate([
                    np.asarray(_extract_column(data, name), dtype=np.float64)
                    for name in self._response_name
                ])
            else:
                y = np.asarray(self._observed_response(data), dtype=np.float64)
            weights = np.asarray(self.weights, dtype=np.float64)
            if weights.size != n:
                weights = np.ones(n, dtype=np.float64)
            _offsets, grads, _risk = lss_family_diagnostics(
                self._family._rust_name,
                len(self.param_names),
                np.ascontiguousarray(y),
                [np.ascontiguousarray(e) for e in etas],
                np.ascontiguousarray(weights),
            )
            gradient = np.asarray(grads[k], dtype=np.float64)
            if parameter is None:
                return {loc_name: gradient}
            return gradient

        if isinstance(self._response_name, tuple):
            raise NotImplementedError(
                f"LSS {type} residuals are undefined for {family_name} with "
                f"a survival/multivariate response: the response is censored "
                f"or multicomponent, so y - mu is not a residual. Use "
                f"residual_frame(kind='quantile')/quantile_residuals() for "
                f"the censored contract, or type='working' for a negative "
                f"gradient."
            )

        if parameter is not None and parameter != loc_name:
            raise NotImplementedError(
                f"LSS {type} residuals are only defined for the location "
                f"parameter ({loc_name!r}); score residuals for "
                f"{parameter!r} are not implemented. type='working' gives "
                f"the negative gradient for any parameter."
            )
        declared = lss_residual_kinds(self._family)
        if type not in declared:
            raise NotImplementedError(_lss_refuse(self._family, type))

        y = np.asarray(self._observed_response(data), dtype=np.float64)
        mu = np.asarray(
            self.predict(parameter=loc_name, type="response", newdata=data,
                         mstop=mstop),
            dtype=np.float64,
        )
        y = y[: mu.size]
        if type == "response":
            residual = y - mu
        elif family_name == "GaussianLSS":
            sigma = np.asarray(
                self.predict(parameter="sigma", type="response", newdata=data,
                             mstop=mstop),
                dtype=np.float64,
            )
            residual = (y - mu) / np.maximum(sigma[: mu.size], 1e-10)
        else:  # PoissonLSS
            residual = (y - mu) / np.sqrt(np.maximum(mu, 1e-10))
        if parameter is None:
            return {loc_name: residual}
        return residual

    def __repr__(self) -> str:
        return (
            f"LSSModel(family={self._family.__class__.__name__}, "
            f"params={self.param_names}, mstop={self.mstop})"
        )


@dataclass
class LssCvRiskResult(MappingResultMixin):
    """Cross-validated LSS risk over a per-parameter mstop grid."""

    risk_matrix: np.ndarray
    aggregate: np.ndarray
    grid: list[list[int]]
    optimal_mstop: list[int]
    fold_type: str = "custom"
    #: The families, row count, budget and weight policy this curve was
    #: computed under. ``result.describe()`` prints it.
    provenance: object = None
    #: The distribution parameters, in grid order, so `report()` can name
    #: each parameter's budget.
    param_names: tuple = ()
    #: SHA-256 digest of the observation IDs, observed response and effective
    #: weights the curve was computed on (`_comparison_identity.data_identity`);
    #: empty when the response could not be digested.
    data_identity: str = ""

    _MAPPING_KEYS = ("risk", "aggregate", "grid", "mstop", "optimal_mstop",
                     "fold_type", "risk_matrix")
    _MAPPING_ALIASES = {"risk": "risk_matrix", "mstop": "optimal_mstop"}

    def __post_init__(self):
        self.risk_matrix = np.asarray(self.risk_matrix, dtype=np.float64)
        self.aggregate = np.asarray(self.aggregate, dtype=np.float64)
        self.grid = [[int(v) for v in point] for point in self.grid]
        self.optimal_mstop = [int(v) for v in self.optimal_mstop]

    @property
    def standard_error(self) -> np.ndarray:
        """Standard error of the mean risk at each grid point.

        The across-fold standard deviation divided by the square root of the
        fold count. It describes variation across these folds; because the
        folds share training rows, it is not a confidence bound for variation
        across datasets.
        """
        risk = np.asarray(self.risk_matrix, dtype=np.float64)
        if risk.shape[0] < 2:
            return np.zeros(risk.shape[1])
        return risk.std(axis=0, ddof=1) / np.sqrt(risk.shape[0])

    @property
    def mstop_1se(self) -> list[int]:
        """The one-standard-error rule over the grid.

        Adds the standard error at the minimum to the minimum risk, then
        returns the grid point with the smallest total budget whose risk is
        within that threshold.
        """
        aggregate = np.asarray(self.aggregate, dtype=np.float64)
        best = int(np.argmin(aggregate))
        threshold = aggregate[best] + float(self.standard_error[best])
        within = [index for index in range(len(aggregate))
                  if aggregate[index] <= threshold]
        if not within:
            return list(self.grid[best])
        cheapest = min(within, key=lambda index: sum(self.grid[index]))
        return list(self.grid[cheapest])

    def report(self):
        """Fold design, grid, optimum, held-out score and boundary flag as a
        :class:`mboostrs.cvrisk.CvReport`, the shape a standard result reports
        too; the optimum is ``{parameter: steps}``."""
        from mboostrs.cvrisk import CvReport, _score_label

        grid = tuple(tuple(int(v) for v in point) for point in self.grid)
        best = int(np.argmin(np.asarray(self.aggregate, dtype=np.float64)))
        names = tuple(self.param_names) or tuple(
            f"parameter {k + 1}" for k in range(len(self.optimal_mstop)))
        return CvReport(
            folds=(getattr(self.provenance, "folds", None) or str(self.fold_type)),
            n_folds=int(np.asarray(self.risk_matrix).shape[0]),
            grid=grid,
            optimum=dict(zip(names, (int(v) for v in self.optimal_mstop), strict=False)),
            risk_at_optimum=float(np.asarray(self.aggregate)[best]),
            score=_score_label(self.provenance), at_boundary=bool(self.at_boundary()))

    def at_boundary(self) -> bool:
        """Whether the best grid point is the largest budget searched.

        The minimum lies at a grid point with the largest total budget.
        Extend the grid to check whether risk declines further.
        """
        totals = [sum(point) for point in self.grid]
        if len(totals) < 2:
            return False
        best = int(np.argmin(np.asarray(self.aggregate, dtype=np.float64)))
        return totals[best] == max(totals)

    def grid_cost(self) -> dict:
        """How many fits this grid actually cost, per fold.

        Grid points where every parameter shares one budget are **diagonal**,
        and the engine reads them off a single path fitted to the largest of
        them: one path serves all of them. Every other point needs its own
        fit, because a cyclic LSS path is not prefix-consistent -- freezing
        one parameter changes the later updates of the others, so the state
        after 20 mu-steps and 5 sigma-steps is not any prefix of a path that
        took 20 of each.

        So a grid of ten diagonal points costs one path per fold, and a
        10 x 10 product grid costs a hundred. That is the difference this
        reports.
        """
        diagonal = [point for point in self.grid if len(set(point)) <= 1]
        fresh = len(self.grid) - len(diagonal)
        return {
            "points": len(self.grid),
            "diagonal": len(diagonal),
            "fresh_fits": fresh,
            "paths_per_fold": fresh + (1 if diagonal else 0),
            "folds": int(self.risk_matrix.shape[0]),
            "paths_total": (fresh + (1 if diagonal else 0))
                           * int(self.risk_matrix.shape[0]),
        }

    def to_frame(self, output="auto"):
        rows = []
        for fold in range(self.risk_matrix.shape[0]):
            for grid_index, point in enumerate(self.grid):
                row = {
                    "fold": fold,
                    "grid_index": grid_index,
                    "mstop": tuple(point),
                    "risk": float(self.risk_matrix[fold, grid_index]),
                    "aggregate": float(self.aggregate[grid_index]),
                    "optimal": point == self.optimal_mstop,
                }
                for k, value in enumerate(point):
                    row[f"mstop_{k}"] = int(value)
                rows.append(row)
        columns = ("fold", "grid_index", "mstop", "risk", "aggregate", "optimal")
        columns += tuple(f"mstop_{k}" for k in range(len(self.param_names)))
        return _records_frame(rows, output, columns=columns)


    def plot(self):
        from mboostrs.plot import plot_lss_cvrisk

        return plot_lss_cvrisk(self)

    def __repr__(self) -> str:
        return (
            f"LssCvRiskResult(mstop_opt={self.optimal_mstop}, "
            f"folds={self.risk_matrix.shape[0]}, grid={len(self.grid)})"
        )

    def to_dict(self, *, include_arrays=False):
        """A JSON-safe record; the raw fold-risk matrix only on request."""
        return export_result(
            self,
            include_arrays=include_arrays,
            arrays=("risk_matrix",),
            extras={
                "standard_error": self.standard_error,
                "mstop_1se": self.mstop_1se,
            },
        )


@dataclass
class LssStabSelResult(MappingResultMixin):
    """Stability-selection result for combined LSS learner/parameter effects."""

    phat: np.ndarray
    selected: list[str]
    max_: np.ndarray
    cutoff: float
    q: int
    PFER: float
    learner_names: list[str]
    sampling_type: str = "SS"
    assumption: str = "unimodal"
    #: Initial subsamples the error bound was computed on: complementary
    #: *pairs* under ``sampling_type="SS"``, independent half-samples under
    #: ``"MB"``. This is the caller's ``B``, not the number of fits.
    B: int = 0
    #: Models actually fitted. Under ``sampling_type="SS"`` this is ``2 * B``,
    #: because complementary-pairs sampling needs each half-sample and its
    #: complement for the Shah-Samworth error bound; under ``"MB"`` it is ``B``.
    #: ``B`` alone therefore understates the run by half.
    n_fits: int = 0
    #: The families, row count, budget and weight policy the run was
    #: performed under. ``result.describe()`` prints it.
    provenance: object = None

    _MAPPING_KEYS = ("phat", "selected", "max", "cutoff", "q", "PFER", "names",
                     "learner_names", "sampling_type", "assumption", "B",
                     "n_fits", "max_")
    _MAPPING_ALIASES = {"names": "learner_names"}

    def __post_init__(self):
        self.phat = np.asarray(self.phat, dtype=np.float64)
        self.selected = list(self.selected)
        self.max_ = np.asarray(self.max_, dtype=np.float64)
        self.cutoff = float(self.cutoff)
        self.q = int(self.q)
        self.PFER = float(self.PFER)
        self.learner_names = list(self.learner_names)
        self.B = int(self.B)
        self.n_fits = int(self.n_fits)

    @property
    def max(self):
        return {name: float(value) for name, value in zip(self.learner_names, self.max_, strict=False)}

    @property
    def names(self):
        return self.learner_names

    def to_frame(self, output="auto"):
        return _records_frame(
            [
                {
                    "learner": name,
                    "max": float(max_value),
                    "selected": name in set(self.selected),
                    "cutoff": self.cutoff,
                    "q": self.q,
                    "PFER": self.PFER,
                    "sampling_type": self.sampling_type,
                    "assumption": self.assumption,
                }
                for name, max_value in zip(self.learner_names, self.max_, strict=False)
            ],
            output,
            columns=(
                "learner", "max", "selected", "cutoff", "q", "PFER",
                "sampling_type", "assumption",
            ),
        )

    def plot(self):
        from mboostrs.plot import plot_stabsel

        return plot_stabsel(self)

    def __repr__(self) -> str:
        sel_str = ", ".join(self.selected) if self.selected else "(none)"
        return (
            f"LssStabSelResult(selected=[{sel_str}], "
            f"cutoff={self.cutoff:.2f}, q={self.q}, PFER={self.PFER:.3f}, "
            f"n_fits={self.n_fits})"
        )

    def to_dict(self, *, include_arrays=False):
        """A JSON-safe record; the selection-probability paths only on request."""
        return export_result(
            self,
            include_arrays=include_arrays,
            arrays=("phat", "max_"),
            extras={"max": self.max},
        )


@capture_fit_warnings
@fit_checkpoint
def glmboostLSS(formula, data, *, families=None, family=None, control=None,
                offset=None, mstop=None, method="cyclic", weights=None,
                env=None, weight_policy="upstream", stabilization=None,
                nu=None, na_action="error", response_scale="auto"):
    """Fit a linear GAMLSS by component-wise boosting.

    Workflow, pitfalls and worked analyses: ``mboostrs.agent_guide()``, or
    ``python -m mboostrs guide index`` from a shell. After a fit, run
    ``mboostrs.reference_gap`` to measure what a flexible model still gains.

    ``method="cyclic"`` (default) updates every parameter each iteration;
    ``method="noncyclic"`` updates only the single parameter that most reduces
    the risk each step (R's noncyclic, Thomas et al. 2018), so ``mstop`` is a
    single total-step budget and the per-parameter mstop argument is ignored.

    Parameters
    ----------
    formula : str | dict
        One formula shared by every parameter, or ``{param_name: formula}`` to
        give parameters different predictor sets. The response must be the same
        across parameters.
    data : mapping / DataFrame
    families : LSS family instance, optional
        The family fitted to the response. Defaults to ``GaussianLSS()``.
    family : LSS family instance, optional
        Alias for ``families=``. Pass only one of these arguments.
    control : boost_control tuple (default ``boost_control(100, 0.1)``).
    offset : optional list of per-parameter offset overrides.
    mstop : optional per-parameter step budget (R's mstop vector). ``None`` uses
        ``control``'s scalar mstop for every parameter; an int sets the same for
        all; a list (family order) or ``{param_name: int}`` dict sets per
        parameter. Cyclic: parameter k takes exactly its mstop steps.
    weights : array-like of length n, optional
        Observation weights. Must be non-negative. Default is equal weights.
    stabilization : {"none", "MAD", "L2"}, optional
        Partial-gradient stabilization, upstream's ``stabilize_ngradient``.
        ``None`` (default) uses the family's own ``stabilization=`` setting —
        upstream gamboostLSS sets it on the family constructor
        (``GaussianLSS(stabilization="MAD")``); passing one of the three
        values here overrides the family for this fit, an mboostrs extension.

    ``nu=`` sets per-parameter learning rates, upstream's
    ``boost_control(nu = c(...))`` vector: ``None`` (default) keeps the
    control's scalar, a float recycles it, a list is in family order, and a
    ``{param_name: float}`` dict must name every parameter.
    A lower rate slows updates to that parameter relative to the others. If
    the fitted mean and scale curves show different convergence, compare rate
    choices on the same folds with the joint score; the score alone does not
    identify which parameter accounts for residual structure.

    ``na_action`` handles missing values in the columns any parameter
    formula reads, as in :func:`mboostrs.gamboost`: ``"error"`` (default)
    refuses them, ``"omit"`` drops those rows, and ``"impute"`` fills each
    numeric predictor with its training median plus a missing-value
    indicator term in every parameter that reads it (a categorical
    predictor gets a ``"(missing)"`` level). The imputation is part of the
    model's predictor recipe, so ``predict`` handles new rows with missing
    values the same way.
    """
    # `env` resolves symbols inside learner calls -- the `K` of
    # `bmrf(district, K=K_matrix)`, say. A linear LSS formula has no learner
    # calls, so there is nothing for it to resolve; it was accepted and never
    # read, which reads as support for something that does not happen.
    if env is not None:
        raise TypeError(
            "glmboostLSS() does not use env=: a linear formula names columns "
            "directly and has no learner call whose arguments would need "
            "resolving. Use gamboostLSS(..., env=...) for a formula with "
            "learner calls such as bmrf(district, K=K_matrix)."
        )

    if not _HAS_LSS_ENGINE:
        raise RuntimeError("mboostrs Rust LSS engine (fit_glmboostlss) not available")
    if family is not None and families is not None:
        raise TypeError("Specify family= or families=, not both")
    _family = family if family is not None else (families if families is not None else GaussianLSS())
    from mboostrs.families import require_lss_family

    require_lss_family(_family, "glmboostLSS")
    stabilization = _resolve_stabilization(stabilization, _family)
    nu_override = _resolve_nu_per_param(nu, _family)
    k = _family.n_params
    if control is None:
        control = boost_control(100, 0.1)
    mstop_val, nu_scalar = control[0], control[1]
    trace_bool = bool(control[3]) if len(control) > 3 else False

    # Resolve the per-parameter step budget (None -> scalar mstop for all).
    if mstop is None:
        mstop_pp = None
    elif isinstance(mstop, dict):
        _check_parameter_mapping(mstop, _family, "an mstop mapping")
        mstop_pp = [int(mstop[name]) for name in _family.param_names]
    elif isinstance(mstop, (list, tuple, np.ndarray)):
        mstop_pp = [int(v) for v in mstop]
    else:
        mstop_pp = [int(mstop)] * k

    # Resolve per-parameter formulas.
    if isinstance(formula, dict):
        _check_parameter_mapping(formula, _family, "a formula mapping")
        formulas = [formula[name] for name in _family.param_names]
    else:
        formulas = [formula] * k
    data, weights, _na_dropped = _lss_na_action(
        formulas, data, na_action, weights,
        offset if offset is not None else env, "glmboostLSS")
    _raw_inputs = _retained_raw_inputs(formula, data, env, na_action)

    # A `%X%` tensor design has no linear-column form: one term per
    # predictor is the contract here, and a tensor is one joint surface.
    # The refusal names the supported alternative instead of failing inside
    # the formula parser with a duplicate-argument error.
    if any("%X%" in f for f in formulas):
        raise ValueError(
            "glmboostLSS fits one linear term per predictor and cannot carry "
            "a %X% tensor design; use gamboostLSS for the joint surface, or "
            "the bilinear bols(x, by=z, intercept=False) term")
    _interactions = []

    formulas, data, _transform_recipe, _response_transform = (
        _apply_formula_transforms(formulas, data, kind="glm", na_action=na_action))
    _rewritten_formulas = (dict(zip(_family.param_names, formulas, strict=True))
                           if isinstance(formula, dict) else formulas[0])

    _family = _uncensored_survival_family(_family, formulas, data)
    survival = getattr(_family, "_survival", False)
    multivariate = getattr(_family, "_multivariate", False)
    known_variance = getattr(_family, "_known_variance", False)
    response_name = None
    per_param_names = []
    for f in formulas:
        resp, predictors = parse_formula(f)
        if isinstance(resp, tuple) != (survival or multivariate or known_variance):
            raise ValueError(
                "survival families require a Surv(time, status) response, "
                "DirichletLSS requires a cbind(...) response, "
                "GaussianKnownVarLSS requires an (outcome, s2) tuple response, "
                "and other families require a plain response")
        if response_name is None:
            response_name = resp
        elif resp != response_name:
            raise ValueError("all parameter formulas must share the same response")
        per_param_names.append(list(predictors))

    y, n = _build_lss_response(data, response_name, survival, multivariate, known_variance)

    normalized = normalize_weights(weights, n, policy=weight_policy)
    w = normalized.fit
    comparison_identity = fit_identity(
        data=data, response_name=response_name, response_transform=_response_transform,
        weights=w, weight_policy=normalized.policy, dropped=_na_dropped, family=_family)

    # Concatenated design grouped by parameter; per-param column counts.
    blocks, param_ncols, flat_names = [], [], []
    for names in per_param_names:
        param_ncols.append(len(names))
        flat_names.extend(names)
        if names:
            blocks.append(np.column_stack([_extract_column(data, nm) for nm in names]))
    x_all = (
        np.ascontiguousarray(np.column_stack(blocks), dtype=np.float64)
        if blocks else np.empty((n, 0), dtype=np.float64)
    )

    scale = _check_response_scale(response_scale)
    if method == "noncyclic":
        results, risk, selection_order, execution_report, param_risk = call_fit(
            _fit_glmboostlss_noncyclic,
            x_all, param_ncols, flat_names, y, w, mstop_val, nu_scalar,
            _family._rust_name, offset, trace=trace_bool,
            stabilization=stabilization, nu_per_param=nu_override,
            response_scale=scale, **step_kwargs(control),
        )
    elif method == "cyclic":
        results, risk, selection_order, execution_report, param_risk = call_fit(
            _fit_glmboostlss,
            x_all, param_ncols, flat_names, y, w, mstop_val, nu_scalar,
            _family._rust_name, offset, mstop_pp, trace=trace_bool,
            stabilization=stabilization, nu_per_param=nu_override,
            response_scale=scale, **step_kwargs(control),
        )
    else:
        raise ValueError(f"method must be 'cyclic' or 'noncyclic', got {method!r}")
    execution_report = ExecutionReport.from_mapping(execution_report)
    if trace_bool:
        print(execution_report.format_text(), file=sys.stderr)

    # Reconstruct per-parameter centering means (simple column means, matching
    # the core: gamboostLSS centers every covariate by colMeans regardless of
    # weights; weights shape the normal equations and the offset, not the
    # centering).
    params = []
    col0 = 0
    for kk, names in enumerate(per_param_names):
        offset_k, selected_k, coefs_flat_k, fitted_k = results[kk]
        risk_k = param_risk[kk]
        centering = []
        for j in range(len(names)):
            col = x_all[:, col0 + j]
            centering.append(float(col.mean()))
        col0 += len(names)
        params.append({
            "offset": offset_k,
            "selected": list(selected_k),
            "coefs_flat": list(coefs_flat_k),
            "predictor_names": names,
            "centering_means": centering,
            "fitted": np.asarray(fitted_k, dtype=np.float64),
            "risk": np.asarray(risk_k, dtype=np.float64),
        })

    model = LSSModel(
        _family,
        nu_scalar,
        risk,
        params,
        data,
        response_name=response_name,
        weights=w,
        formula=formula,
        control=control,
        offset=offset,
        method=method,
        model_kind="glmboostLSS",
        execution_report=execution_report,
        selection_order=selection_order,
        nu_per_param=nu_override,
        stabilization=stabilization,
    )
    model._response_scale = scale
    model._response_scale_used = _response_scale_used(scale, _family, y, w)
    model._transform_recipe = _transform_recipe if _transform_recipe else None
    model._response_transform = _response_transform
    model._comparison_identity = comparison_identity
    model._na_dropped = _na_dropped
    model._rewritten_formulas = _rewritten_formulas
    model._raw_inputs = _raw_inputs
    model._interactions = _interactions or None
    model._set_weight_policy(normalized)
    return model


def _check_parameter_mapping(mapping, family, what):
    """Refuse a per-parameter mapping that misses or misnames a parameter.

    Indexing the mapping directly raised a bare ``KeyError: 'df'`` when a
    StudentTLSS formula mapping left out ``df``, and an extra or misspelled
    key (``"sgma"``) was ignored silently.
    """
    names = list(family.param_names)
    keys = [str(key) for key in mapping]
    missing = [name for name in names if name not in mapping]
    unknown = [key for key in keys if key not in names]
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"missing {missing}")
        if unknown:
            parts.append(f"unknown {unknown}")
        raise ValueError(
            f"{what} must name every parameter of {type(family).__name__} "
            f"{names}: {'; '.join(parts)}. Use '<response> ~ 1' for a "
            "parameter held constant.")

def _impute_model_before_tensors(formulas, data, split_filled=False):
    """Plan and apply the model-based fill before the ``%X%`` tensors are expanded.

    A tensor margin reads its column when the term is expanded, so an
    ``na_action="impute_model"`` fill must already be in the data for the
    margin to see the model's values. Every parameter formula gets the same
    predictors, the columns of all of them. Returns the rewritten formulas, the
    data with the filled columns, and the planned steps, which
    `_apply_formula_transforms` takes as ``planned=``. ``split_filled`` is
    the ``na_action="impute_model_split"`` split of `impute_missing`.
    """
    from mboostrs.formula import _formula_columns
    from mboostrs.transforms import TransformRecipe, apply_recipe, impute_missing

    responses = {partition_tilde(f)[0].strip() for f in formulas}
    predictors = list(dict.fromkeys(
        c for f in formulas for c in _formula_columns(f, data) if c not in responses))
    rewritten, steps = [], {}
    for f in formulas:
        new_f, imputed = impute_missing(f, data, learner="bols", model_based=True,
                                        split_filled=split_filled, predictors=predictors)
        rewritten.append(new_f)
        for step in imputed.steps:
            steps.setdefault(step.name, step)
    planned = tuple(steps.values())
    if planned:
        data = apply_recipe(TransformRecipe(planned), data)
    return rewritten, data, planned


def _expand_interaction_formulas(formulas, data, env, na_action="error"):
    """Expand ``%X%`` tensor terms per parameter formula, as `gamboost` does.

    Returns ``(rewritten, data, env, interactions)``: each tensor becomes a
    derived `buser` term whose design matrix rides in `env` (a placeholder
    zero column is added to `data` for the term's name), and the interaction
    recipes are appended for the model to rebuild designs on new rows.
    Formulas without a ``%X%`` pass through unchanged, with `env` copied only
    when a matrix actually joins it.
    """
    rewritten = []
    interactions = []
    for f in formulas:
        if "%X%" not in f:
            rewritten.append(f)
            continue
        from mboostrs.interactions import expand_interactions

        new_f, ix_env, recipes = expand_interactions(f, data, na_action=na_action)
        rewritten.append(new_f)
        interactions.extend(recipes)
        n_rows = next(
            (m.shape[0] for m in ix_env.values() if m is not None), None)
        for name, matrix in ix_env.items():
            if matrix is None:
                if n_rows is None:
                    raise ValueError(
                        "interaction expansion produced no design matrix")
                data = (dict(data) if isinstance(data, dict) else {
                    c: np.asarray(data[c]) for c in data.columns
                })
                data[name] = np.zeros(n_rows)
            else:
                env = dict(env) if env else {}
                env[name] = matrix
    return rewritten, data, env, interactions


def _mark_categorical_trees(per_param_learners, recipe):
    """Give each tree its columns' level counts, as `gamboost` does.

    The recipe codes a factor column 0..L-1 with the training vocabulary; a
    tree that knows a column is categorical splits it into two sets of
    levels, and prediction can average over its levels for an unseen one.
    Without this a gamboostLSS tree split the codes as ordered numbers.
    """
    from mboostrs.learners import Btree

    levels = {step.name: len(step.params[0]) for step in (recipe.steps if recipe else ())
              if step.kind == "factor"}
    for learners in per_param_learners:
        for learner in learners:
            if isinstance(learner, Btree):
                learner.categorical = [levels.get(name, 0) for name in learner.names]


def _lower_lss_factor_by(per_param_learners, data, factor_columns, interactions):
    """Lower each parameter's factor ``by=`` learners as `gamboost` does.

    A factor ``by=`` (and a ``bols(f, ...)`` with arguments) becomes a
    ``buser`` term over the stacked per-level design, with a recipe that
    prediction replays (`interactions.expand_by_factors`). Derived names are
    numbered across parameters so two parameters' terms do not collide.
    Returns ``(per_param_learners, interactions)``.
    """
    from mboostrs.interactions import expand_by_factors

    recipes = list(interactions or [])
    start = sum(1 for r in recipes if r.derived.startswith("__by"))
    lowered_all = []
    for learners in per_param_learners:
        lowered, new, _display = expand_by_factors(learners, data, factor_columns,
                                                   start=start)
        start += len(new)
        recipes.extend(new)
        lowered_all.append(lowered)
    return lowered_all, recipes


def _retained_raw_inputs(formula, data, env, na_action):
    """The formula as written and the data before any transform, for a refit.

    The transforms (medians, log floors, centres) and formula interactions
    (`%X%`) are estimated from the rows they are applied to, so a conformal
    refit on proper-training rows starts from these inputs and re-estimates
    both. A fit with an external matrix `env=` is not kept: a row subset
    would misalign those matrices, so the refit cannot use them.
    """
    if env is not None:
        return None
    kept = dict(data) if isinstance(data, dict) else data
    return {"formula": formula, "data": kept, "na_action": na_action}


def _lss_na_action(formulas, data, na_action, weights, offset, frontend):
    """Apply ``na_action`` across every parameter formula of an LSS fit.

    ``"error"`` refuses a missing categorical predictor by name (numeric
    columns are refused later by the finite check); ``"omit"`` drops every
    row with a missing value in any column any parameter reads; ``"impute"``
    drops only rows whose response is missing (predictors are imputed by
    the recipe). Rows dropped here are dropped from ``weights`` too; an
    ``offset`` cannot be realigned, so dropping rows with one is refused.
    Returns ``(data, weights, dropped)``, where ``dropped`` lists the input rows
    removed, which the fit's comparison identity needs.
    """
    from mboostrs.formula import apply_na_action

    if na_action not in ("error", "omit", "impute", "impute_model", "impute_model_split"):
        raise ValueError(
            "na_action must be 'error', 'omit', 'impute', 'impute_model' or 'impute_model_split', "
            f"got {na_action!r}")
    lhs = partition_tilde(formulas[0])[0].strip()
    rhs = " + ".join(partition_tilde(f)[2].strip() for f in formulas
                     if partition_tilde(f)[2].strip())
    combined = f"{lhs} ~ {rhs}" if rhs else f"{lhs} ~ 1"
    data, dropped = apply_na_action(combined, data, na_action)
    if dropped:
        if offset is not None:
            raise ValueError(
                f"{frontend}: na_action={na_action!r} drops {len(dropped)} row(s) with a "
                "missing value, and a supplied offset or env= matrix cannot be realigned "
                "to the kept rows; drop those rows before calling, offset included")
        if weights is not None:
            weights = np.delete(np.asarray(weights, dtype=float), dropped)
    return data, weights, dropped


def _apply_formula_transforms(formulas, data, kind="gam", na_action="error", planned=()):
    """Rewrite each parameter formula's transforms, materialize the merged
    recipe once, and split the response step out.

    Categorical predictors are encoded as the standard frontends encode them:
    ``kind="glm"`` expands a bare factor term into treatment-contrast
    columns, one component each, as `glmboost` does, and ``kind="gam"``
    lowers ``bols(f)`` to its dummy-coded design and records integer codes
    for ``brandom(f)``, as `gamboost` does. A factor ``by=`` is refused for
    ``kind="glm"``; for ``kind="gam"`` the caller lowers it after parsing
    (`_lower_lss_factor_by`).

    Returns ``(rewritten, data, predictor_recipe, response_step)``. Every
    parameter formula names the same response, so a response transform
    appears once in the merged recipe; the predictor recipe that prediction
    data replays never contains it, and the model carries it separately for
    quantile inversion and density Jacobians.

    ``na_action="impute"`` gives every predictor with a missing value the
    median fill and missing-value indicator of `impute_missing`, in every
    parameter formula that reads it; the steps join the predictor recipe, so
    new rows with missing values are predicted too. Each formula's
    imputation is computed from the incoming data, so a column read by two
    parameters gets its indicator term in both. ``"impute_model"`` does the
    same with a model-based fill of each numeric predictor, regressed on the
    columns of all parameter formulas. ``"impute_model_split"`` adds the
    opt-in observed/filled split of a smooth term on a filled column.
    """
    merged: list = []
    seen: set = set()
    rewritten = []
    frontend = "glmboostLSS" if kind == "glm" else "gamboostLSS"
    from mboostrs.formula import refuse_intercept_control

    for f in formulas:
        refuse_intercept_control(f, data, frontend)
    extracted = []
    # Steps already planned by _impute_model_before_tensors; the formulas
    # carry their indicator terms, and planning again would add duplicates.
    imputed_steps: list = list(planned)
    # The model-based fill of a column regresses it on every column any
    # parameter reads, so each parameter's imputation gets the same predictors.
    model_predictors = None
    if na_action in ("impute_model", "impute_model_split"):
        from mboostrs.formula import _formula_columns

        responses = {partition_tilde(f)[0].strip() for f in formulas}
        model_predictors = list(dict.fromkeys(
            c for f in formulas for c in _formula_columns(f, data)
            if c not in responses))
    for f in formulas:
        # Imputation is planned before the transforms are extracted, so the
        # fill of x runs before a transform such as log(x) reads it.
        if na_action in ("impute", "impute_model", "impute_model_split"):
            f, imputed = impute_missing(f, data, learner="" if kind == "glm" else "bols",
                                        model_based=na_action in ("impute_model", "impute_model_split"),
                                        split_filled=na_action == "impute_model_split",
                                        predictors=model_predictors)
            for step in imputed.steps:
                if all(step.name != s.name for s in imputed_steps):
                    imputed_steps.append(step)
        new_f, recipe = extract_transforms(f, data)
        extracted.append((new_f, recipe))
    if imputed_steps:
        # The factor encoding below reads its levels from `data`, so a
        # categorical column's "(missing)" level must already be in place.
        data = apply_recipe(TransformRecipe(tuple(imputed_steps)), data)
        for step in imputed_steps:
            seen.add((step.name, step.kind))
            merged.append(step)
    for new_f, recipe in extracted:
        if kind == "glm":
            reject_factor_by(new_f, data, frontend)
        if kind == "glm":
            new_f, dummies = expand_factor_dummies(new_f, data)
            steps = recipe.steps + dummies.steps
        else:
            new_f = lower_categorical_bols(new_f, data)
            steps = recipe.steps
        # bols(f, ...) on a factor f is lowered to its level design later
        # (_lower_lss_factor_by), as gamboost does; the encoding must not
        # refuse that term first.
        recipe = TransformRecipe(steps + encode_factors(
            new_f, data, factor_bols=kind != "glm").steps)
        rewritten.append(new_f)
        for step in recipe.steps:
            if (step.name, step.kind) not in seen:
                seen.add((step.name, step.kind))
                merged.append(step)
    predictor_recipe = TransformRecipe(tuple(merged))
    response_step = None
    if predictor_recipe:
        predictor_recipe = fit_recipe(predictor_recipe, data)
        data = apply_recipe(predictor_recipe, data)
        from mboostrs.formula import parse_response

        recipe_response = parse_response(rewritten[0])
        if isinstance(recipe_response, str):
            response_steps = [s for s in predictor_recipe.steps
                              if s.name == recipe_response]
            if response_steps:
                response_step = response_steps[0]
                predictor_recipe = TransformRecipe(tuple(
                    s for s in predictor_recipe.steps
                    if s.name != recipe_response))
    return rewritten, data, predictor_recipe, response_step


@capture_fit_warnings
@fit_checkpoint
def gamboostLSS(
    formula,
    data,
    *,
    families=None,
    family=None,
    control=None,
    offset=None,
    mstop=None,
    weights=None,
    method="cyclic",
    bins=None,
    env=None,
    weight_policy="upstream",
    stabilization=None,
    nu=None,
    na_action="error",
    response_scale="auto",
):
    """Fit a distributional additive model with linear, spline, and structured learners.

    Workflow, pitfalls and worked analyses: ``mboostrs.agent_guide()``, or
    ``python -m mboostrs guide index`` from a shell. After a fit, run
    ``mboostrs.reference_gap`` to measure what a flexible model still gains.

    Each distribution parameter owns a component-wise learner competition. A
    single formula is reused for every parameter; a mapping keyed by family
    parameter names supplies distinct formulas. ``method="cyclic"`` updates one
    learner in every active parameter per cycle. ``method="noncyclic"`` chooses
    one parameter/learner pair per global boosting step.

    ``bins=`` sets a global quantile-bin count for every ``bols``/``bbs``/
    ``bspatial``/``brandom`` term that does not carry its own. An LSS fit
    prepares the candidate set once per distribution parameter, so binning
    saves a multiple of what it saves on `gamboost`: a 300,000-row,
    six-spline, two-parameter fit measured 2.9 s exact against 0.44 s at
    ``bins=100``, and its stability selection 82 s against 1.0 s, on the same
    selected set. Terms with no binned design keep their exact storage and are
    named in a warning; see the ``bins`` note on :func:`mboostrs.gamboost`.

    Partial-gradient stabilization is set on the family
    (``GaussianLSS(stabilization="MAD")``, upstream's spelling); passing
    ``stabilization=`` to this fit overrides the family for this fit only.
    ``nu=`` sets per-parameter learning rates, upstream's
    ``boost_control(nu = c(...))`` vector: ``None`` (default) keeps the
    control's scalar, a float recycles, a list is family order, and a
    ``{param_name: float}`` dict names every parameter.
    A lower rate slows updates to that parameter relative to the others. If
    the fitted mean and scale curves show different convergence, compare rate
    choices on the same folds with the joint score; the score alone does not
    identify which parameter accounts for residual structure.

    ``na_action`` handles missing values in the columns any parameter
    formula reads, as in :func:`mboostrs.gamboost`: ``"error"`` (default)
    refuses them, ``"omit"`` drops those rows, and ``"impute"`` fills each
    numeric predictor with its training median plus a missing-value
    indicator term in every parameter that reads it (a categorical
    predictor gets a ``"(missing)"`` level). The imputation is part of the
    model's predictor recipe, so ``predict`` handles new rows with missing
    values the same way.
    """
    if not _HAS_LSS_ENGINE or _fit_gamboostlss_mixed is None:
        raise RuntimeError("mboostrs Rust mixed-learner LSS engine is not available")
    if family is not None and families is not None:
        raise TypeError("Specify family= or families=, not both")
    if method not in ("cyclic", "noncyclic"):
        raise ValueError("method must be 'cyclic' or 'noncyclic'")
    _family = family if family is not None else (families if families is not None else GaussianLSS())
    from mboostrs.families import require_lss_family

    require_lss_family(_family, "gamboostLSS")
    stabilization = _resolve_stabilization(stabilization, _family)
    nu_override = _resolve_nu_per_param(nu, _family)
    k = _family.n_params
    if control is None:
        control = boost_control(100, 0.1)
    mstop_val, nu_scalar = control[0], control[1]
    trace_bool = bool(control[3]) if len(control) > 3 else False

    if method == "noncyclic":
        if isinstance(mstop, (dict, list, tuple, np.ndarray)):
            raise ValueError("noncyclic gamboostLSS requires a scalar mstop")
        if mstop is not None:
            mstop_val = int(mstop)
        mstop_pp = None
    elif mstop is None:
        mstop_pp = None
    elif isinstance(mstop, dict):
        _check_parameter_mapping(mstop, _family, "an mstop mapping")
        mstop_pp = [int(mstop[name]) for name in _family.param_names]
    elif isinstance(mstop, (list, tuple, np.ndarray)):
        mstop_pp = [int(v) for v in mstop]
    else:
        mstop_pp = [int(mstop)] * k

    if isinstance(formula, dict):
        _check_parameter_mapping(formula, _family, "a formula mapping")
    formulas = ([formula[name] for name in _family.param_names]
                if isinstance(formula, dict) else [formula] * k)
    data, weights, _na_dropped = _lss_na_action(
        formulas, data, na_action, weights,
        offset if offset is not None else env, "gamboostLSS")
    _raw_inputs = _retained_raw_inputs(formula, data, env, na_action)

    _model_steps = ()
    if na_action in ("impute_model", "impute_model_split") and any("%X%" in f for f in formulas):
        formulas, data, _model_steps = _impute_model_before_tensors(
            formulas, data, split_filled=na_action == "impute_model_split")
    _interactions = []
    if any("%X%" in f for f in formulas):
        formulas, data, env, _interactions = _expand_interaction_formulas(
            formulas, data, env, na_action=na_action)

    # Taken before the encoding below turns factor columns into codes; a
    # factor by= is lowered from it after parsing.
    from mboostrs.interactions import factor_columns as _factor_columns_of

    _factor_cols = _factor_columns_of(data)
    # Read before the transforms below encode the categorical columns and
    # rewrite the terms, so the labels carry the terms as the user wrote them.
    _factor_labels_by_param = [factor_term_labels(f, data) for f in formulas]
    formulas, data, _transform_recipe, _response_transform = (
        _apply_formula_transforms(formulas, data, na_action=na_action,
                                    planned=_model_steps))
    _rewritten_formulas = (dict(zip(_family.param_names, formulas, strict=True))
                           if isinstance(formula, dict) else formulas[0])

    _family = _uncensored_survival_family(_family, formulas, data)
    survival = getattr(_family, "_survival", False)
    multivariate = getattr(_family, "_multivariate", False)
    known_variance = getattr(_family, "_known_variance", False)
    response_name = None
    per_param_learners = []
    for f in formulas:
        resp, learners = parse_gamboost_formula(f, env=env)
        if isinstance(resp, tuple) != (survival or multivariate or known_variance):
            raise ValueError(
                "survival families require a Surv(time, status) response, "
                "DirichletLSS requires a cbind(...) response, "
                "GaussianKnownVarLSS requires an (outcome, s2) tuple response, "
                "and other families require a plain response")
        if response_name is None:
            response_name = resp
        elif resp != response_name:
            raise ValueError("all parameter formulas must share the same response")
        per_param_learners.append(learners)
    per_param_learners, _interactions = _lower_lss_factor_by(
        per_param_learners, data, _factor_cols, _interactions)
    _mark_categorical_trees(per_param_learners, _transform_recipe)
    per_param_learners = _apply_lss_global_bins(per_param_learners, bins)

    y, n = _build_lss_response(data, response_name, survival, multivariate, known_variance)

    normalized = normalize_weights(weights, n, policy=weight_policy)
    w = normalized.fit
    comparison_identity = fit_identity(
        data=data, response_name=response_name, response_transform=_response_transform,
        weights=w, weight_policy=normalized.policy, dropped=_na_dropped, family=_family)

    learner_specs = _gamboostlss_learner_specs(per_param_learners, data)
    scale = _check_response_scale(response_scale)
    results, risk, selection_order, learner_stores, execution_report, param_risk = (
        call_fit(_fit_gamboostlss_mixed,
            learner_specs,
            y,
            w,
            mstop_val,
            nu_scalar,
            _family._rust_name,
            offset,
            mstop_pp,
            method,
            trace=trace_bool,
            stabilization=stabilization,
            nu_per_param=nu_override,
            response_scale=scale,
            **step_kwargs(control),
        )
    )
    execution_report = ExecutionReport.from_mapping(execution_report)
    if trace_bool:
        print(execution_report.format_text(), file=sys.stderr)

    params = []
    for kk, learners in enumerate(per_param_learners):
        offset_k, selected_k, coefs_k, fitted_k = results[kk]
        risk_k = param_risk[kk]
        params.append({
            "kind": "gam",
            "offset": offset_k,
            "selected": list(selected_k),
            "coefs": [np.asarray(c, dtype=np.float64) for c in coefs_k],
            "predictor_names": [ln.name for ln in learners],
            "factor_labels": _factor_labels_by_param[kk],
            "learners": learners,
            "learner_store": learner_stores[kk],
            "fitted": np.asarray(fitted_k, dtype=np.float64),
            "risk": np.asarray(risk_k, dtype=np.float64),
        })

    model = LSSModel(
        _family,
        nu_scalar,
        risk,
        params,
        data,
        response_name=response_name,
        weights=w,
        formula=formula,
        control=control,
        offset=offset,
        method=method,
        model_kind="gamboostLSS",
        execution_report=execution_report,
        selection_order=selection_order,
        nu_per_param=nu_override,
        bins=bins,
        env=env,
        stabilization=stabilization,
    )
    model._response_scale = scale
    model._response_scale_used = _response_scale_used(scale, _family, y, w)
    model._transform_recipe = _transform_recipe if _transform_recipe else None
    model._response_transform = _response_transform
    model._comparison_identity = comparison_identity
    model._na_dropped = _na_dropped
    model._rewritten_formulas = _rewritten_formulas
    model._raw_inputs = _raw_inputs
    model._interactions = _interactions or None
    model._set_weight_policy(normalized)
    return model


def make_grid_lss(max_mstop, length_out=10, log=True):
    """Per-parameter 1-D mstop grid, mirroring gamboostLSS::make.grid's marginal
    sequence (the Cartesian product over parameters forms the full search grid)."""
    if log:
        g = np.unique(np.round(np.exp(
            np.linspace(np.log(1), np.log(max_mstop), length_out))).astype(int))
    else:
        g = np.unique(np.round(np.linspace(1, max_mstop, length_out)).astype(int))
    return [int(v) for v in g]


def _lss_grid_points(grid, n_params):
    """The cross-validation grid as one integer step count per parameter per point.

    A flat sequence of step counts (``[100, 250, 400]``) is a common grid: each
    point gives every parameter the same count, as `diagonal_grid` does. Any
    other grid is a sequence of points, each a sequence of ``n_params`` counts.
    """
    points = list(grid)
    if points and all(np.ndim(point) == 0 for point in points):
        return [[int(value)] * n_params for value in points]
    out = []
    for point in points:
        if np.ndim(point) != 1 or len(point) != n_params:
            example = ", ".join(["100"] * n_params)
            raise ValueError(
                f"each grid point needs one step count per distribution parameter "
                f"({n_params} here), got {point!r}. Pass a flat list of step counts "
                f"for a common grid, e.g. grid=[100, 250, 400], or one tuple per point, "
                f"e.g. grid=[({example}), ...]. make_grid_lss(max_mstop) returns one "
                f"parameter's sequence, and diagonal_grid(max_mstop, {n_params}) "
                f"builds the whole grid")
        out.append([int(value) for value in point])
    return out


def _warn_lss_grid_boundary(result) -> None:
    """Warn when the best grid point is the largest budget searched.

    The reported optimum then says where the grid ended, not where the risk
    turns, and it reads exactly like a minimum the data chose.
    """
    if not result.at_boundary():
        return
    largest = max(result.grid, key=sum)
    warnings.warn(
        f"the lowest cross-validated risk is at {largest}, the largest budget "
        f"in the grid, so the reported optimum is where the search stopped "
        f"rather than where the risk turns. Extend the grid -- "
        f"mb.diagonal_grid({2 * max(largest)}, {len(largest)}) -- and refit "
        f"with a control whose mstop covers it.",
        RuntimeWarning,
        stacklevel=3,
    )


def diagonal_grid(max_mstop, n_params, length_out=10, log=True):
    """A grid whose points give every distribution parameter the same budget.

    Cross-validating an LSS fit means searching over one `mstop` per
    distribution parameter, and the full Cartesian product of `length_out`
    marginal values costs ``length_out ** n_params`` fits per fold -- 100 for
    two parameters, 1000 for three.

    The diagonal is the subset where all the budgets agree, and it is cheap
    for a mechanical reason: the engine fits **one** path per fold to the
    largest budget and reads every diagonal point off it as a checkpoint. Ten
    diagonal points cost one path per fold rather than ten.

    Nothing off the diagonal can be read off that path. A cyclic LSS fit
    updates the parameters in turn, so freezing one changes what the others do
    next: the state after 20 mu-steps and 5 sigma-steps is not a prefix of any
    path that took 20 of each, and it needs its own fit.

    Start on the diagonal to find the scale, then search a small product grid
    around the winner if the parameters plainly want different budgets --
    which shows up as the diagonal optimum sitting at one end of its range.
    """
    marginal = make_grid_lss(max_mstop, length_out=length_out, log=log)
    return [[int(m)] * int(n_params) for m in marginal]


def _cv_data_identity(data, response_name, response_transform, weights, weight_policy,
                      family) -> str:
    """The `data_identity` of the rows a standalone LSS cvrisk call scores.

    The call drops no rows itself, so the observation IDs are the row
    positions of ``data``. Returns ``""`` when the response cannot be digested.
    """
    return record_data_identity(fit_identity(
        data=data, response_name=response_name, response_transform=response_transform,
        weights=weights, weight_policy=weight_policy, dropped=(), family=family))


def cvrisk_glmboostLSS(formula, data, families=None, family=None, control=None, folds=None,
                       grid=None, weight_policy="upstream", nu=None, stabilization=None,
                       weights=None, response_scale="auto"):
    """Cross-validated risk for a linear GAMLSS over a grid of per-parameter
    mstop combinations (R's ``cvrisk.mboostLSS`` for ``glmboostLSS``).

    Parameters
    ----------
    formula, data, families, control : as in :func:`glmboostLSS`.
    folds : (n, B) array of in-bag weights (one column per fold), a
        :class:`~mboostrs.folds.FoldScheme`, or an integer fold count. An
        integer ``k`` builds k-fold weights with
        ``mb.cv_weights(n, type='kfold', B=k)`` (seed 42). Required.
    grid : list of per-parameter mstop vectors (each length ``family.n_params``).
        The out-of-bag risk is returned for each grid point.

    Returns a dict with ``risk`` (B x len(grid) per-fold OOB risk), ``grid``,
    and ``mstop`` (the grid point minimizing the mean OOB risk).
    

    ``LSSModel.cvrisk()`` is the model-based interface: it obtains the
    specification and default grid from the fitted model. This function
    remains available when folds or a grid must be supplied explicitly.

    ``nu=`` sets per-parameter learning rates, upstream's
    ``boost_control(nu = c(...))`` vector: ``None`` (default) keeps the
    control's scalar, a float recycles it, a list is in family order, and a
    ``{param_name: float}`` dict must name every parameter.
    A lower rate slows updates to that parameter relative to the others. If
    the fitted mean and scale curves show different convergence, compare rate
    choices on the same folds with the joint score; the score alone does not
    identify which parameter accounts for residual structure.
    """
    if not _HAS_LSS_ENGINE or _cvrisk_glmboostlss is None:
        raise RuntimeError("mboostrs Rust LSS engine (cvrisk_glmboostlss) not available")
    if folds is None:
        raise ValueError(
            "folds= is required: pass a fold matrix such as "
            "mb.cv_weights(n, 'kfold', 5), or fit the model and call its "
            ".cvrisk(), which builds folds and a grid itself")
    folds = _resolve_lss_folds(folds, data, where="cvrisk_glmboostLSS")
    from mboostrs.cvrisk import _fold_design

    design = _fold_design(folds, np.asarray(folds, dtype=np.float64))
    folds, _ = _lss_fold_matrix(folds, judge_supported=False,
                                where="cvrisk_glmboostLSS")
    if family is not None and families is not None:
        raise TypeError("Specify family= or families=, not both")
    _family = family if family is not None else (families if families is not None else GaussianLSS())
    k = _family.n_params
    stabilization = _resolve_stabilization(stabilization, _family)
    if control is None:
        control = boost_control(100, 0.1)
    mstop_val, nu_scalar = control[0], control[1]
    if grid is None:
        # The diagonal grid `LSSModel.cvrisk()` uses: every parameter gets the
        # same budget, ten points up to the control's `mstop`, read off one
        # path per fold.
        grid = diagonal_grid(int(mstop_val), k)

    nu_override = _resolve_nu_per_param(nu, _family)
    formulas = ([formula[name] for name in _family.param_names]
                if isinstance(formula, dict) else [formula] * k)
    formulas, data, _transform_recipe, _response_transform = (
        _apply_formula_transforms(formulas, data, kind="glm"))

    _family = _uncensored_survival_family(_family, formulas, data)
    survival = getattr(_family, "_survival", False)
    multivariate = getattr(_family, "_multivariate", False)
    known_variance = getattr(_family, "_known_variance", False)
    response_name = None
    per_param_names = []
    for f in formulas:
        resp, predictors = parse_formula(f)
        if isinstance(resp, tuple) != (survival or multivariate or known_variance):
            raise ValueError(
                "survival families require a Surv(time, status) response, "
                "DirichletLSS requires a cbind(...) response, "
                "GaussianKnownVarLSS requires an (outcome, s2) tuple response, "
                "and other families require a plain response")
        if response_name is None:
            response_name = resp
        elif resp != response_name:
            raise ValueError("all parameter formulas must share the same response")
        per_param_names.append(list(predictors))

    y, _ = _build_lss_response(data, response_name, survival, multivariate, known_variance)
    blocks, param_ncols, flat_names = [], [], []
    n = folds.shape[0]
    for names in per_param_names:
        param_ncols.append(len(names))
        flat_names.extend(names)
        if names:
            blocks.append(np.column_stack([_extract_column(data, nm) for nm in names]))
    x_all = (np.ascontiguousarray(np.column_stack(blocks), dtype=np.float64)
             if blocks else np.empty((n, 0), dtype=np.float64))
    # Observation weights: multiplied into a 0/1 fold mask for each fold's
    # refit and used to score the held-out rows, as the standard cvrisk does.
    # Without them a weighted model was cross-validated unweighted.
    if weights is None:
        w = np.ones(n, dtype=np.float64)
    else:
        from mboostrs.cvrisk import _with_model_weights

        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (n,):
            raise ValueError(f"weights has shape {w.shape}; expected ({n},)")
        folds = _with_model_weights(np.asarray(folds, dtype=np.float64), w)

    grid_list = _lss_grid_points(grid, k)
    folds = rescale_fold_weights(folds, weight_policy)
    scale = _check_response_scale(response_scale)
    risk = np.asarray(call_fit(_cvrisk_glmboostlss,
        x_all, param_ncols, flat_names, y, w, mstop_val, nu_scalar,
        _family._rust_name, np.ascontiguousarray(folds, dtype=np.float64), grid_list,
        stabilization, nu_per_param=nu_override,
        _fit_count=folds.shape[1], response_scale=scale,
        **step_kwargs(control),
    ), dtype=np.float64)
    mean_risk = risk.mean(axis=0)
    best = grid_list[int(np.argmin(mean_risk))]
    result = LssCvRiskResult(
        risk_matrix=risk,
        aggregate=mean_risk,
        grid=grid_list,
        optimal_mstop=best,
        param_names=tuple(_family.param_names),
        data_identity=_cv_data_identity(data, response_name, _response_transform, w,
                                        weight_policy, _family),
        provenance=Provenance(
            operation="cvrisk", model="glmboostLSS", family=type(_family).__name__,
            n_obs=n, n_learners=len(flat_names), mstop=mstop_val,
            weight_policy=weight_policy, effective_weight_total=float(w.sum()),
            folds=design,
            n_fits=int(folds.shape[1]) * len(grid_list),
        ),
    )
    _warn_lss_grid_boundary(result)
    return result


def cvrisk_gamboostLSS(formula, data, families=None, family=None, control=None, folds=None,
                       grid=None, bins=None, env=None, weight_policy="upstream", nu=None,
                       stabilization=None, weights=None, na_action="error", _search=None,
                       response_scale="auto"):
    """Cross-validated risk for a mixed structured ``gamboostLSS`` model.

    ``folds`` is an ``(n, B)`` matrix of in-bag weights, a
    :class:`~mboostrs.folds.FoldScheme`, or an integer fold count. An integer
    ``k`` builds k-fold weights with ``mb.cv_weights(n, type='kfold', B=k)``
    (seed 42), so ``folds=5`` is enough; it is required.

    ``na_action`` is the argument of :func:`gamboostLSS`, with the same values
    and meaning. ``"omit"`` drops a row with a missing value in a column any
    parameter formula reads, from the data, the weights and the fold matrix
    together, so every fold fit and every held-out score sees the same rows.
    ``"error"`` (default) refuses a missing value as the fit does.

    Same interface and return value as :func:`cvrisk_glmboostLSS`, but for smooth
    Learners are prepared against each fold's weights, so centering and
    ``df -> lambda`` are recomputed while fold-invariant spline templates remain
    cached.

    Returns a dict with ``risk`` (B x len(grid) per-fold OOB risk), ``grid``,
    and ``mstop`` (the grid point minimizing the mean OOB risk).

    ``bins=`` applies the same global quantile-bin count as
    :func:`gamboostLSS`, and ``LSSModel.cvrisk()`` inherits the value its fit
    was given. It is the argument that decides whether the per-fold rebuild
    accumulates a bin-weight vector or a full ``X'WX``.
    

    ``LSSModel.cvrisk()`` is the model-based interface: it obtains the
    specification and default grid from the fitted model. This function
    remains available when folds or a grid must be supplied explicitly.

    ``nu=`` sets per-parameter learning rates, upstream's
    ``boost_control(nu = c(...))`` vector: ``None`` (default) keeps the
    control's scalar, a float recycles it, a list is in family order, and a
    ``{param_name: float}`` dict must name every parameter.
    A lower rate slows updates to that parameter relative to the others. If
    the fitted mean and scale curves show different convergence, compare rate
    choices on the same folds with the joint score; the score alone does not
    identify which parameter accounts for residual structure.
    """
    if not _HAS_LSS_ENGINE or _cvrisk_gamboostlss_mixed is None:
        raise RuntimeError("mboostrs Rust mixed-learner LSS CV engine is not available")
    if folds is None:
        raise ValueError(
            "folds= is required: pass a fold matrix such as "
            "mb.cv_weights(n, 'kfold', 5), or fit the model and call its "
            ".cvrisk(), which builds folds and a grid itself")
    folds = _resolve_lss_folds(folds, data, where="cvrisk_gamboostLSS")
    from mboostrs.cvrisk import _fold_design

    design = _fold_design(folds, np.asarray(folds, dtype=np.float64))
    folds, judge = _lss_fold_matrix(folds, where="cvrisk_gamboostLSS")
    if family is not None and families is not None:
        raise TypeError("Specify family= or families=, not both")
    _family = family if family is not None else (families if families is not None else GaussianLSS())
    stabilization = _resolve_stabilization(stabilization, _family)
    k = _family.n_params
    if control is None:
        control = boost_control(100, 0.1)
    mstop_val, nu_scalar = control[0], control[1]
    if grid is None:
        # The diagonal grid `LSSModel.cvrisk()` uses: every parameter gets the
        # same budget, ten points up to the control's `mstop`, read off one
        # path per fold.
        grid = diagonal_grid(int(mstop_val), k)

    nu_override = _resolve_nu_per_param(nu, _family)
    formulas = ([formula[name] for name in _family.param_names]
                if isinstance(formula, dict) else [formula] * k)
    # The rows a missing value removes are removed from the data, the weights
    # and the fold matrix together, as gamboostLSS does for its own rows.
    data, weights, _dropped = _lss_na_action(
        formulas, data, na_action, weights, env, "cvrisk_gamboostLSS")
    if len(_dropped):
        folds, judge = _drop_fold_rows(folds, judge, _dropped, folds.shape[0])
    _model_steps = ()
    if na_action in ("impute_model", "impute_model_split") and any("%X%" in f for f in formulas):
        formulas, data, _model_steps = _impute_model_before_tensors(
            formulas, data, split_filled=na_action == "impute_model_split")
    _interactions = []
    if any("%X%" in f for f in formulas):
        formulas, data, env, _interactions = _expand_interaction_formulas(
            formulas, data, env)
    # Taken before the encoding below turns factor columns into codes; a
    # factor by= is lowered from it after parsing.
    from mboostrs.interactions import factor_columns as _factor_columns_of

    _factor_cols = _factor_columns_of(data)
    formulas, data, _transform_recipe, _response_transform = (
        _apply_formula_transforms(formulas, data, na_action=na_action,
                                    planned=_model_steps))
    _family = _uncensored_survival_family(_family, formulas, data)
    survival = getattr(_family, "_survival", False)
    multivariate = getattr(_family, "_multivariate", False)
    known_variance = getattr(_family, "_known_variance", False)

    response_name = None
    per_param_learners = []
    for f in formulas:
        resp, learners = parse_gamboost_formula(f, env=env)
        if isinstance(resp, tuple) != (survival or multivariate or known_variance):
            raise ValueError(
                "survival families require a Surv(time, status) response, "
                "DirichletLSS requires a cbind(...) response, "
                "GaussianKnownVarLSS requires an (outcome, s2) tuple response, "
                "and other families require a plain response")
        if response_name is None:
            response_name = resp
        elif resp != response_name:
            raise ValueError("all parameter formulas must share the same response")
        per_param_learners.append(learners)
    per_param_learners, _interactions = _lower_lss_factor_by(
        per_param_learners, data, _factor_cols, _interactions)
    _mark_categorical_trees(per_param_learners, _transform_recipe)
    per_param_learners = _apply_lss_global_bins(per_param_learners, bins)

    y, _ = _build_lss_response(data, response_name, survival, multivariate, known_variance)
    n = folds.shape[0]

    learner_specs = _gamboostlss_learner_specs(per_param_learners, data)
    # Observation weights: multiplied into a 0/1 fold mask for each fold's
    # refit and used to score the held-out rows, as the standard cvrisk does.
    # Without them a weighted model was cross-validated unweighted.
    if weights is None:
        w = np.ones(n, dtype=np.float64)
    else:
        from mboostrs.cvrisk import _with_model_weights

        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (n,):
            raise ValueError(f"weights has shape {w.shape}; expected ({n},)")
        folds = _with_model_weights(np.asarray(folds, dtype=np.float64), w)
    grid_list = _lss_grid_points(grid, k)
    folds = rescale_fold_weights(folds, weight_policy)
    judge_arr = None if judge is None else np.ascontiguousarray(judge, dtype=bool)
    scale = _check_response_scale(response_scale)
    if _search is not None:
        # The doubling search of tune_mstop: every fold is fitted once and its
        # path is resumed from one grid to the next (see _tune_mstop_lss).
        risk, curve, grid_points, best, at_boundary, used = call_fit(
            _search_gamboostlss_grid,
            learner_specs, y, w, nu_scalar, _family._rust_name,
            np.ascontiguousarray(folds, dtype=np.float64),
            [int(end) for end in _search["ends"]],
            [[int(value) for value in g] for g in _search["grids"]],
            float(_search["boundary_share"]),
            judge_arr, stabilization, nu_per_param=nu_override,
            _fit_count=folds.shape[1], response_scale=scale,
            method=_search.get("method", "cyclic"),
            **step_kwargs(control),
        )
        return {
            "risk": np.asarray(risk, dtype=np.float64),
            "curve": np.asarray(curve, dtype=np.float64),
            "grid": list(grid_points),
            "mstop": int(best),
            "at_boundary": bool(at_boundary),
            "ends": list(used),
        }
    risk = np.asarray(call_fit(_cvrisk_gamboostlss_mixed,
        learner_specs, y, w, mstop_val, nu_scalar, _family._rust_name,
        np.ascontiguousarray(folds, dtype=np.float64), grid_list,
        judge_arr,
        stabilization, nu_per_param=nu_override,
        _fit_count=folds.shape[1], response_scale=scale,
        **step_kwargs(control),
    ), dtype=np.float64)
    mean_risk = risk.mean(axis=0)
    best = grid_list[int(np.argmin(mean_risk))]
    result = LssCvRiskResult(
        risk_matrix=risk,
        aggregate=mean_risk,
        grid=grid_list,
        optimal_mstop=best,
        param_names=tuple(_family.param_names),
        data_identity=_cv_data_identity(data, response_name, _response_transform, w,
                                        weight_policy, _family),
        provenance=Provenance(
            operation="cvrisk", model="gamboostLSS", family=type(_family).__name__,
            n_obs=n, n_learners=sum(len(s) for s in per_param_learners),
            mstop=mstop_val,
            weight_policy=weight_policy, effective_weight_total=float(w.sum()),
            folds=design,
            n_fits=int(folds.shape[1]) * len(grid_list),
        ),
    )
    _warn_lss_grid_boundary(result)
    return result


def _drop_fold_rows(folds, judge, dropped, n):
    """The fold matrix and judge mask without the rows `dropped` names."""
    keep = np.ones(n, dtype=bool)
    keep[np.asarray(dropped, dtype=int)] = False
    folds = np.asarray(folds, dtype=np.float64)[keep]
    judge = None if judge is None else np.asarray(judge, dtype=bool)[keep]
    return folds, judge


def _drop_fold_rows_any(folds, dropped):
    """A supplied fold matrix or FoldScheme without the rows `dropped` names."""
    if folds is None or len(dropped) == 0:
        return folds
    is_scheme = hasattr(folds, "take")
    n = len(folds) if is_scheme else np.shape(folds)[0]
    keep = np.ones(n, dtype=bool)
    keep[np.asarray(dropped, dtype=int)] = False
    if is_scheme:
        return folds.take(keep)
    return np.asarray(folds)[keep]


def _lss_fold_matrix(folds, *, judge_supported=True, where="LSS cross-validation"):
    """The in-bag weight matrix of `folds` and its judge mask, if any.

    A `FoldScheme` can mark held-out rows that are not scored -- a rolling
    origin leaves the rows beyond each fold's horizon unused -- through its
    `judge` mask. Coercing the scheme to its weight matrix dropped the mask,
    so those rows were scored as held out. Returns ``(matrix, judge)``;
    `judge` is None for a plain matrix or a scheme without one, and a scheme
    with one is refused where `judge_supported` is false.
    """
    from mboostrs.folds import FoldScheme

    if not isinstance(folds, FoldScheme):
        return np.asarray(folds, dtype=np.float64), None
    judge = folds.judge
    if judge is not None and not judge_supported:
        raise NotImplementedError(
            f"{where} cannot restrict scoring to a FoldScheme's judged rows "
            "(its held-out rows include rows that are not scored, as in a "
            "rolling origin). Use gamboostLSS, whose cross-validation does.")
    if folds.base_weights is not None and not np.all(folds.base_weights == 1.0):
        raise NotImplementedError(
            f"{where} does not apply a FoldScheme's base_weights; fit the "
            "model with those weights instead.")
    return np.array(folds.weights, dtype=np.float64), (
        None if judge is None else np.array(judge, dtype=bool))


def _resolve_lss_folds(folds, data, *, where):
    """Resolve an integer ``folds`` to a k-fold weight matrix.

    ``folds=5`` is the obvious way to ask for 5-fold cross-validation, and the
    non-LSS ``cvrisk`` takes a fold count directly, so build the matrix here
    with :func:`mboostrs.cvrisk.cv_weights` instead of making the caller
    assemble one. A fold matrix or a
    :class:`~mboostrs.folds.FoldScheme` passes through unchanged.
    """
    from mboostrs.folds import FoldScheme

    if isinstance(folds, FoldScheme):
        return folds
    if isinstance(folds, bool) or not isinstance(folds, (int, np.integer)):
        return folds
    k = int(folds)
    if k < 2:
        raise ValueError(
            f"{where}: folds must be an integer of at least 2 for k-fold "
            f"cross-validation, or a fold matrix built by "
            f"mb.cv_weights(n, type='kfold', B=5); got folds={k}")
    from mboostrs._refit import _row_count
    from mboostrs.cvrisk import cv_weights

    n = _row_count(data)
    return cv_weights(n, type="kfold", B=k)


def _lss_stabsel_phat(all_selected, p, m, q_val):
    """Selection frequencies per component and iteration, after q-truncation.

    Matches R's ``stabsel.mboost``: once more than ``q`` distinct components
    have appeared, every remaining entry is replaced by the fold's first
    selection, and a component's frequency counts the folds in which it
    appeared at or before each iteration.
    """
    phat = np.zeros((p, m))
    for sel in all_selected:
        if not sel:
            continue
        seen = set()
        trunc = []
        threshold_reached = False
        for idx in sel:
            if threshold_reached:
                trunc.append(sel[0])
                continue
            if idx not in seen:
                seen.add(idx)
                if len(seen) > q_val:
                    threshold_reached = True
                    trunc.append(sel[0])
                    continue
            trunc.append(idx)
        for li in range(p):
            try:
                first = trunc.index(li)
            except ValueError:
                continue
            if first < m:
                phat[li, first:] += 1
    phat /= max(len(all_selected), 1)
    return phat


def _warn_short_folds(all_selected, q_val, B=None, sampling_type="SS"):
    """Warn when a subsample's path never reached ``q`` distinct components.

    The engine stops a fold at its q-th distinct component, so a fold that
    comes back with fewer has exhausted ``mstop`` instead, and its
    contribution to the frequencies is a truncated path. R's `stabsel` warns
    in the same situation.
    """
    from mboostrs.stabsel import _pair_count_phrase

    short = sum(1 for sel in all_selected if len(set(sel)) < q_val)
    if short:
        reached = max((len(set(sel)) for sel in all_selected), default=0)
        counts = (
            _pair_count_phrase(B, len(all_selected), sampling_type)
            if B is not None
            else f"{len(all_selected)} fits"
        )
        warnings.warn(
            "mstop too small in some subsampling replicates to select "
            f"q={q_val} base-learners; Increase mstop before applying stabsel. "
            f"{short} of {len(all_selected)} fits fell short ({counts}) and the best "
            f"reached {reached} distinct learners. A model whose path keeps "
            f"re-selecting the same few terms may not reach q={q_val} at any "
            "mstop, in which case lower q instead: a fold that never reaches "
            "q also never stops early, so the run costs the full mstop.",
            stacklevel=3,
        )


def stabsel_glmboostLSS(formula, data, families=None, family=None, control=None, folds=None,
                        q=None, PFER=None, cutoff=None, B=50,
                        sampling_type="SS", assumption="unimodal", seed=42, weights=None,
                        weight_policy="upstream", nu=None, stabilization=None,
                        fit_budget=None, response_scale="auto"):
    """Stability selection for a linear GAMLSS (R's stabsel.mboostLSS).

    Runs non-cyclic glmboostLSS on ``B`` half-samples, plus their exact
    complements under ``sampling_type="SS"``, tracking how often each *combined*
    base-learner (``"<predictor>.<parameter>"``) is selected, and returns the
    learners whose maximum selection frequency reaches the error-control cutoff.

    The error-control parameters (cutoff/q/PFER) use the same stabs math as
    :func:`mboostrs.stabsel.stabsel`. ``folds`` (an ``n x B`` in-bag weight
    matrix of *initial* subsamples) may be supplied to reproduce a specific
    subsampling; under ``"SS"`` the exact complements are appended here, so
    ``B`` columns produce ``2 * B`` fits and pairing is never inferred from an
    even column count.

    ``control``'s ``mstop`` is an upper bound on each subsample's path, not the
    number of iterations every subsample runs: a fold stops once it has
    selected ``q`` distinct components, because nothing it selects after that
    reaches the reported frequencies. Raising ``mstop`` therefore costs nothing
    on folds that reach ``q`` early, and only guards against folds that do not
    reach it at all -- which is warned about when it happens.

    ``B`` is the number of initial subsamples or complementary pairs used in
    the error bound. Under ``sampling_type="SS"``, each pair fits both halves,
    for ``2 * B`` fits; ``sampling_type="MB"`` fits ``B`` subsamples. The
    result reports the actual number of fits as ``n_fits``.

    ``LSSModel.stabsel()`` is the model-based interface and uses the fitted
    object's specification. This function remains available when folds must
    be supplied explicitly.

    ``nu=`` sets per-parameter learning rates, upstream's
    ``boost_control(nu = c(...))`` vector: ``None`` (default) keeps the
    control's scalar, a float recycles it, a list is in family order, and a
    ``{param_name: float}`` dict must name every parameter.
    A lower rate slows updates to that parameter relative to the others. If
    the fitted mean and scale curves show different convergence, compare rate
    choices on the same folds with the joint score; the score alone does not
    identify which parameter accounts for residual structure.
    """
    from mboostrs._validate import check_fit_budget
    from mboostrs.stabsel import _fit_matrix, _stabsel_parameters, make_folds, stabsel_planned_fits

    if not _HAS_LSS_ENGINE or _stabsel_selected_glmboostlss is None:
        raise RuntimeError("mboostrs Rust LSS engine (stabsel) not available")
    if family is not None and families is not None:
        raise TypeError("Specify family= or families=, not both")
    _family = family if family is not None else (families if families is not None else GaussianLSS())
    stabilization = _resolve_stabilization(stabilization, _family)
    k = _family.n_params
    if control is None:
        control = boost_control(100, 0.1)
    mstop_val, nu_scalar = control[0], control[1]

    nu_override = _resolve_nu_per_param(nu, _family)
    formulas = ([formula[name] for name in _family.param_names]
                if isinstance(formula, dict) else [formula] * k)
    formulas, data, _transform_recipe, _response_transform = (
        _apply_formula_transforms(formulas, data, kind="glm"))
    _family = _uncensored_survival_family(_family, formulas, data)
    survival = getattr(_family, "_survival", False)
    multivariate = getattr(_family, "_multivariate", False)
    known_variance = getattr(_family, "_known_variance", False)
    response_name = None
    per_param_names = []
    for f in formulas:
        resp, predictors = parse_formula(f)
        if isinstance(resp, tuple) != (survival or multivariate or known_variance):
            raise ValueError(
                "survival families require a Surv(time, status) response, "
                "DirichletLSS requires a cbind(...) response, "
                "GaussianKnownVarLSS requires an (outcome, s2) tuple response, "
                "and other families require a plain response")
        if response_name is None:
            response_name = resp
        elif resp != response_name:
            raise ValueError("all parameter formulas must share the same response")
        per_param_names.append(list(predictors))

    y, n = _build_lss_response(data, response_name, survival, multivariate, known_variance)
    blocks, param_ncols = [], []
    combined_names = []
    for pi, names in enumerate(per_param_names):
        param_ncols.append(len(names))
        combined_names.append(f"(Intercept).{_family.param_names[pi]}")
        for nm in names:
            combined_names.append(f"{nm}.{_family.param_names[pi]}")
        if names:
            blocks.append(np.column_stack([_extract_column(data, nm) for nm in names]))
    x_all = (np.ascontiguousarray(np.column_stack(blocks), dtype=np.float64)
             if blocks else np.empty((n, 0), dtype=np.float64))
    p = len(combined_names)

    # The subsample columns carry the case weights: `make_folds` draws in-bag
    # weights from `w_full` and appends complements that sum back to it, so
    # each SS pair reconstructs the model's own sample. Passing unit folds and
    # the weights separately is not the same thing -- the engine multiplies
    # the two, so the weights would be applied twice.
    w_full = (np.ones(n, dtype=np.float64) if weights is None
              else np.asarray(weights, dtype=np.float64).ravel())
    if w_full.shape != (n,):
        raise ValueError(
            f"weights must have one entry per row: got {w_full.shape[0]} for "
            f"{n} rows"
        )
    # Refused before the first subsample is drawn: the count depends on the
    # arguments only.
    check_fit_budget(
        stabsel_planned_fits(B, folds, sampling_type), fit_budget, "stabsel_glmboostLSS",
        remedy="lower B, or pass fewer folds")
    if folds is None:
        folds = make_folds(n, B, sampling_type=sampling_type, weights=w_full,
                           seed=seed)
    else:
        folds = _fit_matrix(folds, sampling_type, weights=w_full)
        B = folds.shape[1] // 2 if sampling_type == "SS" else folds.shape[1]
    n_fits = folds.shape[1]
    # Each column is a weight vector for one refit, so it goes through the same
    # scale policy the model's own weights did. Complementary pairing is a
    # property of the columns *before* this, exactly as upstream builds the
    # pair in `check_folds` and rescales each column inside `mboost_fit`.
    folds = rescale_fold_weights(folds, weight_policy)

    par = _stabsel_parameters(p, cutoff=cutoff, q=q, PFER=PFER, B=B,
                              sampling_type=sampling_type, assumption=assumption)
    cutoff_val, q_val, PFER_val = par["cutoff"], par["q"], par["PFER"]

    scale = _check_response_scale(response_scale)
    all_selected = _stabsel_selected_glmboostlss(
        x_all, param_ncols, y, np.ones(n, dtype=np.float64),
        mstop_val, nu_scalar, _family._rust_name, folds, q_val,
        stabilization, nu_per_param=nu_override, response_scale=scale,
        **step_kwargs(control),
    )
    _warn_short_folds(all_selected, q_val, B=B, sampling_type=sampling_type)

    # q-truncation + phat (same rule as mboostrs.stabsel.stabsel).
    phat = _lss_stabsel_phat(all_selected, p, mstop_val, q_val)
    max_phat = phat.max(axis=1)
    selected = [combined_names[i] for i in range(p) if max_phat[i] >= cutoff_val]
    return LssStabSelResult(
        phat=phat,
        selected=selected,
        max_=max_phat,
        cutoff=cutoff_val,
        q=q_val,
        PFER=PFER_val,
        learner_names=combined_names,
        sampling_type=sampling_type,
        assumption=assumption,
        B=B,
        n_fits=n_fits,
        provenance=Provenance(
            operation="stabsel", model="glmboostLSS", family=type(_family).__name__,
            n_obs=n, n_learners=p, mstop=mstop_val,
            weight_policy=weight_policy,
            effective_weight_total=float(n),
            seed=seed,
            folds=(f"{B} half-sample pairs" if sampling_type == "SS"
                   else f"{B} half-samples"),
            n_fits=n_fits,
        ),
    )


def stabsel_gamboostLSS(formula, data, families=None, family=None, control=None, folds=None,
                        q=None, PFER=None, cutoff=None, B=50,
                        sampling_type="SS", assumption="unimodal", seed=42, weights=None, bins=None,
                        env=None, weight_policy="upstream", nu=None, stabilization=None,
                        fit_budget=None, na_action="error", response_scale="auto"):
    """Stability selection for a mixed structured ``gamboostLSS`` model.

    ``na_action`` is the argument of :func:`gamboostLSS`, with the same values
    and meaning. ``"omit"`` drops a row with a missing value in a column any
    parameter formula reads from the data, the weights and any supplied fold
    matrix, so every subsample refers to the same rows.

    Same interface and error-control math as :func:`stabsel_glmboostLSS`, but for
    structured learners. It runs non-cyclic ``gamboostLSS`` on each subsample
    and tracks every combined ``"<learner>.<parameter>"`` component. Learners
    are rebuilt with the subsample weights, including centering and
    ``df -> lambda``.

    As in :func:`stabsel_glmboostLSS`, ``control``'s ``mstop`` bounds each
    subsample's path rather than fixing it: a fold stops once it has selected
    ``q`` distinct components.

    ``B`` is the number of initial subsamples or complementary pairs used in
    the error bound. Under ``sampling_type="SS"``, each pair fits both halves,
    for ``2 * B`` fits; ``sampling_type="MB"`` fits ``B`` subsamples. The
    result reports the actual number of fits as ``n_fits``.

    ``bins=`` applies the same global quantile-bin count as
    :func:`gamboostLSS`, and ``LSSModel.stabsel()`` inherits the value its fit
    was given. This is the single largest lever on the cost of the run: each
    fold rebuilds every learner against its own weights, and a binned rebuild
    accumulates a bin-weight vector where the exact one copies an ``n x p``
    design and forms ``X'WX`` on it.


    ``LSSModel.stabsel()`` is the model-based interface and uses the fitted
    object's specification. This function remains available when folds must
    be supplied explicitly.

    ``nu=`` sets per-parameter learning rates, upstream's
    ``boost_control(nu = c(...))`` vector: ``None`` (default) keeps the
    control's scalar, a float recycles it, a list is in family order, and a
    ``{param_name: float}`` dict must name every parameter.
    A lower rate slows updates to that parameter relative to the others. If
    the fitted mean and scale curves show different convergence, compare rate
    choices on the same folds with the joint score; the score alone does not
    identify which parameter accounts for residual structure.
    """
    from mboostrs._validate import check_fit_budget
    from mboostrs.stabsel import _fit_matrix, _stabsel_parameters, make_folds, stabsel_planned_fits

    if not _HAS_LSS_ENGINE or _stabsel_selected_gamboostlss_mixed is None:
        raise RuntimeError("mboostrs Rust mixed-learner LSS stabsel engine is not available")
    if family is not None and families is not None:
        raise TypeError("Specify family= or families=, not both")
    _family = family if family is not None else (families if families is not None else GaussianLSS())
    stabilization = _resolve_stabilization(stabilization, _family)
    k = _family.n_params
    if control is None:
        control = boost_control(100, 0.1)
    mstop_val, nu_scalar = control[0], control[1]

    nu_override = _resolve_nu_per_param(nu, _family)
    formulas = ([formula[name] for name in _family.param_names]
                if isinstance(formula, dict) else [formula] * k)
    # Rows a missing value removes leave the data, the weights and any
    # supplied fold matrix together, as in gamboostLSS.
    data, weights, _dropped = _lss_na_action(
        formulas, data, na_action, weights, env, "stabsel_gamboostLSS")
    folds = _drop_fold_rows_any(folds, _dropped)
    _model_steps = ()
    if na_action in ("impute_model", "impute_model_split") and any("%X%" in f for f in formulas):
        formulas, data, _model_steps = _impute_model_before_tensors(
            formulas, data, split_filled=na_action == "impute_model_split")
    _interactions = []
    if any("%X%" in f for f in formulas):
        formulas, data, env, _interactions = _expand_interaction_formulas(
            formulas, data, env)
    # Taken before the encoding below turns factor columns into codes; a
    # factor by= is lowered from it after parsing.
    from mboostrs.interactions import factor_columns as _factor_columns_of

    _factor_cols = _factor_columns_of(data)
    formulas, data, _transform_recipe, _response_transform = (
        _apply_formula_transforms(formulas, data, na_action=na_action,
                                    planned=_model_steps))
    _family = _uncensored_survival_family(_family, formulas, data)
    survival = getattr(_family, "_survival", False)
    multivariate = getattr(_family, "_multivariate", False)
    known_variance = getattr(_family, "_known_variance", False)

    response_name = None
    per_param_learners = []
    for f in formulas:
        resp, learners = parse_gamboost_formula(f, env=env)
        if isinstance(resp, tuple) != (survival or multivariate or known_variance):
            raise ValueError(
                "survival families require a Surv(time, status) response, "
                "DirichletLSS requires a cbind(...) response, "
                "GaussianKnownVarLSS requires an (outcome, s2) tuple response, "
                "and other families require a plain response")
        if response_name is None:
            response_name = resp
        elif resp != response_name:
            raise ValueError("all parameter formulas must share the same response")
        per_param_learners.append(learners)
    per_param_learners, _interactions = _lower_lss_factor_by(
        per_param_learners, data, _factor_cols, _interactions)
    _mark_categorical_trees(per_param_learners, _transform_recipe)
    per_param_learners = _apply_lss_global_bins(per_param_learners, bins)

    y, n = _build_lss_response(data, response_name, survival, multivariate, known_variance)

    learner_specs = _gamboostlss_learner_specs(per_param_learners, data)
    combined_names = []
    for pi, learners in enumerate(per_param_learners):
        for ln in learners:
            combined_names.append(f"{ln.name}.{_family.param_names[pi]}")
    p = len(combined_names)

    # The subsample columns carry the case weights: `make_folds` draws in-bag
    # weights from `w_full` and appends complements that sum back to it, so
    # each SS pair reconstructs the model's own sample. Passing unit folds and
    # the weights separately is not the same thing -- the engine multiplies
    # the two, so the weights would be applied twice.
    w_full = (np.ones(n, dtype=np.float64) if weights is None
              else np.asarray(weights, dtype=np.float64).ravel())
    if w_full.shape != (n,):
        raise ValueError(
            f"weights must have one entry per row: got {w_full.shape[0]} for "
            f"{n} rows"
        )
    # Refused before the first subsample is drawn: the count depends on the
    # arguments only.
    check_fit_budget(
        stabsel_planned_fits(B, folds, sampling_type), fit_budget, "stabsel_gamboostLSS",
        remedy="lower B, or pass fewer folds")
    if folds is None:
        folds = make_folds(n, B, sampling_type=sampling_type, weights=w_full,
                           seed=seed)
    else:
        folds = _fit_matrix(folds, sampling_type, weights=w_full)
        B = folds.shape[1] // 2 if sampling_type == "SS" else folds.shape[1]
    n_fits = folds.shape[1]
    # Each column is a weight vector for one refit, so it goes through the same
    # scale policy the model's own weights did. Complementary pairing is a
    # property of the columns *before* this, exactly as upstream builds the
    # pair in `check_folds` and rescales each column inside `mboost_fit`.
    folds = rescale_fold_weights(folds, weight_policy)

    par = _stabsel_parameters(p, cutoff=cutoff, q=q, PFER=PFER, B=B,
                              sampling_type=sampling_type, assumption=assumption)
    cutoff_val, q_val, PFER_val = par["cutoff"], par["q"], par["PFER"]

    scale = _check_response_scale(response_scale)
    all_selected = _stabsel_selected_gamboostlss_mixed(
        learner_specs, y, np.ones(n, dtype=np.float64), mstop_val, nu_scalar,
        _family._rust_name, folds, q_val,
        stabilization, nu_per_param=nu_override, response_scale=scale,
        **step_kwargs(control),
    )
    _warn_short_folds(all_selected, q_val, B=B, sampling_type=sampling_type)

    # q-truncation + phat (same rule as stabsel_glmboostLSS).
    phat = _lss_stabsel_phat(all_selected, p, mstop_val, q_val)
    max_phat = phat.max(axis=1)
    selected = [combined_names[i] for i in range(p) if max_phat[i] >= cutoff_val]
    return LssStabSelResult(
        phat=phat,
        selected=selected,
        max_=max_phat,
        cutoff=cutoff_val,
        q=q_val,
        PFER=PFER_val,
        learner_names=combined_names,
        sampling_type=sampling_type,
        assumption=assumption,
        B=B,
        n_fits=n_fits,
        provenance=Provenance(
            operation="stabsel", model="gamboostLSS", family=type(_family).__name__,
            n_obs=n, n_learners=p, mstop=mstop_val,
            weight_policy=weight_policy,
            effective_weight_total=float(n),
            seed=seed,
            folds=(f"{B} half-sample pairs" if sampling_type == "SS"
                   else f"{B} half-samples"),
            n_fits=n_fits,
        ),
    )


# `mboostrs.gamlss` is both the module and the direct REML fit: calling it, as
# `mb.gamlss(formulas, data)`, fits a distributional model from one formula per
# parameter (see `mboostrs.direct_gam`). A module object cannot be called by
# default, so its class is replaced by one that can, and attribute access and
# `from mboostrs import gamlss` keep returning the module.
class _CallableModule(type(sys)):
    def __call__(self, *args, **kwargs):
        from mboostrs.direct_gam import gamlss as fit_direct

        return fit_direct(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule
