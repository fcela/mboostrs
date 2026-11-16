"""Does a distributional scale effect survive a more flexible mean? (issue #112).

An underfitted mean leaves structure in its residuals. A scale model with
its own predictor can take that structure up: log(sigma) rises and falls
with the mean's bias, the fitted variance becomes squared bias, and the
fit reports a finite risk with plausible intervals. This module refits the
same distributional model three ways on the same out-of-fold split and
reads where the scale effect comes from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from mboostrs._names import IDENT_SRC as IDENT
from mboostrs._names import quote_name, quote_response, unquote_name

VERDICT_ROBUST = "scale effect robust"
VERDICT_ABSORBS = "scale effect absorbs mean misspecification"
VERDICT_INCONCLUSIVE = "inconclusive"

#: The flexible mean must remove at least this share of the given model's
#: fitted log-sigma spread before the scale effect counts as absorbed. A
#: correct mean removes about 0 (the shares were -0.02 to 0.01 over 30 seeds
#: of a correct heteroskedastic fit), and a perfect mean still leaves the
#: sampling noise of the scale fit in the spread, so the share of a real
#: absorption sits well below 1.
ABSORBED_SHARE = 0.25
#: Below this share, the flexible mean is judged not to have removed the
#: scale effect.
ROBUST_SHARE = 0.10
#: The out-of-fold gain of the flexible mean must reach this t statistic
#: (paired across folds) for the absorption verdict.
GAIN_T = 2.0
#: A given model whose fitted log-sigma spread is below this has no scale
#: effect to absorb, and the verdict is inconclusive.
MIN_SCALE_SPREAD = 0.1

#: Continuous families with a mean parameter `mu` and a scale `sigma`.
_CONTINUOUS_LOCATION_SCALE = frozenset({
    "GaussianLSS", "GammaLSS", "StudentTLSS", "JSULSS", "BCTLSS", "BCPELSS",
    "BCCGLSS", "GumbelLSS", "LogisticLSS", "InverseGaussianLSS",
})

_SMOOTH_CALL = re.compile(r"^\s*(bbs|bspatial)\s*\((.*)\)\s*$", re.DOTALL)
_PLAIN_LINEAR = re.compile(rf"^\s*(?:bols\(\s*)?({IDENT})\s*\)?\s*$")
#: Default `df` of each smooth in this library (`Bbs` 4, `Bspatial` 6).
_DEFAULT_SMOOTH_DF = {"bbs": 4.0, "bspatial": 6.0}


@dataclass(frozen=True)
class ScaleAbsorption:
    """Result of :func:`scale_absorption`.

    ``verdict`` is one of ``"scale effect robust"``,
    ``"scale effect absorbs mean misspecification"`` or ``"inconclusive"``.
    ``rule`` states the thresholds that decided it.

    ``scale`` maps each variant (``"given"``, ``"flexible_mean"``,
    ``"slower_scale"``) to ``sd_log_sigma``, the standard deviation of the
    fitted linear predictor of sigma over the training rows, and
    ``term_ranges``, the range (max minus min over the training rows) of each
    sigma term's fitted effect.

    ``absorbed`` gives, for the flexible mean and the slower scale rate, the
    share of the given model's ``sd_log_sigma`` that the variant removed:
    ``1 - sd_variant / sd_given``. Negative values mean the variant's spread is
    larger.

    ``oof`` maps each variant to ``mean_nll`` (out-of-fold joint negative
    log-likelihood per observation, averaged over the folds) and ``fold_nll``.
    The two alternatives also carry ``gain`` (given minus variant, so a
    positive value means the variant predicts held-out rows better), ``se``
    (standard error of that paired gain across folds) and ``t`` (gain over
    se).

    ``residual_structure`` maps each variant to the standardized residual
    ``(y - mu) / sigma`` screened against the model's predictors:
    ``columns`` holds the curvature of the residual on each predictor
    (``curve_r2 - linear_r2``) and ``pairs`` the out-of-sample explained
    variation of each predictor pair. These diagnostics are reported for
    context; the verdict uses the scale and held-out score criteria above.

    ``specifications`` records the mean formula of the flexible variant and
    the per-parameter learning rates of the slower variant. ``notes`` lists
    conditions the reader should know about.
    """

    verdict: str
    rule: str
    absorbed: dict
    scale: dict
    oof: dict
    residual_structure: dict
    specifications: dict
    folds: int
    seed: int
    notes: tuple

    def to_dict(self, *, include_arrays=False) -> dict:
        """The result as plain dicts, lists and floats, for JSON or a report.

        ``include_arrays`` is accepted for the export convention every result
        record follows. This result holds no per-row arrays, so the flag adds
        nothing to the payload.
        """
        return {
            "verdict": self.verdict,
            "rule": self.rule,
            "absorbed": _plain(self.absorbed),
            "scale": _plain(self.scale),
            "oof": _plain(self.oof),
            "residual_structure": _plain(self.residual_structure),
            "specifications": _plain(self.specifications),
            "folds": int(self.folds),
            "seed": int(self.seed),
            "notes": list(self.notes),
        }


def scale_absorption(model, *, folds=5, seed=0, flexible=None):
    """Test whether a fitted LSS scale effect survives a more flexible mean.

    ``model`` is a cyclic ``gamboostLSS`` fit with a continuous family that
    has a mean ``mu`` and a scale ``sigma`` (``GaussianLSS`` and the other
    location-scale families). The function refits the model's specification
    three ways on the same ``folds``-fold split, all with the given model's
    control:

    - ``given``: the model as it was fitted. Its in-sample scale and its
      out-of-fold score are read from the fit itself.
    - ``flexible_mean``: the mean formula with each ``bbs`` and ``bspatial``
      smooth at twice its degrees of freedom, and each continuous predictor
      that enters the mean only linearly (``x`` or ``bols(x)``) given a
      ``bbs`` smooth at the default df. ``flexible=`` replaces this formula
      with a mean formula the caller writes (a full ``"y ~ ..."`` string).
    - ``slower_scale``: the same formulas with learning rates
      ``{mu: nu, sigma: nu / 4}``, where ``nu`` is the given model's rate for
      each parameter.

    For each variant the function reports the spread of fitted log-sigma
    over the training rows, the range of each sigma term, the out-of-fold
    joint score, and the residual structure left in the mean.

    Verdict rule: ``"scale effect absorbs mean misspecification"`` when the
    flexible mean removes at least 25% of the given fit's log-sigma spread
    and its out-of-fold gain has ``t >= 2``; ``"scale effect robust"`` when
    it removes less than 10%; ``"inconclusive"`` otherwise. A given fit whose
    log-sigma spread is below 0.1 has no scale effect to absorb, and the
    verdict is inconclusive. The slower-scale variant is reported and does
    not enter the verdict: a slower rate with the same ``mstop`` moves sigma
    less by construction, so its shrinkage is not evidence about the mean.

    Parameters
    ----------
    model : LSSModel
        A fit from :func:`mboostrs.gamboostLSS` with a continuous
        location-scale family. Non-cyclic and ``glmboostLSS`` fits are
        refused.
    folds : int, default 5
        Number of folds for the out-of-fold scores.
    seed : int, default 0
        Seed of the fold assignment.
    flexible : str, optional
        Mean formula for the flexible variant, for example
        ``"y ~ bbs(x, df=8)"``. Its response must match the model's.

    Returns
    -------
    ScaleAbsorption
        The verdict, the measures behind it and ``to_dict()``.

    Raises
    ------
    NotImplementedError
        If the family is not a continuous location-scale family, or the fit
        is non-cyclic or from ``glmboostLSS``.
    ValueError
        If no flexible mean can be built from the given mean formula, and
        ``flexible=`` was not supplied.

    Examples
    --------
    >>> fit = mb.gamboostLSS({"mu": "y ~ bols(x)", "sigma": "y ~ bbs(x)"},
    ...                      data=data, families=mb.GaussianLSS())
    >>> result = mb.scale_absorption(fit, folds=5, seed=0)
    >>> result.verdict  # doctest: +SKIP
    'scale effect absorbs mean misspecification'
    """
    from mboostrs import gamlss
    from mboostrs.cvrisk import cv_weights
    from mboostrs.formula import _split_formula

    _check_model(model)
    names = tuple(model._family.param_names)
    formulas = _formulas_by_parameter(model)
    response = model._response_name
    n = len(model.model_weights())
    budget = _budget_point(model)
    base_rates = {name: float(model._parameter_nu(k)) for k, name in enumerate(names)}

    if flexible is None:
        flexible_mu = _flexible_mean_formula(formulas["mu"], model._data)
        if flexible_mu == formulas["mu"]:
            raise ValueError(
                f"the mean formula {formulas['mu']!r} has no smooth to double "
                "and no linear continuous predictor to smooth; pass the "
                "flexible= argument with the mean formula to compare against"
            )
    else:
        flexible_mu = str(flexible)
        flexible_response = _split_formula(flexible_mu)[0]
        if flexible_response != response:
            raise ValueError(
                f"flexible= must have the model's response {response!r}, got "
                f"{flexible_response!r}"
            )

    flexible_formulas = dict(formulas)
    flexible_formulas["mu"] = flexible_mu
    slow_rates = dict(base_rates)
    slow_rates["sigma"] = base_rates["sigma"] / 4.0

    fits = {
        "given": model,
        "flexible_mean": gamlss.gamboostLSS(
            **_refit_arguments(model, flexible_formulas, None)),
        "slower_scale": gamlss.gamboostLSS(
            **_refit_arguments(model, formulas, slow_rates)),
    }

    fold_matrix = cv_weights(n, type="kfold", B=int(folds), seed=int(seed),
                             weights=model.model_weights())
    predictors = _predictor_names(formulas, response)

    scale = {}
    residual = {}
    oof_risk = {}
    for variant, fit in fits.items():
        scale[variant] = _scale_summary(fit)
        residual[variant] = _residual_structure(fit, predictors, response)
        cv = fit.cvrisk(folds=fold_matrix, grid=[budget])
        oof_risk[variant] = np.asarray(cv.risk_matrix, dtype=np.float64)[:, 0]

    oof = {"given": {"mean_nll": float(oof_risk["given"].mean()),
                     "fold_nll": [float(v) for v in oof_risk["given"]]}}
    absorbed = {}
    for variant in ("flexible_mean", "slower_scale"):
        gains = oof_risk["given"] - oof_risk[variant]
        se = float(gains.std(ddof=1) / np.sqrt(len(gains))) if len(gains) > 1 else 0.0
        gain = float(gains.mean())
        oof[variant] = {
            "mean_nll": float(oof_risk[variant].mean()),
            "fold_nll": [float(v) for v in oof_risk[variant]],
            "gain": gain,
            "se": se,
            "t": _t_statistic(gain, se),
        }
        sd_given = scale["given"]["sd_log_sigma"]
        absorbed[variant] = (1.0 - scale[variant]["sd_log_sigma"] / sd_given
                             if sd_given > 0 else 0.0)

    verdict, notes = _decide(scale["given"]["sd_log_sigma"],
                             absorbed["flexible_mean"], oof["flexible_mean"]["t"])
    return ScaleAbsorption(
        verdict=verdict,
        rule=_RULE_TEXT,
        absorbed=absorbed,
        scale=scale,
        oof=oof,
        residual_structure=residual,
        specifications={
            "flexible_mean": {"mu": flexible_mu},
            "slower_scale": {"nu": slow_rates},
        },
        folds=int(folds),
        seed=int(seed),
        notes=notes,
    )


_RULE_TEXT = (
    "absorbs: flexible-mean absorbed >= 0.25 and its out-of-fold gain t >= 2; "
    "robust: flexible-mean absorbed < 0.10; inconclusive otherwise; "
    "inconclusive when the given fit's sd_log_sigma < 0.1"
)


def _decide(sd_given, absorbed_flexible, t_flexible):
    """The verdict and its notes from the given spread and the flexible variant."""
    if not sd_given >= MIN_SCALE_SPREAD:
        return VERDICT_INCONCLUSIVE, (
            f"the given fit's log-sigma spread is {sd_given:.3g}, below "
            f"{MIN_SCALE_SPREAD}: there is no scale effect to absorb",)
    if absorbed_flexible >= ABSORBED_SHARE and t_flexible >= GAIN_T:
        return VERDICT_ABSORBS, ()
    if absorbed_flexible < ROBUST_SHARE:
        return VERDICT_ROBUST, ()
    return VERDICT_INCONCLUSIVE, ()


def _check_model(model) -> None:
    family = getattr(model, "_family", None)
    family_name = type(family).__name__
    if family_name not in _CONTINUOUS_LOCATION_SCALE:
        raise NotImplementedError(
            f"scale_absorption is defined for continuous location-scale families "
            f"(GaussianLSS and the families with a mean mu and a scale sigma), "
            f"not {family_name}; fit one of those with gamboostLSS")
    if getattr(model, "_model_kind", None) != "gamboostLSS":
        raise NotImplementedError(
            "scale_absorption refits with gamboostLSS and does not take a "
            "glmboostLSS fit; refit the model with mb.gamboostLSS and pass that")
    if getattr(model, "_method", "cyclic") != "cyclic":
        raise NotImplementedError(
            "scale_absorption needs a cyclic fit; a non-cyclic fit takes one "
            "total budget. Refit with method='cyclic'")
    if not {"mu", "sigma"} <= set(family.param_names):
        raise NotImplementedError(
            f"{family_name} has parameters {tuple(family.param_names)}; "
            "scale_absorption needs a mean 'mu' and a scale 'sigma'")


def _formulas_by_parameter(model) -> dict:
    """The rewritten formula of every distribution parameter, keyed by name."""
    names = tuple(model._family.param_names)
    stored = getattr(model, "_rewritten_formulas", None)
    if stored is None:
        stored = model._formula
    if isinstance(stored, dict):
        return {name: str(stored[name]) for name in names}
    return {name: str(stored) for name in names}


def _budget_point(model) -> list[int]:
    """The per-parameter update counts of the given fit, as a cvrisk grid point."""
    counts = [len(model._params[k]["selected"]) for k in range(len(model._params))]
    return [int(c) for c in counts]


def _refit_arguments(model, formulas, rates) -> dict:
    """The arguments of a refit of ``model`` with other formulas or rates."""
    arguments = model._refit_arguments()
    arguments["formula"] = dict(formulas)
    if rates is not None:
        arguments["nu"] = dict(rates)
    if getattr(model, "_input_weights", None) is not None:
        arguments["weights"] = model.model_weights()
    return arguments


def _double_smooth(term: str) -> tuple[str, bool]:
    """``term`` with its df doubled when it is a bbs or bspatial smooth."""
    if "%X%" in term:
        return term, False
    match = _SMOOTH_CALL.match(term)
    if match is None:
        return term, False
    head, args = match.group(1), match.group(2)
    if re.search(r"\blambda(_value)?\s*=", args):
        return term, False
    df_match = re.search(r"\bdf\s*=\s*([0-9.eE+-]+)", args)
    if df_match:
        doubled = 2.0 * float(df_match.group(1))
        args = args[:df_match.start(1)] + _number(doubled) + args[df_match.end(1):]
    else:
        doubled = 2.0 * _DEFAULT_SMOOTH_DF[head]
        separator = ", " if args.strip() else ""
        args = f"{args}{separator}df={_number(doubled)}"
    return f"{head}({args})", True


def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def _flexible_mean_formula(formula: str, data) -> str:
    """The mean formula with its smooths at double df and its linear terms smoothed.

    A ``bbs`` or ``bspatial`` term gets twice its degrees of freedom. A plain
    linear term of a continuous predictor, ``x`` or ``bols(x)``, gets a
    ``bbs(x)`` at the default df when no smooth in the mean already uses
    that predictor. Other learners (``bmono``, ``btree``, ``brandom``) are
    left as written, because a smooth added beside them changes their
    meaning.
    """
    from mboostrs.formula import _split_formula, term_variables

    response, terms = _split_formula(formula)
    new_terms = []
    smoothed: set[str] = set()
    linear_names: list[str] = []
    for term in terms:
        stripped = term.strip()
        doubled, is_smooth = _double_smooth(stripped)
        new_terms.append(doubled)
        if is_smooth:
            smoothed.update(term_variables(stripped))
            continue
        plain = _PLAIN_LINEAR.match(stripped)
        if plain is not None and unquote_name(plain.group(1)) != response:
            linear_names.append(unquote_name(plain.group(1)))
    for name in linear_names:
        if name in smoothed:
            continue
        if _is_continuous(data, name):
            new_terms.append(f"bbs({quote_name(name)})")
            smoothed.add(name)
    return f"{quote_response(response)} ~ {' + '.join(new_terms)}"


def _is_continuous(data, name: str) -> bool:
    """Whether a column of the fitted frame is a measurement rather than a factor code.

    A fit stores its factors as integer codes, so an integer-valued column
    cannot be told from a factor here and is read as one. That is the
    conservative side: a smooth is not added to a factor, and an integer
    measurement (a count) is not smoothed either.
    """
    from mboostrs._frame import read_column

    column = read_column(data, name)
    if column.kind != "numeric":
        return False
    values = np.asarray(column.values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return finite.size > 0 and not np.all(finite == np.round(finite))


def _predictor_names(formulas: dict, response: str) -> list[str]:
    from mboostrs.formula import _split_formula, term_variables

    names: list[str] = []
    for formula in formulas.values():
        _, terms = _split_formula(formula)
        for term in terms:
            for name in term_variables(term.strip()):
                if name != response and name not in names:
                    names.append(name)
    return names


def _scale_summary(fit) -> dict:
    eta = np.asarray(fit.predict(parameter="sigma", type="link"),
                     dtype=np.float64).ravel()
    effects = fit.partial_effects("sigma")
    ranges = {str(name): float(np.ptp(np.asarray(values, dtype=np.float64)))
              for name, values in effects.items()}
    return {"sd_log_sigma": float(eta.std()), "term_ranges": ranges}


def _residual_structure(fit, predictors: list[str], response: str) -> dict:
    """Standardized residual screened against the predictors of the fit."""
    from mboostrs.terms import column_screen, screen_interactions

    data = fit._data
    y = np.asarray(fit._observed_response(data), dtype=np.float64).ravel()
    mu = np.asarray(fit.predict(parameter="mu", type="response"),
                    dtype=np.float64).ravel()
    sigma = np.asarray(fit.predict(parameter="sigma", type="response"),
                       dtype=np.float64).ravel()
    z = (y - mu) / sigma

    columns = {}
    if predictors:
        screened = column_screen(_with_column(data, "_scale_absorption_z", z),
                                 "_scale_absorption_z", columns=list(predictors),
                                 output="records")
        for row in screened:
            columns[str(row["column"])] = {
                "linear_r2": float(row["linear_r2"]),
                "curve_r2": float(row["curve_r2"]),
                "curvature": float(row["curvature"]),
                "hint": str(row["hint"]),
            }
    pairs = list(combinations(predictors, 2))
    scored = screen_interactions(data, residual=z, response=response,
                                 pairs=pairs) if pairs else []
    removed = [float(p.removed) for p in scored]
    strongest_pair = None
    if scored:
        top = max(scored, key=lambda p: p.removed)
        strongest_pair = [str(top.left), str(top.right)]
    strongest_column = None
    if columns:
        strongest_column = max(columns, key=lambda c: columns[c]["curvature"])
    return {
        "columns": columns,
        "strongest_column": strongest_column,
        "pairs": {
            "n_pairs": len(scored),
            "n_positive": int(sum(1 for r in removed if r > 0)),
            "max_removed": max(removed) if removed else None,
            "strongest_pair": strongest_pair,
        },
    }


def _with_column(data, name: str, values):
    """``data`` with one extra column, for the frame types the fit accepts."""
    if isinstance(data, dict):
        return {**data, name: values}
    if hasattr(data, "assign"):
        return data.assign(**{name: values})
    if hasattr(data, "with_columns"):
        import polars as pl

        return data.with_columns(pl.Series(name, values))
    raise TypeError(f"cannot add a residual column to a {type(data).__name__}")


def _t_statistic(gain: float, se: float) -> float:
    if se > 0:
        return gain / se
    if gain > 0:
        return float("inf")
    return 0.0


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
