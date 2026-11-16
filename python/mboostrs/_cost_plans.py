"""Fit-count plans of the causal calls and of ``discover`` for ``estimate_cost``.

A tuned nuisance model counts as the fits of ``tune_mstop`` (its inner folds
plus one full-data fit); an untuned one counts as one fit. The plans read each
call's argument defaults from the call's own signature through ``bound``, so a
changed default (``smooth``, ``outcome_model``) changes the estimate with it.
"""
from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np

from mboostrs._names import quote_name, quote_response

#: Step count of the timed fit when the call tunes its own step count.
TIMING_STEPS = 100

#: Folds of the inner step-count search that ``effect_model`` and ``ci_test`` fix.
FIXED_INNER_FOLDS = 5


def _steps(mstop) -> int:
    return int(mstop) if isinstance(mstop, int) and not isinstance(mstop, bool) \
        else TIMING_STEPS


def _fold_count(folds) -> int:
    return int(folds) if isinstance(folds, (int, np.integer)) else int(folds.n_folds)


def _tuned_fits(tune, inner_folds) -> int:
    """Fits of one nuisance model: those of ``tune_mstop`` when its step count
    is tuned, one fit otherwise."""
    from mboostrs.explore import _tune_mstop_fits

    return _tune_mstop_fits(inner_folds) if tune else 1


def _outcome_values(data, spec):
    from mboostrs import causal

    values = causal._columns(data, spec, outcome_missing="ipw", categorical=True)
    return np.asarray(values[spec.outcome], dtype=float)


def _use_response(data, spec, outcome_missing) -> bool:
    """Whether a response model is fitted: ``outcome_missing="ipw"`` and some
    outcome is missing."""
    from mboostrs import causal

    resolved = causal._check_outcome_missing(
        causal._spec_default(spec, "outcome_missing", outcome_missing))
    return resolved == "ipw" and bool(np.isnan(_outcome_values(data, spec)).any())


def _outcome_models(requested, data, spec) -> tuple[int, str]:
    """Outcome models fitted per fold, and a phrase saying how the count was set."""
    from mboostrs import causal

    if requested == "cv":
        y = _outcome_values(data, spec)
        seen = y[~np.isnan(y)]
        zero_inflated = bool(spec.outcome_type == "continuous" and np.all(seen >= 0)
                             and np.any(seen == 0))
        count = 3 + (2 if zero_inflated else 0)
        return count, ("every candidate outcome model is fitted: separate (2), pooled (1)"
                       + (", two-part (2)" if zero_inflated else ""))
    causal._check_outcome_model(requested)
    if requested == "pooled":
        return 1, "one pooled outcome model per fold"
    if requested == "auto":
        return 2, ("two outcome models per fold, an upper bound: 'auto' uses one pooled "
                   "model when the sparse-arm check flags a propensity bin")
    return 2, ("two outcome models per fold "
               + ("(separate, one per arm)" if requested == "separate"
                  else "(employment and log-outcome)"))


def _propensity_runner(data, spec, smooth, df, mstop):
    """One ordinary propensity fit on all rows, at the first step count."""
    from mboostrs import causal
    from mboostrs.families import Binomial

    values = causal._columns(data, spec, outcome_missing="ipw", categorical=True)
    names = causal._smooth_baselines(values, spec.covariates)
    flag = causal._smooth_by_nuisance(smooth)["propensity"]
    rhs = causal._rhs(spec.covariates, flag, df, smooth_names=names)
    treated = (np.asarray(data[spec.treatment]) == spec.treatment_contrast[1]).astype(float)
    train = {spec.treatment: treated, **{name: values[name] for name in spec.covariates}}
    steps = _steps(mstop)
    return lambda: causal._fit_nuisance(
        spec.treatment, train, rhs, Binomial(), smooth=flag, mstop=steps,
        max_mstop=steps, tune=False, inner_folds=2, seed=0)


def plan_aipw_bound(a, data, spec):
    """(fits, runner, explanation) of an aipw call from its bound arguments."""
    from mboostrs import causal

    outcome_missing = causal._check_outcome_missing(
        causal._spec_default(spec, "outcome_missing", a["outcome_missing"]))
    values = causal._columns(data, spec, categorical=True, outcome_missing=outcome_missing)
    folds = a["folds"]
    repeats = a["repeats"] if isinstance(folds, (int, np.integer)) else 1
    n_folds = _fold_count(folds)
    if a["propensity_terms"] is not None:
        propensity = 1
    else:
        flags = causal._smooth_by_nuisance(a["smooth"])
        names = causal._smooth_baselines(values, spec.covariates)
        base = causal._rhs(spec.covariates, flags["propensity"], a["df"], smooth_names=names)
        propensity = len(causal._propensity_candidates(
            a["propensity_learner"], base, values, spec.covariates))
    response = int(_use_response(data, spec, a["outcome_missing"]))
    outcome, outcome_text = _outcome_models(a["outcome_model"], data, spec)
    per_fold = propensity + response + outcome
    models = repeats * n_folds * per_fold + 1
    each = _tuned_fits(a["tune"], a["inner_folds"])
    why = (f"{models} nuisance models ({repeats} repeat(s) x {n_folds} folds x {per_fold} "
           f"models [{propensity} propensity, {response} response, {outcome} outcome] plus "
           "the all-rows propensity model), each "
           + (f"tuned by {a['inner_folds']}-fold cross-validation ({each} fits)"
              if a["tune"] else "one fit at its fixed step count")
           + f"; {outcome_text}")
    run = _propensity_runner(data, spec, a["smooth"], a["df"], a["mstop"])
    if a.get("interval") == "bootstrap":
        passes = 1 + int(a["bootstrap_draws"])
        why += (f"; interval='bootstrap' reruns the whole procedure on "
                f"{a['bootstrap_draws']} resamples, so the count is multiplied by {passes} "
                "(the call itself plus the draws)")
        return models * each * passes, run, why
    return models * each, run, why


def plan_aipw(a):
    return plan_aipw_bound(a, a["data"], a["spec"])


def _effect_model_parts(a, data, spec, *, terms, interval_method, nuisance):
    """Fits of one effect_model pipeline, by part."""
    from mboostrs import causal
    from mboostrs.explore import _tune_mstop_fits

    folds = a["folds"]
    modifiers = tuple(spec.covariates if a["modifiers"] is None else a["modifiers"])
    inner = _tuned_fits(True, FIXED_INNER_FOLDS)
    parts = {}
    if nuisance is None:
        values = causal._columns(data, spec, outcome_missing="ipw", categorical=True,
                                 owner="effect_model")
        base = causal._nuisance_rhs(spec, values, a["df"])
        candidates = len(causal._propensity_candidates(
            a["propensity_learner"], base, values, spec.covariates))
        response = int(_use_response(data, spec, a["outcome_missing"]))
        parts["nuisances"] = folds * (candidates + 1 + response) * inner
    kinds = ("smooth", "stumps", "joint") if terms == "auto" else (terms,)
    if terms == "auto" and len(modifiers) < 2:
        kinds = ("smooth", "stumps")
    second = 0
    for kind in kinds:
        second += folds  # the candidate refitted on each fold's training rows
        if kind != "smooth":
            second += _tune_mstop_fits(folds)
    parts["second stage"] = second
    smooth_possible = "smooth" in kinds
    parts["final smooth fit"] = int(smooth_possible)
    parts["propensity model"] = 1
    if interval_method == "auto":
        interval_method = "second_stage_bootstrap" if smooth_possible else "refit"
    refits = (a["interval_repeats"] - 1 if interval_method == "repeats"
              else a["bootstrap_draws"] if interval_method == "bootstrap" else 0)
    if refits:
        one = _effect_model_parts(a, data, spec, terms="smooth", interval_method="refit",
                                  nuisance=None)
        parts["interval refits"] = refits * sum(one.values())
    return parts


def plan_effect_model(a):
    data, spec = a["data"], a["spec"]
    parts = _effect_model_parts(a, data, spec, terms=a["terms"],
                                interval_method=a["interval_method"],
                                nuisance=a["nuisance"])
    fits = sum(parts.values())
    why = (", ".join(f"{name} {count}" for name, count in parts.items() if count)
           + f" = {fits} fits; a tuned nuisance counts its {FIXED_INNER_FOLDS} inner folds "
           "plus one full-data fit, a second-stage refit counts as one"
           + ("; terms='auto' counts the final smooth fit, which a boosted choice omits"
              if a["terms"] == "auto" else ""))
    return fits, _propensity_runner(data, spec, True, a["df"], TIMING_STEPS), why


def plan_ci_test(a):
    """Count the residual regressions by running ``ci_test`` with the
    regression function replaced by a recorder, so the count follows the
    call's own reduction of the conditioning set and its choice of residual
    functions."""
    from mboostrs import causal

    data, x, y = a["data"], a["x"], a["y"]
    given = tuple(a["given"])
    recorded = []
    original = causal._crossfit_prediction

    def record(_data, _given, values, fold, _seed, **_kwargs):
        recorded.append(len(np.unique(fold)))
        return np.zeros(np.asarray(values).size), False

    causal._crossfit_prediction = record
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            causal.ci_test(**{**a, "given": given})
    finally:
        causal._crossfit_prediction = original
    per = _tuned_fits(True, FIXED_INNER_FOLDS)
    fits = sum(folds * per for folds in recorded)

    def run():
        import mboostrs as mb
        from mboostrs._control import boost_control

        target = x if causal._is_categorical(data, y) else y
        if not recorded or causal._is_categorical(data, target):
            return
        terms = [f"bols({quote_name(g)})" if causal._is_categorical(data, g) else quote_name(g) for g in given]
        mb.gamboost(f"{quote_response(target)} ~ " + " + ".join(terms), data=data, na_action="omit",
                    control=boost_control(TIMING_STEPS, .3))

    if recorded:
        why = (f"{len(recorded)} residual regressions (the response, each residual function "
               "of x, and the response model when outcome_missing='ipw' and outcomes are "
               f"missing), each over {recorded[0]} folds with {FIXED_INNER_FOLDS}-fold "
               f"step tuning ({per} fits per fold)")
    else:
        why = ("no regression is fitted: the conditioning set is empty or the "
               "test is degenerate")
    return fits, run, why


def plan_iv_effect(a):
    data, spec = a["data"], a["spec"]
    n_folds = _fold_count(a["folds"])
    response = int(_use_response(data, spec, a["outcome_missing"]))
    per_fold = 3 + response
    each = _tuned_fits(a["tune"], a["inner_folds"])
    why = (f"{n_folds} folds x {per_fold} nuisance models (outcome, treatment, instrument"
           + (", response" if response else "") + "), each "
           + (f"tuned by {a['inner_folds']}-fold cross-validation ({each} fits)"
              if a["tune"] else "one fit at its fixed step count"))
    run = _propensity_runner(data, spec, a["smooth"], a["df"], a["mstop"])
    return n_folds * per_fold * each, run, why


def plan_confounding_benchmark(a):
    from mboostrs import causal
    from mboostrs.cost import _bound

    data, result = a["data"], a["result"]
    if isinstance(data, causal.AIPWResult):
        data, result = result, data
    if not isinstance(result, causal.AIPWResult):
        raise TypeError("result must be an AIPWResult; the call is "
                        "confounding_benchmark(data, result)")
    available = result.spec.covariates
    selected = available if a["variables"] is None else tuple(a["variables"])
    nuisance = result.nuisance
    options = {
        "folds": a["folds"] if a["folds"] is not None else nuisance["fold_count"],
        "repeats": nuisance["fold_repeats"], "smooth": nuisance["smooth"],
        "df": nuisance["df"], "mstop": nuisance["mstop"],
        "max_mstop": nuisance["max_mstop"], "tune": nuisance["tuned"],
        "outcome_model": nuisance["outcome_model"],
        "outcome_interactions": nuisance["outcome_interactions"],
        "inner_folds": nuisance["inner_folds"] or 5,
        "outcome_missing": nuisance.get("outcome_missing", "error"),
    }
    first = selected[0]
    spec = replace(result.spec, covariates=tuple(c for c in available if c != first))
    bound, _extra = _bound(causal.aipw, (data, spec), options)
    fits, run, text = plan_aipw_bound(bound, data, spec)
    return fits * len(selected), run, (
        f"{len(selected)} refits of aipw, each omitting one covariate: {text}")
