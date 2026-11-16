"""Frontend recipes for cross-validation with training-only preprocessing.

Rust requests one fold at a time, fits and predicts through the native model
API, restores its nuisance path, and computes family risks and aggregation.
This module subsets frames and carries fitted prediction recipes.
"""

from __future__ import annotations

import numpy as np

from mboostrs._frame import take_rows
from mboostrs._refit import require_row_subsettable
from mboostrs._workflow import checkpoint
from mboostrs.families import Multinomial, _family_params, _family_rust_name


def check_preprocessing(value):
    if not isinstance(value, str) or value not in ("fixed", "refit"):
        raise ValueError("preprocessing must be 'fixed' or 'refit'")
    return value


def refitted_cv_path(model, folds_matrix, judge_mask, weights, cache=None):
    """Prepare each fold's raw frame; the core drives and scores the CV path.

    `cache`, when given, is a dict that keeps each fold's fitted model and its
    path predictions between calls. A later call at a larger `model.mstop`
    continues the stored fold fit and predicts only the new steps; the steps
    before are unchanged because a boosting path is prefix-consistent.
    """
    from mboostrs import _rust
    from mboostrs._dispatch import is_glmboost
    from mboostrs.glmboost import (
        _binomial_cbind_response,
        _extract_response,
        _is_binomial_cbind,
        _replay_transforms,
    )
    from mboostrs.transforms import TransformRecipe, apply_recipe

    spec = require_row_subsettable(model, "cvrisk(preprocessing='refit')",
                                  refit_preprocessing=True)
    if isinstance(model.family, Multinomial):
        raise NotImplementedError(
            "preprocessing='refit' is not yet available for Multinomial: "
            "the fold response and class-coordinate mapping must be retained together")
    # The fold weights already hold the model's weights, which for a cbind
    # response include the trial counts; refitting the cbind formula
    # multiplies those counts in again, so they are taken out of the weights
    # handed to the fold fit.
    trials = (_binomial_cbind_response(spec.raw_data, model._response_name)[1]
              if _is_binomial_cbind(model.family, model._response_name) else None)
    response_step = getattr(model, "_response_transform", None)
    if response_step is not None and response_step.kind not in ("log", "log1p", "sqrt"):
        raise NotImplementedError(
            "preprocessing='refit' cannot aggregate a response transform whose "
            "scale changes between folds; use a fixed response transform")
    n, count = folds_matrix.shape
    control = spec.arguments["control"]._replace(mstop=model.mstop, risk="inbag")

    end = int(model.mstop)

    def fold_eta(fitted, held_data, evaluation, positions, options):
        """The linear predictor of one fold at the given path positions."""
        positions = list(positions)
        if options:
            prediction_options = {} if is_glmboost(fitted) else {"unseen_levels": "zero_effect"}
            return np.asarray([fitted.predict(held_data, mstop=k, type="link",
                              **options, **prediction_options) for k in positions], dtype=float)
        if is_glmboost(fitted):
            return fitted.predict_path(positions, newdata=_replay_transforms(fitted, held_data))
        return fitted.predict_over_mstop(positions, newdata=held_data, output="numpy",
                                         unseen_levels="zero_effect")

    def provide_fold(b):
        training = np.flatnonzero(folds_matrix[:, b] > 0)
        held = (folds_matrix[:, b] == 0) if judge_mask is None else judge_mask[:, b]
        evaluation = np.flatnonzero(held & (weights > 0))
        if not training.size or not evaluation.size:
            raise ValueError(f"fold {b} needs positive-weight training and evaluation rows")
        held_data = take_rows(spec.raw_data, evaluation)
        stored = cache.get(b) if cache is not None else None
        options = {}
        offset = spec.raw_arguments.get("offset") if spec.raw_arguments else None
        if offset is not None and np.ndim(offset) > 0:
            options["offset"] = np.asarray(offset)[evaluation]
        if stored is None:
            training_spec = spec.subset(training, refit_preprocessing=True)
            fold_weights = folds_matrix[training, b]
            if trials is not None:
                fold_weights = fold_weights / trials[training]
            fitted = training_spec.build(
                control=control, weights=fold_weights, weight_policy="raw")
            eta = fold_eta(fitted, held_data, evaluation, range(end + 1), options)
        else:
            fitted, stored_eta = stored
            done = stored_eta.shape[0] - 1
            if end > done:
                fitted = fitted.continue_to(end)
                new = fold_eta(fitted, held_data, evaluation, range(done + 1, end + 1), options)
                eta = np.concatenate([stored_eta, np.asarray(new, dtype=float).reshape(end - done, -1)])
            else:
                eta = stored_eta[: end + 1]
        if cache is not None:
            cache[b] = (fitted, eta)
        response_recipe = getattr(fitted, "_response_transform", None)
        scoring_data = held_data
        if response_recipe is not None:
            scoring_data = apply_recipe(TransformRecipe((response_recipe,)), held_data)
        observed = _extract_response(scoring_data, fitted._response_name, fitted.family)
        nuisance = (getattr(fitted, "_nuisance_vec_path", None)
                    if is_glmboost(fitted) else getattr(fitted, "_family_sigma_path", None))
        return (observed.tolist(), weights[evaluation].tolist(),
                np.ascontiguousarray(eta, dtype=np.float64), nuisance or [])

    checkpoint()
    # Each provider fit reserves its own charge. This call only schedules
    # callbacks and scores their paths, so charging it would count fits twice.
    return _rust.cvrisk_foldwise(_family_rust_name(model.family),
                                 _family_params(model.family), count, model.mstop, provide_fold)
