"""Definitions and validation for upstream-compatible stopping criteria."""

import numpy as np

from mboostrs._validate import check_positive


def aic_contract(family, method, k, risk_kind):
    """Validate the requested criterion and return its statistical metadata.

    The corrected Gaussian criterion is per-observation and uses no tunable
    penalty multiplier. Classical AIC uses the reported total family risk.
    Neither the hat trace nor these criteria account for adaptive model
    search; a tree step's trace is conditional on the partition it was
    fitted with and does not count split selection either.
    """
    if method not in ("corrected", "classical", "gmdl"):
        raise ValueError("aic method must be 'corrected', 'classical' or 'gmdl'")
    k = check_positive("k", k)
    if risk_kind != "inbag":
        raise ValueError("aic() requires risk='inbag'; use cvrisk() for held-out evaluation")
    name = type(family).__name__
    if method == "classical" and name in ("Gaussian", "GaussReg"):
        raise NotImplementedError(
            "classical AIC method is not implemented for Gaussian squared-error loss; "
            "its risk is not a negative log-likelihood. Use method='corrected' "
            "with Gaussian, or cvrisk()."
        )
    if method in ("corrected", "gmdl") and name != "Gaussian":
        raise NotImplementedError(
            f"{method} AIC method is only implemented for Gaussian; "
            f"{name} does not supply its Gaussian RSS contract. "
            "Use method='classical' for a supported likelihood family or cvrisk()."
        )
    if method != "classical" and k != 2:
        raise ValueError(f"k only applies to method='classical'; {method} has a fixed penalty")
    return {
        "criterion": f"mboost_{method}",
        "normalization": "total" if method == "classical" else "per_observation",
        "df_convention": "conditional_hat_trace",
        "penalty": k if method == "classical" else None,
    }


def weighted_hat(model, store, selected, coefs=None):
    """mboost's IRLS-weighted boosting hat matrix for a GLM family.

    mboost's `hatvalues` weights each step's update by the family's `fW` at
    the fit after that step (`bhatmat`) for Binomial and Poisson; for Gaussian
    and for families without `fW` it uses the unweighted matrix. Returns
    ``(traces, diagonal)`` from the learner store, or None when the
    unweighted matrix applies. `coefs` is the model's per-step coefficient
    list, which the store needs to build a `btree` step's matrix from the
    partition it was fitted with.
    """
    name = type(model._family).__name__
    if name not in ("Binomial", "Poisson") or not selected:
        return None
    step_fits = np.asarray(model.predict(aggregate="cumsum"), dtype=np.float64)
    return store.compute_hat_weighted(
        list(selected), [row.tolist() for row in step_fits[:len(selected)]],
        name.lower(), coefs=coefs)
