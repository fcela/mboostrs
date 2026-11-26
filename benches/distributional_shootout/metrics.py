"""Scores of a Forecast against held-out responses."""
from __future__ import annotations

import numpy as np

from tasks import LEVELS

CENTRAL = {"50": (0.25, 0.75), "80": (0.10, 0.90), "90": (0.05, 0.95), "98": (0.01, 0.99)}
_IDX = {round(float(t), 2): i for i, t in enumerate(LEVELS)}


def crps_from_quantiles(q, y):
    """Row CRPS approximated by the quantile decomposition
    CRPS = 2 * integral_0^1 QS_tau dtau, with the integral replaced by the mean
    over the 99 levels (Gneiting & Ranjan 2011, Laio & Tamea 2007)."""
    diff = q - y[:, None]
    qs = ((y[:, None] < q).astype(float) - LEVELS[None, :]) * diff
    return 2.0 * qs.mean(axis=1)


def score(forecast, y, truth=None, test_frame=None):
    q, mean = forecast.quantiles, forecast.mean
    if q.shape != (len(y), len(LEVELS)) or not np.all(np.isfinite(q)):
        raise ValueError(f"forecast quantiles malformed: shape {q.shape}, "
                         f"finite {bool(np.all(np.isfinite(q)))}")
    crps = crps_from_quantiles(q, y)
    out = {"crps": float(crps.mean()), "crps_se": float(crps.std(ddof=1) / np.sqrt(len(y)))}
    hits = (y[:, None] <= q).mean(axis=0)
    out["quantile_calibration_mae"] = float(np.mean(np.abs(hits - LEVELS)))
    out["quantile_calibration_max"] = float(np.max(np.abs(hits - LEVELS)))
    sd_y = float(np.std(y))
    for name, (lo, hi) in CENTRAL.items():
        a, b = q[:, _IDX[lo]], q[:, _IDX[hi]]
        inside = (y >= a) & (y <= b)
        out[f"coverage_{name}"] = float(inside.mean())
        out[f"width_{name}_over_sd"] = float(np.mean(b - a) / sd_y) if sd_y > 0 else None
        alpha = 1 - (hi - lo)
        interval = (b - a) + 2 / alpha * ((a - y) * (y < a) + (y - b) * (y > b))
        out[f"interval_score_{name}"] = float(interval.mean())
    out["rmse_mean"] = float(np.sqrt(np.mean((y - mean) ** 2)))
    out["mae_median"] = float(np.mean(np.abs(y - q[:, _IDX[0.5]])))
    if forecast.logpdf is not None:
        with np.errstate(all="ignore"):
            lp = np.asarray(forecast.logpdf(y), float)
        finite = np.isfinite(lp)
        out["nll"] = float(-lp.mean()) if finite.all() else None
        out["nll_nonfinite_rows"] = int((~finite).sum())
        out["nll_finite_rows_mean"] = float(-lp[finite].mean()) if finite.any() else None
    if truth is not None:
        oracle = crps_from_quantiles(truth["quantiles"], y)
        out["oracle_crps"] = float(oracle.mean())
        out["excess_crps"] = float(crps.mean() / oracle.mean() - 1)
        # Distance between predicted and true quantile functions, free of the
        # draw of y: mean absolute quantile error over rows and levels.
        out["quantile_error"] = float(np.mean(np.abs(q - truth["quantiles"])))
        out["oracle_nll"] = float(-np.mean(truth["logpdf"]))
        driver = truth.get("scale_driver")
        if driver is not None and test_frame is not None:
            x = test_frame[driver].to_numpy(float)
            cuts = np.quantile(x, [1 / 3, 2 / 3])
            group = np.digitize(x, cuts)
            a, b = q[:, _IDX[0.05]], q[:, _IDX[0.95]]
            inside = (y >= a) & (y <= b)
            cov = [float(inside[group == g].mean()) for g in range(3)]
            out["coverage_90_by_driver_tercile"] = cov
            out["coverage_90_tercile_max_dev"] = float(np.max(np.abs(np.array(cov) - 0.9)))
    return out
