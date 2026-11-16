"""Response-unit diagnostic for #633; see the frozen protocol in the book."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import lognorm, norm

import mboostrs as mb

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "docs/book/benchmarks/lss_response_scale"


def inputs():
    rng = np.random.default_rng(6319)
    x = rng.uniform(-1, 1, 300)
    data = {"x": x, "y": np.exp(.3 + .6*x + .15*rng.normal(size=300))}
    permutation = np.random.default_rng(9).permutation(300)
    return data, {"x": np.array([-.5, 0., .5])}, permutation[150:], permutation[:150]


def fit(data, transform):
    formula = f"{transform}(y) ~ bols(x)"
    return mb.gamboostLSS({"mu": formula, "sigma": formula}, data,
                         family=mb.GaussianLSS(), control=mb.boost_control(60))


def subset(data, indices):
    return {name: column[indices] for name, column in data.items()}


def reference(model, train, held, query, transform, mstop):
    def quantiles(rows):
        params = model.params(rows, type="response", mstop=mstop)
        mu, sigma = params["mu"], params["sigma"]
        if transform == "log":
            return [lognorm.ppf(p, s=sigma, scale=np.exp(mu)) for p in (.05, .5, .95)]
        mean = train["y"].mean()
        sd = train["y"].std(ddof=1) if transform == "scale" else 1.
        return [(mu + norm.ppf(p)*sigma)*sd + mean for p in (.05, .5, .95)]

    low, _, high = quantiles(held)
    scores = np.maximum(low-held["y"], held["y"]-high)
    rank = int(np.ceil((len(scores)+1)*.9))
    correction = np.sort(scores)[rank-1]
    low, middle, high = quantiles(query)
    low, high = low-correction, high+correction
    return {"lower": low, "predicted": np.clip(middle, low, high), "upper": high,
            "quantile": float(correction)}


def run():
    data, query, train_idx, cal_idx = inputs()
    train, held = subset(data, train_idx), subset(data, cal_idx)
    records = []
    for transform in ("log", "center", "scale"):
        full, proper = fit(data, transform), fit(train, transform)
        for mstop in (None, 30):
            expected = reference(proper, train, held, query, transform, mstop)
            for route in ("internal", "external"):
                try:
                    interval = (mb.conformal_interval(full, query, seed=9, mstop=mstop)
                                if route == "internal" else
                                mb.conformal_interval(proper, query, calibration=held, mstop=mstop))
                except (ValueError, NotImplementedError) as error:
                    records.append({"transform": transform, "mstop": mstop, "route": route,
                                    "max_absolute_error": None, "error": str(error)})
                    continue
                observed = {key: getattr(interval, key).tolist()
                            for key in ("lower", "predicted", "upper")}
                error = max(float(np.max(np.abs(observed[key]-expected[key])))
                            for key in observed)
                records.append({"transform": transform, "mstop": mstop, "route": route,
                                "max_absolute_error": error,
                                "fitted_mean_available": interval.fitted_mean is not None,
                                "observed": observed, "quantile": interval.quantile,
                                "reference": {key: value.tolist() if isinstance(value, np.ndarray)
                                              else value for key, value in expected.items()}})
    source_files = ("python/mboostrs/conformal.py", "python/mboostrs/gamlss.py",
                    "python/mboostrs/transforms.py", "benches/audit_lss_response_scale.py",
                    "docs/book/benchmarks/lss_response_scale/protocol.md")
    return {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                              for name in source_files},
            "data_sha256": hashlib.sha256(np.column_stack(list(data.values())).tobytes()).hexdigest(),
            "training_rows": train_idx.tolist(), "calibration_rows": cal_idx.tolist(),
            "records": records}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps([{key: row[key] for key in ("transform", "mstop", "route", "max_absolute_error")}
                      for row in result["records"]], indent=2))
