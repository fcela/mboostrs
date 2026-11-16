"""Frozen production count-backbone comparison for #660."""

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
import lzma
import os
from pathlib import Path
import time
import traceback
import warnings

os.environ.update(
    {
        key: "1"
        for key in (
            "RAYON_NUM_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
    }
)

import mboostrs as mb
import polars as pl
import numpy as np

from benches.evidence_float_codec import encode_record
from benches.observe_calibration_scores import native_calls, control_predictions
from benches.study_discovery_calibration import fit_band, partition
from benches.study_evidence import data_digest, evidence_document
from benches.study_poisson_conformal import CONTROLS, predicted_record
from benches.study_poisson_tuning_consistency import PANELS as BASE_PANELS, draw as baseline_draw, observe_tuning

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "docs/book/benchmarks/count_mean_backbone"
MODULE = importlib.import_module("mboostrs.discover")


PANELS = (
    *BASE_PANELS,
    "nonlinear_poisson",
    "nonlinear_zero_inflated",
    "null_poisson",
    "null_zero_inflated",
    "noise_zero_inflated",
    "interaction_zero_inflated",
    "interaction_supplied_family",
    "nonlinear_interaction_zero_inflated",
)
ORIGINAL_MEAN = MODULE._Analysis._mean_effects


def draw(panel, seed):
    if panel == "interaction_supplied_family":
        panel = "interaction_zero_inflated"
    if panel in BASE_PANELS:
        return baseline_draw(panel, seed)
    rng = np.random.default_rng(seed)
    n = 1000
    names = ("x1", "x2", "n0", "n1") if panel == "noise_zero_inflated" else ("x1", "x2")
    x = rng.uniform(-1, 1, (n, len(names)))
    x1, x2 = x[:, 0], x[:, 1]
    eta = (
        np.full(n, 0.3)
        if panel.startswith("null")
        else 0.3 + 0.8 * np.sin(np.pi * x1) - 0.4 * x2 + 1.2 * x1 * x2
        if panel == "nonlinear_interaction_zero_inflated"
        else 0.3 + 0.8 * np.sin(np.pi * x1) - 0.4 * x2
        if panel.startswith("nonlinear")
        else 0.3 + 0.6 * x1 - 0.4 * x2 + 1.2 * x1 * x2
        if panel == "interaction_zero_inflated"
        else 0.3 + 0.6 * x1 - 0.4 * x2
    )
    mean = 10 * np.exp(eta)
    if panel.endswith("zero_inflated"):
        keep = rng.uniform(size=n) > 0.3
        response = keep * rng.poisson(mean / 0.7)
        variance = mean + 0.3 / 0.7 * mean**2
    else:
        response = rng.poisson(mean)
        variance = mean
    return pl.DataFrame({**{name: x[:, i] for i, name in enumerate(names)}, "y": response}), mean, np.sqrt(variance)


def fitted(train, seed, method, panel):
    if method.endswith("_discovery"):
        trace = []
        family_kwargs = {"family": mb.NBinomial()} if panel == "interaction_supplied_family" else {}
        if method == "shape_discovery":
            # The primary candidate is the actual production default, without a policy override.
            with observe_tuning(trace):
                result = mb.discover(train, "y", seed=seed, **family_kwargs)
        else:
            with profile(method in ("rate_discovery", "combined_discovery")), observe_tuning(trace):
                result = mb.discover(
                    train, "y", seed=seed, simplify_mean=method == "combined_discovery", **family_kwargs
                )
        model = result.model
        return model, dict(
            family=result.family,
            formula=result.formula,
            mstop=result.mstop,
            nu=model._control.nu,
            tuning=trace,
            log=result.log,
            stage_errors=[
                entry
                for entry in result.log
                if entry["stage"] in ("mean effects", "count mean forms")
                and entry.get("reason", "").startswith("error:")
            ],
            provenance=result.provenance,
        )
    formula_panel = "interaction_zero_inflated" if panel == "interaction_supplied_family" else panel
    panel = formula_panel
    mean = (
        "1"
        if panel.startswith("null")
        else "bbs(x1, df=4) + bols(x2) + bols(x1, by=x2, intercept=False)"
        if panel == "nonlinear_interaction_zero_inflated"
        else "bbs(x1, df=4) + bols(x2)"
        if panel.startswith("nonlinear")
        else "bols(x1) + bols(x2) + bols(x1, by=x2, intercept=False)"
        if panel == "interaction_zero_inflated"
        else "bols(x1) + bols(x2)"
    )
    if method == "supplied_fixed" or mean == "1":
        nu = 0.1 if method == "supplied_fixed" else 0.02
        model = mb.gamboost("y ~ " + mean, train, family=mb.Poisson(), control=mb.boost_control(100, nu))
        return model, dict(family="Poisson", formula=mean, nu=nu, mstop=100, offset_only=mean == "1")
    tuning = mb.tune_mstop(
        "y ~ " + mean, train, family=mb.Poisson(), nu=0.02, folds=5, seed=seed, start=100, max_mstop=400
    )
    return tuning.model, dict(family="Poisson", formula=mean, nu=0.02, mstop=tuning.mstop, risk=tuning.risk)


@contextmanager
def profile(candidate):
    original = MODULE._screening_nu

    def rate(family):
        if candidate and isinstance(family, mb.NBinomial):
            return 0.1
        return 0.1 if isinstance(family, mb.Gaussian) else 0.02

    MODULE._screening_nu = rate
    try:
        yield
    finally:
        MODULE._screening_nu = original


def run(phase, panel, output):
    metadata_path = output.with_suffix(".metadata.json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    seeds = range(6607100, 6607110) if phase == "development" else range(6608100, 6608150)
    document = evidence_document(
        {
            "id": "count-mean-backbone/1",
            "phase": phase,
            "sha256": hashlib.sha256((DIRECTORY / "PROTOCOL.md").read_bytes()).hexdigest(),
        },
        {"seeds": list(seeds), "panel": panel, "test_seed_offset": 100000},
        [
            Path(__file__),
            ROOT / "benches/prototype_count_mean_backbone.py",
            ROOT / "benches/study_poisson_tuning_consistency.py",
            ROOT / "benches/study_discovery_calibration.py",
            ROOT / "benches/observe_calibration_scores.py",
            ROOT / "benches/study_evidence.py",
            ROOT / "benches/evidence_float_codec.py",
        ],
    )
    source = Path(mb.__file__).parent
    document["loaded_sources"] = {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest()
        for name in ("discover.py", "explore.py", "conformal.py", "gamboost.py", "families.py")
    }
    document.pop("results")
    metadata_path.write_text(json.dumps(document, indent=2) + "\n")
    count, digest = 0, hashlib.sha256()
    with lzma.open(output, "xb", filters=[{"id": lzma.FILTER_LZMA2, "preset": 3, "dict_size": 16 * 1024**2}]) as stream:

        def write(row):
            nonlocal count
            line = (json.dumps(encode_record(row), allow_nan=False) + "\n").encode()
            stream.write(line)
            stream.flush()
            digest.update(line)
            count += 1

        for seed in seeds:
            analysis, _, _ = draw(panel, seed)
            test, truth, sd = draw(panel, seed + 100000)
            train, calibration, parts = partition(analysis, seed)
            base = dict(
                panel=panel,
                seed=seed,
                partition=parts,
                test_response=test["y"].to_list(),
                true_mean=truth,
                true_scale=sd,
                calibration_response=calibration["y"].to_list(),
                input_digest=data_digest([analysis.to_numpy(), test.to_numpy(), truth]),
            )
            for method in (
                "baseline_discovery",
                "rate_discovery",
                "shape_discovery",
                "combined_discovery",
                "supplied_fixed",
                "supplied_tuned",
                *CONTROLS,
            ):
                row, events = {**base, "method": method}, []
                started = time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    try:
                        if method in CONTROLS:
                            with control_predictions(events):
                                low, high, point, details = fit_band(method, "poisson", seed, analysis, test)
                            assert details["partition"] == parts
                            cal = [np.asarray(v["prediction"]) for v in events if v["rows"] == 500]
                            y = calibration["y"].to_numpy()
                            scores = (
                                np.abs(y - cal[0]) if method.endswith("split") else np.maximum(cal[0] - y, y - cal[1])
                            )
                            row.update(
                                calibration_scores=scores,
                                calibration_quantile=np.sort(scores)[int(np.ceil(501 * 0.9)) - 1],
                            )
                            row["control_predictions"] = events
                        else:
                            model, details = fitted(train, seed, method, panel)
                            try:
                                details["coefficients"] = model.coef()
                            except Exception as error:
                                details["coefficient_capture_error"] = str(error)
                            with native_calls(events):
                                band = mb.conformal_interval(
                                    model,
                                    test.drop("y"),
                                    calibration=calibration,
                                    seed=seed,
                                    level=0.9,
                                    extrapolation="clip",
                                )
                            low, high, point = band.lower, band.upper, band.fitted_mean
                            row.update(
                                native_calls=events,
                                resolved_score=band.score,
                                quantile=band.quantile,
                                zero_probability=band.zero_probability,
                                reference_dispersion=band.reference_dispersion,
                                randomization_seed=band.randomization_seed,
                            )
                        row.update(details=details, true_mean_mse=float(np.mean((np.asarray(point) - truth) ** 2)))
                        if np.isfinite(np.asarray(low) + high + point).all():
                            row.update(status="ok", **predicted_record(low, high, point, test, sd))
                        else:
                            row.update(status="unbounded", predictions=dict(lower=low, upper=high, point=point))
                    except Exception:
                        row.update(status="failed", traceback=traceback.format_exc(), observed_calls=events)
                write({**row, "elapsed": time.perf_counter() - started, "warnings": [str(w.message) for w in caught]})
            print(json.dumps(dict(panel=panel, seed=seed, records=count)), flush=True)
    document.update(
        records=count,
        compressed_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        uncompressed_sha256=digest.hexdigest(),
    )
    metadata_path.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=PANELS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("development", "confirmation"), required=True)
    args = parser.parse_args()
    run(args.phase, args.panel, args.output)
