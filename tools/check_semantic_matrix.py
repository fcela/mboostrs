#!/usr/bin/env python3
"""Run the end-to-end matrix P1.2 specifies, and hold what it covers.

P1.2 names a minimum matrix that crosses

* `glmboost`, `gamboost`, cyclic LSS and non-cyclic LSS;
* unit, integer, fractional and zero-one weights;
* in-sample, truncated-path and new-data prediction; and
* exact, binned, and serialized/reloaded fits where applicable.

That cross was covered in pieces: a weighted-parity test here, a serialization
round-trip there, a binning test somewhere else, and no single place that says
which of the cells actually run. "Where applicable" is doing real work in that
sentence -- `glmboost` takes no `bins=` at all, because it always appends an
intercept learner and an intercept-carrying `bols` has no binned design -- and
an uncovered cell and an inapplicable one look identical from outside.

This fits every applicable cell and checks the invariant that has to hold in
all of them: **a model asked to predict its own training rows must return its
own fitted values.** Everything the matrix varies is a way of storing or
replaying the fit -- a weight rescaling, a binned design, a JSON round trip, a
truncated path -- and each is a chance to lose the mapping from a row to its
prediction. The invariant does not depend on any of them being right about
upstream; it depends on the model being self-consistent, which is checkable
without an oracle and is what a serialization or replay bug breaks first.

`INAPPLICABLE` records the cells that cannot exist, with the reason. An entry
is a statement about the API, not permission to skip a cell.

Run from the repository root:

    python tools/check_semantic_matrix.py
    python tools/check_semantic_matrix.py --show
    python tools/check_semantic_matrix.py --report
"""

from __future__ import annotations

import argparse
import itertools
import pathlib
import sys
import tempfile
import warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import mboostrs as mb  # noqa: E402

MODELS = ("glmboost", "gamboost", "glmboostLSS", "gamboostLSS")
WEIGHT_KINDS = ("unit", "integer", "fractional", "zeroone")
FIT_MODES = ("exact", "binned", "serialized")
PREDICTIONS = ("in_sample", "truncated", "newdata")

#: Cells the API does not have, and why. Keyed by `(model, fit_mode)`.
INAPPLICABLE = {
    ("glmboost", "binned"):
        "glmboost takes no bins=: it always appends an intercept learner, and "
        "an intercept-carrying bols has no binned design",
    ("glmboostLSS", "binned"):
        "glmboostLSS takes no bins=, for the same reason as glmboost",
}

#: How close a replayed prediction has to be to the fit's own values. This is
#: the same arithmetic on the same design, so the gap is rounding, not
#: approximation -- a binned fit is compared with itself, never with an exact
#: one.
TOLERANCE = 1e-9


def fixture(seed: int = 0, n: int = 150) -> dict:
    rng = np.random.default_rng(seed)
    data = {"x": rng.normal(size=n), "z": rng.normal(size=n)}
    data["y"] = (data["x"] + np.sin(2 * data["z"])
                 + rng.normal(scale=0.3, size=n))
    return data


def weight_vector(kind: str, n: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "unit":
        return np.ones(n)
    if kind == "integer":
        return rng.integers(1, 4, size=n).astype(float)
    if kind == "fractional":
        return rng.uniform(0.3, 2.0, size=n)
    if kind == "zeroone":
        drawn = (rng.random(n) > 0.25).astype(float)
        drawn[0] = 1.0            # never an all-zero vector, which is refused
        return drawn
    raise ValueError(kind)


def build(model: str, data: dict, weights: np.ndarray, bins: int | None):
    control = mb.boost_control(30, 0.1)
    if model == "glmboost":
        return mb.glmboost("y ~ x + z", data, control=control, weights=weights)
    if model == "gamboost":
        return mb.gamboost("y ~ bols(x) + bbs(z, df=4)", data, control=control,
                           weights=weights, bins=bins)
    if model == "glmboostLSS":
        return mb.glmboostLSS("y ~ x + z", data, family=mb.GaussianLSS(),
                              control=control, weights=weights,
                              method="cyclic")
    return mb.gamboostLSS("y ~ bbs(x, df=4) + bbs(z, df=4)", data,
                          family=mb.GaussianLSS(), control=control,
                          weights=weights, method="noncyclic",
                          bins=bins)


def as_array(value) -> np.ndarray:
    """One vector for a standard fit, the parameters concatenated for an LSS
    one, so the same comparison serves both."""
    if isinstance(value, dict):
        return np.concatenate([np.asarray(part).ravel()
                               for _, part in sorted(value.items())])
    return np.asarray(value).ravel()


def fitted_values(model) -> np.ndarray:
    attribute = getattr(model, "fitted", None)
    return as_array(attribute() if callable(attribute) else model.fitted_)


def half_path(model) -> int | dict:
    """A position strictly inside the path, per parameter for an LSS fit."""
    if getattr(model, "_method", "cyclic") == "noncyclic":
        # A non-cyclic fit takes a total and divides it between parameters
        # itself, so the half path is half of that total.
        return max(1, model._budget() // 2)
    mstop = model.mstop
    if isinstance(mstop, dict):
        return {name: max(1, value // 2) for name, value in mstop.items()}
    return max(1, mstop // 2)


def run_cell(model_name: str, weight_kind: str, fit_mode: str,
             prediction: str) -> str | None:
    """`None` when the cell holds, else what went wrong."""
    data = fixture()
    weights = weight_vector(weight_kind, len(data["y"]))
    bins = 32 if fit_mode == "binned" else None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = build(model_name, data, weights, bins)

        if prediction == "truncated":
            position = half_path(fit)
            fit = fit.with_mstop(position)

        target = fit
        if fit_mode == "serialized":
            path = pathlib.Path(tempfile.mkdtemp(prefix="mboostrs-matrix-"))
            artifact = path / "model.json"
            fit.save(str(artifact))
            target = mb.load_model(str(artifact))

        expected = fitted_values(fit)
        if prediction == "in_sample":
            got = as_array(target.predict(type="link"))
        else:
            got = as_array(target.predict(newdata=data, type="link"))

    if expected.shape != got.shape:
        return (f"fitted values have shape {expected.shape} and the "
                f"prediction {got.shape}")
    gap = float(np.max(np.abs(expected - got)))
    if not np.isfinite(gap):
        return "the comparison is not finite"
    if gap > TOLERANCE:
        return (f"predicting its own training rows differs from fitted() by "
                f"{gap:.3e}, over the {TOLERANCE:.0e} tolerance")
    return None


def cells():
    for model, weight_kind, fit_mode, prediction in itertools.product(
            MODELS, WEIGHT_KINDS, FIT_MODES, PREDICTIONS):
        if (model, fit_mode) in INAPPLICABLE:
            continue
        yield model, weight_kind, fit_mode, prediction


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true",
                        help="print every cell, not only the failures")
    parser.add_argument("--report", action="store_true",
                        help="print the matrix and the skipped cells, then stop")
    arguments = parser.parse_args()

    if arguments.report:
        total = len(MODELS) * len(WEIGHT_KINDS) * len(FIT_MODES) * len(PREDICTIONS)
        print(f"{total} cells in the full cross, "
              f"{len(list(cells()))} applicable")
        for (model, mode), reason in sorted(INAPPLICABLE.items()):
            skipped = len(WEIGHT_KINDS) * len(PREDICTIONS)
            print(f"  {model}/{mode}: {skipped} cells skipped — {reason}")
        return 0

    problems = []
    passed = 0
    for model, weight_kind, fit_mode, prediction in cells():
        label = f"{model}/{weight_kind}/{fit_mode}/{prediction}"
        try:
            failure = run_cell(model, weight_kind, fit_mode, prediction)
        except Exception as error:  # noqa: BLE001
            failure = f"{type(error).__name__}: {error}"
        if failure:
            problems.append(f"{label}: {failure}")
        else:
            passed += 1
            if arguments.show:
                print(f"  {label}: replays its own fit")

    if problems:
        print("Semantic matrix failed:", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1
    print(f"Semantic matrix: {passed} cell(s) hold the replay invariant; "
          f"{sum(len(WEIGHT_KINDS) * len(PREDICTIONS) for _ in INAPPLICABLE)} "
          f"inapplicable, with reasons")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
