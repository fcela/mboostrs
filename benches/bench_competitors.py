"""Compare mboostrs against Python-native alternatives.

The existing benchmark suite (`benches/scenarios.json`) answers "is this faster
than R `mboost`?". It does not answer the question a Python modeler actually
asks: "should I use this instead of what I already have?"

That question cannot be answered on speed alone, because these libraries do not
fit the same model. A gradient-boosted tree ensemble and a component-wise
additive model give different things. So every row here reports **held-out
accuracy, time, and what the model can tell you**, and the runner suppresses a
row's timing unless that model cleared an accuracy floor on the same split — a
fast wrong answer is not a benchmark result. The floor is applied per row, so a
scenario prints timings for whichever competitors passed; `render`'s footer says
the same.

Rows where mboostrs is slower are kept. A model that is 3x slower but reports
per-term effects and calibrated intervals is a legitimate trade, and hiding
that row would make this marketing rather than measurement.

**Two caveats that must travel with any number from this file.**

1. *An "iteration" is not comparable across libraries.* 300 mboostrs
   iterations means 300 component-wise updates, each touching one base
   learner; 300 HistGradientBoosting iterations means 300 whole trees over
   all features. Equal iteration counts are a convention here, not a fair
   unit of work.

2. *The speed ratio is dominated by sample size.* On the same additive
   scenario, mboostrs against `HistGradientBoostingRegressor` measures 449x
   at n=1000, 136x at n=10,000, and 8.6x at n=50,000 — the tree ensemble's
   fixed overhead is what shows up on small data, not a 449x algorithmic
   advantage. Quoting the small-n number alone would be indefensible, so the
   runner sweeps sizes and prints all of them.

Competitors that are not installed are skipped and reported as such, never
silently dropped:

    pip install 'mboostrs[bench-compare]'
    python benches/bench_competitors.py
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np

# Minimum held-out performance before a timing number is meaningful.
R2_FLOOR = 0.5
ACC_FLOOR = 0.7
REPEATS = 5


@dataclass
class Row:
    scenario: str
    library: str
    model: str
    fit_ms: float | None = None
    predict_ms: float | None = None
    score: float | None = None
    status: str = "ok"
    note: str = ""
    capabilities: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# capability vocabulary — what a modeler gets besides a prediction
# --------------------------------------------------------------------------

_ADDITIVE = "per-term effects"
_INTERVALS = "calibrated intervals"
_MONOTONE = "shape constraints"
_DISTRIB = "full distribution"
_SELECTION = "variable selection"


def _timed(fn, repeats=REPEATS):
    """Median wall time in ms, plus the last returned value."""
    times, out = [], None
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        times.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(times)), out


# Sample sizes swept for the regression scenario. Small n flatters
# mboostrs (tree ensembles pay fixed overhead there); large n is where the
# comparison gets interesting.
REGRESSION_SIZES = (1_000, 10_000, 50_000)


def _regression_data(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(n, 6))
    y = (
        np.sin(2 * X[:, 0])
        + 0.5 * X[:, 1] ** 2
        + 1.2 * X[:, 2]
        + rng.normal(scale=0.3, size=n)
    )
    cut = n // 2
    return X[:cut], y[:cut], X[cut:], y[cut:]


def _classification_data(n=2000, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(n, 6))
    logit = 2.0 * np.sin(X[:, 0]) + X[:, 1] - 0.8 * X[:, 2]
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    cut = n // 2
    return X[:cut], y[:cut], X[cut:], y[cut:]


def _r2(y_true, y_pred):
    resid = float(np.sum((y_true - y_pred) ** 2))
    total = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - resid / total


# --------------------------------------------------------------------------
# competitors
# --------------------------------------------------------------------------


def _mboostrs_regression(Xtr, ytr, Xte):
    from mboostrs.sklearn import GAMBoostRegressor

    est = GAMBoostRegressor(df=4, mstop=300, nu=0.1)
    fit_ms, _ = _timed(lambda: est.fit(Xtr, ytr))
    pred_ms, pred = _timed(lambda: est.predict(Xte))
    return fit_ms, pred_ms, pred, [_ADDITIVE, _SELECTION, _MONOTONE, _INTERVALS, _DISTRIB]


def _sklearn_hgb_regression(Xtr, ytr, Xte):
    from sklearn.ensemble import HistGradientBoostingRegressor

    est = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, random_state=0)
    fit_ms, _ = _timed(lambda: est.fit(Xtr, ytr))
    pred_ms, pred = _timed(lambda: est.predict(Xte))
    return fit_ms, pred_ms, pred, []


def _pygam_regression(Xtr, ytr, Xte):
    from pygam import LinearGAM, s

    terms = s(0)
    for j in range(1, Xtr.shape[1]):
        terms = terms + s(j)
    est = LinearGAM(terms)
    fit_ms, _ = _timed(lambda: est.gridsearch(Xtr, ytr, progress=False), repeats=1)
    pred_ms, pred = _timed(lambda: est.predict(Xte))
    return fit_ms, pred_ms, pred, [_ADDITIVE, _INTERVALS]


def _lightgbm_regression(Xtr, ytr, Xte):
    import lightgbm as lgb

    est = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.1, verbose=-1)
    fit_ms, _ = _timed(lambda: est.fit(Xtr, ytr))
    pred_ms, pred = _timed(lambda: est.predict(Xte))
    return fit_ms, pred_ms, pred, [_MONOTONE]


# Penalty grid for the statsmodels GAM below. A single scalar alpha shared
# across all smooth terms, picked by in-sample AIC — the same spirit as
# pyGAM's default `gridsearch` (a 1-D penalty sweep scored on the training
# fit, not a separate validation split). statsmodels' own per-term penalty
# selector (`select_penweight`) optimizes one weight per smooth term via
# basinhopping, which is a Cartesian-product-sized search for 6 covariates
# and not tractable to run 3 sizes x 5 repeats within this benchmark; a
# shared-alpha sweep is the practical middle ground a modeler would actually
# reach for.
_SM_ALPHA_GRID = (0.01, 0.1, 0.3, 1.0, 3.0, 10.0)


def _statsmodels_regression(Xtr, ytr, Xte):
    from statsmodels.gam.api import BSplines, GLMGam

    def _fit():
        # df=6, degree=3 per term roughly matches mboostrs' bbs(df=4) and
        # pyGAM's default s() basis in flexibility (a few effective degrees
        # of freedom per smooth term).
        smoother = BSplines(Xtr, df=[6] * Xtr.shape[1], degree=[3] * Xtr.shape[1])
        best = None
        for alpha in _SM_ALPHA_GRID:
            res = GLMGam(ytr, smoother=smoother, alpha=alpha).fit()
            if best is None or res.aic < best[0]:
                best = (res.aic, res)
        return best[1], smoother

    fit_ms, (res, smoother) = _timed(_fit, repeats=1)

    # statsmodels' B-spline basis raises rather than extrapolating for points
    # outside the training knot range (a real, documented limitation:
    # https://github.com/statsmodels/statsmodels GLMGam docstring). Clipping
    # test features to the training range before evaluating the basis is the
    # standard workaround; it never touches the target and only affects the
    # handful of test rows that fall outside the training envelope.
    lo, hi = Xtr.min(axis=0), Xtr.max(axis=0)

    def _predict():
        Xte_clipped = np.clip(Xte, lo, hi)
        return res.predict(exog_smooth=Xte_clipped, transform=True)

    pred_ms, pred = _timed(_predict)
    return fit_ms, pred_ms, np.asarray(pred), [_ADDITIVE, _INTERVALS]


def _mboostrs_classification(Xtr, ytr, Xte):
    from mboostrs.sklearn import GAMBoostClassifier

    est = GAMBoostClassifier(df=4, mstop=300, nu=0.1)
    fit_ms, _ = _timed(lambda: est.fit(Xtr, ytr))
    pred_ms, pred = _timed(lambda: est.predict(Xte))
    return fit_ms, pred_ms, pred, [_ADDITIVE, _SELECTION, _MONOTONE]


def _sklearn_hgb_classification(Xtr, ytr, Xte):
    from sklearn.ensemble import HistGradientBoostingClassifier

    est = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0)
    fit_ms, _ = _timed(lambda: est.fit(Xtr, ytr))
    pred_ms, pred = _timed(lambda: est.predict(Xte))
    return fit_ms, pred_ms, pred, []


REGRESSION = [
    ("mboostrs", "GAMBoostRegressor", _mboostrs_regression),
    ("scikit-learn", "HistGradientBoostingRegressor", _sklearn_hgb_regression),
    ("pyGAM", "LinearGAM", _pygam_regression),
    ("statsmodels", "GLMGam", _statsmodels_regression),
    ("LightGBM", "LGBMRegressor", _lightgbm_regression),
]

CLASSIFICATION = [
    ("mboostrs", "GAMBoostClassifier", _mboostrs_classification),
    ("scikit-learn", "HistGradientBoostingClassifier", _sklearn_hgb_classification),
]


def run_scenario(name, entries, data, scorer, floor) -> list[Row]:
    Xtr, ytr, Xte, yte = data
    rows: list[Row] = []
    for library, model, fn in entries:
        try:
            fit_ms, pred_ms, pred, caps = fn(Xtr, ytr, Xte)
        except ImportError:
            rows.append(
                Row(name, library, model, status="not-installed",
                    note="pip install 'mboostrs[bench-compare]'")
            )
            continue
        except Exception as exc:  # noqa: BLE001
            rows.append(Row(name, library, model, status="error", note=str(exc)[:120]))
            continue

        score = float(scorer(yte, pred))
        row = Row(name, library, model, fit_ms, pred_ms, score, capabilities=caps)
        if score < floor:
            row.status = "below-accuracy-floor"
            row.note = f"held-out score {score:.3f} < floor {floor}"
        rows.append(row)
    return rows


def run_all(sizes=REGRESSION_SIZES) -> list[Row]:
    rows: list[Row] = []
    for n in sizes:
        rows += run_scenario(
            f"regression.additive.n={n}",
            REGRESSION,
            _regression_data(n=2 * n),
            _r2,
            R2_FLOOR,
        )
    rows += run_scenario(
        "classification.additive",
        CLASSIFICATION,
        _classification_data(),
        lambda yt, yp: float(np.mean(yt == yp)),
        ACC_FLOOR,
    )
    return rows


def render(rows: list[Row]) -> str:
    lines = [
        f"{'scenario':30}{'library':14}{'model':32}{'fit ms':>10}{'score':>8}  what it also gives you",
        "-" * 136,
    ]
    for r in rows:
        if r.status != "ok":
            lines.append(
                f"{r.scenario:30}{r.library:14}{r.model:32}{'—':>10}{'—':>8}  [{r.status}] {r.note}"
            )
            continue
        caps = ", ".join(r.capabilities) if r.capabilities else "point predictions only"
        lines.append(
            f"{r.scenario:30}{r.library:14}{r.model:32}{r.fit_ms:10.1f}{r.score:8.3f}  {caps}"
        )

    usable = [r for r in rows if r.status == "ok"]
    skipped = [r for r in rows if r.status == "not-installed"]
    lines.append("")
    lines.append(
        f"{len(usable)} comparable row(s); {len(skipped)} competitor(s) not installed."
    )
    if skipped:
        lines.append(
            "Missing competitors are reported, not dropped: "
            + ", ".join(sorted({r.library for r in skipped}))
        )
    lines.append(
        "Timing is only shown for models that cleared the held-out accuracy "
        "floor on the same split."
    )
    lines.append(
        "An 'iteration' is not comparable across libraries, and the speed "
        "ratio shrinks sharply with n — read the size sweep, not one row."
    )
    lines.append(
        "LightGBM and HistGradientBoosting fit tree ensembles, not structured "
        "additive models — they are not like-for-like with mboostrs/pyGAM/"
        "statsmodels on capability, only comparable on held-out score and speed."
    )
    return "\n".join(lines)


#: Libraries this runner may compare against. A competitor comparison is the
#: claim most sensitive to library version, and this file recorded none: its
#: rows carried no commit, no platform, no library versions, no seed, and no
#: repetition count, while `bench_all.py` publishes all of them through
#: `benchmark_contract`. The reasoning in this file was always the careful part;
#: the plumbing was the gap.
COMPETITOR_PACKAGES = (
    "mboostrs",
    "numpy",
    "scikit-learn",
    "lightgbm",
    "pygam",
    "statsmodels",
)


def provenance() -> dict:
    """Machine, commit, and competitor versions for the rows below.

    Reuses `benchmark_contract.machine_metadata`, which already returns
    everything except the competitor versions, and adds those through the same
    `_package_version` helper. Without this the docstring's `449x / 136x / 8.6x`
    ratios were not regenerable, because nothing recorded what they ran against.
    """
    try:
        from benches.benchmark_contract import _package_version, machine_metadata
    except ModuleNotFoundError:
        from benchmark_contract import _package_version, machine_metadata

    metadata = machine_metadata()
    metadata["versions"] = {
        name: _package_version(name) for name in COMPETITOR_PACKAGES
    }
    return metadata


def main(output_path=None):
    rows = run_all()
    print(render(rows))
    if output_path:
        payload = {
            "kind": "mboostrs-competitor-comparison",
            "provenance": provenance(),
            "rows": [asdict(r) for r in rows],
        }
        with open(output_path, "w") as handle:
            json.dump(payload, handle, indent=2)
    return rows


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else None)
