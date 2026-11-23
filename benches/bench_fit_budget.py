"""Held-out loss and wall-clock time of autospec at several fit budgets.

Measures the evidence behind the budget mapping in `mboostrs.fit`
(`_BUDGET_FITS`; roadmap fcela/mboostrs-dev#913, item 2). Uses the
competitive-shootout panels and splits from `benches/shootout_data.py`:
autospec sees the fit and validation rows with its default folds, as `fit`
does, and the chosen model is scored on the test rows (MSE for regression,
log loss for binary panels). Errors are recorded and kept in the output.

Usage, from the repository root, one thread per process:

    RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        python benches/bench_fit_budget.py concrete,cancer 0,8,16,32,128,512 0,1,2 out.json

The panel cache (`benches/.cache/competitive_shootout`) is built by
`shootout_data.prepare`.
"""
import json
import sys
import time
import warnings

import numpy as np
import polars as pl

sys.path.insert(0, "benches")
import mboostrs as mb  # noqa: E402
from shootout_data import payload  # noqa: E402

def main():
    TASKS = sys.argv[1].split(",")
    BUDGETS = [int(b) for b in sys.argv[2].split(",")]
    SEEDS = [int(s) for s in sys.argv[3].split(",")]
    OUT = sys.argv[4]

    rows = []
    for task in TASKS:
        for seed in SEEDS:
            p = payload(task, seed)
            fit_rows, val, _cal, test = p["frames"]
            dev = pl.concat([fit_rows, val])
            for col in p["categorical"]:
                dev = dev.with_columns(pl.col(col).cast(pl.Utf8).cast(pl.Categorical))
                test = test.with_columns(pl.col(col).cast(pl.Utf8).cast(pl.Categorical))
            family = mb.Binomial() if p["kind"] == "binary" else mb.Gaussian()
            y = test["y"].to_numpy().astype(float)
            for budget in BUDGETS:
                t0 = time.perf_counter()
                status, loss, fits, formula = "ok", None, None, None
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        rec = mb.autospec(dev, "y", family=family, budget=budget, seed=seed)
                        model = rec.fit(dev)
                        pred = np.asarray(model.predict(newdata=test, type="response"), float).ravel()
                    fits, formula = rec.fits, str(rec.formula)
                    if p["kind"] == "binary":
                        q = np.clip(pred, 1e-12, 1 - 1e-12)
                        loss = float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))
                    else:
                        loss = float(np.mean((y - pred) ** 2))
                except Exception as exc:  # recorded, kept in the denominator
                    status = f"error: {type(exc).__name__}: {exc}"[:300]
                elapsed = time.perf_counter() - t0
                row = dict(task=task, seed=seed, budget=budget, status=status, loss=loss,
                           seconds=round(elapsed, 2), fits=fits, formula=formula)
                rows.append(row)
                print(json.dumps(row), flush=True)

    with open(OUT, "w") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
