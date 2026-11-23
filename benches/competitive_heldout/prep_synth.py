"""Planted-truth tasks S1-S7 of PROTOCOL.md: five draws each, fixed train/test rows."""
import json
import pathlib

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "data"
OUT.mkdir(exist_ok=True)
P = 12

# truth: column -> structure type
TRUTH = {
    "S1": {"x0": "nonlinear", "x1": "nonlinear", "x2": "linear"},
    "S2": {"x0": "interaction", "x1": "interaction", "x2": "linear"},
    "S3": {"x0": "linear", "x1": "nonlinear", "x2": "scale_only"},
    "S4": {},
    "S5": {"x0": "nonlinear", "x1": "nonlinear", "x2": "linear"},
    "S6": {"x0": "nonlinear", "x1": "nonlinear", "x2": "linear", "x3": "proxy", "x4": "proxy"},
    "S7": {"x0": "nonlinear", "x1": "linear"},
}


def draw(task, g):
    rng = np.random.default_rng(10_000 * (int(task[1]) + 1) + g)
    n_train = 200 if task == "S5" else 2000
    n = n_train + 2000
    X = rng.uniform(-1, 1, (n, P))
    if task == "S6":
        X[:, 3] = X[:, 0] + rng.normal(0, 0.05, n)
        X[:, 4] = X[:, 1] + rng.normal(0, 0.05, n)
    x0, x1, x2 = X[:, 0], X[:, 1], X[:, 2]
    if task in ("S1", "S5", "S6"):
        y = 2 * np.sin(2 * x0) + 2 * (x1 ** 2 - 1 / 3) + x2 + rng.normal(0, 0.5, n)
    elif task == "S2":
        y = 3 * np.sign(x0 * x1) + x2 + rng.normal(0, 0.5, n)
    elif task == "S3":
        y = x0 + np.sin(3 * x1) + rng.normal(0, 1, n) * (0.25 + 1.5 * (x2 + 1) / 2)
    elif task == "S4":
        y = rng.normal(0, 1, n)
    elif task == "S7":
        p = 1 / (1 + np.exp(-(1.5 * np.sin(2 * x0) + x1 - 0.5)))
        y = (rng.uniform(size=n) < p).astype(float)
    frame = pl.DataFrame({f"x{i}": X[:, i] for i in range(P)} | {"y": y})
    return frame, n_train


tasks = []
for task, truth in TRUTH.items():
    for g in range(5):
        frame, n_train = draw(task, g)
        tid = f"{task}_{g}"
        frame.write_parquet(OUT / f"{tid}.parquet")
        tasks.append(dict(id=tid, name=tid, kind="binary" if task == "S7" else "regression",
                          n=frame.height, p=P, categorical=[], source="planted",
                          n_train=n_train, truth=truth,
                          columns={f"x{i}": f"x{i}" for i in range(P)}))
json.dump(dict(tasks=tasks, skipped=[]), open(OUT / "manifest_synth.json", "w"), indent=1)
print(len(tasks), "planted tasks")
