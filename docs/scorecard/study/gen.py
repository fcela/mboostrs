"""Planted-truth battery for the KDD scorecard study. Generates data and hidden truth."""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

DATA = Path(sys.argv[1])
TRUTH = Path(sys.argv[2])
P = 20
N_TEST = 1000
DATASETS = {
    # name: (n_train, seed, messy)
    "main": (2000, 20260101, False),
    "small": (200, 20260102, False),
    "messy": (2000, 20260103, True),
}


def mean_fn(x):
    # x: (n, 20) clean covariates, columns x1..x20 at indices 0..19
    return (1.0 * x[:, 0] + 0.5 * x[:, 1] + 2.0 * np.sin(x[:, 3])
            + 1.5 * (x[:, 4] > 0) + 1.0 * x[:, 5] * x[:, 6])


def sigma_fn(x):
    return 0.5 * np.exp(0.4 * x[:, 7])


def draw_x(rng, n):
    x = rng.standard_normal((n, P))
    x[:, 3] = rng.uniform(-3, 3, n)
    return x


def contrasts(x):
    """Mean change in f when each covariate moves from its 25th to its 75th percentile."""
    out = {}
    for j in range(P):
        lo, hi = np.quantile(x[:, j], [0.25, 0.75])
        a = x.copy(); a[:, j] = hi
        b = x.copy(); b[:, j] = lo
        out[f"x{j + 1}"] = float(np.mean(mean_fn(a) - mean_fn(b)))
    return out


def write(name, n, seed, messy):
    rng = np.random.default_rng(seed)
    x = draw_x(rng, n)
    contr = contrasts(x)
    eps = rng.standard_normal(n)
    y = mean_fn(x) + sigma_fn(x) * eps
    feats = x.copy()
    region = None
    if messy:
        out_idx = rng.choice(n, size=n // 100, replace=False)
        y[out_idx] += rng.choice([-1, 1], size=len(out_idx)) * 10.0
        region = rng.choice(["a", "b", "c"], size=n)
        miss = rng.random((n, P)) < 0.15
        feats[miss] = np.nan
    cols = {"row_id": np.arange(1, n + 1)}
    cols.update({f"x{j + 1}": feats[:, j] for j in range(P)})
    if messy:
        cols["region"] = region
    train = pl.DataFrame(cols).with_columns(pl.Series("y", y))
    train.write_csv(DATA / f"{name}_train.csv")

    # Test set: fresh draw, same generator, clean response (no outliers), features with the same missingness rule.
    trng = np.random.default_rng(seed + 1000)
    xt = draw_x(trng, N_TEST)
    mt = mean_fn(xt)
    st = sigma_fn(xt)
    yt = mt + st * trng.standard_normal(N_TEST)
    feats_t = xt.copy()
    tcols = {"row_id": np.arange(1, N_TEST + 1)}
    if messy:
        feats_t[trng.random((N_TEST, P)) < 0.15] = np.nan
        tcols["region"] = trng.choice(["a", "b", "c"], size=N_TEST)
    tcols.update({f"x{j + 1}": feats_t[:, j] for j in range(P)})
    pl.DataFrame(tcols).write_csv(DATA / f"{name}_test_features.csv")
    pl.DataFrame({"row_id": np.arange(1, N_TEST + 1), "y": yt, "mean": mt, "sigma": st}).write_csv(
        TRUTH / f"{name}_test_truth.csv")
    return contr


if __name__ == "__main__":
    spec = {"p": P, "n_test": N_TEST, "mean": "1.0*x1 + 0.5*x2 + 2*sin(x4) + 1.5*(x5>0) + 1.0*x6*x7",
            "sigma": "0.5*exp(0.4*x8)", "x4": "U(-3,3); others N(0,1)",
            "planted_main": ["x1", "x2", "x4", "x5"], "planted_pair": ["x6", "x7"],
            "variance_driver": "x8", "null": [f"x{j}" for j in range(1, 21) if j not in (1, 2, 4, 5, 6, 7, 8)],
            "datasets": {}}
    for name, (n, seed, messy) in DATASETS.items():
        contr = write(name, n, seed, messy)
        spec["datasets"][name] = {"n_train": n, "seed": seed, "messy": messy, "contrasts": contr}
    (TRUTH / "truth.json").write_text(json.dumps(spec, indent=2))
    print("ok")
