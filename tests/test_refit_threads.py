"""gamlss_refit gives the same result on one and four rayon threads (#388, item 19).

The refit and its posterior draws run on the rayon pool in the Rust core. The
pool size is read from RAYON_NUM_THREADS when the extension starts, so each
thread count runs in its own interpreter.
"""

import json
import os
import subprocess
import sys

import numpy as np

CHILD = r"""
import json
import warnings

import numpy as np
import polars as pl

import mboostrs as mb

rng = np.random.default_rng(3)
n = 300
x1, x2, c = rng.uniform(0, 1, n), rng.uniform(0, 1, n), rng.normal(0, 1, n)
y = np.sin(3 * x1) + 0.8 * c + c * x2 + rng.normal(0, 0.3 + 0.3 * x2, n)
data = pl.DataFrame({"x1": x1, "x2": x2, "c": c, "y": y})
grid = data.head(40).drop("y")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fit = mb.gamboostLSS(
        {"mu": "y ~ bbs(x1, df=5) + bols(c) + bbs(x2, by=c, df=5)",
         "sigma": "y ~ bbs(x2, df=4)"},
        data=data, families=mb.GaussianLSS(),
        control=mb.boost_control(mstop=60),
    )
    refit = mb.gamlss_refit(fit)
draws = refit.draws(grid, n_draws=50, seed=0)

out = {
    "reml_score": float(refit.reml_score),
    "edf": {p: {b: float(v) for b, v in blocks.items()} for p, blocks in refit.edf.items()},
    "smoothing": {p: {b: (list(v) if isinstance(v, tuple) else float(v))
                      for b, v in blocks.items()}
                  for p, blocks in refit.smoothing.items()},
    "draws": {p: np.asarray(a).tolist() for p, a in draws.items()},
}
print(json.dumps(out))
"""


def _run_refit(threads):
    env = dict(os.environ, RAYON_NUM_THREADS=str(threads))
    proc = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True,
                          text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _assert_close(a, b, label):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert a.shape == b.shape, label
    scale = np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))
    worst = float(np.max(np.abs(a - b) / scale)) if a.size else 0.0
    assert worst <= 1e-10, f"{label}: relative difference {worst:.3e}"


def test_refit_matches_across_rayon_thread_counts():
    one = _run_refit(1)
    four = _run_refit(4)
    _assert_close(one["reml_score"], four["reml_score"], "reml_score")
    for param, blocks in one["edf"].items():
        for block, value in blocks.items():
            _assert_close(value, four["edf"][param][block], f"edf {param} {block}")
    for param, blocks in one["smoothing"].items():
        for block, value in blocks.items():
            _assert_close(value, four["smoothing"][param][block],
                          f"smoothing {param} {block}")
    for param, array in one["draws"].items():
        _assert_close(array, four["draws"][param], f"draws {param}")
