"""Calibration of `compare_specifications(..., mstop="cv")` by simulation.

Each replicate draws n = 200 rows with y = sin(x) + beta * u + N(0, 0.7^2),
compares ``bbs(x)`` against ``bbs(x) + bols(u)`` on five outer folds with the
step count tuned inside every outer fold (five inner folds), and records the
paired statistic z = change / change se of the second row. Under beta = 0 the
added term carries no signal, so z measures how the paired fold standard
error behaves when the candidates are equivalent. The Monte Carlo standard
error of each rate is printed next to it.

    python tools/simulate_nested_comparison.py --replicates 1000 --beta 0
    python tools/simulate_nested_comparison.py --replicates 400 --beta 0.15
"""

import argparse
import warnings

import numpy as np

import mboostrs as mb


def replicate(seed, beta, n=200):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    u = rng.normal(size=n)
    data = {"x": x, "u": u, "y": np.sin(x) + beta * u + rng.normal(scale=0.7, size=n)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        row = mb.compare_specifications(
            {"a": "bbs(x)", "b": "bbs(x) + bols(u)"}, data, "y",
            folds=5, mstop="cv", seed=seed)[1]
    return row["change"] / row["change se"] if row["change se"] > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    z = np.array([replicate(args.seed + r, args.beta) for r in range(args.replicates)])
    count = len(z)

    def rate(mask):
        p = float(np.mean(mask))
        return f"{p:.3f} (Monte Carlo SE {np.sqrt(p * (1 - p) / count):.3f})"

    print(f"beta = {args.beta}, {count} replicates")
    print(f"mean z {z.mean():.2f}, sd z {z.std(ddof=1):.2f} (1 for a calibrated standard error)")
    print(f"P(z < -2)  {rate(z < -2)}  (0.023 for a standard normal z)")
    print(f"P(|z| > 2) {rate(np.abs(z) > 2)}  (0.046 for a standard normal z)")


if __name__ == "__main__":
    main()
