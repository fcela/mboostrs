"""Tasks for the distributional shootout: planted-truth generators and OpenML-CTR23.

A task instance is (task name, seed). It yields a development frame (the rows
an arm may fit and tune on), a test frame, and for planted tasks the true
conditional quantiles of the test rows at LEVELS.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import special, stats

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
LEVELS = np.arange(1, 100) / 100.0
N_FEATURES = 10
SYNTH_TEST = 5000
SYNTH_SIZES = (500, 5000)
REAL_DEV_CAP = 10_000
REAL_TEST_CAP = 5000
REAL_TEST_SHARE = 0.2


# ---------------------------------------------------------------- planted truth
# Each generator takes the predictor matrix (n, 10), uniform on (-1, 1), and a
# generator, and returns (y, true quantiles at LEVELS, true log density at y).

def _normal(mu, sd, rng):
    y = mu + sd * rng.standard_normal(mu.shape)
    q = stats.norm.ppf(LEVELS[None, :], mu[:, None], sd[:, None])
    return y, q, stats.norm.logpdf(y, mu, sd)


def s1_hetero_gauss(x, rng):
    mu = 2 * np.sin(2 * x[:, 0]) + x[:, 1]
    sd = np.exp(-0.7 + 0.8 * x[:, 2])
    return _normal(mu, sd, rng)


def s2_student_t(x, rng):
    mu = 2 * np.sin(2 * x[:, 0]) + x[:, 1]
    scale = np.exp(-0.7 + 0.6 * x[:, 2])
    df = 3.0
    y = mu + scale * rng.standard_t(df, mu.shape)
    q = mu[:, None] + scale[:, None] * stats.t.ppf(LEVELS[None, :], df)
    return y, q, stats.t.logpdf(y, df, mu, scale)


def s3_gamma(x, rng):
    mean = np.exp(0.5 + 0.6 * np.sin(2 * x[:, 0]) + 0.4 * x[:, 1])
    shape = np.exp(1.0 + 0.8 * x[:, 2])
    scale = mean / shape
    y = rng.gamma(shape, scale)
    q = stats.gamma.ppf(LEVELS[None, :], shape[:, None], scale=scale[:, None])
    return y, q, stats.gamma.logpdf(y, shape, scale=scale)


def s4_lognormal(x, rng):
    m = 0.3 + np.sin(2 * x[:, 0]) + 0.3 * x[:, 1]
    s = 0.25 + 0.25 * (x[:, 2] + 1)
    z = m + s * rng.standard_normal(m.shape)
    y = np.exp(z)
    q = np.exp(m[:, None] + s[:, None] * stats.norm.ppf(LEVELS)[None, :])
    return y, q, stats.norm.logpdf(z, m, s) - z


def _mixture_quantiles(weights, means, sd):
    """Quantiles of sum_k w_k N(m_k, sd^2) by bisection, rows independent."""
    lo = means.min(axis=1) - 8 * sd
    hi = means.max(axis=1) + 8 * sd
    lo = np.repeat(lo[:, None], len(LEVELS), axis=1)
    hi = np.repeat(hi[:, None], len(LEVELS), axis=1)
    for _ in range(60):
        mid = (lo + hi) / 2
        cdf = sum(weights[:, k, None] * stats.norm.cdf((mid - means[:, k, None]) / sd)
                  for k in range(means.shape[1]))
        below = cdf < LEVELS[None, :]
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return (lo + hi) / 2


def s5_bimodal(x, rng):
    # Two modes 3 apart; x0 moves mass between them, x1 shifts both.
    p = special.expit(3 * x[:, 0])
    shift = 0.5 * x[:, 1]
    sd = 0.4
    upper = rng.random(len(p)) < p
    y = np.where(upper, 1.5, -1.5) + shift + sd * rng.standard_normal(len(p))
    weights = np.column_stack([p, 1 - p])
    means = np.column_stack([1.5 + shift, -1.5 + shift])
    q = _mixture_quantiles(weights, means, sd)
    dens = p * stats.norm.pdf(y, means[:, 0], sd) + (1 - p) * stats.norm.pdf(y, means[:, 1], sd)
    return y, q, np.log(dens)


def s6_skew_flip(x, rng):
    # Three regimes of x0 with mean 0 and variance 1 each: right-skewed
    # (Exp(1) - 1), Gaussian, left-skewed (1 - Exp(1)). Only the shape changes
    # with x0; x1 shifts the location.
    regime = np.digitize(x[:, 0], [-1 / 3, 1 / 3])
    loc = 0.8 * x[:, 1]
    e = rng.exponential(1.0, len(loc))
    z = rng.standard_normal(len(loc))
    eps = np.select([regime == 0, regime == 1], [e - 1, z], 1 - e)
    y = loc + eps
    right = -np.log1p(-LEVELS) - 1
    left = 1 + np.log(LEVELS)
    gauss = stats.norm.ppf(LEVELS)
    base = np.where((regime == 0)[:, None], right[None, :],
                    np.where((regime == 1)[:, None], gauss[None, :], left[None, :]))
    q = loc[:, None] + base
    r = y - loc
    logd = np.select([regime == 0, regime == 1],
                     [np.where(r > -1, -(r + 1), -np.inf), stats.norm.logpdf(r)],
                     np.where(r < 1, -(1 - r), -np.inf))
    return y, q, logd


def s7_interaction(x, rng):
    mu = 2 * x[:, 0] * x[:, 1] + x[:, 2]
    sd = np.exp(-0.7 + 0.8 * x[:, 0] * x[:, 1])
    return _normal(mu, sd, rng)


def s8_linear(x, rng):
    mu = x[:, 0] + 0.5 * x[:, 1] - 0.5 * x[:, 2]
    sd = np.full(len(mu), 0.5)
    return _normal(mu, sd, rng)


SYNTH = {
    "S1_hetero_gauss": s1_hetero_gauss,
    "S2_student_t3": s2_student_t,
    "S3_gamma": s3_gamma,
    "S4_lognormal": s4_lognormal,
    "S5_bimodal": s5_bimodal,
    "S6_skew_flip": s6_skew_flip,
    "S7_interaction": s7_interaction,
    "S8_linear": s8_linear,
}
# The predictor whose value sets the conditional spread, used to measure
# whether coverage adapts (None where the spread does not vary).
SCALE_DRIVER = {"S1_hetero_gauss": "x2", "S2_student_t3": "x2", "S3_gamma": "x2",
                "S4_lognormal": "x2", "S5_bimodal": "x0", "S6_skew_flip": "x0",
                "S7_interaction": None, "S8_linear": None}


def synth_task(name, n_dev, seed):
    generator = SYNTH[name]
    # Seed space is disjoint per (task, size, seed) and from the real splits.
    key = int(hashlib.sha256(f"{name}|{n_dev}|{seed}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(key)
    n = n_dev + SYNTH_TEST
    x = rng.uniform(-1, 1, (n, N_FEATURES))
    y, q, logd = generator(x, rng)
    frame = pd.DataFrame(x, columns=[f"x{i}" for i in range(N_FEATURES)])
    frame["y"] = y
    dev, test = frame.iloc[:n_dev].reset_index(drop=True), frame.iloc[n_dev:].reset_index(drop=True)
    truth = {"quantiles": q[n_dev:], "logpdf": logd[n_dev:],
             "scale_driver": SCALE_DRIVER[name]}
    return dev, test, truth


# ------------------------------------------------------------------ real data
CTR23 = 353  # OpenML study id of the OpenML-CTR23 regression suite


def prepare_real(out=DATA):
    """Download every CTR23 dataset once into parquet files plus a manifest."""
    import urllib.request

    from sklearn.datasets import fetch_openml

    out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"https://www.openml.org/api/v1/json/study/{CTR23}") as r:
        ids = sorted(int(i) for i in json.load(r)["study"]["data"]["data_id"])
    manifest = []
    for did in ids:
        bunch = fetch_openml(data_id=did, as_frame=True, parser="auto")
        frame = bunch.frame.copy()
        target = bunch.target_names[0]
        # Predictors are renamed first: diamonds has a predictor called "y".
        names = {c: ("y" if c == target else _clean_name(str(c))) for c in frame.columns}
        names = {c: (n + "_x" if n == "y" and c != target else n) for c, n in names.items()}
        frame = frame.rename(columns=names)
        frame["y"] = pd.to_numeric(frame["y"], errors="coerce").astype(float)
        frame = frame[frame["y"].notna()].reset_index(drop=True)
        path = out / f"{did}.parquet"
        frame.to_parquet(path, index=False)
        manifest.append({"data_id": did, "name": bunch.details["name"],
                         "version": bunch.details.get("version"), "target": target,
                         "rows": len(frame), "predictors": frame.shape[1] - 1,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        print(did, bunch.details["name"], frame.shape, flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def _clean_name(name):
    # Formula-safe column names for mboostrs; the mapping is deterministic.
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")
    if not cleaned or cleaned[0].isdigit():
        cleaned = "v_" + cleaned
    return cleaned


def real_manifest():
    return json.loads((DATA / "manifest.json").read_text())


def real_task(data_id, seed):
    frame = pd.read_parquet(DATA / f"{data_id}.parquet")
    key = int(hashlib.sha256(f"ctr23|{data_id}|{seed}".encode()).hexdigest()[:8], 16)
    order = np.random.default_rng(key).permutation(len(frame))
    n_test = min(REAL_TEST_CAP, int(round(REAL_TEST_SHARE * len(frame))))
    test_idx = order[:n_test]
    dev_idx = order[n_test:][:REAL_DEV_CAP]
    dev = frame.iloc[dev_idx].reset_index(drop=True)
    test = frame.iloc[test_idx].reset_index(drop=True)
    return dev, test, None


# --------------------------------------------------------------- task listing
SEEDS = (0, 1, 2)


def all_tasks():
    tasks = [("synth", f"{name}@{n}", seed) for name in SYNTH for n in SYNTH_SIZES
             for seed in SEEDS]
    tasks += [("real", str(m["data_id"]), seed) for m in real_manifest() for seed in SEEDS]
    return tasks


def load(kind, name, seed):
    if kind == "synth":
        base, n = name.split("@")
        return synth_task(base, int(n), seed)
    return real_task(int(name), seed)


if __name__ == "__main__":
    prepare_real()
