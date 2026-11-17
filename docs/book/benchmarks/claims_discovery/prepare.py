"""Generate the claim-cost discovery battery.

The covariates are the 20,640 real 1990 California census block groups. The
outcome is simulated from a known structural function (the key), so every
discovery claim, effect estimate and interval can be graded against the truth.

Writes
  <out>/data/dev/{train,test}.parquet         dev replicate given to analysts (test has no y)
  <hidden>/reps/<rep>/{train,test}.parquet     hidden replicates (test with y)
  <hidden>/reps/<rep>/truth.json               effect-query truths for that replicate
  <hidden>/reps/<rep>/test_truth.parquet       row_id, f (true mean), sd, y
  <hidden>/key.json                            the planted structure
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

SRC = Path(__file__).resolve().parents[0] / "california_housing_1990.csv"
MASTER_SEED = 20261009
NU_T = 5  # Student-t degrees of freedom of the noise

FEATURES = [
    "longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms",
    "population", "households", "median_income", "median_house_value",
    "ave_rooms", "ave_bedrooms", "ave_occup", "roof_class",
]
ROOF_EFFECT = {"A": 0.0, "B": 0.0, "C": 0.15, "D": 0.35}

QUERIES = {
    "Q1": {"set": {"median_income": (2.5, 6.0)}},
    "Q2": {"set": {"housing_median_age": (20.0, 45.0)}},
    "Q3": {"set": {"ave_occup": (2.0, 4.0)}},
    "Q4": {"set": {"roof_class": ("A", "D")}},
    "Q5": {"set": {"ave_rooms": (4.0, 7.0)}},
    "Q6": {"set": {"latitude": (37.8, 34.2), "longitude": (-122.3, -117.4)}},
    "Q7": {"set": {"households": (300.0, 1200.0)}},
    "Q8": {"set": {"median_house_value": (150000.0, 350000.0)}},
}

KEY = {
    "mean_active": ["median_income", "housing_median_age", "ave_occup", "roof_class", "latitude", "longitude"],
    "mean_inactive": ["total_rooms", "total_bedrooms", "population", "households",
                      "median_house_value", "ave_rooms", "ave_bedrooms"],
    "interactions": [["latitude", "longitude"], ["housing_median_age", "median_income"]],
    "variance_drivers": ["households"],
    "households_proxies": ["population", "total_rooms", "total_bedrooms"],
    "shapes": {
        "median_income": ["decreasing"],
        "housing_median_age": ["increasing", "threshold_increase"],
        "ave_occup": ["increasing"],
        "roof_class": ["categorical"],
    },
    "roof_order": "D > C > A = B",
}


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def f_mean(d: dict) -> np.ndarray:
    """True conditional mean of y (log claim-cost index) given the true covariates."""
    inc = np.asarray(d["median_income"], float)
    age = np.asarray(d["housing_median_age"], float)
    occ = np.asarray(d["ave_occup"], float)
    lat = np.asarray(d["latitude"], float)
    lon = np.asarray(d["longitude"], float)
    roof = np.asarray(d["roof_class"])
    f_inc = -0.45 * np.log(inc / 3.5)
    old = sig((age - 35.0) / 2.5)
    f_age = 0.30 * old
    f_int = 0.30 * old * sig((3.0 - inc) / 0.6)
    f_occ = 0.35 * np.log(np.clip(occ, 1.0, 6.0) / 3.0)
    bump_s = 0.45 * np.exp(-((lat - 34.2) ** 2 / (2 * 0.35**2) + (lon + 117.4) ** 2 / (2 * 0.45**2)))
    bump_n = 0.35 * np.exp(-((lat - 39.0) ** 2 / (2 * 0.6**2) + (lon + 121.0) ** 2 / (2 * 0.5**2)))
    f_roof = np.vectorize(ROOF_EFFECT.get)(roof).astype(float)
    return f_inc + f_age + f_int + f_occ + bump_s + bump_n + f_roof


def sd_noise(households: np.ndarray) -> np.ndarray:
    h = np.maximum(np.asarray(households, float), 20.0)
    return 0.12 + 0.30 * np.sqrt(400.0 / h)


def base_covariates() -> pl.DataFrame:
    raw = pl.read_csv(SRC).with_columns(pl.col("median_house_value").cast(pl.Float64))
    rng = np.random.default_rng(MASTER_SEED)
    n = raw.height
    age = raw["housing_median_age"].to_numpy().astype(float)
    lat = raw["latitude"].to_numpy()
    # Simulated roof class: older stock and the north carry more class C/D roofs.
    score = 0.06 * (age - 28.0) + 0.5 * (lat > 37.5) + rng.normal(0, 1.0, n)
    roof = np.select([score < -0.8, score < 0.4, score < 1.4], ["A", "B", "C"], "D")
    d = raw.with_columns(
        pl.Series("roof_class", roof),
        (pl.col("total_rooms") / pl.col("households")).alias("ave_rooms"),
        (pl.col("total_bedrooms") / pl.col("households")).alias("ave_bedrooms"),
        (pl.col("population") / pl.col("households")).alias("ave_occup"),
        pl.col("housing_median_age").cast(pl.Float64),
        pl.col("total_rooms").cast(pl.Float64),
        pl.col("total_bedrooms").cast(pl.Float64),
        pl.col("population").cast(pl.Float64),
        pl.col("households").cast(pl.Float64),
    )
    # Split: test rows drawn with probability rising in income (mild covariate shift).
    z = (d["median_income"].to_numpy() - d["median_income"].mean()) / d["median_income"].std()
    p = np.exp(0.35 * z)
    p = p / p.sum()
    test_idx = rng.choice(n, size=n // 4, replace=False, p=p)
    is_test = np.zeros(n, bool)
    is_test[test_idx] = True
    return d.with_columns(pl.Series("row_id", np.arange(n)), pl.Series("is_test", is_test)).select(
        ["row_id", "is_test"] + FEATURES
    )


def apply_missing(d: pl.DataFrame, rng) -> pl.DataFrame:
    n = d.height
    lat = d["latitude"].to_numpy()
    p_age = 0.04 + 0.12 * (lat > 37.5)  # missing at random given latitude
    miss_age = rng.uniform(size=n) < p_age
    miss_bed = rng.uniform(size=n) < 0.01
    return d.with_columns(
        pl.when(pl.Series(miss_age)).then(None).otherwise(pl.col("housing_median_age")).alias("housing_median_age"),
        pl.when(pl.Series(miss_bed)).then(None).otherwise(pl.col("total_bedrooms")).alias("total_bedrooms"),
        pl.when(pl.Series(miss_bed)).then(None).otherwise(pl.col("ave_bedrooms")).alias("ave_bedrooms"),
    )


def query_truth(train_true: pl.DataFrame, null: bool) -> dict:
    out = {}
    base = {c: train_true[c].to_numpy() for c in FEATURES}
    for q, spec in QUERIES.items():
        if null:
            out[q] = 0.0
            continue
        lo, hi = dict(base), dict(base)
        for var, (a, b) in spec["set"].items():
            lo[var] = np.full(train_true.height, a, dtype=object if var == "roof_class" else float)
            hi[var] = np.full(train_true.height, b, dtype=object if var == "roof_class" else float)
        out[q] = float(np.mean(f_mean(hi) - f_mean(lo)))
    return out


def draw(cov: pl.DataFrame, seed: int, kind: str, n_small: int = 800):
    """One replicate. kind: 'full' (planted DGP), 'null' (no structure), 'small' (full DGP, n_train=n_small)."""
    rng = np.random.default_rng(seed)
    cols = {c: cov[c].to_numpy() for c in FEATURES}
    n = cov.height
    if kind == "null":
        f = np.full(n, 0.3)
        sd = np.full(n, 0.45)
    else:
        f = f_mean(cols)
        sd = sd_noise(cols["households"])
    t = rng.standard_t(NU_T, size=n) / np.sqrt(NU_T / (NU_T - 2))
    y = f + sd * t
    full = cov.with_columns(pl.Series("y", y), pl.Series("f", f), pl.Series("sd", sd))
    train_true = full.filter(~pl.col("is_test"))
    if kind == "small":
        keep = rng.choice(train_true.height, size=n_small, replace=False)
        train_true = train_true[np.sort(keep)]
    test_true = full.filter(pl.col("is_test"))
    train = apply_missing(train_true, rng).select(["row_id"] + FEATURES + ["y"])
    test_obs = apply_missing(test_true, rng)
    test = test_obs.select(["row_id"] + FEATURES)
    truth = query_truth(train_true, null=(kind == "null"))
    test_truth = test_true.select("row_id", "f", "sd", "y").with_columns(
        pl.Series("age_missing", test_obs["housing_median_age"].is_null().to_numpy())
    )
    return train, test, truth, test_truth


REPS = {
    "dev": ("full", MASTER_SEED + 0),
    **{f"H{i}": ("full", MASTER_SEED + 100 + i) for i in range(1, 9)},
    "N1": ("null", MASTER_SEED + 201),
    "N2": ("null", MASTER_SEED + 202),
    "S1": ("small", MASTER_SEED + 301),
    "S2": ("small", MASTER_SEED + 302),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", required=True, help="directory receiving data/dev")
    ap.add_argument("--hidden", required=True, help="directory receiving reps/ and key.json")
    a = ap.parse_args()
    cov = base_covariates()
    pub, hid = Path(a.public), Path(a.hidden)
    for rep, (kind, seed) in REPS.items():
        train, test, truth, test_truth = draw(cov, seed, kind)
        rd = hid / "reps" / rep
        rd.mkdir(parents=True, exist_ok=True)
        train.write_parquet(rd / "train.parquet")
        test.write_parquet(rd / "test.parquet")
        test_truth.write_parquet(rd / "test_truth.parquet")
        (rd / "truth.json").write_text(json.dumps({"kind": kind, "seed": seed, "queries": truth}, indent=2))
        if rep == "dev":
            dd = pub / "data" / "dev"
            dd.mkdir(parents=True, exist_ok=True)
            train.write_parquet(dd / "train.parquet")
            test.write_parquet(dd / "test.parquet")
        print(rep, kind, train.shape, test.shape, {k: round(v, 3) for k, v in truth.items()})
    (hid / "key.json").write_text(json.dumps({**KEY, "queries": {k: {v: list(r) for v, r in s["set"].items()} for k, s in QUERIES.items()}}, indent=2))


if __name__ == "__main__":
    main()
