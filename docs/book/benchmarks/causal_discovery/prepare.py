"""Generate the job-training causal battery.

The applicant covariates are the 28,155 real men in the March 1988 Current
Population Survey (``AER::CPS1988``): weekly wage, years of education, years of
potential experience, ethnicity, metropolitan residence, region and part-time
status. The program data are simulated from a known structural model (the
key): enrolment, follow-up survey response, two post-enrolment variables and
follow-up earnings. Every causal claim, effect estimate and interval can
therefore be graded against the truth.

Writes
  <out>/data/dev/{train,test}.parquet        dev replicate given to analysts
  <hidden>/reps/<rep>/{train,test}.parquet   hidden replicates (same columns)
  <hidden>/reps/<rep>/truth.json             effect-query truths for that replicate
  <hidden>/reps/<rep>/test_truth.parquet     row_id, cate (true expected effect, dollars)
  <hidden>/key.json                          the planted structure
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

MASTER_SEED = 20261010
SRC_SHA256 = "ef5ee9cbf3674f9cf242549517cf7801bffc5e98c1006055c9fdefebaf386251"
N_TRAIN, N_TEST, N_SMALL = 12000, 4000, 1500
N_MC = {"train": 400, "test": 4000}  # Monte Carlo draws per row for the potential-outcome means

KEY = {
    "treatment": "enrolled",
    "outcome": "earnings_post",
    "roles": {
        "prior_earnings": "confounder",
        "education": "confounder",
        "months_unemployed": "confounder",
        "experience": "confounder",
        "age": "confounder",
        "region": "confounder",
        "smsa": "confounder",
        "ethnicity": "treatment_only",
        "distance_km": "instrument",
        "parttime": "outcome_only",
        "certificate": "mediator",
        "hours_post": "mediator",
        "responded": "post_treatment_selection",
        "caseworker_note_len": "irrelevant",
        "enrolled": "treatment",
        "earnings_post": "outcome",
    },
    # Effect modifiers on the dollar scale, judged by the range of the true
    # CATE across quintiles (or levels) of each column, relative to the ATE:
    # at least 0.35 is material; below 0.10 is none; columns in between are
    # not counted either way. Measured on a 12,000-row draw.
    "effect_modifiers": {
        "prior_earnings": "larger_when_higher",
        "education": "larger_when_higher",
        "months_unemployed": "larger_when_lower",
        "parttime": "larger_when_no",
    },
    "effect_modifiers_neutral": ["age", "experience", "smsa", "distance_km", "ethnicity", "region"],
    "effect_modifiers_none": ["caseworker_note_len"],
    "never_adjust": ["certificate", "hours_post", "responded"],
    "overlap_thin": "prior_earnings above 40,000 dollars (enrolment probability below 0.03)",
    "employment_effect": "concentrated where prior_earnings is below about 15,000 dollars",
}


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def load_population(src: Path) -> pl.DataFrame:
    raw = src.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SRC_SHA256:
        raise SystemExit(f"{src}: sha256 does not match AER::CPS1988 ({SRC_SHA256})")
    d = pl.read_csv(src).drop("rownames")
    return d.with_columns(
        (pl.col("experience") + pl.col("education") + 6).alias("age"),
    )


def covariates(pop: pl.DataFrame, idx: np.ndarray, rng) -> dict:
    """Pre-enrolment covariates for the sampled applicants."""
    p = pop[idx]
    n = len(idx)
    wage = p["wage"].to_numpy()
    prior = np.round(wage * 52 * np.exp(rng.normal(0, 0.10, n)), -1)
    educ = p["education"].to_numpy().astype(float)
    exper = p["experience"].to_numpy().astype(float)
    smsa = p["smsa"].to_numpy()
    lam = np.exp(0.9 - 0.55 * (np.log(prior) - np.log(27000)) - 0.06 * (educ - 12))
    months = rng.poisson(np.clip(lam, 0.05, 30))
    dist = np.round(np.exp(np.where(smsa == "yes", 1.7, 3.0) + rng.normal(0, 0.6, n)), 1)
    note = rng.integers(20, 400, n)
    return {
        "prior_earnings": prior,
        "education": educ,
        "experience": exper,
        "age": p["age"].to_numpy().astype(float),
        "ethnicity": p["ethnicity"].to_numpy(),
        "smsa": smsa,
        "region": p["region"].to_numpy(),
        "parttime": p["parttime"].to_numpy(),
        "months_unemployed": months.astype(float),
        "distance_km": dist,
        "caseworker_note_len": note.astype(float),
    }


def propensity(c: dict) -> np.ndarray:
    z = np.log(c["prior_earnings"]) - np.log(27000)
    eta = (-1.55 - 1.0 * z + 0.35 * (c["education"] < 12) + 0.30 * np.sqrt(c["months_unemployed"])
           - 0.55 * (np.log(c["distance_km"]) - 2.3) + 0.30 * (c["ethnicity"] == "afam")
           + 0.20 * (c["region"] == "south") - 0.012 * (c["experience"] - 18))
    eta = eta - 3.0 * (c["prior_earnings"] > 40000)
    return sig(eta)


def tau_parts(c: dict, null: bool):
    """Effect of enrolment on the employment logit and on log earnings of the employed."""
    if null:
        z = np.zeros(len(c["age"]))
        return z, z, z, z
    tau_emp = 0.15 + 0.85 * sig((15000 - c["prior_earnings"]) / 2500)
    tau_log = 0.05 + 0.10 * (c["education"] < 12) + 0.06 * np.clip(30 - c["age"], 0, 15) / 10
    hours_gain = np.full(len(c["age"]), 80.0)
    cert_effect = np.full(len(c["age"]), 0.05)
    return tau_emp, tau_log, hours_gain, cert_effect


def outcome_draws(c: dict, t: np.ndarray, rng, null: bool, draws: int = 1):
    """Draw (certificate, employed, hours, earnings) under treatment vector t.

    With draws > 1 the arrays have shape (draws, n) and t is broadcast.
    """
    n = len(c["age"])
    tau_emp, tau_log, hours_gain, cert_effect = tau_parts(c, null)
    shape = (draws, n)
    z = np.log(c["prior_earnings"]) - np.log(27000)
    pt = (c["parttime"] == "yes").astype(float)
    cert = (rng.random(shape) < t * sig(-0.2 + 0.25 * (c["education"] - 12))).astype(float)
    p_emp = sig(1.3 + 0.7 * z + 0.10 * (c["education"] - 12) - 0.30 * np.sqrt(c["months_unemployed"])
                - 0.004 * (c["experience"] - 20) ** 2 / 4 + t * tau_emp)
    emp = (rng.random(shape) < p_emp).astype(float)
    hours = np.clip(1850 - 750 * pt + t * hours_gain + rng.normal(0, 220, shape), 200, 3000)
    mu = (1.05 + 0.88 * np.log(c["prior_earnings"] + 2000) + 0.025 * (c["education"] - 12)
          + 0.07 * (c["smsa"] == "yes") - 0.04 * (c["region"] == "south")
          + 0.55 * np.log(hours / 1850) + t * tau_log + cert * cert_effect)
    sd = 0.32 + 0.12 * pt
    earn = emp * np.exp(mu + sd * rng.normal(0, 1, shape))
    return cert, emp, hours, earn


def true_effects(c: dict, null: bool, seed: int, draws: int) -> dict:
    """Per-row expected potential outcomes by Monte Carlo with common random numbers."""
    out = {}
    for arm in (0, 1):
        rng = np.random.default_rng(seed)
        t = np.full(len(c["age"]), float(arm))
        _, emp, _, earn = outcome_draws(c, t, rng, null, draws=draws)
        out[arm] = (earn.mean(axis=0), emp.mean(axis=0))
    return {"y1": out[1][0], "y0": out[0][0], "e1": out[1][1], "e0": out[0][1]}


def make_rep(pop, idx_train, idx_test, rng, null: bool, mc_seed: int):
    frames, truths = {}, {}
    for part, idx in (("train", idx_train), ("test", idx_test)):
        c = covariates(pop, idx, rng)
        t = (rng.random(len(idx)) < propensity(c)).astype(float)
        cert, emp, hours, earn = (a[0] for a in outcome_draws(c, t, rng, null))
        resp = rng.random(len(idx)) < sig(1.6 + 0.5 * t - 0.3 * (c["education"] < 12)
                                          + 0.4 * (np.log(c["prior_earnings"]) - np.log(27000)))
        te = true_effects(c, null, mc_seed + (part == "test"), N_MC[part])
        hours_obs = np.where(resp, np.where(emp > 0, np.round(hours), 0.0), np.nan)
        earn_obs = np.where(resp, np.round(earn, -1), np.nan)
        frame = pl.DataFrame({
            "row_id": np.arange(len(idx)) + (0 if part == "train" else 100000),
            "age": c["age"], "education": c["education"], "experience": c["experience"],
            "ethnicity": c["ethnicity"], "smsa": c["smsa"], "region": c["region"],
            "parttime": c["parttime"], "prior_earnings": c["prior_earnings"],
            "months_unemployed": c["months_unemployed"], "distance_km": c["distance_km"],
            "caseworker_note_len": c["caseworker_note_len"],
        })
        if part == "train":
            # Post-enrolment columns exist only for the enrolled cohort; the test
            # rows are new applicants, known only by their pre-enrolment covariates.
            frame = frame.with_columns(
                pl.Series("enrolled", t.astype(int)),
                pl.Series("certificate", cert.astype(int)),
                pl.Series("responded", resp.astype(int)),
                pl.Series("hours_post", hours_obs).fill_nan(None),
                pl.Series("earnings_post", earn_obs).fill_nan(None),
            )
        frames[part] = frame
        truths[part] = (c, t, te)
    return frames, truths


def query_truths(c, t, te) -> dict:
    cate = te["y1"] - te["y0"]
    q = np.quantile(c["prior_earnings"], [0.25, 0.75])
    return {
        "ATE": float(cate.mean()),
        "ATT": float(cate[t == 1].mean()),
        "CATE_educ_lt12": float(cate[c["education"] < 12].mean()),
        "CATE_educ_ge16": float(cate[c["education"] >= 16].mean()),
        "CATE_prior_q1": float(cate[c["prior_earnings"] <= q[0]].mean()),
        "CATE_prior_q4": float(cate[c["prior_earnings"] >= q[1]].mean()),
        "CATE_age_lt25": float(cate[c["age"] < 25].mean()),
        "EMP_ATE": float((te["e1"] - te["e0"]).mean()),
        "prior_quartiles": [float(q[0]), float(q[1])],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="path to AER CPS1988.csv")
    ap.add_argument("--out", required=True, help="benchmark directory (dev data goes to <out>/data/dev)")
    ap.add_argument("--hidden", required=True, help="directory for hidden replicates and the key")
    a = ap.parse_args()
    pop = load_population(Path(a.src))
    hidden = Path(a.hidden)
    (hidden / "reps").mkdir(parents=True, exist_ok=True)
    (hidden / "key.json").write_text(json.dumps(KEY, indent=2))
    reps = ["dev"] + [f"H{i}" for i in range(1, 7)] + ["N1", "N2", "S1", "S2"]
    master = np.random.default_rng(MASTER_SEED)
    # The test cohort is a later intake that over-represents low earners.
    w_test = 1.0 / np.sqrt(pop["wage"].to_numpy())
    w_test = w_test / w_test.sum()
    for k, rep in enumerate(reps):
        rng = np.random.default_rng(master.integers(2**63))
        n_train = N_SMALL if rep.startswith("S") else N_TRAIN
        idx_train = rng.choice(pop.height, n_train, replace=False)
        rest = np.setdiff1d(np.arange(pop.height), idx_train)
        p_rest = w_test[rest] / w_test[rest].sum()
        idx_test = rng.choice(rest, N_TEST, replace=False, p=p_rest)
        null = rep.startswith("N")
        frames, truths = make_rep(pop, idx_train, idx_test, rng, null, mc_seed=1000 + k)
        rd = hidden / "reps" / rep
        rd.mkdir(parents=True, exist_ok=True)
        frames["train"].write_parquet(rd / "train.parquet")
        frames["test"].write_parquet(rd / "test.parquet")
        c, t, te = truths["train"]
        tr = query_truths(c, t, te)
        tr["n_train"] = n_train
        tr["null"] = null
        tr["share_enrolled"] = float(t.mean())
        tr["share_responded"] = float(frames["train"]["responded"].mean())
        (rd / "truth.json").write_text(json.dumps(tr, indent=2))
        ct, tt, tet = truths["test"]
        pl.DataFrame({"row_id": frames["test"]["row_id"], "cate": tet["y1"] - tet["y0"],
                      "emp_effect": tet["e1"] - tet["e0"]}).write_parquet(rd / "test_truth.parquet")
        if rep == "dev":
            dd = Path(a.out) / "data" / "dev"
            dd.mkdir(parents=True, exist_ok=True)
            frames["train"].write_parquet(dd / "train.parquet")
            frames["test"].write_parquet(dd / "test.parquet")
        print(rep, {k2: round(v, 1) if isinstance(v, float) else v for k2, v in tr.items()})


if __name__ == "__main__":
    main()
