"""mboostrs discovering two-way interactions, brute force against a screen.

XGBoost, LightGBM, CatBoost and EBM find interactions without being told which
ones to look for. Holding mboostrs to a purely additive fit and reporting the
gap therefore measures two things at once. This module lets mboostrs search for
interactions too, by two routes, and compares them.

*all pairs* enters every two-way interaction as a base learner and leaves the
choice to boosting's own selection step, which is the mechanism boosting already
uses for main effects.

*screened* ranks the pairs first and enters only the top `SCREEN_K`. The screen
is a split-sample test on the additive model's residuals: bin both predictors
into quantile cells on one half of the training rows, take the cell means of the
residual there, and measure the residual sum of squares they remove on the other
half. A pair that only fits noise removes nothing out of sample and scores at or
below zero, so the ranking is not simply a count of flexibility. It costs no
model fits, only binning, and is the same idea as the FAST heuristic behind
EBM's pair selection {cite}`lou2013accurate`.

The learner for a pair depends on the two margins. A spline basis on a
two-valued column is degenerate, and a tensor product of two such bases is
worse: on credit_g, whose 59 columns are 53 one-hot indicators, expanding every
pair as a tensor product diverges, takes 5.5s and 5.9 GB, and scores 0.7013,
where choosing the learner by margin type is stable, takes 0.6s and 0.9 GB, and
scores 0.7801. So a continuous pair gets a tensor product, a mixed pair a
varying coefficient, and a pair of indicators a product term.

Marginal `df` multiplies: two `df=2` margins give a df=4 term. That was chosen
to match a `bbs(x, df=4)` main effect, so that component-wise boosting would not
prefer the tensor merely for being more flexible. The reasoning was wrong. Four
is the dimension of the tensor penalty's null space, so a df=4 term sits at
maximum shrinkage and can only return a bilinear surface; see
`.remember/enhancements-from-library-benchmarking-2026-08-13.md`. The tuning
grid below also offers `marginal_df=4` (df=16), which is identified, so the
results this file produced remain valid — but `marginal_df=2` contributes
nothing a `bols(a, by=b)` term would not.

Splits, seeds, tuning budget and metrics come from `bench_libraries`.

    python benches/bench_interaction_search.py --out <path>.json
"""

from __future__ import annotations

import argparse
import json
import resource
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import bench_libraries as B
import mboostrs as mb
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

MAIN_DF = 4.0
TENSOR_MARGINAL_DF = 2.0
TENSOR_KNOTS = 5
SCREEN_BINS = 4
SCREEN_K = 20


def levels_of(X):
    return np.array([len(np.unique(X[:, j])) for j in range(X.shape[1])])


def main_terms(levels):
    return [f"bbs(f{j}, df={MAIN_DF})" if levels[j] > 2 else f"bols(f{j})"
            for j in range(len(levels))]


def pair_term(a, b, levels, marginal_df=TENSOR_MARGINAL_DF, knots=TENSOR_KNOTS):
    """Learner for one pair, chosen by the type of each margin."""
    ca, cb = levels[a] > 2, levels[b] > 2
    if ca and cb:
        return (f"bbs(f{a}, df={marginal_df}, knots={knots})"
                f" %X% bbs(f{b}, df={marginal_df}, knots={knots})")
    if ca:
        return f"bbs(f{a}, by=f{b}, df={MAIN_DF})"
    if cb:
        return f"bbs(f{b}, by=f{a}, df={MAIN_DF})"
    return f"bols(f{a}, by=f{b})"


def screen(X, residual, pairs, bins=SCREEN_BINS, seed=0):
    """Rank pairs by out-of-sample residual sum of squares removed.

    Cell means are estimated on half the rows and scored on the other half, so a
    pair that only fits noise scores at or below zero.
    """
    rng = np.random.default_rng(seed)
    half = rng.permutation(len(residual)) < len(residual) // 2
    codes = np.empty(X.shape, dtype=np.int64)
    for j in range(X.shape[1]):
        edges = np.quantile(X[half, j], np.linspace(0, 1, bins + 1)[1:-1])
        codes[:, j] = np.searchsorted(edges, X[:, j])
    ra, rb = residual[half], residual[~half]
    base = float(np.sum((rb - rb.mean()) ** 2))
    scored = []
    for a, b in pairs:
        ka, kb = codes[half, a] * bins + codes[half, b], codes[~half, a] * bins + codes[~half, b]
        total = np.bincount(ka, weights=ra, minlength=bins * bins)
        count = np.bincount(ka, minlength=bins * bins)
        cell = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
        removed = base - float(np.sum((rb - cell[kb]) ** 2))
        scored.append((removed, a, b))
    scored.sort(reverse=True)
    return scored


class SearchGAMBoost(BaseEstimator):
    """gamboost over main effects plus a chosen set of two-way interactions."""

    def __init__(self, task, mode="additive", mstop=300, nu=0.1, screen_k=SCREEN_K,
                 marginal_df=TENSOR_MARGINAL_DF, knots=TENSOR_KNOTS):
        self.task, self.mode, self.mstop, self.nu = task, mode, mstop, nu
        self.screen_k = screen_k
        self.marginal_df, self.knots = marginal_df, knots

    def _frame(self, X):
        return {f"f{j}": np.asarray(X[:, j], dtype=float) for j in range(X.shape[1])}

    def _fit_formula(self, data, rhs, y):
        d = dict(data)
        d["y"] = np.asarray(y, dtype=float)
        family = mb.Gaussian() if self.task == "regression" else mb.Binomial()
        return mb.gamboost(f"y ~ {rhs}", data=d, family=family,
                           control=mb.boost_control(self.mstop, self.nu))

    def fit(self, X, y):
        p = X.shape[1]
        self.n_features_in_ = p
        self.y_lo_, self.y_hi_ = float(np.min(y)), float(np.max(y))
        self.levels_ = levels_of(X)
        data = self._frame(X)
        mains = main_terms(self.levels_)
        every = [(a, b) for a in range(p) for b in range(a + 1, p)]

        self.screen_seconds_ = 0.0
        if self.mode == "additive":
            self.pairs_ = []
        elif self.mode == "all_pairs":
            self.pairs_ = every
        elif self.mode == "screened":
            t0 = time.perf_counter()
            additive = self._fit_formula(data, " + ".join(mains), y)
            raw = np.asarray(additive.predict()).ravel()
            target = (np.asarray(y, float) if self.task == "regression"
                      else np.asarray(y, float) - 1.0 / (1.0 + np.exp(-2.0 * raw)))
            residual = target - raw if self.task == "regression" else target
            ranked = screen(X, residual, every)
            self.ranked_ = ranked
            self.pairs_ = [(a, b) for s, a, b in ranked[:self.screen_k] if s > 0]
            self.screen_seconds_ = time.perf_counter() - t0
        else:
            raise ValueError(self.mode)

        rhs = " + ".join(mains + [pair_term(a, b, self.levels_, self.marginal_df,
                                             self.knots) for a, b in self.pairs_])
        self.n_terms_ = len(mains) + len(self.pairs_)
        self.model_ = self._fit_formula(data, rhs, y)
        risk = np.asarray(self.model_._risk)
        self.diverged_ = bool(np.any(np.diff(risk) > 1e-6 * max(1.0, abs(risk[0]))))
        used = set(self.model_.selected)
        self.pairs_selected_ = sum(
            1 for a, b in self.pairs_
            if any(f"f{a}" in u and f"f{b}" in u for u in used))
        return self

    def _raw(self, X):
        return np.asarray(self.model_.predict(newdata=self._frame(X))).ravel()

    def predict(self, X):
        # Held to the range of the training responses, as in bench_libraries.
        return np.clip(self._raw(X), self.y_lo_, self.y_hi_)

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-2.0 * self._raw(X)))
        return np.column_stack([1.0 - p, p])


def variant(task, mode):
    def build(params):
        return SearchGAMBoost(task, mode=mode, **params)

    def sample(rng):
        params = {"mstop": [400, 1500, 4000, 8000][rng.integers(4)],
                  "nu": [0.05, 0.1, 0.3][rng.integers(3)]}
        if mode != "additive":
            # A tensor's df is the product of its margins'. df=2 margins give
            # df=4, which is the null-space dimension of the penalty and so
            # returns a bilinear surface; df=4 margins give df=16, which is a
            # genuine surface. Both are offered and the inner CV decides.
            params["marginal_df"] = [2.0, 4.0][rng.integers(2)]
            params["knots"] = [5, 10][rng.integers(2)]
        return params

    return build, sample


def run(out_path):
    results = []
    for name, (task, X, y) in B.datasets().items():
        p = X.shape[1]
        for repeat in range(B.REPEATS):
            strat = y if task == "classification" else None
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.25, random_state=B.SEED + repeat, stratify=strat)
            for mode in ("additive", "screened", "all_pairs"):
                build, sample = variant(task, mode)
                # A raw frame is encoded inside each fit, from its training rows.
                build = B.with_encoding(build, X)
                rng = np.random.default_rng(B.SEED + repeat)
                t0 = time.perf_counter()
                try:
                    params, tried, tune_s = B.tune(task, build, sample, Xtr, ytr,
                                                   "configs", rng)
                    if params is None:
                        raise RuntimeError("every configuration failed")
                    pipeline = build(params).fit(Xtr, ytr)
                    final = B.final_estimator(pipeline)
                    held = (r2_score(yte, pipeline.predict(Xte)) if task == "regression"
                            else roc_auc_score(yte, pipeline.predict_proba(Xte)[:, 1]))
                    p = int(final.n_features_in_) if hasattr(final, "n_features_in_") else p
                except BaseException as exc:
                    results.append({
                        "dataset": name, "task": task, "n": int(len(y)), "p": p,
                        "repeat": repeat, "variant": mode, "score": None,
                        "failed": f"{type(exc).__name__}: {str(exc)[:120]}"})
                    print(f"{name:14s} r{repeat} {mode:10s} FAILED "
                          f"{type(exc).__name__}", flush=True)
                    continue
                row = {"dataset": name, "task": task, "n": int(len(y)), "p": p,
                       "repeat": repeat, "variant": mode, "score": float(held),
                       "configs_tried": tried, "tune_seconds": round(tune_s, 2),
                       "screen_seconds": round(final.screen_seconds_, 3),
                       "n_pairs_offered": len(final.pairs_),
                       "n_pairs_possible": p * (p - 1) // 2,
                       "n_pairs_selected": final.pairs_selected_,
                       "n_terms": final.n_terms_, "diverged": final.diverged_,
                       "marginal_df": getattr(final, "marginal_df", None),
                       "knots": getattr(final, "knots", None),
                       "peak_rss_gb": round(
                           resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2),
                       "total_seconds": round(time.perf_counter() - t0, 2)}
                results.append(row)
                print(f"{name:14s} r{repeat} {mode:10s} score {held:.4f}  "
                      f"pairs {row['n_pairs_offered']:5d}/{row['n_pairs_possible']:5d}  "
                      f"{row['total_seconds']:8.1f}s"
                      f"{'  DIVERGED' if final.diverged_ else ''}", flush=True)
                with open(out_path, "w") as fh:
                    json.dump({"results": results}, fh, indent=1)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/book/cases/bench_interaction_search.json")
    args = ap.parse_args()
    rows = run(args.out)
    from _jobs import provenance

    protocol = {"repeats": B.REPEATS, "n_configs": B.N_CONFIGS,
                "inner_folds": B.INNER_FOLDS, "seed": B.SEED,
                "main_df": MAIN_DF,
                "tensor_marginal_df": TENSOR_MARGINAL_DF,
                "tensor_knots": TENSOR_KNOTS,
                "screen_bins": SCREEN_BINS, "screen_k": SCREEN_K}
    with open(args.out, "w") as fh:
        json.dump({"provenance": provenance(
                       protocol=protocol, repeats={"outer_splits": B.REPEATS},
                       uncertainty="per-split scores; summarised by mean and standard error"),
                   "protocol": protocol, "results": rows}, fh, indent=1)
    print(f"\nwrote {args.out}: {len(rows)} rows")
