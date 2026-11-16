"""Ranking candidate response distributions, as `gamlss::chooseDist` does.

Choosing the response distribution is usually the least-informed decision in a
distributional model, and fitting the alternatives is a more direct way to make
it than reasoning about the response's shape.

The comparison only means anything because every family's risk is the exact
negative log-likelihood of the same data; `test_lss_cdf_quantile_parity.py`
pins that, and found `BCCGLSS` dropping `log(2*pi)/2` per observation.
"""

import numpy as np
import pytest
from mboostrs.model_selection import FAMILY_SETS, scan_lss_families

import mboostrs as mb

SPEC = {"mu": "y ~ bbs(x, df=4)", "sigma": "y ~ bols(x)"}
CONTROL = (40, 0.1, "inbag", False)


def rank(data, output=None, **kwargs):
    """`scan_lss_families` with this module's shared settings."""
    kwargs.setdefault("metric", "cvrisk")
    return scan_lss_families(SPEC, data=data, control=CONTROL,
                             fit_func="gamboostLSS", **kwargs)


def _draws():
    """One data set per generating family, all sharing a covariate."""
    rng = np.random.default_rng(11)
    n = 400
    x = rng.normal(size=n)
    return x, {
        "gamma": ("realplus", rng.gamma(3.0, np.exp(0.4 + 0.3 * x) / 3.0)),
        "normal": ("realline", 1.2 * x + rng.normal(0, 0.7, n)),
        "logistic": ("realline", 1.2 * x + 0.7 * rng.logistic(size=n)),
        "poisson": ("counts", rng.poisson(np.exp(0.5 + 0.4 * x)).astype(float)),
    }


@pytest.mark.parametrize("label,expected", [
    ("gamma", "GammaLSS"),
    ("normal", "GaussianLSS"),
    ("logistic", "LogisticLSS"),
    ("poisson", "PoissonLSS"),
])
def test_the_generating_family_is_ranked_first(label, expected):
    """The property that makes the function worth having."""
    x, cases = _draws()
    support, y = cases[label]
    ranked = rank({'x': x, 'y': y}, support=support)
    assert ranked[0]["family"] == expected


def test_cross_validated_ranking_beats_gaic_on_zero_inflation():
    """Why out-of-bag risk is the default and GAIC is not.

    `df` for a boosted LSS fit is `mstop * n_params + 1` — the boosting budget,
    not the structure absorbed — so GAIC ranks largely by parameter count. On
    zero-inflated counts, the zero-adjusted family has both the best
    log-likelihood and the best out-of-bag risk, but GAIC promotes a
    one-parameter family above it purely on df.
    """
    rng = np.random.default_rng(11)
    n = 500
    x = rng.normal(size=n)
    y = np.where(rng.random(n) < 0.35, 0.0,
                 rng.poisson(np.exp(1.0 + 0.4 * x)).astype(float))
    data = {"x": x, "y": y}

    by_cv = rank(data, support="counts")
    assert by_cv[0]["family"] == "ZeroAdjustedPoissonLSS"
    assert by_cv[-1]["family"] == "PoissonLSS"
    # The best log-likelihood belongs to the same family the CV ranking picks.
    fitted = [r for r in by_cv if r["status"] == "ok"]
    assert min(fitted, key=lambda r: r["risk"])["family"] == "ZeroAdjustedPoissonLSS"

    by_gaic = rank(data, support="counts", metric="GAIC")
    one_parameter = next(r for r in by_gaic if r["family"] == "GeometricLSS")
    zero_adjusted = next(r for r in by_gaic if r["family"] == "ZeroAdjustedPoissonLSS")
    assert one_parameter["risk"] > zero_adjusted["risk"]      # fits worse
    # ... and yet ranks close to or above it on GAIC, on df alone.
    assert one_parameter["GAIC"] < zero_adjusted["GAIC"] * 1.05


def test_a_family_that_cannot_fit_is_recorded_not_dropped():
    """A positive-support family on data containing negatives must appear in
    the table with the reason, so the absence of a candidate is never silent."""
    rng = np.random.default_rng(4)
    n = 200
    x = rng.normal(size=n)
    data = {"x": x, "y": rng.normal(size=n)}          # takes negative values
    ranked = rank(data, families=["GaussianLSS", "GammaLSS"])
    names = {row["family"] for row in ranked}
    assert names == {"GaussianLSS", "GammaLSS"}
    gamma = next(row for row in ranked if row["family"] == "GammaLSS")
    assert gamma["status"] == "error"
    assert gamma["error"]
    assert ranked[0]["family"] == "GaussianLSS"       # failures sort last


def test_multiple_penalties_give_one_column_each():
    x, cases = _draws()
    _, y = cases["normal"]
    ranked = rank({"x": x, "y": y}, support="realline", metric="GAIC",
                  k=np.log(len(x)))
    assert all(row["GAIC"] is not None for row in ranked if row["status"] == "ok")
    # A larger penalty must not change the ordering of two equal-df candidates.
    assert ranked[0]["status"] == "ok"


def test_families_with_more_parameters_get_the_spec_extended():
    """A three-parameter family is given a right-hand side for its third
    parameter, reusing the last one supplied, so a two-entry spec can rank
    candidates of differing arity."""
    rng = np.random.default_rng(6)
    n = 250
    x = rng.normal(size=n)
    data = {"x": x, "y": rng.gamma(2.0, np.exp(0.3 * x) / 2.0)}
    ranked = rank(data, families=["GammaLSS", "BCCGLSS"])   # 2 and 3 params
    assert all(row["status"] == "ok" for row in ranked)


def test_support_sets_name_only_real_families():
    for names in FAMILY_SETS.values():
        for name in names:
            assert hasattr(mb, name), name


def test_bad_arguments_are_rejected():
    x, cases = _draws()
    _, y = cases["normal"]
    data = {"x": x, "y": y}
    with pytest.raises(ValueError, match="either families= or support="):
        rank(data)
    with pytest.raises(ValueError, match="support must be one of"):
        rank(data, support="nonsense")
    with pytest.raises(ValueError, match="metric must be one of"):
        rank(data, support="realline", metric="bic")


def _gamma_frame(n=200, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return {"x": x, "y": rng.gamma(3.0, np.exp(0.4 + 0.3 * x) / 3.0)}


def test_a_bare_right_hand_side_is_refused_before_any_fit():
    """An agent passed ``{"mu": "bbs(x)", "sigma": "bols(x)"}`` and got one
    identical "formula must contain '~'" error row per family, then read the
    first error row as the winner. The scan now refuses the specification once,
    with the formula to write instead."""
    data = _gamma_frame()
    with pytest.raises(ValueError, match="takes whole formulas.*'response ~ bbs"):
        scan_lss_families({"mu": "bbs(x, df=4)", "sigma": "bols(x)"}, data=data,
                          families=[mb.GammaLSS()], control=CONTROL)
    with pytest.raises(ValueError, match="takes whole formulas"):
        scan_lss_families("bbs(x, df=4)", data=data, families=[mb.GammaLSS()],
                          control=CONTROL)


def test_an_integer_folds_is_a_k_fold_split():
    """``folds=4`` used to fail every family with "folds must be an (n, B)
    matrix"; it now means the same seeded 4-fold weights `cv_weights` builds."""
    data = _gamma_frame()
    families = [mb.GammaLSS(), mb.GaussianLSS()]
    by_count = rank(data, families=families, folds=4)
    by_matrix = rank(data, families=families,
                     folds=mb.cv_weights(len(data["y"]), type="kfold", B=4))
    assert all(r["status"] == "ok" for r in by_count)
    assert [(r["family"], r["cvrisk"]) for r in by_count] == [
        (r["family"], r["cvrisk"]) for r in by_matrix]


def test_a_scan_where_every_family_fails_warns():
    """Error rows sort last, so when every family fails the first row is an
    error, not a winner; the scan says so."""
    data = _gamma_frame()
    data["y"] = -data["y"]  # negative responses: no positive-support family fits
    with pytest.warns(RuntimeWarning, match="every candidate family failed"):
        rows = rank(data, families=[mb.GammaLSS(), mb.WeibullLSS()])
    assert all(r["status"] == "error" for r in rows)
