"""Penalization drift along a boosting path (issue #94).

A penalized learner selected m times at step nu has accumulated
I - (I - nu S)^m on its basis. Its df rises with m from tr(S) towards the rank
of the basis: a `bbs(x, df=4)` term selected 2,600 times at nu = 0.1 uses
about 10.6 of its 24 df. The fit warns once a learner has used a third of that
gap, and `penalty_drift()` lists every learner.
"""

import warnings

import numpy as np
import pytest

import mboostrs as mb


def _data(n=400, seed=1):
    rng = np.random.default_rng(seed)
    d = {"x0": rng.uniform(-2, 2, n), "x1": rng.uniform(-2, 2, n)}
    d["y"] = np.sin(2 * d["x0"]) + 0.5 * d["x1"] ** 2 + rng.normal(0, 0.3, n)
    return d


def _fit(mstop):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = mb.gamboost("y ~ bbs(x0) + bbs(x1) + bols(x1)", data=_data(),
                          control=mb.boost_control(mstop, 0.1))
    return fit, [str(w.message) for w in caught if "penalty" in str(w.message)]


def test_a_short_path_is_quiet():
    _, messages = _fit(200)
    assert messages == []


def test_a_long_path_warns_and_names_the_terms():
    fit, messages = _fit(10_000)
    assert len(messages) == 1
    assert "bbs(x0)" in messages[0] and "undone much of the penalty" in messages[0]
    rows = fit.penalty_drift(output="records")
    assert [r["learner"] for r in rows] == ["bbs(x0)", "bbs(x1)", "bols(x1)"]
    assert sum(r["selections"] for r in rows) == 10_000
    smooth = [r for r in rows if r["learner"].startswith("bbs")]
    assert all(r["unpenalized_df"] == 24 for r in smooth)
    assert all(r["boosted_df"] > r["single_step_df"] for r in smooth)
    assert max(r["drift_share"] for r in smooth) >= 1 / 3


def test_drift_grows_with_the_path():
    short, _ = _fit(300)
    long, _ = _fit(3_000)
    a = {r["learner"]: r["boosted_df"] for r in short.penalty_drift(output="records")}
    b = {r["learner"]: r["boosted_df"] for r in long.penalty_drift(output="records")}
    assert b["bbs(x0)"] > a["bbs(x0)"]


def test_one_step_df_is_the_calibrated_df():
    # 2 tr(S) - tr(S^2) is the df mboost's df2lambda calibrates; the spectrum
    # behind single_step_df reproduces it through the store's own df_mboost.
    fit, _ = _fit(100)
    row = fit.penalty_drift(output="records")[0]
    assert row["single_step_df"] == pytest.approx(3.41, abs=0.05)
    assert fit._learner_store.extract_df_mboost(0) == pytest.approx(4.0, abs=1e-6)


def test_a_fit_made_by_the_package_itself_is_quiet():
    # Searches, probes and refits fit long paths on purpose; the advice is for
    # a caller choosing mstop. A function whose module is inside mboostrs
    # stands in for them here.
    namespace = {"__name__": "mboostrs._probe_stand_in", "mb": mb, "data": _data()}
    exec(
        "def probe():\n"
        "    return mb.gamboost('y ~ bbs(x0) + bbs(x1)', data=data,\n"
        "                       control=mb.boost_control(10_000, 0.1))\n",
        namespace)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        namespace["probe"]()
    assert not [w for w in caught if "penalty" in str(w.message)]


# ---- issue #420 (b): one warning per process ---------------------------------

def _drift_messages(caught):
    return [str(w.message) for w in caught if "undone much of the penalty" in str(w.message)]


def test_two_separate_fits_warn_once_per_process():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mb.gamboost("y ~ bbs(x0) + bbs(x1) + bols(x1)", data=_data(),
                    control=mb.boost_control(10_000, 0.1))
        mb.gamboost("y ~ bbs(x0) + bbs(x1) + bols(x1)", data=_data(),
                    control=mb.boost_control(10_000, 0.1))
    assert len(_drift_messages(caught)) == 1


def test_collect_warnings_records_every_drift_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with mb.collect_warnings():
            for _ in range(2):
                mb.gamboost("y ~ bbs(x0) + bbs(x1) + bols(x1)", data=_data(),
                            control=mb.boost_control(10_000, 0.1))
    drift = _drift_messages(caught)
    assert len(drift) == 1
    assert "raised 2 times" in drift[0]


def test_reset_shown_summaries_lets_the_drift_warning_show_again():
    from mboostrs._extrapolation import reset_shown_summaries

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mb.gamboost("y ~ bbs(x0) + bbs(x1) + bols(x1)", data=_data(),
                    control=mb.boost_control(10_000, 0.1))
        reset_shown_summaries()
        mb.gamboost("y ~ bbs(x0) + bbs(x1) + bols(x1)", data=_data(),
                    control=mb.boost_control(10_000, 0.1))
    assert len(_drift_messages(caught)) == 2
