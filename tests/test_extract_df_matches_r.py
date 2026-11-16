"""`extract("df")` reports the quantity `mboost` reports.

P1.2 asks for each learner's df-to-lambda mapping to be checked. Doing that
turned up a number that was simply wrong, and a second that was a different
definition wearing the same name.

`extract(what="df")` used to return the *spec's* `df`. That is right when the
caller asked for a df and wrong when they asked for a lambda, because `df`
keeps its default of 4: `bbs(x, lambda_value=5)` reported 4.0 for a smoother
whose df is 9.26.

The fallback underneath it returned `tr(H)`, which is a real trace of the
smoother and not R's df. R counts `tr(2H - HᵀH)`, and its `df2lambda` inverts
that one — which is why a learner asked for `df = 4` has `tr(H) = 3.376`. Both
numbers are honest; only one answers "what did `df = 4` mean".

The reference values come from `mboost` 2.9-11 run on the same 140 rows, which
matters: the trace depends on the design, so a comparison against R on
differently seeded data says nothing.
"""

import warnings

import numpy as np
import pytest

import mboostrs as mb

#: 140 rows, `x ~ N(0, 1)` from NumPy's PCG64 at seed 0 and
#: `y = sin(2x) + N(0, 0.3)`, written to CSV and read by R so both sides fit
#: the same design.
N = 140


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=N)
    return {"x": x, "y": np.sin(2 * x) + rng.normal(scale=0.3, size=N)}


def fit(term, data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mb.gamboost(f"y ~ {term}", data,
                           control=mb.boost_control(20, 0.1))


@pytest.mark.parametrize("term, expected", [
    # `mboost` on the same rows: extract(f, "df")
    ("bbs(x, df=4)", 4.0),
    ("bbs(x, lambda_value=5.0)", 9.264741),
    ("bols(x)", 2.0),
])
def test_it_matches_mboost(term, expected, data):
    value = next(iter(fit(term, data).extract("df").values()))
    assert value == pytest.approx(expected, abs=1e-5)


def test_a_lambda_learner_no_longer_reports_the_default_df(data):
    """The defect. `df` defaults to 4 on the spec, so a learner given an
    explicit lambda reported 4.0 whatever lambda it was given."""
    values = [next(iter(fit(f"bbs(x, lambda_value={lam})", data)
                         .extract("df").values()))
              for lam in (0.1, 5.0, 500.0)]
    assert len(set(np.round(values, 6))) == 3, values
    assert values[0] > values[1] > values[2], "more penalty, fewer df"


def test_the_two_df_definitions_are_both_available_and_differ(data):
    """`tr(H)` is still there and still a different number. Keeping both, with
    the R-matching one on the public `extract`, is the point."""
    model = fit("bbs(x, df=4)", data)
    trace = model._learner_store.extract_df(0)
    mboost_df = model._learner_store.extract_df_mboost(0)
    assert trace == pytest.approx(3.375946, abs=1e-5)
    assert mboost_df == pytest.approx(4.0, abs=1e-6)


def test_a_tree_reports_no_df_the_way_r_does(data):
    """R: "model matrix, penalty matrix, lambda and index do not exist for
    tree base-learners", and `extract(f, "df")` returns NULL."""
    assert next(iter(fit("btree(x)", data).extract("df").values())) is None


def test_df_falls_as_lambda_rises(data):
    """The df-to-lambda mapping P1.2 asks to be checkable, as a monotone
    relation rather than one number."""
    previous = None
    for lam in (0.05, 1.0, 20.0, 400.0, 10000.0):
        value = next(iter(fit(f"bbs(x, lambda_value={lam})", data)
                          .extract("df").values()))
        assert value is not None
        if previous is not None:
            assert value < previous, f"lambda={lam} did not reduce df"
        previous = value


def test_asking_for_a_df_gets_that_df_back(data):
    """The other direction: whatever lambda solves for `df = k`, the reported
    df is `k`. That is the mapping being self-consistent."""
    for requested in (3.0, 4.0, 8.0, 12.0):
        value = next(iter(fit(f"bbs(x, df={requested})", data)
                          .extract("df").values()))
        assert value == pytest.approx(requested, abs=1e-6)
