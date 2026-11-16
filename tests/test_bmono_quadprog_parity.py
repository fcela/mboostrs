"""R parity for the `bmono(type="quad.prog")` active-set solver.

Fixtures come from ``tests/fixtures/generate_bmono_quadprog_fixtures.R``
(mboost 2.9.11) and are committed, so this runs without R.

Both implementations run the same primal active-set QP over the same
``H = X'X + lambda*K``, through different Cholesky code. Agreement is therefore
a numerical claim, and the tolerances below are the measured accumulation rather
than round numbers:

===========================  ==========================  =====================
case                         max |fitted difference|     relative to the span
===========================  ==========================  =====================
increasing, df=8, mstop=60    1.5e-11                     4e-12
increasing, df=12, mstop=60   9.4e-14                     3e-14
convex, df=4, mstop=60        1.2e-10                     3e-11
convex, df=4, mstop=300       6.7e-10                     1.8e-10
===========================  ==========================  =====================

The growth from 1.2e-10 to 6.7e-10 as the path lengthens fivefold is the
signature of accumulation in identical arithmetic, not of a divergence: the
selection path and the active set agree throughout.

The binding assertions matter as much as the agreement ones. A monotone
constraint fitted to data that already respects it never activates a
constraint, and the QP reduces to an unconstrained solve — a version of this
test built on such data would pass no matter what the solver did.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import mboostrs as mb

FIXTURE = Path(__file__).parent / "fixtures" / "bmono_quadprog.json"

#: Per case: absolute tolerance on the fitted values at every iteration of the
#: path, and the share of the fitted curve's steps the constraint must pin for
#: the case to be exercising the active set at all.
TOLERANCES = {
    ("increasing", 8, 60): (1e-9, 0.15),
    ("increasing", 12, 60): (1e-9, 0.30),
    ("convex", 4, 60): (1e-9, None),
    ("convex", 4, 300): (5e-9, None),
}


@pytest.fixture(scope="module")
def reference():
    if not FIXTURE.exists():
        pytest.skip(f"fixture {FIXTURE.name} not generated")
    return json.loads(FIXTURE.read_text())


def frame_of(reference):
    return pl.DataFrame({
        "x": np.asarray(reference["x"], dtype=float),
        "y": np.asarray(reference["y"], dtype=float),
    })


def fit_case(reference, case):
    return mb.gamboost(
        f"y ~ bmono(x, constraint='{case['constraint']}', df={case['df']})",
        data=frame_of(reference), family=mb.Gaussian(),
        control=mb.boost_control(mstop=case["mstop"], nu=reference["nu"]),
    )


def pinned_share(model):
    """Share of the fitted curve's adjacent steps the constraint holds at zero.

    A monotone constraint that binds shows up as flat segments: the
    least-squares fit wants to fall, the constraint permits no drop, and
    successive grid values come back exactly equal.
    """
    effects = model.partial_effects(which=0, n_points=60)
    curve = np.asarray(next(iter(effects.values()))[1], dtype=float)
    steps = np.diff(curve)
    if steps.size == 0:
        return 0.0
    span = max(float(curve.max() - curve.min()), 1e-12)
    return float(np.sum(np.abs(steps) < 1e-9 * span)) / steps.size


def case_ids(reference):
    return [(c["constraint"], c["df"], c["mstop"]) for c in reference["cases"]]


def test_every_fixture_case_is_covered_by_a_tolerance(reference):
    assert set(case_ids(reference)) == set(TOLERANCES)


@pytest.mark.parametrize("index", range(4))
def test_quadprog_path_matches_r(reference, index):
    case = reference["cases"][index]
    key = (case["constraint"], case["df"], case["mstop"])
    atol, min_pinned = TOLERANCES[key]
    model = fit_case(reference, case)

    expected = np.asarray(case["fitted_path"], dtype=float)
    ours = np.asarray(
        [np.asarray(model[:k].fitted(), dtype=float)
         for k in range(1, case["mstop"] + 1)]
    )
    assert ours.shape == expected.shape
    gaps = np.abs(ours - expected).max(axis=1)
    worst = int(np.argmax(gaps))
    assert gaps[worst] <= atol, (
        f"{key}: fitted values differ by {gaps[worst]:.3e} at iteration "
        f"{worst + 1} (tolerance {atol:.0e})")

    # The gap must accumulate, not jump: a step change of orders of magnitude
    # between neighbouring iterations is a divergence, not arithmetic noise.
    growth = np.diff(gaps)
    assert np.max(growth, initial=0.0) <= atol / 4, (
        f"{key}: the gap jumps by {np.max(growth):.3e} in one iteration")

    if min_pinned is not None:
        share = pinned_share(model)
        assert share >= min_pinned, (
            f"{key}: the constraint pins only {share:.0%} of the curve's steps, "
            f"so this case no longer exercises the active set")


@pytest.mark.parametrize("index", range(4))
def test_quadprog_risk_path_matches_r(reference, index):
    case = reference["cases"][index]
    model = fit_case(reference, case)

    risk = np.asarray(model.risk_, dtype=float)
    expected = np.asarray(case["risk"], dtype=float)
    shared = min(len(risk), len(expected))
    # Risk is a sum of squares over 80 observations, so it carries the fitted
    # difference squared and multiplied by n; 5e-8 is the measured worst case.
    np.testing.assert_allclose(risk[:shared], expected[:shared], rtol=0, atol=5e-8)


def test_the_constrained_fit_is_not_the_unconstrained_one(reference):
    """The constraint has to change the fit, or nothing above is a QP test."""
    case = reference["cases"][0]
    frame = frame_of(reference)
    constrained = fit_case(reference, case)
    free = mb.gamboost(
        f"y ~ bbs(x, df={case['df']})", data=frame, family=mb.Gaussian(),
        control=mb.boost_control(mstop=case["mstop"], nu=reference["nu"]),
    )

    a = np.asarray(constrained.fitted(), dtype=float)
    b = np.asarray(free.fitted(), dtype=float)
    assert np.abs(a - b).max() > 0.05, (
        "the increasing constraint leaves the unconstrained fit unchanged; "
        "the fixture data no longer violates monotonicity")
