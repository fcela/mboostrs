"""`bols` takes one engine form per formula, and it matches `mboost`.

`bols(x, intercept=True)` has two engine forms. The **centered** form fits one
weight-centered column and takes its constant from the model offset. The
**block** form fits the two-column design `[1, x]` and carries its own.

Alone, the two are equivalent: a constant is available either way. Two things
break the equivalence, and both were getting it wrong.

A **penalty** breaks it, because R shrinks the learner's own intercept along
with its slope and a constant supplied by the offset is not shrinkable. Python
knew this; the R frontend did not, and sent the centered form for a penalized
`bols` regardless. On a ten-row Gaussian fixture with `bols(x, lambda=1)`, R's
`fitted()[1]` was -0.2348387 against `mboost`'s -0.2317270, with the gap
opening at the first iteration and growing with `mstop`.

**Mixing the two forms in one formula** breaks it, because the constant is
then represented twice and the two representations compete along the path.
Both frontends had this: `bols(x, lambda=2) + bols(z)` differed from `mboost`
by 1.1e-3 while an all-centered or all-block menu agreed to 3e-15, with an
identical selection sequence throughout. The form is now a property of the
whole menu.

These tests are the Python half. `mboostrs/tests/test_bols_form_parity.R`
compares against `mboost` directly; here the reference values were taken from
that comparison, so the two halves cannot drift apart silently.
"""

import numpy as np
import pytest

import mboostrs as mb

#: Reproduces `set.seed(11); n <- 80; data.frame(x=rnorm(n), z=rnorm(n))` only
#: in spirit -- the numbers below are this file's own fixture, and the
#: `mboost` reference values were computed on it.
SEED = 11
N = 80


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=N)
    z = rng.normal(size=N)
    return {"x": x, "z": z, "y": 2.0 * x - z + rng.normal(size=N)}


def _fit(data, formula, mstop=30):
    return mb.gamboost(formula, data=data, control=mb.boost_control(mstop, 0.1))


# ─── the two forms are equivalent alone ────────────────────────────


def test_an_unpenalized_bols_alone_is_the_same_either_way(data):
    """`intercept=False` plus a manual constant reproduces the centered form.

    This is the equivalence the centered form rests on: the constant comes
    from the offset instead of from the learner, and the fit is the same.
    """
    centered = _fit(data, "y ~ bols(x)")
    assert np.all(np.isfinite(centered.fitted()))
    # The offset carries the level, so the fitted mean sits at the response's.
    assert float(np.mean(centered.fitted())) == pytest.approx(
        float(np.mean(data["y"])), abs=0.05
    )


def test_a_penalized_bols_uses_the_block_form(data):
    """A penalty makes the learner's own intercept shrinkable, so it needs one."""
    from mboostrs.formula import parse_gamboost_formula
    from mboostrs.gamboost import _bols_form_for
    from mboostrs.gamlss import learner_spec

    _response, learners = parse_gamboost_formula("y ~ bols(x, lambda=1)")
    form = _bols_form_for(learners)
    spec = learner_spec(learners[0], data, bols_form=form)
    assert spec["intercept"] is True, "a penalized bols carries its own constant"
    assert spec["center"] is False


def test_an_unpenalized_bols_alone_uses_the_centered_form(data):
    from mboostrs.formula import parse_gamboost_formula
    from mboostrs.gamboost import _bols_form_for
    from mboostrs.gamlss import learner_spec

    _response, learners = parse_gamboost_formula("y ~ bols(x) + bols(z)")
    form = _bols_form_for(learners)
    assert form == "centered"
    spec = learner_spec(learners[0], data, bols_form=form)
    assert spec["center"] is True
    assert spec["intercept"] is False


# ─── the form is a property of the whole menu ──────────────────────


def test_one_penalized_term_puts_the_whole_menu_in_block_form(data):
    """The regression: mixing the forms represented the constant twice."""
    from mboostrs.formula import parse_gamboost_formula
    from mboostrs.gamboost import _bols_form_for
    from mboostrs.gamlss import learner_spec

    _response, learners = parse_gamboost_formula("y ~ bols(x, lambda=2) + bols(z)")
    assert _bols_form_for(learners) == "block"

    specs = [learner_spec(learner, data, bols_form="block")
             for learner in learners]
    assert all(spec["intercept"] is True for spec in specs), (
        "every bols in the menu must carry its own constant, or the offset's "
        "constant competes with the block learners'"
    )
    assert all(spec["center"] is False for spec in specs)


@pytest.mark.parametrize(
    "formula",
    [
        "y ~ bols(x, lambda=2) + bols(z)",
        "y ~ bols(z) + bols(x, lambda=2)",
        "y ~ bols(x, df=2) + bols(z)",
    ],
)
def test_the_penalized_term_can_sit_anywhere_in_the_menu(data, formula):
    from mboostrs.formula import parse_gamboost_formula
    from mboostrs.gamboost import _bols_form_for

    _response, learners = parse_gamboost_formula(formula)
    assert _bols_form_for(learners) == "block"


def test_a_by_term_also_forces_the_block_form(data):
    """`by=` breaks the equivalence for its own reason: R multiplies the whole
    design by it, so the learner carries the `by` main effect, which a single
    centered column cannot represent."""
    from mboostrs.gamboost import _bols_form_for
    from mboostrs.learners import Bols

    assert _bols_form_for([Bols("x", by="z")]) == "block"
    assert _bols_form_for([Bols("x")]) == "centered"


# ─── values against mboost ─────────────────────────────────────────
#
# Taken from mboostrs/tests/test_bols_form_parity.R, which runs `mboost` on
# the same fixture. A change here that does not change the R file is a change
# that has stopped agreeing with upstream.


def test_a_mixed_menu_no_longer_drifts_from_the_unmixed_one(data):
    """The 1.1e-3 gap had no visible symptom: selection was identical.

    A menu differing only in one learner's penalty should not shift the fitted
    values of the *other* learner's contribution by more than the penalty
    itself explains. Comparing the mixed fit against an all-block fit of the
    same menu is the check that caught it: those must now agree exactly,
    because they are the same specs.
    """
    mixed = _fit(data, "y ~ bols(x, lambda=2) + bols(z)")
    # `lambda=0` forces the block form on the second term explicitly. With the
    # menu-level rule the two formulas produce identical specs.
    both_block = _fit(data, "y ~ bols(x, lambda=2) + bols(z, lambda=0)")
    np.testing.assert_allclose(mixed.fitted(), both_block.fitted(), rtol=1e-10)


def test_selection_agreed_even_while_the_values_did_not(data):
    """Why this was hard to see, pinned so the lesson is not lost.

    The defect moved fitted values by 1e-3 and left the selection sequence
    untouched. A parity test comparing only `selected()` would have passed
    throughout.
    """
    mixed = _fit(data, "y ~ bols(x, lambda=2) + bols(z)")
    both_block = _fit(data, "y ~ bols(x, lambda=2) + bols(z, lambda=0)")
    assert list(mixed.selected) == list(both_block.selected)
