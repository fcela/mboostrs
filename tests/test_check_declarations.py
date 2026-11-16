"""check_declarations: declared signs and shapes tested against the fitted association.

The simulations have a known direction. A declaration that matches it should
agree, and one that opposes it should contradict. The 30-seed agreement rates
reported with the change are produced by the script in the PR description, not
by these tests, which use fewer seeds to stay fast.
"""

import numpy as np
import pytest

import mboostrs as mb
from mboostrs import _rust
from mboostrs.declarations import _Spec, check_declarations, declaration_items

CONTROL = mb.boost_control(mstop=150, nu=0.1)


def _frame(seed, slope, n=300):
    """A response with a linear association of the given sign on x, and an
    unrelated column z."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    z = rng.normal(size=n)
    y = slope * 2.0 * x + rng.normal(scale=0.5, size=n)
    return {"x": x, "z": z, "y": y}


def _model(data, term):
    return mb.gamboost(f"y ~ {term} + bbs(z, df=4)", data=data,
                       family=mb.Gaussian(), control=CONTROL)


def _one(records, declaration):
    matches = [r for r in records if r.declaration == declaration]
    assert len(matches) == 1, [r.declaration for r in records]
    return matches[0]


def test_increasing_declaration_agrees_when_the_association_increases():
    data = _frame(seed=1, slope=+1)
    model = _model(data, "bmono(x, constraint=increasing, df=4)")
    record = _one(check_declarations(model, data, folds=3), "increasing(x)")
    assert record.verdict == "agrees"
    assert record.test == "shape"
    assert record.evidence["share_in_declared_direction"] >= 0.9
    assert "agrees with the fitted association" in record.agreement


def test_increasing_declaration_contradicts_when_the_association_decreases():
    data = _frame(seed=2, slope=-1)
    model = _model(data, "bmono(x, constraint=increasing, df=4)")
    record = _one(check_declarations(model, data, folds=3), "increasing(x)")
    assert record.verdict == "contradicts"
    assert record.evidence["share_in_declared_direction"] <= 0.1
    assert record.evidence["risk_change_without_constraint"] <= -0.01
    assert "contradicts the fitted association" in record.agreement


def test_positive_declaration_is_tested_on_the_average_contrast():
    data = _frame(seed=3, slope=+1)
    model = _model(data, "bmono(x, constraint=positive, df=4)")
    record = _one(check_declarations(model, data, folds=3), "positive(x)")
    assert record.test == "sign"
    assert record.verdict == "agrees"
    assert record.evidence["contrast"] > 0
    assert record.evidence["folds_agreeing"] == 3
    assert len(record.evidence["fold_contrasts"]) == 3
    assert record.evidence["interval_lower"] > 0
    assert record.evidence["bootstrap_replicates"] == 200


def test_negative_declaration_contradicts_a_positive_association():
    data = _frame(seed=4, slope=+1)
    model = _model(data, "bmono(x, constraint=negative, df=4)")
    record = _one(check_declarations(model, data, folds=3), "negative(x)")
    assert record.verdict == "contradicts"
    assert record.evidence["folds_agreeing"] == 3
    assert record.evidence["interval_lower"] > 0


def test_structural_declaration_is_listed_as_not_tested_with_its_reason():
    data = _frame(seed=5, slope=+1)
    data["g"] = np.arange(len(data["y"])) % 4
    rec = mb.autospec(data, "y", treatment="z", groups="g")
    records = check_declarations(rec, data, folds=3)
    treated = _one(records, "treatment(z)")
    assert treated.verdict == "inconclusive"
    assert treated.test is None
    assert "mboostrs.causal" in treated.reason
    grouped = _one(records, "groups(g)")
    assert grouped.verdict == "inconclusive"
    assert grouped.test is None


def test_model_without_constraints_has_no_declarations():
    data = _frame(seed=6, slope=+1)
    model = mb.gamboost("y ~ bbs(x, df=4) + bols(z)", data=data,
                        family=mb.Gaussian(), control=CONTROL)
    assert check_declarations(model, data, folds=3) == []
    assert declaration_items(model) == []


def test_agreement_wording_is_associational():
    data = _frame(seed=7, slope=-1)
    model = _model(data, "bmono(x, constraint=increasing, df=4)")
    for record in check_declarations(model, data, folds=3):
        text = record.agreement.lower()
        for word in ("causes", "caused", "effect of", "intervention", "would change"):
            assert word not in text


def test_fewer_than_two_folds_is_refused():
    data = _frame(seed=8, slope=+1)
    model = _model(data, "bmono(x, constraint=increasing, df=4)")
    with pytest.raises(ValueError, match="folds must be at least 2"):
        check_declarations(model, data, folds=1)


def test_categorical_declared_variable_is_refused():
    data = _frame(seed=9, slope=+1)
    data["x"] = np.where(data["x"] > 0.5, "high", "low")
    spec = _Spec(formula="y ~ bmono(x, constraint=increasing, df=4) + bols(z)",
                 family=mb.Gaussian(), control=CONTROL, response="y")
    with pytest.raises(ValueError, match="categorical"):
        check_declarations(spec, data, folds=3)


def test_rule_thresholds_in_the_rust_core():
    grid = np.tile(np.linspace(0.0, 1.0, 9)[:, None], (1, 50))
    share_up = _rust.declaration_shape_check(1.0, False, grid, [1.0, 1.0], [1.0, 1.0])
    assert share_up[2] == 1.0 and share_up[-1] == "agrees"
    share_down = _rust.declaration_shape_check(-1.0, False, grid, [1.0, 1.0], [1.0, 1.0])
    # Constant risk gives no risk change, so a curve that goes the other way is
    # not enough to contradict on its own.
    assert share_down[2] == 0.0 and share_down[-1] == "inconclusive"
    flat = _rust.declaration_shape_check(1.0, False, np.ones((9, 50)), [1.0], [1.0])
    assert flat[2] is None and flat[-1] == "inconclusive"
    ones = np.ones(20)
    zeros = np.zeros(20)
    steady = [1.0, 1.1, 0.9, 1.05, 0.95]
    fold_hi = [np.full(20, c) for c in steady]
    fold_lo = [zeros for _ in steady]
    boot = np.linspace(0.8, 1.2, 200)
    agree = _rust.declaration_sign_check(1.0, ones, zeros, fold_hi, fold_lo, boot)
    assert agree[-1] == "agrees" and agree[4] == 5
    assert _rust.declaration_sign_check(-1.0, ones, zeros, fold_hi, fold_lo, boot)[-1] == "contradicts"
    wide = np.linspace(-0.5, 1.5, 200)
    assert _rust.declaration_sign_check(1.0, ones, zeros, fold_hi, fold_lo, wide)[-1] == "inconclusive"
    split = [10.0, 10.0, 10.0, -1.0, -1.0]
    split_hi = [np.full(20, c) for c in split]
    assert _rust.declaration_sign_check(1.0, ones, zeros, split_hi, fold_lo, boot)[-1] == "inconclusive"


def test_recommendation_report_lists_declarations_then_their_checks():
    data = _frame(seed=10, slope=+1)
    rec = mb.autospec(data, "y", monotone={"x": "increasing"})
    assert "declarations, not tested" in rec.report()
    checked = rec.with_declaration_checks(data, folds=3)
    report = checked.report()
    assert "declarations, tested against the data" in report
    assert "increasing(x) agrees with the fitted association" in report


def test_search_session_state_carries_the_check():
    data = _frame(seed=11, slope=+1)
    session = mb.open_search(data, "y", folds=3, mstop=150)
    session.run("set_family", family=mb.Gaussian())
    session.run("set_spec", rhs="bmono(x, constraint=increasing, df=4) + bbs(z, df=4)")
    before = session.state()["declarations"]
    assert before[0]["declaration"] == "increasing(x)"
    assert before[0]["check"] is None
    session.run("check_declarations", folds=3)
    state = session.state()
    check = state["declarations"][0]["check"]
    assert check["verdict"] == "agrees"
    assert "check_declarations" in state["available_actions"]


def test_null_association_is_mostly_inconclusive_for_a_sign_declaration():
    # Eight of the 30 table seeds; the 30-seed table (27 of 30 inconclusive) is in the PR.
    decisions = []
    for seed in range(8):
        data = _frame(seed=100 + seed, slope=0)
        model = _model(data, "bmono(x, constraint=positive, df=4)")
        record = _one(check_declarations(model, data, folds=5, seed=seed), "positive(x)")
        decisions.append(record.verdict)
    assert sum(v != "inconclusive" for v in decisions) <= 1, decisions
