"""Reporting of residual trees, interaction screens and tensor surfaces (#124).

The report separates a screen's lead (what the additive fit leaves behind,
measured on training rows) from the adopted term (what held-out risk says),
states how much of the adopted prediction the tree carries, records whether a
tensor surface has its marginal effects in the formula, and reports how many
candidates each screen generated.
"""

import math
import warnings

import numpy as np
import pytest

import mboostrs as mb
from mboostrs._autospec import interaction_structure
from mboostrs._rust import contribution_shares, leftover_share, paired_mean_se


@pytest.fixture(scope="module")
def threeway_fit():
    """A four-column response whose structure is a three-way interaction."""
    rng = np.random.default_rng(0)
    n = 1500
    X = rng.uniform(-1, 1, (n, 4))
    data = {f"x{i}": X[:, i] for i in range(4)}
    y = X[:, 0] + np.sin(3 * X[:, 1] * X[:, 2] * X[:, 3]) + rng.normal(0, .3, n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = mb.autospec(data | {"y": y}, "y", budget=200, residual_tree=True)
    return rec, data | {"y": y}


@pytest.fixture(scope="module")
def additive_fit():
    """A response with no structure beyond its two main effects."""
    rng = np.random.default_rng(0)
    n = 1500
    X = rng.uniform(-1, 1, (n, 4))
    data = {f"x{i}": X[:, i] for i in range(4)}
    y = X[:, 0] + X[:, 1] ** 2 + rng.normal(0, .3, n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = mb.autospec(data | {"y": y}, "y", budget=200, residual_tree=True)
    return rec


# ── contribution statistics (Rust core, thin binding) ──────────────────────


def test_covariance_shares_of_two_parts_sum_to_one():
    rng = np.random.default_rng(1)
    part = rng.normal(size=200)
    rest = 0.5 * part + rng.normal(size=200) + 3.0
    cov_part, _ = contribution_shares(part, rest)
    cov_rest, _ = contribution_shares(rest, part)
    assert math.isclose(cov_part + cov_rest, 1.0, abs_tol=1e-12)


def test_a_part_with_no_contribution_has_zero_shares():
    rest = np.array([1.0, 2.0, 4.0, 3.0])
    assert contribution_shares(np.zeros(4), rest) == (0.0, 0.0)


def test_an_intercept_does_not_change_the_absolute_share():
    rng = np.random.default_rng(2)
    part = rng.normal(size=100)
    rest = rng.normal(size=100)
    base = contribution_shares(part, rest)[1]
    shifted = contribution_shares(part + 5.0, rest - 2.0)[1]
    assert math.isclose(base, shifted, rel_tol=1e-12)


def test_paired_mean_and_standard_error():
    mean, se = paired_mean_se(np.array([2.0, 4.0, 3.0, 5.0]),
                              np.array([1.0, 1.0, 1.0, 3.0]))
    assert math.isclose(mean, 2.0)
    assert math.isclose(se, math.sqrt((2.0 / 3.0) / 4.0), rel_tol=1e-12)


def test_leftover_share_of_an_exact_and_an_empty_fit():
    y = np.array([1.0, 2.0, 3.0, 6.0])
    assert math.isclose(leftover_share(y, np.zeros(4)), 1.0)
    assert math.isclose(leftover_share(y, y), 0.0, abs_tol=1e-12)


def test_contribution_statistics_reject_mismatched_lengths():
    with pytest.raises(ValueError):
        contribution_shares(np.ones(3), np.ones(4))
    with pytest.raises(ValueError):
        paired_mean_se(np.ones(3), np.ones(2))


# ── residual tree: adoption, held-out value and share ──────────────────────


def test_adopted_tree_records_held_out_gain_and_the_share_it_carries(threeway_fit):
    rec, _ = threeway_fit
    tree = rec.evidence["residual_tree"]
    assert tree["adopted"] is True
    assert tree["held_out_gain"] > 0
    assert math.isfinite(tree["held_out_se"]) and tree["held_out_se"] > 0
    assert tree["folds"] == 5
    assert tree["n_candidates"] == 1
    share = tree["share"]
    assert 0.0 <= share["absolute_share"] <= 1.0
    assert share["covariance_share"] == pytest.approx(1.0, abs=1e-6)
    assert share["rows"] == 1500


def test_rec_fit_carries_the_adopted_tree(threeway_fit):
    from mboostrs.autospec import _tree_contribution_share

    rec, frame = threeway_fit
    model = rec.fit(frame)
    # The formula names the tree; the refitted model must hold it as a learner
    # and carry the same share the recommendation recorded.
    assert "btree(" in rec.formula
    assert len(model.base_learner_names()) == len(rec.formula.split("~", 1)[1].split(" + "))
    share = _tree_contribution_share(model)
    assert share is not None
    assert share["covariance_share"] == pytest.approx(
        rec.evidence["residual_tree"]["share"]["covariance_share"], abs=1e-6)


def test_report_states_how_much_of_the_fit_the_tree_carries(threeway_fit):
    rec, _ = threeway_fit
    report = rec.report()
    line = next(line for line in report.splitlines()
                if line.startswith("residual tree: ") and "added" in line)
    assert "lowers the held-out risk by" in line
    assert "of the variance of the prediction" in line
    assert "of the absolute contribution" in line


def test_declined_tree_records_its_loss_and_states_it_carries_nothing(additive_fit):
    tree = additive_fit.evidence["residual_tree"]
    assert tree["adopted"] is False
    assert tree["held_out_gain"] < 0
    assert "share" not in tree
    report = additive_fit.report()
    assert "residual tree: not added" in report
    assert "the tree carries none of the fit" in report
    assert "raises the held-out risk by" in report


# ── residual tree on the distributional (LSS) route (#839) ─────────────────


@pytest.fixture(scope="module")
def lss_tree_fit():
    """Gamma data whose shape moves with x and whose mean is a sign
    interaction of z and w: the family probe chooses GammaLSS, and the tree
    on the mu predictor fits the interaction the additive terms cannot."""
    rng = np.random.default_rng(0)
    n = 400
    X = rng.uniform(-1.5, 1.5, (n, 3))
    mu = np.exp(0.4 + np.sign(X[:, 1] * X[:, 2]))
    shape = np.exp(1.2 + 0.7 * X[:, 0])
    data = {"y": rng.gamma(shape, mu / shape), "x": X[:, 0], "z": X[:, 1], "w": X[:, 2]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mb.autospec(data, "y", budget=300, mstop=100, seed=0, residual_tree=True)


def test_adopted_lss_tree_records_the_held_out_gain_and_its_standard_error(lss_tree_fit):
    rec = lss_tree_fit
    assert isinstance(rec.family, mb.GammaLSS)
    tree = rec.evidence["residual_tree"]
    assert tree["adopted"] is True
    assert tree["held_out_gain"] > 0
    assert math.isfinite(tree["held_out_se"]) and tree["held_out_se"] > 0
    assert tree["folds"] == 5
    assert "held_out_reason" not in tree
    # The recorded z is the adoption rule's paired statistic; it is the
    # recorded gain over the recorded standard error.
    assert tree["z"] == pytest.approx(tree["held_out_gain"] / tree["held_out_se"], abs=0.02)
    line = next(line for line in rec.report().splitlines()
                if line.startswith("residual tree: ") and "added" in line)
    assert "lowers the held-out risk by" in line


def test_paired_held_out_gain_states_why_the_standard_error_is_undefined():
    from mboostrs.autospec import _paired_held_out_gain

    gain, se, z, reason = _paired_held_out_gain([2.0, 4.0, 3.0, 5.0], [1.0, 1.0, 1.0, 3.0])
    assert (gain, z, reason) == (2.0, pytest.approx(2.0 / se), None)
    assert se == pytest.approx(paired_mean_se(np.array([2.0, 4.0, 3.0, 5.0]),
                                              np.array([1.0, 1.0, 1.0, 3.0]))[1])
    gain, se, z, reason = _paired_held_out_gain([3.0], [1.0])
    assert gain == 2.0 and math.isnan(se) and z == 0.0
    assert reason.startswith("one fold")
    for base, tree in (([1.0, float("nan")], [1.0, 1.0]), ((), [1.0])):
        gain, se, z, reason = _paired_held_out_gain(base, tree)
        assert math.isnan(gain) and math.isnan(se) and z == 0.0
        assert "missing" in reason


# ── detection (screen lead) separate from adoption ─────────────────────────


def test_detection_is_recorded_apart_from_the_adoption(threeway_fit, additive_fit):
    for rec in (threeway_fit[0], additive_fit):
        detection = rec.evidence["residual_tree_detection"]
        assert set(detection) >= {"leftover_share", "tree_r2", "product",
                                  "candidates", "basis"}
        assert detection["candidates"] == 1
        assert detection["product"] == pytest.approx(
            detection["leftover_share"] * detection["tree_r2"])
        assert "training rows" in detection["basis"]
    report = threeway_fit[0].report()
    detected = [i for i, line in enumerate(report.splitlines())
                if line.startswith("residual tree screen (a lead")]
    adopted = [i for i, line in enumerate(report.splitlines())
               if line.startswith("residual tree: ")]
    assert detected and adopted and detected[0] < adopted[0]


def test_detection_leftover_is_a_share_of_the_response_variance(threeway_fit, additive_fit):
    # The leftover is measured on the main-effect fit, so it is a share of the
    # response variance: positive for a fit that misses structure, and small
    # for a fit that does not.
    missed = threeway_fit[0].evidence["residual_tree_detection"]["leftover_share"]
    assert 0.0 < missed <= 1.0
    assert additive_fit.evidence["residual_tree_detection"]["leftover_share"] < missed


# ── candidate counts ───────────────────────────────────────────────────────


def test_each_screen_reports_how_many_candidates_it_generated(threeway_fit):
    rec, _ = threeway_fit
    counts = rec.evidence["interaction_candidates"]
    # Four modelled columns give C(4, 2) = 6 pairs; the screen ranks all of them.
    assert counts["residual_pairs"] == 6
    assert counts["surrogate_pairs"] >= 0
    report = rec.report()
    assert "6 pairs screened on the residual" in report
    assert "candidate columns" in report
    assert "candidate pairs" in report
    assert "1 candidate tree" in report


def test_term_report_reports_its_candidate_specifications():
    rng = np.random.default_rng(3)
    n = 240
    x1, x2 = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
    frame = {"x1": x1, "x2": x2, "y": x1 + 0.5 * x2 + rng.normal(0, .3, n)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = mb.term_report("y ~ bbs(x1) + bbs(x2)", frame, folds=3, mstop=30,
                                shapes=False)
    # The full specification and one removal per term: 1 + 2 candidates.
    assert report.candidates == 3
    assert report.to_dict()["candidates"] == 3
    assert "3 candidate specifications" in repr(report)


def test_insights_pair_leads_report_the_pairs_screened():
    rng = np.random.default_rng(4)
    n = 300
    X = rng.uniform(-1, 1, (n, 3))
    frame = {"x0": X[:, 0], "x1": X[:, 1], "x2": X[:, 2],
             "y": X[:, 0] + X[:, 1] * X[:, 2] + rng.normal(0, .3, n)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        findings = mb.insights(frame, "y")
    leads = [f for f in findings if f.basis == "detected"
             and f.finding.startswith("interaction between")]
    assert leads, "the screen should report at least one pair lead"
    # Three predictor columns give C(3, 2) = 3 candidate pairs.
    assert all(f.evidence["candidates"] == 3 for f in leads)


# ── tensor surfaces: marginal effects and purification ─────────────────────


def test_tensor_surface_is_recorded_with_marginals_absent_and_not_purified():
    formula = "y ~ bbs(x0, df=4) + bbs(x1, df=2.5, knots=5) %X% bbs(x2, df=2.5, knots=5)"
    (record,) = interaction_structure(formula)
    assert record["term"] == "bbs(x1, df=2.5, knots=5) %X% bbs(x2, df=2.5, knots=5)"
    assert record["kind"] == "tensor"
    assert record["columns"] == ("x1", "x2")
    assert record["marginal_in_formula"] == {"x1": False, "x2": False}
    assert record["spans_marginals"] is True
    assert record["purified"] is False


def test_tensor_surface_records_marginals_present_in_the_formula():
    formula = ("y ~ bbs(x1) + bbs(x2) + "
               "bbs(x1, df=2.5, knots=5) %X% bbs(x2, df=2.5, knots=5)")
    records = interaction_structure(formula)
    assert [r["term"] for r in records] == [
        "bbs(x1, df=2.5, knots=5) %X% bbs(x2, df=2.5, knots=5)"]
    assert records[0]["marginal_in_formula"] == {"x1": True, "x2": True}


def test_varying_and_factor_terms_are_not_classified_as_tensors():
    formula = "y ~ bbs(x1) + bbs(x2, by=b) + brandom(g) %X% brandom(b)"
    kinds = {r["term"]: (r["kind"], r["spans_marginals"])
             for r in interaction_structure(formula)}
    assert kinds["bbs(x2, by=b)"] == ("varying", None)
    assert kinds["brandom(g) %X% brandom(b)"] == ("factor-by-factor", None)


def test_adopted_pair_is_listed_with_its_structure_in_the_report():
    rng = np.random.default_rng(1)
    n = 1200
    X = rng.uniform(-1, 1, (n, 3))
    data = {f"x{i}": X[:, i] for i in range(3)}
    y = X[:, 0] + 2 * X[:, 1] * X[:, 2] + rng.normal(0, .3, n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = mb.autospec(data | {"y": y}, "y", budget=300, seed=0,
                          interactions="adopt", residual_tree=False)
    entries = {r["term"]: r for r in rec.evidence["interaction_structure"]}
    assert "bbs(x1, df=2.5, knots=5) %X% bbs(x2, df=2.5, knots=5)" in entries
    report = rec.report()
    # The competing-form picker now retains the additive incumbent's main
    # effects; the report must describe the specification actually adopted.
    assert entries["bbs(x1, df=2.5, knots=5) %X% bbs(x2, df=2.5, knots=5)"][
        "marginal_in_formula"] == {"x1": True, "x2": True}
    assert "tensor; main effects in the formula: x1, x2" in report
    assert "adopted with held-out evidence" in report
    assert "not purified" in report
