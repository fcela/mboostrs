"""Obstacles an agent's specification search met on data with a known truth.

Each test covers one change made after running the documented search loop
(`classify_columns`, `autospec`, `term_report`, `suggest_specifications`,
`compare_specifications`) on simulated frames whose generating process is
known; `cases/agent_specification_search.ipynb` records the search.
"""

import numpy as np
import pytest

import mboostrs as mb
from mboostrs.terms import ScreenedPair


def _coded_groups(seed=0, n=1200, k=40):
    """A grouping code the response reads as labels. The codes are sparse
    (3 apart, not 0..k-1): a 0-based code that spans its whole range reads
    as a count (#313) and never reaches the label logic these tests cover."""
    rng = np.random.default_rng(seed)
    index = rng.integers(0, k, n)
    g = (index * 3).astype(float)
    x = rng.normal(size=n)
    y = (np.sin(2 * x) + rng.normal(size=k)[index]
         + rng.normal(scale=0.8, size=n))
    return {"x": x, "g": g, "y": y}


def _whole_number_age(seed=0, n=1200):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 80, n).astype(float)
    y = 0.002 * (age - 50) ** 2 + 0.03 * age + rng.normal(scale=0.5, size=n)
    return {"age": age, "y": y}


# ─── a whole-number column: grouping code or measurement ───────────────


def test_label_evidence_reads_a_grouping_code_as_labels():
    frame = _coded_groups()
    evidence = mb.label_evidence("g", frame["g"], frame["y"])
    assert evidence.labels
    assert evidence.signal_z > 4
    assert evidence.adjacency_ratio > 0.5
    assert "behave as labels" in evidence.sentence


def test_label_evidence_reads_a_whole_number_age_as_a_measurement():
    frame = _whole_number_age()
    evidence = mb.label_evidence("age", frame["age"], frame["y"])
    assert evidence is not None
    assert not evidence.labels
    assert evidence.adjacency_ratio < 0.5


def test_label_evidence_applies_only_above_the_level_cutoff_and_to_whole_numbers():
    rng = np.random.default_rng(1)
    y = rng.normal(size=300)
    assert mb.label_evidence("few", rng.integers(0, 6, 300).astype(float), y) is None
    assert mb.label_evidence("float", rng.normal(size=300), y) is None


def test_classify_columns_uses_the_response_to_read_a_many_level_code():
    kinds = {c.name: c for c in mb.classify_columns(_coded_groups(), response="y")}
    assert kinds["g"].kind == "categorical"
    assert kinds["g"].term.startswith("brandom(g")
    assert "labels" in kinds["g"].reason

    kinds = {c.name: c for c in mb.classify_columns(_whole_number_age(), response="y")}
    assert kinds["age"].term.startswith("bbs(age")

    # Without a response there is nothing to read the column against.
    predictors = {k: v for k, v in _coded_groups().items() if k != "y"}
    kinds = {c.name: c for c in mb.classify_columns(predictors)}
    assert kinds["g"].kind == "continuous"


def test_autospec_fits_the_code_as_groups_and_names_the_fold_design_it_needs():
    rec = mb.autospec(_coded_groups(), "y")
    assert "brandom(g" in rec.formula
    findings = [h for h in rec.hygiene if h.kind == "integer-labels"]
    assert findings
    # The grouping is detected, not left to the caller: the folds hold
    # whole groups out without a groups= declaration.
    assert "detected as a grouping" in findings[0].consequence
    assert "hold whole groups out" in findings[0].consequence


def test_group_structure_is_detected_without_a_groups_declaration():
    """A column the terms fit as a group names the folds on its own: random
    folds would score rows whose group the fit has seen."""
    frame = _coded_groups(n=600, k=20)
    rec = mb.autospec(frame, "y", budget=60, mstop=100)
    assert rec.groups == ()
    assert "grouped folds" in rec.folds_description
    assert "g" in rec.folds_description
    assert rec.fits > 0


def test_a_groups_declaration_holds_whole_groups_out_in_every_probe():
    frame = _coded_groups(n=600, k=20)
    rec = mb.autospec(frame, "y", budget=60, mstop=100, groups="g")
    assert "brandom(g" in rec.formula
    assert rec.groups == ("g",)
    assert "grouped folds" in rec.folds_description
    assert rec.report().splitlines()[2].startswith("folds: ")
    finding = [h for h in rec.hygiene if h.kind == "integer-labels"]
    assert finding and "declared as groups" in finding[0].consequence


def test_a_groups_declaration_naming_an_unknown_column_is_refused():
    with pytest.raises(ValueError, match="groups= names 'nope'"):
        mb.autospec(_coded_groups(n=300), "y", groups="nope")


# ─── step count ─────────────────────────────────────────────────────────


def test_the_mstop_probe_extends_a_grid_whose_risk_is_still_falling():
    rng = np.random.default_rng(4)
    n = 300
    x = rng.uniform(-3, 3, size=n)
    data = {"x": x, "y": np.sin(1.5 * x) * 3 + rng.normal(scale=0.2, size=n)}
    rec = mb.autospec(data, "y", budget=400, mstop=100)
    row = rec.evidence["mstop"]
    assert "extended" in row.sentence
    assert len(row.grid) > 201


def test_a_budget_without_room_for_an_extension_does_not_extend():
    rng = np.random.default_rng(4)
    n = 300
    x = rng.uniform(-3, 3, size=n)
    data = {"x": x, "y": np.sin(1.5 * x) * 3 + rng.normal(scale=0.2, size=n)}
    # Five fits are one sweep of 5 folds: no room to extend it. (Twenty
    # used to be none either; the shape probe now costs one-term fits, so
    # twenty leave room for an extension.)
    rec = mb.autospec(data, "y", budget=5, mstop=100)
    row = rec.evidence["mstop"]
    assert "extended" not in row.sentence
    assert len(row.grid) == 201


# ─── interaction terms follow the column types ──────────────────────────


@pytest.mark.parametrize(("left", "right", "levels", "groups", "expected"), [
    ("x", "z", {"x": 100, "z": 100}, (), "bbs(x, df=2.5, knots=5) %X% bbs(z, df=2.5, knots=5)"),
    ("x", "b", {"x": 100, "b": 2}, (), "bbs(x, by=b)"),
    ("b", "x", {"x": 100, "b": 2}, (), "bbs(x, by=b)"),
    ("g", "x", {"g": 40, "x": 100}, ("g",), "brandom(g, by=x)"),
    ("a", "b", {"a": 2, "b": 2}, (), "bols(a, by=b)"),
    ("g", "b", {"g": 40, "b": 2}, ("g",), "brandom(g, by=b)"),
    ("g", "h", {"g": 40, "h": 9}, ("g", "h"), "brandom(g) %X% brandom(h)"),
])
def test_pair_term_for_follows_the_column_types(left, right, levels, groups, expected):
    assert mb.pair_term_for(left, right, levels=levels, groups=groups) == expected


def test_a_two_valued_column_is_paired_as_a_switched_smooth():
    rng = np.random.default_rng(3)
    n = 800
    x = rng.normal(size=n)
    b = rng.integers(0, 2, n).astype(float)
    data = {"x": x, "b": b, "z": rng.normal(size=n),
            "y": np.where(b == 1, 1.5 * x, -0.5 * x) + rng.normal(size=n)}
    rec = mb.autospec(data, "y", budget=200, mstop=100)
    leads = [lead for lead in rec.evidence["interactions"]
             if {lead.left, lead.right} == {"x", "b"}]
    assert leads, "the generating interaction was not screened"
    assert leads[0].term == "bbs(x, by=b)"
    assert leads[0].kind == "varying"


def test_a_random_slope_fits_what_a_smooth_multiplied_by_group_codes_cannot():
    """`bbs(x, by=g)` multiplies the smooth by the code; codes are labels."""
    rng = np.random.default_rng(0)
    n = 900
    g = rng.integers(0, 30, n).astype(float)
    x = rng.normal(size=n)
    y = rng.normal(size=30)[g.astype(int)] * x + rng.normal(scale=0.3, size=n)
    frame = {"g": g, "x": x, "y": y}
    random_slope = mb.cv_risk("bols(x) + brandom(g, by=x)", frame, "y", folds=5, mstop=300)
    code_multiplier = mb.cv_risk("bols(x) + bbs(x, by=g)", frame, "y", folds=5, mstop=300)
    assert random_slope < 0.5 * code_multiplier


def test_an_interaction_suggestion_with_a_group_is_a_random_slope():
    frame = _coded_groups(n=600, k=20)
    report = mb.term_report("y ~ bbs(x) + brandom(g)", frame, "y", folds=3,
                            mstop=60, shapes=False)
    pair = ScreenedPair(left="x", right="g", removed=0.4)
    added = [s for s in mb.suggest_specifications(report, screened=[pair])
             if s.action == "add"]
    assert added
    assert added[0].specification.endswith("brandom(g, by=x)")


# ─── measured gains carry their noise ───────────────────────────────────


@pytest.fixture(scope="module")
def curved_with_noise():
    rng = np.random.default_rng(10)
    n = 400
    x, n1, n2 = rng.normal(size=(3, n))
    return {"x": x, "n1": n1, "n2": n2,
            "y": 2 * np.sin(1.5 * x) + rng.normal(size=n)}


def test_term_report_gains_carry_a_paired_standard_error(curved_with_noise):
    report = mb.term_report("y ~ bols(x) + bbs(n1) + bbs(n2)", curved_with_noise,
                            "y", folds=5, mstop=100)
    assert all(np.isfinite(row["se"]) and row["se"] >= 0 for row in report.contributions)
    assert all(np.isfinite(row["se"]) for row in report.shapes)
    assert report.levels["x"] == 400
    assert "se " in repr(report)


def test_a_shape_change_within_noise_is_not_suggested(curved_with_noise):
    report = mb.term_report("y ~ bols(x) + bbs(n1) + bbs(n2)", curved_with_noise,
                            "y", folds=5, mstop=100)
    default = [s for s in mb.suggest_specifications(report) if s.action == "refit"]
    assert any(s.term == "bols(x)" for s in default), "the real shape change was filtered"
    for suggestion in default:
        assert suggestion.expected_gain >= 2 * suggestion.evidence["standard error"]
    everything = [s for s in mb.suggest_specifications(report, noise=0) if s.action == "refit"]
    assert {s.term for s in default} <= {s.term for s in everything}


# ─── a spread that moves with a predictor ───────────────────────────────


def test_a_spread_driven_by_a_predictor_is_named():
    rng = np.random.default_rng(3)
    n = 1200
    x, w = rng.normal(size=(2, n))
    y = 1.5 * x + np.cos(w) + rng.normal(size=n) * np.exp(0.2 + 0.5 * w)
    data = {"x": x, "w": w, "y": y}
    model = mb.gamboost("y ~ bbs(x) + bbs(w)", data=data,
                        control=mb.boost_control(150, 0.1))
    spread = mb.heteroscedasticity(model)
    assert spread["strongest"] == "w"
    assert spread["by_column"]["w"]["t"] > 5
    report = mb.term_report("y ~ bbs(x) + bbs(w)", data, "y", folds=3, mstop=100,
                            shapes=False)
    flagged = [s for s in mb.suggest_specifications(report, model=model)
               if s.action == "model"]
    assert flagged
    assert "`w`" in flagged[0].reason


# ─── columns that are not predictors ────────────────────────────────────


def test_a_text_identifier_is_reported_and_left_out():
    rng = np.random.default_rng(5)
    n = 200
    data = {"row_name": np.array([f"row {i}" for i in range(n)]),
            "x": rng.normal(size=n), "y": rng.normal(size=n)}
    rec = mb.autospec(data, "y")
    assert [h.kind for h in rec.hygiene if h.column == "row_name"] == ["identifier"]
    assert "row_name" not in rec.formula


def test_a_unique_integer_column_is_flagged_but_kept():
    rng = np.random.default_rng(6)
    n = 200
    data = {"record": rng.permutation(n).astype(float) * 7 + 1000,
            "x": rng.normal(size=n), "y": rng.normal(size=n)}
    rec = mb.autospec(data, "y")
    assert [h.kind for h in rec.hygiene if h.column == "record"] == ["unique-integer"]
    assert "record" in rec.formula


def test_columns_named_like_the_response_are_flagged_as_siblings():
    rng = np.random.default_rng(7)
    n = 300
    life = rng.normal(80, 3, size=n)
    data = {"le_q1_M": life,
            "le_q1_F": life + rng.normal(scale=2, size=n),
            "sd_le_q1_M": np.abs(rng.normal(size=n)),
            "income": rng.normal(size=n)}
    rec = mb.autospec(data, "le_q1_M")
    siblings = {h.column: h.detail for h in rec.hygiene if h.kind == "response-sibling"}
    assert set(siblings) == {"le_q1_F", "sd_le_q1_M"}
    assert "m replaced by f" in siblings["le_q1_F"]
    assert "correlation" in siblings["le_q1_F"]


def test_a_second_encoding_of_one_grouping_is_left_out():
    rng = np.random.default_rng(8)
    n = 600
    codes = rng.integers(0, 30, n)
    data = {"district": codes.astype(float),
            "district_name": np.array([f"district {c}" for c in codes]),
            "x": rng.normal(size=n),
            "y": rng.normal(size=30)[codes] + rng.normal(size=n)}
    rec = mb.autospec(data, "y")
    duplicates = [h for h in rec.hygiene if h.kind == "duplicate-encoding"]
    assert [h.column for h in duplicates] == ["district_name"]
    assert "district_name" not in rec.formula
    assert "district" in rec.formula


def test_a_coarse_grouping_nested_in_a_fine_one_is_not_a_duplicate():
    rng = np.random.default_rng(9)
    n = 600
    state = rng.integers(0, 40, n)
    data = {"state": state.astype(float), "division": (state // 5).astype(float),
            "y": rng.normal(size=n)}
    rec = mb.autospec(data, "y")
    assert not [h for h in rec.hygiene if h.kind == "duplicate-encoding"]


# ─── group terms on folds that hold their groups out ────────────────────


def test_a_group_term_is_not_measurable_on_folds_that_hold_its_groups_out():
    frame = _coded_groups(n=600, k=20)
    grouped = mb.grouped_folds(frame["g"], B=4)
    report = mb.term_report("y ~ bbs(x) + brandom(g)", frame, "y", folds=grouped,
                            mstop=60, shapes=False)
    group_row = next(row for row in report.contributions if row["term"] == "brandom(g)")
    assert group_row["measurable"] is False
    assert "how the other terms are fitted" in group_row["note"]
    assert "predicts the average group" in repr(report)
    assert not [s for s in mb.suggest_specifications(report)
                if s.action == "drop" and "brandom(g)" in s.term]

    random_folds = mb.term_report("y ~ bbs(x) + brandom(g)", frame, "y", folds=4,
                                  mstop=60, shapes=False)
    group_row = next(row for row in random_folds.contributions if row["term"] == "brandom(g)")
    assert group_row["measurable"] is True
    assert group_row["earns"] > 0


def test_terms_that_earn_nothing_are_also_offered_as_one_drop():
    rng = np.random.default_rng(12)
    n = 400
    data = {"x": rng.normal(size=n), "n1": rng.normal(size=n),
            "n2": rng.normal(size=n), "n3": rng.normal(size=n)}
    data["y"] = 2 * data["x"] + rng.normal(size=n)
    report = mb.term_report("y ~ bols(x) + bols(n1) + bols(n2) + bols(n3)", data, "y",
                            folds=4, mstop=100, shapes=False)
    joint = [s for s in mb.suggest_specifications(report)
             if s.action == "drop" and s.term.endswith("terms that earn nothing")]
    idle = [row["term"] for row in report.contributions
            if row["earns"] <= 0 or row["earns"] < 2 * row["se"]]
    if len(idle) >= 2:
        assert joint
        assert set(joint[0].evidence["terms"]) == set(idle)
        for term in idle:
            assert term not in joint[0].specification
        assert "bols(x)" in joint[0].specification


def test_a_screen_that_keeps_group_levels_finds_a_random_slope():
    rng = np.random.default_rng(2)
    n = 1500
    clinic = rng.integers(0, 30, n).astype(float)
    x = rng.normal(size=n)
    residual = rng.normal(size=30)[clinic.astype(int)] * x + rng.normal(scale=0.5, size=n)
    frame = {"clinic": clinic, "x": x, "other": rng.normal(size=n)}
    binned = {(p.left, p.right): p.removed for p in mb.screen_interactions(frame, residual)}
    levels = {(p.left, p.right): p.removed
              for p in mb.screen_interactions(frame, residual, groups=["clinic"])}
    assert levels[("clinic", "x")] > 3 * max(binned[("clinic", "x")], 0.01)
    assert levels[("clinic", "x")] == max(levels.values())


# ─── the step count grows until the risk turns ─────────────────────────


def test_tune_mstop_doubles_the_grid_until_the_risk_turns():
    rng = np.random.default_rng(4)
    n = 300
    x = rng.uniform(-3, 3, size=n)
    data = {"x": x, "y": np.sin(1.5 * x) * 3 + rng.normal(scale=0.3, size=n)}
    tuning = mb.tune_mstop("y ~ bbs(x)", data, folds=5, start=50)
    assert tuning.extended
    assert tuning.largest > 50
    # Each doubling resumes the fold paths, so the search is one fit per fold.
    assert tuning.fits == 5
    assert tuning.grid_ends == tuple(50 * 2**k for k in range(len(tuning.grid_ends)))
    assert tuning.grid_ends[-1] == tuning.largest
    assert len(tuning.risk) == tuning.largest + 1
    assert tuning.mstop == int(np.argmin(tuning.risk[1:]) + 1)
    with pytest.warns(RuntimeWarning, match="still falling at 50 steps"):
        once = mb.tune_mstop("y ~ bbs(x)", data, folds=5, start=50, extensions=0)
    assert not once.extended and once.at_boundary


# ─── missing values ─────────────────────────────────────────────────────


def test_autospec_probes_complete_rows_and_its_fit_drops_the_rest():
    rng = np.random.default_rng(13)
    n = 500
    x, z = rng.normal(size=(2, n))
    y = np.sin(2 * x) + 0.8 * z + rng.normal(scale=0.5, size=n)
    z = np.where(rng.uniform(size=n) < 0.1, np.nan, z)
    data = {"x": x, "z": z, "y": y}
    rec = mb.autospec(data, "y", budget=60, mstop=80)
    assert rec.incomplete_rows == int(np.isnan(z).sum())
    assert "incomplete rows left out" in rec.report()
    assert rec.measured
    fit = rec.fit(data)
    assert len(np.asarray(fit.fitted_)) == n - rec.incomplete_rows


def test_na_action_omit_applies_when_learners_are_supplied():
    from mboostrs.formula import parse_gamboost_formula

    rng = np.random.default_rng(14)
    n = 200
    x, z = rng.normal(size=(2, n))
    y = x + z + rng.normal(size=n)
    z = np.where(rng.uniform(size=n) < 0.1, np.nan, z)
    data = {"x": x, "z": z, "y": y}
    formula = "y ~ bbs(x, df=4.0) + bbs(z, df=4.0)"
    _, learners = parse_gamboost_formula(formula)
    with_learners = mb.gamboost(formula, data=data, learners=learners, na_action="omit",
                                control=mb.boost_control(20, 0.1))
    parsed = mb.gamboost(formula, data=data, na_action="omit",
                         control=mb.boost_control(20, 0.1))
    np.testing.assert_allclose(with_learners.fitted_, parsed.fitted_, rtol=0, atol=1e-10)


def test_na_action_omit_refuses_to_misalign_a_supplied_buser_design():
    rng = np.random.default_rng(15)
    n = 120
    x = rng.normal(size=n)
    z = np.where(rng.uniform(size=n) < 0.1, np.nan, rng.normal(size=n))
    X = np.column_stack([np.ones(n), x])
    K = np.diag([0.0, 1.0])
    learners = [mb.Buser("x", X=X, K=K), mb.Bbs("z")]
    with pytest.raises(ValueError, match="Buser learner carries a design matrix"):
        mb.gamboost("y ~ buser(x) + bbs(z)", data={"x": x, "z": z, "y": x + rng.normal(size=n)},
                    learners=learners, na_action="omit", control=mb.boost_control(10, 0.1))


def test_a_group_level_measurement_is_not_a_second_encoding_of_the_group():
    """One income value per region maps one-to-one onto the region code, but
    income is a measurement a new region can be predicted from."""
    rng = np.random.default_rng(16)
    n = 700
    codes = rng.integers(0, 35, n)
    income = rng.normal(size=35)
    data = {"region": codes.astype(float), "income": income[codes],
            "x": rng.normal(size=n)}
    data["y"] = data["x"] + data["income"] + rng.normal(size=n)
    rec = mb.autospec(data, "y")
    assert not [h for h in rec.hygiene if h.kind == "duplicate-encoding"]
    assert "income" in rec.formula


def test_grouped_folds_accepts_python_string_labels():
    """A polars string column's `to_numpy()` is an object array of `str`."""
    labels = np.array(["AL", "AK", "AZ", "AR", "CA", "CO"] * 20, dtype=object)
    scheme = mb.grouped_folds(labels, B=3)
    matrix = np.asarray(scheme)
    assert matrix.shape == (120, 3)
    for fold in range(3):
        held = set(labels[matrix[:, fold] == 0])
        trained = set(labels[matrix[:, fold] > 0])
        assert held and not held & trained


def test_priority_names_response_siblings_first_and_summarises_missing_rows():
    rng = np.random.default_rng(17)
    n = 400
    life = rng.normal(80, 3, size=n)
    heavy = np.where(rng.uniform(size=n) < 0.2, np.nan, rng.normal(size=n))
    light = np.where(rng.uniform(size=n) < 0.01, np.nan, rng.normal(size=n))
    data = {"le_q1_M": life, "le_q1_F": life + rng.normal(scale=2, size=n),
            "heavy": heavy, "light": light}
    rec = mb.autospec(data, "le_q1_M")
    assert rec.priority[0].startswith("data: le_q1_F is named like the response")
    incomplete = [line for line in rec.priority if "rows (" in line and "left out" in line]
    assert len(incomplete) == 1
    assert "heavy" in incomplete[0] and "light" not in incomplete[0]
    assert not [line for line in rec.priority if line.startswith("data: light")]


# ─── a positive response that varies little relative to its level ──────


def _narrow_positive_frame(seed=18, n=500):
    rng = np.random.default_rng(seed)
    x, z = rng.normal(size=(2, n))
    return {"x": x, "z": z, "y": 76.0 + 1.0 * x + 0.5 * np.sin(z) + rng.normal(scale=0.8, size=n)}


def test_a_narrow_positive_response_is_gaussian_without_a_budget():
    rec = mb.autospec(_narrow_positive_frame(), "y")
    assert type(rec.family).__name__ == "Gaussian"


def test_the_family_probe_does_not_move_a_narrow_positive_response_to_gamma():
    rec = mb.autospec(_narrow_positive_frame(), "y", budget=40, mstop=60,
                      family=None)
    assert type(rec.family).__name__ == "Gaussian"
    evidence = rec.evidence.get("family")
    if evidence is not None:
        assert "of its mean" in evidence.sentence


def test_tune_mstop_says_when_every_step_makes_held_out_risk_worse():
    """A Gamma fitted to a response that varies by 1% of its level has a shape
    in the thousands, and boosting it with nu=0.1 diverges from the first step."""
    tuning = mb.tune_mstop("y ~ bbs(x) + bbs(z)", _narrow_positive_frame(),
                           family=mb.GammaReg(), folds=5, start=100)
    assert tuning.mstop == 1 + np.argmin(np.asarray(tuning.risk)[1:])
    assert tuning.risk[0] < tuning.risk[tuning.mstop]
    assert "offset-only" in tuning.sentence
    assert not tuning.extended


def test_a_comparison_row_carries_the_paired_standard_error_of_its_change():
    rng = np.random.default_rng(20)
    n = 400
    x, noise = rng.normal(size=(2, n))
    data = {"x": x, "noise": noise, "y": np.sin(2 * x) + rng.normal(scale=0.5, size=n)}
    rows = mb.compare_specifications({"line": "bols(x)", "curve": "bbs(x)",
                                      "curve and noise": "bbs(x) + bbs(noise)"},
                                     data, "y", folds=5, mstop=100)
    assert rows[0]["change se"] == 0.0
    for row in rows[1:]:
        assert np.isfinite(row["change se"]) and row["change se"] > 0
    curve = rows[1]
    assert -curve["change"] > 2 * curve["change se"]


def test_gamma_is_kept_only_if_it_predicts_as_well_as_gaussian():
    """Body fat's spread grows with its level, so the variance points at
    GammaReg, but boosted GammaReg predicts it far worse than Gaussian."""
    bodyfat = mb.datasets.bodyfat(output="polars")
    rec = mb.autospec(bodyfat, "DEXfat", budget=150, mstop=200)
    evidence = rec.evidence["family"]
    assert type(rec.family).__name__ == "Gaussian"
    assert np.isfinite(evidence.gamma_mse) and np.isfinite(evidence.gaussian_mse)
    assert evidence.gamma_mse > evidence.gaussian_mse
    assert "predicts worse" in evidence.sentence


def test_the_spread_check_reads_the_coordinates_of_a_spatial_term():
    """A spatial learner's name is "lon,lat", which is not a column."""
    import polars as pl

    rng = np.random.default_rng(21)
    n = 300
    lon, lat, x = rng.uniform(-2, 2, size=(3, n))
    frame = pl.DataFrame({"lon": lon, "lat": lat, "x": x,
                          "y": np.sin(lon) * np.cos(lat) + x
                               + rng.normal(size=n) * np.exp(0.4 * lat)})
    model = mb.gamboost("y ~ bbs(x) + bspatial(lon, lat, df=6)", data=frame,
                        control=mb.boost_control(100, 0.1))
    spread = mb.heteroscedasticity(model)
    assert {"lon", "lat", "x"} <= set(spread["by_column"])
    assert spread["strongest"] == "lat"


def test_a_term_whose_gain_is_within_noise_is_suggested_for_removal():
    rng = np.random.default_rng(22)
    n = 400
    data = {"x": rng.normal(size=n), "noise": rng.normal(size=n)}
    data["y"] = np.sin(2 * data["x"]) + rng.normal(scale=0.5, size=n)
    report = mb.term_report("y ~ bbs(x) + bbs(noise)", data, "y", folds=5, mstop=150,
                            shapes=False)
    noise_row = next(row for row in report.contributions if row["term"] == "bbs(noise)")
    drops = {s.term: s for s in mb.suggest_specifications(report) if s.action == "drop"}
    if noise_row["earns"] > 0:
        assert noise_row["earns"] < 2 * noise_row["se"]
        assert "cannot be told from" in drops["bbs(noise)"].reason
    assert "bbs(noise)" in drops
    assert "bbs(x)" not in drops


# ─── models with an interaction term, sliced and tuned ─────────────────


def _interaction_frame(seed=23, n=300):
    rng = np.random.default_rng(seed)
    x, z, w = rng.normal(size=(3, n))
    return {"x": x, "z": z, "w": w, "y": x * z + w + rng.normal(scale=0.5, size=n)}


def test_a_sliced_interaction_model_predicts_on_new_data():
    data = _interaction_frame()
    fit = mb.gamboost("y ~ bbs(w) + bbs(x, df=2.5, knots=5) %X% bbs(z, df=2.5, knots=5)", data=data,
                      control=mb.boost_control(100, 0.1))
    sliced = fit[:50]
    np.testing.assert_allclose(np.asarray(sliced.predict(newdata=data)),
                               np.asarray(fit.predict(newdata=data, mstop=50)),
                               rtol=0, atol=1e-10)


def test_tuning_inside_folds_works_with_an_interaction_term():
    data = _interaction_frame()
    score = mb.cv_risk("bbs(w) + bbs(x, df=2.5, knots=5) %X% bbs(z, df=2.5, knots=5)", data, "y",
                       folds=3, mstop="cv")
    assert np.isfinite(score)


def test_stability_selection_counts_selections_for_a_model_with_an_interaction():
    """Fold refits of a `%X%` model used to fail on every fold, reporting zero."""
    rng = np.random.default_rng(1)
    n = 400
    cols = {f"v{i}": rng.normal(size=n) for i in range(4)}
    cols["y"] = 2 * cols["v0"] + cols["v1"] * cols["v2"] + rng.normal(size=n)
    rhs = "bbs(v0) + bbs(v1) + bbs(v2) + bbs(v3) + bbs(v1, df=2.5, knots=5) %X% bbs(v2, df=2.5, knots=5)"
    fit = mb.gamboost(f"y ~ {rhs}", data=cols, control=mb.boost_control(150, 0.1))
    with _no_refit_failure():
        stab = mb.stabsel(fit, q=3, PFER=1.0, B=10, n_jobs=1)
    frequencies = dict(zip(fit._learner_names, np.asarray(stab.max_), strict=True))
    assert frequencies["v0"] > 0.9
    assert frequencies["bbs(v1, df=2.5, knots=5) %X% bbs(v2, df=2.5, knots=5)"] > 0.5


class _no_refit_failure:
    def __enter__(self):
        import warnings as _w
        self._ctx = _w.catch_warnings(record=True)
        self._caught = self._ctx.__enter__()
        _w.simplefilter("always")
        return self

    def __exit__(self, *exc):
        self._ctx.__exit__(*exc)
        failed = [w for w in self._caught if "refit failed" in str(w.message)]
        assert not failed, str(failed[0].message)
        return False


def test_stability_selection_refuses_when_every_refit_fails(monkeypatch):
    from mboostrs import _refit

    rng = np.random.default_rng(2)
    n = 200
    data = {name: rng.normal(size=n) for name in ("a", "b", "c", "d")}
    data["y"] = data["a"] + rng.normal(size=n)
    fit = mb.gamboost("y ~ bbs(a) + bbs(b) + bbs(c) + bbs(d)", data=data,
                      control=mb.boost_control(50, 0.1))

    def broken_build(self, **overrides):
        raise ValueError("synthetic refit failure")

    monkeypatch.setattr(_refit.RefitSpec, "build", broken_build)
    with pytest.raises(ValueError, match="every one of the .* refits failed"):
        with pytest.warns(UserWarning, match="refit failed"):
            mb.stabsel(fit, q=2, PFER=1.0, B=4, n_jobs=1)


def _labelled_interaction_frame(seed=29, n=300):
    data = _interaction_frame(seed=seed, n=n)
    data["g"] = np.random.default_rng(seed).choice(list("abcdef"), n)
    return data


_LABELLED = "y ~ brandom(g) + bbs(w) + bbs(x, df=2.5, knots=5) %X% bbs(z, df=2.5, knots=5)"


def test_partial_effects_on_new_rows_sum_to_the_prediction():
    data = _labelled_interaction_frame()
    fit = mb.gamboost(_LABELLED, data=data, control=mb.boost_control(100, 0.1))
    new = {name: values[:20] for name, values in data.items()}
    effects = fit.partial_effects(newdata=new)
    total = fit.offset_ + sum(np.asarray(effect) for _, effect in effects.values())
    np.testing.assert_allclose(total, np.asarray(fit.predict(newdata=new)).ravel(),
                               rtol=0, atol=1e-10)


def test_partial_effects_on_a_grid_work_beside_an_interaction_term():
    data = _labelled_interaction_frame()
    fit = mb.gamboost(_LABELLED, data=data, control=mb.boost_control(100, 0.1))
    grid, effect = fit.partial_effects(which="w", n_points=40)["w"]
    rows = {"w": grid, "x": np.full(40, data["x"].mean()),
            "z": np.full(40, data["z"].mean()), "g": np.full(40, "a")}
    _, expected = fit.partial_effects(which="w", newdata=rows)["w"]
    np.testing.assert_allclose(effect, expected, rtol=0, atol=1e-10)
    assert mb.plot.plot_partial_effects(fit) is not None


def test_predict_over_mstop_encodes_labels_on_new_rows():
    data = _labelled_interaction_frame()
    fit = mb.gamboost("y ~ brandom(g) + bbs(w)", data=data, control=mb.boost_control(100, 0.1))
    new = {name: values[:20] for name, values in data.items()}
    grid = fit.predict_over_mstop([30, 100], newdata=new, output="numpy")
    np.testing.assert_allclose(grid[0], np.asarray(fit.predict(newdata=new, mstop=30)).ravel(),
                               rtol=0, atol=1e-10)


def test_repeated_folds_compute_the_paired_standard_error_within_each_partition():
    rng = np.random.default_rng(31)
    n = 240
    data = {"x": rng.normal(size=n), "z": rng.normal(size=n)}
    data["y"] = np.sin(data["x"]) + 0.2 * data["z"] + rng.normal(size=n)
    folds = mb.repeated_kfold_folds(n, k=5, repeats=4, seed=3)
    assert folds.repeats == 4
    rows = mb.compare_specifications({"with z": "bbs(x) + bbs(z)", "without z": "bbs(x)"},
                                     data, "y", folds=folds, mstop=80)
    scores = [np.asarray(mb.evaluate_specification(rhs, data, "y", folds=folds,
                                                   mstop=80).fold_scores)
              for rhs in ("bbs(x) + bbs(z)", "bbs(x)")]
    difference = (scores[1] - scores[0]).reshape(4, 5)
    within = np.sqrt(np.mean(np.var(difference, axis=1, ddof=1)) / 5)
    over_every_column = np.std(difference.ravel(), ddof=1) / np.sqrt(20)
    np.testing.assert_allclose(rows[1]["change se"], within, rtol=1e-10)
    assert rows[1]["change se"] > over_every_column


def test_grouped_folds_deal_the_groups_several_times():
    groups = np.random.default_rng(37).integers(0, 30, 600)
    once = mb.grouped_folds(groups, B=5, seed=4)
    several = mb.grouped_folds(groups, B=5, seed=4, repeats=3)
    assert once.repeats == 1
    assert several.repeats == 3
    assert several.shape == (600, 15)
    np.testing.assert_array_equal(np.asarray(several)[:, :5], np.asarray(once))
    held = several.held_out.reshape(600, 3, 5)
    assert np.all(held.sum(axis=2) == 1)
    for deal in range(3):
        fold = held[:, deal].argmax(axis=1)
        assert all(len(set(fold[groups == g])) == 1 for g in np.unique(groups))
    assert not np.array_equal(held[:, 0], held[:, 1])


def test_a_group_term_is_offered_for_removal_only_when_the_folds_say_it_hurts():
    from mboostrs.explore import TermReport

    report = TermReport(formula="y ~ bbs(x) + brandom(g)", response="y", baseline=1.0)
    report.contributions = [
        {"term": "bbs(x)", "variable": "x", "without it": 1.3, "earns": 0.3,
         "se": 0.05, "measurable": True},
        {"term": "brandom(g)", "variable": "g", "without it": 0.8, "earns": -0.2,
         "se": 0.04, "measurable": False},
    ]
    dropped = [s for s in mb.suggest_specifications(report) if s.action == "drop"]
    assert [s.term for s in dropped] == ["brandom(g)"]
    assert dropped[0].specification == "bbs(x)"
    assert "predicts held-out groups better without it" in dropped[0].reason

    report.contributions[1].update({"without it": 1.0, "earns": 0.0})
    assert not [s for s in mb.suggest_specifications(report) if s.term == "brandom(g)"]


# ─── cumulative fit accounting across the search (W03) ──────────────────
#
# `autospec(budget=)` caps only its probes; the term reports, comparisons,
# per-round refits and the stopping sweep all spend outside it. The library
# carries the count on every piece of evidence the loop reads —
# `TermReport.fits` joins `MstopTuning.fits`, the comparison rows' fit
# counts and `SpecRecommendation.fits` — and the documented loop totals them
# in its transcript, so the search's whole cost is stated rather than
# implied by the one budget argument.

def test_term_report_counts_every_fit_it_spends():
    rng = np.random.default_rng(3)
    n = 120
    data = {"y": np.sin(rng.uniform(-2, 2, n)) + rng.normal(size=n),
            "x": rng.uniform(-2, 2, n), "z": rng.uniform(-2, 2, n)}
    folds, mstop = 4, 80
    report = mb.term_report("y ~ bbs(x, df=4) + bols(z)", data, "y",
                            family=mb.Gaussian(), folds=folds, mstop=mstop,
                            shapes=True)
    # One fit per fold for: the baseline, each removal (2), and each shape
    # candidate of each swappable term (x and z: 3 shapes each).
    assert report.fits == folds * (1 + 2 + 3 + 3)
    assert report.fits > 0
    # Without the shape sweep, only the baseline and the removals.
    plain = mb.term_report("y ~ bbs(x, df=4) + bols(z)", data, "y",
                           family=mb.Gaussian(), folds=folds, mstop=mstop,
                           shapes=False)
    assert plain.fits == folds * (1 + 2)


def test_the_distributional_report_counts_its_fits_too():
    rng = np.random.default_rng(3)
    n = 120
    data = {"y": np.sin(rng.uniform(-2, 2, n)) + rng.normal(size=n),
            "x": rng.uniform(-2, 2, n)}
    report = mb.term_report({"mu": "bbs(x, df=4)", "sigma": "1"}, data, "y",
                            family=mb.GaussianLSS(), folds=3, mstop=80,
                            shapes=True)
    # Baseline, the one removal (mu's bbs becomes the constant 1; sigma's
    # intercept-only term is not a candidate), and mu's 3 shape candidates.
    assert report.fits == 3 * (1 + 1 + 3)


def test_the_documented_loop_states_its_whole_cost():
    from benches.bench_agent_search import BUDGET, agent_loop

    rng = np.random.default_rng(4)
    n = 200
    frame = {"y": 2.0 * rng.uniform(-1, 1, n) + rng.normal(scale=0.4, size=n),
             "x": rng.uniform(-1, 1, n), "z": rng.uniform(-1, 1, n),
             "w": rng.uniform(-1, 1, n)}
    _spec, _family, _mstop, _groups, transcript = agent_loop(frame)
    steps = transcript.steps
    logged = {entry["step"]: entry for entry in steps}
    # Every logged stage carries the fits spent so far, and each stage adds
    # to the running total: the autospec probes alone are within BUDGET,
    # and the search's whole cost is more than it.
    assert logged["autospec"]["fits"] <= BUDGET
    running = [entry["fits"] for entry in steps if "fits" in entry]
    assert running == sorted(running)
    assert running[-1] > logged["autospec"]["fits"]


def _labelled_frame(seed=0):
    rng = np.random.default_rng(seed)
    n = 100
    return {"x": rng.normal(size=n),
            "lab": np.where(rng.normal(size=n) > 0, "high", "low"),
            "lab3": np.array(["a", "b", "c"])[rng.integers(0, 3, n)]}


def test_a_text_response_is_refused_with_its_column_named():
    """The battery's LSS finding: `could not convert string to float` named
    neither the column, what it held, nor the family that encodes labels."""
    data = _labelled_frame()
    control = mb.boost_control(30, 0.1)
    refusals = [
        lambda: mb.gamboostLSS({"mu": "lab ~ bbs(x, df=3)", "sigma": "lab ~ 1"},
                               data=data, family=mb.GaussianLSS()),
        lambda: mb.gamboost("lab ~ bbs(x, df=3)", data=data,
                            family=mb.Binomial(), control=control),
        lambda: mb.glmboost("lab ~ bols(x)", data=data,
                            family=mb.Binomial(), control=control),
        lambda: mb.evaluate_specification("bbs(x, df=3)", data, "lab",
                                          family=mb.GaussianLSS(), mstop=30),
    ]
    for refuse in refusals:
        with pytest.raises(ValueError) as excinfo:
            refuse()
        message = str(excinfo.value)
        assert "'lab'" in message
        assert "'high'" in message and "'low'" in message
        assert "Multinomial" in message


def test_multinomial_still_encodes_text_class_labels():
    """The refusal names the one family that reads labels itself; it keeps
    reading them. Its predictions are the first K-1 class probabilities,
    with the last class as the reference."""
    data = _labelled_frame()
    fit = mb.gamboost("lab3 ~ bbs(x, df=3)", data=data, family=mb.Multinomial(),
                      control=mb.boost_control(30, 0.1))
    probabilities = np.asarray(fit.predict(type="response"))
    assert probabilities.shape == (len(data["x"]), 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


def test_cvrisk_judges_a_step_the_caller_chooses():
    """CF-14: comparing designs or specifications at a prespecified step
    restated the boundary warning the probe's full-path search had already
    resolved. `at_mstop=` truncates the judged curve to the caller's step
    and leaves the boundary question with the caller who answered it."""
    import warnings as _warnings

    bodyfat = mb.datasets.bodyfat()
    path = mb.gamboost("DEXfat ~ bbs(hipcirc)", data=bodyfat,
                       control=mb.boost_control(30, 0.1))
    folds = mb.cv_weights(71, type="kfold", B=5, seed=0)

    with _warnings.catch_warnings(record=True) as full_caught:
        _warnings.simplefilter("always")
        full = path.cvrisk(folds=folds)
    with _warnings.catch_warnings(record=True) as caught_judged:
        _warnings.simplefilter("always")
        judged = path.cvrisk(folds=folds, at_mstop=10)

    # The full path hits its end while falling; the judged step is the
    # caller's, so the boundary question it restated is not asked again.
    assert any("still falling" in str(w.message) for w in full_caught)
    assert not any("still falling" in str(w.message) for w in caught_judged)
    assert judged.grid[-1] == 10
    assert len(judged.aggregate) == 11
    np.testing.assert_allclose(np.asarray(full.aggregate)[:11],
                               judged.aggregate)
    assert 1 <= judged.optimal_mstop <= 10
    with pytest.raises(ValueError, match="not a step count"):
        path.cvrisk(folds=folds, at_mstop=999)


def test_a_grown_grid_equals_one_cross_validation_on_the_final_grid():
    from mboostrs.cvrisk import cv_weights

    rng = np.random.default_rng(4)
    n = 300
    x = rng.uniform(-3, 3, size=n)
    data = {"x": x, "y": np.sin(1.5 * x) * 3 + rng.normal(scale=0.3, size=n)}
    tuning = mb.tune_mstop("y ~ bbs(x)", data, folds=5, start=50, seed=3)
    model = mb.gamboost("y ~ bbs(x)", data=data, control=mb.boost_control(tuning.largest))
    full = model.cvrisk(folds=cv_weights(n, type="kfold", B=5, seed=3),
                        at_mstop=tuning.largest)
    np.testing.assert_allclose(tuning.risk, full.aggregate, rtol=0, atol=1e-12)


def test_patience_stops_after_the_minimum_with_the_full_grid_choice():
    rng = np.random.default_rng(4)
    n = 300
    x = rng.uniform(-3, 3, size=n)
    data = {"x": x, "y": np.sin(1.5 * x) * 3 + rng.normal(scale=0.3, size=n)}
    doubled = mb.tune_mstop("y ~ bbs(x)", data, folds=5, start=50, seed=3)
    patient = mb.tune_mstop("y ~ bbs(x)", data, folds=5, start=50, seed=3, patience=40)
    assert patient.mstop == doubled.mstop
    assert patient.largest == patient.mstop + 40 <= doubled.largest
    assert not patient.at_boundary
    with pytest.raises(NotImplementedError, match="patience"):
        mb.tune_mstop("y ~ bbs(x)", data, folds=5, score="mse", patience=40)
