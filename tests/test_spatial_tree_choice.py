"""A coordinate pair is compared as a surface against a tree, on shared folds (#338)."""

import numpy as np

import mboostrs as mb
from mboostrs._autospec import CoordinateScore, coordinate_beats
from mboostrs.insights import insights


def _location(kind, n=400, seed=0):
    """Response over two coordinates plus one noise column.

    ``patchy`` is piecewise constant over four rectangles, so the effect has
    sharp edges. ``smooth`` is a product of sines, a surface a P-spline fits.
    """
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, 1, (2, n))
    w = rng.normal(size=n)
    if kind == "patchy":
        signal = np.where(x < 0.5, np.where(y < 0.5, 2.0, -1.0),
                          np.where(y < 0.3, 0.0, 3.0))
    else:
        signal = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
    response = signal + 0.5 * w + rng.normal(scale=0.5, size=n)
    return {"resp": response, "x": x, "y": y, "w": w}


def test_patchy_location_effect_picks_the_tree():
    rec = mb.autospec(_location("patchy"), "resp", budget=200, seed=0)
    surface = rec.evidence["coordinate_surface"]
    assert surface["adopted"]
    assert surface["chosen"] == "btree3"
    assert "btree(x, y, maxdepth=3)" in rec.formula
    assert "bspatial(x" not in rec.formula
    assert "coordinate_surface" in rec.measured
    assert "coordinate pair x, y" in rec.report()
    assert "btree(x, y, maxdepth=3)" in rec.report()


def test_smooth_location_effect_keeps_the_surface():
    rec = mb.autospec(_location("smooth"), "resp", budget=200, seed=0)
    surface = rec.evidence["coordinate_surface"]
    assert surface["adopted"]
    assert surface["chosen"] == "bspatial"
    assert "bspatial(x, y, df=6.0)" in rec.formula
    assert "btree(x, y" not in rec.formula


def test_every_candidate_is_scored_on_the_same_folds():
    rec = mb.autospec(_location("patchy"), "resp", budget=200, seed=0)
    names = [c["name"] for c in rec.evidence["coordinate_surface"]["candidates"]]
    assert names == ["additive", "bspatial", "btree", "btree3", "both"]
    assert rec.evidence["coordinate_surface"]["pair"] == ("x", "y")


def test_joint_tree_false_keeps_an_additive_search():
    rec = mb.autospec(_location("patchy"), "resp", budget=200, seed=0, joint_tree=False)
    assert "coordinate_surface" not in rec.evidence
    assert "coordinate_surface" not in rec.measured
    assert "btree(x, y" not in rec.formula
    assert "bspatial(x" not in rec.formula


def test_declared_coordinates_keep_the_declared_surface():
    data = _location("patchy")
    rec = mb.autospec(data, "resp", budget=200, seed=0, coordinates=("x", "y"))
    surface = rec.evidence.get("coordinate_surface")
    assert surface is None or not surface["candidates"]
    assert "bspatial(x, y" in rec.formula


def test_suggest_terms_names_the_tree_for_a_coordinate_pair():
    data = _location("patchy")
    kinds = {k.name: k for k in mb.classify_columns(data, "resp")}
    assert kinds["x"].term == "bbs(x, df=4.0)"
    assert kinds["y"].term == "bbs(y, df=4.0)"
    assert "btree(x, y)" in kinds["x"].reason
    assert "bspatial(x, y)" in kinds["y"].reason
    assert "btree(x, y)" not in kinds["w"].reason


def test_suggest_terms_leaves_a_frame_without_a_pair_unchanged():
    data = _location("patchy")
    del data["y"]
    kinds = mb.classify_columns(data, "resp")
    assert all("btree(x" not in k.reason for k in kinds)


def test_insights_reports_a_tree_that_beats_the_fitted_surface():
    findings = insights(_location("patchy"), "resp", seed=0)
    coordinate = [f for f in findings if "beats bspatial(x, y" in f.finding]
    assert len(coordinate) == 1
    finding = coordinate[0]
    assert finding.basis == "measured"
    assert finding.evidence["columns"] == ["x", "y"]
    assert finding.evidence["paired_difference"] > finding.evidence["paired_se"] > 0


def test_insights_stays_silent_when_the_surface_is_right():
    findings = insights(_location("smooth"), "resp", seed=0)
    assert not [f for f in findings if "beats bspatial" in f.finding]


def test_the_shape_rule_needs_a_margin_or_two_paired_se():
    ones = (1.0, 1.0, 1.0, 1.0, 1.0)
    base = CoordinateScore("a", "", 1.0, ones)
    assert coordinate_beats(base, CoordinateScore("b", "", 1.0, ones))[0] is False

    # A 10% reduction passes the 2% margin.
    big = CoordinateScore("c", "", 0.9, (0.9,) * 5)
    beats, gain, _ = coordinate_beats(base, big)
    assert beats and gain > 0.09

    # A 1% reduction that varies across folds passes neither rule: the mean
    # is 1% below the base, and the paired difference is about 0.7 SE.
    diffs = np.array([0.05, -0.03, 0.04, -0.02, 0.01])
    noisy = CoordinateScore("d", "", 0.99, tuple(1.0 - diffs))
    beats, gain, z = coordinate_beats(base, noisy)
    assert not beats and gain < 0.02 and z < 2.0

    # A 1% reduction that holds in every fold passes the two-SE rule alone.
    steady_diffs = np.array([0.009, 0.011, 0.009, 0.011, 0.010])
    steady = CoordinateScore("e", "", 0.99, tuple(1.0 - steady_diffs))
    beats, gain, z = coordinate_beats(base, steady, margin=False)
    assert beats and gain < 0.02 and z >= 2.0
    assert coordinate_beats(base, steady)[0] is True
