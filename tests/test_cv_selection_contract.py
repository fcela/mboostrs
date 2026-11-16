"""Selection on a CV curve does not become prespecified when minima agree."""

import numpy as np
import pytest

import mboostrs as mb
from mboostrs.cvrisk import CvRiskResult


def curves():
    risk = np.array([[4., 2., 3.], [5., 1., 4.], [3., 2., 5.]])
    a = CvRiskResult(risk, risk.mean(0), np.array([1, 2, 3]), 2, "kfold",
                     fold_fingerprint="same")
    shifted = risk + np.array([.1, .2, .3])[:, None]
    b = CvRiskResult(shifted, shifted.mean(0), np.array([1, 2, 3]), 2, "kfold",
                     fold_fingerprint="same")
    return a, b


def test_matching_selected_minima_are_descriptive():
    a, b = curves()
    result = a.compare_to(b)
    assert result.mstop_a == result.mstop_b == 2
    assert not result.inferential
    assert result.interval is None
    np.testing.assert_allclose(result.differences, [-.1, -.2, -.3])
    assert "descriptive" in result.describe()


def test_explicit_fixed_point_retains_the_approximate_interval():
    a, b = curves()
    result = a.compare_to(b, mstop=2)
    assert result.inferential
    assert result.interval is not None
    assert "approximate" in result.describe()
    assert "chosen before" in result.describe()


def test_default_comparison_uses_the_reported_allowed_optimum():
    risk = np.array([[0., 2., 4.], [0., 3., 5.], [0., 4., 6.]])
    a = CvRiskResult(risk, risk.mean(0), np.array([0, 1, 2]), 1, "kfold",
                     fold_fingerprint="same")
    b = CvRiskResult(risk + .1, (risk + .1).mean(0), np.array([0, 1, 2]), 1, "kfold",
                     fold_fingerprint="same")
    result = a.compare_to(b)
    assert result.mstop_a == result.mstop_b == 1


@pytest.mark.parametrize("which", ["first", "second", "both"])
def test_unknown_fold_identity_cannot_establish_pairing(which):
    a, b = curves()
    if which in ("first", "both"):
        a.fold_fingerprint = ""
    if which in ("second", "both"):
        b.fold_fingerprint = ""
    with pytest.raises(ValueError, match="fold fingerprints"):
        a.compare_to(b, mstop=2)


@pytest.mark.parametrize("level", [0, 1, -.1, np.nan, np.inf, True])
def test_level_is_validated_in_descriptive_and_fixed_point_modes(level):
    a, b = curves()
    for mstop in (None, 2):
        with pytest.raises((TypeError, ValueError), match="level"):
            a.compare_to(b, mstop=mstop, level=level)


@pytest.mark.parametrize("kind", ["glm", "gam"])
def test_evaluation_masks_are_part_of_fold_identity(kind):
    rng = np.random.default_rng(27)
    data = {"x": rng.normal(size=80), "y": rng.normal(size=80)}
    if kind == "glm":
        model = mb.glmboost("y ~ x", data=data, control=mb.boost_control(10))
    else:
        model = mb.gamboost("y ~ bols(x)", data=data, control=mb.boost_control(10))
    matrix = mb.cv_weights(80, type="kfold", B=4, seed=3)
    held = matrix == 0
    judge = held.copy()
    judge[::2] = False
    a = model.cvrisk(folds=mb.FoldScheme(matrix, held, "kfold", judge=held))
    b = model.cvrisk(folds=mb.FoldScheme(matrix, held, "kfold", judge=judge))
    assert a.fold_fingerprint != b.fold_fingerprint
    with pytest.raises(ValueError, match="different folds"):
        a.compare_to(b, mstop=5)


def test_evaluation_weights_are_part_of_fold_identity():
    from mboostrs.cvrisk import fold_fingerprint

    matrix = np.array([[1., 0.], [0., 1.]])
    assert fold_fingerprint(matrix, weights=[1., 1.]) != fold_fingerprint(matrix, weights=[1., 2.])
