"""scale_absorption: does an LSS scale effect survive a more flexible mean? (issue #112).

Two simulated truths decide the verdict:

- (a) heteroskedastic noise around a correct linear mean. The scale effect is
  real, so the verdict is "scale effect robust".
- (b) constant spread around sin(3x), with the mean fitted as ``bols(x)``. The
  scale model takes up the mean's bias, and a flexible mean removes the
  scale effect, so the verdict is "scale effect absorbs mean misspecification".

The seeds below are the first ones of the held-out range (1000 and up) in the
simulation reported in the pull request.
"""

import json

import numpy as np
import pytest

import mboostrs as mb
from mboostrs.absorption import (
    ABSORBED_SHARE,
    GAIN_T,
    ROBUST_SHARE,
    VERDICT_ABSORBS,
    VERDICT_INCONCLUSIVE,
    VERDICT_ROBUST,
    _decide,
    _flexible_mean_formula,
)

FORMULA = {"mu": "y ~ bols(x)", "sigma": "y ~ bbs(x)"}
CONTROL = mb.boost_control(100, 0.1)


def _heteroskedastic(seed, n=300):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    return {"x": x, "y": rng.normal(1 + 2 * x, np.exp(-0.5 + 1.2 * x))}


def _underfitted_mean(seed, n=300):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    return {"x": x, "y": rng.normal(np.sin(3 * x), 0.5)}


def _fit(data, formula=FORMULA, family=None):
    return mb.gamboostLSS(formula, data=data,
                          families=family if family is not None else mb.GaussianLSS(),
                          control=CONTROL)


def test_scale_absorption_is_exported():
    assert callable(mb.scale_absorption)
    assert mb.ScaleAbsorption.__name__ == "ScaleAbsorption"


@pytest.mark.parametrize("seed", [1000, 1001, 1002])
def test_heteroskedastic_truth_with_correct_mean_is_robust(seed):
    result = mb.scale_absorption(_fit(_heteroskedastic(seed)), folds=5, seed=seed)
    assert result.verdict == VERDICT_ROBUST
    assert result.absorbed["flexible_mean"] < ROBUST_SHARE


def test_underfitted_mean_with_constant_spread_absorbs():
    seed = 1000
    result = mb.scale_absorption(_fit(_underfitted_mean(seed)), folds=5, seed=seed)
    assert result.verdict == VERDICT_ABSORBS
    assert result.absorbed["flexible_mean"] >= ABSORBED_SHARE
    assert result.oof["flexible_mean"]["gain"] > 0
    assert result.oof["flexible_mean"]["t"] >= GAIN_T
    # The flexible mean fits the sine: its spread of log-sigma is below the given one.
    assert result.scale["flexible_mean"]["sd_log_sigma"] < result.scale["given"]["sd_log_sigma"]


def test_constant_spread_with_correct_mean_has_no_scale_effect_to_absorb():
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 1, 300)
    data = {"x": x, "y": rng.normal(1 + 2 * x, 0.5)}
    fit = _fit(data, formula={"mu": "y ~ bols(x)", "sigma": "y ~ bols(x)"})
    result = mb.scale_absorption(fit, folds=5, seed=0)
    assert result.scale["given"]["sd_log_sigma"] < 0.1
    assert result.verdict == VERDICT_INCONCLUSIVE
    assert any("no scale effect" in note for note in result.notes)


def test_result_to_dict_is_plain_and_serializable():
    result = mb.scale_absorption(_fit(_heteroskedastic(1000)), folds=5, seed=1000)
    payload = result.to_dict()
    text = json.dumps(payload)
    assert json.loads(text)["verdict"] == result.verdict
    assert payload["folds"] == 5 and payload["seed"] == 1000
    assert set(payload["scale"]) == {"given", "flexible_mean", "slower_scale"}
    assert len(payload["oof"]["given"]["fold_nll"]) == 5
    assert set(payload["absorbed"]) == {"flexible_mean", "slower_scale"}
    assert set(payload["residual_structure"]) == {"given", "flexible_mean", "slower_scale"}


def test_slower_scale_variant_uses_quarter_rate_on_sigma():
    result = mb.scale_absorption(_fit(_heteroskedastic(1000)), folds=5, seed=1000)
    nu = result.specifications["slower_scale"]["nu"]
    assert nu["sigma"] == pytest.approx(nu["mu"] / 4)
    assert nu["mu"] == pytest.approx(0.1)


def test_flexible_mean_formula_doubles_smooths_and_smooths_linear_terms():
    data = _heteroskedastic(1000)
    assert _flexible_mean_formula("y ~ bols(x)", data) == "y ~ bols(x) + bbs(x)"
    assert _flexible_mean_formula("y ~ bbs(x, df=4) + bols(x)", data) == \
        "y ~ bbs(x, df=8) + bols(x)"
    assert _flexible_mean_formula("y ~ bbs(x)", data) == "y ~ bbs(x, df=8)"
    assert _flexible_mean_formula("y ~ bspatial(x, z, df=6)",
                                  {"x": [1.0, 2.0], "z": [1.0, 2.0], "y": [1.0, 2.0]}) == \
        "y ~ bspatial(x, z, df=12)"


def test_flexible_mean_leaves_bmono_and_tree_terms_alone():
    data = _heteroskedastic(1000)
    assert _flexible_mean_formula("y ~ bmono(x, constraint='increasing')", data) == \
        "y ~ bmono(x, constraint='increasing')"


def test_user_flexible_formula_is_used_and_recorded():
    data = _underfitted_mean(1000)
    result = mb.scale_absorption(_fit(data), folds=5, seed=1000,
                                 flexible="y ~ bbs(x, df=10)")
    assert result.specifications["flexible_mean"]["mu"] == "y ~ bbs(x, df=10)"


def test_flexible_formula_with_another_response_is_refused():
    with pytest.raises(ValueError, match="response"):
        mb.scale_absorption(_fit(_underfitted_mean(1000)), flexible="z ~ bbs(x)")


def test_mean_with_nothing_to_flex_is_refused():
    rng = np.random.default_rng(0)
    data = {"g": np.array(["a", "b"] * 150), "y": rng.normal(size=300)}
    fit = mb.gamboostLSS({"mu": "y ~ bols(g)", "sigma": "y ~ bols(g)"}, data=data,
                         families=mb.GaussianLSS(), control=CONTROL)
    with pytest.raises(ValueError, match="flexible="):
        mb.scale_absorption(fit)


def test_non_location_scale_family_is_refused():
    rng = np.random.default_rng(0)
    data = {"x": rng.uniform(size=200), "y": rng.poisson(3.0, size=200).astype(float)}
    fit = mb.gamboostLSS("y ~ bols(x)", data=data, families=mb.PoissonLSS(),
                         control=CONTROL)
    with pytest.raises(NotImplementedError, match="PoissonLSS"):
        mb.scale_absorption(fit)


def test_glmboost_lss_fit_is_refused():
    data = _heteroskedastic(1000)
    fit = mb.glmboostLSS({"mu": "y ~ x", "sigma": "y ~ x"}, data=data,
                         families=mb.GaussianLSS(), control=CONTROL)
    with pytest.raises(NotImplementedError, match="gamboostLSS"):
        mb.scale_absorption(fit)


def test_non_cyclic_fit_is_refused():
    fit = mb.gamboostLSS(FORMULA, data=_heteroskedastic(1000), families=mb.GaussianLSS(),
                         control=CONTROL, method="noncyclic")
    with pytest.raises(NotImplementedError, match="cyclic"):
        mb.scale_absorption(fit)


@pytest.mark.parametrize(
    ("sd_given", "absorbed", "t", "verdict"),
    [
        (0.05, 0.9, 5.0, VERDICT_INCONCLUSIVE),   # no scale effect to absorb
        (0.30, 0.50, 3.0, VERDICT_ABSORBS),
        (0.30, ABSORBED_SHARE, GAIN_T, VERDICT_ABSORBS),
        (0.30, 0.50, 1.0, VERDICT_INCONCLUSIVE),  # out-of-fold gain too noisy
        (0.30, 0.00, -2.0, VERDICT_ROBUST),
        (0.30, ROBUST_SHARE - 0.01, 0.0, VERDICT_ROBUST),
        (0.30, 0.18, 4.0, VERDICT_INCONCLUSIVE),  # between the two shares
    ],
)
def test_verdict_rule(sd_given, absorbed, t, verdict):
    assert _decide(sd_given, absorbed, t)[0] == verdict


def test_polars_frame_is_accepted():
    pl = pytest.importorskip("polars")
    data = _underfitted_mean(1000)
    frame = pl.DataFrame({"x": data["x"], "y": data["y"]})
    fit = _fit(frame)
    result = mb.scale_absorption(fit, folds=5, seed=1000)
    dict_result = mb.scale_absorption(_fit(data), folds=5, seed=1000)
    assert result.verdict == dict_result.verdict
    assert result.absorbed["flexible_mean"] == pytest.approx(
        dict_result.absorbed["flexible_mean"], rel=1e-9)
