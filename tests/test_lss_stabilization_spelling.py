"""The LSS stabilization spelling: set on the family, overridable on the fit.

Upstream gamboostLSS 2.2-0 deprecated `options(gamboostLSS_stab_ngrad=TRUE)`
in favour of `GaussianLSS(stabilization = "MAD")` — family-level state that
threads into every parameter's partial gradient (see `check_stabilization`
in the installed package). Every mboostrs LSS family takes the same argument,
and the fitting functions keep a fit-level `stabilization=` that overrides
the family's value when passed explicitly, defaulting to `None` ("use the
family's") so a family setting is never silently ignored.
"""

import warnings

import numpy as np
import pytest

import mboostrs as mb


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(3)
    n = 120
    x = rng.normal(size=n)
    y = rng.normal(1 + 0.8 * x, np.exp(0.5 * x))
    return {"y": y, "x": x}


@pytest.fixture(scope="module")
def control():
    return mb.boost_control(40, 0.1)


def test_family_level_equals_fit_level(data, control):
    # The two spellings of the same setting must produce the same fit.
    from_family = mb.glmboostLSS(
        "y ~ bols(x)", data, families=mb.GaussianLSS(stabilization="MAD"),
        control=control)
    from_fit = mb.glmboostLSS(
        "y ~ bols(x)", data, control=control, stabilization="MAD")
    assert np.allclose(from_family.risk, from_fit.risk)
    for name in from_family.param_names:
        assert np.allclose(from_family.fitted()[name], from_fit.fitted()[name])


def test_fit_level_overrides_the_family(data, control):
    plain = mb.glmboostLSS("y ~ bols(x)", data, control=control)
    overridden = mb.glmboostLSS(
        "y ~ bols(x)", data, families=mb.GaussianLSS(stabilization="MAD"),
        control=control, stabilization="none")
    assert np.allclose(plain.risk, overridden.risk)


def test_invalid_value_refused_at_both_levels(data, control):
    with pytest.raises(ValueError, match='stabilization must be "none"'):
        mb.GaussianLSS(stabilization="bogus")
    with pytest.raises(ValueError, match='stabilization must be "none"'):
        mb.glmboostLSS("y ~ bols(x)", data, control=control,
                       stabilization="bogus")


def test_casing_is_normalized():
    # "mad"/"l2" are the spellings R's deprecation message shows; the Rust
    # parser has always accepted them, so the Python surface does too.
    assert mb.GaussianLSS(stabilization="mad").stabilization == "MAD"
    assert mb.GaussianLSS(stabilization="l2").stabilization == "L2"
    assert mb.GaussianLSS().stabilization == "none"


def test_every_lss_family_accepts_the_argument():
    families = [mb.GaussianLSS, mb.GaussianKnownVarLSS, mb.StudentTLSS,
                mb.BetaLSS, mb.NBinomialLSS, mb.ZINBLSS, mb.PoissonLSS,
                mb.BinomialLSS, mb.Pareto2LSS]
    for cls in families:
        assert cls(stabilization="L2").stabilization == "L2"
    assert mb.DirichletLSS(K=3, stabilization="L2").stabilization == "L2"
    with pytest.raises(ValueError):
        mb.DirichletLSS(K=3, stabilization="bogus")


def test_setting_travels_through_cvrisk_and_stabsel(data, control):
    plain = mb.glmboostLSS("y ~ bols(x)", data, control=control)
    mad = mb.glmboostLSS("y ~ bols(x)", data,
                         families=mb.GaussianLSS(stabilization="MAD"),
                         control=control)
    folds = mb.cv_weights(120, B=3, type="kfold", seed=1)
    with warnings.catch_warnings():
        # mstop=40 exhausts the CV grid; the boundary warning is off-topic here.
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        risk_plain = np.asarray(plain.cvrisk(folds=folds).risk_matrix)
        risk_mad = np.asarray(mad.cvrisk(folds=folds).risk_matrix)
    assert not np.allclose(risk_plain, risk_mad)
    hat_plain = plain.stabsel(cutoff=0.8, q=1, PFER=1.5, B=4).phat
    hat_mad = mad.stabsel(cutoff=0.8, q=1, PFER=1.5, B=4).phat
    assert not np.allclose(hat_plain, hat_mad)


def test_gamboostlss_family_level_spelling(data, control):
    from_family = mb.gamboostLSS(
        "y ~ bbs(x, df=3)", data, control=control,
        families=mb.GaussianLSS(stabilization="L2"))
    from_fit = mb.gamboostLSS("y ~ bbs(x, df=3)", data, control=control,
                              stabilization="L2")
    assert np.allclose(from_family.risk, from_fit.risk)


def test_fit_level_stabilization_survives_cvrisk_and_stabsel_refit(data, control):
    """A fit-level `stabilization=` override must reach a refit.

    The family is constructed with the default (`none`) and the fit overrides
    it, so a `cvrisk`/`stabsel` that re-resolved the *family* only would drop
    the override. The stored model must carry the resolved value and pass it
    through `_refit_arguments`.
    """
    overridden = mb.glmboostLSS(
        "y ~ bols(x)", data, families=mb.GaussianLSS(),
        control=control, stabilization="MAD")
    family_default = mb.glmboostLSS(
        "y ~ bols(x)", data, families=mb.GaussianLSS(), control=control)
    folds = mb.cv_weights(120, B=3, type="kfold", seed=1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        risk_override = np.asarray(overridden.cvrisk(folds=folds).risk_matrix)
        risk_default = np.asarray(family_default.cvrisk(folds=folds).risk_matrix)
    # The override changes the per-fold refits, so the risk matrices differ;
    # had the refit dropped stabilization they would equal the family-default.
    assert not np.allclose(risk_override, risk_default)
    hat_override = overridden.stabsel(cutoff=0.8, q=1, PFER=1.5, B=4).phat
    hat_default = family_default.stabsel(cutoff=0.8, q=1, PFER=1.5, B=4).phat
    assert not np.allclose(hat_override, hat_default)


def test_fit_level_stabilization_survives_slicing(data, control):
    """`model[:k]` refits, so it must refit with the same stabilization.

    A cyclic distributional path is not prefix-consistent, so slicing is a cold
    refit to `k` rather than a truncation. `update()` rebuilt that fit from the
    stored formula, control, weights and method but not from the resolved
    stabilization, so the result was exactly the `stabilization="none"` fit: on
    this fixture the sliced link predictions differed from a direct MAD fit at
    the same mstop by 0.52.
    """
    overridden = mb.glmboostLSS(
        "y ~ bols(x)", data, families=mb.GaussianLSS(),
        control=control, stabilization="MAD")
    truncated = mb.boost_control(20, control.nu)
    direct = mb.glmboostLSS(
        "y ~ bols(x)", data, families=mb.GaussianLSS(),
        control=truncated, stabilization="MAD")
    plain = mb.glmboostLSS(
        "y ~ bols(x)", data, families=mb.GaussianLSS(), control=truncated)

    for model in (overridden[20], overridden.update(mstop=20)):
        for name in direct.param_names:
            assert np.allclose(model.fitted()[name], direct.fitted()[name]), name
        assert any(
            not np.allclose(model.fitted()[name], plain.fitted()[name])
            for name in direct.param_names
        ), "the refit matches the unstabilized fit, so the setting was dropped"
