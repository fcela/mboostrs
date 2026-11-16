"""Complementary-pairs sampling is built the same way on every stabsel path.

Under ``sampling_type="SS"`` a stability-selection run fits ``B`` half-samples
and each one's *exact complement*: column ``b`` and column ``B + b`` of the
fold matrix sum elementwise to the model's full weights. The Shah-Samworth
error bound is derived on those pairs, so a run that instead draws ``2B``
independent half-samples and reports ``B`` pairs to the bound is claiming an
error control it did not perform. That is what the LSS entry points used to
do, and these tests pin it closed.

``B`` counts initial subsamples -- pairs under ``"SS"`` -- and is the number
the bound is computed on. ``n_fits`` counts models actually fitted, so it is
``2 * B`` under ``"SS"`` and ``B`` under ``"MB"``. The tests below assert both
separately, because a single number cannot be right for both.
"""

import importlib

import numpy as np
import pytest
from mboostrs.stabsel import _append_complements, _fit_matrix, make_folds

import mboostrs as mb

# `mboostrs.stabsel` the module is shadowed by the re-exported function.
stabsel_module = importlib.import_module("mboostrs.stabsel")


def _data(n=200, seed=5):
    rng = np.random.default_rng(seed)
    x = {f"x{j}": rng.normal(size=n) for j in range(5)}
    x["y"] = 2.0 * x["x0"] - 1.5 * x["x1"] + rng.normal(size=n)
    return x


def _lss_data(n=200, seed=6):
    rng = np.random.default_rng(seed)
    x = {f"x{j}": rng.normal(size=n) for j in range(4)}
    x["y"] = rng.normal(1.0 + 0.8 * x["x0"], np.exp(-0.3 + 0.4 * x["x1"]))
    return x


# ─── the fold constructor ──────────────────────────────────────────


@pytest.mark.parametrize("n", [20, 21, 200])
def test_ss_folds_are_complementary_pairs(n):
    B = 4
    folds = make_folds(n, B, sampling_type="SS")
    assert folds.shape == (n, 2 * B)
    for b in range(B):
        np.testing.assert_allclose(folds[:, b] + folds[:, B + b], np.ones(n))


def test_ss_pairs_sum_to_the_model_weights_not_to_one():
    n = 40
    w = np.linspace(0.5, 3.0, n)
    B = 3
    folds = make_folds(n, B, sampling_type="SS", weights=w)
    for b in range(B):
        np.testing.assert_allclose(folds[:, b] + folds[:, B + b], w)


def test_mb_folds_are_independent_half_samples():
    n, B = 40, 5
    folds = make_folds(n, B, sampling_type="MB")
    assert folds.shape == (n, B)
    # Independent draws: no column pairs with another to cover every row.
    sums = {(i, j) for i in range(B) for j in range(i + 1, B)
            if np.allclose(folds[:, i] + folds[:, j], np.ones(n))}
    assert not sums


def test_ss_and_mb_share_the_first_b_columns():
    """SS is MB plus complements, from the same draw and the same seed."""
    n, B = 30, 4
    ss = make_folds(n, B, sampling_type="SS", seed=11)
    mb_ = make_folds(n, B, sampling_type="MB", seed=11)
    np.testing.assert_array_equal(ss[:, :B], mb_)


def test_make_folds_rejects_unknown_sampling_type():
    with pytest.raises(ValueError, match="sampling_type must be"):
        make_folds(10, 2, sampling_type="pairs")


def test_append_complements_checks_the_weight_length():
    with pytest.raises(ValueError, match="length 3 but folds has 10 rows"):
        _append_complements(np.zeros((10, 2)), weights=np.ones(3))


def test_supplied_folds_are_the_initial_subsamples():
    """Caller-supplied folds are B initial columns; SS appends complements.

    This is ``stabs:::check_folds``, which requires ``ncol(folds) == B`` and
    then does ``folds <- cbind(folds, weights - folds)``. Pairing is never
    inferred from an even column count, so an MB matrix that happens to have
    an even number of columns is not silently read as pairs.
    """
    n, B = 24, 3
    supplied = make_folds(n, B, sampling_type="MB", seed=3)
    fitted = _fit_matrix(supplied, "SS")
    assert fitted.shape == (n, 2 * B)
    np.testing.assert_array_equal(fitted[:, :B], supplied)
    for b in range(B):
        np.testing.assert_allclose(fitted[:, b] + fitted[:, B + b], np.ones(n))

    even = make_folds(n, 4, sampling_type="MB", seed=3)
    assert _fit_matrix(even, "MB").shape == (n, 4)
    assert _fit_matrix(even, "SS").shape == (n, 8)


# ─── the standard model path ───────────────────────────────────────


def test_standard_stabsel_reports_pairs_and_fits_separately():
    data = _data()
    model = mb.glmboost("y ~ " + " + ".join(f"x{j}" for j in range(5)),
                        data=data, control=mb.boost_control(80, 0.1))
    ss = mb.stabsel(model, q=2, PFER=1.0, B=5, verbose=False)
    assert (ss.B, ss.n_fits, ss.sampling_type) == (5, 10, "SS")

    mb_res = mb.stabsel(model, q=2, PFER=1.0, B=5, sampling_type="MB",
                        verbose=False)
    assert (mb_res.B, mb_res.n_fits, mb_res.sampling_type) == (5, 5, "MB")


def test_standard_stabsel_bound_uses_pairs_not_fits():
    """The PFER the run reports is the one computed on B, not on 2B."""
    from mboostrs.stabsel import _stabsel_parameters

    data = _data()
    model = mb.glmboost("y ~ " + " + ".join(f"x{j}" for j in range(5)),
                        data=data, control=mb.boost_control(80, 0.1))
    result = mb.stabsel(model, cutoff=0.75, q=2, B=5, verbose=False)
    on_pairs = _stabsel_parameters(p=result.p, cutoff=0.75, q=2, B=5,
                                   sampling_type="SS")
    on_fits = _stabsel_parameters(p=result.p, cutoff=0.75, q=2, B=10,
                                  sampling_type="SS")
    assert result.PFER == pytest.approx(on_pairs["PFER"])
    assert on_pairs["PFER"] != pytest.approx(on_fits["PFER"])


def test_standard_supplied_folds_gain_their_complements():
    data = _data(n=120)
    model = mb.glmboost("y ~ " + " + ".join(f"x{j}" for j in range(5)),
                        data=data, control=mb.boost_control(60, 0.1))
    folds = make_folds(120, 4, sampling_type="MB", seed=8)
    result = mb.stabsel(model, q=2, PFER=1.0, folds=folds, verbose=False)
    assert (result.B, result.n_fits) == (4, 8)


def test_ss_and_mb_agree_when_given_the_same_explicit_folds():
    """MB on the SS fit matrix reproduces SS exactly.

    SS = MB over the pair-completed matrix, with the bound computed on half
    the columns. If the two disagree on ``phat``, the fold construction and
    the fitting loop have drifted apart.
    """
    data = _data(n=150, seed=12)
    formula = "y ~ " + " + ".join(f"x{j}" for j in range(5))
    model = mb.glmboost(formula, data=data, control=mb.boost_control(60, 0.1))

    half = make_folds(150, 4, sampling_type="MB", seed=21)
    ss = mb.stabsel(model, q=2, PFER=1.0, folds=half, sampling_type="SS",
                    verbose=False)
    paired = _append_complements(half, np.ones(150))
    as_mb = mb.stabsel(model, q=2, PFER=1.0, folds=paired, sampling_type="MB",
                       verbose=False)

    np.testing.assert_allclose(ss.phat, as_mb.phat)
    assert ss.selected == as_mb.selected
    assert (ss.B, ss.n_fits) == (4, 8)
    assert (as_mb.B, as_mb.n_fits) == (8, 8)


def test_weighted_model_pairs_sum_to_its_weights():
    """A weighted fit's SS folds partition that model's weights, not units."""
    n = 120
    data = _data(n=n, seed=15)
    w = np.repeat([1.0, 2.0, 3.0], n // 3)
    model = mb.glmboost("y ~ " + " + ".join(f"x{j}" for j in range(5)),
                        data=data, weights=w, control=mb.boost_control(50, 0.1))

    seen = {}
    original = stabsel_module._selected_paths

    def _capture(model_, folds, n_fits, *args, **kwargs):
        seen["folds"] = folds
        return original(model_, folds, n_fits, *args, **kwargs)

    stabsel_module._selected_paths = _capture
    try:
        result = mb.stabsel(model, q=2, PFER=1.0, B=3, verbose=False)
    finally:
        stabsel_module._selected_paths = original

    folds = seen["folds"]
    assert folds.shape == (n, 6)
    for b in range(3):
        np.testing.assert_allclose(folds[:, b] + folds[:, 3 + b], w)
    assert (result.B, result.n_fits) == (3, 6)


# ─── the LSS paths ─────────────────────────────────────────────────


@pytest.mark.parametrize("kind", ["glmboostLSS", "gamboostLSS"])
def test_lss_ss_folds_are_complementary_pairs(kind, monkeypatch):
    """The LSS entry points construct pairs, not 2B independent draws.

    They used to call the fold generator for ``2 * B`` columns and hand
    ``B_eff // 2`` to the bound, so the pairs the bound assumed did not
    exist. This captures the matrix the engine is actually given.
    """
    from mboostrs import gamlss as gamlss_module

    data = _lss_data()
    n = len(data["y"])
    seen = {}
    entry = ("_stabsel_selected_glmboostlss" if kind == "glmboostLSS"
             else "_stabsel_selected_gamboostlss_mixed")
    original = getattr(gamlss_module, entry)

    def _capture(*args, **kwargs):
        # folds sits third from the end: (…, folds, q, stabilization);
        # nu_per_param arrives as a keyword after stabilization.
        seen["folds"] = np.asarray(args[-3])
        return original(*args, **kwargs)

    monkeypatch.setattr(gamlss_module, entry, _capture)

    if kind == "glmboostLSS":
        formula = "y ~ " + " + ".join(f"x{j}" for j in range(4))
        model = mb.glmboostLSS(formula, data=data, families=mb.GaussianLSS(),
                               control=mb.boost_control(40, 0.1),
                               method="noncyclic")
    else:
        formula = "y ~ " + " + ".join(f"bbs(x{j}, df=3)" for j in range(4))
        model = mb.gamboostLSS(formula, data=data, families=mb.GaussianLSS(),
                               control=mb.boost_control(40, 0.1),
                               method="noncyclic")

    result = model.stabsel(q=2, PFER=1.0, B=4)
    folds = seen["folds"]
    assert folds.shape == (n, 8)
    for b in range(4):
        np.testing.assert_allclose(folds[:, b] + folds[:, 4 + b], np.ones(n))
    assert (result.B, result.n_fits, result.sampling_type) == (4, 8, "SS")
    assert result["B"] == 4 and result["n_fits"] == 8


def test_lss_mb_fits_b_models_and_reports_b():
    data = _lss_data()
    formula = "y ~ " + " + ".join(f"x{j}" for j in range(4))
    model = mb.glmboostLSS(formula, data=data, families=mb.GaussianLSS(),
                           control=mb.boost_control(40, 0.1),
                           method="noncyclic")
    result = model.stabsel(q=2, PFER=1.0, B=4, sampling_type="MB",
                           assumption="none")
    assert (result.B, result.n_fits, result.sampling_type) == (4, 4, "MB")


def test_lss_ss_and_mb_agree_when_given_the_same_explicit_folds():
    data = _lss_data(seed=19)
    n = len(data["y"])
    formula = "y ~ " + " + ".join(f"x{j}" for j in range(4))
    model = mb.glmboostLSS(formula, data=data, families=mb.GaussianLSS(),
                           control=mb.boost_control(40, 0.1),
                           method="noncyclic")

    half = make_folds(n, 4, sampling_type="MB", seed=31)
    ss = model.stabsel(q=2, PFER=1.0, folds=half, sampling_type="SS")
    paired = _append_complements(half, np.ones(n))
    as_mb = model.stabsel(q=2, PFER=1.0, folds=paired, sampling_type="MB",
                          assumption="none")

    np.testing.assert_allclose(ss.phat, as_mb.phat)
    assert ss.selected == as_mb.selected
    assert (ss.B, ss.n_fits) == (4, 8)
    assert (as_mb.B, as_mb.n_fits) == (8, 8)


def test_lss_bound_uses_pairs_not_fits():
    from mboostrs.stabsel import _stabsel_parameters

    data = _lss_data(seed=23)
    formula = "y ~ " + " + ".join(f"x{j}" for j in range(4))
    model = mb.glmboostLSS(formula, data=data, families=mb.GaussianLSS(),
                           control=mb.boost_control(40, 0.1),
                           method="noncyclic")
    result = model.stabsel(cutoff=0.75, q=2, B=5)
    p = len(result.learner_names)
    on_pairs = _stabsel_parameters(p=p, cutoff=0.75, q=2, B=5,
                                   sampling_type="SS")
    on_fits = _stabsel_parameters(p=p, cutoff=0.75, q=2, B=10,
                                  sampling_type="SS")
    assert result.PFER == pytest.approx(on_pairs["PFER"])
    assert on_pairs["PFER"] != pytest.approx(on_fits["PFER"])


def test_grouped_menu_controls_learners_not_source_variables():
    """PFER bounds false *learners*, so one variable can enter twice.

    A menu such as ``bols(x) + bbs(x)`` gives one source variable two
    candidate learners. ``p`` counts learners, the bound is computed on that
    ``p``, and both learners appear separately in the result. A user reading
    PFER as a bound on false *variables* is reading it wrong: two learners on
    the same variable each count.
    """
    data = _data(n=200, seed=27)
    terms = [f"bols(x{j})" for j in range(3)] + [f"bbs(x{j}, df=3)" for j in range(3)]
    model = mb.gamboost("y ~ " + " + ".join(terms), data=data,
                        control=mb.boost_control(80, 0.1))
    result = mb.stabsel(model, q=2, PFER=1.0, B=4, verbose=False)

    assert result.p == 6
    # Selection is per learner: a variable's two learners have their own rows,
    # and the bound was computed on p = 6 candidates rather than 3 variables.
    assert result.phat.shape[0] == 6

    # The reported names are the source variables, so the two learners on x0
    # are indistinguishable in the result even though they are scored, bounded
    # and selected separately. Naming transformed and duplicated terms is the
    # plan's P0.6 item; this pins the current behaviour so a change to it is
    # a deliberate one.
    assert result.learner_names == ["x0", "x1", "x2", "x0", "x1", "x2"]
