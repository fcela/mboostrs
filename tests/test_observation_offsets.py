"""Observation offsets: R's `offset=` for the standard frontends (W06).

`gamboost`/`glmboost` used to reject a per-observation offset ("must be a
single number applied to every observation"), so counts over unequal
exposure could not be fitted as a rate model: the plan's row — "subtracting
log exposure from a count response is not a valid workaround" — recorded
both the gap and why the workaround was wrong.

The offset is now one known contribution to the linear predictor per row
(`log(exposure)`), which boosting never re-estimates. R treats it the same
way: the vector is the frozen start, and the intercept column is fitted on
top of it. mboostrs reproduces R to ~1e-9 on glmboost (intercept learner)
and to ~1e-14 on gamboost (bbs carries its own intercept basis; `bols`
switches to the two-column block form, since the centered form borrows its
constant from the model offset and a vector offset is not one).
"""

import numpy as np
import pytest

import mboostrs as mb


def _exposure_frame(seed=2, n=300):
    """Poisson counts over unequal exposure: mu = exposure * exp(0.4 + 0.6x)."""
    rng = np.random.default_rng(seed)
    exposure = rng.uniform(0.5, 4.0, n)
    x = rng.uniform(-1, 1, n)
    y = rng.poisson(exposure * np.exp(0.4 + 0.6 * x))
    return {"y": y, "x": x, "exposure": exposure}


def test_the_offset_enters_with_coefficient_one_and_is_not_estimated():
    """Gaussian y = o + b*x, offset o: the fit starts at o and estimates only
    the rest, so eta - o is the model's own and recovers b and the constant."""
    rng = np.random.default_rng(9)
    n = 120
    o = rng.uniform(-3, 3, n)
    x = rng.uniform(-2, 2, n)
    y = o + 0.5 * x
    fit = mb.glmboost("y ~ x", {"y": y, "x": x}, family=mb.Gaussian(),
                      offset=o, control=mb.boost_control(500, 0.1))
    eta = np.asarray(fit.predict(type="link"))
    np.testing.assert_allclose(eta - o, 0.5 * x, atol=1e-6)


def test_unequal_exposure_rates_track_the_offset():
    """The fitted means divided by exposure are a smooth rate in x, not the
    raw counts: the offset made the model a rate model."""
    frame = _exposure_frame()
    fit = mb.glmboost("y ~ x", frame, family=mb.Poisson(),
                      offset=np.log(frame["exposure"]),
                      control=mb.boost_control(400, 0.1))
    mean = np.asarray(fit.predict(type="response"))
    rate = mean / frame["exposure"]
    # exp(0.4 + 0.6x): rising in x, all rows of the same x share one rate.
    low, high = rate[frame["x"] < -0.5], rate[frame["x"] > 0.5]
    assert high.mean() / low.mean() > np.exp(0.6)
    assert 0.2 < rate.mean() < 3.0


def test_cross_validation_runs_on_an_offset_model():
    frame = _exposure_frame(n=150)
    fit = mb.glmboost("y ~ x", frame, family=mb.Poisson(),
                      offset=np.log(frame["exposure"]),
                      control=mb.boost_control(200, 0.1))
    risk = fit.cvrisk(B=3, seed=1)
    assert 1 <= risk.optimal_mstop <= 200


def test_new_rows_need_their_own_offset():
    frame = _exposure_frame(n=150)
    fit = mb.glmboost("y ~ x", frame, family=mb.Poisson(),
                      offset=np.log(frame["exposure"]),
                      control=mb.boost_control(200, 0.1))
    # Same x, growing exposure: the link prediction grows by log(exposure),
    # the rows' known contributions entering with coefficient one.
    new = {"x": np.array([0.3, 0.3, 0.3]),
           "exposure": np.array([1.0, 2.0, 4.0])}
    with pytest.raises(ValueError, match="needs offset="):
        fit.predict(newdata=new)
    eta = np.asarray(fit.predict(newdata=new, type="link",
                                 offset=np.log(new["exposure"])))
    np.testing.assert_allclose(eta[1] - eta[0], np.log(2.0), rtol=0.05)
    np.testing.assert_allclose(eta[2] - eta[1], np.log(2.0), rtol=0.05)


def test_training_rows_reuse_the_fitted_vector():
    frame = _exposure_frame(n=150)
    fit = mb.glmboost("y ~ x", frame, family=mb.Poisson(),
                      offset=np.log(frame["exposure"]),
                      control=mb.boost_control(200, 0.1))
    np.testing.assert_allclose(
        np.asarray(fit.predict(type="link")), fit.fitted(), rtol=1e-12, atol=1e-12)


def test_zero_exposure_is_refused_with_the_remedy():
    frame = _exposure_frame(n=80)
    frame["exposure"][7] = 0.0
    with np.errstate(divide="ignore"):
        with pytest.raises(ValueError, match="log\\(0\\) of a zero-exposure row"):
            mb.glmboost("y ~ x", frame, family=mb.Poisson(),
                        offset=np.log(frame["exposure"]))


def test_wrong_length_and_nonfinite_offsets_are_refused():
    frame = _exposure_frame(n=80)
    short = np.log(frame["exposure"])[:-1]
    with pytest.raises(ValueError, match="80 rows"):
        mb.glmboost("y ~ x", frame, family=mb.Poisson(), offset=short)
    nan_offset = np.log(frame["exposure"])
    nan_offset[3] = np.nan
    with pytest.raises(ValueError, match="entry 3 is not"):
        mb.glmboost("y ~ x", frame, family=mb.Poisson(), offset=nan_offset)


def test_distributional_response_shapes_refuse_a_vector_offset():
    frame = _exposure_frame(n=80)
    labels = (frame["y"] % 3).astype(int)
    with pytest.raises(ValueError, match="not defined for Multinomial"):
        mb.gamboost("y ~ bols(x)", {"y": labels, "x": frame["x"]},
                    family=mb.Multinomial(n_classes=3),
                    offset=np.log(frame["exposure"]))
    from mboostrs.families import CoxPH

    time = np.abs(frame["y"]) + 1.0
    with pytest.raises(ValueError, match="survival family CoxPH"):
        mb.glmboost("Surv(time, event) ~ x",
                    {"time": time, "event": labels.astype(float),
                     "x": frame["x"]},
                    family=CoxPH(), offset=np.log(frame["exposure"]))


def test_a_constant_vector_collapses_to_the_scalar_it_means():
    from mboostrs.glmboost import _prepare_offset

    assert _prepare_offset(np.full(80, 1.5), 80, mb.Poisson()) == 1.5
    assert _prepare_offset(2.0, 80, mb.Poisson()) == 2.0
    assert _prepare_offset(None, 80, mb.Poisson()) is None


def test_gamboost_bols_carries_its_own_intercept_under_a_vector_offset():
    """The centered bols borrows its constant from the model offset; a vector
    offset is a different number each row, so the fit uses the block form and
    the constant is estimated on top — the R design."""
    frame = _exposure_frame(n=150)
    fit = mb.gamboost("y ~ bols(x)", frame, family=mb.Poisson(),
                      offset=np.log(frame["exposure"]),
                      control=mb.boost_control(200, 0.1))
    eta = np.asarray(fit.predict(type="link"))
    above = eta - np.log(frame["exposure"])
    assert np.ptp(above) > 0.05  # a fitted function of x, not a frozen constant
    assert abs(np.mean(above) - 0.4) < 0.15  # near the generating log-rate


def test_the_artifact_round_trips_a_vector_offset():
    frame = _exposure_frame(n=120)
    fit = mb.glmboost("y ~ x", frame, family=mb.Poisson(),
                      offset=np.log(frame["exposure"]),
                      control=mb.boost_control(100, 0.1))
    import tempfile
    import pathlib

    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "offset-model.json"
        mb.save_model(fit, path)
        loaded = mb.load_model(path)
        np.testing.assert_allclose(
            np.asarray(loaded.predict(type="link")), fit.fitted(),
            rtol=1e-12, atol=1e-12)


@pytest.mark.r_parity
def test_glmboost_offset_matches_r(tmp_path):
    """R parity: the frozen start plus the fitted intercept column reproduce
    R's glmboost(offset=) to better than 1e-8 on the link scale."""
    import polars as pl

    frame = _exposure_frame()
    csv = tmp_path / "offsets.csv"
    np.savetxt(csv, np.column_stack(
        [frame["y"], frame["x"], frame["exposure"]]),
        delimiter=",", header="y,x,exposure", comments="")
    import rpy2.robjects as ro

    ro.r(
        'suppressMessages(library(mboost));'
        'd <- read.csv("{}");'
        'fit <- glmboost(y ~ x, data = d, offset = log(d$exposure),'
        '                family = Poisson(),'
        '                control = boost_control(mstop = 400, nu = 0.1));'
        'write.csv(predict(fit), "{}", row.names = FALSE)'.format(
            str(csv), str(tmp_path / "r-eta.csv"))
    )
    r_eta = np.asarray(pl.read_csv(tmp_path / "r-eta.csv").to_series(0))
    fit = mb.glmboost("y ~ x", frame, family=mb.Poisson(),
                      offset=np.log(frame["exposure"]),
                      control=mb.boost_control(400, 0.1))
    eta = np.asarray(fit.predict(type="link"))
    assert np.abs(eta - r_eta).max() < 1e-8


@pytest.mark.r_parity
def test_gamboost_bbs_offset_matches_r(tmp_path):
    import polars as pl

    frame = _exposure_frame()
    csv = tmp_path / "offsets.csv"
    np.savetxt(csv, np.column_stack(
        [frame["y"], frame["x"], frame["exposure"]]),
        delimiter=",", header="y,x,exposure", comments="")
    import rpy2.robjects as ro

    ro.r(
        'suppressMessages(library(mboost));'
        'd <- read.csv("{}");'
        'fit <- gamboost(y ~ bbs(x, df = 4), data = d,'
        '                offset = log(d$exposure), family = Poisson(),'
        '                control = boost_control(mstop = 100, nu = 0.1));'
        'write.csv(predict(fit), "{}", row.names = FALSE)'.format(
            str(csv), str(tmp_path / "r-eta.csv"))
    )
    r_eta = np.asarray(pl.read_csv(tmp_path / "r-eta.csv").to_series(0))
    fit = mb.gamboost("y ~ bbs(x, df=4)", frame, family=mb.Poisson(),
                      offset=np.log(frame["exposure"]),
                      control=mb.boost_control(100, 0.1))
    eta = np.asarray(fit.predict(type="link"))
    assert np.abs(eta - r_eta).max() < 1e-8


# ------------------------------------------------------------------ resampling
def _cv_folds(n, seed=0):
    return mb.cv_weights(n, "kfold", 5, seed=seed)


def test_cvrisk_keeps_the_per_row_offset_in_every_fold():
    """Every fold's fit starts from the known offset, as the full fit did.
    The folds used to refit without it, so the held-out risk of a Gaussian
    fit with offset sd 5 was about 25 (the offset's variance) and tuned
    `mstop` against the wrong curve."""
    rng = np.random.default_rng(0)
    n = 400
    x = rng.uniform(-1, 1, n)
    o = 5 * rng.normal(size=n)
    y = o + np.sin(3 * x) + rng.normal(0, 0.5, n)
    folds = _cv_folds(n)
    gam = mb.gamboost("y ~ bbs(x)", {"y": y, "x": x}, offset=o,
                      control=mb.boost_control(200, 0.1))
    glm = mb.glmboost("y ~ x", {"y": y, "x": x}, offset=o)
    assert np.nanmin(gam.cvrisk(folds=folds).aggregate) < 0.3
    assert np.nanmin(glm.cvrisk(folds=folds).aggregate) < 0.5


def test_stability_selection_keeps_the_per_row_offset():
    """Stability selection on a rate model refits each subsample from its
    own rows' offsets; it used to fail on every fold (gamboost) or refit
    without the offset (glmboost)."""
    frame = _exposure_frame(seed=4, n=400)
    rng = np.random.default_rng(4)
    for k in range(4):
        frame[f"z{k}"] = rng.uniform(-1, 1, 400)
    offset = np.log(frame["exposure"])
    data = {k: v for k, v in frame.items() if k != "exposure"}
    gam = mb.gamboost("y ~ bbs(x) + " + " + ".join(f"bbs(z{k})" for k in range(4)),
                      data, family=mb.Poisson(), offset=offset)
    assert mb.stabsel(gam, PFER=1, cutoff=0.75, verbose=False).variables == ("x",)
    glm = mb.glmboost("y ~ x + " + " + ".join(f"z{k}" for k in range(4)),
                      data, family=mb.Poisson(), offset=offset)
    assert "x" in mb.stabsel(glm, PFER=1, cutoff=0.75, verbose=False).variables


@pytest.mark.r_parity
def test_cvrisk_with_an_offset_matches_r(tmp_path):
    """mboost's cvrisk refits through `update(weights=)`, which keeps the
    offset vector; the out-of-fold risk path agrees with R."""
    import polars as pl

    frame = _exposure_frame()
    n = len(frame["y"])
    folds = _cv_folds(n, seed=3)
    csv = tmp_path / "offset.csv"
    np.savetxt(csv, np.column_stack([frame["y"], frame["x"], frame["exposure"]]),
               delimiter=",", header="y,x,exposure", comments="")
    np.savetxt(tmp_path / "folds.csv", folds, delimiter=",")
    import rpy2.robjects as ro

    ro.r(
        'suppressMessages(library(mboost));'
        'd <- read.csv("{}"); f <- as.matrix(read.csv("{}", header = FALSE));'
        'fit <- gamboost(y ~ bbs(x, df = 4), data = d,'
        '                offset = log(d$exposure), family = Poisson(),'
        '                control = boost_control(mstop = 100, nu = 0.1));'
        'cv <- cvrisk(fit, folds = f, papply = lapply);'
        'write.csv(colMeans(cv), "{}", row.names = FALSE)'.format(
            str(csv), str(tmp_path / "folds.csv"), str(tmp_path / "r-cv.csv")))
    r_cv = np.asarray(pl.read_csv(tmp_path / "r-cv.csv").to_series(0))
    fit = mb.gamboost("y ~ bbs(x, df=4)", frame, family=mb.Poisson(),
                      offset=np.log(frame["exposure"]),
                      control=mb.boost_control(100, 0.1))
    ours = np.asarray(fit.cvrisk(folds=folds).aggregate)
    assert np.abs(ours - r_cv).max() < 1e-6
