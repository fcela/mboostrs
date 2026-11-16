"""Declared response semantics survive autospec -> rec.fit -> export (#127).

A response's meaning cannot always be read from its dtype: a rounded
measurement stored as whole numbers looks like counts, an exposure makes a
count a rate, and an ordered class coded as integers looks like a count. These
tests pin three things: the recommendation names the response and its
source, the declared family and offset reach the fitted model, and the shape
comparison is either made at a common level of convergence or declined.
"""

import warnings

import numpy as np
import polars as pl
import pytest

import mboostrs as mb
from mboostrs import boost_control, gamboost
from mboostrs._autospec import probe_shapes
from mboostrs._frame import read_column
from mboostrs.autospec import _term_for
from mboostrs.cvrisk import cv_weights
from mboostrs.families import Gaussian, NBinomial, PropOdds

_LABELS = ["low", "mid", "high", "top"]


def _rounded_frame(n=400, seed=0):
    """A continuous measurement rounded to whole units, stored as floats."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    z = rng.normal(size=n)
    y = np.round(50 + 20 * x + 3 * z + rng.normal(scale=2.0, size=n))
    return pl.DataFrame({"y": y, "x": x, "z": z})


def _exposure_frame(n=600, seed=1):
    """Counts whose mean is proportional to a per-row exposure."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    exposure = rng.uniform(0.5, 5.0, n)
    y = rng.poisson(exposure * np.exp(-1 + 2 * x))
    return pl.DataFrame({"y": y, "x": x, "expo": exposure})


def _ordered_frame(n=400, seed=2):
    """A latent continuous score cut into four ordered classes."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    latent = 2 * x + rng.logistic(size=n)
    grade = np.digitize(latent, np.quantile(latent, [0.25, 0.5, 0.75]))
    return grade, x


# ── what the recommendation says about the response ──────────────────


def test_whole_numbers_stored_as_floats_are_read_as_counts_with_a_warning():
    frame = _rounded_frame()
    with pytest.warns(UserWarning, match="stored as floating point"):
        rec = mb.autospec(frame, "y")

    assert type(rec.family).__name__ == "Poisson"
    response = rec.evidence["response"]
    assert response["kind"] == "counts"
    assert response["source"] == "inferred"
    assert "rounded measurements" in response["sentence"]
    assert "response: Poisson counts" in rec.report()


def test_declared_gaussian_is_a_rounded_measurement_and_survives_fit():
    frame = _rounded_frame()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rec = mb.autospec(frame, "y", family=Gaussian())
    assert not [w for w in caught if "stored as floating point" in str(w.message)]

    response = rec.evidence["response"]
    assert response["kind"] == "rounded continuous"
    assert response["source"] == "declared"
    assert "response: Gaussian (declared)" in rec.report()

    model = rec.fit(frame)
    assert type(model._family).__name__ == "Gaussian"
    assert model.summary(as_dict=True)["family"] == "Gaussian"


def test_ordered_enum_response_is_inferred_as_proportional_odds():
    grade, x = _ordered_frame()
    frame = pl.DataFrame({
        "y": pl.Series([_LABELS[g] for g in grade]).cast(pl.Enum(_LABELS)),
        "x": x,
    })
    rec = mb.autospec(frame, "y")

    assert isinstance(rec.family, PropOdds)
    assert rec.evidence["response"]["kind"] == "ordered classes"
    model = rec.fit(frame)
    probabilities = np.asarray(model.predict(type="response"))
    assert probabilities.shape == (frame.height, len(_LABELS))
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert model.summary(as_dict=True)["family"] == "PropOdds"


def test_unordered_labels_stay_multinomial():
    grade, x = _ordered_frame()
    frame = pl.DataFrame({
        "y": pl.Series([_LABELS[g] for g in grade]).cast(pl.Categorical),
        "x": x,
    })
    rec = mb.autospec(frame, "y")
    assert type(rec.family).__name__ == "Multinomial"
    assert rec.evidence["response"]["kind"] == "unordered classes"


def test_integer_ordered_classes_declared_as_proportional_odds_survive_fit():
    grade, x = _ordered_frame()
    frame = pl.DataFrame({"y": grade + 1, "x": x})
    rec = mb.autospec(frame, "y", family=PropOdds(n_levels=4))

    response = rec.evidence["response"]
    assert response["kind"] == "ordered classes"
    assert response["source"] == "declared"
    model = rec.fit(frame)
    assert model.summary(as_dict=True)["family"] == "PropOdds"
    assert np.asarray(model.predict(type="response")).shape == (frame.height, 4)


# ── exposure: a count per unit of exposure ───────────────────────────


def test_exposure_is_an_offset_and_not_a_predictor():
    frame = _exposure_frame()
    rec = mb.autospec(frame, "y", exposure="expo")

    assert rec.exposure == "expo"
    assert "expo" not in rec.formula
    assert type(rec.family).__name__ == "Poisson"
    assert rec.evidence["response"]["kind"] == "count per exposure"
    assert "family" not in rec.defaulted


def test_rec_fit_passes_log_exposure_as_the_offset():
    frame = _exposure_frame()
    rec = mb.autospec(frame, "y", exposure="expo")
    fitted = rec.fit(frame)

    control = rec.control or boost_control(rec.mstop)
    explicit = gamboost(rec.formula, data=frame, family=rec.family, control=control,
                        offset=np.log(frame["expo"].to_numpy()))
    assert np.allclose(np.asarray(fitted.predict(type="link")),
                       np.asarray(explicit.predict(type="link")))


def test_exposure_model_export_records_the_offset_per_observation():
    frame = _exposure_frame()
    model = mb.autospec(frame, "y", exposure="expo").fit(frame)

    record = model.to_dict()
    assert record["family"] == "Poisson"
    assert record["offset"]["per_observation"] == frame.height
    assert np.isfinite(record["offset"]["mean"])
    assert "per observation" in str(model.summary())


def test_exposure_is_refused_for_a_non_count_family():
    frame = _exposure_frame()
    with pytest.raises(ValueError, match="count families"):
        mb.autospec(frame, "y", exposure="expo", family=Gaussian())


def test_exposure_accepts_negative_binomial_as_the_count_family():
    frame = _exposure_frame()
    rec = mb.autospec(frame, "y", exposure="expo", family=NBinomial())
    assert type(rec.family).__name__ == "NBinomial"


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan])
def test_exposure_with_a_nonpositive_or_missing_value_is_an_error(bad):
    frame = _exposure_frame()
    exposure = frame["expo"].to_numpy().copy()
    exposure[3] = bad
    frame = frame.with_columns(pl.Series("expo", exposure))
    with pytest.raises(ValueError, match="expo.*missing or non-positive"):
        mb.autospec(frame, "y", exposure="expo")


def test_exposure_names_a_column_the_data_must_hold():
    frame = _exposure_frame()
    with pytest.raises(ValueError, match="not a column"):
        mb.autospec(frame, "y", exposure="area")


def test_exposure_probes_run_with_the_offset():
    frame = _exposure_frame(n=300)
    rec = mb.autospec(frame, "y", exposure="expo", budget=120)

    assert rec.exposure == "expo"
    assert type(rec.family).__name__ == "Poisson"
    assert "expo" not in rec.formula
    assert rec.fits > 0
    rec.fit(frame)


# ── shape comparison under non-convergence ───────────────────────────


def _sine_frame(n=300, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    y = np.sin(3 * x) + rng.normal(scale=0.3, size=n)
    return pl.DataFrame({"y": y, "x": x})


def _shape_probe(frame, *, mstop, nu, budget=400):
    column = read_column(frame, "x")
    terms = [_term_for(column, 4.0, "bbs")]
    folds = cv_weights(frame.height, type="kfold", B=5, seed=0)
    evidence, _ = probe_shapes(
        frame, "y", [column], terms, Gaussian(), folds=folds, mstop=mstop,
        nu=nu, df=4.0, max_fits=budget, fixed_names=(), env=None)
    return evidence


def test_shapes_still_falling_at_the_last_step_are_not_compared():
    frame = _sine_frame()
    evidence = _shape_probe(frame, mstop=5, nu=0.1)

    assert len(evidence) == 1
    row = evidence[0]
    assert row.comparable is False
    assert any(score.at_cap for score in row.scores)
    assert row.chosen == row.incumbent
    assert "no shape comparison is made" in row.sentence
    assert "Raise mstop or nu" in row.sentence


def test_shapes_with_an_interior_minimum_are_compared_as_before():
    frame = _sine_frame()
    evidence = _shape_probe(frame, mstop=300, nu=0.3)

    assert len(evidence) == 1
    row = evidence[0]
    assert row.comparable is True
    assert not any(score.at_cap for score in row.scores)
    assert "no shape comparison is made" not in row.sentence


def _naval_like_frame(n=300, seed=4):
    """Collinear predictors and a near-deterministic response, the shape of
    the naval propulsion benchmark: the boosted path falls for many steps."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1, 1, n)
    b = rng.uniform(-1, 1, n)
    c = a + 1e-3 * rng.normal(size=n)
    d = 0.5 * a + 0.5 * b + 1e-3 * rng.normal(size=n)
    y = np.sin(3 * a) + 0.5 * b**2 + 0.01 * rng.normal(size=n)
    return pl.DataFrame({"y": y, "a": a, "b": b, "c": c, "d": d})


def test_naval_like_fixture_never_replaces_a_term_on_a_path_still_falling():
    rec = mb.autospec(_naval_like_frame(), "y", budget=160)

    shapes = rec.evidence.get("shapes", ())
    for row in shapes:
        if row.chosen == row.incumbent:
            continue
        chosen = next(score for score in row.scores if score.shape == row.chosen)
        assert chosen.at_cap is False, row.column
        assert row.comparable is True
    for row in shapes:
        if not row.comparable:
            assert "no shape comparison is made" in row.sentence
            assert "no shape comparison is made" in rec.report()
