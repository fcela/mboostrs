"""Two learners of one kind on one column keep separate names.

``bbs(hour) + bbs(hour, by=workingday)`` is the standard hour-of-day profile
with a working-day difference. Both learners read ``hour`` and both are
``bbs``, so the old ``"hour (bbs)"`` label was shared and every accessor keyed
by it -- ``partial_effects``, ``coef``, ``varimp``, ``extract`` -- kept only
one of the two terms.
"""

import numpy as np
import polars as pl
import pytest

import mboostrs as mb


@pytest.fixture(scope="module")
def fit():
    rng = np.random.default_rng(7)
    n = 600
    hour = rng.integers(0, 24, n).astype(float)
    work = rng.integers(0, 2, n).astype(float)
    temp = rng.uniform(0, 30, n)
    eta = (1.0 + 0.6 * np.sin(2 * np.pi * hour / 24)
           + work * 0.8 * np.cos(4 * np.pi * hour / 24) + 0.03 * temp)
    y = rng.poisson(np.exp(eta)).astype(float)
    data = pl.DataFrame({"y": y, "hour": hour, "work": work, "temp": temp})
    return mb.gamboost(
        "y ~ bbs(hour, df=4) + bbs(hour, df=4, by=work) + bbs(temp, df=4)",
        data, family=mb.Poisson(), control=mb.boost_control(150, 0.1))


def test_names_are_distinct(fit):
    names = fit.base_learner_names()
    assert names[:2] == ["hour (bbs)", "hour (bbs, by=work)"]
    assert len(set(names)) == len(names)


def test_partial_effects_keep_both_terms(fit):
    effects = fit.partial_effects()
    assert {"hour (bbs)", "hour (bbs, by=work)"} <= set(effects)
    main = effects["hour (bbs)"][1]
    varying = effects["hour (bbs, by=work)"][1]
    assert not np.allclose(main, varying)
    # The by= term is zero wherever its multiplier is.
    work = np.asarray(fit._data["work"]) if hasattr(fit, "_data") else None
    if work is not None and len(work) == len(varying):
        assert np.allclose(varying[work == 0], 0.0)


def test_partial_effects_sum_to_the_link(fit):
    effects = fit.partial_effects()
    total = sum(values for _, values in effects.values()) + fit._offset
    assert np.allclose(total, fit.predict(type="link"), atol=1e-8)


def test_coef_varimp_extract_keep_both(fit):
    labels = {"hour (bbs)", "hour (bbs, by=work)"}
    assert labels <= set(fit.coef())
    assert labels <= set(fit.varimp()["reduction"])
    assert labels <= set(fit.extract("df"))
    assert labels <= set(fit.extract("lambda", which=list(labels)))


def test_learner_index_accepts_each_label(fit):
    assert fit.learner_index("hour (bbs)") == 0
    assert fit.learner_index("hour (bbs, by=work)") == 1
    assert fit.learner_index("hour (bbs)", kind="BBS") == 0
    assert fit.learner_index("hour (bbs, by=work)", kind="bbs") == 1
    with pytest.raises(ValueError, match="has kind 'bbs'"):
        fit.learner_index("hour (bbs, by=work)", kind="bols")


def test_summary_names_both(fit):
    text = str(fit.summary())
    assert "hour [bbs, by=work]" in text


def test_numbered_terms_keep_their_identity_in_selectors_and_summary():
    x = np.linspace(0, 1, 120)
    data = {"x": x, "y": np.sin(4 * x) + .3 * np.sin(24 * x)}
    model = mb.gamboost("y ~ bbs(x, knots=4, df=4) + bbs(x, knots=20, df=4)",
                       data, control=mb.boost_control(300))
    labels = model.base_learner_names()
    assert labels == ["x (bbs) [1]", "x (bbs) [2]"]
    for index, label in enumerate(labels):
        assert model.learner_index(label, kind="bbs") == index
    counts = np.bincount(model._selected, minlength=2)
    assert np.all(counts > 0), "the fixture must exercise both candidate bases"
    summary = model.summary()
    assert dict(summary.header_pairs)["Learners selected"] == 2
    table = next(table for table in summary.tables if table.title == "Base learners")
    actual = {row[0]: int(row[2]) for row in table.rows}
    assert actual == {f"x [bbs] [{index + 1}]": int(count) for index, count in enumerate(counts)}
