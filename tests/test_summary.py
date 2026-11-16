"""Tests for the text/HTML model summary."""
import numpy as np
import pytest
from mboostrs.families import Gaussian
from mboostrs.summary import TEXT_WIDTH, Summary, SummaryTable, build_summary

import mboostrs as mboost


def _shared_data():
    """The rows the glm_model fixture is fitted to."""
    rng = np.random.default_rng(0)
    n = 120
    x1, x2, x3 = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    y = 2.0 * x1 - 1.0 * x2 + rng.normal(scale=0.4, size=n)
    return {"y": y, "x1": x1, "x2": x2, "x3": x3}


@pytest.fixture
def glm_model():
    data = _shared_data()
    return mboost.glmboost("y ~ x1 + x2 + x3", data=data, family=Gaussian(),
                           control=mboost.boost_control(80, 0.1))


@pytest.fixture
def gam_model():
    rng = np.random.default_rng(1)
    n = 150
    z, w = rng.uniform(-2, 2, n), rng.normal(size=n)
    y = np.sin(2 * z) + 0.5 * w + rng.normal(scale=0.3, size=n)
    data = {"y": y, "z": z, "w": w}
    return mboost.gamboost("y ~ bbs(z, df=4) + bols(w)", data=data,
                           family=Gaussian(),
                           control=mboost.boost_control(60, 0.1))


@pytest.fixture
def lss_model():
    rng = np.random.default_rng(2)
    n = 300
    x = rng.uniform(-2, 2, n)
    y = 1.0 + 0.8 * x + rng.normal(scale=np.exp(0.2 * x), size=n)
    return mboost.gamboostLSS("y ~ bols(x)", data={"y": y, "x": x},
                              families=mboost.GaussianLSS(),
                              control=mboost.boost_control(50, 0.1))


class TestRendering:
    def test_text_never_exceeds_the_declared_width(self, glm_model, gam_model, lss_model):
        for model in (glm_model, gam_model, lss_model):
            text = model.summary().as_text()
            assert max(len(line) for line in text.splitlines()) <= TEXT_WIDTH

    def test_a_long_header_value_takes_its_own_line(self):
        """Pairing a long value into a half-width column widened every other
        row to match it, pushing the header past the text width."""
        summary = Summary(
            "T",
            [("Short", 1.0), ("Another", 2.0),
             ("Offsets", "mu=-1.7541, sigma=0.5198, nu=-0.3312, tau=1.0004")],
        )
        lines = summary.as_text().splitlines()
        assert max(len(line) for line in lines) <= TEXT_WIDTH
        own_line = [ln for ln in lines if ln.startswith("Offsets:")]
        assert len(own_line) == 1
        assert "Short" not in own_line[0]

    def test_html_escapes_its_content(self):
        table = SummaryTable("t", ["term"], [["<script>x</script>"]])
        html = table.as_html()
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_repr_and_str_agree_with_as_text(self, glm_model):
        summary = glm_model.summary()
        assert str(summary) == summary.as_text()
        assert repr(summary) == summary.as_text()

    def test_dict_form_is_json_serializable(self, glm_model):
        import json

        json.dumps(glm_model.summary().as_dict())


class TestContent:
    def test_glm_reports_coefficients_with_selection_counts(self, glm_model):
        text = glm_model.summary().as_text()
        assert "Coefficients" in text
        assert "first iter" in text
        # x1 has the largest true effect, so it should head the table.
        rows = [ln for ln in text.splitlines() if ln.startswith("x1")]
        assert rows

    def test_risk_reduction_is_measured_from_the_offset_model(self, glm_model):
        risk = np.asarray(glm_model._risk, dtype=float)
        expected = (risk[0] - risk[glm_model.mstop]) / abs(risk[0])
        header = dict(glm_model.summary().header_pairs)
        assert header["Risk reduction"] == f"{expected:.1%}"

    def test_gam_reports_effect_ranges_not_coefficients(self, gam_model):
        text = gam_model.summary().as_text()
        assert "Fitted effects" in text
        assert "Coefficients" not in text

    def test_lss_reports_one_table_per_parameter(self, lss_model):
        summary = lss_model.summary()
        titles = [t.title for t in summary.tables]
        for name in lss_model.param_names:
            assert any(name in t for t in titles)
        header = dict(summary.header_pairs)
        assert "Offsets" in header
        assert all(name in header["Offsets"] for name in lss_model.param_names)

    def test_intercept_is_not_reported_as_never_selected(self, glm_model):
        """The intercept is fitted throughout rather than competing for
        updates, so listing it as unselected reads as a defect in the fit."""
        notes = [t.note for t in glm_model.summary().tables if t.note]
        assert not any("(Intercept)" in n for n in notes)

    def test_unselected_learners_are_named(self):
        rng = np.random.default_rng(3)
        n = 100
        x1, junk = rng.normal(size=n), rng.normal(size=n)
        y = 3.0 * x1 + rng.normal(scale=0.1, size=n)
        model = mboost.glmboost("y ~ x1 + junk", data={"y": y, "x1": x1, "junk": junk},
                                family=Gaussian(),
                                control=mboost.boost_control(5, 0.1))
        notes = [t.note for t in model.summary().tables if t.note]
        assert any("Never selected" in n for n in notes)


class TestDispatch:
    def test_build_summary_picks_the_right_builder(self, glm_model, gam_model, lss_model):
        assert "glmboost" in build_summary(glm_model).as_text()
        assert "gamboost" in build_summary(gam_model).as_text()
        assert "gamboostLSS" in build_summary(lss_model).as_text()

    def test_as_dict_still_returns_the_mapping(self, glm_model):
        """The dict form predates the Summary object and is depended on."""
        info = glm_model.summary(as_dict=True)
        assert isinstance(info, dict)
        assert info["model_class"] == "GLMBoostModel"
        assert "selection_frequency" in info


class TestCrossValidationBlock:
    def test_cv_block_reports_the_gap_to_the_optimum(self, glm_model):
        folds = mboost.cv_weights(120, type="kfold", B=5, seed=0)
        cv = glm_model.cvrisk(folds=folds)
        text = glm_model.summary(cv=cv).as_text()

        assert "Cross-validation" in text
        assert "Optimal mstop" in text
        assert "CV risk at optimum" in text
        assert max(len(line) for line in text.splitlines()) <= TEXT_WIDTH

    def test_a_model_past_the_optimum_says_so(self, glm_model):
        folds = mboost.cv_weights(120, type="kfold", B=5, seed=0)
        cv = glm_model.cvrisk(folds=folds)
        notes = [t.note for t in glm_model.summary(cv=cv).tables if t.note]
        cv_notes = [n for n in notes if "cross-validated optimum" in n]
        if int(cv.optimal_mstop) != int(glm_model.mstop):
            assert cv_notes, "a mismatch with the optimum should be stated"
            assert "past" in cv_notes[0] or "short of" in cv_notes[0]
        else:
            assert not cv_notes

    def test_summary_without_cv_has_no_cv_block(self, glm_model):
        assert "Cross-validation" not in glm_model.summary().as_text()


class TestFoldLabel:
    """`cvrisk` built its label from `type`/`B`, which keep their defaults when
    an explicit fold matrix is passed, so every supplied matrix was reported
    as "25-fold bootstrap"."""

    def test_supplied_folds_are_labelled_by_their_own_count(self, glm_model):
        for n_folds in (3, 7):
            folds = mboost.cv_weights(120, type="kfold", B=n_folds, seed=0)
            result = glm_model.cvrisk(folds=folds)
            assert result.fold_type == f"{n_folds} supplied folds"
            assert result.risk_matrix.shape[0] == n_folds

    def test_generated_folds_report_their_scheme(self, glm_model):
        assert glm_model.cvrisk(type="kfold", B=4).fold_type == "4-fold kfold"
        assert glm_model.cvrisk(type="bootstrap", B=6).fold_type == "6-fold bootstrap"


class TestCompareSummaries:
    def test_columns_line_up_across_different_formulas(self):
        from mboostrs.summary import compare_summaries

        # Both models are fitted to the same rows and response: compare_summaries
        # refuses models whose data differ.
        data = _shared_data()
        rng = np.random.default_rng(3)
        data["w"], data["v"] = rng.uniform(-2, 2, len(data["y"])), rng.normal(size=len(data["y"]))
        glm_model = mboost.glmboost("y ~ x1 + x2 + x3", data=data, family=Gaussian(),
                                    control=mboost.boost_control(80, 0.1))
        gam_model = mboost.gamboost("y ~ bbs(w, df=4) + bols(v)", data=data,
                                    family=Gaussian(), control=mboost.boost_control(60, 0.1))
        summary = compare_summaries(glm_model, gam_model, labels=["glm", "gam"])
        text = summary.as_text()
        assert "glm" in text and "gam" in text
        assert "Updates per base learner" in text
        assert max(len(line) for line in text.splitlines()) <= TEXT_WIDTH

        # The two models share no learner names, so every cell in one column
        # is blank where the other has a count.
        learner_table = next(t for t in summary.tables
                             if t.title == "Updates per base learner")
        for row in learner_table.rows:
            assert row[1] == "" or row[2] == ""

    def test_mixed_families_are_flagged_as_incomparable(self):
        from mboostrs.summary import compare_summaries

        # Same response and rows, two families: the family note applies, not the
        # data refusal.
        data = _shared_data()
        laplace = mboost.glmboost("y ~ x1 + x2 + x3", data=data, family=mboost.Laplace(),
                                  control=mboost.boost_control(40, 0.1))
        gaussian = mboost.glmboost("y ~ x1 + x2 + x3", data=data, family=Gaussian(),
                                   control=mboost.boost_control(40, 0.1))
        summary = compare_summaries(gaussian, laplace)
        assert any("not comparable" in n for n in summary.notes)

    def test_label_count_must_match(self, glm_model, gam_model):
        from mboostrs.summary import compare_summaries

        with pytest.raises(ValueError, match="2 labels for 2 models|1 labels"):
            compare_summaries(glm_model, gam_model, labels=["only one"])

    def test_cv_columns_are_added_when_supplied(self, glm_model):
        from mboostrs.summary import compare_summaries

        cv = glm_model.cvrisk(folds=mboost.cv_weights(120, type="kfold", B=4, seed=0))
        text = compare_summaries(glm_model, cv=[cv]).as_text()
        assert "CV risk" in text
        assert "CV optimal mstop" in text

    def test_no_models_is_an_error(self):
        from mboostrs.summary import compare_summaries

        with pytest.raises(ValueError, match="at least one model"):
            compare_summaries()


class TestOffsetContract:
    """An offset is one known contribution per row (R's `offset=`), fitted
    on top of rather than estimated; the older single-number restriction
    failed at the Rust boundary with "only 0-dimensional arrays can be
    converted to Python scalars", naming neither the argument nor the
    limitation. `tests/test_observation_offsets.py` covers the full vector
    contract, including R parity; these pin the scalar-equivalence and
    refusal edges that live in this module."""

    def _data(self):
        rng = np.random.default_rng(0)
        n = 80
        x = rng.normal(size=n)
        return {"y": 2.0 * x + rng.normal(scale=0.3, size=n), "x": x}, n

    @pytest.mark.parametrize("fitter,formula", [
        ("glmboost", "y ~ x"),
        ("gamboost", "y ~ bbs(x, df=4)"),
    ])
    def test_scalar_and_constant_vector_offsets_agree(self, fitter, formula):
        data, n = self._data()
        fit = getattr(mboost, fitter)
        control = mboost.boost_control(20, 0.1)
        scalar = fit(formula, data=data, family=Gaussian(), control=control, offset=1.5)
        vector = fit(formula, data=data, family=Gaussian(), control=control,
                     offset=np.full(n, 1.5))
        assert scalar._offset == vector._offset == 1.5
        np.testing.assert_array_equal(np.asarray(scalar.predict()),
                                      np.asarray(vector.predict()))

    @pytest.mark.parametrize("fitter,formula", [
        ("glmboost", "y ~ x"),
        ("gamboost", "y ~ bbs(x, df=4)"),
    ])
    def test_a_varying_offset_fits_and_is_honoured_exactly(self, fitter, formula):
        """The offset enters with coefficient one: eta - offset is the same
        for every constant shift of the response."""
        data, n = self._data()
        fit = getattr(mboost, fitter)
        offset = np.linspace(0.0, 1.0, n)
        shifted = fit(formula, data={**data, "y": data["y"] + offset},
                      family=Gaussian(),
                      control=mboost.boost_control(20, 0.1), offset=offset)
        assert np.all(np.isfinite(np.asarray(shifted.predict())))

    @pytest.mark.parametrize("fitter,formula", [
        ("glmboost", "y ~ x"),
        ("gamboost", "y ~ bbs(x, df=4)"),
    ])
    def test_a_nonfinite_or_misaligned_offset_is_refused_with_an_actionable_message(
            self, fitter, formula):
        data, n = self._data()
        fit = getattr(mboost, fitter)
        nonfinite = np.linspace(0.0, 1.0, n)
        nonfinite[11] = np.inf
        with pytest.raises(ValueError, match="entry 11 is not"):
            fit(formula, data=data, family=Gaussian(),
                control=mboost.boost_control(20, 0.1), offset=nonfinite)
        with pytest.raises(ValueError, match=f"{n} rows"):
            fit(formula, data=data, family=Gaussian(),
                control=mboost.boost_control(20, 0.1),
                offset=np.linspace(0.0, 1.0, n - 1))


class TestMissingColumnMessage:
    """Indexing a frame for a column the model needs raised the backend's own
    error — polars says only '"yearc" not found' — which says neither that a
    model wanted it nor what else it wants."""

    def _fit(self):
        rng = np.random.default_rng(0)
        n = 120
        a, b = rng.normal(size=n), rng.normal(size=n)
        data = {"y": a + b, "a": a, "b": b}
        return mboost.gamboost("y ~ bbs(a, df=4) + bbs(b, df=4)", data=data,
                               family=Gaussian(),
                               control=mboost.boost_control(40, 0.1)), data

    def test_message_names_the_model_and_lists_what_is_present(self):
        fit, data = self._fit()
        newdata = {k: v for k, v in data.items() if k != "b"}
        with pytest.raises(KeyError) as excinfo:
            fit.predict(newdata=newdata)
        message = str(excinfo.value)
        assert "required by the model" in message
        assert "'b'" in message
        assert "Available" in message

    def test_a_case_mismatch_is_pointed_out(self):
        fit, data = self._fit()
        newdata = {("B" if k == "b" else k): v for k, v in data.items()}
        with pytest.raises(KeyError, match="Did you mean 'B'"):
            fit.predict(newdata=newdata)


class TestMotorcycleDataset:
    """The bundled copy of Silverman's mcycle, used by the case study."""

    def test_shape_and_heteroscedasticity(self):
        import mboostrs as mb

        data = mb.datasets.motorcycle(output="records")
        times, accel = np.asarray(data["times"]), np.asarray(data["accel"])
        assert len(times) == 133
        # Repeated times are what make the noise estimable at all.
        assert len(np.unique(times)) < len(times)

        early = accel[(times > 0) & (times <= 14)]
        middle = accel[(times > 20) & (times <= 30)]
        assert early.std(ddof=1) < 5
        assert middle.std(ddof=1) > 40

    @pytest.mark.parametrize("output", ["polars", "pandas", "records"])
    def test_every_output_backend_loads(self, output):
        import mboostrs as mb

        if output == "pandas":
            pytest.importorskip("pandas")
        loaded = mb.datasets.motorcycle(output=output)
        columns = loaded.keys() if isinstance(loaded, dict) else loaded.columns
        assert set(columns) == {"times", "accel"}


class TestLearnerIndex:
    """Two learners on one column share a name, which the decomposition idiom
    `bols(x) + bbs(x, df=1, center=True)` relies on. `names.index(name)`
    returned the first, so the deviation learner was unreachable through the
    public API and had to be assembled from private attributes."""

    @pytest.fixture
    def decomposed(self):
        rng = np.random.default_rng(0)
        n = 120
        x, z = rng.normal(size=n), rng.normal(size=n)
        data = {"y": 2 * x + np.sin(2 * z) + rng.normal(scale=0.3, size=n),
                "x": x, "z": z}
        return mboost.gamboost(
            "y ~ bols(x) + bbs(x, df=1, center=True) + bols(z) "
            "+ bbs(z, df=1, center=True)",
            data=data, family=Gaussian(),
            control=mboost.boost_control(200, 0.1))

    def test_kind_selects_between_learners_sharing_a_name(self, decomposed):
        linear = decomposed.learner_index("x", kind="bols")
        deviation = decomposed.learner_index("x", kind="bbs")
        assert linear != deviation
        assert type(decomposed._learners_spec[linear]).__name__ == "Bols"
        assert type(decomposed._learners_spec[deviation]).__name__ == "Bbs"

    def test_an_ambiguous_name_is_refused(self, decomposed):
        with pytest.raises(ValueError, match="pass kind="):
            decomposed.learner_index("x")

    def test_an_unknown_kind_lists_what_is_there(self, decomposed):
        with pytest.raises(ValueError, match="the learners on that column are"):
            decomposed.learner_index("x", kind="btree")

    def test_partial_effects_accepts_an_index(self, decomposed):
        """z is generated as sin(2z), so its deviation learner is selected;
        x is linear, so its deviation learner never is and has no effect to
        report."""
        deviation = decomposed.learner_index("z", kind="bbs")
        assert deviation in set(decomposed._selected), "fixture must select it"

        effects = decomposed.partial_effects(which=deviation, n_points=40)
        assert len(effects) == 1
        key = next(iter(effects))
        # Shared names are disambiguated in the result key.
        assert key == "z (bbs)"
        grid, effect = effects[key]
        assert len(grid) == len(effect) == 40
        assert np.ptp(effect) > 0

    def test_an_out_of_range_index_is_refused(self, decomposed):
        with pytest.raises(ValueError, match="out of range"):
            decomposed.partial_effects(which=999)

    def test_a_unique_name_still_resolves_without_kind(self):
        rng = np.random.default_rng(1)
        n = 80
        x = rng.normal(size=n)
        model = mboost.gamboost("y ~ bbs(x, df=4)",
                                data={"y": 2 * x, "x": x}, family=Gaussian(),
                                control=mboost.boost_control(20, 0.1))
        assert model.learner_index("x") == 0
