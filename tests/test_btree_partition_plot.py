"""Tests for the multi-column btree partition plot and table (issue #280).

The plot is a 2-D heatmap of one ``btree`` term's own fitted contribution,
with ``btree_splits()`` cutpoints drawn as rules; a one-column tree falls back
to a step plot. ``btree_partition_table`` is the non-plot companion: one row
per region of the term's fitted partition, with per-column bounds and the
term's value at the region centre.
"""
import numpy as np
import polars as pl
import pytest

import mboostrs as mb
from mboostrs.families import Gaussian
from mboostrs.gamboost import btree_partition_table


def _four_cell_data(seed=7, n=1200):
    """A clean 2-D structure: the effect turns at TEMP=15 and Iws=0.5, so a
    depth-2 tree recovers four cells."""
    rng = np.random.default_rng(seed)
    temp = rng.uniform(0.0, 30.0, n)
    iws = rng.uniform(0.0, 1.0, n)
    y = (2.0 * ((temp > 15.0) & (iws > 0.5))
         - 1.0 * ((temp <= 15.0) & (iws > 0.5))
         + rng.normal(0.0, 0.3, n))
    return {"TEMP": temp, "Iws": iws, "y": y}


@pytest.fixture
def four_cell_model():
    data = _four_cell_data()
    model = mb.gamboost(
        "y ~ btree(TEMP, Iws, maxdepth=2, selection=sse)",
        data=data, family=Gaussian(),
        control=mb.boost_control(1, 0.3),
    )
    return model, data


def _recorded_cutpoints(model):
    splits = next(iter(model.btree_splits().values()))["splits"]
    return [(row["variable"], float(row["cutpoint"])) for row in splits]


def _is_known_bound(value, column, recorded, training):
    if np.isclose(value, float(np.min(training))) or np.isclose(
            value, float(np.max(training))):
        return True
    return any(name == column and np.isclose(value, cut)
               for name, cut in recorded)


class TestPartitionTable:
    def test_table_is_the_four_leaf_tree(self, four_cell_model):
        model, data = four_cell_model
        table = btree_partition_table(model)
        assert isinstance(table, pl.DataFrame)
        assert table.columns == [
            "TEMP_lower", "TEMP_upper", "Iws_lower", "Iws_upper", "value"]
        # A depth-2 tree on two columns has at most four leaves.
        assert 2 <= table.height <= 4

        recorded = _recorded_cutpoints(model)
        assert {name for name, _ in recorded} == {"TEMP", "Iws"}
        training = {"TEMP": data["TEMP"], "Iws": data["Iws"]}

        for row in table.iter_rows(named=True):
            centres = {}
            for column in ("TEMP", "Iws"):
                low = row[f"{column}_lower"]
                high = row[f"{column}_upper"]
                assert low < high
                centres[column] = 0.5 * (low + high)
                for bound in (low, high):
                    assert _is_known_bound(
                        bound, column, recorded, training[column])
            predicted = float(
                model.predict(newdata={c: np.array([v])
                                       for c, v in centres.items()})[0]
                - model.offset)
            assert predicted == pytest.approx(row["value"], abs=1e-9)

        # Every cutpoint the fit recorded bounds at least one region.
        bounds = [(column, bound)
                  for row in table.iter_rows(named=True)
                  for column in ("TEMP", "Iws")
                  for bound in (row[f"{column}_lower"], row[f"{column}_upper"])]
        for column, cut in recorded:
            assert any(name == column and np.isclose(bound, cut)
                       for name, bound in bounds)

    def test_table_method_matches_function(self, four_cell_model):
        model, _ = four_cell_model
        assert model.btree_partition_table().equals(btree_partition_table(model))

    def test_one_column_table_has_intervals(self):
        rng = np.random.default_rng(1)
        n = 300
        x = rng.uniform(0.0, 1.0, n)
        y = 2.0 * (x > 0.5) + rng.normal(0.0, 0.1, n)
        model = mb.gamboost(
            "y ~ btree(x, maxdepth=2, selection=sse)",
            data={"x": x, "y": y}, family=Gaussian(),
            control=mb.boost_control(1, 0.3),
        )
        table = btree_partition_table(model)
        assert table.columns == ["x_lower", "x_upper", "value"]
        assert table.height >= 2
        for row in table.iter_rows(named=True):
            centre = np.array([0.5 * (row["x_lower"] + row["x_upper"])])
            predicted = float(
                model.predict(newdata={"x": centre})[0] - model.offset)
            assert predicted == pytest.approx(row["value"], abs=1e-9)


class TestPlotBtreePartition:
    def test_two_column_heatmap(self, four_cell_model):
        pytest.importorskip("altair")
        model, _ = four_cell_model
        spec = model.plot_btree_partition().to_dict()
        marks = [layer.get("mark", {}).get("type")
                 for layer in spec.get("layer", [])]
        assert "rect" in marks
        text = str(spec)
        assert "TEMP" in text and "Iws" in text

    def test_module_function_accepts_training_data(self, four_cell_model):
        pytest.importorskip("altair")
        from mboostrs.plot import plot_btree_partition

        model, data = four_cell_model
        chart = plot_btree_partition(
            model, data=pl.DataFrame({"TEMP": data["TEMP"], "Iws": data["Iws"]}))
        marks = [layer.get("mark", {}).get("type")
                 for layer in chart.to_dict().get("layer", [])]
        assert "rect" in marks and "circle" in marks

    def test_one_column_falls_back_to_step_plot(self):
        pytest.importorskip("altair")
        from mboostrs.plot import plot_btree_partition

        rng = np.random.default_rng(1)
        n = 300
        x = rng.uniform(0.0, 1.0, n)
        y = 2.0 * (x > 0.5) + rng.normal(0.0, 0.1, n)
        model = mb.gamboost(
            "y ~ btree(x, maxdepth=2, selection=sse)",
            data={"x": x, "y": y}, family=Gaussian(),
            control=mb.boost_control(1, 0.3),
        )
        marks = [layer.get("mark", {}).get("type")
                 for layer in plot_btree_partition(model).to_dict()["layer"]]
        assert "line" in marks and "rect" not in marks

    def test_three_columns_requires_at(self):
        pytest.importorskip("altair")
        from mboostrs.plot import plot_btree_partition

        rng = np.random.default_rng(3)
        n = 400
        a = rng.uniform(0.0, 1.0, n)
        b = rng.uniform(0.0, 1.0, n)
        c = rng.uniform(0.0, 1.0, n)
        y = (a > 0.5) * (b > 0.5) * (c > 0.5) + rng.normal(0.0, 0.1, n)
        model = mb.gamboost(
            "y ~ btree(a, b, c, maxdepth=2, selection=sse)",
            data={"a": a, "b": b, "c": c, "y": y}, family=Gaussian(),
            control=mb.boost_control(1, 0.3),
        )
        with pytest.raises(ValueError, match="at="):
            plot_btree_partition(model)

        spec = plot_btree_partition(model, at={"c": 0.5}).to_dict()
        marks = [layer.get("mark", {}).get("type")
                 for layer in spec.get("layer", [])]
        assert "rect" in marks
        text = str(spec)
        assert "a" in text and "b" in text

    def test_bad_at_column_is_rejected(self):
        pytest.importorskip("altair")
        from mboostrs.plot import plot_btree_partition

        rng = np.random.default_rng(3)
        n = 200
        a = rng.uniform(0.0, 1.0, n)
        b = rng.uniform(0.0, 1.0, n)
        c = rng.uniform(0.0, 1.0, n)
        y = a + b + c + rng.normal(0.0, 0.1, n)
        model = mb.gamboost(
            "y ~ btree(a, b, c, maxdepth=2, selection=sse)",
            data={"a": a, "b": b, "c": c, "y": y}, family=Gaussian(),
            control=mb.boost_control(1, 0.3),
        )
        with pytest.raises(ValueError, match="does not split"):
            plot_btree_partition(model, at={"nope": 0.0})

    def test_grid_must_be_at_least_two(self, four_cell_model):
        pytest.importorskip("altair")
        from mboostrs.plot import plot_btree_partition

        model, _ = four_cell_model
        with pytest.raises(ValueError, match="grid"):
            plot_btree_partition(model, grid=1)

    def test_plot_table_wrapper_matches(self, four_cell_model):
        pytest.importorskip("altair")
        from mboostrs.plot import btree_partition_table as plot_table

        model, _ = four_cell_model
        assert plot_table(model).equals(btree_partition_table(model))


class TestBtreePartitionErrors:
    def test_model_without_a_btree(self):
        rng = np.random.default_rng(0)
        n = 100
        x = rng.normal(size=n)
        y = x + rng.normal(0.0, 0.1, n)
        model = mb.gamboost("y ~ bols(x)", data={"x": x, "y": y},
                            family=Gaussian(),
                            control=mb.boost_control(10, 0.1))
        with pytest.raises(ValueError, match="no btree"):
            btree_partition_table(model)

    def test_model_without_btree_splits(self):
        rng = np.random.default_rng(0)
        n = 100
        x = rng.normal(size=n)
        y = x + rng.normal(0.0, 0.1, n)
        model = mb.glmboost("y ~ x", data={"x": x, "y": y},
                            family=Gaussian(),
                            control=mb.boost_control(10, 0.1))
        with pytest.raises(ValueError, match="btree_splits"):
            btree_partition_table(model)

    def test_term_is_not_a_btree(self):
        rng = np.random.default_rng(2)
        n = 200
        x = rng.uniform(0.0, 3.0, n)
        z = rng.uniform(0.0, 1.0, n)
        y = x * (z >= 0.3) + rng.normal(0.0, 0.12, n)
        model = mb.gamboost(
            "y ~ bols(x) + btree(x, z, maxdepth=1)",
            data={"x": x, "z": z, "y": y}, family=Gaussian(),
            control=mb.boost_control(20, 0.3),
        )
        with pytest.raises(ValueError, match="not a btree"):
            btree_partition_table(model, term="x (bols)")

    def test_ambiguous_term_needs_a_name(self):
        rng = np.random.default_rng(0)
        n = 200
        x = rng.uniform(0.0, 1.0, n)
        z = rng.uniform(0.0, 1.0, n)
        y = (x > 0.5) + (z > 0.5) + rng.normal(0.0, 0.1, n)
        model = mb.gamboost(
            "y ~ btree(x, maxdepth=1) + btree(z, maxdepth=1)",
            data={"x": x, "z": z, "y": y}, family=Gaussian(),
            control=mb.boost_control(5, 0.3),
        )
        with pytest.raises(ValueError, match="btree terms"):
            btree_partition_table(model)

    def test_lss_btree_partition_is_documented_unsupported(self):
        rng = np.random.default_rng(0)
        n = 200
        x = rng.uniform(0.0, 1.0, n)
        z = rng.uniform(0.0, 1.0, n)
        y = x + 0.2 * z + rng.normal(0.0, 0.2, n)
        model = mb.gamboostLSS(
            {"mu": "y ~ btree(x, z, maxdepth=2)", "sigma": "y ~ 1"},
            data={"x": x, "z": z, "y": y}, families=mb.GaussianLSS(),
            control=mb.boost_control(5),
        )
        with pytest.raises(ValueError, match="LSS"):
            btree_partition_table(model)
