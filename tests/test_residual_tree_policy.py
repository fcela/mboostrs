"""Residual-tree adoption policy hook (issue #244).

`residual_tree="auto"` offers the tree on a training-only rule -- a
coordinates-like predictor pair, or adopted pairs leaving residual
structure a shallow tree captures -- and the held-out incremental-value
statistic still decides adoption.
"""

import warnings

import numpy as np
import pytest

import mboostrs as mb
from mboostrs.autospec import _has_coordinate_pair


def test_has_coordinate_pair():
    class Col:
        def __init__(self, name):
            self.name = name

    assert _has_coordinate_pair([Col("latitude"), Col("longitude"), Col("x1")])
    assert _has_coordinate_pair([Col("Lat"), Col("Lon")])
    assert not _has_coordinate_pair([Col("x1"), Col("x2"), Col("x3")])


def test_auto_offers_tree_on_coordinates():
    """On california-style columns the auto rule offers the tree."""
    rng = np.random.default_rng(0)
    n = 200
    lat = rng.uniform(32, 42, n)
    lon = rng.uniform(-124, -114, n)
    # Sharp boundary: price jumps east of -120, which smooth pairs miss.
    y = (lon > -120).astype(float) * 2.0 + rng.normal(scale=0.3, size=n)
    data = {"latitude": lat, "longitude": lon, "y": y}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = mb.autospec(data, "y", budget=120, seed=0, residual_tree="auto",
                          preexploration="off", interactions="report")
    assert rec.evidence.get("residual_tree_offer") == \
        "predictors include a coordinates-like pair"
    # The offer was made; adoption is the held-out statistic's call.
    assert "residual_tree" in rec.evidence


def test_auto_does_not_offer_without_trigger():
    """Pure noise columns: no coordinates, no adopted pairs, no offer."""
    rng = np.random.default_rng(2)
    n = 150
    data = {f"x{j}": rng.normal(size=n) for j in range(4)}
    data["y"] = rng.normal(size=n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = mb.autospec(data, "y", budget=120, seed=0, residual_tree="auto",
                          preexploration="off", interactions="report")
    assert "residual_tree_offer" not in rec.evidence


def test_residual_tree_rejects_bad_values():
    rng = np.random.default_rng(3)
    data = {"x": rng.normal(size=50), "y": rng.normal(size=50)}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mb.autospec(data, "y", budget=20, residual_tree="sometimes")
    except ValueError as e:
        assert 'residual_tree' in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_spatial_cell_codes_form_a_grid():
    """Quantile bins over a coordinate pair give ~10 rows per cell."""
    from mboostrs.autospec import _spatial_cell_codes

    class Col:
        def __init__(self, name):
            self.name = name

    rng = np.random.default_rng(0)
    n = 800
    data = {"lon": rng.uniform(-124, -114, n), "lat": rng.uniform(32, 42, n),
            "y": rng.normal(size=n)}
    codes = _spatial_cell_codes(data, [Col("lon"), Col("lat"), Col("y")])
    assert codes is not None
    n_cells = len(np.unique(codes))
    assert 40 <= n_cells <= 200, n_cells
    assert n / n_cells == pytest.approx(10, abs=5)


def test_spatial_cell_codes_none_without_coordinates():
    """No coordinate pair (or a degenerate one) means no spatial cells."""
    from mboostrs.autospec import _spatial_cell_codes

    class Col:
        def __init__(self, name):
            self.name = name

    rng = np.random.default_rng(0)
    n = 100
    data = {"x1": rng.normal(size=n), "x2": rng.normal(size=n),
            "y": rng.normal(size=n)}
    assert _spatial_cell_codes(data, [Col("x1"), Col("x2")]) is None
    flat = {"lon": np.zeros(n), "lat": rng.uniform(32, 42, n), "y": rng.normal(size=n)}
    assert _spatial_cell_codes(flat, [Col("lon"), Col("lat")]) is None


def test_spatial_cell_folds_decline_a_localized_bump():
    """#266: a tree that only explains sub-cell local bumps must not be
    adopted. Two small elevated patches sit inside single grid cells; random
    row folds validate the tree on neighbouring rows (gain 0.16, z=3.5, so
    the old policy adopts), while spatial-cell grouped folds hold whole
    neighbourhoods out (gain negative, declined)."""
    rng = np.random.default_rng(0)
    n = 1500
    lon = rng.uniform(-124, -114, n)
    lat = rng.uniform(32, 42, n)
    bump = (((np.abs(lon + 119) < 0.35) & (np.abs(lat - 37) < 0.35)) |
            ((np.abs(lon + 116) < 0.35) & (np.abs(lat - 40) < 0.35))).astype(float)
    y = lon * 0.1 + bump * 3.5 + rng.normal(0, 0.5, n)
    data = {"lon": lon, "lat": lat, "y": y}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = mb.autospec(data, "y", budget=300, seed=0, residual_tree=True,
                          preexploration="off", interactions="report")
    ev = rec.evidence["residual_tree"]
    assert ev["validation"].startswith("spatial-cell grouped folds"), ev
    assert ev["adopted"] is False, ev
    # The declined term is the maxdepth=5 residual tree; a joint tree from the
    # shape search (#269) is a separate decision and may still be present.
    assert "maxdepth=5, selection=sse" not in rec.formula, rec.formula
