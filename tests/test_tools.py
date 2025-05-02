# tests/test_tools.py
import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import Point
from rasterio.transform import Affine

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from fluxfootprints import (
    polar_to_cartesian_dataframe,
    aggregate_to_daily_centroid,
    generate_density_raster,
)


# ---------------------------------------------------------------------------
# polar_to_cartesian_dataframe
# ---------------------------------------------------------------------------
def test_polar_to_cartesian_dataframe_basic():
    """Known‑angle sanity check + invalid‑value handling."""
    df = pd.DataFrame(
        {
            "WD": [0, 90, 180, 270, -9999, np.nan],
            "Dist": [1, 1, 1, 1, 1, 1],
        }
    )

    out = polar_to_cartesian_dataframe(df)

    # Expected Cartesian coordinates for the first four valid rows
    exp_x = np.array([0, 1, 0, -1], dtype=float)
    exp_y = np.array([1, 0, -1, 0], dtype=float)

    np.testing.assert_allclose(out["X_Dist"].iloc[:4], exp_x, atol=1e-12)
    np.testing.assert_allclose(out["Y_Dist"].iloc[:4], exp_y, atol=1e-12)

    # Invalid entries should propagate to NaN
    assert out["X_Dist"].iloc[4:].isna().all()
    assert out["Y_Dist"].iloc[4:].isna().all()


# ---------------------------------------------------------------------------
# aggregate_to_daily_centroid
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("weighted", [True, False])
def test_aggregate_to_daily_centroid(weighted):
    """Verify weighted vs. unweighted centroids."""
    df = pd.DataFrame(
        {
            "Timestamp": [
                "2025-05-01 00:00",
                "2025-05-01 00:30",
                "2025-05-02 00:00",
            ],
            "X": [0.0, 10.0, 20.0],
            "Y": [0.0, 10.0, 20.0],
            "ET": [1.0, 3.0, 2.0],
        }
    )

    result = aggregate_to_daily_centroid(df, weighted=weighted)

    # Two distinct days expected
    assert len(result) == 2

    day1 = result[result["Date"] == pd.to_datetime("2025‑05‑01").date()].iloc[0]

    if weighted:
        # (0*1 + 10*3) / 4  = 7.5
        expected = 7.5
    else:
        # (0 + 10) / 2 = 5
        expected = 5.0

    assert day1["X"] == pytest.approx(expected, rel=1e-6)
    assert day1["Y"] == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# generate_density_raster
# ---------------------------------------------------------------------------
def test_generate_density_raster_properties():
    """Minimal smoke test: array dims, transform, bounds, and non‑negativity."""
    # Create ten weighted points in a diagonal line
    pts = [
        Point(x, y) for x, y in zip(np.linspace(0, 100, 10), np.linspace(0, 100, 10))
    ]
    gdf = gpd.GeoDataFrame({"ET": np.ones(10)}, geometry=pts, crs="EPSG:5070")

    density, transform, bounds = generate_density_raster(gdf, resolution=50)

    # Basic array checks
    assert density.ndim == 2 and density.size > 0
    assert np.all(density >= 0)

    # Affine transform should reflect the chosen resolution
    assert isinstance(transform, Affine)
    assert transform.a == pytest.approx(50)  # pixel width
    assert transform.e == pytest.approx(-50)  # pixel height (negative y‑scale)

    # Bounds tuple sanity
    assert isinstance(bounds, tuple) and len(bounds) == 4
