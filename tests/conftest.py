# conftest.py
import numpy as np
import pandas as pd
import xarray as xr
import importlib
import pytest


def _maybe_load_module():
    try:
        return importlib.import_module("fluxfootprints.ffp_xr")
    except Exception as e:
        pytest.skip(f"Could not import fluxfootprints.ffp_xr: {e}")


@pytest.fixture(scope="session")
def ffp_xr():
    return _maybe_load_module()


@pytest.fixture(scope="session")
def compute_fn(ffp_xr):
    """
    Try common function names that return a 2D (or 3D with time) footprint
    as an xarray object. Skip if none exist.
    """
    candidates = [
        "compute_ffp_xr",
        "calc_ffp_xr",
        "ffp_xr_compute",
        "ffp_2d",  # seen in earlier discussions
        "compute_footprint_xr",
        "calc_ffp_2d",
    ]
    for name in candidates:
        if hasattr(ffp_xr, name):
            return getattr(ffp_xr, name)
    pytest.skip("No compute function found in ffp_xr (tried common names).")


@pytest.fixture(scope="session")
def aggregate_fn(ffp_xr):
    """
    Function that aggregates/resamples a time-stacked footprint into daily/monthly composites.
    """
    candidates = [
        "aggregate_footprint",
        "aggregate_ffp_xr",
        "resample_footprint",
        "summarize_footprint",
    ]
    for name in candidates:
        if hasattr(ffp_xr, name):
            return getattr(ffp_xr, name)
    # Not fatal — some modules just expect xarray.resample; tests will branch.
    return None


@pytest.fixture(scope="session")
def contour_fn(ffp_xr):
    """
    Function that generates r% source-area polygons from a 2D footprint field.
    Should return Shapely geometries or GeoDataFrame.
    """
    candidates = [
        "contour_polygons",
        "make_contour_polygons",
        "_make_contour_polygon_from_field_alt",
        "footprint_contours",
    ]
    for name in candidates:
        if hasattr(ffp_xr, name):
            return getattr(ffp_xr, name)
    return None


@pytest.fixture(scope="session")
def export_gpkg_fn(ffp_xr):
    """
    Optional: a function that exports contours to a GeoPackage.
    """
    candidates = ["export_contours_gpkg", "write_contours_gpkg", "save_contours_gpkg"]
    for name in candidates:
        if hasattr(ffp_xr, name):
            return getattr(ffp_xr, name)
    return None


@pytest.fixture
def synthetic_inputs():
    """
    Create a tiny, well-behaved synthetic tower record that should be valid for the
    Kljun et al. (2015) parameterization. We keep numbers simple & physical.
    """
    # 6 timestamps across two days
    t = pd.date_range("2024-07-01 11:00:00", periods=6, freq="6H")

    # Reasonable values
    ds = xr.Dataset(
        data_vars=dict(
            USTAR=("time", np.full(len(t), 0.35)),  # m/s
            V_SIGMA=("time", np.full(len(t), 0.6)),  # m/s
            MO_LENGTH=(
                "time",
                np.array([50, -100, 40, -60, 80, -120], dtype=float),
            ),  # m
            WD=("time", np.array([0, 45, 90, 180, 225, 270], dtype=float)),  # degrees
            WS=("time", np.full(len(t), 4.0)),  # m/s at zm
        ),
        coords=dict(time=t),
        attrs=dict(description="Synthetic EC inputs for unit tests"),
    )
    # heights & roughness — pass as attrs or separate params depending on API
    meta = dict(zm=10.0, z0=0.1, h=600.0)  # m
    return ds, meta
