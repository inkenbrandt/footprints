# test_wang_footprint_adapter.py
import os
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from fluxfootprints.wang_footprint_adapter import WangFootprintModel
from fluxfootprints import ffp_daily_monthly_helper as helper


def make_df(n=3, ol=-80.0):
    """Standardized-name DataFrame matching build_climatology's output columns."""
    return pd.DataFrame(
        {
            "zm": np.full(n, 2.5),
            "h": np.full(n, 1000.0),
            "ol": np.full(n, ol),
            "sigmav": np.full(n, 0.4),
            "umean": np.full(n, 3.0),
        }
    )


# ----------------------------
# _validate_input_df
# ----------------------------
def test_missing_required_column_raises():
    df = make_df().drop(columns=["zm"])
    model = WangFootprintModel(df=df, domain=[-500.0, 500.0, -500.0, 500.0])
    with pytest.raises(ValueError, match="Missing required columns"):
        model._validate_input_df(df)


def test_non_convective_rows_filtered_and_all_invalid_raises():
    df = make_df(ol=50.0)  # stable, not convective
    model = WangFootprintModel(df=df, domain=[-500.0, 500.0, -500.0, 500.0])
    with pytest.raises(ValueError, match="convective conditions"):
        model._validate_input_df(df)


def test_mixed_convective_rows_only_keeps_convective():
    df = make_df(n=4)
    df.loc[0, "ol"] = 50.0  # one stable row mixed in
    model = WangFootprintModel(df=df, domain=[-500.0, 500.0, -500.0, 500.0])
    cleaned = model._validate_input_df(df)
    assert len(cleaned) == 3
    assert (cleaned["ol"] < 0).all()


def test_sentinel_and_nan_rows_dropped():
    df = make_df(n=3)
    df.loc[0, "sigmav"] = -9999
    df.loc[1, "umean"] = np.nan
    model = WangFootprintModel(df=df, domain=[-500.0, 500.0, -500.0, 500.0])
    cleaned = model._validate_input_df(df)
    assert len(cleaned) == 1


# ----------------------------
# run() — direct construction with standardized columns
# ----------------------------
def test_run_produces_normalized_climatology():
    df = make_df(n=3)
    model = WangFootprintModel(
        df=df, domain=[-500.0, 500.0, -500.0, 500.0], dx=10.0, dy=10.0
    )
    results = model.run()

    assert isinstance(results, xr.Dataset)
    assert results.attrs["model"] == "Wang et al. (2006)"
    assert results.attrs["n_timesteps"] == 3

    fclim = results["footprint_climatology"]
    assert fclim.dims == ("x", "y")
    assert np.isfinite(fclim.values).all()
    assert (fclim.values >= 0).all()

    # Grid coordinates must match the data shape (regression check for the
    # previous ny/self.y bugs that caused a coordinate/size mismatch).
    assert fclim.sizes["x"] == model.x.shape[0]
    assert fclim.sizes["y"] == model.y.shape[0]

    # Footprint density integrates to ~1 over the grid.
    total = float(fclim.sum()) * model.dx * model.dy
    assert np.isclose(total, 1.0, atol=1e-2)


def test_run_without_smoothing_still_normalizes():
    df = make_df(n=2)
    model = WangFootprintModel(
        df=df,
        domain=[-500.0, 500.0, -500.0, 500.0],
        dx=10.0,
        dy=10.0,
        smooth_data=False,
    )
    results = model.run()
    fclim = results["footprint_climatology"]
    total = float(fclim.sum()) * model.dx * model.dy
    assert np.isclose(total, 1.0, atol=1e-2)


def test_run_return_result_false_populates_fclim_but_not_results():
    df = make_df(n=2)
    model = WangFootprintModel(df=df, domain=[-500.0, 500.0, -500.0, 500.0])
    out = model.run(return_result=False)
    assert out is None
    assert model.fclim_2d is not None
    assert model.results is None


def test_run_raises_when_all_rows_invalid():
    df = make_df(n=2, ol=50.0)  # non-convective
    model = WangFootprintModel(df=df, domain=[-500.0, 500.0, -500.0, 500.0])
    with pytest.raises(ValueError, match="convective conditions"):
        model.run()


# ----------------------------
# Integration via build_climatology
# ----------------------------
def test_build_climatology_wang_end_to_end():
    n = 4
    df = pd.DataFrame(
        {
            "wind_dir": np.linspace(0, 270, n),
            "umean": np.full(n, 3.0),
            "ustar": np.full(n, 0.3),
            "ol": np.full(n, -80.0),
            "sigmav": np.full(n, 0.5),
            "zm": np.full(n, 2.5),
            "z0": np.full(n, 0.05),
            "h": np.full(n, 1000.0),
        },
        index=pd.date_range("2024-06-01", periods=n, freq="30min"),
    )

    clim = helper.build_climatology(
        df,
        model_type="wang",
        ustar="ustar",
        ol="ol",
        umean="umean",
        sigmav="sigmav",
        wind_dir="wind_dir",
        zm="zm",
        z0="z0",
        h="h",
        dx=10.0,
        dy=10.0,
        domain=(-500.0, 500.0, -500.0, 500.0),
    )

    assert isinstance(clim, WangFootprintModel)
    total = float(clim.fclim_2d.sum()) * clim.dx * clim.dy
    assert np.isclose(total, 1.0, atol=1e-2)
