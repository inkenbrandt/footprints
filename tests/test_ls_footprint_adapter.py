# test_ls_footprint_adapter.py
import os
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from fluxfootprints.ls_footprint_adapter import LSFootprintModelAdapter
from fluxfootprints import ffp_daily_monthly_helper as helper

DOMAIN = [-500.0, 500.0, -500.0, 500.0]
# Small particle count keeps the stochastic simulation fast for tests.
N_PARTICLES = 500


def make_df(n=2):
    """Standardized-name DataFrame matching build_climatology's output columns."""
    return pd.DataFrame(
        {
            "zm": np.full(n, 2.0),
            "z0": np.full(n, 0.1),
            "ustar": np.full(n, 0.4),
            "ol": np.full(n, -50.0),
            "wind_dir": np.full(n, 270.0),
            "h": np.full(n, 1000.0),
        }
    )


# ----------------------------
# _validate_input_df
# ----------------------------
def test_missing_required_column_raises():
    df = make_df().drop(columns=["zm"])
    model = LSFootprintModelAdapter(df=df, domain=DOMAIN)
    with pytest.raises(ValueError, match="Missing required columns"):
        model._validate_input_df(df)


def test_sentinel_and_nan_rows_dropped():
    df = make_df(n=3)
    df.loc[0, "ustar"] = -9999
    df.loc[1, "z0"] = np.nan
    model = LSFootprintModelAdapter(df=df, domain=DOMAIN)
    cleaned = model._validate_input_df(df)
    assert len(cleaned) == 1


# ----------------------------
# run() — direct construction with standardized columns
# ----------------------------
def test_run_produces_normalized_climatology():
    df = make_df(n=2)
    model = LSFootprintModelAdapter(
        df=df,
        domain=DOMAIN,
        dx=25.0,
        dy=25.0,
        n_particles=N_PARTICLES,
    )
    results = model.run()

    assert isinstance(results, xr.Dataset)
    assert results.attrs["model"] == "Lagrangian Stochastic"
    assert results.attrs["n_timesteps"] == 2
    assert results.attrs["n_particles"] == N_PARTICLES

    fclim = results["footprint_climatology"]
    assert fclim.dims == ("x", "y")
    assert np.isfinite(fclim.values).all()
    assert (fclim.values >= 0).all()

    assert fclim.sizes["x"] == model.x.shape[0]
    assert fclim.sizes["y"] == model.y.shape[0]

    # Footprint density integrates to ~1 over the grid.
    total = float(fclim.sum()) * model.dx * model.dy
    assert np.isclose(total, 1.0, atol=5e-2)


def test_run_return_result_false_populates_fclim_but_not_results():
    df = make_df(n=1)
    model = LSFootprintModelAdapter(
        df=df, domain=DOMAIN, dx=25.0, dy=25.0, n_particles=N_PARTICLES
    )
    out = model.run(return_result=False)
    assert out is None
    assert model.fclim_2d is not None
    assert model.results is None


def test_run_raises_when_all_rows_invalid():
    df = make_df(n=2)
    df["ustar"] = np.nan
    model = LSFootprintModelAdapter(df=df, domain=DOMAIN, n_particles=N_PARTICLES)
    with pytest.raises(RuntimeError, match="No valid footprints calculated"):
        model.run()


# ----------------------------
# Integration via build_climatology
# ----------------------------
def test_build_climatology_ls_end_to_end():
    n = 2
    df = pd.DataFrame(
        {
            "wind_dir": np.full(n, 270.0),
            "umean": np.full(n, 3.0),
            "ustar": np.full(n, 0.4),
            "ol": np.full(n, -50.0),
            "sigmav": np.full(n, 0.5),
            "zm": np.full(n, 2.0),
            "z0": np.full(n, 0.1),
            "h": np.full(n, 1000.0),
        },
        index=pd.date_range("2024-06-01", periods=n, freq="30min"),
    )

    clim = helper.build_climatology(
        df,
        model_type="ls",
        ustar="ustar",
        ol="ol",
        umean="umean",
        sigmav="sigmav",
        wind_dir="wind_dir",
        zm="zm",
        z0="z0",
        h="h",
        dx=25.0,
        dy=25.0,
        domain=tuple(DOMAIN),
        n_particles=N_PARTICLES,
    )

    assert isinstance(clim, LSFootprintModelAdapter)
    total = float(clim.fclim_2d.sum()) * clim.dx * clim.dy
    assert np.isclose(total, 1.0, atol=5e-2)
