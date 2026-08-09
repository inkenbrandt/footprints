# test_kormannmeixner_adapter.py
import os
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from fluxfootprints.kormannmeixner_adapter import KormannMeixnerModel
from fluxfootprints import ffp_daily_monthly_helper as helper

DOMAIN = [-1000.0, 1000.0, -1000.0, 1000.0]


def make_df(n=3):
    """Standardized-name DataFrame matching build_climatology's output columns."""
    return pd.DataFrame(
        {
            "zm": np.full(n, 10.0),
            "z0": np.full(n, 0.05),
            "ustar": np.full(n, 0.4),
            "ol": np.full(n, -80.0),
            "sigmav": np.full(n, 0.5),
            "umean": np.full(n, 3.0),
        }
    )


# ----------------------------
# _validate_input_df
# ----------------------------
def test_missing_required_column_raises():
    df = make_df().drop(columns=["zm"])
    model = KormannMeixnerModel(df=df, domain=DOMAIN)
    with pytest.raises(ValueError, match="Missing required columns"):
        model._validate_input_df(df)


def test_sentinel_and_nan_rows_dropped():
    df = make_df(n=3)
    df.loc[0, "ustar"] = -9999
    df.loc[1, "z0"] = np.nan
    model = KormannMeixnerModel(df=df, domain=DOMAIN)
    cleaned = model._validate_input_df(df)
    assert len(cleaned) == 1


# ----------------------------
# run() — direct construction with standardized columns
# ----------------------------
def test_run_produces_normalized_climatology():
    df = make_df(n=3)
    model = KormannMeixnerModel(df=df, domain=DOMAIN, dx=10.0, dy=10.0)
    results = model.run()

    assert isinstance(results, xr.Dataset)
    assert results.attrs["model"] == "Kormann-Meixner (2001)"
    assert results.attrs["n_timesteps"] == 3

    fclim = results["footprint_climatology"]
    assert fclim.dims == ("x", "y")
    assert np.isfinite(fclim.values).all()
    assert (fclim.values >= 0).all()

    # Grid coordinates must match the data shape and requested domain.
    assert fclim.sizes["x"] == model.x.shape[0]
    assert fclim.sizes["y"] == model.y.shape[0]
    assert model.x.min() == pytest.approx(DOMAIN[0])
    assert model.y.min() == pytest.approx(DOMAIN[2])

    # Footprint density integrates to ~1 over the grid.
    total = float(fclim.sum()) * model.dx * model.dy
    assert np.isclose(total, 1.0, atol=1e-1)


def test_run_with_wind_dir_rotates_and_stays_normalized():
    df = make_df(n=2)
    df["wind_dir"] = [0.0, 180.0]
    model = KormannMeixnerModel(df=df, domain=DOMAIN, dx=10.0, dy=10.0)
    results = model.run()

    fclim = results["footprint_climatology"]
    assert np.isfinite(fclim.values).all()
    # Interpolation/smoothing can leave negligible (~1e-15) floating-point
    # noise around zero; assert non-negative up to that tolerance.
    assert (fclim.values >= -1e-10).all()
    total = float(fclim.sum()) * model.dx * model.dy
    assert np.isclose(total, 1.0, atol=1e-1)


def test_run_without_smoothing_still_normalizes():
    df = make_df(n=2)
    model = KormannMeixnerModel(
        df=df, domain=DOMAIN, dx=10.0, dy=10.0, smooth_data=False
    )
    results = model.run()
    fclim = results["footprint_climatology"]
    total = float(fclim.sum()) * model.dx * model.dy
    assert np.isclose(total, 1.0, atol=1e-1)


def test_run_return_result_false_populates_fclim_but_not_results():
    df = make_df(n=2)
    model = KormannMeixnerModel(df=df, domain=DOMAIN)
    out = model.run(return_result=False)
    assert out is None
    assert model.fclim_2d is not None
    assert model.results is None


def test_run_raises_when_all_rows_invalid():
    df = make_df(n=2)
    df["ustar"] = np.nan
    model = KormannMeixnerModel(df=df, domain=DOMAIN)
    with pytest.raises(RuntimeError, match="No valid footprints calculated"):
        model.run()


# ----------------------------
# Integration via build_climatology
# ----------------------------
def test_build_climatology_km_end_to_end():
    n = 4
    df = pd.DataFrame(
        {
            "wind_dir": np.linspace(0, 270, n),
            "umean": np.full(n, 3.0),
            "ustar": np.full(n, 0.3),
            "ol": np.full(n, -80.0),
            "sigmav": np.full(n, 0.5),
            "zm": np.full(n, 10.0),
            "z0": np.full(n, 0.05),
            "h": np.full(n, 1000.0),
        },
        index=pd.date_range("2024-06-01", periods=n, freq="30min"),
    )

    clim = helper.build_climatology(
        df,
        model_type="km",
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
        domain=tuple(DOMAIN),
    )

    assert isinstance(clim, KormannMeixnerModel)
    total = float(clim.fclim_2d.sum()) * clim.dx * clim.dy
    assert np.isclose(total, 1.0, atol=1e-1)


def test_build_climatology_km_missing_z0_raises():
    n = 2
    df = pd.DataFrame(
        {
            "wind_dir": np.full(n, 270.0),
            "umean": np.full(n, 3.0),
            "ustar": np.full(n, 0.3),
            "ol": np.full(n, -80.0),
            "sigmav": np.full(n, 0.5),
            "zm": np.full(n, 10.0),
        },
        index=pd.date_range("2024-06-01", periods=n, freq="30min"),
    )

    # z0 explicitly unset (no "z0" column in df either) so build_climatology's
    # per-model required-column check is what raises, not a KeyError from
    # trying to resolve a default "z0" column name.
    with pytest.raises(ValueError, match="requires the following missing"):
        helper.build_climatology(
            df,
            model_type="km",
            ustar="ustar",
            ol="ol",
            umean="umean",
            sigmav="sigmav",
            wind_dir="wind_dir",
            zm="zm",
            z0=None,
            dx=10.0,
            dy=10.0,
            domain=tuple(DOMAIN),
        )
