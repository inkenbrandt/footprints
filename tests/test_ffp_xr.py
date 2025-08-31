# tests/test_ffp_xr.py
import importlib.util
import pathlib
import sys
import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from fluxfootprints import ffp_climatology_new as FFP

# --- Fixtures -----------------------------------------------------------------
@pytest.fixture
def small_domain():
    # Small grid keeps tests fast: 11 x 11 (from -50..50 step 10)
    return [-50.0, 50.0, -50.0, 50.0]


@pytest.fixture
def df_valid_and_invalid():
    """
    Build a DataFrame with 6 valid and 2 invalid rows to test filtering.
    Required raw columns before renaming in prep_df_fields():
      V_SIGMA, USTAR, wd, MO_LENGTH, ws
    """
    times = pd.date_range("2025-01-01", periods=8, freq="H")
    df = pd.DataFrame(
        {
            # last two rows invalid/filtered: negative V_SIGMA, tiny USTAR, wd > 360, NaN ol
            "V_SIGMA": [0.50, 0.60, 0.40, 0.45, 0.55, 0.52, -0.10, 0.50],
            "USTAR": [0.30, 0.35, 0.25, 0.40, 0.20, 0.15, 0.01, 0.02],
            "wd": [0.0, 90.0, 180.0, 270.0, 45.0, 315.0, 10.0, 400.0],
            "MO_LENGTH": [-100.0, -50.0, 50.0, 100.0, 10.0, -10.0, np.nan, 25.0],
            "ws": [3.0, 4.0, 5.0, 3.0, 2.0, 3.0, 3.0, 4.0],
            # Intentionally omit inst_height/crop_height to use defaults
        },
        index=times,
    )
    return df


@pytest.fixture
def model_unsmoothed(df_valid_and_invalid, small_domain):
    # Use verbosity=0 to avoid noisy logging in pytest output
    return FFP(
        df=df_valid_and_invalid.copy(),
        domain=small_domain,
        dx=10.0,
        dy=10.0,
        smooth_data=False,
        verbosity=0,
    )


@pytest.fixture
def model_smoothed(df_valid_and_invalid, small_domain):
    return FFP(
        df=df_valid_and_invalid.copy(),
        domain=small_domain,
        dx=10.0,
        dy=10.0,
        smooth_data=True,
        verbosity=0,
    )


# --- Tests: input prep / validation -------------------------------------------
def test_prep_df_fields_filters_and_computed_columns(model_unsmoothed):
    """
    After __init__, invalid rows should be dropped and key derived fields present.
    """
    m = model_unsmoothed

    # Two invalid rows should be removed -> 6 valid time steps remain
    assert m.ts_len == 6

    # Derived/renamed columns exist
    for col in ["sigmav", "ustar", "wind_dir", "ol", "z0", "zm", "h", "h_c"]:
        assert col in m.df.columns

    # Physical sanity checks on derived columns
    assert np.all(m.df["zm"].values > 0.0)
    assert np.all(m.df["h"].values > 10.0)
    assert np.all(m.df["zm"].values < m.df["h"].values)


def test_missing_required_parameters_raise():
    """
    If required inputs are missing, the class should raise via raise_ffp_exception(1).
    """
    times = pd.date_range("2025-01-01", periods=4, freq="H")
    df_missing = pd.DataFrame(
        {
            # 'USTAR' intentionally missing
            "V_SIGMA": [0.5, 0.6, 0.4, 0.5],
            "wd": [0.0, 90.0, 180.0, 270.0],
            "MO_LENGTH": [-50.0, -10.0, 10.0, 50.0],
            "ws": [3.0, 4.0, 5.0, 3.0],
        },
        index=times,
    )

    with pytest.raises(ValueError, match=r"FFP Exception 1"):
        _ = FFP(df=df_missing, domain=[-50, 50, -50, 50], dx=10.0, dy=10.0, verbosity=0)


def test_explicit_raise_ffp_exception(model_unsmoothed):
    """
    Exercise the explicit error path to ensure ValueError is raised for fatal codes.
    """
    with pytest.raises(ValueError, match="FFP Exception 4"):
        model_unsmoothed.raise_ffp_exception(4)


# --- Tests: grid and dataset setup --------------------------------------------
def test_grid_creation_and_initial_arrays(model_unsmoothed, small_domain):
    m = model_unsmoothed

    # Grid length from -50..50 step 10 => 11
    assert len(m.x) == 11
    assert len(m.y) == 11

    # 2-D mesh should match (x,y)
    assert m.xv.shape == (11, 11)
    assert m.yv.shape == (11, 11)

    # rho/theta and fclim_2d allocated on the same grid
    assert m.rho.shape == (11, 11)
    assert m.theta.shape == (11, 11)
    assert m.fclim_2d.shape == (11, 11)

    # Dataset created from dataframe with a 'time' dimension
    assert "time" in m.ds.dims
    assert m.ds.dims["time"] == m.ts_len


# --- Tests: footprint calculation ---------------------------------------------
def test_calc_xr_footprint_outputs_nonzero_and_finite(model_unsmoothed):
    m = model_unsmoothed
    m.calc_xr_footprint()

    # fclim_2d should be computed on (x,y) grid with finite values
    assert isinstance(m.fclim_2d, xr.DataArray)
    assert m.fclim_2d.shape == (len(m.x), len(m.y))
    total = float(np.nansum(m.fclim_2d.values))
    assert total > 0.0
    assert np.isfinite(m.fclim_2d.values).all()

    # rotated_theta should broadcast time -> expect (x,y,time)
    assert set(m.rotated_theta.dims) == {"x", "y", "time"}
    assert m.rotated_theta.shape == (len(m.x), len(m.y), m.ts_len)


def test_smoothing_changes_variability_and_preserves_scale(model_unsmoothed, model_smoothed):
    """
    Gaussian smoothing should generally reduce variability; overall scale should be similar.
    """
    m_raw = model_unsmoothed
    m_raw.calc_xr_footprint()

    m_sm = model_smoothed
    m_sm.calc_xr_footprint()

    raw = m_raw.fclim_2d.values
    sm = m_sm.fclim_2d.values

    # Variability decreases with smoothing
    assert np.nanstd(sm) < np.nanstd(raw)

    # Totals should be in the same ballpark (allow some slack due to filter)
    raw_sum = float(np.nansum(raw))
    sm_sum = float(np.nansum(sm))
    ratio = sm_sum / raw_sum if raw_sum > 0 else 1.0
    assert 0.85 <= ratio <= 1.15


# --- Tests: contour post-processing -------------------------------------------
def test_smooth_and_contour_produces_expected_vars(model_smoothed):
    """
    After computing the climatology, smooth_and_contour() should return a dataset
    with expected contour variables on the same grid.
    """
    m = model_smoothed
    m.calc_xr_footprint()

    # Choose a simpler set of contours for assertion
    m.rs = [0.2, 0.5, 0.8]
    ds_contours = m.smooth_and_contour()

    # Variables present and shapes match the grid
    for pct in (20, 50, 80):
        var = f"contour_{pct}"
        assert var in ds_contours.data_vars
        da = ds_contours[var]
        assert da.shape == (len(m.x), len(m.y))
        # At least some non-zero cells exist
        assert bool((da.values > 0).any())
