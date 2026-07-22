# tests/test_ffp_xr.py
import importlib.util
import os
import pathlib
import sys

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
    """Build a DataFrame with 6 valid and 2 invalid rows to test filtering.

    Standardized columns: sigmav, ustar, wind_dir, ol, umean, zm, z0, h
    """
    times = pd.date_range("2025-01-01", periods=8, freq="h")
    df = pd.DataFrame(
        {
            # Last two rows invalid/filtered: negative sigmav, tiny ustar, wind_dir > 360, NaN ol
            "sigmav": [0.50, 0.60, 0.40, 0.45, 0.55, 0.52, -0.10, 0.50],
            "ustar": [0.30, 0.35, 0.25, 0.40, 0.20, 0.15, 0.01, 0.02],
            "wind_dir": [0.0, 90.0, 180.0, 270.0, 45.0, 315.0, 10.0, 400.0],
            "ol": [-100.0, -50.0, 50.0, 100.0, 10.0, -10.0, np.nan, 25.0],
            "umean": [3.0, 4.0, 5.0, 3.0, 2.0, 3.0, 3.0, 4.0],
            "zm": [2.0] * 8,
            "z0": [0.05] * 8,
            "h": [2000.0] * 8,
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
    """After __init__, invalid rows should be dropped and key fields present."""
    m = model_unsmoothed

    # Two invalid rows should be removed -> 6 valid time steps remain
    assert m.ts_len == 6

    # Standardized columns exist
    for col in ["sigmav", "ustar", "wind_dir", "ol", "z0", "zm", "h", "umean"]:
        assert col in m.df.columns

    # Physical sanity checks on columns
    assert np.all(m.df["zm"].values > 0.0)
    assert np.all(m.df["h"].values > 10.0)
    assert np.all(m.df["zm"].values < m.df["h"].values)


def test_missing_required_parameters_raise():
    """If required inputs are missing, the class should raise via raise_ffp_exception(1)."""
    times = pd.date_range("2025-01-01", periods=4, freq="h")
    df_missing = pd.DataFrame(
        {
            # 'ustar' intentionally missing
            "sigmav": [0.5, 0.6, 0.4, 0.5],
            "wind_dir": [0.0, 90.0, 180.0, 270.0],
            "ol": [-50.0, -10.0, 10.0, 50.0],
            "umean": [3.0, 4.0, 5.0, 3.0],
            "zm": [2.0] * 4,
            "z0": [0.05] * 4,
            "h": [2000.0] * 4,
        },
        index=times,
    )

    with pytest.raises(ValueError, match=r"FFP Exception 1"):
        _ = FFP(df=df_missing, domain=[-50, 50, -50, 50], dx=10.0, dy=10.0, verbosity=0)


def test_explicit_raise_ffp_exception(model_unsmoothed):
    """Exercise the explicit error path to ensure ValueError is raised for fatal codes."""
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
    assert "time" in m.ds.sizes
    assert m.ds.sizes["time"] == m.ts_len


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


# --- Tests: wind-direction coordinate convention ------------------------------
def _single_step_footprint(wind_dir, zm=2.0, z0=0.05, h=2000.0):
    """Run the climatology for a single time step at a given wind direction and

    return the footprint DataArray (dims x, y) together with its mass-weighted
    centroid (cx, cy).
    """
    times = pd.date_range("2025-01-01", periods=1, freq="h")
    df = pd.DataFrame(
        {
            "sigmav": [0.5],
            "ustar": [0.4],
            "wind_dir": [wind_dir],
            "ol": [-100.0],
            "umean": [3.0],
            "zm": [zm],
            "z0": [z0],
            "h": [h],
        },
        index=times,
    )
    m = FFP(
        df=df,
        domain=[-200.0, 200.0, -200.0, 200.0],
        dx=5.0,
        dy=5.0,
        smooth_data=False,
        verbosity=0,
    )
    m.calc_xr_footprint()

    f = m.fclim_2d
    w = np.nan_to_num(f.values)
    xv, yv = np.meshgrid(np.asarray(f["x"]), np.asarray(f["y"]), indexing="ij")
    total = w.sum()
    cx = float((xv * w).sum() / total)
    cy = float((yv * w).sum() / total)
    return f, cx, cy


@pytest.mark.parametrize(
    "wind_dir, axis, sign",
    [
        (0.0, "y", +1),  # wind FROM north -> source area upwind to the north (+y)
        (90.0, "x", +1),  # wind FROM east  -> source area to the east (+x)
        (180.0, "y", -1),  # wind FROM south -> source area to the south (-y)
        (270.0, "x", -1),  # wind FROM west  -> source area to the west (-x)
    ],
)
def test_footprint_points_upwind_for_cardinal_winds(wind_dir, axis, sign):
    """Validate the wind-coordinate convention against a known reference behaviour."""
    _, cx, cy = _single_step_footprint(wind_dir)

    if axis == "y":
        assert np.sign(cy) == sign
        assert abs(cy) > 10.0
        assert abs(cx) < 0.1 * abs(cy)
    else:
        assert np.sign(cx) == sign
        assert abs(cx) > 10.0
        assert abs(cy) < 0.1 * abs(cx)


def test_footprint_not_mirrored_for_intercardinal_wind():
    """A wind from the north-east (45deg) must place source area to north-east (+x, +y)."""
    _, cx, cy = _single_step_footprint(45.0)
    assert cx > 0.0
    assert cy > 0.0
    assert cx == pytest.approx(cy, rel=0.05)


def test_smoothing_changes_variability_and_preserves_scale(
    model_unsmoothed, model_smoothed
):
    """Gaussian smoothing should reduce variability; overall scale should be similar."""
    m_raw = model_unsmoothed
    m_raw.calc_xr_footprint()

    m_sm = model_smoothed
    m_sm.calc_xr_footprint()

    raw = m_raw.fclim_2d.values
    sm = m_sm.fclim_2d.values

    assert np.nanstd(sm) < np.nanstd(raw)

    raw_sum = float(np.nansum(raw))
    sm_sum = float(np.nansum(sm))
    ratio = sm_sum / raw_sum if raw_sum > 0 else 1.0
    assert 0.85 <= ratio <= 1.15


# --- Tests: contour post-processing -------------------------------------------
def test_smooth_and_contour_produces_expected_vars(model_smoothed):
    """smooth_and_contour() should return a dataset with expected contour variables."""
    m = model_smoothed
    m.calc_xr_footprint()

    m.rs = [0.2, 0.5, 0.8]
    ds_contours = m.smooth_and_contour()

    for pct in (20, 50, 80):
        var = f"contour_{pct}"
        assert var in ds_contours.data_vars
        da = ds_contours[var]
        assert da.shape == (len(m.x), len(m.y))
        assert bool((da.values > 0).any())


# --- Tests: Kljun validity filtering (issue #8) --------------------------------


def _base_df(n=6, ol=-100.0, wind_dir=180.0, zm=2.0, z0=0.05, h=800.0):
    """Helper: small DataFrame with all physically valid values."""
    times = pd.date_range("2025-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {
            "sigmav": [0.5] * n,
            "ustar": [0.35] * n,
            "wind_dir": [wind_dir] * n,
            "ol": [ol] * n,
            "umean": [3.0] * n,
            "zm": [zm] * n,
            "z0": [z0] * n,
            "h": [h] * n,
        },
        index=times,
    )


_FFP_KW = dict(
    domain=[-200.0, 200.0, -200.0, 200.0],
    dx=20.0,
    dy=20.0,
    smooth_data=False,
    verbosity=0,
)


def test_no_nans_in_climatology_after_validity_masking():
    """fclim_2d must be finite (no NaN/Inf) even when some timesteps are zeroed out."""
    df = _base_df(n=4, ol=-0.05)  # zm/L = 2/-0.05 = -40 << -15.5 → all excluded
    m = FFP(df=df, **_FFP_KW)
    m.calc_xr_footprint()
    assert np.isfinite(
        m.fclim_2d.values
    ).all(), "fclim_2d must not contain NaN/Inf"


def test_extreme_instability_excluded_from_climatology():
    """Extreme instability timestep (zm/L < -15.5) must be excluded from climatology."""
    df_clean = _base_df(n=5, ol=-100.0)

    extra = _base_df(n=1, ol=-0.05)
    extra.index = pd.date_range(
        df_clean.index[-1] + pd.Timedelta("1h"), periods=1, freq="h"
    )
    df_bad = pd.concat([df_clean, extra])

    m_clean = FFP(df=df_clean, **_FFP_KW)
    m_clean.calc_xr_footprint()

    m_bad = FFP(df=df_bad, **_FFP_KW)
    m_bad.calc_xr_footprint()

    assert np.allclose(
        m_clean.fclim_2d.values, m_bad.fclim_2d.values, rtol=0.01
    ), "Extreme-stability timestep (zm/L << -15.5) should be excluded."


def test_in_rsl_timesteps_excluded_when_rslayer_false():
    """When rslayer=False (default), timesteps where zm <= 27.5*z0 must be excluded."""
    df = _base_df(n=4, zm=1.0, z0=0.04)
    m = FFP(df=df, rslayer=False, **_FFP_KW)
    m.calc_xr_footprint()

    assert float(m.fclim_2d.sum()) == pytest.approx(
        0.0
    ), "All in-RSL timesteps should be excluded when rslayer=False"


def test_in_rsl_timesteps_allowed_when_rslayer_true():
    """When rslayer=True, in-RSL timesteps still contribute to climatology."""
    df = _base_df(n=4, zm=1.0, z0=0.04)

    m_off = FFP(df=df, rslayer=False, **_FFP_KW)
    m_off.calc_xr_footprint()

    m_on = FFP(df=df, rslayer=True, **_FFP_KW)
    m_on.calc_xr_footprint()

    assert (
        float(m_on.fclim_2d.sum()) > 0.0
    ), "rslayer=True should allow in-RSL timesteps"
    assert (
        float(m_off.fclim_2d.sum()) == pytest.approx(0.0)
    ), "rslayer=False should exclude in-RSL timesteps"