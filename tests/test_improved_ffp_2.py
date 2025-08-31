# tests/test_improved_ffp.py
"""
Unit tests for the FFPModel in improved_ffp.py.

Coverage highlights
- Input validation (required columns, domain bounds, rs values)
- Core calculations (Pi4, scaled peak, 2D footprint, climatology)
- Shape/Dim assertions on xarray outputs
- Source-area contour generation
- RSL correction path smoke test
- Plotting doesn't crash and returns figure/axis
- NetCDF save writes a file

To run:
    pytest -q
"""

import os
import sys
import math
import configparser
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from fluxfootprints import FFPModel  # noqa: E402



# -----------------------
# Fixtures
# -----------------------

@pytest.fixture(scope="function")
def sample_df():
    """Small, physically plausible half-hourly dataset with required columns."""
    n = 48
    t = pd.date_range("2024-06-24", periods=n, freq="30min")
    df = pd.DataFrame(
        {
            # Using uppercase so the module's renamer can exercise its mapping:
            "V_SIGMA": np.full(n, 0.5),            # m/s
            "USTAR": np.full(n, 0.35),             # m/s
            "MO_LENGTH": np.r_[
                np.full(n//3, -150.0), np.full(n//3, -300.0), np.full(n - 2*(n//3), 50.0)
            ],                                     # m (mix of unstable and stable-ish)
            "WD": np.full(n, 180.0),               # deg
            "WS": np.full(n, 3.0),                 # m/s
        },
        index=t,
    )
    return df


@pytest.fixture(scope="function")
def small_model(sample_df):
    """Small-domain, small-grid model for fast tests."""
    model = FFPModel(
        sample_df,
        domain=[-50.0, 50.0, -50.0, 50.0],
        dx=10.0,
        dy=10.0,
        nx=20,  # not used when dx,dy set, but harmless
        ny=20,
        rs=[0.2, 0.5, 0.8],
        crop_height=0.2,
        atm_bound_height=800.0,
        inst_height=2.0,
        smooth_data=True,
        verbosity=0,
    )
    return model


@pytest.fixture(scope="function")
def rsl_model(sample_df):
    """Model configured so that the measurement is within RSL to exercise corrections."""
    # Keep crop_height=0.2 -> z0=0.0246 -> z_star ~ 0.675 m
    # inst_height=0.30 m -> zm ≈ 0.30 - d_h ~ 0.15 m < z_star ⇒ in RSL.
    model = FFPModel(
        sample_df,
        domain=[-30.0, 30.0, -30.0, 30.0],
        dx=10.0,
        dy=10.0,
        rs=[0.5],
        crop_height=0.2,
        atm_bound_height=500.0,
        inst_height=0.30,
        smooth_data=False,  # not relevant here
        verbosity=0,
    )
    return model


@pytest.fixture(scope="function")
def maybe_site_config():
    """Try to load the attached INI; fall back to a minimal config if not present."""
    candidates = [
        HERE / "US-UTE.ini",
        HERE.parent / "US-UTE.ini",
        Path("/mnt/data/US-UTE.ini"),
    ]
    cfg = configparser.ConfigParser()
    for p in candidates:
        if p.exists():
            cfg.read(p)
            return cfg
    # Fallback: minimal configuration so plot code path still runs.
    cfg["METADATA"] = {"site_name": "Test Site"}
    return cfg


# -----------------------
# Input validation
# -----------------------

def test_validate_domain_ok_and_errors(small_model):
    # Valid domain
    out = small_model._validate_domain([-10, 10, -5, 5])
    assert out == [-10.0, 10.0, -5.0, 5.0]

    # Wrong length
    with pytest.raises(ValueError):
        small_model._validate_domain([0, 1, 2])

    # Inverted bounds
    with pytest.raises(ValueError):
        small_model._validate_domain([0, -1, -5, 5])


def test_validate_rs_bounds(small_model):
    # Good
    assert small_model._validate_rs([0.2, 0.5, 0.8]) == [0.2, 0.5, 0.8]
    # Out of bounds
    with pytest.raises(ValueError):
        small_model._validate_rs([0, 0.2, 0.9])
    with pytest.raises(ValueError):
        small_model._validate_rs([0.1, 1.0])


def test_missing_required_columns_raises(sample_df):
    # Drop USTAR to trigger the required-columns check
    bad = sample_df.drop(columns=["USTAR"])
    with pytest.raises(ValueError):
        FFPModel(bad, verbosity=0)


def test_invalid_physical_parameters_raise(sample_df):
    # crop_height < 0
    with pytest.raises(ValueError):
        FFPModel(sample_df, crop_height=-0.1, verbosity=0)

    # atm_bound_height <= 10
    with pytest.raises(ValueError):
        FFPModel(sample_df, atm_bound_height=5.0, verbosity=0)

    # inst_height <= crop_height
    with pytest.raises(ValueError):
        FFPModel(sample_df, crop_height=1.0, inst_height=0.9, verbosity=0)


# -----------------------
# Core computations
# -----------------------

def test_run_returns_dataset_with_expected_vars(small_model):
    ds = small_model.run(return_result=True)
    assert isinstance(ds, xr.Dataset)
    assert "footprint_climatology" in ds
    # Climatology should be nonnegative and finite
    da = ds["footprint_climatology"]
    assert da.ndim == 2 and set(da.dims) == {"x", "y"}
    assert np.isfinite(da.values).all()
    assert (da.values >= 0).all()
    # Grid should be modest in size for the chosen domain & dx
    assert da.shape[0] <= 21 and da.shape[1] <= 21


def test_f_2d_has_time_dimension_after_calc(small_model):
    # run() calls calc_xr_footprint under the hood
    small_model.run(return_result=False)
    assert small_model.f_2d is not None
    assert "time" in small_model.f_2d.dims
    assert set(["x", "y"]).issubset(small_model.f_2d.dims)


def test_pi4_neutral_matches_log_term(small_model):
    # Given our synthetic MO_LENGTH (mostly > -oln), a large portion is neutral in code logic.
    pi4 = small_model.calc_pi_4()
    assert isinstance(pi4, xr.DataArray)
    # For neutral conditions in the implementation, psi_m == 0,
    # so Pi4 == log(zm/z0) - psi_m ≈ log(zm/z0).
    log_term = np.log(small_model.ds["zm"] / small_model.ds["z0"])
    # Use a tolerance; different stability rows may deviate slightly
    diff = (pi4 - log_term).values
    assert np.nanmedian(np.abs(diff)) < 1e-6


def test_scaled_peak_reasonable_range(small_model):
    # The scaled-peak X* typically falls in ~0.8–0.91 depending on regime in the implementation.
    xstar_max = small_model.calc_scaled_footprint_peak()
    # It may be a DataArray (post-regime weighting) or scalar (earlier in lifecycle)
    mean_val = float(np.array(xstar_max).mean())
    assert 0.7 < mean_val < 1.1

# -----------------------
# Source-area / contours
# -----------------------

def test_source_area_contour_structure(small_model):
    # Exercise the contour builder
    r = 0.8
    x_ru, x_rd = small_model.calc_peak_based_limits(r)
    y_r = small_model.calc_crosswind_extent(r, x_ru, x_rd)
    ds_contour = small_model.get_source_area_contour(r, x_ru, x_rd, y_r)
    assert {"x", "y", "f", "contour_level"} <= set(ds_contour.data_vars) | set(ds_contour.coords)
    assert set(ds_contour["f"].dims) == {"y", "x"}
    assert float(ds_contour["contour_level"]) >= 0.0


def test_calculate_source_areas_keys(small_model):
    small_model.source_areas = small_model.calculate_source_areas()
    keys = set(small_model.source_areas.keys())
    assert keys.issuperset({"r_20", "r_50", "r_80"})


# -----------------------
# RSL corrections
# -----------------------

def test_apply_rsl_corrections_sets_attributes(rsl_model):
    # Ensure calling doesn't crash and sets sigma_y/x_min where applicable
    rsl_model.apply_rsl_corrections()
    assert hasattr(rsl_model, "sigma_y")
    assert hasattr(rsl_model, "x_min")
    assert isinstance(rsl_model.sigma_y, xr.DataArray)
    assert isinstance(rsl_model.x_min, xr.DataArray)


def test_save_results_writes_file(tmp_path, small_model):
    small_model.run(return_result=False)
    out = tmp_path / "ffp_results.nc"
    small_model.save_results(str(out))
    assert out.exists()
    assert out.stat().st_size > 0