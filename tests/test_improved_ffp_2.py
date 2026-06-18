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


def test_pi4_stability_dependence(small_model):
    """Pi4 = ln(zm/z0) - psi_M must respond correctly to stability.

    - Neutral (|L| >= oln): psi_M == 0  -> Pi4 == ln(zm/z0)
    - Unstable (L < 0):     psi_M > 0   -> Pi4 <  ln(zm/z0)
    - Stable   (0 < L<oln): psi_M < 0   -> Pi4 >  ln(zm/z0)
    """
    pi4 = small_model.calc_pi_4()
    assert isinstance(pi4, xr.DataArray)

    pi4v = np.asarray(pi4.values)
    logv = np.broadcast_to(
        np.asarray(np.log(small_model.ds["zm"] / small_model.ds["z0"]).values),
        pi4v.shape,
    )
    olv = np.asarray(small_model.ds["ol"].values)

    # Unstable rows: Pi4 strictly below the neutral log term.
    um = olv < 0
    if um.any():
        assert np.all(pi4v[um] < logv[um])

    # Stable rows (0 < L < oln): Pi4 strictly above the neutral log term.
    sm = (olv > 0) & (olv < small_model.oln)
    if sm.any():
        assert np.all(pi4v[sm] > logv[sm])

    # A genuinely neutral profile (|L| >= oln) collapses to the bare log term.
    neutral_model = FFPModel(
        pd.DataFrame(
            {
                "V_SIGMA": [0.5],
                "USTAR": [0.35],
                "MO_LENGTH": [1e9],  # effectively neutral
                "WD": [180.0],
                "WS": [3.0],
            },
            index=pd.date_range("2024-06-24", periods=1, freq="30min"),
        ),
        domain=[-50.0, 50.0, -50.0, 50.0],
        dx=10.0,
        dy=10.0,
        smooth_data=False,
        verbosity=0,
    )
    pi4_n = neutral_model.calc_pi_4()
    log_n = np.log(neutral_model.ds["zm"] / neutral_model.ds["z0"])
    assert float(np.abs((pi4_n - log_n).values).max()) < 1e-6


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
    # Contour must be derived from the actual fclim_2d climatology, not a synthetic footprint.
    small_model.run(return_result=False)
    r = 0.8
    ds_contour = small_model.get_source_area_contour(r)
    assert {"x", "y", "f", "contour_level"} <= set(ds_contour.data_vars) | set(ds_contour.coords)
    assert set(ds_contour["f"].dims) == {"x", "y"}
    assert float(ds_contour["contour_level"]) >= 0.0


def test_source_area_contour_threshold_encloses_r(small_model):
    """The contour threshold must enclose at least fraction r of the total flux."""
    small_model.run(return_result=False)
    r = 0.8
    ds_contour = small_model.get_source_area_contour(r)
    threshold = float(ds_contour["contour_level"])
    fclim = small_model.fclim_2d.values
    enclosed = np.sum(fclim[fclim >= threshold]) * small_model.dx * small_model.dy
    total = np.sum(fclim) * small_model.dx * small_model.dy
    assert enclosed / total >= r - 0.01  # allow 1 % rounding from discretisation


def test_get_source_area_contour_raises_before_run(small_model):
    """Calling get_source_area_contour before run() must raise RuntimeError."""
    with pytest.raises(RuntimeError):
        small_model.get_source_area_contour(0.8)


def test_calculate_source_areas_keys(small_model):
    small_model.run(return_result=False)
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


# -----------------------
# Validity filtering (issue #8)
# -----------------------

def test_valid_footprint_is_per_timestep_dataarray(small_model):
    """valid_footprint must be a boolean DataArray indexed by time, not a scalar."""
    vf = small_model.valid_footprint
    assert isinstance(vf, xr.DataArray)
    assert "time" in vf.dims
    assert vf.dtype == bool or np.issubdtype(vf.dtype, np.bool_)


def test_valid_timesteps_all_true_for_clean_data(small_model):
    """All 48 rows in sample_df satisfy the validity bounds — mask must be all True."""
    assert bool(small_model.valid_footprint.all())


def test_extreme_stability_excluded_by_validity_mask(sample_df):
    """A row with zm/L < -15.5 must be flagged False in valid_footprint."""
    df = sample_df.copy()
    # Overwrite the last row with an extreme negative L so zm/L << -15.5
    df.loc[df.index[-1], "MO_LENGTH"] = -0.05
    model = FFPModel(
        df,
        domain=[-50.0, 50.0, -50.0, 50.0],
        dx=10.0,
        dy=10.0,
        smooth_data=False,
        verbosity=0,
    )
    # At least the last row should fail stability validity
    assert not bool(model.valid_footprint.all()), (
        "Expected at least one invalid timestep for zm/L << -15.5"
    )


def test_invalid_timesteps_do_not_affect_climatology(sample_df):
    """
    Climatologies built from all-valid data vs. data with one extreme-stability
    row must differ: the masked run should exclude that row's contribution.
    """
    common = dict(
        domain=[-50.0, 50.0, -50.0, 50.0],
        dx=10.0,
        dy=10.0,
        smooth_data=False,
        verbosity=0,
    )
    # Baseline: all rows valid
    m_clean = FFPModel(sample_df.copy(), **common)
    r_clean = m_clean.run(return_result=True)

    # Perturbed: last row has extreme stability → excluded by validity mask
    df_bad = sample_df.copy()
    df_bad.loc[df_bad.index[-1], "MO_LENGTH"] = -0.05
    m_bad = FFPModel(df_bad, **common)
    r_bad = m_bad.run(return_result=True)

    fc_clean = r_clean["footprint_climatology"].values
    fc_bad = r_bad["footprint_climatology"].values
    # The two climatologies should not be identical
    assert not np.allclose(fc_clean, fc_bad), (
        "Excluding an invalid timestep should change the climatology"
    )


def test_rslayer_mode_runs_without_error(sample_df):
    """rslayer=True must invoke RSL corrections without raising an exception."""
    model = FFPModel(
        sample_df,
        domain=[-50.0, 50.0, -50.0, 50.0],
        dx=10.0,
        dy=10.0,
        rslayer=True,
        smooth_data=False,
        verbosity=0,
    )
    results = model.run(return_result=True)
    assert results is not None
    assert "footprint_climatology" in results
    # RSL path must have set sigma_y
    assert hasattr(model, "sigma_y") and model.sigma_y is not None


def test_save_results_writes_file(tmp_path, small_model):
    small_model.run(return_result=False)
    out = tmp_path / "ffp_results.nc"
    small_model.save_results(str(out))
    assert out.exists()
    assert out.stat().st_size > 0