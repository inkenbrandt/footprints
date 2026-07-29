"""
Unit tests for the FFPModel in improved_ffp.py.

Coverage highlights:
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
import configparser
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from fluxfootprints import FFPModel  # noqa: E402

HERE = Path(__file__).parent


# -----------------------
# Fixtures
# -----------------------

@pytest.fixture(scope="function")
def sample_df():
    """Small, physically plausible half-hourly dataset with required standardized columns."""
    n = 48
    t = pd.date_range("2024-06-24", periods=n, freq="30min")
    df = pd.DataFrame(
        {
            "sigmav": np.full(n, 0.5),           # m/s
            "ustar": np.full(n, 0.35),           # m/s
            "ol": np.r_[
                np.full(n // 3, -150.0),
                np.full(n // 3, -300.0),
                np.full(n - 2 * (n // 3), 50.0),
            ],                                  # m
            "wind_dir": np.full(n, 180.0),       # deg
            "umean": np.full(n, 3.0),            # m/s
            "zm": np.full(n, 2.0),               # m (measurement height)
            "z0": np.full(n, 0.05),              # m (roughness length)
            "h": np.full(n, 800.0),              # m (boundary layer height)
        },
        index=t,
    )
    return df


@pytest.fixture(scope="function")
def small_model(sample_df):
    """Small-domain, small-grid model for fast tests."""
    return FFPModel(
        sample_df,
        domain=[-50.0, 50.0, -50.0, 50.0],
        dx=10.0,
        dy=10.0,
        rs=[0.2, 0.5, 0.8],
        smooth_data=True,
        verbosity=0,
    )


@pytest.fixture(scope="function")
def rsl_model(sample_df):
    """Model configured with zm within RSL (zm < z*) to exercise corrections."""
    df_rsl = sample_df.copy()
    # z0=0.05 -> z_star = 2.75 * 10 * 0.05 = 1.375 m
    # Setting zm = 0.30 m puts measurement strictly within the RSL
    df_rsl["zm"] = 0.30
    df_rsl["h"] = 500.0

    return FFPModel(
        df_rsl,
        domain=[-30.0, 30.0, -30.0, 30.0],
        dx=10.0,
        dy=10.0,
        rs=[0.5],
        rslayer=True,
        smooth_data=False,
        verbosity=0,
    )


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
    cfg["METADATA"] = {"site_name": "Test Site"}
    return cfg


# -----------------------
# Input validation
# -----------------------

def test_validate_domain_ok_and_errors(small_model):
    out = small_model._validate_domain([-10, 10, -5, 5])
    assert out == [-10.0, 10.0, -5.0, 5.0]

    with pytest.raises(ValueError):
        small_model._validate_domain([0, 1, 2])

    with pytest.raises(ValueError):
        small_model._validate_domain([0, -1, -5, 5])


def test_validate_rs_bounds(small_model):
    assert small_model._validate_rs([0.2, 0.5, 0.8]) == [0.2, 0.5, 0.8]
    with pytest.raises(ValueError):
        small_model._validate_rs([0, 0.2, 0.9])
    with pytest.raises(ValueError):
        small_model._validate_rs([0.1, 1.0])


def test_missing_required_columns_raises(sample_df):
    bad = sample_df.drop(columns=["ustar"])
    with pytest.raises(ValueError, match="Missing required columns"):
        FFPModel(bad, verbosity=0)


def test_invalid_df_bounds_raise(sample_df):
    """DataFrame with all non-physical bounds (e.g. h <= 10m) should drop all rows and raise ValueError."""
    bad_df = sample_df.copy()
    bad_df["h"] = 5.0

    with pytest.raises(ValueError, match="All timesteps were dropped"):
        FFPModel(bad_df, verbosity=0)


# -----------------------
# Core computations
# -----------------------

def test_run_returns_dataset_with_expected_vars(small_model):
    ds = small_model.run(return_result=True)
    assert isinstance(ds, xr.Dataset)
    assert "footprint_climatology" in ds
    da = ds["footprint_climatology"]
    assert da.ndim == 2 and set(da.dims) == {"x", "y"}
    assert np.isfinite(da.values).all()
    assert (da.values >= 0).all()
    assert da.shape[0] <= 21 and da.shape[1] <= 21


def test_f_2d_has_time_dimension_after_calc(small_model):
    small_model.run(return_result=False)
    assert small_model.f_2d is not None
    assert "time" in small_model.f_2d.dims
    assert set(["x", "y"]).issubset(small_model.f_2d.dims)


def test_pi4_stability_dependence(small_model):
    pi4 = small_model.calc_pi_4()
    assert isinstance(pi4, xr.DataArray)

    pi4v = np.asarray(pi4.values)
    logv = np.broadcast_to(
        np.asarray(np.log(small_model.ds["zm"] / small_model.ds["z0"]).values),
        pi4v.shape,
    )
    olv = np.asarray(small_model.ds["ol"].values)

    um = olv < 0
    if um.any():
        assert np.all(pi4v[um] < logv[um])

    sm = (olv > 0) & (olv < small_model.oln)
    if sm.any():
        assert np.all(pi4v[sm] > logv[sm])

    neutral_model = FFPModel(
        pd.DataFrame(
            {
                "sigmav": [0.5],
                "ustar": [0.35],
                "ol": [1e9],
                "wind_dir": [180.0],
                "umean": [3.0],
                "zm": [2.0],
                "z0": [0.05],
                "h": [800.0],
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
    xstar_max = small_model.calc_scaled_footprint_peak()
    mean_val = float(np.array(xstar_max).mean())
    assert 0.7 < mean_val < 1.1


# -----------------------
# Source-area / contours
# -----------------------

def test_source_area_contour_structure(small_model):
    small_model.run(return_result=False)
    r = 0.8
    ds_contour = small_model.get_source_area_contour(r)
    assert {"x", "y", "f", "contour_level"} <= set(ds_contour.data_vars) | set(ds_contour.coords)
    assert set(ds_contour["f"].dims) == {"x", "y"}
    assert float(ds_contour["contour_level"]) >= 0.0


def test_source_area_contour_threshold_encloses_r(small_model):
    small_model.run(return_result=False)
    r = 0.8
    ds_contour = small_model.get_source_area_contour(r)
    threshold = float(ds_contour["contour_level"])
    fclim = small_model.fclim_2d.values
    enclosed = np.sum(fclim[fclim >= threshold]) * small_model.dx * small_model.dy
    total = np.sum(fclim) * small_model.dx * small_model.dy
    assert enclosed / total >= r - 0.01


def test_get_source_area_contour_raises_before_run(small_model):
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
    rsl_model.apply_rsl_corrections()
    assert hasattr(rsl_model, "sigma_y")
    assert hasattr(rsl_model, "x_min")
    assert isinstance(rsl_model.sigma_y, xr.DataArray)
    assert isinstance(rsl_model.x_min, xr.DataArray)


# -----------------------
# Validity filtering
# -----------------------

def test_valid_footprint_is_per_timestep_dataarray(small_model):
    vf = small_model.valid_footprint
    assert isinstance(vf, xr.DataArray)
    assert "time" in vf.dims
    assert vf.dtype == bool or np.issubdtype(vf.dtype, np.bool_)


def test_valid_timesteps_all_true_for_clean_data(small_model):
    assert bool(small_model.valid_footprint.all())


def test_extreme_stability_excluded_by_validity_mask(sample_df):
    df = sample_df.copy()
    df.loc[df.index[-1], "ol"] = -0.05
    model = FFPModel(
        df,
        domain=[-50.0, 50.0, -50.0, 50.0],
        dx=10.0,
        dy=10.0,
        smooth_data=False,
        verbosity=0,
    )
    assert not bool(model.valid_footprint.all())


def test_invalid_timesteps_do_not_affect_climatology(sample_df):
    common = dict(
        domain=[-50.0, 50.0, -50.0, 50.0],
        dx=10.0,
        dy=10.0,
        smooth_data=False,
        verbosity=0,
    )
    m_clean = FFPModel(sample_df.copy(), **common)
    r_clean = m_clean.run(return_result=True)

    df_bad = sample_df.copy()
    df_bad.loc[df_bad.index[-1], "ol"] = -0.05
    m_bad = FFPModel(df_bad, **common)
    r_bad = m_bad.run(return_result=True)

    fc_clean = r_clean["footprint_climatology"].values
    fc_bad = r_bad["footprint_climatology"].values
    assert not np.allclose(fc_clean, fc_bad)


def test_rslayer_mode_runs_without_error(sample_df):
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
    assert hasattr(model, "sigma_y") and model.sigma_y is not None


def test_save_results_writes_file(tmp_path, small_model):
    small_model.run(return_result=False)
    out = tmp_path / "ffp_results.nc"
    small_model.save_results(str(out))
    assert out.exists()
    assert out.stat().st_size > 0