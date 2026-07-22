import pytest
import pandas as pd
import numpy as np
import xarray as xr

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from fluxfootprints.ffp_xr import ffp_climatology_new


@pytest.fixture
def minimal_df():
    return pd.DataFrame(
        {
            "sigmav": [0.4, 0.5],
            "ustar": [0.3, 0.35],
            "ol": [50.0, 60.0],
            "wind_dir": [180.0, 190.0],
            "umean": [2.5, 3.0],
            "zm": [2.0, 2.0],
            "z0": [0.1, 0.1],
            "h": [2000.0, 2000.0],
        },
        index=pd.date_range("2025-01-01", periods=2, freq="30min"),
    )


@pytest.fixture
def small_model(minimal_df):
    return ffp_climatology_new(
        minimal_df,
        domain=[-100.0, 100.0, -100.0, 100.0],
        dx=10.0,
        dy=10.0,
        smooth_data=False,
        verbosity=0,
    )


def test_init_sets_parameters(small_model):
    m = small_model
    assert m.df is not None
    # domain [-100,100] with dx=10 → 21 points per axis
    assert m.xv.shape == (21, 21)
    assert isinstance(m.ds, xr.Dataset)


def test_prep_df_fields_creates_fields(small_model):
    m = small_model
    assert "zm" in m.df.columns
    assert "h" in m.df.columns
    # Both rows should survive filtering
    assert len(m.df) == 2


def test_raise_ffp_exception_fatal(small_model):
    with pytest.raises(ValueError, match="FFP Exception 1"):
        small_model.raise_ffp_exception(1)


def test_define_domain_sets_grids(small_model):
    m = small_model
    m.define_domain()
    assert isinstance(m.xv, np.ndarray)
    assert isinstance(m.rho, xr.DataArray)
    assert m.rho.shape == m.xv.shape


def test_create_xr_dataset_from_dataframe(small_model):
    m = small_model
    m.create_xr_dataset()
    assert isinstance(m.ds, xr.Dataset)
    assert "sigmav" in m.ds.variables


def test_calc_xr_footprint_executes(small_model):
    m = small_model
    m.calc_xr_footprint()
    assert isinstance(m.f_2d, xr.DataArray)
    assert m.fclim_2d.shape == m.f_2d.isel(time=0).shape


def test_run_returns_dataset(small_model):
    result = small_model.run(return_result=True)
    assert isinstance(result, xr.Dataset)
    assert "footprint_climatology" in result
    assert "domain_x" in result
    assert "domain_y" in result
