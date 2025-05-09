import pytest
import pandas as pd
import numpy as np
import xarray as xr

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from fluxfootprints.new_ffp import FFPclim


@pytest.fixture
def minimal_df():
    return pd.DataFrame(
        {
            "V_SIGMA": [0.4, 0.5],
            "USTAR": [0.3, 0.35],
            "MO_LENGTH": [50, 60],
            "wd": [180, 190],
            "ws": [2.5, 3.0],
        }
    )


def test_init_sets_parameters(minimal_df):
    model = FFPclim(minimal_df)
    assert model.df is not None
    assert model.xv.shape == (1001, 1001)
    assert isinstance(model.ds, xr.Dataset)


def test_prep_df_fields_creates_fields(minimal_df):
    model = FFPclim(minimal_df)
    assert "zm" in model.df.columns
    assert "h" in model.df.columns
    assert not model.df.isnull().any().any()


def test_raise_ffp_exception_fatal():
    model = FFPclim(
        pd.DataFrame(
            {
                "V_SIGMA": [0.4],
                "USTAR": [0.3],
                "MO_LENGTH": [50],
                "wd": [180],
                "ws": [2.5],
            }
        )
    )
    with pytest.raises(ValueError):
        model.raise_ffp_exception(1)


def test_define_domain_sets_grids(minimal_df):
    model = FFPclim(minimal_df)
    model.define_domain()
    assert isinstance(model.xv, np.ndarray)
    assert isinstance(model.rho, xr.DataArray)
    assert model.rho.shape == model.xv.shape


def test_create_xr_dataset_from_dataframe(minimal_df):
    model = FFPclim(minimal_df)
    model.create_xr_dataset()
    assert isinstance(model.ds, xr.Dataset)
    assert "sigmav" in model.ds.variables


def test_calc_xr_footprint_executes(minimal_df):
    model = FFPclim(minimal_df)
    model.calc_xr_footprint()
    assert isinstance(model.f_2d, xr.DataArray)
    assert model.fclim_2d.shape == model.f_2d.isel(time=0).shape


def test_smooth_and_contour_output(minimal_df):
    model = FFPclim(minimal_df)
    model.calc_xr_footprint()
    model.ds["footprint"] = model.f_2d
    climatology = model.smooth_and_contour()
    assert isinstance(climatology, xr.Dataset)
    for key in climatology:
        assert climatology[key].ndim == 2


def test_run_returns_expected_keys(minimal_df):
    model = FFPclim(minimal_df)
    output = model.run()
    assert set(output.keys()) >= {"x_2d", "y_2d", "fclim_2d", "f_2d", "rs"}
