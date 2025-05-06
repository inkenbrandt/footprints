"""
Unit tests for the `ffp_climatology_new` class inside ffp_xr.py

Covered functionality
---------------------
1.  Successful end‑to‑end run (`run`)
2.  Input‑sanity filtering & `ts_len`
3.  Grid‑generation logic (dx vs. nx/ny priority)
4.  Fatal‑error branch when required columns are missing
"""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from fluxfootprints import ffp_xr


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def logger():
    """A quiet logger instance for the model."""
    lg = logging.getLogger("ffp_test")
    lg.addHandler(logging.NullHandler())
    lg.setLevel(logging.CRITICAL)
    return lg


@pytest.fixture
def valid_df():
    """Minimal DataFrame that satisfies all model requirements."""
    return pd.DataFrame(
        {
            # required fields (will be renamed internally)
            "V_SIGMA": [0.4, 0.5],
            "USTAR": [0.3, 0.6],
            "MO_LENGTH": [120.0, 150.0],
            "wd": [45.0, 90.0],
            # optional extras
            "ws": [4.0, 4.5],
            "crop_height": [0.2, 0.2],
            "atm_bound_height": [2000.0, 2000.0],
            "inst_height": [2.0, 2.0],
        },
        index=pd.date_range("2025-05-01",
                            periods=2,
                            freq="30min"),
    )


# ---------------------------------------------------------------------------
# 1. Successful run
# ---------------------------------------------------------------------------
def test_run_basic(valid_df, logger):
    """End‑to‑end smoke test: `run` should produce a non‑negative raster."""
    model = ffp_xr.ffp_climatology_new(
        valid_df,
        domain=[-100.0, 100.0, -100.0, 100.0],
        dx=20.0,
        smooth_data=False,  # keep it light for CI
        logger=logger,
    )
    model.run()

    # `fclim_2d` must exist, match the domain grid, and be non‑negative
    assert isinstance(model.fclim_2d, xr.DataArray)
    assert model.fclim_2d.shape == model.rho.shape
    assert np.all(model.fclim_2d.values >= 0)
    # Should not be all zeros
    assert model.fclim_2d.values.sum() > 0


# ---------------------------------------------------------------------------
# 2. Input filtering & ts_len
# ---------------------------------------------------------------------------
def test_input_filtering_drops_low_ustar(valid_df, logger):
    """Rows with u* ≤ 0.05 are discarded during preprocessing."""
    df = valid_df.copy()
    df.loc[df.index[0], "USTAR"] = 0.03  # below threshold → should be removed
    model = ffp_xr.ffp_climatology_new(df, smooth_data=False, logger=logger)
    # After filtering we expect only one time‑step left
    assert model.ts_len == 1


# ---------------------------------------------------------------------------
# 3. Grid generation priority
# ---------------------------------------------------------------------------
def test_grid_from_nx(logger, valid_df):
    """
    If `dx` is None and `nx` is supplied, the grid should contain `nx+1` nodes
    across the specified domain.
    """
    dom = [-50.0, 50.0, -50.0, 50.0]
    nx = ny = 20
    model = ffp_xr.ffp_climatology_new(
        valid_df,
        domain=dom,
        dx=None,
        nx=nx,
        ny=ny,
        smooth_data=False,
        logger=logger,
    )
    # Each axis includes both endpoints ⇒ length == nx + 1
    assert len(model.x) == nx + 1
    assert len(model.y) == ny + 1
    # Meshgrid dimensions must match those lengths
    assert model.xv.shape == (nx + 1, ny + 1)


# ---------------------------------------------------------------------------
# 4. Missing‑column error path
# ---------------------------------------------------------------------------
def test_missing_required_column_raises(valid_df, logger):
    """Omitting a mandatory field should raise `ValueError` (error code 1)."""
    df = valid_df.drop(columns=["V_SIGMA"])  # removes sigmav surrogate
    with pytest.raises(ValueError, match="FFP Exception 1"):
        ffp_xr.ffp_climatology_new(df, smooth_data=False, logger=logger)
