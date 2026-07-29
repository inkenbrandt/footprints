"""
Focused unit tests for the `FFPModel` class (improved_ffp.py).

Covered items
-------------
1.  Data‑frame / argument validation helpers
2.  Low‑level maths (scaled peak distance, cross‑wind footprint mask)
3.  End‑to‑end `run()`  smoke test on a tiny domain
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

from fluxfootprints.improved_ffp import FFPModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def quiet_logger():
    lg = logging.getLogger("ffp_test")
    lg.addHandler(logging.NullHandler())
    lg.setLevel(logging.CRITICAL)
    return lg


@pytest.fixture
def minimal_df():
    """A 2-row DataFrame with all required standardized meteorological fields."""
    return pd.DataFrame(
        {
            "sigmav": [0.40, 0.45],    # σv [m s⁻¹]
            "ustar": [0.30, 0.32],     # u* [m s⁻¹]
            "ol": [200.0, 150.0],      # L [m]
            "wind_dir": [45.0, 90.0],  # wind direction [°]
            "umean": [4.0, 4.5],       # wind speed [m s⁻¹]
            "zm": [2.5, 2.5],          # measurement height [m]
            "z0": [0.05, 0.05],        # roughness length [m]
            "h": [1000.0, 1000.0],     # boundary layer height [m]
        },
        index=pd.date_range("2025-05-01", periods=2, freq="30min"),
    )


@pytest.fixture
def valid_df():
    """A valid DataFrame with standardized inputs."""
    index = pd.date_range("2024-01-01", periods=2, freq="30min")
    data = {
        "sigmav": [0.5, 0.6],
        "ustar": [0.25, 0.30],
        "ol": [-100.0, -200.0],
        "wind_dir": [180.0, 190.0],
        "umean": [2.0, 2.5],
        "zm": [2.0, 2.0],
        "z0": [0.03, 0.03],
        "h": [1500.0, 1500.0],
    }
    return pd.DataFrame(data, index=index)


@pytest.fixture
def seasonal_df():
    """DataFrame spanning 4 timesteps with time-varying measurement heights (zm)."""
    index = pd.date_range("2025-04-01", periods=4, freq="30min")
    return pd.DataFrame(
        {
            "sigmav": [0.40, 0.45, 0.38, 0.42],
            "ustar": [0.30, 0.32, 0.28, 0.31],
            "ol": [200.0, 150.0, -100.0, 300.0],
            "wind_dir": [45.0, 90.0, 135.0, 180.0],
            "umean": [4.0, 4.5, 3.5, 5.0],
            "zm": [1.8, 1.8, 2.8, 2.8],  # Varying measurement height
            "z0": [0.02, 0.02, 0.02, 0.02],
            "h": [1500.0, 1500.0, 1500.0, 1500.0],
        },
        index=index,
    )

@pytest.fixture
def tiny_model(minimal_df, quiet_logger):
    """FFPModel on a 3 × 3 grid for fast tests."""
    return FFPModel(
        minimal_df,
        domain=[-100.0, 100.0, -100.0, 100.0],
        dx=100.0,  # => 3 points per axis
        dy=100.0,
        smooth_data=False,
        verbosity=0,
        logger=quiet_logger,
    )


# ---------------------------------------------------------------------------
# 1. Validation helpers
# ---------------------------------------------------------------------------
def test_missing_column_raises(minimal_df, quiet_logger):
    bad_df = minimal_df.drop(columns=["ustar"])
    with pytest.raises(ValueError, match="Missing required columns"):
        FFPModel(bad_df, dx=100.0, logger=quiet_logger, smooth_data=False)


@pytest.mark.parametrize(
    "domain",
    [
        [0, 1, 2],  # wrong length
        [10, -10, -50, 50],  # xmin ≥ xmax
    ],
)
def test_validate_domain_errors(tiny_model, domain):
    with pytest.raises(ValueError):
        tiny_model._validate_domain(domain)


def test_validate_rs_sort_and_bounds(tiny_model):
    rs = tiny_model._validate_rs([0.8, 0.1, 0.5])
    assert rs == sorted(rs)  # sorted
    with pytest.raises(ValueError):
        tiny_model._validate_rs([0.0, 0.5])  # includes 0 ⇒ invalid


# ---------------------------------------------------------------------------
# 2. Maths sanity checks
# ---------------------------------------------------------------------------


def test_crosswind_integrated_mask(tiny_model):
    """
    For X* ≤ d (≈ 0.136) the model sets F̂y* = 0.
    The grid point at (0, 0) always satisfies that condition.
    """
    f_star = tiny_model.calc_crosswind_integrated_footprint(tiny_model.rho)
    centre_val = f_star.sel(x=0.0, y=0.0).item()
    assert centre_val == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 3. End‑to‑end run
# ---------------------------------------------------------------------------
def test_run_basic_outputs(tiny_model):
    """Smoke‑test the full workflow on a tiny grid."""
    results = tiny_model.run(return_result=True)

    # Dataset integrity
    assert isinstance(results, xr.Dataset)
    assert "footprint_climatology" in results

    fclim = results["footprint_climatology"]
    assert fclim.shape == (3, 3)  # 3 × 3 grid
    assert np.all(fclim.values >= 0)
    assert fclim.values.sum() > 0


def test_initialization(valid_df):
    model = FFPModel(valid_df)
    assert isinstance(model.df, pd.DataFrame)
    assert model.dx == 10.0
    assert model.dy == 10.0
    assert all(col in model.df.columns for col in ["sigmav", "ustar", "ol"])


def test_calc_scaled_x(valid_df):
    model = FFPModel(valid_df)
    x = np.array([10.0, 20.0])
    result = model.calc_scaled_x(x)
    assert isinstance(result, np.ndarray)
    assert result.shape == x.shape


def test_calc_scaled_x_matches_theory(valid_df):
    """
    ``calc_scaled_x`` must implement Eq. 7 of Kljun et al. (2015), i.e.
    ``X* = x/zm * (1 - zm/h) / Pi_4`` with ``Pi_4 = ln(zm/z0) - psi_M``.

    Two consequences are checked:

    1. Evaluating it at the analytic real-scale peak ``xmax`` (Eq. 22) must
       return the scaled peak ``X*max = -c/b + d = 0.87`` — this guards against
       the historical von-Karman (factor k**2) and missing-psi_M bug.
    2. It must be the exact inverse of ``scale_to_real_distance`` (Eq. 6/7),
       keeping the source-area contour helpers consistent with the main
       climatology path.
    """
    # Build a model and evaluate against its own derived quantities.
    model = FFPModel(valid_df)
    zm = float(model.ds["zm"].mean())
    h = float(model.ds["h"].mean())
    pi4 = float(model.calc_pi_4().mean())

    x_star_max = -model.c / model.b + model.d  # 0.87
    xmax = x_star_max * zm * (1 - zm / h) ** -1 * pi4

    # (1) real peak -> scaled peak
    assert model.calc_scaled_x(xmax) == pytest.approx(x_star_max, rel=1e-6)

    # (2) round-trip inverse of scale_to_real_distance
    back = float(model.scale_to_real_distance(x_star_max).mean())
    assert back == pytest.approx(xmax, rel=1e-6)


def test_calc_crosswind_spread(valid_df):
    model = FFPModel(valid_df)
    x = np.array([10.0, 20.0])
    sigma_y = model.calc_crosswind_spread(x)
    assert isinstance(sigma_y, np.ndarray)
    assert np.all(sigma_y > 0)


def test_calc_crosswind_integrated_footprint(valid_df):
    model = FFPModel(valid_df)
    x_star = xr.DataArray(np.array([1.0, 2.0]))
    result = model.calc_crosswind_integrated_footprint(x_star)
    assert isinstance(result, xr.DataArray)
    assert np.all(result >= 0)


def test_calc_xr_footprint(valid_df):
    model = FFPModel(valid_df)
    fclim = model.calc_xr_footprint()
    assert isinstance(fclim, xr.DataArray)
    assert fclim.shape == (len(model.x), len(model.y))
    assert not np.all(np.isnan(fclim))


# ---------------------------------------------------------------------------
# Direct zm / z0 input (issue #7)
# ---------------------------------------------------------------------------


def test_all_invalid_zm_raises(valid_df, quiet_logger):
    """When all zm values are <= 0, the model should raise a ValueError."""
    bad_df = valid_df.copy()
    bad_df["zm"] = -1.0  # Or 0.0

    with pytest.raises(ValueError, match="All timesteps were dropped"):
        FFPModel(bad_df, verbosity=0, logger=quiet_logger)


def test_partial_invalid_zm_filtered(valid_df, quiet_logger):
    """Non-positive zm in individual rows should be filtered out without crashing."""
    df = valid_df.copy()
    # Make the first row invalid, leave second row valid (2.0)
    df.loc[df.index[0], "zm"] = 0.0

    model = FFPModel(df, verbosity=0, logger=quiet_logger)

    # First row should be dropped, leaving 1 valid timestep
    assert model.ts_len == 1


def test_direct_zm_z0_run_produces_footprint(valid_df, quiet_logger):
    """End-to-end smoke test for the zm/z0 direct-input path."""
    model = FFPModel(
        valid_df,
        zm=1.8,
        z0=0.025,
        atm_bound_height=1500.0,
        domain=[-100.0, 100.0, -100.0, 100.0],
        dx=100.0,
        smooth_data=False,
        verbosity=0,
        logger=quiet_logger,
    )
    results = model.run(return_result=True)
    assert results is not None
    assert "footprint_climatology" in results
    assert results["footprint_climatology"].values.sum() > 0


