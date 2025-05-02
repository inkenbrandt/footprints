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
    """A 2‑row DataFrame with all required meteorological fields."""
    df = pd.DataFrame(
        {
            "V_SIGMA": [0.40, 0.45],  # σv  [m s⁻¹]
            "USTAR": [0.30, 0.32],  # u*  [m s⁻¹]
            "MO_LENGTH": [200.0, 150.0],  # L   [m]
            "WD": [45.0, 90.0],  # wind direction [°]
            "WS": [4.0, 4.5],  # wind speed [m s⁻¹]
        },
        index=pd.date_range("2025-05-01", periods=2, freq="30min"),
    )
    return df


@pytest.fixture
def tiny_model(minimal_df, quiet_logger):
    """FFPModel on a 3 × 3 grid for fast tests."""
    return FFPModel(
        minimal_df,
        domain=[-100.0, 100.0, -100.0, 100.0],
        dx=100.0,  # => 3 points per axis
        dy=100.0,
        smooth_data=False,  # keep CI fast
        verbosity=0,
        logger=quiet_logger,
    )


# ---------------------------------------------------------------------------
# 1. Validation helpers
# ---------------------------------------------------------------------------
def test_missing_column_raises(minimal_df, quiet_logger):
    bad_df = minimal_df.drop(columns=["USTAR"])
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
def test_scaled_peak_constant(tiny_model):
    """Equation 20 gives X*ₘₐₓ ≈ 0.87 (independent of input)."""
    peak_star = tiny_model.calc_scaled_footprint_peak()
    assert peak_star == pytest.approx(0.87, rel=1e-2)


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
