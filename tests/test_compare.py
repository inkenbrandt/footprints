# tests/test_compare.py
"""
Unit tests for compare.py

These tests cover:
1.  Helper metrics  …………………………… _peak_distance, _mask_80, footprint_metrics
2.  Toy model       …………………………… run_km
3.  High‑level API  …………………………… compare_footprints, run_all_and_compare
"""
from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


from fluxfootprints import compare as cmp  # the module under test


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_grid():
    """A 3×3 grid with the peak at (+1, +1)."""
    x_1d = np.linspace(-1, 1, 3)
    xv, yv = np.meshgrid(x_1d, x_1d)
    f = np.zeros_like(xv, dtype=float)
    f[2, 2] = 1.0
    return f, xv, yv


@pytest.fixture(scope="module")
def met_df():
    """Minimal meteorological DataFrame accepted by run_km / run_ffp wrappers."""
    return pd.DataFrame(
        {
            "ol": [200.0],
            "sigmav": [0.5],
            "ustar": [0.3],
            "wind_dir": [0.0],
            "ws": [5.0],
        },
        index=pd.date_range("2025-05-01", periods=1, freq="h"),
    )


# ---------------------------------------------------------------------------
# _peak_distance
# ---------------------------------------------------------------------------
def test_peak_distance_basic(tiny_grid):
    f, xv, yv = tiny_grid
    dist = cmp._peak_distance(f, xv, yv)
    # Hypotenuse of (+1, +1)
    assert dist == pytest.approx(np.sqrt(2), rel=1e-12)


# ---------------------------------------------------------------------------
# _mask_80
# ---------------------------------------------------------------------------
def test_mask_80_threshold():
    """80 % mask should include the four largest cells in this toy array."""
    f = np.array([[4, 1], [1, 0]], dtype=float)
    mask = cmp._mask_80(f)
    assert mask.sum() == 3  # 4+1+1 = 6 (≥ 80 % of total=6)
    assert mask[0, 0] and mask[0, 1] and mask[1, 0]  # top‑three contributors


# ---------------------------------------------------------------------------
# footprint_metrics
# ---------------------------------------------------------------------------
def test_footprint_metrics_identical(tiny_grid):
    f, xv, yv = tiny_grid
    m = cmp.footprint_metrics(f, xv, yv, f, xv, yv)
    # Identical inputs → perfect match
    assert m["RMSE"] == pytest.approx(0.0)
    assert m["Peak_diff"] == pytest.approx(0.0)
    assert m["Overlap80(%)"] == pytest.approx(100.0)


def test_footprint_metrics_mismatched_shape():
    f = np.zeros((3, 3))
    g = np.zeros((4, 4))
    with pytest.raises(ValueError):
        cmp.footprint_metrics(f, f, f, g, g, g)  # shapes differ


# ---------------------------------------------------------------------------
# run_km
# ---------------------------------------------------------------------------
def test_run_km_properties(met_df):
    f, (xv, yv) = cmp.run_km(met_df, domain=(-2, 2, -2, 2), dx=1.0)
    # Output shapes must match
    assert f.shape == xv.shape == yv.shape
    # Footprint must be non‑negative and finitely normalised
    assert np.all(f >= 0)
    assert np.isfinite(f).all()
    # Normalisation check: ∑ f · dx² ≈ 1
    dx = 1.0
    assert (f.sum() * dx**2) == pytest.approx(1.0, rel=1e-5)


# ---------------------------------------------------------------------------
# compare_footprints
# ---------------------------------------------------------------------------
def test_compare_footprints_dataframe(tiny_grid):
    f_ref, xv, yv = tiny_grid
    df = cmp.compare_footprints(
        f_ref,
        (xv, yv),
        {"Self": (f_ref.copy(), (xv.copy(), yv.copy()))},
        plot=False,
    )
    assert list(df.columns) == [
        "RMSE",
        "Peak_ref",
        "Peak_other",
        "Peak_diff",
        "Overlap80(%)",
    ]
    assert df.loc["Self", "RMSE"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# run_all_and_compare  (patched to avoid heavy external deps)
# ---------------------------------------------------------------------------
def dummy_model(*args, **kwargs):
    """Return the same tiny footprint every time."""
    f = np.zeros((3, 3))
    f[1, 1] = 1.0
    xv, yv = np.meshgrid([-1, 0, 1], [-1, 0, 1])
    return f, (xv, yv)


def test_run_all_and_compare_monkeypatched(monkeypatch, met_df):
    """
    Patch run_ffp and run_ffp_xr with lightweight stubs so that
    run_all_and_compare executes without the real external models.
    """
    monkeypatch.setattr(cmp, "run_ffp", dummy_model)
    monkeypatch.setattr(cmp, "run_ffp_xr", dummy_model)

    # Keep the real KM toy model
    df = cmp.run_all_and_compare(
        met_df,
        domain=(-1, 1, -1, 1),
        dx=1.0,
        include_km=True,
        include_xr=True,
        include_volk=False,
    )

    # Expect two comparison rows (KM vs. reference, FFP_xr vs. reference)
    assert set(df.index) == {"FFP_xr", "KM"}
    assert "RMSE" in df.columns
