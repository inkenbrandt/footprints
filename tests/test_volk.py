"""
Unit‑tests for lightweight helpers in volk.py

Covered items
-------------
* norm_minmax_dly_et
* norm_dly_et
* mask_fp_cutoff
* snap_centroid
* read_compiled_input     – happy‑path & missing‑column branch
"""

from __future__ import annotations

import builtins
import csv
import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


# -----------------------------------------------------------------------------
# 0.  Build **minimal stubs** for heavy optional deps so that volk.py imports
#     cleanly in environments where those packages are absent.
# -----------------------------------------------------------------------------
def _install_stub(name: str, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    return mod


# Stubs only if a real library is missing
for _mod in [
    "cv2",
    "rasterio",
    "rasterio.warp",
    "rasterio.features",
    "pyproj",
    "affine",
    "refet",
    "fluxdataqaqc",
    "shapely",
    "shapely.geometry",
    "xarray",
]:
    if _mod not in sys.modules:
        _install_stub(_mod)

# cv2.getAffineTransform dummy (needed by volk.find_transform but not by these tests)
if "cv2" in sys.modules and not hasattr(sys.modules["cv2"], "getAffineTransform"):
    sys.modules["cv2"].getAffineTransform = lambda *a, **k: np.eye(2, 3)

# minimal Affine so volk can import
if "affine" in sys.modules and not hasattr(sys.modules["affine"], "Affine"):

    class _Affine(tuple):
        pass

    sys.modules["affine"].Affine = _Affine

# very small pyproj.Transformer stub
if "pyproj" in sys.modules and not hasattr(sys.modules["pyproj"], "Transformer"):

    class _Transformer:
        @staticmethod
        def from_crs(*_a, **_k):
            class _T:
                def transform(self, lat, lon):  # identity
                    return lat, lon

            return _T()

    sys.modules["pyproj"].Transformer = _Transformer

# -----------------------------------------------------------------------------
# 1.  Import volk **after** stubs are installed
# -----------------------------------------------------------------------------
from fluxfootprints import volk

# volk = importlib.import_module("volk")


# -----------------------------------------------------------------------------
# 2.  Plain‑math helpers
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "x, expected",
    [
        (pd.Series([2.0, 4.0, 6.0]), np.array([0.0, 0.5, 1.0])),
        (np.array([3.0, 3.0, 3.0]), np.array([np.nan, np.nan, np.nan])),
    ],
)
def test_norm_minmax_dly_et(x, expected):
    out = volk.norm_minmax_dly_et(x)
    # NaN‑aware compare
    assert np.allclose(out, expected, equal_nan=True)


def test_norm_dly_et():
    x = np.array([1.0, 2.0, 3.0])
    out = volk.norm_dly_et(x)
    assert out.sum() == pytest.approx(1.0, rel=1e-12)
    np.testing.assert_allclose(out, np.round(x / x.sum(), 4))


# -----------------------------------------------------------------------------
# 3.  mask_fp_cutoff
# -----------------------------------------------------------------------------
def test_mask_fp_cutoff_retains_core():
    """
    For a 2×2 array whose top value contributes ≥75 % of total,
    an 80 % cutoff should keep exactly two cells.
    """
    arr = np.array([[9.0, 1.0], [1.0, 1.0]])
    masked = volk.mask_fp_cutoff(arr, cutoff=0.8)
    assert (masked > 0).sum() == 2  # kept cells
    assert masked[0, 0] == 9.0  # peak untouched
    # All suppressed cells should equal the replacement constant (0.0)
    suppressed = masked[masked == 0.0]
    assert suppressed.size == 2


# -----------------------------------------------------------------------------
# 4.  snap_centroid
# -----------------------------------------------------------------------------
def test_snap_centroid_rules():
    x_in, y_in = 100.3, 200.4  # arbitrary non‑aligned coords
    x_out, y_out = volk.snap_centroid(x_in, y_in)

    # Output should land exactly on multiples of 15 m
    assert x_out % 15 == 0 and y_out % 15 == 0
    # …and specifically on **odd** multiples of 15
    assert ((x_out / 15) % 2) == 1
    assert ((y_out / 15) % 2) == 1
    # Distance moved is ≤ half‑cell (≤ 15 m)
    assert abs(x_out - x_in) <= 15 and abs(y_out - y_in) <= 15


# -----------------------------------------------------------------------------
# 5.  read_compiled_input
# -----------------------------------------------------------------------------
@pytest.fixture
def csv_good(tmp_path: Path) -> Path:
    """Create a minimal, valid CSV that satisfies read_compiled_input."""
    idx = pd.date_range("2025-05-01 00:00", periods=3, freq="30min")
    df = pd.DataFrame(
        {
            "latitude": 40.123,
            "longitude": -111.987,
            "ET_corr": [1.2, 1.1, 1.3],
            "wind_dir": [45, 50, 40],
            "u_star": [0.3, 0.35, 0.32],
            "sigma_v": [0.5, 0.55, 0.52],
            "zm": 2.0,
            "hc": 0.2,
            "d": 0.15,
            "L": 150.0,
            "u_mean": 4.0,
        },
        index=idx,
    )
    path = tmp_path / "good.csv"
    df.to_csv(path, index_label="date")
    return path


@pytest.fixture
def csv_bad(tmp_path: Path) -> Path:
    """CSV missing a required variable ('sigma_v') → should be rejected."""
    idx = pd.date_range("2025-05-01 00:00", periods=1, freq="h")
    df = pd.DataFrame(
        {
            "latitude": 40.0,
            "longitude": -111.9,
            "ET_corr": 1.0,
            "wind_dir": 90,
            "u_star": 0.3,
            # 'sigma_v' intentionally omitted
            "zm": 2.0,
            "hc": 0.2,
            "d": 0.15,
            "L": 100.0,
            "u_mean": 3.0,
        },
        index=idx,
    )
    path = tmp_path / "bad.csv"
    df.to_csv(path, index_label="date")
    return path


def test_read_compiled_input_success(csv_good: Path):
    df, lat, lon = volk.read_compiled_input(csv_good)
    # Returned DataFrame should be hourly‑resampled
    assert isinstance(df, pd.DataFrame) and df.index.freq is None
    # Latitude & longitude echo the first row of original file
    assert lat == pytest.approx(40.123)
    assert lon == pytest.approx(-111.987)
    # All required columns present
    for col in [
        "ET_corr",
        "wind_dir",
        "u_star",
        "sigma_v",
        "d",
        "zm",
        "L",
    ]:
        assert col in df.columns


def test_read_compiled_input_missing_vars(csv_bad: Path):
    ret = volk.read_compiled_input(csv_bad)
    assert ret is None
