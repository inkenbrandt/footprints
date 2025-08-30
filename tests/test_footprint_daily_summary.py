# test_footprint_daily_summary.py
# pytest -q

import math
import sys
import types
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
# Allow import whether the module is installed as a package or present as a file
try:
    from fluxfootprints import footprint_daily_summary as fds
except Exception:
    import importlib

    try:
        fds = importlib.import_module("footprint_daily_summary")
    except Exception as e:
        raise RuntimeError(
            "Could not import 'footprint_daily_summary'. "
            "Place this test next to footprint_daily_summary.py or install the package."
        ) from e


# ----------------------------
# Helper stubs / fixtures
# ----------------------------


class _FakeArray:
    """Minimal xarray.DataArray-like wrapper exposing .values."""

    def __init__(self, arr):
        self.values = np.asarray(arr, dtype=float)


class FakeFFPGood:
    """
    Stub FFPClass that produces a uniform 2x2 footprint over centers (-0.5,0.5).
    With dx=1,dy=1 the final polygon should be a 2x2 square (area=4).
    """

    def __init__(
        self,
        df,
        domain,
        dx,
        dy,
        rs,
        smooth_data,
        crop,
        inst_height,
        canopy_height,
        verbosity,
        logger,
    ):
        self.df = df
        self.domain = domain
        self.dx = dx
        self.dy = dy
        self.rs = rs
        self.logger = logger
        # grid centers
        self.x = np.array([-0.5, 0.5])
        self.y = np.array([-0.5, 0.5])
        # uniform field -> after normalization, level picks all cells
        self._arr = np.ones((2, 2), dtype=float)

    def run(self):
        self.fclim_2d = _FakeArray(self._arr)


class FakeFFPZero(FakeFFPGood):
    """Stub that yields all zeros so the day is skipped."""

    def run(self):
        self.fclim_2d = _FakeArray(np.zeros((2, 2), dtype=float))


# Utility to build a day of half-hourly (48) or hourly (24) records
def _make_df(n_per_day=24, start="2025-08-01"):
    start = pd.to_datetime(start)
    freq = "H" if n_per_day == 24 else "30min"
    idx = pd.date_range(start, periods=n_per_day, freq=freq)
    # minimal columns; FakeFFP doesn't read them, only index length matters
    return pd.DataFrame({"dummy": np.arange(len(idx))}, index=idx)


# ----------------------------
# Tests for private helpers
# ----------------------------


def test_level_for_fraction_all_zero_returns_nan():
    arr = np.zeros((3, 3), dtype=float)
    t = fds._level_for_fraction(arr, 0.8)
    assert math.isnan(t)


def test_mask_to_polygon_area_and_parts():
    xs = np.array([0.0, 1.0])
    ys = np.array([0.0, 1.0])
    mask = np.array([[True, True], [True, True]])
    mp = fds._mask_to_polygon(xs, ys, mask, dx=1.0, dy=1.0)
    # 4 cells of area 1 each -> union area = 4
    assert mp.area == pytest.approx(4.0)
    # merged into a single polygon
    assert len(mp.geoms) == 1


def test_major_minor_axes_square():
    from shapely.geometry import box

    poly = box(-1, -1, 1, 1)  # square, width=2, height=2
    major, minor, angle = fds._major_minor_axes(poly)
    assert major == pytest.approx(2.0)
    assert minor == pytest.approx(2.0)
    # For an axis-aligned square, angle should be ~0 or 90; accept either
    assert any(np.isclose(angle, v, atol=1e-6) for v in (0.0, 90.0, -90.0, 180.0))


# ----------------------------
# Tests for daily_source_area_summary
# ----------------------------


def test_daily_summary_basic_one_day_uniform_field():
    df = _make_df(24, "2025-08-01")
    domain = np.array([-1, 1, -1, 1], dtype=float)

    summary, gpkg = fds.daily_source_area_summary(
        df=df,
        FFPClass=FakeFFPGood,
        domain=domain,
        dx=1.0,
        dy=1.0,
        fraction=0.8,
        min_records=24,
        inst_height=3.66,
        canopy_height=0.5,
        logger=None,
        save_gpkg=None,
    )

    assert gpkg is None
    assert len(summary) == 1
    row = summary.iloc[0]
    # date format ISO
    assert row["date"] == "2025-08-01"
    assert row["n_obs"] == 24
    assert row["area_m2"] == pytest.approx(4.0)
    assert row["centroid_x"] == pytest.approx(0.0)
    assert row["centroid_y"] == pytest.approx(0.0)
    assert row["major_axis_m"] == pytest.approx(2.0)
    assert row["minor_axis_m"] == pytest.approx(2.0)
    assert isinstance(row["orientation_deg_from_x"], float)
    assert row["poly_parts"] == 1


def test_daily_summary_skips_days_with_insufficient_records():
    df = _make_df(10, "2025-08-01")  # fewer than min_records
    domain = np.array([-1, 1, -1, 1], dtype=float)

    summary, _ = fds.daily_source_area_summary(
        df=df,
        FFPClass=FakeFFPGood,
        domain=domain,
        dx=1.0,
        dy=1.0,
        fraction=0.8,
        min_records=24,
        save_gpkg=None,
    )

    assert summary.empty


def test_daily_summary_skips_zero_footprint_days():
    # 24 records but model sum = 0 -> skipped
    df = _make_df(24, "2025-08-01")
    domain = np.array([-1, 1, -1, 1], dtype=float)

    summary, _ = fds.daily_source_area_summary(
        df=df,
        FFPClass=FakeFFPZero,
        domain=domain,
        dx=1.0,
        dy=1.0,
        fraction=0.8,
        min_records=24,
        save_gpkg=None,
    )
    assert summary.empty


def test_daily_summary_multiple_days_mixed_outcomes():
    # Day 1: good; Day 2: zero footprint -> skipped
    df1 = _make_df(24, "2025-08-01")
    df2 = _make_df(24, "2025-08-02")
    df = pd.concat([df1, df2])
    domain = np.array([-1, 1, -1, 1], dtype=float)

    # Select FFP per day by looking at date (simple switch)
    class SwitchFFP(FakeFFPGood):
        def __init__(self, df, **kwargs):
            first_day = df.index[0].normalize().date().isoformat()
            if first_day == "2025-08-02":
                # Override to zero
                self.df = df
                self.domain = kwargs.get("domain")
                self.dx = kwargs.get("dx")
                self.dy = kwargs.get("dy")
                self.x = np.array([-0.5, 0.5])
                self.y = np.array([-0.5, 0.5])
                self._arr = np.zeros((2, 2), dtype=float)
            else:
                super().__init__(df=df, **kwargs)

    summary, _ = fds.daily_source_area_summary(
        df=df,
        FFPClass=SwitchFFP,
        domain=domain,
        dx=1.0,
        dy=1.0,
        fraction=0.8,
        min_records=24,
        save_gpkg=None,
    )

    # Only the first day should appear
    assert len(summary) == 1
    assert summary.iloc[0]["date"] == "2025-08-01"


def test_daily_summary_writes_geopackage_via_monkeypatch(monkeypatch):
    df = _make_df(24, "2025-08-01")
    domain = np.array([-1, 1, -1, 1], dtype=float)

    # Dummy sink that collects writes without touching disk
    class DummySink:
        def __init__(self, *args, **kwargs):
            self.records = []
            self.closed = False

        def write(self, rec):
            self.records.append(rec)

        def close(self):
            self.closed = True

    def fake_fiona_open(*args, **kwargs):
        return DummySink()

    # Patch fiona.open used inside the module
    monkeypatch.setattr(fds.fiona, "open", fake_fiona_open)

    summary, gpkg_path = fds.daily_source_area_summary(
        df=df,
        FFPClass=FakeFFPGood,
        domain=domain,
        dx=1.0,
        dy=1.0,
        fraction=0.8,
        min_records=24,
        save_gpkg="ignored.gpkg",
        layer_name="daily_source_area",
    )

    # We can't access the DummySink instance directly after function returns,
    # but we can at least assert the summary is correct and gpkg_path is echoed.
    assert gpkg_path == "ignored.gpkg"
    assert len(summary) == 1


# ----------------------------
# Validation / error handling
# ----------------------------


def test_daily_summary_requires_datetime_index():
    df = pd.DataFrame({"dummy": [1, 2, 3]}, index=[1, 2, 3])  # not DatetimeIndex
    with pytest.raises(ValueError, match="DatetimeIndex"):
        fds.daily_source_area_summary(
            df=df,
            FFPClass=FakeFFPGood,
            domain=np.array([-1, 1, -1, 1], dtype=float),
            dx=1.0,
            dy=1.0,
        )
