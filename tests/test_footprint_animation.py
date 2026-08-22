# test_footprint_animation.py
# Run with: pytest -q

import os
import sys
import types

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

matplotlib.use("Agg")

# Import project from ../src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from fluxfootprints import footprint_animation as fa


# ----------------------------
# Fixtures
# ----------------------------


def _gaussian_stack(times, x, y, drift=2.0):
    """Footprint-like stack: a Gaussian blob drifting downwind over time."""
    xx, yy = np.meshgrid(x, y, indexing="ij")
    frames = []
    for i in range(len(times)):
        x0 = 20.0 + drift * i
        f = np.exp(-(((xx - x0) ** 2) / (2 * 25.0**2) + (yy**2) / (2 * 15.0**2)))
        frames.append(f / f.sum())
    return np.stack(frames)


@pytest.fixture
def fp_da():
    """Half-hourly (time, x, y) footprint DataArray spanning ~2.5 days."""
    times = pd.date_range("2023-06-01 00:00", periods=120, freq="30min")
    x = np.arange(-100.0, 101.0, 10.0)
    y = np.arange(-100.0, 101.0, 10.0)
    return xr.DataArray(
        _gaussian_stack(times, x, y),
        dims=("time", "x", "y"),
        coords={"time": times, "x": x, "y": y},
        name="footprint",
    )


@pytest.fixture
def fp_monthly_da():
    """Half-hourly footprints spanning three months (coarse grid, few steps)."""
    times = pd.date_range("2023-05-01", periods=90, freq="1D")
    x = np.arange(-50.0, 51.0, 25.0)
    y = np.arange(-50.0, 51.0, 25.0)
    return xr.DataArray(
        _gaussian_stack(times, x, y, drift=0.1),
        dims=("time", "x", "y"),
        coords={"time": times, "x": x, "y": y},
        name="footprint",
    )


class _FakeModel:
    """Minimal stand-in for BaseFootprintModel's timeseries accessor."""

    def __init__(self, da):
        self._da = da

    def get_footprint_timeseries(self):
        return self._da


# ----------------------------
# Frequency handling
# ----------------------------


@pytest.mark.parametrize(
    "given, expected",
    [
        ("hourly", "1h"),
        ("Hourly", "1h"),
        ("daily", "1D"),
        ("monthly", "MS"),
        ("native", None),
        (None, None),
        ("3h", "3h"),  # pandas alias passthrough
    ],
)
def test_resolve_freq(given, expected):
    assert fa.resolve_freq(given) == expected


def test_resolve_freq_rejects_non_string():
    with pytest.raises(ValueError):
        fa.resolve_freq(5)


def test_resample_hourly_halves_frame_count(fp_da):
    out = fa.resample_footprints(fp_da, freq="hourly")
    assert out.sizes["time"] == fp_da.sizes["time"] // 2
    assert set(out.dims) == {"time", "x", "y"}


def test_resample_daily_and_monthly(fp_da, fp_monthly_da):
    daily = fa.resample_footprints(fp_da, freq="daily")
    assert daily.sizes["time"] == 3  # 120 half-hours == 2.5 days

    monthly = fa.resample_footprints(fp_monthly_da, freq="monthly")
    assert monthly.sizes["time"] == 3
    assert pd.to_datetime(monthly["time"].values[0]) == pd.Timestamp("2023-05-01")


def test_resample_native_is_passthrough(fp_da):
    out = fa.resample_footprints(fp_da, freq=None)
    assert out.sizes["time"] == fp_da.sizes["time"]


def test_normalize_each_frame_sums_to_one(fp_da):
    out = fa.resample_footprints(fp_da, freq="daily", normalize_each_frame=True)
    sums = out.sum(dim=("x", "y")).values
    np.testing.assert_allclose(sums, 1.0, rtol=1e-6)


def test_resample_drops_all_nan_periods(fp_da):
    gapped = fp_da.copy()
    # Blank the whole second day so that period has no data at all.
    gapped.loc[{"time": slice("2023-06-02", "2023-06-02 23:59")}] = np.nan
    out = fa.resample_footprints(gapped, freq="daily")
    assert out.sizes["time"] == 2
    assert pd.Timestamp("2023-06-02") not in pd.to_datetime(out["time"].values)


def test_unknown_reducer_raises(fp_da):
    with pytest.raises(ValueError, match="Unknown reducer"):
        fa.resample_footprints(fp_da, freq="daily", reducer="nope")


# ----------------------------
# Input coercion
# ----------------------------


def test_accepts_model_dataset_and_dataarray(fp_da):
    for source in (fp_da, fp_da.to_dataset(), _FakeModel(fp_da)):
        anim = fa.FootprintAnimator(source, freq="daily")
        assert anim.frames.sizes["time"] == 3


def test_dataset_with_two_candidates_requires_var(fp_da):
    ds = xr.Dataset({"a": fp_da, "b": fp_da})
    with pytest.raises(ValueError, match="pass var="):
        fa.FootprintAnimator(ds, freq="daily")
    anim = fa.FootprintAnimator(ds, var="b", freq="daily")
    assert anim.frames.sizes["time"] == 3


def test_missing_dims_raise(fp_da):
    flat = fp_da.isel(time=0)
    with pytest.raises(ValueError, match="missing required dimension"):
        fa.FootprintAnimator(flat)


def test_bad_input_type_raises():
    with pytest.raises(TypeError):
        fa.FootprintAnimator([1, 2, 3])


def test_model_without_timeseries_raises():
    with pytest.raises(ValueError, match="does not expose"):
        fa.FootprintAnimator(_FakeModel(None))


# ----------------------------
# Geometry / georeferencing
# ----------------------------


def test_local_extent_uses_pixel_edges(fp_da):
    anim = fa.FootprintAnimator(fp_da, freq="daily")
    assert anim.crs is None
    # 10 m cells, coords span -100..100 -> edges at -105..105
    assert anim.extent == pytest.approx((-105.0, 105.0, -105.0, 105.0))
    assert anim.frames.dims == ("time", "y", "x")


def test_georeferenced_extent_is_shifted_to_utm(fp_da):
    anim = fa.FootprintAnimator(
        fp_da, freq="daily", station_lat=40.05, station_lon=-113.55
    )
    assert anim.crs is not None and anim.crs.is_projected
    assert anim.crs.to_epsg() == 32612  # UTM 12N
    left, right, bottom, top = anim.extent
    assert right - left == pytest.approx(210.0)
    assert top - bottom == pytest.approx(210.0)
    assert left == pytest.approx(anim.x0 - 105.0)
    assert bottom == pytest.approx(anim.y0 - 105.0)


def test_half_supplied_station_coords_raise(fp_da):
    with pytest.raises(ValueError, match="both station_lat and station_lon"):
        fa.FootprintAnimator(fp_da, freq="daily", station_lat=40.05)


def test_geographic_crs_out_rejected(fp_da):
    with pytest.raises(ValueError, match="projected CRS"):
        fa.FootprintAnimator(
            fp_da,
            freq="daily",
            station_lat=40.05,
            station_lon=-113.55,
            crs_out=4326,
        )


def test_basemap_without_station_coords_raises(fp_da):
    with pytest.raises(ValueError, match="requires station_lat"):
        fa.FootprintAnimator(fp_da, freq="daily", basemap=True)


# ----------------------------
# Color limits, masking, contours
# ----------------------------


def test_log_norm_gets_positive_vmin(fp_da):
    anim = fa.FootprintAnimator(fp_da, freq="daily", norm="log")
    assert anim.vmin > 0
    assert anim.vmax > anim.vmin


def test_log_norm_clamps_range_to_log_decades(fp_da):
    """The low percentile is many decades below vmax; log_decades wins."""
    anim = fa.FootprintAnimator(fp_da, freq="daily", norm="log", log_decades=3)
    assert anim.vmin == pytest.approx(anim.vmax / 1e3)

    wider = fa.FootprintAnimator(fp_da, freq="daily", norm="log", log_decades=6)
    assert wider.vmin == pytest.approx(wider.vmax / 1e6)
    assert wider.vmin < anim.vmin


def test_explicit_log_vmin_overrides_log_decades(fp_da):
    anim = fa.FootprintAnimator(
        fp_da, freq="daily", norm="log", vmin=1e-12, log_decades=2
    )
    assert anim.vmin == 1e-12


def test_linear_norm_ignores_log_decades(fp_da):
    anim = fa.FootprintAnimator(fp_da, freq="daily", log_decades=1)
    # 0th percentile of the positive data, not a clamped decade range
    assert anim.vmin < anim.vmax / 10.0


def test_explicit_vmin_vmax_respected(fp_da):
    anim = fa.FootprintAnimator(fp_da, freq="daily", vmin=1e-6, vmax=1e-3)
    assert (anim.vmin, anim.vmax) == (1e-6, 1e-3)


def test_all_zero_input_falls_back_to_unit_range(fp_da):
    anim = fa.FootprintAnimator(xr.zeros_like(fp_da), freq="daily")
    assert (anim.vmin, anim.vmax) == (0.0, 1.0)


def test_mask_quantile_blanks_low_cells(fp_da):
    anim = fa.FootprintAnimator(fp_da, freq="daily", mask_quantile=0.5)
    arr = anim._frame_array(0)
    assert np.isnan(arr).any()
    assert np.nanmin(arr) > anim.mask_threshold


def test_mask_below_and_quantile_are_exclusive(fp_da):
    with pytest.raises(ValueError, match="not both"):
        fa.FootprintAnimator(fp_da, freq="daily", mask_below=1e-6, mask_quantile=0.5)


def test_source_area_levels_are_sorted_and_unique():
    field = np.array([[4.0, 3.0], [2.0, 1.0]])
    levels = fa._source_area_levels(field, cell_area=1.0, fractions=[0.5, 0.9])
    assert levels == sorted(set(levels))
    assert all(np.isfinite(levels))


def test_source_area_levels_empty_field():
    assert fa._source_area_levels(np.full((3, 3), np.nan), 1.0, [0.5]) == []


def test_bad_timestamp_loc_raises(fp_da):
    anim = fa.FootprintAnimator(fp_da, freq="daily", timestamp_loc="middle")
    with pytest.raises(ValueError, match="timestamp_loc"):
        anim._timestamp_xy()


# ----------------------------
# Rendering / output
# ----------------------------


def test_build_labels_first_frame_with_timestamp(fp_da):
    anim = fa.FootprintAnimator(fp_da, freq="daily")
    assert anim.timestamp_format == "%Y-%m-%d"
    fig, animation = anim.build()
    try:
        texts = [t.get_text() for t in fig.axes[0].texts]
        assert "2023-06-01" in texts
        animation._init_draw()
    finally:
        matplotlib.pyplot.close(fig)


def test_monthly_label_format(fp_monthly_da):
    anim = fa.FootprintAnimator(fp_monthly_da, freq="monthly")
    assert anim._label(0, np.zeros((2, 2))) == "May 2023"


def test_annotation_fn_adds_second_line(fp_da):
    anim = fa.FootprintAnimator(
        fp_da, freq="daily", annotation_fn=lambda ts, arr: f"peak={np.nanmax(arr):.1e}"
    )
    label = anim._label(0, anim._frame_array(0))
    assert label.startswith("2023-06-01\npeak=")


def test_save_gif_writes_file(fp_da, tmp_path):
    out = tmp_path / "fp.gif"
    written = fa.animate_footprint(
        fp_da,
        out,
        freq="daily",
        contour_fractions=(0.5, 0.8),
        fps=2,
        dpi=60,
        figsize=(3.0, 3.0),
    )
    assert written == out
    assert out.exists() and out.stat().st_size > 0


def test_save_gif_with_frames_dir(fp_da, tmp_path):
    out = tmp_path / "fp.gif"
    frames = tmp_path / "frames"
    fa.animate_footprint(
        fp_da, out, freq="daily", frames_dir=frames, dpi=60, figsize=(3.0, 3.0)
    )
    pngs = sorted(frames.glob("*.png"))
    assert len(pngs) == 3
    assert all(p.stat().st_size > 0 for p in pngs)


def test_unsupported_extension_raises(fp_da, tmp_path):
    anim = fa.FootprintAnimator(fp_da, freq="daily")
    with pytest.raises(ValueError, match="Unsupported output extension"):
        anim.save(tmp_path / "fp.avi")


def test_mp4_without_ffmpeg_gives_actionable_error(fp_da, tmp_path, monkeypatch):
    monkeypatch.setattr(fa, "ensure_ffmpeg", lambda: None)
    anim = fa.FootprintAnimator(fp_da, freq="daily")
    with pytest.raises(RuntimeError, match="ffmpeg"):
        anim.save(tmp_path / "fp.mp4")


def test_ensure_ffmpeg_prefers_path_binary(monkeypatch):
    from matplotlib import animation as mplanim

    monkeypatch.setattr(mplanim.writers, "is_available", lambda name: True)
    monkeypatch.setitem(matplotlib.rcParams, "animation.ffmpeg_path", "ffmpeg")
    assert fa.ensure_ffmpeg() == "ffmpeg"


def test_ensure_ffmpeg_falls_back_to_imageio(monkeypatch):
    """With nothing on PATH, the imageio-ffmpeg binary is adopted."""
    from matplotlib import animation as mplanim

    fake_exe = "/opt/bundled/ffmpeg"
    seen = {"checks": 0}

    def fake_is_available(name):
        # False on the first check (PATH), True once rcParams points at the
        # bundled binary.
        seen["checks"] += 1
        return seen["checks"] > 1

    fake_module = types.SimpleNamespace(get_ffmpeg_exe=lambda: fake_exe)
    monkeypatch.setattr(mplanim.writers, "is_available", fake_is_available)
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", fake_module)
    monkeypatch.setitem(matplotlib.rcParams, "animation.ffmpeg_path", "ffmpeg")

    assert fa.ensure_ffmpeg() == fake_exe
    assert matplotlib.rcParams["animation.ffmpeg_path"] == fake_exe


def test_ensure_ffmpeg_returns_none_without_imageio(monkeypatch):
    from matplotlib import animation as mplanim

    monkeypatch.setattr(mplanim.writers, "is_available", lambda name: False)
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
    assert fa.ensure_ffmpeg() is None


def test_time_slice_subsets_frames(fp_da):
    anim = fa.FootprintAnimator(
        fp_da, freq="daily", time_slice=("2023-06-02", "2023-06-03")
    )
    assert anim.frames.sizes["time"] == 2


def test_empty_after_slicing_raises(fp_da):
    with pytest.raises(ValueError, match="No frames left"):
        fa.FootprintAnimator(fp_da, time_slice=("2024-01-01", "2024-01-02"))


# ----------------------------
# from_summary
# ----------------------------


class _FakeSummary:
    def __init__(self, daily=None, monthly=None):
        self.f_daily_mean = daily
        self.f_monthly_mean = monthly
        self.f_daily_et_weighted = None
        self.f_monthly_et_weighted = None


def test_from_summary_uses_layer_label_format(fp_monthly_da):
    monthly = fa.resample_footprints(fp_monthly_da, freq="monthly")
    anim = fa.FootprintAnimator.from_summary(
        _FakeSummary(monthly=monthly), layer="monthly_mean"
    )
    assert anim.timestamp_format == "%B %Y"
    assert anim.frames.sizes["time"] == 3


def test_from_summary_missing_layer_raises():
    with pytest.raises(ValueError, match="is empty"):
        fa.FootprintAnimator.from_summary(_FakeSummary(), layer="daily_etw")


def test_from_summary_unknown_layer_raises():
    with pytest.raises(ValueError, match="Unknown layer"):
        fa.FootprintAnimator.from_summary(_FakeSummary(), layer="hourly_mean")
