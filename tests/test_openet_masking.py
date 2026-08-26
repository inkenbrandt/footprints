# test_openet_masking.py
# Run with: pytest -q

import datetime as dt
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# Import project from ../src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

rasterio = pytest.importorskip("rasterio")
from pyproj import CRS, Transformer  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

from fluxfootprints import ffp_daily_monthly_helper as helper  # noqa: E402
from fluxfootprints import openet_masking as om  # noqa: E402

STATION_LAT = 40.0
STATION_LON = -111.9

# Footprint grid: 21 x 21 cells of 10 m, centred on the tower.
GRID = np.arange(-100.0, 100.0 + 10.0, 10.0)


# ----------------------------
# Helpers
# ----------------------------
def _tower_utm():
    epsg = helper._choose_utm_epsg_pyproj(STATION_LON, STATION_LAT)
    crs = CRS.from_epsg(epsg)
    to_proj = Transformer.from_crs(CRS.from_epsg(4326), crs, always_xy=True)
    x0, y0 = to_proj.transform(STATION_LON, STATION_LAT)
    return crs, x0, y0


def _write_openet(
    path,
    valid_side="east",
    res=10.0,
    half=200.0,
    value=5.0,
    nodata=-9999.0,
):
    """Write a synthetic OpenET raster, valid on one half of the tower domain.

    The grid is offset by half a cell so footprint cell centres never land on a
    source cell boundary, which keeps the nearest-neighbour result exact.
    """
    crs, x0, y0 = _tower_utm()
    n = int(2 * half / res)
    transform = from_origin(x0 - half - res / 2.0, y0 + half + res / 2.0, res, res)

    data = np.full((n, n), float(value), dtype="float32")
    west = slice(0, n // 2)
    east = slice(n // 2, n)
    data[:, west if valid_side == "east" else east] = nodata

    profile = {
        "driver": "GTiff",
        "height": n,
        "width": n,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
    return Path(path)


def _footprint_da(times, dims=("time", "x", "y")):
    """Uniform footprint field so masked fractions are easy to reason about."""
    shape = tuple(
        len(times) if d == "time" else len(GRID) for d in dims
    )
    return xr.DataArray(
        np.ones(shape, dtype="float64"),
        dims=dims,
        coords={"time": pd.to_datetime(times), "x": GRID, "y": GRID},
        name="footprint",
    )


# ----------------------------
# File-name date parsing
# ----------------------------
@pytest.mark.parametrize(
    "name,expected",
    [
        ("ensemble_et_20200615.tif", (dt.date(2020, 6, 15), "day")),
        ("2020-06-15_openet.tif", (dt.date(2020, 6, 15), "day")),
        ("openet_2020_06_15_v2.tif", (dt.date(2020, 6, 15), "day")),
        ("ffp_daily_mean_20240201.tif", (dt.date(2024, 2, 1), "day")),
        ("ffp_monthly_mean_202402.tif", (dt.date(2024, 2, 1), "month")),
        ("no_date_here.tif", None),
        ("openet_20201345.tif", None),  # impossible month/day
    ],
)
def test_parse_raster_date(name, expected):
    assert om.parse_raster_date(name) == expected


def test_parse_raster_date_custom_regex():
    got = om.parse_raster_date(
        "et_d150_y2021.tif", date_regex=r"y(\d{4})_?"
    )
    assert got is None  # single group is not enough to build a date

    got = om.parse_raster_date("et_2021doy_0704.tif", date_regex=r"(\d{4})doy_(\d{2})")
    assert got == (dt.date(2021, 7, 1), "month")


# ----------------------------
# Indexing
# ----------------------------
def test_index_openet_rasters_directory(tmp_path):
    _write_openet(tmp_path / "openet_20240201.tif")
    _write_openet(tmp_path / "openet_20240202.tif")
    (tmp_path / "readme.txt").write_text("not a raster")

    index = om.index_openet_rasters(tmp_path)
    assert list(index) == [dt.date(2024, 2, 1), dt.date(2024, 2, 2)]
    assert all(len(v) == 1 for v in index.values())


def test_index_openet_rasters_tiles_share_a_date(tmp_path):
    _write_openet(tmp_path / "openet_20240201_tileA.tif")
    _write_openet(tmp_path / "openet_20240201_tileB.tif")

    index = om.index_openet_rasters(tmp_path)
    assert list(index) == [dt.date(2024, 2, 1)]
    assert len(index[dt.date(2024, 2, 1)]) == 2


def test_index_openet_rasters_accepts_mapping(tmp_path):
    p = _write_openet(tmp_path / "et.tif")
    index = om.index_openet_rasters({"2024-02-01": p})
    assert index == {dt.date(2024, 2, 1): [p]}


def test_index_openet_rasters_empty_raises(tmp_path):
    with pytest.raises(ValueError, match="No dated OpenET rasters"):
        om.index_openet_rasters(tmp_path)


# ----------------------------
# Grid geometry
# ----------------------------
def test_footprint_grid_geometry_matches_export_convention():
    geom = om.footprint_grid_geometry(GRID, GRID, STATION_LAT, STATION_LON)
    _, x0, y0 = _tower_utm()

    assert geom.width == geom.height == len(GRID)
    assert geom.y_ascending is True
    # Outer edge of the corner cell, i.e. half a cell beyond the centre.
    assert geom.transform.c == pytest.approx(x0 - 105.0)
    assert geom.transform.f == pytest.approx(y0 + 105.0)
    assert geom.transform.a == pytest.approx(10.0)
    assert geom.transform.e == pytest.approx(-10.0)


def test_footprint_grid_geometry_rejects_degenerate_grid():
    with pytest.raises(ValueError, match="at least 2 points"):
        om.footprint_grid_geometry([0.0], [0.0], STATION_LAT, STATION_LON)


# ----------------------------
# Masking a DataArray
# ----------------------------
def test_mask_footprint_dataarray_east_half(tmp_path):
    _write_openet(tmp_path / "openet_20240201.tif", valid_side="east")
    da = _footprint_da(["2024-02-01"])

    res = om.mask_footprint_dataarray(da, tmp_path, STATION_LAT, STATION_LON)

    assert res.data.dims == da.dims
    masked = res.data.isel(time=0)
    assert float(masked.sel(x=slice(0.0, 100.0)).min()) == 1.0
    assert float(masked.sel(x=slice(-100.0, -10.0)).max()) == 0.0
    # 11 of 21 columns survive.
    assert float(res.retained_fraction.isel(time=0)) == pytest.approx(11.0 / 21.0)
    assert res.missing_dates == []
    assert res.data.attrs["openet_masked"] == "true"


def test_mask_footprint_dataarray_west_half(tmp_path):
    _write_openet(tmp_path / "openet_20240201.tif", valid_side="west")
    da = _footprint_da(["2024-02-01"])

    res = om.mask_footprint_dataarray(da, tmp_path, STATION_LAT, STATION_LON)

    masked = res.data.isel(time=0)
    assert float(masked.sel(x=slice(-100.0, -10.0)).min()) == 1.0
    assert float(masked.sel(x=slice(0.0, 100.0)).max()) == 0.0


def test_mask_footprint_dataarray_transposed_dims(tmp_path):
    _write_openet(tmp_path / "openet_20240201.tif")
    da = _footprint_da(["2024-02-01"], dims=("time", "y", "x"))

    res = om.mask_footprint_dataarray(da, tmp_path, STATION_LAT, STATION_LON)

    assert res.data.dims == ("time", "y", "x")
    assert float(res.retained_fraction.isel(time=0)) == pytest.approx(11.0 / 21.0)


def test_mask_footprint_dataarray_per_day_masks_differ(tmp_path):
    _write_openet(tmp_path / "openet_20240201.tif", valid_side="east")
    _write_openet(tmp_path / "openet_20240202.tif", valid_side="west")
    da = _footprint_da(["2024-02-01", "2024-02-02"])

    res = om.mask_footprint_dataarray(da, tmp_path, STATION_LAT, STATION_LON)

    day1 = res.data.sel(time="2024-02-01").squeeze()
    day2 = res.data.sel(time="2024-02-02").squeeze()
    assert float(day1.sel(x=100.0).max()) == 1.0
    assert float(day1.sel(x=-100.0).max()) == 0.0
    assert float(day2.sel(x=100.0).max()) == 0.0
    assert float(day2.sel(x=-100.0).max()) == 1.0


def test_mask_footprint_dataarray_nan_fill(tmp_path):
    _write_openet(tmp_path / "openet_20240201.tif")
    da = _footprint_da(["2024-02-01"])

    res = om.mask_footprint_dataarray(
        da, tmp_path, STATION_LAT, STATION_LON, fill_value=np.nan
    )
    assert bool(res.data.sel(x=-100.0).isnull().all())


def test_mask_footprint_dataarray_renormalize_preserves_total(tmp_path):
    _write_openet(tmp_path / "openet_20240201.tif")
    da = _footprint_da(["2024-02-01"])

    res = om.mask_footprint_dataarray(
        da, tmp_path, STATION_LAT, STATION_LON, renormalize=True
    )
    assert float(res.data.sum()) == pytest.approx(float(da.sum()))
    # retained_fraction still reports the pre-rescaling loss
    assert float(res.retained_fraction.isel(time=0)) == pytest.approx(11.0 / 21.0)


def test_mask_footprint_dataarray_treat_zero_as_nodata(tmp_path):
    # Valid everywhere per nodata, but the east half is exactly 0.
    crs, x0, y0 = _tower_utm()
    n, res = 40, 10.0
    data = np.zeros((n, n), dtype="float32")
    data[:, : n // 2] = 5.0
    profile = {
        "driver": "GTiff",
        "height": n,
        "width": n,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": from_origin(x0 - 205.0, y0 + 205.0, res, res),
        "nodata": -9999.0,
    }
    with rasterio.open(tmp_path / "openet_20240201.tif", "w", **profile) as dst:
        dst.write(data, 1)

    da = _footprint_da(["2024-02-01"])

    kept_all = om.mask_footprint_dataarray(da, tmp_path, STATION_LAT, STATION_LON)
    assert float(kept_all.retained_fraction.isel(time=0)) == pytest.approx(1.0)

    dropped = om.mask_footprint_dataarray(
        da, tmp_path, STATION_LAT, STATION_LON, treat_zero_as_nodata=True
    )
    assert float(dropped.retained_fraction.isel(time=0)) == pytest.approx(10.0 / 21.0)


def test_mask_footprint_dataarray_valid_range(tmp_path):
    _write_openet(tmp_path / "openet_20240201.tif", value=5.0)
    da = _footprint_da(["2024-02-01"])

    res = om.mask_footprint_dataarray(
        da, tmp_path, STATION_LAT, STATION_LON, valid_range=(10.0, None)
    )
    assert float(res.data.sum()) == 0.0


# ----------------------------
# Missing dates
# ----------------------------
def test_missing_date_skip_leaves_slice_untouched(tmp_path):
    _write_openet(tmp_path / "openet_20240201.tif")
    da = _footprint_da(["2024-02-01", "2024-02-02"])

    res = om.mask_footprint_dataarray(da, tmp_path, STATION_LAT, STATION_LON)

    assert [pd.Timestamp(t).date() for t in res.missing_dates] == [dt.date(2024, 2, 2)]
    assert float(res.retained_fraction.sel(time="2024-02-02")) == pytest.approx(1.0)


def test_missing_date_mask_blanks_slice(tmp_path):
    _write_openet(tmp_path / "openet_20240201.tif")
    da = _footprint_da(["2024-02-01", "2024-02-02"])

    res = om.mask_footprint_dataarray(
        da, tmp_path, STATION_LAT, STATION_LON, on_missing="mask"
    )
    assert float(res.data.sel(time="2024-02-02").sum()) == 0.0


def test_missing_date_nearest_borrows_neighbour(tmp_path):
    _write_openet(tmp_path / "openet_20240201.tif", valid_side="east")
    da = _footprint_da(["2024-02-04"])

    res = om.mask_footprint_dataarray(
        da, tmp_path, STATION_LAT, STATION_LON, on_missing="nearest"
    )
    assert float(res.retained_fraction.isel(time=0)) == pytest.approx(11.0 / 21.0)

    outside = om.mask_footprint_dataarray(
        da, tmp_path, STATION_LAT, STATION_LON, on_missing="nearest", max_gap_days=1
    )
    assert float(outside.retained_fraction.isel(time=0)) == pytest.approx(1.0)


def test_missing_date_error_raises(tmp_path):
    _write_openet(tmp_path / "openet_20240201.tif")
    da = _footprint_da(["2024-02-02"])

    with pytest.raises(FileNotFoundError, match="No OpenET raster"):
        om.mask_footprint_dataarray(
            da, tmp_path, STATION_LAT, STATION_LON, on_missing="error"
        )


# ----------------------------
# Monthly slices
# ----------------------------
def test_monthly_slice_combines_days(tmp_path):
    _write_openet(tmp_path / "openet_20240205.tif", valid_side="east")
    _write_openet(tmp_path / "openet_20240215.tif", valid_side="west")
    da = _footprint_da(["2024-02-01", "2024-03-01"])
    _write_openet(tmp_path / "openet_20240305.tif", valid_side="east")

    union = om.mask_footprint_dataarray(
        da, tmp_path, STATION_LAT, STATION_LON, freq="monthly", combine="any"
    )
    assert float(union.retained_fraction.sel(time="2024-02-01")) == pytest.approx(1.0)
    assert float(union.retained_fraction.sel(time="2024-03-01")) == pytest.approx(
        11.0 / 21.0
    )

    intersect = om.mask_footprint_dataarray(
        da, tmp_path, STATION_LAT, STATION_LON, freq="monthly", combine="all"
    )
    assert float(intersect.retained_fraction.sel(time="2024-02-01")) == 0.0


def test_freq_auto_detects_monthly(tmp_path):
    _write_openet(tmp_path / "openet_20240215.tif", valid_side="east")
    _write_openet(tmp_path / "openet_20240315.tif", valid_side="east")
    da = _footprint_da(["2024-02-01", "2024-03-01"])

    res = om.mask_footprint_dataarray(da, tmp_path, STATION_LAT, STATION_LON)

    assert res.data.attrs["openet_freq"] == "monthly"
    assert res.missing_dates == []


def test_freq_auto_detects_daily(tmp_path):
    _write_openet(tmp_path / "openet_20240201.tif")
    _write_openet(tmp_path / "openet_20240202.tif")
    da = _footprint_da(["2024-02-01", "2024-02-02"])

    res = om.mask_footprint_dataarray(da, tmp_path, STATION_LAT, STATION_LON)
    assert res.data.attrs["openet_freq"] == "daily"


# ----------------------------
# GeoTIFF masking
# ----------------------------
def _write_footprint_tif(path, value=1.0, nodata=0.0):
    crs, x0, y0 = _tower_utm()
    n = len(GRID)
    profile = {
        "driver": "GTiff",
        "height": n,
        "width": n,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": from_origin(x0 - 105.0, y0 + 105.0, 10.0, 10.0),
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.full((n, n), value, dtype="float32"), 1)
    return Path(path)


def test_mask_rasters_geotiff_daily(tmp_path):
    et_dir = tmp_path / "openet"
    et_dir.mkdir()
    _write_openet(et_dir / "openet_20240201.tif", valid_side="east")

    ffp_dir = tmp_path / "ffp"
    ffp_dir.mkdir()
    _write_footprint_tif(ffp_dir / "ffp_daily_mean_20240201.tif")

    out = om.mask_rasters_geotiff(ffp_dir, et_dir)

    assert len(out) == 1
    assert out[0].parent == ffp_dir / "openet_masked"
    with rasterio.open(out[0]) as src:
        arr = src.read(1)
    # Column 10 is the tower cell; east of it (inclusive) survives.
    assert np.all(arr[:, 10:] == 1.0)
    assert np.all(arr[:, :10] == 0.0)


def test_mask_rasters_geotiff_monthly_name(tmp_path):
    et_dir = tmp_path / "openet"
    et_dir.mkdir()
    _write_openet(et_dir / "openet_20240205.tif", valid_side="east")
    _write_openet(et_dir / "openet_20240215.tif", valid_side="west")

    ffp_dir = tmp_path / "ffp"
    ffp_dir.mkdir()
    _write_footprint_tif(ffp_dir / "ffp_monthly_mean_202402.tif")

    out = om.mask_rasters_geotiff(ffp_dir, et_dir, out_dir=tmp_path / "masked")

    with rasterio.open(out[0]) as src:
        arr = src.read(1)
    assert np.all(arr == 1.0)  # union of the two half-masks covers the grid


def test_mask_rasters_geotiff_missing_date_skips(tmp_path):
    et_dir = tmp_path / "openet"
    et_dir.mkdir()
    _write_openet(et_dir / "openet_20240201.tif", valid_side="east")

    ffp_dir = tmp_path / "ffp"
    ffp_dir.mkdir()
    _write_footprint_tif(ffp_dir / "ffp_daily_mean_20240209.tif")

    out = om.mask_rasters_geotiff(ffp_dir, et_dir)
    with rasterio.open(out[0]) as src:
        assert np.all(src.read(1) == 1.0)

    with pytest.raises(FileNotFoundError):
        om.mask_rasters_geotiff(ffp_dir, et_dir, on_missing="error")


def test_mask_rasters_geotiff_renormalize(tmp_path):
    et_dir = tmp_path / "openet"
    et_dir.mkdir()
    _write_openet(et_dir / "openet_20240201.tif", valid_side="east")

    ffp_dir = tmp_path / "ffp"
    ffp_dir.mkdir()
    src_path = _write_footprint_tif(ffp_dir / "ffp_daily_mean_20240201.tif")
    with rasterio.open(src_path) as src:
        before = src.read(1).sum()

    out = om.mask_rasters_geotiff(ffp_dir, et_dir, renormalize=True)
    with rasterio.open(out[0]) as src:
        after = src.read(1).sum()

    assert after == pytest.approx(before, rel=1e-5)


# ----------------------------
# Dispatcher and SummaryResult
# ----------------------------
def test_apply_openet_mask_dispatch_dataarray(tmp_path):
    _write_openet(tmp_path / "openet_20240201.tif")
    da = _footprint_da(["2024-02-01"])

    res = om.apply_openet_mask(da, tmp_path, STATION_LAT, STATION_LON)
    assert isinstance(res, om.MaskedFootprint)


def test_apply_openet_mask_requires_station_for_xarray(tmp_path):
    _write_openet(tmp_path / "openet_20240201.tif")
    da = _footprint_da(["2024-02-01"])

    with pytest.raises(ValueError, match="station_lat"):
        om.apply_openet_mask(da, tmp_path)


def test_apply_openet_mask_rejects_unknown_type(tmp_path):
    _write_openet(tmp_path / "openet_20240201.tif")
    with pytest.raises(TypeError, match="Cannot mask object"):
        om.apply_openet_mask(42, tmp_path)


def test_mask_summaries_masks_every_field(tmp_path):
    _write_openet(tmp_path / "openet_20240201.tif", valid_side="east")
    _write_openet(tmp_path / "openet_20240202.tif", valid_side="east")

    daily = _footprint_da(["2024-02-01", "2024-02-02"])
    monthly = _footprint_da(["2024-02-01"])

    summaries = helper.SummaryResult(
        f_daily_mean=daily,
        f_monthly_mean=monthly,
        f_daily_et_weighted=daily * 2.0,
        daily_domain_coverage=pd.DataFrame(
            {"mean_coverage_pct": [90.0, 91.0]},
            index=pd.to_datetime(["2024-02-01", "2024-02-02"]),
        ),
    )

    masked = om.apply_openet_mask(summaries, tmp_path, STATION_LAT, STATION_LON)

    assert float(masked.f_daily_mean.sel(x=-100.0).max()) == 0.0
    assert float(masked.f_monthly_mean.sel(x=-100.0).max()) == 0.0
    assert float(masked.f_daily_et_weighted.sel(x=100.0).max()) == 2.0
    assert masked.f_monthly_et_weighted is None

    cov = masked.daily_domain_coverage
    assert "openet_retained_frac" in cov.columns
    assert "openet_retained_frac_etw" in cov.columns
    assert cov["openet_retained_frac"].iloc[0] == pytest.approx(11.0 / 21.0)
    # The input is left untouched.
    assert "openet_retained_frac" not in summaries.daily_domain_coverage.columns
    assert float(summaries.f_daily_mean.sel(x=-100.0).max()) == 1.0


def test_mask_summaries_end_to_end(tmp_path):
    """Model -> summaries -> mask -> GeoTIFF export."""
    times = pd.to_datetime(
        ["2024-02-01 10:00", "2024-02-01 12:00", "2024-02-02 10:00"]
    )
    df = pd.DataFrame(
        {
            "wind_dir": [90.0, 100.0, 110.0],
            "umean": 3.0,
            "ustar": 0.3,
            "ol": 100.0,
            "sigmav": 0.5,
            "zm": 2.5,
            "z0": 0.05,
            "h": 1000.0,
        },
        index=times,
    )

    model = helper.build_climatology(
        df,
        ustar="ustar",
        ol="ol",
        umean="umean",
        sigmav="sigmav",
        wind_dir="wind_dir",
        zm="zm",
        z0="z0",
        h="h",
        dx=10.0,
        dy=10.0,
        domain=(-100.0, 100.0, -100.0, 100.0),
    )
    summaries = helper.summarize_periods(
        model, df, et_source=None, calc_et_weighted=False, monthly=False
    )

    _write_openet(tmp_path / "openet_20240201.tif", valid_side="east")
    _write_openet(tmp_path / "openet_20240202.tif", valid_side="east")

    masked = om.mask_summaries(summaries, tmp_path, STATION_LAT, STATION_LON)

    assert float(masked.f_daily_mean.sel(x=slice(-100.0, -10.0)).sum()) == 0.0
    frac = masked.daily_domain_coverage["openet_retained_frac"].dropna()
    assert len(frac) > 0
    assert ((frac >= 0.0) & (frac <= 1.0)).all()

    out_dir = helper.export_rasters_geotiff(
        model=model,
        summaries=masked,
        station_lat=STATION_LAT,
        station_lon=STATION_LON,
        out_dir=tmp_path / "out",
        which=("daily_mean",),
    )
    tifs = sorted(Path(out_dir).glob("ffp_daily_mean_*.tif"))
    assert len(tifs) == 2
    with rasterio.open(tifs[0]) as src:
        arr = src.read(1)
    assert np.all(arr[:, :10] == 0.0)
