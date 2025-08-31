# test_ffp_daily_monthly_helper.py
# Run with:  pytest -q

import os
import sys
import types
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import xarray as xr

# Import project from ../src as requested
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from fluxfootprints import ffp_daily_monthly_helper as helper
from fluxfootprints.ffp_xr import ffp_climatology_new


# ----------------------------
# Fixtures
# ----------------------------
@pytest.fixture
def tiny_times():
    return pd.to_datetime([
        "2024-01-31 23:30",
        "2024-02-01 00:00",
        "2024-02-01 00:30",
        "2024-02-01 12:00",
        "2024-02-02 00:00",
        "2024-02-02 00:30",
    ])


@pytest.fixture
def amf_like_df(tiny_times):
    n = len(tiny_times)
    return (pd.DataFrame({
        "TIMESTAMP_START": tiny_times,
        "WD": np.linspace(0, 330, n),     # deg
        "WS": np.full(n, 3.0),            # m/s
        "USTAR": np.full(n, 0.3),         # m/s
        "MO_LENGTH": np.full(n, 100.0),   # m
        "V_SIGMA": np.full(n, 0.5),       # m/s
        "LE": np.linspace(100.0, 200.0, n)  # W/m^2
    }).set_index("TIMESTAMP_START"))


@pytest.fixture
def mini_ini(tmp_path):
    ini = tmp_path / "site.ini"
    ini.write_text(
        "[METADATA]\n"
        "station_latitude = 40.0\n"
        "station_longitude = -111.9\n"
        "missing_data_value = -9999\n"
        "skiprows = 1\n"
        "date_parser = %Y%m%d%H%M\n"
        "\n"
        "[DATA]\n"
        "datestring_col = TIMESTAMP_START\n"
        "wind_dir_col = WD\n"
        "wind_spd_col = WS\n"
        "USTAR = USTAR\n"
        "MO_LENGTH = MO_LENGTH\n"
        "V_SIGMA = V_SIGMA\n"
    )
    return ini


# ----------------------------
# Config & CSV loading
# ----------------------------
def test_load_config(mini_ini):
    cfg = helper.load_config(mini_ini)
    assert cfg["station_latitude"] == 40.0
    assert cfg["station_longitude"] == -111.9
    assert cfg["missing_data_value"] == -9999
    assert cfg["skiprows"] == 1
    assert cfg["date_parser"] == "%Y%m%d%H%M"
    assert cfg["ts_col"] == "TIMESTAMP_START"
    assert cfg["wind_dir_col"] == "WD"
    assert cfg["wind_spd_col"] == "WS"
    assert cfg["ustar_col"] == "USTAR"
    assert cfg["mo_length_col"] == "MO_LENGTH"
    assert cfg["v_sigma_col"] == "V_SIGMA"


def test_load_amf_df(mini_ini, tmp_path):
    csv = tmp_path / "amf.csv"
    csv.write_text(
        "SKIP THIS ROW\n"
        "TIMESTAMP_START,WD,WS,USTAR,MO_LENGTH,V_SIGMA,LE\n"
        "202402010000,90,3,0.3,100,0.5,-9999\n"
        "202402010030,100,3,0.3,100,0.5,150\n"
    )
    cfg = helper.load_config(mini_ini)
    df = helper.load_amf_df(csv, cfg)
    # FIX 1: robust datetime check
    assert isinstance(df.index, pd.DatetimeIndex)
    assert np.isnan(df.loc[pd.Timestamp("2024-02-01 00:00"), "LE"])
    assert df.loc[pd.Timestamp("2024-02-01 00:30"), "LE"] == 150


# ----------------------------
# ffp_xr solver basics
# ----------------------------
def test_ffp_xr_run_direct(amf_like_df):
    # FIX 2: ffp_xr expects 'wd'/'ws' (lowercase) which it renames internally.
    df2 = amf_like_df.copy().rename(columns={"WD": "wd", "WS": "ws"})
    clim = ffp_climatology_new(df=df2, dx=10.0, dy=10.0,
                               domain=[-100.0, 100.0, -100.0, 100.0])
    clim.run()
    assert isinstance(clim.f_2d, xr.DataArray)
    assert set(clim.f_2d.dims) == {"time", "x", "y"}
    assert clim.f_2d.shape[0] == len(df2)
    assert np.all(np.isfinite(clim.fclim_2d.values))


def test_build_climatology_wrapper(amf_like_df):
    clim = helper.build_climatology(
        amf_like_df.copy(), dx=10.0, dy=10.0, domain=(-100.0, 100.0, -100.0, 100.0)
    )
    assert isinstance(clim, ffp_climatology_new)
    assert isinstance(clim.f_2d, xr.DataArray)
    assert clim.f_2d.ndim == 3


# ----------------------------
# Small helpers
# ----------------------------
def test_ensure_time_normalized():
    times = pd.to_datetime(["2024-02-01 00:00", "2024-02-01 00:30"])
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    da = xr.DataArray(
        np.array([[[1.0, 1.0], [1.0, 1.0]], [[2.0, 0.0], [0.0, 0.0]]]),
        coords={"time": times, "x": x, "y": y},
        dims=("time", "x", "y"),
    )
    normed = helper._ensure_time_normalized(da)
    sums = normed.sum(dim=("x", "y")).values
    assert np.allclose(sums, np.array([1.0, 1.0]), atol=1e-12)


def test_et_from_le(amf_like_df):
    et = helper._et_from_le(amf_like_df, le_col="LE")
    assert np.isclose(et.iloc[0], amf_like_df["LE"].iloc[0] / 680.6)


# ----------------------------
# Daily / Monthly summaries
# ----------------------------
def test_summarize_periods(amf_like_df):
    clim = helper.build_climatology(
        amf_like_df.copy(), dx=10.0, dy=10.0, domain=(-100.0, 100.0, -100.0, 100.0)
    )
    res = helper.summarize_periods(
        clim, amf_like_df, et_source="LE", daily=True, monthly=True, normalize_each_time=True
    )
    assert res.f_daily_mean is not None
    assert res.f_monthly_mean is not None
    assert res.f_daily_et_weighted is not None
    assert res.f_monthly_et_weighted is not None
    assert res.f_daily_mean.sizes["time"] >= 2
    months = pd.to_datetime(res.f_monthly_mean["time"].values)
    assert any(ts.month == 1 for ts in months) and any(ts.month == 2 for ts in months)
    assert np.isfinite(res.f_daily_et_weighted.values).any()


# ----------------------------
# UTM EPSG chooser
# ----------------------------
@pytest.mark.parametrize(
    "lon,lat,epsg",
    [
        (-111.9, 40.0, 32612),
        (-105.0, 40.0, 32613),
        (-3.7, 40.4, 32630),
    ],
)
def test_choose_utm_epsg(lon, lat, epsg):
    assert helper._choose_utm_epsg(lon, lat) == epsg


# ----------------------------
# Alternative contour extraction
# ----------------------------
def test_make_contour_polygon_from_field_alt_skimage_or_skip():
    try:
        import skimage  # noqa: F401
    except Exception:
        pytest.skip("skimage not available")
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    z = np.zeros((y.size, x.size))
    z[1:3, 1:3] = 10.0
    verts = helper._make_contour_polygon_from_field_alt(z, x, y, level=0.8, method="skimage")
    assert verts is not None and verts.shape[1] == 2 and len(verts) >= 4

# --------------
# Optional I/O smoke tests
# ----------------------------
def test_export_contours_gpkg_smoke(tmp_path, amf_like_df, monkeypatch):
    # Skip cleanly if optional deps are missing
    try:
        import geopandas as gpd  # noqa: F401
        import shapely  # noqa: F401
        import pyproj  # noqa: F401
    except Exception:
        pytest.skip("Optional geospatial dependencies not available")

    clim = helper.build_climatology(
        amf_like_df.copy(), dx=10.0, dy=10.0, domain=(-100.0, 100.0, -100.0, 100.0)
    )
    res = helper.summarize_periods(clim, amf_like_df, daily=True, monthly=False)

    import geopandas as gpd
    def _fake_to_file(self, path, *args, **kwargs):
        Path(path).touch(exist_ok=True)
        return None
    monkeypatch.setattr(gpd.GeoDataFrame, "to_file", _fake_to_file, raising=True)

    out = helper.export_contours_gpkg(
        clim=clim,
        summaries=res,
        df=amf_like_df,
        station_lat=40.0,
        station_lon=-111.9,
        gpkg_path=tmp_path / "out.gpkg",
        crs_out="auto",
        levels=(0.8,),
        contour_method="rasterio",
    )
    assert out.name.endswith(".gpkg")
    assert out.exists()


@pytest.mark.skipif(bool(__import__("importlib").util.find_spec("rasterio") is None),
                    reason="rasterio not available")
def test_export_rasters_geotiff_smoke(tmp_path, amf_like_df):
    clim = helper.build_climatology(
        amf_like_df.copy(), dx=10.0, dy=10.0, domain=(-100.0, 100.0, -100.0, 100.0)
    )
    res = helper.summarize_periods(clim, amf_like_df, daily=True, monthly=False)
    out_dir = helper.export_rasters_geotiff(
        clim=clim,
        summaries=res,
        station_lat=40.0,
        station_lon=-111.9,
        out_dir=tmp_path,
        which=("daily_mean",),
        prefix="ffp",
    )
    tifs = list(Path(out_dir).glob("ffp_daily_mean_*.tif"))
    assert len(tifs) >= 1


def test_export_contour_stats_csv_smoke(tmp_path, amf_like_df):
    try:
        import geopandas  # noqa: F401
        import shapely    # noqa: F401
        import pyproj     # noqa: F401
    except Exception:
        pytest.skip("Optional geospatial dependencies not available")

    clim = helper.build_climatology(
        amf_like_df.copy(), dx=10.0, dy=10.0, domain=(-100.0, 100.0, -100.0, 100.0)
    )
    res = helper.summarize_periods(clim, amf_like_df, daily=True, monthly=False)
    csv_path = tmp_path / "stats.csv"
    out = helper.export_contour_stats_csv(
        df=amf_like_df,
        clim=clim,
        summaries=res,
        station_lat=40.0,
        station_lon=-111.9,
        csv_path=csv_path,
        levels=(0.8,),
    )
    assert out.exists()
    header = out.read_text().splitlines()[0]
    assert "area_ha" in header and "centroid_lon" in header
