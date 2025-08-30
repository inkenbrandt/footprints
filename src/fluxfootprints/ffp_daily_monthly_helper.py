"""
ffp_daily_monthly_helper.py

Summarize flux footprints from `ffp_xr.ffp_climatology_new` to daily/monthly
periods and (optionally) export 80% source-area contours to a GeoPackage.

Requirements (imported lazily when possible):
- numpy, pandas, xarray
- geopandas, shapely, pyproj, matplotlib (for optional GPKG export)

Usage (example):
    from ffp_daily_monthly_helper import (
        load_config, load_amf_df, build_climatology,
        summarize_periods, export_80pct_contours_gpkg
    )

    cfg = load_config("/mnt/data/US-CRT_config.ini")
    df  = load_amf_df("/mnt/data/AMF_US-CRT_BASE_HH_3-5_abb.csv", cfg)

    clim = build_climatology(df)        # computes clim.f_2d (time, x, y)

    # Daily/monthly means (normalized per time) and ET-weighted footprints
    out = summarize_periods(
        clim,
        df,
        et_source="LE",    # compute ET from LE (W/m2) -> mm/hr
        daily=True,
        monthly=True,
        normalize_each_time=True,
    )

    # Optional: export 80% contours (daily + monthly) to a GeoPackage
    export_80pct_contours_gpkg(
        clim,
        out,
        station_lat=cfg["station_latitude"],
        station_lon=cfg["station_longitude"],
        gpkg_path="US-CRT_footprints_80pct.gpkg",
        crs_out="auto"  # "auto" chooses an appropriate UTM
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Union, Any
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

# Local module (provided by user)
from .ffp_xr import ffp_climatology_new


# ------------------------------
# Config & Data Loading
# ------------------------------


def load_config(ini_path: str | Path) -> Dict[str, Any]:
    """Read a minimal INI for site metadata and column names."""
    import configparser

    cp = configparser.ConfigParser(interpolation=None)
    cp.read(ini_path)

    md = cp["METADATA"]
    data = cp["DATA"]

    def _getfloat(section, key, fallback=None):
        try:
            return cp.getfloat(section, key, fallback=fallback)
        except Exception:
            return fallback

    cfg = dict(
        station_latitude=_getfloat("METADATA", "station_latitude"),
        station_longitude=_getfloat("METADATA", "station_longitude"),
        missing_data_value=_getfloat("METADATA", "missing_data_value", -9999.0),
        skiprows=int(cp.get("METADATA", "skiprows", fallback="0")),
        date_parser=cp.get("METADATA", "date_parser", fallback="%Y%m%d%H%M"),
        ts_col=cp.get("DATA", "datestring_col", fallback="TIMESTAMP_START"),
        # Column names used by the footprint model
        wind_dir_col=cp.get("DATA", "wind_dir_col", fallback="WD"),
        wind_spd_col=cp.get("DATA", "wind_spd_col", fallback="WS"),
        ustar_col=cp.get("DATA", "USTAR", fallback="USTAR"),
        mo_length_col=cp.get("DATA", "MO_LENGTH", fallback="MO_LENGTH"),
        v_sigma_col=cp.get("DATA", "V_SIGMA", fallback="V_SIGMA"),
    )
    return cfg


def load_amf_df(csv_path: str | Path, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Load AMF half-hourly CSV and return a tidy DataFrame indexed by time."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, skiprows=cfg.get("skiprows", 0))
    # Parse timestamps
    ts_col = cfg.get("ts_col", "TIMESTAMP_START")
    date_fmt = cfg.get("date_parser", "%Y%m%d%H%M")
    df[ts_col] = pd.to_datetime(
        df[ts_col].astype(str), format=date_fmt, errors="coerce"
    )
    df = df.set_index(ts_col).sort_index()

    # Replace missing sentinel with NaN
    mv = cfg.get("missing_data_value", -9999.0)
    df = df.replace(mv, np.nan)

    return df


# ------------------------------
# Build & Run Climatology
# ------------------------------


def build_climatology(
    df: pd.DataFrame,
    crop_height: float = 0.2,
    atm_bound_height: float = 2000.0,
    inst_height: float = 2.5,
    dx: float = 10.0,
    dy: float = 10.0,
    domain: Tuple[float, float, float, float] = (-1000.0, 1000.0, -1000.0, 1000.0),
) -> ffp_climatology_new:
    """
    Prepare required columns and run the xarray-based footprint solver.

    Expected columns in df (typical AMF names shown in brackets):
      - wind_dir  [WD]       degrees
      - umean     [WS]       m/s
      - ustar     [USTAR]    m/s
      - ol        [MO_LENGTH] m
      - sigmav    [V_SIGMA]  m/s
    """
    # Map user columns to standardized names
    rename_map = {}
    for logical, ini_key in [
        ("wind_dir", "wind_dir_col"),
        ("umean", "wind_spd_col"),
        ("ustar", "ustar_col"),
        ("ol", "mo_length_col"),
        ("sigmav", "v_sigma_col"),
    ]:
        src = ini_key  # keep consistent with cfg keys
        # tolerate both upper/lower-case spellings
        # (we'll pass the actual names through cfg in summarize_periods)
        rename_map[src] = logical

    # Instead of relying on cfg here, accept typical AMF names directly:
    df2 = df.rename(
        columns={
            "WD": "wind_dir",
            "WS": "umean",
            "USTAR": "ustar",
            "MO_LENGTH": "ol",
            "V_SIGMA": "sigmav",
        }
    ).copy()

    # Filter minimal data presence
    needed = ["wind_dir", "umean", "ustar", "ol", "sigmav"]
    missing_cols = [c for c in needed if c not in df2.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Build and run the climatology object
    clim = ffp_climatology_new(
        df=df2,
        domain=np.array(domain, dtype=float),
        dx=float(dx),  # type: ignore
        dy=float(dy),  # type: ignore
        rs=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        crop_height=float(crop_height),
        atm_bound_height=float(atm_bound_height),
        inst_height=float(inst_height),
    )
    clim.run()  # populates clim.f_2d (time, x, y) and clim.fclim_2d
    return clim


# ------------------------------
# Daily/Monthly Summaries
# ------------------------------


def _ensure_time_normalized(f_2d: xr.DataArray) -> xr.DataArray:
    """Normalize each time-slice so sum over x,y = 1."""
    return f_2d / f_2d.sum(dim=("x", "y"))


def _et_from_le(df: pd.DataFrame, le_col: str = "LE") -> pd.Series:
    """
    Convert latent energy (LE, W/m^2) to ET in mm/hr.
    1 mm water over 1 m^2 in 1 hour corresponds to ~ 680.6 W/m^2.
    """
    if le_col not in df.columns:
        raise ValueError(f"LE column '{le_col}' not found in DataFrame")
    # mm/hr = W/m^2 ÷ 680.6
    return df[le_col] / 680.6


@dataclass
class SummaryResult:
    f_daily_mean: Optional[xr.DataArray] = None
    f_monthly_mean: Optional[xr.DataArray] = None
    f_daily_et_weighted: Optional[xr.DataArray] = None
    f_monthly_et_weighted: Optional[xr.DataArray] = None


def summarize_periods(
    clim: ffp_climatology_new,
    df: pd.DataFrame,
    et_source: str = "LE",  # compute ET from LE
    daily: bool = True,
    monthly: bool = True,
    normalize_each_time: bool = True,
) -> SummaryResult:
    """
    Build daily/monthly summaries from clim.f_2d (time,x,y).

    - Mean footprints: per-period average of per-time normalized slices.
    - ET-weighted: per-time normalized slices weighted by ET (mm/hr), then
      aggregated within each period and normalized by the sum of weights.
    """
    f = clim.f_2d
    if normalize_each_time:
        f = _ensure_time_normalized(f)  # type: ignore

    res = SummaryResult()
    if daily:
        res.f_daily_mean = f.resample(time="1D").mean()  # type: ignore
    if monthly:
        res.f_monthly_mean = f.resample(time="MS").mean()  # type: ignore

    # ET-weighted aggregation
    et_mmhr = _et_from_le(df, le_col=et_source).reindex(f["time"].to_index(), method=None)  # type: ignore
    et_da = xr.DataArray(et_mmhr.values, coords={"time": f["time"]}, dims=("time",))  # type: ignore

    weighted = f * et_da  # type: ignore
    if daily:
        num = weighted.resample(time="1D").sum()
        den = et_da.resample(time="1D").sum()
        res.f_daily_et_weighted = num / den
    if monthly:
        num = weighted.resample(time="MS").sum()
        den = et_da.resample(time="MS").sum()
        res.f_monthly_et_weighted = num / den

    return res


# ------------------------------
# Export 80% contours
# ------------------------------


def _choose_utm_epsg(lon: float, lat: float) -> int:
    """Pick a UTM EPSG (NAD83 / WGS84 UTM) from lon/lat (northern hemisphere only)."""
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    # Use WGS84 UTM north as a default
    return int(f"326{zone:02d}")  # EPSG:326##


def _make_contour_polygon_from_field_alt(
    z2d: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    level: float = 0.8,
    method: str = "auto",
):
    """
    Alternative to plt.contour for extracting a single polygon boundary
    for a cumulative-source threshold.

    Strategy:
      1) Normalize z2d so sum = 1, compute mask where cumulative >= level.
      2) Try scikit-image marching squares (find_contours) on the binary mask.
      3) Fallback: rasterio.features.shapes vectorization on the mask.

    Returns
    -------
    vertices : np.ndarray | None
        (N, 2) array of [x, y] in the same units as x,y inputs,
        or None if no contour found.
    """
    z = np.array(z2d, dtype=float, copy=True)
    z[z < 0] = 0.0
    s = z.sum()
    if not np.isfinite(s) or s <= 0:
        return None
    z /= s

    # Determine threshold by sorting descending
    flat = z.ravel()
    idx = np.argsort(flat)[::-1]
    cum = np.cumsum(flat[idx])
    thr_idx = np.searchsorted(cum, float(level), side="left")
    thr_val = flat[idx[thr_idx]] if thr_idx < flat.size else flat[idx[-1]]
    mask = (z >= thr_val).astype(np.uint8)

    # Index→coordinate maps
    x = np.asarray(x)
    y = np.asarray(y)
    nx, ny = x.size, y.size
    col_ix = np.arange(nx, dtype=float)
    row_ix = np.arange(ny, dtype=float)

    def map_cols_to_x(cols):
        return np.interp(cols, col_ix, x)

    def map_rows_to_y(rows):
        return np.interp(rows, row_ix, y)

    # 1) scikit-image marching squares
    if method in ("auto", "skimage"):
        try:
            from skimage import measure

            conts = measure.find_contours(mask, level=0.5)
            if conts:
                arr = max(conts, key=lambda a: a.shape[0])  # (row, col)
                rows, cols = arr[:, 0], arr[:, 1]
                xv = map_cols_to_x(cols)
                yv = map_rows_to_y(rows)
                return np.column_stack([xv, yv])
        except Exception:
            if method == "skimage":
                return None

    # 2) rasterio polygonization
    if method in ("auto", "rasterio"):
        try:
            from rasterio import features
            from affine import Affine

            if nx < 2 or ny < 2:
                return None
            xres = float(x[1] - x[0])
            yres = float(y[1] - y[0])
            left = x.min() - xres / 2.0
            top = y.max() + yres / 2.0
            transform = Affine.translation(left, top) * Affine.scale(xres, -yres)
            geoms = list(features.shapes(mask, mask=None, transform=transform))
            polys = [g for g, v in geoms if int(v) == 1]
            if not polys:
                return None

            def _poly_area(coords):
                try:
                    exterior = coords[0]
                    A = 0.0
                    for i in range(len(exterior) - 1):
                        x0, y0 = exterior[i]
                        x1, y1 = exterior[i + 1]
                        A += x0 * y1 - x1 * y0
                    return abs(A) * 0.5
                except Exception:
                    return 0.0

            polys.sort(key=lambda g: _poly_area(g["coordinates"]), reverse=True)
            ext = polys[0]["coordinates"][0]
            return np.asarray(ext, dtype=float)
        except Exception:
            if method == "rasterio":
                return None

    return None


def export_contours_gpkg(
    clim: ffp_climatology_new,
    summaries: SummaryResult,
    df: pd.DataFrame,
    station_lat: float,
    station_lon: float,
    gpkg_path: str | Path,
    crs_out: str | int = "auto",
    levels=(0.8,),
    stats_csv: str | Path | None = None,
    centroid_out: str | int = 4326,
    rn_col: str | None = None,
    h_col: str | None = None,
    le_col: str | None = None,
    g_col: str | None = None,
    contour_method: str = "auto",
) -> Path:
    """
    Export source-area contours (configurable rs levels, e.g., [0.5, 0.8])
    for each time slice in the provided summaries to a GeoPackage.

    Also writes optional CSV stats with area, perimeter, centroids (in multiple CRS),
    ET stats (from LE), and energy-balance closure (Rn vs H+LE+G).

    Creates layers like: daily_mean_r50, daily_mean_r80, monthly_etw_r80, ...
    """
    import geopandas as gpd
    from shapely.geometry import Polygon
    from pyproj import CRS, Transformer

    x = clim.x  # model grid (relative meters)
    y = clim.y

    # ---------------- CRS + transforms ----------------
    if crs_out == "auto":
        epsg = _choose_utm_epsg(station_lon, station_lat)
        dst_crs = CRS.from_epsg(epsg)
    else:
        dst_crs = CRS.from_user_input(crs_out)

    wgs84 = CRS.from_epsg(4326)
    to_proj = Transformer.from_crs(wgs84, dst_crs, always_xy=True)
    to_wgs = Transformer.from_crs(dst_crs, wgs84, always_xy=True)

    # For centroid outputs
    try:
        centroid_crs = CRS.from_user_input(centroid_out)
    except Exception:
        centroid_crs = CRS.from_epsg(4326)
    to_centroid = Transformer.from_crs(dst_crs, centroid_crs, always_xy=True)
    to_wgs_cent = Transformer.from_crs(dst_crs, wgs84, always_xy=True)

    # ---------------- EB helpers + column selection ----------------
    def _pick(dfcols, preferred, fallbacks):
        if preferred and preferred in dfcols:
            return preferred
        for f in fallbacks:
            for c in dfcols:
                if f.lower() in c.lower():
                    return c
        return None

    cols = list(df.columns)
    rn_c = _pick(cols, rn_col, ["NETRAD", "RN", "RNET"])
    h_c = _pick(cols, h_col, ["H"])
    le_c = _pick(cols, le_col, ["LE", "LE_F_MDS", "LE_QC"])
    g_c = _pick(cols, g_col, ["G", "G_1_1_1", "SHF", "SOIL_HEAT"])

    def _eb_stats(slice_df: pd.DataFrame) -> dict:
        out = {}
        try:
            rn = pd.to_numeric(slice_df[rn_c], errors="coerce") if rn_c else None
            h = pd.to_numeric(slice_df[h_c], errors="coerce") if h_c else None
            le = pd.to_numeric(slice_df[le_c], errors="coerce") if le_c else None
            g = pd.to_numeric(slice_df[g_c], errors="coerce") if g_c else None

            rn_m = float(np.nanmean(rn)) if rn is not None else np.nan
            h_m = float(np.nanmean(h)) if h is not None else np.nan
            le_m = float(np.nanmean(le)) if le is not None else np.nan
            g_m = float(np.nanmean(g)) if g is not None else np.nan

            resid_m = rn_m - sum(v for v in (h_m, le_m, g_m) if np.isfinite(v))
            out.update(
                dict(
                    Rn_mean_Wm2=rn_m,
                    H_mean_Wm2=h_m,
                    LE_mean_Wm2=le_m,
                    G_mean_Wm2=g_m,
                    Residual_mean_Wm2=resid_m,
                )
            )
            if np.isfinite(rn_m) and rn_m != 0.0:
                total = sum(v for v in (h_m, le_m, g_m) if np.isfinite(v))
                out["Closure_frac"] = total / rn_m
                out["Residual_frac_of_Rn"] = resid_m / rn_m
            else:
                out["Closure_frac"] = np.nan
                out["Residual_frac_of_Rn"] = np.nan
        except Exception:
            out = dict(
                Rn_mean_Wm2=np.nan,
                H_mean_Wm2=np.nan,
                LE_mean_Wm2=np.nan,
                G_mean_Wm2=np.nan,
                Residual_mean_Wm2=np.nan,
                Closure_frac=np.nan,
                Residual_frac_of_Rn=np.nan,
            )
        return out

    # ---------------- I/O prep ----------------
    gpkg_path = Path(gpkg_path)
    if gpkg_path.exists():
        gpkg_path.unlink()

    stats_records = []

    x0, y0 = to_proj.transform(station_lon, station_lat)

    def _to_world_coords(vertices_xy: np.ndarray | None):
        if vertices_xy is None:
            return None
        vx = vertices_xy[:, 0] + x0
        vy = vertices_xy[:, 1] + y0
        return Polygon(np.column_stack([vx, vy]))

    # ---------------- Core layer writer ----------------
    def _collect_layer(da: Optional[xr.DataArray], base_layer: str):
        if da is None:
            return
        for r in levels:
            records = []
            for i in range(da.sizes["time"]):
                z = da.isel(time=i).values  # 2D array
                verts = _make_contour_polygon_from_field_alt(
                    z, x, y, level=float(r), method=contour_method
                )
                poly = _to_world_coords(verts)
                if poly is None or poly.is_empty:
                    continue

                ts = pd.Timestamp(da["time"].values[i]).isoformat()

                # EB stats by period (day or month)
                if base_layer.startswith("daily"):
                    sdf = df.loc[ts[:10]] if ts else df  # YYYY-MM-DD
                else:
                    sdf = df.loc[ts[:7]] if ts else df  # YYYY-MM
                eb = _eb_stats(sdf)  # type: ignore

                rec = {"time": ts, "r": float(r), **eb, "geometry": poly}
                records.append(rec)

            if not records:
                continue

            lname = f"{base_layer}_r{int(round(float(r) * 100))}"
            gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=dst_crs)

            # Areas, perimeters
            gdf["area_m2"] = gdf.geometry.area
            gdf["perimeter_m"] = gdf.geometry.length
            gdf["area_ha"] = gdf["area_m2"] / 10000.0
            gdf["area_acres"] = gdf["area_m2"] / 4046.85642

            # Centroids in multiple CRSs
            cx = gdf.geometry.centroid.x.values
            cy = gdf.geometry.centroid.y.values
            cx_out, cy_out = to_centroid.transform(cx, cy)
            lon, lat = to_wgs_cent.transform(cx, cy)
            gdf["centroid_x"] = cx
            gdf["centroid_y"] = cy
            gdf["centroid_out_x"] = cx_out
            gdf["centroid_out_y"] = cy_out
            gdf["centroid_lon"] = lon
            gdf["centroid_lat"] = lat
            gdf["layer"] = lname

            # ET stats from LE (mm/hr and mm per period) if available
            try:
                if le_c and le_c in df.columns:
                    # Daily vs monthly slice
                    if base_layer.startswith("daily"):
                        tstamp = pd.to_datetime(gdf["time"].iloc[0])
                        et_slice = df.loc[str(tstamp.date()), le_c]
                    else:
                        tstamp = pd.to_datetime(gdf["time"].iloc[0])
                        et_slice = df.loc[tstamp.strftime("%Y-%m"), le_c]

                    et_vals = (
                        pd.to_numeric(et_slice, errors="coerce").dropna().values / 680.6  # type: ignore
                    )
                    if et_vals.size > 0:
                        gdf["ET_mean_mmhr"] = float(np.nanmean(et_vals))
                        # If half-hourly data: convert mm/hr to mm depth per step (0.5 hr) and sum
                        gdf["ET_sum_mm"] = float(np.nansum(et_vals * 0.5))
            except Exception:
                pass

            # Write layer and collect stats rows
            gdf.to_file(gpkg_path, layer=lname, driver="GPKG")
            stats_records.extend(gdf.drop(columns="geometry").to_dict(orient="records"))

    # ---------------- Write all layers ----------------
    _collect_layer(summaries.f_daily_mean, "daily_mean")
    _collect_layer(summaries.f_monthly_mean, "monthly_mean")
    _collect_layer(summaries.f_daily_et_weighted, "daily_etw")
    _collect_layer(summaries.f_monthly_et_weighted, "monthly_etw")

    # Optional stats CSV
    if stats_csv and stats_records:
        stats_df = pd.DataFrame(stats_records)
        stats_df.to_csv(stats_csv, index=False)

    return gpkg_path


# ------------------------------
# GeoTIFF raster export
# ------------------------------


def export_rasters_geotiff(
    clim: ffp_climatology_new,
    summaries: SummaryResult,
    station_lat: float,
    station_lon: float,
    out_dir: str | Path,
    crs_out: str | int = "auto",
    which=("daily_mean", "monthly_mean", "daily_etw", "monthly_etw"),
    prefix="ffp",
    dtype="float32",
    nodata=0.0,
) -> Path:
    """
    Export each time slice of the requested summaries as a GeoTIFF.

    File naming pattern: {prefix}_{layer}_{YYYYMMDD}.tif for daily layers,
    and {prefix}_{layer}_{YYYYMM}.tif for monthly layers.

    Layers names in 'which' must be among:
        'daily_mean', 'monthly_mean', 'daily_etw', 'monthly_etw'
    """
    import numpy as np
    from pathlib import Path
    import rasterio
    from rasterio.transform import from_origin
    from pyproj import CRS, Transformer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Grid and CRS transforms
    x = np.asarray(clim.x)
    y = np.asarray(clim.y)

    # Resolution (assume regular grid)
    if x.size < 2 or y.size < 2:
        raise ValueError("x/y grid must have at least 2 points each")

    xres = float(x[1] - x[0])
    yres = float(y[1] - y[0])

    if crs_out == "auto":
        epsg = _choose_utm_epsg(station_lon, station_lat)
        dst_crs = CRS.from_epsg(epsg)
    else:
        dst_crs = CRS.from_user_input(crs_out)

    wgs84 = CRS.from_epsg(4326)
    to_proj = Transformer.from_crs(wgs84, dst_crs, always_xy=True)
    x0, y0 = to_proj.transform(station_lon, station_lat)

    # Compute top-left corner for from_origin
    left = (x.min() + x0) - xres / 2.0
    top = (y.max() + y0) + yres / 2.0

    transform = from_origin(left, top, xres, yres)

    # Helper to write a stack of 2-D arrays, one per time slice
    def _write_series(da, layername):
        if da is None:
            return
        times = pd.to_datetime(da["time"].values)

        for i, ts in enumerate(times):
            arr = da.isel(time=i).values.astype(dtype)
            # rasterio expects (rows, cols) with row 0 at top; our y asc may be bottom->top
            # Flip in Y if needed so higher y is at top of image
            if y[1] > y[0]:  # ascending
                arr = np.flipud(arr)

            # Build filename by granularity
            if layername.startswith("daily"):
                suf = ts.strftime("%Y%m%d")
            else:
                suf = ts.strftime("%Y%m")
            fp = out_dir / f"{prefix}_{layername}_{suf}.tif"

            with rasterio.open(
                fp,
                "w",
                driver="GTiff",
                height=arr.shape[0],
                width=arr.shape[1],
                count=1,
                dtype=dtype,
                crs=dst_crs,
                transform=transform,
                nodata=nodata,
                compress="deflate",
                predictor=3,
                tiled=True,
                blockxsize=256,
                blockysize=256,
            ) as dst:
                dst.write(arr, 1)

    # Route each requested layer
    layer_map = {
        "daily_mean": summaries.f_daily_mean,
        "monthly_mean": summaries.f_monthly_mean,
        "daily_etw": summaries.f_daily_et_weighted,
        "monthly_etw": summaries.f_monthly_et_weighted,
    }
    for name in which:
        if name not in layer_map:
            raise ValueError(f"Unknown layer '{name}'")
        _write_series(layer_map[name], name)

    return Path(out_dir)


# ------------------------------
# CSV Stats Export
# ------------------------------


def export_contour_stats_csv(
    df: pd.DataFrame,
    clim: ffp_climatology_new,
    summaries: SummaryResult,
    station_lat: float,
    station_lon: float,
    csv_path: str | Path,
    rn_col: str | None = None,
    h_col: str | None = None,
    le_col: str | None = None,
    g_col: str | None = None,
    method: str = "auto",
    crs_out: str | int = "auto",
    levels=(0.8,),
) -> Path:
    """
    Export simple statistics (area in hectares, centroid lat/lon) for each contour
    and each time slice to a CSV file.
    """
    import csv
    import geopandas as gpd
    from shapely.geometry import Polygon
    from pyproj import CRS, Transformer

    x = clim.x
    y = clim.y

    # pick output CRS
    if crs_out == "auto":
        epsg = _choose_utm_epsg(station_lon, station_lat)
        dst_crs = CRS.from_epsg(epsg)
    else:
        dst_crs = CRS.from_user_input(crs_out)

    wgs84 = CRS.from_epsg(4326)
    to_proj = Transformer.from_crs(wgs84, dst_crs, always_xy=True)
    to_wgs = Transformer.from_crs(dst_crs, wgs84, always_xy=True)

    # --- helpers for energy balance closure ---
    def _pick(dfcols, preferred, fallbacks):
        if preferred and preferred in dfcols:
            return preferred
        for f in fallbacks:
            # allow contains
            for c in dfcols:
                if f.lower() in c.lower():
                    return c
        return None

    # choose columns
    cols = list(df.columns)
    rn_c = _pick(cols, rn_col, ["NETRAD", "RN", "RNET"])
    h_c = _pick(cols, h_col, ["H"])
    le_c = _pick(cols, le_col, ["LE", "LE_F_MDS", "LE_QC"])
    g_c = _pick(cols, g_col, ["G", "G_1_1_1", "SHF", "SOIL_HEAT"])

    def _eb_stats(slice_df):
        out = {}
        try:
            rn = slice_df[rn_c].astype(float) if rn_c else None
            h = slice_df[h_c].astype(float) if h_c else None
            le = slice_df[le_c].astype(float) if le_c else None
            g = slice_df[g_c].astype(float) if g_c else None
            # means (W/m2)
            rn_m = float(np.nanmean(rn)) if rn is not None else np.nan
            h_m = float(np.nanmean(h)) if h is not None else np.nan
            le_m = float(np.nanmean(le)) if le is not None else np.nan
            g_m = float(np.nanmean(g)) if g is not None else np.nan
            resid_m = rn_m - (
                (h_m if np.isfinite(h_m) else 0.0)
                + (le_m if np.isfinite(le_m) else 0.0)
                + (g_m if np.isfinite(g_m) else 0.0)
            )
            out.update(
                dict(
                    Rn_mean_Wm2=rn_m,
                    H_mean_Wm2=h_m,
                    LE_mean_Wm2=le_m,
                    G_mean_Wm2=g_m,
                    Residual_mean_Wm2=resid_m,
                )
            )
            # closure fraction (H+LE+G)/Rn
            if np.isfinite(rn_m) and rn_m != 0.0:
                num = 0.0
                for v in (h_m, le_m, g_m):
                    if np.isfinite(v):
                        num += v
                out["Closure_frac"] = num / rn_m
                out["Residual_frac_of_Rn"] = resid_m / rn_m
            else:
                out["Closure_frac"] = np.nan
                out["Residual_frac_of_Rn"] = np.nan
        except Exception:
            out = dict(
                Rn_mean_Wm2=np.nan,
                H_mean_Wm2=np.nan,
                LE_mean_Wm2=np.nan,
                G_mean_Wm2=np.nan,
                Residual_mean_Wm2=np.nan,
                Closure_frac=np.nan,
                Residual_frac_of_Rn=np.nan,
            )
        return out

    # Compute tower origin in projected meters
    x0, y0 = to_proj.transform(station_lon, station_lat)

    def _to_world_coords(vertices_xy):
        if vertices_xy is None:
            return None
        vx = vertices_xy[:, 0] + x0
        vy = vertices_xy[:, 1] + y0
        return Polygon(np.column_stack([vx, vy]))

    # Open CSV writer
    csv_path = Path(csv_path)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["layer", "time", "r", "area_ha", "centroid_lon", "centroid_lat"]
        )

        def _process_layer(da: Optional[xr.DataArray], base_layer: str):
            if da is None:
                return
            for r in levels:
                for i in range(da.sizes["time"]):
                    z = da.isel(time=i).values
                    verts = _make_contour_polygon_from_field_alt(
                        z, x, y, level=float(r), method=method
                    )
                    poly = _to_world_coords(verts)
                    if poly is None or poly.is_empty:
                        continue
                    ts = pd.Timestamp(da["time"].values[i]).isoformat()
                    # area in m2 -> ha
                    area_ha = poly.area / 10000.0
                    cx, cy = poly.centroid.x, poly.centroid.y
                    clon, clat = to_wgs.transform(cx, cy)
                    writer.writerow([base_layer, ts, float(r), area_ha, clon, clat])

        _process_layer(summaries.f_daily_mean, "daily_mean")
        _process_layer(summaries.f_monthly_mean, "monthly_mean")
        _process_layer(summaries.f_daily_et_weighted, "daily_etw")
        _process_layer(summaries.f_monthly_et_weighted, "monthly_etw")

    return csv_path
