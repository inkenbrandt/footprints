"""
ffp_daily_monthly_helper.py (Refactored for FFPModel)

Summarize flux footprints from `improved_ffp.FFPModel` to daily/monthly
periods and (optionally) export 80% source-area contours to a GeoPackage.

Updated to use FFPModel as the canonical implementation.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Union, Any, Sequence, Type
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from rasterio import features
from affine import Affine
import rasterio
from rasterio.transform import from_origin
import csv

import geopandas as gpd
from shapely.geometry import Polygon
from pyproj import CRS, Transformer

# Use FFPModel as canonical implementation
from .improved_ffp import FFPModel
    # Import models
from .kormannmeixner_adapter import KormannMeixnerModel
from .ls_footprint_adapter import LSFootprintModelAdapter
from .wang_footprint_adapter import WangFootprintModel
from .base_footprint_model import BaseFootprintModel
from .ffp_xr import ffp_climatology_new

# Keep backward compatibility option
_LEGACY_MODE = False
try:
    from .ffp_xr import ffp_climatology_new as LegacyFFP
except ImportError:
    LegacyFFP = None
    _LEGACY_MODE = False


# ------------------------------
# Config & Data Loading
# ------------------------------


def load_config(ini_path: str | Path) -> Dict[str, Any]:
    """Read a minimal INI for site metadata and column names."""
    import configparser

    cp = configparser.ConfigParser(interpolation=None)
    cp.read(ini_path)

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
# Calculate zm and zo
# ------------------------------

VEG_PRESETS = {
    "stanhill": {"d_h_method": "stanhill", "z0_fraction": 0.123},
    "alfalfa": {"d_h_fraction": 0.67, "z0_fraction": 0.123},
    "corn_standard": {"d_h_fraction": 0.67, "z0_fraction": 0.140},
    "corn_jacobs": {"d_h_fraction": 0.75, "z0_method": "jacobs_1988"},
    "phragmites": {"d_h_fraction": 0.75, "z0_fraction": 0.090},
    "wetland_grass": {"d_h_fraction": 0.70, "z0_fraction": 0.110},
    "pinyon_juniper": {"d_h_fraction": 0.60, "z0_fraction": 0.100},
}

# Unified Flexible Input Type Alias
FlexibleInput = Union[
    str, float, int, pd.Series, np.ndarray, Sequence[float], None
]

def _resolve_input(
    val: FlexibleInput,
    df: pd.DataFrame,
    param_name: str = "input",
    as_series: bool = True,
) -> Union[pd.Series, float, int, None]:
    """Safely resolve string column names, pd.Series, np.ndarrays, lists, or numeric scalars.

    Parameters
    ----------
    val : str, float, int, pd.Series, np.ndarray, list, or None
        Input value to resolve.
    df : pd.DataFrame
        Reference DataFrame for index alignment and length matching.
    param_name : str, default "input"
        Parameter name used for informative error messages.
    as_series : bool, default True
        If True, converts numeric scalars into a pd.Series broadcasted across df.index.

    Returns
    -------
    pd.Series, float, int, or None
    """
    if val is None:
        return None

    # 1. Handle DataFrame Column Name (String)
    if isinstance(val, str):
        if val not in df.columns:
            raise KeyError(
                f"Column '{val}' specified for '{param_name}' not found in DataFrame."
            )
        return df[val]

    # 2. Handle Pandas Series (Align to df.index)
    if isinstance(val, pd.Series):
        return val.reindex(df.index)

    # 3. Handle Arrays or Lists
    if isinstance(val, (np.ndarray, list, tuple)):
        if len(val) != len(df):
            raise ValueError(
                f"Length of '{param_name}' ({len(val)}) does not match DataFrame length ({len(df)})."
            )
        return pd.Series(val, index=df.index, name=param_name)

    # 4. Handle Numeric Scalars
    if isinstance(val, (int, float, np.number)):
        if as_series:
            return pd.Series(float(val), index=df.index, name=param_name)
        return val

    raise TypeError(
        f"Invalid type for '{param_name}': {type(val)}. Expected str, float, int, pd.Series, np.ndarray, or list."
    )


def compute_aerodynamic_params(
    df: pd.DataFrame,
    inst_height: Union[float, str, pd.Series, np.ndarray],
    crop_height: Union[float, str, pd.Series, np.ndarray],
    veg_type: Optional[str] = None,
    z0_ratio: Optional[Union[float, str, pd.Series, np.ndarray]] = None,
    d_h_ratio: Optional[Union[float, str, pd.Series, np.ndarray]] = None,
) -> pd.DataFrame:
    """
    Calculate effective measurement height (zm) and roughness length (z0).

    Requires inst_height and crop_height, plus EITHER a `veg_type` preset
    OR custom ratio parameters (`z0_ratio` / `d_h_ratio`).

    If both veg_type and z0_ratio/d_h_ratio are provided, the user-provided ratios
    will be used instead of the VEG_PRESETS parameters.
    """
    df_out = df.copy()

    # 1. Resolve inputs
    zm_s = _resolve_input(inst_height, df_out, "inst_height")
    h_c = _resolve_input(crop_height, df_out, "crop_height")
    z0_r = (
        _resolve_input(z0_ratio, df_out, "z0_ratio")
        if z0_ratio is not None
        else None
    )
    d_h_r = (
        _resolve_input(d_h_ratio, df_out, "d_h_ratio")
        if d_h_ratio is not None
        else None
    )

    # 2. Parameter validation & preset lookup
    if veg_type is None and z0_r is None and d_h_r is None:
        raise ValueError(
            "Missing parameter rule! You must specify either:\n"
            "  1) A 'veg_type' preset (e.g., veg_type='alfalfa' or 'corn_jacobs')\n"
            "  2) Custom ratios (e.g., z0_ratio=0.12, d_h_ratio=0.67)"
        )

    if veg_type is not None:
        veg_type_clean = veg_type.lower()
        if veg_type_clean not in VEG_PRESETS:
            raise ValueError(
                f"Unknown veg_type '{veg_type}'. Available options: {list(VEG_PRESETS.keys())}"
            )
        preset = VEG_PRESETS[veg_type_clean]
    else:
        preset = {}

    # 3. Calculate Displacement Height (d_h)
    if d_h_r is not None:
        d_h_val = h_c * d_h_r
    elif preset.get("d_h_method") == "stanhill":
        d_h_val = 10 ** (0.979 * np.log10(h_c) - 0.154)
    elif "d_h_fraction" in preset:
        d_h_val = h_c * preset["d_h_fraction"]
    else:
        raise ValueError(
            "Missing displacement height rule! Specify 'd_h_ratio' or pass a valid 'veg_type'."
        )

    # 4. Calculate Roughness Length (z0)
    if z0_r is not None:
        z0_val = h_c * z0_r
    elif preset.get("z0_method") == "jacobs_1988":
        z0_val = 0.26 * (h_c - d_h_val)
    elif "z0_fraction" in preset:
        z0_val = h_c * preset["z0_fraction"]
    else:
        raise ValueError(
            "Missing roughness length rule! Specify 'z0_ratio' or pass a valid 'veg_type'."
        )

    # 5. Assign output columns
    df_out["zm"] = zm_s - d_h_val
    df_out["z0"] = z0_val

    return df_out
# ------------------------------
# Build & Run Climatology
# ------------------------------

def _resolve_input(
    val: Union[str, float, int, pd.Series, None],
    df: pd.DataFrame,
    name: str = "input",
) -> Union[pd.Series, None]:
    """Helper to convert str column name, single numeric scalar, or pd.Series

    into a consistent pd.Series matched to df.index.
    """
    if val is None:
        return None
    if isinstance(val, str):
        if val not in df.columns:
            raise KeyError(
                f"Column '{val}' specified for {name} not found in DataFrame."
            )
        return df[val]
    elif isinstance(val, (int, float, np.number)):
        # Broadcast a single scalar value to match the DataFrame index length
        return pd.Series(float(val), index=df.index, name=name)
    elif isinstance(val, pd.Series):
        # Align with original df index if needed
        return val.reindex(df.index)
    else:
        raise TypeError(
            f"Invalid type for {name}: {type(val)}. Expected str, float, int, or pd.Series."
        )

FootprintInput = Union[str, float, int, pd.Series, None]

def build_climatology(
    df: pd.DataFrame,
    model_type: str = "ffp",
    ustar: FootprintInput = "ustar_1_1_1",
    ol: FootprintInput = "mo_length_1_1_1",
    umean: FootprintInput = "ws_1_1_1",
    sigmav: FootprintInput = "v_sigma_1_1_1",
    wind_dir: FootprintInput = "wd_1_1_1",
    zm: FootprintInput = "zm",
    z0: FootprintInput = "z0",
    h: FootprintInput = 2000.0,
    dx: float = 10.0,
    dy: float = 10.0,
    domain: Tuple[float, float, float, float] = (
        -1000.0,
        1000.0,
        -1000.0,
        1000.0,
    ),
    rs: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    smooth_data: bool = True,
    verbosity: int = 2,
    logger: Optional[logging.Logger] = None,
    **model_kwargs,
) -> BaseFootprintModel:
    """Prepare inputs, validate per-model requirements, and run the footprint model."""
    if logger is None:
        logger = logging.getLogger(__name__)

    # 1. Model Registry
    models = {
        "ffp": FFPModel,
        "ffp_xr": ffp_climatology_new,
        "kormann-meixner": KormannMeixnerModel,
        "km": KormannMeixnerModel,
        "lagrangian": LSFootprintModelAdapter,
        "ls": LSFootprintModelAdapter,
        "wang": WangFootprintModel,
        "wang2006": WangFootprintModel,
    }

    # 2. Per-Model Variable Requirements
    MODEL_REQUIREMENTS: Dict[str, List[str]] = {
        "ffp": ["wind_dir", "umean", "ustar", "ol", "sigmav", "zm", "z0", "h"],
        "ffp_xr": [
            "wind_dir",
            "umean",
            "ustar",
            "ol",
            "sigmav",
            "zm",
            "z0",
            "h",
        ],
        "kormann-meixner": ["wind_dir", "umean", "ustar", "ol", "sigmav", "zm"],
        "km": ["wind_dir", "umean", "ustar", "ol", "sigmav", "zm"],
        "lagrangian": ["wind_dir", "umean", "ustar", "ol", "sigmav", "zm", "z0"],
        "ls": ["wind_dir", "umean", "ustar", "ol", "sigmav", "zm", "z0"],
        "wang": ["wind_dir", "umean", "ustar", "ol", "zm", "z0"],
        "wang2006": ["wind_dir", "umean", "ustar", "ol", "zm", "z0"],
    }

    key = model_type.lower()
    if key not in models:
        raise ValueError(
            f"Unknown model type '{model_type}'. Choose from: {list(models.keys())}"
        )

    ModelClass = models[key]

    # 3. Resolve all flexible inputs into standard pd.Series inside df_prepared
    df_prepared = pd.DataFrame(index=df.index)

    df_prepared["ustar"] = _resolve_input(ustar, df, "ustar")
    df_prepared["ol"] = _resolve_input(ol, df, "ol")
    df_prepared["umean"] = _resolve_input(umean, df, "umean")
    df_prepared["sigmav"] = _resolve_input(sigmav, df, "sigmav")
    df_prepared["wind_dir"] = _resolve_input(wind_dir, df, "wind_dir")
    df_prepared["zm"] = _resolve_input(zm, df, "zm")
    df_prepared["z0"] = _resolve_input(z0, df, "z0")
    df_prepared["h"] = _resolve_input(h, df, "h")

    # 4. Validate model-specific variable requirements
    required_cols = MODEL_REQUIREMENTS.get(key, [])
    missing_cols = [
        col
        for col in required_cols
        if col not in df_prepared.columns or df_prepared[col].isnull().all()
    ]

    if missing_cols:
        raise ValueError(
            f"Model '{model_type}' requires the following missing parameters/columns: {missing_cols}. "
            f"Please specify them as column names in `df`, constant scalars, or pd.Series."
        )

    # 5. Build parameter dictionary and instantiate model
    common_params = dict(
        df=df_prepared,
        domain=list(domain),
        dx=float(dx),
        dy=float(dy),
        rs=rs,
        smooth_data=smooth_data,
        verbosity=verbosity,
        logger=logger,
    )
    common_params.update(model_kwargs)

    try:
        logger.info(f"Initializing {model_type} model...")
        model = ModelClass(**common_params)

        logger.info("Running footprint calculation...")
        model.run(return_result=True)

        logger.info("Footprint calculation completed successfully.")
        return model

    except Exception as e:
        logger.error(f"Model {model_type} failed: {e}")
        raise

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
    return df[le_col] / 680.6




@dataclass
class SummaryResult:
    f_daily_mean: Optional[xr.DataArray] = None
    f_monthly_mean: Optional[xr.DataArray] = None
    f_daily_et_weighted: Optional[xr.DataArray] = None
    f_monthly_et_weighted: Optional[xr.DataArray] = None


def summarize_periods(
    model: BaseFootprintModel,
    df: pd.DataFrame,
    et_source: str = "LE",
    daily: bool = True,
    monthly: bool = True,
    normalize_each_time: bool = False, # 1. CHANGED TO FALSE BY DEFAULT
) -> SummaryResult:
    """Build daily/monthly summaries matching Kljun canonical denominator logic."""

    if not hasattr(model, "f_2d") or model.f_2d is None:
        raise ValueError("Model must have f_2d attribute computed")

    f_2d = model.get_footprint_timeseries()
    if f_2d is None:
        raise ValueError("Model does not support time-resolved footprints.")

    # 2. CRITICAL FIX FOR DENOMINATOR: 
    # Calculate the sum of each 30-min slice. If it's 0, it means it was an invalid/excluded step.
    slice_sums = f_2d.sum(dim=("x", "y"))
    
    # Convert invalid 0.0 timesteps to NaN. 
    # Xarray's .mean() skips NaNs, meaning it will divide ONLY by the valid footprint count!
    f = f_2d.where(slice_sums > 0, np.nan)

    if normalize_each_time:
        f = _ensure_time_normalized(f)

    res = SummaryResult()

    # Mean footprints (Now correctly dividing only by valid_count)
    if daily:
        res.f_daily_mean = f.resample(time="1D").mean(skipna=True)
    if monthly:
        res.f_monthly_mean = f.resample(time="MS").mean(skipna=True)

    # ET-weighted aggregation
    et_mmhr = _et_from_le(df, le_col=et_source).reindex(f["time"].to_index(), method=None)
    et_da = xr.DataArray(et_mmhr.values, coords={"time": f["time"]}, dims=("time",))
    
    # Mask ET data where footprints are invalid so the weights remain clean
    et_da = et_da.where(slice_sums > 0, np.nan)

    weighted = f * et_da

    if daily:
        num = weighted.resample(time="1D").sum(skipna=True)
        den = et_da.resample(time="1D").sum(skipna=True)
        res.f_daily_et_weighted = num / den.where(den > 0)

    if monthly:
        num = weighted.resample(time="MS").sum(skipna=True)
        den = et_da.resample(time="MS").sum(skipna=True)
        res.f_monthly_et_weighted = num / den.where(den > 0)

    return res

# ------------------------------
# Export utilities
# ------------------------------


def _choose_utm_epsg(lon: float, lat: float) -> int:
    """Pick a UTM EPSG (NAD83 / WGS84 UTM) from lon/lat."""
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    return int(f"326{zone:02d}")


def _make_contour_polygon_from_field_alt(
    z2d: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    level: float = 0.8,
    method: str = "auto",
):
    """
    Extract polygon boundary for cumulative-source threshold using
    scikit-image or rasterio vectorization.
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

    x = np.asarray(x)
    y = np.asarray(y)
    nx, ny = x.size, y.size
    col_ix = np.arange(nx, dtype=float)
    row_ix = np.arange(ny, dtype=float)

    def map_cols_to_x(cols):
        return np.interp(cols, col_ix, x)

    def map_rows_to_y(rows):
        return np.interp(rows, row_ix, y)

    # Try scikit-image first
    if method in ("auto", "skimage"):
        try:
            from skimage import measure

            conts = measure.find_contours(mask, level=0.5)
            if conts:
                arr = max(conts, key=lambda a: a.shape[0])
                rows, cols = arr[:, 0], arr[:, 1]
                xv = map_cols_to_x(cols)
                yv = map_rows_to_y(rows)
                return np.column_stack([xv, yv])
        except Exception:
            if method == "skimage":
                return None

    # Fallback to rasterio
    if method in ("auto", "rasterio"):
        try:


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
    model: FFPModel,
    summaries: SummaryResult,
    df: pd.DataFrame,
    station_lat: float,
    station_lon: float,
    gpkg_path: str | Path,
    crs_out: str | int = "auto",
    levels: tuple = (0.8,),
    stats_csv: str | Path | None = None,
    centroid_out: str | int = 4326,
    rn_col: str | None = None,
    h_col: str | None = None,
    le_col: str | None = None,
    g_col: str | None = None,
    contour_method: str = "auto",
) -> Path:
    """
    Export source-area contours to GeoPackage with energy balance stats.

    Adapted to work with FFPModel output structure.
    """


    # Get grid coordinates from model
    x = model.x
    y = model.y

    # CRS setup
    if crs_out == "auto":
        epsg = _choose_utm_epsg(station_lon, station_lat)
        dst_crs = CRS.from_epsg(epsg)
    else:
        dst_crs = CRS.from_user_input(crs_out)

    wgs84 = CRS.from_epsg(4326)
    to_proj = Transformer.from_crs(wgs84, dst_crs, always_xy=True)
    to_wgs = Transformer.from_crs(dst_crs, wgs84, always_xy=True)

    try:
        centroid_crs = CRS.from_user_input(centroid_out)
    except Exception:
        centroid_crs = CRS.from_epsg(4326)
    to_centroid = Transformer.from_crs(dst_crs, centroid_crs, always_xy=True)
    to_wgs_cent = Transformer.from_crs(dst_crs, wgs84, always_xy=True)

    # Energy balance helpers
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

    # Prepare output
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

    # Layer collection helper
    def _collect_layer(da: Optional[xr.DataArray], base_layer: str):
        if da is None:
            return
        for r in levels:
            records = []
            for i in range(da.sizes["time"]):
                z = da.isel(time=i).values
                verts = _make_contour_polygon_from_field_alt(
                    z, x, y, level=float(r), method=contour_method
                )
                poly = _to_world_coords(verts)
                if poly is None or poly.is_empty:
                    continue

                ts = pd.Timestamp(da["time"].values[i]).isoformat()

                # EB stats
                if base_layer.startswith("daily"):
                    sdf = df.loc[ts[:10]] if ts else df
                else:
                    sdf = df.loc[ts[:7]] if ts else df
                eb = _eb_stats(sdf)

                rec = {"time": ts, "r": float(r), **eb, "geometry": poly}
                records.append(rec)

            if not records:
                continue

            lname = f"{base_layer}_r{int(round(float(r) * 100))}"
            gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=dst_crs)

            # Metrics
            gdf["area_m2"] = gdf.geometry.area
            gdf["perimeter_m"] = gdf.geometry.length
            gdf["area_ha"] = gdf["area_m2"] / 10000.0
            gdf["area_acres"] = gdf["area_m2"] / 4046.85642

            # Centroids
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

            # ET stats
            try:
                if le_c and le_c in df.columns:
                    if base_layer.startswith("daily"):
                        tstamp = pd.to_datetime(gdf["time"].iloc[0])
                        et_slice = df.loc[str(tstamp.date()), le_c]
                    else:
                        tstamp = pd.to_datetime(gdf["time"].iloc[0])
                        et_slice = df.loc[tstamp.strftime("%Y-%m"), le_c]

                    et_vals = (
                        pd.to_numeric(et_slice, errors="coerce").dropna().values / 680.6
                    )
                    if et_vals.size > 0:
                        gdf["ET_mean_mmhr"] = float(np.nanmean(et_vals))
                        gdf["ET_sum_mm"] = float(np.nansum(et_vals * 0.5))
            except Exception:
                pass

            gdf.to_file(gpkg_path, layer=lname, driver="GPKG")
            stats_records.extend(gdf.drop(columns="geometry").to_dict(orient="records"))

    # Write layers
    _collect_layer(summaries.f_daily_mean, "daily_mean")
    _collect_layer(summaries.f_monthly_mean, "monthly_mean")
    _collect_layer(summaries.f_daily_et_weighted, "daily_etw")
    _collect_layer(summaries.f_monthly_et_weighted, "monthly_etw")

    # Optional stats CSV
    if stats_csv and stats_records:
        stats_df = pd.DataFrame(stats_records)
        stats_df.to_csv(stats_csv, index=False)

    return gpkg_path


# Remaining functions (export_rasters_geotiff, export_contour_stats_csv) remain the same
# but should use model.x, model.y instead of clim.x, clim.y

def export_rasters_geotiff(
    model: FFPModel,
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
    """Export each time slice as GeoTIFF (adapted for FFPModel)."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x = np.asarray(model.x)
    y = np.asarray(model.y)

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

    left = (x.min() + x0) - xres / 2.0
    top = (y.max() + y0) + yres / 2.0
    transform = from_origin(left, top, xres, yres)

    def _write_series(da, layername):
        if da is None:
            return
        
        # 1. CRITICAL FIX: Transpose from model layout (time, x, y) to GIS layout (time, y, x)
        da_gis = da.transpose("time", "y", "x")
        times = pd.to_datetime(da_gis["time"].values)

        for i, ts in enumerate(times):
            # Extract the transposed 2D grid slice
            arr = da_gis.isel(time=i).values.astype(dtype)
            
            # 2. CORRECTED FLIP: Since Kljun's internal y vector increases South-to-North, 
            # but GeoTIFF matrix rows read North-to-South, we flip the rows vertically.
            if y[1] > y[0]:
                arr = np.flipud(arr)

            if layername.startswith("daily"):
                suf = ts.strftime("%Y%m%d")
            else:
                suf = ts.strftime("%Y%m")
            fp = out_dir / f"{prefix}_{layername}_{suf}.tif"

            with rasterio.open(
                fp,
                "w",
                driver="GTiff",
                height=arr.shape[0],  # Now correctly maps to len(y)
                width=arr.shape[1],   # Now correctly maps to len(x)
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

def export_contour_stats_csv(
    df: pd.DataFrame,
    model: FFPModel,
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
    """Export contour stats to CSV (adapted for FFPModel)."""


    x = model.x
    y = model.y

    if crs_out == "auto":
        epsg = _choose_utm_epsg(station_lon, station_lat)
        dst_crs = CRS.from_epsg(epsg)
    else:
        dst_crs = CRS.from_user_input(crs_out)

    wgs84 = CRS.from_epsg(4326)
    to_proj = Transformer.from_crs(wgs84, dst_crs, always_xy=True)
    to_wgs = Transformer.from_crs(dst_crs, wgs84, always_xy=True)

    x0, y0 = to_proj.transform(station_lon, station_lat)

    def _to_world_coords(vertices_xy):
        if vertices_xy is None:
            return None
        vx = vertices_xy[:, 0] + x0
        vy = vertices_xy[:, 1] + y0
        return Polygon(np.column_stack([vx, vy]))

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
                    area_ha = poly.area / 10000.0
                    cx, cy = poly.centroid.x, poly.centroid.y
                    clon, clat = to_wgs.transform(cx, cy)
                    writer.writerow([base_layer, ts, float(r), area_ha, clon, clat])

        _process_layer(summaries.f_daily_mean, "daily_mean")
        _process_layer(summaries.f_monthly_mean, "monthly_mean")
        _process_layer(summaries.f_daily_et_weighted, "daily_etw")
        _process_layer(summaries.f_monthly_et_weighted, "monthly_etw")

    return csv_path
