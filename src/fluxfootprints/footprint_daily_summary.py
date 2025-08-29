"""
footprint_daily_summary.py

Utilities to compute daily source-area summaries (e.g., 80% polygons) from a
Kljun et al. (2015)-style footprint climatology computed with a provided class.

Designed to work with `ffp_xr.py`'s `ffp_climatology_new` class.
"""

from __future__ import annotations
import math
import logging
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from shapely.geometry import box, MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.geometry import mapping as shp_mapping
import fiona

from .ffp_xr import ffp_climatology_new as FFPClass  # type: ignore[import]


def _ensure_logger(logger: Optional[logging.Logger]) -> logging.Logger:
    if isinstance(logger, logging.Logger):
        return logger
    lg = logging.getLogger("footprint_daily_summary")
    if not lg.handlers:
        lg.addHandler(logging.StreamHandler())
    lg.setLevel(logging.WARNING)
    return lg


def _level_for_fraction(array2d: np.ndarray, r: float) -> float:
    """
    Given a non-negative 2D array whose sum is > 0, return the threshold `t`
    so that the set {array2d >= t} contains approximately fraction `r`
    of the total mass (Greedy by value).
    """
    vals = array2d.ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0 or np.nansum(vals) == 0:
        return np.nan
    order = np.argsort(vals)[::-1]
    sorted_vals = vals[order]
    csum = np.cumsum(sorted_vals)
    csum /= csum[-1] if csum[-1] != 0 else 1.0
    idx = np.searchsorted(csum, r, side="left")
    idx = min(max(idx, 0), len(sorted_vals) - 1)  # type: ignore
    return float(sorted_vals[idx])


def _mask_to_polygon(
    xs: np.ndarray, ys: np.ndarray, mask: np.ndarray, dx: float, dy: float
) -> MultiPolygon:
    """
    Convert a boolean mask (True where inside) on a rectilinear grid (xs, ys)
    to a dissolved polygon by unioning grid cells where mask is True.
    """
    half_dx = dx / 2.0
    half_dy = dy / 2.0
    ii, jj = np.where(mask)
    if ii.size == 0:
        return MultiPolygon()
    cells = [box(xs[i] - half_dx, ys[j] - half_dy, xs[i] + half_dx, ys[j] + half_dx) for i, j in zip(ii, jj)]  # type: ignore
    merged = unary_union(cells)
    if merged.is_empty:
        return MultiPolygon()
    if isinstance(merged, Polygon):
        merged = merged.buffer(0)  # clean
        return MultiPolygon([merged])
    if isinstance(merged, MultiPolygon):
        # clean each
        parts = [
            p.buffer(0)
            for p in merged.geoms
            if isinstance(p, Polygon) and not p.is_empty
        ]
        return MultiPolygon(parts) if parts else MultiPolygon()
    # Fallback
    return MultiPolygon()


def _major_minor_axes(poly: Polygon) -> Tuple[float, float, float]:
    """
    Compute major/minor axis lengths and orientation (degrees from +x) of the
    minimum rotated rectangle bounding `poly`. Returns (major, minor, angle_deg).
    """
    mrr = poly.minimum_rotated_rectangle
    coords = np.asarray(mrr.exterior.coords)  # type: ignore
    # coords has 5 points (closed); compute edge lengths
    edges = [np.linalg.norm(coords[i + 1] - coords[i]) for i in range(4)]
    # Find two unique edge lengths
    Ls = sorted(edges)  # type: ignore # [minor, minor, major, major]
    minor = Ls[1]
    major = Ls[3]
    # Orientation: angle of the longest edge
    # Find index of longest edge
    k = int(np.argmax(edges))
    vec = coords[k + 1] - coords[k]
    angle = math.degrees(math.atan2(vec[1], vec[0]))
    return float(major), float(minor), float(angle)


def daily_source_area_summary(
    df: pd.DataFrame,
    FFPClass,
    domain: np.ndarray,
    dx: float,
    dy: float,
    fraction: float = 0.8,
    min_records: int = 24,
    inst_height=3.66,
    canopy_height=0.5,
    logger: Optional[logging.Logger] = None,
    save_gpkg: Optional[str] = None,
    layer_name: str = "daily_source_area",
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Compute a daily summary of the source area at a given cumulative fraction
    (80% by default). For each day:
      - Fit a footprint climatology over that day's records.
      - Derive the `fraction`-mass region (by iso-value threshold).
      - Convert to polygon(s) and compute geometry metrics.
      - Optionally append polygons into a GeoPackage.

    Parameters
    ----------
    df : pandas.DataFrame
        Half-hourly/hourly input with a DateTimeIndex and required columns
        for the provided footprint class.
    FFPClass : class
        The footprint model class callable like FFPClass(df=..., domain=..., dx=..., dy=...)
        and providing attributes `x`, `y`, and `fclim_2d` after `run()`.
    domain : np.ndarray
        [minx, maxx, miny, maxy] domain in meters.
    dx, dy : float
        Grid resolution in meters.
    fraction : float, default 0.8
        Source-area mass fraction for contour (0–1).
    min_records : int, default 24
        Skip days with fewer than this many input rows.
    logger : logging.Logger, optional
        Logger for the model and this function.
    save_gpkg : str, optional
        If provided, append daily polygons to this GeoPackage path.
    layer_name : str, default "daily_source_area"
        Output layer name if saving to a GeoPackage.

    Returns
    -------
    (summary_df, gpkg_path) : (pandas.DataFrame, Optional[str])
        summary_df columns:
          - date
          - n_obs
          - area_m2
          - centroid_x
          - centroid_y
          - major_axis_m
          - minor_axis_m
          - orientation_deg_from_x
          - poly_parts
    """
    lg = _ensure_logger(logger)

    # Group by day
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index must be a pandas.DatetimeIndex")

    groups = df.groupby(df.index.normalize())

    # Prepare optional gpkg sink
    sink = None
    if save_gpkg:
        schema = {
            "geometry": "MultiPolygon",
            "properties": {
                "date": "str",
                "fraction": "float",
                "n_obs": "int",
                "area_m2": "float",
                "centroid_x": "float",
                "centroid_y": "float",
                "major_m": "float",
                "minor_m": "float",
                "orientation_deg": "float",
            },
        }
        sink = fiona.open(
            save_gpkg,
            mode="w",
            driver="GPKG",
            layer=layer_name,
            schema=schema,
            crs=None,
        )

    rows: List[dict] = []
    try:
        for date, sub in groups:
            n = len(sub)
            if n < min_records:
                lg.info(f"Skipping {date.date()} (only {n} records)")
                continue
            # Build and run model
            model = FFPClass(
                df=sub,
                domain=np.array(domain, dtype=float),
                dx=float(dx),
                dy=float(dy),
                rs=[fraction],
                smooth_data=True,
                crop=False,
                inst_height=inst_height,
                canopy_height=canopy_height,
                verbosity=0,
                logger=lg,
            )
            model.run()

            f = model.fclim_2d
            f_arr = f.values.astype(float)
            total = np.nansum(f_arr)
            if not np.isfinite(total) or total <= 0:
                lg.warning(f"No valid footprint for {date.date()}")
                continue
            f_norm = f_arr / total

            level = _level_for_fraction(f_norm, fraction)
            if not np.isfinite(level) or level <= 0:
                lg.warning(f"Could not determine level for {date.date()}")
                continue

            mask = f_norm >= level
            mp = _mask_to_polygon(model.x, model.y, mask, dx=float(dx), dy=float(dy))
            if mp.is_empty:
                lg.warning(f"Empty polygon for {date.date()}")
                continue

            # Metrics
            area = mp.area
            cx, cy = mp.centroid.x, mp.centroid.y
            # Use the largest part for axis metrics
            largest = max(mp.geoms, key=lambda p: p.area)
            major, minor, angle = _major_minor_axes(largest)

            row = {
                "date": pd.Timestamp(date).date().isoformat(),
                "n_obs": int(n),
                "area_m2": float(area),
                "centroid_x": float(cx),
                "centroid_y": float(cy),
                "major_axis_m": float(major),
                "minor_axis_m": float(minor),
                "orientation_deg_from_x": float(angle),
                "poly_parts": int(len(mp.geoms)),
            }
            rows.append(row)

            # Optional write
            if sink is not None:
                sink.write(
                    {
                        "geometry": shp_mapping(mp),
                        "properties": {
                            "date": row["date"],
                            "fraction": float(fraction),
                            "n_obs": int(n),
                            "area_m2": float(area),
                            "centroid_x": float(cx),
                            "centroid_y": float(cy),
                            "major_m": float(major),
                            "minor_m": float(minor),
                            "orientation_deg": float(angle),
                        },
                    }
                )

    finally:
        if sink is not None:
            sink.close()

    return pd.DataFrame(rows), (save_gpkg if save_gpkg else None)
