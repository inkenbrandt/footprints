# -*- coding: utf-8 -*-
"""
footprint_daily_summary.py  —  utilities to summarize source areas
now with ET-weighted footprints for daily, monthly, and whole POR.
"""

from __future__ import annotations
import math, logging
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
from shapely.geometry import box, MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.geometry import mapping as shp_mapping
import fiona
from .ffp_xr import ffp_climatology_new as FFPClass  # type: ignore[import]


def _ensure_logger(logger=None):
    if isinstance(logger, logging.Logger):
        return logger
    lg = logging.getLogger("footprint_summary")
    if not lg.handlers:
        lg.addHandler(logging.StreamHandler())
    lg.setLevel(logging.WARNING)
    return lg


def _level_for_fraction(array2d: np.ndarray, r: float) -> float:
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


def _mask_to_multipolygon(xs, ys, mask, dx, dy):
    half_dx, half_dy = dx / 2.0, dy / 2.0
    ii, jj = np.where(mask)
    if ii.size == 0:
        return MultiPolygon()
    cells = [
        box(xs[i] - half_dx, ys[j] - half_dy, xs[i] + half_dx, ys[j] + half_dy)
        for i, j in zip(ii, jj)
    ]
    merged = unary_union(cells)
    if merged.is_empty:
        return MultiPolygon()
    if isinstance(merged, Polygon):
        return MultiPolygon([merged.buffer(0)])
    if isinstance(merged, MultiPolygon):
        return MultiPolygon([p.buffer(0) for p in merged.geoms])
    return MultiPolygon()


def _major_minor_axes(poly: Polygon):
    mrr = poly.minimum_rotated_rectangle
    coords = np.asarray(mrr.exterior.coords)  # type: ignore
    edges = [np.linalg.norm(coords[i + 1] - coords[i]) for i in range(4)]
    Ls = sorted(edges)  # type: ignore
    minor, major = Ls[1], Ls[3]
    k = int(np.argmax(edges))
    vec = coords[k + 1] - coords[k]
    angle = np.degrees(np.arctan2(vec[1], vec[0]))
    return float(major), float(minor), float(angle)


def _et_series(df: pd.DataFrame, et_col=None):
    if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 1:
        dt_seconds = df.index.to_series().diff().median().total_seconds()
        if not np.isfinite(dt_seconds) or dt_seconds <= 0:
            dt_seconds = 1800.0
    else:
        dt_seconds = 1800.0
    dt_hours = dt_seconds / 3600.0
    if et_col and et_col in df.columns:
        et_rate = pd.to_numeric(df[et_col], errors="coerce")
        return (et_rate * dt_hours).fillna(0.0)
    for c in df.columns:
        if c.upper().startswith("ET"):
            et_rate = pd.to_numeric(df[c], errors="coerce")
            return (et_rate * dt_hours).fillna(0.0)
    le_candidates = [c for c in df.columns if c.upper().startswith("LE")]
    if le_candidates:
        le = pd.to_numeric(df[le_candidates[0]], errors="coerce")
        Lv = 2.45e6
        et_mm_h = le * 3600.0 / Lv
        return (et_mm_h * dt_hours).fillna(0.0)
    return pd.Series(1.0, index=df.index)


def _run_model(FFPClass, df, domain, dx, dy, logger):
    model = FFPClass(
        df=df,
        domain=np.array(domain, float),
        dx=float(dx),
        dy=float(dy),
        rs=[0.8],
        smooth_data=True,
        crop=False,
        verbosity=0,
        logger=logger,
    )
    model.run()
    f = np.array(model.fclim_2d.values, float)
    return np.array(model.x, float), np.array(model.y, float), f


def _weighted_climatology(FFPClass, df, weights, domain, dx, dy, logger):
    w = pd.to_numeric(weights.reindex(df.index), errors="coerce").fillna(0.0)
    pos = (w > 0) & np.isfinite(w)
    if pos.sum() == 0:
        return _run_model(FFPClass, df, domain, dx, dy, logger)
    xs = ys = None
    acc = None
    wsum = 0.0
    for idx, row in df[pos].iterrows():
        wi = float(w.loc[idx])
        if wi <= 0 or not np.isfinite(wi):
            continue
        xi, yi, fi = _run_model(FFPClass, row.to_frame().T, domain, dx, dy, logger)
        if xs is None:
            xs, ys, acc = xi, yi, wi * fi
        else:
            acc += wi * fi  # type: ignore
        wsum += wi
    if acc is None or wsum <= 0:
        return _run_model(FFPClass, df, domain, dx, dy, logger)
    return xs, ys, acc / wsum


def et_weighted_summaries(
    df,
    FFPClass,
    domain,
    dx,
    dy,
    fraction=0.8,
    min_records=24,
    et_col=None,
    logger=None,
    save_gpkg_daily=None,
    layer_daily="daily_footprint_w",
    save_gpkg_monthly=None,
    layer_monthly="monthly_footprint_w",
    save_gpkg_por=None,
    layer_por="por_footprint_w",
):
    lg = _ensure_logger(logger)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index must be DateTimeIndex")
    et_mm = _et_series(df, et_col=et_col).clip(lower=0).fillna(0.0)

    # DAILY
    daily_rows = []
    sink_d = None
    if save_gpkg_daily:
        schema = {
            "geometry": "MultiPolygon",
            "properties": {
                "date": "str",
                "fraction": "float",
                "n_obs": "int",
                "et_mm": "float",
                "area_m2": "float",
                "centroid_x": "float",
                "centroid_y": "float",
                "major_m": "float",
                "minor_m": "float",
                "orientation_deg": "float",
            },
        }
        sink_d = fiona.open(
            save_gpkg_daily,
            "w",
            driver="GPKG",
            layer=layer_daily,
            schema=schema,
            crs=None,
        )
    for date, sub in df.groupby(df.index.normalize()):
        n = len(sub)
        if n < min_records:
            continue
        w = et_mm.loc[sub.index]
        if (w > 0).sum() == 0:
            continue
        x, y, f = _weighted_climatology(FFPClass, sub, w, domain, dx, dy, lg)
        total = np.nansum(f)
        if not np.isfinite(total) or total <= 0:
            continue
        f_norm = f / total
        level = _level_for_fraction(f_norm, fraction)
        if not np.isfinite(level) or level <= 0:
            continue
        mp = _mask_to_multipolygon(x, y, (f_norm >= level), dx, dy)
        if mp.is_empty:
            continue
        area = mp.area
        cx, cy = mp.centroid.x, mp.centroid.y
        largest = max(mp.geoms, key=lambda p: p.area)
        # major/minor
        mrr = largest.minimum_rotated_rectangle
        coords = np.asarray(mrr.exterior.coords)  # type: ignore
        edges = [np.linalg.norm(coords[i + 1] - coords[i]) for i in range(4)]
        Ls = sorted(edges)  # type: ignore
        minor, major = Ls[1], Ls[3]
        k = int(np.argmax(edges))
        vec = coords[k + 1] - coords[k]
        angle = np.degrees(np.arctan2(vec[1], vec[0]))
        row = {
            "date": pd.Timestamp(date).date().isoformat(),
            "n_obs": int(n),
            "et_mm": float(w.sum()),
            "area_m2": float(area),
            "centroid_x": float(cx),
            "centroid_y": float(cy),
            "major_axis_m": float(major),
            "minor_axis_m": float(minor),
            "orientation_deg_from_x": float(angle),
            "poly_parts": int(len(mp.geoms)),
        }
        daily_rows.append(row)
        if sink_d is not None:
            sink_d.write(
                {
                    "geometry": shp_mapping(mp),
                    "properties": {
                        "date": row["date"],
                        "fraction": float(fraction),
                        "n_obs": int(n),
                        "et_mm": float(w.sum()),
                        "area_m2": float(area),
                        "centroid_x": float(cx),
                        "centroid_y": float(cy),
                        "major_m": float(major),
                        "minor_m": float(minor),
                        "orientation_deg": float(angle),
                    },
                }
            )
    if sink_d is not None:
        sink_d.close()
    daily_df = pd.DataFrame(daily_rows)

    # MONTHLY
    monthly_rows = []
    sink_m = None
    if save_gpkg_monthly:
        schema = {
            "geometry": "MultiPolygon",
            "properties": {
                "month": "str",
                "fraction": "float",
                "n_obs": "int",
                "et_mm": "float",
                "area_m2": "float",
                "centroid_x": "float",
                "centroid_y": "float",
                "major_m": "float",
                "minor_m": "float",
                "orientation_deg": "float",
            },
        }
        sink_m = fiona.open(
            save_gpkg_monthly,
            "w",
            driver="GPKG",
            layer=layer_monthly,
            schema=schema,
            crs=None,
        )
    for month, sub in df.groupby(df.index.to_period("M")):
        n = len(sub)
        w = et_mm.loc[sub.index]
        if n < min_records or (w > 0).sum() == 0:
            continue
        x, y, f = _weighted_climatology(FFPClass, sub, w, domain, dx, dy, lg)
        total = np.nansum(f)
        if not np.isfinite(total) or total <= 0:
            continue
        f_norm = f / total
        level = _level_for_fraction(f_norm, fraction)
        if not np.isfinite(level) or level <= 0:
            continue
        mp = _mask_to_multipolygon(x, y, (f_norm >= level), dx, dy)
        if mp.is_empty:
            continue
        area = mp.area
        cx, cy = mp.centroid.x, mp.centroid.y
        # major/minor
        mrr = (
            mp.geoms[0].minimum_rotated_rectangle
            if len(mp.geoms) > 0
            else mp.minimum_rotated_rectangle
        )
        coords = np.asarray(mrr.exterior.coords)  # type: ignore
        edges = [np.linalg.norm(coords[i + 1] - coords[i]) for i in range(4)]
        Ls = sorted(edges)  # type: ignore
        minor, major = Ls[1], Ls[3]
        k = int(np.argmax(edges))
        vec = coords[k + 1] - coords[k]
        angle = np.degrees(np.arctan2(vec[1], vec[0]))
        key = str(pd.Period(month, freq="M"))
        row = {
            "month": key,
            "n_obs": int(n),
            "et_mm": float(w.sum()),
            "area_m2": float(area),
            "centroid_x": float(cx),
            "centroid_y": float(cy),
            "major_axis_m": float(major),
            "minor_axis_m": float(minor),
            "orientation_deg_from_x": float(angle),
            "poly_parts": int(len(mp.geoms)),
        }
        monthly_rows.append(row)
        if sink_m is not None:
            sink_m.write(
                {
                    "geometry": shp_mapping(mp),
                    "properties": {
                        "month": key,
                        "fraction": float(fraction),
                        "n_obs": int(n),
                        "et_mm": float(w.sum()),
                        "area_m2": float(area),
                        "centroid_x": float(cx),
                        "centroid_y": float(cy),
                        "major_m": float(major),
                        "minor_m": float(minor),
                        "orientation_deg": float(angle),
                    },
                }
            )
    if sink_m is not None:
        sink_m.close()
    monthly_df = pd.DataFrame(monthly_rows)

    # POR
    por_rows = []
    sink_p = None
    if save_gpkg_por:
        schema = {
            "geometry": "MultiPolygon",
            "properties": {
                "period": "str",
                "fraction": "float",
                "n_obs": "int",
                "et_mm": "float",
                "area_m2": "float",
                "centroid_x": "float",
                "centroid_y": "float",
                "major_m": "float",
                "minor_m": "float",
                "orientation_deg": "float",
            },
        }
        sink_p = fiona.open(
            save_gpkg_por, "w", driver="GPKG", layer=layer_por, schema=schema, crs=None
        )
    w = et_mm
    if (w > 0).sum() > 0 and len(df) >= min_records:
        x, y, f = _weighted_climatology(FFPClass, df, w, domain, dx, dy, lg)
        total = np.nansum(f)
        if np.isfinite(total) and total > 0:
            f_norm = f / total
            level = _level_for_fraction(f_norm, fraction)
            if np.isfinite(level) and level > 0:
                mp = _mask_to_multipolygon(x, y, (f_norm >= level), dx, dy)
                if not mp.is_empty:
                    area = mp.area
                    cx, cy = mp.centroid.x, mp.centroid.y
                    mrr = (
                        mp.geoms[0].minimum_rotated_rectangle
                        if len(mp.geoms) > 0
                        else mp.minimum_rotated_rectangle
                    )
                    coords = np.asarray(mrr.exterior.coords)  # type: ignore
                    edges = [
                        np.linalg.norm(coords[i + 1] - coords[i]) for i in range(4)
                    ]
                    Ls = sorted(edges)  # type: ignore
                    minor, major = Ls[1], Ls[3]
                    k = int(np.argmax(edges))
                    vec = coords[k + 1] - coords[k]
                    angle = np.degrees(np.arctan2(vec[1], vec[0]))
                    row = {
                        "period": "full_record",
                        "n_obs": int(len(df)),
                        "et_mm": float(w.sum()),
                        "area_m2": float(area),
                        "centroid_x": float(cx),
                        "centroid_y": float(cy),
                        "major_axis_m": float(major),
                        "minor_axis_m": float(minor),
                        "orientation_deg_from_x": float(angle),
                        "poly_parts": int(len(mp.geoms)),
                    }
                    por_rows.append(row)
                    if sink_p is not None:
                        sink_p.write(
                            {
                                "geometry": shp_mapping(mp),
                                "properties": {
                                    "period": "full_record",
                                    "fraction": float(fraction),
                                    "n_obs": int(len(df)),
                                    "et_mm": float(w.sum()),
                                    "area_m2": float(area),
                                    "centroid_x": float(cx),
                                    "centroid_y": float(cy),
                                    "major_m": float(major),
                                    "minor_m": float(minor),
                                    "orientation_deg": float(angle),
                                },
                            }
                        )
    if sink_p is not None:
        sink_p.close()
    por_df = pd.DataFrame(por_rows)
    return daily_df, monthly_df, por_df
